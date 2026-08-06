#!/usr/bin/env python3
import hashlib
import json
import pathlib
import sys
import time
import urllib.error
import urllib.request

BASE_URL = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:18083"
RESULT_DIR = pathlib.Path(sys.argv[2] if len(sys.argv) > 2 else "results")
MODEL = pathlib.Path(sys.argv[3] if len(sys.argv) > 3 else "models/gemma-4-e2b-it.gguf")
MODE = sys.argv[4] if len(sys.argv) > 4 else "run"
TOKEN = "phase7-report-token"
PRIMARY = "gemma-4-e2b-it"
SECONDARY = "gemma-phase7-secondary"
COVERAGE = json.loads((RESULT_DIR / "phase7-coverage.json").read_text(encoding="utf-8"))
header = pathlib.Path("native/include/cusco_executor.h").read_text(encoding="utf-8")
abi_line = next(
    line for line in header.splitlines() if line.startswith("#define CUSCO_EXECUTOR_ABI_VERSION ")
)
BUILD = {
    "llama_cpp_tag": pathlib.Path("llama.cpp-version.txt").read_text(encoding="utf-8").strip(),
    "executor_abi": int(abi_line.split()[-1].removesuffix("u")),
}


def request(method, path, body=None, timeout=600):
    encoded = None if body is None else json.dumps(body).encode()
    headers = {"authorization": f"Bearer {TOKEN}", "content-type": "application/json"}
    return urllib.request.urlopen(
        urllib.request.Request(BASE_URL + path, data=encoded, headers=headers, method=method),
        timeout=timeout,
    )


def wait_for_server():
    deadline = time.monotonic() + 180
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(BASE_URL + "/openapi.json", timeout=5):
                return
        except (OSError, urllib.error.URLError):
            time.sleep(1)
    raise SystemExit("server did not become ready within 180 seconds")


def digest(path):
    value = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            value.update(chunk)
    return value.hexdigest()


def status():
    with request("GET", "/native/status") as response:
        return json.load(response)


def models():
    with request("GET", "/native/models") as response:
        return json.load(response)


def complete(model, prompt, context_id=None):
    body = {"model": model, "prompt": prompt, "max_tokens": 8, "stream": False}
    if context_id is not None:
        body["context_id"] = context_id
    started = time.perf_counter()
    with request("POST", "/v1/completions", body) as response:
        payload = json.load(response)
    return payload, round((time.perf_counter() - started) * 1000, 2)


def resident_rows(snapshot):
    residency = snapshot.get("residency") or {}
    return residency.get("models", [])


wait_for_server()
output = RESULT_DIR / "phase7-server.json"
if MODE == "restart-check":
    artifact = json.loads(output.read_text(encoding="utf-8"))
    restarted = status()
    restarted_catalog = models()
    restarted_models = restarted_catalog.get("data", [])
    continuation, wall_ms = complete(PRIMARY, "After restart, continue with one concise sentence.")
    rows = resident_rows(restarted)
    previous_primary = next(
        model for model in artifact["catalog"]["data"] if model["id"] == PRIMARY
    )
    restarted_primary = next(model for model in restarted_models if model["id"] == PRIMARY)
    artifact["restart"] = {
        "status": restarted,
        "models": restarted_catalog,
        "continuation": continuation,
        "wall_ms": wall_ms,
    }
    artifact["build"] = BUILD
    artifact["coverage"] = COVERAGE
    artifact["checks"]["restart retained durable catalog"] = any(
        model["id"] == PRIMARY for model in restarted_models
    )
    artifact["checks"]["restart preserved the immutable model epoch"] = (
        restarted_primary["epoch"] == previous_primary["epoch"]
    )
    artifact["checks"]["restart rebuilt a usable resident epoch"] = (
        continuation.get("usage", {}).get("generated_tokens", 0) > 0
    )
    artifact["checks"]["restart exposed residency accounting"] = bool(rows)
    output.write_text(json.dumps(artifact, indent=2) + "\n", encoding="utf-8")
    failed = [name for name, passed in artifact["checks"].items() if not passed]
    if failed:
        raise SystemExit("Phase 7 restart checks failed: " + ", ".join(failed))
    print(f"Phase 7 restart check: PASS. Artifact: `{output}`")
    raise SystemExit(0)

model_sha256 = digest(MODEL)
initial = status()
secondary_record = {
    "id": SECONDARY,
    "revision": model_sha256,
    "path": "/models/gemma-4-e2b-it.gguf",
    "sha256": model_sha256,
    "family": "gemma-4-e2b-it",
}
with request("POST", "/native/models", secondary_record) as response:
    secondary = json.load(response)
after_load = status()
primary_response, primary_ms = complete(PRIMARY, "Name one invariant of transactional model residency.")
secondary_response, secondary_ms = complete(SECONDARY, "Name one invariant of transactional context tiering.")

reload_record = dict(secondary_record)
reload_record["revision"] = model_sha256 + "-reload"
with request("POST", "/native/models", reload_record) as response:
    reloaded = json.load(response)
after_reload = status()
reloaded_response, reload_ms = complete(SECONDARY, "Confirm that this immutable epoch is usable.")
with request("DELETE", f"/native/models/{SECONDARY}"):
    pass
after_remove = status()
catalog = models()

load_rows = resident_rows(after_load)
reload_rows = resident_rows(after_reload)
remove_rows = resident_rows(after_remove)
metrics = (after_remove.get("residency") or {}).get("metrics", {})
config = (after_remove.get("residency") or {}).get("config", {})
checks = {
    "secondary model received a monotonic immutable epoch": secondary.get("epoch", 0) > 0,
    "two model epochs became resident": len(load_rows) >= 2,
    "both resident models executed real inference": primary_response.get("usage", {}).get("generated_tokens", 0) > 0
    and secondary_response.get("usage", {}).get("generated_tokens", 0) > 0,
    "reload advanced the model epoch": reloaded.get("epoch", 0) > secondary.get("epoch", 0),
    "reloaded epoch executed real inference": reloaded_response.get("usage", {}).get("generated_tokens", 0) > 0,
    "old epoch retired after reload": all(
        row.get("epoch") != secondary.get("epoch") for row in reload_rows if row.get("id") == SECONDARY
    ),
    "removal reclaimed the secondary resident": all(row.get("id") != SECONDARY for row in remove_rows),
    "device accounting stayed within budget": metrics.get("device_bytes", 0) <= config.get("device_bytes", 0),
    "host accounting stayed within budget": metrics.get("host_bytes", 0) <= config.get("host_bytes", 0),
    "storage accounting stayed within budget": metrics.get("storage_bytes", 0) <= config.get("storage_bytes", 0),
    "lifecycle metrics recorded reuse unload and reload": metrics.get("reuses", 0) > 0 and metrics.get("unloads", 0) > 0 and metrics.get("reloads", 0) > 0,
}
gpu_csv = (RESULT_DIR / "phase7-gpu.csv").read_text(encoding="utf-8").strip()
artifact = {
    "catalog": catalog,
    "phase": "7",
    "model": {"path": str(MODEL), "sha256": model_sha256},
    "build": BUILD,
    "coverage": COVERAGE,
    "gpu": gpu_csv,
    "initial": initial,
    "after_load": after_load,
    "after_reload": after_reload,
    "after_remove": after_remove,
    "epochs": {"secondary": secondary, "reloaded": reloaded},
    "latency_ms": {"primary": primary_ms, "secondary": secondary_ms, "reload": reload_ms},
    "responses": {
        "primary": primary_response,
        "secondary": secondary_response,
        "reloaded": reloaded_response,
    },
    "checks": checks,
}
RESULT_DIR.mkdir(parents=True, exist_ok=True)
output.write_text(json.dumps(artifact, indent=2) + "\n", encoding="utf-8")
failed = [name for name, passed in checks.items() if not passed]
if failed:
    raise SystemExit("Phase 7 checks failed: " + ", ".join(failed))
print(f"Phase 7 live residency check: PASS. Artifact: `{output}`")
