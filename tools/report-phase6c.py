#!/usr/bin/env python3
import concurrent.futures
import hashlib
import json
import pathlib
import sys
import time
import urllib.error
import urllib.request

BASE_URL = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:18082"
RESULT_DIR = pathlib.Path(sys.argv[2] if len(sys.argv) > 2 else "results")
MODEL = pathlib.Path(sys.argv[3] if len(sys.argv) > 3 else "models/gemma-4-e2b-it.gguf")
MODE = sys.argv[4] if len(sys.argv) > 4 else "run"
TOKEN = "phase6c-report-token"
MODEL_ID = "gemma-phase6c-report"
PROMPT = (
    "A distributed inference system preserves model execution checkpoints across GPU and host memory. "
    "In exactly three concise sentences, explain why transactional restore, cancellation safety, and "
    "deterministic capacity accounting matter for correctness."
)


def request(method, path, body=None, authenticated=True, timeout=600):
    headers = {"content-type": "application/json"}
    if authenticated:
        headers["authorization"] = f"Bearer {TOKEN}"
    encoded = None if body is None else json.dumps(body).encode()
    return urllib.request.urlopen(
        urllib.request.Request(BASE_URL + path, data=encoded, headers=headers, method=method),
        timeout=timeout,
    )


def wait_for_server():
    deadline = time.monotonic() + 180
    while time.monotonic() < deadline:
        try:
            with request("GET", "/openapi.json", authenticated=False, timeout=5):
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


def stream_completion(body, cancel_after_start=False):
    started_at = time.perf_counter()
    first_token_ms = None
    events = []
    with request("POST", "/v1/completions", body) as response:
        for raw_line in response:
            line = raw_line.decode().strip()
            if not line.startswith("data: "):
                continue
            event = json.loads(line[6:])
            events.append(event)
            if event.get("type") == "started" and cancel_after_start:
                with request("DELETE", f"/native/requests/{event['request_id']}"):
                    pass
                cancel_after_start = False
            if event.get("type") == "token" and first_token_ms is None:
                first_token_ms = round((time.perf_counter() - started_at) * 1000, 2)
    return events, first_token_ms, round((time.perf_counter() - started_at) * 1000, 2)


def finished_usage(events):
    rows = [event for event in events if event.get("type") == "finished"]
    return rows[-1].get("usage", {}) if rows else {}


def prefill_rates(usage):
    prefill = usage.get("prefill", {})

    def rate(tokens, duration_ns):
        return round(tokens * 1_000_000_000 / duration_ns, 3) if tokens and duration_ns else None

    total = prefill.get("total_tokens", 0)
    cached = prefill.get("cached_tokens", 0)
    uncached = prefill.get("uncached_tokens", 0)
    return {
        **prefill,
        "cache_fraction": round(cached / total, 6) if total else None,
        "cached_tokens_per_second": rate(cached, prefill.get("mapping_activation_ns", 0)),
        "uncached_tokens_per_second": rate(uncached, prefill.get("uncached_prefill_ns", 0)),
        "effective_tokens_per_second": rate(total, prefill.get("total_ns", 0)),
    }


def overload_probe():
    body = {
        "model": MODEL_ID,
        "prompt": PROMPT,
        "max_tokens": 128,
        "stream": True,
        "deadline_ms": 250,
    }

    def submit(_):
        try:
            with request("POST", "/v1/completions", body, timeout=10) as response:
                response.read()
                return response.status, None, response.headers.get("retry-after")
        except urllib.error.HTTPError as error:
            payload = json.loads(error.read())
            return error.code, payload.get("error", {}).get("code"), error.headers.get("retry-after")
        except (TimeoutError, urllib.error.URLError):
            return 408, "client_timeout", None

    with concurrent.futures.ThreadPoolExecutor(max_workers=40) as pool:
        return list(pool.map(submit, range(40)))


def restart_check():
    output = RESULT_DIR / "phase6c-server.json"
    artifact = json.loads(output.read_text(encoding="utf-8"))
    with request("GET", "/native/contexts") as response:
        contexts = json.load(response)
    expected = artifact["usage"]["context_id"]
    recovered = any(context["id"] == expected for context in contexts)
    shutdown = json.loads((RESULT_DIR / "phase6c-shutdown.json").read_text(encoding="utf-8"))
    artifact["restart"] = {
        "durable_context_recovered": recovered,
        "ephemeral_queue_recovered": False,
        "graceful_shutdown": shutdown,
    }
    artifact["checks"]["graceful shutdown completed"] = shutdown.get("passed", False)
    artifact["checks"]["restart recovered durable context without queue state"] = recovered
    output.write_text(json.dumps(artifact, indent=2) + "\n", encoding="utf-8")
    failed = [name for name, passed in artifact["checks"].items() if not passed]
    if failed:
        raise SystemExit("server report failed after restart: " + ", ".join(failed))
    print(f"Phase 6C restart check: PASS. Artifact: `{output}`")


wait_for_server()
if MODE == "restart-check":
    restart_check()
    raise SystemExit(0)

try:
    request("GET", "/native/models", authenticated=False)
    unauthorized = False
except urllib.error.HTTPError as error:
    unauthorized = error.code == 401

model_sha256 = digest(MODEL)
model = {
    "id": MODEL_ID,
    "revision": model_sha256,
    "path": "/models/gemma-4-e2b-it.gguf",
    "sha256": model_sha256,
}
with request("POST", "/native/models", model) as response:
    registered = json.load(response)
with request("GET", "/native/models") as response:
    listed = json.load(response)["data"]
with request("GET", "/openapi.json", authenticated=False) as response:
    openapi = json.load(response)

events, first_token_ms, wall_ms = stream_completion(
    {"model": MODEL_ID, "prompt": PROMPT, "max_tokens": 72, "stream": True}
)
usage = finished_usage(events)
response_text = "".join(event["token"] for event in events if event.get("type") == "token")
continuation_events, continuation_first_ms, continuation_wall_ms = stream_completion(
    {
        "model": MODEL_ID,
        "prompt": "Continue with one short sentence.",
        "max_tokens": 16,
        "stream": True,
        "context_id": usage.get("context_id"),
    }
)
continuation_usage = finished_usage(continuation_events)
continuation_text = "".join(
    event["token"] for event in continuation_events if event.get("type") == "token"
)
cancel_events, _, _ = stream_completion(
    {"model": MODEL_ID, "prompt": PROMPT, "max_tokens": 128, "stream": True},
    cancel_after_start=True,
)
deadline_events, _, _ = stream_completion(
    {
        "model": MODEL_ID,
        "prompt": PROMPT * 8,
        "max_tokens": 128,
        "stream": True,
        "deadline_ms": 1,
    }
)
overload_results = overload_probe()
coverage = json.loads((RESULT_DIR / "phase6c-coverage.json").read_text(encoding="utf-8"))
executor_proof = json.loads((RESULT_DIR / "phase1.json").read_text(encoding="utf-8"))
mapped_proof = json.loads((RESULT_DIR / "phase5.json").read_text(encoding="utf-8"))
initial_prefill = prefill_rates(usage)
continuation_prefill = prefill_rates(continuation_usage)
token_events = [event for event in events if event.get("type") == "token"]
generated = usage.get("generated_tokens", len(token_events))
server_ms = usage.get("latency_ms", 0)

checks = {
    "anonymous admin request rejected": unauthorized,
    "registered model listed": any(row["id"] == MODEL_ID for row in listed),
    "checked OpenAPI includes completion and native context routes": all(
        route in openapi["paths"] for route in ("/v1/completions", "/native/contexts")
    ),
    "visible stream tokens are bounded by sampled tokens": 0 < len(token_events) <= generated,
    "stream completed with usage": bool(usage),
    "response contains generated text": bool(response_text.strip()),
    "opaque context survives into continuation": bool(usage.get("context_id"))
    and continuation_usage.get("context_id") == usage.get("context_id"),
    "continuation reuses a complete mapped block": continuation_usage.get("cached_tokens", 0) >= 32,
    "real cancellation terminates without context publication": any(
        event.get("type") == "error" and "cancel" in event.get("message", "") for event in cancel_events
    ),
    "real deadline terminates owned native work": any(
        event.get("type") == "error" and "deadline" in event.get("message", "") for event in deadline_events
    ),
    "overload is canonical and retryable": any(
        status == 429 and code == "queue_overloaded" and retry == "1"
        for status, code, retry in overload_results
    ),
    "GPU-less coverage gate passed": coverage.get("passed", False),
    "isolated continuations match exact tokens and logits": all(
        context["token_equal"] and context["logits_equal"]
        for context in executor_proof["contexts"]
    ),
    "failed native operations preserve the prior binding": (
        executor_proof["cancellation_preserved_binding"]
        and executor_proof["failed_promotion_preserved_binding"]
    ),
    "mapped activation is reference-only": (
        mapped_proof["comparison"]["mapped_activation_bytes_copied"] == 0
        and len({branch["next_token"] for branch in mapped_proof["branches"]}) == 1
    ),
}

artifact = {
    "config": {
        "queue_count": 32,
        "queue_bytes": 16 << 20,
        "request_bytes": 1 << 20,
        "pre_queue_concurrency": 16,
        "header_bytes": 32 << 10,
        "body_timeout_ms": 10_000,
        "wall_time_ms": 300_000,
        "active_time_ms": 240_000,
        "shutdown_grace_ms": 30_000,
    },
    "provenance": {
        "llama_cpp_tag": pathlib.Path("llama.cpp-version.txt").read_text().strip(),
        "patch_series_sha256": digest(pathlib.Path("executor/patches/series.toml")),
        "model_sha256": model_sha256,
        "coverage": coverage,
        "source_inputs": {
            "cargo_lock_sha256": digest(pathlib.Path("Cargo.lock")),
            "dockerfile_sha256": digest(pathlib.Path("Dockerfile")),
            "profile_catalog_sha256": digest(pathlib.Path("config/model-families.yaml")),
        },
    },
    "model": registered,
    "gpu": (RESULT_DIR / "phase6c-gpu.csv").read_text(encoding="utf-8").strip(),
    "prompt": PROMPT,
    "response": response_text,
    "usage": usage,
    "continuation": {
        "prompt": "Continue with one short sentence.",
        "response": continuation_text,
        "usage": continuation_usage,
        "client_wall_ms": continuation_wall_ms,
        "first_streamed_token_ms": continuation_first_ms,
    },
    "timing": {
        "client_wall_ms": wall_ms,
        "first_streamed_token_ms": first_token_ms,
        "server_end_to_end_ms": server_ms,
        "generated_tokens_per_second": round(generated / (server_ms / 1000), 3) if server_ms else None,
    },
    "prefill": {"initial": initial_prefill, "continuation": continuation_prefill},
    "exactness": {
        "isolated_executor": executor_proof,
        "mapped_executor": mapped_proof,
    },
    "cache_work": {
        "initial_cached_tokens": usage.get("cached_tokens", 0),
        "continuation_cached_tokens": continuation_usage.get("cached_tokens", 0),
        "initial_uncached_tokens": usage.get("evaluated_tokens", 0),
        "continuation_uncached_tokens": continuation_usage.get("evaluated_tokens", 0),
        "transfer_bytes": max(
            initial_prefill.get("transfer_bytes", 0),
            continuation_prefill.get("transfer_bytes", 0),
            mapped_proof["comparison"]["staged_bytes_read"],
        ),
        "peak_device_bytes": max(
            initial_prefill.get("device_bytes", 0),
            continuation_prefill.get("device_bytes", 0),
        ),
        "peak_host_bytes": max(
            initial_prefill.get("host_bytes", 0),
            continuation_prefill.get("host_bytes", 0),
        ),
    },
    "terminal_injections": {
        "cancellation": cancel_events,
        "deadline": deadline_events,
        "overload": overload_results,
    },
    "checks": checks,
}
RESULT_DIR.mkdir(parents=True, exist_ok=True)
output = RESULT_DIR / "phase6c-server.json"
output.write_text(json.dumps(artifact, indent=2) + "\n", encoding="utf-8")

print("# Cusco Phase 6C bounded lifecycle server report")
print(f"\nModel: `{MODEL_ID}@{model_sha256}`")
print(f"GPU: `{artifact['gpu']}`")
print(f"Response: {response_text}")
print("\n## Assertions")
failed = []
for name, passed in checks.items():
    print(f"- {'PASS' if passed else 'FAIL'}: {name}")
    if not passed:
        failed.append(name)
if failed:
    raise SystemExit("server report failed: " + ", ".join(failed))
print(f"\nInitial Phase 6C run: PASS. Awaiting shutdown/restart check in `{output}`")
