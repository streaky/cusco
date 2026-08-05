#!/usr/bin/env python3
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
TOKEN = "phase4-report-token"
PROMPT = (
    "A distributed inference system preserves model execution checkpoints across GPU and host memory. "
    "In exactly three concise sentences, explain why transactional restore, cancellation safety, and "
    "deterministic capacity accounting matter for correctness."
)


def request(method, path, body=None, authenticated=True):
    headers = {"content-type": "application/json"}
    if authenticated:
        headers["authorization"] = f"Bearer {TOKEN}"
    encoded = None if body is None else json.dumps(body).encode()
    return urllib.request.urlopen(
        urllib.request.Request(BASE_URL + path, data=encoded, headers=headers, method=method),
        timeout=600,
    )


def wait_for_server():
    deadline = time.monotonic() + 180
    while time.monotonic() < deadline:
        try:
            with request("GET", "/openapi.json", authenticated=False):
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


wait_for_server()
try:
    request("GET", "/native/models", authenticated=False)
    unauthorized = False
except urllib.error.HTTPError as error:
    unauthorized = error.code == 401

model_sha256 = digest(MODEL)
model = {
    "id": "gemma-phase4-report",
    "revision": "0314792d7f1f7e229411f620751375812bb9faf2",
    "path": "/models/gemma-4-e2b-it.gguf",
    "sha256": model_sha256,
}
with request("POST", "/native/models", model) as response:
    registered = json.load(response)
with request("GET", "/native/models") as response:
    listed = json.load(response)["data"]
with request("GET", "/openapi.json", authenticated=False) as response:
    openapi = json.load(response)

started = time.perf_counter()
first_token_ms = None
events = []
with request(
    "POST",
    "/v1/completions",
    {"model": model["id"], "prompt": PROMPT, "max_tokens": 72, "stream": True},
) as response:
    for raw_line in response:
        line = raw_line.decode().strip()
        if not line.startswith("data: ") or line == "data: [DONE]":
            continue
        event = json.loads(line[6:])
        events.append(event)
        if event.get("type") == "token" and first_token_ms is None:
            first_token_ms = round((time.perf_counter() - started) * 1000, 2)
wall_ms = round((time.perf_counter() - started) * 1000, 2)

token_events = [event for event in events if event.get("type") == "token"]
usage_events = [event for event in events if event.get("type") == "usage"]
response_text = "".join(event["token"] for event in token_events)
usage = usage_events[-1]["usage"] if usage_events else {}
generated = usage.get("generated_tokens", len(token_events))
server_ms = usage.get("latency_ms", 0)
artifact = {
    "model": registered,
    "gpu": (RESULT_DIR / "phase4-gpu.csv").read_text(encoding="utf-8").strip(),
    "prompt": PROMPT,
    "response": response_text,
    "timing": {
        "client_wall_ms": wall_ms,
        "first_streamed_token_ms": first_token_ms,
        "server_end_to_end_ms": server_ms,
        "generated_tokens_per_second": round(generated / (server_ms / 1000), 3) if server_ms else None,
    },
    "usage": usage,
    "stream": {
        "event_count": len(events),
        "token_event_count": len(token_events),
        "event_kinds": [event.get("type") for event in events],
    },
    "checks": {
        "anonymous admin request rejected": unauthorized,
        "registered model listed": any(row["id"] == model["id"] for row in listed),
        "checked OpenAPI includes completion and native context routes": all(
            route in openapi["paths"] for route in ("/v1/completions", "/native/contexts")
        ),
        "stream emitted one event per generated token": len(token_events) == generated and generated > 0,
        "stream completed with usage": bool(usage_events) and events[-1].get("type") == "finished",
        "response contains generated text": bool(response_text.strip()),
        "server recorded timing and model identity": server_ms > 0 and usage.get("model_revision") == model["revision"],
    },
}
RESULT_DIR.mkdir(parents=True, exist_ok=True)
output = RESULT_DIR / "phase4-server.json"
output.write_text(json.dumps(artifact, indent=2) + "\n", encoding="utf-8")

print("# Cusco Phase 4 real-model server report")
print(f"\nModel: `{model['id']}@{model['revision']}`")
print(f"Digest: `{model_sha256}`")
print(f"GPU: `{artifact['gpu']}`")
print("\n## Prompt\n")
print(PROMPT)
print("\n## Response\n")
print(response_text)
print("\n## Timing and usage\n")
print("| Client wall | First streamed token | Server end-to-end | Generated tokens | Tokens/s |")
print("|---:|---:|---:|---:|---:|")
rate = artifact["timing"]["generated_tokens_per_second"]
print(f"| {wall_ms:.2f} ms | {first_token_ms:.2f} ms | {server_ms} ms | {generated} | {rate:.3f} |")
print("\n## Assertions")
failed = []
for name, passed in artifact["checks"].items():
    print(f"- {'PASS' if passed else 'FAIL'}: {name}")
    if not passed:
        failed.append(name)
if failed:
    raise SystemExit("server report failed: " + ", ".join(failed))
print(f"\nResult: PASS — authenticated HTTP streaming completed through the real llama.cpp executor. Artifact: `{output}`")
