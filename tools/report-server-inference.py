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
TOKEN = "phase6b-report-token"
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


def stream_completion(body):
    started = time.perf_counter()
    first_token_ms = None
    events = []
    with request("POST", "/v1/completions", body) as response:
        for raw_line in response:
            line = raw_line.decode().strip()
            if not line.startswith("data: ") or line == "data: [DONE]":
                continue
            event = json.loads(line[6:])
            events.append(event)
            if event.get("type") == "token" and first_token_ms is None:
                first_token_ms = round((time.perf_counter() - started) * 1000, 2)
    return events, first_token_ms, round((time.perf_counter() - started) * 1000, 2)

def prefill_rates(usage):
    prefill = usage.get("prefill", {})
    total = prefill.get("total_tokens", 0)
    cached = prefill.get("cached_tokens", 0)
    uncached = prefill.get("uncached_tokens", 0)

    def tokens_per_second(tokens, duration_ns):
        if not tokens or not duration_ns:
            return None
        return round(tokens * 1_000_000_000 / duration_ns, 3)

    return {
        **prefill,
        "cache_fraction": round(cached / total, 6) if total else None,
        "cached_tokens_per_second": tokens_per_second(
            cached, prefill.get("mapping_activation_ns", 0)
        ),
        "uncached_tokens_per_second": tokens_per_second(
            uncached, prefill.get("uncached_prefill_ns", 0)
        ),
        "effective_tokens_per_second": tokens_per_second(
            total, prefill.get("total_ns", 0)
        ),
    }


def milliseconds(nanoseconds):
    return nanoseconds / 1_000_000


def display_rate(rate):
    return "n/a" if rate is None else f"{rate:.3f}"


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

events, first_token_ms, wall_ms = stream_completion(
    {"model": model["id"], "prompt": PROMPT, "max_tokens": 72, "stream": True}
)
token_events = [event for event in events if event.get("type") == "token"]
finished_events = [event for event in events if event.get("type") == "finished"]
response_text = "".join(event["token"] for event in token_events)
usage = finished_events[-1]["usage"] if finished_events else {}
generated = usage.get("generated_tokens", len(token_events))
server_ms = usage.get("latency_ms", 0)

continuation_events, continuation_first_ms, continuation_wall_ms = stream_completion(
    {
        "model": model["id"],
        "prompt": "Continue with one short sentence.",
        "max_tokens": 16,
        "stream": True,
        "context_id": usage.get("context_id"),
    }
)
continuation_tokens = [
    event for event in continuation_events if event.get("type") == "token"
]
continuation_finished_events = [
    event for event in continuation_events if event.get("type") == "finished"
]
continuation_usage = (
    continuation_finished_events[-1]["usage"] if continuation_finished_events else {}
)
continuation_text = "".join(event["token"] for event in continuation_tokens)
initial_prefill = prefill_rates(usage)
continuation_prefill = prefill_rates(continuation_usage)

artifact = {
    "model": registered,
    "gpu": (RESULT_DIR / "phase6b-gpu.csv").read_text(encoding="utf-8").strip(),
    "prompt": PROMPT,
    "response": response_text,
    "continuation": {
        "prompt": "Continue with one short sentence.",
        "response": continuation_text,
        "usage": continuation_usage,
        "client_wall_ms": continuation_wall_ms,
        "first_streamed_token_ms": continuation_first_ms,
        "token_event_count": len(continuation_tokens),
    },
    "timing": {
        "client_wall_ms": wall_ms,
        "first_streamed_token_ms": first_token_ms,
        "server_end_to_end_ms": server_ms,
        "generated_tokens_per_second": round(generated / (server_ms / 1000), 3)
        if server_ms
        else None,
    },
    "prefill": {
        "initial": initial_prefill,
        "continuation": continuation_prefill,
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
            route in openapi["paths"]
            for route in ("/v1/completions", "/native/contexts")
        ),
        "visible stream tokens are bounded by sampled tokens": 0
        < len(token_events)
        <= generated,
        "stream completed with usage": bool(finished_events)
        and bool(events[-1].get("usage")),
        "response contains generated text": bool(response_text.strip()),
        "server recorded timing and model identity": server_ms > 0
        and usage.get("model_revision") == model["revision"],
        "opaque context survives into continuation": bool(usage.get("context_id"))
        and continuation_usage.get("context_id") == usage.get("context_id"),
        "continuation reuses a complete mapped block": continuation_usage.get(
            "cached_tokens", 0
        )
        >= 32,
        "continuation reports generated text and usage": bool(continuation_text.strip())
        and 0 < len(continuation_tokens)
        <= continuation_usage.get("generated_tokens", -1),
        "initial request records uncached prefill timing": initial_prefill.get(
            "uncached_tokens", 0
        )
        > 0
        and initial_prefill.get("uncached_prefill_ns", 0) > 0
        and initial_prefill.get("uncached_tokens_per_second") is not None,
        "mapped continuation records cache activation timing": continuation_prefill.get(
            "cached_tokens", 0
        )
        >= 32
        and continuation_prefill.get("mapping_activation_ns", 0) > 0
        and continuation_prefill.get("cached_tokens_per_second") is not None,
        "overall prompt timing covers measured phases": all(
            prefill.get("total_ns", 0)
            >= max(
                prefill.get("tokenization_ns", 0),
                prefill.get("prefix_lookup_ns", 0),
                prefill.get("mapping_activation_ns", 0),
                prefill.get("uncached_prefill_ns", 0),
            )
            for prefill in (initial_prefill, continuation_prefill)
        ),
    },
}
RESULT_DIR.mkdir(parents=True, exist_ok=True)
output = RESULT_DIR / "phase6b-server.json"
output.write_text(json.dumps(artifact, indent=2) + "\n", encoding="utf-8")

print("# Cusco Phase 6B incremental generation server report")
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
print("\n## Prefill work\n")
print(
    "| Case | Total tokens | Cached | Uncached | Cache fraction | Tokenization | Lookup | "
    "Mapping activation | Uncached prefill | Prompt total | Cached tok/s | Uncached tok/s | "
    "Effective tok/s |"
)
print("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
for name, prefill in (
    ("Initial", initial_prefill),
    ("Continuation", continuation_prefill),
):
    print(
        f"| {name} | {prefill['total_tokens']} | {prefill['cached_tokens']} | "
        f"{prefill['uncached_tokens']} | {prefill['cache_fraction']:.3f} | "
        f"{milliseconds(prefill['tokenization_ns']):.3f} ms | "
        f"{milliseconds(prefill['prefix_lookup_ns']):.3f} ms | "
        f"{milliseconds(prefill['mapping_activation_ns']):.3f} ms | "
        f"{milliseconds(prefill['uncached_prefill_ns']):.3f} ms | "
        f"{milliseconds(prefill['total_ns']):.3f} ms | "
        f"{display_rate(prefill['cached_tokens_per_second'])} | "
        f"{display_rate(prefill['uncached_tokens_per_second'])} | "
        f"{display_rate(prefill['effective_tokens_per_second'])} |"
    )
print("\n## Assertions")
failed = []
for name, passed in artifact["checks"].items():
    print(f"- {'PASS' if passed else 'FAIL'}: {name}")
    if not passed:
        failed.append(name)
if failed:
    raise SystemExit("server report failed: " + ", ".join(failed))
print(
    "\nResult: PASS — authenticated HTTP streaming and mapped context reuse completed "
    f"through one persistent llama.cpp executor. Artifact: `{output}`"
)
