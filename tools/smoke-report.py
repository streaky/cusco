#!/usr/bin/env python3
from pathlib import Path
import json
import os
import statistics
import sys
import time
import urllib.error
import urllib.request
from openai import APIStatusError, OpenAI
BASE = os.environ.get("CUSCO_SMOKE_BASE_URL", "http://127.0.0.1:8080").rstrip("/")
TOKEN = os.environ.get("CUSCO_SMOKE_TOKEN", "smoke-report-token")
OUT = Path(os.environ.get("CUSCO_SMOKE_OUTPUT", "/results/smoke-report.json"))
MODEL = os.environ.get(
    "CUSCO_SMOKE_MODEL",
    "hf://unsloth/gemma-4-E2B-it-GGUF/gemma-4-E2B-it-Q3_K_M.gguf",
)
SEMANTIC_WORKLOAD = Path(
    os.environ.get(
        "CUSCO_SEMANTIC_WORKLOAD",
        "/work/config/phase10-semantic-workload.json",
    )
)
OPENAI = OpenAI(
    api_key=TOKEN,
    base_url=f"{BASE}/openai/v1",
    max_retries=0,
    timeout=120.0,
)


def sdk_body(response):
    parsed = response.parse()
    return parsed.model_dump(mode="json") if hasattr(parsed, "model_dump") else parsed


def openai_call(path, payload):
    payload = dict(payload or {})
    if path == "/openai/v1/models":
        response = OPENAI.models.with_raw_response.list()
    elif path == "/openai/v1/completions":
        extra_body = {
            key: payload.pop(key)
            for key in ("context_id", "compaction")
            if key in payload
        }
        response = OPENAI.completions.with_raw_response.create(
            **payload,
            extra_body=extra_body or None,
        )
    elif path == "/openai/v1/chat/completions":
        extra_body = {
            key: payload.pop(key)
            for key in ("context_id", "compaction")
            if key in payload
        }
        response = OPENAI.chat.completions.with_raw_response.create(
            **payload,
            extra_body=extra_body or None,
        )
    elif path == "/openai/v1/responses":
        extra_body = {
            key: payload.pop(key)
            for key in ("seed", "context_id", "compaction")
            if key in payload
        }
        response = OPENAI.responses.with_raw_response.create(
            **payload,
            extra_body=extra_body or None,
        )
    else:
        raise ValueError(f"unsupported OpenAI SDK path: {path}")
    return response.status_code, response.headers.get("content-type", ""), sdk_body(response)


def openai_stream_call(path, payload):
    payload = dict(payload)
    payload.pop("stream", None)
    if path == "/openai/v1/chat/completions":
        stream = OPENAI.chat.completions.create(**payload, stream=True)
    elif path == "/openai/v1/responses":
        extra_body = {"seed": payload.pop("seed")} if "seed" in payload else None
        stream = OPENAI.responses.create(
            **payload,
            stream=True,
            extra_body=extra_body,
        )
    else:
        raise ValueError(f"unsupported OpenAI SDK stream path: {path}")
    return [event.model_dump(mode="json") for event in stream]


def call(method, path, payload=None):
    if path != "/openai/v1/openapi.json" and path.startswith("/openai/v1/"):
        started = time.perf_counter_ns()
        try:
            status, content_type, body = openai_call(path, payload)
        except APIStatusError as exc:
            status = exc.status_code
            content_type = exc.response.headers.get("content-type", "")
            try:
                body = exc.response.json()
            except ValueError:
                body = exc.response.text
        except Exception as exc:
            elapsed = (time.perf_counter_ns() - started) / 1_000_000
            return 0, "application/json", {"error": str(exc)}, elapsed
        elapsed = (time.perf_counter_ns() - started) / 1_000_000
        return status, content_type, body, elapsed
    data = None if payload is None else json.dumps(payload, separators=(",", ":")).encode()
    headers = {"Authorization": f"Bearer {TOKEN}", "Accept": "application/json"}
    if data is not None:
        headers["Content-Type"] = "application/json"

    start = time.perf_counter_ns()
    request = urllib.request.Request(BASE + path, data=data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            status = response.status
            content_type = response.headers.get("content-type", "")
            raw = response.read()
    except urllib.error.HTTPError as exc:
        status = exc.code
        content_type = exc.headers.get("content-type", "")
        raw = exc.read()
    except urllib.error.URLError as exc:
        elapsed = (time.perf_counter_ns() - start) / 1_000_000
        return 0, "application/json", {"error": str(exc)}, elapsed
    except Exception as exc:
        elapsed = (time.perf_counter_ns() - start) / 1_000_000
        return 0, "application/json", {"error": str(exc)}, elapsed

    elapsed = (time.perf_counter_ns() - start) / 1_000_000
    try:
        body = json.loads(raw) if raw else None
    except json.JSONDecodeError:
        body = raw.decode("utf-8", "replace")
    return status, content_type, body, elapsed

def stream_call(path, payload, accept):
    assert accept == "text/event-stream"
    started = time.perf_counter_ns()
    events = openai_stream_call(path, payload)
    elapsed = (time.perf_counter_ns() - started) / 1_000_000
    return 200, "text/event-stream", events, elapsed


def stable_stream_ids(events):
    keys = ("correlation_id", "inference_id", "execution_session_id")
    observed = {key: set() for key in keys}
    request_ids = set()
    for event in events:
        if isinstance(event, dict):
            if event.get("id"):
                request_ids.add(event["id"])
            cusco = event.get("cusco")
            if isinstance(cusco, dict):
                for key in keys:
                    if cusco.get(key):
                        observed[key].add(cusco[key])
            response = event.get("response")
            if isinstance(response, dict):
                if response.get("id"):
                    request_ids.add(response["id"])
                metadata = response.get("metadata")
                if isinstance(metadata, dict):
                    for key in keys:
                        if metadata.get(key):
                            observed[key].add(metadata[key])
    assert events, "stream emitted no events"
    assert len(request_ids) == 1, f"stream request IDs changed: {request_ids}"
    for key, values in observed.items():
        assert len(values) == 1, f"stream {key} changed or was missing: {values}"
    return {"request_id": next(iter(request_ids)), **{key: next(iter(values)) for key, values in observed.items()}}


def status_ok(status):
    assert status == 200, f"expected HTTP 200, got {status}"


def has(status, body, key):
    status_ok(status)
    assert isinstance(body, dict) and key in body, f"missing {key}"


def list_field(status, body, key):
    has(status, body, key)
    assert isinstance(body[key], list), f"{key} is not a list"


def imported(status, body):
    has(status, body, "tokens")
    assert body["tokens"] == ["smoke", "context"], "wrong imported tokens"


def merge_usage_fields(stats, body):
    if not isinstance(body, dict):
        return

    usage = body.get("usage")
    if isinstance(usage, dict):
        stats["input_tokens"] += int(usage.get("prompt_tokens", usage.get("input_tokens", 0)) or 0)
        stats["output_tokens"] += int(usage.get("completion_tokens", usage.get("output_tokens", 0)) or 0)

    cusco_usage = body.get("cusco")
    if isinstance(cusco_usage, dict):
        exec_usage = cusco_usage.get("usage")
        if isinstance(exec_usage, dict):
            stats["cached_tokens"] += int(exec_usage.get("cached_tokens", 0) or 0)
            stats["evaluated_tokens"] += int(exec_usage.get("evaluated_tokens", 0) or 0)
            prefill = exec_usage.get("prefill")
            if isinstance(prefill, dict):
                stats["prefill_cached_tokens"] += int(prefill.get("cached_tokens", 0) or 0)
                stats["prefill_uncached_tokens"] += int(prefill.get("uncached_tokens", 0) or 0)


def compact_cusco_usage(cusco):
    if not isinstance(cusco, dict):
        return None
    usage = cusco.get("usage")
    if not isinstance(usage, dict):
        return None
    return {
        "input_tokens": usage.get("input_tokens", 0),
        "generated_tokens": usage.get("generated_tokens", usage.get("output_tokens", 0)),
        "evaluated_tokens": usage.get("evaluated_tokens", 0),
        "cached_tokens": usage.get("cached_tokens", 0),
        "prefill": usage.get("prefill", {}),
    }


def compact_openai_usage(usage):
    if not isinstance(usage, dict):
        return None
    return {
        "prompt_tokens": usage.get("prompt_tokens", usage.get("input_tokens", 0)),
        "completion_tokens": usage.get("completion_tokens", usage.get("output_tokens", 0)),
        "total_tokens": usage.get("total_tokens", 0),
    }
def response_text(body):
    if not isinstance(body, dict):
        return ""
    output = body.get("output", [])
    text = "".join(
        part.get("text", "")
        for item in output
        if isinstance(item, dict)
        for part in item.get("content", [])
        if isinstance(part, dict) and part.get("type") == "output_text"
    )
    if text:
        return text
    return "".join(
        choice.get("text", "")
        or choice.get("message", {}).get("content", "")
        for choice in body.get("choices", [])
        if isinstance(choice, dict)
    )


def normalized_text(text):
    return " ".join(text.strip().split())


def semantic_request(case, context_id, compact):
    payload = {
        "model": MODEL,
        "messages": case["messages"],
        "max_tokens": 16,
        "temperature": 0,
        "seed": 10,
        "context_id": context_id,
    }
    if compact:
        payload["compaction"] = {
            "strategy_preferences": ["window_tail"],
            "target_tokens": case["target_tokens"],
        }
    return payload


def run_semantic_comparisons(stats, latencies):
    workload = json.loads(SEMANTIC_WORKLOAD.read_text())
    comparisons = []
    for case in workload["cases"]:
        variants = {}
        passed = True
        error = None
        for variant, compact in (("control", False), ("compacted", True)):
            import_status, _, import_body, import_elapsed_ms = call(
                "POST",
                "/cusco/v1/contexts/import",
                {"tokens": case.get("tokens", [])},
            )
            latencies.append(import_elapsed_ms)
            stats["requests"] += 1
            context_id = (
                import_body.get("id")
                if import_status == 200 and isinstance(import_body, dict)
                else None
            )
            for seed_prompt in case["seed_prompts"]:
                seed_status, _, seed_body, seed_elapsed_ms = call(
                    "POST",
                    "/openai/v1/chat/completions",
                    {
                        "model": MODEL,
                        "messages": [{"role": "user", "content": seed_prompt}],
                        "context_id": context_id,
                        "max_tokens": 1,
                        "temperature": 0,
                        "seed": 10,
                    },
                )
                latencies.append(seed_elapsed_ms)
                stats["requests"] += 1
                merge_usage_fields(stats, seed_body)
                if seed_status != 200:
                    context_id = None
                    break
            status, _, body, elapsed_ms = call(
                "POST",
                "/openai/v1/chat/completions",
                semantic_request(case, context_id, compact),
            )
            latencies.append(elapsed_ms)
            stats["requests"] += 1
            merge_usage_fields(stats, body)
            text = response_text(body)
            normalized = normalized_text(text)
            fragments = case.get("required_fragments", [])
            variant_passed = (
                status == 200
                and bool(normalized)
                and all(
                    fragment.casefold() in normalized.casefold()
                    for fragment in fragments
                )
            )
            if "exact_normalized" in case:
                variant_passed = (
                    variant_passed and normalized == case["exact_normalized"]
                )
            compaction_result = (
                body.get("cusco", {}).get("compaction_result")
                if isinstance(body, dict)
                else None
            )
            if compact:
                variant_passed = (
                    variant_passed
                    and isinstance(compaction_result, dict)
                    and compaction_result.get("success") is True
                    and compaction_result.get("selected_strategy_id") == "window_tail:v1"
                )
            variants[variant] = {
                "status": status,
                "latency_ms": round(elapsed_ms, 3),
                "text": text,
                "normalized_text": normalized,
                "passed": variant_passed,
                "openai_usage": compact_openai_usage(
                    body.get("usage") if isinstance(body, dict) else None
                ),
                "cusco_usage": compact_cusco_usage(
                    body.get("cusco") if isinstance(body, dict) else None
                ),
                "compaction_result": compaction_result,
                "response_error": (
                    body.get("error") if isinstance(body, dict) else None
                ),
            }
            if context_id is not None:
                call("DELETE", f"/cusco/v1/contexts/{context_id}")
            if not variant_passed:
                passed = False
                error = f"{variant} failed observable semantic assertions"
        if passed and (
            variants["control"]["normalized_text"]
            != variants["compacted"]["normalized_text"]
        ):
            passed = False
            error = "compacted output diverged from the uncompacted control"
        stats["passed" if passed else "failed"] += 1
        comparisons.append(
            {
                "name": case["name"],
                "required_fragments": case.get("required_fragments", []),
                "exact_normalized": case.get("exact_normalized"),
                "passed": passed,
                "error": error,
                "variants": variants,
            }
        )
    return {
        "schema_version": workload["schema_version"],
        "workload_version": workload["workload_version"],
        "workload": str(SEMANTIC_WORKLOAD),
        "comparisons": comparisons,
        "passed": all(comparison["passed"] for comparison in comparisons),
    }




def run():
    cases = [
        (
            "model_pull",
            "POST",
            "/cusco/v1/api/pull",
            {"model": MODEL, "stream": False},
            lambda status, body: has(status, body, "status"),
        ),
        ("openapi", "GET", "/openai/v1/openapi.json", None, lambda status, body: has(status, body, "paths")),
        ("models", "GET", "/openai/v1/models", None, lambda status, body: list_field(status, body, "data")),
        (
            "completion",
            "POST",
            "/openai/v1/completions",
            {
                "model": MODEL,
                "prompt": "Reply with exactly: smoke-ok",
                "max_tokens": 4,
                "temperature": 0,
                "seed": 10,
            },
            lambda status, body: list_field(status, body, "choices"),
        ),
        (
            "chat",
            "POST",
            "/openai/v1/chat/completions",
            {
                "model": MODEL,
                "messages": [{"role": "user", "content": "Reply with exactly: smoke-ok"}],
                "max_tokens": 4,
                "temperature": 0,
                "seed": 10,
            },
            lambda status, body: list_field(status, body, "choices"),
        ),
        (
            "responses",
            "POST",
            "/openai/v1/responses",
            {
                "model": MODEL,
                "input": "Reply with exactly: smoke-ok",
                "max_output_tokens": 4,
                "temperature": 0,
                "seed": 10,
            },
            lambda status, body: has(status, body, "output"),
        ),
        ("context_create", "POST", "/cusco/v1/contexts", {}, lambda status, body: has(status, body, "id")),
        ("context_import", "POST", "/cusco/v1/contexts/import", {"tokens": ["smoke", "context"]}, imported),
        (
            "compaction_strategies",
            "GET",
            "/cusco/v1/compaction/strategies",
            None,
            lambda status, body: list_field(status, body, "strategies"),
        ),
        ("status", "GET", "/cusco/v1/status", None, lambda status, body: has(status, body, "residency")),
    ]

    results = []
    latencies = []
    context_id = None
    created_context_ids = set()
    stats = {
        "requests": 0,
        "passed": 0,
        "failed": 0,
        "input_tokens": 0,
        "output_tokens": 0,
        "cached_tokens": 0,
        "evaluated_tokens": 0,
        "prefill_cached_tokens": 0,
        "prefill_uncached_tokens": 0,
    }

    for name, method, path, payload, check in cases:
        status, _, body, elapsed_ms = call(method, path, payload)
        latencies.append(elapsed_ms)
        stats["requests"] += 1
        error = None
        merge_usage_fields(stats, body)

        try:
            check(status, body)
            passed = True
            stats["passed"] += 1
            if name in {"context_create", "context_import"} and isinstance(body, dict):
                created_id = body.get("id")
                if created_id is not None:
                    created_context_ids.add(created_id)
                if name == "context_import":
                    context_id = created_id
        except AssertionError as exc:
            passed = False
            error = str(exc)
            stats["failed"] += 1

        entry = {
            "name": name,
            "method": method,
            "path": path,
            "status": status,
            "latency_ms": round(elapsed_ms, 3),
            "passed": passed,
            "error": error,
        }
        if isinstance(body, dict):
            if isinstance(body.get("usage"), dict):
                entry["openai_usage"] = compact_openai_usage(body["usage"])
            if isinstance(body.get("cusco"), dict):
                entry["cusco_usage"] = compact_cusco_usage(body["cusco"])
        results.append(entry)

    if context_id is not None:
        status, _, body, elapsed_ms = call(
            "POST",
            "/openai/v1/completions",
            {
                "model": MODEL,
                "prompt": "This is a short context-seed paragraph used only for cache-reuse smoke checks. "
                          "It discusses determinism, admission control, and residency with deterministic behavior.",
                "context_id": context_id,
                "max_tokens": 1,
                "temperature": 0,
                "seed": 10,
            },
        )
        latencies.append(elapsed_ms)
        stats["requests"] += 1
        error = None
        merge_usage_fields(stats, body)
        try:
            list_field(status, body, "choices")
            passed = True
            stats["passed"] += 1
        except AssertionError as exc:
            passed = False
            error = str(exc)
            stats["failed"] += 1
        entry = {
            "name": "context_reuse_seed",
            "method": "POST",
            "path": "/openai/v1/completions",
            "status": status,
            "latency_ms": round(elapsed_ms, 3),
            "passed": passed,
            "error": error,
        }
        if isinstance(body, dict):
            if isinstance(body.get("usage"), dict):
                entry["openai_usage"] = compact_openai_usage(body["usage"])
            if isinstance(body.get("cusco"), dict):
                entry["cusco_usage"] = compact_cusco_usage(body["cusco"])
        results.append(entry)

        status, _, body, elapsed_ms = call(
            "POST",
            "/openai/v1/completions",
            {
                "model": MODEL,
                "prompt": "Repeat the same point in one short sentence.",
                "context_id": context_id,
                "max_tokens": 1,
                "temperature": 0,
                "seed": 10,
            },
        )
        latencies.append(elapsed_ms)
        stats["requests"] += 1
        error = None
        merge_usage_fields(stats, body)
        try:
            list_field(status, body, "choices")
            passed = True
            stats["passed"] += 1
        except AssertionError as exc:
            passed = False
            error = str(exc)
            stats["failed"] += 1
        entry = {
            "name": "context_reuse_followup",
            "method": "POST",
            "path": "/openai/v1/completions",
            "status": status,
            "latency_ms": round(elapsed_ms, 3),
            "passed": passed,
            "error": error,
        }
        if isinstance(body, dict):
            if isinstance(body.get("usage"), dict):
                entry["openai_usage"] = compact_openai_usage(body["usage"])
            if isinstance(body.get("cusco"), dict):
                entry["cusco_usage"] = compact_cusco_usage(body["cusco"])
        results.append(entry)

    if context_id is not None:
        status, _, body, elapsed_ms = call(
            "POST",
            "/openai/v1/completions",
            {
                "model": MODEL,
                "prompt": "compact this context",
                "context_id": context_id,
                "max_tokens": 1,
                "temperature": 0,
                "seed": 10,
                "compaction": {
                    "strategy_preferences": ["window_tail"],
                    "target_tokens": 1,
                },
            },
        )
        latencies.append(elapsed_ms)
        stats["requests"] += 1
        error = None
        merge_usage_fields(stats, body)
        try:
            has(status, body, "cusco")
            result = body["cusco"].get("compaction_result")
            assert result and result["success"] and result["selected_strategy_id"] == "window_tail:v1", "missing successful window-tail result"
            passed = True
            stats["passed"] += 1
        except AssertionError as exc:
            passed = False
            error = str(exc)
            stats["failed"] += 1

        entry = {
            "name": "window_tail_compaction",
            "method": "POST",
            "path": "/openai/v1/completions",
            "status": status,
            "latency_ms": round(elapsed_ms, 3),
            "passed": passed,
            "error": error,
        }
        if isinstance(body, dict):
            if isinstance(body.get("usage"), dict):
                entry["openai_usage"] = compact_openai_usage(body["usage"])
            if isinstance(body.get("cusco"), dict):
                entry["cusco_usage"] = compact_cusco_usage(body["cusco"])
        results.append(entry)

        status, _, body, elapsed_ms = call(
            "POST",
            "/openai/v1/completions",
            {
                "model": MODEL,
                "prompt": "continue after compaction",
                "context_id": context_id,
                "max_tokens": 1,
                "temperature": 0,
                "seed": 10,
            },
        )
        latencies.append(elapsed_ms)
        stats["requests"] += 1
        error = None
        merge_usage_fields(stats, body)
        try:
            list_field(status, body, "choices")
            passed = True
            stats["passed"] += 1
        except AssertionError as exc:
            passed = False
            error = str(exc)
            stats["failed"] += 1

        entry = {
            "name": "compacted_successor_continuation",
            "method": "POST",
            "path": "/openai/v1/completions",
            "status": status,
            "latency_ms": round(elapsed_ms, 3),
            "passed": passed,
            "error": error,
        }
        if isinstance(body, dict):
            if isinstance(body.get("usage"), dict):
                entry["openai_usage"] = compact_openai_usage(body["usage"])
            if isinstance(body.get("cusco"), dict):
                entry["cusco_usage"] = compact_cusco_usage(body["cusco"])
        results.append(entry)


    stream_cases = [
        (
            "chat_sse_reconstruction",
            "/openai/v1/chat/completions",
            {
                "model": MODEL,
                "messages": [{"role": "user", "content": "Reply with exactly: stream-ok"}],
                "max_tokens": 4,
                "temperature": 0,
                "seed": 10,
                "stream": True,
                "stream_options": {"include_usage": True},
            },
            lambda events: any(
                isinstance(event.get("usage"), dict)
                and any(choice.get("finish_reason") is not None for choice in event.get("choices", []))
                for event in events
            ),
        ),
        (
            "responses_sse_reconstruction",
            "/openai/v1/responses",
            {
                "model": MODEL,
                "input": "Reply with exactly: stream-ok",
                "max_output_tokens": 4,
                "temperature": 0,
                "seed": 10,
                "stream": True,
            },
            lambda events: any(
                event.get("type") == "response.completed"
                and isinstance(event.get("response", {}).get("usage"), dict)
                for event in events
            ),
        ),
    ]
    for name, path, payload, terminal_check in stream_cases:
        started = time.perf_counter_ns()
        status = 0
        try:
            status, content_type, events, elapsed_ms = stream_call(path, payload, "text/event-stream")
            status_ok(status)
            assert "text/event-stream" in content_type, f"unexpected stream content type: {content_type}"
            ids = stable_stream_ids(events)
            assert terminal_check(events), "stream has no terminal event with usage"
            passed, error = True, None
            stats["passed"] += 1
        except (AssertionError, ValueError, urllib.error.URLError) as exc:
            events = []
            elapsed_ms = (time.perf_counter_ns() - started) / 1_000_000
            ids = None
            passed, error = False, str(exc)
            stats["failed"] += 1
        stats["requests"] += 1
        latencies.append(elapsed_ms)
        results.append(
            {
                "name": name,
                "method": "POST",
                "path": path,
                "status": status if "status" in locals() else 0,
                "latency_ms": round(elapsed_ms, 3),
                "passed": passed,
                "error": error,
                "event_count": len(events),
                "stable_ids": ids,
            }
        )
    for created_id in created_context_ids:
        call("DELETE", f"/cusco/v1/contexts/{created_id}")
    call("DELETE", "/cusco/v1/api/delete", {"model": MODEL})
    reload_status, _, _, _ = call(
        "POST",
        "/cusco/v1/api/pull",
        {"model": MODEL, "stream": False},
    )
    if reload_status != 200:
        stats["failed"] += 1
    if "semantic_evidence" not in locals():
        semantic_evidence = run_semantic_comparisons(stats, latencies)
    total_prefill_work = stats["prefill_cached_tokens"] + stats["prefill_uncached_tokens"]
    if total_prefill_work:
        stats["cache_ratio"] = round(stats["prefill_cached_tokens"] / total_prefill_work, 6)
    else:
        stats["cache_ratio"] = 0.0

    stats["latency_ms"] = {
        "count": len(latencies),
        "min": round(min(latencies), 3),
        "median": round(statistics.median(latencies), 3),
        "max": round(max(latencies), 3),
    }

    report = {
        "schema_version": 1,
        "deterministic": True,
        "base_url": BASE,
        "scenarios": results,
        "stats": stats,
        "semantic_evidence": semantic_evidence,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return 0 if stats["failed"] == 0 else 1


if __name__ == "__main__":
    sys.exit(run())
