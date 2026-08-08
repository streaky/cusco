#!/usr/bin/env python3
from pathlib import Path
import json
import os
import statistics
import sys
import time
import urllib.error
import urllib.request
BASE = os.environ.get("CUSCO_SMOKE_BASE_URL", "http://127.0.0.1:8080").rstrip("/")
TOKEN = os.environ.get("CUSCO_SMOKE_TOKEN", "smoke-report-token")
OUT = Path(os.environ.get("CUSCO_SMOKE_OUTPUT", "/results/smoke-report.json"))


def call(method, path, payload=None):
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


def run():
    cases = [
        ("openapi", "GET", "/openai/v1/openapi.json", None, lambda status, body: has(status, body, "paths")),
        ("models", "GET", "/openai/v1/models", None, lambda status, body: list_field(status, body, "data")),
        (
            "completion",
            "POST",
            "/openai/v1/completions",
            {
                "model": "gemma-4-e2b-it",
                "prompt": "Reply with exactly: smoke-ok",
                "max_tokens": 8,
                "temperature": 0,
            },
            lambda status, body: list_field(status, body, "choices"),
        ),
        (
            "chat",
            "POST",
            "/openai/v1/chat/completions",
            {
                "model": "gemma-4-e2b-it",
                "messages": [{"role": "user", "content": "Reply with exactly: smoke-ok"}],
                "max_tokens": 8,
                "temperature": 0,
            },
            lambda status, body: list_field(status, body, "choices"),
        ),
        (
            "responses",
            "POST",
            "/openai/v1/responses",
            {
                "model": "gemma-4-e2b-it",
                "input": "Reply with exactly: smoke-ok",
                "max_output_tokens": 8,
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
            if name == "context_import" and isinstance(body, dict):
                context_id = body.get("id")
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
                "model": "gemma-4-e2b-it",
                "prompt": "This is a short context-seed paragraph used only for cache-reuse smoke checks. "
                          "It discusses determinism, admission control, and residency with deterministic behavior.",
                "context_id": context_id,
                "max_tokens": 1,
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
                "model": "gemma-4-e2b-it",
                "prompt": "Repeat the same point in one short sentence.",
                "context_id": context_id,
                "max_tokens": 1,
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
                "model": "gemma-4-e2b-it",
                "prompt": "compact this context",
                "context_id": context_id,
                "max_tokens": 1,
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
                "model": "gemma-4-e2b-it",
                "prompt": "continue after compaction",
                "context_id": context_id,
                "max_tokens": 1,
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
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return 0 if stats["failed"] == 0 else 1


if __name__ == "__main__":
    sys.exit(run())
