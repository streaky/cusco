#!/usr/bin/env python3
"""Run pinned oai-lens and classify probe failures separately from harness failures."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--gate-report", type=Path, required=True)
    parser.add_argument("--expectations", type=Path, required=True)
    parser.add_argument("--revision-file", type=Path, required=True)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if args.command and args.command[0] == "--":
        args.command = args.command[1:]
    if not args.command:
        parser.error("runner command is required after --")
    return args


def timestamp() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def load_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def write_json_atomic(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def validate_expectations(value: Any, revision: str) -> tuple[str, dict[str, str]]:
    if not isinstance(value, dict) or value.get("schema_version") != 1:
        raise ValueError("expectations must use schema_version 1")
    if value.get("runner_revision") != revision:
        raise ValueError("expectations runner_revision does not match oai-lens-version.txt")
    profile = value.get("profile")
    probes = value.get("probes")
    if not isinstance(profile, str) or not profile:
        raise ValueError("expectations profile must be a non-empty string")
    if not isinstance(probes, dict) or not probes:
        raise ValueError("expectations probes must be a non-empty object")
    if any(not isinstance(name, str) or status not in {"pass", "fail"} for name, status in probes.items()):
        raise ValueError("every expected probe must map to pass or fail")
    return profile, probes


def validate_report(value: Any, profile: str) -> dict[str, str]:
    if not isinstance(value, dict) or value.get("version") != 1:
        raise ValueError("oai-lens report must use version 1")
    if value.get("profile") != profile:
        raise ValueError("oai-lens report profile does not match expectations")
    results = value.get("results")
    if not isinstance(results, list) or not results:
        raise ValueError("oai-lens report has no probe results")
    observed: dict[str, str] = {}
    for result in results:
        if not isinstance(result, dict):
            raise ValueError("oai-lens probe result must be an object")
        name = result.get("name")
        status = result.get("status")
        if not isinstance(name, str) or status not in {"pass", "fail"}:
            raise ValueError("oai-lens probe result has an invalid name or status")
        if name in observed:
            raise ValueError(f"duplicate oai-lens probe result: {name}")
        observed[name] = status
    return observed


def main() -> int:
    args = parse_args()
    started_at = timestamp()
    revision = args.revision_file.read_text(encoding="utf-8").strip()
    base: dict[str, Any] = {
        "schema_version": 1,
        "runner_revision": revision,
        "command": args.command,
        "started_at": started_at,
        "report_path": str(args.report),
    }

    try:
        profile, expected = validate_expectations(load_json(args.expectations), revision)
        args.report.unlink(missing_ok=True)
        completed = subprocess.run(args.command, check=False)
        base["runner_exit_code"] = completed.returncode
        if not args.report.is_file():
            raise ValueError("oai-lens did not produce its report")
        observed = validate_report(load_json(args.report), profile)
        names = sorted(set(expected) | set(observed))
        deltas = {
            "newly_passing": [name for name in names if expected.get(name) == "fail" and observed.get(name) == "pass"],
            "newly_failing": [name for name in names if expected.get(name) == "pass" and observed.get(name) == "fail"],
            "unchanged_failing": [name for name in names if expected.get(name) == observed.get(name) == "fail"],
            "unchanged_passing": [name for name in names if expected.get(name) == observed.get(name) == "pass"],
            "added": [name for name in names if name not in expected],
            "missing": [name for name in names if name not in observed],
        }
        newly_failing = deltas["newly_failing"]
        failed = sorted(name for name, status in observed.items() if status == "fail")
        base.update(
            {
                "classification": (
                    "regression"
                    if newly_failing
                    else "runner_completed_with_probe_failures"
                    if failed
                    else "runner_passed"
                ),
                "profile": profile,
                "probe_counts": {"total": len(observed), "passed": len(observed) - len(failed), "failed": len(failed)},
                "deltas": deltas,
                "upstream_report_sha256": digest(args.report),
                "completed_at": timestamp(),
            }
        )
        write_json_atomic(args.gate_report, base)
        print(json.dumps(base, indent=2, sort_keys=True))
        return 1 if newly_failing else 0
    except (OSError, ValueError, json.JSONDecodeError) as error:
        base.update(
            {
                "classification": "harness_failed",
                "error": f"{type(error).__name__}: {error}",
                "completed_at": timestamp(),
            }
        )
        write_json_atomic(args.gate_report, base)
        print(json.dumps(base, indent=2, sort_keys=True), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
