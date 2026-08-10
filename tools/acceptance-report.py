#!/usr/bin/env python3
import argparse
import datetime as dt
import hashlib
import json
import os
import platform
import shutil
import urllib.request
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS = Path(os.environ.get("CUSCO_RESULT_DIR", "/results"))
MANIFEST = Path(os.environ.get("CUSCO_ACCEPTANCE_MANIFEST", ROOT / "config/acceptance.json"))


def digest(path):
    path = Path(path)
    value = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            value.update(block)
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": value.hexdigest()}


def capture(command):
    try:
        return subprocess.run(command, cwd=ROOT, text=True, capture_output=True, check=False).stdout.strip() or None
    except (OSError, subprocess.SubprocessError):
        return None


def run_gate(name, command, contract=None, artifact=None, workload=None, env=None):
    started_at = dt.datetime.now(dt.timezone.utc).isoformat()
    started = time.monotonic()
    result = subprocess.run(command, cwd=ROOT, text=True, env=env)
    elapsed = round((time.monotonic() - started) * 1000, 3)
    row = {
        "name": name,
        "contract": contract,
        "command": command,
        "started_at": started_at,
        "duration_ms": elapsed,
        "exit_code": result.returncode,
        "passed": result.returncode == 0,
    }
    if workload:
        row["workload"] = digest(workload)
    if artifact and result.returncode == 0:
        try:
            body = json.loads(Path(artifact).read_text())
            asserted = body.get("passed", body.get("stats", {}).get("failed", 0) == 0)
            row["artifact"] = digest(artifact)
            row["artifact_assertion"] = bool(asserted)
            row["passed"] = row["passed"] and bool(asserted)
        except (OSError, ValueError) as error:
            row["artifact_error"] = str(error)
            row["passed"] = False
    return row


def provenance(manifest, model=None):
    revision = capture(["git", "rev-parse", "HEAD"])
    dirty = capture(["git", "status", "--porcelain"])
    value = {
        "source_revision": revision,
        "source_dirty": None if dirty is None else bool(dirty),
        "python": sys.version.split()[0],
        "rustc": capture(["rustc", "--version"]),
        "platform": platform.platform(),
        "cuda_architectures": os.environ.get("CUSCO_CUDA_ARCHITECTURES"),
        "requested_host_gpu": os.environ.get("CUSCO_REQUESTED_HOST_GPU"),
        "gpu": capture(["nvidia-smi", "--query-gpu=uuid,name,compute_cap,driver_version", "--format=csv,noheader"]),
        "manifest": digest(MANIFEST),
    }
    for item in manifest.get("real_model", []):
        workload = item.get("workload")
        if workload:
            value.setdefault("workloads", []).append(digest(workload))
    semantic_workload = manifest.get("semantic_workload")
    if semantic_workload:
        value.setdefault("workloads", []).append(digest(semantic_workload))
    if model:
        resolved = Path(model)
        if resolved.is_file():
            value["model"] = {"identity": manifest["model"], **digest(resolved)}
        else:
            value["model"] = {"identity": manifest["model"], "path": model}
    return value

def advisory_evidence():
    gate_path = RESULTS / "oai-lens-gate.json"
    upstream_path = RESULTS / "oai-lens-report.json"
    if not gate_path.is_file():
        return {"oai_lens": {"classification": "not_run"}}
    try:
        gate = json.loads(gate_path.read_text())
        evidence = {
            "classification": gate["classification"],
            "runner_revision": gate["runner_revision"],
            "probe_counts": gate.get("probe_counts"),
            "deltas": gate.get("deltas"),
            "gate_report": digest(gate_path),
        }
        if upstream_path.is_file():
            evidence["upstream_report"] = digest(upstream_path)
        return {"oai_lens": evidence}
    except (KeyError, OSError, ValueError) as error:
        return {
            "oai_lens": {
                "classification": "invalid_artifact",
                "error": str(error),
                "gate_report": digest(gate_path),
            }
        }



def write_report(mode, manifest, gates, model=None):
    report = {
        "schema_version": 1,
        "test_set_version": manifest["test_set_version"],
        "mode": mode,
        "passed": all(gate["passed"] for gate in gates),
        "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "provenance": provenance(manifest, model),
        "gates": gates,
        "advisory_evidence": advisory_evidence(),
    }
    RESULTS.mkdir(parents=True, exist_ok=True)
    output = RESULTS / f"acceptance-{mode}-report.json"
    output.write_text(json.dumps(report, indent=2) + "\n")
    if mode == "real-model":
        shutil.copyfile(output, RESULTS / "acceptance-report.json")
    print(json.dumps(report, indent=2))
    return 0 if report["passed"] else 1


def model_free(manifest):
    gates = []
    for item in manifest["model_free"]:
        command = ["cargo", "test", "-p", item["crate"], item["test"]]
        gates.append(run_gate(item["test"], command, item["contract"]))
        if not gates[-1]["passed"]:
            break
    return write_report("model-free", manifest, gates)
def api_smoke_gate(semantic_workload):
    for directory in (Path("/data/db"), Path("/data/spill"), Path("/data/user-models"), Path("/data/models")):
        directory.mkdir(parents=True, exist_ok=True)
    Path("/data/user.yaml").write_text("models: []\n")
    env = os.environ.copy()
    env["CUSCO_BEARER_TOKEN"] = "smoke-report-token"
    server = subprocess.Popen(
        ["cargo", "run", "-p", "cusco", "--", "serve", "--config", "/work/config/test.yaml"],
        cwd=ROOT,
        env=env,
    )
    try:
        deadline = time.monotonic() + 120
        while time.monotonic() < deadline:
            if server.poll() is not None:
                return {
                    "name": "api-smoke",
                    "contract": "real-model API, streaming, continuation, and compaction",
                    "command": ["python3", "tools/smoke-report.py"],
                    "duration_ms": 0,
                    "exit_code": server.returncode,
                    "passed": False,
                    "error": "server exited before readiness",
                }
            try:
                request = urllib.request.Request(
                    "http://127.0.0.1:8080/cusco/v1/status",
                    headers={"Authorization": "Bearer smoke-report-token"},
                )
                with urllib.request.urlopen(request, timeout=2):
                    break
            except OSError:
                time.sleep(1)
        else:
            raise RuntimeError("server readiness timed out")
        smoke_env = env | {
            "CUSCO_SMOKE_BASE_URL": "http://127.0.0.1:8080",
            "CUSCO_SMOKE_OUTPUT": str(RESULTS / "smoke-report.json"),
            "CUSCO_SMOKE_TOKEN": "smoke-report-token",
            "CUSCO_SEMANTIC_WORKLOAD": semantic_workload,
        }
        return run_gate(
            "api-smoke",
            ["python3", "tools/smoke-report.py"],
            "real-model API, streaming, continuation, compaction, and semantic fidelity",
            str(RESULTS / "smoke-report.json"),
            workload=semantic_workload,
            env=smoke_env,
        )
    finally:
        server.terminate()
        try:
            server.wait(timeout=10)
        except subprocess.TimeoutExpired:
            server.kill()
            server.wait()




def real_model(manifest, model):
    gates = []
    for item in manifest["real_model"]:
        command = [part.format(model=model) for part in item["command"]]
        gates.append(run_gate(item["name"], command, artifact=item["artifact"], workload=item.get("workload")))
        if not gates[-1]["passed"]:
            break
    if len(gates) == len(manifest["real_model"]):
        gates.append(api_smoke_gate(manifest["semantic_workload"]))
    return write_report("real-model", manifest, gates, model)


def main():
    parser = argparse.ArgumentParser(description="Run Cusco's versioned acceptance test set")
    parser.add_argument("--mode", choices=("model-free", "real-model"), required=True)
    parser.add_argument("--model")
    args = parser.parse_args()
    manifest = json.loads(MANIFEST.read_text())
    if args.mode == "model-free":
        return model_free(manifest)
    if not args.model:
        parser.error("--model is required in real-model mode")
    return real_model(manifest, args.model)


if __name__ == "__main__":
    raise SystemExit(main())
