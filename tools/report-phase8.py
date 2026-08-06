#!/usr/bin/env python3
import hashlib
import json
import pathlib
import sys

RESULT_DIR = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else "results")
MODEL = pathlib.Path(sys.argv[2] if len(sys.argv) > 2 else "models/gemma-4-e2b-it.gguf")
ARTIFACT = RESULT_DIR / "phase8-server.json"
WORKLOAD = pathlib.Path("config/phase8-workload.json")


def digest(path):
    value = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            value.update(chunk)
    return value.hexdigest()


artifact = json.loads(ARTIFACT.read_text(encoding="utf-8"))
coverage = json.loads((RESULT_DIR / "phase8-coverage.json").read_text(encoding="utf-8"))
workload_bytes = WORKLOAD.read_bytes()
header = pathlib.Path("native/include/cusco_executor.h").read_text(encoding="utf-8")
abi_line = next(
    line for line in header.splitlines() if line.startswith("#define CUSCO_EXECUTOR_ABI_VERSION ")
)
artifact["model"] = {"path": str(MODEL), "sha256": digest(MODEL)}
artifact["workload_provenance"] = {
    "path": str(WORKLOAD),
    "sha256": hashlib.sha256(workload_bytes).hexdigest(),
}
artifact["build"] = {
    "llama_cpp_tag": pathlib.Path("llama.cpp-version.txt").read_text(encoding="utf-8").strip(),
    "executor_abi": int(abi_line.split()[-1].removesuffix("u")),
}
artifact["gpu"] = (RESULT_DIR / "phase8-gpu.csv").read_text(encoding="utf-8").strip()
artifact["coverage"] = coverage
artifact["checks"] = {
    "CLI scheduler gates passed": artifact.get("passed") is True,
    "coverage gate passed": coverage.get("passed") is True,
    "GPU provenance recorded": bool(artifact["gpu"]),
    "mixed workload includes all classes": {
        row.get("class") for row in artifact.get("mixed", [])
    } == {"interactive", "standard", "batch"},
    "multiple principals competed": len(
        {row.get("principal") for row in artifact.get("mixed", [])}
    ) >= 3,
    "cancellation was observed": any(
        row.get("observed") == "cancelled" for row in artifact.get("mixed", [])
    ),
    "deadline expiry was observed": any(
        row.get("observed") == "deadline" for row in artifact.get("mixed", [])
    ),
    "capacity recovered": artifact.get("capacity_recovery", {}).get("observed") == "complete",
    "diagnostic records were lossless": artifact.get("scheduler", {})
        .get("metrics", {})
        .get("diagnostic_records_lost") == 0,
}
ARTIFACT.write_text(json.dumps(artifact, indent=2) + "\n", encoding="utf-8")
failed = [name for name, passed in artifact["checks"].items() if not passed]
if failed:
    raise SystemExit("Phase 8 proof failed: " + ", ".join(failed))
print(f"Phase 8 proof: PASS. Artifact: `{ARTIFACT}`")
