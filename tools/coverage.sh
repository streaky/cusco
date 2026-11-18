#!/bin/sh
set -eu
# Share one instrumentation environment across Rust unit tests and the external
# Terrane protocol fixture. The fixture's Cargo target lives outside this
# workspace, so include its exact executable when exporting the merged profile.
eval "$(cargo llvm-cov show-env --export-prefix)"
cargo llvm-cov clean --profraw-only
status=0
# Preserve failures while still measuring the remaining independent surfaces.
test_output="$CARGO_LLVM_COV_TARGET_DIR/test-artifacts.jsonl"
cargo test --workspace --all-features --no-fail-fast \
    --message-format=json-render-diagnostics >"$test_output" || status=1
# Cargo 1.100 may put test executables under build/*/out, which older
# cargo-llvm-cov versions intentionally skip. Use Cargo's artifact stream.
objects=$(python3 - "$test_output" <<'PY'
import json
import sys

objects = set()
with open(sys.argv[1]) as source:
    for line in source:
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            sys.stderr.write(line)
            continue
        reason = event.get("reason") if isinstance(event, dict) else None
        if reason == "compiler-artifact":
            if event.get("executable"):
                objects.add(event["executable"])
        elif reason == "compiler-message":
            sys.stderr.write(event["message"].get("rendered") or "")
        elif reason != "build-finished":
            sys.stderr.write(line)
print(" ".join("--object=" + path for path in sorted(objects)))
PY
)
artifact_file="$CARGO_LLVM_COV_TARGET_DIR/protocol-artifact-path"
CUSCO_PROTOCOL_ARTIFACT_FILE="$artifact_file" python3 server/tests/test_protocol_service.py -v || status=1
artifact=$(cat "$artifact_file")
test -f "$artifact"
service_artifact=$(terrane build server)
python3 tools/test-hybrid-service.py "$service_artifact" -v || status=1
LLVM_COV_FLAGS="${LLVM_COV_FLAGS:+$LLVM_COV_FLAGS }$objects --object=$artifact --object=$service_artifact" \
    cargo llvm-cov report --json --output-path target/llvm-cov.json || status=1
python3 tools/check_coverage.py target/llvm-cov.json || status=1
exit "$status"
