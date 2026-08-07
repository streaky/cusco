#!/bin/sh
set -eu

compose() {
    docker compose -f compose.test.yaml "$@"
}

result_dir=${CUSCO_RESULT_DIR:-./results}
mkdir -p "$result_dir"
compose run --build --rm model-fetch
compose run --build --rm executor-proof
python3 tools/report-inference-proof.py "$result_dir/phase1.json"
