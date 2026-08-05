#!/bin/sh
set -eu

result_dir=${CUSCO_RESULT_DIR:-./results}
mkdir -p "$result_dir"
docker compose run --build --rm executor-proof
python3 tools/report-inference-proof.py "$result_dir/phase1.json"
