#!/bin/sh
set -eu

result_dir=${CUSCO_RESULT_DIR:-./results}
model_dir=${CUSCO_MODEL_DIR:-./models}
port=${CUSCO_SERVER_REPORT_PORT:-18082}
docker compose run --build --rm --no-deps --entrypoint chmod server-report a+rwx /results
mkdir -p "$result_dir"
rm -f "$result_dir/phase6a-gpu.csv" "$result_dir/phase6a-server.json" "$result_dir/phase6a-state.json"
cleanup() {
    docker compose stop server-report >/dev/null 2>&1 || true
    docker compose rm -f server-report >/dev/null 2>&1 || true
}
trap cleanup EXIT INT TERM

docker compose up --detach server-report
python3 tools/report-server-inference.py \
    "http://127.0.0.1:$port" \
    "$result_dir" \
    "$model_dir/gemma-4-e2b-it.gguf"
