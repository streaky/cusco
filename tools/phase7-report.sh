#!/bin/sh
set -eu

compose() {
    docker compose -f compose.test.yaml "$@"
}

result_dir=${CUSCO_RESULT_DIR:-./results}
model_dir=${CUSCO_MODEL_DIR:-./models}
port=${CUSCO_PHASE7_REPORT_PORT:-18083}
service=phase7-report

mkdir -p "$result_dir"
rm -rf "$result_dir/phase7-spill"
rm -f \
    "$result_dir/phase7-coverage.json" \
    "$result_dir/phase7-gpu.csv" \
    "$result_dir/phase7-server.json" \
    "$result_dir/phase7-state.json"

cleanup() {
    compose stop -t 35 "$service" >/dev/null 2>&1 || true
    compose rm -f "$service" >/dev/null 2>&1 || true
}
trap cleanup EXIT INT TERM

compose run --build --rm test
printf '%s\n' '{"passed":true,"command":"docker compose -f compose.test.yaml run --build --rm test","per_file_line_floor_percent":80}' \
    > "$result_dir/phase7-coverage.json"
compose run --build --rm --no-deps --entrypoint chmod "$service" a+rwx /results
compose up --build --detach "$service"
python3 tools/report-phase7.py \
    "http://127.0.0.1:$port" \
    "$result_dir" \
    "$model_dir/gemma-4-e2b-it.gguf"
compose stop -t 35 "$service"
compose up --detach "$service"
python3 tools/report-phase7.py \
    "http://127.0.0.1:$port" \
    "$result_dir" \
    "$model_dir/gemma-4-e2b-it.gguf" \
    restart-check
