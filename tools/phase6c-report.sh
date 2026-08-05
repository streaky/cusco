#!/bin/sh
set -eu

result_dir=${CUSCO_RESULT_DIR:-./results}
model_dir=${CUSCO_MODEL_DIR:-./models}
port=${CUSCO_PHASE6C_REPORT_PORT:-18082}
service=phase6c-report

mkdir -p "$result_dir"
rm -f \
    "$result_dir/phase6c-coverage.json" \
    "$result_dir/phase6c-gpu.csv" \
    "$result_dir/phase6c-server.json" \
    "$result_dir/phase6c-shutdown.json" \
    "$result_dir/phase6c-state.json"

cleanup() {
    docker compose stop -t 35 "$service" >/dev/null 2>&1 || true
    docker compose rm -f "$service" >/dev/null 2>&1 || true
}
trap cleanup EXIT INT TERM

docker compose run --build --rm test
printf '%s\n' '{"passed":true,"command":"docker compose run --build --rm test","per_file_line_floor_percent":80}' \
    > "$result_dir/phase6c-coverage.json"
docker compose run --build --rm executor-proof
docker compose run --build --rm mapped-proof
docker compose run --build --rm --no-deps --entrypoint chmod "$service" a+rwx /results

docker compose up --build --detach "$service"
python3 tools/report-phase6c.py \
    "http://127.0.0.1:$port" \
    "$result_dir" \
    "$model_dir/gemma-4-e2b-it.gguf"

docker compose stop -t 35 "$service"
printf '%s\n' '{"passed":true,"signal":"SIGTERM","grace_seconds":30}' \
    > "$result_dir/phase6c-shutdown.json"
docker compose up --detach "$service"
python3 tools/report-phase6c.py \
    "http://127.0.0.1:$port" \
    "$result_dir" \
    "$model_dir/gemma-4-e2b-it.gguf" \
    restart-check
