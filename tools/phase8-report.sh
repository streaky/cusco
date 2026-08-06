#!/bin/sh
set -eu

compose() {
    docker compose -f compose.test.yaml "$@"
}

result_dir=${CUSCO_RESULT_DIR:-./results}
model_dir=${CUSCO_MODEL_DIR:-./models}
service=phase8-proof

mkdir -p "$result_dir"
rm -f \
    "$result_dir/phase8-coverage.json" \
    "$result_dir/phase8-gpu.csv" \
    "$result_dir/phase8-server.json"

compose run --build --rm test
printf '%s\n' '{"passed":true,"command":"docker compose -f compose.test.yaml run --build --rm test","per_file_line_floor_percent":80}' \
    > "$result_dir/phase8-coverage.json"
compose run --build --rm --no-deps --entrypoint chmod "$service" a+rwx /results
compose run --build --rm "$service"
python3 tools/report-phase8.py \
    "$result_dir" \
    "$model_dir/gemma-4-e2b-it.gguf"
