#!/bin/sh
set -eu
cargo llvm-cov --workspace --all-features --json --output-path target/llvm-cov.json
python3 tools/check_coverage.py target/llvm-cov.json
