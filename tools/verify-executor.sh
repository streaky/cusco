#!/bin/sh
set -eu
tag=$(cat llama.cpp-version.txt)
test "$(git -C "${1:-vendor/llama.cpp}" describe --tags --exact-match)" = "$tag"
test "$(sed -n 's/^patches = //p' executor/patches/series.toml)" = "[]"
