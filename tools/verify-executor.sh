#!/bin/sh
set -eu
tag=b10273
revision=a6aa6f5450eaad18b3c86631b5c3fff330f5a46e
test "$(git -C "${1:-vendor/llama.cpp}" rev-parse HEAD)" = "$revision"
test "$(sed -n 's/^patches = //p' executor/patches/series.toml)" = "[]"
