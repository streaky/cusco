#!/bin/sh
set -eu
tag=$(cat llama.cpp-version.txt)
checkout=${1:-vendor/llama.cpp}
repository=$(sed -n 's/^repository = "\(.*\)"/\1/p' executor/llama.lock.toml)
test "$(git -C "$checkout" describe --tags --exact-match)" = "$tag"
test "$(git -C "$checkout" remote get-url origin)" = "$repository"
test -z "$(git -C "$checkout" status --porcelain --untracked-files=all)"
test "$(sed -n 's/^patches = //p' executor/patches/series.toml)" = "[]"
