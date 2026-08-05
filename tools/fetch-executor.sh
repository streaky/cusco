#!/bin/sh
set -eu
tag=$(cat llama.cpp-version.txt)
repository=https://github.com/ggml-org/llama.cpp.git
destination=${1:-vendor/llama.cpp}
if [ -e "$destination/.git" ]; then git -C "$destination" fetch --depth 1 origin "refs/tags/$tag:refs/tags/$tag"; else git clone --filter=blob:none --branch "$tag" --depth 1 "$repository" "$destination"; fi
git -C "$destination" checkout --detach "$tag"
