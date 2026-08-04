#!/bin/sh
set -eu
tag=b10273
revision=a6aa6f5450eaad18b3c86631b5c3fff330f5a46e
repository=https://github.com/ggml-org/llama.cpp.git
destination=${1:-vendor/llama.cpp}
if [ -e "$destination/.git" ]; then test "$(git -C "$destination" rev-parse HEAD)" = "$revision"; else git clone --filter=blob:none "$repository" "$destination";git -C "$destination" checkout --detach "$revision";fi
