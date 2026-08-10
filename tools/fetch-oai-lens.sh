#!/bin/sh
set -eu

root=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
revision=$(tr -d '\r\n' < "$root/oai-lens-version.txt")
destination="$root/.tools/oai-lens"
remote=https://github.com/streaky/oai-lens.git

if [ "${#revision}" -ne 40 ]; then
    echo "oai-lens-version.txt must contain one lowercase 40-character commit SHA" >&2
    exit 2
fi
case "$revision" in
    *[!0-9a-f]*) echo "oai-lens-version.txt must contain one lowercase 40-character commit SHA" >&2; exit 2 ;;
esac

mkdir -p "$root/.tools"
fresh_checkout=false
if [ ! -d "$destination/.git" ]; then
    if [ -e "$destination" ]; then
        echo "$destination exists but is not an oai-lens Git checkout" >&2
        exit 2
    fi
    git clone --filter=blob:none --no-checkout "$remote" "$destination"
    fresh_checkout=true
fi

actual_remote=$(git -C "$destination" remote get-url origin)
if [ "$actual_remote" != "$remote" ]; then
    echo "unexpected oai-lens origin: $actual_remote" >&2
    exit 2
fi
if [ "$fresh_checkout" = false ] && [ -n "$(git -C "$destination" status --porcelain)" ]; then
    echo "refusing to modify dirty oai-lens checkout at $destination" >&2
    exit 2
fi

git -C "$destination" fetch --no-tags origin "$revision"
git -C "$destination" checkout --detach "$revision"
actual_revision=$(git -C "$destination" rev-parse HEAD)
if [ "$actual_revision" != "$revision" ]; then
    echo "oai-lens revision mismatch: expected $revision, got $actual_revision" >&2
    exit 2
fi

printf 'oai-lens ready at %s (%s)\n' "$destination" "$actual_revision"
