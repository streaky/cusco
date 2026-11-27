#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
data=$(realpath -- "${CUSCO_DATA_DIR:-./data-dev}")
[[ -r "$data/config.yaml" ]] || { echo "Missing $data/config.yaml (copy config.example.yaml there first)" >&2; exit 1; }

if [[ -n ${CUSCO_SERVICE_BIN:-} ]]; then
    service=$CUSCO_SERVICE_BIN
else
    launcher=
    if [[ ${TERRANE_SCCACHE:-1} == 1 ]] && launcher=$(command -v sccache); then
        # Terrane's contained Cargo cannot reach the default TCP server or a /tmp socket.
        socket_dir=${XDG_RUNTIME_DIR:-"$HOME/.cache"}/cusco
        export SCCACHE_SERVER_UDS=${SCCACHE_SERVER_UDS:-"$socket_dir/sccache.sock"}
        mkdir -p -m 700 -- "$(dirname -- "$SCCACHE_SERVER_UDS")"
        if [[ ! -S $SCCACHE_SERVER_UDS ]]; then
            sccache --start-server
        fi
        export TERRANE_SCCACHE=1 CARGO_INCREMENTAL=0 TMPDIR=/tmp
    fi
    launchers=("-DCMAKE_C_COMPILER_LAUNCHER=$launcher" "-DCMAKE_CXX_COMPILER_LAUNCHER=$launcher")
    terrane toolchain install
    backend=${CUSCO_GGML_CUDA:-ON}
    build="$PWD/target/host-$backend"
    tag=$(< llama.cpp-version.txt)
    src="$PWD/target/llama-src/$tag"
    if [[ ! -d $src/.git ]]; then
        tools/fetch-executor.sh "$src"
    fi
    cuda_args=()
    if [[ $backend == ON ]]; then
        cuda_args+=("-DCMAKE_CUDA_ARCHITECTURES=${CUSCO_CUDA_ARCHITECTURES:-native}")
    fi
    cmake -S "$src" -B "$build/llama" -G Ninja -DCMAKE_BUILD_TYPE=Release \
        "${launchers[@]}" "-DCMAKE_CUDA_COMPILER_LAUNCHER=$launcher" -DGGML_CCACHE=OFF \
        "-DGGML_CUDA=$backend" "${cuda_args[@]}" -DLLAMA_BUILD_TESTS=OFF \
        -DLLAMA_BUILD_EXAMPLES=OFF -DLLAMA_BUILD_TOOLS=OFF
    cmake --build "$build/llama" --target llama
    cmake -S native -B "$build/shim" -G Ninja -DCMAKE_BUILD_TYPE=Release \
        "${launchers[@]}" \
        "-DLLAMA_SOURCE_DIR=$src" "-DLLAMA_BUILD_DIR=$build/llama"
    cmake --build "$build/shim" --target cusco_executor
    export CUSCO_NATIVE_LIB_DIR="$build/shim" CUSCO_LLAMA_LIB_DIR="$build/llama/bin"
    terrane fetch server
    export LD_LIBRARY_PATH="$CUSCO_LLAMA_LIB_DIR${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
    # Run from the package so Cargo sees server/.cargo's local backend patch.
    service=$(cd server && terrane build --release .)
    [[ $service == /* ]] || service="$PWD/server/$service"
fi

# Keep /data paths in existing configs and SQLite model records valid without
# creating /data on the host. Only the server sees this mount; no image or sudo.
mounts=()
for path in /*; do
    [[ $path != /data && -r $path ]] || continue
    if [[ -L $path ]]; then
        mounts+=(--symlink "$(readlink -- "$path")" "$path")
    else
        mounts+=(--dev-bind "$path" "$path")
    fi
done
exec bwrap "${mounts[@]}" --bind "$data" /data -- setpriv --pdeathsig TERM \
    "$service" --config /data/config.yaml "$@"
