# syntax=docker/dockerfile:1.7
# Build with the Terrane checkout as context (Compose configures this).
FROM nvidia/cuda:12.8.1-devel-ubuntu24.04@sha256:520292dbb4f755fd360766059e62956e9379485d9e073bbd2f6e3c20c270ed66 AS development
ARG CUSCO_CUDA_ARCHITECTURES="61;70"
RUN apt-get update && apt-get install -y --no-install-recommends ccache curl ca-certificates cmake ninja-build python3 python3-venv clang llvm git build-essential pkg-config && rm -rf /var/lib/apt/lists/* && python3 -m venv /opt/cusco-venv && /opt/cusco-venv/bin/pip install --no-cache-dir openai==2.11.0 && curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --default-toolchain none
ENV PATH=/opt/cusco-venv/bin:/root/.cargo/bin:$PATH
WORKDIR /terrane
COPY rust-toolchain.toml /terrane/rust-toolchain.toml
# Cargo installs/selects Terrane's exact stable pin, including its components.
RUN cargo --version \
 && rustup component add llvm-tools-preview \
 && cargo install cargo-llvm-cov --version 0.6.16 --locked
WORKDIR /terrane/projects/cusco
# Existing proof workloads use /work; retain that deployment path, not a second source copy.
RUN ln -s /terrane/projects/cusco /work
COPY projects/cusco/llama.cpp-version.txt /work/llama.cpp-version.txt
RUN --mount=type=cache,id=cusco-ccache-gpu,target=/root/.cache/ccache LLAMA_CPP_TAG="$(cat /work/llama.cpp-version.txt)" \
 && git clone --filter=blob:none --branch "$LLAMA_CPP_TAG" --depth 1 https://github.com/ggml-org/llama.cpp.git /opt/llama.cpp \
 && cmake -S /opt/llama.cpp -B /opt/llama-build -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_C_COMPILER_LAUNCHER=ccache -DCMAKE_CXX_COMPILER_LAUNCHER=ccache -DCMAKE_CUDA_COMPILER_LAUNCHER=ccache -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES="${CUSCO_CUDA_ARCHITECTURES}" -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=OFF -DLLAMA_BUILD_TOOLS=OFF \
 && cmake --build /opt/llama-build --target llama --parallel
COPY projects/cusco/native /work/native
RUN cmake -S native -B /opt/cusco-native -G Ninja -DCMAKE_BUILD_TYPE=Release -DLLAMA_SOURCE_DIR=/opt/llama.cpp -DLLAMA_BUILD_DIR=/opt/llama-build -DCMAKE_INSTALL_PREFIX=/opt/cusco-install \
 && cmake --build /opt/cusco-native --target install
RUN ln -s libcuda.so /usr/local/cuda/lib64/stubs/libcuda.so.1
COPY . /terrane
ENV CUSCO_NATIVE_LIB_DIR=/opt/cusco-install/lib CUSCO_LLAMA_LIB_DIR=/opt/llama-build/bin LD_LIBRARY_PATH=/opt/llama-build/bin:/usr/local/cuda/lib64/stubs
RUN cargo build --locked --release --manifest-path /terrane/Cargo.toml --bin terrane \
 && install -m 0555 /terrane/target/release/terrane /usr/local/bin/terrane
RUN terrane toolchain install
RUN python3 server/prefetch_native.py server/terrane-dependencies.lock /terrane/packages/http-server/consumer/terrane-dependencies.lock \
 && artifact="$(terrane build --release server)" \
 && install -m 0555 "$artifact" /usr/local/bin/cusco-service
# Refresh copied inputs for proof jobs using persistent Cargo target volumes.
RUN touch Cargo.toml native/include/*.h native/shim/*.cpp \
 && find crates -type f \( -name Cargo.toml -o -name build.rs -o -name '*.rs' \) -exec touch {} +

FROM nvidia/cuda:12.8.1-base-ubuntu24.04@sha256:133c78a0575303be34164d0b90137a042172bdf60696af01a3c424ab402d86e2 AS production
RUN apt-get update && apt-get install -y --no-install-recommends ca-certificates curl libgomp1 libstdc++6 && rm -rf /var/lib/apt/lists/*
# The base supplies libcudart; the host supplies libcuda. Ship the cuBLAS and
# NCCL versions linked by ggml, not the full CUDA math library collection.
COPY --from=development /usr/local/cuda/lib64/libcublas*.so.12.* /usr/local/lib/
COPY --from=development /usr/lib/*-linux-gnu/libnccl.so.2.* /usr/local/lib/
COPY --from=development /usr/share/doc/libcublas-12-8/copyright /usr/share/doc/libcublas-12-8/copyright
COPY --from=development /usr/share/doc/libnccl2/copyright /usr/share/doc/libnccl2/copyright
COPY --from=development /opt/llama.cpp/LICENSE /usr/share/doc/llama.cpp/LICENSE
# Copy only the real versioned libraries; ldconfig creates their SONAME links.
COPY --from=development /opt/llama-build/bin/*.so.*.* /usr/local/lib/
RUN ldconfig
COPY --from=development /usr/local/bin/cusco-service /usr/local/bin/
CMD ["/usr/local/bin/cusco-service"]
