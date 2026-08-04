# syntax=docker/dockerfile:1.7
ARG RUST_IMAGE=rust:1.85-bookworm@sha256:e51d0265072d2d9d5d320f6a44dde6b9ef13653b035098febd68cce8fa7c0bc4
FROM ${RUST_IMAGE} AS cpu
RUN apt-get update && apt-get install -y --no-install-recommends ccache cmake ninja-build python3 clang llvm git && rm -rf /var/lib/apt/lists/* && cargo install cargo-llvm-cov --version 0.6.16 --locked
WORKDIR /work
RUN --mount=type=cache,id=cusco-ccache-cpu,target=/root/.cache/ccache,sharing=locked git clone --filter=blob:none --branch b10273 --depth 1 https://github.com/ggml-org/llama.cpp.git /opt/llama.cpp \
 && test "$(git -C /opt/llama.cpp rev-parse HEAD)" = a6aa6f5450eaad18b3c86631b5c3fff330f5a46e \
 && cmake -S /opt/llama.cpp -B /opt/llama-build -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_C_COMPILER_LAUNCHER=ccache -DCMAKE_CXX_COMPILER_LAUNCHER=ccache -DGGML_CUDA=OFF -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=OFF -DLLAMA_BUILD_TOOLS=OFF \
 && cmake --build /opt/llama-build --target llama --parallel
COPY native /work/native
RUN cmake -S native -B /opt/cusco-native -G Ninja -DCMAKE_BUILD_TYPE=Release -DLLAMA_SOURCE_DIR=/opt/llama.cpp -DLLAMA_BUILD_DIR=/opt/llama-build -DCMAKE_INSTALL_PREFIX=/opt/cusco-install \
 && cmake --build /opt/cusco-native --target install
COPY . .
ENV CUSCO_NATIVE_LIB_DIR=/opt/cusco-install/lib CUSCO_LLAMA_LIB_DIR=/opt/llama-build/bin LD_LIBRARY_PATH=/opt/llama-build/bin
FROM nvidia/cuda:12.8.1-devel-ubuntu24.04@sha256:520292dbb4f755fd360766059e62956e9379485d9e073bbd2f6e3c20c270ed66 AS gpu
ARG CUSCO_CUDA_ARCHITECTURES="61;70;75;80;86;87;89;90;100;120"
RUN apt-get update && apt-get install -y --no-install-recommends ccache curl ca-certificates cmake ninja-build python3 clang llvm git build-essential pkg-config && rm -rf /var/lib/apt/lists/* && curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --default-toolchain 1.85.0 && /root/.cargo/bin/cargo install cargo-llvm-cov --version 0.6.16 --locked
ENV PATH=/root/.cargo/bin:$PATH
WORKDIR /work
RUN --mount=type=cache,id=cusco-ccache-gpu,target=/root/.cache/ccache,sharing=locked git clone --filter=blob:none --branch b10273 --depth 1 https://github.com/ggml-org/llama.cpp.git /opt/llama.cpp \
 && test "$(git -C /opt/llama.cpp rev-parse HEAD)" = a6aa6f5450eaad18b3c86631b5c3fff330f5a46e \
 && cmake -S /opt/llama.cpp -B /opt/llama-build -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_C_COMPILER_LAUNCHER=ccache -DCMAKE_CXX_COMPILER_LAUNCHER=ccache -DCMAKE_CUDA_COMPILER_LAUNCHER=ccache -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES="${CUSCO_CUDA_ARCHITECTURES}" -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=OFF -DLLAMA_BUILD_TOOLS=OFF \
 && cmake --build /opt/llama-build --target llama --parallel
COPY native /work/native
RUN cmake -S native -B /opt/cusco-native -G Ninja -DCMAKE_BUILD_TYPE=Release -DLLAMA_SOURCE_DIR=/opt/llama.cpp -DLLAMA_BUILD_DIR=/opt/llama-build -DCMAKE_INSTALL_PREFIX=/opt/cusco-install \
 && cmake --build /opt/cusco-native --target install
COPY . .
ENV CUSCO_NATIVE_LIB_DIR=/opt/cusco-install/lib CUSCO_LLAMA_LIB_DIR=/opt/llama-build/bin LD_LIBRARY_PATH=/opt/llama-build/bin
