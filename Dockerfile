# syntax=docker/dockerfile:1.7
FROM nvidia/cuda:12.8.1-devel-ubuntu24.04@sha256:520292dbb4f755fd360766059e62956e9379485d9e073bbd2f6e3c20c270ed66
ARG CUSCO_CUDA_ARCHITECTURES="61;70"
RUN apt-get update && apt-get install -y --no-install-recommends ccache curl ca-certificates cmake ninja-build python3 clang llvm git build-essential pkg-config && rm -rf /var/lib/apt/lists/* && curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --default-toolchain 1.85.0 && /root/.cargo/bin/cargo install cargo-llvm-cov --version 0.6.16 --locked
ENV PATH=/root/.cargo/bin:$PATH
WORKDIR /work
COPY llama.cpp-version.txt /work/llama.cpp-version.txt
RUN --mount=type=cache,id=cusco-ccache-gpu,target=/root/.cache/ccache LLAMA_CPP_TAG="$(cat /work/llama.cpp-version.txt)" \
 && git clone --filter=blob:none --branch "$LLAMA_CPP_TAG" --depth 1 https://github.com/ggml-org/llama.cpp.git /opt/llama.cpp \
 && cmake -S /opt/llama.cpp -B /opt/llama-build -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_C_COMPILER_LAUNCHER=ccache -DCMAKE_CXX_COMPILER_LAUNCHER=ccache -DCMAKE_CUDA_COMPILER_LAUNCHER=ccache -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES="${CUSCO_CUDA_ARCHITECTURES}" -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=OFF -DLLAMA_BUILD_TOOLS=OFF \
 && cmake --build /opt/llama-build --target llama --parallel
COPY native /work/native
RUN cmake -S native -B /opt/cusco-native -G Ninja -DCMAKE_BUILD_TYPE=Release -DLLAMA_SOURCE_DIR=/opt/llama.cpp -DLLAMA_BUILD_DIR=/opt/llama-build -DCMAKE_INSTALL_PREFIX=/opt/cusco-install \
 && cmake --build /opt/cusco-native --target install
RUN ln -s libcuda.so /usr/local/cuda/lib64/stubs/libcuda.so.1
COPY . .
ENV CUSCO_NATIVE_LIB_DIR=/opt/cusco-install/lib CUSCO_LLAMA_LIB_DIR=/opt/llama-build/bin LD_LIBRARY_PATH=/opt/llama-build/bin:/usr/local/cuda/lib64/stubs
