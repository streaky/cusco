# Cusco

Cusco is an experimental persistent, tiered, branch-aware model-state server built around llama.cpp. Its goal is to make model execution state a durable resource that can be captured, displaced from scarce accelerator memory, restored later, and continued exactly.

The project separates responsibilities deliberately: Rust will manage logical contexts, scheduling, storage tiers, and transactional state, while a narrow native boundary lets llama.cpp retain ownership of model-specific execution details and optimized kernels.

## Current state

Cusco has completed Phases 1 through 5 and the Phase 6A persistent mapped-core
milestone. The executor proof established exact checkpoint continuation for a
hybrid/recurrent Gemma model, and the Rust layers now provide durable logical
contexts, capacity-accounted physical state, transactional mapped activation,
an authenticated HTTP API, and immutable model registration.

Phase 6A connects those pieces on the live request path: one validated Gemma
profile owns a process-persistent llama.cpp model and executor slot; opaque
context continuations reuse complete mapped blocks without model reload or
valid-prefix reevaluation; and admission accounts for context, prompt, output,
device, and host capacity before decode. Phase 6B will add the incremental
generation frontier, followed by bounded lifecycle and acceptance work in
Phase 6C.

## Run the real-model inference integration test

With the validation GGUF at `models/gemma-4-e2b-it.gguf`, Docker's NVIDIA
runtime configured, and an available NVIDIA GPU, run:

```sh
tools/inference-integration-test.sh
```

Set `CUSCO_GPU_DEVICE_ID`, `CUSCO_MODEL_DIR`, or `CUSCO_RESULT_DIR` to override
those defaults. The command evaluates four prompts with the real Gemma model,
captures each complete execution checkpoint, displaces the active slot,
restores through host memory, and verifies both the inferred token and every
logit bit against uninterrupted execution. It also verifies that cancellation
and a deliberately failed restore preserve the prior binding. The command
exits nonzero on any failed assertion and prints a short Markdown report with
the prompts, inferred token IDs, checkpoint sizes, and exactness results.

## Run the mapped-execution proof

With the validation GGUF and NVIDIA runtime available, run:

```sh
docker compose run --rm mapped-proof
```

The command forks four device-resident sequence mappings, activates and
continues each one with identical results, verifies zero bytes copied by
activation, and writes staged-versus-mapped timing, bytes, and prompt work
avoided to `results/phase5.json`. The proof does not claim graph/cache reuse:
llama.cpp's public API does not expose whether its backend rebuilt a graph.

## Run the real-model server report

To exercise authenticated HTTP inference and a successive opaque-context
continuation through one persistent mapped executor, run:

```sh
tools/server-inference-report.sh
```

The workflow verifies token streaming, durable context identity, and reuse of
at least one complete mapped block on the second request. It writes the
machine-readable evidence to `results/phase6a-server.json`. GPU 1 and port
18082 are defaults; `CUSCO_GPU_DEVICE_ID`, `CUSCO_SERVER_REPORT_PORT`,
`CUSCO_MODEL_DIR`, and `CUSCO_RESULT_DIR` override them.


## Run the minimal server

The unauthenticated development provider is restricted to loopback:

```sh
cargo run -p cusco -- serve \
  --listen 127.0.0.1:8080 \
  --state ./data/cusco-state.json
```

Pass `--bearer-token` to require authentication. Listening anonymously on a
non-loopback address is rejected unless
`--unsafe-public-unauthenticated` is explicitly supplied.

The checked OpenAPI document is served at `/openapi.json`. OpenAI-compatible
entry points are `/v1/completions`, `/v1/chat/completions`, and `/v1/models`.
Native `/native/models`, `/native/contexts`, and `/native/requests` operations
cover model lifecycle, durable contexts and branches, imports, and
cancellation. Server inference uses registered local model paths and does not
implicitly fetch models.

## Future goals

Development is planned to proceed from the proven mapped-execution boundary
toward:

- integration of mapped execution with live server scheduling;
- production hardening of the inference and model-management server;
- broader compatibility, operational hardening, and recovery behavior;
- optional semantic context compaction once the underlying state system is proven reliable.

Each stage is intended to remain gated by correctness and measurable capacity results. The full design and phased acceptance criteria are documented in `docs/outline.md`.
