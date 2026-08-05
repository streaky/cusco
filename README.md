# Cusco

Cusco is an experimental persistent, tiered, branch-aware model-state server built around llama.cpp. Its goal is to make model execution state a durable resource that can be captured, displaced from scarce accelerator memory, restored later, and continued exactly.

The project separates responsibilities deliberately: Rust will manage logical contexts, scheduling, storage tiers, and transactional state, while a narrow native boundary lets llama.cpp retain ownership of model-specific execution details and optimized kernels.

## Current state

Cusco has completed Phases 1 through 7. The executor proof established exact
checkpoint continuation for a hybrid/recurrent Gemma model, and the Rust layers
now provide durable logical contexts, capacity-accounted physical state,
transactional mapped activation, an authenticated HTTP API, immutable model
epochs, bounded live inference, and dynamic model residency.

Phase 7 replaces the one-model process with a capacity-admitted residency
scheduler. Executor-reported operating points account model weights, context
capacity, and device/host placement; model loads, epoch reloads, retirement,
and unload use transactional publication and active-reference draining.
Pressure selects idle LRU victims, inactive mapped contexts can spill through
the native sequence-state ABI and restore exactly, and `/native/status`
reports configured budgets, resident epochs, and lifecycle/tier metrics.

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

## Run the Phase 6C acceptance report

With the validation GGUF and NVIDIA runtime available, run:

```sh
CUSCO_GPU_DEVICE_ID=0 tools/phase6c-report.sh
```

Choose an available GPU index. `CUSCO_MODEL_DIR`, `CUSCO_RESULT_DIR`, and
`CUSCO_PHASE6C_REPORT_PORT` override the remaining defaults. The workflow runs
the GPU-less 80%-per-file coverage gate, exact executor and mapped-execution
proofs, and an authenticated live-server matrix. It injects cancellation,
deadline, and queue overload, verifies mapped continuation and accounting,
performs a graceful stop and restart, and confirms that durable contexts—but
not ephemeral queue state—recover. Machine-readable evidence, including
configuration, provenance, selected GPU UUID, exactness results, prompt phase
timings, cache work, transfers, and device/host accounting, is written to
`results/phase6c-server.json`.

## Run the Phase 7 residency report

With the validation GGUF and NVIDIA runtime available, run:

```sh
CUSCO_GPU_DEVICE_ID=0 tools/phase7-report.sh
```

The workflow runs the GPU-less per-file coverage gate, then exercises the live
resident server with two immutable model identities, a transactional epoch
reload, model removal, accounting checks, graceful restart, and post-restart
inference. Machine-readable GPU, epoch, resident-set, capacity, lifecycle, and
latency evidence is written to `results/phase7-server.json`.


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

Development now proceeds from dynamic residency toward:

- workload scheduling and operational hardening;
- broader protocol and model compatibility, recovery, and deployment behavior;
- optional semantic context compaction once the underlying state system is proven reliable.

Each stage is intended to remain gated by correctness and measurable capacity results. The full design and phased acceptance criteria are documented in `docs/outline.md`.
