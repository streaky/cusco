# Cusco

Cusco is an experimental persistent, tiered, branch-aware model-state server built around llama.cpp. Its goal is to make model execution state a durable resource that can be captured, displaced from scarce accelerator memory, restored later, and continued exactly.

The project separates responsibilities deliberately: Rust will manage logical contexts, scheduling, storage tiers, and transactional state, while a narrow native boundary lets llama.cpp retain ownership of model-specific execution details and optimized kernels.

## Current state

Cusco has completed Phases 1 through 9. In addition to exact checkpoint
continuation, transactional mapped execution, multi-model residency, and
priority-aware scheduling, the server now exposes cleanly namespaced OpenAI,
Ollama, and Cusco APIs. Phase 9 adds strict compatibility request validation,
OpenAI- and Ollama-native streaming frames, Responses, bounded inline images,
tool and structured-output controls, immutable Hub resolution, a migrated
SQLite model catalog, versioned daemon configuration, and a production
Compose profile.

Model identity, immutable revision, aliases, and lifecycle operation records
survive restart. Logical contexts, active requests, queues, native execution
state, and spill state are intentionally disposable in the v1 restart model.

## Run the real-model inference integration test

The workflow uses
`hf://unsloth/gemma-4-E2B-it-GGUF/gemma-4-E2B-it-Q4_K_M.gguf`.
It keeps the immutable Hugging Face artifact under `models/cache` and
materializes the stable `models/gemma-4-e2b-it.gguf` test path. The cache is
reused across runs; the model is downloaded only when it is absent or invalid.
With Docker's NVIDIA runtime configured and an available NVIDIA GPU, run:

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
docker compose -f compose.test.yaml run --rm mapped-proof
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


## Run the Phase 8 scheduler report

With the validation GGUF and NVIDIA runtime available, run:

```sh
CUSCO_GPU_DEVICE_ID=0 tools/phase8-report.sh
```

The workflow runs the GPU-less 80%-per-file coverage gate and the versioned
real-GPU workload in `config/phase8-workload.json`. It records the isolated
baseline, mixed interactive/standard/batch results, cancellation and deadline
injections, post-workload capacity recovery, scheduler policy and counters,
fully drained attributed decision records, latency/starvation thresholds, build
and workload provenance, and selected GPU in `results/phase8-server.json`.

## Run the unified API smoke report

```sh
CUSCO_GPU_DEVICE_ID=0 docker compose -f compose.test.yaml run --rm api-smoke
```

This deterministic, model-backed smoke suite exercises the primary OpenAI,
Cusco context/compaction, and status APIs against the standard Gemma test model.
It writes `results/smoke-report.json` with per-scenario status and latency,
request/token totals, and aggregate min/median/max latency. The report is the
common high-level API gate; the phase-specific reports remain available for
focused scheduler, residency, and executor evidence.

## Run the production Compose service

Copy `config/config.yaml` to an operator-owned location, configure
`config/user.yaml` with local model declarations, provide an authentication
token, and start the release service:

```sh
mkdir -p data/models data/state data/spill data/user-models
CUSCO_BEARER_TOKEN='replace-with-a-secret' docker compose up --build -d server
```

The production definition mounts the versioned daemon configuration read-only,
listens on `127.0.0.1:8080` by default, persists the migrated SQLite catalog
under `data/state`, and uses bounded storage under `data/spill`. Override
`CUSCO_LISTEN_ADDRESS`, `CUSCO_PORT`, and `CUSCO_GPU_DEVICE_ID` as needed.
The entire `data/` tree is ignored by Git and excluded from image build
contexts.

## Run the daemon directly

The daemon accepts one versioned configuration path:

```sh
cargo run -p cusco -- serve --config ./config/config.yaml
```

Unknown, missing, invalid, or unsupported-version configuration is rejected
before model execution starts. `CUSCO_BEARER_TOKEN` may supply the secret
without placing it in the configuration file. Anonymous serving is restricted
to loopback unless the explicit unsafe-public setting is enabled.

HTTP transport diagnostics have three levels selected with
`--http-debug <off|safe|full>` or `CUSCO_HTTP_DEBUG`. An explicit CLI value
overrides the environment; the default is `off`. `safe` writes correlated
JSON-line request, response, and streaming-chunk records to stderr without
buffering the response. It omits request headers, redacts every JSON string
value, preserves only JSON structure and non-string scalars, and omits
non-JSON or body data above 64 KiB. `full` records the complete URI (including
the query), request and response headers, and unredacted request/response body
frames; non-UTF-8 values are represented as hexadecimal. Full mode exposes
credentials and generated content and must be enabled only in a controlled
diagnostic environment. Traced responses include the correlation ID in
`x-request-id`.

The checked OpenAPI documents are served at
`/openai/v1/openapi.json`, `/ollama/api/openapi.json`, and
`/cusco/v1/openapi.json`. OpenAI-compatible endpoints live under
`/openai/v1/*`; Ollama-native endpoints live under `/ollama/api/*`; durable
Cusco lifecycle and diagnostics endpoints live under `/cusco/v1/*`.
The former unprefixed `/v1/*` and `/native/*` routes do not exist.

## Future goals

Development now proceeds from the Phase 9 product contract toward:

- broader model compatibility and later execution-policy optimization;
- optional semantic context compaction once the underlying state system is proven reliable.

Each stage is intended to remain gated by correctness and measurable capacity results. The full design and phased acceptance criteria are documented in `docs/outline.md`.
