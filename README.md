# Cusco

Cusco is an experimental persistent, tiered, branch-aware model-state server built around llama.cpp. Its goal is to make model execution state a durable resource that can be captured, displaced from scarce accelerator memory, restored later, and continued exactly.

The project separates responsibilities deliberately: Rust manages logical
contexts, scheduling, storage tiers, and transactional state, while a narrow
native boundary lets llama.cpp retain ownership of model-specific execution
details and optimized kernels.

## Features

- Exact checkpoint capture, displacement, restoration, and continuation through
  a versioned llama.cpp C ABI.
- Persistent logical contexts with branching and evaluated-prefix reuse.
- Device, host, and storage residency accounting with mapped llama sequence
  activation and local spill support.
- Dynamic model registration, immutable revisions, aliases, loading, reloading,
  retirement, and removal.
- Priority-aware request scheduling with bounded admission, cancellation,
  deadlines, output backpressure, and diagnostic records.
- OpenAI-compatible completions, chat completions, and Responses APIs.
- An Ollama-compatible model-management profile for discovery, inspection,
  pulling, aliases, deletion, and residency reporting.
- Cusco-native context, compaction, lifecycle, status, and OpenAPI endpoints.
- Buffered and streaming generation, deterministic sampling controls, stop
  handling, and request-shape validation. Optional `auto`/`none` tool
  definitions are accepted for text generation but are not yet passed to the
  model; required or explicitly selected tool calls remain unsupported.
- Deterministic `window_tail` context compaction with transactional successor
  publication and replay metadata.
- Versioned YAML configuration, bearer authentication, transport diagnostics,
  SQLite model catalog, and production Docker Compose packaging.

## Architecture

Rust owns logical contexts, request scheduling, model residency policy,
admission, persistence, and protocol adapters. A narrow versioned C ABI owns
llama.cpp model execution, tokenization, sampling, sequence mappings, and
checkpoint import/export.

The server keeps installed-model and lifecycle metadata in SQLite and currently
keeps logical context records in its local JSON state. Native executor mappings,
active requests, queues, and spill contents are process-local and are rebuilt
or discarded after restart.

The workspace is split into focused crates:

- `context-store` provides immutable token branches and evaluated-prefix
  mappings;
- `physical-manager` tracks tier capacity, representations, transitions, and
  active bindings;
- `executor-sys` and `executor` expose the native ABI safely to Rust;
- `model-registry` resolves immutable Hugging Face artifacts and local models;
- `server` contains inference services, scheduling, residency, persistence, and
  HTTP adapters;
- `cli` provides proof, model-management, and serving commands.

## Platform support

Cusco currently targets Linux. Native Windows and macOS support is not
currently planned; development, packaging, and verification assume a Linux
host and the documented Docker Compose workflow.

## Run the acceptance gates

The canonical GPU-less contract gate is:

```sh
docker compose -f compose.test.yaml run --rm acceptance-model-free
```

The canonical real-model/GPU gate is:

```sh
docker compose -f compose.test.yaml run --rm acceptance
```

Both replay the versioned test set in `config/acceptance.json`. The real gate
runs executor continuation, mapped publication, representation scaling,
sustained scheduler, and API smoke workloads, then writes a provenance-indexed
`results/acceptance-report.json`. Focused proof services remain useful
diagnostics, but do not replace the complete gate.

## Run the OpenAI SDK conformance gate

Bootstrap the exact reviewed `oai-lens` revision, then run it against a Cusco
server listening on the configured host port:

```sh
tools/fetch-oai-lens.sh
CUSCO_OAI_LENS_TOKEN=your-token \
  docker compose -f compose.test.yaml run --rm oai-lens
```

The runner source remains in the ignored `.tools/oai-lens/` checkout. The
complete upstream exchange report is written to `results/oai-lens-report.json`;
`results/oai-lens-gate.json` records its digest, runner provenance, probe
counts, and changes from `config/oai-lens-expectations.json`. Probe failures
are initially advisory and leave the gate successful. A runner, configuration,
reporting, or artifact failure is a blocking harness failure.

## Run the representation measurement proof

With the validation GGUF and NVIDIA runtime available, run:

```sh
docker compose -f compose.test.yaml run --rm representation-proof
```

The versioned workload in `config/representation-workload.json` crosses several
represented-prefix boundaries. The proof requires exact source-versus-successor
tokens and logits before reporting publication latency and payload-copy deltas
for each boundary. It also verifies export/import continuation and records graph
recapture telemetry as unsupported when the backend exposes no truthful signal.
The reusable correctness and performance artifact is written to
`results/representation-proof.json`.

The narrower `mapped-proof` service remains available as a staged-restore versus
mapped-activation diagnostic. It writes `results/mapped-proof.json` and does not claim
kernel-level graph or cache reuse.

All real-model gates share one cached artifact:
`hf://unsloth/gemma-4-E2B-it-GGUF/gemma-4-E2B-it-Q3_K_M.gguf`.
Fetch it once with `docker compose -f compose.test.yaml run --rm model-fetch`.
Proof services accept the `hf://` identity directly, and the smoke gate installs
that identity through the model-management API. The registry resolves and
validates the cached local artifact internally; tests do not depend on cache
layout or create copied, hard-linked, or symlinked paths. The model supports
vision and tool use and is the standard fixture for both capability gates.


## Run the unified API smoke report

```sh
CUSCO_GPU_DEVICE_ID=0 docker compose -f compose.test.yaml run --rm api-smoke
```

This deterministic, model-backed smoke suite exercises buffered OpenAI
generation plus the primary Cusco context, compaction, and status APIs against
the standard Gemma test model. It writes `results/smoke-report.json` with
per-scenario status and latency, request/token totals, and aggregate
min/median/max latency. The executor and mapped-execution Compose proofs are
available as lower-level engineering diagnostics.

## Run the production Compose service

Copy the fully documented `config.example.yaml` to the ignored root
`config.yaml`, configure `config/user.yaml` with local model declarations,
provide an authentication token, and start the release service:

```sh
cp config.example.yaml config.yaml
mkdir -p data/models data/db data/spill data/user-models
CUSCO_BEARER_TOKEN='replace-with-a-secret' docker compose up --build -d server
```

The production definition mounts the root `config.yaml` read-only, listens on
`127.0.0.1:8080` by default, persists the migrated SQLite catalog under
`data/db`, and uses bounded storage under `data/spill`. The typed Responses
lifecycle has a file-backed resource store beside the database under
`data/db/responses`; current `store: false` requests do not publish there,
while the store provides atomic-write and deterministic-recovery foundations
for durable response resources. Override `CUSCO_LISTEN_ADDRESS`, `CUSCO_PORT`,
and `CUSCO_GPU_DEVICE_ID` as needed.
The entire `data/` tree is ignored by Git and excluded from image build
contexts.

## Run the daemon directly

The daemon accepts one versioned configuration path:

```sh
cargo run -p cusco -- serve --config ./config.yaml
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

The checked OpenAPI documents are served at `/openai/v1/openapi.json` and
`/cusco/v1/openapi.json`. OpenAI-compatible inference endpoints live under
`/openai/v1/*`. Cusco lifecycle and diagnostics endpoints live under
`/cusco/v1/*`, including an Ollama-compatible model-management profile under
`/cusco/v1/api/*`. The standalone `/ollama/*`, former unprefixed `/v1/*`, and
`/native/*` routes do not exist; the management profile intentionally does not
provide `/api/chat` or `/api/generate`.

For OpenWebUI, keep two connections configured:

- an OpenAI connection with base URL `http://HOST:PORT/openai/v1`, enabled for
  normal inference;
- an Ollama connection with base URL `http://HOST:PORT/cusco/v1`, normally
  disabled.

Enable the Ollama connection only while pulling, inspecting, copying, deleting,
or checking residency for models, then disable it and refresh OpenWebUI's model
list. Duplicate model entries may appear while both connections are enabled.
The Ollama-shaped endpoints are a management compatibility profile, not a
second inference API.

## Known limitations

- Each resident model currently owns one native execution slot, and the
  workload scheduler serializes native quanta across the process.
- Image inputs are validated and bounded, but inference rejects them because
  the executor does not yet expose a compatible vision-projector path.
- Embeddings are not available until the executor exposes embedding output.
- Physical tier accounting does not yet correspond to independently movable
  native KV and recurrent-state blocks.
- Context records currently use whole-state JSON persistence rather than the
  SQLite catalog used for installed models and lifecycle operations.
- The bundled execution profile targets the standard Gemma validation model;
  broader architecture support requires explicit compatible profiles.

The detailed design is documented in `docs/outline.md`. A source-grounded
assessment of architectural and performance constraints is available in
`docs/dump/audit.md`.
