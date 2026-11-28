# Cusco service and verification guide

Run the commands below from the Cusco repository root. For a first run, start with the [quick start](../README.md#quick-start).

## Architecture

The executable application is `server/package.toml`. Its Terrane `main` loads
the Rust backend, binds the listener, mounts Terrane protocol controllers through
`packages/http-server`, and coordinates graceful shutdown. `packages/template`
compiles the HTML templates into the executable. There is no second backend
process, proxy hop, or endpoint-dispatch JSON bridge between the two languages.

`server/src/site.trn` owns the public `/` and `/health` GET/HEAD handlers.
`inference*.trn` owns OpenAI completion/chat/Responses and context-update HTTP
contracts; `management.trn` owns management routes. `policy.trn` owns authorization
and HTTP errors, and `trace.trn` owns diagnostic redaction. Generic body transport,
backpressure, disconnect finalization, and frame observation live in the HTTP package.
The homepage and health endpoint expose no private service state.

Rust exposes typed generation sessions, admission permits, management operations,
and domain results without a Rust HTTP router or HTTP request/response types.
It owns logical contexts, request scheduling, model residency policy,
admission, persistence, prompt preparation, and semantic generation finalization.
A narrow versioned C ABI owns
llama.cpp model execution, tokenization, sampling, sequence mappings, and
checkpoint import/export.

The wall deadline includes planning, hosted search, queueing and execution.
Each inference's active-execution allowance starts only after its admission;
queue wait does not consume it. Hosted-search selection gets its own active
allowance, while selection and provider work also retain the parent wall
deadline and cancellation. Requests invoking hosted search when no provider
is enabled are rejected as `400 invalid_request` before selection inference.
Hosted web search supplies evidence and citation guidance for the final answer;
only declared function tools add function-call JSON instructions.

Unknown request fields are rejected as `400 invalid_request` with the offending
field name; a non-object request reports that an object is required.

The server keeps installed-model and lifecycle metadata in SQLite and currently
keeps logical context records in its local JSON state. Native executor mappings,
active requests, queues, and spill contents are process-local and are rebuilt
or discarded after restart.

Stored Responses are projected as ordinary OpenAI response resources and remain
replayable after restart. Extension-aware clients can discover
`cusco.context_update.v1` at `GET /cusco/v1/capabilities` and submit a typed
`fold` to `POST /cusco/v1/context-updates`. A successful fold atomically
publishes a new stored response with a monotonic lineage revision and portable
message output; stale revisions and competing successors return `409`, while
validation or cancellation before commit leaves the base resource unchanged.
Clients that do not negotiate the extension continue to use standard stored
Responses and `previous_response_id`.

Output-budget exhaustion is retained as `status: incomplete`, including stored
retrieval and `incomplete_details.reason: max_output_tokens`. Such finished
partial responses remain valid lineage predecessors. Streaming distinguishes
`response.incomplete` from `response.completed`; function-call streams announce
empty arguments before their delta/done events so SDK accumulation is portable.

Management results, stored Responses and typed session events cross the native
Serde-to-document boundary with a fixed 16 MiB encoding budget. Oversized values
fail with `state_error` (HTTP 500 before headers), never truncated success. In
particular, context listings include token vectors and are not paginated; keep
their aggregate size within this bound. Pagination or smaller context summaries
would require a separate API contract change; neither is implemented here.

The workspace is split into focused crates:

- `context-store` provides immutable token branches and evaluated-prefix
  mappings;
- `physical-manager` tracks tier capacity, representations, transitions, and
  active bindings;
- `executor-sys` and `executor` expose the native ABI safely to Rust;
- `model-registry` resolves immutable Hugging Face artifacts and local models;
- `server` contains HTTP-free inference services, scheduling, residency, persistence,
  and typed host-application APIs;
- `cli` provides proof and model-management commands; it no longer runs the daemon.

Cached-ratio encoding retains a bounded scalar parse: the exact integer and
decimal constructors accept disjoint normalized number domains, and the core
document API does not expose a general float constructor. Substituting the
decimal constructor alone breaks ordinary ratios of zero or one.

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

The model-free gate executes the Terrane controller suite through the pinned
official OpenAI SDK as well as selected Rust contracts. Rust selectors must
match at least one executable test; an empty filter is a gate failure.
`docker compose -f compose.test.yaml run --rm test` merges Rust unit-test and
instrumented protocol-fixture coverage before enforcing the 80% per-file
threshold. The extracted host-boundary files must appear in that report.

## Run the OpenAI SDK conformance gate

Bootstrap the reviewed `oai-lens` revision. Compose starts a dedicated
model-backed server and waits for its model pull before launching the runner:

```sh
tools/fetch-oai-lens.sh
docker compose -f compose.test.yaml run --rm oai-lens
```

The runner source remains in the ignored `.tools/oai-lens/` checkout. The
complete upstream exchange report is written to `results/oai-lens-report.json`;
`results/oai-lens-gate.json` records its digest, runner provenance, probe
counts, and changes from `config/oai-lens-expectations.json`. Probe failures
are initially advisory and leave the gate successful. A runner, configuration,
reporting, or artifact failure is a blocking harness failure.

Interpret advisory failures by supported contract, not just their total. File
uploads/search, shell/container tools, encrypted reasoning replay,
`parallel_tool_calls` and `prompt_cache_key` are not accepted capabilities.
Requests containing unsupported fields fail before inference; a failed cache
probe with such a field does not measure cache behavior. Vision projection is
unavailable at the native executor boundary. Deterministic fixture output is
not evidence of real-model structured output or function-call behavior.
Function parameter declarations are checked as Draft 2020-12 JSON Schemas;
valid declarations are not restricted to the structured-generation grammar
subset. Reclassify real-model probe output separately before claiming broader
compatibility, without relaxing the pinned expectations.

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
Import telemetry includes backend-specific validation and rollback copies.
The common lower bound is the imported payload size: a backend that validates
without disturbing active state need not perform a snapshot/restore cycle.

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
vision and tool use, but the current executor gate exercises tool use only;
image projection remains unavailable until the native projector path lands.


## Run the unified API smoke report

```sh
CUSCO_GPU_DEVICE_ID=0 docker compose -f compose.test.yaml run --rm api-smoke
```

This deterministic, model-backed smoke suite exercises buffered and streaming
OpenAI generation, strict structured output, required tool calls, response
lifecycle operations, and the primary Cusco context, compaction, and status APIs
against the standard Gemma test model. It writes `results/smoke-report.json`
with per-scenario status and latency, request/token totals, semantic-compaction
evidence, and aggregate min/median/max latency. The executor and
mapped-execution Compose proofs remain lower-level engineering diagnostics.

## Run the production Compose service

Choose a writable host data directory (`./data` by default). Put your daemon
configuration there as `config.yaml` and, if needed, local model declarations
as `user.yaml`. For a new installation:

```sh
mkdir -p "${CUSCO_DATA_DIR:-./data}"
cp config.example.yaml "${CUSCO_DATA_DIR:-./data}/config.yaml"
CUSCO_BEARER_TOKEN='replace-with-a-secret' docker compose up --build -d server
```

The production service mounts that single directory at `/data`, reads
`/data/config.yaml`, and creates missing storage directories. It listens on
`127.0.0.1:11443` on the host by default (port `8080` inside the container),
persists the SQLite catalog under `/data/db`, and uses bounded storage under
`/data/spill`. Responses with `store: true` are atomically published under
`/data/db/responses` and survive daemon restart; retrieval, deletion,
`previous_response_id` replay, and typed function-call output continuation use
those resources. Requests with `store: false` remain stateless and do not
publish there. Override `CUSCO_LISTEN_ADDRESS`, `CUSCO_PORT`, and
`CUSCO_GPU_DEVICE_ID` as needed.
The default `data/` and `data-dev/` directories are ignored by Git and excluded
from the Docker build context. Keep custom data directories outside the checkout:
setting `CUSCO_DATA_DIR` does not create a Docker or Git ignore rule. The test
stack uses `models/` and `user-models/` under the same Compose data root but
does not share the production catalog or spill directory.

The single writable mount also makes `user-models/` writable by the service.
For read-only operator weights, overlay `/data/user-models` with a separate
read-only bind mount in your Compose override.

The build context is the surrounding Terrane checkout, not Cusco alone. Compose
selects `projects/cusco/Dockerfile`; its Dockerfile-specific ignore rules include
only the compiler, required packages, and Cusco sources while excluding caches,
operator configuration, and data. Production copies `cusco-service`, llama.cpp,
ggml, cuBLAS, and NCCL into the matching CUDA base image. The NVIDIA container
runtime supplies the host driver; no SDK, compiler, checkout, or model
weights are shipped. The acceptance services use the same executable in the
development stage, where the build and proof tools remain available.
Runtime `.state/` directories are excluded at every depth, including package
database test volumes. Unrelated packages, projects, and tools remain excluded
when their parent directories are reopened for the source allowlist. Do not
change database-volume permissions to make them readable by the Docker builder.

## Run the daemon directly

`./run-host.trn` is a local development convenience, not a production supervisor
or a Docker entrypoint. For deployment, use the [production Compose service](#run-the-production-compose-service):
the image builds the binary ahead of time and starts `cusco-service` directly.

Build from the Cusco directory in the Terrane checkout, with `terrane` on PATH
and the sibling `packages/http-server` and `packages/template` checkouts present.
Install CMake, Ninja, Git, a C++ compiler, and the CUDA development
toolkit for GPU builds. Use a Terrane CLI with child process group and signal
support. `./run-host.trn` provisions Terrane's pinned Rust tools,
fetches the pinned llama.cpp source, and builds llama.cpp and the Cusco shim.
It runs `terrane fetch server` to prepare locked dependencies for the service
and its build-time consumers, then builds and starts Cusco. Build outputs stay under `target/` and
Terrane's `server/.trn/` cache. Subsequent runs rebuild changed inputs only.
Set `CUSCO_GGML_CUDA=OFF` for an explicit CPU build without the CUDA toolkit,
or `CUSCO_CUDA_ARCHITECTURES` to override the default native GPU architecture.

When `sccache` is on PATH, the launcher uses it for C, C++, CUDA, and Terrane's
cacheable Rust compilations. It starts a host daemon on a Unix socket accessible
to Terrane's isolated Cargo builds. Use this for trusted builds only: the
daemon runs compiler requests outside the sandbox. Set `TERRANE_SCCACHE=0`
to disable it. To inspect this daemon's statistics, run
`SCCACHE_SERVER_UDS="${SCCACHE_SERVER_UDS:-${XDG_RUNTIME_DIR:-$HOME/.cache}/cusco/sccache.sock}" sccache --show-stats`.

Cusco's native Cargo workspace deliberately inherits the surrounding Terrane
checkout's `rust-toolchain.toml`; run Cargo from Cusco or the checkout root.
The Terrane service uses `rust-toolchain = "pinned"`, so its Rust dependencies
compile with that Terrane version's exact stable compiler too. Docker copies the
same toolchain file before invoking Cargo instead of maintaining a separate
stable-version literal. The host launcher and Docker both run `terrane toolchain
install`. Nightly generates Rustdoc metadata, not the native service.

Run the launcher from the Cusco repository root. It passes `./data-dev/config.yaml`
to the service without changing its working directory or mounting `/data`. Relative
storage paths resolve from the config file's directory; absolute paths are unchanged.
`CUSCO_DATA_DIR` selects a different host directory. Install the
`sqlite3` CLI for the read-only compatibility check on an existing catalog.

**Migrating from the shell launcher?** Keep the original data directory and
its config intact. If its config or model records refer to `/data`, the
launcher refuses to start. Choose a fresh directory, copy the host example
there, then pull or re-import your models using host paths. The launcher does
not rewrite a Compose catalog. Do not run host and Compose against the same
database.

```sh
mkdir -p data-dev
cp config.host.example.yaml data-dev/config.yaml  # first setup only
./run-host.trn
```

By default the script builds the service incrementally with the host Terrane
toolchain and native libraries described above. To skip compilation, set
`CUSCO_SERVICE_BIN` to a host-compatible executable with its shared libraries
available to the loader. Relative executable paths resolve from the caller's
working directory. Flags such as `--http-debug safe` are forwarded.
Export `CUSCO_BEARER_TOKEN` in the shell; Compose's `.env` and port mapping
are not applied. The host example binds only `127.0.0.1:11443`.

The daemon accepts `--config PATH` (default `/data/config.yaml`), `--http-debug
off|safe|full`, and `--help`. `cargo run -p cusco -- serve` has been removed.
SIGINT/SIGTERM stop admission and new connections, allow in-flight responses to
drain, and cancel remaining inference after `server.shutdown_grace_ms`. Set the
container/process supervisor's stop timeout longer than that grace period.
Ctrl+C, TERM, and terminal hangup request a graceful stop. The launcher waits
for the service to finish; its stdout and stderr are relayed to the terminal.
The daemon is noninteractive and receives EOF on stdin.

For a model-free hybrid boundary check against the built executable:

```sh
python3 tools/test-hybrid-service.py "$(terrane build server)" -v
```

This uses temporary state and an actual native-linked service without loading a
model. It checks the Terrane HTML/health routes, authenticated API access,
context creation/retrieval, startup policy, and SIGTERM exit. Model-backed inference
and GPU acceptance still use the separate gates above.

For model-free protocol regressions against the same Terrane controllers, run
`python3 server/tests/test_protocol_service.py -v`. This builds a fixture using
a feature-gated deterministic Rust engine; it is not the production engine.
The driver stages the fixture with the production controller sources under
`server/.trn/protocol-fixture`; it does not maintain a second route implementation.
Small temporary copies are acceptable when they keep Terrane straightforward.
Typed Serde and canonical socket-address formatting still need native bridges.
The anonymous-listener policy uses the projected address's loopback predicate.
The fragment-joining helper also retains a preallocated native buffer: its output
can span the request, and current source `+` lowering allocates a new buffer per
iteration. Revisit it when source append/builder operations can reuse that buffer.

No storage migration is part of this cutover: the existing SQLite and file-backed
stores remain authoritative. `dbkit` is deliberately not introduced until a
Terrane-owned storage feature needs it.

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

If stderr writing fails, further diagnostic capture stops. Requests and graceful
draining continue; after shutdown, the service reports the saved diagnostic
failure through a non-zero process exit. This is deliberate for an explicitly
enabled diagnostic sink, not a claim that every request was captured.

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
- Hosted web search currently supports a SearXNG-compatible JSON backend only.
  Shell/container execution, page retrieval, file search, code interpretation,
  vector stores, and other server-executed tools are not available.
- Embeddings are not available until the executor exposes embedding output.
- Physical tier accounting does not yet correspond to independently movable
  native KV and recurrent-state blocks.
- Context records currently use whole-state JSON persistence rather than the
  SQLite catalog used for installed models and lifecycle operations.
- Native model/context allocation is measured for each immutable model,
  executor configuration, accelerator identity, and runtime provenance, then
  persisted in the SQLite catalog and reused for admission. The first load of a
  new profile still requires a conservative bootstrap reservation.

See the [design outline](outline.md) for the storage model and executor design.
