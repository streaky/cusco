# Agent guide

## Project direction

Cusco is intended to become a persistent, tiered, branch-aware model-state server built around llama.cpp. Rust owns logical contexts, scheduling, tiering, and transactional state; a narrow native C ABI around llama.cpp owns model-specific execution and checkpoint mechanics.

The detailed design, phase plan, and acceptance gates live in `docs/outline.md`. Treat that document as the source of truth for intended behavior, but verify the current implementation before changing it.

Keep this `AGENTS.md` up to date whenever development workflows, architecture, supported targets, project state, or repository conventions change.

## Current state

Cusco implements exact checkpoint continuation, shared logical branches, tier accounting, transactional mappings, mapped execution, bounded live generation, multi-model residency, resumable execution sessions, and priority-aware deficit round-robin scheduling with monotonic promotion. The public API provides OpenAI-compatible inference under `/openai/v1/*` and a Cusco control plane under `/cusco/v1/*`, including bounded Ollama-compatible model management under `/cusco/v1/api/*`. It also includes strict compatibility controls, native wire streaming, bounded text-plus-image admission, immutable Hub resolution, metadata-driven execution profiles, a SQLite model catalog, versioned daemon configuration, production Compose packaging, request-tied deterministic `window_tail` context compaction with atomic successor publication and OpenAI replay metadata, durable stored Responses continuation, and a negotiated transactional `cusco.context_update.v1` fold operation. Model identity, configuration, aliases, lifecycle operation records, and stored OpenAI Responses resources survive restart in the v1 profile; logical Cusco contexts and native execution state are disposable.

The native executor derives each loaded model's required ordinary-KV, sliding-window, and recurrent state components at runtime. Cusco does not maintain a family allowlist for execution-state geometry; Gemma 4 and Qwen 3.5 MoE retain exact checkpoint-continuation and mapped-execution proof coverage, while metadata-derived Qwen 3.5/3.6 MoE chat-template rendering supports OpenAI text generation. The canonical acceptance fixture remains Gemma.

The repository currently contains:

- `crates/context-store`: Rust logical contexts, structurally shared token sequences, and evaluated-prefix mappings;
- `crates/physical-manager`: physical representations, tier capacity, ownership references, transfers, bindings, and transactional transitions;
- `crates/executor-sys`: native FFI declarations and linking;
- `crates/executor`: safe Rust executor, token-piece, and checkpoint wrappers;
- `crates/model-registry`: immutable Hugging Face resolution and verified local registration;
- `crates/server`: protocol-neutral inference/model services, durable catalog, scheduler, auth, HTTP adapters, and OpenAPI;
- `crates/cli`: proof, model-management, and `serve` commands;
- `native`: the C ABI header and llama.cpp shim;
- `executor`: upstream and patch metadata;
- `tools`: fetch, verification, integration-report, and coverage helpers.

The server dynamically admits and reuses multiple model epochs within configured device, host, and storage budgets. Each resident model currently owns one native execution slot. The scheduler interleaves request-owned prefill and decode quanta with per-principal/class fairness within that slot, while distinct resident models can execute independently. Suspended sessions retain their model reference but release slot occupancy between native quanta; cancellation, deadline, and shutdown callbacks remain registered through each native ownership fence. HTTP transport diagnostics default off and support privacy-safe and fully unredacted levels through configuration; full mode exposes headers, query values, credentials, and body content. Compatibility is intentionally bounded: embeddings await executor support, images are admitted and retained but the current executor does not yet project them, and broader architecture coverage remains prospective.


## Build and dependency conventions

- Docker Compose is the primary development, test, proof, and production interface. `compose.test.yaml` owns development and verification services; every test or proof command must select it explicitly with `docker compose -f compose.test.yaml`. `compose.yaml` is reserved for the production-oriented `server` service, which runs the release binary with persistent bind mounts under the ignored `data/` tree. Keep CPU-only and GPU execution supported by the same image; CPU-only checks should omit GPU passthrough rather than use a separate build.
- Test Compose services mount project-scoped `cargo-registry`, `cargo-git`, and `cargo-target` named volumes so repeated runs reuse downloaded crates and compiled artifacts. Preserve these mounts on new Rust-running test services; do not remove the volumes during routine cleanup. Test and production services share the ignored `data/models` artifact cache by default so real-model verification does not duplicate downloads; `CUSCO_MODEL_DIR` may override that host path. Production state must use the `data/models`, `data/db`, and `data/spill` bind mounts rather than named or anonymous volumes.
- Local builds must support CUDA architectures `sm_61` and `sm_70`. Use `CUSCO_CUDA_ARCHITECTURES="61;70"` for normal local builds.
- Reserve the broad, full CUDA architecture build for production releases. Do not spend local development time compiling every supported CUDA target unless release validation specifically requires it.
- `llama.cpp-version.txt` is the sole source of truth for the llama.cpp version. It contains a release tag only. Build and fetch tooling must read it; never duplicate the tag or record the corresponding commit hash.
- Keep llama.cpp changes behind the versioned C ABI in `native/include/cusco_executor.h` (currently ABI version 15). Rust should not depend directly on unstable llama.cpp internals.
- Model files and generated proof results are local artifacts and must not be committed.
- Files matched by `.gitignore` are intentionally local artifacts. Never force-add, stage, or commit them; if an ignored artifact contains durable project guidance, move that guidance into an appropriate tracked document instead.
- `config.example.yaml` is the exhaustive, documented operator configuration template. Keep it synchronized with every supported configuration field and update its comments and sensible deployment defaults whenever the schema or behavior changes; `config.yaml` is the ignored operator-local copy mounted by production Compose. Verification services and harnesses must use the tracked `config/test.yaml`, whose test-specific limits and feature choices must not leak into the operator example.

A normal local image build is:

```sh
docker compose -f compose.test.yaml build --build-arg CUSCO_CUDA_ARCHITECTURES="61;70"
```

Select the proof GPU with `CUSCO_GPU_DEVICE_ID`; do not assume a particular host GPU index is available.

## Testing and verification

- Write behavioral tests alongside permanent changes.
 - Every measured Rust source file must maintain at least 80% line coverage. `tools/coverage.sh` runs the tests and enforces the per-file threshold.
 - Run the GPU-less coverage path with `docker compose -f compose.test.yaml run --rm test`.
 - The canonical GPU-less contract gate is `docker compose -f compose.test.yaml run --rm acceptance-model-free`. The canonical real-model/GPU gate is `docker compose -f compose.test.yaml run --rm acceptance`; it replays the versioned `config/acceptance.json` test set and writes `results/acceptance-report.json` with per-gate outcomes and complete available source, model, workload, build, runtime, and device provenance.
 - The focused API smoke, executor, mapped-execution, representation-measurement, and scheduler proofs remain diagnostics when changing those subsystems; they do not replace the canonical acceptance gates.
- Python tests of `/openai/v1/*` behavior must use the pinned official `openai` Python client for models, completions, chat completions, Responses, and streaming. Direct HTTP remains appropriate for Cusco control-plane routes and the generated OpenAPI document.
- Changes to `/openai/v1/*` behavior must run the pinned `oai-lens` SDK-conformance gate. Bootstrap it explicitly with `tools/fetch-oai-lens.sh`, then run `docker compose -f compose.test.yaml run --rm oai-lens` against the separately running candidate. Probe failures are advisory until their contracts graduate into blocking acceptance coverage; bootstrap, build, execution, report parsing, and artifact-preservation failures are blocking. Review `results/oai-lens-gate.json` against `config/oai-lens-expectations.json`. Never use a neighboring checkout or modify the ignored `.tools/oai-lens/` source.
- Real-model tests and proofs use the single canonical artifact `hf://unsloth/gemma-4-E2B-it-GGUF/gemma-4-E2B-it-Q3_K_M.gguf`. Pass that URI directly to proof and smoke interfaces; the model registry resolves its immutable revision, validates or populates the persistent cache under `${CUSCO_MODEL_DIR:-./data/models}`, and supplies the resolved local path only at the executor boundary. Tests must not inspect the cache layout, copy, hard-link, symlink, or independently redownload the artifact. This Gemma model supports vision and tool use, so their real-model acceptance coverage should use the same artifact. Executor-boundary changes require the real model proof, not only deterministic model-free tests, and must write machine-readable evidence under `results/`.
- Mapped-executor changes require `docker compose -f compose.test.yaml run --rm mapped-proof`; it writes staged-versus-mapped evidence to `results/mapped-proof.json`. Representation changes additionally require `docker compose -f compose.test.yaml run --rm representation-proof`; it replays `config/representation-workload.json` and writes exact-continuation, publication-scaling, copy, state-movement, and graph-telemetry evidence to `results/representation-proof.json`.
- Verify failure behavior transactionally: cancellation, preparation failure, transfer failure, validation failure, and commit failure must leave the prior binding usable.
- For behavioral work, exercise the changed path end to end. A successful compile alone is not sufficient.

## Warnings and linting

- New or modified code must not introduce compiler or linter warnings.
- Before considering a change complete, run `cargo check` and `cargo clippy` for the relevant crates, targets, and features through `compose.test.yaml`.
- Fix the underlying issue rather than suppressing its warning.
- Use `#[allow(...)]`, `#[expect(...)]`, or equivalent suppressions only when the warning is intentional, narrowly scoped, and the reason is clear from the surrounding code.
- Do not broaden an otherwise unrelated change solely to clean up pre-existing warnings.
- CI may enforce warning-free code with `cargo clippy -- -D warnings`.

## Engineering expectations

- Preserve the ownership boundary: Rust owns policy and logical state; llama.cpp owns model-specific tensors, graphs, and kernels.
- Prefer transactional prepare/commit operations. Never invalidate a working binding before replacement state is fully prepared and validated.
- Keep slots disposable. Logical contexts survive requests and slot changes but are disposable on daemon restart in the v1 profile.
- Use exact comparisons in deterministic checkpoint validation; do not weaken bitwise-logit or token equality into approximate checks.
- Reuse existing repository patterns and keep changes scoped to the requested behavior.
- Update tests, high-level documentation, and this guide when a change makes them inaccurate.
