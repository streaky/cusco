# Agent guide

## Project direction

Cusco is intended to become a persistent, tiered, branch-aware model-state server built around llama.cpp. Rust owns logical contexts, scheduling, tiering, and transactional state; a narrow native C ABI around llama.cpp owns model-specific execution and checkpoint mechanics.

The detailed design, phase plan, and acceptance gates live in `docs/outline.md`. Treat that document as the source of truth for intended behavior, but verify the current implementation before changing it.

Keep this `AGENTS.md` up to date whenever development workflows, architecture, supported targets, project state, or repository conventions change.

## Current state

Phases 1 through 10 are implemented. The real Gemma executor proof demonstrated exact checkpoint continuation. The logical context store and physical manager own shared logical branches, tier capacity, transactional mappings, transfers, and bindings. Phases 5 through 8 add mapped execution, bounded live generation, multi-model residency, resumable execution sessions, priority-aware deficit round-robin with monotonic promotion, and real-GPU acceptance evidence. Phase 9 adds the clean `/openai/v1/*` and `/cusco/v1/*` API cutover, strict compatibility controls, native wire streaming, bounded text-plus-image admission, immutable Hub resolution, metadata-driven execution profiles, a migrated SQLite model catalog, versioned daemon configuration, and production Compose packaging. OpenAI is the sole inference surface; `/cusco/v1/api/*` is a bounded Ollama-compatible model-management profile without chat or generation, and no standalone `/ollama/*` routes exist. Phase 10 adds request-tied deterministic `window_tail` context compaction, bounded declarations, atomic successor publication, and OpenAI replay metadata. Only model identity, configuration, aliases, and lifecycle operation records survive restart in the v1 profile; contexts and native execution state are disposable.

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
- Test Compose services mount project-scoped `cargo-registry`, `cargo-git`, and `cargo-target` named volumes so repeated runs reuse downloaded crates and compiled artifacts. Preserve these mounts on new Rust-running test services; do not remove the volumes during routine cleanup. Production state must use the `data/models`, `data/state`, and `data/spill` bind mounts rather than named or anonymous volumes.
- Local builds must support CUDA architectures `sm_61` and `sm_70`. Use `CUSCO_CUDA_ARCHITECTURES="61;70"` for normal local builds.
- Reserve the broad, full CUDA architecture build for production releases. Do not spend local development time compiling every supported CUDA target unless release validation specifically requires it.
- `llama.cpp-version.txt` is the sole source of truth for the llama.cpp version. It contains a release tag only. Build and fetch tooling must read it; never duplicate the tag or record the corresponding commit hash.
- Keep llama.cpp changes behind the versioned C ABI in `native/include/cusco_executor.h` (currently ABI version 9). Rust should not depend directly on unstable llama.cpp internals.
- Model files and generated proof results are local artifacts and must not be committed.

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
- Real-model tests and proofs use the single canonical artifact `hf://unsloth/gemma-4-E2B-it-GGUF/gemma-4-E2B-it-Q3_K_M.gguf`. Pass that URI directly to proof and smoke interfaces; the model registry resolves its immutable revision, validates or populates the persistent cache under `${CUSCO_MODEL_DIR:-./models}/cache`, and supplies the resolved local path only at the executor boundary. Tests must not inspect the cache layout, copy, hard-link, symlink, or independently redownload the artifact. This Gemma model supports vision and tool use, so their real-model acceptance coverage should use the same artifact. Executor-boundary changes require the real model proof, not only deterministic model-free tests, and must write machine-readable evidence under `results/`.
- Mapped-executor changes require `docker compose -f compose.test.yaml run --rm mapped-proof`; it writes staged-versus-mapped evidence to `results/phase5.json`. Representation changes additionally require `docker compose -f compose.test.yaml run --rm representation-proof`; it replays `config/representation-workload.json` and writes exact-continuation, publication-scaling, copy, state-movement, and graph-telemetry evidence to `results/representation-proof.json`.
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
- Reuse existing repository patterns and keep changes scoped to the active implementation phase.
- Update tests, high-level documentation, and this guide when a change makes them inaccurate.
