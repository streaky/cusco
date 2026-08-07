# Agent guide

## Project direction

Cusco is intended to become a persistent, tiered, branch-aware model-state server built around llama.cpp. Rust owns logical contexts, scheduling, tiering, and transactional state; a narrow native C ABI around llama.cpp owns model-specific execution and checkpoint mechanics.

The detailed design, phase plan, and acceptance gates live in `docs/outline.md`. Treat that document as the source of truth for intended behavior, but verify the current implementation before changing it.

Keep this `AGENTS.md` up to date whenever development workflows, architecture, supported targets, project state, or repository conventions change.

## Current state

Phases 1 through 9 are implemented. The real Gemma executor proof demonstrated exact checkpoint continuation. The logical context store and physical manager own shared logical branches, tier capacity, transactional mappings, transfers, and bindings. Phases 5 through 8 add mapped execution, bounded live generation, multi-model residency, resumable execution sessions, priority-aware deficit round-robin with monotonic promotion, and real-GPU acceptance evidence. Phase 9 adds the clean `/openai/v1/*`, `/ollama/api/*`, and `/cusco/v1/*` API cutover, strict compatibility controls, native wire streaming, bounded text-plus-image admission, immutable Hub resolution, metadata-driven execution profiles, a migrated SQLite model catalog, versioned daemon configuration, and production Compose packaging. Only model identity, configuration, aliases, and lifecycle operation records survive restart in the v1 profile; contexts and native execution state are disposable.

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
- Keep llama.cpp changes behind the versioned C ABI in `native/include/cusco_executor.h` (currently ABI version 6). Rust should not depend directly on unstable llama.cpp internals.
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
- Phase 6 lifecycle or live-executor changes require `CUSCO_GPU_DEVICE_ID=<index> tools/phase6c-report.sh`; it runs coverage plus the executor, mapped, cancellation/deadline/overload, graceful-shutdown, and restart gates and writes `results/phase6c-server.json`.
- Phase 7 residency or model-lifecycle changes require `CUSCO_GPU_DEVICE_ID=<index> tools/phase7-report.sh`; it runs coverage plus the multi-model load/reuse/reload/remove/restart matrix and writes `results/phase7-server.json`.
- Real-model tests and proofs use `hf://unsloth/gemma-4-E2B-it-GGUF/gemma-4-E2B-it-Q4_K_M.gguf`. Run the Compose `model-fetch` service before them; it reuses the persistent host cache under `${CUSCO_MODEL_DIR:-./models}/cache`, validates the cached artifact, and only downloads when it is absent or invalid. The stable test path is `models/gemma-4-e2b-it.gguf`. Executor-boundary changes require the real model proof, not only deterministic model-free tests, and must write machine-readable evidence under `results/`.
- Mapped-executor changes require `docker compose -f compose.test.yaml run --rm mapped-proof`; it writes staged-versus-mapped evidence to `results/phase5.json`.
- Verify failure behavior transactionally: cancellation, preparation failure, transfer failure, validation failure, and commit failure must leave the prior binding usable.
- For behavioral work, exercise the changed path end to end. A successful compile alone is not sufficient.

## Engineering expectations

- Preserve the ownership boundary: Rust owns policy and logical state; llama.cpp owns model-specific tensors, graphs, and kernels.
- Prefer transactional prepare/commit operations. Never invalidate a working binding before replacement state is fully prepared and validated.
- Keep slots disposable and logical contexts durable.
- Use exact comparisons in deterministic checkpoint validation; do not weaken bitwise-logit or token equality into approximate checks.
- Reuse existing repository patterns and keep changes scoped to the active implementation phase.
- Update tests, high-level documentation, and this guide when a change makes them inaccurate.
