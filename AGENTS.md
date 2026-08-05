# Agent guide

## Project direction

Cusco is intended to become a persistent, tiered, branch-aware model-state server built around llama.cpp. Rust owns logical contexts, scheduling, tiering, and transactional state; a narrow native C ABI around llama.cpp owns model-specific execution and checkpoint mechanics.

The detailed design, phase plan, and acceptance gates live in `docs/outline.md`. Treat that document as the source of truth for intended behavior, but verify the current implementation before changing it.

Keep this `AGENTS.md` up to date whenever development workflows, architecture, supported targets, project state, or repository conventions change.

## Current state

Phases 1 through 5 are implemented. The real Gemma executor proof demonstrated exact checkpoint continuation. The logical context store owns shared token branches and transactional mappings. The physical manager owns tier capacity, representations, transfers, bindings, prepared transitions, eviction, observability, and committed device block tables. The minimal server adds durable opaque external context IDs, atomic state recovery, model lifecycle APIs, bounded transition-cost scheduling, shared streaming events, canonical usage, cancellation/deadlines, authentication policy, OpenAI completion/chat adapters, and checked OpenAPI. Phase 5 adds transactional llama.cpp sequence mappings, reference-only activation, byte-copy metrics, and a real-GPU staged-versus-mapped proof. Graph/cache reuse remains unmeasured because llama.cpp exposes no public rebuild signal.

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

The Phase 4 server remains minimal: it loads a registered llama.cpp model per inference request and has not yet integrated live executor slots with the physical manager or Phase 5 mapped mappings. Do not represent production hardening or later phases as implemented. Phase 6 is the next architectural work and is divided into three reviewable milestones under one acceptance gate: 6A integrates one manually authored, schema-validated Gemma profile with a persistent mapped executor and exact admission; 6B adds the incremental normalized generation frontier; and 6C adds bounded queueing, cancellation, deadlines, shutdown, and the complete real-GPU acceptance artifact. Dynamic residency, concurrent native execution, compatibility breadth, and production hardening follow later.

## Build and dependency conventions

- Docker Compose is the primary development, test, and proof interface. Keep CPU-only and GPU execution supported by the same image; CPU-only checks should omit GPU passthrough rather than use a separate build.
- Local builds must support CUDA architectures `sm_61` and `sm_70`. Use `CUSCO_CUDA_ARCHITECTURES="61;70"` for normal local builds.
- Reserve the broad, full CUDA architecture build for production releases. Do not spend local development time compiling every supported CUDA target unless release validation specifically requires it.
- `llama.cpp-version.txt` is the sole source of truth for the llama.cpp version. It contains a release tag only. Build and fetch tooling must read it; never duplicate the tag or record the corresponding commit hash.
- Keep llama.cpp changes behind the versioned C ABI in `native/include/cusco_executor.h` (currently ABI version 3). Rust should not depend directly on unstable llama.cpp internals.
- Model files and generated proof results are local artifacts and must not be committed.

A normal local image build is:

```sh
docker compose build --build-arg CUSCO_CUDA_ARCHITECTURES="61;70"
```

Select the proof GPU with `CUSCO_GPU_DEVICE_ID`; do not assume a particular host GPU index is available.

## Testing and verification

- Write behavioral tests alongside permanent changes.
- Every measured Rust source file must maintain at least 80% line coverage. `tools/coverage.sh` runs the tests and enforces the per-file threshold.
- Run the GPU-less coverage path with `docker compose run --rm test`.
- Executor-boundary changes require the real model proof, not only the deterministic model-free tests. The proof uses the external `models/gemma-4-e2b-it.gguf` asset and writes a machine-readable result under `results/`.
- Mapped-executor changes require `docker compose run --rm mapped-proof`; it writes staged-versus-mapped evidence to `results/phase5.json`.
- Verify failure behavior transactionally: cancellation, preparation failure, transfer failure, validation failure, and commit failure must leave the prior binding usable.
- For behavioral work, exercise the changed path end to end. A successful compile alone is not sufficient.

## Engineering expectations

- Preserve the ownership boundary: Rust owns policy and logical state; llama.cpp owns model-specific tensors, graphs, and kernels.
- Prefer transactional prepare/commit operations. Never invalidate a working binding before replacement state is fully prepared and validated.
- Keep slots disposable and logical contexts durable.
- Use exact comparisons in deterministic checkpoint validation; do not weaken bitwise-logit or token equality into approximate checks.
- Reuse existing repository patterns and keep changes scoped to the active implementation phase.
- Update tests, high-level documentation, and this guide when a change makes them inaccurate.
