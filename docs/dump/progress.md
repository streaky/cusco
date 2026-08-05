# Cusco progress and capability evidence

_Last updated: 2026-08-05 on branch `phase-4-minimal-server`._

## Executive summary

Cusco has completed the first four implementation gates in `docs/outline.md`:

1. **Phase 1 — executor proof:** a real hybrid/recurrent Gemma checkpoint was captured, displaced, copied through host memory, restored, and continued exactly across four logical contexts.
2. **Phase 2 — Rust logical context store:** Rust represents logical contexts, structurally shared token branches, dependency-aware evaluated-prefix mappings, transactional publication, and logical reference accounting.
3. **Phase 3 — tiered physical manager:** Rust accounts for device, pinned-host, and storage representations; guarded capacity; asynchronous transfer ownership; transactional binding transitions; deterministic eviction; and capacity observability.
4. **Phase 4 — minimal server:** an authenticated HTTP service exposes OpenAI completion/chat adapters and native model/context lifecycle APIs over bounded admission, durable opaque contexts, canonical usage, cancellation/deadline handling, and the real llama.cpp executor.

The strongest executor-boundary result remains the exact Phase 1 checkpoint proof. A second repeatable GPU-backed report now exercises the Phase 4 server end to end: authenticated model registration, a complex 72-token streaming completion through the real Gemma model, OpenAPI route checks, canonical usage, and client/server timing.

Cusco remains experimental. The Phase 4 server persists its catalog and logical contexts across process restarts, but its SSE events are currently buffered until generation completes, and live executor slots are not yet integrated with the Phase 3 physical manager. It does not establish production readiness, sustained-load performance, or a stable public API.

## Capability status

| Capability | Status | Evidence |
|---|---|---|
| Versioned native C ABI around llama.cpp | Demonstrated | The current ABI is version 2; declarations and ownership contracts, including token-to-piece conversion, live in `native/include/cusco_executor.h`. |
| Composite checkpoint capture and restore | Demonstrated on the pinned Gemma artifact | `results/phase1.json` records four restored contexts with non-empty checkpoints. |
| Exact token continuation | Demonstrated | `token_equal: true` for all four contexts in `results/phase1.json`. |
| Bitwise-identical logit continuation | Demonstrated | `logits_equal: true` for all four contexts in `results/phase1.json`; the proof uses exact float-bit comparison rather than tolerance-based comparison. |
| Slot displacement before restore | Demonstrated | The Phase 1 proof replaces each executor's active state with unrelated tokens before restoration and comparison. |
| Device-to-host-to-device checkpoint movement | Demonstrated | `host_round_trip: true` in `results/phase1.json`. |
| Cancellation preserves the active binding | Demonstrated | `cancellation_preserved_binding: true` in `results/phase1.json`. |
| Failed promotion preserves the active binding | Demonstrated | `failed_promotion_preserved_binding: true` in `results/phase1.json`. |
| Concurrent logical-context proof | Demonstrated | Four independently owned executor contexts complete the proof; all four recorded comparisons pass. |
| Locked model identity and digest verification | Implemented and exercised | The proof artifact records the immutable `hf://` identity, expected SHA-256, local path, and file size. The CLI rejects a mismatched digest before executor startup. |
| Persistent/immutable token sequence structure | Implemented and tested | `crates/context-store` uses shared `Arc` sequence nodes; branch tests verify shared prefixes and isolated tails. Here “persistent” describes the immutable data-structure property, not disk persistence. |
| Stable content-derived branch identity | Implemented and tested | Branch IDs are SHA-256-derived from the prior branch hash, token, and fixed-width position encoding, so identities are portable across 32-bit and 64-bit targets. |
| Evaluated-prefix dependency identity | Implemented and tested | Mapping identity includes model epoch, adapter epoch, parent mapping, branch prefix, fixed-width represented end, required components, evaluation parameters, and lineage hash. |
| Longest valid evaluated-prefix lookup | Implemented and tested | Lookup uses an epoch-and-branch index, then checks branch identity, required checkpoint components, and the complete parent dependency chain without materializing token vectors during validation. |
| Transactional mapping publication | Implemented and tested | Publication is prepared separately and committed only if the context's active mapping is unchanged; stale concurrent publication returns `PublicationConflict`. |
| Logical reference accounting and reclamation | Implemented and tested | Catalog, context, and dependent references prevent premature reclamation; dependency-chain cleanup is iterative so reclamation depth does not consume call stack. |
| Per-file Rust coverage enforcement | Active | `tools/coverage.sh` generates workspace coverage and `tools/check_coverage.py` rejects any measured Rust source file below 80% line coverage. |
| Physical device/host/storage tier manager | Implemented and tested | `crates/physical-manager` accounts for residency and guarded capacity, models transfers separately from prepared transitions, commits bindings revision-transactionally, and emits capacity metrics and structured trace events. |
| Durable cross-process context persistence | Implemented and tested | Phase 4 atomically persists the model catalog, logical contexts, branches, revisions, and opaque UUIDs; restart tests verify recovery and non-reuse. Physical representation storage remains separate ownership/accounting metadata. |
| Inference server, scheduler, streaming API | Implemented and GPU-exercised | `cusco-server` provides bounded admission, transition-cost selection, OpenAI completion/chat adapters, native lifecycle routes, shared stream events, cancellation/deadline handling, canonical usage, and checked OpenAPI. `results/phase4-server.json` records a real-model authenticated streaming run. |

## Phase 1 real-model proof

### Validated model

The locally retained proof artifact identifies the external validation model as:

- Identity: `hf://models/unsloth/gemma-4-E2B-it-GGUF@0314792d7f1f7e229411f620751375812bb9faf2/gemma-4-E2B-it-Q3_K_M.gguf`
- SHA-256: `90293b8cdaf9c973012bf4df8a1e92bde7d74ad66a4fe56cf905ccd563d660c5`
- Size: `3,356,037,216` bytes
- Reported vocabulary: `262,144` tokens
- Reported checkpoint components: global KV, sliding-window attention state, and recurrent state
- llama.cpp source tag: read from the repository-wide `llama.cpp-version.txt` source of truth

Model weights are external and are not committed to the repository.

### Recorded integration report

The machine-readable evidence is `results/phase1.json`. The human-readable report is generated and asserted by `tools/report-inference-proof.py`.

| Context | Prompt | Input token | Inferred next token | Checkpoint bytes | Checksum | Exact after restore |
|---:|---|---:|---:|---:|---:|:---:|
| 0 | `The capital of France is [0]` | 236,842 | 107 | 166,506 | 15,855,946,795,732,230,871 | PASS |
| 1 | `The capital of France is [1]` | 236,842 | 9,079 | 166,506 | 7,424,184,562,009,432,005 | PASS |
| 2 | `The capital of France is [2]` | 236,842 | 9,079 | 166,506 | 10,193,984,636,459,084,218 | PASS |
| 3 | `The capital of France is [3]` | 236,842 | 9,079 | 166,506 | 17,792,484,377,050,697,218 | PASS |

Additional observed outcomes:

- Four independent contexts: **PASS**
- Real model vocabulary available (`262,144` tokens): **PASS**
- Restored tokens identical in every context: **PASS**
- Restored logits bitwise-identical in every context: **PASS**
- Device-to-host-to-device round trip: **PASS**
- Cancellation preserved the prior binding: **PASS**
- Deliberately failed restore preserved the prior binding: **PASS**
- Total proof time: **100,009 ms**
- CUDA device: **NVIDIA GeForce GTX 1080 Ti**, compute capability 6.1
- Model placement: **36/36 layers offloaded to GPU**

The differing checksums demonstrate four distinct captured states. The different next-token IDs also show that the report records real prompt-dependent model evaluation rather than only exercising checkpoint serialization.

### Reproduction command

With the verified model at `models/gemma-4-e2b-it.gguf`, Docker's NVIDIA runtime configured, and GPU 1 available:

```sh
tools/inference-integration-test.sh
```

The script rebuilds the current executor image, runs the Compose proof service, verifies the configured model SHA-256, confirms GPU visibility, writes `results/phase1.json`, validates every assertion, prints the concise Markdown report above, and exits nonzero on failure. `CUSCO_GPU_DEVICE_ID`, `CUSCO_MODEL_DIR`, and `CUSCO_RESULT_DIR` override the defaults.

## Phase 2 logical-context-store evidence

The `cusco-context-store` crate implements:

- logical context IDs independent of execution slots;
- immutable token sequences with structurally shared prefix nodes;
- cheap branching at any valid token boundary;
- stable SHA-256-derived branch, lineage, and evaluated-prefix identities;
- model and adapter epochs that invalidate incompatible evaluated state;
- composite checkpoint masks for global KV, SWA, and recurrent components;
- parent-linked evaluated-prefix mappings;
- deterministic longest-valid-prefix selection through an epoch-and-branch index;
- two-step prepared/committed publication;
- conflict rejection when another publication wins first; and
- catalog, active-context, and dependent reference counts with iterative reclamation.

The focused test suite currently has nine tests, including a property test over arbitrary branch prefixes. On 2026-08-05 the following command passed:

```sh
cargo +1.85.1 test -p cusco-context-store
```

Observed result: **9 passed, 0 failed**.

The tests exercise:

- exact shared-prefix structure and private branch tails;
- arbitrary prefix preservation;
- atomic publication and epoch invalidation;
- stale concurrent-publication rejection;
- longest-valid-prefix behavior across diverging branches;
- dependency-aware reference reclamation; and
- rejection of incomplete, out-of-bounds, and invalid-parent publications.

## Phase 3 tiered-physical-manager evidence

The new `cusco-physical-manager` crate implements:

- content-independent physical representation IDs and explicit device, pinned-host, and storage residency;
- separate logical, active-binding, transition-reservation, transfer, and growth-reservation ownership;
- exact composite validation across global KV, sliding-window attention, and recurrent components;
- guarded growth and transition capacity plus opportunistic warm-device capacity;
- reference-only, non-destructive, quiesced, and recompute transition classes;
- asynchronous transfer completion whose ownership survives transition cancellation;
- revision-guarded atomic binding commits that preserve the prior binding on failure;
- device-to-host demotion and host/storage-to-device promotion;
- deterministic value/LRU warm-state eviction; and
- capacity metrics and structured prepared, committed, aborted, transfer, promotion, demotion, recomputation, and eviction events.

The focused suite passed **9 tests, 0 failed**. It covers guarded admission, deterministic eviction, host promotion and demotion, reference-only switching, stale revision rejection, failed and cancelled transitions, detached-transfer capacity reservation, transfer lifetime, recomputation, explicit unbinding/reference release, and composite completeness and boundary agreement.

## Phase 4 minimal-server evidence

The `cusco-server` crate and `cusco serve` command implement:

- authenticated OpenAI-compatible completion, chat, and model-list routes;
- native administrative model lifecycle and durable context/branch/import APIs;
- loopback-safe anonymous development mode and bearer authentication for public listeners;
- bounded admission and deterministic transition-cost slot selection;
- transactional durable context commits with cancellation and deadline rejection;
- canonical usage records and a shared started/token/usage/finished event sequence;
- checked OpenAPI publication; and
- real llama.cpp token generation and token-to-piece conversion through C ABI version 2.

The reproducible GPU report is:

```sh
tools/server-inference-report.sh
```

The latest observed run used the pinned Gemma artifact on an NVIDIA GeForce GTX 1080 Ti. It generated 72 tokens for a multi-sentence distributed-checkpoint prompt in **2,541 ms** of server-reported end-to-end time and **2,543.12 ms** of client wall time, or **28.335 generated tokens/s** when model loading and prompt evaluation are included. The first buffered SSE token was observed at **2,542.35 ms**. All report assertions passed: anonymous administration was rejected, authenticated registration and lookup succeeded, required OpenAPI routes were present, every generated token had one stream event, terminal usage/finished events were present, and model revision and timing were recorded.

The machine-readable result is `results/phase4-server.json`. This measurement is an end-to-end integration observation, not a decode-only benchmark: the server currently opens the model per request and emits buffered SSE events after generation.

## Test and coverage evidence

The repository's containerized gate is:

```sh
docker compose build test
docker compose run --rm test
```

The most recent observed full gate on 2026-08-05 passed all runnable workspace tests. The GPU-backed integration report was run separately because the coverage service intentionally has no GPU or external model requirement.

Per-file line coverage reported by that run:

| Rust source file | Line coverage |
|---|---:|
| `crates/cli/src/main.rs` | 85.82% |
| `crates/context-store/src/lib.rs` | 97.07% |
| `crates/executor/src/lib.rs` | 95.75% |
| `crates/model-registry/src/lib.rs` | 92.86% |
| `crates/physical-manager/src/lib.rs` | 98.82% |
| `crates/server/src/lib.rs` | 95.03% |

Every measured Rust source file exceeded the required **80%** threshold.

## Reproducibility and build constraints

- `llama.cpp-version.txt` is the sole llama.cpp version source and contains a release tag rather than a duplicated commit hash.
- Docker and executor fetch/verification tooling consume that version file.
- Local CUDA builds default to `sm_61` and `sm_70`, covering the GTX 1080 Ti and Tesla GV100-class devices used for development.
- Broader CUDA architecture lists remain configurable through `CUSCO_CUDA_ARCHITECTURES` for release builds.
- CPU-only tests use the same CUDA-capable image without GPU passthrough, with CUDA's stub driver library available for linking/loading.
- Model assets and generated proof results are mounted separately from the image build and are not required for model-free lifecycle and logical-store tests.

## Evidence index

| Evidence | Location |
|---|---|
| Phase 1 machine-readable real-model result | `results/phase1.json` |
| Phase 1 integration scenario | `crates/executor/tests/phase1.rs` |
| Proof orchestration and result generation | `crates/cli/src/main.rs` |
| Safe Rust executor/checkpoint API | `crates/executor/src/lib.rs` |
| Native ABI contract | `native/include/cusco_executor.h` |
| Native llama.cpp implementation | `native/shim/cusco_executor.cpp` |
| Logical context store and behavioral tests | `crates/context-store/src/lib.rs` |
| Tiered physical manager and behavioral tests | `crates/physical-manager/src/lib.rs` |
| Runnable real-model report | `tools/inference-integration-test.sh`, `tools/report-inference-proof.py` |
| Phase 4 machine-readable real-model server result | `results/phase4-server.json` |
| Runnable Phase 4 server report | `tools/server-inference-report.sh`, `tools/report-server-inference.py` |
| Minimal server, HTTP adapters, scheduler, persistence, and tests | `crates/server/src/lib.rs` |
| Immutable model registry | `crates/model-registry/src/lib.rs` |
| Containerized proof/test services | `compose.yaml` |
| Per-file coverage gate | `tools/coverage.sh`, `tools/check_coverage.py` |
| llama.cpp release tag source | `llama.cpp-version.txt` |
| Full phased design and acceptance criteria | `docs/outline.md` |

## Current boundary and next gate

The evidence now supports the executor-boundary hypothesis, logical-state and transactional physical-tier models, durable minimal-server state, authenticated model/context APIs, and real-model HTTP generation. It does not establish production readiness, sustained-load behavior, true incremental token delivery, or a stable public API. The current server opens a model for each request, buffers generation before emitting SSE events, and does not yet bind Phase 3 physical-manager transitions to live executor slots.

The next architectural gate is mapped execution if staged measurements justify it; otherwise the immediate work is integrating the minimal scheduler with the physical manager and then compatibility and production hardening. Any such work must preserve the transactional cancellation, capacity, persistence, and checkpoint guarantees already demonstrated.
