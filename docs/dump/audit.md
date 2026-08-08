# Phases 1–10 implementation and structural audit

**Audit date:** 2026-08-08; updated after the Phase 10 merge and API surface cutover

## Scope and interpretation

This audit compares the current repository with the intent and exit criteria in `docs/outline.md` for Phases 1 through 10. Later-phase decisions are treated as superseding earlier ones. In particular, Phase 9's v1 restart contract supersedes Phase 4's durable-context goal: model identity, installed-model metadata, configuration, aliases, and lifecycle operations should survive restart, while contexts and native execution state may be discarded.

The audit is based on the current source, Compose service graph, checked-in/local proof artifacts under `results/`, and the named tests and fixtures actually present in the tree. After the API cutover, the focused namespace/OpenAPI/management-route tests and the unified real-model smoke gate were rerun successfully; the smoke report recorded 13/13 passing scenarios. Existing historical Phase 6–8 artifacts do not by themselves establish that every current source change still passes those broader gates.

Severity means:

- **Critical:** the current structure contradicts a core ownership/correctness model or creates a scaling limit that will require invasive redesign.
- **High:** a promised product or operational contract is incomplete, or the implementation can become inconsistent under an ordinary failure.
- **Medium:** significant completeness, maintainability, or observability debt that should be addressed before the surface grows further.

## Executive summary

The repository contains a great deal of real implementation, not scaffolding: the native checkpoint proof, logical context machinery, mapped llama sequence reuse, bounded admission, streaming generation, model lifecycle, fairness policy, API adapters, SQLite catalog, Compose packaging, and deterministic `window_tail` compaction all exist.

It is not accurate, however, to say that Phases 1–10 are complete **as fully as the outline intended**. The largest gaps are below.

1. **The physical-state architecture is not yet real or atomic.** `PhysicalManager` mostly accounts for metadata; its transfers mutate tier flags and counters but move no executor bytes. The mapped server registers invented cumulative byte sizes unrelated to the actual native mapping. Logical publication can commit before physical publication and has no rollback.
2. **All model execution is globally serialized by one scheduler thread.** This defeats the Phase 7 requirement that distinct resident models execute independently and makes the Phase 8 scheduler a process-wide execution bottleneck.
3. **The logical context representation will scale poorly.** It allocates and hashes once per token, copies every full evaluated prefix, and repeatedly walks token ancestry. Successive publications of growing prefixes trend toward quadratic copied/hashed work and retained memory.
4. **Residency admission is based on estimates rather than trustworthy native capacity.** The native operating point proportionally assigns model bytes by layer count and treats serialized state size as context memory. The pre-load estimator is coarser still. The scheduler therefore cannot prove the capacity guarantees the outline requires.
5. **Phase 9 persistence is both broader and less transactional than its final contract.** Contexts are still reloaded from a JSON state file; every context mutation rewrites that whole file synchronously. Models are duplicated between JSON and SQLite, and a catalog-finalization failure can restore memory without restoring the already-written JSON file.
6. **The checked-in acceptance story has regressed.** The Phase 6/7/8 report scripts and Compose gates named by the outline no longer exist. The unified real-model smoke gate is useful but does not replace their mixed-load, streaming, management-lifecycle-fault, correlation, or transactional checks.
7. **Phase 10's mechanism is present, but its declared semantic and end-to-end acceptance matrix is not.** The fixture tree named by the outline is absent, semantic tests are structural token-retention checks rather than model-output regression checks, and the unified smoke does not exercise streaming ID correlation.

These are architecture issues worth resolving before adding multi-GPU placement, external compaction workers, more model families, or substantially higher concurrency. Those features would otherwise harden assumptions that currently need to change.

## Phase-by-phase status

| Phase | Assessment | Evidence and material gaps |
|---|---|---|
| **1. Executor proof** | **Substantially implemented** | `crates/executor/tests/phase1.rs`, the `executor-proof` Compose service, and `results/phase1.json` cover exact continuation, multiple logical contexts, host round-trip, cancellation, and failed-promotion preservation. The native boundary is real and versioned. This is the strongest match between intent and implementation. The local artifact is evidence of a prior run, not a freshly executed audit gate. |
| **2. Rust logical context store** | **Implemented with chunked storage** | Persistent branches, identities, epochs, deterministic longest-valid-prefix lookup, immutable prepared publication, references, and monotonic revisions exist in `crates/context-store/src/lib.rs`. The remediation now stores tokens in bounded immutable 256-token chunks, keeps mapping boundaries as structurally shared sequence snapshots rather than cumulative `Vec<Token>` copies, hashes only newly appended tokens, and confirms literal tokens without allocation after indexed lookup. Long-context allocation and latency benchmarks remain desirable acceptance evidence, but the token-per-node and cumulative mapping-copy architecture has been removed. |
| **3. Tiered physical manager** | **Partial** | Capacity classes, representations, transitions, references, eviction decisions, metrics, and trace records exist in `crates/physical-manager/src/lib.rs`. Typed lifetime ownership, real transfers, an outside-the-lock transfer coordinator, and real tier storage do not. Raw IDs and manual release methods remain the ownership API. |
| **4. Minimal server and scheduler** | **Broadly superseded; two goals remain incomplete** | Later APIs supersede the minimal routes and Phase 9 supersedes durable contexts. The current server has authentication, admission, streaming, context APIs, model management, usage, cancellation, deadlines, and OpenAPI. The intended protocol-neutral service boundary has collapsed into a large `lib.rs`, and independent benchmark-harness integration is not present as a runnable first-class gate. |
| **5. Mapped execution** | **Exact native-mapping proof complete; historical block-table claim not implemented** | `mapped-proof` and `results/phase5.json` compare staged and mapped paths. Native mappings use llama sequence IDs and can switch references, but the historical physical-block-table/kernel-resolution design is not present and current publication performs sequence copies. The selected Approach A remediation no longer treats a Rust-owned block table as the immediate completion criterion: it requires truthful opaque native ownership, copy-free linear publication, stable graph reuse, and meaningful copy telemetry. Native-allocated shared blocks remain the conditional B1 evolution. |
| **6. Live execution integration** | **Functional path present; capacity and transaction gates incomplete** | One-model mapped execution, bounded admission, incremental output, request-owned sampling/frontier state, stops, cancellation, deadlines, and shutdown machinery exist. The prior Phase 6 artifacts remain under `results/`. The current physical publication sequence can leave logical and physical state divergent, and capacity is based on fictitious representation sizes. The documented `tools/phase6c-report.sh` gate no longer exists. |
| **7. Residency and lifecycle scheduling** | **Dynamic lifecycle present; core independence/capacity goals incomplete** | Dynamic epochs, load/reuse/reload/remove/retire behavior, spill paths, status, and lifecycle tests exist in `residency.rs` and the catalog. Admission does not choose among executor-reported operating points, accounting is estimated, and all resident models ultimately execute through one scheduler worker. The documented Phase 7 report gate no longer exists. |
| **8. Workload scheduling and hardening** | **Fairness policy implemented; execution architecture and acceptance incomplete** | Priority-aware deficit round robin, FIFO equivalence, monotonic promotion, bounded prefill/decode quanta, IDs, diagnostics, and model-free tests exist in `scheduler.rs`. `WorkloadScheduler` synchronously invokes every session on one OS thread, so the policy also globally serializes unrelated model slots. Historical Phase 8 artifacts exist, but the report script/service named in the outline is gone and the unified smoke has no sustained mixed-load or fault workload. |
| **9. Compatibility, persistence, packaging** | **Broad but incomplete** | The clean surface now consists of OpenAI-compatible inference under `/openai/v1/*` and the Cusco control plane under `/cusco/v1/*`, including a bounded Ollama-compatible model-management profile under `/cusco/v1/api/*`. Standalone `/ollama/*` routes and Ollama inference operations have been removed. Strict controls, native streaming, model catalog, model resolution, config, and production/test Compose exist. Persistence violates the final disposable-context contract and is split across JSON and SQLite. Some blocking lifecycle work still runs on Tokio. Images are bounded and decoded but always rejected because no projector execution path exists. Compatibility breadth therefore falls short of the declared text-plus-image contract. |
| **10. Semantic context compaction** | **Baseline mechanism implemented; acceptance overclaimed** | The registry, deterministic `window_tail`, declarations, bounded workers, successor preparation/publication, replay metadata, terminal-sequence suppression, and compaction smoke scenarios exist. The named semantic-quality and end-to-end stream-correlation fixtures/tests in the outline do not. Current evidence proves deterministic trimming and continuation, not the full semantic quality/correlation matrix. |

## Critical structural and performance findings

### C1. Logical token storage now uses bounded immutable chunks

**Resolved**

- `PersistentTokenSequence` stores bounded immutable chunks of at most 256 tokens and allocates one `Arc` per chunk rather than per token (`crates/context-store/src/lib.rs`).
- Sequence identity is independent of append segmentation: each chunk carries an incremental SHA-256 state, so append work hashes only newly supplied tokens while arbitrary prefix identities remain deterministic.
- `EvaluatedPrefix` retains a structurally shared immutable sequence boundary instead of a cumulative `Vec<Token>`. Publication lineage hashes the parent lineage and logical boundary identity in constant metadata space.
- Longest-prefix lookup uses an epoch-scoped ordered boundary index rather than walking one node per token. A hash candidate is still confirmed by literal token iteration without allocating, preserving collision safety.
- Chunking remains a logical identity and sharing choice only. It makes no claim that a chunk is one native allocation or that executor components have identical physical geometry.

Focused tests cover bounded chunk formation, identity equivalence across different append segmentations, structural branch sharing, partial-chunk prefixes, collision confirmation, lineage validation, deterministic longest-prefix selection, and publication conflict behavior. A long-context benchmark should still record allocations, retained bytes, publication latency, and lookup latency before later cache-index work, but it is evidence work rather than an unresolved storage redesign.

### C2. The physical manager does not own physical data

**Evidence**

- A `PhysicalRepresentation` consists of IDs, byte counts, tier booleans, and reference counters (`crates/physical-manager/src/lib.rs:67-76`). There is no backing buffer, executor handle, transfer fence, or storage location.
- `complete_transfer` establishes device residency by setting `representation.device = true` and incrementing counters (`lib.rs:528-570`); demotion similarly toggles `host`, `device`, or `storage` flags (`lib.rs:715-761`). No bytes move.
- The physical-manager layer still exposes IDs and manual methods such as `release_logical_reference`, `release_growth`, `release_binding`, `abort_transition`, and `evict` (`lib.rs:401-427`, `668-687`, `765-789`). ABI 9 has removed raw llama sequence IDs from the Rust executor API and replaced them with RAII `RepresentationHandle` and `PreparedMapping` owners, but that lifetime model has not yet replaced the physical manager's metadata ownership API.
- Trace records accumulate in an unbounded `Vec<TraceEvent>` (`lib.rs:227-258` and every transition operation).

The outline required “typed transfer, reservation, binding, and transition ownership whose lifetimes ... protect physical residency” and a single-writer coordinator that performs transfers outside its critical section (`docs/outline.md:1493-1497`). The current manager is useful as a state-machine prototype, but it cannot enforce the lifetime of the native state it claims to represent.

**Recommended direction**

Decide before further tier work whether this layer will actually own executor storage and transfer fences. If yes, replace raw/manual ownership with RAII guards tied to real native mapping/storage handles; make transfer completion carry a verified native result; use a bounded diagnostic sink rather than an in-memory history. If no, remove fictitious byte/tier authority and make the native executor/residency layer the sole capacity owner. Two independent “truths” about residency will not remain correct under eviction, concurrency, or multi-GPU placement.

### C3. Logical and physical publication was not atomic — remediated

The audit originally found `publish_block` committing the logical mapping before
the physical transition and device table, so a later failure could expose a
logical successor without an executable binding.

The ownership cutover now validates the immutable logical prepared publication
before mutation, completes and commits the physical transition, publishes the
device table, and only then linearizes the logical successor while
`MappedState` is exclusively locked. The final logical commit repeats validation
and is infallible under that lock. Focused context-store publication-conflict
tests and the mapped-engine prior-mapping failure test cover the boundary. The
remaining work is to replace the physical manager's projected representation
records with authoritative native descriptors and fences; atomic ordering no
longer makes those records more truthful than their source.
### C4. Mapped capacity accounting is fictitious and cumulative

**Evidence**

- `publish_block` assigns `represented_end * 1024` bytes **per component**, regardless of model shape, data type, recurrent state, actual llama allocation, or incremental block size (`mapped.rs:864-887`).
- It registers a fresh set of representations for every cumulative evaluated prefix (`mapped.rs:893-929`). Native mappings, however, are llama sequence IDs inside one context, not three independent cumulative KV/SWA/recurrent allocations matching those records.
- `cusco_executor_mapped_bytes_copied()` returns a `mapped_bytes_copied` field (`native/shim/cusco_executor.cpp:806-807`), but no source increments that field. “Zero copied bytes” is therefore not an observed measurement.

This means capacity rejection, promotion metrics, and Phase 5 copy claims are not grounded in the actual resource. The accounting can reject feasible work or admit infeasible work, and cumulative full-prefix registration compounds the logical-store growth problem.

**Recommended direction**

Expose authoritative native per-mapping/per-block allocation and transfer telemetry through the ABI, including allocator class, bytes, tier/device, and fences. Account for incremental blocks, not cumulative logical prefix length. If llama sequence mappings share one arena and cannot expose independent block ownership, model that truth explicitly rather than projecting a block-store abstraction onto them.

### C5. One scheduler worker serializes every resident model

**Evidence**

- `WorkloadScheduler` owns one command channel and one worker join handle (`crates/server/src/scheduler.rs:381-389`).
- Construction starts exactly one `cusco-workload-scheduler` OS thread (`scheduler.rs:411-436`).
- The loop chooses one job and directly calls `execute_quantum` (`scheduler.rs:622-660`).
- `execute_quantum` synchronously calls either `inner.start_session(request)` or `session.step()` on that same thread (`scheduler.rs:747-767`). These operations include model preparation/load and native prefill/decode.

Fair interleaving within a native slot is sound, but a process-wide synchronous executor contradicts Phase 7's “one independently locked native slot per resident model epoch” design and the project guide's statement that distinct resident models can execute independently. A slow load or long native quantum for model A stalls model B and scheduler command handling.

**Recommended direction**

Keep a central policy actor if useful, but dispatch selected work to a bounded per-resident-model executor/slot actor. Track each slot's in-flight fence and return completion observations to the policy actor. This preserves global fairness/accounting while allowing one native quantum per independent model slot concurrently. Model loading and spill/restore should use a separate bounded lifecycle pool so they cannot block scheduling decisions.

### C6. Residency decisions rely on untrusted operating-point estimates

**Evidence**

- The native operating point reads `llama_model_size` and `llama_state_get_size`, then assigns model bytes to device in proportion to `gpu_layers / model_layers` (`native/shim/cusco_executor.cpp:181-206`). Serialized state size is not the same as live allocator residency, and proportional layer size is generally false for heterogeneous layers/tensors.
- The pre-load residency estimate puts the entire model plus context reserve on device whenever `gpu_layers > 0`, or entirely on host otherwise (`crates/server/src/residency.rs:179-209`).
- The Rust ABI exposes one `operating_point()`, not a set of selectable executor-reported operating points (`crates/executor/src/lib.rs:184-194`).

Phase 7 required choosing among executor-reported operating points and keeping a competent model floor distinct from elastic residency. Current admission cannot prove either property. This is a correctness risk under real memory pressure, not merely inaccurate telemetry.

**Recommended direction**

Add native prepare/probe APIs that report actual planned tensor placement, context arena/pool reservations, scratch high-water requirements, and optional elastic ranges before commit. Return multiple supported operating points or accept a proposed point and return an authoritative reservation. After load, reconcile against allocator telemetry and mark underestimation unhealthy rather than learning capacity through OOM.

## High-severity completeness and consistency findings

### H1. Phase 9 persistence contradicts the superseding restart contract

Phase 9 explicitly says contexts, branches, evaluated prefixes, native mappings, caches, and in-flight state may be discarded on restart (`docs/outline.md:1795`) and calls for SQLite-backed model/lifecycle persistence (`outline.md:1824-1825`). Current behavior retains the earlier Phase 4 JSON design:

- `Server::open` deserializes the complete `DurableState` when the JSON path exists (`crates/server/src/lib.rs:1108-1136`). `DurableState` includes contexts and models.
- `persist()` serializes the whole state with `serde_json::to_vec_pretty` and synchronously writes/renames a temporary file (`lib.rs:1186-1196`).
- Context creation, branching, mutation, deletion, model operations, and inference completion call this full rewrite (`lib.rs:1221-1285`, `1408-1549`, `2061-2064`).

This creates synchronous $O(\text{all durable state})$ filesystem work on the inference path and restores state the final v1 contract deliberately made disposable.

**Recommended direction**

Make SQLite the sole durable model/lifecycle/configuration authority. Stop serializing contexts in the v1 profile; initialize a fresh installation-scoped context namespace on process start. If durable contexts return later, give them a purpose-built append/transaction schema and explicit compatibility version rather than reviving the whole-state JSON snapshot.

### H2. Model registration can split memory, JSON, SQLite, and native state

`register_model_with` clones the complete model map, prepares the native model, updates the in-memory map, persists JSON, and then invokes the catalog finalizer (`lib.rs:1408-1484`). If finalization fails, it restores the in-memory map and retires the native model, but it does **not** rewrite the JSON file that was already successfully persisted. A restart can therefore resurrect the model that the SQLite catalog rejected. The whole-map clone is also avoidable $O(\text{model count})$ work.

**Recommended direction**

Use one SQLite transaction for operation state and model publication. Prepare/probe native state before it, publish durable metadata once, then atomically swap the in-memory epoch. On post-publication native failure, record a failed/unavailable lifecycle state transactionally rather than attempting cross-store rollback. Eliminate the duplicate JSON model registry.

### H3. The async-runtime blocking boundary is incomplete

Phase 9 requires fetch, hashing, filesystem persistence, residency load/reload/unload, spill/restore, and native lifecycle work to execute in bounded blocking workers (`docs/outline.md:1798`, `1817`). The Ollama-compatible management pull implementation, exposed at `/cusco/v1/api/pull`, correctly uses `spawn_blocking` for model fetch and GGUF probing, but it calls synchronous SQLite methods before and after those blocks and calls `register_model_with` directly on the Tokio task. Registration can perform filesystem metadata, native model preparation/load, full JSON persistence, and SQLite publication. The API namespace cutover did not change this execution boundary.

**Recommended direction**

Move the entire lifecycle transaction behind a bounded lifecycle executor, not only the obvious download/probe calls. Keep progress delivery and cancellation orchestration on Tokio. Measure queue occupancy and lifecycle-worker time separately from inference scheduling.

### H4. Historical acceptance gates are no longer runnable as documented

`docs/outline.md` names:

- `tools/phase6c-report.sh` (`outline.md:1543`),
- `tools/phase7-report.sh` (`outline.md:1629`),
- `tools/phase8-report.sh` (`outline.md:1733`).

None exists. `docker compose -f compose.test.yaml config --services` currently exposes only `model-fetch`, `test`, `api-smoke`, `executor-proof`, and `mapped-proof`. Historical Phase 6–8 JSON artifacts remain under `results/`, but there is no current reproducible command that regenerates their complete gates.

The unified `tools/smoke-report.py` currently exercises OpenAPI, model listing, buffered completion/chat/Responses, context operations, status, reuse, compaction, and successor continuation. The post-cutover real-model run passed all 13 scenarios. It does not exercise streaming, the Ollama-compatible management lifecycle, mixed multi-model load, lifecycle races/faults, cancellation/deadline storms, diagnostic overflow, or capacity recovery. It is therefore a useful high-level smoke, not a replacement for Phases 6–8 acceptance.

**Recommended direction**

Reintroduce the missing acceptance workloads as reusable test-set wrappers under the current harness, or explicitly revise the outline and Compose contract to point to equivalent maintained commands. Keep a fast deterministic semantic gate and a separately recorded real-GPU performance/fault gate. Do not infer current completion from stale artifacts.

### H5. Phase 10 acceptance evidence does not match its checklist

The outline declares fixture-backed tests for semantic pronoun continuity, instruction retention, tool-call consistency, follow-up fidelity, cancellation/disconnect races, and end-to-end Responses stream ID correlation (`docs/outline.md:1864-1901`). No `phase10/fixtures/...` tree or equivalent YAML/JSONL fixtures exists.

Current `context_strategy.rs` tests prove anchor retention, supported anchor formats, and deterministic output. The server tests cover declarations, worker isolation, successor commit, replay payload, and rollback. These are valuable mechanics tests, but they do not run the real model against semantic controls. The smoke report checks a buffered `window_tail` result and a continuation, not streaming lifecycle reconstruction or distinct transport/inference/session ID propagation.

**Recommended direction**

Implement the named fixtures as deterministic reusable scenarios. Run semantic cases against both uncompacted controls and compacted successors; make pass/fail assertions about the observable answer or structured output, not retained source strings. Add an actual streamed Responses reconstruction test that observes all lifecycle event types and verifies distinct, stable IDs end to end.

## Medium-severity findings

### M1. Image support is admission-only, not end-to-end multimodal support

The image layer performs meaningful safety work: bounded data-URI decoding, MIME/format checks, dimensions/pixels, animation rejection, and digesting. However, chat handling validates an image and then unconditionally returns `image_unsupported: selected model has no compatible vision projector` (`crates/server/src/lib.rs:2402-2415`). There is no projector execution path or model capability advertisement that can make the declared Phase 9 text-plus-image contract succeed.

This is an acceptable explicit limitation only if Phase 9 is marked incomplete for multimodal support. It should not be described as delivered “bounded text-plus-image admission” without clarifying that all images are rejected after validation.

### M2. The server crate lacks the durable service boundaries described by the outline

`crates/server/src/lib.rs` remains roughly 6,000 lines and owns domain records, persistence, admission, auth integration, lifecycle, inference coordination, all HTTP request/response types, route handlers, streaming encoders, and a large test module. Removing the redundant Ollama inference adapter reduced the surface, and some important subsystems have been split (`scheduler.rs`, `residency.rs`, `mapped.rs`, `catalog.rs`, `generation.rs`, `vision.rs`, `context_strategy.rs`), but the protocol-neutral `InferenceService` and model-management boundaries described in the outline are not durable interfaces in the implementation.

This makes adapter conformance, alternative transports, persistence replacement, and isolated lifecycle testing harder. Splitting by line count alone would not help; extraction should follow ownership boundaries: canonical application services, persistence repositories/transactions, and thin protocol adapters.

### M3. Physical tracing is unbounded while scheduler diagnostics are bounded

The scheduler correctly uses a bounded diagnostic sink with explicit loss counters. `PhysicalManager`, by contrast, stores every event forever in `events: Vec<TraceEvent>`. A long-running state server will leak memory proportional to physical activity. Route these records through the same bounded/non-blocking diagnostic contract or expose a bounded ring buffer.

### M4. Copy telemetry cannot substantiate mapped-execution claims

The native `mapped_bytes_copied` counter is exposed through C and Rust but is never incremented. Any proof that reports zero activation bytes copied is measuring a constant, not observed copy behavior. Instrument sequence copy/fork/import/export operations at the native boundary, with definitions that distinguish metadata-only reference switches from KV/state copying.

### M5. ABI documentation drift is corrected

The audit first found `AGENTS.md` describing executor ABI version 6 while the
header defined version 7. The measurement baseline advanced the boundary to ABI
8, and the opaque representation-handle cutover now advances both the header and
repository guidance to ABI 9. The version still has two manually synchronized
locations; a generated or mechanically checked source of truth remains
preferable.
## What should be fixed before harder-to-reverse expansion

Recommended order:

1. **Define one truthful physical-state and transaction model.** Resolve C2–C4 together; isolated patches will otherwise add more metadata around a non-atomic boundary.
2. **Replace global synchronous execution with per-model slot workers.** Preserve the current fairness policy as the global dispatcher, but make independent residency actually concurrent.
3. **Make native capacity reservation authoritative.** Then update residency admission and performance evidence to use real bytes and operating points.
4. **Complete the Phase 9 clean persistence cutover.** Remove JSON contexts/models, use SQLite transactions, and move lifecycle work entirely behind the blocking boundary.
5. **Restore reproducible acceptance gates.** Reuse stable scenario sets across local and future cluster harnesses; preserve correctness, fault, and performance layers separately.
6. **Close Phase 10 semantic and stream-correlation evidence.** Only then mark its baseline checklist complete.
7. **Extract application-service and repository boundaries from `server::lib`.** Do this alongside the persistence/lifecycle work, not as a cosmetic module shuffle.

## Settled remediation decisions

The following decisions are now part of the remediation contract rather than open design questions:

1. **Physical ownership:** use Approach A. Native code owns model-specific physical allocations and hot-path layout; Rust owns logical identity, opaque typed handles, policy, reservations, lifetimes, and transactional publication. Preserve an ABI evolution path to B1; treat fully Rust-owned B2 allocation as possible but unlikely.
2. **Publication and branching:** ordinary linear continuation must publish without forking or copying a full native sequence. A physical fork may initially occur only for a genuine branch. Measure branch cost separately; require copy-on-write branching later if that cost is material.
3. **Logical storage unit:** use immutable logical token chunks with executor-described component coverage. Do not require a logical chunk to equal one native allocation or assume global KV, SWA, and recurrent state share one physical geometry.
4. **Atomicity:** readers observe either the complete old logical context and executable binding or the complete successor and executable binding. Candidate native state remains private and abortable until one Rust-owned publication record linearizes success; old state remains retained through outstanding references and native fences.
5. **Capacity:** distinguish hard correctness and competent-model floors, per-slot/per-request execution reserve, required transient transition or reload reserve, and reclaimable elastic acceleration/cache capacity. Admission uses hard plus required transient reservations; elastic state cannot consume capacity promised to admitted work.
6. **Concurrency unit:** serialize native ownership per execution slot, not process-wide and not permanently per model. A residency object may own multiple slots later.
7. **Initial concurrency scope:** allow distinct resident models to execute concurrently while retaining one serialized slot per model initially. Same-model multi-slot batching and multi-GPU placement remain deferred.
8. **Startup residency:** models are not loaded at startup by default. A `config.yaml` startup-pinning option accepts one model name or a list, defaults to empty, loads those models after catalog reconciliation, and holds them resident regardless of ordinary idle-unload policy. Missing, inadmissible, or failed pinned models make startup readiness fail clearly rather than silently weakening the pin.
9. **Performance gates:** use hardware-independent invariants and scaling comparisons before fixed throughput targets. Linear publication must copy no full-prefix KV state; publication cost must not grow with represented-prefix length; a logical publication boundary alone must not force CUDA graph recapture; failures preserve the prior exact continuation; and distinct-model execution must demonstrate overlap.
10. **GPU support:** local development builds gate `sm_61` and `sm_70` to keep iteration practical. Production release builds target the useful CUDA architecture range from `sm_61` through `sm_120`, including intervening architectures supported by the pinned toolchain and executor. Runtime behavior should progressively use newer-GPU capabilities rather than reducing every device to the oldest common implementation. Evidence records the exact compiled targets and actual GPU.
11. **Configured local-model ownership:** `user.yaml` is the sole mutable authority for its configured `LocalFile` subset. HTTP and CLI management may list and inspect those records but must reject pull, copy/alias, delete, rename, replacement, or name/alias collisions involving them. API-managed Hub records remain independently mutable.

## Dependency-aware remediation sequence

The order below is intentionally not the original phase order and is not a severity ranking. It is ordered to establish contracts that downstream work will consume, so later changes do not have to be implemented twice. Each step should leave a runnable vertical path; this is a migration sequence, not permission to accumulate disconnected infrastructure.

### 1. Establish measurement before changing representation

First make the current costs observable: increment native copy telemetry for sequence forks and state movement, record mapping-publication latency and bytes by represented prefix length, count CUDA graph recaptures where the backend exposes that information, and preserve a small deterministic workload that crosses several publication boundaries. Add reusable correctness and performance layers to the benchmark harness rather than recreating all historical phase scripts verbatim.

This does not require preserving the current architecture. It provides the before/after evidence needed to tell whether the representation work actually removes KV copying, graph churn, and long-context scaling. Deferring instrumentation until after the redesign would lose the baseline and invite counters shaped around the new implementation.

**Implemented baseline:** `config/representation-workload.json` and the
`representation-proof` Compose gate now preserve a deterministic
multi-boundary trace, require exact token and logit continuation before
performance reporting, and write per-boundary publication latency and actual
fork/export/import copy deltas to `results/representation-proof.json`. ABI 8
also represents unavailable graph-recapture telemetry explicitly rather than
inventing a zero count.

### 2. Implement the selected physical ownership model

The remediation uses **Approach A: native-owned opaque physical state**. llama.cpp and the native shim retain ownership of model-specific allocation layouts, KV and recurrent state, graph-sensitive buffers, and hot-path address resolution. Rust owns logical identity, typed opaque handles, scheduling and tiering policy, authoritative reservations reported by native code, lifetime management, and transactional publication.

**Implemented first cut:** ABI 9 replaces public llama sequence/mapping IDs with
opaque reference-counted `cusco_representation` handles and uniquely owned
prepared handles. Rust `RepresentationHandle` clones retain native ownership,
drops release it, and retain the executor lifetime; `PreparedMapping` aborts on
drop. The server and proof paths now carry these handles end to end. Mapped
publication validates logical state before mutation and exposes the logical
successor only after its physical binding and device table are committed.

The next ownership work is authoritative native allocation/placement and
reservation descriptors, explicit asynchronous completion fences, and removal
of the physical manager's fictitious cumulative byte/tier records. Routine
linear generation also still forks at publication boundaries and must be
changed only against the preserved measurement baseline.
#### Potential evolution: B1, native-allocated blocks with Rust-owned identity

**B1 is a plausible and potentially likely later path if measurements show that branch copying, coarse eviction, or inability to share evaluated prefixes remains material.** Native code would still allocate and lay out model-specific blocks, while Rust would own stable block identity, sharing policy, references, tier placement, and mapping composition through opaque handles. Native code and kernels would retain hot-path address resolution.

The Approach A ABI should therefore preserve an evolutionary route to B1: versioned opaque representation handles, component masks, retain/release semantics, authoritative placement and size data, completion fences, and transactional mapping descriptors. The first implementation must not promise that one handle always means one llama sequence. A future native implementation may back the same contract with several shared blocks.

B1 should be adopted only with evidence that A's remaining copy or placement costs justify the additional cache-manager integration. A proof must demonstrate direct execution or reference-only activation from the composed blocks; creating block objects and then copying them into a conventional llama sequence would add complexity without delivering the intended benefit.

#### Unlikely fallback: B2, fully Rust-owned physical allocation

**B2 remains possible but is not a planned direction.** In B2, Rust would allocate host/device buffers and provide physical descriptors to native execution. This would require Rust to reproduce or continuously track llama.cpp's alignment, cache-format, backend, graph-stability, scratch, SWA, recurrent-state, and architecture-specific requirements. It risks duplicating unstable native internals and becoming a maintained execution backend rather than a narrow ABI integration.

B2 should be reconsidered only if both A and B1 are shown inadequate and the measured value of complete allocation control justifies owning a deep llama.cpp integration. Such a decision would require a separate design and maintenance commitment; it must not emerge incrementally through one-off raw-buffer escape hatches in the Approach A ABI.

Everything below depends on the selected Approach A contract. Capacity cannot be authoritative until native allocations and reservations are visible; scheduler concurrency cannot be safe until ownership and fences are explicit; and logical identities should not be redesigned around fictitious physical blocks.

### 3. Align logical context storage with the chosen physical unit — implemented

`PersistentTokenSequence` now uses bounded immutable 256-token chunks with structural sharing and append-segmentation-independent cumulative identity. Arbitrary prefix views retain a chunk plus a visible tail length, so branching at a non-chunk boundary remains cheap. Evaluated mappings retain those immutable logical boundaries rather than complete copied token vectors; lineage depends on the parent lineage and boundary identity rather than rehashing the entire prefix.

Lookup is indexed by model epoch, adapter epoch, and represented length, searches deterministically from the longest eligible boundary, and confirms both the branch digest and literal tokens without allocation. Logical chunks remain independent of executor allocation geometry, preserving the Approach A contract and its B1 evolution path. Long-context scaling measurements remain to be added to the reusable benchmark harness, but the storage and identity cutover itself is complete.

### 4. Rebuild mapped publication as one atomic transaction

Implement physical preparation and binding construction against the new logical/physical contracts. A publication should prepare all required representations, validate lineage and capacity, construct the candidate executor binding, and only then atomically publish both logical and physical state. Every failure and cancellation path must leave the previous binding executable.

At the same time, remove routine full-sequence forks from ordinary block publication. Fork native state only for a genuine branch if that remains necessary, or publish reference/block-table changes without copying KV state. This step should close the current logical-before-physical commit gap and the mapped-execution zero-copy fiction together; fixing either in isolation would likely be discarded by the other.

### 5. Make capacity and operating points authoritative

Once real allocations and bindings exist, add native prepare/probe reservations that report planned tensor placement, context pools, scratch requirements, transfer headroom, and elastic ranges. Reconcile reservations with post-load allocator telemetry. Then replace proportional layer-count and serialized-state estimates in residency admission.

Capacity work belongs here rather than earlier because estimates cannot be made authoritative over placeholder physical representations. It must precede scheduler concurrency: allowing multiple model workers to load and execute concurrently while admission remains approximate would multiply the current oversubscription and rollback risks.

### 6. Introduce independent model-slot execution

Keep the global fairness scheduler as an admission and dispatch policy, but move native preparation/prefill/decode work to independently runnable per-model or per-slot executors. Define slot occupancy, suspension, cancellation, shutdown, and model-epoch draining using the ownership fences and capacity reservations established above.

This ordering lets concurrency reuse trustworthy state transitions rather than adding locks around the current single-threaded path and later replacing them. Re-run mixed-model fairness and capacity-recovery workloads as soon as the first concurrent implementation exists.

### 7. Complete lifecycle persistence and extract its service boundary together

Make SQLite the sole durable authority for model identity, aliases, revisions, and lifecycle operations; delete whole-state JSON persistence for disposable contexts and native state. Move the complete fetch/probe/prepare/catalog-publication transaction behind a bounded lifecycle executor. Extract a protocol-neutral lifecycle service and repository transaction boundary while doing this, then keep `/cusco/v1/*` and its Ollama-compatible management profile as thin projections.

Persistence and service extraction should be one change sequence because the useful service interface is the transaction boundary. Extracting HTTP-shaped methods first would merely preserve the current split transaction behind another layer. This work can be prototyped alongside Steps 3–4, but final registration/load semantics should be based on the authoritative capacity and ownership contracts from Step 5.

### 8. Rebuild the full acceptance stack against the settled architecture

Promote the reusable workloads introduced in Step 1 into maintained gates:

- deterministic logical, publication, rollback, and exact-continuation checks;
- cancellation, delayed-fence, transfer, allocation, and catalog-failure injection;
- streaming and ID-correlation checks;
- sustained mixed-model fairness, deadline, backpressure, and diagnostic-overflow workloads;
- long-context copy, cache-hit, graph-reuse, latency, and throughput measurements;
- real-GPU evidence with declared model, build, device, workload, and provenance.

Tests should be added alongside each preceding contract change, but the final gate structure comes here because its stable dimensions depend on the new ownership, capacity, scheduler, and lifecycle boundaries. Historical phase-numbered scripts should not be restored if they encode superseded architecture.

### 9. Close Phase 10 semantic evidence on top of the stable execution path

Finally add the named semantic-quality fixtures, follow-up fidelity checks, and end-to-end Responses stream-correlation coverage. Compaction depends on context representation, publication, scheduling, persistence policy, and telemetry; completing its broader evidence before those foundations settle would create fixtures and measurements that need immediate migration.

The deterministic `window_tail` mechanism can remain available throughout the remediation. The sequencing recommendation is only to defer claims about complete semantic and operational acceptance until the underlying execution path is stable.

### Parallel work that will not create much rework

Documentation corrections, bounded physical trace storage, warning cleanup, and model-free protocol conformance can proceed alongside the sequence. The benchmark/test-set format can also be developed early as long as scenarios describe observable behavior rather than current internal types. Multi-GPU placement, external compaction workers, broader model-family support, and durable context formats should wait: each would otherwise bind new functionality to contracts this sequence is intended to replace.


## Bottom line

Phases 1–10 have produced a credible vertical system and several good local mechanisms, but the current completion language is ahead of the implementation in the areas that matter most for a persistent tiered model-state server: physical ownership, atomic publication, authoritative capacity, independent model execution, and reproducible acceptance evidence.

The foundational ownership decision is now explicit: llama.cpp and the native shim own model-specific physical allocations, while Rust owns logical identity, policy, reservations, typed opaque handles, and transactional publication. The immediate work is to make that Approach A boundary truthful and atomic rather than layering more policy over estimates. The ABI should leave room for a measured evolution to native-allocated shared blocks under B1, without committing Cusco to the unlikely B2 path of reproducing llama.cpp's allocator and execution-layout knowledge in Rust.
