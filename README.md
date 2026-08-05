# Cusco

Cusco is an experimental persistent, tiered, branch-aware model-state server built around llama.cpp. Its goal is to make model execution state a durable resource that can be captured, displaced from scarce accelerator memory, restored later, and continued exactly.

The project separates responsibilities deliberately: Rust will manage logical contexts, scheduling, storage tiers, and transactional state, while a narrow native boundary lets llama.cpp retain ownership of model-specific execution details and optimized kernels.

## Current state

Cusco has completed its executor proof, Rust logical-context-store, and tiered physical-manager phases. The Phase 1 proof established the project’s central technical premise on a hybrid/recurrent Gemma model: captured execution state can be replaced, moved through host memory, restored, and continued with identical tokens and bitwise-identical logits. The proof also covers concurrent logical contexts and verifies that cancellation or failed promotion does not destroy the previously valid state.

Phase 2 adds Rust-owned logical contexts with immutable, structurally shared token branches; model and adapter epochs; dependency-valid evaluated-prefix mappings; longest-valid-prefix lookup; explicit reference accounting; and revision-guarded transactional mapping publication. Phase 3 adds capacity-accounted device, pinned-host, and storage representations; guarded growth and transition reservations; asynchronous transfer ownership; atomic prepared transitions; deterministic warm-state eviction; and structured capacity metrics and traces. The repository also includes the native llama.cpp boundary, immutable model registration, reproducible container build, command-line proof driver, and coverage-enforced tests. It does not yet provide a network service or a stable user-facing API.

## Run the real-model inference integration test

With the validation GGUF at `models/gemma-4-e2b-it.gguf`, Docker's NVIDIA
runtime configured, and GPU 1 available, run:

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

## Future goals

Development is planned to proceed from the proven executor boundary toward:

- tiered movement of model state across device, host, and storage capacity;
- a minimal inference and model-management server with scheduling and streaming;
- improved mapped execution to reduce the cost of switching active contexts;
- broader compatibility, operational hardening, and recovery behavior;
- optional semantic context compaction once the underlying state system is proven reliable.

Each stage is intended to remain gated by correctness and measurable capacity results. The full design and phased acceptance criteria are documented in `docs/outline.md`.
