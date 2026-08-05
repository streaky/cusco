# Cusco

Cusco is an experimental persistent, tiered, branch-aware model-state server built around llama.cpp. Its goal is to make model execution state a durable resource that can be captured, displaced from scarce accelerator memory, restored later, and continued exactly.

The project separates responsibilities deliberately: Rust will manage logical contexts, scheduling, storage tiers, and transactional state, while a narrow native boundary lets llama.cpp retain ownership of model-specific execution details and optimized kernels.

## Current state

Cusco has completed its executor proof and Rust logical-context-store phases. The Phase 1 proof established the project’s central technical premise on a hybrid/recurrent Gemma model: captured execution state can be replaced, moved through host memory, restored, and continued with identical tokens and bitwise-identical logits. The proof also covers concurrent logical contexts and verifies that cancellation or failed promotion does not destroy the previously valid state.

Phase 2 adds Rust-owned logical contexts with immutable, structurally shared token branches; model and adapter epochs; dependency-valid evaluated-prefix mappings; longest-valid-prefix lookup; explicit reference accounting; and transactional mapping publication. The repository also includes the native llama.cpp boundary, immutable model registration, reproducible container build, command-line proof driver, and coverage-enforced tests. It does not yet provide physical tier management, a network service, or a stable user-facing API.

## Future goals

Development is planned to proceed from the proven executor boundary toward:

- tiered movement of model state across device, host, and storage capacity;
- a minimal inference and model-management server with scheduling and streaming;
- improved mapped execution to reduce the cost of switching active contexts;
- broader compatibility, operational hardening, and recovery behavior;
- optional semantic context compaction once the underlying state system is proven reliable.

Each stage is intended to remain gated by correctness and measurable capacity results. The full design and phased acceptance criteria are documented in `docs/outline.md`.
