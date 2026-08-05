# Cusco

Cusco is an experimental persistent, tiered, branch-aware model-state server built around llama.cpp. Its goal is to make model execution state a durable resource that can be captured, displaced from scarce accelerator memory, restored later, and continued exactly.

The project separates responsibilities deliberately: Rust will manage logical contexts, scheduling, storage tiers, and transactional state, while a narrow native boundary lets llama.cpp retain ownership of model-specific execution details and optimized kernels.

## Current state

Cusco has completed its executor proof, Rust logical-context-store, tiered physical-manager, and minimal server phases. The Phase 1 proof established the project’s central technical premise on a hybrid/recurrent Gemma model: captured execution state can be replaced, moved through host memory, restored, and continued with identical tokens and bitwise-identical logits. The proof also covers concurrent logical contexts and verifies that cancellation or failed promotion does not destroy the previously valid state.

Phase 2 adds Rust-owned logical contexts and transactional evaluated-prefix publication. Phase 3 adds capacity-accounted physical representations, transfers, transitions, eviction, and observability. Phase 4 adds a protocol-neutral inference core, transition-cost scheduler, durable opaque contexts, model lifecycle operations, canonical usage and streaming events, cancellation and deadlines, authenticated native APIs, useful OpenAI completion/chat adapters, and a checked OpenAPI document. `cusco serve` uses the native llama.cpp executor; inference only resolves models already installed through the administrative API.

## Run the real-model inference integration test

With the validation GGUF at `models/gemma-4-e2b-it.gguf`, Docker's NVIDIA
runtime configured, and an available NVIDIA GPU, run:

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

## Run the minimal server

The unauthenticated development provider is restricted to loopback:

```sh
cargo run -p cusco -- serve \
  --listen 127.0.0.1:8080 \
  --state ./data/cusco-state.json
```

Pass `--bearer-token` to require authentication. Listening anonymously on a
non-loopback address is rejected unless
`--unsafe-public-unauthenticated` is explicitly supplied.

The checked OpenAPI document is served at `/openapi.json`. OpenAI-compatible
entry points are `/v1/completions`, `/v1/chat/completions`, and `/v1/models`.
Native `/native/models`, `/native/contexts`, and `/native/requests` operations
cover model lifecycle, durable contexts and branches, imports, and
cancellation. Server inference uses registered local model paths and does not
implicitly fetch models.

## Future goals

Development is planned to proceed from the proven executor boundary toward:

- integration of the minimal scheduler with increasingly efficient physical context movement;
- production hardening of the inference and model-management server;
- improved mapped execution to reduce the cost of switching active contexts;
- broader compatibility, operational hardening, and recovery behavior;
- optional semantic context compaction once the underlying state system is proven reliable.

Each stage is intended to remain gated by correctness and measurable capacity results. The full design and phased acceptance criteria are documented in `docs/outline.md`.
