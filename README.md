# Cusco

Cusco is an experimental persistent, tiered, branch-aware model-state server built around llama.cpp. Its goal is to make model execution state a durable resource that can be captured, displaced from scarce accelerator memory, restored later, and continued exactly.

The project separates responsibilities deliberately: Rust will manage logical contexts, scheduling, storage tiers, and transactional state, while a narrow native boundary lets llama.cpp retain ownership of model-specific execution details and optimized kernels.

## Current state

Cusco has completed its executor proof, Rust logical-context-store, tiered
physical-manager, minimal server, and mapped-execution phases. The Phase 1
proof established the project’s central technical premise on a
hybrid/recurrent Gemma model: captured execution state can be replaced, moved
through host memory, restored, and continued with identical tokens and
bitwise-identical logits.

Phase 2 adds Rust-owned logical contexts and transactional evaluated-prefix
publication. Phase 3 adds capacity-accounted physical representations and
transactional tier transitions. Phase 4 adds the authenticated inference and
model-management server. Phase 5 adds transactional device-resident sequence
mappings, reference-only activation, physical block-table publication, and
mapped-versus-staged measurements without exposing llama.cpp internals to Rust.

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

## Run the mapped-execution proof

With the validation GGUF and NVIDIA runtime available, run:

```sh
docker compose run --rm mapped-proof
```

The command forks four device-resident sequence mappings, activates and
continues each one with identical results, verifies graph reuse and zero bytes
copied by activation, and writes staged-versus-mapped timing, bytes, and prompt
work avoided to `results/phase5.json`.

## Run the real-model server report

To exercise the authenticated Phase 4 HTTP surface with the same validation
GGUF and print a prompt, generated response, streaming assertions, usage, and
end-to-end timing data, run:

```sh
tools/server-inference-report.sh
```

The report is also written to `results/phase4-server.json`. The workflow uses
GPU 1 and port 18082 by default; `CUSCO_GPU_DEVICE_ID`,
`CUSCO_SERVER_REPORT_PORT`, `CUSCO_MODEL_DIR`, and `CUSCO_RESULT_DIR` override
those defaults. Server timing includes model loading and generation, and the
current SSE adapter emits its buffered token events after generation completes,
so the reported first streamed token is an end-to-end observation rather than
decode-only time-to-first-token.


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

Development is planned to proceed from the proven mapped-execution boundary
toward:

- integration of mapped execution with live server scheduling;
- production hardening of the inference and model-management server;
- broader compatibility, operational hardening, and recovery behavior;
- optional semantic context compaction once the underlying state system is proven reliable.

Each stage is intended to remain gated by correctness and measurable capacity results. The full design and phased acceptance criteria are documented in `docs/outline.md`.
