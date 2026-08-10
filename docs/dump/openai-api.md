# OpenAI Responses API Capability Review

## Scope

This note records Cusco's implemented `/openai/v1/responses` contract as of 2026-08-09 and the remaining compatibility boundary. It is a capability assessment, not a commitment to mirror every hosted OpenAI feature.

The implementation now provides a coherent, portable Responses core rather than only a stateless text wrapper:

- string and structured message input;
- buffered and typed streaming text output;
- OpenAI SDK-compatible `ResponseOutputMessage`/`ResponseOutputText` envelopes and `output_text` reconstruction;
- durable `store: true` response resources with retrieval, deletion, cancellation, restart recovery, quotas, and lineage-aware cleanup;
- stateless `store: false` operation;
- `previous_response_id` continuation with replay of prior input and output items;
- typed function-call output, streamed argument deltas, and client-submitted `function_call_output` continuation;
- `tool_choice` modes `auto`, `none`, `required`, and a named function, with pre-generation validation;
- preserved omission of `max_output_tokens` and a late effective ceiling derived from real executor context headroom;
- deterministic sampling, usage, request correlation, compaction metadata, authentication, admission, cancellation, and deadlines; and
- strict rejection of unsupported capabilities before admission.

The endpoint shares canonical generation and scheduling machinery with the other inference adapters, but owns Responses-specific request validation, item projection, event lifecycle, resource persistence, and lineage rules.

## Implemented contract

### Output items and streaming lifecycle

Buffered text responses contain ordinary typed message output items, so the official OpenAI Python SDK exposes generated text through `response.output_text`. Streaming uses stable IDs and an ordered typed lifecycle:

- `response.created`;
- `response.in_progress`;
- `response.output_item.added`;
- `response.content_part.added`;
- `response.output_text.delta`;
- `response.output_text.done`;
- `response.content_part.done`;
- `response.output_item.done`; and
- `response.completed`.

Terminal response reconstruction is consistent with the deltas and completed output items. Function calls use the corresponding typed function-call item and argument-delta lifecycle. Failure, cancellation, and incomplete results use terminal response states rather than an ad hoc text row.

### Stored response resources and continuation

`store: true` atomically publishes a versioned file-backed resource under the configured response-store directory. Stored resources survive daemon restart and support:

- `GET /openai/v1/responses/{response_id}`;
- `DELETE /openai/v1/responses/{response_id}`;
- `POST /openai/v1/responses/{response_id}/cancel`; and
- continuation through `previous_response_id`.

Publication is bounded by configured per-principal/global quotas. Recovery validates persisted records, cleanup is lineage-aware, and replay reconstructs prior portable input/output items rather than depending on process-local native execution state. `store: false` remains genuinely stateless and does not publish a resource.

This resource durability is distinct from native executor durability. Active requests, queues, mappings, and model execution state remain process-local; continuation after restart is rebuilt from the stored portable response lineage.

### Function calling and `tool_choice`

Cusco supports the client-executed function-call loop:

1. the client submits function definitions;
2. the model may emit a typed function-call item with a stable call ID and JSON arguments;
3. the client executes the function; and
4. a later request submits a typed `function_call_output`, normally with `previous_response_id`.

`tool_choice` accepts:

- `auto`: tools are available but a text answer is allowed;
- `none`: tool instructions are suppressed;
- `required`: the model is instructed to produce a tool call; and
- `{"type":"function","name":"..."}`: the named declared function is required.

Malformed choices, unsupported choice types, `required` without tools, and named functions absent from `tools` fail with a stable `400 invalid_request` before generation. This is client-side tool execution support; Cusco does not execute arbitrary client-provided code.

### Output-limit semantics

An omitted `max_output_tokens` remains omitted through request normalization. Cusco derives the effective limit only after prompt construction and exact tokenization, using the context capacity of the resident executor and any explicit operator ceiling. An explicit client limit is enforced rather than silently replaced by the historical small compatibility default.

The execution record distinguishes the requested limit, effective ceiling, and limiting source so length termination remains diagnosable.

### Public API and Codex profiles

The public OpenAI API and the Codex backend are materially different profiles:

| Area | Public API profile | Codex backend profile |
| --- | --- | --- |
| Input | String or structured item list | Structured item list required |
| Streaming | Optional | Required |
| Storage | Supported and configurable | Must be disabled |
| Output limit | `max_output_tokens` accepted | Upstream rejects it; Cusco may accept it as an extension |
| Final output | Completed `Response` contains output | Streamed items may be authoritative |
| Continuation | `previous_response_id` | Replay prior input and output items |
| Chat Completions | Native endpoint | Translated over Responses |

Cusco's checked profile is currently the public-API profile. A separately selected and independently tested Codex profile remains future work; payload-shape guessing is not an acceptable substitute.

## Remaining gaps

### Conversations API

Cusco has response lineage but does not yet expose the OpenAI Conversations resource surface. Conversation ownership, item mutation, listing, deletion, quotas, and interaction with response-resource cleanup still need a defined contract.

### Structured outputs

The canonical API smoke exercises strict structured output, but compatibility remains narrower than the hosted API's complete formatting surface. JSON Schema behavior should continue to be checked through the official SDK, including schema rejection, generated-output validation, streaming interaction, and model-capability failures. Unsupported formats must continue to fail explicitly rather than degrade to unconstrained text.

### Vision and other modalities

The parser and admission path accept bounded text-plus-image inputs and decode data URLs, but the executor does not yet project image content into model execution. The pinned Gemma test model supports vision; the missing piece is the demonstrated projector/executor boundary, not a second test model.

Remote image URLs, audio, video, image generation, file uploads, and cross-model media pipelines are outside the current profile. Unsupported media or models must fail before generation.

### Reasoning and hosted tools

`reasoning_effort: "none"` is a compatibility boundary, not full reasoning support. Cusco does not expose reasoning items, encrypted reasoning content, reasoning-token accounting, hosted web/file search, computer use, code execution, or server-side arbitrary tools.

### Background execution

Stored resources and cancellation do not imply OpenAI-style detached background execution. Creating a background response, polling an independently running job, expiration policy, and restart-resumable active work remain unimplemented. Live generation is still request-tied and bounded by admission, deadlines, cancellation, and shutdown.

### Full typed item taxonomy

Messages and function calls are implemented. Reasoning items, hosted-tool results, annotations, citations, file items, computer-use actions, and modality-specific output items are not.

## Compatibility evidence

The canonical real-model gate is:

```sh
docker compose -f compose.test.yaml run --rm acceptance
```

The latest recorded run in `results/acceptance-report.json` passed all canonical gates with the pinned model `hf://unsloth/gemma-4-E2B-it-GGUF/gemma-4-E2B-it-Q3_K_M.gguf`.

The advisory SDK-conformance gate is:

```sh
docker compose -f compose.test.yaml run --rm oai-lens
```

The latest `results/oai-lens-gate.json` records **6 passed / 1 failed**. Passing probes cover Chat Completions buffered and streaming behavior, Responses string and message input, Responses streaming reconstruction, and function-call/result continuation. The sole unchanged failure is `vision.data_url`, matching the explicit executor limitation above. There were no newly failing probes.

## Compatibility boundary

Cusco may accurately describe its current surface as a durable, typed Responses implementation supporting portable text generation, streaming reconstruction, stored-resource lifecycle, lineage continuation, and client-executed function calls. It should not be described as full OpenAI Responses parity.

The next useful compatibility work is API depth rather than another persistence redesign: Conversations on the proven item/lineage model, stronger SDK-backed structured-output coverage, then real vision behind an executor capability. A distinct Codex profile and optional Cusco-native context-update extensions should remain explicit, versioned profiles rather than silently changing the public OpenAI contract.
