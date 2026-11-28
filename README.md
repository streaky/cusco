# Cusco

![Two continuations branching from one evaluated prefix at a checkpoint.](docs/assets/context-branches.svg)

Cusco is an experimental inference server for llama.cpp that lets you branch a model context without starting from scratch. Two requests can share the work of reading the same prompt, then generate different continuations. Cusco manages that evaluated state separately from the GPU slot that produced it: it can capture a checkpoint, make room for another context, and restore the checkpoint to continue.

[Quick start](#quick-start) · [Client APIs](#client-apis) · [Service and verification guide](docs/service-guide.md) · [Design outline](docs/outline.md)

> **Experimental · Linux only.** The supported deployment builds from source inside the surrounding Terrane checkout. The production Compose service requests an NVIDIA GPU; the model-free acceptance tests do not need one.

## Quick start

Run these commands from `projects/cusco` in a Terrane checkout containing `packages/http-server` and `packages/template`. You’ll need Docker Compose and an NVIDIA GPU available to containers. Set `CUSCO_GPU_DEVICE_ID` if you want a device other than `0`.

Compose mounts one writable host directory at `/data`. On first setup, put the [example configuration](config.example.yaml) inside that directory as `config.yaml`. Adjust its memory and storage budgets for your machine. Cusco creates `db/`, `models/`, `spill/`, and `user-models/` on startup; it does not generate or overwrite the config.

```sh
mkdir -p "${CUSCO_DATA_DIR:-./data}"
cp config.example.yaml "${CUSCO_DATA_DIR:-./data}/config.yaml"
export CUSCO_BEARER_TOKEN="$(openssl rand -hex 32)"
docker compose up --build --wait server
```

**Upgrading from separate mounts?** Copy your existing root `config.yaml` into the data directory instead of copying the example. If you use local model declarations, also copy `config/user.yaml` there as `user.yaml`. Existing `./data` subdirectories stay in place; if you used custom model directories, make their contents available under the new data root before restarting.

**Developing on the host?** From the Cusco checkout, use [`./run-host.trn`](docs/service-guide.md#run-the-daemon-directly). On first setup, copy `config.host.example.yaml` to `./data-dev/config.yaml`. The host example listens on `127.0.0.1:11443` and uses a separate development catalog. Compose still uses `./data` and starts the compiled `cusco-service` directly.

Check the service before downloading a model. The default host port is **11443**; `8080` is the port *inside* the container.

```sh
base=http://127.0.0.1:11443
auth="Authorization: Bearer $CUSCO_BEARER_TOKEN"
curl -fsS "$base/health"                       # ready
curl -fsS -H "$auth" "$base/openai/v1/models"   # installed models
```

### Store a response and continue it

From the same shell, pull the GGUF used by the real-model acceptance tests. If you already have local weights, put them under `user-models/` in the data directory and declare them in `user.yaml` there instead. Omit `user.yaml` if you don’t need local imports.

```sh
model='hf://unsloth/gemma-4-E2B-it-GGUF/gemma-4-E2B-it-Q3_K_M.gguf'
curl -fsS -H "$auth" -H 'Content-Type: application/json' \
  -d "{\"model\":\"$model\",\"stream\":false}" "$base/cusco/v1/api/pull"
```

With `jq` installed, save the ID from one response and use it as the predecessor of the next. The second request supplies only the new input; it doesn’t resend the conversation.

```sh
respond() {
  curl -fsS -H "$auth" -H 'Content-Type: application/json' \
    --data-binary @- "$base/openai/v1/responses"
}
id=$(jq -n --arg m "$model" \
  '{model:$m, input:"Name a planet with rings.", store:true}' | respond | jq -er .id)
jq -n --arg m "$model" --arg id "$id" \
  '{model:$m, previous_response_id:$id, input:"What are its rings made of?"}' | respond
```

## What makes it Cusco?

**A context is not a GPU allocation.** Cusco records token branches and maps evaluated prefixes to executor checkpoints. A branch can reuse a matching prefix rather than evaluate those tokens again. A checkpoint captures the state the loaded executor actually needs, including sliding-window or recurrent state where applicable. This is distinct from *compaction*, which deliberately rewrites retained context and can change the answer.

**Stored history outlives resident state.** Installed models and aliases live in SQLite; logical contexts and stored Responses are file-backed. A response published with `store: true` can be retrieved and continued after a restart. Live requests, queues, executor mappings, and spill contents do **not** survive as resumable GPU work. Continuing a stored conversation after restart does not promise a warm cache.

**One service, three layers.** The [Terrane application](server/) owns the listener, authentication, controllers, and streaming. The [Rust backend](crates/server/) schedules work and manages models, context residency, and persistence. A [versioned native interface](native/) connects it to llama.cpp for tokenization, sampling, and checkpoint capture and restore. All three run in one process; Rust is not a second HTTP service.

## Client APIs

| Client | Base path | What works today |
| --- | --- | --- |
| OpenAI-compatible clients | `/openai/v1` | Model listing, Completions, Chat Completions, and Responses, including streaming. Use Responses for stored continuation and function calls. |
| Ollama-compatible model managers | `/cusco/v1/api` | Discover, pull, inspect, copy, and delete models; inspect residency. This is **not** an Ollama chat or generate endpoint. |
| Cusco-native clients | `/cusco/v1` | Work with contexts and branches, inspect status and capabilities, and publish transactional context updates. |

For an OpenAI SDK, set the base URL to `http://127.0.0.1:11443/openai/v1` and use the bearer token as the API key. The running service exposes its OpenAPI documents at `/openai/v1/openapi.json` and `/cusco/v1/openapi.json`. Unknown request fields are rejected rather than silently ignored.

Compatibility is selective. Chat Completions does not return tool calls. Responses supports function-call/result continuation and optional operator-enabled web search through SearXNG, not arbitrary server-side tool execution. Vision inference and embeddings are not implemented. Each resident model has one native execution slot; admitting concurrent requests does not mean their native execution steps run in parallel.

## Verify or explore

- **Without a GPU:** `docker compose -f compose.test.yaml run --rm acceptance-model-free` exercises the Terrane controllers with a deterministic engine and checks selected Rust contracts. It tests the protocol, not model quality.
- **With a GPU:** `CUSCO_GPU_DEVICE_ID=0 docker compose -f compose.test.yaml run --rm acceptance` fetches the validation model and checks real execution, including checkpoint continuation, scheduling, and the HTTP API. The report is written to `results/acceptance-report.json`.
- **Operate or develop the service:** the [service guide](docs/service-guide.md) covers direct builds, OpenWebUI setup, diagnostics, shutdown, SDK conformance, and the individual performance proofs. The [design outline](docs/outline.md) explains the model-state architecture in depth.

## License

Cusco is licensed under the [MIT License](LICENSE).
