#include "cusco_executor.h"
#include "llama.h"
#include "ggml-backend.h"
#include <algorithm>
#include <atomic>
#include <charconv>
#include <cstring>
#include <memory>
#include <new>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

static uint64_t free_accelerator_bytes() {
    uint64_t available = 0;
    const size_t count = ggml_backend_dev_count();
    for (size_t index = 0; index < count; ++index) {
        ggml_backend_dev_t device = ggml_backend_dev_get(index);
        const auto type = ggml_backend_dev_type(device);
        if (type != GGML_BACKEND_DEVICE_TYPE_GPU
            && type != GGML_BACKEND_DEVICE_TYPE_IGPU) {
            continue;
        }
        size_t free = 0;
        size_t total = 0;
        ggml_backend_dev_memory(device, &free, &total);
        available += static_cast<uint64_t>(free);
    }
    return available;
}

struct cusco_checkpoint {
    std::vector<uint8_t> bytes;
    uint64_t model_identity;
    size_t position;
};

struct cusco_prepared_restore {
    std::vector<uint8_t> bytes;
    uint64_t model_identity;
    size_t position;
};

struct cusco_executor;

struct cusco_representation {
    cusco_executor * owner;
    uint32_t mapping;
    uint64_t identity;
    std::atomic_uint32_t references;
    uint64_t completion_fence;
    mutable std::vector<uint8_t> state;
};

struct cusco_prepared_mapping {
    cusco_executor * owner;
    uint32_t mapping;
    bool committed;
    std::vector<uint8_t> state;
};
struct cusco_executor {
    llama_model * model;
    llama_context * ctx;
    const llama_vocab * vocab;
    uint64_t model_identity;
    std::vector<float> logits;
    std::vector<int32_t> mock_state;
    std::unordered_map<uint32_t, std::vector<int32_t>> mock_mappings;
    std::unordered_map<uint32_t, int32_t> block_table;
    std::unordered_map<uint32_t, size_t> positions;
    std::unordered_set<uint32_t> published_mappings;
    std::unordered_map<uint32_t, cusco_representation *> representations;
    uint32_t active_mapping;
    uint32_t next_mapping;
    // Logical mappings lease native sequence zero only while active.
    uint64_t mapping_epoch;
    uint64_t measured_device_bytes;
    uint64_t reference_switches;
    uint64_t mapping_fork_bytes_copied;
    uint64_t mapping_export_bytes_copied;
    uint64_t mapping_import_bytes_copied;
    uint64_t completion_fence;
    std::atomic_bool cancel;
    int32_t gpu_layers;
};

struct cusco_sampler {
    cusco_executor * owner;
    llama_sampler * raw;
};

static uint64_t hash_bytes(const uint8_t * p, size_t n) {
    uint64_t h = 1469598103934665603ULL;
    for (size_t i = 0; i < n; ++i) {
        h ^= p[i];
        h *= 1099511628211ULL;
    }
    return h;
}

static bool is_mock(const cusco_executor * e) {
    return e->model == nullptr;
}

static bool abort_decode(void * p) {
    return static_cast<cusco_executor *>(p)->cancel.exchange(false);
}

cusco_status cusco_executor_open(
    const char * path,
    uint32_t n_ctx,
    int32_t gpu_layers,
    cusco_executor ** out) try {
    if (!path || !out) {
        return CUSCO_INVALID;
    }
    *out = nullptr;
    // Deterministic backend for model-free ABI lifecycle tests.
    if (strcmp(path, "mock://deterministic") == 0) {
        auto * executor = new (std::nothrow) cusco_executor{};
        if (!executor) {
            return CUSCO_NOMEM;
        }
        executor->model_identity =
            hash_bytes(reinterpret_cast<const uint8_t *>(path), strlen(path));
        executor->block_table.emplace(0, 0);
        executor->positions.emplace(0, 0);
        executor->published_mappings.insert(0);
        executor->representations.emplace(
            0, new cusco_representation{executor, 0, 0, 1, 0, {}});
        executor->active_mapping = 0;
        executor->next_mapping = 1;
        executor->gpu_layers = 0;
        executor->mapping_epoch = 1;
        executor->completion_fence = 0;
        *out = executor;
        return CUSCO_OK;
    }

    llama_backend_init();
    const uint64_t accelerator_bytes_before = free_accelerator_bytes();
    auto model_params = llama_model_default_params();
    model_params.n_gpu_layers = gpu_layers;
    model_params.check_tensors = true;
    auto * model = llama_model_load_from_file(path, model_params);
    if (!model) {
        return CUSCO_BACKEND;
    }

    auto context_params = llama_context_default_params();
    context_params.n_ctx = n_ctx;
    context_params.n_batch = n_ctx;
    context_params.n_seq_max = 1;
    context_params.swa_full = true;
    auto * context = llama_init_from_model(model, context_params);
    if (!context) {
        llama_model_free(model);
        return CUSCO_BACKEND;
    }
    const uint64_t accelerator_bytes_after = free_accelerator_bytes();

    auto * executor = new (std::nothrow) cusco_executor{};
    if (!executor) {
        llama_free(context);
        llama_model_free(model);
        return CUSCO_NOMEM;
    }
    executor->model = model;
    executor->ctx = context;
    executor->vocab = llama_model_get_vocab(model);
    executor->model_identity =
        hash_bytes(reinterpret_cast<const uint8_t *>(path), strlen(path));
    executor->block_table.emplace(0, 0);
    executor->positions.emplace(0, 0);
    executor->published_mappings.insert(0);
    executor->representations.emplace(
        0, new cusco_representation{executor, 0, 0, 1, 0, {}});
    executor->active_mapping = 0;
    executor->next_mapping = 1;
    executor->gpu_layers = gpu_layers;
    executor->measured_device_bytes =
        accelerator_bytes_before > accelerator_bytes_after
            ? accelerator_bytes_before - accelerator_bytes_after
            : 0;
    executor->mapping_epoch = 1;
    executor->completion_fence = 0;
    llama_set_abort_callback(context, abort_decode, executor);
    *out = executor;
    return CUSCO_OK;
} catch (const std::bad_alloc &) {
    return CUSCO_NOMEM;
} catch (...) {
    return CUSCO_BACKEND;
}

void cusco_executor_close(cusco_executor * executor) {
    if (!executor) {
        return;
    }
    for (const auto & [_, representation] : executor->representations) {
        representation->owner = nullptr;
        delete representation;
    }
    if (!is_mock(executor)) {
        llama_free(executor->ctx);
        llama_model_free(executor->model);
    }
    delete executor;
}

cusco_capabilities cusco_executor_capabilities(const cusco_executor * executor) {
    if (is_mock(executor)) {
        return {CUSCO_EXECUTOR_ABI_VERSION, 1, 1, 1, 256, 1, 1, UINT32_MAX};
    }
    return {
        CUSCO_EXECUTOR_ABI_VERSION,
        1,
        llama_model_n_swa(executor->model) > 0 ? 1u : 0u,
        1,
        llama_vocab_n_tokens(executor->vocab),
        1,
        1,
        static_cast<uint32_t>(std::max(0, llama_model_n_ctx_train(executor->model))),
    };
}

cusco_status cusco_executor_model_architecture(
    const cusco_executor * executor,
    char * buffer,
    size_t capacity,
    size_t * size) try {
    if (!executor || !size || (!buffer && capacity != 0)) {
        return CUSCO_INVALID;
    }
    if (is_mock(executor)) {
        constexpr char architecture[] = "gemma4";
        constexpr size_t required = sizeof(architecture) - 1;
        *size = required;
        if (capacity < required) {
            return CUSCO_BUFFER_TOO_SMALL;
        }
        std::memcpy(buffer, architecture, required);
        return CUSCO_OK;
    }
    const int32_t written = llama_model_meta_val_str(
        executor->model, "general.architecture", buffer, capacity);
    if (written < 0) {
        *size = static_cast<size_t>(-written);
        return CUSCO_BUFFER_TOO_SMALL;
    }
    if (written == 0) {
        return CUSCO_BACKEND;
    }
    *size = static_cast<size_t>(written);
    return CUSCO_OK;
} catch (...) {
    return CUSCO_BACKEND;
}

cusco_operating_point cusco_executor_operating_point(const cusco_executor * executor) {
    if (is_mock(executor)) {
        return {0, 0, 0, 0, 0, 0, 1};
    }
    const uint64_t model_bytes = llama_model_size(executor->model);
    const uint64_t context_bytes = llama_state_get_size(executor->ctx);
    const int32_t model_layers = llama_model_n_layer(executor->model);
    const int32_t placed_layers =
        std::max(0, std::min(executor->gpu_layers, model_layers));
    const uint64_t host_bytes = placed_layers == 0
        ? model_bytes + context_bytes
        : model_bytes;
    return {
        model_bytes,
        context_bytes,
        executor->measured_device_bytes,
        host_bytes,
        executor->gpu_layers,
        model_layers,
        placed_layers >= model_layers ? 1u : 0u,
    };
}

cusco_status cusco_executor_tokenize(
    cusco_executor * executor,
    const char * text,
    int32_t ** out,
    size_t * count) try {
    if (!executor || !text || !out || !count) {
        return CUSCO_INVALID;
    }
    *out = nullptr;
    *count = 0;
    if (is_mock(executor)) {
        const size_t size = strlen(text);
        if (size == 0) {
            return CUSCO_OK;
        }
        auto * tokens = new (std::nothrow) int32_t[size];
        if (!tokens) {
            return CUSCO_NOMEM;
        }
        for (size_t i = 0; i < size; ++i) {
            tokens[i] = static_cast<uint8_t>(text[i]);
        }
        *out = tokens;
        *count = size;
        return CUSCO_OK;
    }

    int size = llama_tokenize(
        executor->vocab,
        text,
        strlen(text),
        nullptr,
        0,
        llama_vocab_get_add_bos(executor->vocab),
        true);
    if (size >= 0) {
        return CUSCO_BACKEND;
    }
    size = -size;
    auto * tokens = new (std::nothrow) int32_t[size];
    if (!tokens) {
        return CUSCO_NOMEM;
    }
    const int written = llama_tokenize(
        executor->vocab,
        text,
        strlen(text),
        tokens,
        size,
        llama_vocab_get_add_bos(executor->vocab),
        true);
    if (written < 0) {
        delete[] tokens;
        return CUSCO_BACKEND;
    }
    *out = tokens;
    *count = written;
    return CUSCO_OK;
} catch (const std::bad_alloc &) {
    return CUSCO_NOMEM;
} catch (...) {
    return CUSCO_BACKEND;
}

uint32_t cusco_executor_token_is_eog(const cusco_executor * executor, int32_t token) {
    if (!executor) {
        return 0;
    }
    if (is_mock(executor)) {
        return token == 1 || token == 106 ? 1u : 0u;
    }
    return llama_vocab_is_eog(executor->vocab, token) ? 1u : 0u;
}

void cusco_executor_tokens_free(int32_t * tokens) {
    delete[] tokens;
}

cusco_status cusco_executor_render_token(
    cusco_executor * executor,
    int32_t token,
    uint8_t * buffer,
    size_t capacity,
    size_t * size) try {
    if (!executor || !size || (!buffer && capacity != 0)) {
        return CUSCO_INVALID;
    }
    if (is_mock(executor)) {
        char encoded[32];
        const auto result = std::to_chars(encoded, encoded + sizeof(encoded), token);
        if (result.ec != std::errc()) {
            return CUSCO_BACKEND;
        }
        const size_t required = static_cast<size_t>(result.ptr - encoded);
        *size = required;
        if (capacity < required) {
            return CUSCO_BUFFER_TOO_SMALL;
        }
        memcpy(buffer, encoded, required);
        return CUSCO_OK;
    }
    const int32_t required =
        llama_token_to_piece(executor->vocab, token, nullptr, 0, 0, true);
    if (required >= 0) {
        return CUSCO_BACKEND;
    }
    *size = static_cast<size_t>(-required);
    if (capacity < *size) {
        return CUSCO_BUFFER_TOO_SMALL;
    }
    const int32_t written = llama_token_to_piece(
        executor->vocab, token, reinterpret_cast<char *>(buffer), capacity, 0, true);
    if (written < 0) {
        return CUSCO_BACKEND;
    }
    *size = static_cast<size_t>(written);
    return CUSCO_OK;
} catch (...) {
    return CUSCO_BACKEND;
}

cusco_status cusco_sampler_create(
    cusco_executor * executor,
    const cusco_sampler_config * config,
    cusco_sampler ** out) try {
    if (!executor || !config || !out || config->top_p <= 0.0F || config->top_p > 1.0F) {
        return CUSCO_INVALID;
    }
    *out = nullptr;
    llama_sampler * raw = nullptr;
    if (!is_mock(executor)) {
        const bool constrained = config->grammar && config->grammar[0] != '\0';
        if (!constrained && config->temperature <= 0.0F) {
            raw = llama_sampler_init_greedy();
        } else {
            raw = llama_sampler_chain_init(llama_sampler_chain_default_params());
            if (!raw) {
                return CUSCO_NOMEM;
            }
            if (constrained) {
                auto * grammar = llama_sampler_init_grammar(
                    executor->vocab, config->grammar, "root");
                if (!grammar) {
                    llama_sampler_free(raw);
                    return CUSCO_INVALID;
                }
                llama_sampler_chain_add(raw, grammar);
            }
            if (config->temperature <= 0.0F) {
                auto * greedy = llama_sampler_init_greedy();
                if (!greedy) {
                    llama_sampler_free(raw);
                    return CUSCO_NOMEM;
                }
                llama_sampler_chain_add(raw, greedy);
            } else {
                auto * top_p = llama_sampler_init_top_p(config->top_p, 1);
                auto * temp = llama_sampler_init_temp(config->temperature);
                auto * dist = llama_sampler_init_dist(config->seed);
                if (!top_p || !temp || !dist) {
                    llama_sampler_free(top_p);
                    llama_sampler_free(temp);
                    llama_sampler_free(dist);
                    llama_sampler_free(raw);
                    return CUSCO_NOMEM;
                }
                llama_sampler_chain_add(raw, top_p);
                llama_sampler_chain_add(raw, temp);
                llama_sampler_chain_add(raw, dist);
            }
        }
    }
    if (!is_mock(executor) && !raw) {
        return CUSCO_NOMEM;
    }
    auto * sampler = new (std::nothrow) cusco_sampler{executor, raw};
    if (!sampler) {
        llama_sampler_free(raw);
        return CUSCO_NOMEM;
    }
    *out = sampler;
    return CUSCO_OK;
} catch (...) {
    return CUSCO_BACKEND;
}

void cusco_sampler_free(cusco_sampler * sampler) {
    if (sampler) {
        if (sampler->raw) {
            llama_sampler_free(sampler->raw);
        }
        delete sampler;
    }
}

cusco_status cusco_sampler_sample(
    cusco_sampler * sampler,
    const float * logits,
    size_t logits_len,
    int32_t * token) try {
    if (!sampler || !logits || logits_len == 0 || !token) {
        return CUSCO_INVALID;
    }
    if (is_mock(sampler->owner)) {
        size_t selected = 0;
        for (size_t i = 1; i < logits_len; ++i) {
            if (logits[i] > logits[selected]) {
                selected = i;
            }
        }
        *token = static_cast<int32_t>(selected);
    } else {
        const int32_t vocab_size = llama_vocab_n_tokens(sampler->owner->vocab);
        if (vocab_size <= 0 || logits_len != static_cast<size_t>(vocab_size)) {
            return CUSCO_INVALID;
        }
        std::vector<llama_token_data> candidates;
        candidates.reserve(logits_len);
        for (size_t i = 0; i < logits_len; ++i) {
            candidates.push_back(
                llama_token_data{static_cast<llama_token>(i), logits[i], 0.0F});
        }
        llama_token_data_array array{
            candidates.data(), candidates.size(), -1, false};
        llama_sampler_apply(sampler->raw, &array);
        if (array.selected < 0 ||
            static_cast<size_t>(array.selected) >= array.size) {
            return CUSCO_BACKEND;
        }
        *token = array.data[array.selected].id;
        llama_sampler_accept(sampler->raw, *token);
    }
    return CUSCO_OK;
} catch (...) {
    return CUSCO_BACKEND;
}

cusco_status cusco_executor_decode(
    cusco_executor * executor,
    const int32_t * tokens,
    size_t count,
    cusco_decode_result * out) try {
    if (!executor || !tokens || count == 0 || !out) {
        return CUSCO_INVALID;
    }
    if (executor->cancel.exchange(false)) {
        return CUSCO_CANCELLED;
    }
    if (is_mock(executor)) {
        executor->mock_state.insert(executor->mock_state.end(), tokens, tokens + count);
        executor->mock_mappings[executor->active_mapping] = executor->mock_state;
        executor->positions[executor->active_mapping] = executor->mock_state.size();
        const uint64_t hash = hash_bytes(
            reinterpret_cast<const uint8_t *>(executor->mock_state.data()),
            executor->mock_state.size() * sizeof(int32_t));
        executor->logits.resize(8);
        size_t token = 0;
        for (size_t i = 0; i < executor->logits.size(); ++i) {
            executor->logits[i] = static_cast<float>((hash >> (i * 8)) & 255) / 255.0f;
            if (executor->logits[i] > executor->logits[token]) {
                token = i;
            }
        }
        *out = {executor->logits.data(), executor->logits.size(), static_cast<int32_t>(token)};
        return CUSCO_OK;
    }

    auto batch = llama_batch_init(static_cast<int32_t>(count), 0, 1);
    batch.n_tokens = static_cast<int32_t>(count);
    const int32_t sequence = executor->block_table.at(executor->active_mapping);
    const size_t position = executor->positions.at(executor->active_mapping);
    for (size_t i = 0; i < count; ++i) {
        batch.token[i] = tokens[i];
        batch.pos[i] = static_cast<llama_pos>(position + i);
        batch.n_seq_id[i] = 1;
        batch.seq_id[i][0] = sequence;
        batch.logits[i] = i + 1 == count;
    }
    const int result = llama_decode(executor->ctx, batch);
    llama_batch_free(batch);

    if (result == 2) {
        return CUSCO_CANCELLED;
    }
    if (result != 0) {
        return CUSCO_BACKEND;
    }
    executor->positions[executor->active_mapping] = position + count;
    llama_synchronize(executor->ctx);
    const float * logits = llama_get_logits_ith(executor->ctx, -1);
    const int32_t vocabulary = llama_vocab_n_tokens(executor->vocab);
    executor->logits.assign(logits, logits + vocabulary);
    int32_t token = 0;
    for (int32_t i = 1; i < vocabulary; ++i) {
        if (executor->logits[i] > executor->logits[token]) {
            token = i;
        }
    }
    *out = {executor->logits.data(), executor->logits.size(), token};
    return CUSCO_OK;
} catch (const std::bad_alloc &) {
    return CUSCO_NOMEM;
} catch (...) {
    return CUSCO_BACKEND;
}

cusco_status cusco_executor_capture(
    cusco_executor * executor,
    cusco_checkpoint ** out) try {
    if (!executor || !out) {
        return CUSCO_INVALID;
    }
    *out = nullptr;
    auto checkpoint = std::make_unique<cusco_checkpoint>();
    checkpoint->model_identity = executor->model_identity;
    checkpoint->position = executor->positions.at(executor->active_mapping);
    if (is_mock(executor)) {
        checkpoint->bytes.resize(executor->mock_state.size() * sizeof(int32_t));
        memcpy(
            checkpoint->bytes.data(),
            executor->mock_state.data(),
            checkpoint->bytes.size());
    } else {
        checkpoint->bytes.resize(llama_state_get_size(executor->ctx));
        const size_t written = llama_state_get_data(
            executor->ctx, checkpoint->bytes.data(), checkpoint->bytes.size());
        if (written == 0) {
            return CUSCO_BACKEND;
        }
        checkpoint->bytes.resize(written);
    }
    *out = checkpoint.release();
    return CUSCO_OK;
} catch (const std::bad_alloc &) {
    return CUSCO_NOMEM;
} catch (...) {
    return CUSCO_BACKEND;
}

void cusco_checkpoint_free(cusco_checkpoint * checkpoint) {
    delete checkpoint;
}

size_t cusco_checkpoint_size(const cusco_checkpoint * checkpoint) {
    return checkpoint ? checkpoint->bytes.size() : 0;
}

uint64_t cusco_checkpoint_checksum(const cusco_checkpoint * checkpoint) {
    return checkpoint ? hash_bytes(checkpoint->bytes.data(), checkpoint->bytes.size()) : 0;
}

cusco_status cusco_executor_prepare_restore(
    cusco_executor * executor,
    const cusco_checkpoint * checkpoint,
    uint64_t checksum,
    cusco_prepared_restore ** out) try {
    if (!executor || !checkpoint || !out) {
        return CUSCO_INVALID;
    }
    *out = nullptr;
    if (checkpoint->model_identity != executor->model_identity
        || cusco_checkpoint_checksum(checkpoint) != checksum) {
        return CUSCO_INCOMPATIBLE;
    }
    auto prepared = std::make_unique<cusco_prepared_restore>();
    prepared->bytes = checkpoint->bytes;
    prepared->model_identity = checkpoint->model_identity;
    prepared->position = checkpoint->position;
    *out = prepared.release();
    return CUSCO_OK;
} catch (const std::bad_alloc &) {
    return CUSCO_NOMEM;
} catch (...) {
    return CUSCO_BACKEND;
}

void cusco_prepared_restore_free(cusco_prepared_restore * prepared) {
    delete prepared;
}

cusco_status cusco_executor_commit_restore(
    cusco_executor * executor,
    cusco_prepared_restore * prepared_raw) try {
    std::unique_ptr<cusco_prepared_restore> prepared(prepared_raw);
    if (!executor || !prepared) {
        return CUSCO_INVALID;
    }
    if (prepared->model_identity != executor->model_identity) {
        return CUSCO_INCOMPATIBLE;
    }
    if (is_mock(executor)) {
        std::vector<int32_t> replacement(prepared->bytes.size() / sizeof(int32_t));
        memcpy(replacement.data(), prepared->bytes.data(), prepared->bytes.size());
        executor->mock_state.swap(replacement);
        executor->positions[executor->active_mapping] = prepared->position;
        return CUSCO_OK;
    }

    std::vector<uint8_t> prior(llama_state_get_size(executor->ctx));
    const size_t prior_size = llama_state_get_data(
        executor->ctx, prior.data(), prior.size());
    if (prior_size == 0) {
        return CUSCO_BACKEND;
    }
    prior.resize(prior_size);

    const size_t restored = llama_state_set_data(
        executor->ctx, prepared->bytes.data(), prepared->bytes.size());
    if (restored == prepared->bytes.size()) {
        executor->positions[executor->active_mapping] = prepared->position;
        return CUSCO_OK;
    }

    const size_t rolled_back = llama_state_set_data(
        executor->ctx, prior.data(), prior.size());
    return rolled_back == prior.size() ? CUSCO_BACKEND : CUSCO_ROLLBACK_FAILED;
} catch (const std::bad_alloc &) {
    return CUSCO_NOMEM;
} catch (...) {
    return CUSCO_BACKEND;
}

static void discard_mapping(cusco_prepared_mapping * prepared) {
    if (!prepared || prepared->committed || !prepared->owner) {
        return;
    }
    auto * executor = prepared->owner;
    executor->block_table.erase(prepared->mapping);
    executor->positions.erase(prepared->mapping);
    executor->mock_mappings.erase(prepared->mapping);
}

static bool valid_representation(
    const cusco_executor * executor, const cusco_representation * representation) {
    return executor && representation && representation->owner == executor
        && executor->published_mappings.count(representation->mapping) != 0
        && executor->representations.at(representation->mapping) == representation;
}

static void reclaim_representation(
    cusco_executor * executor, cusco_representation * representation) {
    const uint32_t mapping = representation->mapping;
    executor->published_mappings.erase(mapping);
    executor->block_table.erase(mapping);
    executor->positions.erase(mapping);
    executor->mock_mappings.erase(mapping);
    executor->representations.erase(mapping);
    representation->owner = nullptr;
    delete representation;
}

static void reclaim_unreferenced_representations(cusco_executor * executor) {
    std::vector<cusco_representation *> reclaimable;
    for (const auto & [mapping, representation] : executor->representations) {
        if (mapping != executor->active_mapping
            && representation->references.load(std::memory_order_acquire) == 0) {
            reclaimable.push_back(representation);
        }
    }
    for (auto * representation : reclaimable) {
        reclaim_representation(executor, representation);
    }
}

static bool snapshot_active_sequence(
    cusco_executor * executor, std::vector<uint8_t> & state) {
    const size_t required = llama_state_seq_get_size(executor->ctx, 0);
    state.resize(required);
    if (required == 0) {
        return true;
    }
    const size_t copied =
        llama_state_seq_get_data(executor->ctx, state.data(), state.size(), 0);
    if (copied != required) {
        state.clear();
        return false;
    }
    return true;
}

cusco_status cusco_executor_active_representation(
    cusco_executor * executor, cusco_representation ** out) {
    if (!executor || !out) {
        return CUSCO_INVALID;
    }
    auto * representation = executor->representations.at(executor->active_mapping);
    cusco_representation_retain(representation);
    *out = representation;
    return CUSCO_OK;
}

void cusco_representation_retain(cusco_representation * representation) {
    if (representation && representation->owner) {
        representation->references.fetch_add(1, std::memory_order_relaxed);
    }
}

void cusco_representation_release(cusco_representation * representation) {
    if (representation && representation->owner) {
        representation->references.fetch_sub(1, std::memory_order_acq_rel);
    }
}

uint64_t cusco_representation_identity(const cusco_representation * representation) {
    return representation ? representation->identity : 0;
}

cusco_status cusco_representation_describe(
    const cusco_representation * representation,
    cusco_representation_descriptor * out) {
    if (!representation || !representation->owner || !out) {
        return CUSCO_INVALID;
    }
    auto * executor = representation->owner;
    const size_t position = executor->positions.at(representation->mapping);
    size_t bytes;
    if (is_mock(executor)) {
        bytes = (representation->mapping == executor->active_mapping
            ? executor->mock_state.size()
            : executor->mock_mappings.at(representation->mapping).size()) * sizeof(int32_t);
    } else if (representation->mapping == executor->active_mapping) {
        bytes = llama_state_seq_get_size(executor->ctx, 0);
    } else {
        bytes = representation->state.size();
    }
    *out = {representation->identity, 7, 0, position, bytes,
        representation->completion_fence};
    return CUSCO_OK;
}

cusco_status cusco_executor_prepare_mapping_fork(
    cusco_executor * executor,
    const cusco_representation * source,
    cusco_prepared_mapping ** out) try {
    reclaim_unreferenced_representations(executor);
    if (!valid_representation(executor, source) || !out) {
        return CUSCO_INVALID;
    }
    *out = nullptr;
    const uint32_t source_mapping = source->mapping;
    const uint32_t mapping = executor->next_mapping++;
    auto prepared = std::unique_ptr<
        cusco_prepared_mapping, decltype(&cusco_prepared_mapping_free)>(
        new cusco_prepared_mapping{executor, mapping, false, {}},
        cusco_prepared_mapping_free);
    if (is_mock(executor)) {
        const auto & state = source_mapping == executor->active_mapping
            ? executor->mock_state : executor->mock_mappings.at(source_mapping);
        executor->mock_mappings.emplace(mapping, state);
        executor->mapping_fork_bytes_copied += state.size() * sizeof(int32_t);
    } else {
        if (source_mapping == executor->active_mapping) {
            if (!snapshot_active_sequence(executor, prepared->state)) {
                return CUSCO_BACKEND;
            }
        } else {
            prepared->state = source->state;
        }
        executor->mapping_fork_bytes_copied += prepared->state.size();
    }
    executor->block_table.emplace(mapping, 0);
    executor->positions.emplace(mapping, executor->positions.at(source_mapping));
    *out = prepared.release();
    return CUSCO_OK;
} catch (const std::bad_alloc &) {
    return CUSCO_NOMEM;
} catch (...) {
    return CUSCO_BACKEND;
}

void cusco_prepared_mapping_free(cusco_prepared_mapping * prepared) {
    discard_mapping(prepared);
    delete prepared;
}

cusco_status cusco_executor_commit_mapping(
    cusco_executor * executor,
    cusco_prepared_mapping * prepared_raw,
    cusco_representation ** out) try {
    std::unique_ptr<cusco_prepared_mapping> prepared(prepared_raw);
    if (!executor || !prepared || !out || prepared->owner != executor
        || prepared->committed) {
        discard_mapping(prepared.get());
        return CUSCO_INVALID;
    }
    const uint64_t identity = ++executor->mapping_epoch;
    auto representation = std::unique_ptr<cusco_representation>(
        new cusco_representation{executor, prepared->mapping, identity, 1,
            ++executor->completion_fence, std::move(prepared->state)});
    executor->published_mappings.insert(prepared->mapping);
    executor->representations.emplace(prepared->mapping, representation.get());
    prepared->committed = true;
    *out = representation.release();
    return CUSCO_OK;
} catch (const std::bad_alloc &) {
    return CUSCO_NOMEM;
} catch (...) {
    return CUSCO_BACKEND;
}

cusco_status cusco_executor_activate_mapping(
    cusco_executor * executor, const cusco_representation * representation) try {
    if (!valid_representation(executor, representation)) {
        return CUSCO_INVALID;
    }
    const uint32_t mapping = representation->mapping;
    if (mapping == executor->active_mapping) {
        return CUSCO_OK;
    }
    const uint32_t previous = executor->active_mapping;
    const auto previous_it = executor->representations.find(previous);
    if (previous_it == executor->representations.end()) {
        return CUSCO_BACKEND;
    }
    auto * previous_representation = previous_it->second;
    if (is_mock(executor)) {
        executor->mock_mappings[previous] = executor->mock_state;
        executor->mock_state = executor->mock_mappings.at(mapping);
        executor->mock_mappings.erase(mapping);
    } else {
        std::vector<uint8_t> prior;
        if (!snapshot_active_sequence(executor, prior)) {
            return CUSCO_BACKEND;
        }
        llama_memory_seq_rm(llama_get_memory(executor->ctx), 0, -1, -1);
        const auto & target = representation->state;
        if (!target.empty()
            && llama_state_seq_set_data(executor->ctx, target.data(), target.size(), 0)
                != target.size()) {
            llama_memory_seq_rm(llama_get_memory(executor->ctx), 0, -1, -1);
            const bool rolled_back = prior.empty()
                || llama_state_seq_set_data(
                    executor->ctx, prior.data(), prior.size(), 0) == prior.size();
            return rolled_back ? CUSCO_BACKEND : CUSCO_ROLLBACK_FAILED;
        }
        representation->state.clear();
        representation->state.shrink_to_fit();
        previous_representation->state = std::move(prior);
    }
    executor->active_mapping = mapping;
    executor->reference_switches++;
    auto * prior = previous_representation;
    if (previous != 0 && prior->references.load(std::memory_order_acquire) == 0) {
        reclaim_representation(executor, prior);
    }
    return CUSCO_OK;
} catch (const std::bad_alloc &) {
    return CUSCO_NOMEM;
} catch (...) {
    return CUSCO_BACKEND;
}

size_t cusco_executor_mapping_state_size(
    cusco_executor * executor, const cusco_representation * representation) {
    if (!valid_representation(executor, representation)) {
        return 0;
    }
    if (is_mock(executor)) {
        const auto & state = representation->mapping == executor->active_mapping
            ? executor->mock_state : executor->mock_mappings.at(representation->mapping);
        return state.size() * sizeof(int32_t);
    }
    return representation->mapping == executor->active_mapping
        ? llama_state_seq_get_size(executor->ctx, 0)
        : representation->state.size();
}

cusco_status cusco_executor_export_mapping(
    cusco_executor * executor, const cusco_representation * representation,
    uint8_t * buffer, size_t capacity, size_t * written, size_t * position) try {
    if (!valid_representation(executor, representation) || !written || !position) {
        return CUSCO_INVALID;
    }
    const size_t required = cusco_executor_mapping_state_size(executor, representation);
    *written = required;
    *position = executor->positions.at(representation->mapping);
    if (capacity < required || (required != 0 && !buffer)) {
        return CUSCO_BUFFER_TOO_SMALL;
    }
    if (is_mock(executor)) {
        const auto & state = representation->mapping == executor->active_mapping
            ? executor->mock_state : executor->mock_mappings.at(representation->mapping);
        if (required != 0) memcpy(buffer, state.data(), required);
    } else if (representation->mapping == executor->active_mapping) {
        const size_t copied =
            llama_state_seq_get_data(executor->ctx, buffer, capacity, 0);
        if (copied != required) return CUSCO_BACKEND;
    } else if (required != 0) {
        memcpy(buffer, representation->state.data(), required);
    }
    executor->mapping_export_bytes_copied += required;
    return CUSCO_OK;
} catch (...) {
    return CUSCO_BACKEND;
}

cusco_status cusco_executor_import_mapping(
    cusco_executor * executor, const uint8_t * buffer, size_t size,
    size_t position, cusco_representation ** out) try {
    if (!executor || !out || (size != 0 && !buffer)) return CUSCO_INVALID;
    reclaim_unreferenced_representations(executor);
    *out = nullptr;
    const uint32_t mapping = executor->next_mapping++;
    std::vector<uint8_t> imported;
    if (is_mock(executor)) {
        if (size % sizeof(int32_t) != 0) return CUSCO_INCOMPATIBLE;
        std::vector<int32_t> state(size / sizeof(int32_t));
        if (size != 0) memcpy(state.data(), buffer, size);
        executor->mock_mappings.emplace(mapping, std::move(state));
    } else {
        auto context_params = llama_context_default_params();
        context_params.n_ctx = llama_n_ctx(executor->ctx);
        context_params.n_batch = context_params.n_ctx;
        context_params.n_seq_max = 1;
        context_params.swa_full = true;
        llama_context * validation =
            llama_init_from_model(executor->model, context_params);
        if (!validation) return CUSCO_NOMEM;
        const size_t consumed =
            llama_state_seq_set_data(validation, buffer, size, 0);
        llama_free(validation);
        if (consumed != size) return CUSCO_INCOMPATIBLE;
        imported.resize(size);
        if (size != 0) memcpy(imported.data(), buffer, size);
    }
    try {
        executor->block_table.emplace(mapping, 0);
        executor->positions.emplace(mapping, position);
        const uint64_t identity = ++executor->mapping_epoch;
        auto representation = std::unique_ptr<cusco_representation>(
            new cusco_representation{executor, mapping, identity, 1,
                ++executor->completion_fence, std::move(imported)});
        executor->published_mappings.insert(mapping);
        executor->representations.emplace(mapping, representation.get());
        *out = representation.release();
        executor->mapping_import_bytes_copied += size;
        return CUSCO_OK;
    } catch (...) {
        executor->published_mappings.erase(mapping);
        executor->representations.erase(mapping);
        executor->positions.erase(mapping);
        executor->block_table.erase(mapping);
        executor->mock_mappings.erase(mapping);
        throw;
    }
} catch (const std::bad_alloc &) {
    return CUSCO_NOMEM;
} catch (...) {
    return CUSCO_BACKEND;
}

uint64_t cusco_executor_active_mapping_identity(const cusco_executor * executor) {
    return executor ? executor->representations.at(executor->active_mapping)->identity : 0;
}
size_t cusco_executor_mapping_count(const cusco_executor * executor) {
    if (!executor) return 0;
    size_t count = 0;
    for (const auto & [mapping, representation] : executor->representations) {
        if (mapping == executor->active_mapping
            || representation->references.load(std::memory_order_acquire) != 0) {
            count++;
        }
    }
    return count;
}
uint64_t cusco_executor_reference_switches(const cusco_executor * executor) {
    return executor ? executor->reference_switches : 0;
}
uint64_t cusco_executor_mapping_fork_bytes_copied(const cusco_executor * executor) {
    return executor ? executor->mapping_fork_bytes_copied : 0;
}
uint64_t cusco_executor_mapping_export_bytes_copied(const cusco_executor * executor) {
    return executor ? executor->mapping_export_bytes_copied : 0;
}
uint64_t cusco_executor_mapping_import_bytes_copied(const cusco_executor * executor) {
    return executor ? executor->mapping_import_bytes_copied : 0;
}
uint64_t cusco_executor_mapping_bytes_copied(const cusco_executor * executor) {
    return executor
        ? executor->mapping_fork_bytes_copied
            + executor->mapping_export_bytes_copied
            + executor->mapping_import_bytes_copied
        : 0;
}
uint32_t cusco_executor_graph_recaptures_supported(const cusco_executor *) {
    return 0;
}
uint64_t cusco_executor_graph_recaptures(const cusco_executor *) {
    return 0;
}

cusco_status cusco_executor_replace_state_for_proof(
    cusco_executor * executor,
    const int32_t * tokens,
    size_t count) try {
    if (!executor) {
        return CUSCO_INVALID;
    }
    if (is_mock(executor)) {
        executor->mock_state.clear();
    } else {
        llama_memory_clear(llama_get_memory(executor->ctx), true);
    }
    executor->positions[executor->active_mapping] = 0;
    if (count == 0) {
        return CUSCO_OK;
    }
    cusco_decode_result ignored{};
    return cusco_executor_decode(executor, tokens, count, &ignored);
} catch (const std::bad_alloc &) {
    return CUSCO_NOMEM;
} catch (...) {
    return CUSCO_BACKEND;
}

void cusco_executor_cancel(cusco_executor * executor) {
    if (executor) {
        executor->cancel = true;
    }
}
void cusco_executor_reset_cancel(cusco_executor * executor) {
    if (executor) {
        executor->cancel = false;
    }
}


void cusco_executor_cancel_next_decode_for_proof(cusco_executor * executor) {
    cusco_executor_cancel(executor);
}
