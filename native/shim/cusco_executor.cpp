#include "cusco_executor.h"
#include "llama.h"

#include <atomic>
#include <charconv>
#include <cstring>
#include <memory>
#include <new>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

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

struct cusco_prepared_mapping {
    cusco_executor * owner;
    uint32_t mapping;
    int32_t sequence;
    bool committed;
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
    uint32_t active_mapping;
    uint32_t next_mapping;
    int32_t next_sequence;
    uint64_t mapping_epoch;
    uint64_t reference_switches;
    uint64_t mapped_bytes_copied;
    std::atomic_bool cancel;
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
    // Phase 1-only deterministic backend for model-free ABI lifecycle tests.
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
        executor->active_mapping = 0;
        executor->next_mapping = 1;
        executor->next_sequence = 1;
        executor->mapping_epoch = 1;
        *out = executor;
        return CUSCO_OK;
    }

    llama_backend_init();
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
    context_params.n_seq_max = 64;
    context_params.swa_full = true;
    auto * context = llama_init_from_model(model, context_params);
    if (!context) {
        llama_model_free(model);
        return CUSCO_BACKEND;
    }

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
    executor->active_mapping = 0;
    executor->next_mapping = 1;
    executor->next_sequence = 1;
    executor->mapping_epoch = 1;
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
    if (!is_mock(executor)) {
        llama_free(executor->ctx);
        llama_model_free(executor->model);
    }
    delete executor;
}

cusco_capabilities cusco_executor_capabilities(const cusco_executor * executor) {
    if (is_mock(executor)) {
        return {CUSCO_EXECUTOR_ABI_VERSION, 1, 1, 1, 256, 1, 64};
    }
    return {
        CUSCO_EXECUTOR_ABI_VERSION,
        1,
        llama_model_n_swa(executor->model) > 0 ? 1u : 0u,
        1,
        llama_vocab_n_tokens(executor->vocab),
        1,
        64,
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

    int size = llama_tokenize(executor->vocab, text, strlen(text), nullptr, 0, true, true);
    if (size >= 0) {
        return CUSCO_BACKEND;
    }
    size = -size;
    auto * tokens = new (std::nothrow) int32_t[size];
    if (!tokens) {
        return CUSCO_NOMEM;
    }
    const int written = llama_tokenize(
        executor->vocab, text, strlen(text), tokens, size, true, true);
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

cusco_status cusco_sampler_greedy(
    cusco_executor * executor,
    cusco_sampler ** out) try {
    if (!executor || !out) {
        return CUSCO_INVALID;
    }
    *out = nullptr;
    llama_sampler * raw = is_mock(executor) ? nullptr : llama_sampler_init_greedy();
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
    cusco_executor * executor,
    int32_t * token) try {
    if (!sampler || !executor || !token || sampler->owner != executor) {
        return CUSCO_INVALID;
    }
    if (is_mock(executor)) {
        if (executor->logits.empty()) {
            return CUSCO_INVALID;
        }
        size_t selected = 0;
        for (size_t i = 1; i < executor->logits.size(); ++i) {
            if (executor->logits[i] > executor->logits[selected]) {
                selected = i;
            }
        }
        *token = static_cast<int32_t>(selected);
    } else {
        *token = llama_sampler_sample(sampler->raw, executor->ctx, -1);
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
    if (!is_mock(executor)) {
        llama_memory_seq_rm(
            llama_get_memory(executor->ctx), prepared->sequence, -1, -1);
    }
    executor->block_table.erase(prepared->mapping);
    executor->positions.erase(prepared->mapping);
    executor->mock_mappings.erase(prepared->mapping);
}

cusco_status cusco_executor_prepare_mapping_fork(
    cusco_executor * executor,
    uint32_t source_mapping,
    cusco_prepared_mapping ** out) try {
    if (!executor || !out
        || executor->published_mappings.count(source_mapping) == 0
        || executor->block_table.size() >= 64) {
        return CUSCO_INVALID;
    }
    *out = nullptr;
    const uint32_t mapping = executor->next_mapping++;
    const int32_t sequence = executor->next_sequence++;
    const int32_t source_sequence = executor->block_table.at(source_mapping);
    std::unique_ptr<cusco_prepared_mapping, decltype(&cusco_prepared_mapping_free)>
        prepared(
            new cusco_prepared_mapping{executor, mapping, sequence, false},
            cusco_prepared_mapping_free);
    if (is_mock(executor)) {
        const auto & state = source_mapping == executor->active_mapping
            ? executor->mock_state
            : executor->mock_mappings.at(source_mapping);
        executor->mock_mappings.emplace(mapping, state);
    } else {
        llama_memory_seq_cp(
            llama_get_memory(executor->ctx), source_sequence, sequence, -1, -1);
    }
    executor->block_table.emplace(mapping, sequence);
    executor->positions.emplace(mapping, executor->positions.at(source_mapping));
    // Keep the unpublished mapping under rollback-aware RAII ownership until
    // every fallible allocation has completed.
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
    uint32_t * mapping) try {
    std::unique_ptr<cusco_prepared_mapping> prepared(prepared_raw);
    if (!executor || !prepared || !mapping || prepared->owner != executor
        || prepared->committed) {
        discard_mapping(prepared.get());
        return CUSCO_INVALID;
    }
    executor->published_mappings.insert(prepared->mapping);
    prepared->committed = true;
    *mapping = prepared->mapping;
    return CUSCO_OK;
} catch (...) {
    return CUSCO_BACKEND;
}

cusco_status cusco_executor_activate_mapping(
    cusco_executor * executor, uint32_t mapping) {
    if (!executor || executor->published_mappings.count(mapping) == 0) {
        return CUSCO_INVALID;
    }
    if (mapping == executor->active_mapping) {
        return CUSCO_OK;
    }
    if (is_mock(executor)) {
        executor->mock_mappings[executor->active_mapping] = executor->mock_state;
        executor->mock_state = executor->mock_mappings.at(mapping);
    }
    executor->active_mapping = mapping;
    executor->reference_switches++;
    return CUSCO_OK;
}

cusco_status cusco_executor_remove_mapping(
    cusco_executor * executor, uint32_t mapping) {
    if (!executor || mapping == 0 || mapping == executor->active_mapping
        || executor->published_mappings.erase(mapping) == 0) {
        return CUSCO_INVALID;
    }
    if (!is_mock(executor)) {
        llama_memory_seq_rm(
            llama_get_memory(executor->ctx), executor->block_table.at(mapping), -1, -1);
    }
    executor->block_table.erase(mapping);
    executor->positions.erase(mapping);
    executor->mock_mappings.erase(mapping);
    return CUSCO_OK;
}

uint32_t cusco_executor_active_mapping(const cusco_executor * executor) {
    return executor ? executor->active_mapping : 0;
}
size_t cusco_executor_mapping_count(const cusco_executor * executor) {
    return executor ? executor->published_mappings.size() : 0;
}
uint64_t cusco_executor_reference_switches(const cusco_executor * executor) {
    return executor ? executor->reference_switches : 0;
}
uint64_t cusco_executor_mapped_bytes_copied(const cusco_executor * executor) {
    return executor ? executor->mapped_bytes_copied : 0;
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
