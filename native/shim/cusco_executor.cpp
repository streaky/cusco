#include "cusco_executor.h"
#include "llama.h"

#include <atomic>
#include <cstring>
#include <memory>
#include <new>
#include <vector>

struct cusco_checkpoint {
    std::vector<uint8_t> bytes;
    uint64_t model_identity;
};

struct cusco_prepared_restore {
    std::vector<uint8_t> bytes;
    uint64_t model_identity;
};

struct cusco_executor {
    llama_model * model;
    llama_context * ctx;
    const llama_vocab * vocab;
    uint64_t model_identity;
    std::vector<float> logits;
    std::vector<int32_t> mock_state;
    std::atomic_bool cancel;
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
        *out = new (std::nothrow) cusco_executor{
            nullptr,
            nullptr,
            nullptr,
            hash_bytes(reinterpret_cast<const uint8_t *>(path), strlen(path)),
            {},
            {},
            false,
        };
        return *out ? CUSCO_OK : CUSCO_NOMEM;
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
    context_params.n_seq_max = 1;
    context_params.swa_full = true;
    auto * context = llama_init_from_model(model, context_params);
    if (!context) {
        llama_model_free(model);
        return CUSCO_BACKEND;
    }

    auto * executor = new (std::nothrow) cusco_executor{
        model,
        context,
        llama_model_get_vocab(model),
        hash_bytes(reinterpret_cast<const uint8_t *>(path), strlen(path)),
        {},
        {},
        false,
    };
    if (!executor) {
        llama_free(context);
        llama_model_free(model);
        return CUSCO_NOMEM;
    }
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
        return {CUSCO_EXECUTOR_ABI_VERSION, 1, 1, 1, 256};
    }
    return {
        CUSCO_EXECUTOR_ABI_VERSION,
        1,
        llama_model_n_swa(executor->model) > 0 ? 1u : 0u,
        1,
        llama_vocab_n_tokens(executor->vocab),
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
        const uint64_t hash = hash_bytes(
            reinterpret_cast<const uint8_t *>(executor->mock_state.data()),
            executor->mock_state.size() * sizeof(int32_t));
        executor->logits.resize(8);
        for (size_t i = 0; i < executor->logits.size(); ++i) {
            executor->logits[i] = static_cast<float>((hash >> (i * 8)) & 255) / 255.0f;
        }
        *out = {executor->logits.data(), executor->logits.size(), static_cast<int32_t>(hash % 256)};
        return CUSCO_OK;
    }

    std::vector<int32_t> copy(tokens, tokens + count);
    const int result = llama_decode(
        executor->ctx, llama_batch_get_one(copy.data(), static_cast<int>(copy.size())));
    if (result == 2) {
        return CUSCO_CANCELLED;
    }
    if (result != 0) {
        return CUSCO_BACKEND;
    }
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
        return CUSCO_OK;
    }

    const size_t rolled_back = llama_state_set_data(
        executor->ctx, prior.data(), prior.size());
    (void) rolled_back;
    return CUSCO_BACKEND;
} catch (const std::bad_alloc &) {
    return CUSCO_NOMEM;
} catch (...) {
    return CUSCO_BACKEND;
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
    cusco_decode_result ignored{};
    return cusco_executor_decode(executor, tokens, count, &ignored);
} catch (const std::bad_alloc &) {
    return CUSCO_NOMEM;
} catch (...) {
    return CUSCO_BACKEND;
}

void cusco_executor_cancel_next_decode_for_proof(cusco_executor * executor) {
    if (executor) {
        executor->cancel = true;
    }
}
