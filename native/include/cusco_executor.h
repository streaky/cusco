#ifndef CUSCO_EXECUTOR_H
#define CUSCO_EXECUTOR_H
#include <stddef.h>
#include <stdint.h>
#ifdef __cplusplus
extern "C" {
#endif

#define CUSCO_EXECUTOR_ABI_VERSION 15u

typedef struct cusco_executor cusco_executor;
typedef struct cusco_representation cusco_representation;

/* Opaque, uniquely owned handles. Only the cancellation signal is thread-safe. */
typedef struct cusco_checkpoint cusco_checkpoint;
typedef struct cusco_prepared_restore cusco_prepared_restore;
typedef struct cusco_prepared_mapping cusco_prepared_mapping;
typedef struct cusco_sampler cusco_sampler;

typedef struct {
    uint32_t abi_version;
    uint32_t has_global_kv;
    uint32_t has_swa;
    uint32_t has_recurrent;
    int32_t n_vocab;
    uint32_t has_mapped_execution;
    uint32_t max_mappings;
    uint32_t training_context_tokens;
} cusco_capabilities;

/* Executor-reported operating point selected at open. Device bytes are the
 * observed accelerator allocation delta across model and context creation;
 * host bytes remain a conservative capacity envelope. */
typedef struct {
    uint64_t model_bytes;
    uint64_t context_bytes;
    uint64_t device_bytes;
    uint64_t host_bytes;
    int32_t gpu_layers;
    int32_t model_layers;
    uint32_t competent;
} cusco_operating_point;
typedef struct {
    float temperature;
    float top_p;
    uint32_t seed;
    /* Optional UTF-8 GBNF grammar. NULL selects unconstrained sampling. */
    const char * grammar;
} cusco_sampler_config;


/* logits is borrowed from the executor and remains valid only until the next
 * mutating executor call or cusco_executor_close. The caller must not free it. */
typedef struct {
    const float * logits;
    size_t logits_len;
    int32_t token;
} cusco_decode_result;

typedef enum {
    CUSCO_OK = 0,
    CUSCO_INVALID = 1,
    CUSCO_NOMEM = 2,
    CUSCO_BACKEND = 3,
    CUSCO_CANCELLED = 4,
    CUSCO_INCOMPATIBLE = 5,
    CUSCO_ROLLBACK_FAILED = 6,
    CUSCO_BUFFER_TOO_SMALL = 7
} cusco_status;

/* On success, writes a uniquely owned executor to out. The caller must close it. */
cusco_status cusco_executor_open(const char *, uint32_t, int32_t, cusco_executor ** out);
void cusco_executor_close(cusco_executor *);
cusco_capabilities cusco_executor_capabilities(const cusco_executor *);
cusco_operating_point cusco_executor_operating_point(const cusco_executor *);
/* Returns the GGUF general.architecture value reported by llama.cpp. On
 * CUSCO_BUFFER_TOO_SMALL, size receives the required capacity. */
cusco_status cusco_executor_model_architecture(
    const cusco_executor *, char * buffer, size_t capacity, size_t * size);

/* On success, tokens receives a uniquely owned allocation (or NULL when count is
 * zero). Release it exactly once with cusco_executor_tokens_free. */
cusco_status cusco_executor_tokenize(cusco_executor *, const char *, int32_t ** tokens, size_t * count);
void cusco_executor_tokens_free(int32_t * tokens);
/* Renders one token into caller-owned reusable storage. On
 * CUSCO_BUFFER_TOO_SMALL, size receives the required capacity. */
cusco_status cusco_executor_render_token(
    cusco_executor *, int32_t token, uint8_t * buffer, size_t capacity, size_t * size);
/* Returns nonzero when token is model-vocabulary end-of-generation. */
uint32_t cusco_executor_token_is_eog(const cusco_executor *, int32_t token);

/* A sampler is request-owned and tied to the executor that created it. A
 * non-positive temperature selects exact greedy sampling. */
cusco_status cusco_sampler_create(
    cusco_executor *, const cusco_sampler_config *, cusco_sampler ** out);
void cusco_sampler_free(cusco_sampler *);
cusco_status cusco_sampler_sample(
    cusco_sampler *, const float * logits, size_t logits_len, int32_t * token);
/* Mutates executor state. Input tokens are borrowed for the duration of the call. */
cusco_status cusco_executor_decode(cusco_executor *, const int32_t *, size_t, cusco_decode_result *);

/* Captures an immutable, independently owned snapshot. The caller must free it. */
cusco_status cusco_executor_capture(cusco_executor *, cusco_checkpoint ** out);
void cusco_checkpoint_free(cusco_checkpoint *);
size_t cusco_checkpoint_size(const cusco_checkpoint *);
uint64_t cusco_checkpoint_checksum(const cusco_checkpoint *);

/* Preparation validates and copies checkpoint without changing executor state.
 * On success, out owns a restore tied to the executor's model identity. */
cusco_status cusco_executor_prepare_restore(cusco_executor *, const cusco_checkpoint *, uint64_t, cusco_prepared_restore ** out);
void cusco_prepared_restore_free(cusco_prepared_restore *);

/* Always consumes prepared, on success or failure. Publication is transactional:
 * success installs the prepared state; ordinary failure leaves the prior binding
 * valid. CUSCO_ROLLBACK_FAILED reports that restoring the prior binding also failed. */
cusco_status cusco_executor_commit_restore(cusco_executor *, cusco_prepared_restore * prepared);

/* Published physical state is exposed only through opaque, reference-counted
 * handles. Handles are executor-owned and keep a stable diagnostic identity;
 * release is the only destruction operation. */
typedef struct {
    uint64_t identity;
    uint32_t component_mask;
    uint32_t tier;
    size_t represented_position;
    size_t serialized_bytes;
    uint64_t completion_fence;
} cusco_representation_descriptor;
cusco_status cusco_executor_active_representation(
    cusco_executor *, cusco_representation ** out);
void cusco_representation_retain(cusco_representation *);
void cusco_representation_release(cusco_representation *);
uint64_t cusco_representation_identity(const cusco_representation *);
cusco_status cusco_representation_describe(
    const cusco_representation *, cusco_representation_descriptor * out);
cusco_status cusco_executor_prepare_mapping_fork(
    cusco_executor *, const cusco_representation *, cusco_prepared_mapping ** out);
void cusco_prepared_mapping_free(cusco_prepared_mapping *);
cusco_status cusco_executor_commit_mapping(
    cusco_executor *, cusco_prepared_mapping *, cusco_representation ** out);
cusco_status cusco_executor_activate_mapping(
    cusco_executor *, const cusco_representation *);
size_t cusco_executor_mapping_state_size(
    cusco_executor *, const cusco_representation *);
cusco_status cusco_executor_export_mapping(
    cusco_executor *, const cusco_representation *, uint8_t * buffer,
    size_t capacity, size_t * written, size_t * position);
cusco_status cusco_executor_import_mapping(
    cusco_executor *, const uint8_t * buffer, size_t size, size_t position,
    cusco_representation ** out);
uint64_t cusco_executor_active_mapping_identity(const cusco_executor *);
size_t cusco_executor_mapping_count(const cusco_executor *);
uint64_t cusco_executor_reference_switches(const cusco_executor *);
uint64_t cusco_executor_mapping_fork_bytes_copied(const cusco_executor *);
uint64_t cusco_executor_mapping_export_bytes_copied(const cusco_executor *);
uint64_t cusco_executor_mapping_import_bytes_copied(const cusco_executor *);
uint64_t cusco_executor_mapping_bytes_copied(const cusco_executor *);
/* The pinned llama.cpp public API exposes no graph-recapture signal. These
 * calls make that absence explicit instead of reporting a fabricated zero. */
uint32_t cusco_executor_graph_recaptures_supported(const cusco_executor *);
uint64_t cusco_executor_graph_recaptures(const cusco_executor *);
/* Thread-safe request abort signal. The next active or subsequent decode
 * observes cancellation at llama.cpp's abort callback without mutating the
 * last completed sequence state. */
void cusco_executor_cancel(cusco_executor *);
/* Clear a stale abort signal while the caller exclusively owns the executor. */
void cusco_executor_reset_cancel(cusco_executor *);
/* Deterministic proof hooks, not production executor operations. */
cusco_status cusco_executor_replace_state_for_proof(cusco_executor *, const int32_t *, size_t);
void cusco_executor_cancel_next_decode_for_proof(cusco_executor *);

#ifdef __cplusplus
}
#endif
#endif
