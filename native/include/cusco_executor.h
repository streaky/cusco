#ifndef CUSCO_EXECUTOR_H
#define CUSCO_EXECUTOR_H
#include <stddef.h>
#include <stdint.h>
#ifdef __cplusplus
extern "C" {
#endif

#define CUSCO_EXECUTOR_ABI_VERSION 2u

/* Opaque, uniquely owned handles. None is thread-safe. */
typedef struct cusco_executor cusco_executor;
typedef struct cusco_checkpoint cusco_checkpoint;
typedef struct cusco_prepared_restore cusco_prepared_restore;

typedef struct {
    uint32_t abi_version;
    uint32_t has_global_kv;
    uint32_t has_swa;
    uint32_t has_recurrent;
    int32_t n_vocab;
} cusco_capabilities;

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
    CUSCO_ROLLBACK_FAILED = 6
} cusco_status;

/* On success, writes a uniquely owned executor to out. The caller must close it. */
cusco_status cusco_executor_open(const char *, uint32_t, int32_t, cusco_executor ** out);
void cusco_executor_close(cusco_executor *);
cusco_capabilities cusco_executor_capabilities(const cusco_executor *);

/* On success, tokens receives a uniquely owned allocation (or NULL when count is
 * zero). Release it exactly once with cusco_executor_tokens_free. */
cusco_status cusco_executor_tokenize(cusco_executor *, const char *, int32_t ** tokens, size_t * count);
void cusco_executor_tokens_free(int32_t * tokens);
/* Converts one token to an owned UTF-8 byte sequence. Release it exactly once
 * with cusco_executor_piece_free. */
cusco_status cusco_executor_token_to_piece(
    cusco_executor *, int32_t token, char ** piece, size_t * size);
void cusco_executor_piece_free(char * piece);

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

/* Phase 1 proof hooks, not production executor operations. */
cusco_status cusco_executor_replace_state_for_proof(cusco_executor *, const int32_t *, size_t);
void cusco_executor_cancel_next_decode_for_proof(cusco_executor *);

#ifdef __cplusplus
}
#endif
#endif
