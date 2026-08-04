#ifndef CUSCO_EXECUTOR_H
#define CUSCO_EXECUTOR_H
#include <stddef.h>
#include <stdint.h>
#ifdef __cplusplus
extern "C" {
#endif
#define CUSCO_EXECUTOR_ABI_VERSION 1u
typedef struct cusco_executor cusco_executor;
typedef struct cusco_checkpoint cusco_checkpoint;
typedef struct { uint32_t abi_version; uint32_t has_global_kv; uint32_t has_swa; uint32_t has_recurrent; int32_t n_vocab; } cusco_capabilities;
typedef struct { const float * logits; size_t logits_len; int32_t token; } cusco_decode_result;
typedef enum { CUSCO_OK=0, CUSCO_INVALID=1, CUSCO_NOMEM=2, CUSCO_BACKEND=3, CUSCO_CANCELLED=4, CUSCO_INCOMPATIBLE=5 } cusco_status;
uint32_t cusco_executor_abi_version(void);
cusco_status cusco_executor_open(const char *, uint32_t, int32_t, cusco_executor **);
void cusco_executor_close(cusco_executor *);
cusco_capabilities cusco_executor_capabilities(const cusco_executor *);
cusco_status cusco_executor_tokenize(cusco_executor *, const char *, int32_t **, size_t *);
void cusco_executor_tokens_free(int32_t *);
cusco_status cusco_executor_decode(cusco_executor *, const int32_t *, size_t, cusco_decode_result *);
cusco_status cusco_executor_capture(cusco_executor *, cusco_checkpoint **);
void cusco_checkpoint_free(cusco_checkpoint *);
size_t cusco_checkpoint_size(const cusco_checkpoint *);
uint64_t cusco_checkpoint_checksum(const cusco_checkpoint *);
cusco_status cusco_executor_prepare_restore(cusco_executor *, const cusco_checkpoint *, uint64_t, cusco_checkpoint **);
cusco_status cusco_executor_commit_restore(cusco_executor *, cusco_checkpoint *);
cusco_status cusco_executor_replace(cusco_executor *, const int32_t *, size_t);
void cusco_executor_cancel_next(cusco_executor *);
#ifdef __cplusplus
}
#endif
#endif
