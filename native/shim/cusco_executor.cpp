#include "cusco_executor.h"
#include "llama.h"
#include <atomic>
#include <cstring>
#include <new>
#include <vector>

struct cusco_checkpoint { std::vector<uint8_t> bytes; uint64_t model_identity; };
struct cusco_executor {
    llama_model * model;
    llama_context * ctx;
    const llama_vocab * vocab;
    uint64_t model_identity;
    std::vector<float> logits;
    std::vector<int32_t> mock_state;
    std::atomic_bool cancel;
};
static uint64_t hash_bytes(const uint8_t * p, size_t n) { uint64_t h = 1469598103934665603ULL; for (size_t i=0;i<n;i++) { h ^= p[i]; h *= 1099511628211ULL; } return h; }
static bool is_mock(const cusco_executor * e) { return e->model == nullptr; }
static bool abort_decode(void * p) { return static_cast<cusco_executor *>(p)->cancel.exchange(false); }
uint32_t cusco_executor_abi_version(void) { return CUSCO_EXECUTOR_ABI_VERSION; }
cusco_status cusco_executor_open(const char * path,uint32_t n_ctx,int32_t gpu_layers,cusco_executor ** out) {
    if (!path || !out) return CUSCO_INVALID; *out = nullptr;
    if (strcmp(path,"mock://deterministic") == 0) { *out = new(std::nothrow) cusco_executor{nullptr,nullptr,nullptr,hash_bytes((const uint8_t *)path,strlen(path)),{},{},false}; return *out ? CUSCO_OK : CUSCO_NOMEM; }
    llama_backend_init(); auto mp=llama_model_default_params(); mp.n_gpu_layers=gpu_layers; mp.check_tensors=true;
    auto * m=llama_model_load_from_file(path,mp); if(!m)return CUSCO_BACKEND;
    auto cp=llama_context_default_params(); cp.n_ctx=n_ctx; cp.n_batch=n_ctx; cp.n_seq_max=1; cp.swa_full=true;
    auto * ctx=llama_init_from_model(m,cp); if(!ctx){llama_model_free(m);return CUSCO_BACKEND;}
    auto * e=new(std::nothrow) cusco_executor{m,ctx,llama_model_get_vocab(m),hash_bytes((const uint8_t *)path,strlen(path)),{},{},false};
    if(!e){llama_free(ctx);llama_model_free(m);return CUSCO_NOMEM;} llama_set_abort_callback(ctx,abort_decode,e); *out=e; return CUSCO_OK;
}
void cusco_executor_close(cusco_executor * e){if(!e)return;if(!is_mock(e)){llama_free(e->ctx);llama_model_free(e->model);}delete e;}
cusco_capabilities cusco_executor_capabilities(const cusco_executor * e){if(is_mock(e))return{CUSCO_EXECUTOR_ABI_VERSION,1,1,1,256};return{CUSCO_EXECUTOR_ABI_VERSION,1u,llama_model_n_swa(e->model)>0?1u:0u,1u,llama_vocab_n_tokens(e->vocab)};}
cusco_status cusco_executor_tokenize(cusco_executor * e,const char * text,int32_t ** out,size_t * count){if(!e||!text||!out||!count)return CUSCO_INVALID;if(is_mock(e)){size_t n=strlen(text);auto*p=new(std::nothrow)int32_t[n];if(!p)return CUSCO_NOMEM;for(size_t i=0;i<n;i++)p[i]=(uint8_t)text[i];*out=p;*count=n;return CUSCO_OK;}int n=llama_tokenize(e->vocab,text,(int)strlen(text),nullptr,0,true,true);if(n>=0)return CUSCO_BACKEND;n=-n;auto*p=new(std::nothrow)int32_t[n];if(!p)return CUSCO_NOMEM;int got=llama_tokenize(e->vocab,text,(int)strlen(text),p,n,true,true);if(got<0){delete[]p;return CUSCO_BACKEND;}*out=p;*count=got;return CUSCO_OK;}
void cusco_executor_tokens_free(int32_t * p){delete[]p;}
cusco_status cusco_executor_decode(cusco_executor * e,const int32_t * tokens,size_t count,cusco_decode_result * out){if(!e||!tokens||!count||!out)return CUSCO_INVALID;if(e->cancel.exchange(false))return CUSCO_CANCELLED;if(is_mock(e)){e->mock_state.insert(e->mock_state.end(),tokens,tokens+count);uint64_t h=hash_bytes((const uint8_t*)e->mock_state.data(),e->mock_state.size()*sizeof(int32_t));e->logits.resize(8);for(size_t i=0;i<8;i++)e->logits[i]=(float)((h>>(i*8))&255)/255.0f;*out={e->logits.data(),e->logits.size(),(int32_t)(h%256)};return CUSCO_OK;}std::vector<int32_t> copy(tokens,tokens+count);int rc=llama_decode(e->ctx,llama_batch_get_one(copy.data(),(int)copy.size()));if(rc==2)return CUSCO_CANCELLED;if(rc!=0)return CUSCO_BACKEND;llama_synchronize(e->ctx);int n=llama_vocab_n_tokens(e->vocab);const float*src=llama_get_logits_ith(e->ctx,-1);if(!src)return CUSCO_BACKEND;e->logits.assign(src,src+n);auto*sampler=llama_sampler_init_greedy();int32_t token=llama_sampler_sample(sampler,e->ctx,-1);llama_sampler_free(sampler);*out={e->logits.data(),e->logits.size(),token};return CUSCO_OK;}
cusco_status cusco_executor_capture(cusco_executor * e,cusco_checkpoint ** out){if(!e||!out)return CUSCO_INVALID;auto*c=new(std::nothrow)cusco_checkpoint;if(!c)return CUSCO_NOMEM;c->model_identity=e->model_identity;if(is_mock(e)){c->bytes.resize(e->mock_state.size()*sizeof(int32_t));memcpy(c->bytes.data(),e->mock_state.data(),c->bytes.size());}else{c->bytes.resize(llama_state_get_size(e->ctx));size_t n=llama_state_get_data(e->ctx,c->bytes.data(),c->bytes.size());if(!n){delete c;return CUSCO_BACKEND;}c->bytes.resize(n);}*out=c;return CUSCO_OK;}
void cusco_checkpoint_free(cusco_checkpoint*c){delete c;}size_t cusco_checkpoint_size(const cusco_checkpoint*c){return c?c->bytes.size():0;}uint64_t cusco_checkpoint_checksum(const cusco_checkpoint*c){return c?hash_bytes(c->bytes.data(),c->bytes.size()):0;}
cusco_status cusco_executor_prepare_restore(cusco_executor*e,const cusco_checkpoint*c,uint64_t sum,cusco_checkpoint**out){if(!e||!c||!out)return CUSCO_INVALID;if(c->model_identity!=e->model_identity||cusco_checkpoint_checksum(c)!=sum)return CUSCO_INCOMPATIBLE;auto*p=new(std::nothrow)cusco_checkpoint(*c);if(!p)return CUSCO_NOMEM;*out=p;return CUSCO_OK;}
cusco_status cusco_executor_commit_restore(cusco_executor*e,cusco_checkpoint*p){if(!e||!p)return CUSCO_INVALID;if(p->model_identity!=e->model_identity)return CUSCO_INCOMPATIBLE;if(is_mock(e)){e->mock_state.resize(p->bytes.size()/sizeof(int32_t));memcpy(e->mock_state.data(),p->bytes.data(),p->bytes.size());delete p;return CUSCO_OK;}size_t n=llama_state_set_data(e->ctx,p->bytes.data(),p->bytes.size());delete p;return n?CUSCO_OK:CUSCO_BACKEND;}
cusco_status cusco_executor_replace(cusco_executor*e,const int32_t*tokens,size_t count){if(!e)return CUSCO_INVALID;if(is_mock(e))e->mock_state.clear();else llama_memory_clear(llama_get_memory(e->ctx),true);cusco_decode_result ignored{};return cusco_executor_decode(e,tokens,count,&ignored);}
void cusco_executor_cancel_next(cusco_executor*e){if(e)e->cancel=true;}
