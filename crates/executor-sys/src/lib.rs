use std::ffi::{c_char, c_int, c_uint, c_ulonglong, c_void};
#[repr(C)]
pub struct CuscoExecutor {
    _private: [u8; 0],
}
#[repr(C)]
pub struct CuscoCheckpoint {
    _private: [u8; 0],
}
#[repr(C)]
#[derive(Clone, Copy)]
pub struct Capabilities {
    pub abi_version: c_uint,
    pub has_global_kv: c_uint,
    pub has_swa: c_uint,
    pub has_recurrent: c_uint,
    pub n_vocab: c_int,
}
#[repr(C)]
pub struct DecodeResult {
    pub logits: *const f32,
    pub logits_len: usize,
    pub token: i32,
}
pub const OK: i32 = 0;
pub const CANCELLED: i32 = 4;
pub const INCOMPATIBLE: i32 = 5;
unsafe extern "C" {
    pub fn cusco_executor_abi_version() -> c_uint;
    pub fn cusco_executor_open(
        path: *const c_char,
        n_ctx: c_uint,
        gpu_layers: c_int,
        out: *mut *mut CuscoExecutor,
    ) -> c_int;
    pub fn cusco_executor_close(e: *mut CuscoExecutor);
    pub fn cusco_executor_capabilities(e: *const CuscoExecutor) -> Capabilities;
    pub fn cusco_executor_tokenize(
        e: *mut CuscoExecutor,
        text: *const c_char,
        tokens: *mut *mut i32,
        count: *mut usize,
    ) -> c_int;
    pub fn cusco_executor_tokens_free(tokens: *mut i32);
    pub fn cusco_executor_decode(
        e: *mut CuscoExecutor,
        tokens: *const i32,
        count: usize,
        out: *mut DecodeResult,
    ) -> c_int;
    pub fn cusco_executor_capture(e: *mut CuscoExecutor, out: *mut *mut CuscoCheckpoint) -> c_int;
    pub fn cusco_checkpoint_free(c: *mut CuscoCheckpoint);
    pub fn cusco_checkpoint_size(c: *const CuscoCheckpoint) -> usize;
    pub fn cusco_checkpoint_checksum(c: *const CuscoCheckpoint) -> c_ulonglong;
    pub fn cusco_executor_prepare_restore(
        e: *mut CuscoExecutor,
        c: *const CuscoCheckpoint,
        sum: c_ulonglong,
        out: *mut *mut CuscoCheckpoint,
    ) -> c_int;
    pub fn cusco_executor_commit_restore(e: *mut CuscoExecutor, c: *mut CuscoCheckpoint) -> c_int;
    pub fn cusco_executor_replace(e: *mut CuscoExecutor, tokens: *const i32, count: usize)
    -> c_int;
    pub fn cusco_executor_cancel_next(e: *mut CuscoExecutor);
}
const _: *const c_void = std::ptr::null();
