use std::ffi::{c_char, c_int, c_uint, c_ulonglong};

#[repr(C)]
pub struct CuscoExecutor {
    _private: [u8; 0],
}
#[repr(C)]
pub struct CuscoCheckpoint {
    _private: [u8; 0],
}
#[repr(C)]
pub struct CuscoPreparedRestore {
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
pub const ROLLBACK_FAILED: i32 = 6;

unsafe extern "C" {
    pub fn cusco_executor_open(
        path: *const c_char,
        n_ctx: c_uint,
        gpu_layers: c_int,
        out: *mut *mut CuscoExecutor,
    ) -> c_int;
    pub fn cusco_executor_close(executor: *mut CuscoExecutor);
    pub fn cusco_executor_capabilities(executor: *const CuscoExecutor) -> Capabilities;
    pub fn cusco_executor_tokenize(
        executor: *mut CuscoExecutor,
        text: *const c_char,
        tokens: *mut *mut i32,
        count: *mut usize,
    ) -> c_int;
    pub fn cusco_executor_tokens_free(tokens: *mut i32);
    pub fn cusco_executor_token_to_piece(
        executor: *mut CuscoExecutor,
        token: i32,
        piece: *mut *mut c_char,
        size: *mut usize,
    ) -> c_int;
    pub fn cusco_executor_piece_free(piece: *mut c_char);
    pub fn cusco_executor_decode(
        executor: *mut CuscoExecutor,
        tokens: *const i32,
        count: usize,
        out: *mut DecodeResult,
    ) -> c_int;
    pub fn cusco_executor_capture(
        executor: *mut CuscoExecutor,
        out: *mut *mut CuscoCheckpoint,
    ) -> c_int;
    pub fn cusco_checkpoint_free(checkpoint: *mut CuscoCheckpoint);
    pub fn cusco_checkpoint_size(checkpoint: *const CuscoCheckpoint) -> usize;
    pub fn cusco_checkpoint_checksum(checkpoint: *const CuscoCheckpoint) -> c_ulonglong;
    pub fn cusco_executor_prepare_restore(
        executor: *mut CuscoExecutor,
        checkpoint: *const CuscoCheckpoint,
        checksum: c_ulonglong,
        out: *mut *mut CuscoPreparedRestore,
    ) -> c_int;
    pub fn cusco_prepared_restore_free(prepared: *mut CuscoPreparedRestore);
    pub fn cusco_executor_commit_restore(
        executor: *mut CuscoExecutor,
        prepared: *mut CuscoPreparedRestore,
    ) -> c_int;
    pub fn cusco_executor_replace_state_for_proof(
        executor: *mut CuscoExecutor,
        tokens: *const i32,
        count: usize,
    ) -> c_int;
    pub fn cusco_executor_cancel_next_decode_for_proof(executor: *mut CuscoExecutor);
}
