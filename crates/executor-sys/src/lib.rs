use std::ffi::{c_char, c_int, c_uint, c_ulonglong};

pub const ABI_VERSION: &str = env!("CUSCO_EXECUTOR_ABI_VERSION");

#[repr(C)]
pub struct CuscoExecutor {
    _private: [u8; 0],
}
#[repr(C)]
pub struct CuscoRepresentation {
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
pub struct CuscoPreparedMapping {
    _private: [u8; 0],
}
#[repr(C)]
pub struct CuscoSampler {
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
    pub has_mapped_execution: c_uint,
    pub max_mappings: c_uint,
    pub training_context_tokens: c_uint,
}
#[repr(C)]
#[derive(Clone, Copy)]
pub struct RepresentationDescriptor {
    pub identity: c_ulonglong,
    pub component_mask: c_uint,
    pub tier: c_uint,
    pub represented_position: usize,
    pub serialized_bytes: usize,
    pub completion_fence: c_ulonglong,
}
#[repr(C)]
#[derive(Clone, Copy)]
pub struct OperatingPoint {
    pub model_bytes: u64,
    pub context_bytes: u64,
    pub device_bytes: u64,
    pub host_bytes: u64,
    pub gpu_layers: c_int,
    pub model_layers: c_int,
    pub competent: c_uint,
}
#[repr(C)]
#[derive(Clone, Copy)]
pub struct SamplerConfig {
    pub temperature: f32,
    pub top_p: f32,
    pub seed: c_uint,
    pub grammar: *const c_char,
}

#[repr(C)]
pub struct DecodeResult {
    pub logits: *const f32,
    pub logits_len: usize,
    pub token: i32,
}
pub const OK: i32 = 0;
pub const INVALID: i32 = 1;
pub const CANCELLED: i32 = 4;
pub const INCOMPATIBLE: i32 = 5;
pub const ROLLBACK_FAILED: i32 = 6;
pub const BUFFER_TOO_SMALL: i32 = 7;

unsafe extern "C" {
    pub fn cusco_executor_open(
        path: *const c_char,
        n_ctx: c_uint,
        gpu_layers: c_int,
        out: *mut *mut CuscoExecutor,
    ) -> c_int;
    pub fn cusco_executor_close(executor: *mut CuscoExecutor);
    pub fn cusco_executor_capabilities(executor: *const CuscoExecutor) -> Capabilities;
    pub fn cusco_executor_operating_point(executor: *const CuscoExecutor) -> OperatingPoint;
    pub fn cusco_executor_free_accelerator_bytes() -> u64;
    pub fn cusco_executor_model_architecture(
        executor: *const CuscoExecutor,
        buffer: *mut c_char,
        capacity: usize,
        size: *mut usize,
    ) -> c_int;
    pub fn cusco_executor_tokenize(
        executor: *mut CuscoExecutor,
        text: *const c_char,
        tokens: *mut *mut i32,
        count: *mut usize,
    ) -> c_int;
    pub fn cusco_executor_tokens_free(tokens: *mut i32);
    pub fn cusco_executor_render_token(
        executor: *mut CuscoExecutor,
        token: i32,
        buffer: *mut u8,
        capacity: usize,
        size: *mut usize,
    ) -> c_int;
    pub fn cusco_executor_token_is_eog(executor: *const CuscoExecutor, token: i32) -> c_uint;
    pub fn cusco_sampler_create(
        executor: *mut CuscoExecutor,
        config: *const SamplerConfig,
        out: *mut *mut CuscoSampler,
    ) -> c_int;
    pub fn cusco_sampler_free(sampler: *mut CuscoSampler);
    pub fn cusco_sampler_sample(
        sampler: *mut CuscoSampler,
        logits: *const f32,
        logits_len: usize,
        token: *mut i32,
    ) -> c_int;
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
    pub fn cusco_executor_active_representation(
        executor: *mut CuscoExecutor,
        out: *mut *mut CuscoRepresentation,
    ) -> c_int;
    pub fn cusco_representation_retain(representation: *mut CuscoRepresentation);
    pub fn cusco_representation_release(representation: *mut CuscoRepresentation);
    pub fn cusco_representation_identity(representation: *const CuscoRepresentation)
    -> c_ulonglong;
    pub fn cusco_representation_describe(
        representation: *const CuscoRepresentation,
        out: *mut RepresentationDescriptor,
    ) -> c_int;
    pub fn cusco_executor_prepare_mapping_fork(
        executor: *mut CuscoExecutor,
        source: *const CuscoRepresentation,
        out: *mut *mut CuscoPreparedMapping,
    ) -> c_int;
    pub fn cusco_prepared_mapping_free(prepared: *mut CuscoPreparedMapping);
    pub fn cusco_executor_commit_mapping(
        executor: *mut CuscoExecutor,
        prepared: *mut CuscoPreparedMapping,
        out: *mut *mut CuscoRepresentation,
    ) -> c_int;
    pub fn cusco_executor_activate_mapping(
        executor: *mut CuscoExecutor,
        representation: *const CuscoRepresentation,
    ) -> c_int;
    pub fn cusco_executor_mapping_state_size(
        executor: *mut CuscoExecutor,
        representation: *const CuscoRepresentation,
    ) -> usize;
    pub fn cusco_executor_export_mapping(
        executor: *mut CuscoExecutor,
        representation: *const CuscoRepresentation,
        buffer: *mut u8,
        capacity: usize,
        written: *mut usize,
        position: *mut usize,
    ) -> c_int;
    pub fn cusco_executor_import_mapping(
        executor: *mut CuscoExecutor,
        buffer: *const u8,
        size: usize,
        position: usize,
        out: *mut *mut CuscoRepresentation,
    ) -> c_int;
    pub fn cusco_executor_active_mapping_identity(executor: *const CuscoExecutor) -> c_ulonglong;
    pub fn cusco_executor_mapping_count(executor: *const CuscoExecutor) -> usize;
    pub fn cusco_executor_reference_switches(executor: *const CuscoExecutor) -> c_ulonglong;
    pub fn cusco_executor_mapping_fork_bytes_copied(executor: *const CuscoExecutor) -> c_ulonglong;
    pub fn cusco_executor_mapping_export_bytes_copied(
        executor: *const CuscoExecutor,
    ) -> c_ulonglong;
    pub fn cusco_executor_mapping_import_bytes_copied(
        executor: *const CuscoExecutor,
    ) -> c_ulonglong;
    pub fn cusco_executor_mapping_bytes_copied(executor: *const CuscoExecutor) -> c_ulonglong;
    pub fn cusco_executor_graph_recaptures_supported(executor: *const CuscoExecutor) -> c_uint;
    pub fn cusco_executor_graph_recaptures(executor: *const CuscoExecutor) -> c_ulonglong;
    pub fn cusco_executor_cancel(executor: *mut CuscoExecutor);
    pub fn cusco_executor_reset_cancel(executor: *mut CuscoExecutor);

    pub fn cusco_executor_replace_state_for_proof(
        executor: *mut CuscoExecutor,
        tokens: *const i32,
        count: usize,
    ) -> c_int;
    pub fn cusco_executor_cancel_next_decode_for_proof(executor: *mut CuscoExecutor);
}
