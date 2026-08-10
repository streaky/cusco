use cusco_executor_sys as sys;
use serde::Serialize;
use std::{ffi::CString, ptr::NonNull, sync::Arc};
use thiserror::Error;

pub const ABI_VERSION: &str = sys::ABI_VERSION;

#[derive(Debug, Error, PartialEq)]
pub enum Error {
    #[error("invalid path")]
    InvalidPath,
    #[error("operation cancelled")]
    Cancelled,
    #[error("incompatible checkpoint")]
    Incompatible,
    #[error("restore failed and the prior binding could not be recovered")]
    RollbackFailed,
    #[error("executor backend error {0}")]
    Backend(i32),
}

fn status(code: i32) -> Result<(), Error> {
    match code {
        sys::OK => Ok(()),
        sys::CANCELLED => Err(Error::Cancelled),
        sys::INCOMPATIBLE => Err(Error::Incompatible),
        sys::ROLLBACK_FAILED => Err(Error::RollbackFailed),
        n => Err(Error::Backend(n)),
    }
}

#[derive(Clone, Copy, Debug, Serialize)]
pub struct Capabilities {
    pub abi_version: u32,
    pub global_kv: bool,
    pub swa: bool,
    pub recurrent: bool,
    pub vocabulary: i32,
    pub mapped_execution: bool,
    pub max_mappings: u32,
    pub training_context_tokens: u32,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
pub struct OperatingPoint {
    pub model_bytes: u64,
    pub context_bytes: u64,
    pub device_bytes: u64,
    pub host_bytes: u64,
    pub gpu_layers: i32,
    pub model_layers: i32,
    pub competent: bool,
}

#[derive(Clone, Debug, Serialize)]
pub struct Decode {
    pub logits: Vec<f32>,
    pub token: i32,
}

pub struct RepresentationHandle {
    raw: NonNull<sys::CuscoRepresentation>,
    identity: u64,
    _lifetime: Arc<ExecutorLifetime>,
}

// SAFETY: the native reference count is atomic, and the executor lifetime is
// retained by Arc. Dropping a handle only decrements that count; reclamation
// occurs on a later exclusively borrowed executor operation.
unsafe impl Send for RepresentationHandle {}
impl Clone for RepresentationHandle {
    fn clone(&self) -> Self {
        ffi::retain_representation(self.raw);
        Self {
            raw: self.raw,
            identity: self.identity,
            _lifetime: self._lifetime.clone(),
        }
    }
}

impl std::fmt::Debug for RepresentationHandle {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_tuple("RepresentationHandle")
            .field(&self.identity)
            .finish()
    }
}
impl PartialEq for RepresentationHandle {
    fn eq(&self, other: &Self) -> bool {
        self.identity == other.identity && self.raw == other.raw
    }
}
impl Eq for RepresentationHandle {}
impl std::hash::Hash for RepresentationHandle {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.identity.hash(state);
        self.raw.hash(state);
    }
}
impl Drop for RepresentationHandle {
    fn drop(&mut self) {
        ffi::release_representation(self.raw);
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
pub struct RepresentationDescriptor {
    pub identity: u64,
    pub component_mask: u32,
    pub tier: u32,
    pub represented_position: usize,
    pub serialized_bytes: usize,
    pub completion_fence: u64,
}
impl RepresentationHandle {
    pub fn identity(&self) -> u64 {
        self.identity
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct MappingState {
    pub bytes: Vec<u8>,
    pub position: usize,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
pub struct MappingMetrics {
    pub active_identity: u64,
    pub resident_mappings: usize,
    pub reference_switches: u64,
    pub fork_bytes_copied: u64,
    pub export_bytes_copied: u64,
    pub import_bytes_copied: u64,
    pub total_bytes_copied: u64,
    pub graph_recaptures: Option<u64>,
}

#[derive(Debug)]
/// Uniquely owns one native execution slot. The slot is mutated only through `&mut self`.
struct ExecutorLifetime {
    raw: NonNull<sys::CuscoExecutor>,
}

// SAFETY: the lifetime object never accesses the executor except to close it
// after every owner is gone. Native request cancellation is independently
// atomic; all other executor access remains uniquely borrowed through Executor.
unsafe impl Send for ExecutorLifetime {}
unsafe impl Sync for ExecutorLifetime {}

impl Drop for ExecutorLifetime {
    fn drop(&mut self) {
        ffi::close(self.raw);
    }
}

/// A thread-safe signal handle that cannot outlive its native executor.
#[derive(Clone)]
pub struct CancellationHandle {
    raw: NonNull<sys::CuscoExecutor>,
    _lifetime: Arc<ExecutorLifetime>,
}

// SAFETY: CancellationHandle exposes only the native atomic cancel signal.
unsafe impl Send for CancellationHandle {}
unsafe impl Sync for CancellationHandle {}

impl CancellationHandle {
    pub fn cancel(&self) {
        ffi::cancel(self.raw);
    }
}

pub struct Executor {
    raw: NonNull<sys::CuscoExecutor>,
    lifetime: Arc<ExecutorLifetime>,
}
/// The native slot has unique ownership and all mutation requires `&mut self`.
/// Moving that ownership between threads is safe; concurrent access is not.
unsafe impl Send for Executor {}

/// An immutable, independently owned snapshot of one executor state.
pub struct Checkpoint {
    raw: NonNull<sys::CuscoCheckpoint>,
    pub bytes: usize,
    pub checksum: u64,
}

/// A validated restore candidate. `commit_restore` consumes it exactly once.
#[derive(Debug)]
pub struct PreparedRestore {
    raw: NonNull<sys::CuscoPreparedRestore>,
}

#[derive(Debug)]
pub struct PreparedMapping {
    raw: NonNull<sys::CuscoPreparedMapping>,
    _lifetime: Arc<ExecutorLifetime>,
}

/// A request-owned native sampler. Sampling requires the executor that created it.
pub struct Sampler {
    raw: NonNull<sys::CuscoSampler>,
}

// SAFETY: a sampler is request-owned and all access still requires an exclusive
// borrow of the executor that created it. Moving a suspended request between
// scheduler threads does not permit concurrent native sampler access.
unsafe impl Send for Sampler {}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SamplingConfig {
    pub temperature: f32,
    pub top_p: f32,
    pub seed: u32,
}

impl Default for SamplingConfig {
    fn default() -> Self {
        Self {
            temperature: 0.0,
            top_p: 1.0,
            seed: 0,
        }
    }
}

impl Executor {
    pub fn open(path: &str, n_ctx: u32, gpu_layers: i32) -> Result<Self, Error> {
        let path = CString::new(path).map_err(|_| Error::InvalidPath)?;
        let raw = ffi::open(&path, n_ctx, gpu_layers)?;
        Ok(Self {
            raw,
            lifetime: Arc::new(ExecutorLifetime { raw }),
        })
    }

    pub fn capabilities(&self) -> Capabilities {
        let c = ffi::capabilities(self.raw);
        Capabilities {
            abi_version: c.abi_version,
            global_kv: c.has_global_kv != 0,
            swa: c.has_swa != 0,
            recurrent: c.has_recurrent != 0,
            vocabulary: c.n_vocab,
            mapped_execution: c.has_mapped_execution != 0,
            max_mappings: c.max_mappings,
            training_context_tokens: c.training_context_tokens,
        }
    }
    pub fn model_architecture(&self) -> Result<String, Error> {
        let bytes = ffi::model_architecture(self.raw)?;
        String::from_utf8(bytes).map_err(|_| Error::Backend(sys::INCOMPATIBLE))
    }
    pub fn operating_point(&self) -> OperatingPoint {
        let point = unsafe { sys::cusco_executor_operating_point(self.raw.as_ptr()) };
        OperatingPoint {
            model_bytes: point.model_bytes,
            context_bytes: point.context_bytes,
            device_bytes: point.device_bytes,
            host_bytes: point.host_bytes,
            gpu_layers: point.gpu_layers,
            model_layers: point.model_layers,
            competent: point.competent != 0,
        }
    }
    pub fn cancellation_handle(&self) -> CancellationHandle {
        CancellationHandle {
            raw: self.raw,
            _lifetime: self.lifetime.clone(),
        }
    }
    pub fn reset_cancellation(&mut self) {
        ffi::reset_cancel(self.raw);
    }

    pub fn tokenize(&mut self, text: &str) -> Result<Vec<i32>, Error> {
        let text = CString::new(text).map_err(|_| Error::InvalidPath)?;
        ffi::tokenize(self.raw, &text)
    }
    pub fn render_token<'a>(
        &mut self,
        token: i32,
        buffer: &'a mut Vec<u8>,
    ) -> Result<&'a [u8], Error> {
        ffi::render_token(self.raw, token, buffer)?;
        Ok(buffer)
    }

    pub fn token_is_eog(&self, token: i32) -> bool {
        // SAFETY: raw is live for the duration of the call.
        unsafe { sys::cusco_executor_token_is_eog(self.raw.as_ptr(), token) != 0 }
    }

    pub fn token_to_piece(&mut self, token: i32) -> Result<String, Error> {
        let mut buffer = Vec::with_capacity(32);
        self.render_token(token, &mut buffer)?;
        Ok(String::from_utf8_lossy(&buffer).into_owned())
    }

    pub fn sampler(&mut self, config: SamplingConfig) -> Result<Sampler, Error> {
        self.sampler_with_grammar(config, None)
    }

    pub fn sampler_with_grammar(
        &mut self,
        config: SamplingConfig,
        grammar: Option<&str>,
    ) -> Result<Sampler, Error> {
        Ok(Sampler {
            raw: ffi::sampler(self.raw, config, grammar)?,
        })
    }

    pub fn decode(&mut self, tokens: &[i32]) -> Result<Decode, Error> {
        let out = ffi::decode(self.raw, tokens)?;
        Ok(Decode {
            logits: out.logits,
            token: out.token,
        })
    }

    /// Capture the current binding without changing it.
    pub fn capture_checkpoint(&mut self) -> Result<Checkpoint, Error> {
        let raw = ffi::capture_checkpoint(self.raw)?;
        Ok(Checkpoint {
            bytes: ffi::checkpoint_size(raw),
            checksum: ffi::checkpoint_checksum(raw),
            raw,
        })
    }

    /// Validate and copy a checkpoint without changing the active binding.
    pub fn prepare_restore(
        &mut self,
        checkpoint: &Checkpoint,
        checksum: u64,
    ) -> Result<PreparedRestore, Error> {
        Ok(PreparedRestore {
            raw: ffi::prepare_restore(self.raw, checkpoint.raw, checksum)?,
        })
    }

    /// Transactionally publish a prepared restore. On error the prior binding remains valid.
    pub fn commit_restore(&mut self, prepared: PreparedRestore) -> Result<(), Error> {
        let prepared = std::mem::ManuallyDrop::new(prepared);
        let raw = prepared.raw;
        ffi::commit_restore(self.raw, raw)
    }

    pub fn active_representation(&mut self) -> Result<RepresentationHandle, Error> {
        let raw = ffi::active_representation(self.raw)?;
        Ok(RepresentationHandle {
            identity: ffi::representation_identity(raw),
            raw,
            _lifetime: self.lifetime.clone(),
        })
    }

    /// Prepare an unpublished device-resident representation fork.
    pub fn prepare_mapping_fork(
        &mut self,
        source: &RepresentationHandle,
    ) -> Result<PreparedMapping, Error> {
        Ok(PreparedMapping {
            raw: ffi::prepare_mapping_fork(self.raw, source.raw)?,
            _lifetime: self.lifetime.clone(),
        })
    }

    pub fn commit_mapping(
        &mut self,
        prepared: PreparedMapping,
    ) -> Result<RepresentationHandle, Error> {
        let prepared = std::mem::ManuallyDrop::new(prepared);
        let raw = ffi::commit_mapping(self.raw, prepared.raw)?;
        Ok(RepresentationHandle {
            identity: ffi::representation_identity(raw),
            raw,
            _lifetime: self.lifetime.clone(),
        })
    }

    pub fn activate_mapping(&mut self, representation: &RepresentationHandle) -> Result<(), Error> {
        ffi::activate_mapping(self.raw, representation.raw)
    }

    pub fn describe_representation(
        &self,
        representation: &RepresentationHandle,
    ) -> Result<RepresentationDescriptor, Error> {
        ffi::describe_representation(representation.raw)
    }

    pub fn export_mapping(
        &mut self,
        representation: &RepresentationHandle,
    ) -> Result<MappingState, Error> {
        ffi::export_mapping(self.raw, representation.raw)
    }

    pub fn import_mapping(&mut self, state: &MappingState) -> Result<RepresentationHandle, Error> {
        let raw = ffi::import_mapping(self.raw, state)?;
        Ok(RepresentationHandle {
            identity: ffi::representation_identity(raw),
            raw,
            _lifetime: self.lifetime.clone(),
        })
    }

    pub fn mapping_metrics(&self) -> MappingMetrics {
        ffi::mapping_metrics(self.raw)
    }

    /// Proof hook: clear the slot and decode unrelated state.
    /// This is not a production state-management operation.
    pub fn replace_state_for_proof(&mut self, tokens: &[i32]) -> Result<(), Error> {
        ffi::replace_state_for_proof(self.raw, tokens)
    }

    /// Proof hook: make the next decode return cancellation before mutation.
    pub fn cancel_next_decode_for_proof(&mut self) {
        ffi::cancel_next_decode_for_proof(self.raw)
    }
}

impl Sampler {
    pub fn sample(&mut self, decode: &Decode) -> Result<i32, Error> {
        ffi::sample(self.raw, &decode.logits)
    }
}

impl Drop for Checkpoint {
    fn drop(&mut self) {
        ffi::free_checkpoint(self.raw)
    }
}
impl Drop for PreparedRestore {
    fn drop(&mut self) {
        ffi::free_prepared_restore(self.raw)
    }
}
impl Drop for PreparedMapping {
    fn drop(&mut self) {
        ffi::free_prepared_mapping(self.raw)
    }
}
impl Drop for Sampler {
    fn drop(&mut self) {
        ffi::free_sampler(self.raw)
    }
}

pub fn logits_identical(a: &[f32], b: &[f32]) -> bool {
    a.len() == b.len() && a.iter().zip(b).all(|(x, y)| x.to_bits() == y.to_bits())
}

struct OwnedDecode {
    logits: Vec<f32>,
    token: i32,
}

/// The only module allowed to interpret native pointers and borrowed buffers.
mod ffi {
    use super::{Error, OwnedDecode, status};
    use crate::sys;
    use std::{
        ffi::{CStr, CString},
        ptr::NonNull,
        slice,
    };

    pub(super) fn open(
        path: &CStr,
        n_ctx: u32,
        gpu_layers: i32,
    ) -> Result<NonNull<sys::CuscoExecutor>, Error> {
        let mut raw = std::ptr::null_mut();
        // SAFETY: path is NUL-terminated and out points to writable storage.
        status(unsafe { sys::cusco_executor_open(path.as_ptr(), n_ctx, gpu_layers, &mut raw) })?;
        NonNull::new(raw).ok_or(Error::Backend(3))
    }

    pub(super) fn close(raw: NonNull<sys::CuscoExecutor>) {
        // SAFETY: raw is uniquely owned and this is its only close.
        unsafe { sys::cusco_executor_close(raw.as_ptr()) }
    }

    pub(super) fn capabilities(raw: NonNull<sys::CuscoExecutor>) -> sys::Capabilities {
        // SAFETY: raw is live for the duration of the call.
        unsafe { sys::cusco_executor_capabilities(raw.as_ptr()) }
    }

    pub(super) fn model_architecture(raw: NonNull<sys::CuscoExecutor>) -> Result<Vec<u8>, Error> {
        let mut buffer = vec![0; 32];
        let mut size = 0;
        // SAFETY: raw is live and buffer is initialized writable storage.
        let mut code = unsafe {
            sys::cusco_executor_model_architecture(
                raw.as_ptr(),
                buffer.as_mut_ptr().cast(),
                buffer.len(),
                &mut size,
            )
        };
        if code == sys::BUFFER_TOO_SMALL {
            buffer.resize(size, 0);
            // SAFETY: resizing provides the capacity requested by the ABI.
            code = unsafe {
                sys::cusco_executor_model_architecture(
                    raw.as_ptr(),
                    buffer.as_mut_ptr().cast(),
                    buffer.len(),
                    &mut size,
                )
            };
        }
        status(code)?;
        buffer.truncate(size);
        Ok(buffer)
    }

    pub(super) fn tokenize(
        raw: NonNull<sys::CuscoExecutor>,
        text: &CStr,
    ) -> Result<Vec<i32>, Error> {
        let mut tokens = std::ptr::null_mut();
        let mut count = 0;
        // SAFETY: all pointers are live and outputs point to writable storage.
        status(unsafe {
            sys::cusco_executor_tokenize(raw.as_ptr(), text.as_ptr(), &mut tokens, &mut count)
        })?;
        let owned = if count == 0 {
            Vec::new()
        } else {
            // SAFETY: the ABI returns count initialized i32 values on success.
            unsafe { slice::from_raw_parts(tokens, count) }.to_vec()
        };
        // SAFETY: the ABI permits NULL and this allocation is released exactly once.
        unsafe { sys::cusco_executor_tokens_free(tokens) };
        Ok(owned)
    }

    pub(super) fn render_token(
        raw: NonNull<sys::CuscoExecutor>,
        token: i32,
        buffer: &mut Vec<u8>,
    ) -> Result<(), Error> {
        if buffer.capacity() == 0 {
            buffer.reserve(32);
        }
        buffer.resize(buffer.capacity(), 0);
        let mut size = 0;
        // SAFETY: raw is live and the vector exposes capacity initialized writable bytes.
        let mut code = unsafe {
            sys::cusco_executor_render_token(
                raw.as_ptr(),
                token,
                buffer.as_mut_ptr(),
                buffer.len(),
                &mut size,
            )
        };
        if code == sys::BUFFER_TOO_SMALL {
            buffer.resize(size, 0);
            // SAFETY: resizing provides at least the capacity requested by the ABI.
            code = unsafe {
                sys::cusco_executor_render_token(
                    raw.as_ptr(),
                    token,
                    buffer.as_mut_ptr(),
                    buffer.len(),
                    &mut size,
                )
            };
        }
        status(code)?;
        buffer.truncate(size);
        Ok(())
    }

    pub(super) fn sampler(
        raw: NonNull<sys::CuscoExecutor>,
        config: crate::SamplingConfig,
        grammar: Option<&str>,
    ) -> Result<NonNull<sys::CuscoSampler>, Error> {
        let grammar = grammar
            .map(CString::new)
            .transpose()
            .map_err(|_| Error::Backend(sys::INVALID))?;
        let mut sampler = std::ptr::null_mut();
        let config = sys::SamplerConfig {
            temperature: config.temperature,
            top_p: config.top_p,
            seed: config.seed,
            grammar: grammar
                .as_ref()
                .map_or(std::ptr::null(), |value| value.as_ptr()),
        };
        // SAFETY: raw is live, config and its optional grammar are borrowed for
        // the call, and the output points to writable storage.
        status(unsafe { sys::cusco_sampler_create(raw.as_ptr(), &config, &mut sampler) })?;
        NonNull::new(sampler).ok_or(Error::Backend(3))
    }

    pub(super) fn sample(
        sampler: NonNull<sys::CuscoSampler>,
        logits: &[f32],
    ) -> Result<i32, Error> {
        let mut token = 0;
        // SAFETY: the sampler is live and logits is borrowed for the call.
        status(unsafe {
            sys::cusco_sampler_sample(sampler.as_ptr(), logits.as_ptr(), logits.len(), &mut token)
        })?;
        Ok(token)
    }

    pub(super) fn decode(
        raw: NonNull<sys::CuscoExecutor>,
        tokens: &[i32],
    ) -> Result<OwnedDecode, Error> {
        let mut out = sys::DecodeResult {
            logits: std::ptr::null(),
            logits_len: 0,
            token: 0,
        };
        // SAFETY: the token slice and output storage live through the call.
        status(unsafe {
            sys::cusco_executor_decode(raw.as_ptr(), tokens.as_ptr(), tokens.len(), &mut out)
        })?;
        let logits = if out.logits_len == 0 {
            Vec::new()
        } else {
            // SAFETY: logits is borrowed from raw with out.logits_len elements. Copying it
            // before another executor call makes the safe result independent of that borrow.
            unsafe { slice::from_raw_parts(out.logits, out.logits_len) }.to_vec()
        };
        Ok(OwnedDecode {
            logits,
            token: out.token,
        })
    }

    pub(super) fn capture_checkpoint(
        raw: NonNull<sys::CuscoExecutor>,
    ) -> Result<NonNull<sys::CuscoCheckpoint>, Error> {
        let mut checkpoint = std::ptr::null_mut();
        // SAFETY: raw is live and out points to writable storage.
        status(unsafe { sys::cusco_executor_capture(raw.as_ptr(), &mut checkpoint) })?;
        NonNull::new(checkpoint).ok_or(Error::Backend(3))
    }

    pub(super) fn checkpoint_size(raw: NonNull<sys::CuscoCheckpoint>) -> usize {
        // SAFETY: raw is a live immutable checkpoint.
        unsafe { sys::cusco_checkpoint_size(raw.as_ptr()) }
    }

    pub(super) fn checkpoint_checksum(raw: NonNull<sys::CuscoCheckpoint>) -> u64 {
        // SAFETY: raw is a live immutable checkpoint.
        unsafe { sys::cusco_checkpoint_checksum(raw.as_ptr()) }
    }

    pub(super) fn prepare_restore(
        executor: NonNull<sys::CuscoExecutor>,
        checkpoint: NonNull<sys::CuscoCheckpoint>,
        checksum: u64,
    ) -> Result<NonNull<sys::CuscoPreparedRestore>, Error> {
        let mut prepared = std::ptr::null_mut();
        // SAFETY: both handles are live and out points to writable storage.
        status(unsafe {
            sys::cusco_executor_prepare_restore(
                executor.as_ptr(),
                checkpoint.as_ptr(),
                checksum,
                &mut prepared,
            )
        })?;
        NonNull::new(prepared).ok_or(Error::Backend(3))
    }

    pub(super) fn commit_restore(
        executor: NonNull<sys::CuscoExecutor>,
        prepared: NonNull<sys::CuscoPreparedRestore>,
    ) -> Result<(), Error> {
        // SAFETY: both handles are live. The ABI consumes prepared on every result.
        status(unsafe { sys::cusco_executor_commit_restore(executor.as_ptr(), prepared.as_ptr()) })
    }

    pub(super) fn active_representation(
        executor: NonNull<sys::CuscoExecutor>,
    ) -> Result<NonNull<sys::CuscoRepresentation>, Error> {
        let mut representation = std::ptr::null_mut();
        status(unsafe {
            sys::cusco_executor_active_representation(executor.as_ptr(), &mut representation)
        })?;
        NonNull::new(representation).ok_or(Error::Backend(3))
    }

    pub(super) fn retain_representation(raw: NonNull<sys::CuscoRepresentation>) {
        unsafe { sys::cusco_representation_retain(raw.as_ptr()) }
    }

    pub(super) fn release_representation(raw: NonNull<sys::CuscoRepresentation>) {
        unsafe { sys::cusco_representation_release(raw.as_ptr()) }
    }

    pub(super) fn representation_identity(raw: NonNull<sys::CuscoRepresentation>) -> u64 {
        unsafe { sys::cusco_representation_identity(raw.as_ptr()) }
    }

    pub(super) fn describe_representation(
        raw: NonNull<sys::CuscoRepresentation>,
    ) -> Result<super::RepresentationDescriptor, Error> {
        let mut descriptor = sys::RepresentationDescriptor {
            identity: 0,
            component_mask: 0,
            tier: 0,
            represented_position: 0,
            serialized_bytes: 0,
            completion_fence: 0,
        };
        status(unsafe { sys::cusco_representation_describe(raw.as_ptr(), &mut descriptor) })?;
        Ok(super::RepresentationDescriptor {
            identity: descriptor.identity,
            component_mask: descriptor.component_mask,
            tier: descriptor.tier,
            represented_position: descriptor.represented_position,
            serialized_bytes: descriptor.serialized_bytes,
            completion_fence: descriptor.completion_fence,
        })
    }

    pub(super) fn prepare_mapping_fork(
        executor: NonNull<sys::CuscoExecutor>,
        source: NonNull<sys::CuscoRepresentation>,
    ) -> Result<NonNull<sys::CuscoPreparedMapping>, Error> {
        let mut prepared = std::ptr::null_mut();
        status(unsafe {
            sys::cusco_executor_prepare_mapping_fork(
                executor.as_ptr(),
                source.as_ptr(),
                &mut prepared,
            )
        })?;
        NonNull::new(prepared).ok_or(Error::Backend(3))
    }

    pub(super) fn commit_mapping(
        executor: NonNull<sys::CuscoExecutor>,
        prepared: NonNull<sys::CuscoPreparedMapping>,
    ) -> Result<NonNull<sys::CuscoRepresentation>, Error> {
        let mut representation = std::ptr::null_mut();
        status(unsafe {
            sys::cusco_executor_commit_mapping(
                executor.as_ptr(),
                prepared.as_ptr(),
                &mut representation,
            )
        })?;
        NonNull::new(representation).ok_or(Error::Backend(3))
    }

    pub(super) fn activate_mapping(
        executor: NonNull<sys::CuscoExecutor>,
        representation: NonNull<sys::CuscoRepresentation>,
    ) -> Result<(), Error> {
        status(unsafe {
            sys::cusco_executor_activate_mapping(executor.as_ptr(), representation.as_ptr())
        })
    }

    pub(super) fn export_mapping(
        executor: NonNull<sys::CuscoExecutor>,
        representation: NonNull<sys::CuscoRepresentation>,
    ) -> Result<super::MappingState, Error> {
        let size = unsafe {
            sys::cusco_executor_mapping_state_size(executor.as_ptr(), representation.as_ptr())
        };
        let mut bytes = vec![0; size];
        let mut written = size;
        let mut position = 0;
        status(unsafe {
            sys::cusco_executor_export_mapping(
                executor.as_ptr(),
                representation.as_ptr(),
                bytes.as_mut_ptr(),
                bytes.len(),
                &mut written,
                &mut position,
            )
        })?;
        bytes.truncate(written);
        Ok(super::MappingState { bytes, position })
    }

    pub(super) fn import_mapping(
        executor: NonNull<sys::CuscoExecutor>,
        state: &super::MappingState,
    ) -> Result<NonNull<sys::CuscoRepresentation>, Error> {
        let mut representation = std::ptr::null_mut();
        status(unsafe {
            sys::cusco_executor_import_mapping(
                executor.as_ptr(),
                state.bytes.as_ptr(),
                state.bytes.len(),
                state.position,
                &mut representation,
            )
        })?;
        NonNull::new(representation).ok_or(Error::Backend(3))
    }

    pub(super) fn mapping_metrics(executor: NonNull<sys::CuscoExecutor>) -> super::MappingMetrics {
        // SAFETY: executor is live for all read-only metric calls.
        unsafe {
            let graph_recaptures =
                (sys::cusco_executor_graph_recaptures_supported(executor.as_ptr()) != 0)
                    .then(|| sys::cusco_executor_graph_recaptures(executor.as_ptr()));
            super::MappingMetrics {
                active_identity: sys::cusco_executor_active_mapping_identity(executor.as_ptr()),
                resident_mappings: sys::cusco_executor_mapping_count(executor.as_ptr()),
                reference_switches: sys::cusco_executor_reference_switches(executor.as_ptr()),
                fork_bytes_copied: sys::cusco_executor_mapping_fork_bytes_copied(executor.as_ptr()),
                export_bytes_copied: sys::cusco_executor_mapping_export_bytes_copied(
                    executor.as_ptr(),
                ),
                import_bytes_copied: sys::cusco_executor_mapping_import_bytes_copied(
                    executor.as_ptr(),
                ),
                total_bytes_copied: sys::cusco_executor_mapping_bytes_copied(executor.as_ptr()),
                graph_recaptures,
            }
        }
    }

    pub(super) fn free_prepared_mapping(raw: NonNull<sys::CuscoPreparedMapping>) {
        // SAFETY: raw is uniquely owned and released exactly once.
        unsafe { sys::cusco_prepared_mapping_free(raw.as_ptr()) }
    }

    pub(super) fn replace_state_for_proof(
        raw: NonNull<sys::CuscoExecutor>,
        tokens: &[i32],
    ) -> Result<(), Error> {
        // SAFETY: raw and the token slice are live through the call.
        status(unsafe {
            sys::cusco_executor_replace_state_for_proof(raw.as_ptr(), tokens.as_ptr(), tokens.len())
        })
    }
    pub(super) fn cancel(raw: NonNull<sys::CuscoExecutor>) {
        // SAFETY: the cancellation handle retains executor lifetime and this
        // production operation only sets the native atomic abort flag.
        unsafe { sys::cusco_executor_cancel(raw.as_ptr()) }
    }
    pub(super) fn reset_cancel(raw: NonNull<sys::CuscoExecutor>) {
        // SAFETY: the caller exclusively borrows the executor.
        unsafe { sys::cusco_executor_reset_cancel(raw.as_ptr()) }
    }

    pub(super) fn cancel_next_decode_for_proof(raw: NonNull<sys::CuscoExecutor>) {
        // SAFETY: raw is live and uniquely borrowed by the caller.
        unsafe { sys::cusco_executor_cancel_next_decode_for_proof(raw.as_ptr()) }
    }

    pub(super) fn free_sampler(raw: NonNull<sys::CuscoSampler>) {
        // SAFETY: raw is uniquely owned and released exactly once.
        unsafe { sys::cusco_sampler_free(raw.as_ptr()) }
    }

    pub(super) fn free_checkpoint(raw: NonNull<sys::CuscoCheckpoint>) {
        // SAFETY: raw is uniquely owned and released exactly once.
        unsafe { sys::cusco_checkpoint_free(raw.as_ptr()) }
    }

    pub(super) fn free_prepared_restore(raw: NonNull<sys::CuscoPreparedRestore>) {
        // SAFETY: raw is uniquely owned and released exactly once.
        unsafe { sys::cusco_prepared_restore_free(raw.as_ptr()) }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn exact_logits_compare_bits() {
        assert!(logits_identical(&[0.0, -1.0], &[0.0, -1.0]));
        assert!(!logits_identical(&[0.0], &[-0.0]));
        assert!(!logits_identical(&[0.0], &[0.0, 1.0]));
    }

    #[test]
    fn maps_statuses() {
        assert_eq!(status(0), Ok(()));
        assert_eq!(status(4), Err(Error::Cancelled));
        assert_eq!(status(5), Err(Error::Incompatible));
        assert_eq!(status(6), Err(Error::RollbackFailed));
        assert_eq!(status(3), Err(Error::Backend(3)));
    }

    #[test]
    fn model_free_lifecycle_is_transactional() {
        let mut executor = Executor::open("mock://deterministic", 128, 0).unwrap();
        let capabilities = executor.capabilities();
        assert!(capabilities.global_kv && capabilities.swa && capabilities.recurrent);
        assert_eq!(capabilities.training_context_tokens, u32::MAX);
        assert_eq!(executor.model_architecture().unwrap(), "gemma4");
        assert!(executor.token_is_eog(1));
        assert!(executor.token_is_eog(106));
        assert!(!executor.token_is_eog(42));
        let prefix = executor.tokenize("prefix").unwrap();
        assert_eq!(executor.token_to_piece(42).unwrap(), "42");
        executor.replace_state_for_proof(&prefix).unwrap();
        let checkpoint = executor.capture_checkpoint().unwrap();
        assert!(checkpoint.bytes > 0);
        let continuation = [7];
        let expected = executor.decode(&continuation).unwrap();
        let unrelated = executor.tokenize("other").unwrap();
        executor.replace_state_for_proof(&unrelated).unwrap();
        executor.replace_state_for_proof(&[]).unwrap();
        assert_eq!(executor.capture_checkpoint().unwrap().bytes, 0);
        assert_eq!(
            executor
                .prepare_restore(&checkpoint, checkpoint.checksum ^ 1)
                .unwrap_err(),
            Error::Incompatible
        );
        let prepared = executor
            .prepare_restore(&checkpoint, checkpoint.checksum)
            .unwrap();
        executor.commit_restore(prepared).unwrap();
        let actual = executor.decode(&continuation).unwrap();
        assert_eq!(actual.token, expected.token);
        assert!(logits_identical(&actual.logits, &expected.logits));
        executor.cancel_next_decode_for_proof();
        assert_eq!(
            executor.decode(&continuation).unwrap_err(),
            Error::Cancelled
        );
        assert!(matches!(
            Executor::open("bad\0path", 1, 0),
            Err(Error::InvalidPath)
        ));
    }

    #[test]
    fn production_cancellation_handle_is_thread_safe_and_retains_executor() {
        let mut executor = Executor::open("mock://deterministic", 64, 0).unwrap();
        let cancellation = executor.cancellation_handle();
        let worker = std::thread::spawn({
            let cancellation = cancellation.clone();
            move || cancellation.cancel()
        });
        worker.join().unwrap();
        assert_eq!(executor.decode(&[1]).unwrap_err(), Error::Cancelled);
        executor.reset_cancellation();
        assert!(executor.decode(&[1]).is_ok());
        drop(executor);
        cancellation.cancel();
    }

    #[test]
    fn native_sampler_and_renderer_are_request_owned() {
        let mut executor = Executor::open("mock://deterministic", 128, 0).unwrap();
        let decoded = executor.decode(&[11]).unwrap();
        let mut sampler = executor.sampler(SamplingConfig::default()).unwrap();
        assert_eq!(sampler.sample(&decoded).unwrap(), decoded.token);

        let mut piece = Vec::with_capacity(32);
        executor.render_token(decoded.token, &mut piece).unwrap();
        let allocation = piece.as_ptr();
        assert_eq!(piece, decoded.token.to_string().as_bytes());
        executor.render_token(-1234, &mut piece).unwrap();
        assert_eq!(piece, b"-1234");
        assert_eq!(piece.as_ptr(), allocation);

        let mut other = Executor::open("mock://deterministic", 128, 0).unwrap();
        let other_decoded = other.decode(&[12]).unwrap();
        assert_eq!(sampler.sample(&other_decoded).unwrap(), other_decoded.token);
    }
    #[test]
    fn mapped_forks_publish_transactionally_and_switch_by_reference() {
        let mut executor = Executor::open("mock://deterministic", 128, 0).unwrap();
        assert!(executor.capabilities().mapped_execution);
        let prefix = executor.tokenize("shared prefix").unwrap();
        executor.decode(&prefix).unwrap();

        let root = executor.active_representation().unwrap();
        let aborted = executor.prepare_mapping_fork(&root).unwrap();
        drop(aborted);
        assert_eq!(executor.mapping_metrics().resident_mappings, 1);

        let prepared = executor.prepare_mapping_fork(&root).unwrap();
        let branch = executor.commit_mapping(prepared).unwrap();
        assert_eq!(executor.mapping_metrics().resident_mappings, 2);
        let continuation = [7, 8];
        let staged = executor.decode(&continuation).unwrap();
        let spilled = executor.export_mapping(&branch).unwrap();
        drop(branch);
        assert_eq!(executor.mapping_metrics().resident_mappings, 1);
        let restored = executor.import_mapping(&spilled).unwrap();
        executor.activate_mapping(&restored).unwrap();
        let mapped = executor.decode(&continuation).unwrap();
        assert_eq!(mapped.token, staged.token);
        assert!(logits_identical(&mapped.logits, &staged.logits));

        let metrics = executor.mapping_metrics();
        assert_eq!(metrics.reference_switches, 1);
        assert_eq!(
            metrics.fork_bytes_copied,
            (2 * prefix.len() * std::mem::size_of::<i32>()) as u64
        );
        assert_eq!(metrics.export_bytes_copied, spilled.bytes.len() as u64);
        assert_eq!(metrics.import_bytes_copied, spilled.bytes.len() as u64);
        assert_eq!(
            metrics.total_bytes_copied,
            metrics.fork_bytes_copied + metrics.export_bytes_copied + metrics.import_bytes_copied
        );
        assert_eq!(metrics.graph_recaptures, None);
        executor.activate_mapping(&root).unwrap();
        drop(restored);
        assert_eq!(executor.mapping_metrics().resident_mappings, 1);
    }
}
