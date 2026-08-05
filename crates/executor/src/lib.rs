use cusco_executor_sys as sys;
use serde::Serialize;
use std::{ffi::CString, ptr::NonNull, sync::Arc};
use thiserror::Error;

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
}

#[derive(Clone, Debug, Serialize)]
pub struct Decode {
    pub logits: Vec<f32>,
    pub token: i32,
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq, Serialize)]
pub struct MappingId(pub u32);

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
pub struct MappingMetrics {
    pub active: MappingId,
    pub resident_mappings: usize,
    pub reference_switches: u64,
    pub activation_bytes_copied: u64,
}

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
}

/// A request-owned native sampler. Sampling requires the executor that created it.
pub struct GreedySampler {
    raw: NonNull<sys::CuscoSampler>,
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

    pub fn token_to_piece(&mut self, token: i32) -> Result<String, Error> {
        let mut buffer = Vec::with_capacity(32);
        self.render_token(token, &mut buffer)?;
        Ok(String::from_utf8_lossy(&buffer).into_owned())
    }

    pub fn greedy_sampler(&mut self) -> Result<GreedySampler, Error> {
        Ok(GreedySampler {
            raw: ffi::greedy_sampler(self.raw)?,
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

    /// Prepare an unpublished device-resident sequence mapping fork.
    pub fn prepare_mapping_fork(&mut self, source: MappingId) -> Result<PreparedMapping, Error> {
        Ok(PreparedMapping {
            raw: ffi::prepare_mapping_fork(self.raw, source.0)?,
        })
    }

    /// Publish a prepared mapping. The mapping cannot be activated before this call.
    pub fn commit_mapping(&mut self, prepared: PreparedMapping) -> Result<MappingId, Error> {
        let prepared = std::mem::ManuallyDrop::new(prepared);
        ffi::commit_mapping(self.raw, prepared.raw).map(MappingId)
    }

    /// Select a resident mapping without restoring checkpoint bytes.
    pub fn activate_mapping(&mut self, mapping: MappingId) -> Result<(), Error> {
        ffi::activate_mapping(self.raw, mapping.0)
    }

    pub fn remove_mapping(&mut self, mapping: MappingId) -> Result<(), Error> {
        ffi::remove_mapping(self.raw, mapping.0)
    }

    pub fn mapping_metrics(&self) -> MappingMetrics {
        ffi::mapping_metrics(self.raw)
    }

    /// Phase 1 proof hook: clear the slot and decode unrelated state.
    /// This is not a production state-management operation.
    pub fn replace_state_for_proof(&mut self, tokens: &[i32]) -> Result<(), Error> {
        ffi::replace_state_for_proof(self.raw, tokens)
    }

    /// Phase 1 proof hook: make the next decode return cancellation before mutation.
    pub fn cancel_next_decode_for_proof(&mut self) {
        ffi::cancel_next_decode_for_proof(self.raw)
    }
}

impl GreedySampler {
    pub fn sample(&mut self, executor: &mut Executor) -> Result<i32, Error> {
        ffi::sample(self.raw, executor.raw)
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
impl Drop for GreedySampler {
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
    use std::{ffi::CStr, ptr::NonNull, slice};

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

    pub(super) fn greedy_sampler(
        raw: NonNull<sys::CuscoExecutor>,
    ) -> Result<NonNull<sys::CuscoSampler>, Error> {
        let mut sampler = std::ptr::null_mut();
        // SAFETY: raw is live and the output points to writable storage.
        status(unsafe { sys::cusco_sampler_greedy(raw.as_ptr(), &mut sampler) })?;
        NonNull::new(sampler).ok_or(Error::Backend(3))
    }

    pub(super) fn sample(
        sampler: NonNull<sys::CuscoSampler>,
        executor: NonNull<sys::CuscoExecutor>,
    ) -> Result<i32, Error> {
        let mut token = 0;
        // SAFETY: both uniquely owned handles are live for the call.
        status(unsafe {
            sys::cusco_sampler_sample(sampler.as_ptr(), executor.as_ptr(), &mut token)
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

    pub(super) fn prepare_mapping_fork(
        executor: NonNull<sys::CuscoExecutor>,
        source: u32,
    ) -> Result<NonNull<sys::CuscoPreparedMapping>, Error> {
        let mut prepared = std::ptr::null_mut();
        // SAFETY: executor is live and out points to writable storage.
        status(unsafe {
            sys::cusco_executor_prepare_mapping_fork(executor.as_ptr(), source, &mut prepared)
        })?;
        NonNull::new(prepared).ok_or(Error::Backend(3))
    }

    pub(super) fn commit_mapping(
        executor: NonNull<sys::CuscoExecutor>,
        prepared: NonNull<sys::CuscoPreparedMapping>,
    ) -> Result<u32, Error> {
        let mut mapping = 0;
        // SAFETY: both handles are live; the ABI consumes prepared on every result.
        status(unsafe {
            sys::cusco_executor_commit_mapping(executor.as_ptr(), prepared.as_ptr(), &mut mapping)
        })?;
        Ok(mapping)
    }

    pub(super) fn activate_mapping(
        executor: NonNull<sys::CuscoExecutor>,
        mapping: u32,
    ) -> Result<(), Error> {
        // SAFETY: executor is live and uniquely borrowed by the caller.
        status(unsafe { sys::cusco_executor_activate_mapping(executor.as_ptr(), mapping) })
    }

    pub(super) fn remove_mapping(
        executor: NonNull<sys::CuscoExecutor>,
        mapping: u32,
    ) -> Result<(), Error> {
        // SAFETY: executor is live and uniquely borrowed by the caller.
        status(unsafe { sys::cusco_executor_remove_mapping(executor.as_ptr(), mapping) })
    }

    pub(super) fn mapping_metrics(executor: NonNull<sys::CuscoExecutor>) -> super::MappingMetrics {
        // SAFETY: executor is live for all read-only metric calls.
        unsafe {
            super::MappingMetrics {
                active: super::MappingId(sys::cusco_executor_active_mapping(executor.as_ptr())),
                resident_mappings: sys::cusco_executor_mapping_count(executor.as_ptr()),
                reference_switches: sys::cusco_executor_reference_switches(executor.as_ptr()),
                activation_bytes_copied: sys::cusco_executor_mapped_bytes_copied(executor.as_ptr()),
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
        let mut sampler = executor.greedy_sampler().unwrap();
        assert_eq!(sampler.sample(&mut executor).unwrap(), decoded.token);

        let mut piece = Vec::with_capacity(32);
        executor.render_token(decoded.token, &mut piece).unwrap();
        let allocation = piece.as_ptr();
        assert_eq!(piece, decoded.token.to_string().as_bytes());
        executor.render_token(-1234, &mut piece).unwrap();
        assert_eq!(piece, b"-1234");
        assert_eq!(piece.as_ptr(), allocation);

        let mut other = Executor::open("mock://deterministic", 128, 0).unwrap();
        other.decode(&[11]).unwrap();
        assert_eq!(sampler.sample(&mut other), Err(Error::Backend(1)));
    }
    #[test]
    fn mapped_forks_publish_transactionally_and_switch_by_reference() {
        let mut executor = Executor::open("mock://deterministic", 128, 0).unwrap();
        assert!(executor.capabilities().mapped_execution);
        let prefix = executor.tokenize("shared prefix").unwrap();
        executor.decode(&prefix).unwrap();

        let aborted = executor.prepare_mapping_fork(MappingId(0)).unwrap();
        assert_eq!(
            executor.activate_mapping(MappingId(1)),
            Err(Error::Backend(1))
        );
        drop(aborted);
        assert_eq!(executor.mapping_metrics().resident_mappings, 1);

        let prepared = executor.prepare_mapping_fork(MappingId(0)).unwrap();
        let branch = executor.commit_mapping(prepared).unwrap();
        assert_eq!(executor.mapping_metrics().resident_mappings, 2);
        let continuation = [7, 8];
        let staged = executor.decode(&continuation).unwrap();
        executor.activate_mapping(branch).unwrap();
        let mapped = executor.decode(&continuation).unwrap();
        assert_eq!(mapped.token, staged.token);
        assert!(logits_identical(&mapped.logits, &staged.logits));

        let metrics = executor.mapping_metrics();
        assert_eq!(metrics.reference_switches, 1);
        assert_eq!(metrics.activation_bytes_copied, 0);
        executor.activate_mapping(MappingId(0)).unwrap();
        executor.remove_mapping(branch).unwrap();
        assert_eq!(executor.mapping_metrics().resident_mappings, 1);
    }
}
