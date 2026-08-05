use cusco_executor_sys as sys;
use serde::Serialize;
use std::{ffi::CString, ptr::NonNull, slice};
use thiserror::Error;
#[derive(Debug, Error, PartialEq)]
pub enum Error {
    #[error("invalid path")]
    InvalidPath,
    #[error("operation cancelled")]
    Cancelled,
    #[error("incompatible checkpoint")]
    Incompatible,
    #[error("executor backend error {0}")]
    Backend(i32),
}
fn status(code: i32) -> Result<(), Error> {
    match code {
        sys::OK => Ok(()),
        sys::CANCELLED => Err(Error::Cancelled),
        sys::INCOMPATIBLE => Err(Error::Incompatible),
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
}
#[derive(Clone, Debug, Serialize)]
pub struct Decode {
    pub logits: Vec<f32>,
    pub token: i32,
}
pub struct Executor {
    raw: NonNull<sys::CuscoExecutor>,
}
pub struct Checkpoint {
    raw: NonNull<sys::CuscoCheckpoint>,
    pub bytes: usize,
    pub checksum: u64,
}
#[derive(Debug)]
pub struct Prepared {
    raw: NonNull<sys::CuscoCheckpoint>,
}
impl Executor {
    pub fn open(path: &str, n_ctx: u32, gpu_layers: i32) -> Result<Self, Error> {
        let path = CString::new(path).map_err(|_| Error::InvalidPath)?;
        let mut raw = std::ptr::null_mut();
        status(unsafe { sys::cusco_executor_open(path.as_ptr(), n_ctx, gpu_layers, &mut raw) })?;
        Ok(Self {
            raw: NonNull::new(raw).ok_or(Error::Backend(3))?,
        })
    }
    pub fn capabilities(&self) -> Capabilities {
        let c = unsafe { sys::cusco_executor_capabilities(self.raw.as_ptr()) };
        Capabilities {
            abi_version: c.abi_version,
            global_kv: c.has_global_kv != 0,
            swa: c.has_swa != 0,
            recurrent: c.has_recurrent != 0,
            vocabulary: c.n_vocab,
        }
    }
    pub fn tokenize(&mut self, text: &str) -> Result<Vec<i32>, Error> {
        let text = CString::new(text).map_err(|_| Error::InvalidPath)?;
        let mut p = std::ptr::null_mut();
        let mut n = 0;
        status(unsafe {
            sys::cusco_executor_tokenize(self.raw.as_ptr(), text.as_ptr(), &mut p, &mut n)
        })?;
        let v = if n == 0 {
            Vec::new()
        } else {
            unsafe { slice::from_raw_parts(p, n) }.to_vec()
        };
        unsafe { sys::cusco_executor_tokens_free(p) };
        Ok(v)
    }
    pub fn decode(&mut self, tokens: &[i32]) -> Result<Decode, Error> {
        let mut out = sys::DecodeResult {
            logits: std::ptr::null(),
            logits_len: 0,
            token: 0,
        };
        status(unsafe {
            sys::cusco_executor_decode(self.raw.as_ptr(), tokens.as_ptr(), tokens.len(), &mut out)
        })?;
        let logits = if out.logits_len == 0 {
            Vec::new()
        } else {
            unsafe { slice::from_raw_parts(out.logits, out.logits_len) }.to_vec()
        };
        Ok(Decode {
            logits,
            token: out.token,
        })
    }
    pub fn capture(&mut self) -> Result<Checkpoint, Error> {
        let mut p = std::ptr::null_mut();
        status(unsafe { sys::cusco_executor_capture(self.raw.as_ptr(), &mut p) })?;
        let raw = NonNull::new(p).ok_or(Error::Backend(3))?;
        Ok(Checkpoint {
            bytes: unsafe { sys::cusco_checkpoint_size(p) },
            checksum: unsafe { sys::cusco_checkpoint_checksum(p) },
            raw,
        })
    }
    pub fn prepare(&mut self, c: &Checkpoint, checksum: u64) -> Result<Prepared, Error> {
        let mut p = std::ptr::null_mut();
        status(unsafe {
            sys::cusco_executor_prepare_restore(self.raw.as_ptr(), c.raw.as_ptr(), checksum, &mut p)
        })?;
        Ok(Prepared {
            raw: NonNull::new(p).ok_or(Error::Backend(3))?,
        })
    }
    pub fn commit(&mut self, p: Prepared) -> Result<(), Error> {
        let raw = p.raw.as_ptr();
        std::mem::forget(p);
        status(unsafe { sys::cusco_executor_commit_restore(self.raw.as_ptr(), raw) })
    }
    pub fn replace(&mut self, tokens: &[i32]) -> Result<(), Error> {
        status(unsafe {
            sys::cusco_executor_replace(self.raw.as_ptr(), tokens.as_ptr(), tokens.len())
        })
    }
    pub fn cancel_next(&mut self) {
        unsafe { sys::cusco_executor_cancel_next(self.raw.as_ptr()) }
    }
}
impl Drop for Executor {
    fn drop(&mut self) {
        unsafe { sys::cusco_executor_close(self.raw.as_ptr()) }
    }
}
impl Drop for Checkpoint {
    fn drop(&mut self) {
        unsafe { sys::cusco_checkpoint_free(self.raw.as_ptr()) }
    }
}
impl Drop for Prepared {
    fn drop(&mut self) {
        unsafe { sys::cusco_checkpoint_free(self.raw.as_ptr()) }
    }
}
pub fn logits_identical(a: &[f32], b: &[f32]) -> bool {
    a.len() == b.len() && a.iter().zip(b).all(|(x, y)| x.to_bits() == y.to_bits())
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
        assert_eq!(status(3), Err(Error::Backend(3)));
    }
    #[test]
    fn model_free_lifecycle_is_transactional() {
        let mut executor = Executor::open("mock://deterministic", 128, 0).unwrap();
        let capabilities = executor.capabilities();
        assert!(capabilities.global_kv && capabilities.swa && capabilities.recurrent);
        let prefix = executor.tokenize("prefix").unwrap();
        executor.replace(&prefix).unwrap();
        let checkpoint = executor.capture().unwrap();
        assert!(checkpoint.bytes > 0);
        let continuation = [7];
        let expected = executor.decode(&continuation).unwrap();
        let unrelated = executor.tokenize("other").unwrap();
        executor.replace(&unrelated).unwrap();
        assert_eq!(
            executor
                .prepare(&checkpoint, checkpoint.checksum ^ 1)
                .unwrap_err(),
            Error::Incompatible
        );
        let prepared = executor.prepare(&checkpoint, checkpoint.checksum).unwrap();
        executor.commit(prepared).unwrap();
        let actual = executor.decode(&continuation).unwrap();
        assert_eq!(actual.token, expected.token);
        assert!(logits_identical(&actual.logits, &expected.logits));
        executor.cancel_next();
        assert_eq!(
            executor.decode(&continuation).unwrap_err(),
            Error::Cancelled
        );
        assert!(matches!(
            Executor::open("bad\0path", 1, 0),
            Err(Error::InvalidPath)
        ));
    }
}
