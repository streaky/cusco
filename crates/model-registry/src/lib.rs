use serde::Serialize;
use sha2::{Digest, Sha256};
use std::{
    fs::{self, File},
    io::{Read, Write},
    path::{Path, PathBuf},
};
use thiserror::Error;
pub const GEMMA_URI: &str = "hf://models/unsloth/gemma-4-E2B-it-GGUF@0314792d7f1f7e229411f620751375812bb9faf2/gemma-4-E2B-it-Q3_K_M.gguf";
#[derive(Debug, Error)]
pub enum Error {
    #[error("invalid hf uri")]
    InvalidUri,
    #[error("artifact digest differs: expected {expected}, got {actual}")]
    Digest { expected: String, actual: String },
    #[error(transparent)]
    Io(#[from] std::io::Error),
    #[error("download failed: {0}")]
    Http(#[from] ureq::Error),
}
#[derive(Clone, Debug, Serialize)]
pub struct ModelRecord {
    pub identity: String,
    pub path: PathBuf,
    pub sha256: String,
    pub size: u64,
}
fn digest(path: &Path) -> Result<(String, u64), Error> {
    let mut f = File::open(path)?;
    let mut h = Sha256::new();
    let mut buf = [0; 1024 * 1024];
    let mut size = 0;
    loop {
        let n = f.read(&mut buf)?;
        if n == 0 {
            break;
        }
        h.update(&buf[..n]);
        size += n as u64;
    }
    Ok((hex::encode(h.finalize()), size))
}
pub fn register_local(
    path: impl AsRef<Path>,
    identity: &str,
    expected: Option<&str>,
) -> Result<ModelRecord, Error> {
    let path = fs::canonicalize(path)?;
    let (sha256, size) = digest(&path)?;
    if expected.is_some_and(|v| !v.eq_ignore_ascii_case(&sha256)) {
        return Err(Error::Digest {
            expected: expected.unwrap().to_owned(),
            actual: sha256,
        });
    }
    Ok(ModelRecord {
        identity: identity.to_owned(),
        path,
        sha256,
        size,
    })
}
fn parse_hf(uri: &str) -> Result<(String, String, String), Error> {
    let rest = uri.strip_prefix("hf://models/").ok_or(Error::InvalidUri)?;
    let (org, remainder) = rest.split_once('/').ok_or(Error::InvalidUri)?;
    let (model_revision, file) = remainder.split_once('/').ok_or(Error::InvalidUri)?;
    let (model, revision) = model_revision.rsplit_once('@').ok_or(Error::InvalidUri)?;
    if org.is_empty() || model.is_empty() || revision.len() != 40 || file.is_empty() {
        return Err(Error::InvalidUri);
    }
    Ok((
        format!("{org}/{model}"),
        revision.to_owned(),
        file.to_owned(),
    ))
}
pub fn fetch_hf(uri: &str, cache: &Path, expected: Option<&str>) -> Result<ModelRecord, Error> {
    let (repo, revision, file) = parse_hf(uri)?;
    let dir = cache
        .join("models")
        .join(repo.replace('/', "--"))
        .join(&revision);
    fs::create_dir_all(&dir)?;
    let destination = dir.join(&file);
    if !destination.exists() {
        let url = format!("https://huggingface.co/{repo}/resolve/{revision}/{file}");
        let tmp = destination.with_extension("partial");
        let mut response = ureq::get(url).call()?;
        let mut output = File::create(&tmp)?;
        std::io::copy(&mut response.body_mut().as_reader(), &mut output)?;
        output.flush()?;
        fs::rename(tmp, &destination)?;
    }
    register_local(destination, uri, expected)
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn registers_and_checks_local() {
        let dir = std::env::temp_dir().join(format!("cusco-registry-{}", std::process::id()));
        fs::create_dir_all(&dir).unwrap();
        let p = dir.join("m.gguf");
        fs::write(&p, b"model").unwrap();
        let record = register_local(&p, "test", None).unwrap();
        assert_eq!(record.size, 5);
        assert_eq!(
            record.sha256,
            "9372c470eeadd5ecd9c3c74c2b3cb633f8e2f2fad799250a0f70d652b6b825e4"
        );
        assert!(matches!(
            register_local(&p, "test", Some("bad")),
            Err(Error::Digest { .. })
        ));
        fs::remove_dir_all(dir).unwrap();
    }
    #[test]
    fn parses_locked_uri() {
        assert_eq!(
            parse_hf(GEMMA_URI).unwrap().1,
            "0314792d7f1f7e229411f620751375812bb9faf2"
        );
        assert!(matches!(parse_hf("https://bad"), Err(Error::InvalidUri)));
        assert!(matches!(
            parse_hf("hf://models/a/b@short/f"),
            Err(Error::InvalidUri)
        ));
    }
}
