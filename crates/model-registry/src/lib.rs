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
    #[error("invalid GGUF metadata: {0}")]
    InvalidMetadata(String),
    #[error("invalid Hub metadata: {0}")]
    Json(#[from] serde_json::Error),
}
#[derive(Clone, Debug, Serialize)]
pub struct ModelRecord {
    pub identity: String,
    pub path: PathBuf,
    pub sha256: String,
    pub size: u64,
}

#[derive(Clone, Debug, Serialize, PartialEq, Eq)]
pub struct ModelMetadata {
    pub architecture: String,
    pub name: Option<String>,
}

fn read_u32(reader: &mut impl Read) -> Result<u32, Error> {
    let mut bytes = [0; 4];
    reader.read_exact(&mut bytes)?;
    Ok(u32::from_le_bytes(bytes))
}
fn read_u64(reader: &mut impl Read) -> Result<u64, Error> {
    let mut bytes = [0; 8];
    reader.read_exact(&mut bytes)?;
    Ok(u64::from_le_bytes(bytes))
}
fn read_string(reader: &mut impl Read) -> Result<String, Error> {
    let length = usize::try_from(read_u64(reader)?)
        .map_err(|_| Error::InvalidMetadata("string length exceeds address space".into()))?;
    if length > 16 << 20 {
        return Err(Error::InvalidMetadata(
            "metadata string exceeds 16 MiB".into(),
        ));
    }
    let mut bytes = vec![0; length];
    reader.read_exact(&mut bytes)?;
    String::from_utf8(bytes)
        .map_err(|_| Error::InvalidMetadata("metadata string is not UTF-8".into()))
}
fn skip_value(reader: &mut impl Read, kind: u32) -> Result<(), Error> {
    let bytes = match kind {
        0 | 1 | 7 => 1,
        2 | 3 => 2,
        4 | 5 | 6 => 4,
        10 | 11 | 12 => 8,
        8 => {
            let _ = read_string(reader)?;
            return Ok(());
        }
        9 => {
            let element = read_u32(reader)?;
            let count = read_u64(reader)?;
            if count > 1_000_000 {
                return Err(Error::InvalidMetadata(
                    "metadata array is unreasonably large".into(),
                ));
            }
            for _ in 0..count {
                skip_value(reader, element)?;
            }
            return Ok(());
        }
        _ => {
            return Err(Error::InvalidMetadata(format!(
                "unknown metadata type {kind}"
            )));
        }
    };
    let mut buffer = [0; 8];
    reader.read_exact(&mut buffer[..bytes])?;
    Ok(())
}

pub fn probe_gguf(path: impl AsRef<Path>) -> Result<ModelMetadata, Error> {
    let mut reader = File::open(path)?;
    let mut magic = [0; 4];
    reader.read_exact(&mut magic)?;
    if &magic != b"GGUF" {
        return Err(Error::InvalidMetadata("missing GGUF magic".into()));
    }
    let version = read_u32(&mut reader)?;
    if !(2..=3).contains(&version) {
        return Err(Error::InvalidMetadata(format!(
            "unsupported GGUF version {version}"
        )));
    }
    let _tensor_count = read_u64(&mut reader)?;
    let metadata_count = read_u64(&mut reader)?;
    if metadata_count > 1_000_000 {
        return Err(Error::InvalidMetadata(
            "metadata entry count is unreasonably large".into(),
        ));
    }
    let mut architecture = None;
    let mut name = None;
    for _ in 0..metadata_count {
        let key = read_string(&mut reader)?;
        let kind = read_u32(&mut reader)?;
        if (key == "general.architecture" || key == "general.name") && kind == 8 {
            let value = read_string(&mut reader)?;
            if key == "general.architecture" {
                architecture = Some(value);
            } else {
                name = Some(value);
            }
        } else {
            skip_value(&mut reader, kind)?;
        }
    }
    Ok(ModelMetadata {
        architecture: architecture
            .ok_or_else(|| Error::InvalidMetadata("general.architecture is absent".into()))?,
        name,
    })
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
    let simple_file = Path::new(file)
        .components()
        .all(|component| matches!(component, std::path::Component::Normal(_)))
        && Path::new(file).components().count() == 1;
    if org.is_empty()
        || model.is_empty()
        || revision.is_empty()
        || revision.contains('/')
        || file.is_empty()
        || !simple_file
    {
        return Err(Error::InvalidUri);
    }
    Ok((
        format!("{org}/{model}"),
        revision.to_owned(),
        file.to_owned(),
    ))
}
fn resolve_revision(repo: &str, revision: &str) -> Result<String, Error> {
    if revision.len() == 40 && revision.bytes().all(|byte| byte.is_ascii_hexdigit()) {
        return Ok(revision.to_ascii_lowercase());
    }
    let url = format!("https://huggingface.co/api/models/{repo}/revision/{revision}");
    let mut response = ureq::get(url).call()?;
    let value: serde_json::Value = serde_json::from_reader(response.body_mut().as_reader())?;
    let sha = value
        .get("sha")
        .and_then(serde_json::Value::as_str)
        .ok_or(Error::InvalidUri)?;
    if sha.len() != 40 || !sha.bytes().all(|byte| byte.is_ascii_hexdigit()) {
        return Err(Error::InvalidUri);
    }
    Ok(sha.to_ascii_lowercase())
}
pub fn fetch_hf(uri: &str, cache: &Path, expected: Option<&str>) -> Result<ModelRecord, Error> {
    let (repo, requested_revision, file) = parse_hf(uri)?;
    let revision = resolve_revision(&repo, &requested_revision)?;
    let dir = cache
        .join("models")
        .join(repo.replace('/', "--"))
        .join(&revision);
    fs::create_dir_all(&dir)?;
    let destination = dir.join(&file);
    if destination.exists() && register_local(&destination, uri, expected).is_err() {
        fs::remove_file(&destination)?;
    }
    if !destination.exists() {
        let url = format!("https://huggingface.co/{repo}/resolve/{revision}/{file}");
        let tmp = destination.with_extension("partial");
        let mut response = ureq::get(url).call()?;
        let mut output = File::create(&tmp)?;
        std::io::copy(&mut response.body_mut().as_reader(), &mut output)?;
        output.flush()?;
        fs::rename(tmp, &destination)?;
    }
    let identity = format!("hf://models/{repo}@{revision}/{file}");
    register_local(destination, &identity, expected)
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
        assert_eq!(parse_hf("hf://models/a/b@main/f").unwrap().1, "main");
        assert!(matches!(
            parse_hf("hf://models/a/b@0123456789012345678901234567890123456789/../escape.gguf"),
            Err(Error::InvalidUri)
        ));
    }
    #[test]
    fn probes_architecture_from_gguf_metadata() {
        let path = std::env::temp_dir().join(format!("cusco-probe-{}.gguf", std::process::id()));
        let mut bytes = Vec::new();
        bytes.extend_from_slice(b"GGUF");
        bytes.extend_from_slice(&3_u32.to_le_bytes());
        bytes.extend_from_slice(&0_u64.to_le_bytes());
        bytes.extend_from_slice(&1_u64.to_le_bytes());
        bytes.extend_from_slice(&20_u64.to_le_bytes());
        bytes.extend_from_slice(b"general.architecture");
        bytes.extend_from_slice(&8_u32.to_le_bytes());
        bytes.extend_from_slice(&6_u64.to_le_bytes());
        bytes.extend_from_slice(b"gemma3");
        fs::write(&path, bytes).unwrap();
        assert_eq!(probe_gguf(&path).unwrap().architecture, "gemma3");
        fs::remove_file(path).unwrap();
    }
    #[test]
    fn rejects_invalid_gguf_headers_and_metadata_values() {
        let mut scalar = std::io::Cursor::new(vec![0_u8; 8]);
        for kind in [0, 2, 4, 7, 10, 12] {
            scalar.set_position(0);
            skip_value(&mut scalar, kind).unwrap();
        }
        let mut string = Vec::new();
        string.extend_from_slice(&3_u64.to_le_bytes());
        string.extend_from_slice(b"abc");
        skip_value(&mut std::io::Cursor::new(string), 8).unwrap();
        assert!(matches!(
            skip_value(&mut std::io::Cursor::new(Vec::<u8>::new()), 99),
            Err(Error::InvalidMetadata(_))
        ));

        let path = std::env::temp_dir().join(format!("cusco-invalid-{}.gguf", std::process::id()));
        fs::write(&path, b"nope").unwrap();
        assert!(matches!(probe_gguf(&path), Err(Error::InvalidMetadata(_))));
        let mut header = Vec::new();
        header.extend_from_slice(b"GGUF");
        header.extend_from_slice(&1_u32.to_le_bytes());
        fs::write(&path, &header).unwrap();
        assert!(matches!(probe_gguf(&path), Err(Error::InvalidMetadata(_))));
        header[4..8].copy_from_slice(&3_u32.to_le_bytes());
        header.extend_from_slice(&0_u64.to_le_bytes());
        header.extend_from_slice(&0_u64.to_le_bytes());
        fs::write(&path, &header).unwrap();
        assert!(matches!(probe_gguf(&path), Err(Error::InvalidMetadata(_))));
        fs::remove_file(path).unwrap();
    }
}
