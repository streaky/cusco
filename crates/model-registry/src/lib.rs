use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::{
    fs::{self, File, OpenOptions},
    io::{Read, Write},
    path::{Path, PathBuf},
};
use thiserror::Error;
pub const GEMMA_URI: &str = "hf://unsloth/gemma-4-E2B-it-GGUF/gemma-4-E2B-it-Q3_K_M.gguf";
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

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DownloadProgress {
    pub total: Option<u64>,
    pub completed: u64,
}

#[derive(Deserialize, Serialize)]
struct CacheStamp {
    sha256: String,
    size: u64,
    modified_ns: u128,
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
        4..=6 => 4,
        10..=12 => 8,
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
fn cached_record(
    path: &Path,
    identity: &str,
    expected: Option<&str>,
) -> Result<Option<ModelRecord>, Error> {
    let metadata = fs::metadata(path)?;
    let modified_ns = metadata
        .modified()?
        .duration_since(std::time::UNIX_EPOCH)
        .map_err(std::io::Error::other)?
        .as_nanos();
    let stamp_path = path.with_extension("cache.json");
    let Ok(stamp) = fs::read(&stamp_path)
        .ok()
        .and_then(|bytes| serde_json::from_slice::<CacheStamp>(&bytes).ok())
        .ok_or(())
    else {
        return Ok(None);
    };
    if stamp.size != metadata.len()
        || stamp.modified_ns != modified_ns
        || expected.is_some_and(|value| !value.eq_ignore_ascii_case(&stamp.sha256))
    {
        return Ok(None);
    }
    Ok(Some(ModelRecord {
        identity: identity.to_owned(),
        path: fs::canonicalize(path)?,
        sha256: stamp.sha256,
        size: stamp.size,
    }))
}
fn parse_hf(uri: &str) -> Result<(String, String, String), Error> {
    let rest = uri
        .strip_prefix("hf://models/")
        .or_else(|| uri.strip_prefix("hf://"))
        .ok_or(Error::InvalidUri)?;
    let (org, remainder) = rest.split_once('/').ok_or(Error::InvalidUri)?;
    let (model_revision, file) = remainder.split_once('/').ok_or(Error::InvalidUri)?;
    let (model, revision) = model_revision
        .rsplit_once('@')
        .map_or((model_revision, "main"), |(model, revision)| {
            (model, revision)
        });
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
fn resolve_revision(repo: &str, revision: &str, cache: &Path) -> Result<String, Error> {
    if revision.len() == 40 && revision.bytes().all(|byte| byte.is_ascii_hexdigit()) {
        return Ok(revision.to_ascii_lowercase());
    }
    let reference = cache
        .join("refs")
        .join(repo.replace('/', "--"))
        .join(revision);
    if let Ok(value) = fs::read_to_string(&reference) {
        let value = value.trim();
        if value.len() == 40 && value.bytes().all(|byte| byte.is_ascii_hexdigit()) {
            return Ok(value.to_ascii_lowercase());
        }
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
    let sha = sha.to_ascii_lowercase();
    if let Some(parent) = reference.parent() {
        fs::create_dir_all(parent)?;
    }
    fs::write(reference, &sha)?;
    Ok(sha)
}
fn response_total(response: &ureq::http::Response<ureq::Body>, offset: u64) -> Option<u64> {
    response
        .headers()
        .get("content-range")
        .and_then(|value| value.to_str().ok())
        .and_then(|value| value.rsplit_once('/'))
        .and_then(|(_, total)| total.parse().ok())
        .or_else(|| {
            response
                .headers()
                .get("content-length")
                .and_then(|value| value.to_str().ok())
                .and_then(|value| value.parse::<u64>().ok())
                .map(|length| length.saturating_add(offset))
        })
}

fn download_resumable(
    url: &str,
    partial: &Path,
    progress: &mut impl FnMut(DownloadProgress),
) -> Result<(), Error> {
    let requested_offset = fs::metadata(partial).map_or(0, |metadata| metadata.len());
    let mut request = ureq::get(url);
    if requested_offset > 0 {
        request = request.header("Range", &format!("bytes={requested_offset}-"));
    }
    let mut response = request.call()?;
    let resumes_at_offset = response.status().as_u16() == 206
        && response
            .headers()
            .get("content-range")
            .and_then(|value| value.to_str().ok())
            .is_some_and(|value| value.starts_with(&format!("bytes {requested_offset}-")));
    let offset = if resumes_at_offset {
        requested_offset
    } else {
        0
    };
    let total = response_total(&response, offset);
    let mut output = if offset > 0 {
        OpenOptions::new().append(true).open(partial)?
    } else {
        File::create(partial)?
    };
    let mut completed = offset;
    progress(DownloadProgress { total, completed });
    let mut buffer = [0; 256 * 1024];
    let mut input = response.body_mut().as_reader();
    loop {
        let read = input.read(&mut buffer)?;
        if read == 0 {
            break;
        }
        output.write_all(&buffer[..read])?;
        completed = completed.saturating_add(read as u64);
        progress(DownloadProgress { total, completed });
    }
    output.flush()?;
    Ok(())
}

pub fn fetch_hf(uri: &str, cache: &Path, expected: Option<&str>) -> Result<ModelRecord, Error> {
    fetch_hf_with_progress(uri, cache, expected, |_| {})
}

pub fn fetch_hf_with_progress(
    uri: &str,
    cache: &Path,
    expected: Option<&str>,
    mut progress: impl FnMut(DownloadProgress),
) -> Result<ModelRecord, Error> {
    let (repo, requested_revision, file) = parse_hf(uri)?;
    let revision = resolve_revision(&repo, &requested_revision, cache)?;
    let dir = cache
        .join("models")
        .join(repo.replace('/', "--"))
        .join(&revision);
    fs::create_dir_all(&dir)?;
    let destination = dir.join(&file);
    if destination.exists()
        && cached_record(&destination, uri, expected)?.is_none()
        && register_local(&destination, uri, expected).is_err()
    {
        fs::remove_file(&destination)?;
    }
    if !destination.exists() {
        let url = format!("https://huggingface.co/{repo}/resolve/{revision}/{file}");
        let tmp = destination.with_extension("partial");
        download_resumable(&url, &tmp, &mut progress)?;
        fs::rename(tmp, &destination)?;
    } else {
        let size = fs::metadata(&destination)?.len();
        progress(DownloadProgress {
            total: Some(size),
            completed: size,
        });
    }
    let identity = format!("hf://{repo}@{revision}/{file}");
    if let Some(record) = cached_record(&destination, &identity, expected)? {
        return Ok(record);
    }
    let record = register_local(&destination, &identity, expected)?;
    let metadata = fs::metadata(&destination)?;
    let stamp = CacheStamp {
        sha256: record.sha256.clone(),
        size: record.size,
        modified_ns: metadata
            .modified()?
            .duration_since(std::time::UNIX_EPOCH)
            .map_err(std::io::Error::other)?
            .as_nanos(),
    };
    fs::write(
        destination.with_extension("cache.json"),
        serde_json::to_vec(&stamp)?,
    )?;
    Ok(record)
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
    fn reuses_cached_symbolic_revision_without_network() {
        let cache = std::env::temp_dir().join(format!("cusco-hf-ref-{}", std::process::id()));
        let reference = cache.join("refs").join("a--b").join("main");
        fs::create_dir_all(reference.parent().unwrap()).unwrap();
        fs::write(&reference, "0123456789012345678901234567890123456789\n").unwrap();
        assert_eq!(
            resolve_revision("a/b", "main", &cache).unwrap(),
            "0123456789012345678901234567890123456789"
        );
        fs::remove_dir_all(cache).unwrap();
    }
    fn download_from(
        response: &'static str,
        initial: &[u8],
    ) -> (Vec<u8>, String, Vec<DownloadProgress>) {
        use std::{
            io::{Read as _, Write as _},
            net::TcpListener,
            sync::mpsc,
            thread,
        };
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let address = listener.local_addr().unwrap();
        let (sender, receiver) = mpsc::channel();
        let server = thread::spawn(move || {
            let (mut stream, _) = listener.accept().unwrap();
            let mut request = [0_u8; 4096];
            let length = stream.read(&mut request).unwrap();
            sender
                .send(String::from_utf8_lossy(&request[..length]).into_owned())
                .unwrap();
            stream.write_all(response.as_bytes()).unwrap();
        });
        let partial = std::env::temp_dir().join(format!(
            "cusco-download-{}-{}",
            std::process::id(),
            address.port()
        ));
        fs::write(&partial, initial).unwrap();
        let mut progress = Vec::new();
        download_resumable(
            &format!("http://{address}/model"),
            &partial,
            &mut |update| {
                progress.push(update);
            },
        )
        .unwrap();
        let bytes = fs::read(&partial).unwrap();
        fs::remove_file(partial).unwrap();
        server.join().unwrap();
        (bytes, receiver.recv().unwrap(), progress)
    }

    #[test]
    fn resumes_partial_downloads_with_a_valid_content_range() {
        let (bytes, request, progress) = download_from(
            "HTTP/1.1 206 Partial Content\r\nContent-Range: bytes 5-10/11\r\nContent-Length: 6\r\nConnection: close\r\n\r\n world",
            b"hello",
        );
        assert_eq!(bytes, b"hello world");
        assert!(request.to_ascii_lowercase().contains("range: bytes=5-"));
        assert_eq!(
            progress,
            vec![
                DownloadProgress {
                    total: Some(11),
                    completed: 5
                },
                DownloadProgress {
                    total: Some(11),
                    completed: 11
                }
            ]
        );
    }

    #[test]
    fn restarts_partial_download_when_the_server_ignores_range() {
        let (bytes, request, progress) = download_from(
            "HTTP/1.1 200 OK\r\nContent-Length: 11\r\nConnection: close\r\n\r\nhello world",
            b"stale",
        );
        assert_eq!(bytes, b"hello world");
        assert!(request.to_ascii_lowercase().contains("range: bytes=5-"));
        assert_eq!(progress.last().unwrap().completed, 11);
        assert_eq!(progress.last().unwrap().total, Some(11));
    }
    #[test]
    fn parses_locked_and_default_revision_uris() {
        assert_eq!(parse_hf(GEMMA_URI).unwrap().1, "main");
        assert!(matches!(parse_hf("https://bad"), Err(Error::InvalidUri)));
        assert_eq!(parse_hf("hf://models/a/b@main/f").unwrap().1, "main");
        assert_eq!(parse_hf("hf://a/b/f").unwrap().1, "main");
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
        bytes.extend_from_slice(b"gemma4");
        fs::write(&path, bytes).unwrap();
        assert_eq!(probe_gguf(&path).unwrap().architecture, "gemma4");
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
