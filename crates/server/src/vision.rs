use crate::VisionConfig;
use base64::{Engine, engine::general_purpose::STANDARD};
use image::{AnimationDecoder, GenericImageView, ImageFormat, ImageReader};
use parking_lot::Mutex;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::{collections::HashMap, io::Cursor, sync::Arc};
use thiserror::Error;

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct AdmittedImage {
    pub digest: String,
    pub mime_type: String,
    pub width: u32,
    pub height: u32,
    pub pixels: u64,
    pub has_alpha: bool,
    #[serde(skip)]
    pub bytes: Arc<[u8]>,
}

#[derive(Clone, Debug, Error, Serialize, PartialEq, Eq)]
#[error("{code}: {message}")]
pub struct VisionError {
    pub code: &'static str,
    pub message: String,
    pub field: &'static str,
    pub observed: Option<u64>,
    pub limit: Option<u64>,
    pub unit: Option<&'static str>,
}
impl VisionError {
    fn malformed(message: impl Into<String>) -> Self { Self { code: "image_malformed", message: message.into(), field: "image", observed: None, limit: None, unit: None } }
    fn limit(field: &'static str, observed: u64, limit: u64, unit: &'static str) -> Self { Self { code: "image_limit_exceeded", message: format!("{field} exceeds configured limit"), field, observed: Some(observed), limit: Some(limit), unit: Some(unit) } }
    pub fn unsupported(message: impl Into<String>) -> Self { Self { code: "image_unsupported", message: message.into(), field: "image", observed: None, limit: None, unit: None } }
}

#[derive(Default)]
struct CacheState { bytes: u64, values: HashMap<String, AdmittedImage> }
#[derive(Clone)]
pub struct ImageAdmission { config: VisionConfig, cache: Arc<Mutex<CacheState>> }

impl ImageAdmission {
    pub fn new(config: VisionConfig) -> Self { Self { config, cache: Arc::new(Mutex::new(CacheState::default())) } }
    pub fn admit_data_uri(&self, value: &str) -> Result<AdmittedImage, VisionError> {
        let (header, encoded) = value.split_once(',').ok_or_else(|| VisionError::malformed("image_url must be a base64 data URI"))?;
        let declared = header.strip_prefix("data:").and_then(|value| value.strip_suffix(";base64")).ok_or_else(|| VisionError::malformed("only base64 data URIs are supported"))?;
        self.admit_base64(declared, encoded)
    }
    pub fn admit_base64(&self, declared_mime: &str, encoded: &str) -> Result<AdmittedImage, VisionError> {
        if encoded.len() > self.config.max_encoded_bytes { return Err(VisionError::limit("encoded_bytes", encoded.len() as u64, self.config.max_encoded_bytes as u64, "bytes")); }
        let normalized: String = encoded.chars().filter(|character| !character.is_ascii_whitespace()).collect();
        let bytes = STANDARD.decode(normalized.as_bytes()).map_err(|_| VisionError::malformed("invalid base64 image data"))?;
        if bytes.len() > self.config.max_decoded_bytes { return Err(VisionError::limit("decoded_bytes", bytes.len() as u64, self.config.max_decoded_bytes as u64, "bytes")); }
        let format = image::guess_format(&bytes).map_err(|_| VisionError::unsupported("unsupported image format"))?;
        let actual_mime = match format { ImageFormat::Jpeg => "image/jpeg", ImageFormat::Png => "image/png", ImageFormat::WebP => "image/webp", ImageFormat::Gif => "image/gif", _ => return Err(VisionError::unsupported("only JPEG, PNG, WebP, and static GIF are supported")) };
        if declared_mime != actual_mime { return Err(VisionError::malformed(format!("declared MIME type {declared_mime} does not match {actual_mime}"))); }
        if format == ImageFormat::Gif {
            let decoder = image::codecs::gif::GifDecoder::new(Cursor::new(&bytes)).map_err(|_| VisionError::malformed("invalid GIF"))?;
            if decoder.into_frames().take(2).count() > 1 { return Err(VisionError::unsupported("animated GIF images are unsupported")); }
        }
        let reader = ImageReader::with_format(Cursor::new(&bytes), format);
        let image = reader.decode().map_err(|_| VisionError::malformed("image could not be decoded"))?;
        let (width, height) = image.dimensions();
        if width > self.config.max_dimension { return Err(VisionError::limit("width", width as u64, self.config.max_dimension as u64, "pixels")); }
        if height > self.config.max_dimension { return Err(VisionError::limit("height", height as u64, self.config.max_dimension as u64, "pixels")); }
        let pixels = u64::from(width).saturating_mul(u64::from(height));
        if pixels > self.config.max_total_pixels { return Err(VisionError::limit("total_pixels", pixels, self.config.max_total_pixels, "pixels")); }
        let digest = format!("{:x}", Sha256::digest(&bytes));
        let mut cache = self.cache.lock();
        if let Some(existing) = cache.values.get(&digest) { if existing.bytes.as_ref() == bytes { return Ok(existing.clone()); } }
        let next = cache.bytes.saturating_add(bytes.len() as u64);
        if next > self.config.retention_capacity.0 { return Err(VisionError::limit("retention_bytes", next, self.config.retention_capacity.0, "bytes")); }
        let admitted = AdmittedImage { digest: digest.clone(), mime_type: actual_mime.into(), width, height, pixels, has_alpha: image.color().has_alpha(), bytes: bytes.into() };
        cache.bytes = next;
        cache.values.insert(digest, admitted.clone());
        Ok(admitted)
    }
    pub fn retained_bytes(&self) -> u64 { self.cache.lock().bytes }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn admission() -> ImageAdmission { ImageAdmission::new(VisionConfig::default()) }
    #[test]
    fn validates_mime_and_deduplicates_exact_bytes() {
        let png = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M/wHwAF/gL+AvzZ6QAAAABJRU5ErkJggg==";
        let images = admission();
        let first = images.admit_data_uri(&format!("data:image/png;base64,{png}")).unwrap();
        let retained = images.retained_bytes();
        assert_eq!(images.admit_base64("image/png", png).unwrap().digest, first.digest);
        assert_eq!(images.retained_bytes(), retained);
        assert_eq!(images.admit_base64("image/jpeg", png).unwrap_err().code, "image_malformed");
    }
    #[test]
    fn rejects_remote_and_unsupported_content() {
        assert_eq!(admission().admit_data_uri("https://example.test/a.png").unwrap_err().code, "image_malformed");
        assert_eq!(admission().admit_base64("image/svg+xml", "PHN2Zy8+").unwrap_err().code, "image_unsupported");
    }
}
