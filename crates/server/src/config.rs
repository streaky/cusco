use crate::{HttpDebugLevel, SchedulerPolicyConfig, ServerConfig};
use serde::{Deserialize, Serialize};
use std::{
    fs,
    net::SocketAddr,
    path::{Path, PathBuf},
    str::FromStr,
};
use thiserror::Error;

const CONFIG_VERSION: u32 = 1;

#[derive(Debug, Error)]
pub enum ConfigError {
    #[error("read configuration: {0}")]
    Io(#[from] std::io::Error),
    #[error("parse configuration: {0}")]
    Yaml(#[from] serde_yaml::Error),
    #[error("unsupported configuration version {0}")]
    Version(u32),
    #[error("invalid configuration: {0}")]
    Invalid(String),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ByteSize(pub u64);

impl<'de> Deserialize<'de> for ByteSize {
    fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let value = String::deserialize(deserializer)?;
        value.parse().map_err(serde::de::Error::custom)
    }
}
impl Serialize for ByteSize {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        serializer.serialize_str(&format!("{} B", self.0))
    }
}
impl FromStr for ByteSize {
    type Err = String;
    fn from_str(value: &str) -> Result<Self, Self::Err> {
        let value = value.trim();
        let split = value
            .find(char::is_whitespace)
            .ok_or_else(|| "size must contain a unit (for example `8 GiB`)".to_string())?;
        let amount: f64 = value[..split]
            .trim()
            .parse()
            .map_err(|_| "size amount must be numeric".to_string())?;
        if !amount.is_finite() || amount < 0.0 {
            return Err("size amount must be finite and non-negative".into());
        }
        let multiplier = match value[split..].trim().to_ascii_lowercase().as_str() {
            "b" => 1.0,
            "kib" => 1024.0,
            "mib" => 1024.0_f64.powi(2),
            "gib" => 1024.0_f64.powi(3),
            "tib" => 1024.0_f64.powi(4),
            "kb" => 1000.0,
            "mb" => 1000.0_f64.powi(2),
            "gb" => 1000.0_f64.powi(3),
            "tb" => 1000.0_f64.powi(4),
            _ => {
                return Err(
                    "unsupported size unit; use B, KiB, MiB, GiB, TiB, KB, MB, GB, or TB".into(),
                );
            }
        };
        let bytes = amount * multiplier;
        if bytes > u64::MAX as f64 {
            return Err("size exceeds u64 byte accounting".into());
        }
        Ok(Self(bytes.round() as u64))
    }
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct DaemonConfig {
    pub version: u32,
    #[serde(default = "default_listen")]
    pub listen: SocketAddr,
    #[serde(default)]
    pub unsafe_public_unauthenticated: bool,
    #[serde(default)]
    pub bearer_token: Option<String>,
    #[serde(default)]
    pub http_debug: HttpDebugLevel,
    pub paths: DataPaths,
    pub execution: ExecutionConfig,
    #[serde(default)]
    pub server: ServerConfig,
    #[serde(default)]
    pub scheduler: SchedulerPolicyConfig,
    #[serde(default)]
    pub vision: VisionConfig,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct DataPaths {
    pub database: PathBuf,
    pub models: PathBuf,
    pub spill: PathBuf,
    pub user_models: PathBuf,
    pub user_config: PathBuf,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ExecutionConfig {
    pub device_capacity: ByteSize,
    pub host_capacity: ByteSize,
    pub storage_capacity: ByteSize,
    pub context_reserve: ByteSize,
    #[serde(default = "default_context")]
    pub context_tokens: u32,
    #[serde(default = "default_gpu_layers")]
    pub gpu_layers: i32,
    #[serde(default)]
    pub require_competent: bool,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct VisionConfig {
    #[serde(default = "default_image_count")]
    pub max_images: usize,
    #[serde(default = "default_encoded_bytes")]
    pub max_encoded_bytes: usize,
    #[serde(default = "default_decoded_bytes")]
    pub max_decoded_bytes: usize,
    #[serde(default = "default_dimension")]
    pub max_dimension: u32,
    #[serde(default = "default_pixels")]
    pub max_total_pixels: u64,
    #[serde(default = "default_retention")]
    pub retention_capacity: ByteSize,
    #[serde(default)]
    pub metadata_allowlist: Vec<String>,
}

impl Default for VisionConfig {
    fn default() -> Self {
        Self {
            max_images: default_image_count(),
            max_encoded_bytes: default_encoded_bytes(),
            max_decoded_bytes: default_decoded_bytes(),
            max_dimension: default_dimension(),
            max_total_pixels: default_pixels(),
            retention_capacity: default_retention(),
            metadata_allowlist: vec![],
        }
    }
}
fn default_listen() -> SocketAddr {
    "127.0.0.1:8080".parse().expect("static address")
}
fn default_context() -> u32 {
    4096
}
fn default_gpu_layers() -> i32 {
    99
}
fn default_image_count() -> usize {
    8
}
fn default_encoded_bytes() -> usize {
    16 << 20
}
fn default_decoded_bytes() -> usize {
    12 << 20
}
fn default_dimension() -> u32 {
    8192
}
fn default_pixels() -> u64 {
    32_000_000
}
fn default_retention() -> ByteSize {
    ByteSize(256 << 20)
}

impl DaemonConfig {
    pub fn load(path: impl AsRef<Path>) -> Result<Self, ConfigError> {
        let config: Self = serde_yaml::from_slice(&fs::read(path)?)?;
        config.validate()
    }
    pub fn validate(self) -> Result<Self, ConfigError> {
        if self.version != CONFIG_VERSION {
            return Err(ConfigError::Version(self.version));
        }
        if self.execution.device_capacity.0 == 0
            || self.execution.host_capacity.0 == 0
            || self.execution.storage_capacity.0 == 0
        {
            return Err(ConfigError::Invalid(
                "execution capacities must be nonzero".into(),
            ));
        }
        if self.execution.context_reserve.0 > self.execution.storage_capacity.0 {
            return Err(ConfigError::Invalid(
                "context reserve exceeds storage capacity".into(),
            ));
        }
        if self.execution.context_tokens == 0 {
            return Err(ConfigError::Invalid(
                "context_tokens must be nonzero".into(),
            ));
        }
        if self.vision.max_images == 0
            || self.vision.max_encoded_bytes == 0
            || self.vision.max_decoded_bytes == 0
            || self.vision.max_dimension == 0
            || self.vision.max_total_pixels == 0
        {
            return Err(ConfigError::Invalid("vision limits must be nonzero".into()));
        }
        if self.vision.retention_capacity.0 < self.vision.max_decoded_bytes as u64 {
            return Err(ConfigError::Invalid(
                "vision retention capacity is smaller than one decoded image limit".into(),
            ));
        }
        Ok(self)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn byte_sizes_require_units_and_allow_fractions() {
        assert_eq!(
            "1.5 GiB".parse::<ByteSize>().unwrap(),
            ByteSize(1_610_612_736)
        );
        assert!("1024".parse::<ByteSize>().is_err());
    }
    #[test]
    fn rejects_unknown_configuration_fields() {
        let yaml = "version: 1\nlisten: 127.0.0.1:8080\npaths: {database: db, models: models, spill: spill, user_models: local, user_config: user.yaml}\nexecution: {device_capacity: '1 GiB', host_capacity: '1 GiB', storage_capacity: '2 GiB', context_reserve: '1 GiB', surprise: true}\n";
        assert!(serde_yaml::from_str::<DaemonConfig>(yaml).is_err());
    }
    fn valid() -> DaemonConfig {
        DaemonConfig {
            version: 1,
            listen: default_listen(),
            unsafe_public_unauthenticated: false,
            bearer_token: None,
            http_debug: HttpDebugLevel::Off,
            paths: DataPaths {
                database: "db".into(),
                models: "models".into(),
                spill: "spill".into(),
                user_models: "local".into(),
                user_config: "user.yaml".into(),
            },
            execution: ExecutionConfig {
                device_capacity: ByteSize(1),
                host_capacity: ByteSize(1),
                storage_capacity: ByteSize(2),
                context_reserve: ByteSize(1),
                context_tokens: 1,
                gpu_layers: 0,
                require_competent: false,
            },
            server: ServerConfig::default(),
            scheduler: SchedulerPolicyConfig::default(),
            vision: VisionConfig::default(),
        }
    }
    #[test]
    fn validates_versions_capacities_and_vision_limits() {
        assert!(valid().validate().is_ok());
        let mut config = valid();
        config.version = 2;
        assert!(matches!(config.validate(), Err(ConfigError::Version(2))));
        for mutate in [
            (|c: &mut DaemonConfig| c.execution.device_capacity = ByteSize(0))
                as fn(&mut DaemonConfig),
            |c: &mut DaemonConfig| c.execution.host_capacity = ByteSize(0),
            |c: &mut DaemonConfig| c.execution.storage_capacity = ByteSize(0),
        ] {
            let mut config = valid();
            mutate(&mut config);
            assert!(matches!(config.validate(), Err(ConfigError::Invalid(_))));
        }
        let mut config = valid();
        config.execution.context_reserve = ByteSize(3);
        assert!(matches!(config.validate(), Err(ConfigError::Invalid(_))));
        let mut config = valid();
        config.execution.context_tokens = 0;
        assert!(matches!(config.validate(), Err(ConfigError::Invalid(_))));
        let mut config = valid();
        config.vision.max_images = 0;
        assert!(matches!(config.validate(), Err(ConfigError::Invalid(_))));
        let mut config = valid();
        config.vision.retention_capacity = ByteSize(1);
        assert!(matches!(config.validate(), Err(ConfigError::Invalid(_))));
    }
    #[test]
    fn loads_round_tripped_configuration_and_rejects_bad_sizes() {
        for (text, bytes) in [
            ("1 B", 1),
            ("1 KiB", 1024),
            ("1 MB", 1_000_000),
            ("1 TB", 1_000_000_000_000),
        ] {
            assert_eq!(text.parse::<ByteSize>().unwrap(), ByteSize(bytes));
        }
        for text in [
            "bad B",
            "-1 B",
            "NaN B",
            "1 XB",
            "999999999999999999999999999999999 TiB",
        ] {
            assert!(text.parse::<ByteSize>().is_err(), "{text}");
        }
        let path = std::env::temp_dir().join(format!("cusco-config-{}.yaml", uuid::Uuid::new_v4()));
        fs::write(&path, serde_yaml::to_string(&valid()).unwrap()).unwrap();
        assert_eq!(DaemonConfig::load(&path).unwrap().version, 1);
        fs::remove_file(path).unwrap();
        assert_eq!(serde_yaml::to_string(&ByteSize(3)).unwrap().trim(), "3 B");
    }
}
