use crate::{
    DaemonConfig, Error, ModelCatalog, ModelRecord, ResidencyConfig, ResidentEngine,
    SearxngHostedTools, Server, WebSearchPolicy, WorkloadScheduler, load_user_models,
};
use std::{fs, sync::Arc, time::Duration};
use thiserror::Error as ThisError;

/// Owns configured Cusco backend services for composition into a host application.
///
/// This retained Rust boundary integrates inference, scheduling, and storage
/// in-process. HTTP routing and protocol representation belong to Terrane.
#[derive(Clone)]
pub struct Daemon {
    server: Server,
    configuration: DaemonConfig,
    shutdown_grace: Duration,
    shutdown_watchdog: Arc<ShutdownWatchdog>,
}

#[derive(Default)]
struct ShutdownWatchdog(parking_lot::Mutex<Option<tokio::task::JoinHandle<()>>>);

impl Drop for ShutdownWatchdog {
    fn drop(&mut self) {
        if let Some(watchdog) = self.0.get_mut().take() {
            watchdog.abort();
        }
    }
}

#[derive(Debug, ThisError)]
pub enum DaemonError {
    #[error(transparent)]
    Config(#[from] crate::config::ConfigError),
    #[error(transparent)]
    Catalog(#[from] crate::catalog::CatalogError),
    #[error(transparent)]
    HostedTools(#[from] crate::HostedToolError),
    #[error(transparent)]
    Executor(#[from] cusco_executor::Error),
    #[error(transparent)]
    Registry(#[from] cusco_model_registry::Error),

    #[error(transparent)]
    Server(#[from] Error),
    #[error("daemon startup: {0}")]
    Startup(String),
}

impl Daemon {
    pub fn open(config_path: &str) -> Result<Self, DaemonError> {
        let config = DaemonConfig::load(config_path)?;
        cusco_executor::set_debug_logging(config.native_debug);
        let catalog = ModelCatalog::open(&config.paths.database)?;
        let engine = ResidentEngine::open_with_spill(
            ResidencyConfig {
                device_bytes: config.execution.device_capacity.0,
                host_bytes: config.execution.host_capacity.0,
                storage_bytes: config.execution.storage_capacity.0,
                context_reserve_bytes: config.execution.context_reserve.0,
                n_ctx: config.execution.context_tokens,
                publication_interval_tokens: config.execution.publication_interval_tokens,
                gpu_layers: config.execution.gpu_layers,
                require_competent: config.execution.require_competent,
            },
            &config.paths.spill,
        )
        .map_err(|error| DaemonError::Startup(error.to_string()))?;
        engine.attach_catalog(catalog.clone());
        let engine = WorkloadScheduler::new(engine, config.scheduler)?;
        let transient_state = config.paths.database.with_extension("runtime.json");
        if transient_state.exists() {
            fs::remove_file(&transient_state).map_err(|error| {
                DaemonError::Startup(format!(
                    "discard transient state {}: {error}",
                    transient_state.display()
                ))
            })?;
        }
        let server = Server::open(&transient_state, engine)?;
        server.configure(config.server)?;
        server.configure_vision(config.vision.clone());
        if config.hosted_tools.web_search.enabled {
            let search = &config.hosted_tools.web_search;
            server.configure_hosted_tools(Arc::new(SearxngHostedTools::new(WebSearchPolicy {
                endpoint: search
                    .endpoint
                    .clone()
                    .expect("validated enabled web-search endpoint"),
                timeout: Duration::from_millis(search.timeout_ms),
                max_results: search.max_results,
                max_response_bytes: search.max_response_bytes,
                allowed_domains: search.allowed_domains.clone(),
            })?));
        }
        server.attach_catalog(catalog.clone(), config.paths.models.clone());
        for model in catalog.models()? {
            server.register_catalog_model(model)?;
        }
        let declared = load_user_models(&config.paths.user_config, &config.paths.user_models)
            .map_err(|error| DaemonError::Startup(error.to_string()))?;
        for declaration in declared.models {
            let registered = cusco_model_registry::register_local(
                &declaration.path,
                &declaration.name,
                declaration.sha256.as_deref(),
            )
            .map_err(|error| DaemonError::Startup(error.to_string()))?;
            let metadata = cusco_model_registry::probe_gguf(&registered.path)
                .map_err(|error| DaemonError::Startup(error.to_string()))?;
            server.register_model_with(
                |model| {
                    catalog
                        .publish(model)
                        .map_err(|error| Error::State(error.to_string()))
                },
                ModelRecord {
                    id: declaration.name,
                    revision: registered.sha256.clone(),
                    path: registered.path,
                    sha256: registered.sha256,
                    aliases: declaration.aliases,
                    family: metadata.architecture,
                    size_bytes: registered.size,
                    block_count: metadata.block_count.unwrap_or(0),
                    epoch: 0,
                },
            )?;
        }
        Ok(Self {
            server,
            shutdown_grace: Duration::from_millis(config.server.shutdown_grace_ms),
            shutdown_watchdog: Arc::default(),
            configuration: config,
        })
    }

    pub fn configuration(&self) -> DaemonConfig {
        self.configuration.clone()
    }
    pub fn server(&self) -> Server {
        self.server.clone()
    }

    /// Wait for an OS shutdown signal, stop admission, and arrange bounded cancellation.
    pub async fn shutdown(&self) -> Result<(), DaemonError> {
        shutdown_signal().await;
        self.server.begin_shutdown();
        let server = self.server.clone();
        let grace = self.shutdown_grace;
        let watchdog = tokio::spawn(async move {
            tokio::time::sleep(grace).await;
            server.cancel_remaining();
        });
        if let Some(previous) = self.shutdown_watchdog.0.lock().replace(watchdog) {
            previous.abort();
        }
        Ok(())
    }

    /// Release the grace timer after the host has finished draining native work.
    pub fn finish_shutdown(&self) {
        if let Some(watchdog) = self.shutdown_watchdog.0.lock().take() {
            watchdog.abort();
        }
    }
}

async fn shutdown_signal() {
    #[cfg(unix)]
    {
        let mut terminate =
            tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate())
                .expect("install SIGTERM handler");
        tokio::select! {
            _ = tokio::signal::ctrl_c() => {}
            _ = terminate.recv() => {}
        }
    }
    #[cfg(not(unix))]
    {
        let _ = tokio::signal::ctrl_c().await;
    }
}
#[cfg(test)]
mod tests {
    use super::*;

    fn configured_file() -> (std::path::PathBuf, String) {
        let root = std::env::temp_dir().join(format!("cusco-daemon-{}", uuid::Uuid::new_v4()));
        fs::create_dir_all(root.join("db")).unwrap();
        fs::create_dir_all(root.join("models")).unwrap();
        fs::create_dir_all(root.join("spill")).unwrap();
        fs::create_dir_all(root.join("user-models")).unwrap();
        fs::write(root.join("user.yaml"), "models: []\n").unwrap();
        let mut config = include_str!("../../../config/test.yaml").to_owned();
        for (from, to) in [
            ("/data/db/catalog.sqlite", root.join("db/catalog.sqlite")),
            ("/data/models", root.join("models")),
            ("/data/spill", root.join("spill")),
            ("/data/user-models", root.join("user-models")),
            ("/data/user.yaml", root.join("user.yaml")),
        ] {
            config = config.replace(from, &to.display().to_string());
        }
        let path = root.join("config.yaml");
        fs::write(&path, config).unwrap();
        (root, path.display().to_string())
    }

    #[test]
    fn opens_configured_services_and_exposes_configuration() {
        let (root, path) = configured_file();
        let daemon = Daemon::open(&path).unwrap();
        assert_eq!(daemon.configuration().server.active_requests, 4);
        assert!(daemon.server().models().is_empty());
        daemon.finish_shutdown();
        drop(daemon);
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn configured_local_models_are_published_and_restored_on_restart() {
        let (root, path) = configured_file();
        let model = root.join("user-models/local.gguf");
        let mut bytes = Vec::new();
        bytes.extend_from_slice(b"GGUF");
        bytes.extend_from_slice(&3_u32.to_le_bytes());
        bytes.extend_from_slice(&0_u64.to_le_bytes());
        bytes.extend_from_slice(&2_u64.to_le_bytes());
        bytes.extend_from_slice(&20_u64.to_le_bytes());
        bytes.extend_from_slice(b"general.architecture");
        bytes.extend_from_slice(&8_u32.to_le_bytes());
        bytes.extend_from_slice(&6_u64.to_le_bytes());
        bytes.extend_from_slice(b"gemma4");
        bytes.extend_from_slice(&18_u64.to_le_bytes());
        bytes.extend_from_slice(b"gemma4.block_count");
        bytes.extend_from_slice(&4_u32.to_le_bytes());
        bytes.extend_from_slice(&62_u32.to_le_bytes());
        fs::write(&model, bytes).unwrap();
        fs::write(
            root.join("user.yaml"),
            "models:\n  - name: local\n    path: local.gguf\n    aliases: [local-alias]\n",
        )
        .unwrap();

        let daemon = Daemon::open(&path).unwrap();
        let registered = daemon.server().models().pop().unwrap();
        assert_eq!(registered.id, "local");
        assert_eq!(registered.path, model);
        assert_eq!(registered.aliases, ["local-alias"]);
        assert_eq!(registered.family, "gemma4");
        assert_eq!(registered.block_count, 62);
        drop(daemon);

        let restarted = Daemon::open(&path).unwrap();
        let restored = restarted.server().models();
        assert_eq!(restored.len(), 1);
        assert_eq!(restored[0].sha256, registered.sha256);
        assert_eq!(restored[0].aliases, registered.aliases);
        drop(restarted);
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn startup_reports_unremovable_transient_state() {
        let (root, path) = configured_file();
        let transient = root.join("db/catalog.runtime.json");
        fs::create_dir_all(&transient).unwrap();
        assert!(matches!(
            Daemon::open(&path),
            Err(DaemonError::Startup(message)) if message.contains("discard transient state")
        ));
        fs::remove_dir_all(root).unwrap();
    }

    #[tokio::test]
    async fn shutdown_watchdog_drop_aborts_owned_task() {
        let (started_sender, started_receiver) = tokio::sync::oneshot::channel();
        let (dropped_sender, dropped_receiver) = tokio::sync::oneshot::channel::<()>();
        let watchdog = ShutdownWatchdog(parking_lot::Mutex::new(Some(tokio::spawn(async move {
            let _sender = dropped_sender;
            let _ = started_sender.send(());
            std::future::pending::<()>().await;
        }))));
        started_receiver.await.unwrap();
        drop(watchdog);
        assert!(
            tokio::time::timeout(Duration::from_secs(2), dropped_receiver)
                .await
                .unwrap()
                .is_err()
        );
    }
}
