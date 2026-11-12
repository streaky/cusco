use crate::{
    AnonymousAdmin, AuthProvider, BearerAuth, DaemonConfig, Error, HttpDebug, HttpDebugLevel,
    ModelCatalog, ModelRecord, ResidencyConfig, ResidentEngine, SearxngHostedTools, Server,
    WebSearchPolicy, WorkloadScheduler, load_user_models, router, router_with_http_debug,
};
use axum::Router;
use std::{fs, sync::Arc, time::Duration};
use thiserror::Error as ThisError;

/// Owns configured Cusco backend services for composition into a host application.
///
/// This retained Rust boundary integrates inference, scheduling, protocol routing, and
/// storage in-process; it is not a workaround for a Terrane compiler/framework gap.
#[derive(Clone)]
pub struct Daemon {
    server: Server,
    address: String,
    router: Router,
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
    pub fn open(config_path: &str, http_debug: &str) -> Result<Self, DaemonError> {
        let config = DaemonConfig::load(config_path)?;
        let cli_debug = if http_debug.is_empty() {
            None
        } else {
            Some(
                http_debug
                    .parse::<HttpDebugLevel>()
                    .map_err(DaemonError::Startup)?,
            )
        };
        let http_debug = config
            .resolve_http_debug(cli_debug, std::env::var("CUSCO_HTTP_DEBUG").ok().as_deref())?;
        cusco_executor::set_debug_logging(config.native_debug);
        let bearer_token = std::env::var("CUSCO_BEARER_TOKEN")
            .ok()
            .or_else(|| config.bearer_token.clone());
        Server::validate_listener(
            config.listen,
            bearer_token.is_none(),
            config.unsafe_public_unauthenticated,
        )?;
        let auth: Arc<dyn AuthProvider> = match bearer_token {
            Some(token) => Arc::new(BearerAuth::new(token)),
            None => Arc::new(AnonymousAdmin),
        };
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
        let server = Server::open(&transient_state, auth, engine)?;
        server.configure(config.server)?;
        server.configure_vision(config.vision);
        server.configure_openapi(config.openapi);
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
        let address = config.listen.to_string();
        let router = if http_debug == HttpDebugLevel::Off {
            router(server.clone())
        } else {
            router_with_http_debug(server.clone(), HttpDebug::stderr(http_debug))
        };
        Ok(Self {
            server,
            address,
            router,
            shutdown_grace: Duration::from_millis(config.server.shutdown_grace_ms),
            shutdown_watchdog: Arc::default(),
        })
    }

    pub fn address(&self) -> String {
        self.address.clone()
    }

    pub fn router(&self) -> Router {
        self.router.clone()
    }

    /// Wait for an OS shutdown signal, stop admission, and arrange bounded cancellation.
    /// Returns after initiating host-managed HTTP graceful shutdown.
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

    /// Release the grace timer only after the host has drained HTTP responses.
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
