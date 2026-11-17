use std::{fs, path::PathBuf, sync::Arc, thread, time::Duration};

use cusco_server::{
    DaemonConfig, DeterministicEngine, EngineOutput, EngineRequest, Error, ExecutionSession,
    HostedToolError, HostedToolExecutor, InferenceEngine, ModelCatalog, ModelRecord,
    PrefillMetrics, QuantumKind, QuantumObservation, RequestControl, Server, SessionStep,
    WebSearchRequest, WebSearchResult,
};

struct FixtureEngine {
    deterministic: DeterministicEngine,
    gate_directory: PathBuf,
}

struct FixtureSession {
    inner: Box<dyn ExecutionSession>,
    gate: Option<(PathBuf, PathBuf)>,
    control: Arc<RequestControl>,
}

struct ReplySession {
    request: EngineRequest,
    reply: Option<String>,
}

impl ExecutionSession for ReplySession {
    fn step(&mut self) -> Result<SessionStep, Error> {
        self.request.control.check()?;
        if let Some(reply) = self.reply.take() {
            return Ok(SessionStep::Token {
                id: -1,
                piece: reply.into_bytes(),
                terminal_or_control: false,
                observation: QuantumObservation {
                    kind: QuantumKind::Decode,
                    charged_tokens: 1,
                    context_placement: "model_free".into(),
                    executor_slot_occupied: false,
                    transition_cost_bytes: 0,
                    capacity_reserved_bytes: 0,
                },
            });
        }
        self.finish().map(SessionStep::Finished)
    }

    fn finish(&mut self) -> Result<EngineOutput, Error> {
        self.request.control.check()?;
        let input_tokens = self.request.prompt.split_whitespace().count();
        Ok(EngineOutput {
            successor_tokens: self.request.prior_tokens.clone(),
            input_tokens,
            cached_tokens: 0,
            evaluated_tokens: input_tokens,
            prefill: PrefillMetrics {
                total_tokens: input_tokens,
                uncached_tokens: input_tokens,
                ..Default::default()
            },
        })
    }
}

struct FixtureHostedTools;

impl HostedToolExecutor for FixtureHostedTools {
    fn web_search(
        &self,
        request: &WebSearchRequest,
    ) -> Result<Vec<WebSearchResult>, HostedToolError> {
        assert_eq!(request.query, "rust ownership");
        Ok(vec![WebSearchResult {
            title: "Rust ownership".into(),
            url: "https://example.com/result".into(),
            snippet: "Ownership documentation".into(),
        }])
    }
}

impl InferenceEngine for FixtureEngine {
    fn start_session(&self, request: EngineRequest) -> Result<Box<dyn ExecutionSession>, Error> {
        let gate_id = request
            .prompt
            .split_once("[[gate:")
            .and_then(|(_, tail)| tail.split_once("]]"))
            .map(|(id, _)| id.to_owned())
            .filter(|id| {
                !id.is_empty()
                    && id
                        .chars()
                        .all(|c| c.is_ascii_alphanumeric() || c == '-' || c == '_')
            });
        let control = Arc::clone(&request.control);
        let reply = if request.prompt.contains("Output only JSON") {
            Some(r#"{"use_search":true,"query":"rust ownership"}"#.to_owned())
        } else if request.prompt.contains("Hosted web search completed") {
            Some("Answer https://example.com/result".to_owned())
        } else if request.grammar.is_some() {
            Some("{}".to_owned())
        } else {
            None
        };
        let inner: Box<dyn ExecutionSession> = if let Some(reply) = reply {
            Box::new(ReplySession {
                request,
                reply: Some(reply),
            })
        } else {
            self.deterministic.start_session(request)?
        };
        let gate = gate_id.map(|id| {
            (
                self.gate_directory.join(format!("{id}.started")),
                self.gate_directory.join(format!("{id}.release")),
            )
        });
        Ok(Box::new(FixtureSession {
            inner,
            gate,
            control,
        }))
    }
}

impl ExecutionSession for FixtureSession {
    fn step(&mut self) -> Result<SessionStep, Error> {
        if let Some((started, release)) = self.gate.take() {
            fs::write(&started, b"ready").map_err(|error| Error::State(error.to_string()))?;
            while !release.exists() {
                if let Err(error) = self.control.check() {
                    let _ = fs::write(started.with_extension("cancelled"), b"cancelled");
                    return Err(error);
                }
                thread::sleep(Duration::from_millis(2));
            }
        }
        self.inner.step()
    }

    fn finish(&mut self) -> Result<EngineOutput, Error> {
        self.inner.finish()
    }
}

pub fn open(config_path: &str) -> Result<(Server, DaemonConfig), String> {
    let settings = DaemonConfig::load(config_path).map_err(|error| error.to_string())?;
    fs::create_dir_all(&settings.paths.models).map_err(|error| error.to_string())?;
    fs::create_dir_all(&settings.paths.spill).map_err(|error| error.to_string())?;
    let gate_directory = settings.paths.database.with_extension("fixture-gates");
    fs::create_dir_all(&gate_directory).map_err(|error| error.to_string())?;

    let model_path = settings.paths.models.join("deterministic.gguf");
    fs::write(&model_path, b"deterministic fixture model").map_err(|error| error.to_string())?;
    let state_path = settings
        .paths
        .database
        .with_extension("fixture-runtime.json");
    let server = Server::open_with_engine(
        state_path
            .to_str()
            .ok_or_else(|| "state path is not Unicode".to_owned())?,
        Arc::new(FixtureEngine {
            deterministic: DeterministicEngine,
            gate_directory,
        }),
    )
    .map_err(|error| error.to_string())?;
    server
        .configure(settings.server.clone())
        .map_err(|error| error.to_string())?;
    server.configure_vision(settings.vision.clone());
    server.configure_hosted_tools(Arc::new(FixtureHostedTools));

    let catalog =
        ModelCatalog::open(&settings.paths.database).map_err(|error| error.to_string())?;
    server.attach_catalog(catalog, PathBuf::from(&settings.paths.models));
    server
        .register_model(ModelRecord {
            id: "m".into(),
            revision: "fixture-r1".into(),
            path: model_path,
            sha256: "deterministic-fixture".into(),
            aliases: vec!["latest".into()],
            family: "gemma4".into(),
            size_bytes: 27,
            block_count: 1,
            epoch: 0,
        })
        .map_err(|error| error.to_string())?;

    Ok((server, settings))
}
