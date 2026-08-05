use axum::{
    Json, Router,
    extract::{Path as AxumPath, State},
    http::{HeaderMap, StatusCode},
    response::{IntoResponse, Response, Sse, sse::Event},
    routing::{delete, get, post},
};
use futures_util::stream;
use parking_lot::Mutex;
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use std::{
    collections::HashMap,
    convert::Infallible,
    fs,
    io::Read,
    net::SocketAddr,
    path::{Path, PathBuf},
    sync::Arc,
    time::{Duration, Instant},
};
use thiserror::Error;
use uuid::Uuid;
mod mapped;

pub use mapped::{ExecutionProfile, MappedEngine, MappedMetrics};

#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(transparent)]
pub struct ContextId(String);
impl ContextId {
    fn new() -> Self {
        Self(Uuid::new_v4().to_string())
    }
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct RequestContext {
    pub principal: String,
    pub credential: String,
    pub request_id: String,
    pub scope: Scope,
}
#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum Scope {
    Inference,
    Admin,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct Usage {
    pub input_tokens: usize,
    pub generated_tokens: usize,
    pub evaluated_tokens: usize,
    pub cached_tokens: usize,
    pub model: String,
    pub model_revision: String,
    pub context_id: ContextId,
    pub latency_ms: u128,
    pub status: String,
}
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum StreamEvent {
    Started {
        request_id: String,
        context_id: ContextId,
    },
    Token {
        token: String,
        index: usize,
    },
    Usage {
        usage: Usage,
    },
    Finished {
        reason: String,
    },
    Error {
        message: String,
    },
}
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct InferRequest {
    pub model: String,
    pub prompt: String,
    #[serde(default = "default_tokens")]
    pub max_tokens: usize,
    #[serde(default)]
    pub context_id: Option<ContextId>,
    #[serde(default)]
    pub deadline_ms: Option<u64>,
    #[serde(default)]
    pub priority: i32,
}
fn default_tokens() -> usize {
    16
}
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct InferResponse {
    pub id: String,
    pub text: String,
    pub usage: Usage,
}
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct ContextRecord {
    pub id: ContextId,
    pub revision: u64,
    pub tokens: Vec<String>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub native_tokens: Vec<i32>,
}
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct ModelRecord {
    pub id: String,
    pub revision: String,
    pub path: PathBuf,
    pub sha256: String,
    pub aliases: Vec<String>,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
struct DurableState {
    installation: Uuid,
    contexts: HashMap<ContextId, ContextRecord>,
    models: HashMap<String, ModelRecord>,
}

#[derive(Debug, Error)]
pub enum Error {
    #[error("authentication required")]
    Unauthorized,
    #[error("admin scope required")]
    Forbidden,
    #[error("model not installed: {0}")]
    ModelNotFound(String),
    #[error("context not found")]
    ContextNotFound,
    #[error("deadline exceeded")]
    Deadline,
    #[error("request cancelled")]
    Cancelled,
    #[error("scheduler admission capacity exhausted")]
    Busy,
    #[error("unsafe unauthenticated listener: {0}")]
    UnsafeListener(SocketAddr),
    #[error("state error: {0}")]
    State(String),
}
impl IntoResponse for Error {
    fn into_response(self) -> Response {
        let status = match self {
            Self::Unauthorized => StatusCode::UNAUTHORIZED,
            Self::Forbidden => StatusCode::FORBIDDEN,
            Self::ModelNotFound(_) | Self::ContextNotFound => StatusCode::NOT_FOUND,
            Self::Deadline => StatusCode::REQUEST_TIMEOUT,
            Self::Busy => StatusCode::TOO_MANY_REQUESTS,
            Self::Cancelled => StatusCode::CONFLICT,
            _ => StatusCode::BAD_REQUEST,
        };
        (status, Json(json!({"error": self.to_string()}))).into_response()
    }
}

pub trait AuthProvider: Send + Sync {
    fn authenticate(&self, headers: &HeaderMap, scope: Scope) -> Result<RequestContext, Error>;
}
#[derive(Default)]
pub struct AnonymousAdmin;
impl AuthProvider for AnonymousAdmin {
    fn authenticate(&self, _: &HeaderMap, scope: Scope) -> Result<RequestContext, Error> {
        Ok(RequestContext {
            principal: "anonymous-admin".into(),
            credential: "anonymous".into(),
            request_id: Uuid::new_v4().to_string(),
            scope,
        })
    }
}
pub struct BearerAuth {
    token: String,
}
impl BearerAuth {
    pub fn new(token: impl Into<String>) -> Self {
        Self {
            token: token.into(),
        }
    }
}
impl AuthProvider for BearerAuth {
    fn authenticate(&self, headers: &HeaderMap, scope: Scope) -> Result<RequestContext, Error> {
        let supplied = headers.get("authorization").and_then(|v| v.to_str().ok());
        if supplied != Some(&format!("Bearer {}", self.token)) {
            return Err(Error::Unauthorized);
        }
        Ok(RequestContext {
            principal: "token-user".into(),
            credential: "bearer".into(),
            request_id: Uuid::new_v4().to_string(),
            scope,
        })
    }
}

pub struct EngineRequest<'a> {
    pub model: &'a ModelRecord,
    pub prompt: &'a str,
    pub max_tokens: usize,
    pub prior_tokens: &'a [i32],
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct EngineOutput {
    pub pieces: Vec<String>,
    pub successor_tokens: Vec<i32>,
    pub input_tokens: usize,
    pub cached_tokens: usize,
    pub evaluated_tokens: usize,
}

pub trait InferenceEngine: Send + Sync {
    fn generate(&self, request: EngineRequest<'_>) -> Result<EngineOutput, Error>;
}
#[derive(Default)]
pub struct DeterministicEngine;
impl InferenceEngine for DeterministicEngine {
    fn generate(&self, request: EngineRequest<'_>) -> Result<EngineOutput, Error> {
        let input_tokens = request.prompt.split_whitespace().count();
        let pieces = request
            .prompt
            .split_whitespace()
            .rev()
            .cycle()
            .take(request.max_tokens)
            .enumerate()
            .map(|(index, piece)| {
                if index == 0 {
                    piece.to_owned()
                } else {
                    format!(" {piece}")
                }
            })
            .collect();
        Ok(EngineOutput {
            pieces,
            successor_tokens: request.prior_tokens.to_vec(),
            input_tokens,
            cached_tokens: 0,
            evaluated_tokens: input_tokens,
        })
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SlotCandidate {
    pub slot: usize,
    pub valid_prefix: usize,
    pub transfer_bytes: usize,
    pub rollback_tokens: usize,
    pub quiesce_cost: usize,
    pub growth_bytes: usize,
    pub priority: i32,
    pub wait_ms: u64,
    pub decode_tokens: usize,
}
pub fn select_slot(candidates: &[SlotCandidate]) -> Option<usize> {
    candidates
        .iter()
        .min_by_key(|c| {
            let costs = c
                .transfer_bytes
                .saturating_add(c.rollback_tokens * 1024)
                .saturating_add(c.quiesce_cost)
                .saturating_add(c.growth_bytes);
            (
                costs.saturating_sub(c.valid_prefix * 1024),
                std::cmp::Reverse(c.priority),
                std::cmp::Reverse(c.wait_ms),
                c.decode_tokens,
                c.slot,
            )
        })
        .map(|c| c.slot)
}

struct Inner {
    durable: DurableState,
    cancelled: HashMap<String, bool>,
    active: usize,
    admission_limit: usize,
}
#[derive(Clone)]
pub struct Server {
    state_path: PathBuf,
    inner: Arc<Mutex<Inner>>,
    auth: Arc<dyn AuthProvider>,
    engine: Arc<dyn InferenceEngine>,
}
impl Server {
    pub fn open(
        path: impl AsRef<Path>,
        auth: Arc<dyn AuthProvider>,
        engine: Arc<dyn InferenceEngine>,
    ) -> Result<Self, Error> {
        let path = path.as_ref().to_owned();
        let durable = if path.exists() {
            serde_json::from_slice(&fs::read(&path).map_err(state_err)?).map_err(state_err)?
        } else {
            DurableState {
                installation: Uuid::new_v4(),
                contexts: HashMap::new(),
                models: HashMap::new(),
            }
        };
        let server = Self {
            state_path: path,
            inner: Arc::new(Mutex::new(Inner {
                durable,
                cancelled: HashMap::new(),
                active: 0,
                admission_limit: 4,
            })),
            auth,
            engine,
        };
        server.persist()?;
        Ok(server)
    }
    fn persist(&self) -> Result<(), Error> {
        let guard = self.inner.lock();
        let bytes = serde_json::to_vec_pretty(&guard.durable).map_err(state_err)?;
        if let Some(parent) = self.state_path.parent() {
            fs::create_dir_all(parent).map_err(state_err)?;
        }
        let tmp = self
            .state_path
            .with_extension(format!("tmp-{}", Uuid::new_v4()));
        fs::write(&tmp, bytes).map_err(state_err)?;
        fs::rename(tmp, &self.state_path).map_err(state_err)
    }
    fn authorize(&self, headers: &HeaderMap, scope: Scope) -> Result<RequestContext, Error> {
        self.auth.authenticate(headers, scope)
    }
    pub fn validate_listener(
        addr: SocketAddr,
        anonymous: bool,
        unsafe_public: bool,
    ) -> Result<(), Error> {
        if anonymous && !addr.ip().is_loopback() && !unsafe_public {
            Err(Error::UnsafeListener(addr))
        } else {
            Ok(())
        }
    }
    pub fn create_context(&self) -> Result<ContextRecord, Error> {
        let mut guard = self.inner.lock();
        let id = ContextId::new();
        let record = ContextRecord {
            id: id.clone(),
            revision: 0,
            tokens: vec![],
            native_tokens: vec![],
        };
        guard.durable.contexts.insert(id, record.clone());
        drop(guard);
        self.persist()?;
        Ok(record)
    }
    pub fn branch_context(&self, source: &ContextId) -> Result<ContextRecord, Error> {
        let mut record = self
            .inner
            .lock()
            .durable
            .contexts
            .get(source)
            .cloned()
            .ok_or(Error::ContextNotFound)?;
        record.id = ContextId::new();
        record.revision = 0;
        self.inner
            .lock()
            .durable
            .contexts
            .insert(record.id.clone(), record.clone());
        self.persist()?;
        Ok(record)
    }
    pub fn context(&self, id: &ContextId) -> Result<ContextRecord, Error> {
        self.inner
            .lock()
            .durable
            .contexts
            .get(id)
            .cloned()
            .ok_or(Error::ContextNotFound)
    }
    pub fn contexts(&self) -> Vec<ContextRecord> {
        self.inner
            .lock()
            .durable
            .contexts
            .values()
            .cloned()
            .collect()
    }
    pub fn import_context(&self, tokens: Vec<String>) -> Result<ContextRecord, Error> {
        let record = ContextRecord {
            id: ContextId::new(),
            revision: 0,
            tokens,
            native_tokens: vec![],
        };
        self.inner
            .lock()
            .durable
            .contexts
            .insert(record.id.clone(), record.clone());
        self.persist()?;
        Ok(record)
    }
    pub fn delete_context(&self, id: &ContextId) -> Result<(), Error> {
        self.inner
            .lock()
            .durable
            .contexts
            .remove(id)
            .ok_or(Error::ContextNotFound)?;
        self.persist()
    }
    pub fn register_model(&self, mut model: ModelRecord) -> Result<ModelRecord, Error> {
        model.aliases.sort();
        model.aliases.dedup();
        self.inner
            .lock()
            .durable
            .models
            .insert(model.id.clone(), model.clone());
        self.persist()?;
        Ok(model)
    }
    pub fn models(&self) -> Vec<ModelRecord> {
        self.inner.lock().durable.models.values().cloned().collect()
    }
    pub fn model(&self, id: &str) -> Result<ModelRecord, Error> {
        let guard = self.inner.lock();
        guard
            .durable
            .models
            .get(id)
            .or_else(|| {
                guard
                    .durable
                    .models
                    .values()
                    .find(|m| m.aliases.iter().any(|a| a == id))
            })
            .cloned()
            .ok_or_else(|| Error::ModelNotFound(id.into()))
    }
    pub fn alias_model(&self, id: &str, alias: String) -> Result<ModelRecord, Error> {
        let mut guard = self.inner.lock();
        let model = guard
            .durable
            .models
            .get_mut(id)
            .ok_or_else(|| Error::ModelNotFound(id.into()))?;
        if !model.aliases.contains(&alias) {
            model.aliases.push(alias);
        }
        let out = model.clone();
        drop(guard);
        self.persist()?;
        Ok(out)
    }
    pub fn remove_model(&self, id: &str) -> Result<(), Error> {
        self.inner
            .lock()
            .durable
            .models
            .remove(id)
            .ok_or_else(|| Error::ModelNotFound(id.into()))?;
        self.persist()
    }
    pub fn verify_model(&self, id: &str) -> Result<bool, Error> {
        let model = self.model(id)?;
        let mut file = fs::File::open(model.path).map_err(state_err)?;
        let mut digest = Sha256::new();
        let mut buffer = [0; 1024 * 1024];
        loop {
            let count = file.read(&mut buffer).map_err(state_err)?;
            if count == 0 {
                break;
            }
            digest.update(&buffer[..count]);
        }
        Ok(format!("{:x}", digest.finalize()) == model.sha256)
    }
    pub fn check_update(&self, id: &str, revision: &str) -> Result<bool, Error> {
        Ok(self.model(id)?.revision != revision)
    }
    pub fn cancel(&self, request: &str) {
        self.inner.lock().cancelled.insert(request.into(), true);
    }
    pub fn set_admission_limit(&self, limit: usize) {
        self.inner.lock().admission_limit = limit;
    }
    pub fn infer(
        &self,
        request_id: &str,
        req: InferRequest,
    ) -> Result<(InferResponse, Vec<StreamEvent>), Error> {
        {
            let mut guard = self.inner.lock();
            if guard.active >= guard.admission_limit {
                return Err(Error::Busy);
            }
            guard.active += 1;
        }
        let result = self.infer_admitted(request_id, req);
        self.inner.lock().active -= 1;
        result
    }
    fn infer_admitted(
        &self,
        request_id: &str,
        req: InferRequest,
    ) -> Result<(InferResponse, Vec<StreamEvent>), Error> {
        let started = Instant::now();
        if req.deadline_ms == Some(0) {
            return Err(Error::Deadline);
        }
        let model = self.model(&req.model)?;
        if self.inner.lock().cancelled.remove(request_id).is_some() {
            return Err(Error::Cancelled);
        }
        let context = match &req.context_id {
            Some(id) => Some(self.context(id)?),
            None => None,
        };
        let prior_tokens = context
            .as_ref()
            .map_or(&[][..], |record| record.native_tokens.as_slice());
        let generated = self.engine.generate(EngineRequest {
            model: &model,
            prompt: &req.prompt,
            max_tokens: req.max_tokens,
            prior_tokens,
        })?;
        if self.inner.lock().cancelled.remove(request_id).is_some() {
            return Err(Error::Cancelled);
        }
        if req
            .deadline_ms
            .is_some_and(|ms| started.elapsed() > Duration::from_millis(ms))
        {
            return Err(Error::Deadline);
        }
        let input: Vec<_> = req.prompt.split_whitespace().map(str::to_owned).collect();
        let EngineOutput {
            pieces: generated_pieces,
            successor_tokens,
            input_tokens,
            cached_tokens,
            evaluated_tokens,
        } = generated;
        let mut guard = self.inner.lock();
        let (context_id, revision) = if let Some(context) = context {
            let stored = guard
                .durable
                .contexts
                .get_mut(&context.id)
                .ok_or(Error::ContextNotFound)?;
            stored.tokens.extend(input.iter().cloned());
            stored.tokens.extend(generated_pieces.iter().cloned());
            stored.native_tokens = successor_tokens;
            stored.revision += 1;
            (stored.id.clone(), stored.revision)
        } else {
            let id = ContextId::new();
            let mut tokens = input.clone();
            tokens.extend(generated_pieces.iter().cloned());
            guard.durable.contexts.insert(
                id.clone(),
                ContextRecord {
                    id: id.clone(),
                    revision: 1,
                    tokens,
                    native_tokens: successor_tokens,
                },
            );
            (id, 1)
        };
        drop(guard);
        self.persist()?;
        let usage = Usage {
            input_tokens,
            generated_tokens: generated_pieces.len(),
            evaluated_tokens,
            cached_tokens,
            model: model.id,
            model_revision: model.revision,
            context_id: context_id.clone(),
            latency_ms: started.elapsed().as_millis(),
            status: "completed".into(),
        };
        let mut events = vec![StreamEvent::Started {
            request_id: request_id.into(),
            context_id,
        }];
        events.extend(generated_pieces.iter().enumerate().map(|(index, token)| {
            StreamEvent::Token {
                token: token.clone(),
                index,
            }
        }));
        events.push(StreamEvent::Usage {
            usage: usage.clone(),
        });
        events.push(StreamEvent::Finished {
            reason: format!("stop@revision-{revision}"),
        });
        Ok((
            InferResponse {
                id: request_id.into(),
                text: generated_pieces.concat(),
                usage,
            },
            events,
        ))
    }
}
fn state_err(error: impl std::fmt::Display) -> Error {
    Error::State(error.to_string())
}
#[cfg(test)]
fn hex_digest(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

#[derive(Clone, Deserialize)]
struct AliasRequest {
    alias: String,
}
#[derive(Clone, Deserialize)]
struct RegisterRequest {
    id: String,
    revision: String,
    path: PathBuf,
    sha256: String,
    #[serde(default)]
    aliases: Vec<String>,
}
#[derive(Clone, Deserialize)]
struct FetchRequest {
    uri: String,
    cache: PathBuf,
    #[serde(default)]
    sha256: Option<String>,
}
#[derive(Deserialize)]
struct ImportContextRequest {
    tokens: Vec<String>,
}
#[derive(Deserialize)]
struct CompletionRequest {
    model: String,
    prompt: String,
    #[serde(default)]
    max_tokens: Option<usize>,
    #[serde(default)]
    stream: bool,
    #[serde(default)]
    context_id: Option<ContextId>,
}
#[derive(Deserialize)]
struct ChatRequest {
    model: String,
    messages: Vec<ChatMessage>,
    #[serde(default)]
    max_tokens: Option<usize>,
    #[serde(default)]
    stream: bool,
    #[serde(default)]
    context_id: Option<ContextId>,
}
#[derive(Deserialize)]
struct ChatMessage {
    content: String,
}

pub fn router(server: Server) -> Router {
    Router::new()
        .route("/openapi.json", get(openapi))
        .route("/v1/completions", post(completion))
        .route("/v1/chat/completions", post(chat))
        .route("/v1/models", get(list_models))
        .route(
            "/native/models",
            get(list_native_models).post(register_model),
        )
        .route("/native/models/fetch", post(fetch_model))
        .route(
            "/native/models/{id}",
            get(inspect_model).delete(remove_model),
        )
        .route("/native/models/{id}/verify", post(verify_model))
        .route(
            "/native/models/{id}/check-update/{revision}",
            get(check_update),
        )
        .route("/native/models/{id}/aliases", post(alias_model))
        .route("/native/contexts", get(list_contexts).post(create_context))
        .route("/native/contexts/import", post(import_context))
        .route(
            "/native/contexts/{id}",
            get(get_context).delete(delete_context),
        )
        .route("/native/contexts/{id}/branches", post(branch_context))
        .route("/native/requests/{id}", delete(cancel_request))
        .with_state(server)
}
fn auth(server: &Server, headers: &HeaderMap, scope: Scope) -> Result<RequestContext, Error> {
    server.authorize(headers, scope)
}
async fn completion(
    State(s): State<Server>,
    headers: HeaderMap,
    Json(r): Json<CompletionRequest>,
) -> Result<Response, Error> {
    auth(&s, &headers, Scope::Inference)?;
    infer_response(&s, r.model, r.prompt, r.max_tokens, r.stream, r.context_id)
}
async fn chat(
    State(s): State<Server>,
    headers: HeaderMap,
    Json(r): Json<ChatRequest>,
) -> Result<Response, Error> {
    auth(&s, &headers, Scope::Inference)?;
    infer_response(
        &s,
        r.model,
        r.messages
            .into_iter()
            .map(|m| m.content)
            .collect::<Vec<_>>()
            .join("\n"),
        r.max_tokens,
        r.stream,
        r.context_id,
    )
}
fn infer_response(
    s: &Server,
    model: String,
    prompt: String,
    max_tokens: Option<usize>,
    streaming: bool,
    context_id: Option<ContextId>,
) -> Result<Response, Error> {
    let id = Uuid::new_v4().to_string();
    let (response, events) = s.infer(
        &id,
        InferRequest {
            model,
            prompt,
            max_tokens: max_tokens.unwrap_or_else(default_tokens),
            context_id,
            deadline_ms: None,
            priority: 0,
        },
    )?;
    if streaming {
        let rows = events
            .into_iter()
            .map(|event| Ok::<_, Infallible>(Event::default().json_data(event).unwrap()));
        Ok(Sse::new(stream::iter(rows)).into_response())
    } else {
        Ok(Json(json!({"id":response.id,"object":"text_completion","choices":[{"text":response.text}],"usage":response.usage})).into_response())
    }
}
async fn list_models(State(s): State<Server>, headers: HeaderMap) -> Result<Json<Value>, Error> {
    auth(&s, &headers, Scope::Inference)?;
    Ok(Json(json!({"data":s.models()})))
}
async fn list_native_models(
    State(s): State<Server>,
    headers: HeaderMap,
) -> Result<Json<Value>, Error> {
    auth(&s, &headers, Scope::Admin)?;
    Ok(Json(json!({"data":s.models()})))
}
async fn register_model(
    State(s): State<Server>,
    headers: HeaderMap,
    Json(r): Json<RegisterRequest>,
) -> Result<Json<ModelRecord>, Error> {
    auth(&s, &headers, Scope::Admin)?;
    Ok(Json(s.register_model(ModelRecord {
        id: r.id,
        revision: r.revision,
        path: r.path,
        sha256: r.sha256,
        aliases: r.aliases,
    })?))
}
async fn fetch_model(
    State(s): State<Server>,
    headers: HeaderMap,
    Json(r): Json<FetchRequest>,
) -> Result<Json<ModelRecord>, Error> {
    auth(&s, &headers, Scope::Admin)?;
    let fetched =
        cusco_model_registry::fetch_hf(&r.uri, &r.cache, r.sha256.as_deref()).map_err(state_err)?;
    let revision = r
        .uri
        .split_once('@')
        .and_then(|(_, value)| value.split_once('/'))
        .map(|(value, _)| value)
        .unwrap_or("unknown")
        .to_owned();
    Ok(Json(s.register_model(ModelRecord {
        id: fetched.identity,
        revision,
        path: fetched.path,
        sha256: fetched.sha256,
        aliases: vec![],
    })?))
}
async fn inspect_model(
    State(s): State<Server>,
    headers: HeaderMap,
    AxumPath(id): AxumPath<String>,
) -> Result<Json<ModelRecord>, Error> {
    auth(&s, &headers, Scope::Admin)?;
    Ok(Json(s.model(&id)?))
}
async fn verify_model(
    State(s): State<Server>,
    headers: HeaderMap,
    AxumPath(id): AxumPath<String>,
) -> Result<Json<Value>, Error> {
    auth(&s, &headers, Scope::Admin)?;
    Ok(Json(json!({"valid":s.verify_model(&id)?})))
}
async fn check_update(
    State(s): State<Server>,
    headers: HeaderMap,
    AxumPath((id, revision)): AxumPath<(String, String)>,
) -> Result<Json<Value>, Error> {
    auth(&s, &headers, Scope::Admin)?;
    Ok(Json(
        json!({"update_available":s.check_update(&id,&revision)?}),
    ))
}
async fn alias_model(
    State(s): State<Server>,
    headers: HeaderMap,
    AxumPath(id): AxumPath<String>,
    Json(r): Json<AliasRequest>,
) -> Result<Json<ModelRecord>, Error> {
    auth(&s, &headers, Scope::Admin)?;
    Ok(Json(s.alias_model(&id, r.alias)?))
}
async fn remove_model(
    State(s): State<Server>,
    headers: HeaderMap,
    AxumPath(id): AxumPath<String>,
) -> Result<StatusCode, Error> {
    auth(&s, &headers, Scope::Admin)?;
    s.remove_model(&id)?;
    Ok(StatusCode::NO_CONTENT)
}
async fn create_context(
    State(s): State<Server>,
    headers: HeaderMap,
) -> Result<Json<ContextRecord>, Error> {
    auth(&s, &headers, Scope::Inference)?;
    Ok(Json(s.create_context()?))
}
async fn list_contexts(
    State(s): State<Server>,
    headers: HeaderMap,
) -> Result<Json<Vec<ContextRecord>>, Error> {
    auth(&s, &headers, Scope::Inference)?;
    Ok(Json(s.contexts()))
}
async fn import_context(
    State(s): State<Server>,
    headers: HeaderMap,
    Json(r): Json<ImportContextRequest>,
) -> Result<Json<ContextRecord>, Error> {
    auth(&s, &headers, Scope::Inference)?;
    Ok(Json(s.import_context(r.tokens)?))
}
fn parse_context(id: String) -> ContextId {
    ContextId(id)
}
async fn get_context(
    State(s): State<Server>,
    headers: HeaderMap,
    AxumPath(id): AxumPath<String>,
) -> Result<Json<ContextRecord>, Error> {
    auth(&s, &headers, Scope::Inference)?;
    Ok(Json(s.context(&parse_context(id))?))
}
async fn branch_context(
    State(s): State<Server>,
    headers: HeaderMap,
    AxumPath(id): AxumPath<String>,
) -> Result<Json<ContextRecord>, Error> {
    auth(&s, &headers, Scope::Inference)?;
    Ok(Json(s.branch_context(&parse_context(id))?))
}
async fn delete_context(
    State(s): State<Server>,
    headers: HeaderMap,
    AxumPath(id): AxumPath<String>,
) -> Result<StatusCode, Error> {
    auth(&s, &headers, Scope::Inference)?;
    s.delete_context(&parse_context(id))?;
    Ok(StatusCode::NO_CONTENT)
}
async fn cancel_request(
    State(s): State<Server>,
    headers: HeaderMap,
    AxumPath(id): AxumPath<String>,
) -> Result<StatusCode, Error> {
    auth(&s, &headers, Scope::Inference)?;
    s.cancel(&id);
    Ok(StatusCode::ACCEPTED)
}
async fn openapi() -> Json<Value> {
    Json(openapi_document())
}
pub fn openapi_document() -> Value {
    json!({"openapi":"3.1.0","info":{"title":"Cusco API","version":"0.1.0"},"paths":{
        "/v1/completions":{"post":{}},"/v1/chat/completions":{"post":{}},"/v1/models":{"get":{}},
        "/native/models":{"get":{},"post":{}},"/native/models/fetch":{"post":{}},
        "/native/models/{id}":{"get":{},"delete":{}},"/native/models/{id}/verify":{"post":{}},
        "/native/models/{id}/check-update/{revision}":{"get":{}},"/native/models/{id}/aliases":{"post":{}},
        "/native/contexts":{"get":{},"post":{}},"/native/contexts/import":{"post":{}},
        "/native/contexts/{id}":{"get":{},"delete":{}},"/native/contexts/{id}/branches":{"post":{}},
        "/native/requests/{id}":{"delete":{}}
    }})
}

pub async fn serve(
    server: Server,
    addr: SocketAddr,
    anonymous: bool,
    unsafe_public: bool,
) -> Result<(), Error> {
    Server::validate_listener(addr, anonymous, unsafe_public)?;
    let listener = tokio::net::TcpListener::bind(addr)
        .await
        .map_err(state_err)?;
    axum::serve(listener, router(server))
        .await
        .map_err(state_err)
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::{
        body::{Body, to_bytes},
        http::Request,
    };
    use std::net::IpAddr;
    use tower::ServiceExt;
    fn dir() -> PathBuf {
        std::env::temp_dir().join(format!("cusco-server-{}", Uuid::new_v4()))
    }
    fn setup(auth: Arc<dyn AuthProvider>) -> (Server, PathBuf) {
        let d = dir();
        fs::create_dir_all(&d).unwrap();
        let model = d.join("m.gguf");
        fs::write(&model, b"model").unwrap();
        let s = Server::open(d.join("state.json"), auth, Arc::new(DeterministicEngine)).unwrap();
        s.register_model(ModelRecord {
            id: "m".into(),
            revision: "r1".into(),
            path: model,
            sha256: hex_digest(b"model"),
            aliases: vec!["latest".into()],
        })
        .unwrap();
        (s, d)
    }

    struct FailingEngine;
    impl InferenceEngine for FailingEngine {
        fn generate(&self, _: EngineRequest<'_>) -> Result<EngineOutput, Error> {
            Err(Error::State("generation failed".into()))
        }
    }
    #[test]
    fn persistence_branching_and_ids_survive_restart() {
        let (s, d) = setup(Arc::new(AnonymousAdmin));
        let a = s.create_context().unwrap();
        let b = s.branch_context(&a.id).unwrap();
        assert_ne!(a.id, b.id);
        drop(s);
        let s = Server::open(
            d.join("state.json"),
            Arc::new(AnonymousAdmin),
            Arc::new(DeterministicEngine),
        )
        .unwrap();
        assert_eq!(s.context(&a.id).unwrap(), a);
        let c = s.create_context().unwrap();
        assert_ne!(a.id, c.id);
        assert_ne!(b.id, c.id);
        fs::remove_dir_all(d).unwrap()
    }
    #[test]
    fn inference_usage_events_deadline_and_cancel_are_transactional() {
        let (s, d) = setup(Arc::new(AnonymousAdmin));
        let req = InferRequest {
            model: "latest".into(),
            prompt: "one two".into(),
            max_tokens: 2,
            context_id: None,
            deadline_ms: Some(1000),
            priority: 2,
        };
        let (out, events) = s.infer("r", req).unwrap();
        assert_eq!(out.text, "two one");
        assert_eq!(out.usage.input_tokens, 2);
        assert_eq!(events.len(), 5);
        let before = s.context(&out.usage.context_id).unwrap();
        s.cancel("cancelled");
        let err = s
            .infer(
                "cancelled",
                InferRequest {
                    model: "m".into(),
                    prompt: "x".into(),
                    max_tokens: 1,
                    context_id: Some(before.id.clone()),
                    deadline_ms: None,
                    priority: 0,
                },
            )
            .unwrap_err();
        assert!(matches!(err, Error::Cancelled));
        assert_eq!(s.context(&before.id).unwrap(), before);
        assert!(matches!(
            s.infer(
                "late",
                InferRequest {
                    model: "m".into(),
                    prompt: "x".into(),
                    max_tokens: 1,
                    context_id: Some(before.id),
                    deadline_ms: Some(0),
                    priority: 0
                }
            ),
            Err(Error::Deadline)
        ));
        s.set_admission_limit(0);
        assert!(matches!(
            s.infer(
                "busy",
                InferRequest {
                    model: "m".into(),
                    prompt: "x".into(),
                    max_tokens: 1,
                    context_id: None,
                    deadline_ms: None,
                    priority: 0,
                },
            ),
            Err(Error::Busy)
        ));
        let contexts_before = s.contexts();
        drop(s);
        let failing = Server::open(
            d.join("state.json"),
            Arc::new(AnonymousAdmin),
            Arc::new(FailingEngine),
        )
        .unwrap();
        assert!(matches!(
            failing.infer(
                "failed",
                InferRequest {
                    model: "m".into(),
                    prompt: "x".into(),
                    max_tokens: 1,
                    context_id: None,
                    deadline_ms: None,
                    priority: 0,
                },
            ),
            Err(Error::State(_))
        ));
        assert_eq!(failing.contexts(), contexts_before);
        fs::remove_dir_all(d).unwrap()
    }
    #[test]
    fn model_lifecycle_and_listener_policy() {
        let (s, d) = setup(Arc::new(AnonymousAdmin));
        assert!(s.verify_model("m").unwrap());
        assert!(!s.check_update("m", "r1").unwrap());
        assert!(s.check_update("m", "r2").unwrap());
        s.alias_model("m", "stable".into()).unwrap();
        assert_eq!(s.model("stable").unwrap().id, "m");
        s.remove_model("m").unwrap();
        assert!(matches!(s.model("m"), Err(Error::ModelNotFound(_))));
        let public = SocketAddr::new(IpAddr::from([0, 0, 0, 0]), 8080);
        assert!(matches!(
            Server::validate_listener(public, true, false),
            Err(Error::UnsafeListener(_))
        ));
        assert!(Server::validate_listener(public, true, true).is_ok());
        fs::remove_dir_all(d).unwrap()
    }
    #[test]
    fn scheduler_uses_transition_cost_priority_and_wait() {
        let c = vec![
            SlotCandidate {
                slot: 1,
                valid_prefix: 1,
                transfer_bytes: 5000,
                rollback_tokens: 0,
                quiesce_cost: 0,
                growth_bytes: 0,
                priority: 0,
                wait_ms: 0,
                decode_tokens: 2,
            },
            SlotCandidate {
                slot: 2,
                valid_prefix: 3,
                transfer_bytes: 0,
                rollback_tokens: 0,
                quiesce_cost: 0,
                growth_bytes: 0,
                priority: 0,
                wait_ms: 10,
                decode_tokens: 2,
            },
        ];
        assert_eq!(select_slot(&c), Some(2));
        let mut low = c[1].clone();
        low.priority = i32::MIN;
        let mut high = low.clone();
        high.slot = 3;
        high.priority = i32::MAX;
        assert_eq!(select_slot(&[low, high]), Some(3));
        assert_eq!(select_slot(&[]), None)
    }
    #[tokio::test]
    async fn http_auth_openapi_completion_and_context_api() {
        let (s, d) = setup(Arc::new(BearerAuth::new("secret")));
        let app = router(s);
        let denied = app
            .clone()
            .oneshot(
                Request::builder()
                    .method("GET")
                    .uri("/v1/models")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(denied.status(), StatusCode::UNAUTHORIZED);
        let req = Request::builder()
            .method("POST")
            .uri("/v1/completions")
            .header("authorization", "Bearer secret")
            .header("content-type", "application/json")
            .body(Body::from(
                r#"{"model":"m","prompt":"hello world","max_tokens":2}"#,
            ))
            .unwrap();
        let response = app.clone().oneshot(req).await.unwrap();
        assert_eq!(response.status(), StatusCode::OK);
        let body: Value =
            serde_json::from_slice(&to_bytes(response.into_body(), usize::MAX).await.unwrap())
                .unwrap();
        assert_eq!(body["choices"][0]["text"], "world hello");
        let context_id = body["usage"]["context_id"].as_str().unwrap();
        let continuation = Request::builder()
            .method("POST")
            .uri("/v1/completions")
            .header("authorization", "Bearer secret")
            .header("content-type", "application/json")
            .body(Body::from(
                json!({
                    "model": "m",
                    "prompt": "again",
                    "max_tokens": 1,
                    "context_id": context_id
                })
                .to_string(),
            ))
            .unwrap();
        let continuation = app.clone().oneshot(continuation).await.unwrap();
        assert_eq!(continuation.status(), StatusCode::OK);
        let continuation: Value = serde_json::from_slice(
            &to_bytes(continuation.into_body(), usize::MAX)
                .await
                .unwrap(),
        )
        .unwrap();
        assert_eq!(continuation["usage"]["context_id"], context_id);
        let spec = openapi_document();
        assert_eq!(spec["openapi"], "3.1.0");
        assert!(spec["paths"]["/v1/chat/completions"].is_object());
        fs::remove_dir_all(d).unwrap()
    }

    fn request(method: &str, uri: &str, body: Value) -> Request<Body> {
        Request::builder()
            .method(method)
            .uri(uri)
            .header("content-type", "application/json")
            .body(Body::from(body.to_string()))
            .unwrap()
    }

    #[tokio::test]
    async fn native_lifecycle_chat_streaming_and_context_routes() {
        let (s, d) = setup(Arc::new(AnonymousAdmin));
        let app = router(s);
        let second = d.join("second.gguf");
        fs::write(&second, b"second").unwrap();
        let register = json!({
            "id":"second",
            "revision":"r2",
            "path":second,
            "sha256":hex_digest(b"second")
        });
        for (method, uri, body, expected) in [
            ("POST", "/native/models", register, StatusCode::OK),
            ("GET", "/native/models", json!(null), StatusCode::OK),
            ("GET", "/native/models/second", json!(null), StatusCode::OK),
            (
                "POST",
                "/native/models/second/aliases",
                json!({"alias":"stable"}),
                StatusCode::OK,
            ),
            (
                "GET",
                "/native/models/second/check-update/r1",
                json!(null),
                StatusCode::OK,
            ),
            (
                "POST",
                "/native/models/second/verify",
                json!(null),
                StatusCode::OK,
            ),
        ] {
            assert_eq!(
                app.clone()
                    .oneshot(request(method, uri, body))
                    .await
                    .unwrap()
                    .status(),
                expected
            );
        }
        assert_eq!(
            app.clone()
                .oneshot(request("GET", "/native/contexts", json!(null)))
                .await
                .unwrap()
                .status(),
            StatusCode::OK
        );
        let imported = app
            .clone()
            .oneshot(request(
                "POST",
                "/native/contexts/import",
                json!({"tokens":["durable","state"]}),
            ))
            .await
            .unwrap();
        let imported: ContextRecord =
            serde_json::from_slice(&to_bytes(imported.into_body(), usize::MAX).await.unwrap())
                .unwrap();
        assert_eq!(imported.tokens, vec!["durable", "state"]);
        let created = app
            .clone()
            .oneshot(request("POST", "/native/contexts", json!(null)))
            .await
            .unwrap();
        let created: ContextRecord =
            serde_json::from_slice(&to_bytes(created.into_body(), usize::MAX).await.unwrap())
                .unwrap();
        let context_uri = format!("/native/contexts/{}", created.id.0);
        assert_eq!(
            app.clone()
                .oneshot(request("GET", &context_uri, json!(null)))
                .await
                .unwrap()
                .status(),
            StatusCode::OK
        );
        let branch_uri = format!("{context_uri}/branches");
        assert_eq!(
            app.clone()
                .oneshot(request("POST", &branch_uri, json!(null)))
                .await
                .unwrap()
                .status(),
            StatusCode::OK
        );
        assert_eq!(
            app.clone()
                .oneshot(request("DELETE", &context_uri, json!(null)))
                .await
                .unwrap()
                .status(),
            StatusCode::NO_CONTENT
        );
        assert_eq!(
            app.clone()
                .oneshot(request("DELETE", "/native/requests/pending", json!(null)))
                .await
                .unwrap()
                .status(),
            StatusCode::ACCEPTED
        );
        let chat = json!({"model":"m","messages":[{"content":"hello world"}],"max_tokens":2,"stream":true});
        let streamed = app
            .clone()
            .oneshot(request("POST", "/v1/chat/completions", chat))
            .await
            .unwrap();
        assert_eq!(streamed.status(), StatusCode::OK);
        assert!(
            streamed.headers()["content-type"]
                .to_str()
                .unwrap()
                .starts_with("text/event-stream")
        );
        assert_eq!(
            app.clone()
                .oneshot(request("DELETE", "/native/models/second", json!(null)))
                .await
                .unwrap()
                .status(),
            StatusCode::NO_CONTENT
        );
        fs::remove_dir_all(d).unwrap();
    }
}
