use axum::{
    Json, Router,
    body::{Body, to_bytes},
    extract::{FromRequest, Path as AxumPath, Request, State},
    http::{HeaderMap, HeaderValue, StatusCode, header::RETRY_AFTER},
    middleware::{self, Next},
    response::{IntoResponse, Response, Sse, sse::Event},
    routing::{delete, get, post},
};
use futures_util::{StreamExt, stream};
use http_body_util::BodyExt;
use parking_lot::Mutex;
use serde::{Deserialize, Serialize, de::DeserializeOwned};
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use std::{
    collections::{HashMap, HashSet, VecDeque},
    convert::Infallible,
    fs,
    future::Future,
    io::Read,
    net::SocketAddr,
    path::{Path, PathBuf},
    sync::{
        Arc,
        atomic::{AtomicU8, Ordering},
    },
    time::{Duration, Instant},
};
mod residency;
use thiserror::Error;
use uuid::Uuid;
mod generation;
mod mapped;

pub use generation::{
    FinishReason, FrontierControl, GenerationFrontier, MAX_STOP_BYTES, MAX_STOP_SEQUENCES,
    StopAlignment,
};
pub use mapped::{ExecutionProfile, MappedEngine, MappedMetrics};
pub use residency::{ResidencyConfig, ResidencyMetrics, ResidentEngine, ResidentModelStatus};

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

#[derive(Clone, Copy, Debug, Default, Serialize, Deserialize, PartialEq, Eq)]
pub struct PrefillMetrics {
    pub total_tokens: usize,
    pub cached_tokens: usize,
    pub uncached_tokens: usize,
    pub tokenization_ns: u64,
    pub prefix_lookup_ns: u64,
    pub mapping_activation_ns: u64,
    pub uncached_prefill_ns: u64,
    pub total_ns: u64,
    pub transfer_bytes: u64,
    pub device_bytes: usize,
    pub host_bytes: usize,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct Usage {
    pub input_tokens: usize,
    pub generated_tokens: usize,
    pub evaluated_tokens: usize,
    pub cached_tokens: usize,
    pub prefill: PrefillMetrics,
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
    Finished {
        reason: FinishReason,
        usage: Usage,
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
    pub stop: Vec<String>,
    #[serde(default)]
    pub raw_continuation: bool,
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
    #[serde(default)]
    pub aliases: Vec<String>,
    #[serde(default = "default_model_family")]
    pub family: String,
    #[serde(default)]
    pub size_bytes: u64,
    #[serde(default)]
    pub epoch: u64,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
struct DurableState {
    installation: Uuid,
    contexts: HashMap<ContextId, ContextRecord>,
    models: HashMap<String, ModelRecord>,
    #[serde(default = "initial_model_epoch")]
    next_model_epoch: u64,
}

fn initial_model_epoch() -> u64 {
    1
}

fn default_model_family() -> String {
    "gemma-4-e2b-it".into()
}

#[derive(Clone, Copy, Debug, Deserialize, PartialEq, Eq, Serialize)]
pub struct ServerConfig {
    pub active_requests: usize,
    pub queue_count: usize,
    pub queue_bytes: usize,
    pub request_bytes: usize,
    pub pre_queue_concurrency: usize,
    pub header_bytes: usize,
    pub body_timeout_ms: u64,
    pub wall_time_ms: u64,
    pub active_time_ms: u64,
    pub stream_buffer: usize,
    pub shutdown_grace_ms: u64,
}
impl Default for ServerConfig {
    fn default() -> Self {
        Self {
            active_requests: 1,
            queue_count: 32,
            queue_bytes: 16 << 20,
            request_bytes: 1 << 20,
            pre_queue_concurrency: 16,
            header_bytes: 32 << 10,
            body_timeout_ms: 10_000,
            wall_time_ms: 300_000,
            active_time_ms: 240_000,
            stream_buffer: 8,
            shutdown_grace_ms: 30_000,
        }
    }
}

const HTTP_DEBUG_BODY_LIMIT: usize = 64 << 10;

#[derive(Clone)]
pub struct HttpDebug {
    sink: Arc<dyn Fn(&str) + Send + Sync>,
}

impl HttpDebug {
    pub fn stderr() -> Self {
        Self::new(|line| eprintln!("{line}"))
    }

    pub fn new(sink: impl Fn(&str) + Send + Sync + 'static) -> Self {
        Self {
            sink: Arc::new(sink),
        }
    }

    fn emit(&self, record: Value) {
        (self.sink)(&record.to_string());
    }
}

#[derive(Default)]
struct HttpBodyCapture {
    bytes: Vec<u8>,
    total: usize,
    truncated: bool,
}

impl HttpBodyCapture {
    fn push(&mut self, bytes: &[u8]) {
        self.total = self.total.saturating_add(bytes.len());
        if self.truncated {
            return;
        }
        if self.bytes.len().saturating_add(bytes.len()) > HTTP_DEBUG_BODY_LIMIT {
            self.bytes.clear();
            self.truncated = true;
        } else {
            self.bytes.extend_from_slice(bytes);
        }
    }

    fn redacted(&self, content_type: Option<&str>) -> Value {
        if self.truncated {
            return json!({
                "bytes": self.total,
                "omitted": "body exceeds 65536-byte HTTP debug limit"
            });
        }
        if self.bytes.is_empty() {
            return json!({"bytes": 0});
        }
        let is_sse = content_type.is_some_and(|value| value.starts_with("text/event-stream"));
        if is_sse {
            let events = String::from_utf8_lossy(&self.bytes)
                .lines()
                .filter_map(|line| line.strip_prefix("data:"))
                .map(str::trim)
                .map(|data| {
                    serde_json::from_str(data).map_or_else(
                        |_| json!({"omitted": "non-JSON SSE data"}),
                        |mut value| {
                            redact_body_values(&mut value);
                            value
                        },
                    )
                })
                .collect::<Vec<_>>();
            return json!({"bytes": self.total, "events": events});
        }
        let Ok(mut value) = serde_json::from_slice(&self.bytes) else {
            return json!({"bytes": self.total, "omitted": "non-JSON body"});
        };
        redact_body_values(&mut value);
        json!({"bytes": self.total, "json": value})
    }
}

fn redact_body_values(value: &mut Value) {
    match value {
        Value::Object(fields) => {
            for (key, value) in fields {
                let key = key.to_ascii_lowercase().replace('-', "_");
                if matches!(
                    key.as_str(),
                    "authorization"
                        | "cookie"
                        | "set_cookie"
                        | "password"
                        | "secret"
                        | "api_key"
                        | "bearer_token"
                        | "access_token"
                        | "refresh_token"
                ) {
                    *value = Value::String("[REDACTED]".into());
                } else {
                    redact_body_values(value);
                }
            }
        }
        Value::Array(items) => items.iter_mut().for_each(redact_body_values),
        Value::String(value) if value == "[DONE]" => {}
        Value::String(value) => *value = "[REDACTED]".into(),
        _ => {}
    }
}
#[derive(Clone, Copy, Debug, Default, Deserialize, PartialEq, Eq, Serialize)]
pub struct AdmissionMetrics {
    pub active: usize,
    pub queued: usize,
    pub queued_bytes: usize,
}
#[derive(Debug, Error)]
pub enum Error {
    #[error("unauthorized")]
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
    #[error("server is shutting down")]
    ShuttingDown,
    #[error("request exceeds the configured byte limit")]
    PayloadTooLarge,
    #[error("request headers exceed the configured byte limit")]
    HeadersTooLarge,
    #[error("request body timed out")]
    BodyTimeout,
    #[error("invalid request: {0}")]
    BadRequest(String),
    #[error("unsafe unauthenticated listener: {0}")]
    UnsafeListener(SocketAddr),
    #[error("state error: {0}")]
    State(String),
}
impl Error {
    fn code(&self) -> &'static str {
        match self {
            Self::Unauthorized => "unauthorized",
            Self::Forbidden => "forbidden",
            Self::ModelNotFound(_) => "model_not_found",
            Self::ContextNotFound => "context_not_found",
            Self::Deadline => "deadline_exceeded",
            Self::Cancelled => "cancelled",
            Self::Busy => "queue_overloaded",
            Self::ShuttingDown => "server_shutting_down",
            Self::PayloadTooLarge => "payload_too_large",
            Self::HeadersTooLarge => "headers_too_large",
            Self::BodyTimeout => "body_timeout",
            Self::BadRequest(_) => "invalid_request",
            Self::UnsafeListener(_) => "unsafe_listener",
            Self::State(_) => "state_error",
        }
    }
}
impl IntoResponse for Error {
    fn into_response(self) -> Response {
        let status = match &self {
            Self::Unauthorized => StatusCode::UNAUTHORIZED,
            Self::Forbidden => StatusCode::FORBIDDEN,
            Self::ModelNotFound(_) | Self::ContextNotFound => StatusCode::NOT_FOUND,
            Self::Deadline | Self::BodyTimeout => StatusCode::REQUEST_TIMEOUT,
            Self::Cancelled | Self::BadRequest(_) => StatusCode::BAD_REQUEST,
            Self::Busy => StatusCode::TOO_MANY_REQUESTS,
            Self::ShuttingDown => StatusCode::SERVICE_UNAVAILABLE,
            Self::PayloadTooLarge => StatusCode::PAYLOAD_TOO_LARGE,
            Self::HeadersTooLarge => StatusCode::REQUEST_HEADER_FIELDS_TOO_LARGE,
            Self::UnsafeListener(_) | Self::State(_) => StatusCode::INTERNAL_SERVER_ERROR,
        };
        let retry = matches!(self, Self::Busy).then_some("1");
        let mut response = (
            status,
            Json(json!({"error":{"code":self.code(),"message":self.to_string()}})),
        )
            .into_response();
        if let Some(value) = retry {
            response
                .headers_mut()
                .insert(RETRY_AFTER, value.parse().expect("static header value"));
        }
        response
    }
}

const CONTROL_RUNNING: u8 = 0;
const CONTROL_CANCELLED: u8 = 1;
const CONTROL_DEADLINE: u8 = 2;
const CONTROL_COMPLETE: u8 = 3;

pub struct RequestControl {
    state: AtomicU8,
    abort: Mutex<Option<Arc<dyn Fn() + Send + Sync>>>,
}
impl RequestControl {
    fn new() -> Self {
        Self {
            state: AtomicU8::new(CONTROL_RUNNING),
            abort: Mutex::new(None),
        }
    }

    pub fn check(&self) -> Result<(), Error> {
        match self.state.load(Ordering::Acquire) {
            CONTROL_RUNNING => Ok(()),
            CONTROL_DEADLINE => Err(Error::Deadline),
            _ => Err(Error::Cancelled),
        }
    }

    fn cancel(&self) {
        if self
            .state
            .compare_exchange(
                CONTROL_RUNNING,
                CONTROL_CANCELLED,
                Ordering::AcqRel,
                Ordering::Acquire,
            )
            .is_ok()
        {
            self.abort();
        }
    }

    fn expire(&self) {
        if self
            .state
            .compare_exchange(
                CONTROL_RUNNING,
                CONTROL_DEADLINE,
                Ordering::AcqRel,
                Ordering::Acquire,
            )
            .is_ok()
        {
            self.abort();
        }
    }

    fn complete(&self) {
        let _ = self.state.compare_exchange(
            CONTROL_RUNNING,
            CONTROL_COMPLETE,
            Ordering::AcqRel,
            Ordering::Acquire,
        );
    }

    fn is_complete(&self) -> bool {
        self.state.load(Ordering::Acquire) == CONTROL_COMPLETE
    }

    fn register_abort(&self, abort: Arc<dyn Fn() + Send + Sync>) -> AbortRegistration<'_> {
        *self.abort.lock() = Some(abort.clone());
        if self.check().is_err() {
            abort();
        }
        AbortRegistration { control: self }
    }

    fn abort(&self) {
        let abort = self.abort.lock().clone();
        if let Some(abort) = abort {
            abort();
        }
    }
}

struct AbortRegistration<'a> {
    control: &'a RequestControl,
}

impl Drop for AbortRegistration<'_> {
    fn drop(&mut self) {
        self.control.abort.lock().take();
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
    pub control: &'a RequestControl,
}

pub type TokenSink<'a> = dyn FnMut(i32, &[u8], bool) -> Result<FrontierControl, Error> + 'a;

#[derive(Clone, Debug, Default, PartialEq)]
pub struct EngineOutput {
    pub successor_tokens: Vec<i32>,
    pub input_tokens: usize,
    pub cached_tokens: usize,
    pub evaluated_tokens: usize,
    pub prefill: PrefillMetrics,
}

pub trait InferenceEngine: Send + Sync {
    fn generate(
        &self,
        request: EngineRequest<'_>,
        sink: &mut TokenSink<'_>,
    ) -> Result<EngineOutput, Error>;

    fn prepare_model(&self, _model: &ModelRecord) -> Result<(), Error> {
        Ok(())
    }

    fn commit_model(&self, _model: &ModelRecord, _replaced_epoch: Option<u64>) {}

    fn retire_model(&self, _id: &str, _epoch: u64) {}

    fn demote_inactive(&self) -> Result<usize, Error> {
        Ok(0)
    }

    fn residency_status(&self) -> Option<Value> {
        None
    }
}
#[derive(Default)]
pub struct DeterministicEngine;
impl InferenceEngine for DeterministicEngine {
    fn generate(
        &self,
        request: EngineRequest<'_>,
        sink: &mut TokenSink<'_>,
    ) -> Result<EngineOutput, Error> {
        request.control.check()?;
        let input_tokens = request.prompt.split_whitespace().count();
        for (index, piece) in request
            .prompt
            .split_whitespace()
            .rev()
            .cycle()
            .take(request.max_tokens)
            .enumerate()
        {
            request.control.check()?;
            let piece = if index == 0 {
                piece.to_owned()
            } else {
                format!(" {piece}")
            };
            if sink(-(index as i32) - 1, piece.as_bytes(), false)? == FrontierControl::Stop {
                break;
            }
        }
        Ok(EngineOutput {
            successor_tokens: request.prior_tokens.to_vec(),
            input_tokens,
            cached_tokens: 0,
            evaluated_tokens: input_tokens,
            prefill: PrefillMetrics {
                total_tokens: input_tokens,
                uncached_tokens: input_tokens,
                ..PrefillMetrics::default()
            },
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

struct QueueEntry {
    ticket: u64,
    request_id: String,
    bytes: usize,
    ready: tokio::sync::oneshot::Sender<()>,
}
struct Inner {
    durable: DurableState,
    controls: HashMap<String, Arc<RequestControl>>,
    pending_cancelled: HashSet<String>,
    active: usize,
    admission_limit: usize,
    queue: VecDeque<QueueEntry>,
    queued_bytes: usize,
    next_ticket: u64,
    shutting_down: bool,
    config: ServerConfig,
}
struct PrequeueState {
    limit: usize,
    retire_on_drop: usize,
}

struct PrequeueGate {
    semaphore: Arc<tokio::sync::Semaphore>,
    state: Mutex<PrequeueState>,
}

impl PrequeueGate {
    fn new(limit: usize) -> Arc<Self> {
        Arc::new(Self {
            semaphore: Arc::new(tokio::sync::Semaphore::new(limit)),
            state: Mutex::new(PrequeueState {
                limit,
                retire_on_drop: 0,
            }),
        })
    }

    async fn acquire(self: &Arc<Self>) -> Result<PrequeuePermit, Error> {
        let permit = self
            .semaphore
            .clone()
            .acquire_owned()
            .await
            .map_err(|_| Error::ShuttingDown)?;
        Ok(PrequeuePermit {
            gate: self.clone(),
            permit: Some(permit),
        })
    }

    fn set_limit(&self, limit: usize) {
        let mut state = self.state.lock();
        if limit < state.limit {
            let reduction = state.limit - limit;
            let retired = self.semaphore.forget_permits(reduction);
            state.retire_on_drop += reduction - retired;
        } else {
            let increase = limit - state.limit;
            let cancelled_retirements = increase.min(state.retire_on_drop);
            state.retire_on_drop -= cancelled_retirements;
            self.semaphore.add_permits(increase - cancelled_retirements);
        }
        state.limit = limit;
    }
}

struct PrequeuePermit {
    gate: Arc<PrequeueGate>,
    permit: Option<tokio::sync::OwnedSemaphorePermit>,
}

impl Drop for PrequeuePermit {
    fn drop(&mut self) {
        let permit = self.permit.take().expect("pre-queue permit exists");
        let mut state = self.gate.state.lock();
        if state.retire_on_drop == 0 {
            drop(state);
            drop(permit);
        } else {
            state.retire_on_drop -= 1;
            permit.forget();
        }
    }
}

#[derive(Clone)]
pub struct Server {
    state_path: PathBuf,
    inner: Arc<Mutex<Inner>>,
    pre_queue: Arc<PrequeueGate>,
    model_lifecycle: Arc<Mutex<()>>,
    auth: Arc<dyn AuthProvider>,
    engine: Arc<dyn InferenceEngine>,
}
struct AdmissionGuard {
    server: Server,
    ticket: Option<u64>,
    request_id: Option<String>,
    control: Arc<RequestControl>,
    active: bool,
}
impl Drop for AdmissionGuard {
    fn drop(&mut self) {
        self.server.release_or_cancel_admission(
            self.ticket.take(),
            self.request_id.take(),
            self.active,
        );
    }
}
impl Server {
    pub fn open(
        path: impl AsRef<Path>,
        auth: Arc<dyn AuthProvider>,
        engine: Arc<dyn InferenceEngine>,
    ) -> Result<Self, Error> {
        let path = path.as_ref().to_owned();
        let mut durable: DurableState = if path.exists() {
            serde_json::from_slice(&fs::read(&path).map_err(state_err)?).map_err(state_err)?
        } else {
            DurableState {
                installation: Uuid::new_v4(),
                contexts: HashMap::new(),
                models: HashMap::new(),
                next_model_epoch: initial_model_epoch(),
            }
        };
        durable.next_model_epoch = durable
            .models
            .values()
            .map(|model| model.epoch)
            .max()
            .unwrap_or(0)
            .saturating_add(1)
            .max(durable.next_model_epoch);
        let config = ServerConfig::default();
        let server = Self {
            state_path: path,
            inner: Arc::new(Mutex::new(Inner {
                durable,
                controls: HashMap::new(),
                pending_cancelled: HashSet::new(),
                active: 0,
                admission_limit: config.active_requests,
                queue: VecDeque::new(),
                queued_bytes: 0,
                next_ticket: 0,
                shutting_down: false,
                config,
            })),
            pre_queue: PrequeueGate::new(config.pre_queue_concurrency),
            model_lifecycle: Arc::new(Mutex::new(())),
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
        let _lifecycle = self.model_lifecycle.lock();
        model.aliases.sort();
        model.aliases.dedup();
        if model.family.is_empty() {
            model.family = default_model_family();
        }
        if model.size_bytes == 0 {
            model.size_bytes = fs::metadata(&model.path).map_err(state_err)?.len();
        }
        let existing = self
            .inner
            .lock()
            .durable
            .models
            .get(&model.id)
            .cloned()
            .filter(|existing| {
                existing.revision == model.revision
                    && existing.path == model.path
                    && existing.sha256 == model.sha256
                    && existing.family == model.family
                    && existing.size_bytes == model.size_bytes
            });
        if let Some(existing) = existing {
            self.engine.prepare_model(&existing)?;
            return Ok(existing);
        }
        let (epoch, previous, previous_models) = {
            let mut guard = self.inner.lock();
            let epoch = guard.durable.next_model_epoch;
            guard.durable.next_model_epoch = epoch
                .checked_add(1)
                .ok_or_else(|| Error::State("model epoch space exhausted".into()))?;
            (
                epoch,
                guard.durable.models.get(&model.id).cloned(),
                guard.durable.models.clone(),
            )
        };
        model.epoch = epoch;
        if let Err(error) = self.engine.prepare_model(&model) {
            self.inner.lock().durable.next_model_epoch = epoch;
            return Err(error);
        }
        {
            let mut guard = self.inner.lock();
            for existing in guard.durable.models.values_mut() {
                existing
                    .aliases
                    .retain(|alias| !model.aliases.contains(alias));
            }
            guard.durable.models.insert(model.id.clone(), model.clone());
        }
        if let Err(error) = self.persist() {
            let mut guard = self.inner.lock();
            guard.durable.models = previous_models;
            guard.durable.next_model_epoch = epoch;
            drop(guard);
            self.engine.retire_model(&model.id, model.epoch);
            return Err(error);
        }
        self.engine
            .commit_model(&model, previous.as_ref().map(|record| record.epoch));
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
        let _lifecycle = self.model_lifecycle.lock();
        let mut guard = self.inner.lock();
        if !guard.durable.models.contains_key(id) {
            return Err(Error::ModelNotFound(id.into()));
        }
        let previous = guard.durable.models.clone();
        for model in guard.durable.models.values_mut() {
            model.aliases.retain(|existing| existing != &alias);
        }
        let model = guard.durable.models.get_mut(id).unwrap();
        model.aliases.push(alias);
        model.aliases.sort();
        let out = model.clone();
        drop(guard);
        if let Err(error) = self.persist() {
            self.inner.lock().durable.models = previous;
            return Err(error);
        }
        Ok(out)
    }
    pub fn remove_model(&self, id: &str) -> Result<(), Error> {
        let _lifecycle = self.model_lifecycle.lock();
        let model = self
            .inner
            .lock()
            .durable
            .models
            .remove(id)
            .ok_or_else(|| Error::ModelNotFound(id.into()))?;
        if let Err(error) = self.persist() {
            self.inner
                .lock()
                .durable
                .models
                .insert(model.id.clone(), model);
            return Err(error);
        }
        self.engine.retire_model(&model.id, model.epoch);
        Ok(())
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
        let mut guard = self.inner.lock();
        if let Some(control) = guard.controls.get(request).cloned() {
            control.cancel();
        } else {
            guard.pending_cancelled.insert(request.to_owned());
        }
        if let Some(index) = guard
            .queue
            .iter()
            .position(|entry| entry.request_id == request)
        {
            let entry = guard.queue.remove(index).expect("queued request exists");
            guard.queued_bytes -= entry.bytes;
        }
        Self::promote_queued(&mut guard);
    }
    pub fn set_admission_limit(&self, limit: usize) {
        let mut guard = self.inner.lock();
        guard.admission_limit = limit;
        guard.config.active_requests = limit;
        Self::promote_queued(&mut guard);
    }
    pub fn configure(&self, config: ServerConfig) -> Result<(), Error> {
        if config.active_requests == 0
            || config.queue_count == 0
            || config.queue_bytes == 0
            || config.request_bytes == 0
            || config.pre_queue_concurrency == 0
            || config.header_bytes == 0
            || config.body_timeout_ms == 0
            || config.wall_time_ms == 0
            || config.active_time_ms == 0
            || config.stream_buffer == 0
            || config.shutdown_grace_ms == 0
        {
            return Err(Error::State("server limits must be nonzero".into()));
        }
        let mut guard = self.inner.lock();
        self.pre_queue.set_limit(config.pre_queue_concurrency);
        guard.admission_limit = config.active_requests;
        guard.config = config;
        Self::promote_queued(&mut guard);
        Ok(())
    }
    pub fn admission_metrics(&self) -> AdmissionMetrics {
        let guard = self.inner.lock();
        AdmissionMetrics {
            active: guard.active,
            queued: guard.queue.len(),
            queued_bytes: guard.queued_bytes,
        }
    }
    pub fn config(&self) -> ServerConfig {
        self.inner.lock().config
    }
    pub fn infer(
        &self,
        request_id: &str,
        req: InferRequest,
    ) -> Result<(InferResponse, Vec<StreamEvent>), Error> {
        let control = Arc::new(RequestControl::new());
        let admission = self.try_admit(request_id, control)?;
        let result = self.infer_reserved(request_id, req);
        drop(admission);
        result
    }
    fn begin_shutdown(&self) {
        self.inner.lock().shutting_down = true;
    }

    fn cancel_remaining(&self) {
        let mut guard = self.inner.lock();
        for control in guard.controls.values() {
            control.cancel();
        }
        guard.queue.clear();
        guard.queued_bytes = 0;
    }

    fn infer_reserved(
        &self,
        request_id: &str,
        req: InferRequest,
    ) -> Result<(InferResponse, Vec<StreamEvent>), Error> {
        let frontier =
            GenerationFrontier::new(&req.stop, req.raw_continuation).map_err(state_err)?;
        let successor_id = req.context_id.clone().unwrap_or_else(ContextId::new);
        let mut events = vec![StreamEvent::Started {
            request_id: request_id.into(),
            context_id: successor_id.clone(),
        }];
        let (response, terminal) =
            self.infer_admitted(request_id, req, successor_id, frontier, |event| {
                events.push(event);
                Ok(())
            })?;
        events.push(terminal);
        Ok((response, events))
    }
    async fn infer_stream_reserved(
        &self,
        request_id: String,
        req: InferRequest,
        admission: AdmissionGuard,
    ) -> Result<(StreamEvent, tokio::sync::mpsc::Receiver<StreamEvent>), Error> {
        let frontier =
            GenerationFrontier::new(&req.stop, req.raw_continuation).map_err(state_err)?;
        let successor_id = req.context_id.clone().unwrap_or_else(ContextId::new);
        let started = StreamEvent::Started {
            request_id: request_id.clone(),
            context_id: successor_id.clone(),
        };
        let stream_buffer = self.inner.lock().config.stream_buffer;
        let (sender, receiver) = tokio::sync::mpsc::channel(stream_buffer);
        let server = self.clone();
        let control = admission.control.clone();
        tokio::task::spawn_blocking(move || {
            let _admission = admission;
            let emit_control = control.clone();
            let result = server.infer_admitted(&request_id, req, successor_id, frontier, |event| {
                sender.blocking_send(event).map_err(|_| {
                    emit_control.cancel();
                    Error::Cancelled
                })
            });
            let event = match result {
                Ok((_, terminal)) => terminal,
                Err(error) => StreamEvent::Error {
                    message: error.to_string(),
                },
            };
            let _ = sender.blocking_send(event);
        });
        Ok((started, receiver))
    }

    fn try_admit(
        &self,
        request_id: &str,
        control: Arc<RequestControl>,
    ) -> Result<AdmissionGuard, Error> {
        let mut guard = self.inner.lock();
        if guard.pending_cancelled.remove(request_id) {
            return Err(Error::Cancelled);
        }
        if guard.shutting_down {
            return Err(Error::ShuttingDown);
        }
        if guard.active >= guard.admission_limit || !guard.queue.is_empty() {
            return Err(Error::Busy);
        }
        guard.active += 1;
        guard.controls.insert(request_id.into(), control.clone());
        Ok(AdmissionGuard {
            server: self.clone(),
            ticket: None,
            request_id: Some(request_id.into()),
            control,
            active: true,
        })
    }

    fn promote_queued(guard: &mut Inner) {
        while guard.active < guard.admission_limit {
            let Some(entry) = guard.queue.pop_front() else {
                break;
            };
            guard.queued_bytes -= entry.bytes;
            if entry.ready.send(()).is_ok() {
                guard.active += 1;
            }
        }
    }

    fn release_or_cancel_admission(
        &self,
        ticket: Option<u64>,
        request_id: Option<String>,
        active: bool,
    ) {
        let mut guard = self.inner.lock();
        if let Some(request_id) = request_id {
            if let Some(control) = guard.controls.remove(&request_id) {
                control.complete();
            }
        }
        if active {
            guard.active = guard.active.saturating_sub(1);
        } else if let Some(ticket) = ticket {
            if let Some(index) = guard.queue.iter().position(|entry| entry.ticket == ticket) {
                let entry = guard.queue.remove(index).expect("queued ticket exists");
                guard.queued_bytes -= entry.bytes;
            } else {
                guard.active = guard.active.saturating_sub(1);
            }
        }
        Self::promote_queued(&mut guard);
    }

    async fn reserve_admission(
        &self,
        request_id: &str,
        control: Arc<RequestControl>,
        bytes: usize,
        deadline: Duration,
    ) -> Result<AdmissionGuard, Error> {
        let (ticket, receiver) = {
            let mut guard = self.inner.lock();
            if guard.pending_cancelled.remove(request_id) {
                return Err(Error::Cancelled);
            }
            if guard.shutting_down {
                return Err(Error::ShuttingDown);
            }
            if bytes > guard.config.request_bytes {
                return Err(Error::PayloadTooLarge);
            }
            guard
                .controls
                .insert(request_id.to_owned(), control.clone());
            if guard.active < guard.admission_limit && guard.queue.is_empty() {
                guard.active += 1;
                return Ok(AdmissionGuard {
                    server: self.clone(),
                    ticket: None,
                    request_id: Some(request_id.to_owned()),
                    control,
                    active: true,
                });
            }
            if guard.queue.len() >= guard.config.queue_count
                || bytes > guard.config.queue_bytes.saturating_sub(guard.queued_bytes)
            {
                guard.controls.remove(request_id);
                return Err(Error::Busy);
            }
            let ticket = guard.next_ticket;
            guard.next_ticket = guard
                .next_ticket
                .checked_add(1)
                .ok_or_else(|| Error::State("admission ticket space exhausted".into()))?;
            let (ready, receiver) = tokio::sync::oneshot::channel();
            guard.queue.push_back(QueueEntry {
                ticket,
                request_id: request_id.to_owned(),
                bytes,
                ready,
            });
            guard.queued_bytes += bytes;
            (ticket, receiver)
        };
        let mut admission = AdmissionGuard {
            server: self.clone(),
            ticket: Some(ticket),
            request_id: Some(request_id.to_owned()),
            control: control.clone(),
            active: false,
        };
        match tokio::time::timeout(deadline, receiver).await {
            Ok(Ok(())) => {
                control.check()?;
                admission.ticket = None;
                admission.active = true;
                Ok(admission)
            }
            Ok(Err(_)) => {
                admission.ticket = None;
                control.check().and(Err(Error::Cancelled))
            }
            Err(_) => {
                control.expire();
                Err(Error::Deadline)
            }
        }
    }

    fn infer_admitted(
        &self,
        request_id: &str,
        req: InferRequest,
        successor_id: ContextId,
        mut frontier: GenerationFrontier,
        mut emit: impl FnMut(StreamEvent) -> Result<(), Error>,
    ) -> Result<(InferResponse, StreamEvent), Error> {
        let started = Instant::now();
        let deadline = req
            .deadline_ms
            .and_then(|milliseconds| started.checked_add(Duration::from_millis(milliseconds)));
        if req.deadline_ms == Some(0) {
            return Err(Error::Deadline);
        }
        let control = self
            .inner
            .lock()
            .controls
            .get(request_id)
            .cloned()
            .ok_or(Error::Cancelled)?;
        control.check()?;
        let model = self.model(&req.model)?;
        let context = match &req.context_id {
            Some(id) => Some(self.context(id)?),
            None => None,
        };
        let prior_tokens = context
            .as_ref()
            .map_or(&[][..], |record| record.native_tokens.as_slice());
        let mut generated_pieces = Vec::new();
        let mut delta_index = 0;
        let generated = self.engine.generate(
            EngineRequest {
                model: &model,
                prompt: &req.prompt,
                max_tokens: req.max_tokens,
                prior_tokens,
                control: &control,
            },
            &mut |_id, piece, terminal_or_control| {
                if deadline.is_some_and(|deadline| Instant::now() >= deadline) {
                    control.expire();
                }
                control.check()?;
                frontier.push(piece, terminal_or_control, |delta| {
                    let token = delta.to_owned();
                    emit(StreamEvent::Token {
                        token: token.clone(),
                        index: delta_index,
                    })?;
                    delta_index += 1;
                    generated_pieces.push(token);
                    Ok(())
                })
            },
        )?;
        let frontier = frontier.finish(|delta| {
            let token = delta.to_owned();
            emit(StreamEvent::Token {
                token: token.clone(),
                index: delta_index,
            })?;
            delta_index += 1;
            generated_pieces.push(token);
            Ok(())
        })?;
        control.check()?;
        if req
            .deadline_ms
            .is_some_and(|ms| started.elapsed() > Duration::from_millis(ms))
        {
            return Err(Error::Deadline);
        }
        let input: Vec<_> = req.prompt.split_whitespace().map(str::to_owned).collect();
        let EngineOutput {
            successor_tokens,
            input_tokens,
            cached_tokens,
            evaluated_tokens,
            prefill,
        } = generated;
        let mut guard = self.inner.lock();
        let context_id = if let Some(context) = context {
            let stored = guard
                .durable
                .contexts
                .get_mut(&context.id)
                .ok_or(Error::ContextNotFound)?;
            stored.tokens.extend(input.iter().cloned());
            stored.tokens.extend(generated_pieces.iter().cloned());
            stored.native_tokens = successor_tokens;
            stored.revision += 1;
            stored.id.clone()
        } else {
            let mut tokens = input.clone();
            tokens.extend(generated_pieces.iter().cloned());
            guard.durable.contexts.insert(
                successor_id.clone(),
                ContextRecord {
                    id: successor_id.clone(),
                    revision: 1,
                    tokens,
                    native_tokens: successor_tokens,
                },
            );
            successor_id
        };
        drop(guard);
        self.persist()?;
        let usage = Usage {
            input_tokens,
            generated_tokens: frontier.generated_tokens,
            evaluated_tokens,
            cached_tokens,
            prefill,
            model: model.id,
            model_revision: model.revision,
            context_id: context_id.clone(),
            latency_ms: started.elapsed().as_millis(),
            status: "completed".into(),
        };
        Ok((
            InferResponse {
                id: request_id.into(),
                text: frontier.text,
                usage: usage.clone(),
            },
            StreamEvent::Finished {
                reason: frontier.finish_reason,
                usage,
            },
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
    #[serde(default = "default_model_family")]
    family: String,
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
struct PrequeueJson<T> {
    headers: HeaderMap,
    value: T,
    retained_bytes: usize,
    _permit: PrequeuePermit,
}

impl<T> FromRequest<Server> for PrequeueJson<T>
where
    T: DeserializeOwned,
{
    type Rejection = Error;

    async fn from_request(request: Request, state: &Server) -> Result<Self, Self::Rejection> {
        let config = state.config();
        let header_bytes = request
            .headers()
            .iter()
            .fold(0usize, |total, (name, value)| {
                total
                    .saturating_add(name.as_str().len())
                    .saturating_add(value.as_bytes().len())
            });
        if header_bytes > config.header_bytes {
            return Err(Error::HeadersTooLarge);
        }
        let permit = state.pre_queue.acquire().await?;
        let (parts, body) = request.into_parts();
        let bytes = tokio::time::timeout(
            Duration::from_millis(config.body_timeout_ms),
            to_bytes(body, config.request_bytes),
        )
        .await
        .map_err(|_| Error::BodyTimeout)?
        .map_err(|_| Error::PayloadTooLarge)?;
        let retained_bytes = bytes.len();
        let value =
            serde_json::from_slice(&bytes).map_err(|error| Error::BadRequest(error.to_string()))?;
        Ok(Self {
            headers: parts.headers,
            value,
            retained_bytes,
            _permit: permit,
        })
    }
}
#[derive(Deserialize)]
struct ImportContextRequest {
    tokens: Vec<String>,
}
#[derive(Deserialize)]
#[serde(untagged)]
enum StopInput {
    One(String),
    Many(Vec<String>),
}
impl StopInput {
    fn into_vec(self) -> Vec<String> {
        match self {
            Self::One(stop) => vec![stop],
            Self::Many(stops) => stops,
        }
    }
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
    #[serde(default)]
    stop: Option<StopInput>,
    #[serde(default)]
    raw_continuation: bool,
    #[serde(default)]
    deadline_ms: Option<u64>,
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
    #[serde(default)]
    stop: Option<StopInput>,
    #[serde(default)]
    deadline_ms: Option<u64>,
}
#[derive(Deserialize)]
struct ChatMessage {
    content: String,
}

pub fn router(server: Server) -> Router {
    routes(server)
}

pub fn router_with_http_debug(server: Server, debug: HttpDebug) -> Router {
    routes(server).layer(middleware::from_fn_with_state(debug, http_debug_middleware))
}

fn routes(server: Server) -> Router {
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
        .route("/native/status", get(native_status))
        .route("/native/contexts/{id}/branches", post(branch_context))
        .route("/native/requests/{id}", delete(cancel_request))
        .with_state(server)
}

async fn http_debug_middleware(
    State(debug): State<HttpDebug>,
    request: Request,
    next: Next,
) -> Response {
    let request_id = Uuid::new_v4().to_string();
    let method = request.method().to_string();
    let path = request.uri().path().to_owned();
    let request_content_type = request
        .headers()
        .get("content-type")
        .and_then(|value| value.to_str().ok())
        .map(str::to_owned);
    let request_content_length = request
        .headers()
        .get("content-length")
        .and_then(|value| value.to_str().ok())
        .map(str::to_owned);
    let request_body = Arc::new(Mutex::new(HttpBodyCapture::default()));
    let body_capture = request_body.clone();
    let (parts, body) = request.into_parts();
    let body = Body::new(body.map_frame(move |frame| {
        if let Some(bytes) = frame.data_ref() {
            body_capture.lock().push(bytes);
        }
        frame
    }));
    let started = Instant::now();
    let mut response = next.run(Request::from_parts(parts, body)).await;
    debug.emit(json!({
        "type": "http_debug",
        "direction": "in",
        "request_id": request_id,
        "method": method,
        "path": path,
        "content_type": request_content_type,
        "content_length": request_content_length,
        "body": request_body.lock().redacted(request_content_type.as_deref())
    }));

    let status = response.status();
    let response_content_type = response
        .headers()
        .get("content-type")
        .and_then(|value| value.to_str().ok())
        .map(str::to_owned);
    response.headers_mut().insert(
        "x-request-id",
        HeaderValue::from_str(&request_id).expect("UUID is a valid header value"),
    );
    debug.emit(json!({
        "type": "http_debug",
        "direction": "out",
        "request_id": request_id,
        "method": method,
        "path": path,
        "status": status.as_u16(),
        "duration_ms": started.elapsed().as_millis()
    }));

    let (parts, body) = response.into_parts();
    let mut chunk_index = 0_u64;
    let body = Body::new(body.map_frame(move |frame| {
        if let Some(bytes) = frame.data_ref() {
            let mut capture = HttpBodyCapture::default();
            capture.push(bytes);
            debug.emit(json!({
                "type": "http_debug",
                "direction": "out_body",
                "request_id": request_id,
                "chunk_index": chunk_index,
                "body": capture.redacted(response_content_type.as_deref())
            }));
            chunk_index += 1;
        }
        frame
    }));
    Response::from_parts(parts, body)
}
fn auth(server: &Server, headers: &HeaderMap, scope: Scope) -> Result<RequestContext, Error> {
    server.authorize(headers, scope)
}
async fn completion(
    State(s): State<Server>,
    PrequeueJson {
        headers,
        value: r,
        retained_bytes,
        _permit: permit,
    }: PrequeueJson<CompletionRequest>,
) -> Result<Response, Error> {
    auth(&s, &headers, Scope::Inference)?;
    s.model(&r.model)?;
    drop(permit);
    infer_response(
        s,
        r.model,
        r.prompt,
        r.max_tokens,
        r.stream,
        r.context_id,
        r.stop.map(StopInput::into_vec).unwrap_or_default(),
        r.raw_continuation,
        r.deadline_ms,
        retained_bytes,
    )
    .await
}
async fn chat(
    State(s): State<Server>,
    PrequeueJson {
        headers,
        value: r,
        retained_bytes,
        _permit: permit,
    }: PrequeueJson<ChatRequest>,
) -> Result<Response, Error> {
    auth(&s, &headers, Scope::Inference)?;
    s.model(&r.model)?;
    drop(permit);
    infer_response(
        s,
        r.model,
        r.messages
            .into_iter()
            .map(|m| m.content)
            .collect::<Vec<_>>()
            .join("\n"),
        r.max_tokens,
        r.stream,
        r.context_id,
        r.stop.map(StopInput::into_vec).unwrap_or_default(),
        false,
        r.deadline_ms,
        retained_bytes,
    )
    .await
}
struct DisconnectGuard {
    control: Arc<RequestControl>,
    armed: bool,
}

impl DisconnectGuard {
    fn new(control: Arc<RequestControl>) -> Self {
        Self {
            control,
            armed: true,
        }
    }

    fn disarm(&mut self) {
        self.armed = false;
    }
}

impl Drop for DisconnectGuard {
    fn drop(&mut self) {
        if self.armed {
            self.control.cancel();
        }
    }
}

fn start_deadline_watchdogs(
    control: &Arc<RequestControl>,
    wall_remaining: Duration,
    active_limit: Duration,
) {
    for deadline in [wall_remaining, active_limit] {
        let control = control.clone();
        tokio::spawn(async move {
            tokio::time::sleep(deadline).await;
            if !control.is_complete() {
                control.expire();
            }
        });
    }
}

async fn infer_response(
    server: Server,
    model: String,
    prompt: String,
    max_tokens: Option<usize>,
    streaming: bool,
    context_id: Option<ContextId>,
    stop: Vec<String>,
    raw_continuation: bool,
    deadline_ms: Option<u64>,
    retained_bytes: usize,
) -> Result<Response, Error> {
    server.model(&model)?;
    let id = Uuid::new_v4().to_string();
    let config = server.config();
    let wall_limit = Duration::from_millis(
        deadline_ms
            .unwrap_or(config.wall_time_ms)
            .min(config.wall_time_ms),
    );
    let waiting_since = Instant::now();
    let control = Arc::new(RequestControl::new());
    let admission = server
        .reserve_admission(&id, control.clone(), retained_bytes, wall_limit)
        .await?;
    let wall_remaining = wall_limit
        .checked_sub(waiting_since.elapsed())
        .ok_or(Error::Deadline)?;
    start_deadline_watchdogs(
        &control,
        wall_remaining,
        Duration::from_millis(config.active_time_ms),
    );
    let request = InferRequest {
        model,
        prompt,
        max_tokens: max_tokens.unwrap_or_else(default_tokens),
        context_id,
        deadline_ms: Some(
            u64::try_from(wall_remaining.as_millis())
                .unwrap_or(u64::MAX)
                .max(1),
        ),
        stop,
        raw_continuation,
        priority: 0,
    };
    if streaming {
        let (started, receiver) = server.infer_stream_reserved(id, request, admission).await?;
        let first = stream::once(async move { started });
        let disconnect = DisconnectGuard::new(control);
        let rest = stream::unfold(
            (receiver, disconnect),
            |(mut receiver, mut disconnect)| async move {
                match receiver.recv().await {
                    Some(event) => Some((event, (receiver, disconnect))),
                    None => {
                        disconnect.disarm();
                        None
                    }
                }
            },
        );
        let rows = first
            .chain(rest)
            .map(|event| Ok::<_, Infallible>(Event::default().json_data(event).unwrap()));
        Ok(Sse::new(rows).into_response())
    } else {
        let mut disconnect = DisconnectGuard::new(control);
        let result = tokio::task::spawn_blocking(move || {
            let _admission = admission;
            server.infer_reserved(&id, request)
        })
        .await
        .map_err(state_err)?;
        disconnect.disarm();
        let (response, _) = result?;
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
        family: r.family,
        size_bytes: 0,
        epoch: 0,
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
        family: "gemma-phase6".into(),
        size_bytes: fetched.size,
        epoch: 0,
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
async fn native_status(
    State(server): State<Server>,
    headers: HeaderMap,
) -> Result<Json<Value>, Error> {
    auth(&server, &headers, Scope::Admin)?;
    Ok(Json(json!({
        "config": server.config(),
        "admission": server.admission_metrics(),
        "residency": server.engine.residency_status(),
    })))
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
        "/native/requests/{id}":{"delete":{}},"/native/status":{"get":{}}
    }})
}

pub async fn serve(
    server: Server,
    addr: SocketAddr,
    anonymous: bool,
    unsafe_public: bool,
) -> Result<(), Error> {
    serve_until(
        server,
        addr,
        anonymous,
        unsafe_public,
        None,
        shutdown_signal(),
    )
    .await
}

pub async fn serve_with_http_debug(
    server: Server,
    addr: SocketAddr,
    anonymous: bool,
    unsafe_public: bool,
) -> Result<(), Error> {
    serve_until(
        server,
        addr,
        anonymous,
        unsafe_public,
        Some(HttpDebug::stderr()),
        shutdown_signal(),
    )
    .await
}

async fn serve_until(
    server: Server,
    addr: SocketAddr,
    anonymous: bool,
    unsafe_public: bool,
    http_debug: Option<HttpDebug>,
    shutdown: impl Future<Output = ()>,
) -> Result<(), Error> {
    Server::validate_listener(addr, anonymous, unsafe_public)?;
    let grace = Duration::from_millis(server.config().shutdown_grace_ms);
    let lifecycle = server.clone();
    let listener = tokio::net::TcpListener::bind(addr)
        .await
        .map_err(state_err)?;
    let (begin_shutdown, shutdown_requested) = tokio::sync::oneshot::channel();
    let serving = async move {
        let app = match http_debug {
            Some(debug) => router_with_http_debug(server, debug),
            None => router(server),
        };
        axum::serve(listener, app)
            .with_graceful_shutdown(async move {
                let _ = shutdown_requested.await;
            })
            .await
    };
    tokio::pin!(serving);
    tokio::select! {
        result = &mut serving => result.map_err(state_err),
        () = shutdown => {
            lifecycle.begin_shutdown();
            let _ = begin_shutdown.send(());
            match tokio::time::timeout(grace, &mut serving).await {
                Ok(result) => result.map_err(state_err),
                Err(_) => {
                    lifecycle.cancel_remaining();
                    (&mut serving).await.map_err(state_err)
                }
            }
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
    use axum::{
        body::{Body, Bytes, to_bytes},
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
            family: "gemma-4-e2b-it".into(),
            size_bytes: 5,
            epoch: 0,
        })
        .unwrap();
        (s, d)
    }

    struct FailingEngine;
    impl InferenceEngine for FailingEngine {
        fn generate(
            &self,
            _: EngineRequest<'_>,
            _: &mut TokenSink<'_>,
        ) -> Result<EngineOutput, Error> {
            Err(Error::State("generation failed".into()))
        }
    }

    struct SlowEngine {
        started: Arc<tokio::sync::Notify>,
    }
    impl InferenceEngine for SlowEngine {
        fn generate(
            &self,
            request: EngineRequest<'_>,
            sink: &mut TokenSink<'_>,
        ) -> Result<EngineOutput, Error> {
            for index in 0..request.max_tokens {
                sink(index as i32, b"x", false)?;
                if index == 0 {
                    self.started.notify_one();
                }
                std::thread::sleep(Duration::from_millis(20));
            }
            Ok(EngineOutput {
                successor_tokens: (0..request.max_tokens as i32).collect(),
                input_tokens: request.prompt.split_whitespace().count(),
                cached_tokens: 0,
                evaluated_tokens: request.prompt.split_whitespace().count(),
                prefill: PrefillMetrics {
                    total_tokens: request.prompt.split_whitespace().count(),
                    uncached_tokens: request.prompt.split_whitespace().count(),
                    ..PrefillMetrics::default()
                },
            })
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
            stop: vec![],
            raw_continuation: false,
        };
        let (out, events) = s.infer("r", req).unwrap();
        assert_eq!(out.text, "two one");
        assert_eq!(out.usage.input_tokens, 2);
        assert_eq!(events.len(), 4);
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
                    stop: vec![],
                    raw_continuation: false,
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
                    priority: 0,
                    stop: vec![],
                    raw_continuation: false,
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
                    stop: vec![],
                    raw_continuation: false,
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
                    stop: vec![],
                    raw_continuation: false,
                }
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
        let original = s.model("m").unwrap();
        assert_eq!(
            s.register_model(original.clone()).unwrap().epoch,
            original.epoch
        );

        let second_path = d.join("second.gguf");
        fs::write(&second_path, b"second").unwrap();
        s.register_model(ModelRecord {
            id: "second".into(),
            revision: "r2".into(),
            path: second_path,
            sha256: hex_digest(b"second"),
            aliases: vec!["latest".into()],
            family: "gemma-4-e2b-it".into(),
            size_bytes: 6,
            epoch: 0,
        })
        .unwrap();
        s.alias_model("m", "stable".into()).unwrap();
        assert_eq!(s.model("stable").unwrap().id, "m");
        assert_eq!(s.model("latest").unwrap().id, "second");
        assert!(
            s.model("m")
                .unwrap()
                .aliases
                .iter()
                .all(|alias| alias != "latest")
        );
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
        assert!(spec["paths"]["/native/status"].is_object());
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
            ("GET", "/native/status", json!(null), StatusCode::OK),
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
            let response = app
                .clone()
                .oneshot(request(method, uri, body))
                .await
                .unwrap();
            if response.status() != expected {
                let bytes = to_bytes(response.into_body(), usize::MAX).await.unwrap();
                panic!("{method} {uri}: {}", String::from_utf8_lossy(&bytes));
            }
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
        to_bytes(streamed.into_body(), usize::MAX).await.unwrap();
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

    async fn wait_for_queue(server: &Server, queued: usize) {
        tokio::time::timeout(Duration::from_secs(1), async {
            loop {
                if server.admission_metrics().queued == queued {
                    break;
                }
                tokio::task::yield_now().await;
            }
        })
        .await
        .unwrap();
    }

    #[tokio::test]
    async fn bounded_admission_is_fifo() {
        let (server, dir) = setup(Arc::new(AnonymousAdmin));
        server
            .configure(ServerConfig {
                active_requests: 1,
                queue_count: 2,
                queue_bytes: 16,
                request_bytes: 16,
                stream_buffer: 2,
                shutdown_grace_ms: 100,
                ..ServerConfig::default()
            })
            .unwrap();
        let first = server
            .reserve_admission(
                "first",
                Arc::new(RequestControl::new()),
                1,
                Duration::from_secs(1),
            )
            .await
            .unwrap();

        let (second_acquired_tx, second_acquired_rx) = tokio::sync::oneshot::channel();
        let (second_release_tx, second_release_rx) = tokio::sync::oneshot::channel();
        let second_server = server.clone();
        let second = tokio::spawn(async move {
            let _admission = second_server
                .reserve_admission(
                    "second",
                    Arc::new(RequestControl::new()),
                    2,
                    Duration::from_secs(1),
                )
                .await
                .unwrap();
            second_acquired_tx.send(()).unwrap();
            second_release_rx.await.unwrap();
        });
        wait_for_queue(&server, 1).await;

        let (third_acquired_tx, mut third_acquired_rx) = tokio::sync::oneshot::channel();
        let (third_release_tx, third_release_rx) = tokio::sync::oneshot::channel();
        let third_server = server.clone();
        let third = tokio::spawn(async move {
            let _admission = third_server
                .reserve_admission(
                    "third",
                    Arc::new(RequestControl::new()),
                    3,
                    Duration::from_secs(1),
                )
                .await
                .unwrap();
            third_acquired_tx.send(()).unwrap();
            third_release_rx.await.unwrap();
        });
        wait_for_queue(&server, 2).await;

        drop(first);
        tokio::time::timeout(Duration::from_secs(1), second_acquired_rx)
            .await
            .unwrap()
            .unwrap();
        assert!(
            tokio::time::timeout(Duration::from_millis(10), &mut third_acquired_rx)
                .await
                .is_err()
        );
        second_release_tx.send(()).unwrap();
        tokio::time::timeout(Duration::from_secs(1), &mut third_acquired_rx)
            .await
            .unwrap()
            .unwrap();
        third_release_tx.send(()).unwrap();
        second.await.unwrap();
        third.await.unwrap();
        assert_eq!(server.admission_metrics(), AdmissionMetrics::default());
        fs::remove_dir_all(dir).unwrap();
    }

    #[tokio::test]
    async fn admission_limits_reject_and_timed_out_entries_are_removed() {
        let (server, dir) = setup(Arc::new(AnonymousAdmin));
        assert!(server.configure(ServerConfig::default()).is_ok());
        assert!(matches!(
            server
                .reserve_admission(
                    "large",
                    Arc::new(RequestControl::new()),
                    1024 * 1024 + 1,
                    Duration::from_secs(1),
                )
                .await,
            Err(Error::PayloadTooLarge)
        ));
        server
            .configure(ServerConfig {
                active_requests: 1,
                queue_count: 1,
                queue_bytes: 4,
                request_bytes: 8,
                stream_buffer: 1,
                shutdown_grace_ms: 100,
                ..ServerConfig::default()
            })
            .unwrap();
        let active = server
            .reserve_admission(
                "active",
                Arc::new(RequestControl::new()),
                1,
                Duration::from_secs(1),
            )
            .await
            .unwrap();
        assert!(matches!(
            server
                .reserve_admission(
                    "timeout",
                    Arc::new(RequestControl::new()),
                    2,
                    Duration::from_millis(5),
                )
                .await,
            Err(Error::Deadline)
        ));
        assert_eq!(server.admission_metrics().queued, 0);

        let queued_server = server.clone();
        let queued = tokio::spawn(async move {
            queued_server
                .reserve_admission(
                    "queued",
                    Arc::new(RequestControl::new()),
                    4,
                    Duration::from_secs(1),
                )
                .await
        });
        wait_for_queue(&server, 1).await;
        assert!(matches!(
            server
                .reserve_admission(
                    "busy",
                    Arc::new(RequestControl::new()),
                    1,
                    Duration::from_secs(1),
                )
                .await,
            Err(Error::Busy)
        ));
        queued.abort();
        assert!(matches!(queued.await, Err(error) if error.is_cancelled()));
        wait_for_queue(&server, 0).await;
        drop(active);
        assert_eq!(server.admission_metrics(), AdmissionMetrics::default());
        fs::remove_dir_all(dir).unwrap();
    }

    #[tokio::test]
    async fn disconnected_stream_releases_admission() {
        let (server, dir) = setup(Arc::new(AnonymousAdmin));
        server
            .configure(ServerConfig {
                active_requests: 1,
                queue_count: 1,
                queue_bytes: 1024,
                request_bytes: 1024,
                stream_buffer: 1,
                shutdown_grace_ms: 100,
                ..ServerConfig::default()
            })
            .unwrap();
        let admission = server
            .reserve_admission(
                "disconnect",
                Arc::new(RequestControl::new()),
                16,
                Duration::from_secs(1),
            )
            .await
            .unwrap();
        let (_, receiver) = server
            .infer_stream_reserved(
                "disconnect".into(),
                InferRequest {
                    model: "m".into(),
                    prompt: "one two three".into(),
                    max_tokens: 32,
                    context_id: None,
                    deadline_ms: None,
                    stop: vec![],
                    raw_continuation: false,
                    priority: 0,
                },
                admission,
            )
            .await
            .unwrap();
        drop(receiver);
        tokio::time::timeout(Duration::from_secs(1), async {
            while server.admission_metrics().active != 0 {
                tokio::task::yield_now().await;
            }
        })
        .await
        .unwrap();
        assert!(server.contexts().is_empty());
        fs::remove_dir_all(dir).unwrap();
    }

    #[tokio::test]
    async fn one_event_stream_buffer_applies_backpressure_without_deadlock() {
        let (server, dir) = setup(Arc::new(AnonymousAdmin));
        server
            .configure(ServerConfig {
                stream_buffer: 1,
                ..ServerConfig::default()
            })
            .unwrap();
        let admission = server
            .reserve_admission(
                "bounded-stream",
                Arc::new(RequestControl::new()),
                16,
                Duration::from_secs(1),
            )
            .await
            .unwrap();
        let (_, mut receiver) = server
            .infer_stream_reserved(
                "bounded-stream".into(),
                InferRequest {
                    model: "m".into(),
                    prompt: "one two".into(),
                    max_tokens: 3,
                    context_id: None,
                    deadline_ms: None,
                    stop: vec![],
                    raw_continuation: false,
                    priority: 0,
                },
                admission,
            )
            .await
            .unwrap();
        let events = tokio::time::timeout(Duration::from_secs(1), async {
            let mut events = Vec::new();
            while let Some(event) = receiver.recv().await {
                let terminal = matches!(
                    event,
                    StreamEvent::Finished { .. } | StreamEvent::Error { .. }
                );
                events.push(event);
                if terminal {
                    break;
                }
            }
            events
        })
        .await
        .unwrap();
        assert_eq!(
            events
                .iter()
                .filter(|event| matches!(event, StreamEvent::Token { .. }))
                .count(),
            3
        );
        assert!(matches!(events.last(), Some(StreamEvent::Finished { .. })));
        fs::remove_dir_all(dir).unwrap();
    }

    #[tokio::test]
    async fn mid_generation_cancellation_and_deadline_do_not_publish_contexts() {
        let (base, dir) = setup(Arc::new(AnonymousAdmin));
        drop(base);
        let started = Arc::new(tokio::sync::Notify::new());
        let server = Server::open(
            dir.join("state.json"),
            Arc::new(AnonymousAdmin),
            Arc::new(SlowEngine {
                started: started.clone(),
            }),
        )
        .unwrap();
        let worker = {
            let server = server.clone();
            tokio::task::spawn_blocking(move || {
                server.infer(
                    "cancel-during-generation",
                    InferRequest {
                        model: "m".into(),
                        prompt: "prompt".into(),
                        max_tokens: 8,
                        context_id: None,
                        deadline_ms: None,
                        stop: vec![],
                        raw_continuation: false,
                        priority: 0,
                    },
                )
            })
        };
        started.notified().await;
        server.cancel("cancel-during-generation");
        assert!(matches!(worker.await.unwrap(), Err(Error::Cancelled)));
        assert!(server.contexts().is_empty());
        assert!(matches!(
            server.infer(
                "deadline-during-generation",
                InferRequest {
                    model: "m".into(),
                    prompt: "prompt".into(),
                    max_tokens: 8,
                    context_id: None,
                    deadline_ms: Some(5),
                    stop: vec![],
                    raw_continuation: false,
                    priority: 0,
                },
            ),
            Err(Error::Deadline)
        ));
        assert!(server.contexts().is_empty());
        fs::remove_dir_all(dir).unwrap();
    }

    #[tokio::test]
    async fn configured_server_stops_when_shutdown_is_requested() {
        let (server, dir) = setup(Arc::new(AnonymousAdmin));
        serve_until(
            server,
            "127.0.0.1:0".parse().unwrap(),
            true,
            false,
            None,
            async {},
        )
        .await
        .unwrap();
        fs::remove_dir_all(dir).unwrap();
    }
    #[test]
    fn phase6c_defaults_match_the_fixed_operating_contract() {
        assert_eq!(
            ServerConfig::default(),
            ServerConfig {
                active_requests: 1,
                queue_count: 32,
                queue_bytes: 16 << 20,
                request_bytes: 1 << 20,
                pre_queue_concurrency: 16,
                header_bytes: 32 << 10,
                body_timeout_ms: 10_000,
                wall_time_ms: 300_000,
                active_time_ms: 240_000,
                stream_buffer: 8,
                shutdown_grace_ms: 30_000,
            }
        );
    }

    #[tokio::test]
    async fn overload_is_canonical_and_model_resolution_precedes_admission() {
        let (server, dir) = setup(Arc::new(AnonymousAdmin));
        server
            .configure(ServerConfig {
                queue_count: 1,
                queue_bytes: 1024,
                request_bytes: 1024,
                ..ServerConfig::default()
            })
            .unwrap();
        let active = server
            .reserve_admission(
                "active",
                Arc::new(RequestControl::new()),
                1,
                Duration::from_secs(1),
            )
            .await
            .unwrap();
        let queued_server = server.clone();
        let queued = tokio::spawn(async move {
            queued_server
                .reserve_admission(
                    "queued",
                    Arc::new(RequestControl::new()),
                    1,
                    Duration::from_secs(2),
                )
                .await
        });
        wait_for_queue(&server, 1).await;

        let invalid = router(server.clone())
            .oneshot(
                Request::post("/v1/completions")
                    .header("content-type", "application/json")
                    .body(Body::from(r#"{"model":"missing","prompt":"hello"}"#))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(invalid.status(), StatusCode::NOT_FOUND);

        let overloaded = router(server.clone())
            .oneshot(
                Request::post("/v1/completions")
                    .header("content-type", "application/json")
                    .body(Body::from(r#"{"model":"m","prompt":"hello"}"#))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(overloaded.status(), StatusCode::TOO_MANY_REQUESTS);
        assert_eq!(overloaded.headers()[RETRY_AFTER], "1");
        let body = to_bytes(overloaded.into_body(), usize::MAX).await.unwrap();
        assert_eq!(
            serde_json::from_slice::<Value>(&body).unwrap()["error"]["code"],
            "queue_overloaded"
        );

        server.cancel("queued");
        assert!(matches!(queued.await.unwrap(), Err(Error::Cancelled)));
        drop(active);
        assert_eq!(server.admission_metrics(), AdmissionMetrics::default());
        fs::remove_dir_all(dir).unwrap();
    }

    #[tokio::test]
    async fn transport_limits_reject_before_admission() {
        let (server, dir) = setup(Arc::new(AnonymousAdmin));
        server
            .configure(ServerConfig {
                header_bytes: 16,
                request_bytes: 32,
                ..ServerConfig::default()
            })
            .unwrap();
        let headers = router(server.clone())
            .oneshot(
                Request::post("/v1/completions")
                    .header("content-type", "application/json")
                    .header("x-oversized", "01234567890123456789")
                    .body(Body::from("{}"))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(
            headers.status(),
            StatusCode::REQUEST_HEADER_FIELDS_TOO_LARGE
        );
        assert_eq!(server.admission_metrics(), AdmissionMetrics::default());

        server
            .configure(ServerConfig {
                header_bytes: 1024,
                request_bytes: 32,
                ..ServerConfig::default()
            })
            .unwrap();
        let body = router(server.clone())
            .oneshot(
                Request::post("/v1/completions")
                    .header("content-type", "application/json")
                    .body(Body::from("x".repeat(33)))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(body.status(), StatusCode::PAYLOAD_TOO_LARGE);
        assert_eq!(server.admission_metrics(), AdmissionMetrics::default());

        server
            .configure(ServerConfig {
                body_timeout_ms: 5,
                ..ServerConfig::default()
            })
            .unwrap();
        let slow_body = Body::from_stream(futures_util::stream::once(async {
            tokio::time::sleep(Duration::from_millis(20)).await;
            Ok::<_, Infallible>(Bytes::from_static(b"{}"))
        }));
        let timed_out = router(server.clone())
            .oneshot(
                Request::post("/v1/completions")
                    .header("content-type", "application/json")
                    .body(slow_body)
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(timed_out.status(), StatusCode::REQUEST_TIMEOUT);
        assert_eq!(server.admission_metrics(), AdmissionMetrics::default());
        fs::remove_dir_all(dir).unwrap();
    }

    #[tokio::test]
    async fn prequeue_reconfiguration_preserves_configured_capacity_with_active_permits() {
        let (server, dir) = setup(Arc::new(AnonymousAdmin));
        server
            .configure(ServerConfig {
                pre_queue_concurrency: 3,
                ..ServerConfig::default()
            })
            .unwrap();
        let first = server.pre_queue.acquire().await.unwrap();
        let second = server.pre_queue.acquire().await.unwrap();
        assert_eq!(server.pre_queue.semaphore.available_permits(), 1);

        server
            .configure(ServerConfig {
                pre_queue_concurrency: 3,
                ..ServerConfig::default()
            })
            .unwrap();
        assert_eq!(server.pre_queue.semaphore.available_permits(), 1);

        server
            .configure(ServerConfig {
                pre_queue_concurrency: 1,
                ..ServerConfig::default()
            })
            .unwrap();
        assert_eq!(server.pre_queue.semaphore.available_permits(), 0);
        drop(first);
        assert_eq!(server.pre_queue.semaphore.available_permits(), 0);
        drop(second);
        assert_eq!(server.pre_queue.semaphore.available_permits(), 1);

        let third = server.pre_queue.acquire().await.unwrap();
        server
            .configure(ServerConfig {
                pre_queue_concurrency: 3,
                ..ServerConfig::default()
            })
            .unwrap();
        assert_eq!(server.pre_queue.semaphore.available_permits(), 2);
        drop(third);
        assert_eq!(server.pre_queue.semaphore.available_permits(), 3);
        fs::remove_dir_all(dir).unwrap();
    }

    #[tokio::test]
    async fn shutdown_rejects_new_work_and_restart_recovers_only_durable_state() {
        let (server, dir) = setup(Arc::new(AnonymousAdmin));
        let context = server.create_context().unwrap();
        let active = server
            .reserve_admission(
                "live",
                Arc::new(RequestControl::new()),
                1,
                Duration::from_secs(1),
            )
            .await
            .unwrap();
        server.begin_shutdown();
        assert!(matches!(
            server
                .reserve_admission(
                    "late",
                    Arc::new(RequestControl::new()),
                    1,
                    Duration::from_secs(1),
                )
                .await,
            Err(Error::ShuttingDown)
        ));
        server.cancel_remaining();
        assert!(matches!(active.control.check(), Err(Error::Cancelled)));
        drop(active);

        let restarted = Server::open(
            dir.join("state.json"),
            Arc::new(AnonymousAdmin),
            Arc::new(DeterministicEngine),
        )
        .unwrap();
        assert_eq!(restarted.context(&context.id).unwrap(), context);
        assert_eq!(restarted.admission_metrics(), AdmissionMetrics::default());
        assert!(!restarted.inner.lock().shutting_down);
        fs::remove_dir_all(dir).unwrap();
    }

    #[tokio::test]
    async fn http_debug_traces_redacted_json_without_credentials_or_content() {
        let (server, dir) = setup(Arc::new(AnonymousAdmin));
        let records = Arc::new(Mutex::new(Vec::new()));
        let captured = records.clone();
        let app = router_with_http_debug(
            server,
            HttpDebug::new(move |line| captured.lock().push(line.to_owned())),
        );
        let response = app
            .oneshot(
                Request::post("/v1/completions")
                    .header("authorization", "Bearer header-secret")
                    .header("content-type", "application/json")
                    .body(Body::from(
                        r#"{"model":"m","prompt":"private prompt","max_tokens":2,"api_key":"body-secret"}"#,
                    ))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::OK);
        let request_id = response.headers()["x-request-id"]
            .to_str()
            .unwrap()
            .to_owned();
        let _ = to_bytes(response.into_body(), usize::MAX).await.unwrap();

        let records = records
            .lock()
            .iter()
            .map(|line| serde_json::from_str::<Value>(line).unwrap())
            .collect::<Vec<_>>();
        assert_eq!(records.len(), 3);
        assert_eq!(records[0]["direction"], "in");
        assert_eq!(records[0]["request_id"], request_id);
        assert_eq!(records[0]["method"], "POST");
        assert_eq!(records[0]["path"], "/v1/completions");
        assert_eq!(records[0]["body"]["json"]["model"], "[REDACTED]");
        assert_eq!(records[0]["body"]["json"]["prompt"], "[REDACTED]");
        assert_eq!(records[0]["body"]["json"]["api_key"], "[REDACTED]");
        assert_eq!(records[0]["body"]["json"]["max_tokens"], 2);
        assert_eq!(records[1]["direction"], "out");
        assert_eq!(records[1]["status"], 200);
        assert!(records[1]["duration_ms"].is_number());
        assert_eq!(records[2]["direction"], "out_body");
        assert_eq!(
            records[2]["body"]["json"]["choices"][0]["text"],
            "[REDACTED]"
        );
        let serialized = serde_json::to_string(&records).unwrap();
        for secret in ["header-secret", "body-secret", "private prompt"] {
            assert!(!serialized.contains(secret));
        }
        fs::remove_dir_all(dir).unwrap();
    }

    #[tokio::test]
    async fn http_debug_observes_stream_chunks_without_buffering_the_response() {
        let (server, dir) = setup(Arc::new(AnonymousAdmin));
        let records = Arc::new(Mutex::new(Vec::new()));
        let captured = records.clone();
        let app = router_with_http_debug(
            server,
            HttpDebug::new(move |line| captured.lock().push(line.to_owned())),
        );
        let response = app
            .oneshot(
                Request::post("/v1/completions")
                    .header("content-type", "application/json")
                    .body(Body::from(
                        r#"{"model":"m","prompt":"stream secret","max_tokens":2,"stream":true}"#,
                    ))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::OK);
        assert_eq!(records.lock().len(), 2);
        let wire_body = to_bytes(response.into_body(), usize::MAX).await.unwrap();
        assert!(String::from_utf8_lossy(&wire_body).contains("data:"));

        let records = records.lock().clone();
        let body_records = records
            .iter()
            .map(|line| serde_json::from_str::<Value>(line).unwrap())
            .filter(|record| record["direction"] == "out_body")
            .collect::<Vec<_>>();
        assert!(body_records.len() >= 2);
        for (index, record) in body_records.iter().enumerate() {
            assert_eq!(record["chunk_index"], index as u64);
        }
        let serialized = records.join("\n");
        assert!(!serialized.contains("stream secret"));
        assert!(!serialized.contains("\"token\":\"world\""));
        fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn http_debug_omits_oversized_and_non_json_bodies() {
        let mut oversized = HttpBodyCapture::default();
        oversized.push(&vec![b'x'; HTTP_DEBUG_BODY_LIMIT + 1]);
        let trace = oversized.redacted(Some("application/json"));
        assert_eq!(trace["bytes"], HTTP_DEBUG_BODY_LIMIT + 1);
        assert!(trace["omitted"].as_str().unwrap().contains("exceeds"));

        let mut binary = HttpBodyCapture::default();
        binary.push(b"private binary data");
        let trace = binary.redacted(Some("application/octet-stream"));
        assert_eq!(trace["omitted"], "non-JSON body");
        assert!(!trace.to_string().contains("private binary data"));
    }

    #[tokio::test]
    async fn http_debug_runs_on_the_live_http_transport() {
        let (server, dir) = setup(Arc::new(AnonymousAdmin));
        let probe = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let address = probe.local_addr().unwrap();
        drop(probe);
        let records = Arc::new(Mutex::new(Vec::new()));
        let captured = records.clone();
        let (shutdown, shutdown_requested) = tokio::sync::oneshot::channel();
        let serving = tokio::spawn(serve_until(
            server,
            address,
            true,
            false,
            Some(HttpDebug::new(move |line| {
                captured.lock().push(line.to_owned())
            })),
            async move {
                let _ = shutdown_requested.await;
            },
        ));
        let response = tokio::task::spawn_blocking(move || {
            let mut socket = (0..100)
                .find_map(|_| match std::net::TcpStream::connect(address) {
                    Ok(socket) => Some(socket),
                    Err(_) => {
                        std::thread::sleep(Duration::from_millis(5));
                        None
                    }
                })
                .expect("HTTP server became ready");
            std::io::Write::write_all(
                &mut socket,
                b"GET /openapi.json HTTP/1.1\r\nHost: localhost\r\nConnection: close\r\n\r\n",
            )
            .unwrap();
            let mut response = String::new();
            std::io::Read::read_to_string(&mut socket, &mut response).unwrap();
            response
        })
        .await
        .unwrap();
        assert!(response.starts_with("HTTP/1.1 200 OK\r\n"));
        assert!(response.to_ascii_lowercase().contains("\r\nx-request-id: "));
        shutdown.send(()).unwrap();
        serving.await.unwrap().unwrap();
        let records = records.lock().clone();
        assert_eq!(records.len(), 3);
        assert!(records.iter().all(|line| line.contains("\"http_debug\"")));
        fs::remove_dir_all(dir).unwrap();
    }

    #[tokio::test]
    async fn wall_and_active_deadline_watchdogs_abort_owned_work() {
        let wall = Arc::new(RequestControl::new());
        let active = Arc::new(RequestControl::new());
        start_deadline_watchdogs(&wall, Duration::from_millis(5), Duration::from_secs(1));
        start_deadline_watchdogs(&active, Duration::from_secs(1), Duration::from_millis(5));
        tokio::time::sleep(Duration::from_millis(20)).await;
        assert!(matches!(wall.check(), Err(Error::Deadline)));
        assert!(matches!(active.check(), Err(Error::Deadline)));
    }
}
