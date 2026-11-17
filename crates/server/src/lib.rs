use cusco_executor::SamplingConfig;
use parking_lot::Mutex;
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use std::{
    collections::{HashMap, HashSet, VecDeque},
    path::{Path, PathBuf},
    sync::{
        Arc,
        atomic::{AtomicU8, Ordering},
    },
    time::{Duration, Instant, SystemTime, UNIX_EPOCH},
};
use thiserror::Error;

use uuid::Uuid;

mod catalog;
mod config;
mod context_strategy;
mod context_updates;
mod daemon;
mod generation;
mod generation_plan;
mod generation_tools;
mod hosted_tools;
mod lifecycle;
mod mapped;
mod prompt;
pub mod native;
pub mod management;
mod residency;
pub mod response_store;
pub mod responses;
mod scheduler;
mod structured;
mod vision;
pub(crate) use generation_tools::{
    HostedWebSearchExecution, append_tool_instructions, append_web_search_evidence,
    execute_hosted_web_search, tool_names, tool_value, validate_tools,
};
pub use catalog::{CatalogError, ModelCatalog, load_user_models};
pub use config::{
    ConfigError, DaemonConfig, HostedToolsConfig, OpenApiConfig, VisionConfig, WebSearchConfig,
};
pub use daemon::{Daemon, DaemonError};
pub use context_strategy::{
    CompactionDeclaration, CompactionNoMatchFallback, CompactionProposal, CompactionRequest,
    CompactionResult, CompactionResultReason, CompactionStrategyCatalog, CompactionStrategyInfo,
    CompactionTrigger, CreateCompactionDeclaration, WINDOW_TAIL_STRATEGY_ID,
    apply_window_tail_strategy, deterministic_strategy_id, select_strategy, strategy_catalog,
};
pub use generation::{
    FinishReason, FrontierControl, GenerationFrontier, MAX_STOP_BYTES, MAX_STOP_SEQUENCES,
    StopAlignment,
};
pub use hosted_tools::{
    DisabledHostedTools, HostedToolError, HostedToolExecutor, SearxngHostedTools, WebSearchPolicy,
    WebSearchRequest, WebSearchResult,
};
pub use mapped::{MappedEngine, MappedMetrics};
pub use residency::{ResidencyConfig, ResidencyMetrics, ResidentEngine, ResidentModelStatus};
pub use scheduler::{SchedulerMetrics, SchedulerStatus, WorkloadScheduler};
pub use vision::{AdmittedImage, ImageAdmission, VisionError};

#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(transparent)]
pub struct ContextId(String);
impl ContextId {
    fn new() -> Self { Self(Uuid::new_v4().to_string()) }
    pub fn from_string(value: impl Into<String>) -> Self { Self(value.into()) }
    pub fn as_str(&self) -> &str { &self.0 }
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
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Usage {
    pub input_tokens: usize,
    pub generated_tokens: usize,
    pub evaluated_tokens: usize,
    pub cached_tokens: usize,
    pub prefill: PrefillMetrics,
    pub model: String,
    pub model_revision: String,
    pub context_id: ContextId,
    pub compaction_result: Option<CompactionResult>,
    pub latency_ms: u128,
    pub status: String,
    pub correlation_id: String,
    pub inference_id: String,
    pub execution_session_id: String,
}
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum StreamEvent {
    Started {
        request_id: String,
        context_id: ContextId,
        correlation_id: String,
        inference_id: String,
        execution_session_id: String,
    },
    Token {
        token: String,
        index: usize,
    },
    Finished {
        reason: FinishReason,
        usage: Box<Usage>,
    },
    Error {
        #[serde(default)]
        code: String,
        message: String,
    },
}
#[derive(Clone, Copy, Debug, Default, Deserialize, Eq, Hash, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum SchedulingClass {
    Interactive,
    #[default]
    Standard,
    Batch,
}

#[derive(Clone, Copy, Debug, Default, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum PrioritySource {
    #[default]
    AdapterDefault,
    ControlledWorkload,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct SchedulingMetadata {
    pub class: SchedulingClass,
    pub source: PrioritySource,
    pub principal: String,
    pub correlation_id: String,
    pub inference_id: String,
}

impl Default for SchedulingMetadata {
    fn default() -> Self {
        Self {
            class: SchedulingClass::Standard,
            source: PrioritySource::AdapterDefault,
            principal: "anonymous-admin".into(),
            correlation_id: String::new(),
            inference_id: String::new(),
        }
    }
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
    pub compaction: Option<CompactionRequest>,
    #[serde(skip, default)]
    pub sampling: SamplingConfig,
    #[serde(skip, default)]
    pub grammar: Option<String>,
    #[serde(skip, default)]
    pub scheduling: SchedulingMetadata,
}
fn default_true() -> bool {
    true
}
fn default_tokens() -> usize {
    16
}
fn default_compaction_workers() -> usize {
    1
}
fn default_compaction_declarations() -> usize {
    128
}
fn default_compaction_declaration_rate() -> usize {
    32
}
fn default_compaction_lifetime_ms() -> u64 {
    3_600_000
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
    pub block_count: u32,
    #[serde(default)]
    pub epoch: u64,
}
#[derive(Clone, Debug)]
struct RuntimeState {
    contexts: HashMap<ContextId, ContextRecord>,
}
fn default_model_family() -> String {
    "gemma4".into()
}

#[derive(Clone, Copy, Debug, Deserialize, PartialEq, Eq, Serialize)]
pub struct ServerConfig {
    pub active_requests: usize,
    pub queue_count: usize,
    pub queue_bytes: usize,
    pub request_bytes: usize,
    pub pre_queue_concurrency: usize,
    #[serde(default = "default_tokens")]
    pub default_output_tokens: usize,
    pub header_bytes: usize,
    pub body_timeout_ms: u64,
    pub wall_time_ms: u64,
    pub active_time_ms: u64,
    pub stream_buffer: usize,
    pub shutdown_grace_ms: u64,
    #[serde(default = "default_true")]
    pub compaction_enabled: bool,
    #[serde(default = "default_compaction_workers")]
    pub compaction_workers: usize,
    #[serde(default = "default_compaction_declarations")]
    pub compaction_declarations: usize,
    #[serde(default = "default_compaction_declaration_rate")]
    pub compaction_declaration_rate_per_minute: usize,
    #[serde(default = "default_compaction_lifetime_ms")]
    pub compaction_declaration_lifetime_ms: u64,
}
impl Default for ServerConfig {
    fn default() -> Self {
        Self {
            active_requests: 1,
            queue_count: 32,
            queue_bytes: 16 << 20,
            request_bytes: 1 << 20,
            pre_queue_concurrency: 16,
            default_output_tokens: default_tokens(),
            header_bytes: 32 << 10,
            body_timeout_ms: 10_000,
            wall_time_ms: 300_000,
            active_time_ms: 240_000,
            stream_buffer: 8,
            shutdown_grace_ms: 30_000,
            compaction_enabled: true,
            compaction_workers: default_compaction_workers(),
            compaction_declarations: default_compaction_declarations(),
            compaction_declaration_rate_per_minute: default_compaction_declaration_rate(),
            compaction_declaration_lifetime_ms: default_compaction_lifetime_ms(),
        }
    }
}

#[derive(Clone, Copy, Debug, Deserialize, PartialEq, Eq, Serialize)]
pub struct SchedulerPolicyConfig {
    pub version: u32,
    pub interactive_weight: u32,
    pub standard_weight: u32,
    pub batch_weight: u32,
    pub deficit_refill: u32,
    pub prefill_tokens: usize,
    pub promotion_rounds: u64,
    #[serde(default)]
    pub diagnostics_enabled: bool,
    pub diagnostic_capacity: usize,
}

impl Default for SchedulerPolicyConfig {
    fn default() -> Self {
        Self {
            version: 1,
            interactive_weight: 4,
            standard_weight: 2,
            batch_weight: 1,
            deficit_refill: 1,
            prefill_tokens: 32,
            promotion_rounds: 64,
            diagnostics_enabled: false,
            diagnostic_capacity: 1024,
        }
    }
}

impl SchedulerPolicyConfig {
    fn validate(self) -> Result<Self, Error> {
        if self.version != 1 {
            return Err(Error::State(format!(
                "unsupported scheduler policy version {}",
                self.version
            )));
        }
        if self.interactive_weight == 0
            || self.standard_weight == 0
            || self.batch_weight == 0
            || self.deficit_refill == 0
            || self.prefill_tokens == 0
            || self.promotion_rounds == 0
            || self.diagnostic_capacity == 0
        {
            return Err(Error::State(
                "scheduler policy parameters must be nonzero".into(),
            ));
        }
        Ok(self)
    }
}
#[derive(Clone, Copy, Debug, Default, Deserialize, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum HttpDebugLevel {
    #[default]
    Off,
    Safe,
    Full,
}

#[derive(Clone, Copy, Debug, Default, Deserialize, PartialEq, Eq, Serialize)]
pub struct AdmissionMetrics {
    pub active: usize,
    pub queued: usize,
    pub queued_bytes: usize,
}
#[derive(Debug, Error)]
pub enum Error {
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
    #[error("invalid request: {0}")]
    BadRequest(String),
    #[error("lineage conflict: {0}")]
    Conflict(String),
    #[error("state error: {0}")]
    State(String),
}
impl Error {
    /// Stable, transport-neutral machine-readable domain error identifier.
    pub fn code(&self) -> &'static str {
        match self {
            Self::Forbidden => "forbidden",
            Self::ModelNotFound(_) => "model_not_found",
            Self::ContextNotFound => "context_not_found",
            Self::Deadline => "deadline_exceeded",
            Self::Cancelled => "cancelled",
            Self::Busy => "queue_overloaded",
            Self::ShuttingDown => "server_shutting_down",
            Self::PayloadTooLarge => "payload_too_large",
            Self::BadRequest(_) => "invalid_request",
            Self::Conflict(_) => "lineage_conflict",
            Self::State(_) => "state_error",
        }
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
    pub fn new() -> Self {
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

    pub fn cancel(&self) {
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

    pub fn expire(&self) {
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

impl Default for RequestControl {
    fn default() -> Self {
        Self::new()
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


#[derive(Clone)]
pub struct EngineRequest {
    pub model: ModelRecord,
    pub prompt: String,
    pub max_tokens: usize,
    pub prior_tokens: Vec<i32>,
    pub sampling: SamplingConfig,
    pub grammar: Option<String>,
    pub control: Arc<RequestControl>,
    pub scheduling: SchedulingMetadata,
    pub prefill_chunk_tokens: usize,
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

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum QuantumKind {
    Preparation,
    Prefill,
    Decode,
    Publication,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct QuantumObservation {
    pub kind: QuantumKind,
    pub charged_tokens: usize,
    pub context_placement: String,
    pub executor_slot_occupied: bool,
    pub transition_cost_bytes: u64,
    pub capacity_reserved_bytes: u64,
}

impl QuantumObservation {
    fn model_free(kind: QuantumKind, charged_tokens: usize) -> Self {
        Self {
            kind,
            charged_tokens,
            context_placement: "model_free".into(),
            executor_slot_occupied: false,
            transition_cost_bytes: 0,
            capacity_reserved_bytes: 0,
        }
    }
}

pub enum SessionStep {
    Progress(QuantumObservation),
    Token {
        id: i32,
        piece: Vec<u8>,
        terminal_or_control: bool,
        observation: QuantumObservation,
    },
    Finished(EngineOutput),
}

pub trait ExecutionSession: Send {
    fn step(&mut self) -> Result<SessionStep, Error>;
    fn finish(&mut self) -> Result<EngineOutput, Error>;
}

pub trait InferenceEngine: Send + Sync {
    fn start_session(&self, request: EngineRequest) -> Result<Box<dyn ExecutionSession>, Error>;

    fn generate(
        &self,
        request: EngineRequest,
        sink: &mut TokenSink<'_>,
    ) -> Result<EngineOutput, Error> {
        let mut session = self.start_session(request)?;
        loop {
            match session.step()? {
                SessionStep::Progress(_) => {}
                SessionStep::Token {
                    id,
                    piece,
                    terminal_or_control,
                    ..
                } => {
                    if sink(id, &piece, terminal_or_control)? == FrontierControl::Stop {
                        return session.finish();
                    }
                }
                SessionStep::Finished(output) => return Ok(output),
            }
        }
    }

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

struct DeterministicSession {
    request: EngineRequest,
    pieces: Vec<String>,
    index: usize,
    prepared: bool,
}

impl ExecutionSession for DeterministicSession {
    fn step(&mut self) -> Result<SessionStep, Error> {
        self.request.control.check()?;
        if !self.prepared {
            self.prepared = true;
            return Ok(SessionStep::Progress(QuantumObservation::model_free(
                QuantumKind::Preparation,
                self.pieces.len(),
            )));
        }
        if self.index >= self.request.max_tokens || self.pieces.is_empty() {
            return self.finish().map(SessionStep::Finished);
        }
        let piece = &self.pieces[self.index % self.pieces.len()];
        let piece = if self.index == 0 {
            piece.clone()
        } else {
            format!(" {piece}")
        };
        let id = -(self.index as i32) - 1;
        self.index += 1;
        Ok(SessionStep::Token {
            id,
            piece: piece.into_bytes(),
            terminal_or_control: false,
            observation: QuantumObservation::model_free(QuantumKind::Decode, 1),
        })
    }

    fn finish(&mut self) -> Result<EngineOutput, Error> {
        self.request.control.check()?;
        let input_tokens = self.pieces.len();
        Ok(EngineOutput {
            successor_tokens: self.request.prior_tokens.clone(),
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

impl InferenceEngine for DeterministicEngine {
    fn start_session(&self, request: EngineRequest) -> Result<Box<dyn ExecutionSession>, Error> {
        request.control.check()?;
        let pieces = request
            .prompt
            .split_whitespace()
            .rev()
            .map(str::to_owned)
            .collect();
        Ok(Box::new(DeterministicSession {
            request,
            pieces,
            index: 0,
            prepared: false,
        }))
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
    durable: RuntimeState,
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

    fn try_acquire(self: &Arc<Self>) -> Result<PrequeuePermit, Error> {
        let permit = self
            .semaphore
            .clone()
            .try_acquire_owned()
            .map_err(|_| Error::Busy)?;
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

struct DeclarationRecord {
    declaration: CompactionDeclaration,
    principal: String,
}

struct DeclarationState {
    records: HashMap<String, DeclarationRecord>,
    creation_times: HashMap<String, VecDeque<u64>>,
}

#[derive(Clone)]
pub struct Server {
    inner: Arc<Mutex<Inner>>,
    pre_queue: Arc<PrequeueGate>,
    compaction_workers: Arc<PrequeueGate>,
    declarations: Arc<Mutex<DeclarationState>>,
    engine: Arc<dyn InferenceEngine>,
    lifecycle: lifecycle::ModelLifecycleService,
    vision: Arc<Mutex<VisionConfig>>,
    hosted_tools: Arc<Mutex<Arc<dyn HostedToolExecutor>>>,
    response_service: responses::ResponseService,
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
    pub fn open(path: impl AsRef<Path>, engine: Arc<dyn InferenceEngine>) -> Result<Self, Error> {
        let config = ServerConfig::default();
        let state_root = path.as_ref().parent().unwrap_or_else(|| Path::new("."));
        let response_store = response_store::FileResponseResourceStore::open(state_root)
            .map_err(|error| Error::State(error.to_string()))?;
        Ok(Self {
            inner: Arc::new(Mutex::new(Inner {
                durable: RuntimeState {
                    contexts: HashMap::new(),
                },
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
            compaction_workers: PrequeueGate::new(config.compaction_workers),
            declarations: Arc::new(Mutex::new(DeclarationState {
                records: HashMap::new(),
                creation_times: HashMap::new(),
            })),
            lifecycle: lifecycle::ModelLifecycleService::new(engine.clone()),
            engine,
            vision: Arc::new(Mutex::new(VisionConfig::default())),
            hosted_tools: Arc::new(Mutex::new(hosted_tools::disabled_hosted_tools())),
            response_service: responses::ResponseService::new(Arc::new(response_store)),
        })
    }
    pub fn attach_catalog(&self, catalog: ModelCatalog, model_directory: impl Into<PathBuf>) {
        self.lifecycle
            .attach_catalog(catalog, model_directory.into());
    }
    fn catalog(&self) -> Option<ModelCatalog> {
        self.lifecycle.catalog()
    }
    pub fn configure_vision(&self, config: VisionConfig) {
        *self.vision.lock() = config;
    }
    pub fn configure_hosted_tools(&self, executor: Arc<dyn HostedToolExecutor>) {
        *self.hosted_tools.lock() = executor;
    }
    fn vision_config(&self) -> VisionConfig {
        self.vision.lock().clone()
    }
    fn model_directory(&self) -> PathBuf {
        self.lifecycle.model_directory()
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
        Ok(record)
    }
    pub fn delete_context(&self, id: &ContextId) -> Result<(), Error> {
        self.inner
            .lock()
            .durable
            .contexts
            .remove(id)
            .ok_or(Error::ContextNotFound)?;
        Ok(())
    }
    pub fn create_compaction_declaration(
        &self,
        request: CreateCompactionDeclaration,
    ) -> Result<CompactionDeclaration, Error> {
        self.create_compaction_declaration_for("local", request)
    }

    fn create_compaction_declaration_for(
        &self,
        principal: &str,
        request: CreateCompactionDeclaration,
    ) -> Result<CompactionDeclaration, Error> {
        let config = self.config();
        if !config.compaction_enabled {
            return Err(Error::BadRequest("compaction is disabled".into()));
        }
        if request.expires_in_ms == 0
            || request.expires_in_ms > config.compaction_declaration_lifetime_ms
        {
            return Err(Error::BadRequest(
                "invalid compaction declaration lifetime".into(),
            ));
        }

        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map_err(|_| Error::State("system clock before unix epoch".into()))?
            .as_millis() as u64;
        let declaration = CompactionDeclaration {
            declaration_id: Uuid::new_v4().to_string(),
            context_id: request.context_id,
            strategy_preferences: request.strategy_preferences,
            target_tokens: request.target_tokens,
            expires_at_ms: now.saturating_add(request.expires_in_ms),
        };

        let mut state = self.declarations.lock();
        state
            .records
            .retain(|_, record| record.declaration.expires_at_ms > now);
        let stale_cutoff_ms = now.saturating_sub(60_000);
        state.creation_times.retain(|_, recent| {
            while recent
                .front()
                .is_some_and(|created| now.saturating_sub(*created) >= 60_000)
            {
                recent.pop_front();
            }
            !recent.is_empty()
        });
        if state.records.len() >= config.compaction_declarations {
            return Err(Error::Busy);
        }
        let recent = state
            .creation_times
            .entry(principal.to_owned())
            .or_default();
        while recent
            .front()
            .is_some_and(|created| *created <= stale_cutoff_ms)
        {
            recent.pop_front();
        }
        if recent.len() >= config.compaction_declaration_rate_per_minute {
            return Err(Error::Busy);
        }
        recent.push_back(now);
        state.records.insert(
            declaration.declaration_id.clone(),
            DeclarationRecord {
                declaration: declaration.clone(),
                principal: principal.to_owned(),
            },
        );
        Ok(declaration)
    }

    pub fn compaction_declaration(&self, id: &str) -> Result<CompactionDeclaration, Error> {
        self.compaction_declaration_for("local", id)
    }

    fn compaction_declaration_for(
        &self,
        principal: &str,
        id: &str,
    ) -> Result<CompactionDeclaration, Error> {
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map_err(|_| Error::State("system clock before unix epoch".into()))?
            .as_millis() as u64;
        let mut state = self.declarations.lock();
        let record = state
            .records
            .get(id)
            .ok_or_else(|| Error::BadRequest("compaction declaration not found".into()))?;
        if record.declaration.expires_at_ms <= now {
            state.records.remove(id);
            return Err(Error::BadRequest("compaction declaration expired".into()));
        }
        if record.principal != principal {
            return Err(Error::Forbidden);
        }
        Ok(record.declaration.clone())
    }

    fn revoke_compaction_declaration_for(&self, principal: &str, id: &str) -> Result<(), Error> {
        let mut state = self.declarations.lock();
        let record = state
            .records
            .get(id)
            .ok_or_else(|| Error::BadRequest("compaction declaration not found".into()))?;
        if record.principal != principal {
            return Err(Error::Forbidden);
        }
        state.records.remove(id);
        Ok(())
    }
    pub fn register_model(&self, model: ModelRecord) -> Result<ModelRecord, Error> {
        self.lifecycle.register(model, |_| Ok(()))
    }

    pub fn register_catalog_model(&self, model: ModelRecord) -> Result<ModelRecord, Error> {
        self.lifecycle.register_from_catalog(model)
    }

    pub fn register_model_with<F>(
        &self,
        finalize: F,
        model: ModelRecord,
    ) -> Result<ModelRecord, Error>
    where
        F: FnOnce(&ModelRecord) -> Result<(), Error>,
    {
        self.lifecycle.register(model, finalize)
    }
    fn effective_stop_sequences(&self, req: &InferRequest) -> Result<Vec<String>, Error> {
        let mut stops = req.stop.clone();
        let model = self.model(&req.model)?;
        for marker in prompt::terminal_markers(&model.family) {
            if !stops.iter().any(|stop| stop == marker) {
                stops.push((*marker).into());
            }
        }
        Ok(stops)
    }

    pub fn models(&self) -> Vec<ModelRecord> {
        self.lifecycle.models()
    }
    pub fn model(&self, id: &str) -> Result<ModelRecord, Error> {
        self.lifecycle.model(id)
    }
    pub fn alias_model(&self, id: &str, alias: String) -> Result<ModelRecord, Error> {
        self.lifecycle.alias(id, alias)
    }
    pub fn remove_model(&self, id: &str) -> Result<(), Error> {
        self.lifecycle.remove(id)
    }
    pub fn verify_model(&self, id: &str) -> Result<bool, Error> {
        self.lifecycle.verify(id)
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
            || config.default_output_tokens == 0
            || config.header_bytes == 0
            || config.body_timeout_ms == 0
            || config.wall_time_ms == 0
            || config.active_time_ms == 0
            || config.stream_buffer == 0
            || config.shutdown_grace_ms == 0
            || config.compaction_workers == 0
            || config.compaction_declarations == 0
            || config.compaction_declaration_rate_per_minute == 0
            || config.compaction_declaration_lifetime_ms == 0
        {
            return Err(Error::State("server limits must be nonzero".into()));
        }
        let mut guard = self.inner.lock();
        self.pre_queue.set_limit(config.pre_queue_concurrency);
        self.compaction_workers.set_limit(config.compaction_workers);
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
    pub(crate) fn begin_shutdown(&self) {
        self.inner.lock().shutting_down = true;
    }

    pub(crate) fn cancel_remaining(&self) {
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
        let stops = self.effective_stop_sequences(&req)?;
        let frontier =
            GenerationFrontier::new_effective(&stops, req.raw_continuation).map_err(state_err)?;
        let successor_id = req.context_id.clone().unwrap_or_else(ContextId::new);
        let execution_session_id = req.scheduling.inference_id.clone();
        let mut events = vec![StreamEvent::Started {
            request_id: request_id.into(),
            context_id: successor_id.clone(),
            correlation_id: req.scheduling.correlation_id.clone(),
            inference_id: req.scheduling.inference_id.clone(),
            execution_session_id,
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
        let (started, receiver, _worker) = self.infer_stream_reserved_with_compaction(request_id, req, admission, None).await?;
        Ok((started, receiver))
    }
    pub(crate) async fn infer_stream_reserved_with_compaction(
        &self,
        request_id: String,
        req: InferRequest,
        admission: AdmissionGuard,
        compaction_permit: Option<PrequeuePermit>,
    ) -> Result<(StreamEvent, tokio::sync::mpsc::Receiver<StreamEvent>, tokio::task::JoinHandle<Result<InferResponse, Error>>), Error> {
        let stops = self.effective_stop_sequences(&req)?;
        let frontier =
            GenerationFrontier::new_effective(&stops, req.raw_continuation).map_err(state_err)?;
        let successor_id = req.context_id.clone().unwrap_or_else(ContextId::new);
        let started = StreamEvent::Started {
            request_id: request_id.clone(),
            context_id: successor_id.clone(),
            correlation_id: req.scheduling.correlation_id.clone(),
            inference_id: req.scheduling.inference_id.clone(),
            execution_session_id: req.scheduling.inference_id.clone(),
        };
        let stream_buffer = self.inner.lock().config.stream_buffer;
        let (sender, receiver) = tokio::sync::mpsc::channel(stream_buffer);
        let server = self.clone();
        let control = admission.control.clone();
        let worker = tokio::task::spawn_blocking(move || {
            let _admission = admission;
            let _compaction_permit = compaction_permit;
            let emit_control = control.clone();
            match server.infer_admitted(&request_id, req, successor_id, frontier, |event| {
                sender.blocking_send(event).map_err(|_| {
                    emit_control.cancel();
                    Error::Cancelled
                })
            }) {
                Ok((response, terminal)) => {
                    let _ = sender.blocking_send(terminal);
                    Ok(response)
                }
                Err(error) => {
                    let terminal = StreamEvent::Error {
                        code: error.code().to_owned(),
                        message: error.to_string(),
                    };
                    let _ = sender.blocking_send(terminal);
                    Err(error)
                }
            }
        });
        Ok((started, receiver, worker))
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
        mut req: InferRequest,
        successor_id: ContextId,
        mut frontier: GenerationFrontier,
        mut emit: impl FnMut(StreamEvent) -> Result<(), Error>,
    ) -> Result<(InferResponse, StreamEvent), Error> {
        if let Some(compaction) = req.compaction.as_mut() {
            if let Some(declaration_id) = compaction.declaration_id.clone() {
                let declaration =
                    self.compaction_declaration_for(&req.scheduling.principal, &declaration_id)?;
                let context_id = req.context_id.as_ref().ok_or_else(|| {
                    Error::BadRequest("compaction declaration requires context".into())
                })?;
                if context_id.0 != declaration.context_id {
                    return Err(Error::BadRequest(
                        "compaction declaration context mismatch".into(),
                    ));
                }
                compaction.strategy_preferences = declaration.strategy_preferences;
                if compaction.target_tokens.is_none() {
                    compaction.target_tokens = declaration.target_tokens;
                }
            }
        }
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
        let mut logical_context_tokens: Vec<String> = context
            .as_ref()
            .map_or_else(Vec::new, |record| record.tokens.clone());
        let mut prior_tokens = context
            .as_ref()
            .map_or_else(Vec::new, |record| record.native_tokens.clone());
        let mut compaction_result = None;
        if let Some(compaction) = req.compaction.take() {
            let source = context.as_ref().ok_or_else(|| {
                Error::BadRequest("compaction requires an existing context".into())
            })?;
            let selected = match select_strategy(&compaction.strategy_preferences) {
                Some(selected) => selected,
                None => {
                    if matches!(
                        compaction.fallback_when_no_match,
                        CompactionNoMatchFallback::Reject
                    ) {
                        return Err(Error::BadRequest("unsupported_compaction_strategy".into()));
                    }
                    compaction_result = Some(CompactionResult {
                        compact_mode: "window_tail".into(),
                        compact_reason: CompactionResultReason::None,
                        requested_strategy_ids: compaction.strategy_preferences,
                        selected_strategy_id: None,
                        requested_strategy_budget: compaction.target_tokens.unwrap_or(3072),
                        resulting_budget: logical_context_tokens.len(),
                        retained_indices: (0..logical_context_tokens.len()).collect(),
                        retained_message_count: logical_context_tokens.len(),
                        source_message_count: logical_context_tokens.len(),
                        retained_anchor_count: 0,
                        success: false,
                        fallback: true,
                        resulting_context_epoch: source.revision.saturating_add(1),
                    });
                    String::new()
                }
            };
            if !selected.is_empty() {
                if selected != WINDOW_TAIL_STRATEGY_ID {
                    return Err(Error::BadRequest(
                        "unsupported compaction strategy for this release".into(),
                    ));
                }
                let proposal = apply_window_tail_strategy(
                    &logical_context_tokens,
                    compaction.target_tokens.unwrap_or(3072),
                    &[selected.clone()],
                )
                .map_err(Error::BadRequest)?;
                control.check()?;
                let mut result = proposal.result;
                result.selected_strategy_id = Some(deterministic_strategy_id(&selected));
                result.resulting_context_epoch = source.revision.saturating_add(1);
                if proposal.resulting_tokens != logical_context_tokens {
                    let prepared = self.engine.generate(
                        EngineRequest {
                            model: model.clone(),
                            prompt: proposal.resulting_tokens.join(" "),
                            max_tokens: 0,
                            prior_tokens: Vec::new(),
                            sampling: req.sampling,
                            grammar: req.grammar.clone(),
                            control: control.clone(),
                            scheduling: req.scheduling.clone(),
                            prefill_chunk_tokens: 32,
                        },
                        &mut |_, _, _| {
                            Err(Error::State(
                                "compaction preparation unexpectedly generated output".into(),
                            ))
                        },
                    )?;
                    control.check()?;
                    logical_context_tokens = proposal.resulting_tokens;
                    prior_tokens = prepared.successor_tokens;
                }
                compaction_result = Some(result);
            }
        }
        let mut delta_index = 0;
        let generated = self.engine.generate(
            EngineRequest {
                model: model.clone(),
                prompt: req.prompt.clone(),
                max_tokens: req.max_tokens,
                prior_tokens,
                sampling: req.sampling,
                grammar: req.grammar,
                control: control.clone(),
                scheduling: req.scheduling.clone(),
                prefill_chunk_tokens: 32,
            },
            &mut |_id, piece, terminal_or_control| {
                if deadline.is_some_and(|deadline| Instant::now() >= deadline) {
                    control.expire();
                }
                control.check()?;
                frontier.push(piece, terminal_or_control, |delta| {
                    emit(StreamEvent::Token {
                        token: delta.to_owned(),
                        index: delta_index,
                    })?;
                    delta_index += 1;
                    Ok(())
                })
            },
        )?;
        let frontier = frontier.finish(|delta| {
            emit(StreamEvent::Token {
                token: delta.to_owned(),
                index: delta_index,
            })?;
            delta_index += 1;
            Ok(())
        })?;
        control.check()?;
        if req
            .deadline_ms
            .is_some_and(|ms| started.elapsed() > Duration::from_millis(ms))
        {
            return Err(Error::Deadline);
        }
        logical_context_tokens.extend(req.prompt.split_whitespace().map(str::to_owned));
        logical_context_tokens.extend(frontier.text.split_whitespace().map(str::to_owned));
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
            if stored.revision != context.revision {
                return Err(Error::State("context changed during compaction".into()));
            }
            stored.tokens = logical_context_tokens;
            stored.native_tokens = successor_tokens;
            stored.revision += 1;
            stored.id.clone()
        } else {
            guard.durable.contexts.insert(
                successor_id.clone(),
                ContextRecord {
                    id: successor_id.clone(),
                    revision: 1,
                    tokens: logical_context_tokens,
                    native_tokens: successor_tokens,
                },
            );
            successor_id
        };
        drop(guard);
        let usage = Usage {
            input_tokens,
            generated_tokens: frontier.generated_tokens,
            evaluated_tokens,
            cached_tokens,
            prefill,
            model: model.id,
            model_revision: model.revision,
            context_id: context_id.clone(),
            compaction_result,
            latency_ms: started.elapsed().as_millis(),
            status: "completed".into(),
            correlation_id: req.scheduling.correlation_id.clone(),
            inference_id: req.scheduling.inference_id.clone(),
            execution_session_id: req.scheduling.inference_id.clone(),
        };
        Ok((
            InferResponse {
                id: request_id.into(),
                text: frontier.text,
                usage: usage.clone(),
            },
            StreamEvent::Finished {
                reason: frontier.finish_reason,
                usage: Box::new(usage),
            },
        ))
    }
}

fn state_err(error: impl std::fmt::Display) -> Error {
    Error::State(error.to_string())
}
#[cfg(test)]
fn hex_digest(bytes: &[u8]) -> String {
    use sha2::{Digest, Sha256};
    format!("{:x}", Sha256::digest(bytes))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    fn dir() -> PathBuf {
        std::env::temp_dir().join(format!("cusco-server-{}", Uuid::new_v4()))
    }
    fn setup() -> (Server, PathBuf) {
        setup_engine(Arc::new(DeterministicEngine))
    }
    fn setup_engine(engine: Arc<dyn InferenceEngine>) -> (Server, PathBuf) {
        let d = dir();
        fs::create_dir_all(&d).unwrap();
        let model = d.join("m.gguf");
        fs::write(&model, b"model").unwrap();
        let s = Server::open(d.join("state.json"), engine).unwrap();
        s.register_model(ModelRecord {
            id: "m".into(),
            revision: "r1".into(),
            path: model,
            sha256: hex_digest(b"model"),
            aliases: vec!["latest".into()],
            family: "gemma4".into(),
            size_bytes: 5,
            block_count: 1,
            epoch: 0,
        })
        .unwrap();
        (s, d)
    }

    #[test]
    fn gemma_effective_stops_include_terminal_markers() {
        let (server, directory) = setup();
        let request = InferRequest {
            model: "m".into(),
            prompt: "hello".into(),
            max_tokens: 1,
            context_id: None,
            compaction: None,
            deadline_ms: None,
            scheduling: SchedulingMetadata::default(),
            stop: vec!["custom".into()],
            raw_continuation: false,
            sampling: SamplingConfig::default(),
            grammar: None,
        };

        assert_eq!(
            server.effective_stop_sequences(&request).unwrap(),
            vec![
                "custom",
                "<end_of_turn>",
                "</end_of_turn>",
                "</start_of_turn>",
            ]
        );
        fs::remove_dir_all(directory).unwrap();
    }

    struct CompactionProofEngine;

    struct CompactionProofSession {
        request: EngineRequest,
    }

    impl ExecutionSession for CompactionProofSession {
        fn step(&mut self) -> Result<SessionStep, Error> {
            self.finish().map(SessionStep::Finished)
        }

        fn finish(&mut self) -> Result<EngineOutput, Error> {
            self.request.control.check()?;
            if self.request.max_tokens != 0 && self.request.prompt == "fail continuation" {
                return Err(Error::State("continuation failed after compaction".into()));
            }
            let successor_tokens = if self.request.max_tokens == 0 {
                assert!(self.request.prior_tokens.is_empty());
                assert!(self.request.prompt.contains("policy"));
                vec![41, 42]
            } else {
                assert_eq!(self.request.prior_tokens, vec![41, 42]);
                vec![41, 42, 43]
            };
            Ok(EngineOutput {
                successor_tokens,
                input_tokens: self.request.prompt.split_whitespace().count(),
                cached_tokens: usize::from(self.request.max_tokens != 0) * 2,
                evaluated_tokens: self.request.prompt.split_whitespace().count(),
                prefill: PrefillMetrics::default(),
            })
        }
    }

    impl InferenceEngine for CompactionProofEngine {
        fn start_session(
            &self,
            request: EngineRequest,
        ) -> Result<Box<dyn ExecutionSession>, Error> {
            Ok(Box::new(CompactionProofSession { request }))
        }
    }

    struct FailingEngine;
    impl InferenceEngine for FailingEngine {
        fn start_session(&self, _: EngineRequest) -> Result<Box<dyn ExecutionSession>, Error> {
            Err(Error::State("generation failed".into()))
        }
    }


    struct SlowEngine {
        started: Arc<tokio::sync::Notify>,
    }

    struct SlowSession {
        request: EngineRequest,
        started: Arc<tokio::sync::Notify>,
        index: usize,
    }

    impl SlowSession {
        fn output(&self) -> EngineOutput {
            let input_tokens = self.request.prompt.split_whitespace().count();
            EngineOutput {
                successor_tokens: (0..self.request.max_tokens as i32).collect(),
                input_tokens,
                cached_tokens: 0,
                evaluated_tokens: input_tokens,
                prefill: PrefillMetrics {
                    total_tokens: input_tokens,
                    uncached_tokens: input_tokens,
                    ..PrefillMetrics::default()
                },
            }
        }
    }

    impl ExecutionSession for SlowSession {
        fn step(&mut self) -> Result<SessionStep, Error> {
            self.request.control.check()?;
            if self.index >= self.request.max_tokens {
                return Ok(SessionStep::Finished(self.output()));
            }
            let id = self.index as i32;
            self.index += 1;
            if id == 0 {
                self.started.notify_one();
            }
            std::thread::sleep(Duration::from_millis(20));
            Ok(SessionStep::Token {
                id,
                piece: b"x".to_vec(),
                terminal_or_control: false,
                observation: QuantumObservation::model_free(QuantumKind::Decode, 1),
            })
        }

        fn finish(&mut self) -> Result<EngineOutput, Error> {
            Ok(self.output())
        }
    }
    impl InferenceEngine for SlowEngine {
        fn start_session(
            &self,
            request: EngineRequest,
        ) -> Result<Box<dyn ExecutionSession>, Error> {
            Ok(Box::new(SlowSession {
                request,
                started: self.started.clone(),
                index: 0,
            }))
        }
    }
    #[test]
    fn contexts_are_disposable_across_restart() {
        let (server, directory) = setup();
        let context = server.create_context().unwrap();
        server.branch_context(&context.id).unwrap();
        drop(server);
        let restarted = Server::open(directory.join("state.json"), Arc::new(DeterministicEngine))
        .unwrap();
        assert!(restarted.contexts().is_empty());
        assert!(matches!(
            restarted.context(&context.id),
            Err(Error::ContextNotFound)
        ));
        fs::remove_dir_all(directory).unwrap()
    }
    #[test]
    fn inference_usage_events_deadline_and_cancel_are_transactional() {
        let (s, d) = setup();
        let req = InferRequest {
            model: "latest".into(),
            prompt: "one two".into(),
            max_tokens: 2,
            context_id: None,
            compaction: None,
            deadline_ms: Some(1000),
            scheduling: SchedulingMetadata::default(),
            stop: vec![],
            raw_continuation: false,
            sampling: SamplingConfig::default(),
            grammar: None,
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
                    compaction: None,
                    deadline_ms: None,
                    scheduling: SchedulingMetadata::default(),
                    stop: vec![],
                    raw_continuation: false,
                    sampling: SamplingConfig::default(),
                    grammar: None,
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
                    compaction: None,
                    deadline_ms: Some(0),
                    scheduling: SchedulingMetadata::default(),
                    stop: vec![],
                    raw_continuation: false,
                    sampling: SamplingConfig::default(),
                    grammar: None,
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
                    compaction: None,
                    deadline_ms: None,
                    scheduling: SchedulingMetadata::default(),
                    stop: vec![],
                    raw_continuation: false,
                    sampling: SamplingConfig::default(),
                    grammar: None,
                },
            ),
            Err(Error::Busy)
        ));
        let model = s.model("m").unwrap();
        drop(s);
        let failing = Server::open(d.join("state.json"), Arc::new(FailingEngine))
        .unwrap();
        failing.register_model(model).unwrap();
        assert!(matches!(
            failing.infer(
                "failed",
                InferRequest {
                    model: "m".into(),
                    prompt: "x".into(),
                    max_tokens: 1,
                    compaction: None,
                    context_id: None,
                    deadline_ms: None,
                    scheduling: SchedulingMetadata::default(),
                    stop: vec![],
                    raw_continuation: false,
                    sampling: SamplingConfig::default(),
                    grammar: None,
                }
            ),
            Err(Error::State(_))
        ));
        assert!(failing.contexts().is_empty());
        fs::remove_dir_all(d).unwrap()
    }
    #[test]
    fn model_lifecycle_preserves_aliases_and_revisions() {
        let (s, d) = setup();
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
            family: "gemma4".into(),
            size_bytes: 6,
            block_count: 1,
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
        fs::remove_dir_all(d).unwrap()
    }

    #[test]
    fn responses_tool_instructions_remain_inside_the_user_turn() {
        let mut prompt = prompt::apply_chat_template(
            "gemma4",
            vec![prompt::Message {
                role: "user".into(),
                content: "Use the tool.".into(),
            }],
        )
        .unwrap();
        append_tool_instructions(
            "gemma4",
            &mut prompt,
            &[native::NativeToolDefinition {
                kind: native::NativeToolKind::Function,
                name: Some("lookup".into()),
                description: None,
                parameters: Some(json!({"type": "object"})),
                strict: false,
                allowed_domains: Vec::new(),
                search_context_size: None,
            }],
            &native::NativeToolChoice::Function { name: "lookup".into() },
            false,
        )
        .unwrap();
        assert!(
            prompt.contains(
                "You must call the `lookup` function.<end_of_turn>\n<start_of_turn>model\n"
            )
        );
        assert!(!prompt.ends_with("function."));
    }
    #[test]
    fn responses_tool_result_instructions_prevent_duplicate_calls() {
        let mut prompt = "<start_of_turn>user\nTool result for call_1: blue<end_of_turn>\n\
             <start_of_turn>model\n"
            .to_string();
        append_tool_instructions(
            "gemma4",
            &mut prompt,
            &[native::NativeToolDefinition {
                kind: native::NativeToolKind::Function,
                name: Some("lookup".into()),
                description: None,
                parameters: Some(json!({"type": "object"})),
                strict: false,
                allowed_domains: Vec::new(),
                search_context_size: None,
            }],
            &native::NativeToolChoice::Auto,
            true,
        )
        .unwrap();
        assert!(
            prompt.contains(
                "Use it to answer the user; do not repeat a completed call.<end_of_turn>"
            )
        );
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
        let (server, dir) = setup();
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
        let (server, dir) = setup();
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
        let (server, dir) = setup();
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
                    compaction: None,
                    deadline_ms: None,
                    stop: vec![],
                    raw_continuation: false,
                    sampling: SamplingConfig::default(),
                    grammar: None,
                    scheduling: SchedulingMetadata::default(),
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
        let (server, dir) = setup();
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
                    compaction: None,
                    context_id: None,
                    deadline_ms: None,
                    stop: vec![],
                    raw_continuation: false,
                    sampling: SamplingConfig::default(),
                    grammar: None,
                    scheduling: SchedulingMetadata::default(),
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
        let started = Arc::new(tokio::sync::Notify::new());
        let (server, dir) = setup_engine(Arc::new(SlowEngine {
            started: started.clone(),
        }));
        let worker = {
            let server = server.clone();
            tokio::task::spawn_blocking(move || {
                server.infer(
                    "cancel-during-generation",
                    InferRequest {
                        model: "m".into(),
                        prompt: "prompt".into(),
                        max_tokens: 8,
                        compaction: None,
                        context_id: None,
                        deadline_ms: None,
                        stop: vec![],
                        raw_continuation: false,
                        sampling: SamplingConfig::default(),
                        grammar: None,
                        scheduling: SchedulingMetadata::default(),
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
                    compaction: None,
                    deadline_ms: Some(5),
                    stop: vec![],
                    raw_continuation: false,
                    sampling: SamplingConfig::default(),
                    grammar: None,
                    scheduling: SchedulingMetadata::default(),
                },
            ),
            Err(Error::Deadline)
        ));
        assert!(server.contexts().is_empty());
        fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn defaults_match_the_fixed_operating_contract() {
        assert_eq!(
            ServerConfig::default(),
            ServerConfig {
                active_requests: 1,
                queue_count: 32,
                queue_bytes: 16 << 20,
                request_bytes: 1 << 20,
                pre_queue_concurrency: 16,
                default_output_tokens: default_tokens(),
                header_bytes: 32 << 10,
                body_timeout_ms: 10_000,
                wall_time_ms: 300_000,
                active_time_ms: 240_000,
                stream_buffer: 8,
                shutdown_grace_ms: 30_000,
                compaction_enabled: true,
                compaction_workers: default_compaction_workers(),
                compaction_declarations: default_compaction_declarations(),
                compaction_declaration_rate_per_minute: default_compaction_declaration_rate(),
                compaction_declaration_lifetime_ms: default_compaction_lifetime_ms(),
            }
        );
    }

    #[tokio::test]
    async fn prequeue_reconfiguration_preserves_configured_capacity_with_active_permits() {
        let (server, dir) = setup();
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
    async fn shutdown_rejects_new_work_and_restart_discards_runtime_state() {
        let (server, dir) = setup();
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

        let restarted = Server::open(dir.join("state.json"), Arc::new(DeterministicEngine))
        .unwrap();
        assert!(restarted.contexts().is_empty());
        assert!(matches!(
            restarted.context(&context.id),
            Err(Error::ContextNotFound)
        ));
        assert_eq!(restarted.admission_metrics(), AdmissionMetrics::default());
        assert!(!restarted.inner.lock().shutting_down);
        fs::remove_dir_all(dir).unwrap();
    }

    #[tokio::test]
    async fn wall_and_active_deadline_watchdogs_abort_owned_work() {
        let wall = Arc::new(RequestControl::new());
        let active = Arc::new(RequestControl::new());
        native::start_deadline_watchdogs(&wall, Duration::from_millis(5), Duration::from_secs(1));
        native::start_deadline_watchdogs(&active, Duration::from_secs(1), Duration::from_millis(5));
        tokio::time::sleep(Duration::from_millis(100)).await;
        assert!(matches!(wall.check(), Err(Error::Deadline)));
        assert!(matches!(active.check(), Err(Error::Deadline)));
    }

    #[test]
    fn compaction_declarations_are_context_bound_and_expire() {
        let (server, directory) = setup();
        let context = server.create_context().unwrap();
        let declaration = server
            .create_compaction_declaration(CreateCompactionDeclaration {
                context_id: context.id.0.clone(),
                strategy_preferences: vec![WINDOW_TAIL_STRATEGY_ID.into()],
                target_tokens: Some(8),
                expires_in_ms: 1_000,
            })
            .unwrap();
        assert_eq!(
            server
                .compaction_declaration(&declaration.declaration_id)
                .unwrap(),
            declaration
        );
        assert!(matches!(
            server.compaction_declaration_for("another-principal", &declaration.declaration_id),
            Err(Error::Forbidden)
        ));
        let mut unauthorized = compacting_request(context.id.clone());
        unauthorized.scheduling.principal = "another-principal".into();
        unauthorized.compaction.as_mut().unwrap().declaration_id =
            Some(declaration.declaration_id.clone());
        assert!(matches!(
            server.infer("declaration-owner", unauthorized),
            Err(Error::Forbidden)
        ));
        assert_eq!(server.context(&context.id).unwrap(), context);
        server
            .revoke_compaction_declaration_for("local", &declaration.declaration_id)
            .unwrap();
        assert!(matches!(
            server.compaction_declaration(&declaration.declaration_id),
            Err(Error::BadRequest(message)) if message == "compaction declaration not found"
        ));
        let mismatch = CompactionRequest {
            declaration_id: Some(declaration.declaration_id),
            ..CompactionRequest::default()
        };
        assert_eq!(mismatch.target_tokens, None);
        drop(server);
        fs::remove_dir_all(directory).unwrap();
    }

    #[test]
    fn compaction_operator_limits_bound_workers_declarations_and_rate() {
        let (server, directory) = setup();
        let mut config = ServerConfig {
            compaction_workers: 1,
            compaction_declarations: 1,
            compaction_declaration_rate_per_minute: 1,
            compaction_declaration_lifetime_ms: 100,
            ..ServerConfig::default()
        };
        server.configure(config).unwrap();
        let worker = server.compaction_workers.try_acquire().unwrap();
        assert!(matches!(
            server.compaction_workers.try_acquire(),
            Err(Error::Busy)
        ));
        drop(worker);
        assert!(server.compaction_workers.try_acquire().is_ok());

        let context = server.create_context().unwrap();
        let declaration = CreateCompactionDeclaration {
            context_id: context.id.0,
            strategy_preferences: vec![WINDOW_TAIL_STRATEGY_ID.into()],
            target_tokens: Some(8),
            expires_in_ms: 100,
        };
        server
            .create_compaction_declaration_for("principal-a", declaration.clone())
            .unwrap();
        assert!(matches!(
            server.create_compaction_declaration_for("principal-a", declaration.clone()),
            Err(Error::Busy)
        ));
        assert!(matches!(
            server.create_compaction_declaration_for("principal-b", declaration.clone()),
            Err(Error::Busy)
        ));

        config.compaction_enabled = false;
        server.configure(config).unwrap();
        assert!(matches!(
            server.create_compaction_declaration_for("principal-c", declaration),
            Err(Error::BadRequest(message)) if message == "compaction is disabled"
        ));
        fs::remove_dir_all(directory).unwrap();
    }

    #[test]
    fn compaction_creation_times_prune_stale_principals() {
        let (server, directory) = setup();
        let context = server.create_context().unwrap();
        let declaration = CreateCompactionDeclaration {
            context_id: context.id.0,
            strategy_preferences: vec![WINDOW_TAIL_STRATEGY_ID.into()],
            target_tokens: Some(8),
            expires_in_ms: 10_000,
        };
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_millis() as u64;
        {
            let mut state = server.declarations.lock();
            state.creation_times.insert(
                "stale-principal".into(),
                VecDeque::from(vec![now - 120_000]),
            );
            state
                .creation_times
                .insert("active-principal".into(), VecDeque::from(vec![now]));
            state.creation_times.insert(
                "stale-principal-2".into(),
                VecDeque::from(vec![now - 120_000]),
            );
        }
        server
            .create_compaction_declaration_for("fresh-principal", declaration)
            .unwrap();
        let state = server.declarations.lock();
        assert!(!state.creation_times.contains_key("stale-principal"));
        assert!(!state.creation_times.contains_key("stale-principal-2"));
        assert!(state.creation_times.contains_key("active-principal"));
        assert!(state.creation_times.contains_key("fresh-principal"));
        fs::remove_dir_all(directory).unwrap();
    }

    fn compacting_request(context_id: ContextId) -> InferRequest {
        InferRequest {
            model: "m".into(),
            prompt: "new question".into(),
            max_tokens: 2,
            context_id: Some(context_id),
            compaction: Some(CompactionRequest {
                target_tokens: Some(4),
                ..CompactionRequest::default()
            }),
            deadline_ms: None,
            stop: vec![],
            raw_continuation: false,
            sampling: SamplingConfig::default(),
            grammar: None,
            scheduling: SchedulingMetadata::default(),
        }
    }

    #[test]
    fn compaction_success_successor_commit() {
        let (server, directory) = setup();
        let source = server
            .import_context(vec![
                "<start_of_turn>system".into(),
                "policy".into(),
                "<start_of_turn>user".into(),
                "older question".into(),
            ])
            .unwrap();
        let (response, _) = server
            .infer("compact-success", compacting_request(source.id.clone()))
            .unwrap();
        let generated_text = response.text.clone();
        assert!(!generated_text.is_empty());
        let result = response.usage.compaction_result.unwrap();
        let successor = server.context(&source.id).unwrap();
        assert!(result.success);
        assert_eq!(
            result.selected_strategy_id.as_deref(),
            Some("window_tail:v1")
        );
        assert_eq!(successor.revision, source.revision + 1);
        assert!(
            successor
                .tokens
                .iter()
                .any(|token| token.contains("system"))
        );
        assert!(successor.tokens.iter().any(|token| token == "new"));
        assert!(
            generated_text
                .split_whitespace()
                .all(|token| successor.tokens.iter().any(|stored| stored == token))
        );
        fs::remove_dir_all(directory).unwrap();
    }

    #[test]
    fn compaction_result_payload_for_openai_replay() {
        let (server, directory) = setup();
        let source = server
            .import_context(vec![
                "<start_of_turn>system".into(),
                "policy".into(),
                "<start_of_turn>user".into(),
                "older question".into(),
            ])
            .unwrap();
        let (response, _) = server
            .infer("compact-replay", compacting_request(source.id))
            .unwrap();
        let result = response.usage.compaction_result.unwrap();
        assert_eq!(result.selected_strategy_id.as_deref(), Some("window_tail:v1"));
        assert!(result.success);
        fs::remove_dir_all(directory).unwrap();
    }
    #[test]
    fn compacted_successor_is_evaluated_before_continuation() {
        let directory = dir();
        fs::create_dir_all(&directory).unwrap();
        let model_path = directory.join("m.gguf");
        fs::write(&model_path, b"model").unwrap();
        let server = Server::open(directory.join("state.json"), Arc::new(CompactionProofEngine))
        .unwrap();
        server
            .register_model(ModelRecord {
                id: "m".into(),
                revision: "r1".into(),
                path: model_path,
                sha256: hex_digest(b"model"),
                aliases: Vec::new(),
                family: "gemma4".into(),
                size_bytes: 5,
                block_count: 1,
                epoch: 0,
            })
            .unwrap();
        let source = server
            .import_context(vec![
                "<start_of_turn>system".into(),
                "policy".into(),
                "<start_of_turn>user".into(),
                "discard me".into(),
            ])
            .unwrap();
        let (response, _) = server
            .infer("compaction-proof", compacting_request(source.id.clone()))
            .unwrap();
        let successor = server.context(&response.usage.context_id).unwrap();
        assert!(response.usage.compaction_result.unwrap().success);
        assert_eq!(successor.native_tokens, vec![41, 42, 43]);
        assert!(successor.tokens.iter().any(|token| token == "policy"));
        assert_eq!(server.context(&source.id).unwrap(), source);
        fs::remove_dir_all(directory).unwrap();
    }

    #[test]
    fn prepared_compaction_rolls_back_when_continuation_fails() {
        let directory = dir();
        fs::create_dir_all(&directory).unwrap();
        let model_path = directory.join("m.gguf");
        fs::write(&model_path, b"model").unwrap();
        let server = Server::open(directory.join("state.json"), Arc::new(CompactionProofEngine))
        .unwrap();
        server
            .register_model(ModelRecord {
                id: "m".into(),
                revision: "r1".into(),
                path: model_path,
                sha256: hex_digest(b"model"),
                aliases: Vec::new(),
                family: "gemma4".into(),
                size_bytes: 5,
                block_count: 1,
                epoch: 0,
            })
            .unwrap();
        let source = server
            .import_context(vec![
                "<start_of_turn>system".into(),
                "policy".into(),
                "<start_of_turn>user".into(),
                "discard me".into(),
            ])
            .unwrap();
        let mut request = compacting_request(source.id.clone());
        request.prompt = "fail continuation".into();
        assert!(matches!(
            server.infer("compaction-rollback", request),
            Err(Error::State(message)) if message == "continuation failed after compaction"
        ));
        assert_eq!(server.context(&source.id).unwrap(), source);
        fs::remove_dir_all(directory).unwrap();
    }

    #[test]
    fn compaction_failure_rolls_back_source_context() {
        let (server, directory) = setup();
        let source = server
            .import_context(vec!["one".into(), "two".into()])
            .unwrap();
        let mut request = compacting_request(source.id.clone());
        request.compaction.as_mut().unwrap().strategy_preferences = vec!["missing".into()];
        request.compaction.as_mut().unwrap().fallback_when_no_match =
            CompactionNoMatchFallback::Reject;
        assert!(matches!(
            server.infer("compact-failure", request),
            Err(Error::BadRequest(_))
        ));
        assert_eq!(server.context(&source.id).unwrap(), source);
        fs::remove_dir_all(directory).unwrap();
    }

    #[test]
    fn compaction_cancel_disconnect_race() {
        let (server, directory) = setup();
        let source = server
            .import_context(vec!["one".into(), "two".into()])
            .unwrap();
        server.cancel("compact-cancel");
        assert!(matches!(
            server.infer("compact-cancel", compacting_request(source.id.clone())),
            Err(Error::Cancelled)
        ));
        assert_eq!(server.context(&source.id).unwrap(), source);
        fs::remove_dir_all(directory).unwrap();
    }

    fn assert_semantic_fixture(name: &str) {
        let fixtures: Value =
            serde_json::from_str(include_str!("../tests/fixtures/semantic_compaction.json"))
                .unwrap();
        let fixture = &fixtures[name];
        let tokens = fixture["tokens"]
            .as_array()
            .unwrap()
            .iter()
            .map(|token| token.as_str().unwrap().to_owned())
            .collect::<Vec<_>>();
        let target_tokens = fixture["target_tokens"].as_u64().unwrap() as usize;
        let proposal = apply_window_tail_strategy(&tokens, target_tokens, &[]).unwrap();
        for required in fixture["required_fragments"].as_array().unwrap() {
            let required = required.as_str().unwrap();
            assert!(
                proposal
                    .resulting_tokens
                    .iter()
                    .any(|token| token.contains(required)),
                "fixture {name} lost required fragment {required}"
            );
        }
        let repeated = apply_window_tail_strategy(&tokens, target_tokens, &[]).unwrap();
        assert_eq!(proposal, repeated, "fixture {name} is not deterministic");
    }

    #[test]
    fn semantic_regression_pronoun_continuity() {
        assert_semantic_fixture("pronoun_continuity");
    }

    #[test]
    fn semantic_regression_instruction_retention() {
        assert_semantic_fixture("instruction_retention");
    }

    #[test]
    fn semantic_regression_tool_call_consistency() {
        assert_semantic_fixture("tool_call_consistency");
    }

    #[test]
    fn semantic_regression_followup_fidelity() {
        assert_semantic_fixture("followup_fidelity");
    }

}
