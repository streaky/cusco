pub use crate::context_updates::ContextUpdateOperation;
pub use crate::context_updates::{ContextUpdateRequest, ContextUpdateResult};
use crate::{
    ContextId, Error, InferRequest, InferResponse, RequestControl, Server, StreamEvent, Usage,
};
use std::{sync::Arc, time::Duration};
use tokio::sync::mpsc;

pub(super) fn start_deadline_watchdogs(
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

fn sampling_config(
    temperature: Option<f32>,
    top_p: Option<f32>,
    seed: Option<u64>,
) -> Result<crate::SamplingConfig, Error> {
    let seed = seed
        .map(u32::try_from)
        .transpose()
        .map_err(|_| Error::BadRequest("seed must be at most 4294967295".into()))?
        .unwrap_or(u32::MAX);
    Ok(crate::SamplingConfig {
        temperature: temperature.unwrap_or(1.0),
        top_p: top_p.unwrap_or(1.0),
        seed,
    })
}

#[derive(Clone, Copy, Debug, Default, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum NativeToolKind {
    #[default]
    Function,
    WebSearch,
}
#[derive(Clone, Debug, serde::Deserialize)]
pub struct NativeToolDefinition {
    #[serde(default)]
    pub kind: NativeToolKind,
    #[serde(default)]
    pub name: Option<String>,
    #[serde(default)]
    pub description: Option<String>,
    #[serde(default)]
    pub parameters: Option<serde_json::Value>,
    #[serde(default)]
    pub strict: bool,
    #[serde(default)]
    pub allowed_domains: Vec<String>,
    #[serde(default)]
    pub search_context_size: Option<String>,
}
#[derive(Clone, Debug, Default, serde::Deserialize)]
pub enum NativeToolChoice {
    #[default]
    Auto,
    None,
    Required,
    Function {
        name: String,
    },
}
#[derive(Clone, Debug, serde::Deserialize)]
pub enum NativeResponseFormat {
    Text,
    JsonObject,
    JsonSchema { schema: serde_json::Value },
}
#[derive(Clone, Debug, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum NativeReasoningEffort {
    None,
    Low,
    Medium,
    High,
    Max,
}
/// Protocol-neutral semantic inputs for response-oriented generation.
#[derive(Clone, Debug, serde::Deserialize)]
#[serde(default)]
pub struct GenerationSemantics {
    pub tools: Vec<NativeToolDefinition>,
    pub tool_choice: NativeToolChoice,
    pub format_constraints: Vec<NativeResponseFormat>,
    pub reasoning_effort: Option<NativeReasoningEffort>,
    pub continuation: Option<Continuation>,
    pub store: bool,
    pub previous_response_id: Option<String>,
    pub input_items: Vec<crate::responses::ResponseInputItem>,
    pub record_output: bool,
    pub expose_reasoning: bool,
    pub tool_snapshot: Vec<serde_json::Value>,
    pub tool_choice_snapshot: Option<serde_json::Value>,
}
impl Default for GenerationSemantics {
    fn default() -> Self {
        Self {
            tools: Vec::new(),
            tool_choice: NativeToolChoice::Auto,
            format_constraints: Vec::new(),
            reasoning_effort: None,
            continuation: None,
            store: true,
            previous_response_id: None,
            input_items: Vec::new(),
            record_output: false,
            expose_reasoning: false,
            tool_snapshot: Vec::new(),
            tool_choice_snapshot: None,
        }
    }
}
#[derive(Clone, Debug, serde::Deserialize)]
pub struct Continuation {
    pub previous_response_id: String,
}
#[derive(Clone, Debug, serde::Deserialize)]
pub enum GenerationInput {
    Prompt(String),
    Messages(Vec<PromptMessage>),
    Ledger(Vec<crate::responses::ResponseInputItem>),
}
#[derive(Clone, Debug, serde::Deserialize)]
pub struct PromptMessage {
    pub role: String,
    pub content: PromptContent,
}
#[derive(Clone, Debug, serde::Deserialize)]
pub enum PromptContent {
    Text(String),
    Parts(Vec<PromptPart>),
}
#[derive(Clone, Debug, serde::Deserialize)]
pub enum PromptPart {
    Text(String),
    ImageUri(String),
}
#[derive(Clone, Debug, Default, serde::Deserialize)]
#[serde(default)]
pub struct GenerationOptions {
    pub max_tokens: Option<usize>,
    pub temperature: Option<f32>,
    pub top_p: Option<f32>,
    pub seed: Option<u64>,
    pub stop: Vec<String>,
    pub context_id: Option<ContextId>,
    pub deadline_ms: Option<u64>,
    pub grammar: Option<String>,
    pub compaction: Option<crate::CompactionRequest>,
    pub raw_continuation: bool,
    pub semantics: GenerationSemantics,
}
#[derive(Clone, Debug, serde::Deserialize)]
pub struct GenerationRequest {
    pub model: String,
    pub input: GenerationInput,
    pub options: GenerationOptions,
    pub principal: String,
    pub request_id: String,
    pub retained_bytes: usize,
}
#[derive(Clone, Debug, serde::Serialize)]
pub struct DomainFailure {
    pub code: String,
    pub message: String,
}

#[derive(Clone, Debug, serde::Serialize)]
pub struct DomainOutcome<T> {
    pub value: Option<T>,
    pub failure: Option<DomainFailure>,
}
impl<T> DomainOutcome<T> {
    fn success(value: T) -> Self {
        Self {
            value: Some(value),
            failure: None,
        }
    }
    fn failed(error: Error) -> Self {
        Self {
            value: None,
            failure: Some(DomainFailure {
                code: error.code().into(),
                message: error.to_string(),
            }),
        }
    }
}
pub fn new_request_id() -> String {
    uuid::Uuid::new_v4().to_string()
}
#[derive(Clone, Debug, serde::Serialize)]
pub struct GenerationStarted {
    pub request_id: String,
    pub context_id: ContextId,
    pub correlation_id: String,
    pub inference_id: String,
    pub execution_session_id: String,
    pub created_at: u64,
    pub buffer_output: bool,
    pub response: Option<crate::responses::ResponseResource>,
}
#[derive(Clone, Debug, serde::Serialize)]
pub struct GenerationToken {
    pub token: String,
    pub index: usize,
}
#[derive(Clone, Debug, serde::Serialize)]
pub struct GenerationOutput {
    pub text: String,
    pub function_call: Option<crate::responses::ResponseFunctionCall>,
    pub reasoning: Option<crate::responses::ResponseReasoning>,
    pub web_search_call: Option<crate::responses::ResponseWebSearchCall>,
}
#[derive(Clone, Debug, serde::Serialize)]
pub struct GenerationFinished {
    pub reason: crate::FinishReason,
    pub usage: Box<Usage>,
    pub response: Option<crate::responses::ResponseResource>,
    pub output: Option<GenerationOutput>,
}
#[derive(Clone, Debug, serde::Serialize)]
pub enum GenerationEvent {
    Started(GenerationStarted),
    Token(GenerationToken),
    Finished(GenerationFinished),
    Error(DomainFailure),
}

/// Permit held while the caller reads and validates a request body.
pub struct NativePermit(Option<crate::PrequeuePermit>);
impl NativePermit {
    pub(crate) fn take(&mut self) -> Option<crate::PrequeuePermit> {
        self.0.take()
    }
}
struct CancellationOnDrop {
    control: Arc<RequestControl>,
    armed: bool,
}
impl Drop for CancellationOnDrop {
    fn drop(&mut self) {
        if self.armed {
            self.control.cancel();
        }
    }
}

/// Shared session stream; dropping its final owner cancels execution and closes the channel.
#[derive(Clone)]
pub struct GenerationSession {
    inner: Arc<GenerationSessionInner>,
}
struct GenerationSessionInner {
    started: tokio::sync::Mutex<Option<GenerationStarted>>,
    receiver: tokio::sync::Mutex<mpsc::Receiver<StreamEvent>>,
    worker: tokio::sync::Mutex<Option<tokio::task::JoinHandle<Result<InferResponse, Error>>>>,
    control: Arc<RequestControl>,
    finalizer: tokio::sync::Mutex<Option<crate::generation_plan::ResourcePlan>>,
    output_text: tokio::sync::Mutex<Option<String>>,
}
impl Drop for GenerationSessionInner {
    fn drop(&mut self) {
        self.control.cancel();
    }
}
impl GenerationSession {
    pub async fn next(&self) -> Option<GenerationEvent> {
        if let Some(event) = self.inner.started.lock().await.take() {
            return Some(GenerationEvent::Started(event));
        }
        let event = self.inner.receiver.lock().await.recv().await?;
        let mut output = None;
        let mut response_resource = None;
        if let StreamEvent::Finished { reason, .. } = &event {
            let worker = self.inner.worker.lock().await.take()?;
            let response = match worker.await {
                Ok(Ok(response)) => response,
                Ok(Err(error)) => {
                    return Some(GenerationEvent::Error(DomainFailure {
                        code: error.code().into(),
                        message: error.to_string(),
                    }));
                }
                Err(error) => {
                    let error = Error::State(format!("inference worker failed: {error}"));
                    return Some(GenerationEvent::Error(DomainFailure {
                        code: error.code().into(),
                        message: error.to_string(),
                    }));
                }
            };
            if let Some(finalizer) = self.inner.finalizer.lock().await.take() {
                match finalizer.complete(response, *reason) {
                    Ok(completed) => {
                        response_resource = completed.response;
                        output = Some(completed.output);
                    }
                    Err(error) => {
                        return Some(GenerationEvent::Error(DomainFailure {
                            code: error.code().into(),
                            message: error.to_string(),
                        }));
                    }
                }
            } else {
                *self.inner.output_text.lock().await = Some(response.text);
            }
        }
        Some(convert_event(event, response_resource, output))
    }
    pub async fn take_output_text(&self) -> Option<String> {
        self.inner.output_text.lock().await.take()
    }
    pub fn cancel(&self) {
        self.inner.control.cancel();
    }
}

fn convert_event(
    event: StreamEvent,
    response: Option<crate::responses::ResponseResource>,
    output: Option<GenerationOutput>,
) -> GenerationEvent {
    match event {
        StreamEvent::Started { .. } => {
            unreachable!("started event is emitted before receiver events")
        }
        StreamEvent::Token { token, index } => {
            GenerationEvent::Token(GenerationToken { token, index })
        }
        StreamEvent::Finished { reason, usage } => GenerationEvent::Finished(GenerationFinished {
            reason,
            usage,
            response,
            output,
        }),
        StreamEvent::Error { code, message } => {
            GenerationEvent::Error(DomainFailure { code, message })
        }
    }
}
fn epoch_seconds() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs()
}

impl Server {
    #[cfg(feature = "test-fixtures")]
    pub fn open_with_engine(
        path: impl AsRef<std::path::Path>,
        engine: Arc<dyn crate::InferenceEngine>,
    ) -> Result<Self, Error> {
        Self::open(path, engine)
    }
    pub async fn acquire_prequeue_permit(&self) -> Result<NativePermit, Error> {
        Ok(NativePermit(Some(self.pre_queue.acquire().await?)))
    }

    pub async fn prepare_generation(
        &self,
        request: GenerationRequest,
        mut permit: NativePermit,
    ) -> Result<GenerationSession, Error> {
        let GenerationRequest {
            model,
            input,
            options,
            principal,
            request_id,
            retained_bytes,
        } = request;
        let model_record = self.model(&model)?.clone();
        let mut plan = crate::generation_plan::prepare(
            self,
            &request_id,
            &principal,
            &model_record,
            input,
            options.semantics,
        )
        .await?;
        let grammar = match (options.grammar, plan.grammar) {
            (Some(_), Some(_)) => {
                return Err(Error::BadRequest(
                    "grammar and response_format cannot both be supplied".into(),
                ));
            }
            (grammar, format) => grammar.or(format),
        };
        let config = self.config();
        let compaction_permit = if options.compaction.is_some() {
            if !config.compaction_enabled {
                return Err(Error::BadRequest("compaction is disabled".into()));
            }
            Some(self.compaction_workers.try_acquire()?)
        } else {
            None
        };
        let deadline = Duration::from_millis(
            options
                .deadline_ms
                .unwrap_or(config.wall_time_ms)
                .min(config.wall_time_ms),
        );
        let control = Arc::new(RequestControl::new());
        let waiting_since = std::time::Instant::now();
        drop(permit.take());
        let admission = self
            .reserve_admission(&request_id, control.clone(), retained_bytes, deadline)
            .await?;
        let wall_remaining = deadline
            .checked_sub(waiting_since.elapsed())
            .ok_or(Error::Deadline)?;
        start_deadline_watchdogs(
            &control,
            wall_remaining,
            Duration::from_millis(config.active_time_ms),
        );
        let mut req = InferRequest {
            model,
            prompt: plan.prompt,
            max_tokens: options.max_tokens.unwrap_or(config.default_output_tokens),
            context_id: options.context_id,
            deadline_ms: Some(
                u64::try_from(wall_remaining.as_millis())
                    .unwrap_or(u64::MAX)
                    .max(1),
            ),
            stop: options.stop,
            raw_continuation: options.raw_continuation,
            compaction: options.compaction,
            sampling: sampling_config(options.temperature, options.top_p, options.seed)?,
            grammar,
            scheduling: Default::default(),
        };
        req.scheduling.principal = principal;
        req.scheduling.correlation_id = request_id.clone();
        req.scheduling.inference_id = uuid::Uuid::new_v4().to_string();
        let (started_event, receiver, worker) = self
            .infer_stream_reserved_with_compaction(
                request_id.clone(),
                req,
                admission,
                compaction_permit,
            )
            .await?;
        let StreamEvent::Started {
            request_id: started_request_id,
            context_id,
            correlation_id,
            inference_id,
            execution_session_id,
        } = started_event
        else {
            unreachable!()
        };
        let created_at = epoch_seconds();
        let buffer_output = plan.buffer_output;
        let response = plan
            .resource
            .as_mut()
            .filter(|resource| resource.persist())
            .map(|resource| {
                resource.started_resource(
                    created_at,
                    crate::responses::ResponseMetadata {
                        correlation_id: correlation_id.clone(),
                        inference_id: inference_id.clone(),
                        execution_session_id: execution_session_id.clone(),
                    },
                )
            });
        let started = GenerationStarted {
            request_id: started_request_id,
            context_id,
            correlation_id,
            inference_id,
            execution_session_id,
            created_at,
            buffer_output,
            response,
        };
        Ok(GenerationSession {
            inner: Arc::new(GenerationSessionInner {
                started: tokio::sync::Mutex::new(Some(started)),
                receiver: tokio::sync::Mutex::new(receiver),
                worker: tokio::sync::Mutex::new(Some(worker)),
                control,
                finalizer: tokio::sync::Mutex::new(plan.resource),
                output_text: tokio::sync::Mutex::new(None),
            }),
        })
    }

    pub async fn acquire_prequeue_permit_outcome(&self) -> DomainOutcome<NativePermit> {
        match self.acquire_prequeue_permit().await {
            Ok(value) => DomainOutcome::success(value),
            Err(error) => DomainOutcome::failed(error),
        }
    }
    pub async fn prepare_generation_outcome(
        &self,
        request: GenerationRequest,
        permit: NativePermit,
    ) -> DomainOutcome<GenerationSession> {
        match self.prepare_generation(request, permit).await {
            Ok(value) => DomainOutcome::success(value),
            Err(error) => DomainOutcome::failed(error),
        }
    }
    pub fn retrieve_response(
        &self,
        id: &str,
        owner: &str,
    ) -> DomainOutcome<crate::responses::ResponseResource> {
        match self
            .response_service
            .retrieve(id, owner)
            .map_err(crate::responses::store_error)
        {
            Ok(value) => DomainOutcome::success(value),
            Err(error) => DomainOutcome::failed(error),
        }
    }
    pub fn delete_response(&self, id: &str, owner: &str) -> DomainOutcome<()> {
        match self
            .response_service
            .delete(id, owner)
            .map_err(crate::responses::store_error)
        {
            Ok(()) => DomainOutcome::success(()),
            Err(error) => DomainOutcome::failed(error),
        }
    }
    pub async fn apply_context_update_outcome(
        &self,
        principal: String,
        mut request: crate::context_updates::ContextUpdateRequest,
        mut permit: NativePermit,
        retained_bytes: usize,
    ) -> DomainOutcome<crate::context_updates::ContextUpdateResult> {
        let operation_id = request
            .operation_id
            .get_or_insert_with(|| format!("ctxupd_{}", uuid::Uuid::new_v4()))
            .clone();
        let correlation_id = uuid::Uuid::new_v4().to_string();
        let control = Arc::new(RequestControl::new());
        let mut cancellation = CancellationOnDrop {
            control: control.clone(),
            armed: true,
        };
        let config = self.config();
        let waiting_since = std::time::Instant::now();
        drop(permit.take());
        let deadline = Duration::from_millis(config.wall_time_ms);
        let admission = match self
            .reserve_admission(&operation_id, control.clone(), retained_bytes, deadline)
            .await
        {
            Ok(admission) => admission,
            Err(error) => return DomainOutcome::failed(error),
        };
        let remaining = match deadline.checked_sub(waiting_since.elapsed()) {
            Some(remaining) => remaining,
            None => return DomainOutcome::failed(Error::Deadline),
        };
        start_deadline_watchdogs(
            &control,
            remaining,
            Duration::from_millis(config.active_time_ms),
        );
        let service = self.response_service.clone();
        let worker_control = control.clone();
        let worker = tokio::task::spawn_blocking(move || {
            let _admission = admission;
            crate::context_updates::apply(
                &service,
                &principal,
                correlation_id,
                request,
                &worker_control,
            )
        });
        let result = match worker.await {
            Ok(result) => result,
            Err(error) => Err(Error::State(format!(
                "context update worker failed: {error}"
            ))),
        };
        cancellation.armed = false;
        match result {
            Ok(result) => DomainOutcome::success(result),
            Err(error) => DomainOutcome::failed(error),
        }
    }
    pub fn complete_response(
        &self,
        params: crate::responses::CompleteResponse<'_>,
    ) -> DomainOutcome<crate::responses::ResponseResource> {
        let resource = self.response_service.complete(params);
        match self
            .response_service
            .remember(&resource)
            .map_err(crate::responses::store_error)
        {
            Ok(()) => DomainOutcome::success(resource),
            Err(error) => DomainOutcome::failed(error),
        }
    }
}
