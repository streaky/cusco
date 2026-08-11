use crate::response_store::{ResponseResourceStore, StoreError};
use crate::{FinishReason, StreamEvent, Usage};
use parking_lot::Mutex;
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use std::{
    collections::{HashMap, VecDeque},
    time::{SystemTime, UNIX_EPOCH},
};
use uuid::Uuid;
pub const RESPONSE_SCHEMA_VERSION: u32 = 1;

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct ResponseTextPart {
    pub text: String,
    pub annotations: Vec<Value>,
    pub logprobs: Vec<Value>,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct ResponseReasoningSummaryPart {
    pub text: String,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct ResponseReasoning {
    pub id: String,
    pub status: String,
    pub summary: Vec<ResponseReasoningSummaryPart>,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct ResponseMessage {
    pub id: String,
    pub role: String,
    pub status: String,
    pub content: Vec<ResponseTextPart>,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct ResponseFunctionCall {
    pub id: String,
    pub call_id: String,
    pub name: String,
    pub arguments: String,
    pub status: String,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct ResponseWebSearchSource {
    pub r#type: String,
    pub url: String,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct ResponseWebSearchAction {
    pub r#type: String,
    pub query: String,
    pub sources: Vec<ResponseWebSearchSource>,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct ResponseWebSearchCall {
    pub id: String,
    pub status: String,
    pub action: ResponseWebSearchAction,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ResponseInputItem {
    Message {
        role: String,
        text: String,
    },
    FunctionCallOutput {
        call_id: String,
        output: String,
    },
    FunctionCall {
        call_id: String,
        name: String,
        arguments: String,
    },
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ResponseOutputItem {
    Message(ResponseMessage),
    FunctionCall(ResponseFunctionCall),
    Reasoning(ResponseReasoning),
    WebSearchCall(ResponseWebSearchCall),
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct ResponseResource {
    #[serde(default = "schema_version")]
    pub schema_version: u32,
    pub id: String,
    pub owner: String,
    pub model: String,
    pub created_at: u64,
    pub status: String,
    pub store: bool,
    pub previous_response_id: Option<String>,
    pub input: Vec<ResponseInputItem>,
    pub output: Vec<ResponseOutputItem>,
    #[serde(default)]
    pub tools: Vec<Value>,
    #[serde(default = "default_tool_choice")]
    pub tool_choice: Value,
    pub finish_reason: Option<FinishReason>,
    pub usage: Option<ResponseUsage>,
    pub metadata: ResponseMetadata,
    #[serde(default)]
    pub lineage_revision: u64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub context_update: Option<ContextUpdateMetadata>,
}

fn schema_version() -> u32 {
    RESPONSE_SCHEMA_VERSION
}

fn default_tool_choice() -> Value {
    Value::String("auto".into())
}

#[derive(Clone, Debug, Default, Deserialize, PartialEq, Serialize)]
pub struct ResponseMetadata {
    pub correlation_id: String,
    pub inference_id: String,
    pub execution_session_id: String,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct ContextUpdateMetadata {
    pub operation_id: String,
    pub operation: String,
    pub base_response_id: String,
    pub base_revision: u64,
    pub correlation_id: String,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct ResponseUsage {
    pub input_tokens: usize,
    pub cached_tokens: usize,
    pub output_tokens: usize,
    #[serde(default)]
    pub reasoning_tokens: usize,
}

const TRANSIENT_RESPONSE_CAPACITY: usize = 1024;

#[derive(Default)]
struct TransientResponses {
    resources: HashMap<String, ResponseResource>,
    insertion_order: VecDeque<String>,
}

pub struct StreamResponse {
    pub model: String,
    pub owner: String,
    pub store: bool,
    pub previous_response_id: Option<String>,
    pub input: Vec<ResponseInputItem>,
    pub tools: Vec<Value>,
    pub tool_choice: Value,
    pub buffer_output: bool,
    pub expose_reasoning: bool,
    pub strip_reasoning: bool,
    pub web_search_call: Option<ResponseWebSearchCall>,
    pub lineage_revision: u64,
}

#[derive(Clone)]
pub struct ResponseService {
    store: std::sync::Arc<dyn ResponseResourceStore>,
    transient: std::sync::Arc<Mutex<TransientResponses>>,
}

impl ResponseService {
    pub fn new(store: std::sync::Arc<dyn ResponseResourceStore>) -> Self {
        Self {
            store,
            transient: std::sync::Arc::new(Mutex::new(TransientResponses::default())),
        }
    }
    pub fn projection(&self, model: String) -> ResponseProjection {
        ResponseProjection::new(model)
    }
    pub fn stream_projection(&self, request: StreamResponse) -> ResponseProjection {
        ResponseProjection::new(request.model.clone()).with_request(request)
    }

    pub fn complete(&self, params: CompleteResponse<'_>) -> ResponseResource {
        let (reasoning, text) = if params.strip_reasoning {
            split_native_reasoning(params.text)
        } else {
            (None, params.text)
        };
        let message = || {
            let annotations = params.web_search_call.map_or_else(Vec::new, |call| {
                call.action
                    .sources
                    .iter()
                    .filter_map(|source| {
                        let byte_start = text.find(&source.url)?;
                        let start_index = text[..byte_start].chars().count();
                        let end_index = start_index + source.url.chars().count();
                        Some(json!({
                            "type": "url_citation",
                            "url": source.url,
                            "title": source.url,
                            "start_index": start_index,
                            "end_index": end_index
                        }))
                    })
                    .collect()
            });
            ResponseOutputItem::Message(ResponseMessage {
                id: "msg_0".into(),
                role: "assistant".into(),
                status: "completed".into(),
                content: vec![ResponseTextPart {
                    text: text.into(),
                    annotations,
                    logprobs: vec![],
                }],
            })
        };
        let reasoning_item = || ResponseOutputItem::Reasoning(ResponseReasoning {
            id: "rs_0".into(),
            status: "completed".into(),
            summary: vec![ResponseReasoningSummaryPart {
                text: reasoning.unwrap_or_default().into(),
            }],
        });
        let output = if let Some(call) = params.function_call {
            vec![ResponseOutputItem::FunctionCall(call)]
        } else {
            let mut output = Vec::new();
            if params.expose_reasoning && reasoning.is_some() {
                output.push(reasoning_item());
            }
            if let Some(call) = params.web_search_call {
                output.push(ResponseOutputItem::WebSearchCall(call.clone()));
            }
            output.push(message());
            output
        };
        ResponseResource {
            schema_version: RESPONSE_SCHEMA_VERSION,
            id: params.id.into(),
            owner: params.owner.into(),
            model: params.model.into(),
            created_at: now(),
            status: "completed".into(),
            store: params.store,
            previous_response_id: params.previous_response_id.map(str::to_owned),
            input: params.input.to_vec(),
            tools: params.tools.to_vec(),
            tool_choice: params.tool_choice.clone(),
            output,
            finish_reason: Some(params.reason),
            usage: Some(ResponseUsage::from(params.usage)),
            metadata: ResponseMetadata {
                correlation_id: params.usage.correlation_id.clone(),
                inference_id: params.usage.inference_id.clone(),
                execution_session_id: params.usage.execution_session_id.clone(),
            },
            lineage_revision: params.lineage_revision,
            context_update: None,
        }
    }

    pub fn remember(&self, resource: &ResponseResource) -> Result<(), StoreError> {
        if resource.store {
            if let Some(base_id) = resource.previous_response_id.as_deref() {
                return self.store.put_successor(
                    base_id,
                    resource.lineage_revision.saturating_sub(1),
                    resource,
                );
            }
            return self.store.put(resource);
        }
        let mut transient = self.transient.lock();
        if !transient.resources.contains_key(&resource.id) {
            transient.insertion_order.push_back(resource.id.clone());
        }
        transient
            .resources
            .insert(resource.id.clone(), resource.clone());
        while transient.resources.len() > TRANSIENT_RESPONSE_CAPACITY {
            let Some(id) = transient.insertion_order.pop_front() else {
                break;
            };
            transient.resources.remove(&id);
        }
        Ok(())
    }
    pub fn persist(&self, resource: &ResponseResource) -> Result<(), StoreError> {
        self.store.put(resource)
    }
    pub fn commit_successor(
        &self,
        base_id: &str,
        expected_revision: u64,
        resource: &ResponseResource,
    ) -> Result<(), StoreError> {
        self.store
            .put_successor(base_id, expected_revision, resource)
    }
    pub fn retrieve(&self, id: &str, owner: &str) -> Result<ResponseResource, StoreError> {
        let resource = self
            .transient
            .lock()
            .resources
            .get(id)
            .cloned()
            .map_or_else(|| self.store.get(id), Ok)?;
        if resource.owner != owner {
            return Err(StoreError::NotFound(id.into()));
        }
        Ok(resource)
    }
    pub fn delete(&self, id: &str, owner: &str) -> Result<(), StoreError> {
        self.retrieve(id, owner)?;
        let mut transient = self.transient.lock();
        if transient
            .resources
            .values()
            .any(|candidate| candidate.previous_response_id.as_deref() == Some(id))
        {
            return Err(StoreError::Conflict(id.into()));
        }
        if transient.resources.remove(id).is_some() {
            transient
                .insertion_order
                .retain(|candidate| candidate != id);
            return Ok(());
        }
        drop(transient);
        self.store.delete(id)
    }
}

pub struct CompleteResponse<'a> {
    pub id: &'a str,
    pub owner: &'a str,
    pub model: &'a str,
    pub text: &'a str,
    pub reason: FinishReason,
    pub usage: &'a Usage,
    pub store: bool,
    pub previous_response_id: Option<&'a str>,
    pub tools: &'a [Value],
    pub tool_choice: &'a Value,
    pub lineage_revision: u64,
    pub input: &'a [ResponseInputItem],
    pub function_call: Option<ResponseFunctionCall>,
    pub expose_reasoning: bool,
    pub strip_reasoning: bool,
    pub web_search_call: Option<&'a ResponseWebSearchCall>,
}

impl From<&Usage> for ResponseUsage {
    fn from(usage: &Usage) -> Self {
        Self {
            input_tokens: usage.input_tokens,
            cached_tokens: usage.cached_tokens,
            output_tokens: usage.generated_tokens,
            reasoning_tokens: 0,
        }
    }
}

pub fn new_function_call(name: String, arguments: String) -> ResponseFunctionCall {
    ResponseFunctionCall {
        id: format!("fc_{}", Uuid::new_v4()),
        call_id: format!("call_{}", Uuid::new_v4()),
        name,
        arguments,
        status: "completed".into(),
    }
}

pub fn new_web_search_call(
    query: String,
    urls: impl IntoIterator<Item = String>,
) -> ResponseWebSearchCall {
    ResponseWebSearchCall {
        id: format!("ws_{}", Uuid::new_v4()),
        status: "completed".into(),
        action: ResponseWebSearchAction {
            r#type: "search".into(),
            query,
            sources: urls
                .into_iter()
                .map(|url| ResponseWebSearchSource {
                    r#type: "url".into(),
                    url,
                })
                .collect(),
        },
    }
}

pub fn project_resource(resource: &ResponseResource) -> Value {
    let usage = resource.usage.as_ref().map(|u| json!({"input_tokens":u.input_tokens,"input_tokens_details":{"cached_tokens":u.cached_tokens},"output_tokens":u.output_tokens,"output_tokens_details":{"reasoning_tokens":u.reasoning_tokens},"total_tokens":u.input_tokens+u.output_tokens})).unwrap_or(Value::Null);
    let output = resource
        .output
        .iter()
        .map(project_output_item)
        .collect::<Vec<_>>();
    json!({"id":resource.id,"object":"response","created_at":resource.created_at,"status":resource.status,"background":false,"error":null,"incomplete_details":null,"instructions":null,"max_output_tokens":null,"metadata":resource.metadata,"model":resource.model,"output":output,"parallel_tool_calls":true,"previous_response_id":resource.previous_response_id,"prompt_cache_key":null,"reasoning":null,"safety_identifier":null,"service_tier":"default","store":resource.store,"temperature":1.0,"text":{"format":{"type":"text"},"verbosity":"medium"},"tool_choice":resource.tool_choice,"tools":resource.tools,"top_logprobs":0,"top_p":1.0,"truncation":"disabled","usage":usage,"finish_reason":resource.finish_reason,"cusco":{"lineage_revision":resource.lineage_revision,"context_update":resource.context_update}})
}

fn project_output_item(item: &ResponseOutputItem) -> Value {
    match item {
        ResponseOutputItem::Message(message) => project_message(message),
        ResponseOutputItem::FunctionCall(call) => {
            json!({"id":call.id,"type":"function_call","call_id":call.call_id,"name":call.name,"arguments":call.arguments,"status":call.status})
        }
        ResponseOutputItem::WebSearchCall(call) => {
            json!({"id":call.id,"type":"web_search_call","status":call.status,"action":call.action})
        }
        ResponseOutputItem::Reasoning(reasoning) => {
            json!({"id":reasoning.id,"type":"reasoning","status":reasoning.status,"summary":reasoning.summary.iter().map(|part| json!({"type":"summary_text","text":part.text})).collect::<Vec<_>>()})
        }
    }
}
fn project_part(part: &ResponseTextPart) -> Value {
    json!({"type":"output_text","text":part.text,"annotations":part.annotations,"logprobs":part.logprobs})
}
fn project_message(message: &ResponseMessage) -> Value {
    json!({"id":message.id,"type":"message","role":message.role,"status":message.status,"content":message.content.iter().map(project_part).collect::<Vec<_>>()})
}
pub(crate) fn now() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |duration| duration.as_secs())
}

fn split_native_reasoning(text: &str) -> (Option<&str>, &str) {
    let trimmed = text.trim_start();
    let Some(reasoning) = trimmed.strip_prefix("<think>") else {
        return (None, text);
    };
    let separated = match reasoning.split_once("</think>") {
        Some((reasoning, answer)) => (reasoning, answer.trim_start_matches(['\r', '\n'])),
        None => (reasoning, ""),
    };
    let reasoning = separated.0.trim();
    ((!reasoning.is_empty()).then_some(reasoning), separated.1)
}

#[derive(Clone, Debug, PartialEq)]
pub enum ResponseEvent {
    Created(ResponseResource),
    OutputItemAdded(ResponseMessage),
    FunctionCallAdded(ResponseFunctionCall),
    WebSearchCallAdded(ResponseWebSearchCall),
    ReasoningAdded(ResponseReasoning),
    ReasoningSummaryAdded(ResponseReasoningSummaryPart),
    ReasoningSummaryDelta(String),
    ReasoningSummaryDone(ResponseReasoningSummaryPart),
    ReasoningDone(ResponseReasoning),
    ContentPartAdded(ResponseTextPart),
    TextDelta(String),
    TextDone(ResponseTextPart),
    ContentPartDone(ResponseTextPart),
    OutputItemDone(ResponseMessage),
    FunctionCallDone(ResponseFunctionCall),
    WebSearchCallDone(ResponseWebSearchCall),
    Completed(ResponseResource),
    Error(String),
}

pub struct ResponseProjection {
    resource: ResponseResource,
    sequence: usize,
    buffer_output: bool,
    function_call: Option<ResponseFunctionCall>,
    web_search_call: Option<ResponseWebSearchCall>,
    expose_reasoning: bool,
    strip_reasoning: bool,
}
impl ResponseProjection {
    pub fn new(model: String) -> Self {
        Self {
            resource: ResponseResource {
                schema_version: RESPONSE_SCHEMA_VERSION,
                id: String::new(),
                owner: String::new(),
                model,
                created_at: now(),
                status: "in_progress".into(),
                store: false,
                previous_response_id: None,
                input: vec![],
                output: vec![],
                tools: vec![],
                tool_choice: default_tool_choice(),
                finish_reason: None,
                usage: None,
                metadata: ResponseMetadata::default(),
                lineage_revision: 0,
                context_update: None,
            },
            sequence: 0,
            buffer_output: false,
            function_call: None,
            web_search_call: None,
            expose_reasoning: false,
            strip_reasoning: false,
        }
    }
    fn with_request(mut self, request: StreamResponse) -> Self {
        self.resource.owner = request.owner;
        self.resource.store = request.store;
        self.resource.previous_response_id = request.previous_response_id;
        self.resource.input = request.input;
        self.resource.tools = request.tools;
        self.resource.tool_choice = request.tool_choice;
        self.resource.lineage_revision = request.lineage_revision;
        self.web_search_call = request.web_search_call;
        self.expose_reasoning = request.expose_reasoning;
        self.strip_reasoning = request.strip_reasoning;
        self.buffer_output = request.buffer_output || request.expose_reasoning;
        self
    }
    pub fn completed_resource(&self) -> Option<&ResponseResource> {
        (self.resource.status == "completed").then_some(&self.resource)
    }
    pub fn generated_text(&self) -> &str {
        self.resource
            .output
            .iter()
            .find_map(|item| match item {
                ResponseOutputItem::Message(message) => {
                    message.content.first().map(|part| part.text.as_str())
                }
                _ => None,
            })
            .unwrap_or("")
    }
    pub fn set_function_call(&mut self, call: Option<ResponseFunctionCall>) {
        self.function_call = call;
    }
    fn message(&self, status: &str) -> ResponseMessage {
        let text = self.generated_text();
        let annotations = self.web_search_call.as_ref().map_or_else(Vec::new, |call| {
            call.action
                .sources
                .iter()
                .filter_map(|source| {
                    let byte_start = text.find(&source.url)?;
                    let start_index = text[..byte_start].chars().count();
                    Some(json!({
                        "type": "url_citation",
                        "url": source.url,
                        "title": source.url,
                        "start_index": start_index,
                        "end_index": start_index + source.url.chars().count()
                    }))
                })
                .collect()
        });
        ResponseMessage {
            id: "msg_0".into(),
            role: "assistant".into(),
            status: status.into(),
            content: vec![ResponseTextPart {
                text: text.into(),
                annotations,
                logprobs: vec![],
            }],
        }
    }
    pub fn project(&mut self, event: StreamEvent) -> String {
        let events = match event {
            StreamEvent::Started {
                request_id,
                correlation_id,
                inference_id,
                execution_session_id,
                ..
            } => {
                self.resource.id = request_id;
                self.resource.metadata = ResponseMetadata {
                    correlation_id,
                    inference_id,
                    execution_session_id,
                };
                let message = self.message("in_progress");
                self.resource.output = self
                    .web_search_call
                    .iter()
                    .cloned()
                    .map(ResponseOutputItem::WebSearchCall)
                    .chain(std::iter::once(ResponseOutputItem::Message(
                        message.clone(),
                    )))
                    .collect();
                if self.buffer_output {
                    vec![ResponseEvent::Created(self.resource.clone())]
                } else {
                    vec![
                        ResponseEvent::Created(self.resource.clone()),
                        ResponseEvent::OutputItemAdded(message),
                        ResponseEvent::ContentPartAdded(ResponseTextPart {
                            text: String::new(),
                            annotations: vec![],
                            logprobs: vec![],
                        }),
                    ]
                }
            }
            StreamEvent::Token { token, .. } => {
                if let Some(message) = self.resource.output.iter_mut().find_map(|item| match item {
                    ResponseOutputItem::Message(message) => Some(message),
                    _ => None,
                }) {
                    message.content[0].text.push_str(&token)
                }
                if self.buffer_output {
                    vec![]
                } else {
                    vec![ResponseEvent::TextDelta(token)]
                }
            }
            StreamEvent::Finished { reason, usage } => {
                self.resource.status = "completed".into();
                self.resource.finish_reason = Some(reason);
                self.resource.usage = Some(ResponseUsage::from(usage.as_ref()));
                if let Some(call) = self.function_call.take() {
                    self.resource.output = vec![ResponseOutputItem::FunctionCall(call.clone())];
                    vec![
                        ResponseEvent::FunctionCallAdded(call.clone()),
                        ResponseEvent::FunctionCallDone(call),
                        ResponseEvent::Completed(self.resource.clone()),
                    ]
                } else {
                    let raw_text = self.generated_text().to_owned();
                    let (reasoning, answer) = if self.strip_reasoning {
                        split_native_reasoning(&raw_text)
                    } else {
                        (None, raw_text.as_str())
                    };
                    let message = ResponseMessage {
                        id: "msg_0".into(),
                        role: "assistant".into(),
                        status: "completed".into(),
                        content: vec![ResponseTextPart {
                            text: answer.into(),
                            annotations: vec![],
                            logprobs: vec![],
                        }],
                    };
                    let mut output = Vec::new();
                    let reasoning_item = reasoning.map(|text| ResponseReasoning {
                        id: "rs_0".into(),
                        status: "completed".into(),
                        summary: vec![ResponseReasoningSummaryPart { text: text.into() }],
                    });
                    if self.expose_reasoning {
                        if let Some(item) = reasoning_item.clone() {
                            output.push(ResponseOutputItem::Reasoning(item));
                        }
                    }
                    if let Some(call) = self.web_search_call.clone() {
                        output.push(ResponseOutputItem::WebSearchCall(call));
                    }
                    output.push(ResponseOutputItem::Message(message.clone()));
                    self.resource.output = output;
                    let part = message.content[0].clone();
                    let mut events = Vec::new();
                    if self.buffer_output {
                        if self.expose_reasoning {
                            if let Some(item) = reasoning_item {
                                let summary = item.summary[0].clone();
                                events.push(ResponseEvent::ReasoningAdded(item.clone()));
                                events.push(ResponseEvent::ReasoningSummaryAdded(
                                    ResponseReasoningSummaryPart {
                                        text: String::new(),
                                    },
                                ));
                                events.push(ResponseEvent::ReasoningSummaryDelta(
                                    summary.text.clone(),
                                ));
                                events.push(ResponseEvent::ReasoningSummaryDone(summary));
                                events.push(ResponseEvent::ReasoningDone(item));
                            }
                        }
                        if let Some(call) = self.web_search_call.clone() {
                            events.push(ResponseEvent::WebSearchCallAdded(call.clone()));
                            events.push(ResponseEvent::WebSearchCallDone(call));
                        }
                        events.push(ResponseEvent::OutputItemAdded(message.clone()));
                        events.push(ResponseEvent::ContentPartAdded(ResponseTextPart {
                            text: String::new(),
                            annotations: vec![],
                            logprobs: vec![],
                        }));
                        events.push(ResponseEvent::TextDelta(part.text.clone()));
                    }
                    events.extend([
                        ResponseEvent::TextDone(part.clone()),
                        ResponseEvent::ContentPartDone(part),
                        ResponseEvent::OutputItemDone(message),
                        ResponseEvent::Completed(self.resource.clone()),
                    ]);
                    events
                }
            }
            StreamEvent::Error { message } => vec![ResponseEvent::Error(message)],
        };
        events.into_iter().fold(String::new(), |mut rows, event| {
            use std::fmt::Write as _;
            let value = self.project_event(event);
            write!(rows, "data: {value}\n\n").expect("String write");
            rows
        })
    }
    fn project_event(&mut self, event: ResponseEvent) -> Value {
        let n = self.sequence;
        self.sequence += 1;
        let reasoning_offset = usize::from(
            self.expose_reasoning
                && self
                    .resource
                    .output
                    .iter()
                    .any(|item| matches!(item, ResponseOutputItem::Reasoning(_))),
        );
        let message_index = usize::from(self.web_search_call.is_some()) + reasoning_offset;
        match event {
            ResponseEvent::Created(r) => {
                json!({"type":"response.created","sequence_number":n,"response":project_resource(&r)})
            }
            ResponseEvent::OutputItemAdded(m) => {
                json!({"type":"response.output_item.added","sequence_number":n,"item":project_message(&m),"output_index":message_index})
            }
            ResponseEvent::FunctionCallAdded(call) => {
                json!({"type":"response.output_item.added","sequence_number":n,"item":project_output_item(&ResponseOutputItem::FunctionCall(call)),"output_index":0})
            }
            ResponseEvent::WebSearchCallAdded(call) => {
                json!({"type":"response.output_item.added","sequence_number":n,"item":project_output_item(&ResponseOutputItem::WebSearchCall(call)),"output_index":0})
            }
            ResponseEvent::ReasoningAdded(reasoning) => {
                json!({"type":"response.output_item.added","sequence_number":n,"item":project_output_item(&ResponseOutputItem::Reasoning(reasoning)),"output_index":0})
            }
            ResponseEvent::ReasoningSummaryAdded(part) => {
                json!({"type":"response.reasoning_summary_part.added","sequence_number":n,"item_id":"rs_0","output_index":0,"summary_index":0,"part":{"type":"summary_text","text":part.text}})
            }
            ResponseEvent::ReasoningSummaryDelta(delta) => {
                json!({"type":"response.reasoning_summary_text.delta","sequence_number":n,"item_id":"rs_0","output_index":0,"summary_index":0,"delta":delta})
            }
            ResponseEvent::ReasoningSummaryDone(part) => {
                json!({"type":"response.reasoning_summary_text.done","sequence_number":n,"item_id":"rs_0","output_index":0,"summary_index":0,"text":part.text})
            }
            ResponseEvent::ReasoningDone(reasoning) => {
                json!({"type":"response.output_item.done","sequence_number":n,"item":project_output_item(&ResponseOutputItem::Reasoning(reasoning)),"output_index":0})
            }
            ResponseEvent::ContentPartAdded(p) => {
                json!({"type":"response.content_part.added","sequence_number":n,"item_id":"msg_0","output_index":message_index,"content_index":0,"part":project_part(&p)})
            }
            ResponseEvent::TextDelta(delta) => {
                json!({"type":"response.output_text.delta","sequence_number":n,"item_id":"msg_0","output_index":message_index,"content_index":0,"delta":delta,"logprobs":[]})
            }
            ResponseEvent::TextDone(p) => {
                json!({"type":"response.output_text.done","sequence_number":n,"item_id":"msg_0","output_index":message_index,"content_index":0,"text":p.text,"logprobs":[]})
            }
            ResponseEvent::ContentPartDone(p) => {
                json!({"type":"response.content_part.done","sequence_number":n,"item_id":"msg_0","output_index":message_index,"content_index":0,"part":project_part(&p)})
            }
            ResponseEvent::OutputItemDone(m) => {
                json!({"type":"response.output_item.done","sequence_number":n,"item":project_message(&m),"output_index":message_index})
            }
            ResponseEvent::FunctionCallDone(call) => {
                json!({"type":"response.output_item.done","sequence_number":n,"item":project_output_item(&ResponseOutputItem::FunctionCall(call)),"output_index":0})
            }
            ResponseEvent::WebSearchCallDone(call) => {
                json!({"type":"response.output_item.done","sequence_number":n,"item":project_output_item(&ResponseOutputItem::WebSearchCall(call)),"output_index":0})
            }
            ResponseEvent::Completed(r) => {
                json!({"type":"response.completed","sequence_number":n,"response":project_resource(&r)})
            }
            ResponseEvent::Error(message) => {
                json!({"type":"error","sequence_number":n,"code":"server_error","message":message,"param":null})
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn response_projections_preserve_requested_tools() {
        let tools = vec![json!({
            "type": "function",
            "name": "lookup",
            "description": "Look something up",
            "parameters": {"type": "object"}
        })];
        let tool_choice = json!({"type": "function", "name": "lookup"});
        let mut projection = ResponseProjection::new("m".into()).with_request(StreamResponse {
            model: "m".into(),
            owner: "owner".into(),
            store: false,
            previous_response_id: None,
            input: vec![],
            tools: tools.clone(),
            tool_choice: tool_choice.clone(),
            buffer_output: false,
            expose_reasoning: false,
            strip_reasoning: false,
            web_search_call: None,
            lineage_revision: 0,
        });

        let projected = project_resource(&projection.resource);
        assert_eq!(projected["tools"], json!(tools));
        assert_eq!(projected["tool_choice"], tool_choice);

        let streamed = projection.project(StreamEvent::Started {
            request_id: "resp_1".into(),
            context_id: crate::ContextId::new(),
            correlation_id: "corr_1".into(),
            inference_id: "infer_1".into(),
            execution_session_id: "session_1".into(),
        });
        let first_row = streamed.split("\n\n").next().unwrap();
        let created: Value = serde_json::from_str(first_row.trim_start_matches("data: ")).unwrap();
        assert_eq!(created["response"]["tools"], json!(tools));
        assert_eq!(created["response"]["tool_choice"], tool_choice);
    }

    #[test]
    fn native_reasoning_is_separated_only_at_the_leading_channel_marker() {
        assert_eq!(
            split_native_reasoning("<think>\nprivate work\n</think>\nFinal answer"),
            (Some("private work"), "Final answer")
        );
        assert_eq!(
            split_native_reasoning("<think>unfinished private work"),
            (Some("unfinished private work"), "")
        );
        assert_eq!(
            split_native_reasoning("<think>\n</think>\nhello"),
            (None, "hello")
        );
        assert_eq!(
            split_native_reasoning("Literal <think> tag"),
            (None, "Literal <think> tag")
        );
    }

    #[test]
    fn reasoning_projection_buffers_native_tokens_before_completion() {
        let mut projection = ResponseProjection::new("m".into()).with_request(StreamResponse {
            model: "m".into(),
            owner: "owner".into(),
            store: false,
            previous_response_id: None,
            input: vec![],
            tools: vec![],
            tool_choice: json!("auto"),
            buffer_output: true,
            expose_reasoning: true,
            web_search_call: None,
            strip_reasoning: true,
            lineage_revision: 0,
        });
        let started = projection.project(StreamEvent::Started {
            request_id: "resp_1".into(),
            context_id: crate::ContextId::new(),
            correlation_id: "corr_1".into(),
            inference_id: "infer_1".into(),
            execution_session_id: "session_1".into(),
        });
        assert!(started.contains("response.created"));
        assert_eq!(
            projection.project(StreamEvent::Token {
                token: "<think>private".into(),
                index: 1,
            }),
            ""
        );
    }
}
