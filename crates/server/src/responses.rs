use crate::response_store::{ResponseResourceStore, StoreError};
use crate::{FinishReason, Usage};
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
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct ResponseSnapshot {
    pub id: String,
    pub model: String,
    pub created_at: u64,
    pub status: String,
    pub store: bool,
    pub previous_response_id: Option<String>,
    pub output: Vec<ResponseOutputItem>,
    pub tools: Vec<Value>,
    pub tool_choice: Value,
    pub finish_reason: Option<FinishReason>,
    pub usage: Option<ResponseUsage>,
    pub metadata: ResponseMetadata,
    pub lineage_revision: u64,
    pub context_update: Option<ContextUpdateMetadata>,
}

impl From<&ResponseResource> for ResponseSnapshot {
    fn from(resource: &ResponseResource) -> Self {
        Self {
            id: resource.id.clone(),
            model: resource.model.clone(),
            created_at: resource.created_at,
            status: resource.status.clone(),
            store: resource.store,
            previous_response_id: resource.previous_response_id.clone(),
            output: resource.output.clone(),
            tools: resource.tools.clone(),
            tool_choice: resource.tool_choice.clone(),
            finish_reason: resource.finish_reason,
            usage: resource.usage.clone(),
            metadata: resource.metadata.clone(),
            lineage_revision: resource.lineage_revision,
            context_update: resource.context_update.clone(),
        }
    }
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
pub(crate) fn in_progress_resource(
    id: &str,
    owner: &str,
    model: &str,
    created_at: u64,
    store: bool,
    previous_response_id: Option<&str>,
    input: &[ResponseInputItem],
    tools: &[Value],
    tool_choice: &Value,
    lineage_revision: u64,
    metadata: ResponseMetadata,
) -> ResponseResource {
    ResponseResource {
        schema_version: RESPONSE_SCHEMA_VERSION,
        id: id.to_owned(),
        owner: owner.to_owned(),
        model: model.to_owned(),
        created_at,
        status: "in_progress".into(),
        store,
        previous_response_id: previous_response_id.map(str::to_owned),
        input: input.to_vec(),
        output: Vec::new(),
        tools: tools.to_vec(),
        tool_choice: tool_choice.clone(),
        finish_reason: None,
        usage: None,
        metadata,
        lineage_revision,
        context_update: None,
    }
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

    pub fn complete(&self, params: CompleteResponse<'_>) -> ResponseResource {
        let reasoning = params.reasoning;
        let text = params.text;
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
                id: format!("msg_{}", params.id),
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
            created_at: params.created_at.unwrap_or_else(now),
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
    pub created_at: Option<u64>,
    pub text: &'a str,
    pub reasoning: Option<&'a str>,
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

pub(crate) fn now() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |duration| duration.as_secs())
}

pub(crate) fn split_native_reasoning(text: &str) -> (Option<&str>, &str) {
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


pub(crate) fn store_error(error: crate::response_store::StoreError) -> crate::Error {
    match error {
        crate::response_store::StoreError::NotFound(_) => crate::Error::ContextNotFound,
        crate::response_store::StoreError::StaleRevision { .. }
        | crate::response_store::StoreError::Conflict(_) => crate::Error::Conflict(error.to_string()),
        other => crate::Error::State(other.to_string()),
    }
}

/// Reconstructs a response lineage into the model family's native chat prompt.
pub(crate) fn lineage_prompt(
    service: &ResponseService,
    mut resource: ResponseResource,
    owner: &str,
    family: &str,
) -> Result<String, crate::Error> {
    let mut lineage = Vec::new();
    for _ in 0..1024 {
        let previous = resource.previous_response_id.clone();
        lineage.push(resource);
        let Some(id) = previous else { break };
        resource = service.retrieve(&id, owner).map_err(store_error)?;
    }
    if lineage.last().is_some_and(|resource| resource.previous_response_id.is_some()) {
        return Err(crate::Error::BadRequest("response lineage exceeds 1024 resources".into()));
    }
    lineage.reverse();
    let mut messages = Vec::new();
    for resource in lineage {
        for item in resource.input {
            match item {
                ResponseInputItem::Message { role, text } => messages.push(crate::prompt::Message { role, content: text }),
                ResponseInputItem::FunctionCall { name, arguments, .. } => {
                    let call = json!({
                        "name": name,
                        "arguments": serde_json::from_str::<Value>(&arguments).unwrap_or(Value::String(arguments)),
                    });
                    messages.push(crate::prompt::Message { role: "assistant".into(), content: call.to_string() });
                }
                ResponseInputItem::FunctionCallOutput { call_id, output } => {
                    messages.push(crate::prompt::Message { role: "user".into(), content: format!("Tool result for {call_id}: {output}") });
                }
            }
        }
        for item in resource.output {
            match item {
                ResponseOutputItem::Message(message) => messages.push(crate::prompt::Message {
                    role: "assistant".into(),
                    content: message.content.into_iter().map(|part| part.text).collect(),
                }),
                ResponseOutputItem::FunctionCall(call) => {
                    let call = json!({
                        "name": call.name,
                        "arguments": serde_json::from_str::<Value>(&call.arguments).unwrap_or(Value::String(call.arguments)),
                    });
                    messages.push(crate::prompt::Message { role: "assistant".into(), content: call.to_string() });
                }
                ResponseOutputItem::WebSearchCall(_) | ResponseOutputItem::Reasoning(_) => {}
            }
        }
    }
    crate::prompt::apply_chat_template(family, normalize_lineage_messages(messages))
}

pub(crate) fn normalize_lineage_messages(messages: Vec<crate::prompt::Message>) -> Vec<crate::prompt::Message> {
    let mut normalized: Vec<crate::prompt::Message> = Vec::with_capacity(messages.len());
    for message in messages {
        if normalized.last().is_some_and(|previous| previous.role == message.role) {
            let previous = normalized.last_mut().expect("last message was present");
            if !previous.content.is_empty() && !message.content.is_empty() {
                previous.content.push('\n');
            }
            previous.content.push_str(&message.content);
        } else {
            normalized.push(message);
        }
    }
    normalized
}

pub(crate) fn lineage_has_call(
    service: &ResponseService,
    previous: Option<&str>,
    owner: &str,
    call_id: &str,
) -> Result<bool, crate::Error> {
    let Some(id) = previous else { return Ok(false) };
    let mut resource = service.retrieve(id, owner).map_err(store_error)?;
    for _ in 0..1024 {
        if resource.output.iter().any(|item| matches!(item, ResponseOutputItem::FunctionCall(call) if call.call_id == call_id)) {
            return Ok(true);
        }
        let Some(id) = resource.previous_response_id.clone() else { return Ok(false) };
        resource = service.retrieve(&id, owner).map_err(store_error)?;
    }
    Err(crate::Error::BadRequest("response lineage exceeds 1024 resources".into()))
}

#[cfg(test)]
mod tests {
    use super::*;

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
    fn lineage_coalesces_adjacent_assistant_items() {
        let messages = normalize_lineage_messages(vec![
            crate::prompt::Message { role: "user".into(), content: "question".into() },
            crate::prompt::Message { role: "assistant".into(), content: "answer".into() },
            crate::prompt::Message {
                role: "assistant".into(),
                content: r#"{"name":"lookup","arguments":{}}"#.into(),
            },
        ]);
        assert_eq!(messages.len(), 2);
        assert_eq!(messages[1].role, "assistant");
        assert_eq!(messages[1].content, "answer\n{\"name\":\"lookup\",\"arguments\":{}}");
        crate::prompt::apply_chat_template("gemma4", messages).unwrap();
    }
}
