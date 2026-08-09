use crate::response_store::{ResponseResourceStore, StoreError};
use crate::{FinishReason, StreamEvent, Usage};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use std::time::{SystemTime, UNIX_EPOCH};

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct ResponseTextPart {
    pub text: String,
    pub annotations: Vec<Value>,
    pub logprobs: Vec<Value>,
}

#[derive(Clone)]
pub struct ResponseService {
    store: std::sync::Arc<dyn ResponseResourceStore>,
}

impl ResponseService {
    pub fn new(store: std::sync::Arc<dyn ResponseResourceStore>) -> Self { Self { store } }
    pub fn complete(&self, id: String, model: String, text: String, reason: FinishReason, usage: &Usage) -> ResponseResource { ResponseLifecycle::complete(id, model, text, reason, usage) }
    pub fn lifecycle(&self, model: String) -> ResponseLifecycle { ResponseLifecycle::new(model) }
    pub fn projection(&self, model: String) -> ResponseProjection { ResponseProjection::new(model) }
    pub fn persist(&self, resource: &ResponseResource) -> Result<(), StoreError> { self.store.put(resource) }
    pub fn retrieve(&self, id: &str) -> Result<ResponseResource, StoreError> { self.store.get(id) }
    pub fn delete(&self, id: &str) -> Result<(), StoreError> { self.store.delete(id) }
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct ResponseMessage {
    pub id: String,
    pub role: String,
    pub status: String,
    pub content: Vec<ResponseTextPart>,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct ResponseResource {
    pub id: String,
    pub model: String,
    pub created_at: u64,
    pub status: String,
    pub output: Vec<ResponseMessage>,
    pub finish_reason: Option<FinishReason>,
    pub usage: Option<ResponseUsage>,
    pub metadata: ResponseMetadata,
}

#[derive(Clone, Debug, Default, Deserialize, PartialEq, Serialize)]
pub struct ResponseMetadata {
    pub correlation_id: String,
    pub inference_id: String,
    pub execution_session_id: String,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct ResponseUsage {
    pub input_tokens: usize,
    pub cached_tokens: usize,
    pub output_tokens: usize,
}

#[derive(Clone, Debug, PartialEq)]
pub enum ResponseEvent {
    Created(ResponseResource),
    OutputItemAdded(ResponseMessage),
    ContentPartAdded(ResponseTextPart),
    TextDelta(String),
    TextDone(ResponseTextPart),
    ContentPartDone(ResponseTextPart),
    OutputItemDone(ResponseMessage),
    Completed(ResponseResource),
    Error(String),
}

pub struct ResponseLifecycle {
    resource: ResponseResource,
}

impl ResponseLifecycle {
    pub fn new(model: String) -> Self {
        Self {
            resource: ResponseResource {
                id: String::new(),
                model,
                created_at: SystemTime::now().duration_since(UNIX_EPOCH).map_or(0, |d| d.as_secs()),
                status: "in_progress".into(),
                output: Vec::new(),
                finish_reason: None,
                usage: None,
                metadata: ResponseMetadata::default(),
            },
        }
    }

    fn part(&self) -> ResponseTextPart {
        ResponseTextPart { text: self.resource.output.first().and_then(|m| m.content.first()).map_or_else(String::new, |p| p.text.clone()), annotations: Vec::new(), logprobs: Vec::new() }
    }

    fn message(&self, status: &str) -> ResponseMessage {
        ResponseMessage { id: "msg_0".into(), role: "assistant".into(), status: status.into(), content: vec![self.part()] }
    }

    pub fn apply(&mut self, event: StreamEvent) -> Vec<ResponseEvent> {
        match event {
            StreamEvent::Started { request_id, correlation_id, inference_id, execution_session_id, .. } => {
                self.resource.id = request_id;
                self.resource.metadata = ResponseMetadata { correlation_id, inference_id, execution_session_id };
                self.resource.output = vec![self.message("in_progress")];
                vec![ResponseEvent::Created(self.resource.clone()), ResponseEvent::OutputItemAdded(self.message("in_progress")), ResponseEvent::ContentPartAdded(self.part())]
            }
            StreamEvent::Token { token, .. } => {
                if self.resource.output.is_empty() { self.resource.output.push(self.message("in_progress")); }
                self.resource.output[0].content[0].text.push_str(&token);
                vec![ResponseEvent::TextDelta(token)]
            }
            StreamEvent::Finished { reason, usage } => {
                self.resource.status = "completed".into();
                self.resource.finish_reason = Some(reason);
                self.resource.usage = Some(ResponseUsage::from(usage.as_ref()));
                self.resource.output[0].status = "completed".into();
                let part = self.part();
                vec![ResponseEvent::TextDone(part.clone()), ResponseEvent::ContentPartDone(part), ResponseEvent::OutputItemDone(self.resource.output[0].clone()), ResponseEvent::Completed(self.resource.clone())]
            }
            StreamEvent::Error { message } => vec![ResponseEvent::Error(message)],
        }
    }

    pub fn complete(id: String, model: String, text: String, reason: FinishReason, usage: &Usage) -> ResponseResource {
        let mut lifecycle = Self::new(model);
        lifecycle.resource.id = id;
        lifecycle.resource.metadata = ResponseMetadata { correlation_id: usage.correlation_id.clone(), inference_id: usage.inference_id.clone(), execution_session_id: usage.execution_session_id.clone() };
        lifecycle.resource.output = vec![ResponseMessage { id: "msg_0".into(), role: "assistant".into(), status: "completed".into(), content: vec![ResponseTextPart { text, annotations: vec![], logprobs: vec![] }] }];
        lifecycle.resource.status = "completed".into();
        lifecycle.resource.finish_reason = Some(reason);
        lifecycle.resource.usage = Some(ResponseUsage::from(usage));
        lifecycle.resource
    }
}

impl From<&Usage> for ResponseUsage {
    fn from(usage: &Usage) -> Self { Self { input_tokens: usage.input_tokens, cached_tokens: usage.cached_tokens, output_tokens: usage.generated_tokens } }
}

pub fn project_resource(resource: &ResponseResource) -> Value {
    let usage = resource.usage.as_ref().map(|u| json!({"input_tokens":u.input_tokens,"input_tokens_details":{"cached_tokens":u.cached_tokens},"output_tokens":u.output_tokens,"output_tokens_details":{"reasoning_tokens":0},"total_tokens":u.input_tokens+u.output_tokens})).unwrap_or(Value::Null);
    let output = resource.output.iter().map(|m| json!({"id":m.id,"type":"message","role":m.role,"status":m.status,"content":m.content.iter().map(|p| json!({"type":"output_text","text":p.text,"annotations":p.annotations,"logprobs":p.logprobs})).collect::<Vec<_>>() })).collect::<Vec<_>>();
    json!({"id":resource.id,"object":"response","created_at":resource.created_at,"status":resource.status,"background":false,"error":null,"incomplete_details":null,"instructions":null,"max_output_tokens":null,"metadata":resource.metadata,"model":resource.model,"output":output,"parallel_tool_calls":true,"previous_response_id":null,"prompt_cache_key":null,"reasoning":null,"safety_identifier":null,"service_tier":"default","store":false,"temperature":1.0,"text":{"format":{"type":"text"},"verbosity":"medium"},"tool_choice":"auto","tools":[],"top_logprobs":0,"top_p":1.0,"truncation":"disabled","usage":usage,"finish_reason":resource.finish_reason})
}

pub struct ResponseProjection { lifecycle: ResponseLifecycle, sequence: usize }
impl ResponseProjection {
    pub fn new(model: String) -> Self { Self { lifecycle: ResponseLifecycle::new(model), sequence: 0 } }
    pub fn project(&mut self, event: StreamEvent) -> String {
        self.lifecycle.apply(event).into_iter().fold(String::new(), |mut rows, event| { use std::fmt::Write as _; let value = self.project_event(event); write!(rows, "data: {value}\n\n").expect("String write"); rows })
    }
    fn project_event(&mut self, event: ResponseEvent) -> Value {
        let sequence_number = self.sequence; self.sequence += 1;
        match event {
            ResponseEvent::Created(r) => json!({"type":"response.created","sequence_number":sequence_number,"response":project_resource(&r)}),
            ResponseEvent::OutputItemAdded(m) => json!({"type":"response.output_item.added","sequence_number":sequence_number,"item":project_message(&m),"output_index":0}),
            ResponseEvent::ContentPartAdded(p) => json!({"type":"response.content_part.added","sequence_number":sequence_number,"item_id":"msg_0","output_index":0,"content_index":0,"part":project_part(&p)}),
            ResponseEvent::TextDelta(delta) => json!({"type":"response.output_text.delta","sequence_number":sequence_number,"item_id":"msg_0","output_index":0,"content_index":0,"delta":delta,"logprobs":[]}),
            ResponseEvent::TextDone(p) => json!({"type":"response.output_text.done","sequence_number":sequence_number,"item_id":"msg_0","output_index":0,"content_index":0,"text":p.text,"logprobs":[]}),
            ResponseEvent::ContentPartDone(p) => json!({"type":"response.content_part.done","sequence_number":sequence_number,"item_id":"msg_0","output_index":0,"content_index":0,"part":project_part(&p)}),
            ResponseEvent::OutputItemDone(m) => json!({"type":"response.output_item.done","sequence_number":sequence_number,"item":project_message(&m),"output_index":0}),
            ResponseEvent::Completed(r) => json!({"type":"response.completed","sequence_number":sequence_number,"response":project_resource(&r)}),
            ResponseEvent::Error(message) => json!({"type":"error","sequence_number":sequence_number,"code":"server_error","message":message,"param":null}),
        }
    }
}
fn project_part(p: &ResponseTextPart) -> Value { json!({"type":"output_text","text":p.text,"annotations":p.annotations,"logprobs":p.logprobs}) }
fn project_message(m: &ResponseMessage) -> Value { json!({"id":m.id,"type":"message","role":m.role,"status":m.status,"content":m.content.iter().map(project_part).collect::<Vec<_>>()}) }

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{ContextId, PrefillMetrics};

    fn usage() -> Box<Usage> {
        Box::new(Usage {
            input_tokens: 2,
            generated_tokens: 1,
            evaluated_tokens: 3,
            cached_tokens: 0,
            prefill: PrefillMetrics::default(),
            model: "m".into(),
            model_revision: "r".into(),
            context_id: ContextId::new(),
            compaction_result: None,
            latency_ms: 1,
            status: "completed".into(),
            correlation_id: "correlation".into(),
            inference_id: "inference".into(),
            execution_session_id: "session".into(),
        })
    }

    #[test]
    fn streamed_terminal_resource_matches_buffered_resource() {
        let usage = usage();
        let buffered = ResponseLifecycle::complete(
            "resp_1".into(),
            "m".into(),
            "hello".into(),
            FinishReason::Stop,
            &usage,
        );
        let mut streamed = ResponseLifecycle::new("m".into());
        streamed.apply(StreamEvent::Started {
            request_id: "resp_1".into(),
            context_id: ContextId::new(),
            correlation_id: "correlation".into(),
            inference_id: "inference".into(),
            execution_session_id: "session".into(),
        });
        streamed.apply(StreamEvent::Token {
            token: "hello".into(),
            index: 0,
        });
        let events = streamed.apply(StreamEvent::Finished {
            reason: FinishReason::Stop,
            usage,
        });
        let ResponseEvent::Completed(terminal) = events.last().unwrap() else {
            panic!("stream must end with a completed response");
        };

        assert_eq!(project_resource(terminal), project_resource(&buffered));
    }
}
