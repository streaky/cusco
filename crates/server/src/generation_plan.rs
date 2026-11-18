use crate::native::{
    GenerationInput, GenerationSemantics, NativeToolChoice, PromptContent, PromptPart,
};
use crate::responses::{ResponseInputItem, ResponseResource};
use crate::{Error, InferResponse, ModelRecord, Server};
use serde_json::{Value, json};

pub(crate) struct PreparedGeneration {
    pub prompt: String,
    pub grammar: Option<String>,
    pub buffer_output: bool,
    pub resource: Option<ResourcePlan>,
}
pub(crate) struct CompletedGeneration {
    pub response: Option<ResponseResource>,
    pub output: crate::native::GenerationOutput,
}

fn generation_output(resource: &ResponseResource, text: String) -> crate::native::GenerationOutput {
    let mut function_call = None;
    let mut reasoning = None;
    let mut web_search_call = None;
    for item in &resource.output {
        match item {
            crate::responses::ResponseOutputItem::FunctionCall(call) => {
                function_call = Some(call.clone());
            }
            crate::responses::ResponseOutputItem::Reasoning(item) => reasoning = Some(item.clone()),
            crate::responses::ResponseOutputItem::WebSearchCall(call) => {
                web_search_call = Some(call.clone())
            }
            crate::responses::ResponseOutputItem::Message(_) => {}
        }
    }
    crate::native::GenerationOutput {
        text,
        function_call,
        reasoning,
        web_search_call,
    }
}

pub(crate) struct ResourcePlan {
    service: crate::responses::ResponseService,
    id: String,
    owner: String,
    model: String,
    created_at: Option<u64>,
    store: bool,
    previous_response_id: Option<String>,
    lineage_revision: u64,
    input: Vec<ResponseInputItem>,
    tools: Vec<Value>,
    tool_choice: Value,
    tool_names: Vec<String>,
    choice: NativeToolChoice,
    hosted_search: Option<crate::HostedWebSearchExecution>,
    expose_reasoning: bool,
    strip_reasoning: bool,
    persist: bool,
}

impl ResourcePlan {
    pub(crate) fn persist(&self) -> bool {
        self.persist
    }
    pub(crate) fn started_resource(
        &mut self,
        created_at: u64,
        metadata: crate::responses::ResponseMetadata,
    ) -> ResponseResource {
        self.created_at = Some(created_at);
        ResponseResource {
            schema_version: crate::responses::RESPONSE_SCHEMA_VERSION,
            id: self.id.clone(),
            owner: self.owner.clone(),
            model: self.model.clone(),
            created_at,
            status: "in_progress".into(),
            store: self.store,
            previous_response_id: self.previous_response_id.clone(),
            input: self.input.clone(),
            output: Vec::new(),
            tools: self.tools.clone(),
            tool_choice: self.tool_choice.clone(),
            finish_reason: None,
            usage: None,
            metadata,
            lineage_revision: self.lineage_revision,
            context_update: None,
        }
    }
    pub(crate) fn complete(
        self,
        response: InferResponse,
        reason: crate::FinishReason,
    ) -> Result<CompletedGeneration, Error> {
        let (reasoning, text) = if self.strip_reasoning {
            crate::responses::split_native_reasoning(&response.text)
        } else {
            (None, response.text.as_str())
        };
        let function_call = self.function_call(text)?;
        let web_search_call = self.hosted_search.as_ref().map(|search| {
            crate::responses::new_web_search_call(
                search.query.clone(),
                search.results.iter().map(|result| result.url.clone()),
            )
        });
        let resource = self.service.complete(crate::responses::CompleteResponse {
            id: &self.id,
            created_at: self.created_at,
            owner: &self.owner,
            model: &self.model,
            text,
            reasoning,
            reason,
            usage: &response.usage,
            store: self.persist && self.store,
            previous_response_id: self.previous_response_id.as_deref(),
            tools: &self.tools,
            tool_choice: &self.tool_choice,
            lineage_revision: self.lineage_revision,
            input: &self.input,
            function_call,
            expose_reasoning: self.expose_reasoning,
            web_search_call: web_search_call.as_ref(),
        });
        let output_text = if self.persist {
            String::new()
        } else {
            text.to_owned()
        };
        let output = generation_output(&resource, output_text);
        let response = if self.persist {
            self.service
                .remember(&resource)
                .map_err(crate::responses::store_error)?;
            Some(resource)
        } else {
            None
        };
        Ok(CompletedGeneration { response, output })
    }
    fn function_call(
        &self,
        text: &str,
    ) -> Result<Option<crate::responses::ResponseFunctionCall>, Error> {
        if self.hosted_search.is_some()
            || self.tool_names.is_empty()
            || matches!(self.choice, NativeToolChoice::None)
        {
            return Ok(None);
        }
        let parsed = serde_json::from_str::<Value>(text.trim()).ok();
        let call = parsed
            .as_ref()
            .and_then(Value::as_object)
            .and_then(|object| {
                let name = object.get("name")?.as_str()?.to_owned();
                let arguments = object.get("arguments")?;
                Some((
                    name,
                    arguments
                        .as_str()
                        .map(str::to_owned)
                        .unwrap_or_else(|| arguments.to_string()),
                ))
            });
        if let Some((name, arguments)) = call {
            if !self.tool_names.contains(&name) {
                if matches!(
                    self.choice,
                    NativeToolChoice::Required | NativeToolChoice::Function { .. }
                ) {
                    return Err(Error::BadRequest(format!(
                        "model selected undeclared function `{name}`"
                    )));
                }
                return Ok(None);
            }
            if let NativeToolChoice::Function { name: required } = &self.choice
                && &name != required
            {
                return Err(Error::BadRequest(format!(
                    "model selected `{name}` instead of required function `{required}`"
                )));
            }
            return Ok(Some(crate::responses::new_function_call(name, arguments)));
        }
        if matches!(
            self.choice,
            NativeToolChoice::Required | NativeToolChoice::Function { .. }
        ) {
            return Err(Error::BadRequest(
                "model did not produce the required function call".into(),
            ));
        }
        Ok(None)
    }
}

fn render_incremental_ledger(
    family: &str,
    messages: Vec<crate::prompt::Message>,
    has_lineage: bool,
) -> Result<String, Error> {
    if messages.is_empty() && has_lineage {
        return Ok(String::new());
    }
    crate::prompt::apply_chat_template(
        family,
        crate::responses::normalize_lineage_messages(messages),
    )
}
pub(crate) struct GenerationPlanRequest<'a> {
    pub server: &'a Server,
    pub request_id: &'a str,
    pub owner: &'a str,
    pub model: &'a ModelRecord,
    pub input: GenerationInput,
    pub semantics: GenerationSemantics,
    pub control: std::sync::Arc<crate::RequestControl>,
    pub deadline: std::time::Instant,
}

pub(crate) async fn prepare(
    request: GenerationPlanRequest<'_>,
) -> Result<PreparedGeneration, Error> {
    let GenerationPlanRequest {
        server,
        request_id,
        owner,
        model,
        input,
        semantics,
        control,
        deadline,
    } = request;
    let family = &model.family;
    let continuation_id = semantics
        .continuation
        .as_ref()
        .map(|continuation| continuation.previous_response_id.clone());
    if semantics
        .previous_response_id
        .as_ref()
        .zip(continuation_id.as_ref())
        .is_some_and(|(a, b)| a != b)
    {
        return Err(Error::BadRequest(
            "continuation and previous_response_id disagree".into(),
        ));
    }
    let previous_response_id = semantics.previous_response_id.clone().or(continuation_id);
    let mut prompt = String::new();
    let mut lineage_revision = 0;
    if let Some(previous_id) = &previous_response_id {
        let predecessor = server
            .response_service
            .retrieve(previous_id, owner)
            .map_err(crate::responses::store_error)?;
        if predecessor.model != model.id {
            return Err(Error::BadRequest(
                "previous_response_id uses an incompatible model".into(),
            ));
        }
        if !predecessor.store {
            return Err(Error::BadRequest(
                "previous_response_id requires a stored response".into(),
            ));
        }
        if !matches!(predecessor.status.as_str(), "completed" | "incomplete") {
            return Err(Error::BadRequest(
                "previous_response_id requires a finished response".into(),
            ));
        }
        lineage_revision = predecessor.lineage_revision.saturating_add(1);
        prompt.push_str(&crate::responses::lineage_prompt(
            &server.response_service,
            predecessor,
            owner,
            family,
        )?);
    }

    let mut input_ledger = semantics.input_items.clone();
    let (input_prompt, input_items) = match input {
        GenerationInput::Prompt(text) => (
            text.clone(),
            vec![ResponseInputItem::Message {
                role: "user".into(),
                text,
            }],
        ),
        GenerationInput::Messages(messages) => {
            let mut prompt_messages = Vec::with_capacity(messages.len());
            let mut ledger = Vec::with_capacity(messages.len());
            for message in messages {
                if !matches!(message.role.as_str(), "system" | "user" | "assistant") {
                    return Err(Error::BadRequest(format!(
                        "unsupported message role {}",
                        message.role
                    )));
                }
                let text = match message.content {
                    PromptContent::Text(text) => text,
                    PromptContent::Parts(parts) => {
                        let mut text = String::new();
                        for part in parts {
                            match part {
                                PromptPart::Text(part) => text.push_str(&part),
                                PromptPart::ImageUri(uri) => {
                                    if message.role != "user" {
                                        return Err(Error::BadRequest(
                                            "images are accepted only in user messages".into(),
                                        ));
                                    }
                                    crate::ImageAdmission::new(server.vision_config())
                                        .admit_data_uri(&uri)
                                        .map_err(|error| Error::BadRequest(error.to_string()))?;
                                    return Err(Error::BadRequest("image_unsupported: selected model has no compatible vision projector".into()));
                                }
                            }
                        }
                        text
                    }
                };
                ledger.push(ResponseInputItem::Message {
                    role: message.role.clone(),
                    text: text.clone(),
                });
                prompt_messages.push(crate::prompt::Message {
                    role: message.role,
                    content: text,
                });
            }
            (
                crate::prompt::apply_chat_template(family, prompt_messages)?,
                ledger,
            )
        }
        GenerationInput::Ledger(items) => {
            if !input_ledger.is_empty() && input_ledger != items {
                return Err(Error::BadRequest(
                    "generation ledger disagrees with semantic input items".into(),
                ));
            }
            input_ledger = items.clone();
            let mut call_ids = std::collections::HashSet::new();
            let mut messages = Vec::new();
            for item in items {
                match item {
                    ResponseInputItem::Message { role, text } => {
                        messages.push(crate::prompt::Message {
                            role,
                            content: text,
                        })
                    }
                    ResponseInputItem::FunctionCall {
                        call_id,
                        name,
                        arguments,
                    } => {
                        call_ids.insert(call_id);
                        let arguments = serde_json::from_str::<Value>(&arguments)
                            .unwrap_or(Value::String(arguments));
                        messages.push(crate::prompt::Message {
                            role: "assistant".into(),
                            content: json!({"name":name,"arguments":arguments}).to_string(),
                        });
                    }
                    ResponseInputItem::FunctionCallOutput { call_id, output } => {
                        if !call_ids.contains(&call_id)
                            && !crate::responses::lineage_has_call(
                                &server.response_service,
                                previous_response_id.as_deref(),
                                owner,
                                &call_id,
                            )?
                        {
                            return Err(Error::BadRequest(format!(
                                "function_call_output references unknown call_id `{call_id}`"
                            )));
                        }
                        messages.push(crate::prompt::Message {
                            role: "user".into(),
                            content: format!("Tool result for {call_id}: {output}"),
                        });
                    }
                }
            }
            let input_prompt =
                render_incremental_ledger(family, messages, previous_response_id.is_some())?;
            (input_prompt, input_ledger.clone())
        }
    };
    if input_ledger.is_empty() {
        input_ledger = input_items.clone();
    } else if input_ledger != input_items {
        return Err(Error::BadRequest(
            "generation input disagrees with semantic input items".into(),
        ));
    }
    if !input_items.is_empty() {
        if prompt.is_empty() {
            prompt = input_prompt;
        } else {
            crate::prompt::append_rendered(family, &mut prompt, &input_prompt)?;
        }
    }

    let tool_values = semantics
        .tools
        .iter()
        .map(crate::tool_value)
        .collect::<Vec<_>>();
    crate::validate_tools(&semantics.tools, &semantics.tool_choice)?;
    let choice = semantics.tool_choice.clone();
    let tool_choice_value = match &choice {
        NativeToolChoice::Auto => json!("auto"),
        NativeToolChoice::None => json!("none"),
        NativeToolChoice::Required => json!("required"),
        NativeToolChoice::Function { name } => json!({"type":"function","name":name}),
    };
    let has_tool_output = input_ledger
        .iter()
        .any(|item| matches!(item, ResponseInputItem::FunctionCallOutput { .. }));
    let responses_tools = semantics.record_output;
    let hosted_search = if responses_tools {
        crate::execute_hosted_web_search(
            server,
            model,
            &prompt,
            &semantics.tools,
            &choice,
            crate::generation_tools::HostedSearchContext {
                principal: owner,
                control: &control,
                deadline,
            },
        )
        .await?
    } else {
        None
    };
    if let Some(search) = &hosted_search {
        crate::append_web_search_evidence(family, &mut prompt, search)?;
    }
    if responses_tools {
        crate::append_tool_instructions(
            family,
            &mut prompt,
            &semantics.tools,
            &choice,
            has_tool_output,
        )?;
    }
    let mut grammar: Option<String> = None;
    for format in &semantics.format_constraints {
        if let Some(compiled) = crate::generation_tools::response_format_grammar(format)? {
            if grammar
                .as_ref()
                .is_some_and(|existing| existing != &compiled)
            {
                return Err(Error::BadRequest(
                    "response_format constraints conflict".into(),
                ));
            }
            grammar.get_or_insert(compiled);
        }
    }
    let strips_native_reasoning = matches!(family.as_str(), "qwen35moe" | "qwen3moe");
    let strip_reasoning =
        strips_native_reasoning && (semantics.record_output || semantics.expose_reasoning);
    let tool_names = if responses_tools {
        crate::tool_names(&semantics.tools)
    } else {
        Vec::new()
    };
    let buffer_output = !tool_names.is_empty() || hosted_search.is_some() || strip_reasoning;
    let resource_tools = if semantics.tool_snapshot.is_empty() {
        tool_values.clone()
    } else {
        semantics.tool_snapshot.clone()
    };
    let resource_tool_choice = semantics
        .tool_choice_snapshot
        .clone()
        .unwrap_or_else(|| tool_choice_value.clone());
    let resource = (semantics.record_output || semantics.expose_reasoning).then(|| ResourcePlan {
        service: server.response_service.clone(),
        id: request_id.to_owned(),
        owner: owner.to_owned(),
        model: model.id.clone(),
        created_at: None,
        store: semantics.store,
        previous_response_id,
        lineage_revision,
        input: input_ledger,
        tools: resource_tools,
        tool_choice: resource_tool_choice,
        tool_names,
        choice,
        hosted_search,
        expose_reasoning: semantics.expose_reasoning && strips_native_reasoning,
        strip_reasoning,
        persist: semantics.record_output,
    });
    Ok(PreparedGeneration {
        prompt,
        grammar,
        buffer_output,
        resource,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn response_test_usage() -> crate::Usage {
        crate::Usage {
            input_tokens: 0,
            generated_tokens: 0,
            evaluated_tokens: 0,
            cached_tokens: 0,
            prefill: Default::default(),
            model: "model".into(),
            model_revision: "revision".into(),
            context_id: crate::ContextId::new(),
            compaction_result: None,
            latency_ms: 0,
            status: "completed".into(),
            correlation_id: "correlation".into(),
            inference_id: "inference".into(),
            execution_session_id: "session".into(),
        }
    }

    #[test]
    fn generated_function_selection_respects_required_tools() {
        let root = std::env::temp_dir().join(format!("cusco-tool-plan-{}", uuid::Uuid::new_v4()));
        let store = crate::response_store::FileResponseResourceStore::open(&root).unwrap();
        let mut plan = ResourcePlan {
            service: crate::responses::ResponseService::new(std::sync::Arc::new(store)),
            id: "response".into(),
            owner: "owner".into(),
            model: "model".into(),
            created_at: None,
            store: false,
            previous_response_id: None,
            lineage_revision: 0,
            input: Vec::new(),
            tools: Vec::new(),
            tool_choice: json!("auto"),
            tool_names: vec!["lookup".into()],
            choice: NativeToolChoice::Auto,
            hosted_search: None,
            expose_reasoning: false,
            strip_reasoning: false,
            persist: false,
        };
        let unknown = r#"{"name":"describe","arguments":{"subject":"function calls"}}"#;
        assert!(plan.function_call(unknown).unwrap().is_none());
        plan.choice = NativeToolChoice::Required;
        assert!(
            matches!(plan.function_call(unknown), Err(Error::BadRequest(message))
            if message.contains("undeclared function `describe`"))
        );
        let (reasoning, visible) = crate::responses::split_native_reasoning(
            "<think>checking the tool selection</think>\n{\"name\":\"lookup\",\"arguments\":{\"query\":\"rust\"}}",
        );
        assert_eq!(reasoning, Some("checking the tool selection"));
        let call = plan.function_call(visible).unwrap().unwrap();
        assert_eq!(call.name, "lookup");
        assert_eq!(
            serde_json::from_str::<Value>(&call.arguments).unwrap(),
            json!({"query":"rust"})
        );
        plan.choice = NativeToolChoice::Auto;
        assert!(plan.function_call(visible).unwrap().is_some());
        std::fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn empty_incremental_ledger_keeps_existing_lineage_prompt() {
        assert_eq!(
            render_incremental_ledger("gemma4", Vec::new(), true).unwrap(),
            ""
        );
        assert!(render_incremental_ledger("gemma4", Vec::new(), false).is_err());
    }

    #[test]
    fn complete_parses_tool_json_after_stripping_reasoning() {
        let root =
            std::env::temp_dir().join(format!("cusco-reasoning-tool-{}", uuid::Uuid::new_v4()));
        let store = crate::response_store::FileResponseResourceStore::open(&root).unwrap();
        let service = crate::responses::ResponseService::new(std::sync::Arc::new(store));
        for choice in [NativeToolChoice::Auto, NativeToolChoice::Required] {
            let plan = ResourcePlan {
                service: service.clone(),
                id: uuid::Uuid::new_v4().to_string(),
                owner: "owner".into(),
                model: "model".into(),
                created_at: None,
                store: false,
                previous_response_id: None,
                lineage_revision: 0,
                input: Vec::new(),
                tools: Vec::new(),
                tool_choice: json!("auto"),
                tool_names: vec!["lookup".into()],
                choice,
                hosted_search: None,
                expose_reasoning: true,
                strip_reasoning: true,
                persist: true,
            };
            let completed = plan
                .complete(
                    InferResponse {
                        id: "inference".into(),
                        text: "<think>private reasoning</think>\n{\"name\":\"lookup\",\"arguments\":{\"query\":\"rust\"}}".into(),
                        usage: response_test_usage(),
                    },
                    crate::FinishReason::Stop,
                )
                .unwrap();
            let response = completed.response.unwrap();
            assert_eq!(response.output.len(), 1);
            assert!(matches!(
                &response.output[0],
                crate::responses::ResponseOutputItem::FunctionCall(call)
                    if call.name == "lookup"
            ));
            assert!(completed.output.function_call.is_some());
            assert!(completed.output.reasoning.is_none());
            assert!(completed.output.text.is_empty());
        }
        let plan = ResourcePlan {
            service: service.clone(),
            id: uuid::Uuid::new_v4().to_string(),
            owner: "owner".into(),
            model: "model".into(),
            created_at: None,
            store: false,
            previous_response_id: None,
            lineage_revision: 0,
            input: Vec::new(),
            tools: Vec::new(),
            tool_choice: json!("auto"),
            tool_names: Vec::new(),
            choice: NativeToolChoice::Auto,
            hosted_search: None,
            expose_reasoning: true,
            strip_reasoning: true,
            persist: true,
        };
        let completed = plan
            .complete(
                InferResponse {
                    id: "inference".into(),
                    text: "<think>private reasoning</think>\nvisible answer".into(),
                    usage: response_test_usage(),
                },
                crate::FinishReason::Stop,
            )
            .unwrap();
        let response = completed.response.unwrap();
        assert_eq!(response.output.len(), 2);
        assert!(matches!(
            &response.output[0],
            crate::responses::ResponseOutputItem::Reasoning(reasoning)
                if reasoning.summary[0].text == "private reasoning"
        ));
        assert!(matches!(
            &response.output[1],
            crate::responses::ResponseOutputItem::Message(message)
                if message.content[0].text == "visible answer"
                    && !message.content[0].text.contains("<think>")
        ));
        std::fs::remove_dir_all(root).unwrap();
    }
}
