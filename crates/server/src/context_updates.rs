use crate::responses::{
    ContextUpdateMetadata, RESPONSE_SCHEMA_VERSION, ResponseInputItem, ResponseMessage,
    ResponseMetadata, ResponseOutputItem, ResponseResource, ResponseService, ResponseTextPart,
};
use crate::{Error, FinishReason, RequestControl};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};
use uuid::Uuid;

pub const CONTEXT_UPDATE_CAPABILITY: &str = "cusco.context_update.v1";

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ContextUpdateRequest {
    pub base_response_id: String,
    pub expected_revision: u64,
    #[serde(default)]
    pub operation_id: Option<String>,
    pub operation: ContextUpdateOperation,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case", deny_unknown_fields)]
pub enum ContextUpdateOperation {
    Fold { items: Vec<ResponseInputItem> },
}

impl ContextUpdateOperation {
    fn name(&self) -> &'static str {
        match self {
            Self::Fold { .. } => "fold",
        }
    }

    fn items(&self) -> &[ResponseInputItem] {
        match self {
            Self::Fold { items } => items,
        }
    }
}

#[derive(Clone, Debug, Serialize)]
pub struct ContextUpdateResult {
    pub capability: &'static str,
    pub operation_id: String,
    pub correlation_id: String,
    pub base_response_id: String,
    pub base_revision: u64,
    pub revision: u64,
    pub response: Value,
}

pub fn apply(
    service: &ResponseService,
    owner: &str,
    correlation_id: String,
    request: ContextUpdateRequest,
    control: &Arc<RequestControl>,
) -> Result<ContextUpdateResult, Error> {
    if request.base_response_id.is_empty() {
        return Err(Error::BadRequest(
            "base_response_id must not be empty".into(),
        ));
    }
    let items = request.operation.items();
    if items.is_empty() {
        return Err(Error::BadRequest("fold items must not be empty".into()));
    }
    if items
        .iter()
        .any(|item| !matches!(item, ResponseInputItem::Message { .. }))
    {
        return Err(Error::BadRequest(
            "fold items must be portable message items".into(),
        ));
    }

    let base = service
        .retrieve(&request.base_response_id, owner)
        .map_err(crate::response_store_error)?;
    if !base.store {
        return Err(Error::BadRequest(
            "context updates require a stored base response".into(),
        ));
    }
    control.check()?;

    let operation_id = request
        .operation_id
        .unwrap_or_else(|| format!("ctxupd_{}", Uuid::new_v4()));
    if operation_id.is_empty() {
        return Err(Error::BadRequest("operation_id must not be empty".into()));
    }
    let operation_name = request.operation.name().to_owned();
    let output = items
        .iter()
        .enumerate()
        .map(|(index, item)| match item {
            ResponseInputItem::Message { text, .. } => {
                ResponseOutputItem::Message(ResponseMessage {
                    id: format!("msg_{}_{index}", Uuid::new_v4()),
                    role: "assistant".into(),
                    status: "completed".into(),
                    content: vec![ResponseTextPart {
                        text: text.clone(),
                        annotations: vec![],
                        logprobs: vec![],
                    }],
                })
            }
            ResponseInputItem::FunctionCallOutput { .. } => unreachable!("validated above"),
        })
        .collect();
    let revision = request.expected_revision.saturating_add(1);
    let response = ResponseResource {
        schema_version: RESPONSE_SCHEMA_VERSION,
        id: format!("resp_{}", Uuid::new_v4()),
        owner: owner.into(),
        model: base.model,
        created_at: now(),
        status: "completed".into(),
        store: true,
        previous_response_id: Some(request.base_response_id.clone()),
        input: items.to_vec(),
        output,
        finish_reason: Some(FinishReason::Stop),
        usage: None,
        metadata: ResponseMetadata {
            correlation_id: correlation_id.clone(),
            inference_id: operation_id.clone(),
            execution_session_id: String::new(),
        },
        lineage_revision: revision,
        context_update: Some(ContextUpdateMetadata {
            operation_id: operation_id.clone(),
            operation: operation_name,
            base_response_id: request.base_response_id.clone(),
            base_revision: request.expected_revision,
            correlation_id: correlation_id.clone(),
        }),
    };
    control.check()?;
    service
        .commit_successor(
            &request.base_response_id,
            request.expected_revision,
            &response,
        )
        .map_err(crate::response_store_error)?;

    Ok(ContextUpdateResult {
        capability: CONTEXT_UPDATE_CAPABILITY,
        operation_id,
        correlation_id,
        base_response_id: request.base_response_id,
        base_revision: request.expected_revision,
        revision,
        response: crate::responses::project_resource(&response),
    })
}

fn now() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |duration| duration.as_secs())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::response_store::{FileResponseResourceStore, ResponseResourceStore, StoreError};
    use crate::responses::ResponseUsage;
    use std::fs;

    fn base(id: &str) -> ResponseResource {
        ResponseResource {
            schema_version: RESPONSE_SCHEMA_VERSION,
            id: id.into(),
            owner: "owner".into(),
            model: "model".into(),
            created_at: 1,
            status: "completed".into(),
            store: true,
            previous_response_id: None,
            input: vec![],
            output: vec![],
            finish_reason: Some(FinishReason::Stop),
            usage: Some(ResponseUsage {
                input_tokens: 1,
                cached_tokens: 0,
                output_tokens: 1,
            }),
            metadata: ResponseMetadata::default(),
            lineage_revision: 0,
            context_update: None,
        }
    }

    fn request(base_response_id: &str, expected_revision: u64) -> ContextUpdateRequest {
        ContextUpdateRequest {
            base_response_id: base_response_id.into(),
            expected_revision,
            operation_id: Some("ctxupd_test".into()),
            operation: ContextUpdateOperation::Fold {
                items: vec![ResponseInputItem::Message {
                    role: "user".into(),
                    text: "portable summary".into(),
                }],
            },
        }
    }

    #[test]
    fn update_commits_once_and_recovers_after_restart() {
        let directory =
            std::env::temp_dir().join(format!("cusco-context-update-{}", Uuid::new_v4()));
        let store = Arc::new(FileResponseResourceStore::open(&directory).unwrap());
        store.put(&base("resp_base")).unwrap();
        let service = ResponseService::new(store);
        let result = apply(
            &service,
            "owner",
            "corr_test".into(),
            request("resp_base", 0),
            &Arc::new(RequestControl::new()),
        )
        .unwrap();
        assert_eq!(result.revision, 1);
        assert_eq!(result.response["cusco"]["lineage_revision"], 1);

        let reopened = FileResponseResourceStore::open(&directory).unwrap();
        let recovered = reopened
            .get(result.response["id"].as_str().unwrap())
            .unwrap();
        assert_eq!(recovered.previous_response_id.as_deref(), Some("resp_base"));
        assert_eq!(
            recovered
                .context_update
                .as_ref()
                .map(|metadata| metadata.operation_id.as_str()),
            Some("ctxupd_test")
        );
        fs::remove_dir_all(directory).unwrap();
    }

    #[test]
    fn stale_conflicting_and_cancelled_updates_publish_nothing() {
        let directory =
            std::env::temp_dir().join(format!("cusco-context-update-{}", Uuid::new_v4()));
        let store = Arc::new(FileResponseResourceStore::open(&directory).unwrap());
        store.put(&base("resp_base")).unwrap();
        let service = ResponseService::new(store.clone());

        let stale = apply(
            &service,
            "owner",
            "corr_stale".into(),
            request("resp_base", 1),
            &Arc::new(RequestControl::new()),
        )
        .unwrap_err();
        assert!(matches!(stale, Error::Conflict(_)));

        apply(
            &service,
            "owner",
            "corr_first".into(),
            request("resp_base", 0),
            &Arc::new(RequestControl::new()),
        )
        .unwrap();
        let conflict = apply(
            &service,
            "owner",
            "corr_second".into(),
            request("resp_base", 0),
            &Arc::new(RequestControl::new()),
        )
        .unwrap_err();
        assert!(matches!(conflict, Error::Conflict(_)));

        store.put(&base("resp_cancel")).unwrap();
        let cancelled = Arc::new(RequestControl::new());
        cancelled.cancel();
        assert!(matches!(
            apply(
                &service,
                "owner",
                "corr_cancel".into(),
                request("resp_cancel", 0),
                &cancelled,
            ),
            Err(Error::Cancelled)
        ));
        assert!(matches!(
            store.get("resp_cancel"),
            Ok(ResponseResource {
                lineage_revision: 0,
                ..
            })
        ));
        let mut successor = base("unused");
        successor.previous_response_id = Some("resp_cancel".into());
        successor.lineage_revision = 1;
        assert!(store.put_successor("resp_cancel", 0, &successor).is_ok());
        let mut conflicting = base("unused_again");
        conflicting.previous_response_id = Some("resp_cancel".into());
        conflicting.lineage_revision = 1;
        assert!(matches!(
            store.put_successor("resp_cancel", 0, &conflicting),
            Err(StoreError::Conflict(_))
        ));
        fs::remove_dir_all(directory).unwrap();
    }
}
