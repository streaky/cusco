//! Transport-neutral model/context management used by host applications.
use crate::native::{DomainFailure, DomainOutcome};
use crate::{
    AdmissionMetrics, CompactionDeclaration, CompactionStrategyCatalog, ContextId, ContextRecord,
    CreateCompactionDeclaration, Error, ModelRecord, Server, ServerConfig,
};
use serde::Serialize;
use std::sync::Arc;
use tokio::sync::{Mutex, mpsc};

fn outcome<T>(result: Result<T, Error>) -> DomainOutcome<T> {
    match result {
        Ok(value) => DomainOutcome {
            value: Some(value),
            failure: None,
        },
        Err(error) => DomainOutcome {
            value: None,
            failure: Some(DomainFailure {
                code: error.code().into(),
                message: error.to_string(),
            }),
        },
    }
}
#[derive(Clone, Serialize)]
pub struct StatusRecord {
    pub config: ServerConfig,
    pub admission: AdmissionMetrics,
    pub residency: Option<serde_json::Value>,
}
#[derive(Clone)]
pub struct Management {
    server: Server,
}
impl Management {
    pub fn new(server: &Server) -> Self {
        Self {
            server: server.clone(),
        }
    }
    pub fn version(&self) -> String {
        env!("CARGO_PKG_VERSION").into()
    }
    pub fn models(&self) -> Vec<ModelRecord> {
        self.server.models()
    }
    pub fn model(&self, name: &str) -> DomainOutcome<ModelRecord> {
        outcome(self.server.model(name))
    }
    pub fn alias_model(&self, name: &str, alias: String) -> DomainOutcome<ModelRecord> {
        outcome(self.server.alias_model(name, alias))
    }
    pub fn remove_model(&self, name: &str) -> DomainOutcome<bool> {
        outcome(self.server.remove_model(name).map(|()| true))
    }
    pub fn contexts(&self) -> Vec<ContextRecord> {
        self.server.contexts()
    }
    pub fn context(&self, id: String) -> DomainOutcome<ContextRecord> {
        outcome(self.server.context(&ContextId(id)))
    }
    pub fn create_context(&self) -> DomainOutcome<ContextRecord> {
        outcome(self.server.create_context())
    }
    pub fn import_context(&self, tokens: Vec<String>) -> DomainOutcome<ContextRecord> {
        outcome(self.server.import_context(tokens))
    }
    pub fn branch_context(&self, id: String) -> DomainOutcome<ContextRecord> {
        outcome(self.server.branch_context(&ContextId(id)))
    }
    pub fn delete_context(&self, id: String) -> DomainOutcome<bool> {
        outcome(self.server.delete_context(&ContextId(id)).map(|()| true))
    }
    pub fn cancel(&self, id: &str) {
        self.server.cancel(id);
    }
    pub fn status(&self) -> StatusRecord {
        StatusRecord {
            config: self.server.config(),
            admission: self.server.admission_metrics(),
            residency: self.server.engine.residency_status(),
        }
    }
    pub fn strategies(&self) -> CompactionStrategyCatalog {
        crate::strategy_catalog()
    }
    pub fn declare(
        &self,
        principal: &str,
        request: CreateCompactionDeclaration,
    ) -> DomainOutcome<CompactionDeclaration> {
        outcome(
            self.server
                .create_compaction_declaration_for(principal, request),
        )
    }
    pub fn revoke(&self, principal: &str, id: &str) -> DomainOutcome<bool> {
        outcome(
            self.server
                .revoke_compaction_declaration_for(principal, id)
                .map(|()| true),
        )
    }
    pub fn pull(&self, name: String, expected: Option<String>) -> DomainOutcome<PullSession> {
        outcome(self.begin_pull(name, expected))
    }
    fn begin_pull(&self, name: String, expected: Option<String>) -> Result<PullSession, Error> {
        let catalog = self
            .server
            .catalog()
            .ok_or_else(|| Error::State("model catalog is not configured".into()))?;
        let operation = uuid::Uuid::new_v4().to_string();
        catalog
            .begin_operation(&operation, &name, "pull")
            .map_err(crate::state_err)?;
        let (sender, receiver) = mpsc::channel(16);
        let server = self.server.clone();
        tokio::spawn(async move {
            if let Err(error) =
                pull_model(server, catalog, operation, name, expected, &sender).await
            {
                let _ = sender
                    .send(PullProgress::Failed {
                        code: error.code().into(),
                        message: error.to_string(),
                    })
                    .await;
            }
        });
        Ok(PullSession {
            receiver: Arc::new(Mutex::new(receiver)),
        })
    }
}

#[derive(Clone, Debug, Serialize)]
#[serde(tag = "stage", rename_all = "snake_case")]
pub enum PullProgress {
    Manifest,
    Transfer { completed: u64, total: Option<u64> },
    Verification { digest: String },
    Publication,
    Complete,
    Failed { code: String, message: String },
}
#[derive(Clone)]
pub struct PullSession {
    receiver: Arc<Mutex<mpsc::Receiver<PullProgress>>>,
}
impl PullSession {
    pub async fn next(&self) -> Option<PullProgress> {
        self.receiver.lock().await.recv().await
    }
    pub fn cancel(&self) {
        if let Ok(mut receiver) = self.receiver.try_lock() {
            receiver.close();
        }
    }
}
async fn pull_model(
    server: Server,
    catalog: crate::ModelCatalog,
    operation: String,
    name: String,
    expected: Option<String>,
    sender: &mpsc::Sender<PullProgress>,
) -> Result<(), Error> {
    let _ = sender.send(PullProgress::Manifest).await;
    let progress = sender.clone();
    let cache = server.model_directory();
    let fetched = tokio::task::spawn_blocking({
        let name = name.clone();
        move || {
            cusco_model_registry::fetch_hf_with_progress(
                &name,
                &cache,
                expected.as_deref(),
                |update| {
                    let _ = progress.blocking_send(PullProgress::Transfer {
                        completed: update.completed,
                        total: update.total,
                    });
                },
            )
        }
    })
    .await
    .map_err(crate::state_err)
    .and_then(|value| value.map_err(crate::state_err));
    let fetched = match fetched {
        Ok(value) => value,
        Err(error) => {
            catalog
                .finish_operation(&operation, "failed", Some(&error.to_string()))
                .map_err(crate::state_err)?;
            return Err(error);
        }
    };
    let _ = sender
        .send(PullProgress::Verification {
            digest: fetched.sha256.clone(),
        })
        .await;
    let _ = sender.send(PullProgress::Publication).await;
    let completed = tokio::task::spawn_blocking({
        let catalog = catalog.clone();
        let operation = operation.clone();
        move || {
            let metadata =
                cusco_model_registry::probe_gguf(&fetched.path).map_err(crate::state_err)?;
            server.register_model_with(
                |model| {
                    catalog
                        .publish_and_finish_operation(model, &operation)
                        .map_err(crate::state_err)
                },
                ModelRecord {
                    id: name,
                    revision: fetched.identity,
                    path: fetched.path,
                    sha256: fetched.sha256,
                    aliases: vec![],
                    family: metadata.architecture,
                    size_bytes: fetched.size,
                    block_count: metadata.block_count.unwrap_or(0),
                    epoch: 0,
                },
            )
        }
    })
    .await
    .map_err(crate::state_err)
    .and_then(|value| value);
    if let Err(error) = completed {
        catalog
            .finish_operation(&operation, "failed", Some(&error.to_string()))
            .map_err(crate::state_err)?;
        return Err(error);
    }
    let _ = sender.send(PullProgress::Complete).await;
    Ok(())
}
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn domain_outcomes_preserve_success_and_failure_contracts() {
        assert_eq!(outcome(Ok(7)).value, Some(7));
        let failure = outcome::<()>(Err(Error::BadRequest("bad request".into())));
        assert!(failure.value.is_none());
        let failure = failure.failure.unwrap();
        assert_eq!(failure.code, "invalid_request");
    }

    #[tokio::test]
    async fn cancelling_pull_session_closes_progress_delivery() {
        let (sender, receiver) = mpsc::channel(1);
        let session = PullSession {
            receiver: Arc::new(Mutex::new(receiver)),
        };
        session.cancel();
        assert!(session.next().await.is_none());
        drop(sender);
    }
    #[test]
    fn context_and_declaration_management_preserves_domain_results() {
        let root = std::env::temp_dir().join(format!("cusco-management-{}", uuid::Uuid::new_v4()));
        std::fs::create_dir_all(&root).unwrap();
        let server = Server::open(
            root.join("state.json"),
            Arc::new(crate::DeterministicEngine),
        )
        .unwrap();
        let management = Management::new(&server);

        let created = management.create_context().value.unwrap();
        assert_eq!(
            management.context(created.id.0.clone()).value,
            Some(created.clone())
        );
        assert_eq!(management.contexts().len(), 1);
        let branch = management
            .branch_context(created.id.0.clone())
            .value
            .unwrap();
        assert!(management.delete_context(branch.id.0).value.unwrap());
        let declaration = management
            .declare(
                "principal",
                CreateCompactionDeclaration {
                    context_id: created.id.0.clone(),
                    strategy_preferences: vec![crate::WINDOW_TAIL_STRATEGY_ID.into()],
                    target_tokens: Some(8),
                    expires_in_ms: 1_000,
                },
            )
            .value
            .unwrap();
        assert!(
            management
                .revoke("principal", &declaration.declaration_id)
                .value
                .unwrap()
        );
        assert_eq!(management.strategies().strategies.len(), 1);
        assert!(management.status().residency.is_none());
        assert!(!management.version().is_empty());
        management.cancel("no-active-request");
        assert!(management.delete_context(created.id.0).value.unwrap());
        assert!(management.model("missing").failure.is_some());
        assert!(matches!(
            management.pull("model".into(), None).failure,
            Some(failure) if failure.code == "state_error"
        ));
        drop(management);
        drop(server);
        std::fs::remove_dir_all(root).unwrap();
    }
    #[tokio::test]
    async fn failed_pull_publishes_terminal_progress() {
        let root = std::env::temp_dir().join(format!("cusco-pull-{}", uuid::Uuid::new_v4()));
        std::fs::create_dir_all(&root).unwrap();
        let server = Server::open(
            root.join("state.json"),
            Arc::new(crate::DeterministicEngine),
        )
        .unwrap();
        let catalog = crate::ModelCatalog::open(root.join("catalog.sqlite")).unwrap();
        server.attach_catalog(catalog.clone(), root.join("models"));
        let session = Management::new(&server)
            .pull("invalid".into(), None)
            .value
            .unwrap();

        assert!(matches!(session.next().await, Some(PullProgress::Manifest)));
        assert!(matches!(
            session.next().await,
            Some(PullProgress::Failed { code, message })
                if code == "state_error" && message.contains("invalid")
        ));

        drop(session);
        drop(server);
        std::fs::remove_dir_all(root).unwrap();
    }
}
