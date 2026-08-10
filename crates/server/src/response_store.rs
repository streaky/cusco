use crate::responses::ResponseResource;
use parking_lot::RwLock;
use std::{
    collections::HashMap,
    fs, io,
    path::{Path, PathBuf},
    sync::Arc,
    time::{SystemTime, UNIX_EPOCH},
};
use uuid::Uuid;

const RESPONSE_RETENTION_SECONDS: u64 = 30 * 24 * 60 * 60;

fn now() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs()
}

fn expired(resource: &ResponseResource, timestamp: u64) -> bool {
    timestamp.saturating_sub(resource.created_at) >= RESPONSE_RETENTION_SECONDS
}

#[derive(Debug, thiserror::Error)]
pub enum StoreError {
    #[error("response resource {0} was not found")]
    NotFound(String),
    #[error("response store I/O failed: {0}")]
    Io(#[from] io::Error),
    #[error("response resource is malformed: {0}")]
    Malformed(String),
    #[error("response lineage revision is stale: expected {expected}, current {current}")]
    StaleRevision { expected: u64, current: u64 },
    #[error("response resource {0} already has a committed successor")]
    Conflict(String),
}

pub trait ResponseResourceStore: Send + Sync {
    fn put(&self, resource: &ResponseResource) -> Result<(), StoreError>;
    fn get(&self, id: &str) -> Result<ResponseResource, StoreError>;
    fn put_successor(
        &self,
        base_id: &str,
        expected_revision: u64,
        resource: &ResponseResource,
    ) -> Result<(), StoreError>;
    fn delete(&self, id: &str) -> Result<(), StoreError>;
}

#[derive(Clone)]
pub struct FileResponseResourceStore {
    resources: PathBuf,
    temporary: PathBuf,
    recovered: Arc<RwLock<HashMap<String, ResponseResource>>>,
}

impl FileResponseResourceStore {
    pub fn open(state_root: impl AsRef<Path>) -> Result<Self, StoreError> {
        let root = state_root.as_ref().join("responses");
        let resources = root.join("resources");
        let temporary = root.join("tmp");
        fs::create_dir_all(&resources)?;
        fs::create_dir_all(&temporary)?;
        let mut recovered = HashMap::new();
        let mut entries = fs::read_dir(&resources)?.collect::<Result<Vec<_>, _>>()?;
        entries.sort_by_key(|entry| entry.file_name());
        for entry in entries {
            let path = entry.path();
            if path.extension().and_then(|v| v.to_str()) != Some("json") {
                continue;
            }
            let bytes = fs::read(&path)?;
            let resource: ResponseResource = serde_json::from_slice(&bytes)
                .map_err(|error| StoreError::Malformed(format!("{}: {error}", path.display())))?;
            let expected = format!("{}.json", resource.id);
            if entry.file_name() != expected.as_str() {
                return Err(StoreError::Malformed(format!(
                    "{} does not match resource id {}",
                    path.display(),
                    resource.id
                )));
            }
            if expired(&resource, now()) {
                fs::remove_file(path)?;
                continue;
            }
            recovered.insert(resource.id.clone(), resource);
        }
        Ok(Self {
            resources,
            temporary,
            recovered: Arc::new(RwLock::new(recovered)),
        })
    }

    fn resource_path(&self, id: &str) -> PathBuf {
        self.resources.join(format!("{id}.json"))
    }

    fn write_resource(&self, resource: &ResponseResource) -> Result<(), StoreError> {
        let bytes = serde_json::to_vec_pretty(resource)
            .map_err(|error| StoreError::Malformed(error.to_string()))?;
        let temp = self
            .temporary
            .join(format!("{}.{}.tmp", resource.id, Uuid::new_v4()));
        let mut file = fs::File::create(&temp)?;
        use std::io::Write as _;
        file.write_all(&bytes)?;
        file.sync_all()?;
        fs::rename(&temp, self.resource_path(&resource.id))?;
        fs::File::open(&self.resources)?.sync_all()?;
        Ok(())
    }

    fn prune_expired(&self) -> Result<(), StoreError> {
        let timestamp = now();
        let mut recovered = self.recovered.write();
        let expired_ids = recovered
            .values()
            .filter(|resource| expired(resource, timestamp))
            .map(|resource| resource.id.clone())
            .collect::<Vec<_>>();
        for id in &expired_ids {
            fs::remove_file(self.resource_path(id))?;
            recovered.remove(id);
        }
        if !expired_ids.is_empty() {
            fs::File::open(&self.resources)?.sync_all()?;
        }
        Ok(())
    }
}

impl ResponseResourceStore for FileResponseResourceStore {
    fn put(&self, resource: &ResponseResource) -> Result<(), StoreError> {
        self.prune_expired()?;
        self.write_resource(resource)?;
        self.recovered
            .write()
            .insert(resource.id.clone(), resource.clone());
        Ok(())
    }
    fn put_successor(
        &self,
        base_id: &str,
        expected_revision: u64,
        resource: &ResponseResource,
    ) -> Result<(), StoreError> {
        self.prune_expired()?;
        let mut recovered = self.recovered.write();
        let base = recovered
            .get(base_id)
            .ok_or_else(|| StoreError::NotFound(base_id.into()))?;
        if base.lineage_revision != expected_revision {
            return Err(StoreError::StaleRevision {
                expected: expected_revision,
                current: base.lineage_revision,
            });
        }
        if recovered
            .values()
            .any(|candidate| candidate.previous_response_id.as_deref() == Some(base_id))
        {
            return Err(StoreError::Conflict(base_id.into()));
        }
        self.write_resource(resource)?;
        recovered.insert(resource.id.clone(), resource.clone());
        Ok(())
    }
    fn get(&self, id: &str) -> Result<ResponseResource, StoreError> {
        self.recovered
            .read()
            .get(id)
            .cloned()
            .ok_or_else(|| StoreError::NotFound(id.into()))
    }
    fn delete(&self, id: &str) -> Result<(), StoreError> {
        let mut recovered = self.recovered.write();
        if !recovered.contains_key(id) {
            return Err(StoreError::NotFound(id.into()));
        }
        if recovered
            .values()
            .any(|candidate| candidate.previous_response_id.as_deref() == Some(id))
        {
            return Err(StoreError::Conflict(id.into()));
        }
        fs::remove_file(self.resource_path(id))?;
        fs::File::open(&self.resources)?.sync_all()?;
        recovered.remove(id);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::responses::{ResponseMetadata, ResponseResource};

    fn resource(id: &str) -> ResponseResource {
        ResponseResource {
            schema_version: 1,
            id: id.into(),
            owner: "owner".into(),
            model: "m".into(),
            created_at: now(),
            status: "completed".into(),
            store: true,
            previous_response_id: None,
            input: vec![],
            output: vec![],
            tools: vec![],
            tool_choice: serde_json::Value::String("auto".into()),
            finish_reason: None,
            usage: None,
            metadata: ResponseMetadata::default(),
            lineage_revision: 0,
            context_update: None,
        }
    }

    #[test]
    fn recovers_committed_resources_and_ignores_temporary_files() {
        let dir = std::env::temp_dir().join(format!("cusco-response-store-{}", Uuid::new_v4()));
        let store = FileResponseResourceStore::open(&dir).unwrap();
        store.put(&resource("resp_1")).unwrap();
        fs::write(store.temporary.join("interrupted.tmp"), b"incomplete").unwrap();
        let reopened = FileResponseResourceStore::open(&dir).unwrap();
        assert_eq!(reopened.get("resp_1").unwrap().id, "resp_1");
        fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn expired_resources_are_pruned_on_recovery() {
        let dir = std::env::temp_dir().join(format!("cusco-response-expiry-{}", Uuid::new_v4()));
        let store = FileResponseResourceStore::open(&dir).unwrap();
        let mut expired = resource("resp_expired");
        expired.created_at = now().saturating_sub(RESPONSE_RETENTION_SECONDS);
        store.put(&expired).unwrap();

        let reopened = FileResponseResourceStore::open(&dir).unwrap();
        assert!(matches!(
            reopened.get("resp_expired"),
            Err(StoreError::NotFound(_))
        ));
        assert!(!reopened.resource_path("resp_expired").exists());
        fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn malformed_committed_resource_fails_recovery() {
        let dir = std::env::temp_dir().join(format!("cusco-response-store-{}", Uuid::new_v4()));
        let resources = dir.join("responses/resources");
        fs::create_dir_all(&resources).unwrap();
        fs::write(resources.join("broken.json"), b"{").unwrap();
        assert!(matches!(
            FileResponseResourceStore::open(&dir),
            Err(StoreError::Malformed(_))
        ));
        fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn write_failure_does_not_publish_resource() {
        let dir = std::env::temp_dir().join(format!("cusco-response-store-{}", Uuid::new_v4()));
        let store = FileResponseResourceStore::open(&dir).unwrap();
        fs::remove_dir_all(&store.temporary).unwrap();
        assert!(matches!(
            store.put(&resource("resp_1")),
            Err(StoreError::Io(_))
        ));
        assert!(matches!(store.get("resp_1"), Err(StoreError::NotFound(_))));
        fs::remove_dir_all(dir).unwrap();
    }
}
