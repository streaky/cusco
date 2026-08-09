use crate::responses::ResponseResource;
use std::{collections::HashMap, fs, io, path::{Path, PathBuf}, sync::Arc};
use parking_lot::RwLock;
use uuid::Uuid;

#[derive(Debug, thiserror::Error)]
pub enum StoreError {
    #[error("response resource {0} was not found")]
    NotFound(String),
    #[error("response store I/O failed: {0}")]
    Io(#[from] io::Error),
    #[error("response resource is malformed: {0}")]
    Malformed(String),
}

pub trait ResponseResourceStore: Send + Sync {
    fn put(&self, resource: &ResponseResource) -> Result<(), StoreError>;
    fn get(&self, id: &str) -> Result<ResponseResource, StoreError>;
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
            if path.extension().and_then(|v| v.to_str()) != Some("json") { continue; }
            let bytes = fs::read(&path)?;
            let resource: ResponseResource = serde_json::from_slice(&bytes).map_err(|error| StoreError::Malformed(format!("{}: {error}", path.display())))?;
            let expected = format!("{}.json", resource.id);
            if entry.file_name() != expected.as_str() { return Err(StoreError::Malformed(format!("{} does not match resource id {}", path.display(), resource.id))); }
            recovered.insert(resource.id.clone(), resource);
        }
        Ok(Self { resources, temporary, recovered: Arc::new(RwLock::new(recovered)) })
    }

    fn resource_path(&self, id: &str) -> PathBuf { self.resources.join(format!("{id}.json")) }
}

impl ResponseResourceStore for FileResponseResourceStore {
    fn put(&self, resource: &ResponseResource) -> Result<(), StoreError> {
        let bytes = serde_json::to_vec_pretty(resource).map_err(|error| StoreError::Malformed(error.to_string()))?;
        let temp = self.temporary.join(format!("{}.{}.tmp", resource.id, Uuid::new_v4()));
        let mut file = fs::File::create(&temp)?;
        use std::io::Write as _;
        file.write_all(&bytes)?;
        file.sync_all()?;
        fs::rename(&temp, self.resource_path(&resource.id))?;
        fs::File::open(&self.resources)?.sync_all()?;
        self.recovered.write().insert(resource.id.clone(), resource.clone());
        Ok(())
    }
    fn get(&self, id: &str) -> Result<ResponseResource, StoreError> { self.recovered.read().get(id).cloned().ok_or_else(|| StoreError::NotFound(id.into())) }
    fn delete(&self, id: &str) -> Result<(), StoreError> {
        if self.recovered.write().remove(id).is_none() { return Err(StoreError::NotFound(id.into())); }
        fs::remove_file(self.resource_path(id))?;
        fs::File::open(&self.resources)?.sync_all()?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::responses::{ResponseMetadata, ResponseResource};

    fn resource(id: &str) -> ResponseResource { ResponseResource { id:id.into(), model:"m".into(), created_at:1, status:"completed".into(), output:vec![], finish_reason:None, usage:None, metadata:ResponseMetadata::default() } }

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
    fn malformed_committed_resource_fails_recovery() {
        let dir = std::env::temp_dir().join(format!("cusco-response-store-{}", Uuid::new_v4()));
        let resources = dir.join("responses/resources");
        fs::create_dir_all(&resources).unwrap();
        fs::write(resources.join("broken.json"), b"{").unwrap();
        assert!(matches!(FileResponseResourceStore::open(&dir), Err(StoreError::Malformed(_))));
        fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn write_failure_does_not_publish_resource() {
        let dir = std::env::temp_dir().join(format!("cusco-response-store-{}", Uuid::new_v4()));
        let store = FileResponseResourceStore::open(&dir).unwrap();
        fs::remove_dir_all(&store.temporary).unwrap();
        assert!(matches!(store.put(&resource("resp_1")), Err(StoreError::Io(_))));
        assert!(matches!(store.get("resp_1"), Err(StoreError::NotFound(_))));
        fs::remove_dir_all(dir).unwrap();
    }
}
