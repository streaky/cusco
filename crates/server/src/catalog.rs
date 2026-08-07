use crate::ModelRecord;
use parking_lot::Mutex;
use rusqlite::{Connection, OptionalExtension, params};
use rusqlite_migration::{M, Migrations};
use serde::{Deserialize, Serialize};
use std::{
    fs,
    path::{Path, PathBuf},
    sync::Arc,
};
use thiserror::Error;

#[derive(Debug, Error)]
pub enum CatalogError {
    #[error("catalog database: {0}")]
    Sql(#[from] rusqlite::Error),
    #[error("catalog migration: {0}")]
    Migration(#[from] rusqlite_migration::Error),
    #[error("catalog filesystem: {0}")]
    Io(#[from] std::io::Error),
    #[error("catalog data: {0}")]
    Data(String),
}

#[derive(Clone)]
pub struct ModelCatalog {
    connection: Arc<Mutex<Connection>>,
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct UserModelConfig {
    pub name: String,
    pub path: PathBuf,
    #[serde(default)]
    pub sha256: Option<String>,
    #[serde(default)]
    pub aliases: Vec<String>,
}
#[derive(Clone, Debug, Default, Deserialize, Serialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct UserModels {
    #[serde(default)]
    pub models: Vec<UserModelConfig>,
}

fn migrations() -> Migrations<'static> {
    Migrations::new(vec![
        M::up("CREATE TABLE models (id TEXT PRIMARY KEY NOT NULL, revision TEXT NOT NULL, path TEXT NOT NULL, sha256 TEXT NOT NULL, family TEXT NOT NULL, size_bytes INTEGER NOT NULL CHECK(size_bytes >= 0), epoch INTEGER NOT NULL CHECK(epoch > 0), aliases_json TEXT NOT NULL, installed_at INTEGER NOT NULL DEFAULT (unixepoch())); CREATE UNIQUE INDEX model_path_revision ON models(path, revision); CREATE TABLE lifecycle_operations (id TEXT PRIMARY KEY NOT NULL, model_id TEXT NOT NULL, kind TEXT NOT NULL, status TEXT NOT NULL CHECK(status IN ('running','complete','failed','cancelled')), detail TEXT, started_at INTEGER NOT NULL DEFAULT (unixepoch()), finished_at INTEGER);")
            .down("DROP TABLE lifecycle_operations; DROP TABLE models;"),
    ])
}

impl ModelCatalog {
    pub fn open(path: impl AsRef<Path>) -> Result<Self, CatalogError> {
        let path = path.as_ref();
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent)?;
        }
        let mut connection = Connection::open(path)?;
        connection.pragma_update(None, "journal_mode", "WAL")?;
        connection.pragma_update(None, "foreign_keys", "ON")?;
        migrations().to_latest(&mut connection)?;
        connection.execute("UPDATE lifecycle_operations SET status='failed', detail='server restarted before operation completed', finished_at=unixepoch() WHERE status='running'", [])?;
        Ok(Self {
            connection: Arc::new(Mutex::new(connection)),
        })
    }

    pub fn models(&self) -> Result<Vec<ModelRecord>, CatalogError> {
        let connection = self.connection.lock();
        let mut query = connection.prepare("SELECT id, revision, path, sha256, aliases_json, family, size_bytes, epoch FROM models ORDER BY id")?;
        let rows = query.query_map([], |row| {
            let aliases: String = row.get(4)?;
            let size: i64 = row.get(6)?;
            let epoch: i64 = row.get(7)?;
            Ok(ModelRecord {
                id: row.get(0)?,
                revision: row.get(1)?,
                path: PathBuf::from(row.get::<_, String>(2)?),
                sha256: row.get(3)?,
                aliases: serde_json::from_str(&aliases).map_err(|error| {
                    rusqlite::Error::FromSqlConversionFailure(
                        4,
                        rusqlite::types::Type::Text,
                        Box::new(error),
                    )
                })?,
                family: row.get(5)?,
                size_bytes: size.try_into().map_err(|error| {
                    rusqlite::Error::FromSqlConversionFailure(
                        6,
                        rusqlite::types::Type::Integer,
                        Box::new(error),
                    )
                })?,
                epoch: epoch.try_into().map_err(|error| {
                    rusqlite::Error::FromSqlConversionFailure(
                        7,
                        rusqlite::types::Type::Integer,
                        Box::new(error),
                    )
                })?,
            })
        })?;
        rows.collect::<Result<_, _>>().map_err(Into::into)
    }

    pub fn model(&self, id: &str) -> Result<Option<ModelRecord>, CatalogError> {
        Ok(self
            .models()?
            .into_iter()
            .find(|model| model.id == id || model.aliases.iter().any(|alias| alias == id)))
    }

    pub fn publish(&self, model: &ModelRecord) -> Result<(), CatalogError> {
        if model.epoch == 0 {
            return Err(CatalogError::Data("model epoch must be nonzero".into()));
        }
        let aliases = serde_json::to_string(&model.aliases)
            .map_err(|error| CatalogError::Data(error.to_string()))?;
        let size = i64::try_from(model.size_bytes)
            .map_err(|_| CatalogError::Data("model size exceeds SQLite integer".into()))?;
        let epoch = i64::try_from(model.epoch)
            .map_err(|_| CatalogError::Data("model epoch exceeds SQLite integer".into()))?;
        self.connection.lock().execute("INSERT INTO models(id,revision,path,sha256,family,size_bytes,epoch,aliases_json) VALUES(?1,?2,?3,?4,?5,?6,?7,?8) ON CONFLICT(id) DO UPDATE SET revision=excluded.revision,path=excluded.path,sha256=excluded.sha256,family=excluded.family,size_bytes=excluded.size_bytes,epoch=excluded.epoch,aliases_json=excluded.aliases_json", params![model.id, model.revision, model.path.to_string_lossy(), model.sha256, model.family, size, epoch, aliases])?;
        Ok(())
    }

    pub fn remove(&self, id: &str) -> Result<bool, CatalogError> {
        Ok(self
            .connection
            .lock()
            .execute("DELETE FROM models WHERE id=?1", [id])?
            == 1)
    }

    pub fn begin_operation(&self, id: &str, model: &str, kind: &str) -> Result<(), CatalogError> {
        self.connection.lock().execute(
            "INSERT INTO lifecycle_operations(id,model_id,kind,status) VALUES(?1,?2,?3,'running')",
            params![id, model, kind],
        )?;
        Ok(())
    }
    pub fn finish_operation(
        &self,
        id: &str,
        status: &str,
        detail: Option<&str>,
    ) -> Result<(), CatalogError> {
        let changed = self.connection.lock().execute("UPDATE lifecycle_operations SET status=?2, detail=?3, finished_at=unixepoch() WHERE id=?1 AND status='running'", params![id, status, detail])?;
        if changed != 1 {
            return Err(CatalogError::Data(format!(
                "operation {id} is absent or terminal"
            )));
        }
        Ok(())
    }
    pub fn operation_status(&self, id: &str) -> Result<Option<String>, CatalogError> {
        Ok(self
            .connection
            .lock()
            .query_row(
                "SELECT status FROM lifecycle_operations WHERE id=?1",
                [id],
                |row| row.get(0),
            )
            .optional()?)
    }
}

pub fn load_user_models(
    path: impl AsRef<Path>,
    root: impl AsRef<Path>,
) -> Result<UserModels, CatalogError> {
    let bytes = fs::read(path)?;
    let mut config: UserModels =
        serde_yaml::from_slice(&bytes).map_err(|error| CatalogError::Data(error.to_string()))?;
    let root = fs::canonicalize(root)?;
    let mut names = std::collections::HashSet::new();
    for model in &mut config.models {
        if !names.insert(model.name.clone()) {
            return Err(CatalogError::Data(format!(
                "duplicate local model name {}",
                model.name
            )));
        }
        let candidate = if model.path.is_absolute() {
            model.path.clone()
        } else {
            root.join(&model.path)
        };
        let canonical = fs::canonicalize(candidate)?;
        if !canonical.starts_with(&root) {
            return Err(CatalogError::Data(format!(
                "model {} escapes user-models root",
                model.name
            )));
        }
        model.path = canonical;
    }
    Ok(config)
}

#[cfg(test)]
mod tests {
    use super::*;
    use uuid::Uuid;
    fn database() -> (PathBuf, ModelCatalog) {
        let path = std::env::temp_dir().join(format!("cusco-catalog-{}.sqlite", Uuid::new_v4()));
        let catalog = ModelCatalog::open(&path).unwrap();
        (path, catalog)
    }
    #[test]
    fn persists_models_and_recovers_running_operations() {
        let (path, catalog) = database();
        let model = ModelRecord {
            id: "m".into(),
            revision: "r".into(),
            path: "/model.gguf".into(),
            sha256: "abc".into(),
            aliases: vec!["latest".into()],
            family: "gemma".into(),
            size_bytes: 7,
            epoch: 1,
        };
        catalog.publish(&model).unwrap();
        catalog.begin_operation("op", "m", "pull").unwrap();
        drop(catalog);
        let reopened = ModelCatalog::open(&path).unwrap();
        assert_eq!(reopened.model("latest").unwrap(), Some(model));
        assert_eq!(
            reopened.operation_status("op").unwrap().as_deref(),
            Some("failed")
        );
        let _ = fs::remove_file(path);
    }
    #[test]
    fn validates_catalog_mutations_and_user_model_roots() {
        let (path, catalog) = database();
        let mut model = ModelRecord {
            id: "m".into(),
            revision: "r".into(),
            path: "/m.gguf".into(),
            sha256: "abc".into(),
            aliases: vec!["latest".into()],
            family: "gemma3".into(),
            size_bytes: 7,
            epoch: 0,
        };
        assert!(matches!(
            catalog.publish(&model),
            Err(CatalogError::Data(_))
        ));
        model.epoch = 1;
        catalog.publish(&model).unwrap();
        assert_eq!(catalog.models().unwrap(), vec![model.clone()]);
        assert!(!catalog.remove("absent").unwrap());
        catalog.begin_operation("done", "m", "pull").unwrap();
        catalog.finish_operation("done", "complete", None).unwrap();
        assert_eq!(
            catalog.operation_status("done").unwrap().as_deref(),
            Some("complete")
        );
        assert!(matches!(
            catalog.finish_operation("done", "failed", None),
            Err(CatalogError::Data(_))
        ));
        assert!(catalog.remove("m").unwrap());

        let root = std::env::temp_dir().join(format!("cusco-user-models-{}", Uuid::new_v4()));
        fs::create_dir_all(&root).unwrap();
        fs::write(root.join("m.gguf"), b"model").unwrap();
        let config = root.join("models.yaml");
        fs::write(
            &config,
            "models:\n  - {name: local, path: m.gguf, aliases: [latest]}\n",
        )
        .unwrap();
        let loaded = load_user_models(&config, &root).unwrap();
        assert_eq!(
            loaded.models[0].path,
            fs::canonicalize(root.join("m.gguf")).unwrap()
        );
        fs::write(
            &config,
            "models:\n  - {name: same, path: m.gguf}\n  - {name: same, path: m.gguf}\n",
        )
        .unwrap();
        assert!(matches!(
            load_user_models(&config, &root),
            Err(CatalogError::Data(_))
        ));
        let outside = root
            .parent()
            .unwrap()
            .join(format!("outside-{}.gguf", Uuid::new_v4()));
        fs::write(&outside, b"outside").unwrap();
        fs::write(
            &config,
            format!(
                "models:\n  - {{name: escape, path: {}}}\n",
                outside.display()
            ),
        )
        .unwrap();
        assert!(matches!(
            load_user_models(&config, &root),
            Err(CatalogError::Data(_))
        ));
        let _ = fs::remove_file(outside);
        let _ = fs::remove_dir_all(root);
        let _ = fs::remove_file(path);
    }
}
