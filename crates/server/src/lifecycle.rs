use crate::{Error, InferenceEngine, ModelCatalog, ModelRecord, state_err};
use parking_lot::Mutex;
use sha2::{Digest, Sha256};
use std::{collections::HashMap, fs, io::Read, path::PathBuf, sync::Arc};

#[derive(Clone)]
pub struct ModelLifecycleService {
    state: Arc<Mutex<State>>,
    operation: Arc<Mutex<()>>,
    engine: Arc<dyn InferenceEngine>,
    catalog: Arc<Mutex<Option<ModelCatalog>>>,
    model_directory: Arc<Mutex<PathBuf>>,
}

struct State {
    models: HashMap<String, ModelRecord>,
    next_epoch: u64,
}

impl ModelLifecycleService {
    pub fn new(engine: Arc<dyn InferenceEngine>) -> Self {
        Self {
            state: Arc::new(Mutex::new(State {
                models: HashMap::new(),
                next_epoch: 1,
            })),
            operation: Arc::new(Mutex::new(())),
            engine,
            catalog: Arc::new(Mutex::new(None)),
            model_directory: Arc::new(Mutex::new(PathBuf::from("models"))),
        }
    }

    pub fn attach_catalog(&self, catalog: ModelCatalog, directory: PathBuf) {
        *self.catalog.lock() = Some(catalog);
        *self.model_directory.lock() = directory;
    }

    pub fn catalog(&self) -> Option<ModelCatalog> {
        self.catalog.lock().clone()
    }
    pub fn model_directory(&self) -> PathBuf {
        self.model_directory.lock().clone()
    }
    pub fn models(&self) -> Vec<ModelRecord> {
        self.state.lock().models.values().cloned().collect()
    }

    pub fn model(&self, id: &str) -> Result<ModelRecord, Error> {
        let state = self.state.lock();
        state
            .models
            .get(id)
            .or_else(|| {
                state
                    .models
                    .values()
                    .find(|m| m.aliases.iter().any(|a| a == id))
            })
            .cloned()
            .ok_or_else(|| Error::ModelNotFound(id.into()))
    }

    pub fn register<F>(&self, mut model: ModelRecord, publish: F) -> Result<ModelRecord, Error>
    where
        F: FnOnce(&ModelRecord) -> Result<(), Error>,
    {
        let _operation = self.operation.lock();
        model.aliases.sort();
        model.aliases.dedup();
        if model.family.is_empty() {
            model.family = crate::default_model_family();
        }
        if model.size_bytes == 0 {
            model.size_bytes = fs::metadata(&model.path).map_err(state_err)?.len();
        }
        let existing = self
            .state
            .lock()
            .models
            .get(&model.id)
            .cloned()
            .filter(|existing| {
                existing.revision == model.revision
                    && existing.path == model.path
                    && existing.sha256 == model.sha256
                    && existing.family == model.family
                    && existing.size_bytes == model.size_bytes
            });
        if let Some(existing) = existing {
            self.engine.prepare_model(&existing)?;
            return Ok(existing);
        }
        let (epoch, previous_epoch) = {
            let state = self.state.lock();
            (
                state.next_epoch,
                state.models.get(&model.id).map(|record| record.epoch),
            )
        };
        model.epoch = epoch;
        self.engine.prepare_model(&model)?;
        if let Err(error) = publish(&model) {
            self.engine.retire_model(&model.id, model.epoch);
            return Err(error);
        }
        {
            let mut state = self.state.lock();
            for existing in state.models.values_mut() {
                existing
                    .aliases
                    .retain(|alias| !model.aliases.contains(alias));
            }
            state.models.insert(model.id.clone(), model.clone());
            state.next_epoch = epoch
                .checked_add(1)
                .ok_or_else(|| Error::State("model epoch space exhausted".into()))?;
        }
        self.engine.commit_model(&model, previous_epoch);
        Ok(model)
    }

    pub fn register_from_catalog(&self, model: ModelRecord) -> Result<ModelRecord, Error> {
        let _operation = self.operation.lock();
        if model.epoch == 0 {
            return Err(Error::State("catalog model epoch must be nonzero".into()));
        }
        self.engine.prepare_model(&model)?;
        let previous_epoch = {
            let mut state = self.state.lock();
            let previous = state.models.get(&model.id).map(|record| record.epoch);
            for existing in state.models.values_mut() {
                existing
                    .aliases
                    .retain(|alias| !model.aliases.contains(alias));
            }
            state.next_epoch = state.next_epoch.max(model.epoch.saturating_add(1));
            state.models.insert(model.id.clone(), model.clone());
            previous
        };
        self.engine.commit_model(&model, previous_epoch);
        Ok(model)
    }

    pub fn alias(&self, id: &str, alias: String) -> Result<ModelRecord, Error> {
        let _operation = self.operation.lock();
        let mut candidate = self.state.lock().models.clone();
        if !candidate.contains_key(id) {
            return Err(Error::ModelNotFound(id.into()));
        }
        for model in candidate.values_mut() {
            model.aliases.retain(|existing| existing != &alias);
        }
        let model = candidate.get_mut(id).expect("model existence checked");
        model.aliases.push(alias);
        model.aliases.sort();
        model.aliases.dedup();
        let out = model.clone();
        if let Some(catalog) = self.catalog() {
            catalog.publish(&out).map_err(state_err)?;
        }
        self.state.lock().models = candidate;
        Ok(out)
    }

    pub fn remove(&self, id: &str) -> Result<(), Error> {
        let _operation = self.operation.lock();
        let model = self
            .state
            .lock()
            .models
            .get(id)
            .cloned()
            .ok_or_else(|| Error::ModelNotFound(id.into()))?;
        if let Some(catalog) = self.catalog() {
            catalog.remove(id).map_err(state_err)?;
        }
        self.state.lock().models.remove(id);
        self.engine.retire_model(&model.id, model.epoch);
        Ok(())
    }

    pub fn verify(&self, id: &str) -> Result<bool, Error> {
        let model = self.model(id)?;
        let mut file = fs::File::open(model.path).map_err(state_err)?;
        let mut digest = Sha256::new();
        let mut buffer = [0; 1024 * 1024];
        loop {
            let count = file.read(&mut buffer).map_err(state_err)?;
            if count == 0 {
                break;
            }
            digest.update(&buffer[..count]);
        }
        Ok(format!("{:x}", digest.finalize()) == model.sha256)
    }
}
