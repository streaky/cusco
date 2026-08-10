use crate::{
    EngineRequest, Error, ExecutionSession, InferenceEngine, MappedEngine, ModelCatalog,
    ModelRecord, SessionStep,
    catalog::{
        MEASURED_EXECUTION_PROFILE_SCHEMA_MAJOR, MeasuredExecutionProfile, MeasurementSource,
    },
};
use cusco_context_store::ModelEpoch;
use cusco_executor::{ABI_VERSION, OperatingPoint};
use parking_lot::Mutex;
use serde::Serialize;
use sha2::{Digest, Sha256};
use std::{
    collections::HashMap,
    fs,
    path::{Path, PathBuf},
    sync::{
        Arc,
        atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering},
    },
};

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
pub struct ResidencyConfig {
    pub device_bytes: u64,
    pub host_bytes: u64,
    pub storage_bytes: u64,
    pub context_reserve_bytes: u64,
    pub n_ctx: u32,
    pub gpu_layers: i32,
    pub require_competent: bool,
}

impl ResidencyConfig {
    pub fn validate(self) -> Result<Self, Error> {
        if self.device_bytes == 0
            || self.host_bytes == 0
            || self.storage_bytes == 0
            || self.n_ctx == 0
        {
            return Err(Error::State(
                "residency budgets and context size must be nonzero".into(),
            ));
        }
        Ok(self)
    }
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, Serialize)]
pub struct ResidencyMetrics {
    pub loads: u64,
    pub load_failures: u64,
    pub reuses: u64,
    pub unloads: u64,
    pub reloads: u64,
    pub context_spills: u64,
    pub context_reloads: u64,
    pub device_bytes: u64,
    pub host_bytes: u64,
    pub storage_bytes: u64,
    pub active_slots: usize,
    pub sessions: usize,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct ResidentModelStatus {
    pub id: String,
    pub revision: String,
    pub epoch: u64,
    pub active_slots: usize,
    pub sessions: usize,
    pub retiring: bool,
    pub context_resident: bool,
    pub last_used: u64,
    pub operating_point: OperatingPoint,
}

trait ModelLoader: Send + Sync {
    fn load(
        &self,
        model: &ModelRecord,
        config: ResidencyConfig,
    ) -> Result<(Arc<dyn InferenceEngine>, OperatingPoint), Error>;
}

struct NativeLoader {
    spill_dir: PathBuf,
}

impl ModelLoader for NativeLoader {
    fn load(
        &self,
        model: &ModelRecord,
        config: ResidencyConfig,
    ) -> Result<(Arc<dyn InferenceEngine>, OperatingPoint), Error> {
        let spill_dir = self.spill_dir.join(format!("{}-{}", model.id, model.epoch));
        let engine = MappedEngine::open_at_epoch_with_spill(
            &model.path,
            config.n_ctx,
            config.gpu_layers,
            usize::try_from(config.device_bytes)
                .map_err(|_| Error::State("device budget exceeds address space".into()))?,
            usize::try_from(config.host_bytes)
                .map_err(|_| Error::State("host budget exceeds address space".into()))?,
            ModelEpoch(model.epoch),
            Some(spill_dir),
            usize::try_from(config.context_reserve_bytes)
                .map_err(|_| Error::State("context reserve exceeds address space".into()))?,
        )?;
        let mut point = engine.operating_point();
        point.model_bytes = model.size_bytes;
        Ok((engine, point))
    }
}

struct ResidentModel {
    record: ModelRecord,
    engine: Arc<dyn InferenceEngine>,
    point: OperatingPoint,
    active: AtomicUsize,
    sessions: AtomicUsize,
    retiring: AtomicBool,
    context_resident: AtomicBool,
    last_used: AtomicU64,
}

impl ResidentModel {
    fn status(&self) -> ResidentModelStatus {
        ResidentModelStatus {
            id: self.record.id.clone(),
            revision: self.record.revision.clone(),
            epoch: self.record.epoch,
            active_slots: self.active.load(Ordering::Acquire),
            sessions: self.sessions.load(Ordering::Acquire),
            retiring: self.retiring.load(Ordering::Acquire),
            context_resident: self.context_resident.load(Ordering::Acquire),
            last_used: self.last_used.load(Ordering::Acquire),
            operating_point: self.point,
        }
    }
}

#[derive(Default)]
struct ResidencyState {
    entries: HashMap<(String, u64), Arc<ResidentModel>>,
    clock: u64,
    metrics: ResidencyMetrics,
}

pub struct ResidentEngine {
    config: ResidencyConfig,
    state: Arc<Mutex<ResidencyState>>,
    loader: Arc<dyn ModelLoader>,
    catalog: Mutex<Option<ModelCatalog>>,
}

impl ResidentEngine {
    pub fn open(config: ResidencyConfig) -> Result<Arc<Self>, Error> {
        Self::open_with_spill(config, std::env::temp_dir().join("cusco-spill"))
    }

    pub fn open_with_spill(
        config: ResidencyConfig,
        spill_dir: impl AsRef<Path>,
    ) -> Result<Arc<Self>, Error> {
        let spill_dir = spill_dir.as_ref().to_owned();
        fs::create_dir_all(&spill_dir).map_err(super::state_err)?;
        Ok(Arc::new(Self {
            config: config.validate()?,
            state: Arc::new(Mutex::new(ResidencyState::default())),
            loader: Arc::new(NativeLoader { spill_dir }),
            catalog: Mutex::new(None),
        }))
    }

    #[cfg(test)]
    fn with_loader(config: ResidencyConfig, loader: Arc<dyn ModelLoader>) -> Arc<Self> {
        Arc::new(Self {
            config: config.validate().unwrap(),
            state: Arc::new(Mutex::new(ResidencyState::default())),
            loader,
            catalog: Mutex::new(None),
        })
    }

    pub fn attach_catalog(&self, catalog: ModelCatalog) {
        *self.catalog.lock() = Some(catalog);
    }

    fn profile_key(&self, model: &ModelRecord) -> (String, String, String, String) {
        let model_identity = if model.sha256.is_empty() {
            format!("{}@{}", model.id, model.revision)
        } else {
            model.sha256.clone()
        };
        let config = format!(
            "abi={ABI_VERSION};llama={};n_ctx={};n_batch={};gpu_layers={};mapped=true",
            include_str!("../../../llama.cpp-version.txt").trim(),
            self.config.n_ctx,
            self.config.n_ctx,
            self.config.gpu_layers,
        );
        let config_hash = hex::encode(Sha256::digest(config.as_bytes()));
        let gpu_id = if self.config.gpu_layers > 0 {
            format!(
                "configured-gpu:{}",
                std::env::var("CUSCO_GPU_DEVICE_ID").unwrap_or_else(|_| "unspecified".into())
            )
        } else {
            "cpu".into()
        };
        let provenance = format!(
            "executor-abi={ABI_VERSION};llama={}",
            include_str!("../../../llama.cpp-version.txt").trim()
        );
        (model_identity, config_hash, gpu_id, provenance)
    }

    fn cached_profile(&self, model: &ModelRecord) -> Result<Option<OperatingPoint>, Error> {
        let Some(catalog) = self.catalog.lock().clone() else {
            return Ok(None);
        };
        let (model_identity, config_hash, gpu_id, provenance) = self.profile_key(model);
        let record = catalog
            .measured_execution_profile(&model_identity, &config_hash, &gpu_id, &provenance)
            .map_err(|error| Error::State(error.to_string()))?;
        Ok(record.map(|record| OperatingPoint {
            model_bytes: record.profile.model_bytes,
            context_bytes: record.profile.context_state_bytes,
            device_bytes: record.profile.device_model_bytes
                + record.profile.device_execution_reserve_bytes
                + record.profile.allocator_headroom_bytes,
            host_bytes: record.profile.host_model_bytes + record.profile.host_staging_bytes,
            gpu_layers: self.config.gpu_layers,
            model_layers: record.profile.model_layers,
            competent: record.profile.competent,
        }))
    }

    fn publish_profile(&self, model: &ModelRecord, point: OperatingPoint) -> Result<(), Error> {
        let Some(catalog) = self.catalog.lock().clone() else {
            return Ok(());
        };
        let (model_identity, config_hash, gpu_id, provenance) = self.profile_key(model);
        let profile = MeasuredExecutionProfile {
            schema_major: MEASURED_EXECUTION_PROFILE_SCHEMA_MAJOR,
            schema_minor: 0,
            model_bytes: point.model_bytes,
            device_model_bytes: point.device_bytes,
            host_model_bytes: point.host_bytes,
            context_state_bytes: point.context_bytes,
            device_execution_reserve_bytes: 0,
            host_staging_bytes: 0,
            model_layers: point.model_layers,
            competent: point.competent,
            allocator_headroom_bytes: point.device_bytes / 20,
            measurement_source: if self.config.gpu_layers > 0 {
                MeasurementSource::BackendAllocator
            } else {
                MeasurementSource::ConservativeFallback
            },
            provenance,
        };
        catalog
            .publish_measured_execution_profile(&model_identity, &config_hash, &gpu_id, &profile)
            .map_err(|error| Error::State(error.to_string()))
    }

    fn estimate(&self, model: &ModelRecord) -> Result<OperatingPoint, Error> {
        if let Some(profile) = self.cached_profile(model)? {
            return Ok(profile);
        }
        let model_bytes = if model.size_bytes == 0 {
            fs::metadata(&model.path).map_err(super::state_err)?.len()
        } else {
            model.size_bytes
        };
        if model_bytes > self.config.storage_bytes {
            return Err(Error::State(
                "model exceeds the configured storage residency budget".into(),
            ));
        }
        let (device_bytes, host_bytes) = if self.config.gpu_layers > 0 {
            (
                model_bytes.saturating_add(self.config.context_reserve_bytes),
                0,
            )
        } else {
            (
                0,
                model_bytes.saturating_add(self.config.context_reserve_bytes),
            )
        };
        Ok(OperatingPoint {
            model_bytes,
            context_bytes: self.config.context_reserve_bytes,
            device_bytes,
            host_bytes,
            gpu_layers: self.config.gpu_layers,
            model_layers: 0,
            competent: !self.config.require_competent || self.config.gpu_layers > 0,
        })
    }
    fn fits_with(
        config: ResidencyConfig,
        point: OperatingPoint,
        retained: impl Iterator<Item = OperatingPoint>,
    ) -> bool {
        let (device, host, storage) = retained.fold(
            (point.device_bytes, point.host_bytes, point.model_bytes),
            |(device, host, storage), item| {
                (
                    device.saturating_add(item.device_bytes),
                    host.saturating_add(item.host_bytes),
                    storage.saturating_add(item.model_bytes),
                )
            },
        );
        device <= config.device_bytes
            && host <= config.host_bytes
            && storage <= config.storage_bytes
    }

    fn fits(&self, point: OperatingPoint, retained: impl Iterator<Item = OperatingPoint>) -> bool {
        Self::fits_with(self.config, point, retained)
    }

    fn effective_point(entry: &ResidentModel) -> OperatingPoint {
        let mut point = entry.point;
        if !entry.context_resident.load(Ordering::Acquire) {
            if point.device_bytes == 0 {
                point.host_bytes = point.host_bytes.saturating_sub(point.context_bytes);
            } else {
                point.device_bytes = point.device_bytes.saturating_sub(point.context_bytes);
            }
            point.model_bytes = point.model_bytes.saturating_add(point.context_bytes);
        }
        point
    }

    fn load(
        &self,
        model: &ModelRecord,
        control: Option<&crate::RequestControl>,
    ) -> Result<Arc<ResidentModel>, Error> {
        if let Some(control) = control {
            control.check()?;
        }
        let key = (model.id.clone(), model.epoch);
        {
            let mut state = self.state.lock();
            if let Some(entry) = state.entries.get(&key).cloned() {
                state.clock = state.clock.wrapping_add(1);
                entry.last_used.store(state.clock, Ordering::Release);
                state.metrics.reuses += 1;
                return Ok(entry);
            }
        }

        let estimate = self.estimate(model)?;
        let candidate_victims = {
            let mut state = self.state.lock();
            let mut idle = state
                .entries
                .values()
                .filter(|entry| {
                    entry.active.load(Ordering::Acquire) == 0
                        && entry.sessions.load(Ordering::Acquire) == 0
                        && entry.record.id != model.id
                })
                .cloned()
                .collect::<Vec<_>>();
            idle.sort_by_key(|entry| (entry.last_used.load(Ordering::Acquire), entry.record.epoch));
            while !self.fits(
                estimate,
                state
                    .entries
                    .values()
                    .map(|entry| Self::effective_point(entry)),
            ) {
                let Some(candidate) = idle
                    .iter()
                    .find(|entry| entry.context_resident.load(Ordering::Acquire))
                else {
                    break;
                };
                let count = candidate.engine.demote_inactive()?;
                if count == 0 {
                    break;
                }
                candidate.context_resident.store(false, Ordering::Release);
                state.metrics.context_spills =
                    state.metrics.context_spills.saturating_add(count as u64);
            }
            let mut retained = state.entries.values().cloned().collect::<Vec<_>>();
            let mut victims = Vec::new();
            while !self.fits(
                estimate,
                retained.iter().map(|entry| Self::effective_point(entry)),
            ) {
                let victim = idle
                    .get(victims.len())
                    .cloned()
                    .ok_or_else(|| Error::State("residency capacity is exhausted".into()))?;
                retained.retain(|entry| !Arc::ptr_eq(entry, &victim));
                victims.push(victim);
            }
            victims
        };

        let loaded = self.loader.load(model, self.config);
        let (engine, point) = match loaded {
            Ok(loaded) => loaded,
            Err(error) => {
                self.state.lock().metrics.load_failures += 1;
                return Err(error);
            }
        };
        // A backend-wide free-memory delta can include unrelated devices and
        // concurrent allocations. Never let it weaken the conservative
        // admission estimate derived before loading.
        let measured_point = OperatingPoint {
            model_bytes: point.model_bytes.max(estimate.model_bytes),
            context_bytes: point.context_bytes.max(estimate.context_bytes),
            device_bytes: point.device_bytes.max(estimate.device_bytes),
            host_bytes: point.host_bytes.max(estimate.host_bytes),
            ..point
        };
        let point = OperatingPoint {
            device_bytes: measured_point
                .device_bytes
                .saturating_add(measured_point.device_bytes / 20),
            ..measured_point
        };
        if let Some(control) = control {
            if let Err(error) = control.check() {
                self.state.lock().metrics.load_failures += 1;
                return Err(error);
            }
        }
        if self.config.require_competent && !point.competent {
            self.state.lock().metrics.load_failures += 1;
            return Err(Error::State(
                "executor operating point is below the competent floor".into(),
            ));
        }

        let mut state = self.state.lock();
        if let Some(entry) = state.entries.get(&key).cloned() {
            state.metrics.reuses += 1;
            return Ok(entry);
        }
        let victim_keys = candidate_victims
            .iter()
            .map(|entry| (entry.record.id.clone(), entry.record.epoch))
            .collect::<Vec<_>>();
        let retained = state
            .entries
            .iter()
            .filter(|(key, entry)| {
                !victim_keys.contains(key) || entry.active.load(Ordering::Acquire) != 0
            })
            .map(|(_, entry)| Self::effective_point(entry));
        if !self.fits(point, retained) {
            state.metrics.load_failures += 1;
            return Err(Error::State(
                "measured operating point exceeds residency capacity".into(),
            ));
        }
        if let Err(error) = self.publish_profile(model, measured_point) {
            state.metrics.load_failures += 1;
            return Err(error);
        }
        for victim in candidate_victims {
            let victim_key = (victim.record.id.clone(), victim.record.epoch);
            if victim.active.load(Ordering::Acquire) == 0
                && state.entries.remove(&victim_key).is_some()
            {
                state.metrics.unloads += 1;
            }
        }
        state.clock = state.clock.wrapping_add(1);
        let last_used = state.clock;
        let entry = Arc::new(ResidentModel {
            record: model.clone(),
            engine,
            point,
            active: AtomicUsize::new(0),
            sessions: AtomicUsize::new(0),
            retiring: AtomicBool::new(false),
            context_resident: AtomicBool::new(true),
            last_used: AtomicU64::new(last_used),
        });
        state.entries.insert(key, entry.clone());
        state.metrics.loads += 1;
        Self::refresh_metrics(&mut state);
        Ok(entry)
    }

    fn refresh_metrics(state: &mut ResidencyState) {
        state.metrics.device_bytes = state
            .entries
            .values()
            .map(|entry| Self::effective_point(entry).device_bytes)
            .sum();
        state.metrics.host_bytes = state
            .entries
            .values()
            .map(|entry| Self::effective_point(entry).host_bytes)
            .sum();
        state.metrics.storage_bytes = state
            .entries
            .values()
            .map(|entry| Self::effective_point(entry).model_bytes)
            .sum();
        state.metrics.active_slots = state
            .entries
            .values()
            .map(|entry| entry.active.load(Ordering::Acquire))
            .sum();
        state.metrics.sessions = state
            .entries
            .values()
            .map(|entry| entry.sessions.load(Ordering::Acquire))
            .sum();
    }

    fn retire_other_epochs(&self, model: &ModelRecord) {
        let mut state = self.state.lock();
        let keys = state
            .entries
            .iter()
            .filter(|((id, epoch), _)| id == &model.id && *epoch != model.epoch)
            .map(|(key, entry)| (key.clone(), entry.clone()))
            .collect::<Vec<_>>();
        for (key, entry) in keys {
            entry.retiring.store(true, Ordering::Release);
            if entry.active.load(Ordering::Acquire) == 0
                && entry.sessions.load(Ordering::Acquire) == 0
                && state.entries.remove(&key).is_some()
            {
                state.metrics.unloads += 1;
                state.metrics.reloads += 1;
            }
        }
        Self::refresh_metrics(&mut state);
    }
    fn acquire_entry(
        config: ResidencyConfig,
        shared: &Arc<Mutex<ResidencyState>>,
        target: &Arc<ResidentModel>,
    ) -> Result<bool, Error> {
        let mut state = shared.lock();
        let key = (target.record.id.clone(), target.record.epoch);
        let Some(current) = state.entries.get(&key) else {
            return Ok(false);
        };
        if !Arc::ptr_eq(current, target) {
            return Ok(false);
        }
        if !target.context_resident.load(Ordering::Acquire) {
            let mut idle = state
                .entries
                .values()
                .filter(|entry| {
                    !Arc::ptr_eq(entry, target)
                        && entry.active.load(Ordering::Acquire) == 0
                        && entry.sessions.load(Ordering::Acquire) == 0
                        && entry.context_resident.load(Ordering::Acquire)
                })
                .cloned()
                .collect::<Vec<_>>();
            idle.sort_by_key(|entry| (entry.last_used.load(Ordering::Acquire), entry.record.epoch));
            loop {
                let retained = state
                    .entries
                    .values()
                    .filter(|entry| !Arc::ptr_eq(entry, target))
                    .map(|entry| Self::effective_point(entry));
                if Self::fits_with(config, target.point, retained) {
                    target.context_resident.store(true, Ordering::Release);
                    state.metrics.context_reloads += 1;
                    break;
                }
                let Some(candidate) = idle
                    .iter()
                    .find(|entry| entry.context_resident.load(Ordering::Acquire))
                else {
                    return Err(Error::State("residency capacity is exhausted".into()));
                };
                let count = candidate.engine.demote_inactive()?;
                if count == 0 {
                    return Err(Error::State("residency capacity is exhausted".into()));
                }
                candidate.context_resident.store(false, Ordering::Release);
                state.metrics.context_spills =
                    state.metrics.context_spills.saturating_add(count as u64);
            }
        }
        target.active.fetch_add(1, Ordering::AcqRel);
        state.clock = state.clock.wrapping_add(1);
        target.last_used.store(state.clock, Ordering::Release);
        Self::refresh_metrics(&mut state);
        Ok(true)
    }

    fn acquire(&self, target: &Arc<ResidentModel>) -> Result<bool, Error> {
        Self::acquire_entry(self.config, &self.state, target)
    }

    fn release_entry(shared: &Arc<Mutex<ResidencyState>>, entry: &Arc<ResidentModel>) {
        entry.active.fetch_sub(1, Ordering::AcqRel);
        let mut state = shared.lock();
        if entry.retiring.load(Ordering::Acquire)
            && entry.active.load(Ordering::Acquire) == 0
            && entry.sessions.load(Ordering::Acquire) == 0
        {
            let key = (entry.record.id.clone(), entry.record.epoch);
            if state.entries.remove(&key).is_some() {
                state.metrics.unloads += 1;
            }
        }
        Self::refresh_metrics(&mut state);
    }

    fn release(&self, entry: &Arc<ResidentModel>) {
        Self::release_entry(&self.state, entry);
    }

    pub fn metrics(&self) -> ResidencyMetrics {
        let mut state = self.state.lock();
        Self::refresh_metrics(&mut state);
        state.metrics
    }

    pub fn status(&self) -> Vec<ResidentModelStatus> {
        let mut statuses = self
            .state
            .lock()
            .entries
            .values()
            .map(|entry| entry.status())
            .collect::<Vec<_>>();
        statuses.sort_by_key(|status| (status.id.clone(), status.epoch));
        statuses
    }
}

struct ResidentSession {
    config: ResidencyConfig,
    entry: Arc<ResidentModel>,
    state: Arc<Mutex<ResidencyState>>,
    inner: Box<dyn ExecutionSession>,
}

struct ResidentQuantumLease {
    entry: Arc<ResidentModel>,
    state: Arc<Mutex<ResidencyState>>,
}

impl Drop for ResidentQuantumLease {
    fn drop(&mut self) {
        ResidentEngine::release_entry(&self.state, &self.entry);
    }
}

impl ResidentSession {
    fn acquire(&self) -> Result<ResidentQuantumLease, Error> {
        if !ResidentEngine::acquire_entry(self.config, &self.state, &self.entry)? {
            return Err(Error::State(
                "resident model disappeared while its session was suspended".into(),
            ));
        }
        Ok(ResidentQuantumLease {
            entry: self.entry.clone(),
            state: self.state.clone(),
        })
    }
}

impl ExecutionSession for ResidentSession {
    fn step(&mut self) -> Result<SessionStep, Error> {
        let lease = self.acquire()?;
        let result = self.inner.step();
        drop(lease);
        result
    }

    fn finish(&mut self) -> Result<crate::EngineOutput, Error> {
        let lease = self.acquire()?;
        let result = self.inner.finish();
        drop(lease);
        result
    }
}

impl Drop for ResidentSession {
    fn drop(&mut self) {
        let previous = self.entry.sessions.fetch_sub(1, Ordering::AcqRel);
        debug_assert!(previous > 0, "resident session count underflow");
        let mut state = self.state.lock();
        if self.entry.retiring.load(Ordering::Acquire)
            && self.entry.active.load(Ordering::Acquire) == 0
            && self.entry.sessions.load(Ordering::Acquire) == 0
        {
            let key = (self.entry.record.id.clone(), self.entry.record.epoch);
            if state.entries.remove(&key).is_some() {
                state.metrics.unloads += 1;
            }
        }
        ResidentEngine::refresh_metrics(&mut state);
    }
}

impl InferenceEngine for ResidentEngine {
    fn prepare_model(&self, model: &ModelRecord) -> Result<(), Error> {
        self.load(model, None)?;
        Ok(())
    }

    fn commit_model(&self, model: &ModelRecord, _replaced_epoch: Option<u64>) {
        self.retire_other_epochs(model);
    }

    fn retire_model(&self, id: &str, epoch: u64) {
        let mut state = self.state.lock();
        let key = (id.to_owned(), epoch);
        if let Some(entry) = state.entries.get(&key).cloned() {
            entry.retiring.store(true, Ordering::Release);
            if entry.active.load(Ordering::Acquire) == 0
                && entry.sessions.load(Ordering::Acquire) == 0
            {
                state.entries.remove(&key);
                state.metrics.unloads += 1;
            }
        }
        Self::refresh_metrics(&mut state);
    }

    fn residency_status(&self) -> Option<serde_json::Value> {
        Some(serde_json::json!({
            "config": self.config,
            "metrics": self.metrics(),
            "models": self.status(),
        }))
    }

    fn start_session(&self, request: EngineRequest) -> Result<Box<dyn ExecutionSession>, Error> {
        let entry = loop {
            let entry = self.load(&request.model, Some(&request.control))?;
            if self.acquire(&entry)? {
                break entry;
            }
        };
        let started = entry.engine.start_session(request);
        match started {
            Ok(inner) => {
                entry.sessions.fetch_add(1, Ordering::AcqRel);
                self.release(&entry);
                Ok(Box::new(ResidentSession {
                    config: self.config,
                    entry,
                    state: self.state.clone(),
                    inner,
                }))
            }
            Err(error) => {
                self.release(&entry);
                Err(error)
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{DeterministicEngine, FrontierControl, RequestControl, SchedulingMetadata};
    use std::{path::PathBuf, sync::Barrier, thread};

    struct FixtureLoader {
        point: OperatingPoint,
        failures: Mutex<Vec<String>>,
        blocker: Option<(Arc<Barrier>, Arc<Barrier>)>,
    }

    struct BlockingEngine {
        entered: Arc<Barrier>,
        release: Arc<Barrier>,
    }

    struct BlockingSession {
        entered: Arc<Barrier>,
        release: Arc<Barrier>,
        inner: Box<dyn ExecutionSession>,
        blocked: bool,
    }

    impl ExecutionSession for BlockingSession {
        fn step(&mut self) -> Result<SessionStep, Error> {
            if !self.blocked {
                self.entered.wait();
                self.release.wait();
                self.blocked = true;
            }
            self.inner.step()
        }

        fn finish(&mut self) -> Result<crate::EngineOutput, Error> {
            self.inner.finish()
        }
    }

    impl InferenceEngine for BlockingEngine {
        fn start_session(
            &self,
            request: EngineRequest,
        ) -> Result<Box<dyn ExecutionSession>, Error> {
            Ok(Box::new(BlockingSession {
                entered: self.entered.clone(),
                release: self.release.clone(),
                inner: DeterministicEngine.start_session(request)?,
                blocked: false,
            }))
        }
    }

    struct DemotableEngine;

    impl InferenceEngine for DemotableEngine {
        fn start_session(
            &self,
            request: EngineRequest,
        ) -> Result<Box<dyn ExecutionSession>, Error> {
            DeterministicEngine.start_session(request)
        }

        fn demote_inactive(&self) -> Result<usize, Error> {
            Ok(1)
        }
    }

    fn test_request(
        model: &ModelRecord,
        prompt: impl Into<String>,
        max_tokens: usize,
        control: Arc<RequestControl>,
    ) -> EngineRequest {
        EngineRequest {
            model: model.clone(),
            prompt: prompt.into(),
            max_tokens,
            prior_tokens: vec![],
            sampling: Default::default(),
            grammar: None,
            control,
            scheduling: SchedulingMetadata::default(),
            prefill_chunk_tokens: 32,
        }
    }

    struct DemotableLoader {
        point: OperatingPoint,
    }

    impl ModelLoader for DemotableLoader {
        fn load(
            &self,
            _model: &ModelRecord,
            _config: ResidencyConfig,
        ) -> Result<(Arc<dyn InferenceEngine>, OperatingPoint), Error> {
            Ok((Arc::new(DemotableEngine), self.point))
        }
    }

    impl ModelLoader for FixtureLoader {
        fn load(
            &self,
            model: &ModelRecord,
            _config: ResidencyConfig,
        ) -> Result<(Arc<dyn InferenceEngine>, OperatingPoint), Error> {
            if self
                .failures
                .lock()
                .iter()
                .any(|revision| revision == &model.revision)
            {
                return Err(Error::State("injected load failure".into()));
            }
            let engine: Arc<dyn InferenceEngine> = match &self.blocker {
                Some((entered, release)) if model.revision == "slow" => Arc::new(BlockingEngine {
                    entered: entered.clone(),
                    release: release.clone(),
                }),
                _ => Arc::new(DeterministicEngine),
            };
            Ok((engine, self.point))
        }
    }

    fn config(capacity: u64) -> ResidencyConfig {
        ResidencyConfig {
            device_bytes: capacity,
            host_bytes: capacity,
            storage_bytes: capacity,
            context_reserve_bytes: 1,
            n_ctx: 128,
            gpu_layers: 1,
            require_competent: true,
        }
    }

    fn point(bytes: u64) -> OperatingPoint {
        OperatingPoint {
            model_bytes: bytes,
            context_bytes: 1,
            device_bytes: bytes,
            host_bytes: 0,
            gpu_layers: 1,
            model_layers: 1,
            competent: true,
        }
    }

    fn model(id: &str, revision: &str, epoch: u64, bytes: u64) -> ModelRecord {
        ModelRecord {
            id: id.into(),
            revision: revision.into(),
            path: PathBuf::from("mock://deterministic"),
            sha256: revision.into(),
            aliases: vec![],
            family: "gemma4".into(),
            size_bytes: bytes,
            epoch,
        }
    }

    #[test]
    fn persisted_profile_cannot_weaken_conservative_admission() {
        let database = std::env::temp_dir().join(format!(
            "cusco-residency-profile-{}.sqlite",
            uuid::Uuid::new_v4()
        ));
        let catalog = ModelCatalog::open(&database).unwrap();
        let loader = Arc::new(FixtureLoader {
            point: point(10),
            failures: Mutex::new(vec![]),
            blocker: None,
        });
        let measured = ResidentEngine::with_loader(config(110), loader.clone());
        measured.attach_catalog(catalog.clone());
        let large_declaration = model("profiled", "checksum", 1, 100);
        measured.prepare_model(&large_declaration).unwrap();
        drop(measured);

        let reused = ResidentEngine::with_loader(config(20), loader);
        reused.attach_catalog(catalog);
        assert!(matches!(
            reused.prepare_model(&large_declaration),
            Err(Error::State(message)) if message == "residency capacity is exhausted"
        ));

        let _ = fs::remove_file(database);
    }

    #[test]
    fn persisted_profile_preserves_measured_competence() {
        let database = std::env::temp_dir().join(format!(
            "cusco-residency-competence-{}.sqlite",
            uuid::Uuid::new_v4()
        ));
        let catalog = ModelCatalog::open(&database).unwrap();
        let loader = Arc::new(FixtureLoader {
            point: OperatingPoint {
                competent: false,
                ..point(10)
            },
            failures: Mutex::new(vec![]),
            blocker: None,
        });
        let mut permissive = config(100);
        permissive.require_competent = false;
        let measured = ResidentEngine::with_loader(permissive, loader.clone());
        measured.attach_catalog(catalog.clone());
        let record = model("profiled", "checksum", 1, 10);
        measured.prepare_model(&record).unwrap();
        drop(measured);

        let mut required = config(100);
        required.require_competent = true;
        let reused = ResidentEngine::with_loader(required, loader);
        reused.attach_catalog(catalog);
        assert!(matches!(
            reused.prepare_model(&record),
            Err(Error::State(message))
                if message == "executor operating point is below the competent floor"
        ));

        let _ = fs::remove_file(database);
    }

    #[test]
    fn suspended_session_releases_its_native_slot_without_becoming_evictable() {
        let loader = Arc::new(FixtureLoader {
            point: point(10),
            failures: Mutex::new(vec![]),
            blocker: None,
        });
        let engine = ResidentEngine::with_loader(config(21), loader);
        let model = model("suspended", "r", 1, 10);
        let mut session = engine
            .start_session(test_request(
                &model,
                "yield",
                2,
                Arc::new(RequestControl::new()),
            ))
            .unwrap();

        assert_eq!(engine.status()[0].active_slots, 0);
        assert_eq!(engine.status()[0].sessions, 1);
        let _ = session.step().unwrap();
        assert_eq!(engine.status()[0].active_slots, 0);
        assert_eq!(engine.metrics().active_slots, 0);
        drop(session);
        assert_eq!(engine.status()[0].sessions, 0);
    }
    #[test]
    fn failed_reload_preserves_the_published_epoch() {
        let loader = Arc::new(FixtureLoader {
            point: point(10),
            failures: Mutex::new(vec!["bad".into()]),
            blocker: None,
        });
        let engine = ResidentEngine::with_loader(config(21), loader);
        let old = model("m", "old", 1, 10);
        engine.prepare_model(&old).unwrap();
        engine.commit_model(&old, None);

        assert!(engine.prepare_model(&model("m", "bad", 2, 10)).is_err());
        assert_eq!(
            engine
                .status()
                .iter()
                .map(|status| status.epoch)
                .collect::<Vec<_>>(),
            vec![1]
        );

        let new = model("m", "new", 3, 10);
        engine.prepare_model(&new).unwrap();
        engine.commit_model(&new, Some(1));
        assert_eq!(engine.status()[0].epoch, 3);
        assert_eq!(engine.metrics().reloads, 1);
    }

    #[test]
    fn pressure_evicts_the_least_recently_used_idle_model() {
        let loader = Arc::new(FixtureLoader {
            point: point(10),
            failures: Mutex::new(vec![]),
            blocker: None,
        });
        let engine = ResidentEngine::with_loader(config(21), loader);
        let first = model("first", "r", 1, 10);
        let second = model("second", "r", 2, 10);
        let third = model("third", "r", 3, 10);
        engine.prepare_model(&first).unwrap();
        engine.prepare_model(&second).unwrap();

        let control = Arc::new(RequestControl::new());
        let mut sink = |_: i32, _: &[u8], _: bool| Ok(FrontierControl::Continue);
        engine
            .generate(test_request(&first, "touch", 1, control), &mut sink)
            .unwrap();
        engine.prepare_model(&third).unwrap();

        let ids = engine
            .status()
            .into_iter()
            .map(|status| status.id)
            .collect::<Vec<_>>();
        assert_eq!(ids, vec!["first", "third"]);
        assert_eq!(engine.metrics().unloads, 1);
    }

    #[test]
    fn evicted_entry_cannot_be_acquired_by_a_delayed_request() {
        let loader = Arc::new(FixtureLoader {
            point: point(10),
            failures: Mutex::new(vec![]),
            blocker: None,
        });
        let engine = ResidentEngine::with_loader(config(11), loader);
        let first = engine.load(&model("first", "r", 1, 10), None).unwrap();

        engine.prepare_model(&model("second", "r", 2, 10)).unwrap();

        assert!(!engine.acquire(&first).unwrap());
        assert_eq!(engine.status()[0].id, "second");
    }

    #[test]
    fn active_epoch_drains_after_reload_commit() {
        let entered = Arc::new(Barrier::new(2));
        let release = Arc::new(Barrier::new(2));
        let loader = Arc::new(FixtureLoader {
            point: point(10),
            failures: Mutex::new(vec![]),
            blocker: Some((entered.clone(), release.clone())),
        });
        let engine = ResidentEngine::with_loader(config(21), loader);
        let old = model("m", "slow", 1, 10);
        engine.prepare_model(&old).unwrap();
        engine.commit_model(&old, None);

        let worker_engine = engine.clone();
        let worker = thread::spawn(move || {
            let control = Arc::new(RequestControl::new());
            let mut sink = |_: i32, _: &[u8], _: bool| Ok(FrontierControl::Continue);
            worker_engine
                .generate(test_request(&old, "old epoch", 1, control), &mut sink)
                .unwrap();
        });
        entered.wait();

        let new = model("m", "new", 2, 10);
        engine.prepare_model(&new).unwrap();
        engine.commit_model(&new, Some(1));
        let statuses = engine.status();
        assert_eq!(statuses.len(), 2);
        assert!(
            statuses
                .iter()
                .any(|status| status.epoch == 1 && status.retiring)
        );

        release.wait();
        worker.join().unwrap();
        assert_eq!(engine.status()[0].epoch, 2);
    }

    #[test]
    fn reload_requires_transient_capacity_without_evicting_the_published_epoch() {
        let loader = Arc::new(FixtureLoader {
            point: point(10),
            failures: Mutex::new(vec![]),
            blocker: None,
        });
        let engine = ResidentEngine::with_loader(config(11), loader);
        let old = model("m", "old", 1, 10);
        engine.prepare_model(&old).unwrap();
        engine.commit_model(&old, None);

        assert!(engine.prepare_model(&model("m", "new", 2, 10)).is_err());
        assert_eq!(engine.status()[0].epoch, 1);
        assert!(!engine.status()[0].retiring);
    }

    #[test]
    fn storage_and_competence_floors_reject_before_publication() {
        let loader = Arc::new(FixtureLoader {
            point: OperatingPoint {
                competent: false,
                ..point(10)
            },
            failures: Mutex::new(vec![]),
            blocker: None,
        });
        let mut storage_limited = config(100);
        storage_limited.storage_bytes = 9;
        let engine = ResidentEngine::with_loader(storage_limited, loader.clone());
        assert!(engine.prepare_model(&model("large", "r", 1, 10)).is_err());
        assert!(engine.status().is_empty());

        let mut competent = config(100);
        competent.require_competent = true;
        let engine = ResidentEngine::with_loader(competent, loader);
        assert!(engine.prepare_model(&model("weak", "r", 2, 10)).is_err());
        assert!(engine.status().is_empty());
    }

    #[test]
    fn cpu_context_spill_releases_host_capacity() {
        let point = OperatingPoint {
            model_bytes: 10,
            context_bytes: 5,
            device_bytes: 0,
            host_bytes: 15,
            gpu_layers: 0,
            model_layers: 1,
            competent: true,
        };
        let mut limits = config(25);
        limits.storage_bytes = 25;
        limits.gpu_layers = 0;
        let engine = ResidentEngine::with_loader(limits, Arc::new(DemotableLoader { point }));

        engine.prepare_model(&model("first", "r", 1, 10)).unwrap();
        engine.prepare_model(&model("second", "r", 2, 10)).unwrap();

        assert_eq!(engine.status().len(), 2);
        assert_eq!(engine.metrics().context_spills, 1);
        assert_eq!(engine.metrics().host_bytes, 25);
        assert_eq!(engine.metrics().storage_bytes, 25);
    }

    #[test]
    fn cancelled_load_does_not_publish_a_resident_model() {
        let loader = Arc::new(FixtureLoader {
            point: point(10),
            failures: Mutex::new(vec![]),
            blocker: None,
        });
        let engine = ResidentEngine::with_loader(config(100), loader);
        let control = Arc::new(RequestControl::new());
        control.cancel();
        let mut sink = |_: i32, _: &[u8], _: bool| Ok(FrontierControl::Continue);
        let error = engine
            .generate(
                test_request(&model("cancelled", "r", 1, 10), "cancel", 1, control),
                &mut sink,
            )
            .unwrap_err();
        assert!(matches!(error, Error::Cancelled));
        assert!(engine.status().is_empty());
    }
}
