use crate::{
    EngineOutput, EngineRequest, Error, ExecutionSession, InferenceEngine, PrefillMetrics,
    QuantumKind, QuantumObservation, SessionStep,
};
use cusco_context_store::{
    AdapterEpoch, ComponentMask, ContextStore, EvaluatedPrefixId, LogicalContextId, ModelEpoch,
    PersistentTokenSequence,
};
use cusco_executor::{Decode, Executor, GreedySampler, MappingId, MappingState, OperatingPoint};
use cusco_physical_manager::{
    Capacity, Component, PhysicalManager, PhysicalRepresentationId, Tier,
};
use parking_lot::Mutex;
use serde::{Deserialize, Serialize};
use std::{
    collections::HashMap,
    fs,
    path::{Path, PathBuf},
    sync::Arc,
    time::Instant,
};

const ADAPTER_EPOCH: AdapterEpoch = AdapterEpoch(0);
const EXECUTION_SLOT: LogicalContextId = LogicalContextId(u64::MAX);

#[derive(Clone, Debug, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
struct ProfileCatalog {
    schema_version: u32,
    families: Vec<ExecutionProfile>,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct ExecutionProfile {
    pub id: String,
    pub architecture: String,
    pub model_type: String,
    pub block_size: usize,
    pub recurrent_checkpoint_interval: usize,
    pub context_limit: usize,
    required_components: Vec<ProfileComponent>,
    tokenizer: TokenizerProfile,
    terminal_tokens: Vec<i32>,
    sampler: SamplerProfile,
}

#[derive(Clone, Copy, Debug, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
enum ProfileComponent {
    Kv,
    Swa,
    Recurrent,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
struct TokenizerProfile {
    add_bos: bool,
    parse_special: bool,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
struct SamplerProfile {
    kind: String,
}
impl ExecutionProfile {
    pub fn bundled_gemma() -> Result<Self, Error> {
        let catalog_source = include_str!("../../../config/model-families.yaml");
        let schema: serde_json::Value =
            serde_json::from_str(include_str!("../../../config/model-families.schema.json"))
                .map_err(state_error)?;
        let catalog_value: serde_json::Value =
            serde_yaml::from_str(catalog_source).map_err(state_error)?;
        let validator = jsonschema::JSONSchema::options()
            .with_draft(jsonschema::Draft::Draft202012)
            .compile(&schema)
            .map_err(|error| Error::State(format!("invalid model-family schema: {error}")))?;
        if let Err(mut errors) = validator.validate(&catalog_value) {
            let error = errors.next().expect("schema validation returned an error");
            return Err(Error::State(format!(
                "invalid model-family catalog: {error}"
            )));
        }
        let catalog: ProfileCatalog = serde_json::from_value(catalog_value).map_err(state_error)?;
        if catalog.families.len() != 1 {
            return Err(Error::State("unsupported model-family catalog".into()));
        }
        let profile = catalog.families.into_iter().next().unwrap();
        profile.validate()?;
        Ok(profile)
    }

    fn validate(&self) -> Result<(), Error> {
        if self.id.is_empty()
            || self.architecture != "gemma3"
            || self.model_type != "gemma3_text"
            || self.block_size == 0
            || self.recurrent_checkpoint_interval == 0
            || self.context_limit < self.block_size
            || self.sampler.kind != "greedy"
            || !self.tokenizer.add_bos
            || !self.tokenizer.parse_special
        {
            return Err(Error::State("invalid Gemma execution profile".into()));
        }
        let required = self.required_mask();
        if !required.contains(ComponentMask::GLOBAL_KV)
            || !required.contains(ComponentMask::SWA)
            || !required.contains(ComponentMask::RECURRENT)
        {
            return Err(Error::State(
                "Gemma profile omits a required execution component".into(),
            ));
        }
        Ok(())
    }

    fn required_mask(&self) -> ComponentMask {
        self.required_components
            .iter()
            .fold(ComponentMask::EMPTY, |mask, component| {
                mask.union(match component {
                    ProfileComponent::Kv => ComponentMask::GLOBAL_KV,
                    ProfileComponent::Swa => ComponentMask::SWA,
                    ProfileComponent::Recurrent => ComponentMask::RECURRENT,
                })
            })
    }

    fn components(&self) -> impl Iterator<Item = Component> + '_ {
        self.required_components
            .iter()
            .map(|component| match component {
                ProfileComponent::Kv => Component::GlobalKv,
                ProfileComponent::Swa => Component::SlidingWindow,
                ProfileComponent::Recurrent => Component::Recurrent,
            })
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize)]
pub struct MappedMetrics {
    pub requests: u64,
    pub cache_hits: u64,
    pub cached_tokens: u64,
    pub decoded_tokens: u64,
    pub published_blocks: u64,
    pub reference_switches: u64,
    pub activation_bytes_copied: u64,
}

#[derive(Clone)]
struct ResidentMapping {
    native: Option<MappingId>,
    spill: Option<SpilledMapping>,
    representations: Vec<PhysicalRepresentationId>,
    continuation: Decode,
}

#[derive(Clone)]
struct SpilledMapping {
    path: PathBuf,
    bytes: usize,
    position: usize,
}

struct MappedState {
    executor: Executor,
    logical: ContextStore,
    physical: PhysicalManager,
    model_epoch: ModelEpoch,
    resident: HashMap<EvaluatedPrefixId, ResidentMapping>,
    metrics: MappedMetrics,
    spill_bytes: usize,
}

pub struct MappedEngine {
    profile: ExecutionProfile,
    model_path: String,
    context_capacity: usize,
    spill_dir: Option<PathBuf>,
    spill_capacity: usize,
    state: Arc<Mutex<MappedState>>,
}

impl MappedEngine {
    pub fn open(
        model_family_id: &str,
        model_path: impl AsRef<Path>,
        n_ctx: u32,
        gpu_layers: i32,
        device_bytes: usize,
        host_bytes: usize,
    ) -> Result<Arc<Self>, Error> {
        Self::open_at_epoch(
            model_family_id,
            model_path,
            n_ctx,
            gpu_layers,
            device_bytes,
            host_bytes,
            ModelEpoch(1),
        )
    }

    pub fn open_at_epoch(
        model_family_id: &str,
        model_path: impl AsRef<Path>,
        n_ctx: u32,
        gpu_layers: i32,
        device_bytes: usize,
        host_bytes: usize,
        model_epoch: ModelEpoch,
    ) -> Result<Arc<Self>, Error> {
        Self::open_at_epoch_with_spill(
            model_family_id,
            model_path,
            n_ctx,
            gpu_layers,
            device_bytes,
            host_bytes,
            model_epoch,
            None,
            0,
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub fn open_at_epoch_with_spill(
        model_family_id: &str,
        model_path: impl AsRef<Path>,
        n_ctx: u32,
        gpu_layers: i32,
        device_bytes: usize,
        host_bytes: usize,
        model_epoch: ModelEpoch,
        spill_dir: Option<PathBuf>,
        spill_capacity: usize,
    ) -> Result<Arc<Self>, Error> {
        let profile = ExecutionProfile::bundled_gemma()?;
        if model_family_id != profile.id {
            return Err(Error::State(format!(
                "unsupported model family: {model_family_id}"
            )));
        }
        if n_ctx as usize > profile.context_limit {
            return Err(Error::State(
                "configured context exceeds the Gemma profile limit".into(),
            ));
        }
        if let Some(path) = &spill_dir {
            reset_spill_directory(path)?;
        }
        let model_path = model_path
            .as_ref()
            .to_str()
            .ok_or_else(|| Error::State("model path is not UTF-8".into()))?
            .to_owned();
        let executor = Executor::open(&model_path, n_ctx, gpu_layers).map_err(state_error)?;
        let capabilities = executor.capabilities();
        if !capabilities.mapped_execution
            || !capabilities.global_kv
            || !capabilities.swa
            || !capabilities.recurrent
        {
            return Err(Error::State(
                "executor does not satisfy the Gemma execution profile".into(),
            ));
        }

        Ok(Arc::new(Self {
            profile,
            context_capacity: n_ctx as usize,
            model_path,
            spill_dir,
            spill_capacity,
            state: Arc::new(Mutex::new(MappedState {
                executor,
                logical: ContextStore::default(),
                physical: PhysicalManager::new(Capacity {
                    device_bytes,
                    host_bytes,
                }),
                resident: HashMap::new(),
                model_epoch,
                metrics: MappedMetrics::default(),
                spill_bytes: 0,
            })),
        }))
    }

    pub fn metrics(&self) -> MappedMetrics {
        self.state.lock().metrics
    }

    pub fn spill_inactive_mappings(&self) -> Result<usize, Error> {
        let Some(spill_dir) = &self.spill_dir else {
            return Ok(0);
        };
        let mut state = self.state.lock();
        let active = state.executor.mapping_metrics().active;
        let candidates = state
            .resident
            .iter()
            .filter_map(|(id, resident)| {
                resident
                    .native
                    .filter(|native| *native != active)
                    .map(|native| (*id, native))
            })
            .collect::<Vec<_>>();
        if candidates.is_empty() {
            return Ok(0);
        }
        let _ = state.physical.release_binding(EXECUTION_SLOT);
        let mut spilled = 0;
        for (id, native) in candidates {
            let mapping = state.executor.export_mapping(native).map_err(state_error)?;
            if state.spill_bytes.saturating_add(mapping.bytes.len()) > self.spill_capacity {
                continue;
            }
            let name =
                id.0.iter()
                    .map(|byte| format!("{byte:02x}"))
                    .collect::<String>();
            let path = spill_dir.join(format!("{name}.seq"));
            let temporary = path.with_extension("seq.tmp");
            fs::write(&temporary, &mapping.bytes).map_err(state_error)?;
            fs::rename(&temporary, &path).map_err(state_error)?;
            state.executor.remove_mapping(native).map_err(state_error)?;
            let representations = state
                .resident
                .get(&id)
                .expect("spill candidate remains resident")
                .representations
                .clone();
            for representation in representations {
                state
                    .physical
                    .demote_to_storage(representation)
                    .map_err(state_error)?;
            }
            let bytes = mapping.bytes.len();
            let resident = state.resident.get_mut(&id).unwrap();
            resident.native = None;
            resident.spill = Some(SpilledMapping {
                path,
                bytes,
                position: mapping.position,
            });
            state.spill_bytes += bytes;
            spilled += 1;
        }
        Ok(spilled)
    }

    pub fn profile(&self) -> &ExecutionProfile {
        &self.profile
    }

    pub fn model_path(&self) -> &str {
        &self.model_path
    }

    pub fn operating_point(&self) -> OperatingPoint {
        self.state.lock().executor.operating_point()
    }
}

struct MappedSession {
    profile: ExecutionProfile,
    state: Arc<Mutex<MappedState>>,
    request: EngineRequest,
    prompt_started: Instant,
    stage: MappedStage,
    tokens: Vec<i32>,
    logical_context: Option<LogicalContextId>,
    active_mapping: Option<MappingId>,
    parent: Option<EvaluatedPrefixId>,
    next: Option<Decode>,
    sampler: Option<GreedySampler>,
    input_tokens: usize,
    cached: usize,
    evaluated_tokens: usize,
    evaluated: usize,
    generated: usize,
    prefill: PrefillMetrics,
    finished: Option<EngineOutput>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum MappedStage {
    Preparation,
    Prefill,
    Decode,
    Publication,
    Finished,
}

impl MappedSession {
    fn prepare(&mut self) -> Result<SessionStep, Error> {
        self.request.control.check()?;
        let tokenization_started = Instant::now();
        let mut state = self.state.lock();
        state.executor.reset_cancellation();
        let cancellation = state.executor.cancellation_handle();
        let _abort = self.request.control.register_abort(Arc::new(move || {
            cancellation.cancel();
        }));
        let prompt = state
            .executor
            .tokenize(&self.request.prompt)
            .map_err(state_error)?;
        self.request.control.check()?;
        self.prefill.tokenization_ns = elapsed_ns(tokenization_started);
        self.input_tokens = prompt.len();
        let total = self
            .request
            .prior_tokens
            .len()
            .checked_add(prompt.len())
            .and_then(|value| value.checked_add(self.request.max_tokens))
            .ok_or_else(|| Error::State("request token count overflow".into()))?;
        if total > self.profile.context_limit {
            return Err(Error::State(
                "request exceeds model context capacity".into(),
            ));
        }

        self.tokens = self.request.prior_tokens.clone();
        self.tokens.extend_from_slice(&prompt);
        let sequence = PersistentTokenSequence::default().append(&self.tokens);
        let model_epoch = state.model_epoch;
        let logical_context = state.logical.create(sequence, model_epoch, ADAPTER_EPOCH);
        let prefix_lookup_started = Instant::now();
        let prefix = state
            .logical
            .longest_valid_prefix(logical_context)
            .map_err(state_error)?;
        self.prefill.prefix_lookup_ns = elapsed_ns(prefix_lookup_started);
        self.cached = prefix.as_ref().map_or(0, |mapping| mapping.represented_end);
        self.evaluated_tokens = self.tokens.len().saturating_sub(self.cached);
        self.evaluated = self.cached;
        self.parent = prefix.as_ref().map(|mapping| mapping.id);
        self.next = prefix
            .as_ref()
            .and_then(|mapping| state.resident.get(&mapping.id))
            .map(|resident| resident.continuation.clone());

        let activation_started = Instant::now();
        let transfer_before = state.physical.metrics().transfer_bytes;
        activate_prefix(&mut state, logical_context, prefix.as_deref())?;
        let source = state.executor.mapping_metrics().active;
        self.active_mapping = Some(fork_and_activate(&mut state.executor, source)?);
        self.prefill.mapping_activation_ns = elapsed_ns(activation_started);
        let transfer_after = state.physical.metrics().transfer_bytes;
        self.logical_context = Some(logical_context);
        self.prefill.total_tokens = self.tokens.len();
        self.prefill.cached_tokens = self.cached;
        self.prefill.uncached_tokens = self.evaluated_tokens;
        self.stage = MappedStage::Prefill;
        Ok(SessionStep::Progress(QuantumObservation {
            kind: QuantumKind::Preparation,
            charged_tokens: 1,
            context_placement: "device".into(),
            executor_slot_occupied: true,
            transition_cost_bytes: transfer_after.saturating_sub(transfer_before),
            capacity_reserved_bytes: 0,
        }))
    }

    fn prefill(&mut self) -> Result<SessionStep, Error> {
        if self.evaluated >= self.tokens.len() {
            if self.request.max_tokens > 0 && self.next.is_none() {
                return Err(Error::State(
                    "an exact cached prefix cannot supply uncached logits".into(),
                ));
            }
            let mut state = self.state.lock();
            self.sampler = Some(state.executor.greedy_sampler().map_err(state_error)?);
            self.stage = MappedStage::Decode;
            return Ok(SessionStep::Progress(QuantumObservation::model_free(
                QuantumKind::Prefill,
                1,
            )));
        }

        let started = Instant::now();
        let mut state = self.state.lock();
        self.request.control.check()?;
        state.executor.reset_cancellation();
        let cancellation = state.executor.cancellation_handle();
        let _abort = self.request.control.register_abort(Arc::new(move || {
            cancellation.cancel();
        }));
        self.request.control.check()?;
        self.activate_owned(&mut state)?;
        let next_block = ((self.evaluated / self.profile.block_size) + 1)
            .saturating_mul(self.profile.block_size);
        let end = self.tokens.len().min(next_block).min(
            self.evaluated
                .saturating_add(self.request.prefill_chunk_tokens),
        );
        let charged = end.saturating_sub(self.evaluated);
        self.next = Some(
            state
                .executor
                .decode(&self.tokens[self.evaluated..end])
                .map_err(state_error)?,
        );
        self.request.control.check()?;
        state.metrics.decoded_tokens += charged as u64;
        self.evaluated = end;
        if self.evaluated % self.profile.block_size == 0 {
            self.parent = Some(publish_block(
                &mut state,
                &self.profile,
                self.logical_context.expect("prepared context exists"),
                self.evaluated,
                self.parent,
                self.active_mapping.expect("prepared mapping exists"),
                self.next.as_ref().expect("decode result exists"),
            )?);
            let source = self.active_mapping.expect("published mapping exists");
            self.active_mapping = Some(fork_and_activate(&mut state.executor, source)?);
        }
        self.prefill.uncached_prefill_ns = self
            .prefill
            .uncached_prefill_ns
            .saturating_add(elapsed_ns(started));
        Ok(SessionStep::Progress(QuantumObservation {
            kind: QuantumKind::Prefill,
            charged_tokens: charged.max(1),
            context_placement: "device".into(),
            executor_slot_occupied: true,
            transition_cost_bytes: 0,
            capacity_reserved_bytes: 0,
        }))
    }

    fn decode(&mut self) -> Result<SessionStep, Error> {
        if self.generated >= self.request.max_tokens {
            self.stage = MappedStage::Publication;
            return self.publish().map(SessionStep::Finished);
        }
        let mut state = self.state.lock();
        self.request.control.check()?;
        state.executor.reset_cancellation();
        let cancellation = state.executor.cancellation_handle();
        let _abort = self.request.control.register_abort(Arc::new(move || {
            cancellation.cancel();
        }));
        self.request.control.check()?;
        self.activate_owned(&mut state)?;
        let sampled = self
            .sampler
            .as_mut()
            .expect("decode owns sampler")
            .sample(&mut state.executor)
            .map_err(state_error)?;
        let terminal_or_control = self.profile.terminal_tokens.contains(&sampled);
        let mut piece = Vec::with_capacity(32);
        if !terminal_or_control {
            state
                .executor
                .render_token(sampled, &mut piece)
                .map_err(state_error)?;
        }
        self.request.control.check()?;
        self.tokens.push(sampled);
        state
            .logical
            .append(
                self.logical_context.expect("prepared context exists"),
                &[sampled],
            )
            .map_err(state_error)?;
        self.generated += 1;
        if terminal_or_control || self.generated == self.request.max_tokens {
            self.stage = MappedStage::Publication;
        } else {
            self.next = Some(state.executor.decode(&[sampled]).map_err(state_error)?);
            self.request.control.check()?;
            state.metrics.decoded_tokens += 1;
            self.evaluated += 1;
            if self.evaluated % self.profile.block_size == 0 {
                self.parent = Some(publish_block(
                    &mut state,
                    &self.profile,
                    self.logical_context.expect("prepared context exists"),
                    self.evaluated,
                    self.parent,
                    self.active_mapping.expect("prepared mapping exists"),
                    self.next.as_ref().expect("decode result exists"),
                )?);
                let source = self.active_mapping.expect("published mapping exists");
                self.active_mapping = Some(fork_and_activate(&mut state.executor, source)?);
            }
        }
        Ok(SessionStep::Token {
            id: sampled,
            piece,
            terminal_or_control,
            observation: QuantumObservation {
                kind: QuantumKind::Decode,
                charged_tokens: 1,
                context_placement: "device".into(),
                executor_slot_occupied: true,
                transition_cost_bytes: 0,
                capacity_reserved_bytes: 0,
            },
        })
    }

    fn activate_owned(&self, state: &mut MappedState) -> Result<(), Error> {
        state
            .executor
            .activate_mapping(self.active_mapping.expect("prepared mapping exists"))
            .map_err(state_error)
    }

    fn publish(&mut self) -> Result<EngineOutput, Error> {
        if let Some(output) = &self.finished {
            return Ok(output.clone());
        }
        self.request.control.check()?;
        self.sampler = None;
        let mut state = self.state.lock();
        if let Some(active) = self.active_mapping.take() {
            let fallback = self
                .parent
                .and_then(|id| state.resident.get(&id))
                .and_then(|resident| resident.native)
                .unwrap_or(MappingId(0));
            if active != fallback {
                state
                    .executor
                    .activate_mapping(fallback)
                    .map_err(state_error)?;
                state.executor.remove_mapping(active).map_err(state_error)?;
            }
        }
        let native_metrics = state.executor.mapping_metrics();
        state.metrics.requests += 1;
        state.metrics.cached_tokens += self.cached as u64;
        state.metrics.cache_hits += u64::from(self.cached != 0);
        state.metrics.reference_switches = native_metrics.reference_switches;
        state.metrics.activation_bytes_copied = native_metrics.activation_bytes_copied;
        let physical_metrics = state.physical.metrics();
        self.prefill.transfer_bytes = physical_metrics.transfer_bytes;
        self.prefill.device_bytes = physical_metrics.device_total;
        self.prefill.host_bytes = physical_metrics.host_used;
        self.prefill.total_ns = elapsed_ns(self.prompt_started);
        let output = EngineOutput {
            successor_tokens: self.tokens.clone(),
            input_tokens: self.input_tokens,
            cached_tokens: self.cached,
            evaluated_tokens: self.evaluated_tokens,
            prefill: self.prefill,
        };
        self.finished = Some(output.clone());
        self.stage = MappedStage::Finished;
        Ok(output)
    }
}

impl ExecutionSession for MappedSession {
    fn step(&mut self) -> Result<SessionStep, Error> {
        match self.stage {
            MappedStage::Preparation => self.prepare(),
            MappedStage::Prefill => self.prefill(),
            MappedStage::Decode => self.decode(),
            MappedStage::Publication => self.publish().map(SessionStep::Finished),
            MappedStage::Finished => Ok(SessionStep::Finished(
                self.finished.clone().expect("finished output exists"),
            )),
        }
    }

    fn finish(&mut self) -> Result<EngineOutput, Error> {
        self.publish()
    }
}

impl Drop for MappedSession {
    fn drop(&mut self) {
        self.sampler = None;
        if let Some(active) = self.active_mapping.take() {
            let mut state = self.state.lock();
            let fallback = self
                .parent
                .and_then(|id| state.resident.get(&id))
                .and_then(|resident| resident.native)
                .unwrap_or(MappingId(0));
            if active != fallback {
                let _ = state.executor.activate_mapping(fallback);
                let _ = state.executor.remove_mapping(active);
            }
        }
    }
}

impl InferenceEngine for MappedEngine {
    fn start_session(&self, request: EngineRequest) -> Result<Box<dyn ExecutionSession>, Error> {
        request.control.check()?;
        if request.prefill_chunk_tokens == 0 {
            return Err(Error::State("prefill chunk size must be nonzero".into()));
        }
        if request.model.path.to_str() != Some(self.model_path()) {
            return Err(Error::State(
                "Phase 8 admits only a resident process-owned model".into(),
            ));
        }
        let context_limit = self.context_capacity.min(self.profile.context_limit);
        let mut profile = self.profile.clone();
        profile.context_limit = context_limit;
        Ok(Box::new(MappedSession {
            profile,
            state: self.state.clone(),
            request,
            prompt_started: Instant::now(),
            stage: MappedStage::Preparation,
            tokens: Vec::new(),
            logical_context: None,
            active_mapping: None,
            parent: None,
            next: None,
            sampler: None,
            input_tokens: 0,
            cached: 0,
            evaluated_tokens: 0,
            evaluated: 0,
            generated: 0,
            prefill: PrefillMetrics::default(),
            finished: None,
        }))
    }

    fn demote_inactive(&self) -> Result<usize, Error> {
        self.spill_inactive_mappings()
    }
}

fn activate_prefix(
    state: &mut MappedState,
    context: LogicalContextId,
    prefix: Option<&cusco_context_store::EvaluatedPrefix>,
) -> Result<(), Error> {
    let Some(prefix) = prefix else {
        state
            .executor
            .activate_mapping(MappingId(0))
            .map_err(state_error)?;
        return Ok(());
    };
    let resident = state
        .resident
        .get(&prefix.id)
        .cloned()
        .ok_or_else(|| Error::State("logical mapping has no resident native mapping".into()))?;
    let restored = resident.native.is_none();
    let native = if let Some(native) = resident.native {
        native
    } else {
        let spilled = resident
            .spill
            .as_ref()
            .ok_or_else(|| Error::State("mapping has neither native nor spilled state".into()))?;
        let bytes = fs::read(&spilled.path).map_err(state_error)?;
        if bytes.len() != spilled.bytes {
            return Err(Error::State("spilled mapping size changed".into()));
        }
        state
            .executor
            .import_mapping(&MappingState {
                bytes,
                position: spilled.position,
            })
            .map_err(state_error)?
    };
    let revision = state
        .logical
        .context(context)
        .ok_or_else(|| Error::State("logical context disappeared".into()))?
        .revision;
    let previous = state.executor.mapping_metrics().active;
    let (prepared, _, transfers) = state
        .physical
        .prepare_transition(
            EXECUTION_SLOT,
            revision,
            prefix.id,
            prefix.represented_end,
            prefix.required_components,
            &resident.representations,
            false,
        )
        .map_err(|error| {
            if restored {
                let _ = state.executor.remove_mapping(native);
            }
            state_error(error)
        })?;
    for transfer in transfers {
        if let Err(error) = state.physical.complete_transfer(transfer, true) {
            let _ = state.physical.abort_transition(prepared);
            if restored {
                let _ = state.executor.remove_mapping(native);
            }
            return Err(state_error(error));
        }
    }
    if let Err(error) = state.executor.activate_mapping(native) {
        let _ = state.physical.abort_transition(prepared);
        if restored {
            let _ = state.executor.remove_mapping(native);
        }
        return Err(state_error(error));
    }
    match state.physical.commit_transition(prepared, revision) {
        Ok(binding) => {
            state
                .physical
                .publish_device_block_table(EXECUTION_SLOT, binding, native.0)
                .map_err(state_error)?;
            if restored {
                let spilled = state
                    .resident
                    .get_mut(&prefix.id)
                    .expect("restored mapping remains resident")
                    .spill
                    .take()
                    .expect("restored mapping has spill metadata");
                let _ = fs::remove_file(spilled.path);
                state.spill_bytes = state.spill_bytes.saturating_sub(spilled.bytes);
                state.resident.get_mut(&prefix.id).unwrap().native = Some(native);
            }
            Ok(())
        }
        Err(error) => {
            let _ = state.executor.activate_mapping(previous);
            if restored {
                let _ = state.executor.remove_mapping(native);
            }
            Err(state_error(error))
        }
    }
}

fn fork_and_activate(executor: &mut Executor, source: MappingId) -> Result<MappingId, Error> {
    let prepared = executor.prepare_mapping_fork(source).map_err(state_error)?;
    let mapping = executor.commit_mapping(prepared).map_err(state_error)?;
    executor.activate_mapping(mapping).map_err(state_error)?;
    Ok(mapping)
}

fn publish_block(
    state: &mut MappedState,
    profile: &ExecutionProfile,
    context: LogicalContextId,
    represented_end: usize,
    parent: Option<EvaluatedPrefixId>,
    native: MappingId,
    continuation: &Decode,
) -> Result<EvaluatedPrefixId, Error> {
    let required = profile.required_mask();
    let bytes_per_component = represented_end.saturating_mul(1024);
    let required_bytes = bytes_per_component
        .checked_mul(profile.required_components.len())
        .ok_or_else(|| Error::State("representation size overflow".into()))?;
    let capacity = state.physical.metrics();
    let unavailable = capacity
        .device_active
        .saturating_add(capacity.device_growth_reserved)
        .saturating_add(capacity.device_transition_reserved)
        .saturating_add(capacity.device_detached_transfer_reserved);
    if required_bytes > capacity.device_total.saturating_sub(unavailable) {
        return Err(Error::State(
            "device capacity cannot publish the completed block".into(),
        ));
    }
    let rollback = parent
        .and_then(|id| state.resident.get(&id))
        .and_then(|resident| resident.native)
        .unwrap_or(MappingId(0));
    let prepared_publication = state
        .logical
        .prepare_publication(
            context,
            represented_end,
            parent,
            required,
            required,
            [0; 32],
        )
        .map_err(state_error)?;
    let mapping_id = prepared_publication.mapping_id();
    let existing = state.resident.get(&mapping_id).cloned();
    let mut registered = Vec::new();
    let representations = if let Some(resident) = &existing {
        resident.representations.clone()
    } else {
        registered.reserve(profile.required_components.len());
        for component in profile.components() {
            match state.physical.register(
                mapping_id,
                component,
                represented_end,
                bytes_per_component,
                Tier::Host,
                represented_end as u64,
            ) {
                Ok(id) => registered.push(id),
                Err(error) => {
                    release_representations(&mut state.physical, &registered);
                    let _ = state.executor.activate_mapping(rollback);
                    return Err(state_error(error));
                }
            }
        }
        registered.clone()
    };
    let revision = state
        .logical
        .context(context)
        .ok_or_else(|| Error::State("logical context disappeared".into()))?
        .revision;
    let expected_revision = revision
        .checked_add(1)
        .ok_or_else(|| Error::State("logical context revision exhausted".into()))?;
    let (transition, _, transfers) = match state.physical.prepare_transition(
        EXECUTION_SLOT,
        expected_revision,
        mapping_id,
        represented_end,
        required,
        &representations,
        false,
    ) {
        Ok(prepared) => prepared,
        Err(error) => {
            release_representations(&mut state.physical, &registered);
            let _ = state.executor.activate_mapping(rollback);
            return Err(state_error(error));
        }
    };
    for transfer in transfers {
        if let Err(error) = state.physical.complete_transfer(transfer, true) {
            let _ = state.physical.abort_transition(transition);
            release_representations(&mut state.physical, &registered);
            let _ = state.executor.activate_mapping(rollback);
            return Err(state_error(error));
        }
    }
    let mapping = match state.logical.commit_publication(prepared_publication) {
        Ok(mapping) => mapping,
        Err(error) => {
            let _ = state.physical.abort_transition(transition);
            release_representations(&mut state.physical, &registered);
            let _ = state.executor.activate_mapping(rollback);
            return Err(state_error(error));
        }
    };
    let binding = match state
        .physical
        .commit_transition(transition, expected_revision)
    {
        Ok(binding) => binding,
        Err(error) => {
            release_representations(&mut state.physical, &registered);
            let _ = state.executor.activate_mapping(rollback);
            return Err(state_error(error));
        }
    };
    let published_native = existing
        .as_ref()
        .and_then(|resident| resident.native)
        .unwrap_or(native);
    if let Err(error) =
        state
            .physical
            .publish_device_block_table(EXECUTION_SLOT, binding, published_native.0)
    {
        let _ = state.executor.activate_mapping(rollback);
        return Err(state_error(error));
    }
    if existing.is_none() {
        state.resident.insert(
            mapping.id,
            ResidentMapping {
                native: Some(native),
                spill: None,
                representations,
                continuation: continuation.clone(),
            },
        );
        state.metrics.published_blocks += 1;
    }
    Ok(mapping.id)
}

fn release_representations(
    physical: &mut PhysicalManager,
    representations: &[PhysicalRepresentationId],
) {
    for representation in representations {
        let _ = physical.release_logical_reference(*representation);
    }
}

fn elapsed_ns(started: Instant) -> u64 {
    started.elapsed().as_nanos().min(u64::MAX as u128) as u64
}

fn reset_spill_directory(path: &Path) -> Result<(), Error> {
    fs::create_dir_all(path).map_err(state_error)?;
    for entry in fs::read_dir(path).map_err(state_error)? {
        let entry = entry.map_err(state_error)?;
        if !entry.file_type().map_err(state_error)?.is_file() {
            continue;
        }
        let name = entry.file_name();
        let name = name.to_string_lossy();
        if name.ends_with(".seq") || name.ends_with(".seq.tmp") {
            fs::remove_file(entry.path()).map_err(state_error)?;
        }
    }
    Ok(())
}

fn state_error(error: impl std::fmt::Display) -> Error {
    Error::State(error.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{FrontierControl, ModelRecord, RequestControl, SchedulingMetadata};
    use std::{path::PathBuf, sync::Arc};

    fn model() -> ModelRecord {
        ModelRecord {
            id: "gemma".into(),
            revision: "test".into(),
            path: PathBuf::from("mock://deterministic"),
            sha256: "mock".into(),
            aliases: vec![],
            family: "gemma3".into(),
            size_bytes: 1,
            epoch: 1,
        }
    }

    fn test_request(
        model: &ModelRecord,
        prompt: impl Into<String>,
        max_tokens: usize,
        prior_tokens: &[i32],
    ) -> EngineRequest {
        EngineRequest {
            model: model.clone(),
            prompt: prompt.into(),
            max_tokens,
            prior_tokens: prior_tokens.to_vec(),
            control: Arc::new(RequestControl::new()),
            scheduling: SchedulingMetadata::default(),
            prefill_chunk_tokens: 32,
        }
    }

    #[derive(Debug)]
    struct CollectedOutput {
        pieces: Vec<Vec<u8>>,
        successor_tokens: Vec<i32>,
        cached_tokens: usize,
        evaluated_tokens: usize,
    }

    trait GenerateCollected {
        fn generate_collected(&self, request: EngineRequest) -> Result<CollectedOutput, Error>;
    }
    impl GenerateCollected for MappedEngine {
        fn generate_collected(&self, request: EngineRequest) -> Result<CollectedOutput, Error> {
            let mut pieces = Vec::new();
            let output = self.generate(request, &mut |_, piece, _| {
                pieces.push(piece.to_vec());
                Ok(FrontierControl::Continue)
            })?;
            Ok(CollectedOutput {
                pieces,
                successor_tokens: output.successor_tokens,
                cached_tokens: output.cached_tokens,
                evaluated_tokens: output.evaluated_tokens,
            })
        }
    }

    #[test]
    fn bundled_profile_is_strict_and_complete() {
        let profile = ExecutionProfile::bundled_gemma().unwrap();
        assert_eq!(profile.block_size, 32);
        assert_eq!(profile.recurrent_checkpoint_interval, 8);
        assert!(profile.required_mask().contains(ComponentMask::RECURRENT));
        let schema: serde_json::Value =
            serde_json::from_str(include_str!("../../../config/model-families.schema.json"))
                .unwrap();
        assert_eq!(schema["properties"]["schema_version"]["const"], 1);
    }

    #[test]
    fn profile_rejects_missing_global_kv() {
        let mut missing_kv = ExecutionProfile::bundled_gemma().unwrap();
        missing_kv
            .required_components
            .retain(|component| *component != ProfileComponent::Kv);
        assert!(
            !missing_kv
                .required_mask()
                .contains(ComponentMask::GLOBAL_KV)
        );
        assert!(matches!(
            missing_kv.validate(),
            Err(Error::State(message))
                if message == "Gemma profile omits a required execution component"
        ));
    }

    #[test]
    fn mapped_state_is_reused_without_activation_copying() {
        let engine =
            MappedEngine::open("gemma3", "mock://deterministic", 4096, 0, 1 << 30, 1 << 30)
                .unwrap();
        let model = model();
        let first = engine
            .generate_collected(test_request(&model, &"a".repeat(40), 2, &[]))
            .unwrap();
        let second = engine
            .generate_collected(test_request(&model, "z", 1, &first.successor_tokens))
            .unwrap();
        assert!(!second.pieces.is_empty());
        let metrics = engine.metrics();
        assert_eq!(metrics.requests, 2);
        assert_eq!(metrics.cache_hits, 1);
        assert!(metrics.cached_tokens >= 32);
        assert_eq!(metrics.activation_bytes_copied, 0);
        assert!(metrics.reference_switches >= 2);
    }

    #[test]
    fn restart_removes_orphaned_runtime_spills() {
        let spill_dir =
            std::env::temp_dir().join(format!("cusco-spill-restart-{}", uuid::Uuid::new_v4()));
        fs::create_dir_all(&spill_dir).unwrap();
        fs::write(spill_dir.join("stale.seq"), b"checkpoint").unwrap();
        fs::write(spill_dir.join("stale.seq.tmp"), b"temporary").unwrap();
        fs::write(spill_dir.join("operator-note"), b"keep").unwrap();

        let _engine = MappedEngine::open_at_epoch_with_spill(
            "gemma3",
            "mock://deterministic",
            4096,
            0,
            1 << 30,
            1 << 30,
            ModelEpoch(1),
            Some(spill_dir.clone()),
            1 << 20,
        )
        .unwrap();

        assert!(!spill_dir.join("stale.seq").exists());
        assert!(!spill_dir.join("stale.seq.tmp").exists());
        assert!(spill_dir.join("operator-note").exists());
        fs::remove_dir_all(spill_dir).unwrap();
    }

    #[test]
    fn spilled_mapping_restores_exact_continuation() {
        let spill_dir = std::env::temp_dir().join(format!("cusco-spill-{}", uuid::Uuid::new_v4()));
        let engine = MappedEngine::open_at_epoch_with_spill(
            "gemma3",
            "mock://deterministic",
            4096,
            0,
            1 << 30,
            1 << 30,
            ModelEpoch(1),
            Some(spill_dir.clone()),
            1 << 20,
        )
        .unwrap();
        let model = model();
        let first = engine
            .generate_collected(test_request(&model, &"a".repeat(40), 2, &[]))
            .unwrap();
        engine
            .generate_collected(test_request(&model, &"b".repeat(40), 1, &[]))
            .unwrap();
        let control =
            MappedEngine::open("gemma3", "mock://deterministic", 4096, 0, 1 << 30, 1 << 30)
                .unwrap();
        let control_first = control
            .generate_collected(test_request(&model, &"a".repeat(40), 2, &[]))
            .unwrap();
        let baseline = control
            .generate_collected(test_request(
                &model,
                "z",
                2,
                &control_first.successor_tokens,
            ))
            .unwrap();
        assert_eq!(engine.spill_inactive_mappings().unwrap(), 1);
        assert_eq!(fs::read_dir(&spill_dir).unwrap().count(), 1);

        let resumed = engine
            .generate_collected(test_request(&model, "z", 2, &first.successor_tokens))
            .unwrap();
        assert_eq!(resumed.cached_tokens, 32);
        assert_eq!(resumed.pieces, baseline.pieces);
        assert_eq!(resumed.successor_tokens, baseline.successor_tokens);
        assert_eq!(fs::read_dir(&spill_dir).unwrap().count(), 0);
        fs::remove_dir_all(spill_dir).unwrap();
    }

    #[test]
    fn capacity_rejection_precedes_decode_and_preserves_metrics() {
        let engine =
            MappedEngine::open("gemma3", "mock://deterministic", 64, 0, 1 << 20, 1 << 20).unwrap();
        let model = model();
        let error = engine
            .generate_collected(test_request(&model, &"x".repeat(65), 1, &[]))
            .unwrap_err();
        assert!(error.to_string().contains("context capacity"));
        assert_eq!(engine.metrics(), MappedMetrics::default());
    }

    #[test]
    fn publication_failure_keeps_the_prior_mapping_reusable() {
        let engine =
            MappedEngine::open("gemma3", "mock://deterministic", 4096, 0, 250_000, 1 << 20)
                .unwrap();
        let model = model();
        let first = engine
            .generate_collected(test_request(&model, &"a".repeat(40), 1, &[]))
            .unwrap();
        let failed = engine.generate_collected(test_request(
            &model,
            &"b".repeat(25),
            1,
            &first.successor_tokens,
        ));
        assert!(failed.unwrap_err().to_string().contains("cannot publish"));
        let resumed = engine
            .generate_collected(test_request(&model, "c", 1, &first.successor_tokens))
            .unwrap();
        assert_eq!(resumed.pieces.len(), 1);
        assert_eq!(engine.metrics().requests, 2);
        assert!(engine.metrics().cache_hits >= 1);
    }

    #[test]
    fn exact_cached_prefix_resumes_from_its_saved_logits() {
        let engine =
            MappedEngine::open("gemma3", "mock://deterministic", 4096, 0, 1 << 30, 1 << 30)
                .unwrap();
        let model = model();
        let primed = engine
            .generate_collected(test_request(&model, &"p".repeat(32), 0, &[]))
            .unwrap();
        let resumed = engine
            .generate_collected(test_request(&model, "", 1, &primed.successor_tokens))
            .unwrap();
        assert_eq!(resumed.pieces.len(), 1);
        assert_eq!(resumed.cached_tokens, 32);
        assert_eq!(resumed.evaluated_tokens, 0);
    }

    #[test]
    fn server_context_persists_native_tokens_and_reports_cache_work() {
        use crate::{AnonymousAdmin, InferRequest, Server};
        let directory =
            std::env::temp_dir().join(format!("cusco-mapped-server-{}", std::process::id()));
        std::fs::create_dir_all(&directory).unwrap();
        let engine =
            MappedEngine::open("gemma3", "mock://deterministic", 4096, 0, 1 << 30, 1 << 30)
                .unwrap();
        let server = Server::open(
            directory.join("state.json"),
            Arc::new(AnonymousAdmin),
            engine,
        )
        .unwrap();
        server.register_model(model()).unwrap();
        let first = server
            .infer(
                "first",
                InferRequest {
                    model: "gemma".into(),
                    prompt: "a".repeat(40),
                    max_tokens: 1,
                    context_id: None,
                    deadline_ms: None,
                    scheduling: SchedulingMetadata::default(),
                    stop: vec![],
                    raw_continuation: false,
                },
            )
            .unwrap()
            .0;
        let durable = server.context(&first.usage.context_id).unwrap();
        assert!(!durable.native_tokens.is_empty());
        let second = server
            .infer(
                "second",
                InferRequest {
                    model: "gemma".into(),
                    prompt: "c".into(),
                    max_tokens: 1,
                    context_id: Some(durable.id),
                    deadline_ms: None,
                    scheduling: SchedulingMetadata::default(),
                    stop: vec![],
                    raw_continuation: false,
                },
            )
            .unwrap()
            .0;
        assert!(second.usage.cached_tokens >= 32);
        assert!(second.usage.evaluated_tokens < durable.native_tokens.len());
        for usage in [&first.usage, &second.usage] {
            assert_eq!(usage.prefill.cached_tokens, usage.cached_tokens);
            assert_eq!(usage.prefill.uncached_tokens, usage.evaluated_tokens);
            assert_eq!(
                usage.prefill.total_tokens,
                usage.cached_tokens + usage.evaluated_tokens
            );
            assert!(usage.prefill.total_ns >= usage.prefill.tokenization_ns);
            assert!(usage.prefill.total_ns >= usage.prefill.prefix_lookup_ns);
            assert!(usage.prefill.total_ns >= usage.prefill.mapping_activation_ns);
            assert!(usage.prefill.total_ns >= usage.prefill.uncached_prefill_ns);
        }
        assert!(serde_json::to_value(&second.usage).unwrap()["prefill"]["total_ns"].is_u64());
        std::fs::remove_dir_all(directory).unwrap();
    }
}
