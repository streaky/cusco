use crate::{
    EngineOutput, EngineRequest, Error, ExecutionSession, InferenceEngine, PrefillMetrics,
    QuantumKind, QuantumObservation, SessionStep,
};
use cusco_context_store::{
    AdapterEpoch, ComponentMask, ContextStore, EvaluatedPrefixId, LogicalContextId, ModelEpoch,
    PersistentTokenSequence,
};
use cusco_executor::{
    Capabilities, Decode, Executor, MappingState, OperatingPoint, RepresentationHandle, Sampler,
};
use cusco_physical_manager::{
    Capacity, Component, PhysicalManager, PhysicalRepresentationId, Tier,
};
use parking_lot::Mutex;
use serde::Serialize;
use std::{
    collections::HashMap,
    fs,
    path::{Path, PathBuf},
    sync::Arc,
    time::Instant,
};

const ADAPTER_EPOCH: AdapterEpoch = AdapterEpoch(0);
const EXECUTION_SLOT: LogicalContextId = LogicalContextId(u64::MAX);
const DEFAULT_PUBLICATION_INTERVAL_TOKENS: usize = 32;

#[derive(Clone, Copy, Debug)]
struct ExecutionProfile {
    publication_interval_tokens: usize,
    required_components: ComponentMask,
}

impl ExecutionProfile {
    fn from_capabilities(
        capabilities: &Capabilities,
        publication_interval_tokens: usize,
    ) -> Result<Self, Error> {
        if publication_interval_tokens == 0 {
            return Err(Error::State(
                "prefix publication interval must be nonzero".into(),
            ));
        }
        if !capabilities.mapped_execution {
            return Err(Error::State(
                "executor does not support mapped execution".into(),
            ));
        }
        let mut required_components = ComponentMask::EMPTY;
        if capabilities.global_kv {
            required_components = required_components.union(ComponentMask::GLOBAL_KV);
        }
        if capabilities.swa {
            required_components = required_components.union(ComponentMask::SWA);
        }
        if capabilities.recurrent {
            required_components = required_components.union(ComponentMask::RECURRENT);
        }
        if required_components == ComponentMask::EMPTY {
            return Err(Error::State(
                "executor did not report any checkpoint state components".into(),
            ));
        }
        Ok(Self {
            publication_interval_tokens,
            required_components,
        })
    }

    fn required_mask(self) -> ComponentMask {
        self.required_components
    }

    fn components(self) -> impl Iterator<Item = Component> {
        [
            (ComponentMask::GLOBAL_KV, Component::GlobalKv),
            (ComponentMask::SWA, Component::SlidingWindow),
            (ComponentMask::RECURRENT, Component::Recurrent),
        ]
        .into_iter()
        .filter_map(move |(mask, component)| {
            self.required_components.contains(mask).then_some(component)
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
    pub fork_bytes_copied: u64,
    pub export_bytes_copied: u64,
    pub import_bytes_copied: u64,
    pub total_bytes_copied: u64,
    pub graph_recaptures: Option<u64>,
}

#[derive(Clone)]
struct ResidentMapping {
    native: Option<RepresentationHandle>,
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
    root: RepresentationHandle,
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
        model_path: impl AsRef<Path>,
        n_ctx: u32,
        gpu_layers: i32,
        device_bytes: usize,
        host_bytes: usize,
    ) -> Result<Arc<Self>, Error> {
        Self::open_at_epoch(
            model_path,
            n_ctx,
            gpu_layers,
            device_bytes,
            host_bytes,
            ModelEpoch(1),
        )
    }

    pub fn open_at_epoch(
        model_path: impl AsRef<Path>,
        n_ctx: u32,
        gpu_layers: i32,
        device_bytes: usize,
        host_bytes: usize,
        model_epoch: ModelEpoch,
    ) -> Result<Arc<Self>, Error> {
        Self::open_at_epoch_with_spill(
            model_path,
            n_ctx,
            gpu_layers,
            device_bytes,
            host_bytes,
            model_epoch,
            None,
            0,
            DEFAULT_PUBLICATION_INTERVAL_TOKENS,
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub fn open_at_epoch_with_spill(
        model_path: impl AsRef<Path>,
        n_ctx: u32,
        gpu_layers: i32,
        device_bytes: usize,
        host_bytes: usize,
        model_epoch: ModelEpoch,
        spill_dir: Option<PathBuf>,
        spill_capacity: usize,
        publication_interval_tokens: usize,
    ) -> Result<Arc<Self>, Error> {
        if let Some(path) = &spill_dir {
            reset_spill_directory(path)?;
        }
        let model_path = model_path
            .as_ref()
            .to_str()
            .ok_or_else(|| Error::State("model path is not UTF-8".into()))?
            .to_owned();
        let mut executor = Executor::open(&model_path, n_ctx, gpu_layers).map_err(state_error)?;
        let capabilities = executor.capabilities();
        let profile =
            ExecutionProfile::from_capabilities(&capabilities, publication_interval_tokens)?;
        if capabilities.training_context_tokens == 0 {
            return Err(Error::State(
                "executor did not report the model context capacity".into(),
            ));
        }
        let root = executor.active_representation().map_err(state_error)?;

        Ok(Arc::new(Self {
            profile,
            context_capacity: n_ctx as usize,
            model_path,
            spill_dir,
            spill_capacity,
            state: Arc::new(Mutex::new(MappedState {
                executor,
                root,
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
        let active = state.executor.mapping_metrics().active_identity;
        let candidates = state
            .resident
            .iter()
            .filter_map(|(id, resident)| {
                resident
                    .native
                    .as_ref()
                    .filter(|native| native.identity() != active)
                    .cloned()
                    .map(|native| (*id, native))
            })
            .collect::<Vec<_>>();
        if candidates.is_empty() {
            return Ok(0);
        }
        let _ = state.physical.release_binding(EXECUTION_SLOT);
        let mut spilled = 0;
        for (id, native) in candidates {
            let mapping = state
                .executor
                .export_mapping(&native)
                .map_err(state_error)?;
            if state.spill_bytes.saturating_add(mapping.bytes.len()) > self.spill_capacity {
                continue;
            }
            let name = hex::encode(id.0);
            let path = spill_dir.join(format!("{name}.seq"));
            let temporary = path.with_extension("seq.tmp");
            fs::write(&temporary, &mapping.bytes).map_err(state_error)?;
            fs::rename(&temporary, &path).map_err(state_error)?;
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

    pub fn model_path(&self) -> &str {
        &self.model_path
    }

    pub fn operating_point(&self) -> OperatingPoint {
        self.state.lock().executor.operating_point()
    }
}

struct MappedSession {
    profile: ExecutionProfile,
    context_capacity: usize,
    state: Arc<Mutex<MappedState>>,
    request: EngineRequest,
    prompt_started: Instant,
    stage: MappedStage,
    tokens: Vec<i32>,
    logical_context: Option<LogicalContextId>,
    active_mapping: Option<RepresentationHandle>,
    parent: Option<EvaluatedPrefixId>,
    next: Option<Decode>,
    sampler: Option<Sampler>,
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
        if total > self.context_capacity {
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
        let source = state
            .executor
            .active_representation()
            .map_err(state_error)?;
        self.active_mapping = Some(fork_for_request_branch(&mut state.executor, &source)?);
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
            self.sampler = Some(
                state
                    .executor
                    .sampler_with_grammar(self.request.sampling, self.request.grammar.as_deref())
                    .map_err(state_error)?,
            );
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
        let next_block = ((self.evaluated / self.profile.publication_interval_tokens) + 1)
            .saturating_mul(self.profile.publication_interval_tokens);
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
        if self.evaluated % self.profile.publication_interval_tokens == 0 {
            let snapshot = snapshot_active_mapping(
                &mut state.executor,
                self.active_mapping
                    .as_ref()
                    .expect("prepared mapping exists"),
            )?;
            self.parent = Some(publish_block(
                &mut state,
                &self.profile,
                self.logical_context.expect("prepared context exists"),
                self.evaluated,
                self.parent,
                snapshot,
                self.next.as_ref().expect("decode result exists"),
            )?);
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
            .sample(self.next.as_ref().expect("decode owns next-token logits"))
            .map_err(state_error)?;
        let terminal_or_control = state.executor.token_is_eog(sampled);
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
            if self.evaluated % self.profile.publication_interval_tokens == 0 {
                let snapshot = snapshot_active_mapping(
                    &mut state.executor,
                    self.active_mapping
                        .as_ref()
                        .expect("prepared mapping exists"),
                )?;
                self.parent = Some(publish_block(
                    &mut state,
                    &self.profile,
                    self.logical_context.expect("prepared context exists"),
                    self.evaluated,
                    self.parent,
                    snapshot,
                    self.next.as_ref().expect("decode result exists"),
                )?);
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
            .activate_mapping(
                self.active_mapping
                    .as_ref()
                    .expect("prepared mapping exists"),
            )
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
                .and_then(|resident| resident.native.clone())
                .unwrap_or(
                    state
                        .executor
                        .active_representation()
                        .map_err(state_error)?,
                );
            if active != fallback {
                state
                    .executor
                    .activate_mapping(&fallback)
                    .map_err(state_error)?;
            }
        }
        let native_metrics = state.executor.mapping_metrics();
        state.metrics.requests += 1;
        state.metrics.cached_tokens += self.cached as u64;
        state.metrics.cache_hits += u64::from(self.cached != 0);
        state.metrics.reference_switches = native_metrics.reference_switches;
        state.metrics.fork_bytes_copied = native_metrics.fork_bytes_copied;
        state.metrics.export_bytes_copied = native_metrics.export_bytes_copied;
        state.metrics.import_bytes_copied = native_metrics.import_bytes_copied;
        state.metrics.total_bytes_copied = native_metrics.total_bytes_copied;
        state.metrics.graph_recaptures = native_metrics.graph_recaptures;
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
                .and_then(|resident| resident.native.clone())
                .unwrap_or_else(|| state.root.clone());
            if active != fallback {
                let _ = state.executor.activate_mapping(&fallback);
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
                "mapped execution admits only a resident process-owned model".into(),
            ));
        }
        let profile = self.profile;
        Ok(Box::new(MappedSession {
            profile,
            context_capacity: self.context_capacity,
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
            .activate_mapping(&state.root)
            .map_err(state_error)?;
        return Ok(());
    };
    let resident = state
        .resident
        .get(&prefix.id)
        .cloned()
        .ok_or_else(|| Error::State("logical mapping has no resident native mapping".into()))?;
    let restored = resident.native.is_none();
    let native = if let Some(native) = resident.native.clone() {
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
    let previous = state
        .executor
        .active_representation()
        .map_err(state_error)?;
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
        .map_err(state_error)?;
    for transfer in transfers {
        if let Err(error) = state.physical.complete_transfer(transfer, true) {
            let _ = state.physical.abort_transition(prepared);
            return Err(state_error(error));
        }
    }
    if let Err(error) = state.executor.activate_mapping(&native) {
        let _ = state.physical.abort_transition(prepared);
        return Err(state_error(error));
    }
    match state.physical.commit_mapped_transition(
        prepared,
        revision,
        u32::try_from(native.identity())
            .map_err(|_| Error::State("native representation identity exhausted".into()))?,
    ) {
        Ok(_) => {
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
            let _ = state.executor.activate_mapping(&previous);
            Err(state_error(error))
        }
    }
}

fn fork_for_request_branch(
    executor: &mut Executor,
    source: &RepresentationHandle,
) -> Result<RepresentationHandle, Error> {
    let prepared = executor.prepare_mapping_fork(source).map_err(state_error)?;
    let mapping = executor.commit_mapping(prepared).map_err(state_error)?;
    executor.activate_mapping(&mapping).map_err(state_error)?;
    Ok(mapping)
}

fn snapshot_active_mapping(
    executor: &mut Executor,
    active: &RepresentationHandle,
) -> Result<RepresentationHandle, Error> {
    let prepared = executor.prepare_mapping_fork(active).map_err(state_error)?;
    executor.commit_mapping(prepared).map_err(state_error)
}

fn publish_block(
    state: &mut MappedState,
    profile: &ExecutionProfile,
    context: LogicalContextId,
    represented_end: usize,
    parent: Option<EvaluatedPrefixId>,
    native: RepresentationHandle,
    continuation: &Decode,
) -> Result<EvaluatedPrefixId, Error> {
    let required = profile.required_mask();
    let serialized_bytes = state
        .executor
        .describe_representation(&native)
        .map_err(state_error)?
        .serialized_bytes;
    let component_count = profile.components().count();
    let bytes_per_component = serialized_bytes
        .checked_add(component_count.saturating_sub(1))
        .and_then(|bytes| bytes.checked_div(component_count))
        .ok_or_else(|| Error::State("representation size overflow".into()))?;
    let required_bytes = bytes_per_component
        .checked_mul(component_count)
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
        .and_then(|resident| resident.native.clone())
        .unwrap_or(
            state
                .executor
                .active_representation()
                .map_err(state_error)?,
        );
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
    state
        .logical
        .validate_publication(&prepared_publication)
        .map_err(state_error)?;
    let mapping_id = prepared_publication.mapping_id();
    let existing = state.resident.get(&mapping_id).cloned();
    let mut registered = Vec::new();
    let representations = if let Some(resident) = &existing {
        resident.representations.clone()
    } else {
        registered.reserve(component_count);
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
                    let _ = state.executor.activate_mapping(&rollback);
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
            let _ = state.executor.activate_mapping(&rollback);
            return Err(state_error(error));
        }
    };
    for transfer in transfers {
        if let Err(error) = state.physical.complete_transfer(transfer, true) {
            let _ = state.physical.abort_transition(transition);
            release_representations(&mut state.physical, &registered);
            let _ = state.executor.activate_mapping(&rollback);
            return Err(state_error(error));
        }
    }
    let published_native = existing
        .as_ref()
        .and_then(|resident| resident.native.clone())
        .unwrap_or_else(|| native.clone());
    if let Err(error) = state.physical.commit_mapped_transition(
        transition,
        expected_revision,
        u32::try_from(published_native.identity())
            .map_err(|_| Error::State("native representation identity exhausted".into()))?,
    ) {
        release_representations(&mut state.physical, &registered);
        let _ = state.executor.activate_mapping(&rollback);
        return Err(state_error(error));
    }
    let mapping = state
        .logical
        .commit_publication(prepared_publication)
        .expect("validated publication cannot conflict while mapped state is exclusively locked");
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
            family: "gemma4".into(),
            size_bytes: 1,
            block_count: 1,
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
            sampling: Default::default(),
            grammar: None,
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
    fn profile_uses_executor_reported_components_and_validates_interval() {
        let capabilities = Executor::open("mock://deterministic", 128, 0)
            .unwrap()
            .capabilities();
        let profile = ExecutionProfile::from_capabilities(&capabilities, 64).unwrap();
        assert_eq!(profile.publication_interval_tokens, 64);
        assert_eq!(
            profile.required_mask(),
            ComponentMask::GLOBAL_KV
                .union(ComponentMask::SWA)
                .union(ComponentMask::RECURRENT)
        );
        assert!(ExecutionProfile::from_capabilities(&capabilities, 0).is_err());
    }

    #[test]
    fn mapped_state_is_reused_without_activation_copying() {
        let engine = MappedEngine::open("mock://deterministic", 4096, 0, 1 << 30, 1 << 30).unwrap();
        let model = model();
        let first = engine
            .generate_collected(test_request(&model, "a".repeat(40), 2, &[]))
            .unwrap();
        let second = engine
            .generate_collected(test_request(&model, "z", 1, &first.successor_tokens))
            .unwrap();
        assert!(!second.pieces.is_empty());
        let metrics = engine.metrics();
        assert_eq!(metrics.requests, 2);
        assert_eq!(metrics.cache_hits, 1);
        assert!(metrics.cached_tokens >= 32);
        assert_eq!(metrics.graph_recaptures, None);
        assert!(metrics.reference_switches >= 2);
    }

    #[test]
    fn unrelated_requests_start_from_the_root_mapping() {
        let shared = MappedEngine::open("mock://deterministic", 4096, 0, 1 << 30, 1 << 30).unwrap();
        let fresh = MappedEngine::open("mock://deterministic", 4096, 0, 1 << 30, 1 << 30).unwrap();
        let model = model();
        shared
            .generate_collected(test_request(&model, "first unrelated prompt", 2, &[]))
            .unwrap();
        let reused = shared
            .generate_collected(test_request(&model, "second prompt", 2, &[]))
            .unwrap();
        let baseline = fresh
            .generate_collected(test_request(&model, "second prompt", 2, &[]))
            .unwrap();
        assert_eq!(reused.pieces, baseline.pieces);
        assert_eq!(reused.successor_tokens, baseline.successor_tokens);
        assert_eq!(reused.cached_tokens, 0);
    }

    #[test]
    fn repeated_identical_requests_reuse_an_immutable_prefix() {
        let engine = MappedEngine::open("mock://deterministic", 4096, 0, 1 << 30, 1 << 30).unwrap();
        let model = model();
        let prompt = "tool schema and request payload ".repeat(96);
        let cold = engine
            .generate_collected(test_request(&model, &prompt, 8, &[]))
            .unwrap();
        let mut expected_cached = None;
        for _ in 0..3 {
            let reused = engine
                .generate_collected(test_request(&model, &prompt, 8, &[]))
                .unwrap();
            assert_eq!(reused.pieces, cold.pieces);
            assert_eq!(reused.successor_tokens, cold.successor_tokens);
            assert!(reused.cached_tokens > 0);
            assert_eq!(
                *expected_cached.get_or_insert(reused.cached_tokens),
                reused.cached_tokens
            );
        }
    }

    #[test]
    #[ignore = "requires the pinned external Gemma GGUF"]
    fn repeated_real_model_requests_reuse_an_immutable_prefix() {
        let model_path =
            std::env::var("CUSCO_TEST_MODEL").expect("CUSCO_TEST_MODEL must name the Gemma GGUF");
        let engine = MappedEngine::open(&model_path, 8192, 99, 1 << 40, 1 << 40).unwrap();
        let model = ModelRecord {
            path: PathBuf::from(&model_path),
            ..model()
        };
        let prompt = "tool schema and request payload ".repeat(768);
        let cold = engine
            .generate_collected(test_request(&model, &prompt, 8, &[]))
            .unwrap();
        let mut expected_cached = None;
        for _ in 0..3 {
            let reused = engine
                .generate_collected(test_request(&model, &prompt, 8, &[]))
                .unwrap();
            assert_eq!(reused.pieces, cold.pieces);
            assert_eq!(reused.successor_tokens, cold.successor_tokens);
            assert!(reused.cached_tokens > 0);
            assert_eq!(
                *expected_cached.get_or_insert(reused.cached_tokens),
                reused.cached_tokens
            );
        }
    }

    #[test]
    fn dropping_unpublished_session_reactivates_root_mapping() {
        let engine = MappedEngine::open("mock://deterministic", 4096, 0, 1 << 30, 1 << 30).unwrap();
        let root_identity = engine.state.lock().root.identity();
        let mut session = engine
            .start_session(test_request(&model(), "cancel before publication", 2, &[]))
            .unwrap();
        assert!(matches!(session.step().unwrap(), SessionStep::Progress(_)));
        drop(session);
        let state = engine.state.lock();
        assert_eq!(
            state.executor.mapping_metrics().active_identity,
            root_identity
        );
    }

    #[test]
    fn linear_publication_snapshots_do_not_switch_the_active_branch() {
        let engine = MappedEngine::open("mock://deterministic", 4096, 0, 1 << 30, 1 << 30).unwrap();
        engine
            .generate_collected(test_request(&model(), "a".repeat(160), 2, &[]))
            .unwrap();
        let metrics = engine.metrics();
        assert!(metrics.published_blocks >= 5);
        assert_eq!(metrics.reference_switches, 2);
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
            "mock://deterministic",
            4096,
            0,
            1 << 30,
            1 << 30,
            ModelEpoch(1),
            Some(spill_dir.clone()),
            1 << 20,
            DEFAULT_PUBLICATION_INTERVAL_TOKENS,
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
            "mock://deterministic",
            4096,
            0,
            1 << 30,
            1 << 30,
            ModelEpoch(1),
            Some(spill_dir.clone()),
            1 << 20,
            DEFAULT_PUBLICATION_INTERVAL_TOKENS,
        )
        .unwrap();
        let model = model();
        let first = engine
            .generate_collected(test_request(&model, "a".repeat(40), 2, &[]))
            .unwrap();
        engine
            .generate_collected(test_request(&model, "b".repeat(40), 1, &[]))
            .unwrap();
        let control =
            MappedEngine::open("mock://deterministic", 4096, 0, 1 << 30, 1 << 30).unwrap();
        let control_first = control
            .generate_collected(test_request(&model, "a".repeat(40), 2, &[]))
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
        let engine = MappedEngine::open("mock://deterministic", 64, 0, 1 << 20, 1 << 20).unwrap();
        let model = model();
        let error = engine
            .generate_collected(test_request(&model, "x".repeat(65), 1, &[]))
            .unwrap_err();
        assert!(error.to_string().contains("context capacity"));
        assert_eq!(engine.metrics(), MappedMetrics::default());
    }

    #[test]
    fn publication_failure_keeps_the_prior_mapping_reusable() {
        let engine = MappedEngine::open("mock://deterministic", 4096, 0, 250_000, 1 << 20).unwrap();
        let model = model();
        let first = engine
            .generate_collected(test_request(&model, "a".repeat(40), 1, &[]))
            .unwrap();
        let failed = engine.generate_collected(test_request(
            &model,
            "b".repeat(25),
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
        let engine = MappedEngine::open("mock://deterministic", 4096, 0, 1 << 30, 1 << 30).unwrap();
        let model = model();
        let primed = engine
            .generate_collected(test_request(&model, "p".repeat(32), 0, &[]))
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
        let engine = MappedEngine::open("mock://deterministic", 4096, 0, 1 << 30, 1 << 30).unwrap();
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
                    compaction: None,
                    deadline_ms: None,
                    scheduling: SchedulingMetadata::default(),
                    stop: vec![],
                    raw_continuation: false,
                    sampling: Default::default(),
                    grammar: None,
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
                    compaction: None,
                    context_id: Some(durable.id),
                    deadline_ms: None,
                    scheduling: SchedulingMetadata::default(),
                    stop: vec![],
                    raw_continuation: false,
                    sampling: Default::default(),
                    grammar: None,
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
