use crate::{
    EngineOutput, EngineRequest, Error, FrontierControl, InferenceEngine, PrefillMetrics, TokenSink,
};
use cusco_context_store::{
    AdapterEpoch, ComponentMask, ContextStore, EvaluatedPrefixId, LogicalContextId, ModelEpoch,
    PersistentTokenSequence,
};
use cusco_executor::{Decode, Executor, MappingId};
use cusco_physical_manager::{
    Capacity, Component, PhysicalManager, PhysicalRepresentationId, Tier,
};
use parking_lot::Mutex;
use serde::Deserialize;
use std::{collections::HashMap, path::Path, sync::Arc, time::Instant};

const MODEL_EPOCH: ModelEpoch = ModelEpoch(1);
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

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
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
    native: MappingId,
    representations: Vec<PhysicalRepresentationId>,
    continuation: Decode,
}

struct MappedState {
    executor: Executor,
    logical: ContextStore,
    physical: PhysicalManager,
    resident: HashMap<EvaluatedPrefixId, ResidentMapping>,
    metrics: MappedMetrics,
}

pub struct MappedEngine {
    profile: ExecutionProfile,
    model_path: String,
    context_capacity: usize,
    state: Mutex<MappedState>,
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
            state: Mutex::new(MappedState {
                executor,
                logical: ContextStore::default(),
                physical: PhysicalManager::new(Capacity {
                    device_bytes,
                    host_bytes,
                }),
                resident: HashMap::new(),
                metrics: MappedMetrics::default(),
            }),
        }))
    }

    pub fn metrics(&self) -> MappedMetrics {
        self.state.lock().metrics
    }

    pub fn profile(&self) -> &ExecutionProfile {
        &self.profile
    }

    pub fn model_path(&self) -> &str {
        &self.model_path
    }
}

impl InferenceEngine for MappedEngine {
    fn generate(
        &self,
        request: EngineRequest<'_>,
        sink: &mut TokenSink<'_>,
    ) -> Result<EngineOutput, Error> {
        if request.model.path.to_str() != Some(self.model_path()) {
            return Err(Error::State(
                "Phase 6A admits only the process-owned model".into(),
            ));
        }
        let prompt_started = Instant::now();
        let mut state = self.state.lock();
        let tokenization_started = Instant::now();
        let prompt = state
            .executor
            .tokenize(request.prompt)
            .map_err(state_error)?;
        let tokenization_ns = elapsed_ns(tokenization_started);
        let input_tokens = prompt.len();
        let total = request
            .prior_tokens
            .len()
            .checked_add(prompt.len())
            .and_then(|value| value.checked_add(request.max_tokens))
            .ok_or_else(|| Error::State("request token count overflow".into()))?;
        if total > self.context_capacity {
            return Err(Error::State(
                "request exceeds model context capacity".into(),
            ));
        }

        let mut tokens = request.prior_tokens.to_vec();
        tokens.extend_from_slice(&prompt);
        let sequence = PersistentTokenSequence::default().append(&tokens);
        let logical_context = state.logical.create(sequence, MODEL_EPOCH, ADAPTER_EPOCH);
        let prefix_lookup_started = Instant::now();
        let prefix = state
            .logical
            .longest_valid_prefix(logical_context)
            .map_err(state_error)?;
        let prefix_lookup_ns = elapsed_ns(prefix_lookup_started);
        let cached = prefix.as_ref().map_or(0, |mapping| mapping.represented_end);
        let evaluated_tokens = tokens.len().saturating_sub(cached);
        let mapping_activation_started = Instant::now();
        activate_prefix(&mut state, logical_context, prefix.as_deref())?;

        let mut active_mapping = state.executor.mapping_metrics().active;
        if cached == 0 {
            active_mapping = fork_and_activate(&mut state.executor, active_mapping)?;
        }
        let mapping_activation_ns = elapsed_ns(mapping_activation_started);
        let uncached_prefill_started = Instant::now();
        let mut evaluated = cached;
        let mut parent = prefix.as_ref().map(|mapping| mapping.id);
        let mut next = prefix
            .as_ref()
            .and_then(|mapping| state.resident.get(&mapping.id))
            .map(|resident| resident.continuation.clone());
        while evaluated < tokens.len() {
            let end = tokens
                .len()
                .min(((evaluated / self.profile.block_size) + 1) * self.profile.block_size);
            next = Some(
                state
                    .executor
                    .decode(&tokens[evaluated..end])
                    .map_err(state_error)?,
            );
            state.metrics.decoded_tokens += (end - evaluated) as u64;
            evaluated = end;
            if evaluated % self.profile.block_size == 0 {
                parent = Some(publish_block(
                    &mut state,
                    &self.profile,
                    logical_context,
                    evaluated,
                    parent,
                    active_mapping,
                    next.as_ref().expect("decode result exists"),
                )?);
                active_mapping = fork_and_activate(&mut state.executor, active_mapping)?;
            }
        }
        let uncached_prefill_ns = elapsed_ns(uncached_prefill_started);
        if request.max_tokens > 0 && next.is_none() {
            return Err(Error::State(
                "an exact cached prefix cannot supply uncached logits".into(),
            ));
        }
        let prefill = PrefillMetrics {
            total_tokens: tokens.len(),
            cached_tokens: cached,
            uncached_tokens: evaluated_tokens,
            tokenization_ns,
            prefix_lookup_ns,
            mapping_activation_ns,
            uncached_prefill_ns,
            total_ns: elapsed_ns(prompt_started),
        };
        let mut sampler = state.executor.greedy_sampler().map_err(state_error)?;
        let mut piece = Vec::with_capacity(32);
        for index in 0..request.max_tokens {
            let sampled = sampler.sample(&mut state.executor).map_err(state_error)?;
            let terminal_or_control = self.profile.terminal_tokens.contains(&sampled);
            if terminal_or_control {
                piece.clear();
            } else {
                state
                    .executor
                    .render_token(sampled, &mut piece)
                    .map_err(state_error)?;
            }
            tokens.push(sampled);
            state
                .logical
                .append(logical_context, &[sampled])
                .map_err(state_error)?;
            let control = sink(sampled, &piece, terminal_or_control)?;
            if control == FrontierControl::Stop
                || terminal_or_control
                || index + 1 == request.max_tokens
            {
                break;
            }
            next = Some(state.executor.decode(&[sampled]).map_err(state_error)?);
            state.metrics.decoded_tokens += 1;
            evaluated += 1;
            if evaluated % self.profile.block_size == 0 {
                parent = Some(publish_block(
                    &mut state,
                    &self.profile,
                    logical_context,
                    evaluated,
                    parent,
                    active_mapping,
                    next.as_ref().expect("decode result exists"),
                )?);
                active_mapping = fork_and_activate(&mut state.executor, active_mapping)?;
            }
        }
        let native_metrics = state.executor.mapping_metrics();
        state.metrics.requests += 1;
        state.metrics.cached_tokens += cached as u64;
        state.metrics.cache_hits += u64::from(cached != 0);
        state.metrics.reference_switches = native_metrics.reference_switches;
        state.metrics.activation_bytes_copied = native_metrics.activation_bytes_copied;
        Ok(EngineOutput {
            successor_tokens: tokens,
            input_tokens,
            cached_tokens: cached,
            evaluated_tokens,
            prefill,
        })
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
        .map_err(state_error)?;
    for transfer in transfers {
        if let Err(error) = state.physical.complete_transfer(transfer, true) {
            let _ = state.physical.abort_transition(prepared);
            return Err(state_error(error));
        }
    }
    if let Err(error) = state.executor.activate_mapping(resident.native) {
        let _ = state.physical.abort_transition(prepared);
        return Err(state_error(error));
    }
    match state.physical.commit_transition(prepared, revision) {
        Ok(binding) => {
            state
                .physical
                .publish_device_block_table(EXECUTION_SLOT, binding, resident.native.0)
                .map_err(state_error)?;
            Ok(())
        }
        Err(error) => {
            let _ = state.executor.activate_mapping(previous);
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
        .map_or(MappingId(0), |resident| resident.native);
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
    let published_native = existing.as_ref().map_or(native, |resident| resident.native);
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
                native,
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

fn state_error(error: impl std::fmt::Display) -> Error {
    Error::State(error.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ModelRecord;
    use std::path::PathBuf;

    fn model() -> ModelRecord {
        ModelRecord {
            id: "gemma".into(),
            revision: "test".into(),
            path: PathBuf::from("mock://deterministic"),
            sha256: "mock".into(),
            aliases: vec![],
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
        fn generate_collected(&self, request: EngineRequest<'_>) -> Result<CollectedOutput, Error>;
    }
    impl GenerateCollected for MappedEngine {
        fn generate_collected(&self, request: EngineRequest<'_>) -> Result<CollectedOutput, Error> {
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
        let engine = MappedEngine::open(
            "gemma-4-e2b-it",
            "mock://deterministic",
            4096,
            0,
            1 << 30,
            1 << 30,
        )
        .unwrap();
        let model = model();
        let first = engine
            .generate_collected(EngineRequest {
                model: &model,
                prompt: &"a".repeat(40),
                max_tokens: 2,
                prior_tokens: &[],
            })
            .unwrap();
        let second = engine
            .generate_collected(EngineRequest {
                model: &model,
                prompt: "z",
                max_tokens: 1,
                prior_tokens: &first.successor_tokens,
            })
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
    fn capacity_rejection_precedes_decode_and_preserves_metrics() {
        let engine = MappedEngine::open(
            "gemma-4-e2b-it",
            "mock://deterministic",
            64,
            0,
            1 << 20,
            1 << 20,
        )
        .unwrap();
        let model = model();
        let error = engine
            .generate_collected(EngineRequest {
                model: &model,
                prompt: &"x".repeat(65),
                max_tokens: 1,
                prior_tokens: &[],
            })
            .unwrap_err();
        assert!(error.to_string().contains("context capacity"));
        assert_eq!(engine.metrics(), MappedMetrics::default());
    }

    #[test]
    fn publication_failure_keeps_the_prior_mapping_reusable() {
        let engine = MappedEngine::open(
            "gemma-4-e2b-it",
            "mock://deterministic",
            4096,
            0,
            250_000,
            1 << 20,
        )
        .unwrap();
        let model = model();
        let first = engine
            .generate_collected(EngineRequest {
                model: &model,
                prompt: &"a".repeat(40),
                max_tokens: 1,
                prior_tokens: &[],
            })
            .unwrap();
        let failed = engine.generate_collected(EngineRequest {
            model: &model,
            prompt: &"b".repeat(25),
            max_tokens: 1,
            prior_tokens: &first.successor_tokens,
        });
        assert!(failed.unwrap_err().to_string().contains("cannot publish"));
        let resumed = engine
            .generate_collected(EngineRequest {
                model: &model,
                prompt: "c",
                max_tokens: 1,
                prior_tokens: &first.successor_tokens,
            })
            .unwrap();
        assert_eq!(resumed.pieces.len(), 1);
        assert_eq!(engine.metrics().requests, 2);
        assert!(engine.metrics().cache_hits >= 1);
    }

    #[test]
    fn exact_cached_prefix_resumes_from_its_saved_logits() {
        let engine = MappedEngine::open(
            "gemma-4-e2b-it",
            "mock://deterministic",
            4096,
            0,
            1 << 30,
            1 << 30,
        )
        .unwrap();
        let model = model();
        let primed = engine
            .generate_collected(EngineRequest {
                model: &model,
                prompt: &"p".repeat(32),
                max_tokens: 0,
                prior_tokens: &[],
            })
            .unwrap();
        let resumed = engine
            .generate_collected(EngineRequest {
                model: &model,
                prompt: "",
                max_tokens: 1,
                prior_tokens: &primed.successor_tokens,
            })
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
        let engine = MappedEngine::open(
            "gemma-4-e2b-it",
            "mock://deterministic",
            4096,
            0,
            1 << 30,
            1 << 30,
        )
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
                    priority: 0,
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
                    priority: 0,
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
        assert!(
            serde_json::to_value(&second.usage).unwrap()["prefill"]["total_ns"].is_u64()
        );
        std::fs::remove_dir_all(directory).unwrap();
    }
}
