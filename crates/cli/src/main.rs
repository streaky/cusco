use anyhow::{Context, Result, ensure};
use clap::{Parser, Subcommand, ValueEnum};
use cusco_executor::{Executor, logits_identical};
use cusco_model_registry::{GEMMA_URI, ModelRecord, fetch_hf, register_local};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use std::{
    fs,
    path::PathBuf,
    sync::{Arc, Barrier, Mutex},
    thread,
    time::{Duration, Instant},
};

#[derive(Parser)]
struct Args {
    #[command(subcommand)]
    command: Command,
}
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, ValueEnum)]
enum HttpDebugLevelArg {
    #[default]
    Off,
    Safe,
    Full,
}

#[derive(Subcommand)]
enum Command {
    Fetch {
        #[arg(default_value=GEMMA_URI)]
        uri: String,
        #[arg(long, default_value = "/models/cache")]
        cache: PathBuf,
        #[arg(long)]
        sha256: Option<String>,
    },
    Register {
        path: PathBuf,
        #[arg(long)]
        sha256: Option<String>,
    },
    Proof {
        model: PathBuf,
        #[arg(long, required_unless_present = "allow_unverified_model")]
        sha256: Option<String>,
        #[arg(long, hide = true)]
        allow_unverified_model: bool,
        #[arg(long, default_value_t = 4096)]
        context: u32,
        #[arg(long, default_value_t = 99)]
        gpu_layers: i32,
        #[arg(long, default_value = "The capital of France is")]
        prefix: String,
        #[arg(long, default_value = "Unrelated replacement context")]
        replacement: String,
        #[arg(long, default_value = "/results/phase1.json")]
        output: PathBuf,
    },
    MappedProof {
        model: PathBuf,
        #[arg(long, default_value_t = 4096)]
        context: u32,
        #[arg(long, default_value_t = 99)]
        gpu_layers: i32,
        #[arg(
            long,
            default_value = "Mapped execution proves reference-only branch switching"
        )]
        prefix: String,
        #[arg(long, default_value = "/results/phase5.json")]
        output: PathBuf,
    },
    /// Run the versioned real-model Phase 8 scheduler acceptance workload.
    SchedulerProof {
        model: PathBuf,
        #[arg(long, default_value = "/work/config/phase8-workload.json")]
        workload: PathBuf,
        #[arg(long, default_value = "/results/phase8-server.json")]
        output: PathBuf,
        #[arg(long, default_value_t = 4096)]
        context: u32,
        #[arg(long, default_value_t = 99)]
        gpu_layers: i32,
        #[arg(long, default_value_t = 8_589_934_592)]
        device_bytes: usize,
        #[arg(long, default_value_t = 17_179_869_184)]
        host_bytes: usize,
    },
    Serve {
        /// Versioned daemon configuration.
        #[arg(long, default_value = "/data/config.yaml")]
        config: PathBuf,
    },
}
fn main() -> Result<()> {
    run(Args::parse().command)
}
fn run(command: Command) -> Result<()> {
    match command {
        Command::Fetch { uri, cache, sha256 } => println!(
            "{}",
            serde_json::to_string_pretty(&fetch_hf(&uri, &cache, sha256.as_deref())?)?
        ),
        Command::Register { path, sha256 } => println!(
            "{}",
            serde_json::to_string_pretty(&register_local(path, "local", sha256.as_deref())?)?
        ),
        Command::Proof {
            model,
            sha256,
            allow_unverified_model,
            context,
            gpu_layers,
            prefix,
            replacement,
            output,
        } => proof(
            model,
            sha256.as_deref(),
            allow_unverified_model,
            context,
            gpu_layers,
            &prefix,
            &replacement,
            output,
        )?,
        Command::MappedProof {
            model,
            context,
            gpu_layers,
            prefix,
            output,
        } => mapped_proof(model, context, gpu_layers, &prefix, output)?,
        Command::SchedulerProof {
            model,
            workload,
            output,
            context,
            gpu_layers,
            device_bytes,
            host_bytes,
        } => scheduler_proof(
            model,
            workload,
            output,
            context,
            gpu_layers,
            device_bytes,
            host_bytes,
        )?,
        Command::Serve { config } => {
            use cusco_server::{
                AnonymousAdmin, AuthProvider, BearerAuth, DaemonConfig, HttpDebugLevel,
                ModelCatalog, ModelRecord, ResidencyConfig, ResidentEngine, Server,
                WorkloadScheduler, load_user_models,
            };
            let config = DaemonConfig::load(config)?;
            let bearer_token = std::env::var("CUSCO_BEARER_TOKEN")
                .ok()
                .or_else(|| config.bearer_token.clone());
            let anonymous = bearer_token.is_none();
            let auth: Arc<dyn AuthProvider> = match bearer_token {
                Some(token) => Arc::new(BearerAuth::new(token)),
                None => Arc::new(AnonymousAdmin),
            };
            let engine = ResidentEngine::open_with_spill(
                ResidencyConfig {
                    device_bytes: config.execution.device_capacity.0,
                    host_bytes: config.execution.host_capacity.0,
                    storage_bytes: config.execution.storage_capacity.0,
                    context_reserve_bytes: config.execution.context_reserve.0,
                    n_ctx: config.execution.context_tokens,
                    gpu_layers: config.execution.gpu_layers,
                    require_competent: config.execution.require_competent,
                },
                &config.paths.spill,
            )?;
            let engine = WorkloadScheduler::new(engine, config.scheduler)?;
            let transient_state = config.paths.database.with_extension("runtime.json");
            if transient_state.exists() {
                fs::remove_file(&transient_state).with_context(|| {
                    format!("discard transient state {}", transient_state.display())
                })?;
            }
            let server = Server::open(&transient_state, auth, engine)?;
            server.configure(config.server)?;
            let catalog = ModelCatalog::open(&config.paths.database)?;
            server.attach_catalog(catalog.clone(), config.paths.models.clone());
            for model in catalog.models()? {
                server.register_model(model)?;
            }
            let declared = load_user_models(&config.paths.user_config, &config.paths.user_models)?;
            for declaration in declared.models {
                let registered = register_local(
                    &declaration.path,
                    &declaration.name,
                    declaration.sha256.as_deref(),
                )?;
                let metadata = cusco_model_registry::probe_gguf(&registered.path)?;
                let model = server.register_model(ModelRecord {
                    id: declaration.name,
                    revision: registered.sha256.clone(),
                    path: registered.path,
                    sha256: registered.sha256,
                    aliases: declaration.aliases,
                    family: metadata.architecture,
                    size_bytes: registered.size,
                    epoch: 0,
                })?;
                catalog.publish(&model)?;
            }
            let runtime = tokio::runtime::Runtime::new()?;
            match config.http_debug {
                HttpDebugLevel::Off => runtime.block_on(cusco_server::serve(
                    server,
                    config.listen,
                    anonymous,
                    config.unsafe_public_unauthenticated,
                ))?,
                level => runtime.block_on(cusco_server::serve_with_http_debug(
                    server,
                    config.listen,
                    anonymous,
                    config.unsafe_public_unauthenticated,
                    level,
                ))?,
            }
        }
    }
    Ok(())
}
#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct SchedulerProofWorkload {
    version: u32,
    name: String,
    model_family: String,
    policy: cusco_server::SchedulerPolicyConfig,
    baseline: SchedulerProofCase,
    mixed: Vec<SchedulerProofCase>,
    thresholds: SchedulerProofThresholds,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct SchedulerProofCase {
    id: String,
    principal: String,
    class: cusco_server::SchedulingClass,
    prompt: String,
    prompt_repetitions: usize,
    max_tokens: usize,
    arrival_delay_ms: u64,
    expected: SchedulerProofOutcome,
}

#[derive(Clone, Copy, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
enum SchedulerProofOutcome {
    Complete,
    Cancel,
    Deadline,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct SchedulerProofThresholds {
    max_queue_age_rounds: u64,
    max_first_event_baseline_multiplier: u128,
    max_first_event_additive_ms: u128,
    max_quantum_baseline_multiplier: u128,
    max_quantum_additive_ms: u128,
}

#[derive(Debug, Serialize)]
struct SchedulerProofResult {
    id: String,
    principal: String,
    class: cusco_server::SchedulingClass,
    expected: SchedulerProofOutcome,
    observed: String,
    first_event_ms: Option<u128>,
    total_ms: u128,
    token_wait_ms: Vec<u128>,
}

#[allow(clippy::too_many_arguments)]
fn scheduler_proof(
    model_path: PathBuf,
    workload_path: PathBuf,
    output: PathBuf,
    n_ctx: u32,
    gpu_layers: i32,
    device_bytes: usize,
    host_bytes: usize,
) -> Result<()> {
    use cusco_server::{MappedEngine, WorkloadScheduler};

    let proof_started = Instant::now();
    let workload_bytes = fs::read(&workload_path)
        .with_context(|| format!("read workload {}", workload_path.display()))?;
    let workload: SchedulerProofWorkload =
        serde_json::from_slice(&workload_bytes).context("parse scheduler workload")?;
    ensure!(
        matches!(workload.version, 1 | 2),
        "unsupported scheduler workload version"
    );
    ensure!(!workload.mixed.is_empty(), "mixed workload is empty");
    ensure!(
        workload
            .mixed
            .iter()
            .any(|case| case.expected == SchedulerProofOutcome::Cancel),
        "mixed workload has no cancellation injection"
    );
    ensure!(
        workload
            .mixed
            .iter()
            .any(|case| case.expected == SchedulerProofOutcome::Deadline),
        "mixed workload has no deadline injection"
    );

    let model_size = if model_path.to_string_lossy().starts_with("mock://") {
        1
    } else {
        fs::metadata(&model_path)
            .with_context(|| format!("stat model {}", model_path.display()))?
            .len()
    };
    let engine = MappedEngine::open(
        &workload.model_family,
        &model_path,
        n_ctx,
        gpu_layers,
        device_bytes,
        host_bytes,
    )
    .map_err(|error| anyhow::anyhow!(error.to_string()))?;
    let model = cusco_server::ModelRecord {
        id: "phase8-proof-model".into(),
        revision: "proof".into(),
        path: model_path.clone(),
        sha256: "verified-by-proof-wrapper".into(),
        aliases: Vec::new(),
        family: workload.model_family.clone(),
        size_bytes: model_size,
        epoch: 1,
    };
    let diagnostics = Arc::new(Mutex::new(Vec::<Value>::new()));
    let diagnostic_rows = diagnostics.clone();
    let scheduler = WorkloadScheduler::new_with_diagnostics(
        engine.clone(),
        workload.policy,
        Some(move |line: &str| {
            if let Ok(value) = serde_json::from_str(line) {
                diagnostic_rows.lock().expect("diagnostic lock").push(value);
            }
        }),
    )
    .map_err(|error| anyhow::anyhow!(error.to_string()))?;

    let baseline = run_scheduler_proof_case(
        scheduler.clone(),
        model.clone(),
        workload.baseline.clone(),
        None,
    );
    ensure!(
        baseline.observed == "complete",
        "isolated baseline did not complete"
    );
    let baseline_first = baseline
        .first_event_ms
        .context("isolated baseline emitted no token")?;
    let baseline_quantum = percentile(&baseline.token_wait_ms, 95).max(1);

    let barrier = Arc::new(Barrier::new(workload.mixed.len() + 1));
    let mut workers = Vec::with_capacity(workload.mixed.len());
    for case in workload.mixed.clone() {
        let scheduler = scheduler.clone();
        let model = model.clone();
        let barrier = barrier.clone();
        workers.push(thread::spawn(move || {
            barrier.wait();
            if case.arrival_delay_ms > 0 {
                thread::sleep(Duration::from_millis(case.arrival_delay_ms));
            }
            run_scheduler_proof_case(scheduler, model, case, None)
        }));
    }
    barrier.wait();
    let mixed = workers
        .into_iter()
        .map(|worker| worker.join().expect("scheduler proof worker panicked"))
        .collect::<Vec<_>>();

    ensure!(
        wait_for_scheduler_idle(&scheduler, Duration::from_secs(5)),
        "scheduler did not become idle after the mixed workload"
    );
    let before_recovery_status = scheduler.status();
    let before_recovery_metrics = engine.metrics();
    let recovery_case = SchedulerProofCase {
        id: "capacity-recovery".into(),
        principal: "recovery".into(),
        class: cusco_server::SchedulingClass::Interactive,
        prompt: "Capacity recovered.".into(),
        prompt_repetitions: 1,
        max_tokens: 2,
        arrival_delay_ms: 0,
        expected: SchedulerProofOutcome::Complete,
    };
    let recovery = run_scheduler_proof_case(scheduler.clone(), model, recovery_case, None);
    ensure!(
        wait_for_scheduler_idle(&scheduler, Duration::from_secs(5)),
        "scheduler did not become idle after the recovery request"
    );
    ensure!(
        scheduler.flush_diagnostics(Duration::from_secs(5)),
        "scheduler diagnostic sink did not drain"
    );
    let status = scheduler.status();
    let after_recovery_metrics = engine.metrics();
    let diagnostic_rows = diagnostics.lock().expect("diagnostic lock").clone();
    let decisions = diagnostic_rows
        .iter()
        .filter(|row| row["type"] == "scheduler_decision")
        .cloned()
        .collect::<Vec<_>>();
    let max_queue_age_rounds = decisions
        .iter()
        .filter_map(|row| row["queue_age_rounds"].as_u64())
        .max()
        .unwrap_or(0);
    let mixed_first = mixed
        .iter()
        .filter_map(|result| result.first_event_ms)
        .collect::<Vec<_>>();
    let mixed_quantums = mixed
        .iter()
        .flat_map(|result| result.token_wait_ms.iter().copied())
        .collect::<Vec<_>>();
    let p95_first_event_ms = percentile(&mixed_first, 95);
    let p95_token_wait_ms = percentile(&mixed_quantums, 95);
    let first_event_limit_ms = baseline_first
        .saturating_mul(workload.thresholds.max_first_event_baseline_multiplier)
        .saturating_add(workload.thresholds.max_first_event_additive_ms);
    let quantum_limit_ms = baseline_quantum
        .saturating_mul(workload.thresholds.max_quantum_baseline_multiplier)
        .saturating_add(workload.thresholds.max_quantum_additive_ms);
    let outcomes_match = mixed.iter().all(|result| {
        result.observed
            == match result.expected {
                SchedulerProofOutcome::Complete => "complete",
                SchedulerProofOutcome::Cancel => "cancelled",
                SchedulerProofOutcome::Deadline => "deadline",
            }
    });
    let capacity_recovered = recovery.observed == "complete"
        && before_recovery_status.metrics.runnable == 0
        && before_recovery_status.metrics.waiting_for_consumer == 0
        && status.metrics.runnable == 0
        && status.metrics.waiting_for_consumer == 0
        && after_recovery_metrics.requests == before_recovery_metrics.requests.saturating_add(1);
    let gates = json!({
        "outcomes_match": outcomes_match,
        "capacity_recovered": capacity_recovered,
        "diagnostics_lossless": status.metrics.diagnostic_records_lost == 0
            && status.metrics.diagnostic_records == status.metrics.diagnostic_records_delivered,
        "starvation_round_bound": max_queue_age_rounds <= workload.thresholds.max_queue_age_rounds,
        "first_event_latency_bound": p95_first_event_ms <= first_event_limit_ms,
        "quantum_latency_bound": p95_token_wait_ms <= quantum_limit_ms,
    });
    let passed = gates
        .as_object()
        .expect("gates object")
        .values()
        .all(|value| value == &Value::Bool(true));
    let artifact = json!({
        "phase": "8",
        "passed": passed,
        "workload": workload,
        "provenance": {
            "model_path": model_path,
            "model_bytes": model_size,
            "context": n_ctx,
            "gpu_layers": gpu_layers,
            "device_bytes": device_bytes,
            "host_bytes": host_bytes,
        },
        "baseline": baseline,
        "mixed": mixed,
        "capacity_recovery": recovery,
        "capacity_recovery_evidence": {
            "before": {
                "scheduler": before_recovery_status,
                "mapped_metrics": before_recovery_metrics,
            },
            "after": {
                "scheduler": status.clone(),
                "mapped_metrics": after_recovery_metrics,
            },
        },
        "measurements": {
            "max_queue_age_rounds": max_queue_age_rounds,
            "p95_first_event_ms": p95_first_event_ms,
            "p95_token_wait_ms": p95_token_wait_ms,
            "first_event_limit_ms": first_event_limit_ms,
            "quantum_limit_ms": quantum_limit_ms,
            "elapsed_ms": proof_started.elapsed().as_millis(),
        },
        "scheduler": status,
        "mapped_metrics": engine.metrics(),
        "diagnostics": diagnostic_rows,
        "gates": gates,
    });
    if let Some(parent) = output.parent() {
        fs::create_dir_all(parent)?;
    }
    fs::write(&output, serde_json::to_vec_pretty(&artifact)?)?;
    ensure!(passed, "Phase 8 scheduler proof gates failed");
    println!("{}", output.display());
    Ok(())
}

fn run_scheduler_proof_case(
    scheduler: Arc<cusco_server::WorkloadScheduler>,
    model: cusco_server::ModelRecord,
    case: SchedulerProofCase,
    start_barrier: Option<Arc<Barrier>>,
) -> SchedulerProofResult {
    use cusco_server::{
        EngineRequest, Error as ServerError, FrontierControl, InferenceEngine, RequestControl,
        SchedulingMetadata,
    };

    if let Some(barrier) = start_barrier {
        barrier.wait();
    }
    let control = Arc::new(RequestControl::new());
    let sink_control = control.clone();
    let expected = case.expected;
    let started = Instant::now();
    let mut last_token = started;
    let mut first_event_ms = None;
    let mut token_wait_ms = Vec::new();
    let prompt = case.prompt.repeat(case.prompt_repetitions);
    let result = scheduler.generate(
        EngineRequest {
            model,
            prompt,
            max_tokens: case.max_tokens,
            prior_tokens: Vec::new(),
            control,
            scheduling: SchedulingMetadata {
                class: case.class,
                source: cusco_server::PrioritySource::ControlledWorkload,
                principal: case.principal.clone(),
                correlation_id: format!("phase8-transport-{}", case.id),
                inference_id: format!("phase8-inference-{}", case.id),
            },
            prefill_chunk_tokens: scheduler.status().policy.prefill_tokens,
        },
        &mut |_, _, _| {
            let now = Instant::now();
            if first_event_ms.is_some() {
                token_wait_ms.push(now.duration_since(last_token).as_millis());
            } else {
                first_event_ms = Some(now.duration_since(started).as_millis());
            }
            last_token = now;
            match expected {
                SchedulerProofOutcome::Complete => {}
                SchedulerProofOutcome::Cancel => sink_control.cancel(),
                SchedulerProofOutcome::Deadline => sink_control.expire(),
            }
            Ok(FrontierControl::Continue)
        },
    );
    let observed = match result {
        Ok(_) => "complete",
        Err(ServerError::Cancelled) => "cancelled",
        Err(ServerError::Deadline) => "deadline",
        Err(_) => "failed",
    }
    .to_owned();
    SchedulerProofResult {
        id: case.id,
        principal: case.principal,
        class: case.class,
        expected,
        observed,

        first_event_ms,
        total_ms: started.elapsed().as_millis(),
        token_wait_ms,
    }
}
fn wait_for_scheduler_idle(scheduler: &cusco_server::WorkloadScheduler, timeout: Duration) -> bool {
    let started = Instant::now();
    loop {
        let metrics = scheduler.status().metrics;
        if metrics.runnable == 0 && metrics.waiting_for_consumer == 0 {
            return true;
        }
        if started.elapsed() >= timeout {
            return false;
        }
        thread::sleep(Duration::from_millis(1));
    }
}

fn percentile(values: &[u128], percentile: usize) -> u128 {
    if values.is_empty() {
        return 0;
    }
    let mut sorted = values.to_vec();
    sorted.sort_unstable();
    let index = (sorted.len() - 1).saturating_mul(percentile) / 100;
    sorted[index]
}

fn mapped_proof(
    model: PathBuf,
    n_ctx: u32,
    gpu_layers: i32,
    prefix: &str,
    output: PathBuf,
) -> Result<()> {
    let started = Instant::now();
    let model_path = model.to_str().context("model path is not UTF-8")?;
    let mut executor = Executor::open(model_path, n_ctx, gpu_layers)?;
    ensure!(
        executor.capabilities().mapped_execution,
        "executor does not support mapped execution"
    );
    let tokens = executor.tokenize(prefix)?;
    ensure!(
        !tokens.is_empty(),
        "mapped proof prefix tokenized to nothing"
    );
    executor.decode(&tokens)?;
    let checkpoint = executor.capture_checkpoint()?;
    let staged_started = Instant::now();
    for _ in 0..4 {
        let prepared = executor.prepare_restore(&checkpoint, checkpoint.checksum)?;
        executor.commit_restore(prepared)?;
    }
    let staged_restore_ns = staged_started.elapsed().as_nanos();
    let mut mappings = Vec::with_capacity(4);
    for _ in 0..4 {
        let prepared = executor.prepare_mapping_fork(cusco_executor::MappingId(0))?;
        mappings.push(executor.commit_mapping(prepared)?);
    }
    let continuation = *tokens.last().unwrap();
    let mut expected: Option<(i32, Vec<f32>)> = None;
    let mut results = Vec::with_capacity(mappings.len());
    let mut activation_ns = 0u128;
    for mapping in mappings {
        let activation_started = Instant::now();
        executor.activate_mapping(mapping)?;
        activation_ns += activation_started.elapsed().as_nanos();
        let decoded = executor.decode(&[continuation])?;
        if let Some((token, logits)) = &expected {
            ensure!(
                *token == decoded.token && logits_identical(logits, &decoded.logits),
                "mapped branches diverged"
            );
        } else {
            expected = Some((decoded.token, decoded.logits.clone()));
        }
        results.push(json!({"mapping": mapping.0, "next_token": decoded.token}));
    }
    let metrics = executor.mapping_metrics();
    ensure!(
        metrics.activation_bytes_copied == 0,
        "mapping activation copied device bytes"
    );
    let artifact = json!({
        "model": model,
        "branches": results,
        "metrics": metrics,
        "comparison": {
            "staged_restore_ns": staged_restore_ns,
            "mapped_activation_ns": activation_ns,
            "staged_bytes_read": checkpoint.bytes * 4,
            "mapped_activation_bytes_copied": metrics.activation_bytes_copied,
            "prompt_tokens_avoided": tokens.len() * 4
        },
        "elapsed_ms": started.elapsed().as_millis()
    });
    if let Some(parent) = output.parent() {
        fs::create_dir_all(parent)?;
    }
    fs::write(&output, serde_json::to_vec_pretty(&artifact)?)?;
    println!("{}", serde_json::to_string_pretty(&artifact)?);
    Ok(())
}

fn proof(
    model: PathBuf,
    expected_sha256: Option<&str>,
    allow_unverified_model: bool,
    n_ctx: u32,
    gpu_layers: i32,
    prefix: &str,
    replacement: &str,
    output: PathBuf,
) -> Result<()> {
    let started = Instant::now();
    let model_path = model.to_str().context("model path is not UTF-8")?;
    let record = if model_path == "mock://deterministic" {
        ModelRecord {
            identity: "mock".into(),
            path: model.clone(),
            sha256: "model-free".into(),
            size: 0,
        }
    } else {
        ensure!(
            expected_sha256.is_some() || allow_unverified_model,
            "the Phase 1 proof requires --sha256 for the pinned Gemma artifact"
        );
        register_local(&model, GEMMA_URI, expected_sha256)?
    };
    let mut executor = Executor::open(model_path, n_ctx, gpu_layers)?;
    let capabilities = executor.capabilities();
    ensure!(
        capabilities.global_kv && capabilities.swa && capabilities.recurrent,
        "model lacks a complete composite checkpoint capability"
    );
    let replacement = executor.tokenize(replacement)?;
    let mut contexts = Vec::new();
    for i in 0..4 {
        let prompt = format!("{prefix} [{i}]");
        let tokens = executor.tokenize(&prompt)?;
        executor.replace_state_for_proof(&tokens)?;
        let checkpoint = executor.capture_checkpoint()?;
        let continuation = tokens[tokens.len() - 1];
        let uninterrupted = executor.decode(&[continuation])?;
        contexts.push((i, prompt, checkpoint, continuation, uninterrupted));
    }
    let mut comparisons = Vec::new();
    for (context_index, prompt, checkpoint, continuation, expected) in &contexts {
        executor.replace_state_for_proof(&replacement)?;
        let prepared = executor.prepare_restore(checkpoint, checkpoint.checksum)?;
        executor.commit_restore(prepared)?;
        let restored = executor.decode(&[*continuation])?;
        comparisons.push(json!({
            "context_index": context_index,
            "prompt": prompt,
            "continuation_input_token": continuation,
            "next_token": restored.token,
            "token_equal": expected.token == restored.token,
            "logits_equal": logits_identical(&expected.logits, &restored.logits),
            "checkpoint_bytes": checkpoint.bytes,
            "checksum": checkpoint.checksum
        }));
    }
    ensure!(
        comparisons
            .iter()
            .all(|v| v["token_equal"] == true && v["logits_equal"] == true),
        "restored execution differs"
    );
    let cancellation_checkpoint = executor.capture_checkpoint()?;
    let failure_continuation = replacement[0];
    let cancellation_expected = executor.decode(&[failure_continuation])?;
    let cancellation_restore =
        executor.prepare_restore(&cancellation_checkpoint, cancellation_checkpoint.checksum)?;
    executor.commit_restore(cancellation_restore)?;
    executor.cancel_next_decode_for_proof();
    ensure!(
        executor.decode(&[failure_continuation]).is_err(),
        "cancellation did not fire"
    );
    let cancellation_actual = executor.decode(&[failure_continuation])?;
    let cancellation_preserved = cancellation_expected.token == cancellation_actual.token
        && logits_identical(&cancellation_expected.logits, &cancellation_actual.logits);
    ensure!(
        cancellation_preserved,
        "cancellation changed the active binding"
    );

    let promotion_checkpoint = executor.capture_checkpoint()?;
    let promotion_expected = executor.decode(&[failure_continuation])?;
    let promotion_restore =
        executor.prepare_restore(&promotion_checkpoint, promotion_checkpoint.checksum)?;
    executor.commit_restore(promotion_restore)?;
    ensure!(
        executor
            .prepare_restore(&promotion_checkpoint, promotion_checkpoint.checksum ^ 1)
            .is_err(),
        "corrupt promotion succeeded"
    );
    let promotion_actual = executor.decode(&[failure_continuation])?;
    let failed_promotion_preserved = promotion_expected.token == promotion_actual.token
        && logits_identical(&promotion_expected.logits, &promotion_actual.logits);
    ensure!(
        failed_promotion_preserved,
        "failed promotion changed the active binding"
    );
    let artifact = json!({"model":record,"capabilities":capabilities,"contexts":comparisons,"host_round_trip":true,"cancellation_preserved_binding":cancellation_preserved,"failed_promotion_preserved_binding":failed_promotion_preserved,"elapsed_ms":started.elapsed().as_millis()});
    if let Some(parent) = output.parent() {
        fs::create_dir_all(parent)?
    }
    fs::write(&output, serde_json::to_vec_pretty(&artifact)?)?;
    println!("{}", output.display());
    Ok(())
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn serve_cli_accepts_only_versioned_configuration_path() {
        let args = Args::try_parse_from(["cusco", "serve", "--config", "operator.yaml"]).unwrap();
        let Command::Serve { config } = args.command else {
            panic!("serve command expected")
        };
        assert_eq!(config, PathBuf::from("operator.yaml"));
        assert!(Args::try_parse_from(["cusco", "serve", "model.gguf"]).is_err());
    }

    #[test]
    fn scheduler_proof_cli_parses_versioned_workload_inputs() {
        let args = Args::try_parse_from([
            "cusco",
            "scheduler-proof",
            "model.gguf",
            "--workload",
            "workload.json",
            "--output",
            "artifact.json",
            "--context",
            "2048",
        ])
        .unwrap();
        let Command::SchedulerProof {
            model,
            workload,
            output,
            context,
            ..
        } = args.command
        else {
            panic!("scheduler-proof command expected")
        };
        assert_eq!(model, PathBuf::from("model.gguf"));
        assert_eq!(workload, PathBuf::from("workload.json"));
        assert_eq!(output, PathBuf::from("artifact.json"));
        assert_eq!(context, 2048);
        assert_eq!(percentile(&[40, 10, 30, 20], 95), 30);
    }

    #[test]
    fn serve_config_path_has_production_default() {
        use clap::CommandFactory;
        let args = Args::try_parse_from(["cusco", "serve"]).unwrap();
        let Command::Serve { config } = args.command else {
            panic!("serve command expected")
        };
        assert_eq!(config, PathBuf::from("/data/config.yaml"));
        let command = Args::command();
        let serve = command
            .get_subcommands()
            .find(|command| command.get_name() == "serve")
            .unwrap();
        assert!(
            serve
                .get_arguments()
                .all(|argument| argument.get_id() == "config" || argument.get_id() == "help")
        );
    }

    #[test]
    fn commands_and_proof_execute_model_free() {
        let root = std::env::temp_dir().join(format!("cusco-cli-{}", std::process::id()));
        fs::create_dir_all(&root).unwrap();
        let local = root.join("local.gguf");
        fs::write(&local, b"model").unwrap();
        run(Command::Register {
            path: local,
            sha256: None,
        })
        .unwrap();
        let (uri, revision, file) = (
            GEMMA_URI,
            "0314792d7f1f7e229411f620751375812bb9faf2",
            "gemma-4-E2B-it-Q3_K_M.gguf",
        );
        let cached = root
            .join("models")
            .join("unsloth--gemma-4-E2B-it-GGUF")
            .join(revision);
        fs::create_dir_all(&cached).unwrap();
        fs::write(cached.join(file), b"model").unwrap();
        run(Command::Fetch {
            uri: uri.into(),
            cache: root.clone(),
            sha256: None,
        })
        .unwrap();
        let output = root.join("proof.json");
        run(Command::Proof {
            model: PathBuf::from("mock://deterministic"),
            sha256: None,
            allow_unverified_model: true,
            context: 128,
            gpu_layers: 0,
            prefix: "prefix".into(),
            replacement: "replacement".into(),
            output: output.clone(),
        })
        .unwrap();
        let artifact: serde_json::Value =
            serde_json::from_slice(&fs::read(&output).unwrap()).unwrap();
        assert_eq!(artifact["contexts"].as_array().unwrap().len(), 4);
        assert_eq!(artifact["contexts"][0]["prompt"], "prefix [0]");
        assert!(artifact["contexts"][0]["next_token"].is_number());
        assert_eq!(artifact["failed_promotion_preserved_binding"], true);
        let mapped_output = root.join("mapped.json");
        run(Command::MappedProof {
            model: PathBuf::from("mock://deterministic"),
            context: 128,
            gpu_layers: 0,
            prefix: "mapped prefix".into(),
            output: mapped_output.clone(),
        })
        .unwrap();
        let mapped: serde_json::Value =
            serde_json::from_slice(&fs::read(mapped_output).unwrap()).unwrap();
        assert_eq!(mapped["branches"].as_array().unwrap().len(), 4);
        assert_eq!(mapped["metrics"]["activation_bytes_copied"], 0);
        assert!(mapped["comparison"]["staged_bytes_read"].as_u64().unwrap() > 0);
        assert_eq!(mapped["comparison"]["mapped_activation_bytes_copied"], 0);
        assert!(
            mapped["comparison"]["prompt_tokens_avoided"]
                .as_u64()
                .unwrap()
                > 0
        );
        let workload = root.join("scheduler-workload.json");
        fs::write(
            &workload,
            serde_json::to_vec(&json!({
                "version": 1,
                "name": "model-free",
                "model_family": "gemma3",
                "policy": {
                    "version": 1,
                    "interactive_weight": 4,
                    "standard_weight": 2,
                    "batch_weight": 1,
                    "promotion_rounds": 8,
                    "deficit_refill": 1,
                    "prefill_tokens": 8,
                    "diagnostic_capacity": 1024
                },
                "baseline": {
                    "id": "baseline",
                    "principal": "baseline",
                    "class": "standard",
                    "prompt": "baseline",
                    "prompt_repetitions": 1,
                    "max_tokens": 2,
                    "arrival_delay_ms": 0,
                    "expected": "complete"
                },
                "mixed": [
                    {
                        "id": "interactive",
                        "principal": "one",
                        "class": "interactive",
                        "prompt": "interactive",
                        "prompt_repetitions": 1,
                        "max_tokens": 2,
                        "arrival_delay_ms": 0,
                        "expected": "complete"
                    },
                    {
                        "id": "cancel",
                        "principal": "two",
                        "class": "standard",
                        "prompt": "cancel",
                        "prompt_repetitions": 1,
                        "max_tokens": 2,
                        "arrival_delay_ms": 0,
                        "expected": "cancel"
                    },
                    {
                        "id": "deadline",
                        "principal": "three",
                        "class": "batch",
                        "prompt": "deadline",
                        "prompt_repetitions": 1,
                        "max_tokens": 2,
                        "arrival_delay_ms": 0,
                        "expected": "deadline"
                    }
                ],
                "thresholds": {
                    "max_queue_age_rounds": 100,
                    "max_first_event_baseline_multiplier": 100,
                    "max_first_event_additive_ms": 1000,
                    "max_quantum_baseline_multiplier": 100,
                    "max_quantum_additive_ms": 1000
                }
            }))
            .unwrap(),
        )
        .unwrap();
        let scheduler_output = root.join("scheduler.json");
        run(Command::SchedulerProof {
            model: PathBuf::from("mock://deterministic"),
            workload,
            output: scheduler_output.clone(),
            context: 128,
            gpu_layers: 0,
            device_bytes: 1 << 20,
            host_bytes: 1 << 20,
        })
        .unwrap();
        let scheduler_artifact: Value =
            serde_json::from_slice(&fs::read(scheduler_output).unwrap()).unwrap();
        assert_eq!(scheduler_artifact["passed"], true);
        assert_eq!(scheduler_artifact["mixed"].as_array().unwrap().len(), 3);
        assert!(
            scheduler_artifact["measurements"]["max_queue_age_rounds"]
                .as_u64()
                .unwrap()
                > 0
        );
        assert!(Args::try_parse_from(["cusco", "serve", "model.gguf"]).is_err());
        fs::remove_dir_all(root).unwrap();
    }
}
