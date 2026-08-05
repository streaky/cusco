use anyhow::{Context, Result, ensure};
use clap::{Parser, Subcommand};
use cusco_executor::{Executor, logits_identical};
use cusco_model_registry::{GEMMA_URI, ModelRecord, fetch_hf, register_local};
use serde_json::json;
use std::{fs, net::SocketAddr, path::PathBuf, sync::Arc, time::Instant};

#[derive(Parser)]
struct Args {
    #[command(subcommand)]
    command: Command,
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
    Serve {
        model: PathBuf,
        #[arg(long, default_value = "gemma-4-e2b-it")]
        model_id: String,
        #[arg(long, default_value = "gemma-4-e2b-it")]
        model_family: String,
        #[arg(long, default_value_t = 4096)]
        context: u32,
        #[arg(long, default_value_t = 99)]
        gpu_layers: i32,
        #[arg(long, default_value_t = 8_589_934_592)]
        device_bytes: usize,
        #[arg(long, default_value_t = 17_179_869_184)]
        host_bytes: usize,
        #[arg(long, default_value_t = 68_719_476_736)]
        storage_bytes: u64,
        #[arg(long, default_value_t = 1_073_741_824)]
        context_reserve_bytes: u64,
        #[arg(long, default_value = "/data/cusco-spill")]
        spill_directory: PathBuf,
        #[arg(long)]
        require_competent: bool,
        #[arg(long, default_value = "127.0.0.1:8080")]
        listen: SocketAddr,
        #[arg(long, default_value = "/data/cusco-state.json")]
        state: PathBuf,
        #[arg(long)]
        bearer_token: Option<String>,
        #[arg(long)]
        unsafe_public_unauthenticated: bool,
        #[arg(long, default_value_t = 4)]
        active_requests: usize,
        #[arg(long, default_value_t = 32)]
        queue_count: usize,
        #[arg(long, default_value_t = 16_777_216)]
        queue_bytes: usize,
        #[arg(long, default_value_t = 1_048_576)]
        request_bytes: usize,
        #[arg(long, default_value_t = 16)]
        pre_queue_concurrency: usize,
        #[arg(long, default_value_t = 32_768)]
        header_bytes: usize,
        #[arg(long, default_value_t = 10_000)]
        body_timeout_ms: u64,
        #[arg(long, default_value_t = 300_000)]
        wall_time_ms: u64,
        #[arg(long, default_value_t = 240_000)]
        active_time_ms: u64,
        #[arg(long, default_value_t = 8)]
        stream_buffer: usize,
        #[arg(long, default_value_t = 30_000)]
        shutdown_grace_ms: u64,
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
        Command::Serve {
            model,
            model_id,
            model_family,
            context,
            gpu_layers,
            device_bytes,
            host_bytes,
            storage_bytes,
            context_reserve_bytes,
            spill_directory,
            require_competent,
            listen,
            state,
            bearer_token,
            unsafe_public_unauthenticated,
            active_requests,
            queue_count,
            queue_bytes,
            request_bytes,
            pre_queue_concurrency,
            header_bytes,
            body_timeout_ms,
            wall_time_ms,
            active_time_ms,
            stream_buffer,
            shutdown_grace_ms,
        } => {
            use cusco_server::{
                AnonymousAdmin, AuthProvider, BearerAuth, ModelRecord, ResidencyConfig,
                ResidentEngine, Server, ServerConfig,
            };
            let anonymous = bearer_token.is_none();
            let auth: Arc<dyn AuthProvider> = match bearer_token {
                Some(token) => Arc::new(BearerAuth::new(token)),
                None => Arc::new(AnonymousAdmin),
            };
            let registered = register_local(&model, &model_id, None)?;
            let engine = ResidentEngine::open_with_spill(
                ResidencyConfig {
                    device_bytes: device_bytes as u64,
                    host_bytes: host_bytes as u64,
                    storage_bytes,
                    context_reserve_bytes,
                    n_ctx: context,
                    gpu_layers,
                    require_competent,
                },
                spill_directory,
            )?;
            let server = Server::open(state, auth, engine)?;
            server.configure(ServerConfig {
                active_requests,
                queue_count,
                queue_bytes,
                request_bytes,
                pre_queue_concurrency,
                header_bytes,
                body_timeout_ms,
                wall_time_ms,
                active_time_ms,
                stream_buffer,
                shutdown_grace_ms,
            })?;
            server.register_model(ModelRecord {
                id: model_id.clone(),
                revision: registered.sha256.clone(),
                path: registered.path,
                sha256: registered.sha256,
                aliases: vec![],
                family: model_family,
                size_bytes: registered.size,
                epoch: 0,
            })?;
            tokio::runtime::Runtime::new()?.block_on(cusco_server::serve(
                server,
                listen,
                anonymous,
                unsafe_public_unauthenticated,
            ))?;
        }
    }
    Ok(())
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
    fn serve_cli_parses_bounded_lifecycle_configuration() {
        let args = Args::try_parse_from([
            "cusco",
            "serve",
            "model.gguf",
            "--active-requests",
            "2",
            "--queue-count",
            "3",
            "--queue-bytes",
            "4096",
            "--request-bytes",
            "2048",
            "--stream-buffer",
            "4",
            "--shutdown-grace-ms",
            "250",
        ])
        .unwrap();
        let Command::Serve {
            active_requests,
            queue_count,
            queue_bytes,
            request_bytes,
            stream_buffer,
            shutdown_grace_ms,
            ..
        } = args.command
        else {
            panic!("serve command expected")
        };
        assert_eq!(
            (
                active_requests,
                queue_count,
                queue_bytes,
                request_bytes,
                stream_buffer,
                shutdown_grace_ms,
            ),
            (2, 3, 4096, 2048, 4, 250)
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
        let public: SocketAddr = "0.0.0.0:8080".parse().unwrap();
        assert!(
            run(Command::Serve {
                model: PathBuf::from("mock://deterministic"),
                model_id: "gemma-4-e2b-it".into(),
                model_family: "gemma-4-e2b-it".into(),
                context: 128,
                gpu_layers: 0,
                device_bytes: 1 << 20,
                host_bytes: 1 << 20,
                storage_bytes: 1 << 20,
                context_reserve_bytes: 1,
                spill_directory: root.join("spill"),
                require_competent: false,
                listen: public,
                state: root.join("server.json"),
                bearer_token: None,
                unsafe_public_unauthenticated: false,
                active_requests: 1,
                queue_count: 1,
                queue_bytes: 1024,
                request_bytes: 1024,
                pre_queue_concurrency: 16,
                header_bytes: 32 << 10,
                body_timeout_ms: 10_000,
                wall_time_ms: 300_000,
                active_time_ms: 240_000,
                stream_buffer: 1,
                shutdown_grace_ms: 100,
            })
            .is_err()
        );
        let loopback: SocketAddr = "127.0.0.1:8080".parse().unwrap();
        let unsupported = run(Command::Serve {
            model: root.join("local.gguf"),
            model_id: "unsupported".into(),
            model_family: "unsupported".into(),
            context: 128,
            gpu_layers: 0,
            device_bytes: 1 << 20,
            host_bytes: 1 << 20,
            storage_bytes: 1 << 20,
            context_reserve_bytes: 1,
            spill_directory: root.join("unsupported-spill"),
            require_competent: false,
            listen: loopback,
            state: root.join("unsupported-server.json"),
            bearer_token: None,
            unsafe_public_unauthenticated: false,
            active_requests: 1,
            queue_count: 1,
            queue_bytes: 1024,
            request_bytes: 1024,
            pre_queue_concurrency: 16,
            header_bytes: 32 << 10,
            body_timeout_ms: 10_000,
            wall_time_ms: 300_000,
            active_time_ms: 240_000,
            stream_buffer: 1,
            shutdown_grace_ms: 100,
        })
        .unwrap_err();
        assert!(unsupported.to_string().contains("unsupported model family"));
        fs::remove_dir_all(root).unwrap();
    }
}
