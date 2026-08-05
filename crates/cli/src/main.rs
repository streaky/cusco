use anyhow::{Context, Result, ensure};
use clap::{Parser, Subcommand};
use cusco_executor::{Executor, logits_identical};
use cusco_model_registry::{GEMMA_URI, ModelRecord, fetch_hf, register_local};
use serde_json::json;
use std::{fs, path::PathBuf, time::Instant};
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
    }
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
        fs::remove_dir_all(root).unwrap();
    }
}
