use cusco_executor::{Error, Executor, logits_identical};

#[test]
#[ignore = "requires the pinned external Gemma GGUF"]
fn displaced_host_round_trip_is_exact_and_transactional() {
    let model = std::env::var("CUSCO_TEST_MODEL")
        .expect("CUSCO_TEST_MODEL must name the pinned Gemma GGUF");
    let mut executor = Executor::open(&model, 4096, 99).unwrap();
    let replacement = executor.tokenize("unrelated displacement").unwrap();
    let mut contexts = Vec::new();
    for index in 0..4 {
        let tokens = executor
            .tokenize(&format!("deterministic prefix {index}"))
            .unwrap();
        executor.replace_state_for_proof(&tokens).unwrap();
        let checkpoint = executor.capture_checkpoint().unwrap();
        let continuation = *tokens.last().unwrap();
        let expected = executor.decode(&[continuation]).unwrap();
        contexts.push((checkpoint, continuation, expected));
    }
    for (checkpoint, continuation, expected) in &contexts {
        executor.replace_state_for_proof(&replacement).unwrap();
        let host_copy = executor
            .prepare_restore(checkpoint, checkpoint.checksum)
            .unwrap();
        executor.commit_restore(host_copy).unwrap();
        let actual = executor.decode(&[*continuation]).unwrap();
        assert_eq!(actual.token, expected.token);
        assert!(logits_identical(&actual.logits, &expected.logits));
    }
    let prior = executor.capture_checkpoint().unwrap();
    executor.cancel_next_decode_for_proof();
    assert_eq!(
        executor.decode(&[replacement[0]]).unwrap_err(),
        Error::Cancelled
    );
    assert_eq!(
        executor
            .prepare_restore(&prior, prior.checksum ^ 1)
            .unwrap_err(),
        Error::Incompatible
    );
    let prepared = executor.prepare_restore(&prior, prior.checksum).unwrap();
    executor.commit_restore(prepared).unwrap();
    assert!(executor.decode(&[replacement[0]]).is_ok());
}

#[test]
#[ignore = "requires the pinned external Gemma GGUF"]
fn mapped_fork_chain_can_use_the_full_configured_context() {
    let model = std::env::var("CUSCO_TEST_MODEL")
        .expect("CUSCO_TEST_MODEL must name the pinned Gemma GGUF");
    let mut executor = Executor::open(&model, 4096, 99).unwrap();
    let tokens = executor
        .tokenize(&"mapped context capacity ".repeat(256))
        .unwrap();
    assert!(tokens.len() > 512);

    let mut source = executor.active_representation().unwrap();
    let mut retained = Vec::new();
    for chunk in tokens[..512].chunks(32) {
        let prepared = executor.prepare_mapping_fork(&source).unwrap();
        let successor = executor.commit_mapping(prepared).unwrap();
        executor.activate_mapping(&successor).unwrap();
        executor.decode(chunk).unwrap();
        retained.push(source);
        source = successor;
    }

    assert_eq!(
        executor
            .describe_representation(&source)
            .unwrap()
            .represented_position,
        512
    );
    assert_eq!(retained.len(), 16);
}
