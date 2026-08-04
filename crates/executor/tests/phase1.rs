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
        executor.replace(&tokens).unwrap();
        let checkpoint = executor.capture().unwrap();
        let continuation = *tokens.last().unwrap();
        let expected = executor.decode(&[continuation]).unwrap();
        contexts.push((checkpoint, continuation, expected));
    }
    for (checkpoint, continuation, expected) in &contexts {
        executor.replace(&replacement).unwrap();
        let host_copy = executor.prepare(checkpoint, checkpoint.checksum).unwrap();
        executor.commit(host_copy).unwrap();
        let actual = executor.decode(&[*continuation]).unwrap();
        assert_eq!(actual.token, expected.token);
        assert!(logits_identical(&actual.logits, &expected.logits));
    }
    let prior = executor.capture().unwrap();
    executor.cancel_next();
    assert_eq!(
        executor.decode(&[replacement[0]]).unwrap_err(),
        Error::Cancelled
    );
    assert_eq!(
        executor.prepare(&prior, prior.checksum ^ 1).unwrap_err(),
        Error::Incompatible
    );
    let prepared = executor.prepare(&prior, prior.checksum).unwrap();
    executor.commit(prepared).unwrap();
    assert!(executor.decode(&[replacement[0]]).is_ok());
}
