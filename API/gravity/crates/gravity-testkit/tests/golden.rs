//! Golden-trace parity tests: Rust kernels vs. fixtures captured from Python.

use gravity_providers::chatgpt::pow;
use gravity_providers::glm::sign;
use gravity_testkit::load;

#[test]
fn glm_signatures_match_python() {
    let fx = load();
    assert!(!fx.glm_signatures.is_empty(), "no GLM fixtures");
    for case in &fx.glm_signatures {
        let got = sign::generate_signature(
            &case.signature_prompt,
            case.timestamp,
            &case.request_id,
            &case.user_id,
        );
        assert_eq!(
            got, case.signature,
            "GLM signature mismatch for prompt {:?}",
            case.signature_prompt
        );
    }
}

#[test]
fn chatgpt_pow_matches_python() {
    let fx = load();
    assert!(!fx.chatgpt_pow.is_empty(), "no PoW fixtures");
    for case in &fx.chatgpt_pow {
        let (answer, solved) = pow::generate_answer(&case.seed, &case.diff, &fx.pow_config);
        assert_eq!(answer, case.answer, "PoW answer mismatch for seed {:?}", case.seed);
        assert_eq!(solved, case.solved, "PoW solved flag mismatch for seed {:?}", case.seed);
    }
}
