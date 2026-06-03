//! # gravity-testkit
//!
//! Golden-trace verification harness. Reverse-engineered providers must produce
//! request-signing output **byte-for-byte identical** to the Python original, so
//! parity is pinned with golden fixtures captured from the live Python library
//! (see `scripts/capture_golden.py`) and asserted against the Rust kernels in
//! the crate's integration tests.
//!
//! Determinism is achieved by dependency-injecting the clock and entropy
//! (`gravity_core::Env::fixed`) on the Rust side and freezing the equivalent
//! sources on the Python side, so signatures, proof-of-work tokens, and request
//! IDs are reproducible across both implementations.

#![forbid(unsafe_code)]

use serde::Deserialize;

/// The full golden fixture set loaded from `fixtures/golden.json`.
#[derive(Debug, Deserialize)]
pub struct GoldenFixtures {
    /// GLM HMAC signature vectors.
    pub glm_signatures: Vec<GlmSignatureCase>,
    /// ChatGPT proof-of-work vectors.
    pub chatgpt_pow: Vec<PowCase>,
    /// The 18-element PoW config used to produce [`Self::chatgpt_pow`].
    pub pow_config: Vec<serde_json::Value>,
}

/// One GLM signature golden vector.
#[derive(Debug, Deserialize)]
pub struct GlmSignatureCase {
    /// The signed prompt.
    pub signature_prompt: String,
    /// Millisecond timestamp.
    pub timestamp: u64,
    /// Request id.
    pub request_id: String,
    /// User id.
    pub user_id: String,
    /// Expected hex signature.
    pub signature: String,
}

/// One ChatGPT proof-of-work golden vector.
#[derive(Debug, Deserialize)]
pub struct PowCase {
    /// Challenge seed.
    pub seed: String,
    /// Difficulty hex.
    pub diff: String,
    /// Expected base64 answer.
    pub answer: String,
    /// Whether the challenge was solved.
    pub solved: bool,
}

/// Load the checked-in golden fixtures.
pub fn load() -> GoldenFixtures {
    let raw = include_str!("../fixtures/golden.json");
    serde_json::from_str(raw).expect("golden fixtures are valid JSON")
}
