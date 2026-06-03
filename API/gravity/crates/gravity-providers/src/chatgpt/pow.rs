//! ChatGPT Sentinel Proof-of-Work.
//!
//! Ports `_generate_answer` / `build_proof_token` from `gravity/chatgpt/pow.py`.
//! The server validates the SHA3-512 digest against a difficulty target, so the
//! payload byte-layout must match Python exactly — hence the golden vector test.
//!
//! Payload layout per brute-force iteration `i`:
//! `static_1 ‖ str(i) ‖ static_2 ‖ str(i>>1) ‖ static_3`, base64'd, then
//! `digest = SHA3-512(seed ‖ base64)` and accept when
//! `digest[..diff.len()] <= hex_decode(diff)` (lexicographic byte compare).

use base64::engine::general_purpose::STANDARD;
use base64::Engine as _;
use serde_json::Value;
use sha3::{Digest, Sha3_512};

/// Max brute-force iterations before falling back (matches Python `500000`).
const MAX_ITERS: u64 = 500_000;

/// The Python fallback prefix used when no solution is found in time.
const FALLBACK_PREFIX: &str = "wQ8Lk5FbGpA2NcR9dShT6gYjU7VxZ4D";

/// Build the three static payload segments from the 18-element config.
///
/// Mirrors the exact `json.dumps(separators=(',',':'))` + string-slicing of the
/// Python kernel.
fn static_segments(config: &[Value]) -> (Vec<u8>, Vec<u8>, Vec<u8>) {
    // static_1 = dumps(config[:3])[:-1] + ","
    let s1 = serde_json::to_string(&config[0..3]).unwrap();
    let static_1 = format!("{},", &s1[..s1.len() - 1]);

    // static_2 = "," + dumps(config[4:9])[1:-1] + ","
    let s2 = serde_json::to_string(&config[4..9]).unwrap();
    let static_2 = format!(",{},", &s2[1..s2.len() - 1]);

    // static_3 = "," + dumps(config[10:])[1:]
    let s3 = serde_json::to_string(&config[10..]).unwrap();
    let static_3 = format!(",{}", &s3[1..]);

    (static_1.into_bytes(), static_2.into_bytes(), static_3.into_bytes())
}

/// Decode a hex string to bytes (small, dependency-free).
fn hex_decode(s: &str) -> Vec<u8> {
    let bytes = s.as_bytes();
    let mut out = Vec::with_capacity(bytes.len() / 2);
    let mut i = 0;
    while i + 1 < bytes.len() {
        let hi = (bytes[i] as char).to_digit(16);
        let lo = (bytes[i + 1] as char).to_digit(16);
        match (hi, lo) {
            (Some(h), Some(l)) => out.push(((h << 4) | l) as u8),
            _ => break,
        }
        i += 2;
    }
    out
}

/// Solve the PoW. Returns `(answer_base64, solved)`.
pub fn generate_answer(seed: &str, diff: &str, config: &[Value]) -> (String, bool) {
    let diff_len = diff.len();
    let target = hex_decode(diff);
    let seed_bytes = seed.as_bytes();
    let (static_1, static_2, static_3) = static_segments(config);

    for i in 0..MAX_ITERS {
        let json_i = i.to_string();
        let json_j = (i >> 1).to_string();
        let mut payload =
            Vec::with_capacity(static_1.len() + static_2.len() + static_3.len() + json_i.len() + json_j.len());
        payload.extend_from_slice(&static_1);
        payload.extend_from_slice(json_i.as_bytes());
        payload.extend_from_slice(&static_2);
        payload.extend_from_slice(json_j.as_bytes());
        payload.extend_from_slice(&static_3);

        let encoded = STANDARD.encode(&payload);

        let mut hasher = Sha3_512::new();
        hasher.update(seed_bytes);
        hasher.update(encoded.as_bytes());
        let digest = hasher.finalize();

        // Compare the first `diff_len` bytes against the target, with Python's
        // bytes-comparison semantics (lexicographic, then length).
        let take = diff_len.min(digest.len());
        if digest[..take] <= target[..] {
            return (encoded, true);
        }
    }

    let fallback = format!(
        "{FALLBACK_PREFIX}{}",
        STANDARD.encode(format!("\"{seed}\"").as_bytes())
    );
    (fallback, false)
}

/// Build the Sentinel proof token (`gAAAAAB` + answer).
pub fn build_proof_token(seed: &str, difficulty: &str, config: &[Value]) -> (String, bool) {
    let (answer, solved) = generate_answer(seed, difficulty, config);
    (format!("gAAAAAB{answer}"), solved)
}

/// Build a requirements token (`gAAAAAC` + answer) from a (caller-provided) seed.
pub fn build_requirements_token(seed: &str, config: &[Value]) -> String {
    let (answer, _) = generate_answer(seed, "0fffff", config);
    format!("gAAAAAC{answer}")
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn fixed_config() -> Vec<Value> {
        json!([
            3000,
            "Mon Jan 01 2024 00:00:00 GMT-0500 (Eastern Standard Time)",
            4294705152u64,
            0,
            "UA/1.0",
            "https://chatgpt.com/s.js",
            "dpl-1",
            "en-US",
            "en-US,en",
            0,
            "navkey-x",
            "location",
            "document",
            123.0,
            "uuid-fixed",
            "",
            8,
            0.0
        ])
        .as_array()
        .unwrap()
        .clone()
    }

    #[test]
    fn matches_python_golden_answer() {
        let (answer, solved) = generate_answer("seed-xyz", "ff", &fixed_config());
        assert!(solved);
        assert_eq!(
            answer,
            "WzMwMDAsIk1vbiBKYW4gMDEgMjAyNCAwMDowMDowMCBHTVQtMDUwMCAoRWFzdGVybiBTdGFuZGFyZCBUaW1lKSIsNDI5NDcwNTE1MiwwLCJVQS8xLjAiLCJodHRwczovL2NoYXRncHQuY29tL3MuanMiLCJkcGwtMSIsImVuLVVTIiwiZW4tVVMsZW4iLDAsIm5hdmtleS14IiwibG9jYXRpb24iLCJkb2N1bWVudCIsMTIzLjAsInV1aWQtZml4ZWQiLCIiLDgsMC4wXQ=="
        );
    }

    #[test]
    fn proof_token_prefix_matches_python() {
        let (tok, solved) = build_proof_token("seed-xyz", "ff", &fixed_config());
        assert!(solved);
        assert!(tok.starts_with("gAAAAAB"));
        assert!(tok.contains("WzMwMDAs"));
    }
}
