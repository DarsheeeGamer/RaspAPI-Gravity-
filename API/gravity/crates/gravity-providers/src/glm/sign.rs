//! GLM / chat.z.ai request signing.
//!
//! Ports `_generate_signature` from `gravity/glm/client.py` exactly — the server
//! validates the `X-Signature` over precise bytes, so this is golden-traced.
//!
//! Algorithm (given `signature_prompt`, `timestamp` ms, `request_id`, `user_id`):
//! 1. `payload_b64 = base64(signature_prompt)`
//! 2. `sorted_payload = "requestId,<id>,timestamp,<ts>,user_id,<uid>"`
//!    (the three pairs sorted alphabetically by key, flattened, comma-joined)
//! 3. `data = "<sorted_payload>|<payload_b64>|<timestamp>"`
//! 4. `time_step = timestamp / 300000` (integer; 5-minute bucket)
//! 5. `step_key = hex(HMAC-SHA256(MASTER_KEY, str(time_step)))`
//! 6. `signature = hex(HMAC-SHA256(step_key, data))`

use base64::engine::general_purpose::STANDARD;
use base64::Engine as _;
use hmac::{Hmac, Mac};
use sha2::Sha256;

type HmacSha256 = Hmac<Sha256>;

/// The reverse-engineered master key (`gravity/glm/client.py`).
const MASTER_KEY: &[u8] = b"key-@@@@)))()((9))-xxxx&&&%%%%%";

/// 5-minute signing bucket in milliseconds.
const TIME_STEP_MS: u64 = 300_000;

fn hmac_hex(key: &[u8], msg: &[u8]) -> String {
    let mut mac = HmacSha256::new_from_slice(key).expect("HMAC accepts any key length");
    mac.update(msg);
    let bytes = mac.finalize().into_bytes();
    let mut s = String::with_capacity(bytes.len() * 2);
    for b in bytes {
        use std::fmt::Write;
        let _ = write!(s, "{b:02x}");
    }
    s
}

/// Compute the `X-Signature` for a GLM chat request.
pub fn generate_signature(
    signature_prompt: &str,
    timestamp: u64,
    request_id: &str,
    user_id: &str,
) -> String {
    let payload_b64 = STANDARD.encode(signature_prompt.as_bytes());

    // Pairs sorted alphabetically by key: requestId, timestamp, user_id.
    let ts = timestamp.to_string();
    let sorted_payload = format!(
        "requestId,{request_id},timestamp,{ts},user_id,{user_id}"
    );

    let data = format!("{sorted_payload}|{payload_b64}|{timestamp}");

    let time_step = (timestamp / TIME_STEP_MS).to_string();
    let step_key = hmac_hex(MASTER_KEY, time_step.as_bytes());
    hmac_hex(step_key.as_bytes(), data.as_bytes())
}

#[cfg(test)]
mod tests {
    use super::*;

    // Golden vectors computed from the Python `generate_signature`
    // (gravity/zlm/utils.py / gravity/glm/client.py) under fixed inputs.
    #[test]
    fn matches_python_golden_vector() {
        let sig = generate_signature("Hello world", 1_700_000_000_000, "req-123", "user-9");
        assert_eq!(
            sig,
            "a607c14fce186d369f231d3ba0862e32eea0422d35b108ae992c15f844c0607b"
        );
    }

    #[test]
    fn time_bucket_is_5_minutes() {
        // Two timestamps in the same 5-min bucket → same step key component →
        // signatures differ only via the `data` timestamp, but a boundary
        // crossing changes the bucket. Verify the bucket math directly.
        assert_eq!(1_700_000_000_000u64 / 300_000, 5_666_666);
        assert_eq!(1_700_000_299_999u64 / 300_000, 5_666_667);
    }

    #[test]
    fn empty_prompt_still_signs() {
        let sig = generate_signature("", 1_700_000_000_000, "r", "u");
        assert_eq!(sig.len(), 64); // hex sha256
    }
}
