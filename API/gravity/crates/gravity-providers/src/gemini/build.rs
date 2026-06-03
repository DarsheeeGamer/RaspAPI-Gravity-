//! Gemini web `f.req` payload assembly.
//!
//! Ports `_build_generate_payload` from `gravity/gemini/client.py`: the request
//! is a 69-element array (`inner_req_list`) whose indices are reverse-engineered
//! and must match exactly, wrapped as `[null, json(inner_req_list)]` and posted
//! as the `f.req` form field to the `StreamGenerate` endpoint.

use crate::util::uuid_v4;
use serde_json::{json, Value};

/// `StreamGenerate` endpoint.
pub const GENERATE_URL: &str =
    "https://gemini.google.com/_/BardChatUi/data/assistant.lamda.BardFrontendService/StreamGenerate";

/// Generic `batchexecute` endpoint (used for the `otAQ7b` GetUserStatus RPC
/// that returns the account's available-model list).
pub const BATCH_EXEC_URL: &str = "https://gemini.google.com/_/BardChatUi/data/batchexecute";

/// RPC id for `GetUserStatus` — its response carries the available-model list.
pub const RPC_GET_USER_STATUS: &str = "otAQ7b";

/// Bootstrap page — embeds `SNlM0e`/`cfb2h`/`FdrFJe` session tokens.
pub const APP_URL: &str = "https://gemini.google.com/app";

const TEMPORARY_CHAT_FLAG_INDEX: usize = 45;
const GEM_ID_INDEX: usize = 19;

/// Default metadata when a conversation has none yet.
fn default_metadata() -> Vec<Value> {
    vec![
        json!(""), json!(""), json!(""), Value::Null, Value::Null,
        Value::Null, Value::Null, Value::Null, Value::Null, json!(""),
    ]
}

/// Build the `f.req` payload string and the per-request UUID.
///
/// `metadata` carries `[c_id, r_id, rc_id, ...]` for conversation continuity.
pub fn build_freq(
    env: &gravity_core::Env,
    prompt: &str,
    system_prompt: Option<&str>,
    metadata: &[Value],
    temporary: bool,
    gem_id: Option<&str>,
) -> (String, String) {
    let prompt_text = match system_prompt {
        Some(sp) if !sp.trim().is_empty() => format!(
            "System instructions:\n{}\n\nUser request:\n{}",
            sp.trim(),
            prompt.trim()
        ),
        _ => prompt.trim().to_owned(),
    };

    let message_content = json!([prompt_text, 0, Value::Null, Value::Null, Value::Null, Value::Null, 0]);

    let mut inner: Vec<Value> = vec![Value::Null; 81];
    inner[0] = message_content;
    inner[1] = json!(["en"]);
    inner[2] = if metadata.is_empty() {
        Value::Array(default_metadata())
    } else {
        Value::Array(metadata.to_vec())
    };
    inner[6] = json!([1]);
    inner[7] = json!(1);
    inner[10] = json!(1);
    inner[11] = json!(0);
    inner[17] = json!([[0]]);
    inner[18] = json!(0);
    if let Some(g) = gem_id {
        inner[GEM_ID_INDEX] = json!(g);
    }
    inner[27] = json!(1);
    inner[30] = json!([4]);
    inner[41] = json!([1]);
    inner[53] = json!(0);
    inner[61] = json!([]);
    inner[68] = json!(2);
    inner[79] = json!(1);
    inner[80] = json!(1);
    if temporary {
        inner[TEMPORARY_CHAT_FLAG_INDEX] = json!(1);
    }
    let uuid_val = uuid_v4(env).to_uppercase();
    inner[59] = json!(uuid_val);

    // payload = json([null, json(inner)])
    let inner_str = Value::Array(inner).to_string();
    let payload = json!([Value::Null, inner_str]).to_string();
    (payload, uuid_val)
}

/// Derive a `_reqid` value from injected entropy (non-parity-critical).
pub fn reqid(env: &gravity_core::Env) -> u64 {
    10_000 + (env.entropy.next_u64() % 900_000)
}

/// Known web models exposed by the provider.
pub fn models() -> Vec<String> {
    [
        "gemini-3-flash", "gemini-3-pro", "gemini-3-flash-thinking",
        "gemini-3-pro-plus", "gemini-3-flash-plus",
        "gemini-3-pro-advanced", "gemini-3-flash-advanced",
        "gemini-2.5-pro", "gemini-2.5-flash",
    ].iter().map(|s| s.to_string()).collect()
}

/// The canonical Gemini web model table: `(request_name, model_id_hex, tier)`.
///
/// `tier`: 0=free, 1=standard, 2=advanced, 4=plus. The hex ids match those
/// returned by the `otAQ7b` GetUserStatus RPC, so discovery can reverse-map a
/// discovered hex id back to a request-usable name. Reverse-engineered from
/// `gravity/gemini/constants.py::GEMINI_WEB_MODEL_HEADERS` + HanaokaYuzu/Gemini-API.
pub const MODEL_TABLE: &[(&str, &str, u8)] = &[
    ("gemini-3-flash", "fbb127bbb056c959", 1),
    ("gemini-3-pro", "9d8ca3786ebdfbea", 1),
    ("gemini-3-flash-thinking", "5bf011840784117a", 1),
    ("gemini-3-pro-plus", "e6fa609c3fa255c0", 4),
    ("gemini-3-flash-plus", "56fdd199312815e2", 4),
    ("gemini-3-flash-thinking-plus", "e051ce1aa80aa576", 4),
    ("gemini-3-pro-advanced", "e6fa609c3fa255c0", 2),
    ("gemini-3-flash-advanced", "56fdd199312815e2", 2),
];

/// Reverse-map a hex `model_id` (from the `otAQ7b` RPC) to a request-usable name.
pub fn model_name_for_id(id: &str) -> Option<&'static str> {
    MODEL_TABLE.iter().find(|(_, hex, _)| *hex == id).map(|(name, _, _)| *name)
}

/// Model-specific headers — `x-goog-ext-525001261-jspb` carries the model ID.
///
/// Values reverse-engineered from `gravity/gemini/constants.py::GEMINI_WEB_MODEL_HEADERS`.
/// Build model-specific headers for a request, including the per-request UUID
/// for the `x-goog-ext-525001261-jspb` routing header.
///
/// Format (from HAR): `[1,null,null,null,"<id>",null,null,0,[4,5,6,8],null,null,<tier>,null,null,1,1,"<uuid>"]`
pub fn model_headers_with_uuid(model: &str, uuid: &str) -> Vec<(String, String)> {
    // (model_id, tier): tier 0=free, 1=standard, 2=advanced, 4=plus
    let id_tier: Option<(&str, u8)> = MODEL_TABLE
        .iter()
        .find(|(name, _, _)| *name == model)
        .map(|(_, id, tier)| (*id, *tier));
    let mut headers: Vec<(String, String)> = vec![
        ("x-goog-ext-73010989-jspb".into(), "[0]".into()),
        ("x-goog-ext-73010990-jspb".into(), "[0,0,0]".into()),
    ];
    if let Some((id, tier)) = id_tier {
        headers.push((
            "x-goog-ext-525001261-jspb".into(),
            format!(r#"[1,null,null,null,"{id}",null,null,0,[4,5,6,8],null,null,{tier},null,null,1,1,"{uuid}"]"#),
        ));
    }
    headers
}

/// Static model headers — used by unit tests that don't need a per-request UUID.
#[allow(dead_code)]
pub fn model_headers(model: &str) -> Vec<(&'static str, &'static str)> {
    // Static headers for unit tests; live requests use model_headers_with_uuid.
    let headers: Vec<(&'static str, &'static str)> = vec![
        ("x-goog-ext-73010989-jspb", "[0]"),
        ("x-goog-ext-73010990-jspb", "[0,0,0]"),
    ];
    // 525001261 is omitted here since it requires a per-request UUID;
    // provider.rs uses model_headers_with_uuid for actual requests.
    let _ = model; // used by live path
    headers
}

#[cfg(test)]
mod tests {
    use super::*;
    use gravity_core::Env;

    #[test]
    fn freq_has_69_element_inner_with_fixed_indices() {
        let env = Env::fixed(0, 1);
        let (payload, uuid) = build_freq(&env, "Hello", None, &[], false, None);
        // payload = [null, "<inner json string>"]
        let outer: Value = serde_json::from_str(&payload).unwrap();
        assert!(outer[0].is_null());
        let inner: Value = serde_json::from_str(outer[1].as_str().unwrap()).unwrap();
        let arr = inner.as_array().unwrap();
        assert_eq!(arr.len(), 81);
        assert_eq!(arr[0][0], "Hello"); // prompt at message_content[0]
        assert_eq!(arr[1], json!(["en"]));
        assert_eq!(arr[68], 2);
        assert_eq!(arr[59].as_str().unwrap(), uuid);
        assert_eq!(uuid, uuid.to_uppercase());
    }

    #[test]
    fn system_prompt_is_prefixed() {
        let env = Env::fixed(0, 1);
        let (payload, _) = build_freq(&env, "Question?", Some("Be terse."), &[], false, None);
        let outer: Value = serde_json::from_str(&payload).unwrap();
        let inner: Value = serde_json::from_str(outer[1].as_str().unwrap()).unwrap();
        let prompt = inner[0][0].as_str().unwrap();
        assert!(prompt.starts_with("System instructions:\nBe terse."));
        assert!(prompt.contains("User request:\nQuestion?"));
    }

    #[test]
    fn temporary_flag_sets_index_45() {
        let env = Env::fixed(0, 1);
        let (payload, _) = build_freq(&env, "x", None, &[], true, None);
        let outer: Value = serde_json::from_str(&payload).unwrap();
        let inner: Value = serde_json::from_str(outer[1].as_str().unwrap()).unwrap();
        assert_eq!(inner[45], 1);
    }
}
