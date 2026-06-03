//! GLM request-body, query-string, and header builders.
//!
//! Ports the `_build_body` / `_build_url` / `_build_headers` helpers from
//! `gravity/glm/client.py` and the constants from `gravity/glm/constants.py`.

use crate::util::uuid_v4;
use gravity_core::{Content, Env, Message, Role};
use serde_json::{json, Map, Value};

/// chat.z.ai endpoints.
pub const COMPLETIONS_URL: &str = "https://chat.z.ai/api/v2/chat/completions";
/// Model-list endpoint (OpenWebUI-style; the call chat.z.ai makes to populate
/// its model picker). Returns `{"data":[{"id","name","info":{...}}]}`.
pub const MODELS_URL: &str = "https://chat.z.ai/api/models";
/// Default model.
pub const DEFAULT_MODEL: &str = "glm-5";

/// Static models exposed by the provider.
pub fn models() -> Vec<String> {
    ["glm-5", "glm-4.7", "glm-4-air-250414", "zero", "deep-research"]
        .iter()
        .map(|s| s.to_string())
        .collect()
}

/// `User-Agent` used in headers and the signed query.
pub const USER_AGENT: &str = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) \
AppleWebKit/537.36 (KHTML, like Gecko) Chrome/145.0.0.0 Safari/537.36 OPR/129.0.0.0";

/// Default request headers (excluding auth/signature).
pub fn default_headers() -> Vec<(String, String)> {
    [
        ("Accept", "*/*"),
        ("Accept-Language", "en-US"),
        ("Connection", "keep-alive"),
        ("Content-Type", "application/json"),
        ("Origin", "https://chat.z.ai"),
        ("Referer", "https://chat.z.ai"),
        ("Sec-Fetch-Dest", "empty"),
        ("Sec-Fetch-Mode", "cors"),
        ("Sec-Fetch-Site", "same-origin"),
        ("User-Agent", USER_AGENT),
        ("X-Zai-Client", "web"),
        ("X-FE-Version", "prod-fe-1.1.7"),
        ("sec-ch-ua", "\"Not:A-Brand\";v=\"99\", \"Opera GX\";v=\"129\", \"Chromium\";v=\"145\""),
        ("sec-ch-ua-mobile", "?0"),
        ("sec-ch-ua-platform", "\"Windows\""),
    ]
    .iter()
    .map(|(k, v)| (k.to_string(), v.to_string()))
    .collect()
}

/// The last user message's content — the `signature_prompt`.
pub fn signature_prompt(messages: &[Message]) -> String {
    messages
        .iter()
        .rev()
        .find(|m| m.role == Role::User)
        .map(text_of)
        .unwrap_or_default()
}

fn text_of(m: &Message) -> String {
    match &m.content {
        Some(Content::Text(s)) => s.clone(),
        Some(Content::Parts(parts)) => parts
            .iter()
            .filter_map(|p| match p {
                gravity_core::ContentPart::Text { text } => Some(text.clone()),
                _ => None,
            })
            .collect::<Vec<_>>()
            .join("\n"),
        None => String::new(),
    }
}

/// Build the completion request body (`_build_body`).
pub fn build_body(
    env: &Env,
    messages: &[Message],
    model: &str,
    chat_id: &str,
    user_name: &str,
) -> Value {
    let oai_messages: Vec<Value> = messages
        .iter()
        .map(|m| {
            let role = match m.role {
                Role::System => "system",
                Role::User => "user",
                Role::Assistant => "assistant",
                Role::Tool => "tool",
            };
            json!({ "role": role, "content": text_of(m) })
        })
        .collect();

    let mut variables = datetime_variables(env);
    variables.insert("{{USER_NAME}}".into(), json!(user_name));
    variables.insert("{{USER_LOCATION}}".into(), json!("Unknown"));

    json!({
        "stream": true,
        "model": model,
        "messages": oai_messages,
        "signature_prompt": signature_prompt(messages),
        "params": {},
        "extra": {},
        "features": {
            "image_generation": false, "web_search": false, "auto_web_search": false,
            "preview_mode": true, "flags": [], "vlm_tools_enable": false,
            "vlm_web_search_enable": false, "vlm_website_mode": false, "enable_thinking": true,
        },
        "variables": Value::Object(variables),
        "chat_id": chat_id,
        "id": uuid_v4(env),
        "current_user_message_id": uuid_v4(env),
        "current_user_message_parent_id": Value::Null,
        "background_tasks": { "title_generation": true, "tags_generation": true },
    })
}

/// Datetime substitution variables (`get_datetime_variables`).
fn datetime_variables(env: &Env) -> Map<String, Value> {
    let secs = env.clock.now_secs() as i64;
    let dt = chrono::DateTime::from_timestamp(secs, 0).unwrap_or_default();
    let mut m = Map::new();
    m.insert("{{CURRENT_DATETIME}}".into(), json!(dt.format("%Y-%m-%d %H:%M:%S").to_string()));
    m.insert("{{CURRENT_DATE}}".into(), json!(dt.format("%Y-%m-%d").to_string()));
    m.insert("{{CURRENT_TIME}}".into(), json!(dt.format("%H:%M:%S").to_string()));
    m.insert("{{CURRENT_WEEKDAY}}".into(), json!(dt.format("%A").to_string()));
    m.insert("{{CURRENT_TIMEZONE}}".into(), json!("Asia/Calcutta"));
    m.insert("{{USER_LANGUAGE}}".into(), json!("en-US"));
    m
}

/// Build the signed completion URL with full query params (`_build_url`).
pub fn build_url(token: &str, user_id: &str, timestamp: u64, request_id: &str, chat_id: &str) -> String {
    let ts = timestamp.to_string();
    // Ordered to match the Python dict insertion order for byte-stable queries.
    let params: Vec<(&str, String)> = vec![
        ("version", "0.0.1".into()),
        ("platform", "web".into()),
        ("language", "en-GB".into()),
        ("languages", "en-GB,en-US,en".into()),
        ("timezone", "Asia/Calcutta".into()),
        ("cookie_enabled", "true".into()),
        ("screen_width", "1536".into()),
        ("screen_height", "864".into()),
        ("screen_resolution", "1536x864".into()),
        ("viewport_height", "710".into()),
        ("viewport_width", "936".into()),
        ("viewport_size", "936x710".into()),
        ("color_depth", "32".into()),
        ("pixel_ratio", "1.25".into()),
        ("search", "".into()),
        ("hash", "".into()),
        ("host", "chat.z.ai".into()),
        ("hostname", "chat.z.ai".into()),
        ("protocol", "https:".into()),
        ("referrer", "".into()),
        ("title", "Z.ai - Free AI Chatbot & Agent powered by GLM-5.1 & GLM-5".into()),
        ("timezone_offset", "-330".into()),
        ("is_mobile", "false".into()),
        ("is_touch", "false".into()),
        ("max_touch_points", "0".into()),
        ("browser_name", "Opera".into()),
        ("os_name", "Windows".into()),
        ("timestamp", ts.clone()),
        ("requestId", request_id.into()),
        ("user_id", user_id.into()),
        ("token", token.into()),
        ("user_agent", USER_AGENT.into()),
        ("current_url", format!("https://chat.z.ai/c/{chat_id}")),
        ("pathname", format!("/c/{chat_id}")),
        ("signature_timestamp", ts),
    ];
    let query = params
        .iter()
        .map(|(k, v)| format!("{}={}", urlencode(k), urlencode(v)))
        .collect::<Vec<_>>()
        .join("&");
    format!("{COMPLETIONS_URL}?{query}")
}

/// Minimal `application/x-www-form-urlencoded` component encoder.
fn urlencode(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for b in s.bytes() {
        match b {
            b'A'..=b'Z' | b'a'..=b'z' | b'0'..=b'9' | b'-' | b'_' | b'.' | b'~' => out.push(b as char),
            b' ' => out.push_str("%20"),
            _ => out.push_str(&format!("%{b:02X}")),
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn signature_prompt_picks_last_user() {
        let msgs = vec![
            Message::user("first"),
            Message::assistant("reply"),
            Message::user("second"),
        ];
        assert_eq!(signature_prompt(&msgs), "second");
    }

    #[test]
    fn body_has_required_fields() {
        let env = Env::fixed(1_700_000_000_000, 3);
        let body = build_body(&env, &[Message::user("hi")], "glm-5", "chat-1", "Guest");
        assert_eq!(body["model"], "glm-5");
        assert_eq!(body["signature_prompt"], "hi");
        assert_eq!(body["stream"], true);
        assert_eq!(body["features"]["enable_thinking"], true);
        assert!(body["variables"]["{{CURRENT_DATE}}"].is_string());
    }

    #[test]
    fn url_contains_signed_query_params() {
        let url = build_url("TOK", "u9", 1_700_000_000_000, "req-1", "chat-1");
        assert!(url.contains("token=TOK"));
        assert!(url.contains("requestId=req-1"));
        assert!(url.contains("signature_timestamp=1700000000000"));
        assert!(url.contains("current_url=https%3A%2F%2Fchat.z.ai%2Fc%2Fchat-1"));
    }
}
