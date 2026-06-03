//! Pure JSON ⇄ Gravity-Schema conversion for the C ABI.
//!
//! The C ABI is a flat JSON-string boundary identical to the contract the old
//! `_ffi_bridge.py` served, so existing C/Go/Python callers keep working. This
//! module holds the (safe, unit-tested) parsing/serialization; `lib.rs` only
//! adds the `extern "C"` shell and memory management.

use gravity_client::{build_transient, ResolvedAccount, TransientOptions};
use gravity_core::{
    ChatRequest, ChatResponse, Content, Error, Message, Metadata, ModelId, ProviderName, Result,
    Role, Sampling, ToolDefinition,
};
use gravity_providers::Registry;
use serde_json::{json, Map, Value};
use std::str::FromStr;

/// Parse a `gravity_chat` request JSON into a transient account and request.
pub fn parse_chat_request(
    registry: &Registry,
    body: &str,
) -> Result<(ResolvedAccount, ChatRequest)> {
    let v: Value = serde_json::from_str(body)
        .map_err(|e| Error::validation(format!("invalid request JSON: {e}"), "ffi_bad_json"))?;

    let provider_str = string_field(&v, "provider")
        .ok_or_else(|| Error::validation("missing 'provider'", "ffi_missing_provider"))?;
    let provider = ProviderName::from_str(provider_str)
        .map_err(|_| Error::validation(format!("unknown provider {provider_str:?}"), "ffi_unknown_provider"))?;

    let model_name = string_field(&v, "model")
        .ok_or_else(|| Error::validation("missing 'model'", "ffi_missing_model"))?;
    let model = ModelId::new(provider, model_name)
        .map_err(|e| Error::validation(e.to_string(), "ffi_bad_model"))?;

    let messages = parse_messages(v.get("messages"))?;

    // Credential + transport options.
    let secret = first_string(&v, &["api_key", "session_key", "token", "access_token"]).unwrap_or_default();
    let kind = if matches!(provider, ProviderName::Claude | ProviderName::Chatgpt) {
        Some("session_cookie".to_owned())
    } else {
        Some("api_key".to_owned())
    };
    let opts = TransientOptions {
        secret: secret.clone(),
        kind,
        base_url: string_field(&v, "base_url").map(str::to_owned),
        impersonate: string_field(&v, "impersonate").map(str::to_owned),
        proxy_url: string_field(&v, "proxy").or_else(|| string_field(&v, "proxy_url")).map(str::to_owned),
        timeout_ms: v.get("timeout_ms").and_then(Value::as_u64).map(|n| n as u32),
        device_id: string_field(&v, "device_id").map(str::to_owned),
        org_id: string_field(&v, "org_id").map(str::to_owned),
        extra: Metadata::new(),
    };
    let account = build_transient(registry, provider, opts)?;

    let mut request = ChatRequest::new(model, messages);
    request.system_prompt = string_field(&v, "system_prompt").map(str::to_owned);
    request.sampling = Sampling {
        temperature: v.get("temperature").and_then(Value::as_f64).map(|n| n as f32),
        max_tokens: v.get("max_tokens").and_then(Value::as_u64).map(|n| n as u32),
        top_p: v.get("top_p").and_then(Value::as_f64).map(|n| n as f32),
        stop: Vec::new(),
        seed: v.get("seed").and_then(Value::as_i64),
        frequency_penalty: v.get("frequency_penalty").and_then(Value::as_f64).map(|n| n as f32),
        presence_penalty: v.get("presence_penalty").and_then(Value::as_f64).map(|n| n as f32),
    };
    request.tools = parse_tools(v.get("tools"));
    request.stream = v.get("stream").and_then(Value::as_bool).unwrap_or(false);

    Ok((account, request))
}

fn parse_messages(value: Option<&Value>) -> Result<Vec<Message>> {
    let arr = value
        .and_then(Value::as_array)
        .ok_or_else(|| Error::validation("missing 'messages' array", "ffi_missing_messages"))?;
    let mut out = Vec::with_capacity(arr.len());
    for m in arr {
        let role = match string_field(m, "role").unwrap_or("user") {
            "system" => Role::System,
            "assistant" => Role::Assistant,
            "tool" => Role::Tool,
            _ => Role::User,
        };
        let content = string_field(m, "content").unwrap_or("").to_owned();
        out.push(Message {
            role,
            content: Some(Content::Text(content)),
            name: None,
            tool_calls: Default::default(),
            tool_result: None,
            metadata: Metadata::new(),
        });
    }
    Ok(out)
}

fn parse_tools(value: Option<&Value>) -> Vec<ToolDefinition> {
    let Some(arr) = value.and_then(Value::as_array) else {
        return Vec::new();
    };
    arr.iter()
        .filter_map(|t| {
            let name = string_field(t, "name")?.to_owned();
            Some(ToolDefinition {
                name: name.into(),
                description: string_field(t, "description").unwrap_or("").into(),
                input_schema: t
                    .get("input_schema")
                    .or_else(|| t.get("parameters"))
                    .and_then(Value::as_object)
                    .cloned()
                    .unwrap_or_default(),
            })
        })
        .collect()
}

/// Serialize a [`ChatResponse`] into the flat FFI response JSON.
pub fn response_to_json(resp: &ChatResponse) -> Value {
    let tool_calls: Vec<Value> = resp
        .message
        .tool_calls
        .iter()
        .map(|tc| json!({ "id": tc.id, "name": tc.name, "arguments": tc.arguments }))
        .collect();
    json!({
        "content": resp.text(),
        "model": resp.model,
        "finish_reason": resp.finish_reason.as_deref(),
        "tool_calls": tool_calls,
        "input_tokens": resp.usage.input_tokens,
        "output_tokens": resp.usage.output_tokens,
    })
}

/// Build the `gravity_list_providers` JSON payload from a registry.
pub fn providers_list_json(registry: &Registry) -> Value {
    let arr: Vec<Value> = registry
        .iter_capabilities()
        .map(|(name, caps)| {
            json!({
                "name": name.as_str(),
                "tool_support": tool_support_str(caps.tool_support),
                "structured_output": structured_str(caps.structured_output),
                "supports_system_prompt": caps.supports_system_prompt,
            })
        })
        .collect();
    Value::Array(arr)
}

/// Render an error as the `{"error": ...}` envelope the ABI documents.
pub fn error_json(err: &Error) -> Value {
    json!({ "error": err.to_string() })
}

fn tool_support_str(m: gravity_core::ToolSupportMode) -> &'static str {
    use gravity_core::ToolSupportMode::*;
    match m {
        Native => "native",
        JsonEmulated => "json_emulated",
        None => "none",
    }
}

fn structured_str(m: gravity_core::StructuredOutputMode) -> &'static str {
    use gravity_core::StructuredOutputMode::*;
    match m {
        Native => "native",
        JsonInstruction => "json_instruction",
        None => "none",
    }
}

fn string_field<'a>(v: &'a Value, key: &str) -> Option<&'a str> {
    v.get(key).and_then(Value::as_str)
}

fn first_string(v: &Value, keys: &[&str]) -> Option<String> {
    keys.iter().find_map(|k| string_field(v, k).map(str::to_owned))
}

#[allow(dead_code)]
fn empty_map() -> Map<String, Value> {
    Map::new()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_basic_chat_request() {
        let body = r#"{
            "provider": "claude",
            "model": "claude-sonnet-4-6",
            "api_key": "sk-ant-sid-xyz",
            "device_id": "dev-1",
            "system_prompt": "Be brief.",
            "messages": [{"role":"user","content":"hello"}],
            "temperature": 0.7, "max_tokens": 100
        }"#;
        let (acct, req) = parse_chat_request(gravity_providers::builtin(), body).unwrap();
        assert_eq!(acct.account.provider, ProviderName::Claude);
        assert_eq!(acct.secret, "sk-ant-sid-xyz");
        assert_eq!(&*acct.account.auth.kind, "session_cookie");
        assert_eq!(acct.account.auth.device_id.as_deref(), Some("dev-1"));
        assert_eq!(req.model.full_name(), "claude/claude-sonnet-4-6");
        assert_eq!(req.system_prompt.as_deref(), Some("Be brief."));
        assert_eq!(req.sampling.temperature, Some(0.7));
        assert_eq!(req.messages.len(), 1);
    }

    #[test]
    fn response_serialization_shape() {
        let resp = ChatResponse {
            model: "claude/x".into(),
            message: Message::assistant("hi there"),
            usage: gravity_core::Usage { input_tokens: Some(2), output_tokens: Some(3), total_tokens: None },
            tool_calls_executed: vec![],
            provider_response_id: None,
            finish_reason: Some("stop".into()),
            metadata: Metadata::new(),
        };
        let j = response_to_json(&resp);
        assert_eq!(j["content"], "hi there");
        assert_eq!(j["finish_reason"], "stop");
        assert_eq!(j["input_tokens"], 2);
        assert!(j["tool_calls"].as_array().unwrap().is_empty());
    }

    #[test]
    fn providers_list_includes_claude() {
        let j = providers_list_json(gravity_providers::builtin());
        let names: Vec<&str> = j.as_array().unwrap().iter().map(|p| p["name"].as_str().unwrap()).collect();
        assert!(names.contains(&"claude"));
    }

    #[test]
    fn unknown_provider_errors() {
        let body = r#"{"provider":"nope","model":"x","messages":[]}"#;
        assert!(parse_chat_request(gravity_providers::builtin(), body).is_err());
    }
}
