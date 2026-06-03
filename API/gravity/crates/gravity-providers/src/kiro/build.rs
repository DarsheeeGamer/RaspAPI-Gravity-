//! Kiro (AWS Q Developer / CodeWhisperer) request building.
//!
//! Ports the `conversationState` assembly from `gravity/kiro/client.py` and the
//! `build_history` / `inject_system_prompt` / `convert_tools_to_kiro` helpers
//! from `gravity/kiro/transformers.py`. Faithful for standard conversations;
//! the multi-megabyte truncation knobs are applied conservatively.

use crate::util::uuid_v4;
use gravity_core::{Content, Env, Message, Role, ToolDefinition};
use serde_json::{json, Map, Value};

/// AWS region default.
pub const DEFAULT_REGION: &str = "us-east-1";
const ORIGIN: &str = "AI_EDITOR";
const CHAT_TRIGGER: &str = "MANUAL";
/// Q Developer API version header.
pub const API_VERSION: &str = "2024-07-31";
/// AWS SDK version (header).
pub const SDK_VERSION: &str = "3.738.0";
/// User-agent suffix.
pub const USER_AGENT: &str = "AmazonQ-For-CLI/2.0.1";

/// Resolve a Gravity model id to Kiro's internal model name.
pub fn resolve_model(model: &str) -> Option<&'static str> {
    Some(match model {
        "auto" => "auto",
        "claude-haiku-4-5" | "claude-haiku-4-5-thinking" => "claude-haiku-4.5",
        "claude-sonnet-4-5" | "claude-sonnet-4-5-thinking" => "claude-sonnet-4.5",
        "claude-sonnet-4-5-1m" | "claude-sonnet-4-5-1m-thinking" => "claude-sonnet-4.5-1m",
        "claude-sonnet-4-6" | "claude-sonnet-4-6-thinking" => "claude-sonnet-4.6",
        "claude-sonnet-4-6-1m" | "claude-sonnet-4-6-1m-thinking" => "claude-sonnet-4.6-1m",
        "claude-opus-4-5" | "claude-opus-4-5-thinking" => "claude-opus-4.5",
        "claude-opus-4-6" | "claude-opus-4-6-thinking" => "claude-opus-4.6",
        "claude-opus-4-6-1m" | "claude-opus-4-6-1m-thinking" => "claude-opus-4.6-1m",
        "claude-sonnet-4" => "claude-sonnet-4",
        "claude-3-7-sonnet" => "CLAUDE_3_7_SONNET_20250219_V1_0",
        "nova-swe" => "AGI_NOVA_SWE_V1_5",
        "gpt-oss-120b" => "OPENAI_GPT_OSS_120B_1_0",
        "qwen3-coder-480b" => "QWEN3_CODER_480B_A35B_1_0",
        "minimax-m2" => "MINIMAX_MINIMAX_M2",
        "kimi-k2-thinking" => "MOONSHOT_KIMI_K2_THINKING",
        _ => return None,
    })
}

/// Known Kiro models (for `list_models`).
pub fn models() -> Vec<String> {
    ["auto", "claude-sonnet-4-5", "claude-sonnet-4-6", "claude-opus-4-6", "gpt-oss-120b"]
        .iter()
        .map(|s| s.to_string())
        .collect()
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

fn user_input(content: &str, model: &str) -> Value {
    json!({ "content": content, "modelId": model, "origin": ORIGIN })
}

/// Build the `history` array from all but the last message (`build_history`).
fn build_history(messages: &[Message], model: &str) -> Vec<Value> {
    let mut history: Vec<Value> = Vec::new();
    let push_continue_if_user = |h: &mut Vec<Value>| {
        if h.last().map(|p| p.get("userInputMessage").is_some()).unwrap_or(false) {
            h.push(json!({ "assistantResponseMessage": { "content": "Continue" } }));
        }
    };

    for msg in messages.iter().take(messages.len().saturating_sub(1)) {
        match msg.role {
            Role::User | Role::System => {
                push_continue_if_user(&mut history);
                history.push(json!({ "userInputMessage": user_input(&text_of(msg), model) }));
            }
            Role::Tool => {
                let tr = msg.tool_result.as_ref();
                let content = tr.map(|t| value_to_text(&t.content)).unwrap_or_default();
                let tool_use_id = tr.map(|t| t.tool_call_id.to_string()).unwrap_or_default();
                push_continue_if_user(&mut history);
                history.push(json!({
                    "userInputMessage": {
                        "content": "Tool results provided.",
                        "modelId": model,
                        "origin": ORIGIN,
                        "userInputMessageContext": {
                            "toolResults": [{
                                "content": [{ "text": content }],
                                "status": "success",
                                "toolUseId": tool_use_id,
                            }]
                        }
                    }
                }));
            }
            Role::Assistant => {
                let mut arm = Map::new();
                arm.insert("content".into(), json!(text_of(msg)));
                if !msg.tool_calls.is_empty() {
                    let tool_uses: Vec<Value> = msg
                        .tool_calls
                        .iter()
                        .map(|tc| json!({ "input": tc.arguments, "name": tc.name, "toolUseId": tc.id }))
                        .collect();
                    arm.insert("toolUses".into(), Value::Array(tool_uses));
                }
                let empty = arm.get("content").map(|c| c.as_str() == Some("")).unwrap_or(true)
                    && !arm.contains_key("toolUses");
                if empty {
                    continue;
                }
                // Merge into a preceding assistant message if present.
                if let Some(prev) = history.last_mut().and_then(|p| p.get_mut("assistantResponseMessage")) {
                    merge_assistant(prev, &arm);
                } else {
                    history.push(json!({ "assistantResponseMessage": Value::Object(arm) }));
                }
            }
        }
    }
    history
}

fn merge_assistant(prev: &mut Value, arm: &Map<String, Value>) {
    if let Some(new_content) = arm.get("content").and_then(Value::as_str).filter(|s| !s.is_empty()) {
        let prev_content = prev.get("content").and_then(Value::as_str).unwrap_or("").to_owned();
        let merged = if prev_content.is_empty() {
            new_content.to_owned()
        } else {
            format!("{prev_content}\n\n{new_content}")
        };
        prev["content"] = json!(merged);
    }
    if let Some(Value::Array(new_uses)) = arm.get("toolUses") {
        let existing = prev
            .get("toolUses")
            .and_then(Value::as_array)
            .cloned()
            .unwrap_or_default();
        let mut all = existing;
        all.extend(new_uses.clone());
        prev["toolUses"] = Value::Array(all);
    }
}

/// Prepend a system prompt onto the first user message of `history`
/// (`inject_system_prompt`).
fn inject_system_prompt(history: &mut Vec<Value>, system_prompt: &str, model: &str) {
    if system_prompt.is_empty() {
        return;
    }
    for item in history.iter_mut() {
        if let Some(uim) = item.get_mut("userInputMessage") {
            let old = uim.get("content").and_then(Value::as_str).unwrap_or("").to_owned();
            uim["content"] = json!(if old.is_empty() {
                system_prompt.to_owned()
            } else {
                format!("{system_prompt}\n\n{old}")
            });
            return;
        }
    }
    history.insert(0, json!({ "userInputMessage": user_input(system_prompt, model) }));
}

/// Convert tools to Kiro `toolSpecification` form (`convert_tools_to_kiro`).
fn convert_tools(tools: &[ToolDefinition]) -> Vec<Value> {
    tools
        .iter()
        .map(|t| {
            json!({
                "toolSpecification": {
                    "name": t.name,
                    "description": t.description,
                    "inputSchema": { "json": t.input_schema },
                }
            })
        })
        .collect()
}

fn value_to_text(v: &Value) -> String {
    match v {
        Value::String(s) => s.clone(),
        other => other.to_string(),
    }
}

/// Build the full `generateAssistantResponse` request body.
///
/// Returns `(body, resolved_model)`.
pub fn build_request(
    env: &Env,
    messages: &[Message],
    tools: &[ToolDefinition],
    system_prompt: Option<&str>,
    model: &str,
    profile_arn: Option<&str>,
) -> Option<(Value, &'static str)> {
    let resolved = resolve_model(model)?;
    if messages.is_empty() {
        return None;
    }
    let mut history = build_history(messages, resolved);
    let sys = system_prompt.unwrap_or("").trim();
    inject_system_prompt(&mut history, sys, resolved);

    let current = messages.last().unwrap();
    let mut current_content = text_of(current);
    let mut current_tool_results: Vec<Value> = Vec::new();

    // Collect trailing tool results after the last assistant turn.
    if current.role == Role::Tool {
        for m in messages.iter().rev() {
            match m.role {
                Role::Tool => {
                    if let Some(tr) = &m.tool_result {
                        current_tool_results.insert(
                            0,
                            json!({
                                "content": [{ "text": value_to_text(&tr.content) }],
                                "status": "success",
                                "toolUseId": tr.tool_call_id,
                            }),
                        );
                    }
                }
                Role::Assistant => break,
                _ => {}
            }
        }
        current_content = if current_tool_results.is_empty() {
            "Continue".into()
        } else {
            "Tool results provided.".into()
        };
    }
    if current_content.is_empty() {
        current_content = "Continue".into();
    }

    let mut uim = Map::new();
    uim.insert("content".into(), json!(current_content));
    uim.insert("modelId".into(), json!(resolved));
    uim.insert("origin".into(), json!(ORIGIN));

    let mut ctx = Map::new();
    if !current_tool_results.is_empty() {
        ctx.insert("toolResults".into(), Value::Array(current_tool_results));
    }
    if !tools.is_empty() {
        ctx.insert("tools".into(), Value::Array(convert_tools(tools)));
    }
    if !ctx.is_empty() {
        uim.insert("userInputMessageContext".into(), Value::Object(ctx));
    }

    let mut conv_state = Map::new();
    conv_state.insert("chatTriggerType".into(), json!(CHAT_TRIGGER));
    conv_state.insert("conversationId".into(), json!(uuid_v4(env)));
    conv_state.insert(
        "currentMessage".into(),
        json!({ "userInputMessage": Value::Object(uim) }),
    );
    if !history.is_empty() {
        conv_state.insert("history".into(), Value::Array(history));
    }

    let mut body = Map::new();
    body.insert("conversationState".into(), Value::Object(conv_state));
    if let Some(arn) = profile_arn {
        body.insert("profileArn".into(), json!(arn));
    }
    Some((Value::Object(body), resolved))
}

/// Build the regional endpoint URL.
pub fn url(region: &str) -> String {
    format!("https://q.{region}.amazonaws.com/generateAssistantResponse")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn resolves_models() {
        assert_eq!(resolve_model("claude-sonnet-4-5"), Some("claude-sonnet-4.5"));
        assert_eq!(resolve_model("gpt-oss-120b"), Some("OPENAI_GPT_OSS_120B_1_0"));
        assert!(resolve_model("nope").is_none());
    }

    #[test]
    fn builds_conversation_state_with_history_and_system() {
        let env = Env::fixed(0, 1);
        let msgs = vec![
            Message::user("first question"),
            Message::assistant("first answer"),
            Message::user("second question"),
        ];
        let (body, model) =
            build_request(&env, &msgs, &[], Some("You are helpful."), "claude-sonnet-4-5", None).unwrap();
        assert_eq!(model, "claude-sonnet-4.5");
        let cs = &body["conversationState"];
        assert_eq!(cs["chatTriggerType"], "MANUAL");
        assert_eq!(cs["currentMessage"]["userInputMessage"]["content"], "second question");
        // history holds the first Q/A; system prompt prepended to first user msg
        let hist = cs["history"].as_array().unwrap();
        assert!(hist[0]["userInputMessage"]["content"]
            .as_str()
            .unwrap()
            .starts_with("You are helpful."));
        assert_eq!(hist[1]["assistantResponseMessage"]["content"], "first answer");
    }

    #[test]
    fn tools_become_tool_specifications() {
        let env = Env::fixed(0, 1);
        let tools = vec![ToolDefinition {
            name: "get_weather".into(),
            description: "weather".into(),
            input_schema: serde_json::from_str(r#"{"type":"object"}"#).unwrap(),
        }];
        let (body, _) =
            build_request(&env, &[Message::user("hi")], &tools, None, "auto", None).unwrap();
        let spec = &body["conversationState"]["currentMessage"]["userInputMessage"]
            ["userInputMessageContext"]["tools"][0]["toolSpecification"];
        assert_eq!(spec["name"], "get_weather");
        assert_eq!(spec["inputSchema"]["json"]["type"], "object");
    }
}
