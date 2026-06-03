//! Anthropic Messages API wire types and conversion.

use gravity_core::{
    ChatRequest, ChatResponse, Content, Message, Metadata, Role, ToolCall, Usage,
};
use serde::{Deserialize, Serialize};
use serde_json::{json, Map, Value};

/// Build the Messages-API request body from the canonical request.
///
/// `system` is hoisted to the top level; only `user`/`assistant` turns go in
/// `messages`; tool results become `tool_result` content blocks on a user turn.
pub fn build_request(req: &ChatRequest, default_max_tokens: u32, stream: bool) -> Value {
    let mut body = Map::new();
    body.insert("model".into(), json!(req.model.model));
    body.insert(
        "max_tokens".into(),
        json!(req.sampling.max_tokens.unwrap_or(default_max_tokens)),
    );
    if let Some(sp) = &req.system_prompt {
        body.insert("system".into(), json!(sp));
    }
    if let Some(t) = req.sampling.temperature {
        body.insert("temperature".into(), json!(t));
    }
    if let Some(p) = req.sampling.top_p {
        body.insert("top_p".into(), json!(p));
    }
    // Anthropic doesn't have seed/frequency_penalty/presence_penalty on the
    // Messages API; we silently drop them (they're OAI extensions).
    body.insert("messages".into(), json!(build_messages(&req.messages)));
    if !req.tools.is_empty() {
        let tools: Vec<Value> = req
            .tools
            .iter()
            .map(|t| json!({ "name": t.name, "description": t.description, "input_schema": t.input_schema }))
            .collect();
        body.insert("tools".into(), Value::Array(tools));
    }
    body.insert("stream".into(), json!(stream));
    Value::Object(body)
}

fn build_messages(messages: &[Message]) -> Vec<Value> {
    let mut out = Vec::with_capacity(messages.len());
    for m in messages {
        match m.role {
            Role::System => {} // hoisted to top-level `system`
            Role::Tool => {
                if let Some(tr) = &m.tool_result {
                    out.push(json!({
                        "role": "user",
                        "content": [{
                            "type": "tool_result",
                            "tool_use_id": tr.tool_call_id,
                            "content": value_to_text(&tr.content),
                            "is_error": tr.is_error,
                        }]
                    }));
                }
            }
            Role::User => out.push(json!({ "role": "user", "content": text_of(m) })),
            Role::Assistant => {
                if m.tool_calls.is_empty() {
                    out.push(json!({ "role": "assistant", "content": text_of(m) }));
                } else {
                    let mut blocks: Vec<Value> = Vec::new();
                    let text = text_of(m);
                    if !text.is_empty() {
                        blocks.push(json!({ "type": "text", "text": text }));
                    }
                    for tc in &m.tool_calls {
                        blocks.push(json!({
                            "type": "tool_use",
                            "id": tc.id,
                            "name": tc.name,
                            "input": tc.arguments,
                        }));
                    }
                    out.push(json!({ "role": "assistant", "content": blocks }));
                }
            }
        }
    }
    out
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

fn value_to_text(v: &Value) -> String {
    match v {
        Value::String(s) => s.clone(),
        other => other.to_string(),
    }
}

/// A decoded Messages-API response.
#[derive(Debug, Deserialize)]
pub struct AnthropicResponse {
    /// Message id.
    #[serde(default)]
    pub id: Option<String>,
    /// Content blocks.
    #[serde(default)]
    pub content: Vec<ContentBlock>,
    /// Stop reason.
    #[serde(default)]
    pub stop_reason: Option<String>,
    /// Usage.
    #[serde(default)]
    pub usage: Option<AnthropicUsage>,
}

/// One response content block.
#[derive(Debug, Deserialize, Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ContentBlock {
    /// A text block.
    Text {
        /// The text.
        text: String,
    },
    /// A tool-use block.
    ToolUse {
        /// Tool-use id.
        id: String,
        /// Tool name.
        name: String,
        /// Input object.
        input: Value,
    },
    /// Any other block type (ignored).
    #[serde(other)]
    Other,
}

/// Usage block.
#[derive(Debug, Default, Deserialize)]
pub struct AnthropicUsage {
    /// Input tokens.
    #[serde(default)]
    pub input_tokens: Option<u32>,
    /// Output tokens.
    #[serde(default)]
    pub output_tokens: Option<u32>,
}

/// Convert a decoded response to the canonical [`ChatResponse`].
pub fn to_chat_response(model_full: &str, resp: AnthropicResponse) -> ChatResponse {
    let mut text = String::new();
    let mut tool_calls: Vec<ToolCall> = Vec::new();
    for block in resp.content {
        match block {
            ContentBlock::Text { text: t } => text.push_str(&t),
            ContentBlock::ToolUse { id, name, input } => tool_calls.push(ToolCall {
                id: id.into(),
                name: name.into(),
                arguments: input.as_object().cloned().unwrap_or_default(),
            }),
            ContentBlock::Other => {}
        }
    }
    let message = if tool_calls.is_empty() {
        Message::assistant(text)
    } else {
        Message::assistant_with_tools(Some(Content::Text(text)), tool_calls)
    };
    let usage = resp
        .usage
        .map(|u| Usage {
            input_tokens: u.input_tokens,
            output_tokens: u.output_tokens,
            total_tokens: None,
        })
        .unwrap_or_default();
    ChatResponse {
        model: model_full.into(),
        message,
        usage,
        tool_calls_executed: Vec::new(),
        provider_response_id: resp.id.map(Into::into),
        finish_reason: resp.stop_reason.map(Into::into),
        metadata: Metadata::new(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use gravity_core::ModelId;

    #[test]
    fn request_hoists_system_and_requires_max_tokens() {
        let mut req = ChatRequest::new(
            ModelId::parse("anthropic_api/claude-sonnet-4-5").unwrap(),
            vec![Message::user("hi")],
        );
        req.system_prompt = Some("be brief".into());
        let body = build_request(&req, 4096, false);
        assert_eq!(body["system"], "be brief");
        assert_eq!(body["max_tokens"], 4096);
        assert_eq!(body["model"], "claude-sonnet-4-5");
        // system is NOT in the messages array
        assert_eq!(body["messages"].as_array().unwrap().len(), 1);
        assert_eq!(body["messages"][0]["role"], "user");
    }

    #[test]
    fn response_parses_text_and_tool_use() {
        let raw = r#"{
            "id":"msg_1",
            "content":[
                {"type":"text","text":"Let me check. "},
                {"type":"tool_use","id":"tu1","name":"get_weather","input":{"city":"NYC"}}
            ],
            "stop_reason":"tool_use",
            "usage":{"input_tokens":10,"output_tokens":8}
        }"#;
        let resp: AnthropicResponse = serde_json::from_str(raw).unwrap();
        let chat = to_chat_response("anthropic_api/x", resp);
        assert_eq!(chat.text(), "Let me check. ");
        assert_eq!(chat.message.tool_calls.len(), 1);
        assert_eq!(chat.message.tool_calls[0].arguments["city"], "NYC");
        assert_eq!(chat.usage.input_tokens, Some(10));
        assert_eq!(chat.finish_reason.as_deref(), Some("tool_use"));
    }
}
