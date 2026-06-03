//! OpenAI Chat Completions wire types.
//!
//! Mirrors `gravity/providers/schema.py`'s `unified_to_oai` / `oai_to_unified`.
//! The **request** side borrows from the canonical [`ChatRequest`] (`&str` /
//! [`Cow`]) so building a payload allocates only where it must (e.g. when
//! concatenating multipart text or stringifying tool arguments).

use gravity_core::{ChatRequest, Content, ContentPart, Message, Role, ToolCall};
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};
use std::borrow::Cow;

// ── Request (borrowed) ──────────────────────────────────────────────────────

/// An OpenAI chat-completions request, borrowing from the canonical request.
#[derive(Debug, Serialize)]
pub struct OaiRequest<'a> {
    /// Model name (bare, without the `provider/` prefix).
    pub model: &'a str,
    /// Conversation messages.
    pub messages: Vec<OaiMessage<'a>>,
    /// Advertised tools.
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub tools: Vec<OaiTool<'a>>,
    /// Sampling temperature.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f32>,
    /// Max completion tokens.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_tokens: Option<u32>,
    /// Nucleus sampling.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_p: Option<f32>,
    /// Stop sequences.
    #[serde(skip_serializing_if = "<[_]>::is_empty")]
    pub stop: &'a [String],
    /// Tool selection mode (`"auto"` when tools are present).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_choice: Option<&'static str>,
    /// Structured-output constraint (`{"type":"json_schema", ...}`).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub response_format: Option<Value>,
    /// Deterministic sampling seed.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub seed: Option<i64>,
    /// Frequency penalty.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub frequency_penalty: Option<f32>,
    /// Presence penalty.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub presence_penalty: Option<f32>,
    /// Whether to stream.
    pub stream: bool,
}

/// OpenAI content — either a plain string or an array of typed parts (for multimodal).
#[derive(Debug, Serialize)]
#[serde(untagged)]
pub enum OaiContent<'a> {
    /// Plain text (most messages).
    Text(Cow<'a, str>),
    /// Multimodal parts array (text + image_url).
    Parts(Vec<OaiContentPart<'a>>),
}

/// One part inside a multimodal content array.
#[derive(Debug, Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum OaiContentPart<'a> {
    /// Text segment.
    Text { text: Cow<'a, str> },
    /// Image URL segment.
    ImageUrl { image_url: OaiImageUrl<'a> },
}

/// Image URL payload.
#[derive(Debug, Serialize)]
pub struct OaiImageUrl<'a> {
    /// `https://…` or `data:<mime>;base64,…`
    pub url: Cow<'a, str>,
    /// `"auto"` | `"low"` | `"high"`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub detail: Option<&'static str>,
}

/// One OpenAI message.
#[derive(Debug, Serialize)]
pub struct OaiMessage<'a> {
    /// Role string.
    pub role: &'a str,
    /// Text or multimodal content.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub content: Option<OaiContent<'a>>,
    /// Assistant tool calls.
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub tool_calls: Vec<OaiToolCall<'a>>,
    /// For `role = "tool"`: the originating call id.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_call_id: Option<&'a str>,
}

/// An assistant tool call in the request transcript.
#[derive(Debug, Serialize)]
pub struct OaiToolCall<'a> {
    /// Call id.
    pub id: &'a str,
    /// Always `"function"`.
    #[serde(rename = "type")]
    pub kind: &'static str,
    /// The function call.
    pub function: OaiFunctionCall<'a>,
}

/// The function name + JSON-string arguments of a tool call.
#[derive(Debug, Serialize)]
pub struct OaiFunctionCall<'a> {
    /// Function name.
    pub name: &'a str,
    /// Arguments serialized to a JSON string (OpenAI wire requirement).
    pub arguments: String,
}

/// A tool advertised to the model.
#[derive(Debug, Serialize)]
pub struct OaiTool<'a> {
    /// Always `"function"`.
    #[serde(rename = "type")]
    pub kind: &'static str,
    /// The function schema.
    pub function: OaiToolFunction<'a>,
}

/// A function tool's schema.
#[derive(Debug, Serialize)]
pub struct OaiToolFunction<'a> {
    /// Function name.
    pub name: &'a str,
    /// Description.
    pub description: &'a str,
    /// JSON-Schema parameters object.
    pub parameters: &'a Map<String, Value>,
}

/// Build a borrowed [`OaiRequest`] from the canonical request, prepending the
/// system prompt if present.
pub fn build_request<'a>(req: &'a ChatRequest, stream: bool) -> OaiRequest<'a> {
    let mut messages = Vec::with_capacity(req.messages.len() + 1);
    if let Some(sp) = &req.system_prompt {
        messages.push(OaiMessage {
            role: "system",
            content: Some(OaiContent::Text(Cow::Borrowed(sp.as_str()))),
            tool_calls: Vec::new(),
            tool_call_id: None,
        });
    }
    for m in &req.messages {
        messages.push(to_oai_message(m));
    }
    let tools: Vec<OaiTool<'a>> = req
        .tools
        .iter()
        .map(|t| OaiTool {
            kind: "function",
            function: OaiToolFunction {
                name: &t.name,
                description: &t.description,
                parameters: &t.input_schema,
            },
        })
        .collect();
    let tool_choice = if tools.is_empty() { None } else { Some("auto") };
    let response_format = match &req.response_format {
        gravity_core::ResponseFormat::JsonSchema { json_schema } => Some(serde_json::json!({
            "type": "json_schema",
            "json_schema": { "name": "response", "schema": json_schema, "strict": false }
        })),
        gravity_core::ResponseFormat::Free => None,
    };
    OaiRequest {
        model: &req.model.model,
        messages,
        tools,
        temperature: req.sampling.temperature,
        max_tokens: req.sampling.max_tokens,
        top_p: req.sampling.top_p,
        stop: &req.sampling.stop,
        tool_choice,
        response_format,
        seed: req.sampling.seed,
        frequency_penalty: req.sampling.frequency_penalty,
        presence_penalty: req.sampling.presence_penalty,
        stream,
    }
}

fn to_oai_message(m: &Message) -> OaiMessage<'_> {
    let role = match m.role {
        Role::System => "system",
        Role::User => "user",
        Role::Assistant => "assistant",
        Role::Tool => "tool",
    };
    // Tool result message.
    if let (Role::Tool, Some(tr)) = (m.role, &m.tool_result) {
        return OaiMessage {
            role,
            content: Some(OaiContent::Text(content_string(&tr.content))),
            tool_calls: Vec::new(),
            tool_call_id: Some(&tr.tool_call_id),
        };
    }
    let content = match &m.content {
        Some(Content::Text(s)) => Some(OaiContent::Text(Cow::Borrowed(s.as_str()))),
        Some(Content::Parts(parts)) => {
            let oai_parts: Vec<OaiContentPart<'_>> = parts
                .iter()
                .filter_map(|p| match p {
                    ContentPart::Text { text } => {
                        Some(OaiContentPart::Text { text: Cow::Borrowed(text.as_str()) })
                    }
                    ContentPart::ImageUrl { image_url } => Some(OaiContentPart::ImageUrl {
                        image_url: OaiImageUrl {
                            url: Cow::Borrowed(image_url.url.as_str()),
                            detail: image_url.detail.as_deref().map(|d| match d {
                                "low" => "low",
                                "high" => "high",
                                _ => "auto",
                            }),
                        },
                    }),
                    ContentPart::InputFile { .. } | ContentPart::Json { .. } => None,
                })
                .collect();
            if oai_parts.is_empty() { None } else { Some(OaiContent::Parts(oai_parts)) }
        }
        None => None,
    };
    let tool_calls = m.tool_calls.iter().map(to_oai_tool_call).collect();
    OaiMessage { role, content, tool_calls, tool_call_id: None }
}

fn to_oai_tool_call(tc: &ToolCall) -> OaiToolCall<'_> {
    OaiToolCall {
        id: &tc.id,
        kind: "function",
        function: OaiFunctionCall {
            name: &tc.name,
            arguments: Value::Object(tc.arguments.clone()).to_string(),
        },
    }
}

fn content_string(v: &Value) -> Cow<'_, str> {
    match v {
        Value::String(s) => Cow::Borrowed(s.as_str()),
        other => Cow::Owned(other.to_string()),
    }
}

// ── Response (owned) ────────────────────────────────────────────────────────

/// A decoded OpenAI chat-completions response.
#[derive(Debug, Deserialize)]
pub struct OaiResponse {
    /// Response id.
    #[serde(default)]
    pub id: Option<String>,
    /// Model echoed by the server (parsed for completeness; the canonical
    /// model name is taken from the request).
    #[serde(default)]
    #[allow(dead_code)]
    pub model: Option<String>,
    /// Choices (we read the first).
    #[serde(default)]
    pub choices: Vec<OaiChoice>,
    /// Token usage.
    #[serde(default)]
    pub usage: Option<OaiUsage>,
}

/// One completion choice.
#[derive(Debug, Deserialize)]
pub struct OaiChoice {
    /// The assistant message.
    #[serde(default)]
    pub message: Option<OaiRespMessage>,
    /// Streaming delta.
    #[serde(default)]
    pub delta: Option<OaiRespMessage>,
    /// Finish reason.
    #[serde(default)]
    pub finish_reason: Option<String>,
}

/// A response message / delta.
#[derive(Debug, Default, Deserialize)]
pub struct OaiRespMessage {
    /// Text content.
    #[serde(default)]
    pub content: Option<String>,
    /// Tool calls.
    #[serde(default)]
    pub tool_calls: Vec<OaiRespToolCall>,
}

/// A tool call in a response.
#[derive(Debug, Deserialize)]
pub struct OaiRespToolCall {
    /// Call id.
    #[serde(default)]
    pub id: Option<String>,
    /// Function payload.
    pub function: OaiRespFunction,
}

/// The function name + JSON-string args of a response tool call.
#[derive(Debug, Deserialize)]
pub struct OaiRespFunction {
    /// Function name.
    #[serde(default)]
    pub name: Option<String>,
    /// JSON-string arguments.
    #[serde(default)]
    pub arguments: Option<String>,
}

/// Token usage block.
#[derive(Debug, Default, Deserialize)]
pub struct OaiUsage {
    /// Prompt tokens.
    #[serde(default)]
    pub prompt_tokens: Option<u32>,
    /// Completion tokens.
    #[serde(default)]
    pub completion_tokens: Option<u32>,
    /// Total tokens.
    #[serde(default)]
    pub total_tokens: Option<u32>,
}

#[cfg(test)]
mod tests {
    use super::*;
    use gravity_core::{ModelId, ToolDefinition};

    #[test]
    fn request_borrows_plain_text() {
        let mut req = ChatRequest::new(
            ModelId::parse("groq/llama").unwrap(),
            vec![Message::user("hi"), Message::assistant("hello")],
        );
        req.system_prompt = Some("be nice".into());
        let oai = build_request(&req, false);
        let json = serde_json::to_value(&oai).unwrap();
        assert_eq!(json["model"], "llama");
        assert_eq!(json["messages"][0]["role"], "system");
        assert_eq!(json["messages"][0]["content"], "be nice");
        assert_eq!(json["messages"][1]["content"], "hi");
        assert_eq!(json["stream"], false);
        assert!(json.get("tools").is_none());
    }

    #[test]
    fn tools_serialize_as_functions() {
        let mut req = ChatRequest::new(
            ModelId::parse("groq/x").unwrap(),
            vec![Message::user("q")],
        );
        req.tools = vec![ToolDefinition {
            name: "get_weather".into(),
            description: "weather".into(),
            input_schema: serde_json::from_str(r#"{"type":"object"}"#).unwrap(),
        }];
        let json = serde_json::to_value(build_request(&req, true)).unwrap();
        assert_eq!(json["tools"][0]["type"], "function");
        assert_eq!(json["tools"][0]["function"]["name"], "get_weather");
        assert_eq!(json["stream"], true);
    }

    #[test]
    fn response_parses_choices_and_usage() {
        let raw = r#"{
            "id":"cmpl-1","model":"llama",
            "choices":[{"message":{"content":"hi there"},"finish_reason":"stop"}],
            "usage":{"prompt_tokens":5,"completion_tokens":2,"total_tokens":7}
        }"#;
        let resp: OaiResponse = serde_json::from_str(raw).unwrap();
        assert_eq!(resp.choices[0].message.as_ref().unwrap().content.as_deref(), Some("hi there"));
        assert_eq!(resp.usage.unwrap().total_tokens, Some(7));
    }
}
