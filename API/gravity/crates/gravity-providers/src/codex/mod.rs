//! Codex provider (`ProviderName::Codex`).
//!
//! Ports `gravity/codex/*` — OpenAI Codex on the ChatGPT backend
//! (`/backend-api/codex/responses`), the **Responses API** under an OAuth bearer
//! token. The system prompt is sent as `instructions`; messages map to typed
//! `input` items; the SSE stream emits `response.output_text.delta` events.

mod auth;
use crate::context::ChatCtx;
use crate::provider::{Authenticator, Provider};
use async_trait::async_trait;
use gravity_core::{
    ChatResponse, Content, Error, HistoryMode, Message, Metadata, ProviderCapabilities,
    ProviderError, ProviderErrorKind, ProviderName, RemoteSessionMode, Result, Role,
    StructuredOutputMode, ToolSupportMode, Usage,
};
use gravity_http::{HttpRequest, Profile};
use serde_json::{json, Value};

const URL: &str = "https://chatgpt.com/backend-api/codex/responses";

/// The Codex adapter.
pub struct CodexProvider {
    capabilities: ProviderCapabilities,
}

impl Default for CodexProvider {
    fn default() -> Self {
        CodexProvider::new()
    }
}

impl CodexProvider {
    /// Construct the provider.
    pub fn new() -> Self {
        CodexProvider {
            capabilities: ProviderCapabilities {
                provider: ProviderName::Codex,
                tool_support: ToolSupportMode::Native,
                structured_output: StructuredOutputMode::JsonInstruction,
                remote_session_mode: RemoteSessionMode::None,
                supports_system_prompt: true,
                supports_stateful_history: false,
                requires_remote_checkpoint: false,
                default_history_mode: HistoryMode::Stateless,
                supports_image_generation: false,
            },
        }
    }

    fn build_request(&self, ctx: &ChatCtx<'_>) -> HttpRequest {
        let instructions = ctx
            .request
            .system_prompt
            .clone()
            .filter(|s| !s.trim().is_empty())
            .unwrap_or_else(|| "You are a helpful assistant.".to_owned());
        let mut body = json!({
            "model": ctx.request.model.model,
            "instructions": instructions,
            "input": build_input(&ctx.request.messages),
            "stream": true,
            "store": false,
            "include": ["reasoning.encrypted_content"],
        });
        if !ctx.request.tools.is_empty() {
            let tools: Vec<Value> = ctx
                .request
                .tools
                .iter()
                .map(|t| {
                    json!({
                        "type": "function",
                        "name": t.name,
                        "description": t.description,
                        "parameters": t.input_schema,
                    })
                })
                .collect();
            body["tools"] = Value::Array(tools);
            body["tool_choice"] = json!("auto");
        }
        let mut req = HttpRequest::post(URL)
            .profile(Profile::from_name(ctx.impersonate_or("chrome")))
            .timeout_ms(ctx.timeout_ms())
            .proxy(ctx.proxy())
            .header("content-type", "application/json")
            .header("accept", "text/event-stream")
            .header("authorization", format!("Bearer {}", ctx.secret))
            .header("openai-beta", "responses=v1")
            .header("originator", "codex_cli_rs")
            .header("openai-originator", "codex")
            .json(&body);
        if let Some(acct) = ctx
            .account
            .auth
            .chatgpt_account_id
            .as_deref()
            .or_else(|| ctx.account.auth.inline_secret("chatgpt_account_id"))
        {
            req = req.header("chatgpt-account-id", acct);
        }
        req
    }
}

#[async_trait]
impl Provider for CodexProvider {
    fn capabilities(&self) -> &ProviderCapabilities {
        &self.capabilities
    }

    fn authenticator(&self) -> Option<&dyn Authenticator> {
        static AUTH: auth::CodexAuthenticator = auth::CodexAuthenticator;
        Some(&AUTH)
    }

    fn list_models(&self) -> Vec<String> {
        ["gpt-5-codex", "gpt-5", "o4-mini"].iter().map(|s| s.to_string()).collect()
    }

    async fn chat(&self, ctx: &ChatCtx<'_>) -> Result<ChatResponse> {
        let req = self.build_request(ctx);
        let resp = ctx.http.send(req).await?;
        if !resp.is_success() {
            return Err(Error::Provider(
                ProviderError::new(
                    ProviderName::Codex,
                    ProviderErrorKind::Other,
                    "codex_failed",
                    format!("Codex request failed: {}", resp.text().chars().take(300).collect::<String>()),
                )
                .with_status(resp.status),
            ));
        }
        let text = accumulate_responses_sse(&resp.text());
        Ok(ChatResponse {
            model: ctx.request.model.full_name().into(),
            message: Message::assistant(text),
            usage: Usage::default(),
            tool_calls_executed: Vec::new(),
            provider_response_id: None,
            finish_reason: Some("stop".into()),
            metadata: Metadata::new(),
        })
    }
}

/// Build the Responses-API `input` array from canonical messages.
fn build_input(messages: &[Message]) -> Vec<Value> {
    let mut items = Vec::new();
    for m in messages {
        match m.role {
            Role::System => continue,
            Role::Assistant => {
                items.push(json!({ "type": "message", "role": "assistant", "content": text_of(m) }));
                for tc in &m.tool_calls {
                    items.push(json!({
                        "type": "function_call",
                        "name": tc.name,
                        "arguments": Value::Object(tc.arguments.clone()).to_string(),
                        "call_id": tc.id,
                    }));
                }
            }
            Role::User => {
                items.push(json!({
                    "type": "message", "role": "user",
                    "content": [{ "type": "input_text", "text": text_of(m) }],
                }));
            }
            Role::Tool => {
                if let Some(tr) = &m.tool_result {
                    let output = tr
                        .structured_content
                        .clone()
                        .map(|v| v.to_string())
                        .unwrap_or_else(|| value_to_text(&tr.content));
                    items.push(json!({
                        "type": "function_call_output",
                        "call_id": tr.tool_call_id,
                        "output": output,
                    }));
                }
            }
        }
    }
    items
}

/// Accumulate `response.output_text.delta` events from the Responses SSE.
fn accumulate_responses_sse(body: &str) -> String {
    let mut text = String::new();
    for line in body.lines() {
        let Some(data) = line.strip_prefix("data: ") else {
            continue;
        };
        if data.trim() == "[DONE]" {
            break;
        }
        let Ok(v) = serde_json::from_str::<Value>(data) else {
            continue;
        };
        match v.get("type").and_then(Value::as_str) {
            Some("response.output_text.delta") => {
                if let Some(d) = v.get("delta").and_then(Value::as_str) {
                    text.push_str(d);
                }
            }
            // Some servers emit a final completed item with the full text.
            Some("response.completed") if text.is_empty() => {
                if let Some(t) = v
                    .get("response")
                    .and_then(|r| r.get("output_text"))
                    .and_then(Value::as_str)
                {
                    text = t.to_owned();
                }
            }
            _ => {}
        }
    }
    text
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn input_maps_roles_to_responses_items() {
        let msgs = vec![Message::user("hi"), Message::assistant("hello")];
        let input = build_input(&msgs);
        assert_eq!(input[0]["role"], "user");
        assert_eq!(input[0]["content"][0]["type"], "input_text");
        assert_eq!(input[1]["role"], "assistant");
        assert_eq!(input[1]["content"], "hello");
    }

    #[test]
    fn accumulates_output_text_deltas() {
        let body = "data: {\"type\":\"response.output_text.delta\",\"delta\":\"Hel\"}\n\
data: {\"type\":\"response.output_text.delta\",\"delta\":\"lo\"}\n\
data: [DONE]";
        assert_eq!(accumulate_responses_sse(body), "Hello");
    }
}
