//! Gemini CLI provider (`ProviderName::GeminiCli`).
//!
//! Ports `gravity/gemini_cli/*` — the Google Code Assist API
//! (`cloudcode-pa.googleapis.com`) used by the Gemini CLI under an OAuth bearer
//! token. Messages convert to the Gemini `contents` format; the response is the
//! `generateContent` candidate text. Auth = account access token; the GCP
//! project id comes from `auth.extra.project`.

mod auth;
use crate::context::ChatCtx;
use crate::provider::{Authenticator, Provider};
use crate::util::uuid_v4;
use async_trait::async_trait;
use gravity_core::{
    ChatResponse, Content, Error, HistoryMode, Message, Metadata, ProviderCapabilities,
    ProviderError, ProviderErrorKind, ProviderName, RemoteSessionMode, Result, Role,
    StructuredOutputMode, ToolSupportMode, Usage,
};
use gravity_http::{HttpRequest, Profile};
use serde_json::{json, Value};

const ENDPOINT: &str = "https://cloudcode-pa.googleapis.com";

/// The Gemini CLI adapter.
pub struct GeminiCliProvider {
    capabilities: ProviderCapabilities,
}

impl Default for GeminiCliProvider {
    fn default() -> Self {
        GeminiCliProvider::new()
    }
}

impl GeminiCliProvider {
    /// Construct the provider.
    pub fn new() -> Self {
        GeminiCliProvider {
            capabilities: ProviderCapabilities {
                provider: ProviderName::GeminiCli,
                tool_support: ToolSupportMode::Native,
                structured_output: StructuredOutputMode::Native,
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
        let contents = build_contents(&ctx.request.messages);
        let mut inner = json!({ "contents": contents });
        if let Some(sp) = &ctx.request.system_prompt {
            inner["systemInstruction"] = json!({ "parts": [{ "text": sp }] });
        }
        if !ctx.request.tools.is_empty() {
            let decls: Vec<Value> = ctx
                .request
                .tools
                .iter()
                .map(|t| json!({ "name": t.name, "description": t.description, "parameters": t.input_schema }))
                .collect();
            inner["tools"] = json!([{ "functionDeclarations": decls }]);
        }
        let project = ctx.account.auth.inline_secret("project").unwrap_or("");
        let body = json!({
            "model": ctx.request.model.model,
            "project": project,
            "user_prompt_id": uuid_v4(ctx.env),
            "request": inner,
        });
        let url = format!("{ENDPOINT}/v1internal:generateContent");
        HttpRequest::post(url)
            .profile(Profile::from_name(ctx.impersonate_or("chrome")))
            .timeout_ms(ctx.timeout_ms())
            .proxy(ctx.proxy())
            .header("content-type", "application/json")
            .header("authorization", format!("Bearer {}", ctx.secret))
            .header("user-agent", "google-api-nodejs-client/9.15.1")
            .header("x-goog-api-client", "gl-node/22.17.0")
            .json(&body)
    }
}

#[async_trait]
impl Provider for GeminiCliProvider {
    fn capabilities(&self) -> &ProviderCapabilities {
        &self.capabilities
    }

    fn authenticator(&self) -> Option<&dyn Authenticator> {
        static AUTH: auth::GeminiCliAuthenticator = auth::GeminiCliAuthenticator;
        Some(&AUTH)
    }

    fn list_models(&self) -> Vec<String> {
        ["gemini-2.5-pro", "gemini-2.5-flash"].iter().map(|s| s.to_string()).collect()
    }

    async fn chat(&self, ctx: &ChatCtx<'_>) -> Result<ChatResponse> {
        let req = self.build_request(ctx);
        let resp = ctx.http.send(req).await?;
        if !resp.is_success() {
            return Err(Error::Provider(
                ProviderError::new(
                    ProviderName::GeminiCli,
                    ProviderErrorKind::Other,
                    "gemini_cli_failed",
                    format!("Gemini CLI request failed: {}", resp.text().chars().take(300).collect::<String>()),
                )
                .with_status(resp.status),
            ));
        }
        let v: Value = resp.json()?;
        let text = extract_text(&v);
        let tool_calls = extract_tool_calls(&v);
        let message = if tool_calls.is_empty() {
            Message::assistant(text)
        } else {
            Message::assistant_with_tools(Some(gravity_core::Content::Text(text)), tool_calls)
        };
        Ok(ChatResponse {
            model: ctx.request.model.full_name().into(),
            message,
            usage: Usage::default(),
            tool_calls_executed: Vec::new(),
            provider_response_id: None,
            finish_reason: Some("stop".into()),
            metadata: Metadata::new(),
        })
    }
}

/// Extract `functionCall` parts from a `generateContent` response into tool calls.
fn extract_tool_calls(v: &Value) -> Vec<gravity_core::ToolCall> {
    let root = v.get("response").unwrap_or(v);
    root.get("candidates")
        .and_then(Value::as_array)
        .and_then(|c| c.first())
        .and_then(|c| c.get("content"))
        .and_then(|c| c.get("parts"))
        .and_then(Value::as_array)
        .map(|parts| {
            parts
                .iter()
                .filter_map(|p| p.get("functionCall"))
                .map(|fc| gravity_core::ToolCall {
                    id: fc.get("id").and_then(Value::as_str).unwrap_or("").into(),
                    name: fc.get("name").and_then(Value::as_str).unwrap_or("").into(),
                    arguments: fc.get("args").and_then(Value::as_object).cloned().unwrap_or_default(),
                })
                .filter(|tc| !tc.name.is_empty())
                .collect()
        })
        .unwrap_or_default()
}

/// Convert canonical messages to the Gemini `contents` array.
fn build_contents(messages: &[Message]) -> Vec<Value> {
    let mut out = Vec::new();
    for m in messages {
        if m.role == Role::System {
            continue;
        }
        let mut parts = Vec::new();
        let text = text_of(m);
        if !text.is_empty() {
            parts.push(json!({ "text": text }));
        }
        for tc in &m.tool_calls {
            parts.push(json!({ "functionCall": { "name": tc.name, "args": tc.arguments, "id": tc.id } }));
        }
        if let Some(tr) = &m.tool_result {
            parts.push(json!({
                "functionResponse": {
                    "name": tr.name,
                    "id": tr.tool_call_id,
                    "response": tr.structured_content.clone().unwrap_or_else(|| json!({ "content": tr.content })),
                }
            }));
        }
        if parts.is_empty() {
            continue;
        }
        let role = if m.role == Role::Assistant { "model" } else { "user" };
        out.push(json!({ "role": role, "parts": parts }));
    }
    out
}

/// Pull the assistant text from a `generateContent` response (under an optional
/// `response` wrapper).
fn extract_text(v: &Value) -> String {
    let root = v.get("response").unwrap_or(v);
    root.get("candidates")
        .and_then(Value::as_array)
        .and_then(|c| c.first())
        .and_then(|c| c.get("content"))
        .and_then(|c| c.get("parts"))
        .and_then(Value::as_array)
        .map(|parts| {
            parts
                .iter()
                .filter_map(|p| p.get("text").and_then(Value::as_str))
                .collect::<Vec<_>>()
                .join("")
        })
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn contents_skip_system_and_map_roles() {
        let msgs = vec![
            Message::system("ignored"),
            Message::user("hi"),
            Message::assistant("hello"),
        ];
        let c = build_contents(&msgs);
        assert_eq!(c.len(), 2);
        assert_eq!(c[0]["role"], "user");
        assert_eq!(c[1]["role"], "model");
        assert_eq!(c[0]["parts"][0]["text"], "hi");
    }

    #[test]
    fn extract_text_from_candidates() {
        let v = json!({
            "response": { "candidates": [{ "content": { "parts": [{ "text": "answer" }] } }] }
        });
        assert_eq!(extract_text(&v), "answer");
    }
}
