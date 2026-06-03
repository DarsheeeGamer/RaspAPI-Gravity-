//! Antigravity provider (`ProviderName::Antigravity`).
//!
//! Ports the `generateContent` path of `gravity/antigravity/client.py` — a
//! Google Code Assist sandbox endpoint (`*-cloudcode-pa.sandbox.googleapis.com`)
//! reached under an OAuth bearer token over the **FIPS/boring** Go TLS profile
//! (`go_boring`). Messages convert to the Gemini `contents` format wrapped in a
//! `{project, model, request, requestId}` envelope.

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

const ENDPOINT: &str = "https://daily-cloudcode-pa.sandbox.googleapis.com";

/// The Antigravity adapter.
pub struct AntigravityProvider {
    capabilities: ProviderCapabilities,
}

impl Default for AntigravityProvider {
    fn default() -> Self {
        AntigravityProvider::new()
    }
}

impl AntigravityProvider {
    /// Construct the provider.
    pub fn new() -> Self {
        AntigravityProvider {
            capabilities: ProviderCapabilities {
                provider: ProviderName::Antigravity,
                tool_support: ToolSupportMode::Native,
                structured_output: StructuredOutputMode::Native,
                remote_session_mode: RemoteSessionMode::None,
                supports_system_prompt: true,
                supports_stateful_history: false,
                requires_remote_checkpoint: false,
                default_history_mode: HistoryMode::Stateless,
                supports_image_generation: true,
            },
        }
    }

    fn build_request(&self, ctx: &ChatCtx<'_>) -> HttpRequest {
        let contents = build_contents(&ctx.request.messages);
        let mut request_body = json!({ "contents": contents });
        if let Some(sp) = &ctx.request.system_prompt {
            request_body["systemInstruction"] = json!({ "parts": [{ "text": sp }] });
        }
        let body = json!({
            "project": ctx.account.auth.inline_secret("project").unwrap_or(""),
            "model": ctx.request.model.model,
            "request": request_body,
            "requestId": format!("gravity-{}", uuid_v4(ctx.env)),
        });
        HttpRequest::post(format!("{ENDPOINT}/v1internal:generateContent"))
            // Antigravity's language server uses the Go boringcrypto/FIPS fingerprint.
            .profile(Profile::from_name(ctx.impersonate_or("go_boring")))
            .timeout_ms(ctx.timeout_ms())
            .proxy(ctx.proxy())
            .header("content-type", "application/json")
            .header("authorization", format!("Bearer {}", ctx.secret))
            .json(&body)
    }
}

#[async_trait]
impl Provider for AntigravityProvider {
    fn capabilities(&self) -> &ProviderCapabilities {
        &self.capabilities
    }

    fn authenticator(&self) -> Option<&dyn Authenticator> {
        static AUTH: auth::AntigravityAuthenticator = auth::AntigravityAuthenticator;
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
                    ProviderName::Antigravity,
                    ProviderErrorKind::Other,
                    "antigravity_failed",
                    format!("Antigravity request failed (status {})", resp.status),
                )
                .with_status(resp.status),
            ));
        }
        let v: Value = resp.json()?;
        let text = extract_text(&v);
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

fn build_contents(messages: &[Message]) -> Vec<Value> {
    let mut out = Vec::new();
    for m in messages {
        if m.role == Role::System {
            continue;
        }
        let text = text_of(m);
        if text.is_empty() {
            continue;
        }
        let role = if m.role == Role::Assistant { "model" } else { "user" };
        out.push(json!({ "role": role, "parts": [{ "text": text }] }));
    }
    out
}

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
    fn contents_skip_system() {
        let c = build_contents(&[Message::system("s"), Message::user("hi")]);
        assert_eq!(c.len(), 1);
        assert_eq!(c[0]["role"], "user");
    }

    #[test]
    fn extracts_candidate_text() {
        let v = json!({ "candidates": [{ "content": { "parts": [{ "text": "ans" }] } }] });
        assert_eq!(extract_text(&v), "ans");
    }

    #[test]
    fn capabilities() {
        assert_eq!(AntigravityProvider::new().capabilities().provider, ProviderName::Antigravity);
    }
}
