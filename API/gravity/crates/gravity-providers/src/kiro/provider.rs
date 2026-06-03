//! The AWS Kiro provider.
//!
//! Ports `gravity/kiro/client.py`. Auth is a Bearer access token (the account
//! secret); the response is decoded with the brace-counting [`super::parser`].

use super::{auth, build, parser};
use crate::context::ChatCtx;
use crate::provider::{Authenticator, EventStream, Provider};
use crate::util::uuid_v4;
use async_trait::async_trait;
use gravity_core::{
    ChatResponse, Content, Error, HistoryMode, Message, Metadata, ProviderCapabilities,
    ProviderError, ProviderErrorKind, ProviderName, RemoteSessionMode, Result, StreamEvent,
    StructuredOutputMode, ToolCall, ToolSupportMode, Usage,
};
use gravity_http::{HttpRequest, Profile};
use parser::KiroEvent;

/// The Kiro adapter.
pub struct KiroProvider {
    capabilities: ProviderCapabilities,
}

impl Default for KiroProvider {
    fn default() -> Self {
        KiroProvider::new()
    }
}

impl KiroProvider {
    /// Construct the provider.
    pub fn new() -> Self {
        KiroProvider {
            capabilities: ProviderCapabilities {
                provider: ProviderName::Kiro,
                tool_support: ToolSupportMode::Native,
                structured_output: StructuredOutputMode::None,
                remote_session_mode: RemoteSessionMode::None,
                supports_system_prompt: true,
                supports_stateful_history: false,
                requires_remote_checkpoint: false,
                default_history_mode: HistoryMode::Stateless,
                supports_image_generation: false,
            },
        }
    }

    fn region<'a>(&self, ctx: &'a ChatCtx<'a>) -> &'a str {
        ctx.account.auth.inline_secret("region").unwrap_or(build::DEFAULT_REGION)
    }

    fn build_request(&self, ctx: &ChatCtx<'_>) -> Result<HttpRequest> {
        let profile_arn = ctx.account.auth.inline_secret("profile_arn");
        let (body, _model) = build::build_request(
            ctx.env,
            &ctx.request.messages,
            &ctx.request.tools,
            ctx.request.system_prompt.as_deref(),
            &ctx.request.model.model,
            profile_arn,
        )
        .ok_or_else(|| {
            Error::validation(
                format!("unsupported or empty Kiro request for model {}", ctx.request.model.model),
                "kiro_bad_request",
            )
        })?;
        let bytes = serde_json::to_vec(&body).unwrap_or_default();
        // AWS API Gateway does not fingerprint TLS — use the plain profile.
        Ok(HttpRequest::post(build::url(self.region(ctx)))
            .profile(Profile::from_name(ctx.impersonate_or("go")))
            .timeout_ms(ctx.timeout_ms())
            .proxy(ctx.proxy())
            .header("content-type", "application/json")
            .header("accept", "*/*")
            .header("authorization", format!("Bearer {}", ctx.secret))
            .header("x-amzn-q-developer-api-version", build::API_VERSION)
            .header("amz-sdk-invocation-id", uuid_v4(ctx.env))
            .header("amz-sdk-request", "attempt=1; max=1")
            .header("x-amzn-kiro-agent-mode", "vibe")
            .header(
                "x-amz-user-agent",
                format!("aws-sdk-js/{} {}", build::SDK_VERSION, build::USER_AGENT),
            )
            .body(bytes::Bytes::from(bytes)))
    }

    async fn buffered_events(&self, ctx: &ChatCtx<'_>) -> Result<Vec<KiroEvent>> {
        let req = self.build_request(ctx)?;
        let resp = ctx.http.send(req).await?;
        if !resp.is_success() {
            return Err(Error::Provider(
                ProviderError::new(
                    ProviderName::Kiro,
                    ProviderErrorKind::Other,
                    "kiro_request_failed",
                    format!("Kiro request failed: {}", resp.text().chars().take(300).collect::<String>()),
                )
                .with_status(resp.status),
            ));
        }
        Ok(parser::parse_event_stream_buffer(&resp.text()))
    }
}

#[async_trait]
impl Provider for KiroProvider {
    fn capabilities(&self) -> &ProviderCapabilities {
        &self.capabilities
    }

    fn authenticator(&self) -> Option<&dyn Authenticator> {
        static AUTH: auth::KiroAuthenticator = auth::KiroAuthenticator;
        Some(&AUTH)
    }

    fn list_models(&self) -> Vec<String> {
        build::models()
    }

    async fn chat(&self, ctx: &ChatCtx<'_>) -> Result<ChatResponse> {
        let events = self.buffered_events(ctx).await?;
        let mut text = String::new();
        let mut tool_calls: Vec<ToolCall> = Vec::new();
        for ev in events {
            match ev {
                KiroEvent::Content(c) => text.push_str(&c),
                KiroEvent::ToolUse { name, tool_use_id, input, .. } => {
                    let arguments = serde_json::from_str(&input)
                        .ok()
                        .and_then(|v: serde_json::Value| v.as_object().cloned())
                        .unwrap_or_default();
                    tool_calls.push(ToolCall { id: tool_use_id.into(), name: name.into(), arguments });
                }
                _ => {}
            }
        }
        let message = if tool_calls.is_empty() {
            Message::assistant(text)
        } else {
            Message::assistant_with_tools(Some(Content::Text(text)), tool_calls)
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

    async fn chat_stream(&self, ctx: &ChatCtx<'_>) -> Result<EventStream> {
        // Kiro's framing is not line-oriented SSE; buffer then replay deltas.
        let resp = self.chat(ctx).await?;
        let text = resp.text().to_owned();
        let out = async_stream::stream! {
            if !text.is_empty() {
                yield Ok(StreamEvent::TextDelta(bytes::Bytes::from(text)));
            }
            yield Ok(StreamEvent::Done(Box::new(resp)));
        };
        Ok(Box::pin(out))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn capabilities() {
        let p = KiroProvider::new();
        assert_eq!(p.capabilities().provider, ProviderName::Kiro);
        assert!(!p.list_models().is_empty());
    }
}
