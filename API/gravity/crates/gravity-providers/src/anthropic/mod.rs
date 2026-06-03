//! The Anthropic public API provider (`ProviderName::AnthropicApi`).
//!
//! Ports `gravity/anthropic_api/*`. Unlike the OpenAI-compatible providers,
//! Anthropic's Messages API puts the system prompt at the top level, requires
//! `max_tokens`, and represents tool calls/results as typed content blocks.

mod wire;

use crate::context::ChatCtx;
use crate::provider::{EventStream, Provider};
use async_trait::async_trait;
use gravity_core::{
    ChatResponse, Error, HistoryMode, Message, Metadata, ProviderCapabilities, ProviderError,
    ProviderErrorKind, ProviderName, RemoteSessionMode, Result, StreamEvent, StructuredOutputMode,
    ToolSupportMode, Usage,
};
use gravity_http::{HttpRequest, Profile};

/// Default `max_tokens` when the request doesn't set one (the API requires it).
const DEFAULT_MAX_TOKENS: u32 = 4096;
/// Pinned API version header.
const ANTHROPIC_VERSION: &str = "2023-06-01";
/// Default base URL.
const DEFAULT_BASE_URL: &str = "https://api.anthropic.com/v1";

/// The Anthropic Messages API adapter.
pub struct AnthropicProvider {
    capabilities: ProviderCapabilities,
}

impl Default for AnthropicProvider {
    fn default() -> Self {
        AnthropicProvider::new()
    }
}

impl AnthropicProvider {
    /// Construct the provider.
    pub fn new() -> Self {
        AnthropicProvider {
            capabilities: ProviderCapabilities {
                provider: ProviderName::AnthropicApi,
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

    fn base_url<'a>(&self, ctx: &'a ChatCtx<'a>) -> &'a str {
        let b = ctx.account.transport.base_url.as_str();
        if b.is_empty() {
            DEFAULT_BASE_URL
        } else {
            b
        }
    }

    fn build_request(&self, ctx: &ChatCtx<'_>, stream: bool) -> Result<HttpRequest> {
        let url = format!("{}/messages", self.base_url(ctx).trim_end_matches('/'));
        let body = wire::build_request(ctx.request, DEFAULT_MAX_TOKENS, stream);
        let bytes = serde_json::to_vec(&body)
            .map_err(|e| Error::validation(format!("encode failed: {e}"), "anthropic_encode"))?;
        Ok(HttpRequest::post(url)
            .profile(Profile::from_name(ctx.impersonate_or("chrome")))
            .timeout_ms(ctx.timeout_ms())
            .proxy(ctx.proxy())
            .header("content-type", "application/json")
            .header("x-api-key", ctx.secret)
            .header("anthropic-version", ANTHROPIC_VERSION)
            .body(bytes::Bytes::from(bytes)))
    }
}

#[async_trait]
impl Provider for AnthropicProvider {
    fn capabilities(&self) -> &ProviderCapabilities {
        &self.capabilities
    }

    fn list_models(&self) -> Vec<String> {
        ["claude-opus-4-1", "claude-sonnet-4-5", "claude-3-5-haiku-latest"]
            .iter()
            .map(|s| s.to_string())
            .collect()
    }

    async fn chat(&self, ctx: &ChatCtx<'_>) -> Result<ChatResponse> {
        let req = self.build_request(ctx, false)?;
        let resp = ctx.http.send(req).await?;
        if !resp.is_success() {
            return Err(err(resp.status, &resp.text()));
        }
        let parsed: wire::AnthropicResponse = serde_json::from_slice(&resp.body)
            .map_err(|e| Error::Network(format!("invalid Anthropic response: {e}")))?;
        Ok(wire::to_chat_response(&ctx.request.model.full_name(), parsed))
    }

    async fn chat_stream(&self, ctx: &ChatCtx<'_>) -> Result<EventStream> {
        let req = self.build_request(ctx, true)?;
        let model_full = ctx.request.model.full_name();
        let (status, _h, byte_stream) = ctx.http.stream(req).await?;
        if status != 200 {
            return Err(err(status, "stream failed"));
        }
        let events = gravity_http::sse::events(byte_stream);
        let out = async_stream::try_stream! {
            use futures::StreamExt;
            futures::pin_mut!(events);
            let mut text = String::new();
            let mut usage = Usage::default();
            let mut stop: Option<String> = None;
            let mut resp_id: Option<String> = None;
            // Pending tool call being assembled from streaming deltas.
            let mut pending_tc: Option<(String, String, String)> = None; // (id, name, json_buf)
            let mut tool_calls: Vec<gravity_core::ToolCall> = Vec::new();

            while let Some(ev) = events.next().await {
                let ev = ev?;
                let v: serde_json::Value = match serde_json::from_str(&ev.data) {
                    Ok(v) => v,
                    Err(_) => continue,
                };
                match v.get("type").and_then(|t| t.as_str()) {
                    Some("message_start") => {
                        // Capture response id and input token usage.
                        if let Some(msg) = v.get("message") {
                            resp_id = msg.get("id").and_then(|i| i.as_str()).map(str::to_owned);
                            if let Some(it) = msg.get("usage").and_then(|u| u.get("input_tokens")).and_then(|n| n.as_u64()) {
                                usage.input_tokens = Some(it as u32);
                            }
                        }
                    }
                    Some("content_block_start") => {
                        let block = v.get("content_block");
                        if block.and_then(|b| b.get("type")).and_then(|t| t.as_str()) == Some("tool_use") {
                            let id = block.and_then(|b| b.get("id")).and_then(|i| i.as_str()).unwrap_or("").to_owned();
                            let name = block.and_then(|b| b.get("name")).and_then(|n| n.as_str()).unwrap_or("").to_owned();
                            pending_tc = Some((id, name, String::new()));
                        }
                    }
                    Some("content_block_delta") => {
                        let delta = v.get("delta");
                        match delta.and_then(|d| d.get("type")).and_then(|t| t.as_str()) {
                            Some("text_delta") => {
                                if let Some(t) = delta.and_then(|d| d.get("text")).and_then(|t| t.as_str()) {
                                    text.push_str(t);
                                    yield StreamEvent::TextDelta(bytes::Bytes::from(t.to_owned()));
                                }
                            }
                            Some("input_json_delta") => {
                                if let Some(ref mut tc) = pending_tc {
                                    if let Some(frag) = delta.and_then(|d| d.get("partial_json")).and_then(|j| j.as_str()) {
                                        tc.2.push_str(frag);
                                    }
                                }
                            }
                            _ => {}
                        }
                    }
                    Some("content_block_stop") => {
                        // Finalize any pending tool call.
                        if let Some((id, name, json_buf)) = pending_tc.take() {
                            let arguments: serde_json::Map<String, serde_json::Value> =
                                serde_json::from_str(&json_buf).unwrap_or_default();
                            tool_calls.push(gravity_core::ToolCall {
                                id: id.into(),
                                name: name.into(),
                                arguments,
                            });
                        }
                    }
                    Some("message_delta") => {
                        if let Some(sr) = v.get("delta").and_then(|d| d.get("stop_reason")).and_then(|s| s.as_str()) {
                            stop = Some(sr.to_owned());
                        }
                        if let Some(ot) = v.get("usage").and_then(|u| u.get("output_tokens")).and_then(|n| n.as_u64()) {
                            usage.output_tokens = Some(ot as u32);
                        }
                    }
                    Some("error") => {
                        let msg = v.get("error").and_then(|e| e.get("message")).and_then(|m| m.as_str()).unwrap_or("anthropic stream error");
                        Err(Error::Provider(ProviderError::new(ProviderName::AnthropicApi, ProviderErrorKind::Other, "anthropic_stream_error", msg.to_owned())))?;
                    }
                    _ => {}
                }
            }
            let message = if tool_calls.is_empty() {
                Message::assistant(text)
            } else {
                Message::assistant_with_tools(
                    if text.is_empty() { None } else { Some(gravity_core::Content::Text(text)) },
                    tool_calls,
                )
            };
            let resp = ChatResponse {
                model: model_full.into(),
                message,
                usage,
                tool_calls_executed: Vec::new(),
                provider_response_id: resp_id.map(Into::into),
                finish_reason: stop.map(Into::into),
                metadata: Metadata::new(),
            };
            yield StreamEvent::Done(Box::new(resp));
        };
        Ok(Box::pin(out))
    }
}

fn err(status: u16, body: &str) -> Error {
    let snippet: String = body.chars().take(500).collect();
    Error::Provider(
        ProviderError::new(
            ProviderName::AnthropicApi,
            ProviderErrorKind::Other,
            "anthropic_request_failed",
            format!("request failed ({status}): {snippet}"),
        )
        .with_status(status),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn capabilities() {
        let p = AnthropicProvider::new();
        assert_eq!(p.capabilities().provider, ProviderName::AnthropicApi);
        assert!(!p.list_models().is_empty());
    }
}
