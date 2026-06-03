//! The generic OpenAI-compatible provider.
//!
//! One `OpenAiCompatProvider<C>` serves every provider whose API speaks OpenAI
//! Chat Completions; each concrete provider supplies a small [`OaiConfig`] (base
//! URL, capabilities, auth header, models). Mirrors `OpenAICompatAdapter` in
//! `gravity/providers/openai_compat.py`.

use super::wire::{self, OaiResponse};
use crate::context::ChatCtx;
use crate::provider::{EventStream, Provider};
use async_trait::async_trait;
use gravity_core::{
    ChatResponse, Content, Error, Message, Metadata, ProviderCapabilities, ProviderError,
    ProviderErrorKind, ProviderName, Result, StreamEvent, ToolCall, Usage,
};
use gravity_http::{HttpRequest, Profile};
use serde_json::{Map, Value};

/// Per-provider configuration for the OpenAI-compatible base.
pub trait OaiConfig: Send + Sync + 'static {
    /// The provider this config is for.
    fn provider(&self) -> ProviderName;
    /// The provider's capabilities.
    fn capabilities(&self) -> ProviderCapabilities;
    /// Default API base URL (used when the account doesn't override it).
    fn default_base_url(&self) -> &str;
    /// Statically known models (may be empty).
    fn models(&self) -> Vec<String> {
        Vec::new()
    }
    /// The auth header pair for a resolved secret. Default: Bearer token.
    fn auth_header(&self, secret: &str) -> (String, String) {
        ("authorization".into(), format!("Bearer {secret}"))
    }
    /// Append any extra static headers.
    fn extra_headers(&self, _headers: &mut Vec<(String, String)>) {}
    /// Default impersonation profile name.
    fn default_profile(&self) -> &'static str {
        "chrome"
    }
}

/// The OpenAI-compatible provider, parameterized by a [`OaiConfig`].
pub struct OpenAiCompatProvider<C: OaiConfig> {
    config: C,
    capabilities: ProviderCapabilities,
}

impl<C: OaiConfig> OpenAiCompatProvider<C> {
    /// Build from a config (snapshotting its capabilities).
    pub fn new(config: C) -> Self {
        let capabilities = config.capabilities();
        OpenAiCompatProvider { config, capabilities }
    }

    fn base_url<'a>(&'a self, ctx: &'a ChatCtx<'a>) -> &'a str {
        let acct = ctx.account.transport.base_url.as_str();
        if acct.is_empty() {
            self.config.default_base_url()
        } else {
            acct
        }
    }

    fn profile(&self, ctx: &ChatCtx<'_>) -> Profile {
        Profile::from_name(ctx.impersonate_or(self.config.default_profile()))
    }

    fn build_http_request(&self, ctx: &ChatCtx<'_>, stream: bool) -> Result<HttpRequest> {
        let url = format!("{}/chat/completions", self.base_url(ctx).trim_end_matches('/'));
        let oai = wire::build_request(ctx.request, stream);
        let body = serde_json::to_vec(&oai)
            .map_err(|e| Error::validation(format!("failed to encode request: {e}"), "oai_encode"))?;
        let mut req = HttpRequest::post(url)
            .profile(self.profile(ctx))
            .timeout_ms(ctx.timeout_ms())
            .proxy(ctx.proxy())
            .header("content-type", "application/json")
            .body(bytes::Bytes::from(body));
        // Skip auth header for anonymous access (empty secret or kind="anonymous").
        if !ctx.secret.is_empty() && ctx.account.auth.kind.as_ref() != "anonymous" {
            let (auth_name, auth_val) = self.config.auth_header(ctx.secret);
            req = req.header(auth_name, auth_val);
        }
        let mut extra = Vec::new();
        self.config.extra_headers(&mut extra);
        for (k, v) in extra {
            req = req.header(k, v);
        }
        Ok(req)
    }
}

#[async_trait]
impl<C: OaiConfig> Provider for OpenAiCompatProvider<C> {
    fn capabilities(&self) -> &ProviderCapabilities {
        &self.capabilities
    }

    fn list_models(&self) -> Vec<String> {
        self.config.models()
    }

    /// Fetch models from the OpenAI-compatible `GET /models` endpoint
    /// (`{"data":[{"id":...}]}`), falling back to the static catalog on failure.
    async fn discover_models(&self, ctx: &ChatCtx<'_>) -> Result<Vec<gravity_core::ModelInfo>> {
        use gravity_core::ModelInfo;
        let fallback = || self.config.models().into_iter().map(ModelInfo::bare).collect::<Vec<_>>();

        let url = format!("{}/models", self.base_url(ctx).trim_end_matches('/'));
        let mut req = HttpRequest::get(url)
            .profile(self.profile(ctx))
            .timeout_ms(ctx.timeout_ms())
            .proxy(ctx.proxy());
        if !ctx.secret.is_empty() && ctx.account.auth.kind.as_ref() != "anonymous" {
            let (k, v) = self.config.auth_header(ctx.secret);
            req = req.header(k, v);
        }
        let mut extra = Vec::new();
        self.config.extra_headers(&mut extra);
        for (k, v) in extra {
            req = req.header(k, v);
        }

        let resp = match ctx.http.send(req).await {
            Ok(r) if r.is_success() => r,
            _ => return Ok(fallback()),
        };
        let parsed: Value = match serde_json::from_slice(&resp.body) {
            Ok(v) => v,
            Err(_) => return Ok(fallback()),
        };
        let arr = parsed.get("data").and_then(Value::as_array);
        let models: Vec<ModelInfo> = match arr {
            Some(items) => items
                .iter()
                .filter_map(|m| m.get("id").and_then(Value::as_str))
                .map(ModelInfo::bare)
                .collect(),
            None => return Ok(fallback()),
        };
        if models.is_empty() {
            Ok(fallback())
        } else {
            Ok(models)
        }
    }

    async fn chat(&self, ctx: &ChatCtx<'_>) -> Result<ChatResponse> {
        let req = self.build_http_request(ctx, false)?;
        let resp = ctx.http.send(req).await?;
        if !resp.is_success() {
            return Err(http_error(self.config.provider(), resp.status, &resp.text()));
        }
        let parsed: OaiResponse = serde_json::from_slice(&resp.body)
            .map_err(|e| Error::Network(format!("invalid OpenAI response JSON: {e}")))?;
        response_to_chat(self.config.provider(), &ctx.request.model.full_name(), parsed)
    }

    async fn chat_stream(&self, ctx: &ChatCtx<'_>) -> Result<EventStream> {
        let req = self.build_http_request(ctx, true)?;
        let provider = self.config.provider();
        let model_full = ctx.request.model.full_name();
        let (status, _h, byte_stream) = ctx.http.stream(req).await?;
        if status != 200 {
            return Err(http_error(provider, status, "stream failed"));
        }
        let events = gravity_http::sse::events(byte_stream);
        let out = async_stream::try_stream! {
            use futures::StreamExt;
            futures::pin_mut!(events);
            let mut text = String::new();
            let mut usage = Usage::default();
            let mut finish: Option<String> = None;
            while let Some(ev) = events.next().await {
                let ev = ev?;
                if ev.is_done() { break; }
                let chunk: OaiResponse = match serde_json::from_str(&ev.data) {
                    Ok(c) => c,
                    Err(_) => continue,
                };
                if let Some(u) = chunk.usage.as_ref() {
                    usage = Usage {
                        input_tokens: u.prompt_tokens,
                        output_tokens: u.completion_tokens,
                        total_tokens: u.total_tokens,
                    };
                }
                if let Some(choice) = chunk.choices.into_iter().next() {
                    if let Some(fr) = choice.finish_reason { finish = Some(fr); }
                    if let Some(delta) = choice.delta.or(choice.message) {
                        if let Some(c) = delta.content {
                            if !c.is_empty() {
                                text.push_str(&c);
                                yield StreamEvent::TextDelta(bytes::Bytes::from(c));
                            }
                        }
                    }
                }
            }
            let resp = ChatResponse {
                model: model_full.into(),
                message: Message::assistant(text),
                usage,
                tool_calls_executed: Vec::new(),
                provider_response_id: None,
                finish_reason: finish.map(Into::into),
                metadata: Metadata::new(),
            };
            yield StreamEvent::Done(Box::new(resp));
        };
        Ok(Box::pin(out))
    }
}

/// Convert an [`OaiResponse`] into a canonical [`ChatResponse`].
fn response_to_chat(provider: ProviderName, model_full: &str, resp: OaiResponse) -> Result<ChatResponse> {
    let choice = resp.choices.into_iter().next().ok_or_else(|| {
        Error::Provider(ProviderError::new(
            provider,
            ProviderErrorKind::Other,
            "oai_no_choices",
            "response contained no choices",
        ))
    })?;
    let msg = choice.message.or(choice.delta).unwrap_or_default();

    let tool_calls: Vec<ToolCall> = msg
        .tool_calls
        .into_iter()
        .map(|tc| {
            let arguments = tc
                .function
                .arguments
                .as_deref()
                .and_then(|s| serde_json::from_str::<Value>(s).ok())
                .and_then(|v| v.as_object().cloned())
                .unwrap_or_else(Map::new);
            ToolCall {
                id: tc.id.unwrap_or_default().into(),
                name: tc.function.name.unwrap_or_default().into(),
                arguments,
            }
        })
        .collect();

    let content = msg.content.unwrap_or_default();
    let message = if tool_calls.is_empty() {
        Message::assistant(content)
    } else {
        Message::assistant_with_tools(Some(Content::Text(content)), tool_calls)
    };

    let usage = resp
        .usage
        .map(|u| Usage {
            input_tokens: u.prompt_tokens,
            output_tokens: u.completion_tokens,
            total_tokens: u.total_tokens,
        })
        .unwrap_or_default();

    Ok(ChatResponse {
        model: model_full.into(),
        message,
        usage,
        tool_calls_executed: Vec::new(),
        provider_response_id: resp.id.map(Into::into),
        finish_reason: choice.finish_reason.map(Into::into),
        metadata: Metadata::new(),
    })
}

fn http_error(provider: ProviderName, status: u16, body: &str) -> Error {
    let snippet: String = body.chars().take(500).collect();
    Error::Provider(
        ProviderError::new(
            provider,
            ProviderErrorKind::Other,
            "oai_request_failed",
            format!("request failed ({status}): {snippet}"),
        )
        .with_status(status),
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use gravity_core::{HistoryMode, RemoteSessionMode, StructuredOutputMode, ToolSupportMode};

    struct TestConfig;
    impl OaiConfig for TestConfig {
        fn provider(&self) -> ProviderName {
            ProviderName::Groq
        }
        fn capabilities(&self) -> ProviderCapabilities {
            ProviderCapabilities {
                provider: ProviderName::Groq,
                tool_support: ToolSupportMode::Native,
                structured_output: StructuredOutputMode::JsonInstruction,
                remote_session_mode: RemoteSessionMode::None,
                supports_system_prompt: true,
                supports_stateful_history: false,
                requires_remote_checkpoint: false,
                default_history_mode: HistoryMode::Stateless,
                supports_image_generation: false,
            }
        }
        fn default_base_url(&self) -> &str {
            "https://api.groq.com/openai/v1"
        }
    }

    #[test]
    fn response_conversion_with_tool_calls() {
        let raw = r#"{
            "id":"cmpl-9","model":"llama",
            "choices":[{"message":{"content":null,"tool_calls":[
                {"id":"tc1","type":"function","function":{"name":"get_weather","arguments":"{\"city\":\"NYC\"}"}}
            ]},"finish_reason":"tool_calls"}],
            "usage":{"prompt_tokens":3,"completion_tokens":4}
        }"#;
        let resp: OaiResponse = serde_json::from_str(raw).unwrap();
        let chat = response_to_chat(ProviderName::Groq, "groq/llama", resp).unwrap();
        assert_eq!(chat.message.tool_calls.len(), 1);
        assert_eq!(&*chat.message.tool_calls[0].name, "get_weather");
        assert_eq!(chat.message.tool_calls[0].arguments["city"], "NYC");
        assert_eq!(chat.finish_reason.as_deref(), Some("tool_calls"));
        assert_eq!(chat.usage.total(), 7);
    }

    #[test]
    fn provider_uses_config_capabilities() {
        let p = OpenAiCompatProvider::new(TestConfig);
        assert_eq!(p.capabilities().provider, ProviderName::Groq);
    }
}
