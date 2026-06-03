//! The GLM / chat.z.ai provider.
//!
//! Ports `gravity/glm/client.py`. A guest token is supplied via the account
//! secret; each request is HMAC-signed ([`super::sign`]) and carries the full
//! browser-shaped query string. Streaming uses the `phase`/`delta_content` SSE
//! shape unique to chat.z.ai.

use super::{build, sign};
use crate::context::ChatCtx;
use crate::provider::{EventStream, Provider};
use crate::util::uuid_v4;
use async_trait::async_trait;
use gravity_core::{
    ChatResponse, Error, HistoryMode, Message, Metadata, ProviderCapabilities, ProviderError,
    ProviderErrorKind, ProviderName, RemoteSessionMode, Result, StreamEvent, StructuredOutputMode,
    ToolSupportMode, Usage,
};
use gravity_http::HttpRequest;
use serde_json::Value;

/// The chat.z.ai provider adapter.
pub struct GlmProvider {
    capabilities: ProviderCapabilities,
}

impl Default for GlmProvider {
    fn default() -> Self {
        GlmProvider::new()
    }
}

impl GlmProvider {
    /// Construct the provider.
    pub fn new() -> Self {
        GlmProvider {
            capabilities: ProviderCapabilities {
                provider: ProviderName::Glm,
                tool_support: ToolSupportMode::None,
                structured_output: StructuredOutputMode::JsonInstruction,
                remote_session_mode: RemoteSessionMode::Optional,
                supports_system_prompt: true,
                supports_stateful_history: true,
                requires_remote_checkpoint: false,
                default_history_mode: HistoryMode::Auto,
                supports_image_generation: false,
            },
        }
    }

    /// Build the signed completion HTTP request shared by chat and stream.
    fn build_request(&self, ctx: &ChatCtx<'_>) -> HttpRequest {
        let token = ctx.secret;
        let user_id = ctx.account.auth.inline_secret("user_id").unwrap_or("");
        let timestamp = ctx.env.clock.now_millis();
        let request_id = uuid_v4(ctx.env);
        let chat_id = uuid_v4(ctx.env);
        let model = if ctx.request.model.model.is_empty() {
            build::DEFAULT_MODEL
        } else {
            &ctx.request.model.model
        };

        let body = build::build_body(ctx.env, &ctx.request.messages, model, &chat_id, user_id);
        let sig_prompt = build::signature_prompt(&ctx.request.messages);
        let signature = sign::generate_signature(&sig_prompt, timestamp, &request_id, user_id);
        let url = build::build_url(token, user_id, timestamp, &request_id, &chat_id);
        let body_bytes = serde_json::to_vec(&body).unwrap_or_default();

        let mut req = HttpRequest::post(url)
            .profile(gravity_http::Profile::from_name(ctx.impersonate_or("chrome")))
            .timeout_ms(ctx.timeout_ms())
            .proxy(ctx.proxy())
            .body(bytes::Bytes::from(body_bytes));
        for (k, v) in build::default_headers() {
            req = req.header(k, v);
        }
        req = req
            .header("Authorization", format!("Bearer {token}"))
            .header("X-Signature", signature)
            .header("X-Signature-Timestamp", timestamp.to_string());
        req
    }
}

#[async_trait]
impl Provider for GlmProvider {
    fn capabilities(&self) -> &ProviderCapabilities {
        &self.capabilities
    }

    fn list_models(&self) -> Vec<String> {
        build::models()
    }

    /// Discover models from `GET /api/models` (chat.z.ai is OpenWebUI-based;
    /// response is `{"data":[{"id","name"}]}`). Falls back to the static list.
    async fn discover_models(&self, ctx: &ChatCtx<'_>) -> Result<Vec<gravity_core::ModelInfo>> {
        use gravity_core::ModelInfo;
        let fallback = || build::models().into_iter().map(ModelInfo::bare).collect::<Vec<_>>();

        let mut req = HttpRequest::get(build::MODELS_URL)
            .profile(gravity_http::Profile::from_name(ctx.impersonate_or("chrome")))
            .timeout_ms(ctx.timeout_ms())
            .proxy(ctx.proxy());
        for (k, v) in build::default_headers() {
            req = req.header(k, v);
        }
        if !ctx.secret.is_empty() {
            req = req.header("Authorization", format!("Bearer {}", ctx.secret));
        }
        let resp = match ctx.http.send(req).await {
            Ok(r) if r.is_success() => r,
            _ => return Ok(fallback()),
        };
        let parsed: Value = match serde_json::from_slice(&resp.body) {
            Ok(v) => v,
            Err(_) => return Ok(fallback()),
        };
        // OpenWebUI returns {"data":[...]}; some deployments return a bare list.
        let items = parsed
            .get("data")
            .and_then(Value::as_array)
            .or_else(|| parsed.as_array());
        let Some(items) = items else {
            return Ok(fallback());
        };
        let out: Vec<ModelInfo> = items
            .iter()
            .filter_map(|m| {
                let id = m.get("id").and_then(Value::as_str)?;
                Some(ModelInfo {
                    id: id.to_owned(),
                    display_name: m.get("name").and_then(Value::as_str).map(str::to_owned),
                    description: None,
                    tags: Vec::new(),
                })
            })
            .collect();
        if out.is_empty() {
            Ok(fallback())
        } else {
            Ok(out)
        }
    }

    async fn chat(&self, ctx: &ChatCtx<'_>) -> Result<ChatResponse> {
        let req = self.build_request(ctx);
        let model_full = ctx.request.model.full_name();
        let (status, _h, byte_stream) = ctx.http.stream(req).await?;
        if status != 200 {
            return Err(glm_error("glm_request_failed", status, "GLM request failed"));
        }
        // Buffer the stream into a final text.
        let mut text = String::new();
        let events = gravity_http::sse::events(byte_stream);
        futures::pin_mut!(events);
        use futures::StreamExt;
        while let Some(ev) = events.next().await {
            let ev = ev?;
            if ev.is_done() {
                break;
            }
            match parse_chunk(&ev.data)? {
                Chunk::Delta(d) => text.push_str(&d),
                Chunk::Done => break,
                Chunk::Skip => {}
            }
        }
        Ok(ChatResponse {
            model: model_full.into(),
            message: Message::assistant(text.trim().to_owned()),
            usage: Usage::default(),
            tool_calls_executed: Vec::new(),
            provider_response_id: None,
            finish_reason: Some("stop".into()),
            metadata: Metadata::new(),
        })
    }

    async fn chat_stream(&self, ctx: &ChatCtx<'_>) -> Result<EventStream> {
        let req = self.build_request(ctx);
        let model_full = ctx.request.model.full_name();
        let (status, _h, byte_stream) = ctx.http.stream(req).await?;
        if status != 200 {
            return Err(glm_error("glm_stream_failed", status, "GLM stream failed"));
        }
        let events = gravity_http::sse::events(byte_stream);
        let out = async_stream::try_stream! {
            use futures::StreamExt;
            futures::pin_mut!(events);
            let mut text = String::new();
            while let Some(ev) = events.next().await {
                let ev = ev?;
                if ev.is_done() { break; }
                match parse_chunk(&ev.data)? {
                    Chunk::Delta(d) => {
                        text.push_str(&d);
                        yield StreamEvent::TextDelta(bytes::Bytes::from(d));
                    }
                    Chunk::Done => break,
                    Chunk::Skip => {}
                }
            }
            let resp = ChatResponse {
                model: model_full.into(),
                message: Message::assistant(text.trim().to_owned()),
                usage: Usage::default(),
                tool_calls_executed: Vec::new(),
                provider_response_id: None,
                finish_reason: Some("stop".into()),
                metadata: Metadata::new(),
            };
            yield StreamEvent::Done(Box::new(resp));
        };
        Ok(Box::pin(out))
    }
}

/// One parsed GLM SSE chunk.
#[derive(Debug)]
enum Chunk {
    /// A text delta to append/emit.
    Delta(String),
    /// Terminal `[DONE]`.
    Done,
    /// Non-content frame.
    Skip,
}

/// Parse one chat.z.ai SSE `data:` payload (`_yield_stream_chunks` logic).
///
/// The frame nests a `data` object carrying `phase` / `delta_content` /
/// `content`; an embedded `error` raises a provider error.
fn parse_chunk(raw: &str) -> Result<Chunk> {
    if raw.trim() == "[DONE]" {
        return Ok(Chunk::Done);
    }
    let v: Value = match serde_json::from_str(raw) {
        Ok(v) => v,
        Err(_) => return Ok(Chunk::Skip),
    };
    let data = match v.get("data") {
        Some(Value::Object(_)) => &v["data"],
        _ => return Ok(Chunk::Skip),
    };
    if let Some(err) = data.get("error").and_then(Value::as_object) {
        let detail = err
            .get("detail")
            .and_then(Value::as_str)
            .unwrap_or("GLM returned a streaming error");
        let code = err.get("code").and_then(Value::as_str).unwrap_or("glm_stream_error");
        return Err(Error::Provider(ProviderError::new(
            ProviderName::Glm,
            ProviderErrorKind::Other,
            code.to_lowercase(),
            detail.to_owned(),
        )));
    }
    let phase = data.get("phase").and_then(Value::as_str);
    if let Some(delta) = data.get("delta_content").and_then(Value::as_str) {
        if phase == Some("answer") || phase.is_none() {
            return Ok(Chunk::Delta(delta.to_owned()));
        }
    }
    if let Some(content) = data.get("content").and_then(Value::as_str) {
        if !content.is_empty() {
            return Ok(Chunk::Delta(content.to_owned()));
        }
    }
    Ok(Chunk::Skip)
}

fn glm_error(code: &'static str, status: u16, msg: &str) -> Error {
    Error::Provider(
        ProviderError::new(ProviderName::Glm, ProviderErrorKind::Other, code, msg.to_owned())
            .with_status(status),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_answer_delta() {
        let raw = r#"{"data":{"phase":"answer","delta_content":"Hello"}}"#;
        assert!(matches!(parse_chunk(raw).unwrap(), Chunk::Delta(d) if d == "Hello"));
    }

    #[test]
    fn skips_non_answer_thinking_phase_delta() {
        let raw = r#"{"data":{"phase":"thinking","delta_content":"hmm"}}"#;
        assert!(matches!(parse_chunk(raw).unwrap(), Chunk::Skip));
    }

    #[test]
    fn surfaces_stream_error() {
        let raw = r#"{"data":{"error":{"detail":"rate limited","code":"RATE_LIMIT"}}}"#;
        let err = parse_chunk(raw).unwrap_err();
        match err {
            Error::Provider(p) => assert_eq!(&*p.code, "rate_limit"),
            _ => panic!("expected provider error"),
        }
    }

    #[test]
    fn done_terminates() {
        assert!(matches!(parse_chunk("[DONE]").unwrap(), Chunk::Done));
    }
}
