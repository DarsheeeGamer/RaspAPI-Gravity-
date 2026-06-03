//! The ChatGPT web provider.
//!
//! Ports the chat path of `gravity/chatgpt/client.py`: fetch Sentinel
//! requirements, solve the [`super::pow`] proof-of-work, then POST
//! `/backend-api/conversation` with the sentinel tokens and parse the SSE.
//! Auth (access token, device id) comes from the account. Arkose/Turnstile are
//! attempted only when the server requires them.

use super::pow;
use crate::context::ChatCtx;
use crate::provider::{EventStream, Provider};
use crate::util::uuid_v4;
use async_trait::async_trait;
use gravity_core::{
    ChatResponse, Checkpoint, Error, HistoryMode, Message, Metadata, ProviderCapabilities,
    ProviderError, ProviderErrorKind, ProviderName, RemoteSessionMode, Result, StreamEvent,
    StructuredOutputMode, ToolSupportMode, Usage,
};
use gravity_http::{HttpRequest, Profile};
use serde_json::{json, Value};

const BACKEND_BASE: &str = "https://chatgpt.com/backend-api";
const USER_AGENT: &str = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 \
(KHTML, like Gecko) Chrome/136.0.0.0 Safari/537.36";

/// The ChatGPT web adapter.
pub struct ChatgptProvider {
    capabilities: ProviderCapabilities,
}

impl Default for ChatgptProvider {
    fn default() -> Self {
        ChatgptProvider::new()
    }
}

impl ChatgptProvider {
    /// Construct the provider.
    pub fn new() -> Self {
        ChatgptProvider {
            capabilities: ProviderCapabilities {
                provider: ProviderName::Chatgpt,
                tool_support: ToolSupportMode::JsonEmulated,
                structured_output: StructuredOutputMode::JsonInstruction,
                remote_session_mode: RemoteSessionMode::Required,
                supports_system_prompt: false,
                supports_stateful_history: true,
                requires_remote_checkpoint: true,
                default_history_mode: HistoryMode::Auto,
                supports_image_generation: false,
            },
        }
    }

    fn profile(ctx: &ChatCtx<'_>) -> Profile {
        Profile::from_name(ctx.impersonate_or("chrome"))
    }

    fn base_headers(ctx: &ChatCtx<'_>) -> Vec<(String, String)> {
        let mut h = vec![
            ("user-agent".to_string(), USER_AGENT.to_string()),
            ("accept".to_string(), "*/*".to_string()),
            ("authorization".to_string(), format!("Bearer {}", ctx.secret)),
            ("content-type".to_string(), "application/json".to_string()),
            ("origin".to_string(), "https://chatgpt.com".to_string()),
            ("referer".to_string(), "https://chatgpt.com/".to_string()),
        ];
        if let Some(dev) = ctx.account.auth.device_id.as_deref() {
            h.push(("oai-device-id".to_string(), dev.to_string()));
        }
        if let Some(acct) = ctx
            .account
            .auth
            .chatgpt_account_id
            .as_deref()
            .or_else(|| ctx.account.auth.inline_secret("chatgpt_account_id"))
        {
            h.push(("chatgpt-account-id".to_string(), acct.to_string()));
        }
        h
    }

    /// A default PoW config (fallback build metadata; randomized via env).
    fn pow_config(env: &gravity_core::Env) -> Vec<Value> {
        let cores = [8, 16, 24, 32][(env.entropy.next_u64() % 4) as usize];
        json!([
            if env.entropy.next_u64() % 2 == 0 { 3000 } else { 4000 },
            "Mon Jan 01 2024 00:00:00 GMT-0500 (Eastern Standard Time)",
            4294705152u64, 0, USER_AGENT,
            "https://chatgpt.com/backend-api/sentinel/sdk.js", "",
            "en-US", "en-US,en", 0,
            "webdriver-false", "location", "document",
            (env.clock.now_millis() % 100000) as f64,
            uuid_v4(env), "", cores, 0.0
        ])
        .as_array()
        .unwrap()
        .clone()
    }

    /// Fetch sentinel requirements and solve the PoW, returning the tokens.
    async fn get_requirements(&self, ctx: &ChatCtx<'_>) -> Result<(String, Option<String>)> {
        let config = Self::pow_config(ctx.env);
        let requirements_token = pow::build_requirements_token("0.5", &config);
        let url = format!("{BACKEND_BASE}/sentinel/chat-requirements");
        let mut req = HttpRequest::post(url)
            .profile(Self::profile(ctx))
            .timeout_ms(ctx.timeout_ms())
            .proxy(ctx.proxy())
            .json(&json!({ "p": requirements_token }));
        for (k, v) in Self::base_headers(ctx) {
            req = req.header(k, v);
        }
        let resp = ctx.http.send(req).await?;
        if !resp.is_success() {
            return Err(cg_err("chatgpt_requirements_failed", resp.status, &resp.text()));
        }
        let payload: Value = resp.json()?;
        if payload.get("arkose").and_then(|a| a.get("required")).and_then(Value::as_bool) == Some(true) {
            return Err(Error::Provider(ProviderError::new(
                ProviderName::Chatgpt,
                ProviderErrorKind::Other,
                "chatgpt_arkose_required",
                "ChatGPT requested an Arkose token (not supported)",
            )));
        }
        let mut proof_token = None;
        if let Some(pow_info) = payload.get("proofofwork") {
            if pow_info.get("required").and_then(Value::as_bool) == Some(true) {
                let seed = pow_info.get("seed").and_then(Value::as_str).unwrap_or_default();
                let difficulty = pow_info.get("difficulty").and_then(Value::as_str).unwrap_or_default();
                if seed.is_empty() || difficulty.is_empty() {
                    return Err(cg_err("chatgpt_pow_invalid", 200, "missing seed/difficulty"));
                }
                let (token, solved) = pow::build_proof_token(seed, difficulty, &config);
                if !solved {
                    return Err(cg_err("chatgpt_pow_failed", 200, "could not solve PoW"));
                }
                proof_token = Some(token);
            }
        }
        let token = payload
            .get("token")
            .and_then(Value::as_str)
            .ok_or_else(|| cg_err("chatgpt_missing_token", 200, "no chat-requirements token"))?
            .to_owned();
        Ok((token, proof_token))
    }

    fn conversation_body(&self, ctx: &ChatCtx<'_>) -> Value {
        let messages: Vec<Value> = ctx
            .request
            .messages
            .iter()
            .map(|m| {
                json!({
                    "id": uuid_v4(ctx.env),
                    "author": { "role": role_str(m.role) },
                    "content": { "content_type": "text", "parts": [m.text()] },
                    "metadata": {},
                })
            })
            .collect();
        // Randomize contextual info to match real browser variance (from Python).
        let time_since = 50 + (ctx.env.entropy.next_u64() % 450);
        let page_h = 700 + (ctx.env.entropy.next_u64() % 700);
        let page_w = 1100 + (ctx.env.entropy.next_u64() % 1100);
        let scr_h  = 900 + (ctx.env.entropy.next_u64() % 500);
        let scr_w  = 1400 + (ctx.env.entropy.next_u64() % 1200);

        // Use checkpoint conversation/parent ids for multi-turn continuity.
        let parent_message_id = ctx
            .checkpoint
            .and_then(|c| c.remote_parent_id.as_deref())
            .map(str::to_owned)
            .unwrap_or_else(|| uuid_v4(ctx.env));
        let mut body = json!({
            "action": "next",
            "client_contextual_info": {
                "is_dark_mode": false,
                "time_since_loaded": time_since,
                "page_height": page_h, "page_width": page_w, "pixel_ratio": 1.5,
                "screen_height": scr_h, "screen_width": scr_w,
            },
            "conversation_mode": { "kind": "primary_assistant" },
            "force_paragen": false,
            "force_paragen_model_slug": "",
            "force_rate_limit": false,
            "force_use_sse": true,
            "history_and_training_disabled": false,
            "messages": messages,
            "model": ctx.request.model.model,
            "paragen_cot_summary_display_override": "allow",
            "parent_message_id": parent_message_id,
            "reset_rate_limits": false,
            "suggestions": [],
            "supported_encodings": [],
            "system_hints": [],
            "timezone": "Asia/Calcutta",
            "timezone_offset_min": 330,
            "variant_purpose": "comparison_implicit",
            "websocket_request_id": uuid_v4(ctx.env),
        });
        // Thread conversation_id from checkpoint for continuations.
        if let Some(conv_id) = ctx.checkpoint.and_then(|c| c.remote_conversation_id.as_deref()) {
            body.as_object_mut().unwrap()
                .insert("conversation_id".to_string(), Value::String(conv_id.to_owned()));
        }
        body
    }
}

#[async_trait]
impl Provider for ChatgptProvider {
    fn capabilities(&self) -> &ProviderCapabilities {
        &self.capabilities
    }

    fn list_models(&self) -> Vec<String> {
        ["gpt-5", "gpt-4o", "o4-mini", "auto"].iter().map(|s| s.to_string()).collect()
    }

    /// Fetch the model picker list from `GET /backend-api/models`
    /// (`{"models":[{"slug","title","description","tags"}]}`). Requires a
    /// session token — anonymous accounts have no enumerable list, so they fall
    /// back to the static set.
    async fn discover_models(&self, ctx: &ChatCtx<'_>) -> Result<Vec<gravity_core::ModelInfo>> {
        use gravity_core::ModelInfo;
        let fallback = || self.list_models().into_iter().map(ModelInfo::bare).collect::<Vec<_>>();

        // No token → anonymous; the /models endpoint is auth-only.
        if ctx.secret.is_empty() {
            return Ok(fallback());
        }
        let url = format!("{BACKEND_BASE}/models?history_and_training_disabled=false");
        let mut req = HttpRequest::get(url)
            .profile(Self::profile(ctx))
            .timeout_ms(ctx.timeout_ms())
            .proxy(ctx.proxy());
        for (k, v) in Self::base_headers(ctx) {
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
        let Some(models) = parsed.get("models").and_then(Value::as_array) else {
            return Ok(fallback());
        };
        let out: Vec<ModelInfo> = models
            .iter()
            .filter_map(|m| {
                let id = m.get("slug").and_then(Value::as_str)?;
                let tags = m
                    .get("tags")
                    .and_then(Value::as_array)
                    .map(|a| a.iter().filter_map(|t| t.as_str().map(str::to_owned)).collect())
                    .unwrap_or_default();
                Some(ModelInfo {
                    id: id.to_owned(),
                    display_name: m.get("title").and_then(Value::as_str).map(str::to_owned),
                    description: m.get("description").and_then(Value::as_str).map(str::to_owned),
                    tags,
                })
            })
            .collect();
        if out.is_empty() {
            Ok(fallback())
        } else {
            Ok(out)
        }
    }

    /// Open a placeholder checkpoint so multi-turn conversation IDs can be
    /// threaded. The actual `conversation_id` is extracted from the first SSE
    /// response and stored by the caller in `Checkpoint.remote_conversation_id`.
    async fn open_checkpoint(&self, ctx: &ChatCtx<'_>) -> Result<Option<Checkpoint>> {
        use chrono::Utc;
        let existing = ctx.checkpoint.filter(|c| c.remote_conversation_id.is_some());
        if let Some(c) = existing {
            // Pass through an existing valid checkpoint (second turn+).
            return Ok(Some(c.clone()));
        }
        Ok(Some(Checkpoint {
            conversation_key: ctx.conversation.conversation_key.clone(),
            provider: ProviderName::Chatgpt,
            account_id: ctx.account.account_id.clone(),
            remote_conversation_id: None,
            remote_parent_id: None,
            remote_metadata: gravity_core::Metadata::new(),
            history_version: ctx.conversation.history_version,
            updated_at: Utc::now(),
            is_valid: true,
        }))
    }

    async fn chat(&self, ctx: &ChatCtx<'_>) -> Result<ChatResponse> {
        let (req_token, proof_token) = self.get_requirements(ctx).await?;
        let body = self.conversation_body(ctx);
        let mut req = HttpRequest::post(format!("{BACKEND_BASE}/conversation"))
            .profile(Self::profile(ctx))
            .timeout_ms(ctx.timeout_ms())
            .proxy(ctx.proxy())
            .header("accept", "text/event-stream")
            .json(&body);
        for (k, v) in Self::base_headers(ctx) {
            if k == "accept" {
                continue;
            }
            req = req.header(k, v);
        }
        req = req.header("openai-sentinel-chat-requirements-token", req_token);
        if let Some(pt) = proof_token {
            req = req.header("openai-sentinel-proof-token", pt);
        }

        let resp = ctx.http.send(req).await?;
        if !resp.is_success() {
            return Err(cg_err("chatgpt_chat_failed", resp.status, &resp.text()));
        }
        let (text, msg_id, conv_id) = accumulate_sse(&resp.text())?;
        let mut meta = Metadata::new();
        if let Some(ref cid) = conv_id {
            meta.insert("conversation_id", cid.clone());
        }
        Ok(ChatResponse {
            model: ctx.request.model.full_name().into(),
            message: Message::assistant(text),
            usage: Usage::default(),
            tool_calls_executed: Vec::new(),
            provider_response_id: msg_id.map(Into::into),
            finish_reason: Some("stop".into()),
            metadata: meta,
        })
    }

    async fn chat_stream(&self, ctx: &ChatCtx<'_>) -> Result<EventStream> {
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

/// Accumulate the ChatGPT conversation SSE into (text, message_id, conversation_id).
///
/// Each `data:` frame carries `message.content.parts`; ChatGPT sends the full
/// running text each frame, so the last non-empty `parts[0]` wins.
fn accumulate_sse(body: &str) -> Result<(String, Option<String>, Option<String>)> {
    let mut text = String::new();
    let mut msg_id = None;
    let mut conv_id = None;
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
        if let Some(err) = v.get("error").filter(|e| !e.is_null()) {
            return Err(Error::Provider(ProviderError::new(
                ProviderName::Chatgpt,
                ProviderErrorKind::Other,
                "chatgpt_sse_error",
                err.to_string(),
            )));
        }
        // Extract top-level conversation_id for checkpoint threading.
        if let Some(cid) = v.get("conversation_id").and_then(Value::as_str) {
            conv_id = Some(cid.to_owned());
        }
        let Some(message) = v.get("message").filter(|m| !m.is_null()) else {
            continue;
        };
        if let Some(id) = message.get("id").and_then(Value::as_str) {
            msg_id = Some(id.to_owned());
        }
        let content = message.get("content");
        if content.and_then(|c| c.get("content_type")).and_then(Value::as_str) == Some("text") {
            if let Some(part) = content
                .and_then(|c| c.get("parts"))
                .and_then(Value::as_array)
                .and_then(|p| p.first())
                .and_then(Value::as_str)
            {
                if !part.is_empty() {
                    text = part.to_owned();
                }
            }
        }
    }
    Ok((text, msg_id, conv_id))
}

fn role_str(role: gravity_core::Role) -> &'static str {
    use gravity_core::Role::*;
    match role {
        System => "system",
        User => "user",
        Assistant => "assistant",
        Tool => "tool",
    }
}

fn cg_err(code: &'static str, status: u16, body: &str) -> Error {
    let snippet: String = body.chars().take(400).collect();
    Error::Provider(
        ProviderError::new(ProviderName::Chatgpt, ProviderErrorKind::Other, code, snippet)
            .with_status(status),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn capabilities() {
        let p = ChatgptProvider::new();
        assert_eq!(p.capabilities().provider, ProviderName::Chatgpt);
    }

    #[test]
    fn accumulate_sse_takes_last_full_text() {
        let body = "data: {\"message\":{\"id\":\"m1\",\"content\":{\"content_type\":\"text\",\"parts\":[\"Hel\"]}}}\n\
data: {\"message\":{\"id\":\"m1\",\"content\":{\"content_type\":\"text\",\"parts\":[\"Hello world\"]}}}\n\
data: [DONE]";
        let (text, id, conv_id) = accumulate_sse(body).unwrap();
        assert_eq!(text, "Hello world");
        assert_eq!(id.as_deref(), Some("m1"));
        assert!(conv_id.is_none());
    }

    #[test]
    fn accumulate_sse_surfaces_error() {
        let body = "data: {\"error\":{\"message\":\"rate limited\"}}";
        assert!(accumulate_sse(body).is_err());
    }
}
