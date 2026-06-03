//! The claude.ai web provider.
//!
//! Ports `gravity/claude/*`. Claude is a **stateful** provider: a remote
//! conversation must exist before completing, so [`ClaudeProvider::open_checkpoint`]
//! discovers the organization and creates a conversation, and the resulting
//! [`Checkpoint`] threads `parent_message_uuid` across turns.

mod auth;
mod constants;
mod parser;

use crate::context::ChatCtx;
use crate::provider::{EventStream, Provider};
use crate::util::uuid_v4;
use async_trait::async_trait;
use auth::ClaudeAuth;
use gravity_core::{
    ChatResponse, Checkpoint, Content, Error, Message, ProviderCapabilities, ProviderError,
    ProviderErrorKind, ProviderName, Result, Role, StreamEvent, ToolCall,
};
use gravity_http::{HttpRequest, Method, Profile};
use parser::Accumulator;
use serde_json::{json, Map, Value};

/// Capabilities for the Claude web provider (mirrors `CLAUDE_CAPABILITIES`).
fn capabilities() -> ProviderCapabilities {
    ProviderCapabilities {
        provider: ProviderName::Claude,
        tool_support: gravity_core::ToolSupportMode::Native,
        structured_output: gravity_core::StructuredOutputMode::JsonInstruction,
        remote_session_mode: gravity_core::RemoteSessionMode::Required,
        supports_system_prompt: true,
        supports_stateful_history: true,
        requires_remote_checkpoint: true,
        default_history_mode: gravity_core::HistoryMode::Auto,
        supports_image_generation: false,
    }
}

/// The claude.ai web provider adapter.
pub struct ClaudeProvider {
    capabilities: ProviderCapabilities,
}

impl Default for ClaudeProvider {
    fn default() -> Self {
        ClaudeProvider::new()
    }
}

impl ClaudeProvider {
    /// Construct the provider.
    pub fn new() -> Self {
        ClaudeProvider {
            capabilities: capabilities(),
        }
    }

    fn profile(ctx: &ChatCtx<'_>) -> Profile {
        Profile::from_name(ctx.impersonate_or("chrome"))
    }

    fn url(path: &str, org_id: &str, conversation_id: &str) -> String {
        let p = path
            .replace("{org_id}", org_id)
            .replace("{conversation_id}", conversation_id);
        format!("{}{}", constants::BASE_URL, p)
    }

    /// `GET /organizations` → pick a usable org id (routing-hint match, else first).
    async fn discover_org(&self, ctx: &ChatCtx<'_>, auth: &ClaudeAuth) -> Result<String> {
        if let Some(org) = auth.resolved_org_id() {
            return Ok(org.to_owned());
        }
        let url = format!("{}{}", constants::BASE_URL, constants::paths::ORGANIZATIONS);
        let mut req = HttpRequest::new(Method::GET, url)
            .profile(Self::profile(ctx))
            .timeout_ms(ctx.timeout_ms())
            .proxy(ctx.proxy());
        for (k, v) in auth.headers(false) {
            req = req.header(k, v);
        }
        let resp = ctx.http.send(req).await?;
        if !resp.is_success() {
            return Err(claude_error(
                "claude_org_discovery_failed",
                "failed to discover Claude organizations",
                resp.status,
            ));
        }
        let orgs: Value = resp.json()?;
        let arr = orgs.as_array().filter(|a| !a.is_empty()).ok_or_else(|| {
            claude_error("claude_no_organizations", "Claude returned no organizations", 200)
        })?;
        // routing-hint match first
        if let Some(rh) = &auth.routing_hint {
            if let Some(found) = arr
                .iter()
                .find(|o| o.get("uuid").and_then(Value::as_str) == Some(rh.as_str()))
            {
                if let Some(id) = found.get("uuid").and_then(Value::as_str) {
                    return Ok(id.to_owned());
                }
            }
        }
        arr[0]
            .get("uuid")
            .and_then(Value::as_str)
            .map(str::to_owned)
            .ok_or_else(|| claude_error("claude_no_organizations", "org has no uuid", 200))
    }

    /// `POST /chat_conversations` → create a conversation, returning its uuid.
    async fn create_conversation(
        &self,
        ctx: &ChatCtx<'_>,
        auth: &ClaudeAuth,
        org_id: &str,
    ) -> Result<String> {
        let conversation_id = uuid_v4(ctx.env);
        let url = Self::url(constants::paths::CONVERSATIONS, org_id, "");
        let body = json!({ "uuid": conversation_id, "name": "", "is_temporary": true });
        let mut req = HttpRequest::post(url)
            .profile(Self::profile(ctx))
            .timeout_ms(ctx.timeout_ms())
            .proxy(ctx.proxy())
            .json(&body);
        for (k, v) in auth.headers(false) {
            req = req.header(k, v);
        }
        let resp = ctx.http.send(req).await?;
        if !matches!(resp.status, 200 | 201) {
            return Err(claude_error(
                "claude_create_conversation_failed",
                "failed to create Claude conversation",
                resp.status,
            ));
        }
        Ok(conversation_id)
    }

    /// Build the completion request body, replicating Python `exclude_none`
    /// semantics (empty arrays kept; null scalars dropped).
    fn completion_body(&self, ctx: &ChatCtx<'_>, parent_message_uuid: Option<&str>) -> Value {
        let prompt = latest_prompt(ctx);
        let mut body = Map::new();
        body.insert("prompt".into(), Value::String(prompt));
        body.insert("model".into(), Value::String(ctx.request.model.model.to_string()));
        if let Some(sp) = self.normalize_system_prompt(ctx.request.system_prompt.as_deref()) {
            body.insert("custom_system_prompt".into(), Value::String(sp));
        }
        body.insert("personalized_styles".into(), Value::Array(vec![]));
        body.insert("attachments".into(), Value::Array(vec![]));
        body.insert("files".into(), Value::Array(vec![]));
        body.insert("sync_sources".into(), Value::Array(vec![]));
        // Optional locale/timezone — default to en-US / UTC.
        let timezone = ctx.account.auth.inline_secret("timezone").unwrap_or("UTC");
        let locale = ctx.account.auth.inline_secret("locale").unwrap_or("en-US");
        body.insert("timezone".into(), Value::String(timezone.to_owned()));
        body.insert("locale".into(), Value::String(locale.to_owned()));
        let tools: Vec<Value> = ctx
            .request
            .tools
            .iter()
            .map(|t| {
                json!({
                    "name": t.name,
                    "description": t.description,
                    "input_schema": t.input_schema,
                })
            })
            .collect();
        body.insert("tools".into(), Value::Array(tools));
        body.insert("rendering_mode".into(), Value::String("messages".into()));
        if let Some(parent) = parent_message_uuid {
            body.insert("parent_message_uuid".into(), Value::String(parent.to_owned()));
        }
        body.insert(
            "turn_message_uuids".into(),
            json!({
                "human_message_uuid": uuid_v4(ctx.env),
                "assistant_message_uuid": uuid_v4(ctx.env),
            }),
        );
        Value::Object(body)
    }

    /// Resolve the conversation id: checkpoint, else create one.
    async fn ensure_conversation(
        &self,
        ctx: &ChatCtx<'_>,
        auth: &ClaudeAuth,
        org_id: &str,
    ) -> Result<String> {
        if let Some(cp) = ctx.checkpoint {
            if let Some(id) = &cp.remote_conversation_id {
                return Ok(id.to_string());
            }
        }
        self.create_conversation(ctx, auth, org_id).await
    }

    /// POST tool results to `/tool_result` for each `Role::Tool` message.
    ///
    /// Claude's web API requires a separate endpoint per tool result before the
    /// next completion turn — unlike the Anthropic API which bundles them in
    /// `messages`. Silently skips non-Tool messages.
    async fn submit_tool_results(
        &self,
        ctx: &ChatCtx<'_>,
        auth: &ClaudeAuth,
        org_id: &str,
        conversation_id: &str,
    ) -> Result<()> {
        let tool_messages: Vec<&Message> = ctx
            .request
            .messages
            .iter()
            .filter(|m| m.role == Role::Tool)
            .collect();
        if tool_messages.is_empty() {
            return Ok(());
        }
        let url = Self::url(constants::paths::TOOL_RESULT, org_id, conversation_id);
        for msg in tool_messages {
            let Some(tr) = &msg.tool_result else { continue };
            let content_text = match &tr.content {
                Value::String(s) => s.clone(),
                other => other.to_string(),
            };
            let payload = json!({
                "type": "tool_result",
                "tool_use_id": &*tr.tool_call_id,
                "name": &*tr.name,
                "is_error": tr.is_error,
                "content": [{"type": "text", "text": content_text}],
            });
            let mut req = HttpRequest::post(url.clone())
                .profile(Self::profile(ctx))
                .timeout_ms(ctx.timeout_ms())
                .proxy(ctx.proxy())
                .json(&payload);
            for (k, v) in auth.headers(false) {
                req = req.header(k, v);
            }
            let resp = ctx.http.send(req).await?;
            if !resp.is_success() {
                return Err(claude_error(
                    "claude_tool_result_failed",
                    "Failed to submit Claude tool result",
                    resp.status,
                ));
            }
        }
        Ok(())
    }

    /// POST the completion and buffer the SSE body into a parsed response.
    async fn complete_buffered(
        &self,
        ctx: &ChatCtx<'_>,
        auth: &ClaudeAuth,
        org_id: &str,
        conversation_id: &str,
    ) -> Result<parser::ClaudeParsed> {
        let parent = ctx.checkpoint.and_then(|c| c.remote_parent_id.as_deref());
        let body = self.completion_body(ctx, parent);
        let url = Self::url(constants::paths::COMPLETION, org_id, conversation_id);
        let mut req = HttpRequest::post(url)
            .profile(Self::profile(ctx))
            .timeout_ms(ctx.timeout_ms())
            .proxy(ctx.proxy())
            .json(&body);
        for (k, v) in auth.headers(true) {
            req = req.header(k, v);
        }
        let resp = ctx.http.send(req).await?;
        if !resp.is_success() {
            return Err(claude_error(
                "claude_completion_failed",
                "Claude completion request failed",
                resp.status,
            ));
        }
        let mut acc = Accumulator::new();
        for line in resp.text().lines() {
            if let Some(event) = parse_sse_data_line(line)? {
                acc.push(&event);
            }
        }
        Ok(acc.finish())
    }
}

#[async_trait]
impl Provider for ClaudeProvider {
    fn capabilities(&self) -> &ProviderCapabilities {
        &self.capabilities
    }

    fn list_models(&self) -> Vec<String> {
        [
            "claude-sonnet-4-5-20250929",
            "claude-sonnet-4-6",
            "claude-haiku-4-5-20251001",
            "claude-sonnet-4-7",
            "claude-opus-4-7",
        ]
        .iter()
        .map(|s| s.to_string())
        .collect()
    }

    fn normalize_system_prompt(&self, system_prompt: Option<&str>) -> Option<String> {
        system_prompt
            .map(str::trim)
            .filter(|s| !s.is_empty())
            .map(str::to_owned)
    }

    async fn open_checkpoint(&self, ctx: &ChatCtx<'_>) -> Result<Option<Checkpoint>> {
        let auth = ClaudeAuth::from_account(ctx.account, ctx.secret)?;
        let org_id = self.discover_org(ctx, &auth).await?;
        let conversation_id = self.create_conversation(ctx, &auth, &org_id).await?;
        Ok(Some(Checkpoint {
            conversation_key: ctx.conversation.conversation_key.clone(),
            provider: ProviderName::Claude,
            account_id: ctx.account.account_id.clone(),
            remote_conversation_id: Some(conversation_id.into()),
            remote_parent_id: None,
            remote_metadata: gravity_core::Metadata::new(),
            history_version: ctx.conversation.history_version,
            updated_at: now(ctx),
            is_valid: true,
        }))
    }

    async fn chat(&self, ctx: &ChatCtx<'_>) -> Result<ChatResponse> {
        let auth = ClaudeAuth::from_account(ctx.account, ctx.secret)?;
        let org_id = self.discover_org(ctx, &auth).await?;
        let conversation_id = self.ensure_conversation(ctx, &auth, &org_id).await?;
        self.submit_tool_results(ctx, &auth, &org_id, &conversation_id).await?;
        let parsed = self
            .complete_buffered(ctx, &auth, &org_id, &conversation_id)
            .await?;
        Ok(build_response(ctx, parsed))
    }

    async fn chat_stream(&self, ctx: &ChatCtx<'_>) -> Result<EventStream> {
        // Move everything the stream needs out of the borrowed context.
        let auth = ClaudeAuth::from_account(ctx.account, ctx.secret)?;
        let org_id = self.discover_org(ctx, &auth).await?;
        let conversation_id = self.ensure_conversation(ctx, &auth, &org_id).await?;
        self.submit_tool_results(ctx, &auth, &org_id, &conversation_id).await?;
        let parent = ctx.checkpoint.and_then(|c| c.remote_parent_id.as_deref());
        let body = self.completion_body(ctx, parent);
        let url = Self::url(constants::paths::COMPLETION, &org_id, &conversation_id);
        let profile = Self::profile(ctx);
        let timeout_ms = ctx.timeout_ms();
        let proxy = ctx.proxy();
        let model_full = ctx.request.model.full_name();
        let headers = auth.headers(true);
        let http = ctx.http;

        let mut req = HttpRequest::post(url)
            .profile(profile)
            .timeout_ms(timeout_ms)
            .proxy(proxy)
            .json(&body);
        for (k, v) in headers {
            req = req.header(k, v);
        }

        let (status, _hdrs, byte_stream) = http.stream(req).await?;
        if status != 200 {
            return Err(claude_error(
                "claude_completion_failed",
                "Claude completion request failed",
                status,
            ));
        }

        let event_stream = gravity_http::sse::events(byte_stream);
        let out = async_stream::try_stream! {
            use futures::StreamExt;
            futures::pin_mut!(event_stream);
            let mut acc = Accumulator::new();
            while let Some(sse) = event_stream.next().await {
                let sse = sse?;
                if sse.is_done() { continue; }
                let event: Value = serde_json::from_str(&sse.data)
                    .map_err(|e| Error::Network(format!("invalid Claude SSE JSON: {e}")))?;
                if event.get("type").and_then(Value::as_str) == Some("error") {
                    Err(sse_error_event(&event))?;
                }
                if let Some(delta) = acc.push(&event) {
                    yield StreamEvent::TextDelta(bytes::Bytes::from(delta));
                }
            }
            let parsed = acc.finish();
            let resp = build_response_with_model(&model_full, parsed);
            yield StreamEvent::Done(Box::new(resp));
        };
        Ok(Box::pin(out))
    }
}

// ── free helpers ────────────────────────────────────────────────────────────

fn now(ctx: &ChatCtx<'_>) -> chrono::DateTime<chrono::Utc> {
    chrono::DateTime::from_timestamp_millis(ctx.env.clock.now_millis() as i64)
        .unwrap_or_else(chrono::Utc::now)
}

/// The prompt to send: on a fresh conversation (no remote parent yet) render
/// the FULL transcript so prior history isn't lost; on a continuing checkpoint
/// send only the latest message. Mirrors the `rebuild` branch in
/// `gravity/claude/adapter.py`.
fn latest_prompt(ctx: &ChatCtx<'_>) -> String {
    let rebuild = ctx
        .checkpoint
        .map(|c| c.remote_parent_id.is_none())
        .unwrap_or(true);
    if rebuild && ctx.request.messages.len() > 1 {
        render_transcript(&ctx.request.messages)
    } else {
        ctx.request.messages.last().map(message_to_text).unwrap_or_default()
    }
}

/// Render a transcript as `ROLE: text` blocks (Python `_render_transcript`).
fn render_transcript(messages: &[Message]) -> String {
    let mut lines = Vec::new();
    for m in messages {
        let role = match m.role {
            gravity_core::Role::System => "SYSTEM",
            gravity_core::Role::User => "USER",
            gravity_core::Role::Assistant => "ASSISTANT",
            gravity_core::Role::Tool => "TOOL_RESULT",
        };
        let content = message_to_text(m);
        if content.is_empty() && m.tool_calls.is_empty() {
            continue;
        }
        lines.push(format!("{role}: {content}").trim_end().to_string());
        for tc in &m.tool_calls {
            lines.push(format!("ASSISTANT_TOOL_CALL[{}]: {}", tc.name, serde_json::Value::Object(tc.arguments.clone())));
        }
    }
    lines.join("\n\n").trim().to_string()
}

fn message_to_text(m: &Message) -> String {
    match &m.content {
        Some(Content::Text(s)) => s.clone(),
        Some(Content::Parts(parts)) => parts
            .iter()
            .filter_map(|p| match p {
                gravity_core::ContentPart::Text { text } => Some(text.clone()),
                gravity_core::ContentPart::Json { data } => Some(Value::Object(data.clone()).to_string()),
                _ => None,
            })
            .collect::<Vec<_>>()
            .join("\n"),
        None => m
            .tool_result
            .as_ref()
            .map(|r| r.content.to_string())
            .unwrap_or_default(),
    }
}

fn build_response(ctx: &ChatCtx<'_>, parsed: parser::ClaudeParsed) -> ChatResponse {
    build_response_with_model(&ctx.request.model.full_name(), parsed)
}

fn build_response_with_model(model_full: &str, parsed: parser::ClaudeParsed) -> ChatResponse {
    let tool_calls: Vec<ToolCall> = parsed
        .tool_calls
        .iter()
        .map(|tc| ToolCall {
            id: tc.id.clone().into(),
            name: tc.name.clone().into(),
            arguments: tc.arguments(),
        })
        .collect();

    let message = if tool_calls.is_empty() {
        Message::assistant(parsed.text)
    } else {
        Message::assistant_with_tools(Some(Content::Text(parsed.text)), tool_calls)
    };

    ChatResponse {
        model: model_full.into(),
        message,
        usage: gravity_core::Usage::default(),
        tool_calls_executed: Vec::new(),
        provider_response_id: parsed.assistant_message_uuid.map(Into::into),
        finish_reason: parsed.stop_reason.map(Into::into),
        metadata: gravity_core::Metadata::new(),
    }
}

fn parse_sse_data_line(line: &str) -> Result<Option<Value>> {
    let Some(data) = line.strip_prefix("data: ") else {
        return Ok(None);
    };
    match serde_json::from_str::<Value>(data) {
        Ok(v) => {
            if v.get("type").and_then(Value::as_str) == Some("error") {
                return Err(sse_error_event(&v));
            }
            Ok(Some(v))
        }
        Err(_) => Ok(None), // tolerate non-JSON keep-alives, like the Python loop
    }
}

fn claude_error(code: &'static str, msg: &str, status: u16) -> Error {
    Error::Provider(
        ProviderError::new(ProviderName::Claude, ProviderErrorKind::Other, code, msg.to_owned())
            .with_status(status),
    )
}

fn sse_error_event(event: &Value) -> Error {
    let err = event.get("error").cloned().unwrap_or(Value::Null);
    let msg = err
        .get("message")
        .and_then(Value::as_str)
        .unwrap_or("Claude returned an SSE error event")
        .to_owned();
    Error::Provider(ProviderError::new(
        ProviderName::Claude,
        ProviderErrorKind::Other,
        "claude_sse_error",
        msg,
    ))
}

pub use ClaudeProvider as Adapter;

#[cfg(test)]
mod tests {
    use super::*;
    use gravity_core::{ChatRequest, Conversation, Env, ModelId};

    fn ctx_account() -> gravity_core::ProviderAccount {
        use gravity_core::*;
        ProviderAccount {
            account_id: "a".into(),
            provider: ProviderName::Claude,
            enabled: true,
            pool: "default".into(),
            priority: 100,
            weight: 1,
            auth: AuthConfig {
                kind: "session_cookie".into(),
                secret_ref: "ref".into(),
                account_label: None,
                org_id: Some("org-1".into()),
                device_id: Some("dev-1".into()),
                routing_hint: None,
                chatgpt_account_id: None,
                refresh_secret_ref: None,
                extra: Metadata::new(),
            },
            transport: TransportConfig {
                base_url: constants::BASE_URL.into(),
                timeout_ms: 60000,
                impersonate: None,
                proxy_url: None,
                verify_ssl: true,
                extra_headers: Default::default(),
            },
            rotation_state: Default::default(),
            quota: Default::default(),
            defaults: ProviderDefaults {
                tool_support: ToolSupportMode::Native,
                structured_output: StructuredOutputMode::JsonInstruction,
                remote_session_mode: RemoteSessionMode::Required,
                supports_system_prompt: true,
                default_history_mode: HistoryMode::Auto,
            },
            metadata: Metadata::new(),
        }
    }

    #[test]
    fn completion_body_shape_matches_python() {
        let provider = ClaudeProvider::new();
        let account = ctx_account();
        let conv = Conversation::new("k");
        let http = gravity_http::HttpClient::new();
        let env = Env::fixed(1_700_000_000_000, 5);
        let mut req = ChatRequest::new(
            ModelId::parse("claude/claude-sonnet-4-6").unwrap(),
            vec![Message::user("Hello Claude")],
        );
        req.system_prompt = Some("  Be terse.  ".into());
        let ctx = ChatCtx {
            account: &account,
            secret: "sk-ant-sid-x",
            conversation: &conv,
            checkpoint: None,
            request: &req,
            http: &http,
            env: &env,
        };
        let body = provider.completion_body(&ctx, Some("parent-1"));
        assert_eq!(body["prompt"], "Hello Claude");
        assert_eq!(body["model"], "claude-sonnet-4-6");
        assert_eq!(body["custom_system_prompt"], "Be terse."); // trimmed
        assert_eq!(body["rendering_mode"], "messages");
        assert_eq!(body["parent_message_uuid"], "parent-1");
        assert!(body["personalized_styles"].is_array());
        assert!(body["turn_message_uuids"]["human_message_uuid"].is_string());
    }

    #[test]
    fn capabilities_match_python() {
        use gravity_core::RemoteSessionMode;
        let p = ClaudeProvider::new();
        let c = p.capabilities();
        assert_eq!(c.provider, ProviderName::Claude);
        assert!(c.requires_remote_checkpoint);
        assert_eq!(c.remote_session_mode, RemoteSessionMode::Required);
    }
}
