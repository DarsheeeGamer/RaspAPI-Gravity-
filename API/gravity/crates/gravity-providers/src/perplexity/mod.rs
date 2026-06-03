//! Perplexity web provider — anonymous access via `perplexity.ai`.
//!
//! Ports `gravity/perplexity/adapter.py` + `client.py`.
//!
//! **Protocol**: Two-phase:
//! 1. `GET /api/auth/session` — bootstrap cookies (cookie-store client).
//! 2. `POST /rest/sse/perplexity_ask` — query with SSE streaming response.
//!
//! Unlike the OAI-compat Perplexity adapter (API key, `api.perplexity.ai`),
//! this provider talks to the **web interface** and works without credentials
//! (anonymous mode). Authenticated accounts can pass cookies via
//! `account.auth.extra["cookies"]`.

use crate::context::ChatCtx;
use crate::provider::{EventStream, Provider};
use async_trait::async_trait;
use gravity_core::{
    ChatResponse, Content, Error, HistoryMode, Message, Metadata, ProviderCapabilities,
    ProviderError, ProviderErrorKind, ProviderName, RemoteSessionMode, Result, Role, StreamEvent,
    StructuredOutputMode, ToolCall, ToolSupportMode,
};
use gravity_http::{HttpRequest, Profile};
use serde_json::{json, Value};

const BASE_URL: &str = "https://www.perplexity.ai";
const SSE_ASK_URL: &str = "https://www.perplexity.ai/rest/sse/perplexity_ask";
const AUTH_SESSION_URL: &str = "https://www.perplexity.ai/api/auth/session";
const API_VERSION: &str = "2.18";

const DEFAULT_HEADERS: &[(&str, &str)] = &[
    ("accept", "application/json, text/plain, */*"),
    ("accept-language", "en-US,en;q=0.9"),
    ("content-type", "application/json"),
    ("origin", BASE_URL),
    ("referer", "https://www.perplexity.ai/"),
    (
        "user-agent",
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) \
         AppleWebKit/537.36 (KHTML, like Gecko) Chrome/136.0.0.0 Safari/537.36",
    ),
];

/// Perplexity web provider — works anonymously without an API key.
pub struct PerplexityWebProvider {
    capabilities: ProviderCapabilities,
}

impl Default for PerplexityWebProvider {
    fn default() -> Self {
        PerplexityWebProvider::new()
    }
}

impl PerplexityWebProvider {
    /// Construct the provider.
    pub fn new() -> Self {
        PerplexityWebProvider {
            capabilities: ProviderCapabilities {
                provider: ProviderName::Perplexity,
                tool_support: ToolSupportMode::JsonEmulated,
                structured_output: StructuredOutputMode::JsonInstruction,
                remote_session_mode: RemoteSessionMode::Optional,
                supports_system_prompt: false,
                supports_stateful_history: false,
                requires_remote_checkpoint: false,
                default_history_mode: HistoryMode::Stateless,
                supports_image_generation: false,
            },
        }
    }

    fn profile(ctx: &ChatCtx<'_>) -> Profile {
        Profile::from_name(ctx.impersonate_or("chrome136"))
    }

    /// Bootstrap the session by hitting the auth endpoint so cookies are set.
    async fn bootstrap(&self, ctx: &ChatCtx<'_>) -> Result<()> {
        let req = HttpRequest::get(AUTH_SESSION_URL)
            .profile(Self::profile(ctx))
            .timeout_ms(ctx.timeout_ms())
            .proxy(ctx.proxy())
            .cookie_store(true);
        // best-effort — ignore errors (anonymous mode doesn't strictly need auth cookies)
        let _ = ctx.http.send(req).await;
        Ok(())
    }

    /// Build the query string from messages (history inlined for anonymous).
    fn build_query(ctx: &ChatCtx<'_>) -> String {
        let messages = &ctx.request.messages;
        if messages.len() > 1 {
            render_history(messages)
        } else {
            messages.last().map(message_text).unwrap_or_default()
        }
    }

    /// Build the `perplexity_ask` payload.
    ///
    /// Honors account `extra` overrides:
    /// - `mode`              — `"auto"` (→concise) | `"copilot"` | `"pro"` | `"reasoning"`
    /// - `model_preference`  — `"turbo"` (anon) | `"experimental"` | `"pplx_pro"` etc.
    /// - `sources`           — comma-sep `"web"`,`"scholar"`,`"social"`
    /// - `language`          — BCP-47 tag, e.g. `"fr-FR"`
    /// - `search_recency_filter` — `"day"` | `"week"` | `"month"` | `"hour"`
    /// - `last_backend_uuid` — pass to continue a prior server-side conversation
    fn ask_payload(ctx: &ChatCtx<'_>) -> Value {
        let query = Self::build_query(ctx);
        let incognito = ctx.account.auth.kind.as_ref() == "anonymous";

        // Read per-account overrides.
        let mode_str = ctx.account.auth.inline_secret("mode").unwrap_or("auto");
        let model_pref = ctx.account.auth.inline_secret("model_preference").unwrap_or("default");
        let language = ctx.account.auth.inline_secret("language").unwrap_or("en-US");
        let recency = ctx.account.auth.inline_secret("search_recency_filter");
        let last_uuid = ctx.account.auth.inline_secret("last_backend_uuid");

        // Parse sources: comma-separated string → array.
        let sources: Vec<&str> = ctx.account.auth
            .inline_secret("sources")
            .map(|s| s.split(',').map(str::trim).collect())
            .unwrap_or_else(|| vec!["web"]);

        let server_mode = if mode_str == "auto" || mode_str == "concise" { "concise" } else { "copilot" };

        let mut params = json!({
            "attachments": [],
            "frontend_context_uuid": crate::util::uuid_v4(ctx.env),
            "frontend_uuid": crate::util::uuid_v4(ctx.env),
            "is_incognito": incognito,
            "language": language,
            "last_backend_uuid": last_uuid,
            "mode": server_mode,
            "model_preference": model_pref,
            "source": "default",
            "sources": sources,
            "version": API_VERSION,
        });

        // Optional fields (silently accepted by server; some require auth to take effect).
        if let Some(r) = recency {
            params["search_recency_filter"] = Value::String(r.to_owned());
        }

        json!({ "query_str": query, "params": params })
    }

    /// Send the ask request and parse the SSE response.
    async fn ask(&self, ctx: &ChatCtx<'_>) -> Result<PplParsed> {
        let payload = Self::ask_payload(ctx);
        let mut req = HttpRequest::post(SSE_ASK_URL)
            .profile(Self::profile(ctx))
            .timeout_ms(ctx.timeout_ms())
            .proxy(ctx.proxy())
            .cookie_store(true)
            .header("accept", "text/event-stream")
            .header("content-type", "application/json")
            .header("origin", BASE_URL)
            .header("referer", "https://www.perplexity.ai/")
            .header("user-agent", DEFAULT_HEADERS[5].1);
        // Add cookies from account if available.
        if let Some(cookies) = ctx.account.auth.inline_secret("cookies") {
            req = req.header("cookie", cookies);
        }
        req = req.body(bytes::Bytes::from(serde_json::to_vec(&payload).unwrap()));

        let resp = ctx.http.send(req).await?;
        if !resp.is_success() {
            return Err(ppl_err("perplexity_ask_failed", resp.status, &resp.text()));
        }
        let body = resp.text().into_owned();
        Ok(parse_sse_response(&body))
    }
}

#[async_trait]
impl Provider for PerplexityWebProvider {
    fn capabilities(&self) -> &ProviderCapabilities {
        &self.capabilities
    }

    fn list_models(&self) -> Vec<String> {
        vec![
            "sonar".to_string(),
            "sonar-pro".to_string(),
            "sonar-reasoning".to_string(),
            "sonar-deep-research".to_string(),
        ]
    }

    async fn chat(&self, ctx: &ChatCtx<'_>) -> Result<ChatResponse> {
        self.bootstrap(ctx).await?;
        let parsed = self.ask(ctx).await?;
        let (message, finish_reason) = if !ctx.request.tools.is_empty() {
            // 1. Check native tool_calls from SSE (if Perplexity honored the tools field).
            if !parsed.tool_calls.is_empty() {
                let tcs: Vec<ToolCall> = parsed.tool_calls.iter().map(|tc| ToolCall {
                    id: crate::util::uuid_v4(ctx.env).into(),
                    name: tc.name.clone().into(),
                    arguments: tc.arguments.clone(),
                }).collect();
                (Message::assistant_with_tools(None, tcs), "tool_calls")
            } else {
                // 2. Fall back: plain text (no tool call found — model answered directly).
                (Message::assistant(parsed.answer), "stop")
            }
        } else {
            (Message::assistant(parsed.answer), "stop")
        };
        let mut meta = Metadata::new();
        if let Some(uuid) = &parsed.backend_uuid {
            meta.insert("backend_uuid", uuid.clone());
        }
        Ok(ChatResponse {
            model: ctx.request.model.full_name().into(),
            message,
            usage: gravity_core::Usage::default(),
            tool_calls_executed: Vec::new(),
            provider_response_id: parsed.backend_uuid.map(Into::into),
            finish_reason: Some(finish_reason.into()),
            metadata: meta,
        })
    }

    async fn chat_stream(&self, ctx: &ChatCtx<'_>) -> Result<EventStream> {
        // Buffer into a single Done event — Perplexity SSE is multi-chunk but
        // we extract the final answer so we emit it atomically.
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

// ── Query helpers ────────────────────────────────────────────────────────────

fn message_text(m: &Message) -> String {
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
        None => m.tool_result.as_ref().map(|r| r.content.to_string()).unwrap_or_default(),
    }
}

/// Render multi-turn history as a transcript prepended to the last query.
/// Anonymous Perplexity has no server-side memory, so we inline history.
fn render_history(messages: &[Message]) -> String {
    let mut parts = Vec::new();
    for m in messages {
        let role = match m.role {
            Role::User => "USER",
            Role::Assistant => "ASSISTANT",
            Role::System => "SYSTEM",
            Role::Tool => "TOOL_RESULT",
        };
        let text = message_text(m);
        if !text.is_empty() {
            parts.push(format!("{role}: {text}"));
        }
    }
    parts.join("\n\n")
}

// ── SSE parsing ──────────────────────────────────────────────────────────────

/// A native tool call returned inside a Perplexity SSE frame.
#[derive(Debug, Clone)]
pub struct PplToolCall {
    /// Tool name.
    pub name: String,
    /// Arguments object.
    pub arguments: serde_json::Map<String, Value>,
}

/// Parsed result from one Perplexity SSE response.
#[derive(Debug, Default)]
pub struct PplParsed {
    /// Final answer text.
    pub answer: String,
    /// Native tool calls (present when server honors the `tools` field).
    pub tool_calls: Vec<PplToolCall>,
    /// `backend_uuid` from last event (needed for follow-up requests).
    pub backend_uuid: Option<String>,
}

/// Parse Perplexity's SSE body (`event: message\r\ndata: {...}`, `\r\n\r\n`-delimited).
///
/// Extracts the final answer text, any native tool calls, and the backend_uuid.
/// Each `message` event's `text` field is a JSON string containing a list of steps;
/// the `FINAL` step holds `content.answer` (inner JSON `{"answer":"...", "chunks":[...]}`).
/// Native tool calls appear as `tool_calls` array in the top-level JSON frame.
fn parse_sse_response(body: &str) -> PplParsed {
    let mut parsed = PplParsed::default();
    for event in body.split("\r\n\r\n") {
        let event = event.trim();
        if event.is_empty() || event.contains("end_of_stream") {
            continue;
        }
        let data = if let Some(pos) = event.find("\r\ndata: ") {
            &event[pos + 8..]
        } else if let Some(stripped) = event.strip_prefix("data: ") {
            stripped
        } else {
            continue;
        };
        let Ok(v) = serde_json::from_str::<Value>(data) else { continue };

        // Track backend_uuid for follow-up continuity.
        if let Some(uuid) = v.get("backend_uuid").and_then(Value::as_str) {
            parsed.backend_uuid = Some(uuid.to_owned());
        }

        // Native tool_calls array (if Perplexity honored the tools field).
        if let Some(tcs) = v.get("tool_calls").and_then(Value::as_array) {
            for tc in tcs {
                // OpenAI-style: {id, type, function: {name, arguments}}
                let func = tc.get("function");
                let name = func
                    .and_then(|f| f.get("name"))
                    .and_then(Value::as_str)
                    .unwrap_or("")
                    .to_owned();
                if name.is_empty() { continue; }
                let args_raw = func.and_then(|f| f.get("arguments")).and_then(Value::as_str).unwrap_or("{}");
                let arguments = serde_json::from_str::<serde_json::Map<String, Value>>(args_raw)
                    .unwrap_or_default();
                parsed.tool_calls.push(PplToolCall { name, arguments });
            }
        }

        // Nested path: v.text → JSON steps list → FINAL → content.answer → JSON → .answer
        if let Some(text_raw) = v.get("text").and_then(Value::as_str) {
            if let Ok(steps) = serde_json::from_str::<Vec<Value>>(text_raw) {
                for step in &steps {
                    if step.get("step_type").and_then(Value::as_str) == Some("FINAL") {
                        if let Some(blob) = step.get("content").and_then(|c| c.get("answer")).and_then(Value::as_str) {
                            let text = if let Ok(inner) = serde_json::from_str::<Value>(blob) {
                                inner.get("answer").and_then(Value::as_str).unwrap_or(blob).to_owned()
                            } else {
                                blob.to_owned()
                            };
                            if !text.is_empty() { parsed.answer = text; }
                        }
                        break;
                    }
                }
            }
        }

        // Fallback: top-level answer field. Last non-empty wins.
        if let Some(a) = v.get("answer").and_then(Value::as_str) {
            if !a.is_empty() { parsed.answer = a.to_owned(); }
        }
    }
    parsed
}

fn ppl_err(code: &'static str, status: u16, body: &str) -> Error {
    let snippet: String = body.chars().take(400).collect();
    Error::Provider(
        ProviderError::new(ProviderName::Perplexity, ProviderErrorKind::Other, code, snippet)
            .with_status(status),
    )
}

pub use PerplexityWebProvider as Adapter;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_top_level_answer() {
        let body = "event: message\r\ndata: {\"answer\":\"The answer is 42.\",\"backend_uuid\":\"abc\"}\r\n\r\n\
event: end_of_stream\r\ndata: {}\r\n\r\n";
        let p = parse_sse_response(body);
        assert_eq!(p.answer, "The answer is 42.");
        assert_eq!(p.backend_uuid.as_deref(), Some("abc"));
    }

    #[test]
    fn last_answer_wins() {
        let body = "event: message\r\ndata: {\"answer\":\"partial\"}\r\n\r\n\
event: message\r\ndata: {\"answer\":\"The final answer.\"}\r\n\r\n\
event: end_of_stream\r\ndata: {}\r\n\r\n";
        assert_eq!(parse_sse_response(body).answer, "The final answer.");
    }

    #[test]
    fn parses_native_tool_calls() {
        let body = "event: message\r\ndata: {\"tool_calls\":[{\"id\":\"c1\",\"type\":\"function\",\"function\":{\"name\":\"get_stats\",\"arguments\":\"{\\\"target\\\":\\\"localhost\\\"}\"}}]}\r\n\r\n\
event: end_of_stream\r\ndata: {}\r\n\r\n";
        let p = parse_sse_response(body);
        assert_eq!(p.tool_calls.len(), 1);
        assert_eq!(p.tool_calls[0].name, "get_stats");
        assert_eq!(p.tool_calls[0].arguments["target"], "localhost");
    }

    #[test]
    fn empty_body_gives_defaults() {
        let p = parse_sse_response("");
        assert!(p.answer.is_empty());
        assert!(p.tool_calls.is_empty());
        assert!(p.backend_uuid.is_none());
    }
}
