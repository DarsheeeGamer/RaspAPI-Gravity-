//! The Gemini web provider.
//!
//! Ports the chat path of `gravity/gemini/client.py`. Credentials (the `SNlM0e`
//! access token, `bl` build label, `f.sid` session id, and `__Secure-1PSID*`
//! cookies) come from the account; the `f.req` payload ([`super::build`]) is
//! posted form-encoded and the response decoded with the frame [`super::parser`].

use super::{build, parser};
use crate::context::ChatCtx;
use crate::provider::{EventStream, Provider};
use async_trait::async_trait;
use gravity_core::{
    ChatResponse, Content, Error, HistoryMode, Message, Metadata, ProviderCapabilities,
    ProviderError, ProviderErrorKind, ProviderName, RemoteSessionMode, Result, StreamEvent,
    StructuredOutputMode, ToolSupportMode, Usage,
};
use gravity_http::{HttpRequest, Profile};

/// The Gemini web adapter.
pub struct GeminiProvider {
    capabilities: ProviderCapabilities,
}

impl Default for GeminiProvider {
    fn default() -> Self {
        GeminiProvider::new()
    }
}

impl GeminiProvider {
    /// Construct the provider.
    pub fn new() -> Self {
        GeminiProvider {
            capabilities: ProviderCapabilities {
                provider: ProviderName::Gemini,
                tool_support: ToolSupportMode::JsonEmulated,
                structured_output: StructuredOutputMode::JsonInstruction,
                remote_session_mode: RemoteSessionMode::Required,
                supports_system_prompt: true,
                supports_stateful_history: true,
                requires_remote_checkpoint: false,
                default_history_mode: HistoryMode::Auto,
                supports_image_generation: false,
            },
        }
    }

    fn cookie_header(ctx: &ChatCtx<'_>) -> String {
        let mut cookies = Vec::new();
        if let Some(map) = ctx.account.auth.extra.as_map() {
            for (k, v) in map {
                // Include all Google auth cookies: __Secure-*, SID, SAPISID, NID, SIDCC
                // Also __Secure-STRP (anonymous session cookie from HAR analysis).
                let is_cookie = k.starts_with("__Secure-")
                    || matches!(k.as_str(), "SID" | "SAPISID" | "NID" | "SIDCC" | "SNLM0E");
                if is_cookie {
                    if let Some(s) = v.as_str() {
                        cookies.push(format!("{k}={s}"));
                    }
                }
            }
        }
        // If no cookies at all but a non-empty secret, treat it as __Secure-1PSID.
        if cookies.is_empty() && !ctx.secret.is_empty() {
            cookies.push(format!("__Secure-1PSID={}", ctx.secret));
        }
        cookies.join("; ")
    }

    /// Bootstrap the web session: GET `/app` to harvest the `SNlM0e` access
    /// token (`at`), `cfb2h` build label (`bl`), and `FdrFJe` session id
    /// (`f.sid`) embedded in the page. Mirrors `client.py::bootstrap`.
    ///
    /// Tokens already present in `account.auth.extra` short-circuit the fetch
    /// (the caller pre-resolved a warm session). A logged-in session is
    /// required: anonymous access additionally needs a BotGuard token the web
    /// client mints in JS, which is out of scope here.
    async fn bootstrap(&self, ctx: &ChatCtx<'_>) -> Result<BootstrapTokens> {
        let access_token = ctx
            .account
            .auth
            .inline_secret("access_token")
            .or(ctx.account.auth.inline_secret("SNlM0e"))
            .map(str::to_owned);
        let build_label = ctx.account.auth.inline_secret("build_label").map(str::to_owned);
        let session_id = ctx.account.auth.inline_secret("session_id").map(str::to_owned);
        // If the session metadata is already warm, skip the network round-trip.
        if build_label.is_some() && session_id.is_some() {
            return Ok(BootstrapTokens {
                access_token: access_token.unwrap_or_default(),
                build_label,
                session_id,
            });
        }

        let cookies = Self::cookie_header(ctx);
        let mut get = HttpRequest::get(build::APP_URL)
            .profile(Profile::from_name(ctx.impersonate_or("chrome")))
            .timeout_ms(ctx.timeout_ms())
            .proxy(ctx.proxy())
            .cookie_store(true)
            .header("origin", "https://gemini.google.com")
            .header("referer", "https://gemini.google.com/")
            .header("user-agent", USER_AGENT);
        if !cookies.is_empty() {
            get = get.header("cookie", cookies);
        }
        let resp = ctx.http.send(get).await?;
        if !resp.is_success() {
            return Err(Error::Provider(
                ProviderError::new(
                    ProviderName::Gemini,
                    ProviderErrorKind::Authentication,
                    "gemini_bootstrap_failed",
                    "Gemini bootstrap GET /app failed (check __Secure-1PSID cookie)",
                )
                .with_status(resp.status),
            ));
        }
        let body = resp.text();
        Ok(BootstrapTokens {
            access_token: extract_field(&body, "SNlM0e")
                .map(str::to_owned)
                .or(access_token)
                .unwrap_or_default(),
            build_label: extract_field(&body, "cfb2h").map(str::to_owned).or(build_label),
            session_id: extract_field(&body, "FdrFJe").map(str::to_owned).or(session_id),
        })
    }

    fn build_request(&self, ctx: &ChatCtx<'_>, tokens: &BootstrapTokens) -> HttpRequest {
        let prompt = build_prompt(&ctx.request.messages, ctx.request.system_prompt.as_deref());
        // Stateful threading: if caller supplies c_id/r_id/rcid from a prior
        // response, embed them in inner[2] so the server threads the conversation.
        let conv_metadata = build_conv_metadata(ctx);
        let (payload, uuid) = build::build_freq(
            ctx.env,
            &prompt,
            None, // system_prompt already merged into prompt by build_prompt
            &conv_metadata,
            true,
            None,
        );

        let access_token = if tokens.access_token.is_empty() {
            ctx.secret
        } else {
            &tokens.access_token
        };

        // Query params.
        let mut url = format!("{}?_reqid={}&rt=c", build::GENERATE_URL, build::reqid(ctx.env));
        if let Some(bl) = &tokens.build_label {
            url.push_str(&format!("&bl={}", urlencode(bl)));
        }
        url.push_str("&hl=en-US");
        if let Some(sid) = &tokens.session_id {
            url.push_str(&format!("&f.sid={}", urlencode(sid)));
        }

        // Form body: f.req + at (omit at if empty — anonymous mode).
        let body = if access_token.is_empty() {
            format!("f.req={}", urlencode(&payload))
        } else {
            format!("f.req={}&at={}", urlencode(&payload), urlencode(access_token))
        };

        let model = &ctx.request.model.model;
        let mut req = HttpRequest::post(url)
            .profile(Profile::from_name(ctx.impersonate_or("chrome")))
            .timeout_ms(ctx.timeout_ms())
            .proxy(ctx.proxy())
            .cookie_store(true)
            .header("content-type", "application/x-www-form-urlencoded;charset=UTF-8")
            .header("origin", "https://gemini.google.com")
            .header("referer", "https://gemini.google.com/")
            .header("x-same-domain", "1")
            .header("user-agent", USER_AGENT)
            // Per-request UUID header (also passed as last field of 525001261 model header).
            .header("x-goog-ext-525005358-jspb", format!(r#"["{}",1]"#, uuid));
        // Cookie header — omit if empty (anonymous mode).
        let cookies = Self::cookie_header(ctx);
        if !cookies.is_empty() {
            req = req.header("cookie", cookies);
        }
        let mut req = req
            .body(bytes::Bytes::from(body));
        // Model-specific headers with per-request UUID for correct model routing.
        for (k, v) in build::model_headers_with_uuid(model, &uuid) {
            req = req.header(k, v);
        }
        req
    }
}

/// Session tokens resolved by [`GeminiProvider::bootstrap`].
struct BootstrapTokens {
    access_token: String,
    build_label: Option<String>,
    session_id: Option<String>,
}

const USER_AGENT: &str = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 \
(KHTML, like Gecko) Chrome/145.0.0.0 Safari/537.36";

/// Extract a `"<key>":"<value>"` string field from bootstrap HTML.
///
/// The Gemini app page embeds the session tokens as JSON-ish key/value pairs;
/// this scans for `"<key>":` and reads the following double-quoted string.
fn extract_field<'a>(haystack: &'a str, key: &str) -> Option<&'a str> {
    let needle = format!("\"{key}\":");
    let start = haystack.find(&needle)? + needle.len();
    let rest = haystack[start..].trim_start();
    let rest = rest.strip_prefix('"')?;
    let end = rest.find('"')?;
    let val = &rest[..end];
    (!val.is_empty()).then_some(val)
}

#[async_trait]
impl Provider for GeminiProvider {
    fn capabilities(&self) -> &ProviderCapabilities {
        &self.capabilities
    }

    fn list_models(&self) -> Vec<String> {
        build::models()
    }

    /// Discover the account's available models via the `otAQ7b` GetUserStatus
    /// batchexecute RPC (the same call the web UI makes to populate its model
    /// picker). Each response entry is `[model_id_hex, display_name, description]`;
    /// known hex ids are reverse-mapped to request-usable names. Falls back to
    /// the static list on any failure (including anonymous sessions, which can't
    /// enumerate beyond the base flash model).
    async fn discover_models(&self, ctx: &ChatCtx<'_>) -> Result<Vec<gravity_core::ModelInfo>> {
        use gravity_core::ModelInfo;
        use parser::{nested, NestIndex};
        let fallback = || build::models().into_iter().map(ModelInfo::bare).collect::<Vec<_>>();

        let tokens = match self.bootstrap(ctx).await {
            Ok(t) => t,
            Err(_) => return Ok(fallback()),
        };
        let access_token = if tokens.access_token.is_empty() {
            ctx.secret
        } else {
            &tokens.access_token
        };

        // f.req for a single-RPC batchexecute: [[["otAQ7b","[]",null,"generic"]]]
        let f_req = format!(r#"[[["{}","[]",null,"generic"]]]"#, build::RPC_GET_USER_STATUS);
        let mut url = format!(
            "{}?rpcids={}&_reqid={}&rt=c",
            build::BATCH_EXEC_URL,
            build::RPC_GET_USER_STATUS,
            build::reqid(ctx.env),
        );
        if let Some(sid) = &tokens.session_id {
            url.push_str(&format!("&f.sid={}", urlencode(sid)));
        }
        if let Some(bl) = &tokens.build_label {
            url.push_str(&format!("&bl={}", urlencode(bl)));
        }
        let body = if access_token.is_empty() {
            format!("f.req={}", urlencode(&f_req))
        } else {
            format!("f.req={}&at={}", urlencode(&f_req), urlencode(access_token))
        };
        let mut req = HttpRequest::post(url)
            .profile(Profile::from_name(ctx.impersonate_or("chrome")))
            .timeout_ms(ctx.timeout_ms())
            .proxy(ctx.proxy())
            .cookie_store(true)
            .header("content-type", "application/x-www-form-urlencoded;charset=UTF-8")
            .header("origin", "https://gemini.google.com")
            .header("referer", "https://gemini.google.com/")
            .header("x-same-domain", "1")
            .header("user-agent", USER_AGENT);
        let cookies = Self::cookie_header(ctx);
        if !cookies.is_empty() {
            req = req.header("cookie", cookies);
        }
        let req = req.body(bytes::Bytes::from(body));

        let resp = match ctx.http.send(req).await {
            Ok(r) if r.is_success() => r,
            _ => return Ok(fallback()),
        };
        let frames = parser::extract_json_from_response(&resp.text());

        // Each wrb.fr frame's [2] is a JSON string; parse it and read [15] =
        // the models list, where each entry is [hex_id, display_name, desc].
        let mut out: Vec<ModelInfo> = Vec::new();
        for frame in &frames {
            let Some(inner_str) = nested(frame, &[NestIndex::Idx(2)]).and_then(|v| v.as_str()) else {
                continue;
            };
            let Ok(part) = serde_json::from_str::<serde_json::Value>(inner_str) else {
                continue;
            };
            let Some(models) = nested(&part, &[NestIndex::Idx(15)]).and_then(|v| v.as_array()) else {
                continue;
            };
            for m in models {
                let Some(hex) = nested(m, &[NestIndex::Idx(0)]).and_then(|v| v.as_str()) else {
                    continue;
                };
                let display = nested(m, &[NestIndex::Idx(1)]).and_then(|v| v.as_str());
                let desc = nested(m, &[NestIndex::Idx(2)]).and_then(|v| v.as_str());
                // Prefer a request-usable name; fall back to the hex id.
                let id = build::model_name_for_id(hex).map(str::to_owned).unwrap_or_else(|| hex.to_owned());
                out.push(ModelInfo {
                    id,
                    display_name: display.map(str::to_owned),
                    description: desc.map(str::to_owned),
                    tags: Vec::new(),
                });
            }
        }
        if out.is_empty() {
            Ok(fallback())
        } else {
            Ok(out)
        }
    }

    async fn chat(&self, ctx: &ChatCtx<'_>) -> Result<ChatResponse> {
        let tokens = self.bootstrap(ctx).await?;
        let req = self.build_request(ctx, &tokens);
        let resp = ctx.http.send(req).await?;
        if !resp.is_success() {
            return Err(Error::Provider(
                ProviderError::new(
                    ProviderName::Gemini,
                    ProviderErrorKind::Other,
                    "gemini_request_failed",
                    format!("Gemini request failed: {}", resp.text().chars().take(300).collect::<String>()),
                )
                .with_status(resp.status),
            ));
        }
        let body_text = resp.text();
        let frames = parser::extract_json_from_response(&body_text);
        let parsed = parser::parse_parts(&frames);
        if parsed.text.is_empty() && parsed.rcid.is_none() {
            let preview: String = body_text.chars().take(500).collect();
            return Err(Error::Provider(ProviderError::new(
                ProviderName::Gemini,
                ProviderErrorKind::Other,
                "gemini_empty_response",
                format!("Gemini returned no parseable content. frames={} body_preview={:?}",
                    frames.len(), preview),
            )));
        }
        // Embed conversation state (c_id, r_id, rcid, state_token) so callers
        // can thread subsequent requests natively via inner[2].
        // state_token ("26" field in part_json[2]) must go in inner[2][9].
        let mut meta = Metadata::new();
        if let Some(c_id) = parsed.metadata.first().and_then(Option::as_deref) {
            meta.insert("gemini_c_id", c_id);
        }
        if let Some(r_id) = parsed.metadata.get(1).and_then(Option::as_deref) {
            meta.insert("gemini_r_id", r_id);
        }
        if let Some(rcid) = &parsed.rcid {
            meta.insert("gemini_rcid", rcid.as_str());
        }
        if let Some(st) = &parsed.state_token {
            meta.insert("gemini_state_token", st.as_str());
        }
        Ok(ChatResponse {
            model: ctx.request.model.full_name().into(),
            message: Message::assistant(parsed.text),
            usage: Usage::default(),
            tool_calls_executed: Vec::new(),
            provider_response_id: parsed.rcid.map(Into::into),
            finish_reason: Some("stop".into()),
            metadata: meta,
        })
    }

    async fn chat_stream(&self, ctx: &ChatCtx<'_>) -> Result<EventStream> {
        // The batchexecute response is not incremental SSE; buffer then replay.
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

/// Build the prompt matching Python's `_render_history` + `_build_prompt_and_files`.
///
/// Single-message: user text only (or system prefix + user text).
/// Multi-turn (stateless): render all messages as labelled turns separated by `\n\n`,
/// matching `gravity/gemini/adapter.py::_render_history` exactly.
fn build_prompt(messages: &[Message], system_prompt: Option<&str>) -> String {
    if messages.len() <= 1 {
        let last_text = messages.last().map(text_of).unwrap_or_default();
        if let Some(sp) = system_prompt.filter(|s| !s.trim().is_empty()) {
            return format!("System instructions:\n{}\n\nUser request:\n{}", sp.trim(), last_text.trim());
        }
        return last_text;
    }
    // Multi-turn: full history as text (Python _render_history format).
    // Each turn: "<Role>: <text>", separated by "\n\n".
    let mut lines: Vec<String> = Vec::new();
    if let Some(sp) = system_prompt.filter(|s| !s.trim().is_empty()) {
        lines.push(format!("System: {}", sp.trim()));
    }
    for m in messages {
        let role = match m.role {
            gravity_core::Role::User => "User",
            gravity_core::Role::Assistant => "Assistant",
            gravity_core::Role::Tool => "Tool",
            gravity_core::Role::System => "System",
        };
        let text = text_of(m);
        if !text.trim().is_empty() {
            lines.push(format!("{role}: {text}"));
        }
    }
    lines.join("\n\n")
}

/// Extract conversation threading IDs (c_id, r_id, rcid) from the checkpoint
/// metadata so we can embed them in `inner[2]` for stateful continuation.
fn build_conv_metadata(ctx: &ChatCtx<'_>) -> Vec<serde_json::Value> {
    let chk = match ctx.checkpoint {
        Some(c) => c,
        None => return vec![],
    };
    let c_id = chk.remote_metadata.get("gemini_c_id").and_then(|v| v.as_str()).unwrap_or("");
    let r_id = chk.remote_metadata.get("gemini_r_id").and_then(|v| v.as_str()).unwrap_or("");
    let rcid = chk.remote_metadata.get("gemini_rcid").and_then(|v| v.as_str()).unwrap_or("");
    // State token from part_json[2]["26"] — required in inner[2][9] for threading.
    let state_token = chk.remote_metadata.get("gemini_state_token").and_then(|v| v.as_str()).unwrap_or("");
    if c_id.is_empty() && r_id.is_empty() {
        return vec![]; // no prior state → fresh conversation
    }
    // inner[2] = [c_id, r_id, rcid, null×6, state_token]
    // (HAR-verified: position [9] is the opaque session continuation token)
    use serde_json::{Value, json};
    vec![
        json!(c_id), json!(r_id), json!(rcid),
        Value::Null, Value::Null, Value::Null, Value::Null, Value::Null, Value::Null,
        json!(state_token),
    ]
}

/// Minimal form-component urlencoder.
fn urlencode(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for b in s.bytes() {
        match b {
            b'A'..=b'Z' | b'a'..=b'z' | b'0'..=b'9' | b'-' | b'_' | b'.' | b'~' => out.push(b as char),
            _ => out.push_str(&format!("%{b:02X}")),
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn capabilities() {
        let p = GeminiProvider::new();
        assert_eq!(p.capabilities().provider, ProviderName::Gemini);
        assert!(!p.list_models().is_empty());
    }

    #[test]
    fn urlencode_escapes_reserved() {
        assert_eq!(urlencode("a b&c"), "a%20b%26c");
    }

    #[test]
    fn extract_field_reads_quoted_value() {
        let html = r#"window.WIZ={"SNlM0e":"AB6tok==","cfb2h":"boq_label","FdrFJe":"-12345"};"#;
        assert_eq!(extract_field(html, "SNlM0e"), Some("AB6tok=="));
        assert_eq!(extract_field(html, "cfb2h"), Some("boq_label"));
        assert_eq!(extract_field(html, "FdrFJe"), Some("-12345"));
        assert_eq!(extract_field(html, "missing"), None);
    }

    #[test]
    fn extract_field_handles_whitespace_and_empty() {
        assert_eq!(extract_field(r#""SNlM0e":  "tok""#, "SNlM0e"), Some("tok"));
        assert_eq!(extract_field(r#""SNlM0e":"""#, "SNlM0e"), None);
    }
}
