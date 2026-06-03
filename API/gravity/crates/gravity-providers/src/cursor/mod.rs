//! Cursor provider (`ProviderName::Cursor`).
//!
//! Ports `gravity/cursor/*` — talks to `api2.cursor.sh` over Connect-RPC binary
//! protobuf (`application/connect+proto`). The `proto` codec builds the
//! `InferenceStreamRequest` and parses the streamed response frames. Auth is the
//! account access token; the `x-cursor-checksum` is supplied via the account.

pub(crate) mod proto;
mod auth;

use crate::context::ChatCtx;
use crate::provider::{Authenticator, Provider};
use crate::util::uuid_v4;
use async_trait::async_trait;
use gravity_core::{
    ChatResponse, Content, EmbeddingResponse, Error, HistoryMode, Message, Metadata,
    ProviderCapabilities, ProviderError, ProviderErrorKind, ProviderName, RemoteSessionMode,
    Result, Role, StructuredOutputMode, ToolCall, ToolSupportMode, Usage,
};
use gravity_http::{HttpRequest, Profile};

const STREAM_URL: &str = "https://api2.cursor.sh/aiserver.v1.InferenceService/Stream";
const EMBED_URL: &str = "https://api2.cursor.sh/aiserver.v1.AiService/BulkEmbed";

/// The Cursor adapter.
pub struct CursorProvider {
    capabilities: ProviderCapabilities,
}

impl Default for CursorProvider {
    fn default() -> Self {
        CursorProvider::new()
    }
}

impl CursorProvider {
    /// Construct the provider.
    pub fn new() -> Self {
        CursorProvider {
            capabilities: ProviderCapabilities {
                provider: ProviderName::Cursor,
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

    fn build_request(&self, ctx: &ChatCtx<'_>) -> HttpRequest {
        let proto_body = build_inference_request(
            &ctx.request.messages,
            ctx.request.system_prompt.as_deref(),
            &ctx.request.tools,
            &ctx.request.model.model,
            &uuid_v4(ctx.env),
        );
        let framed = proto::encode_frame(&proto_body);

        // x-cursor-checksum = time-prefix + machineId [/ macMachineId].
        let machine_id = ctx
            .account
            .auth
            .inline_secret("machine_id")
            .map(str::to_owned)
            .unwrap_or_else(|| uuid_v4(ctx.env));
        let checksum = match ctx.account.auth.inline_secret("checksum") {
            Some(c) => c.to_owned(),
            None => match ctx.account.auth.inline_secret("mac_machine_id") {
                Some(mac) => format!("{}{}/{}", checksum_prefix(ctx.env), machine_id, mac),
                None => format!("{}{}", checksum_prefix(ctx.env), machine_id),
            },
        };

        HttpRequest::post(STREAM_URL)
            .profile(Profile::from_name(ctx.impersonate_or("chrome")))
            .timeout_ms(ctx.timeout_ms())
            .proxy(ctx.proxy())
            .header("content-type", "application/connect+proto")
            .header("connect-protocol-version", "1")
            .header("authorization", format!("Bearer {}", ctx.secret))
            .header("x-cursor-checksum", checksum)
            .header("x-cursor-client-version", "3.1.17")
            .header("x-cursor-client-type", "ide")
            .header("x-session-id", uuid_v4(ctx.env))
            .header("user-agent", "Cursor/1.0.0")
            .body(bytes::Bytes::from(framed))
    }
}

#[async_trait]
impl Provider for CursorProvider {
    fn capabilities(&self) -> &ProviderCapabilities {
        &self.capabilities
    }

    fn authenticator(&self) -> Option<&dyn Authenticator> {
        static AUTH: auth::CursorAuthenticator = auth::CursorAuthenticator;
        Some(&AUTH)
    }

    fn list_models(&self) -> Vec<String> {
        ["claude-4.5-sonnet", "gpt-5", "auto"].iter().map(|s| s.to_string()).collect()
    }

    /// Embed texts via `AiService/BulkEmbed` (3072-dim float64, Connect-RPC proto).
    async fn embeddings(&self, ctx: &ChatCtx<'_>, texts: &[String]) -> Result<EmbeddingResponse> {
        if texts.is_empty() {
            return Ok(EmbeddingResponse {
                texts: Vec::new(), embeddings: Vec::new(),
                provider: ProviderName::Cursor.as_str().into(),
                dimensions: 0, metadata: gravity_core::Metadata::new(),
            });
        }
        let text_refs: Vec<&str> = texts.iter().map(String::as_str).collect();
        let payload = proto::build_bulk_embed_request(&text_refs);
        // BulkEmbed is a unary call (no Connect streaming frame needed).
        let req = HttpRequest::post(EMBED_URL)
            .profile(Profile::from_name(ctx.impersonate_or("go_tls")))
            .timeout_ms(ctx.timeout_ms())
            .proxy(ctx.proxy())
            .header("content-type", "application/proto")
            .header("authorization", format!("Bearer {}", ctx.secret))
            .header("x-cursor-client-version", "0.50.0")
            .header("x-cursor-client-type", "ide")
            .body(bytes::Bytes::from(payload));
        let resp = ctx.http.send(req).await?;
        if !resp.is_success() {
            return Err(Error::Provider(
                ProviderError::new(ProviderName::Cursor, ProviderErrorKind::Other,
                    "cursor_embed_failed",
                    format!("Cursor embeddings failed (status {})", resp.status))
            ));
        }
        let vecs = proto::parse_bulk_embed_response(&resp.body);
        let embeddings: Vec<Vec<f32>> = vecs.into_iter()
            .map(|v| v.into_iter().map(|f| f as f32).collect())
            .collect();
        let dim = embeddings.first().map(|v| v.len() as u32).unwrap_or(0);
        Ok(EmbeddingResponse {
            texts: texts.to_vec(),
            embeddings,
            provider: ProviderName::Cursor.as_str().into(),
            dimensions: dim,
            metadata: gravity_core::Metadata::new(),
        })
    }

    async fn chat(&self, ctx: &ChatCtx<'_>) -> Result<ChatResponse> {
        let req = self.build_request(ctx);
        let resp = ctx.http.send(req).await?;
        if !resp.is_success() {
            return Err(Error::Provider(
                ProviderError::new(
                    ProviderName::Cursor,
                    ProviderErrorKind::Other,
                    "cursor_failed",
                    format!("Cursor request failed (status {})", resp.status),
                )
                .with_status(resp.status),
            ));
        }
        let mut text = String::new();
        let mut tool_calls: Vec<ToolCall> = Vec::new();
        for (flag, payload) in proto::decode_frames(&resp.body) {
            if flag & 0x02 != 0 {
                continue; // end-of-stream / trailer frame
            }
            for ev in parse_inference_frame(&payload) {
                match ev {
                    InferenceEvent::Text(t) => text.push_str(&t),
                    InferenceEvent::ToolCall { id, name, args } => {
                        let arguments = serde_json::from_str::<serde_json::Value>(&args)
                            .ok()
                            .and_then(|v| v.as_object().cloned())
                            .unwrap_or_default();
                        tool_calls.push(ToolCall { id: id.into(), name: name.into(), arguments });
                    }
                    InferenceEvent::Error(msg) => {
                        return Err(Error::Provider(ProviderError::new(
                            ProviderName::Cursor,
                            ProviderErrorKind::Other,
                            "cursor_stream_error",
                            msg,
                        )));
                    }
                }
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
}

/// Time-based XOR-obfuscated 6-byte checksum prefix (base64), ported from
/// Cursor's `_compute_checksum_prefix` (R6y/D6y transform).
fn checksum_prefix(env: &gravity_core::Env) -> String {
    use base64::{engine::general_purpose::STANDARD, Engine as _};
    let t = env.clock.now_millis() / 1_000_000;
    let mut raw = [
        ((t >> 40) & 0xFF) as u8,
        ((t >> 32) & 0xFF) as u8,
        ((t >> 24) & 0xFF) as u8,
        ((t >> 16) & 0xFF) as u8,
        ((t >> 8) & 0xFF) as u8,
        (t & 0xFF) as u8,
    ];
    let mut e: u8 = 165;
    for (i, byte) in raw.iter_mut().enumerate() {
        *byte = (*byte ^ e).wrapping_add(i as u8);
        e = *byte;
    }
    STANDARD.encode(raw)
}

/// Build the `InferenceStreamRequest` protobuf body.
fn build_inference_request(
    messages: &[Message],
    system_prompt: Option<&str>,
    tools: &[gravity_core::ToolDefinition],
    model: &str,
    conversation_id: &str,
) -> Vec<u8> {
    let mut req: Vec<u8> = Vec::new();

    // System prompt becomes a leading message with source role 4 (SYSTEM).
    if let Some(sp) = system_prompt.filter(|s| !s.trim().is_empty()) {
        let mut core = proto::field_varint(1, 4);
        core.extend(proto::field_string(2, sp));
        req.extend(proto::field_message(1, &core));
    }

    for m in messages {
        let role = match m.role {
            Role::System => 4u64,
            Role::Assistant => 2u64,
            Role::Tool => 3u64,
            _ => 1u64,
        };
        let mut core = proto::field_varint(1, role);
        if role == 3 {
            // tool result
            if let Some(tr) = &m.tool_result {
                let mut part = proto::field_string(1, &tr.tool_call_id);
                part.extend(proto::field_string(2, &tr.name));
                part.extend(proto::field_message(3, &proto::encode_value(&tr.content)));
                let parts = proto::field_message(1, &part);
                core.extend(proto::field_message(6, &parts));
            }
        } else {
            let text = text_of(m);
            if !text.is_empty() {
                core.extend(proto::field_string(2, &text));
            }
            if role == 2 {
                for tc in &m.tool_calls {
                    let mut call = proto::field_string(1, &tc.id);
                    call.extend(proto::field_string(2, &tc.name));
                    call.extend(proto::field_message(3, &proto::encode_struct(&tc.arguments)));
                    core.extend(proto::field_message(4, &call));
                }
            }
        }
        req.extend(proto::field_message(1, &core));
    }

    // field 2: advertised tools (InferenceAgentTool { name, description, params }).
    for t in tools {
        let mut schema = serde_json::Map::new();
        schema.insert("type".into(), serde_json::json!("object"));
        for (k, v) in &t.input_schema {
            schema.insert(k.clone(), v.clone());
        }
        let mut tproto = proto::field_string(1, &t.name);
        tproto_desc(&mut tproto, &t.description);
        tproto.extend(proto::field_message(3, &proto::encode_struct(&schema)));
        req.extend(proto::field_message(2, &tproto));
    }

    // field 7: requested model { field 1: model_id }
    req.extend(proto::field_message(7, &proto::field_string(1, model)));
    // field 8: conversation id
    req.extend(proto::field_string(8, conversation_id));
    req
}

/// Append a tool description field (skipped when empty, matching Python).
fn tproto_desc(buf: &mut Vec<u8>, description: &str) {
    if !description.is_empty() {
        buf.extend(proto::field_string(2, description));
    }
}

/// An event parsed from one `InferenceStreamResponse` frame.
enum InferenceEvent {
    Text(String),
    ToolCall { id: String, name: String, args: String },
    Error(String),
}

/// Parse one response frame (`parse_inference_frame`): field 1 = text part
/// `{1: text}`, field 2 = tool-call part `{1: id, 2: name, 3: args}`, field 8 =
/// error `{2: message}`.
fn parse_inference_frame(payload: &[u8]) -> Vec<InferenceEvent> {
    let mut events = Vec::new();
    let fields = proto::decode_fields(payload);
    if let Some(text_parts) = fields.get(&1) {
        for raw in text_parts {
            if let Some(bytes) = raw.bytes() {
                let inner = proto::decode_fields(bytes);
                if let Some(t) = inner.get(&1).and_then(|v| v.first()) {
                    let s = t.as_string();
                    if !s.is_empty() {
                        events.push(InferenceEvent::Text(s));
                    }
                }
            }
        }
    }
    if let Some(tool_parts) = fields.get(&2) {
        for raw in tool_parts {
            if let Some(bytes) = raw.bytes() {
                let tc = proto::decode_fields(bytes);
                let id = tc.get(&1).and_then(|v| v.first()).map(|f| f.as_string()).unwrap_or_default();
                let name = tc.get(&2).and_then(|v| v.first()).map(|f| f.as_string()).unwrap_or_default();
                let args = tc.get(&3).and_then(|v| v.first()).map(|f| f.as_string()).unwrap_or_default();
                if !name.is_empty() {
                    events.push(InferenceEvent::ToolCall { id, name, args });
                }
            }
        }
    }
    if let Some(errs) = fields.get(&8) {
        for raw in errs {
            // error message is nested at inner field 2.
            let msg = raw
                .bytes()
                .and_then(|b| proto::decode_fields(b).get(&2).and_then(|v| v.first()).map(|f| f.as_string()))
                .unwrap_or_else(|| raw.as_string());
            events.push(InferenceEvent::Error(msg));
        }
    }
    events
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
    fn request_roundtrips_through_proto_decode() {
        let msgs = vec![Message::user("Hello cursor")];
        let body = build_inference_request(&msgs, Some("Be brief"), &[], "gpt-5", "conv-1");
        let fields = proto::decode_fields(&body);
        // two message envelopes (system + user) under field 1
        assert_eq!(fields[&1].len(), 2);
        // model under field 7, conversation id under field 8
        assert!(fields.contains_key(&7));
        assert_eq!(fields[&8][0].as_string(), "conv-1");
    }

    #[test]
    fn parses_text_frame() {
        // field 1 { field 1: "hi there" }
        let inner = proto::field_string(1, "hi there");
        let frame = proto::field_message(1, &inner);
        let events = parse_inference_frame(&frame);
        assert_eq!(events.len(), 1);
        match &events[0] {
            InferenceEvent::Text(t) => assert_eq!(t, "hi there"),
            _ => panic!("expected text"),
        }
    }

    #[test]
    fn capabilities() {
        assert_eq!(CursorProvider::new().capabilities().provider, ProviderName::Cursor);
    }
}
