//! Windsurf provider (`ProviderName::Windsurf`).
//!
//! Ports `gravity/windsurf/__init__.py` — Codeium Windsurf remote inference over
//! Connect binary protobuf (`GetChatMessage` at `server.codeium.com`). Reuses
//! the shared `cursor::proto` codec and Connect framing; uses the `go_tls` TLS
//! profile. Auth = Bearer api key (`sk-ws-01-…`) + a JWT in the proto metadata
//! (`auth.extra.jwt`).

use crate::context::ChatCtx;
use crate::cursor::proto;
use crate::provider::Provider;
use crate::util::uuid_v4;
use async_trait::async_trait;
use gravity_core::{
    ChatResponse, Content, EmbeddingResponse, Error, HistoryMode, Message, Metadata,
    ProviderCapabilities, ProviderError, ProviderErrorKind, ProviderName, RemoteSessionMode,
    Result, Role, StructuredOutputMode, ToolSupportMode, Usage,
};
use gravity_http::{HttpRequest, Profile};
use sha2::{Digest, Sha256};

const URL: &str = "https://server.codeium.com/exa.api_server_pb.ApiServerService/GetChatMessage";
const EMBED_URL: &str = "https://server.self-serve.windsurf.com/exa.api_server_pb.ApiServerService/GetEmbeddings";
const EXT_VER: &str = "1.48.2";
const IDE_VER: &str = "2.0.61";
const SRC_USER: u64 = 1;
const SRC_ASSISTANT: u64 = 2;
const SRC_TOOL: u64 = 4;

/// The Windsurf adapter.
pub struct WindsurfProvider {
    capabilities: ProviderCapabilities,
}

impl Default for WindsurfProvider {
    fn default() -> Self {
        WindsurfProvider::new()
    }
}

impl WindsurfProvider {
    /// Construct the provider.
    pub fn new() -> Self {
        WindsurfProvider {
            capabilities: ProviderCapabilities {
                provider: ProviderName::Windsurf,
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

    fn build_request(&self, ctx: &ChatCtx<'_>) -> HttpRequest {
        let api_key = ctx.secret;
        let jwt = ctx.account.auth.inline_secret("jwt").unwrap_or("");
        let envelope = build_envelope(
            api_key,
            jwt,
            &ctx.request.messages,
            ctx.request.system_prompt.as_deref(),
            &ctx.request.tools,
            &ctx.request.model.model,
            &uuid_v4(ctx.env),
        );
        HttpRequest::post(URL)
            // Windsurf's language server uses the Go stdlib TLS fingerprint.
            .profile(Profile::from_name(ctx.impersonate_or("go_tls")))
            .timeout_ms(ctx.timeout_ms())
            .proxy(ctx.proxy())
            .header("content-type", "application/connect+proto")
            .header("connect-protocol-version", "1")
            .header("authorization", format!("Bearer {api_key}"))
            .header("user-agent", "connect-go/1.18.1 (go1.26.1)")
            .body(bytes::Bytes::from(envelope))
    }
}

#[async_trait]
impl Provider for WindsurfProvider {
    fn capabilities(&self) -> &ProviderCapabilities {
        &self.capabilities
    }

    fn list_models(&self) -> Vec<String> {
        ["swe-1-6-fast", "swe-1-6", "kimi-k2-6", "devstral"].iter().map(|s| s.to_string()).collect()
    }

    /// Embed texts via `ApiServerService/GetEmbeddings` — 1024-dim float32, standard gRPC framing.
    ///
    /// Wire format (from Python reverse-engineering):
    /// ```text
    /// GetEmbeddingsRequest { field 1: EmbeddingsRequest { field 1: repeated string, field 2: priority=1 } }
    /// GetEmbeddingsResponse { field 1: repeated EmbeddingVector { field 1: packed float32 bytes } }
    /// ```
    async fn embeddings(&self, ctx: &ChatCtx<'_>, texts: &[String]) -> Result<EmbeddingResponse> {
        if texts.is_empty() {
            return Ok(EmbeddingResponse {
                texts: Vec::new(), embeddings: Vec::new(),
                provider: ProviderName::Windsurf.as_str().into(),
                dimensions: 0, metadata: Metadata::new(),
            });
        }
        // Build EmbeddingsRequest { field 1: repeated string, field 2: priority=1 }
        let mut emb_req = Vec::new();
        for t in texts {
            emb_req.extend(proto::field_string(1, t));
        }
        emb_req.extend(proto::field_varint(2, 1)); // HIGH priority
        // Wrap in GetEmbeddingsRequest { field 1: EmbeddingsRequest }
        let body = proto::field_message(1, &emb_req);
        // Standard gRPC frame: [0x00][len:4be][payload]
        let mut grpc_frame = Vec::with_capacity(5 + body.len());
        grpc_frame.push(0x00);
        grpc_frame.extend_from_slice(&(body.len() as u32).to_be_bytes());
        grpc_frame.extend_from_slice(&body);

        let profile = Profile::from_name(ctx.impersonate_or("go_tls"));
        let req = HttpRequest::post(EMBED_URL)
            .profile(profile)
            .timeout_ms(ctx.timeout_ms())
            .proxy(ctx.proxy())
            .header("content-type", "application/grpc")
            .header("te", "trailers")
            .header("authorization", format!("Bearer {}", ctx.secret))
            .header("accept-encoding", "identity")
            .header("user-agent", "connect-go/1.18.1 (go1.26.1)")
            .body(bytes::Bytes::from(grpc_frame));

        let resp = ctx.http.send(req).await?;
        if !resp.is_success() {
            return Err(Error::Provider(
                ProviderError::new(ProviderName::Windsurf, ProviderErrorKind::Other,
                    "windsurf_embed_failed",
                    format!("Windsurf embeddings failed (status {})", resp.status))
            ));
        }

        // Parse gRPC response frames → GetEmbeddingsResponse.
        let mut embeddings: Vec<Vec<f32>> = Vec::new();
        let buf = &resp.body;
        let mut pos = 0usize;
        while pos + 5 <= buf.len() {
            let flag = buf[pos];
            let length = u32::from_be_bytes(buf[pos+1..pos+5].try_into().unwrap()) as usize;
            pos += 5;
            if flag == 0x80 || length == 0 { pos += length; continue; } // trailer or empty
            let payload = &buf[pos..pos+length]; pos += length;
            let resp_fields = proto::decode_fields(payload);
            for emb_raw in resp_fields.get(&1).into_iter().flatten() {
                let Some(emb_bytes) = emb_raw.bytes() else { embeddings.push(vec![]); continue };
                let ev_fields = proto::decode_fields(emb_bytes);
                let raw_bytes: Vec<u8> = ev_fields.get(&1).into_iter().flatten()
                    .filter_map(|f| f.bytes().map(|b| b.to_vec()))
                    .flatten().collect();
                if raw_bytes.len() >= 4 {
                    let n = raw_bytes.len() / 4;
                    let vec: Vec<f32> = (0..n)
                        .map(|i| f32::from_le_bytes(raw_bytes[i*4..(i+1)*4].try_into().unwrap()))
                        .collect();
                    embeddings.push(vec);
                } else {
                    embeddings.push(vec![]);
                }
            }
        }
        let dim = embeddings.first().map(|v| v.len() as u32).unwrap_or(0);
        Ok(EmbeddingResponse {
            texts: texts.to_vec(),
            embeddings,
            provider: ProviderName::Windsurf.as_str().into(),
            dimensions: dim,
            metadata: Metadata::new(),
        })
    }

    async fn chat(&self, ctx: &ChatCtx<'_>) -> Result<ChatResponse> {
        let req = self.build_request(ctx);
        let resp = ctx.http.send(req).await?;
        if !resp.is_success() {
            return Err(Error::Provider(
                ProviderError::new(
                    ProviderName::Windsurf,
                    ProviderErrorKind::Other,
                    "windsurf_failed",
                    format!("Windsurf request failed (status {})", resp.status),
                )
                .with_status(resp.status),
            ));
        }
        // field 3 = response text token; field 6 = tool-call proto.
        let mut text = String::new();
        let mut tool_calls: Vec<gravity_core::ToolCall> = Vec::new();
        for (flag, payload) in proto::decode_frames(&resp.body) {
            if flag & 0x02 != 0 {
                continue;
            }
            let fields = proto::decode_fields(&payload);
            if let Some(parts) = fields.get(&3) {
                for p in parts {
                    text.push_str(&p.as_string());
                }
            }
            if let Some(calls) = fields.get(&6) {
                for raw in calls {
                    if let Some(b) = raw.bytes() {
                        let tc = proto::decode_fields(b);
                        let id = tc.get(&1).and_then(|v| v.first()).map(|f| f.as_string()).unwrap_or_default();
                        let name = tc.get(&2).and_then(|v| v.first()).map(|f| f.as_string()).unwrap_or_default();
                        let args = tc.get(&3).and_then(|v| v.first()).map(|f| f.as_string()).unwrap_or_default();
                        if !name.is_empty() {
                            let arguments = serde_json::from_str::<serde_json::Value>(&args)
                                .ok()
                                .and_then(|v| v.as_object().cloned())
                                .unwrap_or_default();
                            tool_calls.push(gravity_core::ToolCall { id: id.into(), name: name.into(), arguments });
                        }
                    }
                }
            }
        }
        let message = if tool_calls.is_empty() {
            Message::assistant(text)
        } else {
            Message::assistant_with_tools(Some(gravity_core::Content::Text(text)), tool_calls)
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

fn device_fingerprint(api_key: &str) -> String {
    let mut h = Sha256::new();
    h.update(api_key.as_bytes());
    let digest = h.finalize();
    let mut s = String::with_capacity(64);
    for b in digest {
        use std::fmt::Write;
        let _ = write!(s, "{b:02x}");
    }
    s
}

fn meta_bytes(api_key: &str, jwt: &str) -> Vec<u8> {
    let fp = device_fingerprint(api_key);
    let mut out = proto::field_string(1, "windsurf");
    out.extend(proto::field_string(2, EXT_VER));
    out.extend(proto::field_string(3, api_key));
    out.extend(proto::field_string(4, "en"));
    out.extend(proto::field_string(7, IDE_VER));
    out.extend(proto::field_string(12, "Windsurf"));
    out.extend(proto::field_string(21, jwt));
    out.extend(proto::field_string(24, &fp));
    out
}

/// Build the `GetChatMessage` Connect envelope (frame-wrapped).
#[allow(clippy::too_many_arguments)]
fn build_envelope(
    api_key: &str,
    jwt: &str,
    messages: &[Message],
    system_prompt: Option<&str>,
    tools: &[gravity_core::ToolDefinition],
    model: &str,
    cascade_id: &str,
) -> Vec<u8> {
    let mut req = proto::field_message(1, &meta_bytes(api_key, jwt));

    if let Some(sp) = system_prompt.filter(|s| !s.trim().is_empty()) {
        req.extend(proto::field_string(2, sp));
    }

    for m in messages {
        if m.role == Role::System {
            req.extend(proto::field_string(2, &text_of(m)));
            continue;
        }
        let source = match m.role {
            Role::Assistant => SRC_ASSISTANT,
            Role::Tool => SRC_TOOL,
            _ => SRC_USER,
        };
        let mut chat_msg = proto::field_string(1, cascade_id);
        chat_msg.extend(proto::field_varint(2, source));
        chat_msg.extend(proto::field_string(3, &text_of(m)));
        if let Some(tr) = &m.tool_result {
            chat_msg.extend(proto::field_string(7, &tr.tool_call_id));
        }
        for tc in &m.tool_calls {
            let args = serde_json::Value::Object(tc.arguments.clone()).to_string();
            let mut tcm = proto::field_string(1, &tc.id);
            tcm.extend(proto::field_string(2, &tc.name));
            tcm.extend(proto::field_string(3, &args));
            chat_msg.extend(proto::field_message(6, &tcm));
        }
        req.extend(proto::field_message(3, &chat_msg));
    }

    // field 10: ChatToolDefinition { 1: name, 2: description, 3: json(schema) }.
    for t in tools {
        let mut schema = serde_json::Map::new();
        schema.insert(
            "$schema".into(),
            serde_json::json!("https://json-schema.org/draft/2020-12/schema"),
        );
        for (k, v) in &t.input_schema {
            schema.insert(k.clone(), v.clone());
        }
        let schema_json = serde_json::Value::Object(schema).to_string();
        let mut tdef = proto::field_string(1, &t.name);
        tdef.extend(proto::field_string(2, &t.description));
        tdef.extend(proto::field_string(3, &schema_json));
        req.extend(proto::field_message(10, &tdef));
    }

    req.extend(proto::field_varint(7, 5)); // request_type = CASCADE
    req.extend(proto::field_string(16, cascade_id));
    req.extend(proto::field_string(21, model));

    proto::encode_frame(&req)
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
    fn fingerprint_is_sha256_hex() {
        assert_eq!(device_fingerprint("abc").len(), 64);
    }

    #[test]
    fn envelope_decodes_with_metadata_and_model() {
        let env = build_envelope("sk-ws-01-x", "jwt-y", &[Message::user("hi")], Some("sys"), &[], "swe-1-6-fast", "cid-1");
        // strip the Connect frame header (5 bytes)
        let frames = proto::decode_frames(&env);
        assert_eq!(frames.len(), 1);
        let fields = proto::decode_fields(&frames[0].1);
        assert!(fields.contains_key(&1)); // metadata
        assert_eq!(fields[&21][0].as_string(), "swe-1-6-fast"); // model
        assert_eq!(fields[&16][0].as_string(), "cid-1"); // cascade id
    }

    #[test]
    fn capabilities() {
        assert_eq!(WindsurfProvider::new().capabilities().provider, ProviderName::Windsurf);
    }
}
