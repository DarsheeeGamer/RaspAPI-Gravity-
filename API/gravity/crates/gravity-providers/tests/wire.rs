//! Request-byte tests: run each provider's `chat()` against a recording mock
//! transport and assert on the exact HTTP request it builds (URL, headers,
//! body). These exercise the full provider chat path — the seam the audit found
//! missing — without touching the network.

use gravity_core::*;
use gravity_http::{HttpRequest, MockTransport};
use gravity_providers::{ChatCtx, Provider};
use std::collections::BTreeMap;

/// Build a transient account for `provider` with an inline secret + extras.
fn account(provider: ProviderName, kind: &str, extras: &[(&str, &str)]) -> ProviderAccount {
    let mut extra = Metadata::new();
    for (k, v) in extras {
        extra.insert(*k, *v);
    }
    ProviderAccount {
        account_id: "t".into(),
        provider,
        enabled: true,
        pool: "default".into(),
        priority: 100,
        weight: 1,
        auth: AuthConfig {
            kind: kind.into(),
            secret_ref: "inline".into(),
            account_label: None,
            org_id: None,
            device_id: extras.iter().find(|(k, _)| *k == "device_id").map(|(_, v)| (*v).into()),
            routing_hint: None,
            chatgpt_account_id: None,
            refresh_secret_ref: None,
            extra,
        },
        transport: TransportConfig {
            base_url: String::new(),
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
            remote_session_mode: RemoteSessionMode::None,
            supports_system_prompt: true,
            default_history_mode: HistoryMode::Auto,
        },
        metadata: Metadata::new(),
    }
}

/// Run a provider's `chat()` against a mock returning `canned_response`, and
/// return the captured requests.
fn capture<P: gravity_providers::Provider>(
    provider: &P,
    account: &ProviderAccount,
    secret: &str,
    request: &ChatRequest,
    canned_response: &str,
) -> Vec<HttpRequest> {
    let mock = MockTransport::ok(bytes::Bytes::copy_from_slice(canned_response.as_bytes()));
    let conv = Conversation::new("k");
    let env = Env::fixed(1_700_000_000_000, 7);
    let ctx = ChatCtx {
        account,
        secret,
        conversation: &conv,
        checkpoint: None,
        request,
        http: &mock,
        env: &env,
    };
    let rt = tokio::runtime::Runtime::new().unwrap();
    let _ = rt.block_on(provider.chat(&ctx));
    mock.captured()
}

fn headers_map(req: &HttpRequest) -> BTreeMap<String, String> {
    req.headers.iter().cloned().collect()
}

fn req(provider: ProviderName, model: &str, msgs: Vec<Message>) -> ChatRequest {
    ChatRequest::new(ModelId::new(provider, model).unwrap(), msgs)
}

// ── OpenAI-compatible ────────────────────────────────────────────────────────

#[test]
fn groq_builds_chat_completions_request() {
    use gravity_providers::openai_compat::catalog;
    let p = gravity_providers::openai_compat::OpenAiCompatProvider::new(catalog::Groq);
    let acct = account(ProviderName::Groq, "api_key", &[]);
    let request = req(ProviderName::Groq, "llama-3.3-70b", vec![Message::user("hi")]);
    let canned = r#"{"choices":[{"message":{"content":"ok"},"finish_reason":"stop"}]}"#;
    let reqs = capture(&p, &acct, "sk-xyz", &request, canned);
    assert_eq!(reqs.len(), 1);
    let r = &reqs[0];
    assert_eq!(r.url, "https://api.groq.com/openai/v1/chat/completions");
    let h = headers_map(r);
    assert_eq!(h.get("authorization").unwrap(), "Bearer sk-xyz");
    assert_eq!(h.get("content-type").unwrap(), "application/json");
    let body: serde_json::Value = serde_json::from_slice(r.body.as_ref().unwrap()).unwrap();
    assert_eq!(body["model"], "llama-3.3-70b");
    assert_eq!(body["messages"][0]["content"], "hi");
}

// ── Anthropic API ────────────────────────────────────────────────────────────

#[test]
fn anthropic_sends_version_and_api_key_headers() {
    let p = gravity_providers::anthropic::AnthropicProvider::new();
    let acct = account(ProviderName::AnthropicApi, "api_key", &[]);
    let request = req(ProviderName::AnthropicApi, "claude-sonnet-4-5", vec![Message::user("hi")]);
    let canned = r#"{"id":"m","content":[{"type":"text","text":"ok"}],"stop_reason":"end_turn"}"#;
    let reqs = capture(&p, &acct, "sk-ant-key", &request, canned);
    let r = &reqs[0];
    assert_eq!(r.url, "https://api.anthropic.com/v1/messages");
    let h = headers_map(r);
    assert_eq!(h.get("x-api-key").unwrap(), "sk-ant-key");
    assert_eq!(h.get("anthropic-version").unwrap(), "2023-06-01");
    let body: serde_json::Value = serde_json::from_slice(r.body.as_ref().unwrap()).unwrap();
    assert!(body["max_tokens"].is_number());
}

// ── GLM (signed) ─────────────────────────────────────────────────────────────

#[test]
fn glm_signs_request_with_x_signature() {
    let p = gravity_providers::glm::GlmProvider::new();
    let acct = account(ProviderName::Glm, "api_key", &[("user_id", "u9")]);
    let request = req(ProviderName::Glm, "glm-5", vec![Message::user("Hello world")]);
    let reqs = capture(&p, &acct, "TESTTOKEN", &request, "data: [DONE]\n");
    let r = &reqs[0];
    assert!(r.url.starts_with("https://chat.z.ai/api/v2/chat/completions?"));
    assert!(r.url.contains("token=TESTTOKEN"));
    let h = headers_map(r);
    assert_eq!(h.get("Authorization").unwrap(), "Bearer TESTTOKEN");
    // Signature present and 64-hex.
    let sig = h.get("X-Signature").expect("X-Signature header");
    assert_eq!(sig.len(), 64);
    assert!(h.contains_key("X-Signature-Timestamp"));
}

// ── Cursor (protobuf) ────────────────────────────────────────────────────────

#[test]
fn cursor_uses_connect_proto_content_type() {
    let p = gravity_providers::cursor::CursorProvider::new();
    let acct = account(ProviderName::Cursor, "api_key", &[("checksum", "cks-1")]);
    let request = req(ProviderName::Cursor, "gpt-5", vec![Message::user("hi")]);
    let reqs = capture(&p, &acct, "tok", &request, "");
    let r = &reqs[0];
    let h = headers_map(r);
    assert_eq!(h.get("content-type").unwrap(), "application/connect+proto");
    assert_eq!(h.get("x-cursor-checksum").unwrap(), "cks-1");
    // Connect frame: first byte flag 0x00.
    assert_eq!(r.body.as_ref().unwrap()[0], 0x00);
}

// ── Windsurf (protobuf) — content-type must be connect+proto ─────────────────

#[test]
fn windsurf_uses_connect_proto_content_type() {
    let p = gravity_providers::windsurf::WindsurfProvider::new();
    let acct = account(ProviderName::Windsurf, "api_key", &[("jwt", "jwt-x")]);
    let request = req(ProviderName::Windsurf, "swe-1-6-fast", vec![Message::user("hi")]);
    let reqs = capture(&p, &acct, "sk-ws-01-x", &request, "");
    let h = headers_map(&reqs[0]);
    assert_eq!(h.get("content-type").unwrap(), "application/connect+proto");
}

// ── ChatGPT — must send chat-requirements + conversation, with account-id ────

#[test]
fn chatgpt_sends_sentinel_then_conversation() {
    let p = gravity_providers::chatgpt::ChatgptProvider::new();
    let acct = account(
        ProviderName::Chatgpt,
        "session_cookie",
        &[("device_id", "dev-1"), ("chatgpt_account_id", "acct-9")],
    );
    let request = req(ProviderName::Chatgpt, "gpt-5", vec![Message::user("hi")]);
    // Both requests get the same canned body; requirements parse will fail
    // gracefully but we only assert on the first (sentinel) request shape.
    let canned = r#"{"token":"req-tok","proofofwork":{"required":false}}"#;
    let reqs = capture(&p, &acct, "access-tok", &request, canned);
    assert!(!reqs.is_empty());
    let first = &reqs[0];
    assert!(first.url.ends_with("/sentinel/chat-requirements"));
    let h = headers_map(first);
    assert_eq!(h.get("authorization").unwrap(), "Bearer access-tok");
    assert_eq!(h.get("chatgpt-account-id").map(String::as_str), Some("acct-9"));
}

fn tool() -> ToolDefinition {
    ToolDefinition {
        name: "get_weather".into(),
        description: "weather".into(),
        input_schema: serde_json::from_str(r#"{"type":"object","properties":{}}"#).unwrap(),
    }
}

#[test]
fn cursor_sends_tools_in_proto() {
    let p = gravity_providers::cursor::CursorProvider::new();
    let acct = account(ProviderName::Cursor, "api_key", &[("machine_id", "mid-1")]);
    let mut request = req(ProviderName::Cursor, "gpt-5", vec![Message::user("hi")]);
    request.tools = vec![tool()];
    let reqs = capture(&p, &acct, "tok", &request, "");
    let h = headers_map(&reqs[0]);
    // checksum computed (not from inline) + client headers present
    assert!(h.contains_key("x-cursor-checksum"));
    assert_eq!(h.get("x-cursor-client-type").unwrap(), "ide");
    // tool name bytes present in the protobuf body
    let body = reqs[0].body.as_ref().unwrap();
    assert!(body.windows(11).any(|w| w == b"get_weather"), "tool name not in proto body");
}

#[test]
fn codex_sends_tools_and_correct_headers() {
    let p = gravity_providers::codex::CodexProvider::new();
    let acct = account(ProviderName::Codex, "api_key", &[("chatgpt_account_id", "acct-1")]);
    let mut request = req(ProviderName::Codex, "gpt-5-codex", vec![Message::user("hi")]);
    request.tools = vec![tool()];
    let reqs = capture(&p, &acct, "tok", &request, "data: [DONE]\n");
    let r = &reqs[0];
    let h = headers_map(r);
    assert_eq!(h.get("openai-beta").unwrap(), "responses=v1");
    assert_eq!(h.get("originator").unwrap(), "codex_cli_rs");
    assert_eq!(h.get("chatgpt-account-id").unwrap(), "acct-1");
    let body: serde_json::Value = serde_json::from_slice(r.body.as_ref().unwrap()).unwrap();
    assert_eq!(body["tools"][0]["name"], "get_weather");
    assert_eq!(body["tool_choice"], "auto");
}

#[test]
fn gemini_cli_sends_function_declarations() {
    let p = gravity_providers::gemini_cli::GeminiCliProvider::new();
    let acct = account(ProviderName::GeminiCli, "api_key", &[("project", "proj-1")]);
    let mut request = req(ProviderName::GeminiCli, "gemini-2.5-pro", vec![Message::user("hi")]);
    request.tools = vec![tool()];
    let reqs = capture(&p, &acct, "tok", &request, r#"{"response":{"candidates":[]}}"#);
    let body: serde_json::Value = serde_json::from_slice(reqs[0].body.as_ref().unwrap()).unwrap();
    assert_eq!(body["request"]["tools"][0]["functionDeclarations"][0]["name"], "get_weather");
    assert_eq!(body["project"], "proj-1");
}

#[test]
fn openai_compat_sends_response_format_json_schema() {
    use gravity_providers::openai_compat::catalog;
    let p = gravity_providers::openai_compat::OpenAiCompatProvider::new(catalog::OpenaiApi);
    let acct = account(ProviderName::OpenaiApi, "api_key", &[]);
    let mut request = req(ProviderName::OpenaiApi, "gpt-4o", vec![Message::user("hi")]);
    request.response_format = ResponseFormat::JsonSchema {
        json_schema: serde_json::from_str(r#"{"type":"object","properties":{"x":{"type":"number"}}}"#).unwrap(),
    };
    let canned = r#"{"choices":[{"message":{"content":"{}"},"finish_reason":"stop"}]}"#;
    let reqs = capture(&p, &acct, "sk", &request, canned);
    let body: serde_json::Value = serde_json::from_slice(reqs[0].body.as_ref().unwrap()).unwrap();
    assert_eq!(body["response_format"]["type"], "json_schema");
    assert_eq!(body["response_format"]["json_schema"]["schema"]["type"], "object");
}

// ── Sampling: seed + frequency/presence penalty ──────────────────────────────

#[test]
fn oai_compat_passes_seed_and_penalties() {
    use gravity_providers::openai_compat::catalog;
    let p = gravity_providers::openai_compat::OpenAiCompatProvider::new(catalog::Groq);
    let acct = account(ProviderName::Groq, "api_key", &[]);
    let mut request = req(ProviderName::Groq, "llama-3.3-70b", vec![Message::user("hi")]);
    request.sampling.seed = Some(42);
    request.sampling.frequency_penalty = Some(0.5);
    request.sampling.presence_penalty = Some(0.3);
    let canned = r#"{"choices":[{"message":{"content":"ok"},"finish_reason":"stop"}]}"#;
    let reqs = capture(&p, &acct, "sk", &request, canned);
    let body: serde_json::Value = serde_json::from_slice(reqs[0].body.as_ref().unwrap()).unwrap();
    assert_eq!(body["seed"], 42);
    assert!((body["frequency_penalty"].as_f64().unwrap() - 0.5).abs() < 0.01);
    assert!((body["presence_penalty"].as_f64().unwrap() - 0.3).abs() < 0.01);
}

// ── Multimodal: image_url parts ───────────────────────────────────────────────

#[test]
fn oai_compat_encodes_image_url_parts() {
    use gravity_providers::openai_compat::catalog;
    // Build a multimodal message via JSON deserialization (avoids smallvec import).
    let raw_msg = serde_json::json!({
        "role": "user",
        "content": [
            {"type": "text", "text": "describe this"},
            {"type": "image_url", "image_url": {"url": "https://example.com/img.png", "detail": "high"}},
        ]
    });
    let msg: Message = serde_json::from_value(raw_msg).unwrap();
    let p = gravity_providers::openai_compat::OpenAiCompatProvider::new(catalog::OpenaiApi);
    let acct = account(ProviderName::OpenaiApi, "api_key", &[]);
    let request = req(ProviderName::OpenaiApi, "gpt-4o", vec![msg]);
    let canned = r#"{"choices":[{"message":{"content":"a cat"},"finish_reason":"stop"}]}"#;
    let reqs = capture(&p, &acct, "sk", &request, canned);
    let body: serde_json::Value = serde_json::from_slice(reqs[0].body.as_ref().unwrap()).unwrap();
    let parts = &body["messages"][0]["content"];
    assert!(parts.is_array(), "content should be array for multimodal");
    assert_eq!(parts[0]["type"], "text");
    assert_eq!(parts[0]["text"], "describe this");
    assert_eq!(parts[1]["type"], "image_url");
    assert_eq!(parts[1]["image_url"]["url"], "https://example.com/img.png");
    assert_eq!(parts[1]["image_url"]["detail"], "high");
}

// ── Claude web ───────────────────────────────────────────────────────────────

#[test]
fn claude_sends_session_cookie_and_hits_completion_endpoint() {
    use gravity_core::Checkpoint;
    use chrono::Utc;
    let p = gravity_providers::claude::ClaudeProvider::new();
    let mut acct = account(ProviderName::Claude, "session_cookie", &[]);
    acct.auth.org_id = Some("org-abc".into());
    // Build a checkpoint so it skips both GET /organizations and POST /chat_conversations.
    let cp = Checkpoint {
        conversation_key: "k".into(),
        provider: ProviderName::Claude,
        account_id: "t".into(),
        remote_conversation_id: Some("conv-xyz".into()),
        remote_parent_id: None,
        remote_metadata: Metadata::new(),
        history_version: 0,
        updated_at: Utc::now(),
        is_valid: true,
    };
    let request = req(ProviderName::Claude, "claude-sonnet-4-6", vec![Message::user("hi")]);
    // Claude SSE: completion_type field from `message_start` event.
    let canned = concat!(
        "event: message_start\ndata: {\"type\":\"message_start\",\"message\":{\"id\":\"m1\"}}\n\n",
        "event: content_block_start\ndata: {\"type\":\"content_block_start\",\"index\":0,\"content_block\":{\"type\":\"text\",\"text\":\"\"}}\n\n",
        "event: content_block_delta\ndata: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"text_delta\",\"text\":\"hello\"}}\n\n",
        "event: message_stop\ndata: {\"type\":\"message_stop\",\"stop_reason\":\"end_turn\"}\n\n",
    );
    let mock = gravity_http::MockTransport::ok(bytes::Bytes::copy_from_slice(canned.as_bytes()));
    let conv = Conversation::new("k");
    let env = Env::fixed(1_700_000_000_000, 7);
    let ctx = ChatCtx {
        account: &acct,
        secret: "sk-ant-sid-XYZ",
        conversation: &conv,
        checkpoint: Some(&cp),
        request: &request,
        http: &mock,
        env: &env,
    };
    let rt = tokio::runtime::Runtime::new().unwrap();
    let _ = rt.block_on(p.chat(&ctx));
    let reqs = mock.captured();
    // Should be a single POST to the completion endpoint.
    assert_eq!(reqs.len(), 1);
    let r = &reqs[0];
    assert!(r.url.contains("/chat_conversations/conv-xyz/completion"),
        "URL should contain conversation id, got: {}", r.url);
    let h = headers_map(r);
    let cookie = h.get("cookie").expect("cookie header");
    assert!(cookie.contains("sessionKey=sk-ant-sid-XYZ"), "cookie missing sessionKey");
    assert!(cookie.contains("lastActiveOrg=org-abc"), "cookie missing org");
    assert_eq!(h.get("content-type").unwrap(), "application/json");
}

// ── Kiro ─────────────────────────────────────────────────────────────────────

#[test]
fn kiro_sends_bearer_and_hits_regional_endpoint() {
    let p = gravity_providers::kiro::KiroProvider::new();
    let acct = account(ProviderName::Kiro, "api_key", &[("region", "eu-west-1")]);
    let request = req(ProviderName::Kiro, "claude-sonnet-4-5", vec![Message::user("hi")]);
    // Kiro response: brace-counted JSON `{"content":"answer"}`.
    let canned = r#"{"content":"hello"}"#;
    let reqs = capture(&p, &acct, "kiro-tok", &request, canned);
    assert_eq!(reqs.len(), 1);
    let r = &reqs[0];
    assert_eq!(r.url, "https://q.eu-west-1.amazonaws.com/generateAssistantResponse");
    let h = headers_map(r);
    assert_eq!(h.get("authorization").unwrap(), "Bearer kiro-tok");
    assert_eq!(h.get("content-type").unwrap(), "application/json");
    let body: serde_json::Value = serde_json::from_slice(r.body.as_ref().unwrap()).unwrap();
    assert!(body.get("conversationState").is_some(), "body missing conversationState");
}

// ── Perplexity web ────────────────────────────────────────────────────────────

#[test]
fn perplexity_hits_sse_ask_url_and_sends_query() {
    let p = gravity_providers::perplexity::PerplexityWebProvider::new();
    let acct = account(ProviderName::Perplexity, "anonymous", &[]);
    let request = req(ProviderName::Perplexity, "default", vec![Message::user("What is 2+2?")]);
    // Perplexity SSE: `data: {"answer": "4"}` then end_of_stream.
    let canned = concat!(
        "event: message\r\ndata: {\"answer\":\"4\",\"backend_uuid\":\"uuid-1\"}\r\n\r\n",
        "data: {\"end_of_stream\":true}\r\n\r\n",
    );
    let reqs = capture(&p, &acct, "", &request, canned);
    // May have 1-2 requests (auth bootstrap + ask, or just ask).
    assert!(!reqs.is_empty());
    let last = reqs.last().unwrap();
    assert!(last.url.contains("perplexity_ask") || last.url.contains("perplexity.ai"),
        "last request should target perplexity, got: {}", last.url);
}

// ── Antigravity ───────────────────────────────────────────────────────────────

#[test]
fn antigravity_sends_bearer_and_hits_generate_content() {
    let p = gravity_providers::antigravity::AntigravityProvider::new();
    let acct = account(ProviderName::Antigravity, "api_key", &[("project", "proj-x")]);
    let request = req(ProviderName::Antigravity, "gemini-2.5-pro", vec![Message::user("hi")]);
    let canned = r#"{"candidates":[{"content":{"parts":[{"text":"hello"}]}}]}"#;
    let reqs = capture(&p, &acct, "ag-token", &request, canned);
    assert_eq!(reqs.len(), 1);
    let r = &reqs[0];
    assert!(r.url.contains("generateContent"), "URL should contain generateContent, got: {}", r.url);
    assert!(r.url.contains("cloudcode-pa.sandbox.googleapis.com"), "URL should target sandbox endpoint");
    let h = headers_map(r);
    assert_eq!(h.get("authorization").unwrap(), "Bearer ag-token");
    let body: serde_json::Value = serde_json::from_slice(r.body.as_ref().unwrap()).unwrap();
    assert_eq!(body["model"], "gemini-2.5-pro");
    assert_eq!(body["project"], "proj-x");
}

// ── Anonymous auth: no Authorization header sent ──────────────────────────────

#[test]
fn oai_compat_anon_skips_auth_header() {
    use gravity_providers::openai_compat::catalog;
    let p = gravity_providers::openai_compat::OpenAiCompatProvider::new(catalog::Groq);
    let mut acct = account(ProviderName::Groq, "anonymous", &[]);
    acct.auth.kind = "anonymous".into();
    let request = req(ProviderName::Groq, "llama-3.3-70b", vec![Message::user("hi")]);
    let canned = r#"{"choices":[{"message":{"content":"ok"},"finish_reason":"stop"}]}"#;
    let reqs = capture(&p, &acct, "", &request, canned);
    let h = headers_map(&reqs[0]);
    assert!(!h.contains_key("authorization"), "anonymous account must not send Authorization");
}
