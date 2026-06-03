//! `discover_models` tests: run each provider's model-discovery path against a
//! mock transport returning a canned model-list body and assert the parsed
//! [`ModelInfo`] set. Covers the reverse-engineered endpoints (Cursor proto,
//! ChatGPT `/backend-api/models`, Gemini `otAQ7b` batchexecute, GLM `/api/models`)
//! plus the OpenAI-compatible `GET /models`.

use gravity_core::*;
use gravity_http::MockTransport;
use gravity_providers::{ChatCtx, Provider};

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
            device_id: None,
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

/// Run a provider's `discover_models()` against a mock returning `canned`.
fn discover<P: Provider>(
    provider: &P,
    acct: &ProviderAccount,
    secret: &str,
    model: &str,
    canned: &str,
) -> Vec<ModelInfo> {
    let mock = MockTransport::ok(bytes::Bytes::copy_from_slice(canned.as_bytes()));
    let conv = Conversation::new("k");
    let env = Env::fixed(1_700_000_000_000, 7);
    let request = ChatRequest::new(
        ModelId::new(acct.provider, model).unwrap(),
        vec![Message::user("hi")],
    );
    let ctx = ChatCtx {
        account: acct,
        secret,
        conversation: &conv,
        checkpoint: None,
        request: &request,
        http: &mock,
        env: &env,
    };
    let rt = tokio::runtime::Runtime::new().unwrap();
    rt.block_on(provider.discover_models(&ctx)).unwrap()
}

// ── OpenAI-compatible: GET /models → {"data":[{"id":...}]} ────────────────────

#[test]
fn openai_compat_parses_data_array() {
    use gravity_providers::openai_compat::{catalog, OpenAiCompatProvider};
    let p = OpenAiCompatProvider::new(catalog::Groq);
    let acct = account(ProviderName::Groq, "api_key", &[]);
    let canned = r#"{"object":"list","data":[
        {"id":"llama-3.3-70b-versatile","object":"model"},
        {"id":"llama-3.1-8b-instant","object":"model"}
    ]}"#;
    let models = discover(&p, &acct, "gsk_x", "llama-3.3-70b", canned);
    let ids: Vec<&str> = models.iter().map(|m| m.id.as_str()).collect();
    assert!(ids.contains(&"llama-3.3-70b-versatile"));
    assert!(ids.contains(&"llama-3.1-8b-instant"));
}

#[test]
fn openai_compat_falls_back_on_bad_body() {
    use gravity_providers::openai_compat::{catalog, OpenAiCompatProvider};
    let p = OpenAiCompatProvider::new(catalog::Groq);
    let acct = account(ProviderName::Groq, "api_key", &[]);
    // No "data" key → fall back to the static catalog (non-empty).
    let models = discover(&p, &acct, "gsk_x", "llama-3.3-70b", r#"{"oops":true}"#);
    assert!(!models.is_empty());
}

// ── ChatGPT: GET /backend-api/models → {"models":[{"slug","title",...}]} ──────

#[test]
fn chatgpt_parses_models_with_slug_title() {
    let p = gravity_providers::chatgpt::ChatgptProvider::new();
    let acct = account(ProviderName::Chatgpt, "session_cookie", &[]);
    let canned = r#"{"models":[
        {"slug":"gpt-4o","title":"GPT-4o","description":"omni","tags":["gpt4"]},
        {"slug":"o4-mini","title":"o4-mini","description":"fast"}
    ],"categories":[]}"#;
    let models = discover(&p, &acct, "access-tok", "gpt-4o", canned);
    assert_eq!(models[0].id, "gpt-4o");
    assert_eq!(models[0].display_name.as_deref(), Some("GPT-4o"));
    assert_eq!(models[0].tags, vec!["gpt4".to_string()]);
    assert!(models.iter().any(|m| m.id == "o4-mini"));
}

#[test]
fn chatgpt_anonymous_uses_static() {
    let p = gravity_providers::chatgpt::ChatgptProvider::new();
    let acct = account(ProviderName::Chatgpt, "anonymous", &[]);
    // Empty secret → no /models call, static fallback.
    let models = discover(&p, &acct, "", "gpt-4o", "{}");
    assert!(!models.is_empty());
}

// ── GLM: GET /api/models → {"data":[{"id","name"}]} ───────────────────────────

#[test]
fn glm_parses_openwebui_models() {
    let p = gravity_providers::glm::GlmProvider::new();
    let acct = account(ProviderName::Glm, "api_key", &[]);
    let canned = r#"{"data":[
        {"id":"glm-4.6","name":"GLM-4.6"},
        {"id":"glm-4.5-air","name":"GLM-4.5-Air"}
    ]}"#;
    let models = discover(&p, &acct, "TOK", "glm-4.6", canned);
    let ids: Vec<&str> = models.iter().map(|m| m.id.as_str()).collect();
    assert!(ids.contains(&"glm-4.6"));
    assert_eq!(models[0].display_name.as_deref(), Some("GLM-4.6"));
}

// ── Cursor: GetUsableModels proto (field 1 repeated, inner field 1 = name) ────

#[test]
fn cursor_parses_proto_model_names() {
    let p = gravity_providers::cursor::CursorProvider::new();
    let acct = account(ProviderName::Cursor, "api_key", &[("machine_id", "mid")]);
    // Proto: outer field 1 (tag 0x0A) = message; inner field 1 (tag 0x0A) = name.
    // Two models: "auto" (4 bytes) and "gpt-5" (5 bytes).
    // inner_auto = 0A 04 'a''u''t''o'      (6 bytes)
    // outer1     = 0A 06 <inner_auto>      (8 bytes)
    // inner_gpt5 = 0A 05 'g''p''t''-''5'   (7 bytes)
    // outer2     = 0A 07 <inner_gpt5>      (9 bytes)
    let proto: Vec<u8> = vec![
        0x0A, 0x06, 0x0A, 0x04, b'a', b'u', b't', b'o',
        0x0A, 0x07, 0x0A, 0x05, b'g', b'p', b't', b'-', b'5',
    ];
    let mock = MockTransport::ok(bytes::Bytes::from(proto));
    let conv = Conversation::new("k");
    let env = Env::fixed(1, 7);
    let request = ChatRequest::new(ModelId::new(ProviderName::Cursor, "auto").unwrap(), vec![Message::user("hi")]);
    let ctx = ChatCtx {
        account: &acct, secret: "tok", conversation: &conv,
        checkpoint: None, request: &request, http: &mock, env: &env,
    };
    let rt = tokio::runtime::Runtime::new().unwrap();
    let models = rt.block_on(p.discover_models(&ctx)).unwrap();
    let ids: Vec<&str> = models.iter().map(|m| m.id.as_str()).collect();
    assert!(ids.contains(&"auto"), "got {ids:?}");
    assert!(ids.contains(&"gpt-5"), "got {ids:?}");
}

// ── Gemini: otAQ7b batchexecute → frame[2] JSON, [15] = [[hex,name,desc],…] ───

#[test]
fn gemini_parses_otaq7b_models_and_maps_hex() {
    let p = gravity_providers::gemini::GeminiProvider::new();
    // Warm session so bootstrap short-circuits (no GET /app needed).
    let acct = account(
        ProviderName::Gemini,
        "session_cookie",
        &[("build_label", "boq_x"), ("session_id", "sid_x"), ("access_token", "at_x"),
          ("__Secure-1PSID", "psid_x")],
    );

    // inner[14] = status 1000, inner[15] = models list of [hex, display, desc].
    // fbb127bbb056c959 is the known hex for gemini-3-flash → reverse-mapped.
    let mut inner = vec![serde_json::Value::Null; 16];
    inner[14] = serde_json::json!(1000);
    inner[15] = serde_json::json!([
        ["fbb127bbb056c959", "Fast", "Quick responses"],
        ["deadbeefdeadbeef", "Mystery", "Unknown model"]
    ]);
    let inner_str = serde_json::Value::Array(inner).to_string();
    // batchexecute wraps each RPC reply in an outer array; parse_response_frames
    // flattens one level, leaving the ["wrb.fr",rpcid,payload,…] array as a frame.
    let frame = serde_json::json!([["wrb.fr", "otAQ7b", inner_str, null, null, "generic"]]).to_string();
    // Length counts UTF-16 units from the leading `\n` (inclusive): 1 + frame chars.
    let len = 1 + frame.chars().count();
    let canned = format!(")]}}'\n{len}\n{frame}");

    let models = discover(&p, &acct, "at_x", "gemini-2.5-flash", &canned);
    // Known hex maps to the request-usable name.
    assert!(models.iter().any(|m| m.id == "gemini-3-flash"), "got {models:?}");
    assert!(models.iter().any(|m| m.display_name.as_deref() == Some("Fast")));
    // Unknown hex is kept verbatim as the id.
    assert!(models.iter().any(|m| m.id == "deadbeefdeadbeef"));
}
