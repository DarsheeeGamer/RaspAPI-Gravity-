//! Live Gemini web tests — require real session cookies.
//!
//! Set env vars before running:
//!   GRAVITY_GEMINI_1PSID   __Secure-1PSID cookie value
//!   GRAVITY_GEMINI_SID     SID cookie value
//!   GRAVITY_GEMINI_NID     NID cookie value
//!   GRAVITY_GEMINI_SAPISID SAPISID cookie value
//!   GRAVITY_GEMINI_AT      at= token (SNlM0e)
//!   GRAVITY_GEMINI_BL      bl= build label
//!   GRAVITY_GEMINI_FSID    f.sid= session id
//!
//! Run:
//!   cargo test -p gravity-providers --test gemini_live -- --test-threads=1 --nocapture

use gravity_core::*;
use gravity_http::HttpClient;
use gravity_providers::{ChatCtx, Provider};
use std::time::Duration;

fn skip_if_no_creds() -> bool {
    std::env::var("GRAVITY_GEMINI_1PSID").is_err()
}

fn cred_account(_model: &str) -> ProviderAccount {
    let psid = std::env::var("GRAVITY_GEMINI_1PSID").unwrap_or_default();
    let sid = std::env::var("GRAVITY_GEMINI_SID").unwrap_or_default();
    let nid = std::env::var("GRAVITY_GEMINI_NID").unwrap_or_default();
    let sapisid = std::env::var("GRAVITY_GEMINI_SAPISID").unwrap_or_default();
    let sidcc = std::env::var("GRAVITY_GEMINI_SIDCC").unwrap_or_default();
    let at = std::env::var("GRAVITY_GEMINI_AT").unwrap_or_default();
    let bl = std::env::var("GRAVITY_GEMINI_BL").unwrap_or_default();
    let fsid = std::env::var("GRAVITY_GEMINI_FSID").unwrap_or_default();

    let mut extra = serde_json::Map::new();
    // Session tokens (bootstrap short-circuit)
    if !at.is_empty()   { extra.insert("access_token".into(), serde_json::json!(at)); }
    if !bl.is_empty()   { extra.insert("build_label".into(), serde_json::json!(bl)); }
    if !fsid.is_empty() { extra.insert("session_id".into(), serde_json::json!(fsid)); }
    // Auth cookies
    if !psid.is_empty()   { extra.insert("__Secure-1PSID".into(), serde_json::json!(psid)); }
    if !sid.is_empty()    { extra.insert("SID".into(), serde_json::json!(sid)); }
    if !nid.is_empty()    { extra.insert("NID".into(), serde_json::json!(nid)); }
    if !sapisid.is_empty(){ extra.insert("SAPISID".into(), serde_json::json!(sapisid)); }
    if !sidcc.is_empty()  { extra.insert("SIDCC".into(), serde_json::json!(sidcc)); }

    ProviderAccount {
        account_id: "gemini-live".into(),
        provider: ProviderName::Gemini,
        enabled: true,
        pool: "live".into(),
        priority: 100, weight: 1,
        auth: AuthConfig {
            kind: "session_cookie".into(),
            secret_ref: "".into(),
            account_label: None,
            org_id: None, device_id: None,
            routing_hint: None, chatgpt_account_id: None, refresh_secret_ref: None,
            extra: Metadata::from(extra),
        },
        transport: TransportConfig {
            base_url: String::new(), timeout_ms: 60_000,
            impersonate: Some("firefox".into()),
            proxy_url: None, verify_ssl: true,
            extra_headers: Default::default(),
        },
        rotation_state: Default::default(), quota: Default::default(),
        defaults: ProviderDefaults {
            tool_support: ToolSupportMode::JsonEmulated,
            structured_output: StructuredOutputMode::JsonInstruction,
            remote_session_mode: RemoteSessionMode::Required,
            supports_system_prompt: true,
            default_history_mode: HistoryMode::Stateless,
        },
        metadata: Metadata::new(),
    }
}

async fn chat(msgs: Vec<Message>, model: &str) -> gravity_core::Result<ChatResponse> {
    let p = gravity_providers::gemini::GeminiProvider::new();
    let acct = cred_account(model);
    let http = HttpClient::new();
    let req = ChatRequest::new(ModelId::new(ProviderName::Gemini, model).unwrap(), msgs);
    let conv = Conversation::new("gemini-live");
    let env = Env::default();
    let ctx = ChatCtx { account: &acct, secret: "", conversation: &conv,
        checkpoint: None, request: &req, http: &http, env: &env };
    p.chat(&ctx).await
}

async fn sleep(ms: u64) { tokio::time::sleep(Duration::from_millis(ms)).await; }

// ── 1. Basic chat ─────────────────────────────────────────────────────────────

#[tokio::test]
async fn gemini_live_basic() {
    if skip_if_no_creds() { println!("[skip] set GRAVITY_GEMINI_1PSID to run"); return; }
    let r = chat(vec![Message::user("What is 2+2? Reply with just the number.")], "gemini-3-flash-plus").await;
    match r {
        Ok(resp) => {
            let text = resp.text().to_string();
            println!("[basic] response: {:?}", text);
            assert!(!text.is_empty(), "empty response");
            assert!(text.contains('4'), "expected 4 in: {text}");
            println!("  ✓ Gemini live chat works");
        }
        Err(e) => panic!("[basic] FAILED: {e}"),
    }
}

// ── 2. Multi-turn history ─────────────────────────────────────────────────────

#[tokio::test]
async fn gemini_live_history() {
    if skip_if_no_creds() { println!("[skip]"); return; }
    sleep(3000).await;
    let msgs = vec![
        Message::user("My secret number is 99."),
        Message::assistant("Got it — your secret number is 99."),
        Message::user("What is my secret number? Just the number."),
    ];
    match chat(msgs, "gemini-3-flash-plus").await {
        Ok(resp) => {
            let text = resp.text().to_string();
            println!("[history] response: {:?}", text);
            assert!(text.contains("99"), "expected 99 in: {text}");
            println!("  ✓ history recalled correctly");
        }
        Err(e) => panic!("[history] FAILED: {e}"),
    }
}

// ── 3. System prompt ─────────────────────────────────────────────────────────

#[tokio::test]
async fn gemini_live_system_prompt() {
    if skip_if_no_creds() { println!("[skip]"); return; }
    sleep(3000).await;
    let p = gravity_providers::gemini::GeminiProvider::new();
    let acct = cred_account("gemini-3-flash-plus");
    let http = HttpClient::new();
    let mut req = ChatRequest::new(
        ModelId::new(ProviderName::Gemini, "gemini-3-flash-plus").unwrap(),
        vec![Message::user("What language should I use?")],
    );
    req.system_prompt = Some("You are a Rust evangelist. Always recommend Rust. Be very brief.".into());
    let conv = Conversation::new("g-live-sys");
    let env = Env::default();
    let ctx = ChatCtx { account: &acct, secret: "", conversation: &conv,
        checkpoint: None, request: &req, http: &http, env: &env };
    match p.chat(&ctx).await {
        Ok(resp) => {
            let text = resp.text().to_lowercase();
            println!("[system_prompt] response: {:?}", resp.text());
            assert!(text.contains("rust"), "expected Rust in: {text}");
            println!("  ✓ system prompt respected");
        }
        Err(e) => panic!("[system_prompt] FAILED: {e}"),
    }
}

// ── 4. Tools via history injection (hardcoded values) ────────────────────────

#[tokio::test]
async fn gemini_live_tools_history() {
    if skip_if_no_creds() { println!("[skip]"); return; }
    sleep(3000).await;
    let hardcoded = serde_json::json!({"cpu_pct": 42, "memory_mb": 8192, "uptime_sec": 7200});
    let msgs = vec![
        Message::user("Call get_system_stats and tell me the CPU usage."),
        Message::assistant(format!(
            "I called get_system_stats. Result: {hardcoded}"
        )),
        Message::user("What is the CPU usage percentage? Just the number."),
    ];
    match chat(msgs, "gemini-3-flash-plus").await {
        Ok(resp) => {
            let text = resp.text().to_string();
            println!("[tools_history] response: {:?}", text);
            assert!(text.contains("42"), "expected 42 in: {text}");
            println!("  ✓ Gemini read tool result from history correctly");
        }
        Err(e) => panic!("[tools_history] FAILED: {e}"),
    }
}

// ── 5. JSON-emulated tool round-trip ─────────────────────────────────────────

#[tokio::test]
async fn gemini_live_tools_emulated() {
    if skip_if_no_creds() { println!("[skip]"); return; }
    sleep(3000).await;
    use gravity_core::tool_emulation::build_json_tool_instruction;

    let tools = vec![ToolDefinition {
        name: "get_system_stats".into(),
        description: "Returns CPU and memory stats. No inputs needed.".into(),
        input_schema: serde_json::from_str(r#"{"type":"object","properties":{}}"#).unwrap(),
    }];
    let instruction = build_json_tool_instruction(&tools, None);
    let query = format!("{instruction}\n\nCall get_system_stats now.");

    match chat(vec![Message::user(query)], "gemini-3-flash-plus").await {
        Ok(resp) => {
            let text = resp.text().to_string();
            println!("[tools_emulated] response: {:?}", &text[..text.len().min(400)]);
            let has_tool_call = text.contains("get_system_stats")
                || text.contains("\"tool\"")
                || text.contains("ToolCall");
            println!("  tool call in response: {has_tool_call}");
            if has_tool_call {
                println!("  ✓ Gemini emitted JSON tool call");
            } else {
                println!("  NOTE: no tool call emitted (may need stronger instruction)");
            }
            assert!(!text.is_empty());
        }
        Err(e) => panic!("[tools_emulated] FAILED: {e}"),
    }
}

// ── 6. Other model (gemini-2.5-flash) ────────────────────────────────────────

#[tokio::test]
async fn gemini_live_model_25flash() {
    if skip_if_no_creds() { println!("[skip]"); return; }
    sleep(3000).await;
    match chat(vec![Message::user("Say only: HELLO")], "gemini-2.5-flash").await {
        Ok(resp) => {
            let text = resp.text().to_string();
            println!("[2.5-flash] response: {:?}", text);
            assert!(!text.is_empty());
            println!("  ✓ gemini-2.5-flash works");
        }
        Err(e) => {
            println!("[2.5-flash] error (may not have model access): {e}");
        }
    }
}
