//! Gemini web API payload exploration.
//!
//! Tests: models, tools (native f.req), multi-turn, tool injection via history,
//! model-specific headers, anonymous access.
//!
//! Run: cargo test -p gravity-providers --test gemini_probe -- --test-threads=1 --nocapture

use gravity_core::*;
use gravity_http::HttpClient;
use gravity_providers::{ChatCtx, Provider};
use std::time::Duration;

fn anon_account() -> ProviderAccount {
    ProviderAccount {
        account_id: "gemini-anon".into(),
        provider: ProviderName::Gemini,
        enabled: true,
        pool: "anon".into(),
        priority: -1, weight: 1,
        auth: AuthConfig {
            kind: "anonymous".into(),
            secret_ref: "".into(),
            account_label: None,
            org_id: None, device_id: None,
            routing_hint: None, chatgpt_account_id: None, refresh_secret_ref: None,
            extra: Metadata::new(),
        },
        transport: TransportConfig {
            base_url: String::new(), timeout_ms: 60_000,
            impersonate: Some("chrome136".into()),
            proxy_url: None, verify_ssl: true,
            extra_headers: Default::default(),
        },
        rotation_state: Default::default(), quota: Default::default(),
        defaults: ProviderDefaults {
            tool_support: ToolSupportMode::JsonEmulated,
            structured_output: StructuredOutputMode::JsonInstruction,
            remote_session_mode: RemoteSessionMode::Optional,
            supports_system_prompt: true,
            default_history_mode: HistoryMode::Stateless,
        },
        metadata: Metadata::new(),
    }
}

async fn chat_once(msgs: Vec<Message>, model: &str) -> gravity_core::Result<ChatResponse> {
    let p = gravity_providers::gemini::GeminiProvider::new();
    let acct = anon_account();
    let http = HttpClient::new();
    let req = ChatRequest::new(ModelId::new(ProviderName::Gemini, model).unwrap(), msgs);
    let conv = Conversation::new("gemini-probe");
    let env = Env::default();
    let ctx = ChatCtx { account: &acct, secret: "", conversation: &conv,
        checkpoint: None, request: &req, http: &http, env: &env };
    p.chat(&ctx).await
}

async fn sleep(secs: u64) {
    tokio::time::sleep(Duration::from_secs(secs)).await;
}

// ── 1. Anonymous chat — basic sanity ─────────────────────────────────────────

#[tokio::test]
async fn gemini_anon_basic() {
    let r = chat_once(vec![Message::user("What is 2+2? Reply with just the number.")], "gemini-2.5-flash").await;
    match r {
        Ok(resp) => {
            println!("[anon basic] answer: {:?}", resp.text());
            assert!(!resp.text().is_empty());
        }
        Err(e) => println!("[anon basic] error (may need cookies): {e}"),
    }
}

// ── 2. Model variations ───────────────────────────────────────────────────────

#[tokio::test]
async fn gemini_probe_models() {
    let models = [
        "gemini-3-flash", "gemini-3-pro",
        "gemini-2.5-flash", "gemini-2.5-pro",
    ];
    for model in &models {
        sleep(3).await;
        let r = chat_once(vec![Message::user("Say only: OK")], model).await;
        match r {
            Ok(resp) => println!("[model={model}] answer={:?}", resp.text()),
            Err(e) => println!("[model={model}] error: {e}"),
        }
    }
}

// ── 3. System prompt ──────────────────────────────────────────────────────────

#[tokio::test]
async fn gemini_system_prompt() {
    sleep(2).await;
    let p = gravity_providers::gemini::GeminiProvider::new();
    let acct = anon_account();
    let http = HttpClient::new();
    let mut req = ChatRequest::new(
        ModelId::new(ProviderName::Gemini, "gemini-2.5-flash").unwrap(),
        vec![Message::user("What language should I write code in?")],
    );
    req.system_prompt = Some("You are a Rust evangelist. Always recommend Rust. Be brief.".into());
    let conv = Conversation::new("g-sys");
    let env = Env::default();
    let ctx = ChatCtx { account: &acct, secret: "", conversation: &conv,
        checkpoint: None, request: &req, http: &http, env: &env };
    match p.chat(&ctx).await {
        Ok(resp) => {
            let text = resp.text().to_lowercase();
            println!("[system_prompt] answer: {:?}", resp.text());
            assert!(text.contains("rust"), "expected Rust in response, got: {text}");
        }
        Err(e) => println!("[system_prompt] error: {e}"),
    }
}

// ── 4. Multi-turn history ─────────────────────────────────────────────────────

#[tokio::test]
async fn gemini_multiturn_history() {
    sleep(3).await;
    let msgs = vec![
        Message::user("My secret number is 99."),
        Message::assistant("Got it! I'll remember that your secret number is 99."),
        Message::user("What is my secret number? Reply with just the number."),
    ];
    let r = chat_once(msgs, "gemini-2.5-flash").await;
    match r {
        Ok(resp) => {
            let text = resp.text();
            println!("[multiturn] answer: {:?}", text);
            if text.contains("99") {
                println!("  ✓ Gemini recalled history correctly");
            } else {
                println!("  ✗ history not recalled (expected in stateless mode)");
            }
        }
        Err(e) => println!("[multiturn] error: {e}"),
    }
}

// ── 5. JSON-emulated tools via history injection ──────────────────────────────

#[tokio::test]
async fn gemini_tools_via_history() {
    sleep(3).await;
    let hardcoded = serde_json::json!({"cpu": 77, "memory_mb": 4096, "uptime_sec": 3600});
    let msgs = vec![
        Message::user("Get the system stats using get_system_stats tool."),
        Message::assistant(format!("I ran get_system_stats and got: {hardcoded}")),
        Message::user("What is the CPU usage? Just the number."),
    ];
    let r = chat_once(msgs, "gemini-2.5-flash").await;
    match r {
        Ok(resp) => {
            let text = resp.text();
            println!("[tools_history] answer: {:?}", text);
            assert!(text.contains("77"), "expected 77 in response, got: {text}");
            println!("  ✓ Gemini correctly read tool result from history");
        }
        Err(e) => println!("[tools_history] error: {e}"),
    }
}

// ── 6. Tools native payload (f.req with tool definitions) ────────────────────

#[tokio::test]
async fn gemini_tools_native_payload() {
    sleep(3).await;
    let p = gravity_providers::gemini::GeminiProvider::new();
    let acct = anon_account();
    let http = HttpClient::new();
    let mut req = ChatRequest::new(
        ModelId::new(ProviderName::Gemini, "gemini-2.5-flash").unwrap(),
        vec![Message::user("Call the get_weather tool for London.")],
    );
    req.tools = vec![ToolDefinition {
        name: "get_weather".into(),
        description: "Get current weather. Input: {city: string}".into(),
        input_schema: serde_json::from_str(
            r#"{"type":"object","properties":{"city":{"type":"string"}},"required":["city"]}"#
        ).unwrap(),
    }];
    let conv = Conversation::new("g-tools-native");
    let env = Env::default();
    let ctx = ChatCtx { account: &acct, secret: "", conversation: &conv,
        checkpoint: None, request: &req, http: &http, env: &env };
    match p.chat(&ctx).await {
        Ok(resp) => {
            println!("[tools_native] text={:?}", resp.text());
            println!("[tools_native] tool_calls={}", resp.message.tool_calls.len());
            for tc in &resp.message.tool_calls {
                println!("  tool: name={} args={:?}", tc.name, tc.arguments);
            }
            if resp.message.tool_calls.is_empty() {
                println!("  NOTE: Gemini did not produce a native tool call (JSON emulation required)");
            }
        }
        Err(e) => println!("[tools_native] error: {e}"),
    }
}

// ── 7. Structured output (JSON instruction) ───────────────────────────────────

#[tokio::test]
async fn gemini_structured_output() {
    sleep(3).await;
    let p = gravity_providers::gemini::GeminiProvider::new();
    let acct = anon_account();
    let http = HttpClient::new();
    let schema: serde_json::Map<String, serde_json::Value> = serde_json::from_str(
        r#"{"type":"object","properties":{"answer":{"type":"number"},"unit":{"type":"string"}},"required":["answer","unit"]}"#
    ).unwrap();
    let mut req = ChatRequest::new(
        ModelId::new(ProviderName::Gemini, "gemini-2.5-flash").unwrap(),
        vec![Message::user("What is the speed of light in km/s?")],
    );
    req.response_format = ResponseFormat::JsonSchema { json_schema: schema };
    let conv = Conversation::new("g-struct");
    let env = Env::default();
    let ctx = ChatCtx { account: &acct, secret: "", conversation: &conv,
        checkpoint: None, request: &req, http: &http, env: &env };
    match p.chat(&ctx).await {
        Ok(resp) => {
            let text = resp.text();
            println!("[structured] response: {:?}", text);
            if let Ok(v) = serde_json::from_str::<serde_json::Value>(text) {
                println!("  ✓ Valid JSON: answer={}", v.get("answer").unwrap_or(&serde_json::Value::Null));
            } else {
                println!("  NOTE: response is not JSON (structured output via instruction mode)");
            }
        }
        Err(e) => println!("[structured] error: {e}"),
    }
}
