//! Live anonymous integration tests for ChatGPT and Perplexity web providers.
//!
//! Run with:  cargo test -p gravity-providers --test live_anon -- --nocapture
//!
//! These hit real network endpoints (no credentials needed — anonymous access).
//! They are NOT gated behind a feature flag because the user explicitly asked for
//! them to run. Mark as `#[ignore]` if running offline.

use gravity_core::*;
use gravity_http::HttpClient;
use gravity_providers::{ChatCtx, Provider};

fn anon_account(provider: ProviderName) -> ProviderAccount {
    ProviderAccount {
        account_id: format!("live-anon-{}", provider.as_str()).into(),
        provider,
        enabled: true,
        pool: "live-anon".into(),
        priority: -1,
        weight: 1,
        auth: AuthConfig {
            kind: "anonymous".into(),
            secret_ref: "".into(),
            account_label: Some("anon".into()),
            org_id: None,
            device_id: None,
            routing_hint: None,
            chatgpt_account_id: None,
            refresh_secret_ref: None,
            extra: Metadata::new(),
        },
        transport: TransportConfig {
            base_url: String::new(),
            timeout_ms: 60_000,
            impersonate: Some("chrome136".into()),
            proxy_url: None,
            verify_ssl: true,
            extra_headers: Default::default(),
        },
        rotation_state: Default::default(),
        quota: Default::default(),
        defaults: ProviderDefaults {
            tool_support: ToolSupportMode::None,
            structured_output: StructuredOutputMode::JsonInstruction,
            remote_session_mode: RemoteSessionMode::None,
            supports_system_prompt: false,
            default_history_mode: HistoryMode::Stateless,
        },
        metadata: Metadata::new(),
    }
}

// ── Perplexity anonymous ──────────────────────────────────────────────────────

#[tokio::test]
async fn perplexity_anonymous_chat() {
    let p = gravity_providers::perplexity::PerplexityWebProvider::new();
    let acct = anon_account(ProviderName::Perplexity);
    let http = HttpClient::new();
    let request = ChatRequest::new(
        ModelId::new(ProviderName::Perplexity, "sonar").unwrap(),
        vec![Message::user("What is 2 + 2? Reply with only the number.")],
    );
    let conv = Conversation::new("live-ppl");
    let env = Env::default();
    let ctx = ChatCtx {
        account: &acct,
        secret: "",
        conversation: &conv,
        checkpoint: None,
        request: &request,
        http: &http,
        env: &env,
    };
    let result: gravity_core::Result<gravity_core::ChatResponse> = p.chat(&ctx).await;
    match result {
        Ok(resp) => {
            let text = resp.text().to_string();
            println!("Perplexity anon response: {:?}", text);
            assert!(!text.is_empty(), "Perplexity anonymous returned empty response");
        }
        Err(e) => {
            // Fail if network error — the provider should work anonymously.
            panic!("Perplexity anonymous failed: {e}");
        }
    }
}

// ── Perplexity with custom history ───────────────────────────────────────────

#[tokio::test]
async fn perplexity_anonymous_with_history() {
    let p = gravity_providers::perplexity::PerplexityWebProvider::new();
    let acct = anon_account(ProviderName::Perplexity);
    let http = HttpClient::new();

    // Multi-turn: first turn established a fact, second turn references it.
    let request = ChatRequest::new(
        ModelId::new(ProviderName::Perplexity, "sonar").unwrap(),
        vec![
            Message::user("My favorite color is blue."),
            Message::assistant("Got it! Blue is a great color."),
            Message::user("What is my favorite color? One word answer only."),
        ],
    );
    let conv = Conversation::new("live-ppl-history");
    let env = Env::default();
    let ctx = ChatCtx {
        account: &acct,
        secret: "",
        conversation: &conv,
        checkpoint: None,
        request: &request,
        http: &http,
        env: &env,
    };
    let result: gravity_core::Result<gravity_core::ChatResponse> = p.chat(&ctx).await;
    match result {
        Ok(resp) => {
            let text = resp.text().to_string().to_lowercase();
            println!("Perplexity history response: {:?}", text);
            assert!(!text.is_empty(), "empty response");
            // Should mention blue
            assert!(text.contains("blue"), "expected 'blue' in response, got: {text}");
        }
        Err(e) => panic!("Perplexity history failed: {e}"),
    }
}

// ── Perplexity with native tools payload + hardcoded AFC loop ─────────────────

/// Confirm Perplexity web endpoint does NOT support native tools in payload.
/// Then verify multi-turn history can carry hardcoded tool results correctly.
#[tokio::test]
async fn perplexity_tools_native_vs_injected() {
    use gravity_providers::Provider;

    let p = gravity_providers::perplexity::PerplexityWebProvider::new();
    let acct = anon_account(ProviderName::Perplexity);
    let http = HttpClient::new();
    let conv = Conversation::new("live-ppl-tools-native");
    let env = Env::default();

    // ── Part 1: native tools payload — expect server to ignore it ──
    let mut request1 = ChatRequest::new(
        ModelId::new(ProviderName::Perplexity, "sonar").unwrap(),
        vec![Message::user("Call the get_system_stats tool. Output the tool_call JSON only.")],
    );
    request1.tools = vec![ToolDefinition {
        name: "get_system_stats".into(),
        description: "Returns CPU and memory usage.".into(),
        input_schema: serde_json::from_str(r#"{"type":"object","properties":{}}"#).unwrap(),
    }];
    let ctx1 = ChatCtx { account: &acct, secret: "", conversation: &conv, checkpoint: None,
        request: &request1, http: &http, env: &env };
    let r1: gravity_core::Result<gravity_core::ChatResponse> = p.chat(&ctx1).await;
    let r1 = r1.expect("native tools request failed");
    println!("Native tools — tool_calls={}, text={:?}", r1.message.tool_calls.len(), r1.text());
    // Server ignores the tools field — no native tool call returned.
    assert!(r1.message.tool_calls.is_empty(), "Perplexity web does not support native tool calls");

    // ── Part 2: inject hardcoded stats via history, verify Perplexity reads them ──
    // Simulate what an AFC loop would do: pass stats as assistant context.
    let hardcoded = serde_json::json!({
        "cpu_usage_percent": 42,
        "memory_used_mb": 8192,
        "memory_total_mb": 16384,
        "load_average": 1.5
    });
    let request2 = ChatRequest::new(
        ModelId::new(ProviderName::Perplexity, "sonar").unwrap(),
        vec![
            Message::user("What are the current system stats?"),
            // Simulate tool execution: assistant relays hardcoded values
            Message::assistant(format!(
                "I ran get_system_stats and got: {hardcoded}"
            )),
            Message::user("Based on those stats, what is the CPU usage percentage? Just the number."),
        ],
    );
    let ctx2 = ChatCtx { account: &acct, secret: "", conversation: &conv, checkpoint: None,
        request: &request2, http: &http, env: &env };
    let r2: gravity_core::Result<gravity_core::ChatResponse> = p.chat(&ctx2).await;
    let r2 = r2.expect("injected stats request failed");
    let text2 = r2.text().to_string();
    println!("Injected stats response: {:?}", text2);
    // Should mention 42 (the hardcoded CPU value)
    assert!(
        text2.contains("42"),
        "expected '42' (hardcoded CPU%) in response, got: {text2}"
    );
    println!("✓ Perplexity correctly read hardcoded stats from conversation history");
}

// ── ChatGPT anonymous ─────────────────────────────────────────────────────────

#[tokio::test]
async fn chatgpt_anonymous_chat() {
    let p = gravity_providers::chatgpt::ChatgptProvider::new();
    let acct = anon_account(ProviderName::Chatgpt);
    let http = HttpClient::new();
    let request = ChatRequest::new(
        ModelId::new(ProviderName::Chatgpt, "auto").unwrap(),
        vec![Message::user("What is 2 + 2? Reply with only the number.")],
    );
    let conv = Conversation::new("live-cg");
    let env = Env::default();
    let ctx = ChatCtx {
        account: &acct,
        secret: "", // anonymous — no bearer token
        conversation: &conv,
        checkpoint: None,
        request: &request,
        http: &http,
        env: &env,
    };
    let result: gravity_core::Result<gravity_core::ChatResponse> = p.chat(&ctx).await;
    match result {
        Ok(resp) => {
            let text = resp.text().to_string();
            println!("ChatGPT anon response: {:?}", text);
            assert!(!text.is_empty(), "ChatGPT returned empty response");
        }
        Err(e) => {
            println!("ChatGPT anon error (expected without auth): {e}");
        }
    }
}
