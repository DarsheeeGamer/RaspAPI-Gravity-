//! Perplexity web API payload exploration.
//!
//! Probes every known field: models, modes, sources, follow-up multi-turn,
//! incognito, extra fields, alternative endpoints, language variations.
//!
//! Run: cargo test -p gravity-providers --test perplexity_probe -- --test-threads=1 --nocapture
//!
//! Tests skip gracefully on 429 (rate limit) and print what they discover.

use gravity_http::HttpClient;
use serde_json::{json, Value};

const BASE: &str = "https://www.perplexity.ai";
const ASK_URL: &str = "https://www.perplexity.ai/rest/sse/perplexity_ask";
const SESSION_URL: &str = "https://www.perplexity.ai/api/auth/session";
const VERSION: &str = "2.18";
const UA: &str = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) \
    AppleWebKit/537.36 (KHTML, like Gecko) Chrome/136.0.0.0 Safari/537.36";

fn new_http() -> HttpClient { HttpClient::new() }

fn add_headers(req: gravity_http::HttpRequest) -> gravity_http::HttpRequest {
    req.header("accept", "text/event-stream")
       .header("content-type", "application/json")
       .header("origin", BASE)
       .header("referer", "https://www.perplexity.ai/")
       .header("user-agent", UA)
       .cookie_store(true)
}

async fn bootstrap(h: &HttpClient) {
    let _ = h.send(add_headers(gravity_http::HttpRequest::get(SESSION_URL))).await;
}

/// Send payload, retry once on 429 with 10s backoff.
async fn ask_raw(h: &HttpClient, payload: &Value) -> (u16, String) {
    for attempt in 0..2 {
        if attempt > 0 {
            tokio::time::sleep(std::time::Duration::from_secs(10)).await;
        }
        let req = add_headers(gravity_http::HttpRequest::post(ASK_URL))
            .body(bytes::Bytes::from(serde_json::to_vec(payload).unwrap()));
        match h.send(req).await {
            Ok(r) if r.status != 429 => return (r.status, r.text().into_owned()),
            Ok(r) => { println!("  [429 rate-limit, attempt {attempt}]"); let _ = r; }
            Err(e) => return (0, format!("ERROR: {e}")),
        }
    }
    (429, "rate-limited".to_string())
}

/// Extract final answer from SSE body.
fn answer(body: &str) -> String {
    let mut out = String::new();
    for event in body.split("\r\n\r\n") {
        let data = if let Some(p) = event.find("\r\ndata: ") { &event[p+8..] }
                   else if let Some(s) = event.strip_prefix("data: ") { s }
                   else { continue };
        let Ok(v) = serde_json::from_str::<Value>(data) else { continue };
        if let Some(text_raw) = v.get("text").and_then(Value::as_str) {
            if let Ok(steps) = serde_json::from_str::<Vec<Value>>(text_raw) {
                for step in &steps {
                    if step.get("step_type").and_then(Value::as_str) == Some("FINAL") {
                        if let Some(blob) = step.get("content")
                            .and_then(|c| c.get("answer")).and_then(Value::as_str) {
                            let text = if let Ok(inner) = serde_json::from_str::<Value>(blob) {
                                inner.get("answer").and_then(Value::as_str).unwrap_or(blob).to_owned()
                            } else { blob.to_owned() };
                            if !text.is_empty() { out = text; }
                        }
                        break;
                    }
                }
            }
        }
        if let Some(a) = v.get("answer").and_then(Value::as_str) {
            if !a.is_empty() { out = a.to_owned(); }
        }
    }
    out
}

/// Extract backend_uuid + display_model + mode from first events.
fn meta(body: &str) -> (Option<String>, Option<String>, Option<String>) {
    let (mut uuid, mut model, mut mode) = (None, None, None);
    for event in body.split("\r\n\r\n") {
        let data = if let Some(p) = event.find("\r\ndata: ") { &event[p+8..] }
                   else if let Some(s) = event.strip_prefix("data: ") { s }
                   else { continue };
        let Ok(v) = serde_json::from_str::<Value>(data) else { continue };
        if let Some(u) = v.get("backend_uuid").and_then(Value::as_str) { uuid = Some(u.to_owned()); }
        if let Some(m) = v.get("display_model").and_then(Value::as_str) { model = Some(m.to_owned()); }
        if let Some(m) = v.get("mode").and_then(Value::as_str) { mode = Some(m.to_owned()); }
    }
    (uuid, model, mode)
}

// ── 1. Mode variations ────────────────────────────────────────────────────────

#[tokio::test]
async fn probe_mode_concise() {
    let h = new_http(); bootstrap(&h).await;
    let payload = json!({ "query_str": "What is 1+1? One word.", "params": {
        "attachments": [], "frontend_context_uuid": "mc-a", "frontend_uuid": "mc-b",
        "is_incognito": true, "language": "en-US", "last_backend_uuid": null,
        "mode": "concise", "model_preference": "default",
        "source": "default", "sources": ["web"], "version": VERSION
    }});
    let (status, body) = ask_raw(&h, &payload).await;
    let (uuid, model, mode) = meta(&body);
    let ans = answer(&body);
    println!("[mode=concise] status={status} display_model={model:?} server_mode={mode:?}");
    println!("  backend_uuid={uuid:?}");
    println!("  answer: {ans:?}");
    if status == 429 { println!("  SKIP: rate limited"); return; }
    assert_eq!(status, 200);
    assert!(!ans.is_empty(), "expected non-empty answer");
}

#[tokio::test]
async fn probe_mode_copilot() {
    let h = new_http(); bootstrap(&h).await;
    let payload = json!({ "query_str": "What is 2+2? Just the number.", "params": {
        "attachments": [], "frontend_context_uuid": "cop-a", "frontend_uuid": "cop-b",
        "is_incognito": true, "language": "en-US", "last_backend_uuid": null,
        "mode": "copilot", "model_preference": "default",
        "source": "default", "sources": ["web"], "version": VERSION
    }});
    let (status, body) = ask_raw(&h, &payload).await;
    let (_, model, mode) = meta(&body);
    let ans = answer(&body);
    println!("[mode=copilot] status={status} display_model={model:?} server_mode={mode:?} answer={ans:?}");
    if status == 429 { println!("  SKIP: rate limited"); return; }
    println!("  body preview: {}", &body[..body.len().min(300)]);
}

// ── 2. Model preferences ──────────────────────────────────────────────────────

#[tokio::test]
async fn probe_model_preferences() {
    let h = new_http(); bootstrap(&h).await;
    // Try each model_preference from constants.py PERPLEXITY_MODEL_MAPPINGS
    let prefs = [
        "default", "turbo",
        // pro models (may need auth):
        "experimental", "pplx_pro",
        "claude45sonnet", "gpt52",
        // reasoning:
        "pplx_reasoning", "pplx_alpha",
    ];
    for pref in &prefs {
        tokio::time::sleep(std::time::Duration::from_millis(1500)).await;
        let payload = json!({ "query_str": "Say only: OK", "params": {
            "attachments": [], "frontend_context_uuid": format!("pref-{pref}-a"), "frontend_uuid": format!("pref-{pref}-b"),
            "is_incognito": true, "language": "en-US", "last_backend_uuid": null,
            "mode": "concise", "model_preference": pref,
            "source": "default", "sources": ["web"], "version": VERSION
        }});
        let (status, body) = ask_raw(&h, &payload).await;
        let (_, model, _) = meta(&body);
        let ans = answer(&body);
        println!("[model_preference={pref}] status={status} display_model={model:?} answer={ans:?}");
    }
}

// ── 3. Sources ────────────────────────────────────────────────────────────────

#[tokio::test]
async fn probe_sources() {
    let h = new_http(); bootstrap(&h).await;
    let sets: &[(&str, &[&str])] = &[
        ("web_only", &["web"]),
        ("scholar_only", &["scholar"]),
        ("social_only", &["social"]),
        ("web_scholar", &["web", "scholar"]),
        ("all", &["web", "scholar", "social"]),
    ];
    for (label, srcs) in sets {
        tokio::time::sleep(std::time::Duration::from_millis(2000)).await;
        let payload = json!({ "query_str": "latest AI research", "params": {
            "attachments": [], "frontend_context_uuid": format!("src-{label}-a"), "frontend_uuid": format!("src-{label}-b"),
            "is_incognito": true, "language": "en-US", "last_backend_uuid": null,
            "mode": "concise", "model_preference": "default",
            "source": "default", "sources": srcs, "version": VERSION
        }});
        let (status, body) = ask_raw(&h, &payload).await;
        let ans = answer(&body);
        println!("[sources={label}={srcs:?}] status={status} answer_len={}", ans.len());
        if status == 429 { println!("  SKIP"); continue; }
    }
}

// ── 4. follow_up multi-turn via last_backend_uuid ────────────────────────────

#[tokio::test]
async fn probe_followup_multiturn() {
    let h = new_http(); bootstrap(&h).await;

    // Turn 1
    let p1 = json!({ "query_str": "My secret number is 73. Just say 'Noted'.", "params": {
        "attachments": [], "frontend_context_uuid": "fu1-a", "frontend_uuid": "fu1-b",
        "is_incognito": false, "language": "en-US", "last_backend_uuid": null,
        "mode": "concise", "model_preference": "default",
        "source": "default", "sources": ["web"], "version": VERSION
    }});
    let (s1, b1) = ask_raw(&h, &p1).await;
    if s1 == 429 { println!("SKIP: rate limited"); return; }
    let (uuid1, model1, _) = meta(&b1);
    let ans1 = answer(&b1);
    println!("[turn1] status={s1} uuid={uuid1:?} model={model1:?} answer={ans1:?}");
    assert_eq!(s1, 200);

    let Some(backend_uuid) = uuid1 else {
        println!("  No backend_uuid — cannot test follow-up without auth.");
        return;
    };

    tokio::time::sleep(std::time::Duration::from_secs(2)).await;

    // Turn 2: follow-up
    let p2 = json!({ "query_str": "What is my secret number? Just the digits.", "params": {
        "attachments": [], "frontend_context_uuid": "fu2-a", "frontend_uuid": "fu2-b",
        "is_incognito": false, "language": "en-US", "last_backend_uuid": backend_uuid,
        "mode": "concise", "model_preference": "default",
        "source": "default", "sources": ["web"], "version": VERSION
    }});
    let (s2, b2) = ask_raw(&h, &p2).await;
    let ans2 = answer(&b2);
    println!("[turn2 last_backend_uuid=set] status={s2} answer={ans2:?}");
    if s2 == 200 {
        if ans2.contains("73") {
            println!("  ✓ FOLLOW-UP WORKS — server remembered '73' across turns");
        } else {
            println!("  ✗ server did NOT recall prior turn (anon sessions = no memory)");
            println!("  → For follow-up to work, need authenticated session with cookies");
        }
    }
}

// ── 5. Extra fields server may honor ─────────────────────────────────────────

#[tokio::test]
async fn probe_extra_fields() {
    let h = new_http(); bootstrap(&h).await;
    let extras: &[(&str, Value)] = &[
        ("search_recency_filter=day", json!({ "query_str": "latest news today", "params": {
            "attachments": [], "frontend_context_uuid": "ef1a", "frontend_uuid": "ef1b",
            "is_incognito": true, "language": "en-US", "last_backend_uuid": null,
            "mode": "concise", "model_preference": "default",
            "source": "default", "sources": ["web"], "version": VERSION,
            "search_recency_filter": "day"
        }})),
        ("search_recency_filter=hour", json!({ "query_str": "breaking news", "params": {
            "attachments": [], "frontend_context_uuid": "ef2a", "frontend_uuid": "ef2b",
            "is_incognito": true, "language": "en-US", "last_backend_uuid": null,
            "mode": "concise", "model_preference": "default",
            "source": "default", "sources": ["web"], "version": VERSION,
            "search_recency_filter": "hour"
        }})),
        ("timezone=UTC", json!({ "query_str": "What time is it UTC?", "params": {
            "attachments": [], "frontend_context_uuid": "ef3a", "frontend_uuid": "ef3b",
            "is_incognito": true, "language": "en-US", "last_backend_uuid": null,
            "mode": "concise", "model_preference": "default",
            "source": "default", "sources": ["web"], "version": VERSION,
            "timezone": "UTC"
        }})),
        ("search_focus=internet", json!({ "query_str": "rust programming", "params": {
            "attachments": [], "frontend_context_uuid": "ef4a", "frontend_uuid": "ef4b",
            "is_incognito": true, "language": "en-US", "last_backend_uuid": null,
            "mode": "concise", "model_preference": "default",
            "source": "default", "sources": ["web"], "version": VERSION,
            "search_focus": "internet"
        }})),
        ("query_source=home", json!({ "query_str": "hello", "params": {
            "attachments": [], "frontend_context_uuid": "ef5a", "frontend_uuid": "ef5b",
            "is_incognito": true, "language": "en-US", "last_backend_uuid": null,
            "mode": "concise", "model_preference": "default",
            "source": "default", "sources": ["web"], "version": VERSION,
            "query_source": "home"
        }})),
    ];
    for (label, payload) in extras {
        tokio::time::sleep(std::time::Duration::from_millis(2000)).await;
        let (status, body) = ask_raw(&h, payload).await;
        let ans = answer(&body);
        println!("[{label}] status={status} answer={:?}", &ans[..ans.len().min(80)]);
    }
}

// ── 6. Alternative endpoints ──────────────────────────────────────────────────

#[tokio::test]
async fn probe_alternative_endpoints() {
    let h = new_http(); bootstrap(&h).await;
    let oai_body = bytes::Bytes::from(serde_json::to_vec(&json!({
        "model": "sonar",
        "messages": [{"role": "user", "content": "Say hi"}],
        "stream": false
    })).unwrap());
    let ask_body = bytes::Bytes::from(serde_json::to_vec(&json!({
        "query_str": "hello",
        "params": { "attachments": [], "frontend_context_uuid": "alt-a", "frontend_uuid": "alt-b",
            "is_incognito": true, "language": "en-US", "last_backend_uuid": null,
            "mode": "concise", "model_preference": "default",
            "source": "default", "sources": ["web"], "version": VERSION }
    })).unwrap());
    let probes: &[(&str, bytes::Bytes)] = &[
        ("https://www.perplexity.ai/api/v1/chat/completions", oai_body.clone()),
        ("https://www.perplexity.ai/rest/sse/perplexity_ask_pro", ask_body.clone()),
        ("https://www.perplexity.ai/rest/sse/perplexity_ask_labs", ask_body.clone()),
        ("https://www.perplexity.ai/rest/perplexity_ask", ask_body.clone()),
        ("https://www.perplexity.ai/api/ask", ask_body.clone()),
        ("https://www.perplexity.ai/api/chat", oai_body.clone()),
        ("https://www.perplexity.ai/rest/v1/chat/completions", oai_body.clone()),
    ];
    for (url, body) in probes {
        let req = add_headers(gravity_http::HttpRequest::post(*url)).body(body.clone());
        match h.send(req).await {
            Ok(r) => {
                let preview: String = r.text().chars().take(80).collect();
                println!("[{url}] status={} preview={preview:?}", r.status);
            }
            Err(e) => println!("[{url}] error={e}"),
        }
    }
}

// ── 7. Full SSE structure analysis ────────────────────────────────────────────

#[tokio::test]
async fn probe_full_sse_structure() {
    let h = new_http(); bootstrap(&h).await;
    let payload = json!({ "query_str": "Capital of France? One word.", "params": {
        "attachments": [], "frontend_context_uuid": "sse-a", "frontend_uuid": "sse-b",
        "is_incognito": true, "language": "en-US", "last_backend_uuid": null,
        "mode": "concise", "model_preference": "default",
        "source": "default", "sources": ["web"], "version": VERSION
    }});
    let (status, body) = ask_raw(&h, &payload).await;
    if status == 429 { println!("SKIP: rate limited"); return; }
    println!("[full SSE] status={status} total_bytes={}", body.len());

    let mut event_count = 0usize;
    let mut top_level_keys: std::collections::BTreeSet<String> = Default::default();
    let mut step_types: std::collections::BTreeSet<String> = Default::default();
    let mut all_keys: std::collections::BTreeMap<String, usize> = Default::default();

    for event in body.split("\r\n\r\n") {
        let data = if let Some(p) = event.find("\r\ndata: ") { &event[p+8..] }
                   else if let Some(s) = event.strip_prefix("data: ") { s }
                   else { continue };
        event_count += 1;
        let Ok(v) = serde_json::from_str::<Value>(data) else { continue };
        if let Some(obj) = v.as_object() {
            for k in obj.keys() {
                top_level_keys.insert(k.clone());
                *all_keys.entry(k.clone()).or_default() += 1;
            }
        }
        if let Some(text_raw) = v.get("text").and_then(Value::as_str) {
            if let Ok(steps) = serde_json::from_str::<Vec<Value>>(text_raw) {
                for step in &steps {
                    if let Some(st) = step.get("step_type").and_then(Value::as_str) {
                        step_types.insert(st.to_owned());
                    }
                }
            }
        }
    }
    println!("  Total SSE events: {event_count}");
    println!("  Top-level JSON keys (all events): {top_level_keys:?}");
    println!("  Key occurrence counts: {all_keys:?}");
    println!("  Step types in .text[]: {step_types:?}");
    println!("  Answer: {:?}", answer(&body));
    assert_eq!(status, 200);
}

// ── 8. Language + version variations ─────────────────────────────────────────

#[tokio::test]
async fn probe_language_and_version() {
    let h = new_http(); bootstrap(&h).await;
    for (lang, ver, query) in [
        ("en-US", "2.18", "Say hello"),
        ("fr-FR", "2.18", "Dis bonjour"),
        ("de-DE", "2.18", "Sage hallo"),
        ("en-US", "2.0",  "Say hi"),  // old version
        ("en-US", "3.0",  "Say hi"),  // future version
    ] {
        tokio::time::sleep(std::time::Duration::from_millis(2000)).await;
        let payload = json!({ "query_str": query, "params": {
            "attachments": [], "frontend_context_uuid": format!("lv-{lang}-{ver}-a"), "frontend_uuid": format!("lv-{lang}-{ver}-b"),
            "is_incognito": true, "language": lang, "last_backend_uuid": null,
            "mode": "concise", "model_preference": "default",
            "source": "default", "sources": ["web"], "version": ver
        }});
        let (status, body) = ask_raw(&h, &payload).await;
        let ans = answer(&body);
        println!("[lang={lang} version={ver}] status={status} answer={:?}", &ans[..ans.len().min(60)]);
    }
}
