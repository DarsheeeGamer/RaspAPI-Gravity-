//! Gemini payload experimentation — probes native history threading, tools,
//! and other inner_req_list structures.
//!
//! Run:
//!   GRAVITY_GEMINI_1PSID=... GRAVITY_GEMINI_AT=... GRAVITY_GEMINI_BL=... \
//!   GRAVITY_GEMINI_FSID=... cargo test -p gravity-providers --test gemini_payload_probe \
//!   -- --test-threads=1 --nocapture

use gravity_core::*;
use gravity_http::{HttpClient, HttpRequest, Profile};
use gravity_providers::gemini::parser;
use std::time::Duration;

fn skip_if_no_creds() -> bool {
    std::env::var("GRAVITY_GEMINI_1PSID").is_err()
}

fn cred_account() -> ProviderAccount {
    let psid  = std::env::var("GRAVITY_GEMINI_1PSID").unwrap_or_default();
    let sid   = std::env::var("GRAVITY_GEMINI_SID").unwrap_or_default();
    let nid   = std::env::var("GRAVITY_GEMINI_NID").unwrap_or_default();
    let sapisid = std::env::var("GRAVITY_GEMINI_SAPISID").unwrap_or_default();
    let sidcc = std::env::var("GRAVITY_GEMINI_SIDCC").unwrap_or_default();
    let at    = std::env::var("GRAVITY_GEMINI_AT").unwrap_or_default();
    let bl    = std::env::var("GRAVITY_GEMINI_BL").unwrap_or_default();
    let fsid  = std::env::var("GRAVITY_GEMINI_FSID").unwrap_or_default();

    let mut extra = serde_json::Map::new();
    if !at.is_empty()     { extra.insert("access_token".into(), serde_json::json!(at)); }
    if !bl.is_empty()     { extra.insert("build_label".into(), serde_json::json!(bl)); }
    if !fsid.is_empty()   { extra.insert("session_id".into(), serde_json::json!(fsid)); }
    if !psid.is_empty()   { extra.insert("__Secure-1PSID".into(), serde_json::json!(psid)); }
    if !sid.is_empty()    { extra.insert("SID".into(), serde_json::json!(sid)); }
    if !nid.is_empty()    { extra.insert("NID".into(), serde_json::json!(nid)); }
    if !sapisid.is_empty(){ extra.insert("SAPISID".into(), serde_json::json!(sapisid)); }
    if !sidcc.is_empty()  { extra.insert("SIDCC".into(), serde_json::json!(sidcc)); }

    ProviderAccount {
        account_id: "gemini-probe".into(),
        provider: ProviderName::Gemini,
        enabled: true,
        pool: "probe".into(),
        priority: 100, weight: 1,
        auth: AuthConfig {
            kind: "session_cookie".into(),
            secret_ref: "".into(),
            account_label: None, org_id: None, device_id: None,
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

async fn sleep(ms: u64) { tokio::time::sleep(Duration::from_millis(ms)).await; }

fn cookie_header() -> String {
    let psid  = std::env::var("GRAVITY_GEMINI_1PSID").unwrap_or_default();
    let sid   = std::env::var("GRAVITY_GEMINI_SID").unwrap_or_default();
    let nid   = std::env::var("GRAVITY_GEMINI_NID").unwrap_or_default();
    let sapisid = std::env::var("GRAVITY_GEMINI_SAPISID").unwrap_or_default();
    let sidcc = std::env::var("GRAVITY_GEMINI_SIDCC").unwrap_or_default();
    let mut parts = Vec::new();
    if !psid.is_empty()   { parts.push(format!("__Secure-1PSID={psid}")); }
    if !sid.is_empty()    { parts.push(format!("SID={sid}")); }
    if !nid.is_empty()    { parts.push(format!("NID={nid}")); }
    if !sapisid.is_empty(){ parts.push(format!("SAPISID={sapisid}")); }
    if !sidcc.is_empty()  { parts.push(format!("SIDCC={sidcc}")); }
    parts.join("; ")
}

fn url_encode(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for b in s.bytes() {
        match b {
            b'A'..=b'Z' | b'a'..=b'z' | b'0'..=b'9' | b'-' | b'_' | b'.' | b'~' => out.push(b as char),
            _ => out.push_str(&format!("%{b:02X}")),
        }
    }
    out
}

struct SessionTokens { at: String, bl: String, fsid: String }

/// Do a real bootstrap: GET /app with cookies, extract fresh at=, bl=, f.sid.
async fn bootstrap_session(http: &HttpClient) -> std::result::Result<SessionTokens, String> {
    let cookies = cookie_header();
    let req = HttpRequest::get("https://gemini.google.com/app")
        .profile(Profile::Firefox)
        .timeout_ms(30_000)
        .cookie_store(true)
        .header("user-agent", "Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:151.0) Gecko/20100101 Firefox/151.0")
        .header("accept", "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8")
        .header("referer", "https://gemini.google.com/");
    let req = if !cookies.is_empty() { req.header("cookie", cookies) } else { req };
    let resp = http.send(req).await.map_err(|e| e.to_string())?;
    let body = resp.text().into_owned();
    fn extract(html: &str, key: &str) -> Option<String> {
        let needle = format!("\"{key}\":");
        let start = html.find(&needle)? + needle.len();
        let rest = html[start..].trim_start().strip_prefix('"')?;
        let end = rest.find('"')?;
        let v = &rest[..end];
        if v.is_empty() { None } else { Some(v.to_owned()) }
    }
    let at = extract(&body, "SNlM0e").unwrap_or_default();
    let bl = extract(&body, "cfb2h").unwrap_or_default();
    let fsid = extract(&body, "FdrFJe").unwrap_or_default();
    let has_snlm0e = body.contains("SNlM0e");
    println!("[bootstrap] status={} body_len={} has_SNlM0e={} at={:?} bl={:?} fsid={:?}",
        "ok", body.len(), has_snlm0e,
        &at[..at.len().min(20)], &bl[..bl.len().min(30)], &fsid[..fsid.len().min(20)]);
    if !has_snlm0e {
        // Show beginning of body to diagnose
        println!("[bootstrap] body[:500]: {:?}", &body[..body.len().min(500)]);
    }
    Ok(SessionTokens { at, bl, fsid })
}

/// Send a raw f.req payload and return the raw response body for inspection.
async fn raw_post_with(http: &HttpClient, inner: serde_json::Value, tokens: &SessionTokens) -> std::result::Result<String, String> {
    let at = &tokens.at;
    let bl = &tokens.bl;
    let fsid = &tokens.fsid;

    let inner_str = inner.to_string();
    let freq_payload = serde_json::json!([serde_json::Value::Null, inner_str]).to_string();

    let mut url = format!(
        "https://gemini.google.com/_/BardChatUi/data/assistant.lamda.BardFrontendService/StreamGenerate?_reqid={}&rt=c",
        10000 + (rand_u64() % 900000)
    );
    if !bl.is_empty()   { url.push_str(&format!("&bl={}", url_encode(bl))); }
    url.push_str("&hl=en-US");
    if !fsid.is_empty() { url.push_str(&format!("&f.sid={}", url_encode(fsid))); }

    let body = if at.is_empty() {
        format!("f.req={}", url_encode(&freq_payload))
    } else {
        format!("f.req={}&at={}", url_encode(&freq_payload), url_encode(at))
    };

    let cookies = cookie_header();
    let mut req = HttpRequest::post(&url)
        .profile(Profile::Firefox)
        .timeout_ms(60_000)
        .cookie_store(true)
        .header("user-agent", "Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:151.0) Gecko/20100101 Firefox/151.0")
        .header("accept", "*/*")
        .header("accept-language", "en-US,en;q=0.9")
        .header("content-type", "application/x-www-form-urlencoded;charset=utf-8")
        .header("origin", "https://gemini.google.com")
        .header("referer", "https://gemini.google.com/")
        .header("x-same-domain", "1")
        .header("x-goog-ext-73010989-jspb", "[0]")
        .header("x-goog-ext-73010990-jspb", "[0,0,0]")
        .header("sec-fetch-dest", "empty")
        .header("sec-fetch-mode", "cors")
        .header("sec-fetch-site", "same-origin");
    if !cookies.is_empty() { req = req.header("cookie", cookies); }
    req = req.body(bytes::Bytes::from(body));

    let resp = http.send(req).await.map_err(|e| e.to_string())?;
    Ok(resp.text().into_owned())
}

/// Send a raw f.req payload and return the raw response body for inspection.
async fn raw_post(inner: serde_json::Value) -> std::result::Result<String, String> {
    let at  = std::env::var("GRAVITY_GEMINI_AT").unwrap_or_default();
    let bl  = std::env::var("GRAVITY_GEMINI_BL").unwrap_or_default();
    let fsid = std::env::var("GRAVITY_GEMINI_FSID").unwrap_or_default();

    let inner_str = inner.to_string();
    let freq_payload = serde_json::json!([serde_json::Value::Null, inner_str]).to_string();

    let mut url = format!(
        "https://gemini.google.com/_/BardChatUi/data/assistant.lamda.BardFrontendService/StreamGenerate?_reqid={}&rt=c",
        10000 + (rand_u64() % 900000)
    );
    if !bl.is_empty()   { url.push_str(&format!("&bl={}", url_encode(&bl))); }
    url.push_str("&hl=en-US");
    if !fsid.is_empty() { url.push_str(&format!("&f.sid={}", url_encode(&fsid))); }

    let body = if at.is_empty() {
        format!("f.req={}", url_encode(&freq_payload))
    } else {
        format!("f.req={}&at={}", url_encode(&freq_payload), url_encode(&at))
    };

    let http = HttpClient::new();
    let cookies = cookie_header();
    let mut req = HttpRequest::post(&url)
        .profile(Profile::Firefox)
        .timeout_ms(60_000)
        .cookie_store(true)
        .header("content-type", "application/x-www-form-urlencoded;charset=utf-8")
        .header("origin", "https://gemini.google.com")
        .header("referer", "https://gemini.google.com/")
        .header("x-same-domain", "1")
        .header("user-agent", "Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:151.0) Gecko/20100101 Firefox/151.0");
    if !cookies.is_empty() {
        req = req.header("cookie", cookies);
    }
    req = req.body(bytes::Bytes::from(body));

    use gravity_http::Transport;
    let resp = http.send(req).await.map_err(|e| e.to_string())?;
    Ok(resp.text().into_owned())
}

fn rand_u64() -> u64 {
    // Simple LCG seeded by time — only for _reqid, not security-relevant.
    use std::time::{SystemTime, UNIX_EPOCH};
    SystemTime::now().duration_since(UNIX_EPOCH).map(|d| d.subsec_nanos() as u64).unwrap_or(12345)
}

fn base_inner(prompt: &str) -> Vec<serde_json::Value> {
    use serde_json::{json, Value};
    let mut inner: Vec<Value> = vec![Value::Null; 81];
    inner[0] = json!([prompt, 0, Value::Null, Value::Null, Value::Null, Value::Null, 0]);
    inner[1] = json!(["en"]);
    inner[2] = json!(["", "", "", Value::Null, Value::Null, Value::Null, Value::Null, Value::Null, Value::Null, ""]);
    inner[6] = json!([1]);
    inner[7] = json!(1);
    inner[10] = json!(1);
    inner[11] = json!(0);
    inner[17] = json!([[0]]);
    inner[18] = json!(0);
    inner[27] = json!(1);
    inner[30] = json!([4]);
    inner[41] = json!([1]);
    inner[45] = json!(1); // temporary chat
    inner[53] = json!(0);
    let uuid = format!("{:08X}-{:04X}-4{:03X}-{:04X}-{:012X}",
        rand_u64() & 0xFFFFFFFF, rand_u64() & 0xFFFF, rand_u64() & 0x0FFF,
        (rand_u64() & 0x3FFF) | 0x8000, rand_u64() & 0xFFFFFFFFFFFF);
    inner[59] = json!(uuid);
    inner[61] = json!([]);
    inner[68] = json!(2);
    inner[79] = json!(1);
    inner[80] = json!(1);
    inner
}

fn parse_response_body(body: &str) -> (String, Vec<Option<String>>) {
    let frames = parser::extract_json_from_response(body);
    let parsed = parser::parse_parts(&frames);
    (parsed.text, parsed.metadata)
}

// ── 1. Stateful threading: two-turn conversation via c_id/r_id ───────────────
//
// Turn 1: say secret number. Extract c_id/r_id/rcid from response.
// Turn 2: ask for the secret number, passing c_id/r_id in inner[2].
// If Gemini threads correctly, it recalls "42" without re-stating it in turn 2.

#[tokio::test]
async fn probe_stateful_threading() {
    if skip_if_no_creds() { println!("[skip] set GRAVITY_GEMINI_1PSID"); return; }

    // Turn 1 — NOT temporary so the conversation persists server-side.
    let mut inner1 = base_inner("My secret number is 42. Acknowledge with exactly: NOTED");
    inner1[45] = serde_json::Value::Null; // clear temporary flag → persistent conversation
    let body1 = raw_post(serde_json::Value::Array(inner1)).await.unwrap();
    println!("[turn1] body[:500]: {:?}", &body1[..body1.len().min(500)]);

    let frames1 = parser::extract_json_from_response(&body1);
    let parsed1 = parser::parse_parts(&frames1);
    println!("[turn1] text={:?}", &parsed1.text[..parsed1.text.len().min(100)]);
    println!("[turn1] metadata={:?}", &parsed1.metadata);
    println!("[turn1] rcid={:?}", &parsed1.rcid);

    let c_id = parsed1.metadata.get(0).and_then(|v| v.as_deref()).unwrap_or("").to_owned();
    let r_id = parsed1.metadata.get(1).and_then(|v| v.as_deref()).unwrap_or("").to_owned();
    let rcid = parsed1.rcid.as_deref().unwrap_or("").to_owned();
    let state_token = parsed1.state_token.as_deref().unwrap_or("").to_owned();
    println!("[turn1] c_id={c_id:?} r_id={r_id:?} rcid={rcid:?}");
    println!("[turn1] state_token={state_token:?}");

    if c_id.is_empty() {
        println!("[threading] no c_id — cannot thread (session expired or auth issue)");
        return;
    }

    sleep(2000).await;

    use serde_json::{json, Value};

    // Turn 2a: c_id+r_id+rcid but NO state_token in [9]
    let mut inner2a = base_inner("What is my secret number? Just the number.");
    inner2a[45] = Value::Null;
    inner2a[2] = json!([c_id, r_id, rcid, Value::Null, Value::Null, Value::Null, Value::Null, Value::Null, Value::Null, ""]);
    let body2a = raw_post(serde_json::Value::Array(inner2a)).await.unwrap();
    let (text2a, _) = parse_response_body(&body2a);
    println!("[turn2a no_state_token] text={:?}", &text2a[..text2a.len().min(200)]);
    println!("  body[:300]: {:?}", &body2a[..body2a.len().min(300)]);

    sleep(2000).await;

    // Turn 2b: c_id+r_id+rcid+state_token (full HAR-verified format)
    let mut inner2b = base_inner("What is my secret number? Just the number.");
    inner2b[45] = Value::Null;
    inner2b[2] = json!([c_id, r_id, rcid, Value::Null, Value::Null, Value::Null, Value::Null, Value::Null, Value::Null, state_token]);
    let body2b = raw_post(serde_json::Value::Array(inner2b)).await.unwrap();
    let (text2b, _) = parse_response_body(&body2b);
    println!("[turn2b with_state_token] text={:?}", &text2b[..text2b.len().min(200)]);
    println!("  body[:300]: {:?}", &body2b[..body2b.len().min(300)]);

    if text2a.contains("42") {
        println!("  ✓ STATEFUL THREADING (no state_token) WORKS");
    } else if text2b.contains("42") {
        println!("  ✓ STATEFUL THREADING (with state_token) WORKS");
    } else {
        println!("  ✗ neither turn recalled '42'");
        println!("  (Note: state_token from 1st turn empty={:?} — may need fresh at= token)", state_token.is_empty());
    }
}

// ── 2. Probe inner[0] with full message array format ─────────────────────────
//
// Hypothesis: inner[0] might accept a richer format with multiple turns.
// Test: pass a 2-element array at inner[0] to see if server interprets it.

#[tokio::test]
async fn probe_inner0_rich_format() {
    if skip_if_no_creds() { println!("[skip]"); return; }
    sleep(3000).await;
    use serde_json::json;

    // Try: inner[0] = [[user_text, 0,...], [prior_user_text, 0,...]]  (speculative)
    let mut inner = base_inner("");
    inner[0] = json!([
        ["What is my secret number? Reply: the number only.", 0, null, null, null, null, 0],
        // Extra element — hypothesis: prior context messages
        ["My secret number is 77.", 1, null, null, null, null, 0],
    ]);
    let body = raw_post(serde_json::Value::Array(inner)).await.unwrap();
    println!("[inner0_rich] raw_body[:400]: {:?}", &body[..body.len().min(400)]);
    let (text, _) = parse_response_body(&body);
    println!("[inner0_rich] text={:?}", &text[..text.len().min(200)]);
    if text.contains("77") {
        println!("  ✓ inner[0] accepts multi-message array format!");
    } else if !text.is_empty() {
        println!("  NOTE: got response but no '77' — format not recognised");
    } else {
        println!("  no parseable response");
    }
}

// ── 3. Probe inner[35] for history array ─────────────────────────────────────
//
// HAR analysis found [35] unused. Could be a history/context slot.

#[tokio::test]
async fn probe_inner35_history() {
    if skip_if_no_creds() { println!("[skip]"); return; }
    sleep(3000).await;
    use serde_json::json;

    let mut inner = base_inner("What is my secret number? Just the number.");
    // Speculative: inner[35] = prior conversation turns
    inner[35] = json!([["My secret number is 55.", "user"], ["Got it!", "assistant"]]);
    let body = raw_post(serde_json::Value::Array(inner)).await.unwrap();
    let (text, _) = parse_response_body(&body);
    println!("[inner35_history] text={:?}", &text[..text.len().min(200)]);
    if text.contains("55") {
        println!("  ✓ inner[35] history slot works!");
    } else if !text.is_empty() {
        println!("  NOTE: response exists but no recall (inner[35] ignored)");
    } else {
        println!("  no response");
    }
}

// ── 4. Native tools: probe inner[64] and other candidate slots ───────────────
//
// Gemini API uses functionDeclarations. Try passing them in unused indices.

#[tokio::test]
async fn probe_native_tools_slots() {
    if skip_if_no_creds() { println!("[skip]"); return; }
    sleep(3000).await;
    use serde_json::json;

    let tool_def = json!({
        "name": "get_system_stats",
        "description": "Returns CPU and memory stats",
        "parameters": {
            "type": "object",
            "properties": {
                "detail": {"type": "string"}
            }
        }
    });

    // Try inner[64] as function declarations slot (unexplored in HAR)
    let mut inner = base_inner("Call get_system_stats to check system health. What does it return?");
    inner[64] = json!([[tool_def.clone()]]);
    let body = raw_post(serde_json::Value::Array(inner)).await.unwrap();
    let (text, _) = parse_response_body(&body);
    println!("[inner64_tools] text={:?}", &text[..text.len().min(300)]);

    // Try inner[36] as another candidate
    sleep(2000).await;
    let mut inner2 = base_inner("Use get_system_stats tool.");
    inner2[36] = json!([{"functionDeclarations": [tool_def.clone()]}]);
    let body2 = raw_post(serde_json::Value::Array(inner2)).await.unwrap();
    let (text2, _) = parse_response_body(&body2);
    println!("[inner36_tools] text={:?}", &text2[..text2.len().min(300)]);
}

// ── 4b. Two-turn with fresh bootstrap between turns ──────────────────────────
//
// Hypothesis: at= (SNlM0e) token is per-session and needs refresh after use.
// Bootstrap fresh before each turn.

#[tokio::test]
async fn probe_stateful_fresh_bootstrap() {
    if skip_if_no_creds() { println!("[skip]"); return; }
    let http = HttpClient::new();

    // Bootstrap 1 → turn 1
    let tokens1 = match bootstrap_session(&http).await {
        Ok(t) => t,
        Err(e) => { println!("[bootstrap1] error: {e}"); return; }
    };
    if tokens1.fsid.is_empty() {
        println!("[bootstrap1] no f.sid returned (maybe redirect or auth error)");
    }
    println!("[bootstrap1] fsid={:?}", tokens1.fsid);

    let mut inner1 = base_inner("My secret number is 99. Acknowledge: NOTED");
    inner1[45] = serde_json::Value::Null; // persistent
    let body1 = raw_post_with(&http, serde_json::Value::Array(inner1), &tokens1).await.unwrap();
    let frames1 = parser::extract_json_from_response(&body1);
    let parsed1 = parser::parse_parts(&frames1);
    println!("[turn1] text={:?}", &parsed1.text[..parsed1.text.len().min(100)]);
    println!("[turn1] c_id={:?} r_id={:?} rcid={:?}",
        parsed1.metadata.get(0).and_then(|v| v.as_deref()),
        parsed1.metadata.get(1).and_then(|v| v.as_deref()),
        parsed1.rcid.as_deref());
    println!("[turn1] state_token={:?}", parsed1.state_token.as_deref());

    if parsed1.rcid.is_none() {
        println!("[turn1] no rcid — cannot continue");
        return;
    }
    let c_id = parsed1.metadata.get(0).and_then(|v| v.as_deref()).unwrap_or("");
    let r_id = parsed1.metadata.get(1).and_then(|v| v.as_deref()).unwrap_or("");
    let rcid = parsed1.rcid.as_deref().unwrap_or("");
    let state_token = parsed1.state_token.as_deref().unwrap_or("");

    sleep(2000).await;

    use serde_json::{json, Value};

    // Turn 2a: same f.sid, same at= as turn 1
    let mut inner2a = base_inner("What is my secret number? Just the number.");
    inner2a[45] = Value::Null;
    inner2a[2] = json!([c_id, r_id, rcid, Value::Null, Value::Null, Value::Null, Value::Null, Value::Null, Value::Null, state_token]);
    let body2a = raw_post_with(&http, serde_json::Value::Array(inner2a.clone()), &tokens1).await.unwrap();
    let (text2a, _) = parse_response_body(&body2a);
    println!("[turn2a same_fsid+at] text={:?}", &text2a[..text2a.len().min(100)]);
    println!("  body[:200]: {:?}", &body2a[..body2a.len().min(200)]);

    sleep(2000).await;

    // Turn 2b: same f.sid but NO at= (mimicking HAR which never sends at=)
    let tokens_no_at = SessionTokens { at: String::new(), bl: tokens1.bl.clone(), fsid: tokens1.fsid.clone() };
    let body2b = raw_post_with(&http, serde_json::Value::Array(inner2a), &tokens_no_at).await.unwrap();
    let (text2b, _) = parse_response_body(&body2b);
    println!("[turn2b same_fsid_no_at] text={:?}", &text2b[..text2b.len().min(100)]);
    println!("  body[:200]: {:?}", &body2b[..body2b.len().min(200)]);

    if text2a.contains("99") {
        println!("  ✓ STATEFUL THREADING WORKS (same at=)");
    } else if text2b.contains("99") {
        println!("  ✓ STATEFUL THREADING WORKS (no at=) — matches HAR behavior");
    } else {
        println!("  ✗ not recalled — 1097 may require same cookie_jar or botguard");
    }
}

// ── 5. Two-shot via provider API with checkpoint threading ───────────────────
//
// Use the actual GeminiProvider with a mock checkpoint that injects
// c_id/r_id from a first real call.

#[tokio::test]
async fn probe_provider_stateful_twoturn() {
    if skip_if_no_creds() { println!("[skip]"); return; }
    sleep(3000).await;

    let p = gravity_providers::gemini::GeminiProvider::new();
    let acct = cred_account();
    let http = HttpClient::new();
    let env = Env::default();

    // Turn 1: establish secret
    let req1 = ChatRequest::new(
        ModelId::new(ProviderName::Gemini, "gemini-3-flash-plus").unwrap(),
        vec![Message::user("My lucky number is 88. Just say NOTED.")],
    );
    let conv = Conversation::new("stateful-test");

    use gravity_providers::Provider;
    let ctx1 = gravity_providers::ChatCtx {
        account: &acct, secret: "", conversation: &conv,
        checkpoint: None, request: &req1, http: &http, env: &env,
    };
    let resp1 = p.chat(&ctx1).await.unwrap();
    println!("[turn1] text={:?}", resp1.text());

    let c_id = resp1.metadata.get("gemini_c_id").and_then(|v| v.as_str()).unwrap_or("").to_owned();
    let r_id = resp1.metadata.get("gemini_r_id").and_then(|v| v.as_str()).unwrap_or("").to_owned();
    let rcid = resp1.metadata.get("gemini_rcid").and_then(|v| v.as_str()).unwrap_or("").to_owned();
    let state_token = resp1.metadata.get("gemini_state_token").and_then(|v| v.as_str()).unwrap_or("").to_owned();
    println!("[turn1] c_id={:?} r_id={:?} rcid={:?}", c_id, r_id, rcid);
    println!("[turn1] state_token={:?}", state_token);

    if c_id.is_empty() {
        println!("[threading] no c_id returned — skipping turn 2");
        return;
    }

    sleep(2000).await;

    // Turn 2: inject checkpoint metadata with c_id/r_id
    let mut chk_meta = serde_json::Map::new();
    chk_meta.insert("gemini_c_id".into(), serde_json::json!(c_id));
    chk_meta.insert("gemini_r_id".into(), serde_json::json!(r_id));
    chk_meta.insert("gemini_rcid".into(), serde_json::json!(rcid));
    chk_meta.insert("gemini_state_token".into(), serde_json::json!(state_token));
    let checkpoint = Checkpoint {
        conversation_key: "stateful-test".into(),
        account_id: "gemini-probe".into(),
        provider: ProviderName::Gemini,
        remote_conversation_id: Some(c_id.into_boxed_str()),
        remote_parent_id: None,
        remote_metadata: Metadata::from(chk_meta),
        history_version: 0,
        updated_at: chrono::Utc::now(),
        is_valid: true,
    };

    let req2 = ChatRequest::new(
        ModelId::new(ProviderName::Gemini, "gemini-3-flash-plus").unwrap(),
        vec![Message::user("What is my lucky number? Just the number.")],
    );
    let ctx2 = gravity_providers::ChatCtx {
        account: &acct, secret: "", conversation: &conv,
        checkpoint: Some(&checkpoint), request: &req2, http: &http, env: &env,
    };
    let resp2 = p.chat(&ctx2).await.unwrap();
    println!("[turn2] text={:?}", resp2.text());
    if resp2.text().contains("88") {
        println!("  ✓ STATEFUL THREADING VIA PROVIDER WORKS — recalled 88");
    } else {
        println!("  ✗ did not recall 88");
    }
}
