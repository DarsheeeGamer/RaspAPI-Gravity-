//! OpenAI-compatible request handlers.

use crate::AppState;
use axum::extract::State;
use axum::http::{HeaderMap, StatusCode};
use axum::response::sse::{Event, Sse};
use axum::response::{IntoResponse, Response};
use axum::Json;
use gravity_client::{build_transient, ResolvedAccount, TransientOptions};
use gravity_core::{
    ChatRequest, Content, Conversation, Error, Message, Metadata, ModelId, ProviderName, Role,
    Sampling, StreamEvent,
};
use serde_json::{json, Value};
use std::convert::Infallible;
use std::sync::Arc;

/// `GET /health`.
pub async fn health() -> Json<Value> {
    Json(json!({ "status": "ok" }))
}

/// `GET /providers`.
pub async fn providers() -> Json<Value> {
    let reg = gravity_providers::builtin();
    let arr: Vec<Value> = reg
        .iter_capabilities()
        .map(|(name, caps)| {
            json!({
                "name": name.as_str(),
                "tool_support": format!("{:?}", caps.tool_support).to_lowercase(),
                "supports_system_prompt": caps.supports_system_prompt,
                "models": reg.provider(name).map(|p| p.list_models()).unwrap_or_default(),
            })
        })
        .collect();
    Json(Value::Array(arr))
}

/// `GET /v1/models` — OpenAI list shape.
pub async fn models() -> Json<Value> {
    let reg = gravity_providers::builtin();
    let mut data = Vec::new();
    for (name, _) in reg.iter_capabilities() {
        if let Ok(p) = reg.provider(name) {
            for m in p.list_models() {
                data.push(json!({
                    "id": format!("{}/{}", name.as_str(), m),
                    "object": "model",
                    "owned_by": name.as_str(),
                }));
            }
        }
    }
    Json(json!({ "object": "list", "data": data }))
}

/// `POST /v1/chat/completions`.
pub async fn chat_completions(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
    Json(body): Json<Value>,
) -> Response {
    match prepare(&headers, &body) {
        Ok((account, request, stream)) => {
            let conv = Conversation::new("");
            if stream {
                stream_response(state, account, request, conv).await
            } else {
                buffered_response(state, account, request, conv).await
            }
        }
        Err(e) => error_response(&e),
    }
}

/// Parse the request into a transient account + canonical request.
fn prepare(headers: &HeaderMap, body: &Value) -> Result<(ResolvedAccount, ChatRequest, bool), Error> {
    let model_str = body
        .get("model")
        .and_then(Value::as_str)
        .ok_or_else(|| Error::validation("missing 'model'", "missing_model"))?;
    let model = ModelId::parse(model_str)
        .map_err(|e| Error::validation(e.to_string(), "bad_model"))?;
    let provider = model.provider;

    let secret = bearer_token(headers)
        .or_else(|| body.get("api_key").and_then(Value::as_str).map(str::to_owned))
        .unwrap_or_default();

    let kind = if matches!(provider, ProviderName::Claude | ProviderName::Chatgpt) {
        Some("session_cookie".to_owned())
    } else {
        Some("api_key".to_owned())
    };
    let account = build_transient(
        gravity_providers::builtin(),
        provider,
        TransientOptions { secret: secret.clone(), kind, ..Default::default() },
    )?;

    let messages = parse_messages(body.get("messages"))?;
    let mut request = ChatRequest::new(model, messages);
    request.system_prompt = body.get("system").and_then(Value::as_str).map(str::to_owned);
    request.sampling = Sampling {
        temperature: body.get("temperature").and_then(Value::as_f64).map(|n| n as f32),
        max_tokens: body.get("max_tokens").and_then(Value::as_u64).map(|n| n as u32),
        top_p: body.get("top_p").and_then(Value::as_f64).map(|n| n as f32),
        stop: Vec::new(),
        seed: body.get("seed").and_then(Value::as_i64),
        frequency_penalty: body.get("frequency_penalty").and_then(Value::as_f64).map(|n| n as f32),
        presence_penalty: body.get("presence_penalty").and_then(Value::as_f64).map(|n| n as f32),
    };
    let stream = body.get("stream").and_then(Value::as_bool).unwrap_or(false);
    request.stream = stream;
    Ok((account, request, stream))
}

fn bearer_token(headers: &HeaderMap) -> Option<String> {
    headers
        .get("authorization")
        .and_then(|v| v.to_str().ok())
        .and_then(|s| s.strip_prefix("Bearer "))
        .map(str::to_owned)
}

fn parse_messages(value: Option<&Value>) -> Result<Vec<Message>, Error> {
    let arr = value
        .and_then(Value::as_array)
        .ok_or_else(|| Error::validation("missing 'messages'", "missing_messages"))?;
    Ok(arr
        .iter()
        .map(|m| {
            let role = match m.get("role").and_then(Value::as_str).unwrap_or("user") {
                "system" => Role::System,
                "assistant" => Role::Assistant,
                "tool" => Role::Tool,
                _ => Role::User,
            };
            let content = m.get("content").and_then(Value::as_str).unwrap_or("").to_owned();
            Message {
                role,
                content: Some(Content::Text(content)),
                name: None,
                tool_calls: Default::default(),
                tool_result: None,
                metadata: Metadata::new(),
            }
        })
        .collect())
}

async fn buffered_response(
    state: Arc<AppState>,
    account: ResolvedAccount,
    request: ChatRequest,
    conv: Conversation,
) -> Response {
    match state.client.chat_over(&request, std::slice::from_ref(&account), &conv).await {
        Ok(resp) => Json(openai_completion(&resp)).into_response(),
        Err(e) => error_response(&e),
    }
}

async fn stream_response(
    state: Arc<AppState>,
    account: ResolvedAccount,
    request: ChatRequest,
    conv: Conversation,
) -> Response {
    let model = request.model.full_name();
    let stream = match state.client.chat_stream_over(&request, &account, &conv).await {
        Ok(s) => s,
        Err(e) => return error_response(&e),
    };
    use futures::StreamExt;
    // Track whether any incremental delta was emitted; if a provider only emits
    // a terminal Done (non-incremental), surface its full text as one chunk.
    let emitted = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
    let sse = stream.map(move |ev| {
        let chunk_of = |content: std::borrow::Cow<'_, str>| {
            json!({
                "object": "chat.completion.chunk",
                "model": model,
                "choices": [{ "index": 0, "delta": { "content": content } }],
            })
            .to_string()
        };
        let event = match ev {
            Ok(StreamEvent::TextDelta(bytes)) => {
                emitted.store(true, std::sync::atomic::Ordering::SeqCst);
                Event::default().data(chunk_of(String::from_utf8_lossy(&bytes)))
            }
            Ok(StreamEvent::Done(resp)) => {
                if emitted.load(std::sync::atomic::Ordering::SeqCst) {
                    Event::default().data("[DONE]")
                } else {
                    Event::default().data(chunk_of(std::borrow::Cow::Owned(resp.text().to_owned())))
                }
            }
            Ok(_) => Event::default().comment("keepalive"),
            Err(e) => Event::default().data(json!({ "error": e.to_string() }).to_string()),
        };
        Ok::<_, Infallible>(event)
    });
    Sse::new(sse).into_response()
}

/// Build the OpenAI `chat.completion` response body.
fn openai_completion(resp: &gravity_core::ChatResponse) -> Value {
    json!({
        "object": "chat.completion",
        "model": resp.model,
        "choices": [{
            "index": 0,
            "message": { "role": "assistant", "content": resp.text() },
            "finish_reason": resp.finish_reason.as_deref().unwrap_or("stop"),
        }],
        "usage": {
            "prompt_tokens": resp.usage.input_tokens.unwrap_or(0),
            "completion_tokens": resp.usage.output_tokens.unwrap_or(0),
            "total_tokens": resp.usage.total(),
        }
    })
}

fn error_response(e: &Error) -> Response {
    let status = match e {
        Error::Validation { .. } | Error::ModelRouting(_) => StatusCode::BAD_REQUEST,
        Error::ProviderNotRegistered { .. } => StatusCode::NOT_FOUND,
        Error::AccountSelection(_) => StatusCode::SERVICE_UNAVAILABLE,
        _ => StatusCode::BAD_GATEWAY,
    };
    (status, Json(json!({ "error": { "message": e.to_string() } }))).into_response()
}
