//! # gravity-node — native Node.js bindings (napi-rs)
//!
//! A typed Node addon over the pure-Rust Gravity core. Exposes `version()`,
//! `listProviders()`, `chat()` (sync), `chatAsync()` (Promise), and
//! `chatStream()` (callback-based streaming).
//!
//! ```js
//! const gravity = require('./gravity.node');
//!
//! // Sync
//! const out = gravity.chat({ provider: 'groq', model: 'llama-3.3-70b-versatile',
//!   messages: [{ role: 'user', content: 'Hello' }], apiKey: '...' });
//! console.log(out.content);
//!
//! // Async (Promise)
//! gravity.chatAsync({ provider: 'groq', model: 'llama-3.3-70b-versatile',
//!   messages: [{ role: 'user', content: 'Hello' }], apiKey: '...' })
//!   .then(out => console.log(out.content));
//!
//! // Streaming
//! gravity.chatStream(
//!   { provider: 'groq', model: 'llama-3.3-70b-versatile',
//!     messages: [{ role: 'user', content: 'Hello' }], apiKey: '...' },
//!   (err, chunk) => { if (chunk) process.stdout.write(chunk); }
//! );
//! ```

use gravity_client::{build_transient, AsyncGravityClient, TransientOptions};
use gravity_core::{
    ChatRequest, Content, Conversation, Message, Metadata, ModelId, ProviderName, Role, Sampling,
    StreamEvent, ToolDefinition,
};
use napi::bindgen_prelude::*;
use napi::threadsafe_function::{ThreadSafeCallContext, ThreadsafeFunction, ThreadsafeFunctionCallMode};
use napi::{JsFunction, JsObject};
use napi_derive::napi;
use serde_json::{Map, Value};
use std::str::FromStr;
use std::sync::OnceLock;

use futures::StreamExt;

struct Engine {
    runtime: tokio::runtime::Runtime,
    client: AsyncGravityClient,
}

fn engine() -> &'static Engine {
    static ENGINE: OnceLock<Engine> = OnceLock::new();
    ENGINE.get_or_init(|| Engine {
        runtime: tokio::runtime::Builder::new_multi_thread()
            .enable_all()
            .build()
            .expect("failed to build runtime"),
        client: AsyncGravityClient::new(),
    })
}

// ── Public types ─────────────────────────────────────────────────────────────

/// Library version string.
#[napi]
pub fn version() -> String {
    "gravity/2.0.0".to_string()
}

/// A provider description.
#[napi(object)]
pub struct ProviderInfo {
    /// Provider identifier.
    pub name: String,
    /// Tool-support mode.
    pub tool_support: String,
    /// Whether a system prompt is honored.
    pub supports_system_prompt: bool,
}

/// List all registered providers.
#[napi]
pub fn list_providers() -> Vec<ProviderInfo> {
    gravity_providers::builtin()
        .iter_capabilities()
        .map(|(name, caps)| ProviderInfo {
            name: name.as_str().to_string(),
            tool_support: format!("{:?}", caps.tool_support).to_lowercase(),
            supports_system_prompt: caps.supports_system_prompt,
        })
        .collect()
}

/// A single chat message.
#[napi(object)]
pub struct ChatMessage {
    /// Role (`system`/`user`/`assistant`/`tool`).
    pub role: String,
    /// Text content.
    pub content: String,
}

/// A tool definition to advertise to the model.
#[napi(object)]
pub struct ToolInput {
    /// Function name.
    pub name: String,
    /// Function description.
    pub description: String,
    /// JSON-string of the JSON Schema parameters object.
    pub input_schema: String,
}

/// Options for [`chat`] / [`chat_async`] / [`chat_stream`].
#[napi(object)]
pub struct ChatOptions {
    /// Provider name.
    pub provider: String,
    /// Bare model name.
    pub model: String,
    /// Conversation messages.
    pub messages: Vec<ChatMessage>,
    /// API key / session secret.
    pub api_key: Option<String>,
    /// Optional system prompt.
    pub system: Option<String>,
    /// Sampling temperature.
    pub temperature: Option<f64>,
    /// Max output tokens.
    pub max_tokens: Option<u32>,
    /// Deterministic sampling seed.
    pub seed: Option<i64>,
    /// Frequency penalty.
    pub frequency_penalty: Option<f64>,
    /// Presence penalty.
    pub presence_penalty: Option<f64>,
    /// Tools to advertise to the model.
    pub tools: Option<Vec<ToolInput>>,
}

/// A tool call in a response.
#[napi(object)]
pub struct ToolCallResult {
    /// Call id.
    pub id: String,
    /// Function name.
    pub name: String,
    /// JSON-string arguments.
    pub arguments: String,
}

/// The chat result.
#[napi(object)]
pub struct ChatResult {
    /// Assistant text.
    pub content: String,
    /// Resolved model name.
    pub model: String,
    /// Finish reason, if any.
    pub finish_reason: Option<String>,
    /// Prompt tokens.
    pub input_tokens: Option<u32>,
    /// Completion tokens.
    pub output_tokens: Option<u32>,
    /// Tool calls (empty for non-tool responses).
    pub tool_calls: Vec<ToolCallResult>,
}

// ── Helpers ─────────────────────────────────────────────────────────────────

fn parse_role(s: &str) -> Role {
    match s {
        "system" => Role::System,
        "assistant" => Role::Assistant,
        "tool" => Role::Tool,
        _ => Role::User,
    }
}

fn build_request_node(opts: &ChatOptions) -> std::result::Result<(ProviderName, ChatRequest), String> {
    let provider = ProviderName::from_str(&opts.provider)
        .map_err(|_| format!("unknown provider: {}", opts.provider))?;
    let model_id = ModelId::new(provider, opts.model.as_str())
        .map_err(|e| e.to_string())?;
    let msgs: Vec<Message> = opts.messages.iter().map(|m| Message {
        role: parse_role(&m.role),
        content: Some(Content::Text(m.content.clone())),
        name: None,
        tool_calls: Default::default(),
        tool_result: None,
        metadata: Metadata::new(),
    }).collect();
    let tools: Vec<ToolDefinition> = opts.tools.as_deref().unwrap_or(&[]).iter().map(|t| {
        let schema: Map<String, Value> = serde_json::from_str(&t.input_schema)
            .unwrap_or_else(|_| serde_json::from_str(r#"{"type":"object"}"#).unwrap());
        ToolDefinition {
            name: t.name.clone().into(),
            description: t.description.clone().into(),
            input_schema: schema,
        }
    }).collect();
    let mut req = ChatRequest::new(model_id, msgs);
    req.system_prompt = opts.system.clone();
    req.tools = tools;
    req.sampling = Sampling {
        temperature: opts.temperature.map(|t| t as f32),
        max_tokens: opts.max_tokens,
        top_p: None,
        stop: Vec::new(),
        seed: opts.seed,
        frequency_penalty: opts.frequency_penalty.map(|f| f as f32),
        presence_penalty: opts.presence_penalty.map(|p| p as f32),
    };
    Ok((provider, req))
}

fn build_account_node(opts: &ChatOptions, provider: ProviderName)
    -> std::result::Result<gravity_client::ResolvedAccount, String>
{
    let kind = if matches!(provider, ProviderName::Claude | ProviderName::Chatgpt) {
        Some("session_cookie".to_owned())
    } else {
        Some("api_key".to_owned())
    };
    build_transient(
        gravity_providers::builtin(),
        provider,
        TransientOptions {
            secret: opts.api_key.clone().unwrap_or_default(),
            kind,
            ..Default::default()
        },
    )
    .map_err(|e| e.to_string())
}

fn resp_to_node(resp: gravity_core::ChatResponse) -> ChatResult {
    let tool_calls = resp.message.tool_calls.iter().map(|tc| ToolCallResult {
        id: tc.id.to_string(),
        name: tc.name.to_string(),
        arguments: Value::Object(tc.arguments.clone()).to_string(),
    }).collect();
    ChatResult {
        content: resp.text().to_string(),
        model: resp.model.to_string(),
        finish_reason: resp.finish_reason.as_deref().map(str::to_owned),
        input_tokens: resp.usage.input_tokens,
        output_tokens: resp.usage.output_tokens,
        tool_calls,
    }
}

// ── Exported functions ───────────────────────────────────────────────────────

/// Run a synchronous (blocking) chat turn.
#[napi]
pub fn chat(opts: ChatOptions) -> Result<ChatResult> {
    let (provider, req) = build_request_node(&opts).map_err(Error::from_reason)?;
    let account = build_account_node(&opts, provider).map_err(Error::from_reason)?;
    let eng = engine();
    let conv = Conversation::new("");
    let resp = eng.runtime
        .block_on(eng.client.chat_over(&req, std::slice::from_ref(&account), &conv))
        .map_err(|e| Error::from_reason(e.to_string()))?;
    Ok(resp_to_node(resp))
}

/// Run an async chat turn, returning a JS Promise.
#[napi]
pub fn chat_async(opts: ChatOptions) -> AsyncTask<ChatTask> {
    AsyncTask::new(ChatTask { opts })
}

/// Internal task for `chatAsync`.
pub struct ChatTask {
    opts: ChatOptions,
}

impl Task for ChatTask {
    type Output = ChatResult;
    type JsValue = JsObject;

    fn compute(&mut self) -> Result<Self::Output> {
        let (provider, req) = build_request_node(&self.opts).map_err(Error::from_reason)?;
        let account = build_account_node(&self.opts, provider).map_err(Error::from_reason)?;
        let eng = engine();
        let conv = Conversation::new("");
        let resp = eng.runtime
            .block_on(eng.client.chat_over(&req, std::slice::from_ref(&account), &conv))
            .map_err(|e| Error::from_reason(e.to_string()))?;
        Ok(resp_to_node(resp))
    }

    fn resolve(&mut self, env: napi::Env, output: Self::Output) -> Result<Self::JsValue> {
        let mut obj = env.create_object()?;
        obj.set_named_property("content", env.create_string(&output.content)?)?;
        obj.set_named_property("model", env.create_string(&output.model)?)?;
        let fr = match &output.finish_reason {
            Some(s) => env.create_string(s)?.into_unknown(),
            None => env.get_null()?.into_unknown(),
        };
        obj.set_named_property("finishReason", fr)?;
        let it = match output.input_tokens {
            Some(n) => env.create_uint32(n)?.into_unknown(),
            None => env.get_null()?.into_unknown(),
        };
        obj.set_named_property("inputTokens", it)?;
        let ot = match output.output_tokens {
            Some(n) => env.create_uint32(n)?.into_unknown(),
            None => env.get_null()?.into_unknown(),
        };
        obj.set_named_property("outputTokens", ot)?;
        Ok(obj)
    }
}

/// Stream a chat turn, calling `callback(err, chunk)` for each text delta.
/// When the stream ends, `callback(null, null)` is called.
#[napi]
pub fn chat_stream(opts: ChatOptions, callback: JsFunction) -> Result<()> {
    let (provider, mut req) = build_request_node(&opts).map_err(Error::from_reason)?;
    req.stream = true;
    let account = build_account_node(&opts, provider).map_err(Error::from_reason)?;

    let tsfn: ThreadsafeFunction<Option<String>> = callback
        .create_threadsafe_function(0, |ctx: ThreadSafeCallContext<Option<String>>| {
            match ctx.value {
                Some(chunk) => Ok(vec![ctx.env.create_string(&chunk)?.into_unknown()]),
                None => Ok(vec![ctx.env.get_null()?.into_unknown()]),
            }
        })?;

    let eng = engine();
    let conv = Conversation::new("");
    let tsfn_err = tsfn.clone();

    eng.runtime.spawn(async move {
        match eng.client.chat_stream_over(&req, &account, &conv).await {
            Err(e) => {
                tsfn_err.call(Ok(None), ThreadsafeFunctionCallMode::Blocking);
                drop(e);
            }
            Ok(mut stream) => {
                while let Some(event) = stream.next().await {
                    match event {
                        Ok(StreamEvent::TextDelta(bytes)) => {
                            let text = String::from_utf8_lossy(&bytes).into_owned();
                            if !text.is_empty() {
                                tsfn.call(Ok(Some(text)), ThreadsafeFunctionCallMode::Blocking);
                            }
                        }
                        Ok(StreamEvent::Done(resp)) => {
                            let t = resp.text();
                            if !t.is_empty() {
                                tsfn.call(Ok(Some(t.to_owned())), ThreadsafeFunctionCallMode::Blocking);
                            }
                            break;
                        }
                        Err(_) => break,
                        _ => {}
                    }
                }
                // Signal end of stream.
                tsfn.call(Ok(None), ThreadsafeFunctionCallMode::Blocking);
            }
        }
    });

    Ok(())
}
