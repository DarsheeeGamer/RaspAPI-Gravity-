//! # gravity_rs — native Python bindings (PyO3)
//!
//! A typed Python surface over the pure-Rust Gravity core, replacing the old
//! `_ffi_bridge.py`. Exposes `version()`, `list_providers()`, `chat()`, and
//! `chat_stream()` (returns a Python iterator).
//!
//! ```python
//! import gravity_rs
//!
//! # Blocking chat
//! out = gravity_rs.chat(
//!     provider="groq", model="llama-3.3-70b-versatile",
//!     messages=[{"role": "user", "content": "Hello"}],
//!     api_key="...",
//! )
//! print(out["content"])
//!
//! # Streaming
//! for chunk in gravity_rs.chat_stream(
//!     provider="groq", model="llama-3.3-70b-versatile",
//!     messages=[{"role": "user", "content": "Hello"}],
//!     api_key="...",
//! ):
//!     print(chunk, end="", flush=True)
//! ```

use futures::StreamExt;
use gravity_client::{build_transient, AsyncGravityClient, TransientOptions};
use gravity_core::{
    ChatRequest, Content, Conversation, Message, Metadata, ModelId, ProviderName, Role, Sampling,
    StreamEvent, ToolDefinition,
};
use pyo3::exceptions::{PyRuntimeError, PyStopIteration, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};
use serde_json::{Map, Value};
use std::str::FromStr;
use std::sync::{Mutex, OnceLock};

/// Process-wide tokio runtime + async client.
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

// ── Python iterator for streaming ──────────────────────────────────────────

/// Lazy streaming iterator returned by `chat_stream()`.
///
/// Each `next()` call yields the next text delta as a `str`, or raises
/// `StopIteration` when the stream finishes.
#[pyclass]
pub struct GravityStream {
    rx: Mutex<std::sync::mpsc::Receiver<Result<String, String>>>,
}

#[pymethods]
impl GravityStream {
    fn __iter__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> {
        slf
    }

    fn __next__(&self, py: Python<'_>) -> PyResult<String> {
        let rx = self.rx.lock().unwrap();
        match rx.recv() {
            Ok(Ok(text)) => Ok(text),
            Ok(Err(e)) => Err(PyRuntimeError::new_err(e)),
            Err(_) => Err(PyStopIteration::new_err(py.None())),
        }
    }
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

/// Parse a list of message dicts into [`Message`]s.
fn parse_messages_py(messages: &[Bound<'_, PyDict>]) -> PyResult<Vec<Message>> {
    let mut msgs = Vec::with_capacity(messages.len());
    for m in messages {
        let role = m
            .get_item("role")?
            .and_then(|v| v.extract::<String>().ok())
            .unwrap_or_else(|| "user".to_string());
        let content = m
            .get_item("content")?
            .and_then(|v| v.extract::<String>().ok())
            .unwrap_or_default();
        msgs.push(Message {
            role: parse_role(&role),
            content: Some(Content::Text(content)),
            name: None,
            tool_calls: Default::default(),
            tool_result: None,
            metadata: Metadata::new(),
        });
    }
    Ok(msgs)
}

/// Parse a list of tool dicts into [`ToolDefinition`]s.
fn parse_tools_py(tools: Option<Vec<Bound<'_, PyDict>>>) -> PyResult<Vec<ToolDefinition>> {
    let Some(tools) = tools else { return Ok(Vec::new()) };
    let mut out = Vec::with_capacity(tools.len());
    for t in &tools {
        let name = t
            .get_item("name")?
            .and_then(|v| v.extract::<String>().ok())
            .ok_or_else(|| PyValueError::new_err("tool missing 'name'"))?;
        let description = t
            .get_item("description")?
            .and_then(|v| v.extract::<String>().ok())
            .unwrap_or_default();
        let schema_str = t
            .get_item("input_schema")?
            .and_then(|v| v.extract::<String>().ok())
            .unwrap_or_else(|| r#"{"type":"object"}"#.to_owned());
        let input_schema: Map<String, Value> = serde_json::from_str(&schema_str)
            .map_err(|e| PyValueError::new_err(format!("invalid tool schema: {e}")))?;
        out.push(ToolDefinition { name: name.into(), description: description.into(), input_schema });
    }
    Ok(out)
}

/// Build a [`ChatRequest`] from Python arguments.
#[allow(clippy::too_many_arguments)]
fn build_request_py(
    provider: &str,
    model: &str,
    messages: Vec<Bound<'_, PyDict>>,
    system: Option<String>,
    temperature: Option<f32>,
    max_tokens: Option<u32>,
    tools: Option<Vec<Bound<'_, PyDict>>>,
    seed: Option<i64>,
    frequency_penalty: Option<f32>,
    presence_penalty: Option<f32>,
) -> PyResult<(ModelId, ProviderName, ChatRequest)> {
    let pname = ProviderName::from_str(provider)
        .map_err(|_| PyValueError::new_err(format!("unknown provider: {provider}")))?;
    let model_id = ModelId::new(pname, model).map_err(|e| PyValueError::new_err(e.to_string()))?;
    let msgs = parse_messages_py(&messages)?;
    let tool_defs = parse_tools_py(tools)?;
    let mut req = ChatRequest::new(model_id.clone(), msgs);
    req.system_prompt = system;
    req.tools = tool_defs;
    req.sampling = Sampling {
        temperature,
        max_tokens,
        top_p: None,
        stop: Vec::new(),
        seed,
        frequency_penalty,
        presence_penalty,
    };
    Ok((model_id, pname, req))
}

// ── Public Python functions ─────────────────────────────────────────────────

/// Return the library version string.
#[pyfunction]
fn version() -> &'static str {
    "gravity/2.0.0"
}

/// List registered providers as a list of dicts.
#[pyfunction]
fn list_providers(py: Python<'_>) -> PyResult<Py<PyList>> {
    let list = PyList::empty(py);
    for (name, caps) in gravity_providers::builtin().iter_capabilities() {
        let d = PyDict::new(py);
        d.set_item("name", name.as_str())?;
        d.set_item("tool_support", format!("{:?}", caps.tool_support).to_lowercase())?;
        d.set_item("supports_system_prompt", caps.supports_system_prompt)?;
        list.append(d)?;
    }
    Ok(list.unbind())
}

/// Run a blocking chat and return a response dict.
///
/// The dict contains: `content`, `model`, `finish_reason`, `input_tokens`,
/// `output_tokens`, and `tool_calls` (list of `{id, name, arguments}` dicts).
#[pyfunction]
#[pyo3(signature = (
    provider, model, messages, api_key = "",
    system = None, temperature = None, max_tokens = None,
    tools = None, seed = None, frequency_penalty = None, presence_penalty = None,
))]
#[allow(clippy::too_many_arguments)]
fn chat(
    py: Python<'_>,
    provider: &str,
    model: &str,
    messages: Vec<Bound<'_, PyDict>>,
    api_key: &str,
    system: Option<String>,
    temperature: Option<f32>,
    max_tokens: Option<u32>,
    tools: Option<Vec<Bound<'_, PyDict>>>,
    seed: Option<i64>,
    frequency_penalty: Option<f32>,
    presence_penalty: Option<f32>,
) -> PyResult<Py<PyDict>> {
    let (_, pname, req) = build_request_py(
        provider, model, messages, system, temperature, max_tokens, tools,
        seed, frequency_penalty, presence_penalty,
    )?;
    let kind = if matches!(pname, ProviderName::Claude | ProviderName::Chatgpt) {
        Some("session_cookie".to_owned())
    } else {
        Some("api_key".to_owned())
    };
    let account = build_transient(
        gravity_providers::builtin(),
        pname,
        TransientOptions { secret: api_key.to_owned(), kind, ..Default::default() },
    )
    .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;

    let eng = engine();
    let conv = Conversation::new("");
    let result = py.detach(|| {
        eng.runtime.block_on(eng.client.chat_over(&req, std::slice::from_ref(&account), &conv))
    });
    let resp = result.map_err(|e| PyRuntimeError::new_err(e.to_string()))?;

    let out = PyDict::new(py);
    out.set_item("content", resp.text())?;
    out.set_item("model", &*resp.model)?;
    out.set_item("finish_reason", resp.finish_reason.as_deref())?;
    out.set_item("input_tokens", resp.usage.input_tokens)?;
    out.set_item("output_tokens", resp.usage.output_tokens)?;
    let tcs = PyList::empty(py);
    for tc in &resp.message.tool_calls {
        let d = PyDict::new(py);
        d.set_item("id", &*tc.id)?;
        d.set_item("name", &*tc.name)?;
        d.set_item("arguments", Value::Object(tc.arguments.clone()).to_string())?;
        tcs.append(d)?;
    }
    out.set_item("tool_calls", tcs)?;
    Ok(out.unbind())
}

/// Start a streaming chat, returning a Python iterator that yields text deltas.
///
/// ```python
/// for chunk in gravity_rs.chat_stream(
///     provider="groq", model="llama-3.3-70b-versatile",
///     messages=[{"role": "user", "content": "Hello"}],
///     api_key="sk-...",
/// ):
///     print(chunk, end="", flush=True)
/// ```
#[pyfunction]
#[pyo3(signature = (
    provider, model, messages, api_key = "",
    system = None, temperature = None, max_tokens = None,
    tools = None, seed = None, frequency_penalty = None, presence_penalty = None,
))]
#[allow(clippy::too_many_arguments)]
fn chat_stream(
    py: Python<'_>,
    provider: &str,
    model: &str,
    messages: Vec<Bound<'_, PyDict>>,
    api_key: &str,
    system: Option<String>,
    temperature: Option<f32>,
    max_tokens: Option<u32>,
    tools: Option<Vec<Bound<'_, PyDict>>>,
    seed: Option<i64>,
    frequency_penalty: Option<f32>,
    presence_penalty: Option<f32>,
) -> PyResult<Py<GravityStream>> {
    let (_, pname, mut req) = build_request_py(
        provider, model, messages, system, temperature, max_tokens, tools,
        seed, frequency_penalty, presence_penalty,
    )?;
    req.stream = true;
    let kind = if matches!(pname, ProviderName::Claude | ProviderName::Chatgpt) {
        Some("session_cookie".to_owned())
    } else {
        Some("api_key".to_owned())
    };
    let account = build_transient(
        gravity_providers::builtin(),
        pname,
        TransientOptions { secret: api_key.to_owned(), kind, ..Default::default() },
    )
    .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;

    let (tx, rx) = std::sync::mpsc::channel::<Result<String, String>>();
    let eng = engine();
    let conv = Conversation::new("");

    eng.runtime.spawn(async move {
        match eng.client.chat_stream_over(&req, &account, &conv).await {
            Err(e) => { let _ = tx.send(Err(e.to_string())); }
            Ok(mut stream) => {
                while let Some(event) = stream.next().await {
                    match event {
                        Ok(StreamEvent::TextDelta(bytes)) => {
                            let text = String::from_utf8_lossy(&bytes).into_owned();
                            if !text.is_empty() && tx.send(Ok(text)).is_err() {
                                break;
                            }
                        }
                        Ok(StreamEvent::Done(resp)) => {
                            // Emit any trailing text from the Done response.
                            let t = resp.text();
                            if !t.is_empty() {
                                let _ = tx.send(Ok(t.to_owned()));
                            }
                            break;
                        }
                        Err(e) => { let _ = tx.send(Err(e.to_string())); break; }
                        _ => {}
                    }
                }
                // Channel drop signals StopIteration.
            }
        }
    });

    let stream = GravityStream { rx: Mutex::new(rx) };
    Py::new(py, stream).map_err(|e| PyRuntimeError::new_err(e.to_string()))
}

/// The Python module definition.
#[pymodule]
fn gravity_rs(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<GravityStream>()?;
    m.add_function(wrap_pyfunction!(version, m)?)?;
    m.add_function(wrap_pyfunction!(list_providers, m)?)?;
    m.add_function(wrap_pyfunction!(chat, m)?)?;
    m.add_function(wrap_pyfunction!(chat_stream, m)?)?;
    Ok(())
}
