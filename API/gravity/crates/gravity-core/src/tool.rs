//! Tool definitions, calls, outputs, and results.
//!
//! Mirrors the tool model in `gravity/types.py` and `gravity/tools.py`.

use crate::metadata::Metadata;
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};

/// A tool the model may call: name, description, and a JSON-Schema for inputs.
///
/// Mirrors `ToolDefinition` in `gravity/tools.py`. The executable handler is
/// kept separate (see `gravity_client::ToolHandler`) so this type stays a pure,
/// serializable description that can cross the FFI boundary.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ToolDefinition {
    /// Unique tool name.
    pub name: Box<str>,
    /// Human/model-facing description.
    #[serde(default)]
    pub description: Box<str>,
    /// JSON-Schema object describing the accepted arguments.
    pub input_schema: Map<String, Value>,
}

/// A single tool invocation emitted by the model.
///
/// Mirrors `gravity.types.ToolCall`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ToolCall {
    /// Provider-assigned call id (correlates the later [`ToolResult`]).
    pub id: Box<str>,
    /// The tool name to invoke.
    pub name: Box<str>,
    /// Parsed argument object.
    #[serde(default)]
    pub arguments: Map<String, Value>,
}

/// Normalized output of a tool handler.
///
/// Mirrors `gravity.types.ToolOutput`, including the error-detection rules of
/// `from_handler_return`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ToolOutput {
    /// Primary result value.
    pub content: Value,
    /// True when the tool failed.
    #[serde(default)]
    pub is_error: bool,
    /// Machine-readable extract when `content` is an object.
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub structured: Option<Value>,
    /// Handler/provider metadata (timing, source, …).
    #[serde(skip_serializing_if = "Metadata::is_empty", default)]
    pub metadata: Metadata,
}

impl ToolOutput {
    /// A successful output; objects are mirrored into `structured`.
    pub fn ok(content: Value) -> Self {
        let structured = content.is_object().then(|| content.clone());
        ToolOutput {
            content,
            is_error: false,
            structured,
            metadata: Metadata::new(),
        }
    }

    /// An error output carrying `{"error": message}`.
    pub fn error(message: impl Into<String>) -> Self {
        let mut obj = Map::new();
        obj.insert("error".into(), Value::String(message.into()));
        let structured = Value::Object(obj);
        ToolOutput {
            content: structured.clone(),
            is_error: true,
            structured: Some(structured),
            metadata: Metadata::new(),
        }
    }

    /// Normalize any handler return value, applying the Python error heuristics:
    /// an object with an `"error"` key, or `ok == false`, or `success == false`,
    /// is treated as an error.
    pub fn from_handler_return(value: Value) -> Self {
        match &value {
            Value::Object(map) => {
                let is_err = map.contains_key("error")
                    || map.get("ok") == Some(&Value::Bool(false))
                    || map.get("success") == Some(&Value::Bool(false));
                ToolOutput {
                    structured: Some(value.clone()),
                    content: value,
                    is_error: is_err,
                    metadata: Metadata::new(),
                }
            }
            _ => ToolOutput {
                content: value,
                is_error: false,
                structured: None,
                metadata: Metadata::new(),
            },
        }
    }

    /// Serialize `content` into the string injected back into the model context.
    ///
    /// Objects and arrays are JSON-encoded; scalars are stringified — matching
    /// `ToolOutput.to_wire` in Python.
    pub fn to_wire(&self) -> String {
        match &self.content {
            Value::Object(_) | Value::Array(_) => self.content.to_string(),
            Value::String(s) => s.clone(),
            Value::Null => "null".to_owned(),
            other => other.to_string(),
        }
    }
}

/// A fully resolved tool call, surfaced on `ChatResponse.tool_calls_executed`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ExecutedToolCall {
    /// The original call.
    pub call: ToolCall,
    /// The handler output.
    pub output: ToolOutput,
    /// Wall-clock handler duration, if measured.
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub duration_ms: Option<f64>,
}

/// A tool result carried in a `role = "tool"` message.
///
/// Mirrors `gravity.types.ToolResult`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ToolResult {
    /// Correlates with [`ToolCall::id`].
    pub tool_call_id: Box<str>,
    /// The tool name.
    pub name: Box<str>,
    /// Result content.
    pub content: Value,
    /// True when the tool failed.
    #[serde(default)]
    pub is_error: bool,
    /// Structured extract.
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub structured_content: Option<Value>,
}

impl ToolResult {
    /// Build from a call and its output (Python `ToolResult.from_output`).
    pub fn from_output(call: &ToolCall, output: ToolOutput) -> Self {
        ToolResult {
            tool_call_id: call.id.clone(),
            name: call.name.clone(),
            content: output.content,
            is_error: output.is_error,
            structured_content: output.structured,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn from_handler_detects_errors() {
        assert!(ToolOutput::from_handler_return(json!({"error": "x"})).is_error);
        assert!(ToolOutput::from_handler_return(json!({"ok": false})).is_error);
        assert!(ToolOutput::from_handler_return(json!({"success": false})).is_error);
        assert!(!ToolOutput::from_handler_return(json!({"temp": 22})).is_error);
        assert!(!ToolOutput::from_handler_return(json!("plain")).is_error);
    }

    #[test]
    fn to_wire_matches_python_rules() {
        assert_eq!(ToolOutput::ok(json!({"a": 1})).to_wire(), r#"{"a":1}"#);
        assert_eq!(ToolOutput::ok(json!("hello")).to_wire(), "hello");
        assert_eq!(ToolOutput::ok(json!(42)).to_wire(), "42");
    }

    #[test]
    fn result_from_output_copies_ids() {
        let call = ToolCall {
            id: "c1".into(),
            name: "get_weather".into(),
            arguments: Map::new(),
        };
        let r = ToolResult::from_output(&call, ToolOutput::ok(json!({"t": 1})));
        assert_eq!(&*r.tool_call_id, "c1");
        assert_eq!(&*r.name, "get_weather");
        assert!(!r.is_error);
    }
}
