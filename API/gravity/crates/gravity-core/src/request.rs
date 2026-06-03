//! The canonical chat request — the heart of the Gravity Schema.

use crate::capabilities::HistoryMode;
use crate::message::Message;
use crate::metadata::Metadata;
use crate::model_id::ModelId;
use crate::tool::ToolDefinition;
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};

/// Tool-execution mode.
///
/// Mirrors the `function_calling` string field in `gravity/abcs.py`.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum FunctionCalling {
    /// AFC — Gravity executes tool calls and returns final text.
    #[default]
    Auto,
    /// MFC — raw tool calls are returned for the caller to execute.
    Manual,
}

/// How structured output is requested.
///
/// Replaces the Python `response_model` / `response_json_schema` either-or with
/// a single typed field that cannot represent the conflicting state.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(untagged)]
pub enum ResponseFormat {
    /// No structured-output constraint.
    #[default]
    Free,
    /// Constrain output to the given JSON Schema.
    JsonSchema {
        /// The JSON Schema object.
        json_schema: Map<String, Value>,
    },
}

impl ResponseFormat {
    /// True when a schema constraint is present.
    pub fn is_constrained(&self) -> bool {
        matches!(self, ResponseFormat::JsonSchema { .. })
    }
}

/// Generation tuning knobs, all optional so an empty sampling config
/// serializes to nothing.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct Sampling {
    /// Nucleus temperature.
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub temperature: Option<f32>,
    /// Maximum output tokens.
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub max_tokens: Option<u32>,
    /// Top-p.
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub top_p: Option<f32>,
    /// Stop sequences.
    #[serde(skip_serializing_if = "Vec::is_empty", default)]
    pub stop: Vec<String>,
    /// Deterministic sampling seed.
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub seed: Option<i64>,
    /// Frequency penalty.
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub frequency_penalty: Option<f32>,
    /// Presence penalty.
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub presence_penalty: Option<f32>,
}

impl Sampling {
    /// True when no sampling override is set.
    pub fn is_empty(&self) -> bool {
        self.temperature.is_none()
            && self.max_tokens.is_none()
            && self.top_p.is_none()
            && self.stop.is_empty()
            && self.seed.is_none()
            && self.frequency_penalty.is_none()
            && self.presence_penalty.is_none()
    }
}

/// A provider-agnostic chat request.
///
/// Mirrors `ChatRequest` in `gravity/abcs.py`. A [`crate::SchemaConverter`]
/// turns this into each provider's wire format.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ChatRequest {
    /// Target `provider/model`.
    pub model: ModelId,
    /// The conversation messages to send.
    pub messages: Vec<Message>,
    /// Optional system prompt (separate from `messages`).
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub system_prompt: Option<String>,
    /// Tools advertised to the model.
    #[serde(skip_serializing_if = "Vec::is_empty", default)]
    pub tools: Vec<ToolDefinition>,
    /// Tool-execution mode.
    #[serde(default)]
    pub function_calling: FunctionCalling,
    /// Structured-output constraint.
    #[serde(default, skip_serializing_if = "ResponseFormat::is_free")]
    pub response_format: ResponseFormat,
    /// Per-request history-mode override.
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub history_mode: Option<HistoryMode>,
    /// Whether the caller wants a streamed response.
    #[serde(default)]
    pub stream: bool,
    /// Sampling knobs.
    #[serde(default, skip_serializing_if = "Sampling::is_empty")]
    pub sampling: Sampling,
    /// Free-form request metadata.
    #[serde(default, skip_serializing_if = "Metadata::is_empty")]
    pub metadata: Metadata,
}

impl ResponseFormat {
    fn is_free(&self) -> bool {
        matches!(self, ResponseFormat::Free)
    }
}

impl ChatRequest {
    /// Build a minimal request for `model` carrying `messages`.
    pub fn new(model: ModelId, messages: Vec<Message>) -> Self {
        ChatRequest {
            model,
            messages,
            system_prompt: None,
            tools: Vec::new(),
            function_calling: FunctionCalling::Auto,
            response_format: ResponseFormat::Free,
            history_mode: None,
            stream: false,
            sampling: Sampling::default(),
            metadata: Metadata::new(),
        }
    }

    /// Validate structural invariants (delegates to [`Message::validate`]).
    pub fn validate(&self) -> Result<(), &'static str> {
        for m in &self.messages {
            m.validate()?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn minimal_request_omits_empty_fields() {
        let req = ChatRequest::new(
            ModelId::parse("groq/llama-3.3-70b").unwrap(),
            vec![Message::user("hi")],
        );
        let json = serde_json::to_value(&req).unwrap();
        assert_eq!(json["function_calling"], "auto");
        assert!(json.get("tools").is_none());
        assert!(json.get("sampling").is_none());
        assert!(json.get("response_format").is_none());
    }

    #[test]
    fn response_format_is_exclusive_by_construction() {
        let rf = ResponseFormat::JsonSchema {
            json_schema: serde_json::from_str(r#"{"type":"object"}"#).unwrap(),
        };
        assert!(rf.is_constrained());
    }
}
