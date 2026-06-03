//! Chat responses, token usage, and auxiliary response types.

use crate::message::Message;
use crate::metadata::Metadata;
use crate::tool::{ExecutedToolCall, ToolCall};
use bytes::Bytes;
use serde::{Deserialize, Serialize};

/// Token usage for one inference call.
///
/// Mirrors `gravity.types.Usage`.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct Usage {
    /// Prompt tokens.
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub input_tokens: Option<u32>,
    /// Completion tokens.
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub output_tokens: Option<u32>,
    /// Total, if the provider reports one directly.
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub total_tokens: Option<u32>,
}

impl Usage {
    /// Total tokens, computed from input+output when `total_tokens` is absent.
    pub fn total(&self) -> u32 {
        self.total_tokens
            .unwrap_or_else(|| self.input_tokens.unwrap_or(0) + self.output_tokens.unwrap_or(0))
    }
}

/// Why generation stopped.
///
/// Held as a `Box<str>` rather than an enum so unmapped provider-specific
/// reasons pass through losslessly.
pub type FinishReason = Box<str>;

/// The result of a chat call.
///
/// Mirrors `gravity.types.ChatResponse`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ChatResponse {
    /// The model that produced this response.
    pub model: Box<str>,
    /// The assistant message.
    pub message: Message,
    /// Token usage.
    #[serde(default)]
    pub usage: Usage,
    /// Tool calls resolved during AFC, in execution order.
    #[serde(skip_serializing_if = "Vec::is_empty", default)]
    pub tool_calls_executed: Vec<ExecutedToolCall>,
    /// Provider-side response id, when available.
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub provider_response_id: Option<Box<str>>,
    /// Finish reason.
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub finish_reason: Option<FinishReason>,
    /// Free-form metadata.
    #[serde(skip_serializing_if = "Metadata::is_empty", default)]
    pub metadata: Metadata,
}

impl ChatResponse {
    /// The assistant text (empty for tool-only / multipart turns).
    pub fn text(&self) -> &str {
        self.message.text()
    }

    /// Raw tool calls from the model (populated in MFC mode).
    pub fn tool_calls(&self) -> &[ToolCall] {
        &self.message.tool_calls
    }
}

/// A single embedding response.
///
/// Mirrors `gravity.types.EmbeddingResponse`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EmbeddingResponse {
    /// Input texts, in order.
    pub texts: Vec<String>,
    /// One vector per input.
    pub embeddings: Vec<Vec<f32>>,
    /// Provider identifier.
    #[serde(default)]
    pub provider: Box<str>,
    /// Vector dimensionality.
    #[serde(default)]
    pub dimensions: u32,
    /// Metadata.
    #[serde(skip_serializing_if = "Metadata::is_empty", default)]
    pub metadata: Metadata,
}

/// A generated image.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct GeneratedImage {
    /// Raw image bytes.
    #[serde(with = "crate::serde_bytes_b64")]
    pub data: Bytes,
    /// MIME type.
    #[serde(default = "default_png")]
    pub mime_type: Box<str>,
    /// Provider-revised prompt, if any.
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub revised_prompt: Option<String>,
}

fn default_png() -> Box<str> {
    "image/png".into()
}

/// Response to an image-generation request.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ImageResponse {
    /// The model that produced the images.
    pub model: Box<str>,
    /// The generated images.
    pub images: Vec<GeneratedImage>,
    /// Metadata.
    #[serde(skip_serializing_if = "Metadata::is_empty", default)]
    pub metadata: Metadata,
}

/// Request to generate images.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ImageRequest {
    /// Target `provider/model`.
    pub model: crate::model_id::ModelId,
    /// The prompt.
    pub prompt: String,
    /// Optional negative prompt.
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub negative_prompt: Option<String>,
    /// Width in pixels.
    #[serde(default = "default_dim")]
    pub width: u32,
    /// Height in pixels.
    #[serde(default = "default_dim")]
    pub height: u32,
    /// Number of images.
    #[serde(default = "default_n")]
    pub n: u32,
    /// Metadata.
    #[serde(skip_serializing_if = "Metadata::is_empty", default)]
    pub metadata: Metadata,
}

fn default_dim() -> u32 {
    1024
}
fn default_n() -> u32 {
    1
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn usage_total_falls_back_to_sum() {
        let u = Usage {
            input_tokens: Some(10),
            output_tokens: Some(5),
            total_tokens: None,
        };
        assert_eq!(u.total(), 15);
        let u2 = Usage {
            total_tokens: Some(100),
            ..u
        };
        assert_eq!(u2.total(), 100);
    }

    #[test]
    fn response_text_accessor() {
        let resp = ChatResponse {
            model: "groq/x".into(),
            message: Message::assistant("hello"),
            usage: Usage::default(),
            tool_calls_executed: vec![],
            provider_response_id: None,
            finish_reason: Some("stop".into()),
            metadata: Metadata::new(),
        };
        assert_eq!(resp.text(), "hello");
        assert!(resp.tool_calls().is_empty());
    }
}
