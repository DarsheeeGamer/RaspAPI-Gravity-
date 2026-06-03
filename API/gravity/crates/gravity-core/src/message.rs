//! Conversation messages and their content parts.
//!
//! Mirrors the `Message` / `*Part` types in `gravity/types.py`, with
//! allocation-frugal representations:
//! - [`Content::Text`] covers the >95% plain-text case with one `String`.
//! - [`Content::Parts`] uses a [`SmallVec`] inline for 1–2 parts (typical
//!   multimodal turns) before spilling to the heap.
//! - [`InputFile`] stores raw [`Bytes`] once; base64 is derived at the wire
//!   boundary instead of being stored alongside (the Python type held both
//!   `data` and `data_base64`).

use crate::metadata::Metadata;
use crate::tool::{ToolCall, ToolResult};
use bytes::Bytes;
use serde::{Deserialize, Serialize};
use smallvec::SmallVec;

/// The author role of a message.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Role {
    /// System / developer instruction.
    System,
    /// End-user input.
    User,
    /// Model output.
    Assistant,
    /// A tool execution result.
    Tool,
}

/// An `image_url` content part payload.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ImageUrl {
    /// The image URL (may be a `data:` URI).
    pub url: String,
    /// Optional provider detail hint (`"low"`, `"high"`, `"auto"`).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub detail: Option<Box<str>>,
}

/// An `input_file` content part: an inline binary attachment.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct InputFile {
    /// Original file name.
    pub file_name: Box<str>,
    /// MIME type, e.g. `application/pdf`.
    pub mime_type: Box<str>,
    /// Raw file bytes. Serialized as standard base64.
    #[serde(with = "crate::serde_bytes_b64")]
    pub data: Bytes,
}

/// A single part of a multimodal message.
///
/// Serializes with an internal `type` tag matching the Python `Literal`
/// discriminators (`text`, `image_url`, `json`, `input_file`).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ContentPart {
    /// Plain text.
    Text {
        /// The text body.
        text: String,
    },
    /// An image reference.
    ImageUrl {
        /// The image payload.
        image_url: ImageUrl,
    },
    /// An embedded JSON object.
    Json {
        /// The JSON data.
        data: serde_json::Map<String, serde_json::Value>,
    },
    /// An inline file attachment.
    InputFile(InputFile),
}

/// Message body: either a plain string or a list of parts.
///
/// `untagged` so `"hello"` and `[{"type":"text",...}]` both deserialize, exactly
/// like the Python `str | list[MessagePart]` union.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(untagged)]
pub enum Content {
    /// A single text body (the common case).
    Text(String),
    /// An ordered list of parts.
    Parts(SmallVec<[ContentPart; 2]>),
}

impl Content {
    /// Borrow the text when this is a plain-text body, else `""`.
    ///
    /// Equivalent to the Python `Message.text` property.
    pub fn as_text(&self) -> &str {
        match self {
            Content::Text(s) => s,
            Content::Parts(_) => "",
        }
    }
}

impl From<String> for Content {
    fn from(s: String) -> Self {
        Content::Text(s)
    }
}

impl From<&str> for Content {
    fn from(s: &str) -> Self {
        Content::Text(s.to_owned())
    }
}

/// A single conversation message.
///
/// Mirrors `gravity.types.Message`. `tool_calls` uses an inline [`SmallVec`]
/// because a message carries zero or one tool call in the overwhelming majority
/// of turns.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Message {
    /// Author role.
    pub role: Role,
    /// Message body. `None` for e.g. an assistant turn that only emits tools.
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub content: Option<Content>,
    /// Optional speaker name.
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub name: Option<Box<str>>,
    /// Tool calls emitted by the model (assistant turns).
    #[serde(skip_serializing_if = "SmallVec::is_empty", default)]
    pub tool_calls: SmallVec<[ToolCall; 1]>,
    /// The tool result carried by a `role = "tool"` message.
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub tool_result: Option<ToolResult>,
    /// Free-form metadata.
    #[serde(skip_serializing_if = "Metadata::is_empty", default)]
    pub metadata: Metadata,
}

impl Message {
    /// Construct a `user` message.
    pub fn user(content: impl Into<Content>) -> Self {
        Message::bare(Role::User, Some(content.into()))
    }

    /// Construct an `assistant` message.
    pub fn assistant(content: impl Into<Content>) -> Self {
        Message::bare(Role::Assistant, Some(content.into()))
    }

    /// Construct a `system` message.
    pub fn system(content: impl Into<Content>) -> Self {
        Message::bare(Role::System, Some(content.into()))
    }

    /// Wrap a [`ToolResult`] in a `role = "tool"` message.
    pub fn tool_result(result: ToolResult) -> Self {
        Message {
            role: Role::Tool,
            content: None,
            name: None,
            tool_calls: SmallVec::new(),
            tool_result: Some(result),
            metadata: Metadata::new(),
        }
    }

    /// An assistant turn that carries tool calls and (optionally) text.
    pub fn assistant_with_tools(
        content: Option<Content>,
        tool_calls: impl IntoIterator<Item = ToolCall>,
    ) -> Self {
        Message {
            role: Role::Assistant,
            content,
            name: None,
            tool_calls: tool_calls.into_iter().collect(),
            tool_result: None,
            metadata: Metadata::new(),
        }
    }

    fn bare(role: Role, content: Option<Content>) -> Self {
        Message {
            role,
            content,
            name: None,
            tool_calls: SmallVec::new(),
            tool_result: None,
            metadata: Metadata::new(),
        }
    }

    /// The text content as a plain `&str` (empty for multipart/None).
    pub fn text(&self) -> &str {
        self.content.as_ref().map_or("", Content::as_text)
    }

    /// Validate the structural invariant: tool messages must carry a result.
    ///
    /// Mirrors the Python `_validate_tool_shape` model validator.
    pub fn validate(&self) -> Result<(), &'static str> {
        if self.role == Role::Tool && self.tool_result.is_none() {
            return Err("tool messages require `tool_result`");
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn text_message_serializes_with_string_content() {
        let m = Message::user("hi");
        let json = serde_json::to_value(&m).unwrap();
        assert_eq!(json["role"], "user");
        assert_eq!(json["content"], "hi");
        // empty collections are omitted
        assert!(json.get("tool_calls").is_none());
        assert!(json.get("metadata").is_none());
    }

    #[test]
    fn content_deserializes_from_string_or_parts() {
        let s: Content = serde_json::from_str(r#""hello""#).unwrap();
        assert_eq!(s.as_text(), "hello");
        let p: Content =
            serde_json::from_str(r#"[{"type":"text","text":"a"}]"#).unwrap();
        assert!(matches!(p, Content::Parts(_)));
    }

    #[test]
    fn tool_message_validation() {
        let bad = Message::bare(Role::Tool, None);
        assert!(bad.validate().is_err());
    }

    #[test]
    fn input_file_roundtrips_base64() {
        let file = InputFile {
            file_name: "a.bin".into(),
            mime_type: "application/octet-stream".into(),
            data: Bytes::from_static(&[1, 2, 3, 255]),
        };
        let part = ContentPart::InputFile(file);
        let json = serde_json::to_string(&part).unwrap();
        let back: ContentPart = serde_json::from_str(&json).unwrap();
        assert_eq!(part, back);
    }
}
