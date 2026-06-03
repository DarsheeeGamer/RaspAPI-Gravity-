//! Streaming chat events.

use crate::response::{ChatResponse, Usage};
use bytes::Bytes;

/// An incremental event from a streaming chat call.
///
/// Providers translate their wire stream (SSE deltas, Gemini frames, Kiro
/// brace-counted JSON, …) into this uniform sequence. The terminal
/// [`StreamEvent::Done`] carries the fully assembled [`ChatResponse`] so callers
/// that buffer a stream get the same value a non-streaming `chat` would return.
#[derive(Debug, Clone)]
pub enum StreamEvent {
    /// A chunk of assistant text. Held as [`Bytes`] to forward the transport
    /// buffer without copying.
    TextDelta(Bytes),
    /// The model began emitting a tool call with the given name.
    ToolCallStarted {
        /// Tool name.
        name: Box<str>,
    },
    /// A usage update (often only present near the end of a stream).
    Usage(Usage),
    /// Terminal event: the complete response.
    Done(Box<ChatResponse>),
}

impl StreamEvent {
    /// Borrow the text if this is a [`StreamEvent::TextDelta`].
    pub fn as_text(&self) -> Option<&[u8]> {
        match self {
            StreamEvent::TextDelta(b) => Some(b),
            _ => None,
        }
    }
}
