//! The Gravity ⇄ provider-wire conversion contract.

use crate::conversation::Conversation;
use crate::error::Result;
use crate::model_id::ModelId;
use crate::request::ChatRequest;
use crate::response::ChatResponse;
use crate::stream::StreamEvent;
use serde::de::DeserializeOwned;
use serde::Serialize;

/// Converts the canonical [`ChatRequest`] / [`ChatResponse`] to and from a
/// provider's wire schema.
///
/// This is the formalization of the converter functions in
/// `gravity/providers/schema.py`. The ~23 OpenAI-compatible providers share one
/// implementation; bespoke web providers implement it directly.
///
/// `to_wire` borrows from the request/conversation so the wire payload can hold
/// `&str` / `Cow<str>` views instead of cloning every field.
pub trait SchemaConverter {
    /// The serializable wire-request type (often carries a lifetime borrowing
    /// from the [`ChatRequest`]).
    type WireRequest<'a>: Serialize
    where
        Self: 'a;

    /// The deserializable wire-response type.
    type WireResponse: DeserializeOwned;

    /// Build the provider wire request, borrowing from the canonical request.
    fn to_wire<'a>(&'a self, req: &'a ChatRequest, conv: &'a Conversation) -> Self::WireRequest<'a>;

    /// Map a decoded wire response back to the canonical [`ChatResponse`].
    ///
    /// Takes `&self` because conversion may consult provider config; the
    /// `from_` name follows the wire-decode convention, not a constructor.
    #[allow(clippy::wrong_self_convention)]
    fn from_wire(&self, raw: Self::WireResponse, model: &ModelId) -> Result<ChatResponse>;

    /// Parse one line/frame of a streaming response into a [`StreamEvent`].
    ///
    /// Returns `Ok(None)` for lines that carry no event (keep-alives, comments).
    fn parse_stream_line(&self, line: &[u8], model: &ModelId) -> Result<Option<StreamEvent>>;
}
