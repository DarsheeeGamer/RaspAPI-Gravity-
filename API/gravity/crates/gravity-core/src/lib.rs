//! # gravity-core
//!
//! The canonical **Gravity Schema** and shared contracts for the Gravity AI
//! gateway — a pure-Rust port of the Python `gravity` library.
//!
//! This crate is dependency-light (serde + a few frugal primitives) and carries
//! **no I/O**. It defines:
//!
//! - The provider-agnostic request/response types ([`ChatRequest`],
//!   [`ChatResponse`], [`Message`], [`ToolCall`], …).
//! - Provider identity and [`ProviderCapabilities`].
//! - The [`SchemaConverter`] trait (Gravity ⇄ provider wire format) and
//!   [`StreamEvent`].
//! - Account data models ([`ProviderAccount`], [`AuthConfig`], [`AccountsFile`]).
//! - Conversation history and [`Checkpoint`]s.
//! - The [`Error`] hierarchy with centralized retry classification.
//! - Injectable [`Clock`] / [`Entropy`] for reproducible request signing.
//!
//! ## Memory model
//!
//! Public types are owned (so they cross `await` points, FFI, and `Arc` freely),
//! but chosen for frugality: immutable identifiers use [`Box<str>`], binary
//! payloads use [`bytes::Bytes`] (refcounted, zero-copy clones), small
//! collections use [`smallvec::SmallVec`] inline storage, and empty metadata
//! costs a single null pointer. Borrowing (`Cow`/`&str`) is reserved for the hot
//! wire-conversion path in [`SchemaConverter::to_wire`].

#![forbid(unsafe_code)]

pub mod accounts;
pub mod capabilities;
pub mod clock;
pub mod conversation;
pub mod error;
pub mod message;
pub mod metadata;
pub mod model_id;
pub mod request;
pub mod response;
pub mod schema;
pub mod stream;
pub mod structured_output;
pub mod tool;
pub mod tool_emulation;

pub(crate) mod serde_bytes_b64;

// ── Curated public prelude ──────────────────────────────────────────────────

pub use accounts::{
    AccountsFile, AuthConfig, CredentialKind, EmailBundle, ModelRoute, ModelRouteCandidate,
    ProviderAccount, ProviderDefaults, QuotaState, QuotaWindow, RotationState, SecretRecord,
    SecurityConfig, TransportConfig,
};
pub use capabilities::{
    HistoryMode, ProviderCapabilities, ProviderName, RemoteSessionMode, StructuredOutputMode,
    ToolSupportMode, UnknownProvider,
};
pub use clock::{Clock, Entropy, Env, FixedClock, OsEntropy, SeededEntropy, SystemClock};
pub use conversation::{
    CanonicalTurn, Checkpoint, Conversation, ConversationRuntimeState,
};
pub use error::{Error, ProviderError, ProviderErrorKind, Result};
pub use message::{Content, ContentPart, ImageUrl, InputFile, Message, Role};
pub use metadata::Metadata;
pub use model_id::{ModelId, ModelIdParseError};
pub use request::{ChatRequest, FunctionCalling, ResponseFormat, Sampling};
pub use response::{
    ChatResponse, EmbeddingResponse, FinishReason, GeneratedImage, ImageRequest, ImageResponse,
    Usage,
};
pub use schema::SchemaConverter;
pub use stream::StreamEvent;
pub use structured_output::{build_structured_output_instruction, extract_json_payload};
pub use tool::{ExecutedToolCall, ToolCall, ToolDefinition, ToolOutput, ToolResult};
pub use tool_emulation::{build_json_tool_instruction, parse_tool_envelope, ToolEnvelope};
