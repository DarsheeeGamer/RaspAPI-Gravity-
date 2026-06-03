//! OpenAI-compatible provider base, shared by the ~21 API-key providers.
//!
//! Ports `gravity/providers/openai_compat.py` + `schema.py`. A single generic
//! [`OpenAiCompatProvider`] handles the HTTP chat/stream flow; each provider is
//! a small [`OaiConfig`] in [`catalog`].

pub mod catalog;
mod provider;
mod wire;

pub use catalog::all;
pub use provider::{OaiConfig, OpenAiCompatProvider};
