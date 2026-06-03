//! # gravity-providers
//!
//! Provider adapters and the registry for Gravity.
//!
//! Every provider implements the single async [`Provider`] trait (replacing the
//! Python `SyncProviderAdapter`/`AsyncProviderAdapter` split). Adapters are
//! immutable config shared behind `Arc<dyn Provider>`; per-call state travels in
//! a borrowed [`ChatCtx`]. The [`Registry`] is built once and indexed by
//! [`gravity_core::ProviderName`] for O(1), deterministic dispatch.
//!
//! Providers are ported in milestone order; reverse-engineered web providers
//! come first. Currently implemented: **Claude** (`claude.ai`).

#![forbid(unsafe_code)]

mod context;
mod provider;
mod registry;
pub(crate) mod util;

pub mod antigravity;
pub mod anthropic;
pub mod chatgpt;
pub mod claude;
pub mod codex;
pub mod cursor;
pub mod gemini;
pub mod gemini_cli;
pub mod glm;
pub mod kiro;
pub mod openai_compat;
pub mod perplexity;
pub mod windsurf;

pub use context::ChatCtx;
pub use provider::{Authenticator, EventStream, Provider};
pub use registry::{builtin, Registry, RegistryBuilder};
