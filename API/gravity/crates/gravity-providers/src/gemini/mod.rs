//! Gemini web (`ProviderName::Gemini`) provider.
//!
//! Ports `gravity/gemini/*`. The UTF-16-unit length-prefixed **frame parser**
//! ([`parser`]) and nested-index extraction — Gemini's parity-critical
//! reverse-engineered kernel — are complete and tested. The `batchexecute`
//! `f.req` request assembly and cookie/session bootstrap are layered on top in
//! the M2b milestone.

mod build;
pub mod parser;
mod provider;

pub use provider::GeminiProvider;
