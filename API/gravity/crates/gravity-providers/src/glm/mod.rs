//! GLM / chat.z.ai (`ProviderName::Glm`) provider.
//!
//! Ports `gravity/glm/*`. The signing kernel ([`sign`]) — the parity-critical,
//! golden-traced part — is complete; the HTTP chat flow (guest bootstrap, chat
//! creation, signed completion, SSE `phase`/`delta_content` parsing) is built on
//! top of it in the M2a milestone.

mod build;
mod provider;
pub mod sign;

pub use provider::GlmProvider;
