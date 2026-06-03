//! AWS Kiro (`ProviderName::Kiro`) provider.
//!
//! Ports `gravity/kiro/*`. The brace-counting event-stream parser ([`parser`])
//! — Kiro's parity-critical reverse-engineered kernel — is complete and tested.
//! The AWS IDC OAuth device/social flow, token refresh, and CodeWhisperer
//! conversation request are layered on top in the M2d milestone.

pub(crate) mod auth;
mod build;
pub mod parser;
mod provider;

pub use provider::KiroProvider;
