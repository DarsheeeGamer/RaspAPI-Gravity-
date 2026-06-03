//! ChatGPT web (`ProviderName::Chatgpt`) provider.
//!
//! Ports `gravity/chatgpt/*`. The Sentinel **Proof-of-Work** ([`pow`]) — the
//! parity-critical, golden-traced kernel — is complete. The full web flow
//! (Auth0 device session, sentinel requirements, Turnstile, `/backend-api/
//! conversation` SSE) is layered on top in the M2c milestone.

pub mod pow;
mod provider;

pub use provider::ChatgptProvider;
