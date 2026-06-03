//! The per-call context handed to a [`crate::Provider`].

use gravity_core::{ChatRequest, Checkpoint, Conversation, Env, ProviderAccount};
use gravity_http::Transport;

/// Everything a provider needs to service one chat/checkpoint call.
///
/// Bundles the four "keyword arguments" every Python adapter method received
/// (`account`, `conversation`, `checkpoint`, `request`) plus the injected
/// transport and environment. All fields are borrowed: the client owns the
/// state and the provider only reads it, so a `ChatCtx` is cheap to build.
pub struct ChatCtx<'a> {
    /// The selected account (immutable config; credentials + transport).
    pub account: &'a ProviderAccount,
    /// The primary secret already resolved from the account (API key, session
    /// key, …). Empty when the provider authenticates purely from `account`.
    pub secret: &'a str,
    /// Canonical conversation history.
    pub conversation: &'a Conversation,
    /// Active remote-session checkpoint, if any.
    pub checkpoint: Option<&'a Checkpoint>,
    /// The request being serviced.
    pub request: &'a ChatRequest,
    /// Shared HTTP transport (real impersonating client, or a mock in tests).
    pub http: &'a dyn Transport,
    /// Injected clock + entropy (deterministic in tests).
    pub env: &'a Env,
}

impl<'a> ChatCtx<'a> {
    /// The impersonation profile name for this account, defaulting to the
    /// provider's preferred profile when the account doesn't pin one.
    pub fn impersonate_or(&self, default: &'a str) -> &'a str {
        self.account
            .transport
            .impersonate
            .as_deref()
            .unwrap_or(default)
    }

    /// The account's configured proxy, if any.
    pub fn proxy(&self) -> Option<String> {
        self.account.transport.proxy_url.as_ref().map(|p| p.to_string())
    }

    /// The per-request timeout in milliseconds.
    pub fn timeout_ms(&self) -> u32 {
        self.account.transport.timeout_ms
    }
}
