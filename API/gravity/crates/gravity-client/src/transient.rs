//! Building a throwaway account from a flat request (the FFI fast-path).
//!
//! Mirrors `_make_account` in `gravity/_ffi_bridge.py`: an FFI caller passes a
//! provider, a secret (API/session key), and optional transport overrides, and
//! we synthesize a single [`ProviderAccount`] using the provider's capability
//! defaults — no persisted accounts file required.

use crate::client::ResolvedAccount;
use gravity_core::{
    AuthConfig, Metadata, ProviderAccount, ProviderDefaults, ProviderName, Result, TransportConfig,
};
use gravity_providers::Registry;

/// Options for a transient account.
#[derive(Debug, Default, Clone)]
pub struct TransientOptions {
    /// Primary secret (API key / session key).
    pub secret: String,
    /// Auth kind (`api_key` by default; `session_cookie` for Claude, …).
    pub kind: Option<String>,
    /// Base URL override.
    pub base_url: Option<String>,
    /// Impersonation profile override.
    pub impersonate: Option<String>,
    /// Proxy override.
    pub proxy_url: Option<String>,
    /// Request timeout (ms).
    pub timeout_ms: Option<u32>,
    /// Extra inline auth fields (device_id, org_id, …) copied into `auth.extra`.
    pub extra: Metadata,
    /// Explicit device id (Claude/ChatGPT).
    pub device_id: Option<String>,
    /// Explicit org id.
    pub org_id: Option<String>,
}

/// Build a [`ResolvedAccount`] for `provider`, drawing capability defaults from
/// the registry.
pub fn build_transient(
    registry: &Registry,
    provider: ProviderName,
    opts: TransientOptions,
) -> Result<ResolvedAccount> {
    let caps = registry.provider(provider)?.capabilities();
    let defaults = ProviderDefaults {
        tool_support: caps.tool_support,
        structured_output: caps.structured_output,
        remote_session_mode: caps.remote_session_mode,
        supports_system_prompt: caps.supports_system_prompt,
        default_history_mode: caps.default_history_mode,
    };

    let kind = opts.kind.unwrap_or_else(|| "api_key".to_owned());
    let mut extra = opts.extra;
    // Make the secret resolvable inline for both api_key and session flows.
    if !opts.secret.is_empty() {
        let field = if kind.contains("session") { "session_key" } else { "api_key" };
        extra.insert(field, opts.secret.clone());
    }

    let auth = AuthConfig {
        kind: kind.into(),
        secret_ref: "inline".into(),
        account_label: Some("transient".into()),
        org_id: opts.org_id.map(Into::into),
        device_id: opts.device_id.map(Into::into),
        routing_hint: None,
        chatgpt_account_id: None,
        refresh_secret_ref: None,
        extra,
    };

    let transport = TransportConfig {
        base_url: opts.base_url.unwrap_or_default(),
        timeout_ms: opts.timeout_ms.unwrap_or(60_000),
        impersonate: opts.impersonate.map(Into::into),
        proxy_url: opts.proxy_url.map(Into::into),
        verify_ssl: true,
        extra_headers: Default::default(),
    };

    let account = ProviderAccount {
        account_id: "transient".into(),
        provider,
        enabled: true,
        pool: "default".into(),
        priority: 100,
        weight: 1,
        auth,
        transport,
        rotation_state: Default::default(),
        quota: Default::default(),
        defaults,
        metadata: Metadata::new(),
    };

    Ok(ResolvedAccount {
        account,
        secret: opts.secret,
    })
}
