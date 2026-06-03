//! Account, credential, and transport data models.
//!
//! These mirror `gravity/accounts.py` field-for-field so existing
//! `accounts.json` (schema_version 4) files deserialize unchanged. The
//! encrypted-secret store, disk IO, and migration live in `gravity-accounts`;
//! this module is pure data.

use crate::capabilities::{HistoryMode, ProviderName, RemoteSessionMode, StructuredOutputMode, ToolSupportMode};
use crate::metadata::Metadata;
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

/// One rate-limit window (requests or tokens).
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct QuotaWindow {
    /// Window limit.
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub limit: Option<u64>,
    /// Remaining allowance.
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub remaining: Option<u64>,
    /// When the window resets.
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub reset_at: Option<DateTime<Utc>>,
}

/// Quota / cooldown state for an account.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct QuotaState {
    /// Request-count window.
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub requests: Option<QuotaWindow>,
    /// Token-count window.
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub tokens: Option<QuotaWindow>,
    /// Max concurrent in-flight requests.
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub concurrency_limit: Option<u32>,
    /// Cooldown expiry.
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub cooldown_until: Option<DateTime<Utc>>,
    /// Last time we were rate limited.
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub last_rate_limit_at: Option<DateTime<Utc>>,
}

/// Health/rotation state for an account, mutated by the failover loop.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RotationState {
    /// `"healthy"`, `"degraded"`, or `"cooldown"`.
    #[serde(default = "healthy")]
    pub state: Box<str>,
    /// Consecutive failure count.
    #[serde(default)]
    pub consecutive_failures: u32,
    /// Last error code seen.
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub last_error_code: Option<Box<str>>,
    /// Last error timestamp.
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub last_error_at: Option<DateTime<Utc>>,
    /// Last success timestamp.
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub last_ok_at: Option<DateTime<Utc>>,
    /// Cooldown expiry.
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub cooldown_until: Option<DateTime<Utc>>,
}

fn healthy() -> Box<str> {
    "healthy".into()
}

impl Default for RotationState {
    fn default() -> Self {
        RotationState {
            state: healthy(),
            consecutive_failures: 0,
            last_error_code: None,
            last_error_at: None,
            last_ok_at: None,
            cooldown_until: None,
        }
    }
}

/// Per-account HTTP transport configuration.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TransportConfig {
    /// API base URL.
    pub base_url: String,
    /// Request timeout in milliseconds.
    #[serde(default = "default_timeout_ms")]
    pub timeout_ms: u32,
    /// Impersonation profile name (`"chrome"`, `"go_boring"`, …).
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub impersonate: Option<Box<str>>,
    /// Proxy URL.
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub proxy_url: Option<Box<str>>,
    /// Whether to verify TLS certificates.
    #[serde(default = "default_true")]
    pub verify_ssl: bool,
    /// Extra headers added to every request.
    #[serde(default, skip_serializing_if = "std::collections::BTreeMap::is_empty")]
    pub extra_headers: std::collections::BTreeMap<String, String>,
}

fn default_timeout_ms() -> u32 {
    60_000
}
fn default_true() -> bool {
    true
}

/// Credential configuration. Mirrors the Python `AuthConfig` shape exactly
/// (a `kind` discriminator plus typed optionals plus an `extra` bag) so
/// existing files load. [`AuthConfig::kind_parsed`] gives a typed view.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AuthConfig {
    /// Credential kind: `"api_key"`, `"oauth"`, `"session"`, `"signed_hmac"`, …
    pub kind: Box<str>,
    /// Reference into the encrypted secret store.
    pub secret_ref: Box<str>,
    /// Human-friendly label.
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub account_label: Option<Box<str>>,
    /// Organization id (Claude, OpenAI).
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub org_id: Option<Box<str>>,
    /// Stable device id.
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub device_id: Option<Box<str>>,
    /// Routing hint header (Claude).
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub routing_hint: Option<Box<str>>,
    /// ChatGPT account id.
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub chatgpt_account_id: Option<Box<str>>,
    /// Reference to a refresh-token secret.
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub refresh_secret_ref: Option<Box<str>>,
    /// Provider-specific extras (may carry an inline secret for FFI callers).
    #[serde(default, skip_serializing_if = "Metadata::is_empty")]
    pub extra: Metadata,
}

/// Typed view of [`AuthConfig::kind`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CredentialKind {
    /// Static API key.
    ApiKey,
    /// OAuth access/refresh tokens.
    Oauth,
    /// Web session cookies.
    Session,
    /// HMAC request signing seed.
    SignedHmac,
    /// An unrecognized kind (forward-compatible).
    Other,
}

impl AuthConfig {
    /// Classify [`Self::kind`] into a [`CredentialKind`].
    pub fn kind_parsed(&self) -> CredentialKind {
        match &*self.kind {
            "api_key" => CredentialKind::ApiKey,
            "oauth" => CredentialKind::Oauth,
            "session" => CredentialKind::Session,
            "signed_hmac" => CredentialKind::SignedHmac,
            _ => CredentialKind::Other,
        }
    }

    /// Read an inline secret from `extra` (used by FFI callers that pass an
    /// `api_key` directly rather than via the encrypted store).
    pub fn inline_secret(&self, key: &str) -> Option<&str> {
        self.extra.get(key).and_then(|v| v.as_str())
    }
}

/// Capability defaults snapshotted onto an account.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ProviderDefaults {
    /// Tool support.
    pub tool_support: ToolSupportMode,
    /// Structured output.
    pub structured_output: StructuredOutputMode,
    /// Remote session mode.
    pub remote_session_mode: RemoteSessionMode,
    /// System-prompt support.
    pub supports_system_prompt: bool,
    /// Default history mode.
    #[serde(default = "default_history_mode")]
    pub default_history_mode: HistoryMode,
}

fn default_history_mode() -> HistoryMode {
    HistoryMode::Auto
}

/// A single configured provider account.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ProviderAccount {
    /// Unique account id.
    pub account_id: Box<str>,
    /// Which provider this account is for.
    pub provider: ProviderName,
    /// Soft enable/disable.
    #[serde(default = "default_true")]
    pub enabled: bool,
    /// Account pool/group.
    #[serde(default = "default_pool")]
    pub pool: Box<str>,
    /// Selection priority (lower = preferred).
    #[serde(default = "default_priority")]
    pub priority: i32,
    /// Weight within a priority tier.
    #[serde(default = "default_weight")]
    pub weight: u32,
    /// Credentials.
    pub auth: AuthConfig,
    /// Transport config.
    pub transport: TransportConfig,
    /// Mutable rotation/health state.
    #[serde(default)]
    pub rotation_state: RotationState,
    /// Quota state.
    #[serde(default)]
    pub quota: QuotaState,
    /// Capability defaults.
    pub defaults: ProviderDefaults,
    /// Metadata.
    #[serde(default, skip_serializing_if = "Metadata::is_empty")]
    pub metadata: Metadata,
}

fn default_pool() -> Box<str> {
    "default".into()
}
fn default_priority() -> i32 {
    100
}
fn default_weight() -> u32 {
    1
}

impl ProviderAccount {
    /// Whether the account is currently in a cooldown window per `now_millis`.
    pub fn in_cooldown(&self, now_millis: u64) -> bool {
        self.rotation_state
            .cooldown_until
            .map(|t| (t.timestamp_millis() as u64) > now_millis)
            .unwrap_or(false)
    }
}

/// A model-routing candidate.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ModelRouteCandidate {
    /// Candidate provider.
    pub provider: ProviderName,
    /// Pool to draw from.
    #[serde(default = "default_pool")]
    pub pool: Box<str>,
    /// Weight.
    #[serde(default = "default_weight")]
    pub weight: u32,
}

/// A named model route.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ModelRoute {
    /// Selector pattern.
    pub selector: String,
    /// Ordered candidates.
    pub candidates: Vec<ModelRouteCandidate>,
}

/// A bundle of accounts grouped under one email/identity.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EmailBundle {
    /// Identity email.
    pub email: String,
    /// Enabled flag.
    #[serde(default = "default_true")]
    pub enabled: bool,
    /// Display name.
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub display_name: Option<String>,
    /// Accounts in this bundle.
    #[serde(default)]
    pub providers: Vec<ProviderAccount>,
    /// Model routes.
    #[serde(default)]
    pub model_routes: Vec<ModelRoute>,
    /// Metadata.
    #[serde(default, skip_serializing_if = "Metadata::is_empty")]
    pub metadata: Metadata,
}

/// An encrypted secret record.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SecretRecord {
    /// Reference key.
    pub secret_ref: Box<str>,
    /// Ciphertext (Fernet token).
    pub ciphertext: String,
    /// Last update.
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub updated_at: Option<DateTime<Utc>>,
}

/// Security/encryption configuration block.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SecurityConfig {
    /// Whether secrets are encrypted at rest.
    #[serde(default = "default_true")]
    pub secrets_encrypted: bool,
    /// Encryption scheme tag.
    #[serde(default = "default_encryption")]
    pub encryption: Box<str>,
    /// Age recipient, if used.
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub age_recipient: Option<Box<str>>,
}

fn default_encryption() -> Box<str> {
    "age-v1".into()
}

impl Default for SecurityConfig {
    fn default() -> Self {
        SecurityConfig {
            secrets_encrypted: true,
            encryption: default_encryption(),
            age_recipient: None,
        }
    }
}

/// The full on-disk accounts file (schema v4).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AccountsFile {
    /// Schema version.
    #[serde(default = "default_schema_version")]
    pub schema_version: u32,
    /// Security config.
    #[serde(default)]
    pub security: SecurityConfig,
    /// Email bundles.
    #[serde(default)]
    pub emails: Vec<EmailBundle>,
    /// Encrypted secrets.
    #[serde(default)]
    pub secrets: Vec<SecretRecord>,
}

fn default_schema_version() -> u32 {
    4
}

impl Default for AccountsFile {
    fn default() -> Self {
        AccountsFile {
            schema_version: 4,
            security: SecurityConfig::default(),
            emails: Vec::new(),
            secrets: Vec::new(),
        }
    }
}

impl AccountsFile {
    /// Iterate over every account across all bundles.
    pub fn iter_accounts(&self) -> impl Iterator<Item = &ProviderAccount> {
        self.emails.iter().flat_map(|b| b.providers.iter())
    }

    /// Mutably iterate over every account.
    pub fn iter_accounts_mut(&mut self) -> impl Iterator<Item = &mut ProviderAccount> {
        self.emails.iter_mut().flat_map(|b| b.providers.iter_mut())
    }

    /// Find an account by id.
    pub fn find(&self, account_id: &str) -> Option<&ProviderAccount> {
        self.iter_accounts().find(|a| &*a.account_id == account_id)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn auth_kind_classification() {
        let mut a = AuthConfig {
            kind: "api_key".into(),
            secret_ref: "s".into(),
            account_label: None,
            org_id: None,
            device_id: None,
            routing_hint: None,
            chatgpt_account_id: None,
            refresh_secret_ref: None,
            extra: Metadata::new(),
        };
        assert_eq!(a.kind_parsed(), CredentialKind::ApiKey);
        a.kind = "signed_hmac".into();
        assert_eq!(a.kind_parsed(), CredentialKind::SignedHmac);
        a.kind = "weird".into();
        assert_eq!(a.kind_parsed(), CredentialKind::Other);
    }

    #[test]
    fn accounts_file_defaults_and_iteration() {
        let f = AccountsFile::default();
        assert_eq!(f.schema_version, 4);
        assert_eq!(f.iter_accounts().count(), 0);
    }
}
