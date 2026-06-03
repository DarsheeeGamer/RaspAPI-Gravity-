//! In-memory account manager with disk persistence and secret resolution.
//!
//! Mirrors `gravity/account_manager.py` (CRUD over `~/.gravity/accounts.json`)
//! plus the secret-resolution path: an [`AuthConfig`] points at a `secret_ref`,
//! whose ciphertext lives in [`AccountsFile::secrets`] and is decrypted on
//! demand. Inline secrets carried in `auth.extra` (as FFI callers supply) take
//! precedence so a bare API key works without the encrypted store.

use crate::io::{self, AccountsIoError};
use crate::store::{self, StoreError};
use gravity_core::{AccountsFile, AuthConfig, Env, ProviderAccount, ProviderName};
use secrecy::{ExposeSecret, SecretString};
use std::path::{Path, PathBuf};

/// Errors from the account manager.
#[derive(Debug, thiserror::Error)]
pub enum ManagerError {
    /// IO/parse error.
    #[error(transparent)]
    Io(#[from] AccountsIoError),
    /// Crypto/key error.
    #[error(transparent)]
    Store(#[from] StoreError),
    /// A referenced secret could not be resolved.
    #[error("secret {0:?} not found")]
    SecretNotFound(String),
}

/// Owns the loaded accounts file and mediates reads, mutations, and secret
/// decryption.
#[derive(Debug)]
pub struct AccountManager {
    accounts: AccountsFile,
    path: PathBuf,
    env: Env,
    encrypt_on_save: bool,
}

impl AccountManager {
    /// Load the manager from `path`, decrypting if necessary.
    pub fn load(env: Env, path: impl Into<PathBuf>) -> Result<Self, ManagerError> {
        let path = path.into();
        let accounts = io::load_accounts(&env, &path)?;
        // Preserve the file's encryption posture on save.
        let encrypt_on_save = accounts.security.secrets_encrypted || store::key_path().map(|p| p.exists()).unwrap_or(false);
        Ok(AccountManager { accounts, path, env, encrypt_on_save })
    }

    /// Load from the default location (`~/.gravity/accounts.json`).
    pub fn load_default(env: Env) -> Result<Self, ManagerError> {
        let path = store::gravity_dir()?.join("accounts.json");
        AccountManager::load(env, path)
    }

    /// Build a manager around an in-memory file (no disk), for tests/embedding.
    pub fn in_memory(env: Env, accounts: AccountsFile) -> Self {
        AccountManager {
            accounts,
            path: PathBuf::from("/dev/null"),
            env,
            encrypt_on_save: false,
        }
    }

    /// Borrow the underlying accounts file.
    pub fn file(&self) -> &AccountsFile {
        &self.accounts
    }

    /// Mutable access to the underlying accounts file.
    pub fn file_mut(&mut self) -> &mut AccountsFile {
        &mut self.accounts
    }

    /// All accounts for `provider` that are enabled and not in cooldown, in
    /// priority/weight order (lower priority first; higher weight first within a
    /// tier). Accounts inside a disabled email bundle are excluded, and any
    /// account whose `rotation_state.cooldown_until` is still in the future is
    /// skipped — mirroring `_iter_accounts` in `clients.py`.
    pub fn accounts_for(&self, provider: ProviderName) -> Vec<&ProviderAccount> {
        let now = self.env.clock.now_millis();
        let mut v: Vec<&ProviderAccount> = self
            .accounts
            .emails
            .iter()
            .filter(|b| b.enabled)
            .flat_map(|b| b.providers.iter())
            .filter(|a| a.provider == provider && a.enabled && !a.in_cooldown(now))
            .collect();
        v.sort_by(|a, b| a.priority.cmp(&b.priority).then(b.weight.cmp(&a.weight)));
        v
    }

    /// Find an account by id.
    pub fn get(&self, account_id: &str) -> Option<&ProviderAccount> {
        self.accounts.find(account_id)
    }

    /// Resolve the plaintext secret an [`AuthConfig`] refers to.
    ///
    /// Resolution order:
    /// 1. an inline `api_key` / `token` in `auth.extra` (FFI fast-path),
    /// 2. the encrypted `secret_ref` record decrypted with the machine key.
    pub fn resolve_secret(&self, auth: &AuthConfig) -> Result<SecretString, ManagerError> {
        for inline_key in ["api_key", "token", "access_token", "session_key"] {
            if let Some(v) = auth.inline_secret(inline_key) {
                return Ok(SecretString::from(v.to_owned()));
            }
        }
        let rec = self
            .accounts
            .secrets
            .iter()
            .find(|s| s.secret_ref == auth.secret_ref)
            .ok_or_else(|| ManagerError::SecretNotFound(auth.secret_ref.to_string()))?;
        let key = store::get_or_create_key(&self.env)?;
        let plaintext = store::decrypt_str(&key, &rec.ciphertext)?;
        Ok(SecretString::from(plaintext))
    }

    /// Persist the current state to disk.
    pub fn save(&self) -> Result<(), ManagerError> {
        io::save_accounts(&self.env, &self.accounts, &self.path, self.encrypt_on_save)?;
        Ok(())
    }

    /// Reload from disk, discarding in-memory changes.
    pub fn reload(&mut self) -> Result<(), ManagerError> {
        self.accounts = io::load_accounts(&self.env, &self.path)?;
        Ok(())
    }

    /// Mark a successful call: clear failure state and cooldown, stamp `last_ok`.
    /// Mirrors `_mark_account_success` in `clients.py`.
    pub fn mark_success(&mut self, account_id: &str) {
        let now = self.now();
        if let Some(a) = self.account_mut(account_id) {
            a.rotation_state.state = "healthy".into();
            a.rotation_state.consecutive_failures = 0;
            a.rotation_state.cooldown_until = None;
            a.rotation_state.last_error_code = None;
            a.rotation_state.last_ok_at = Some(now);
        }
    }

    /// Mark a failed call: bump failure count, record the error, and set a
    /// cooldown window. Mirrors `_mark_account_failure`. `cooldown` is how long
    /// to skip this account (e.g. from `Retry-After`, else a default).
    pub fn mark_failure(&mut self, account_id: &str, code: &str, cooldown: std::time::Duration) {
        let now = self.now();
        let until = chrono::DateTime::from_timestamp_millis(
            self.env.clock.now_millis() as i64 + cooldown.as_millis() as i64,
        );
        if let Some(a) = self.account_mut(account_id) {
            a.rotation_state.consecutive_failures += 1;
            a.rotation_state.last_error_code = Some(code.into());
            a.rotation_state.last_error_at = Some(now);
            a.rotation_state.cooldown_until = until;
            a.rotation_state.state =
                if cooldown.is_zero() { "degraded".into() } else { "cooldown".into() };
        }
    }

    fn now(&self) -> chrono::DateTime<chrono::Utc> {
        chrono::DateTime::from_timestamp_millis(self.env.clock.now_millis() as i64)
            .unwrap_or_else(chrono::Utc::now)
    }

    fn account_mut(&mut self, account_id: &str) -> Option<&mut ProviderAccount> {
        self.accounts.iter_accounts_mut().find(|a| &*a.account_id == account_id)
    }

    /// Replace an account's [`AuthConfig`] with refreshed credentials.
    ///
    /// Used after a successful `Authenticator::refresh_credentials` call to
    /// persist the new access token / cookies back into the manager.
    pub fn update_auth(&mut self, account_id: &str, new_auth: AuthConfig) -> bool {
        if let Some(a) = self.account_mut(account_id) {
            a.auth = new_auth;
            true
        } else {
            false
        }
    }

    /// Enable/disable an account by id. Returns whether it was found.
    pub fn set_enabled(&mut self, account_id: &str, enabled: bool) -> bool {
        if let Some(a) = self
            .accounts
            .iter_accounts_mut()
            .find(|a| &*a.account_id == account_id)
        {
            a.enabled = enabled;
            true
        } else {
            false
        }
    }

    /// Path the manager persists to.
    pub fn path(&self) -> &Path {
        &self.path
    }
}

/// Helper for callers that hold a resolved secret and need the raw `&str`.
pub fn expose(secret: &SecretString) -> &str {
    secret.expose_secret()
}

#[cfg(test)]
mod tests {
    use super::*;
    use gravity_core::{
        AuthConfig, EmailBundle, HistoryMode, Metadata, ProviderDefaults, RemoteSessionMode,
        StructuredOutputMode, ToolSupportMode, TransportConfig,
    };

    fn account(id: &str, provider: ProviderName, priority: i32) -> ProviderAccount {
        ProviderAccount {
            account_id: id.into(),
            provider,
            enabled: true,
            pool: "default".into(),
            priority,
            weight: 1,
            auth: AuthConfig {
                kind: "api_key".into(),
                secret_ref: format!("ref-{id}").into(),
                account_label: None,
                org_id: None,
                device_id: None,
                routing_hint: None,
                chatgpt_account_id: None,
                refresh_secret_ref: None,
                extra: Metadata::new(),
            },
            transport: TransportConfig {
                base_url: "https://api.example.com".into(),
                timeout_ms: 60000,
                impersonate: None,
                proxy_url: None,
                verify_ssl: true,
                extra_headers: Default::default(),
            },
            rotation_state: Default::default(),
            quota: Default::default(),
            defaults: ProviderDefaults {
                tool_support: ToolSupportMode::Native,
                structured_output: StructuredOutputMode::None,
                remote_session_mode: RemoteSessionMode::None,
                supports_system_prompt: true,
                default_history_mode: HistoryMode::Auto,
            },
            metadata: Metadata::new(),
        }
    }

    fn file_with(accts: Vec<ProviderAccount>) -> AccountsFile {
        AccountsFile {
            emails: vec![EmailBundle {
                email: "e@x".into(),
                enabled: true,
                display_name: None,
                providers: accts,
                model_routes: vec![],
                metadata: Metadata::new(),
            }],
            ..Default::default()
        }
    }

    #[test]
    fn accounts_for_sorts_by_priority_then_weight() {
        let mut a_hi = account("hi", ProviderName::Groq, 10);
        a_hi.weight = 5;
        let a_lo = account("lo", ProviderName::Groq, 100);
        let other = account("x", ProviderName::Claude, 1);
        let mgr = AccountManager::in_memory(Env::fixed(0, 0), file_with(vec![a_lo, a_hi, other]));
        let groq = mgr.accounts_for(ProviderName::Groq);
        assert_eq!(groq.len(), 2);
        assert_eq!(&*groq[0].account_id, "hi"); // lower priority wins
    }

    #[test]
    fn resolves_inline_secret() {
        let mut a = account("a", ProviderName::Groq, 1);
        a.auth.extra.insert("api_key", "sk-inline");
        let mgr = AccountManager::in_memory(Env::fixed(0, 0), file_with(vec![a]));
        let acct = mgr.get("a").unwrap();
        let secret = mgr.resolve_secret(&acct.auth).unwrap();
        assert_eq!(expose(&secret), "sk-inline");
    }

    #[test]
    fn accounts_for_skips_cooled_down() {
        let mut a = account("a", ProviderName::Groq, 1);
        // cooldown until far future relative to the fixed clock (epoch 0).
        a.rotation_state.cooldown_until =
            Some(chrono::DateTime::from_timestamp_millis(10_000_000_000_000).unwrap());
        let b = account("b", ProviderName::Groq, 2);
        let mgr = AccountManager::in_memory(Env::fixed(0, 0), file_with(vec![a, b]));
        let avail = mgr.accounts_for(ProviderName::Groq);
        // 'a' is in cooldown → only 'b' remains.
        assert_eq!(avail.len(), 1);
        assert_eq!(&*avail[0].account_id, "b");
    }

    #[test]
    fn set_enabled_toggles() {
        let mut mgr = AccountManager::in_memory(Env::fixed(0, 0), file_with(vec![account("a", ProviderName::Groq, 1)]));
        assert!(mgr.set_enabled("a", false));
        assert!(!mgr.get("a").unwrap().enabled);
        assert!(!mgr.set_enabled("missing", true));
    }
}
