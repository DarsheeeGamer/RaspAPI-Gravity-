//! Loading and saving the accounts file, with legacy-schema migration.
//!
//! Mirrors `gravity/io.py`:
//! - decrypts the envelope when the file is encrypted,
//! - rewrites the legacy provider name `"zlm"` → `"glm"`,
//! - migrates a v1/v2 top-level `accounts` list into a single `emails` bundle,
//! - then deserializes into the strict [`AccountsFile`] model.

use crate::store::{self, StoreError};
use gravity_core::{AccountsFile, Env};
use serde_json::Value;
use std::path::Path;

/// Errors from accounts IO.
#[derive(Debug, thiserror::Error)]
pub enum AccountsIoError {
    /// Filesystem error.
    #[error("accounts io: {0}")]
    Io(String),
    /// JSON parse/shape error.
    #[error("accounts file is not valid JSON: {0}")]
    Json(String),
    /// Decryption error.
    #[error(transparent)]
    Store(#[from] StoreError),
}

/// Apply the legacy migrations to a raw JSON value, in place.
///
/// This is the equivalent of `_migrate_provider_names` + the
/// `migrate_legacy_schema` model-validator from the Python library.
pub fn migrate_value(mut data: Value) -> Value {
    rewrite_provider_names(&mut data);
    if let Value::Object(map) = &mut data {
        // v1/v2: a top-level `accounts` list → one "Personal" email bundle.
        if !map.contains_key("emails") {
            if let Some(accounts) = map.remove("accounts") {
                let bundle = serde_json::json!({
                    "email": "personal@gravity.local",
                    "display_name": "Personal",
                    "providers": accounts,
                });
                map.insert("emails".into(), Value::Array(vec![bundle]));
            } else {
                map.insert("emails".into(), Value::Array(vec![]));
            }
        }
        map.entry("security").or_insert_with(|| Value::Object(Default::default()));
    }
    data
}

/// Recursively rewrite any `"provider": "zlm"` to `"glm"`.
fn rewrite_provider_names(value: &mut Value) {
    match value {
        Value::Object(map) => {
            if let Some(Value::String(p)) = map.get_mut("provider") {
                if p == "zlm" {
                    *p = "glm".to_owned();
                }
            }
            for v in map.values_mut() {
                rewrite_provider_names(v);
            }
        }
        Value::Array(arr) => arr.iter_mut().for_each(rewrite_provider_names),
        _ => {}
    }
}

/// Parse raw file content (already decrypted) into an [`AccountsFile`].
pub fn parse_accounts(content: &str) -> Result<AccountsFile, AccountsIoError> {
    let raw: Value = serde_json::from_str(content).map_err(|e| AccountsIoError::Json(e.to_string()))?;
    let migrated = migrate_value(raw);
    serde_json::from_value(migrated).map_err(|e| AccountsIoError::Json(e.to_string()))
}

/// Load and decrypt the accounts file at `path`. A missing file yields a
/// default empty [`AccountsFile`].
pub fn load_accounts(env: &Env, path: &Path) -> Result<AccountsFile, AccountsIoError> {
    let content = match std::fs::read_to_string(path) {
        Ok(c) => c,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(AccountsFile::default()),
        Err(e) => return Err(AccountsIoError::Io(e.to_string())),
    };
    let plaintext = if store::is_encrypted_content(&content) {
        let token = store::unwrap_encrypted(&content)
            .ok_or_else(|| AccountsIoError::Json("encrypted envelope missing ciphertext".into()))?;
        let key = store::get_or_create_key(env)?;
        store::decrypt_str(&key, &token)?
    } else {
        content
    };
    parse_accounts(&plaintext)
}

/// Serialize and write the accounts file. When `encrypt` is true (or a key file
/// already exists), the JSON is Fernet-encrypted and wrapped in the envelope.
pub fn save_accounts(
    env: &Env,
    accounts: &AccountsFile,
    path: &Path,
    encrypt: bool,
) -> Result<(), AccountsIoError> {
    let json = serde_json::to_string_pretty(accounts).map_err(|e| AccountsIoError::Json(e.to_string()))?;
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent).map_err(|e| AccountsIoError::Io(e.to_string()))?;
    }
    let out = if encrypt {
        let key = store::get_or_create_key(env)?;
        let token = store::encrypt_str(env, &key, &json)?;
        store::wrap_encrypted(&token)
    } else {
        json
    };
    std::fs::write(path, out).map_err(|e| AccountsIoError::Io(e.to_string()))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn migrates_legacy_accounts_list() {
        let legacy = serde_json::json!({
            "accounts": [
                {"account_id": "a1", "provider": "zlm", "auth": {"kind":"api_key","secret_ref":"s"},
                 "transport": {"base_url":"https://x"},
                 "defaults": {"tool_support":"native","structured_output":"none",
                              "remote_session_mode":"none","supports_system_prompt":true}}
            ]
        });
        let migrated = migrate_value(legacy);
        // zlm rewritten to glm, accounts wrapped into an email bundle
        let acct = &migrated["emails"][0]["providers"][0];
        assert_eq!(acct["provider"], "glm");
        assert_eq!(migrated["emails"][0]["email"], "personal@gravity.local");
    }

    #[test]
    fn parses_migrated_into_strict_model() {
        let legacy = serde_json::json!({
            "accounts": [
                {"account_id": "a1", "provider": "groq",
                 "auth": {"kind":"api_key","secret_ref":"s1"},
                 "transport": {"base_url":"https://api.groq.com"},
                 "defaults": {"tool_support":"native","structured_output":"json_instruction",
                              "remote_session_mode":"none","supports_system_prompt":true}}
            ]
        });
        let file = parse_accounts(&legacy.to_string()).unwrap();
        assert_eq!(file.schema_version, 4);
        assert_eq!(file.iter_accounts().count(), 1);
        assert_eq!(file.find("a1").unwrap().provider, gravity_core::ProviderName::Groq);
    }

    #[test]
    fn empty_when_missing() {
        let env = Env::fixed(0, 0);
        let missing = std::path::Path::new("/nonexistent/path/accounts.json");
        let f = load_accounts(&env, missing).unwrap();
        assert!(f.emails.is_empty());
    }
}
