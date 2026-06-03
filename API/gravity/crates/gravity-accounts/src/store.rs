//! Machine-local key management and the encrypted-file envelope.
//!
//! Mirrors `gravity/crypto.py`: the key lives at `~/.gravity/.gravity.key`
//! (mode 0600), and an encrypted accounts file is wrapped as
//! `{"v": 1, "encrypted": true, "ciphertext": "<fernet_token>"}`.

use crate::crypto::{FernetError, FernetKey};
use base64::engine::general_purpose::URL_SAFE;
use base64::Engine as _;
use gravity_core::Env;
use std::path::{Path, PathBuf};

/// Errors from the key store.
#[derive(Debug, thiserror::Error)]
pub enum StoreError {
    /// Filesystem error.
    #[error("key store io: {0}")]
    Io(String),
    /// Crypto error.
    #[error(transparent)]
    Fernet(#[from] FernetError),
    /// The home directory could not be determined.
    #[error("could not determine home directory")]
    NoHome,
}

/// Path to the Gravity config directory (`~/.gravity`).
pub fn gravity_dir() -> Result<PathBuf, StoreError> {
    home_dir().map(|h| h.join(".gravity")).ok_or(StoreError::NoHome)
}

/// Path to the machine-local Fernet key file.
pub fn key_path() -> Result<PathBuf, StoreError> {
    gravity_dir().map(|d| d.join(".gravity.key"))
}

fn home_dir() -> Option<PathBuf> {
    std::env::var_os("HOME")
        .or_else(|| std::env::var_os("USERPROFILE"))
        .map(PathBuf::from)
}

/// Load the machine key, generating and persisting a fresh one (mode 0600) if
/// none exists. The generated key is a 32-byte url-safe-base64 string, matching
/// `Fernet.generate_key()`.
pub fn get_or_create_key(env: &Env) -> Result<Vec<u8>, StoreError> {
    let path = key_path()?;
    if let Ok(bytes) = std::fs::read(&path) {
        let trimmed: Vec<u8> = bytes
            .iter()
            .copied()
            .filter(|b| !b.is_ascii_whitespace())
            .collect();
        if !trimmed.is_empty() {
            return Ok(trimmed);
        }
    }
    // Generate 32 random bytes → url-safe base64.
    let mut raw = [0u8; 32];
    env.entropy.fill(&mut raw);
    let key = URL_SAFE.encode(raw).into_bytes();
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent).map_err(|e| StoreError::Io(e.to_string()))?;
    }
    std::fs::write(&path, &key).map_err(|e| StoreError::Io(e.to_string()))?;
    set_owner_only(&path);
    Ok(key)
}

#[cfg(unix)]
fn set_owner_only(path: &Path) {
    use std::os::unix::fs::PermissionsExt;
    let _ = std::fs::set_permissions(path, std::fs::Permissions::from_mode(0o600));
}

#[cfg(not(unix))]
fn set_owner_only(_path: &Path) {}

/// Encrypt a UTF-8 string to a Fernet token, drawing the timestamp and IV from
/// `env`.
pub fn encrypt_str(env: &Env, key: &[u8], plaintext: &str) -> Result<String, StoreError> {
    let fk = FernetKey::from_base64(key)?;
    let mut iv = [0u8; 16];
    env.entropy.fill(&mut iv);
    Ok(fk.encrypt_with(plaintext.as_bytes(), env.clock.now_secs(), iv))
}

/// Decrypt a Fernet token to a UTF-8 string.
pub fn decrypt_str(key: &[u8], token: &str) -> Result<String, StoreError> {
    let fk = FernetKey::from_base64(key)?;
    let pt = fk.decrypt(token.as_bytes())?;
    String::from_utf8(pt).map_err(|_| StoreError::Fernet(FernetError::BadPadding))
}

/// Detect whether file content is the encrypted-accounts envelope.
pub fn is_encrypted_content(content: &str) -> bool {
    let trimmed = content.trim_start();
    if !trimmed.starts_with('{') {
        return false;
    }
    serde_json::from_str::<serde_json::Value>(content)
        .ok()
        .and_then(|v| v.get("encrypted").and_then(|e| e.as_bool()))
        .unwrap_or(false)
}

/// Wrap a Fernet ciphertext in the encrypted-file envelope.
pub fn wrap_encrypted(ciphertext: &str) -> String {
    serde_json::json!({ "v": 1, "encrypted": true, "ciphertext": ciphertext }).to_string()
}

/// Extract the ciphertext from the envelope.
pub fn unwrap_encrypted(content: &str) -> Option<String> {
    serde_json::from_str::<serde_json::Value>(content)
        .ok()?
        .get("ciphertext")?
        .as_str()
        .map(str::to_owned)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn envelope_detection_and_roundtrip() {
        let env = wrap_encrypted("TOKEN123");
        assert!(is_encrypted_content(&env));
        assert_eq!(unwrap_encrypted(&env).unwrap(), "TOKEN123");
        assert!(!is_encrypted_content(r#"{"emails":[]}"#));
        assert!(!is_encrypted_content("not json"));
    }

    #[test]
    fn encrypt_decrypt_str_with_fixed_env() {
        let env = Env::fixed(1_700_000_000_000, 99);
        let key = b"cw_0x689RpI-jtRR7oE8h_eQsKImvJapLeSbXpwF4e4=".to_vec();
        let token = encrypt_str(&env, &key, "api-key-value").unwrap();
        assert_eq!(decrypt_str(&key, &token).unwrap(), "api-key-value");
    }
}
