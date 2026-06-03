//! Fernet symmetric encryption — wire-compatible with the Python
//! `cryptography.fernet.Fernet` used by `gravity/crypto.py`.
//!
//! A Fernet key is 32 bytes (url-safe base64), split into a 16-byte signing key
//! and a 16-byte AES-128 key. A token is:
//!
//! ```text
//! base64url( 0x80 ‖ ts[8] ‖ iv[16] ‖ AES-128-CBC(pkcs7, data) ‖ HMAC-SHA256[32] )
//! ```
//!
//! where the HMAC covers everything before it (version ‖ ts ‖ iv ‖ ciphertext)
//! and is keyed by the signing key. This implementation verifies the HMAC in
//! constant time and is byte-for-byte compatible with files written by the
//! Python library (TTL is not enforced, matching Gravity's usage).

use aes::Aes128;
use base64::engine::general_purpose::{STANDARD, URL_SAFE};
use base64::Engine as _;
use cbc::cipher::block_padding::Pkcs7;
use cbc::cipher::{BlockDecryptMut, BlockEncryptMut, KeyIvInit};
use hmac::{Hmac, Mac};
use sha2::Sha256;

type HmacSha256 = Hmac<Sha256>;
type Aes128CbcEnc = cbc::Encryptor<Aes128>;
type Aes128CbcDec = cbc::Decryptor<Aes128>;

const VERSION: u8 = 0x80;

/// A Fernet error.
#[derive(Debug, thiserror::Error)]
pub enum FernetError {
    /// The key was not a valid 32-byte url-safe base64 string.
    #[error("invalid Fernet key")]
    InvalidKey,
    /// The token was malformed (bad base64, wrong length, bad version).
    #[error("malformed Fernet token")]
    MalformedToken,
    /// The HMAC did not verify — wrong key or tampered token.
    #[error("Fernet signature verification failed")]
    BadSignature,
    /// The decrypted plaintext had invalid PKCS7 padding.
    #[error("invalid padding")]
    BadPadding,
}

/// A parsed Fernet key (signing + encryption halves).
pub struct FernetKey {
    signing: [u8; 16],
    encryption: [u8; 16],
}

impl FernetKey {
    /// Parse a url-safe-base64 32-byte Fernet key.
    pub fn from_base64(key: &[u8]) -> Result<Self, FernetError> {
        // Python writes/strips with url-safe base64; accept both alphabets.
        let raw = URL_SAFE
            .decode(trim(key))
            .or_else(|_| STANDARD.decode(trim(key)))
            .map_err(|_| FernetError::InvalidKey)?;
        if raw.len() != 32 {
            return Err(FernetError::InvalidKey);
        }
        let mut signing = [0u8; 16];
        let mut encryption = [0u8; 16];
        signing.copy_from_slice(&raw[..16]);
        encryption.copy_from_slice(&raw[16..]);
        Ok(FernetKey { signing, encryption })
    }

    /// Encrypt `plaintext` into a Fernet token string.
    ///
    /// `timestamp` is the seconds-since-epoch stamped into the token, and `iv`
    /// the 16-byte AES IV — both injected so the operation is deterministic for
    /// golden-trace tests. In production pass the current time and random IV.
    pub fn encrypt_with(&self, plaintext: &[u8], timestamp: u64, iv: [u8; 16]) -> String {
        let ct = Aes128CbcEnc::new(&self.encryption.into(), &iv.into())
            .encrypt_padded_vec_mut::<Pkcs7>(plaintext);

        let mut body = Vec::with_capacity(1 + 8 + 16 + ct.len() + 32);
        body.push(VERSION);
        body.extend_from_slice(&timestamp.to_be_bytes());
        body.extend_from_slice(&iv);
        body.extend_from_slice(&ct);

        let mut mac = HmacSha256::new_from_slice(&self.signing).expect("hmac key length");
        mac.update(&body);
        body.extend_from_slice(&mac.finalize().into_bytes());

        URL_SAFE.encode(body)
    }

    /// Decrypt a Fernet token to plaintext, verifying the HMAC.
    pub fn decrypt(&self, token: &[u8]) -> Result<Vec<u8>, FernetError> {
        let data = URL_SAFE
            .decode(trim(token))
            .or_else(|_| STANDARD.decode(trim(token)))
            .map_err(|_| FernetError::MalformedToken)?;
        // version(1) + ts(8) + iv(16) + ct(>=16) + hmac(32)
        if data.len() < 1 + 8 + 16 + 16 + 32 || data[0] != VERSION {
            return Err(FernetError::MalformedToken);
        }
        let (signed, tag) = data.split_at(data.len() - 32);
        let mut mac = HmacSha256::new_from_slice(&self.signing).expect("hmac key length");
        mac.update(signed);
        mac.verify_slice(tag).map_err(|_| FernetError::BadSignature)?;

        let iv: [u8; 16] = signed[9..25].try_into().expect("iv slice");
        let ct = &signed[25..];
        Aes128CbcDec::new(&self.encryption.into(), &iv.into())
            .decrypt_padded_vec_mut::<Pkcs7>(ct)
            .map_err(|_| FernetError::BadPadding)
    }
}

fn trim(b: &[u8]) -> &[u8] {
    let start = b.iter().position(|c| !c.is_ascii_whitespace()).unwrap_or(0);
    let end = b
        .iter()
        .rposition(|c| !c.is_ascii_whitespace())
        .map(|i| i + 1)
        .unwrap_or(start);
    &b[start..end]
}

#[cfg(test)]
mod tests {
    use super::*;

    // A real 32-byte url-safe-base64 Fernet key.
    const KEY: &[u8] = b"cw_0x689RpI-jtRR7oE8h_eQsKImvJapLeSbXpwF4e4=";

    #[test]
    fn roundtrip() {
        let k = FernetKey::from_base64(KEY).unwrap();
        let token = k.encrypt_with(b"hello gravity", 1_700_000_000, [7u8; 16]);
        let pt = k.decrypt(token.as_bytes()).unwrap();
        assert_eq!(pt, b"hello gravity");
    }

    #[test]
    fn deterministic_with_fixed_iv_and_ts() {
        let k = FernetKey::from_base64(KEY).unwrap();
        let a = k.encrypt_with(b"x", 42, [1u8; 16]);
        let b = k.encrypt_with(b"x", 42, [1u8; 16]);
        assert_eq!(a, b, "fixed ts+iv must produce identical tokens");
    }

    #[test]
    fn tampering_is_rejected() {
        let k = FernetKey::from_base64(KEY).unwrap();
        let mut token = k.encrypt_with(b"secret", 1, [2u8; 16]).into_bytes();
        // flip a byte in the middle
        let mid = token.len() / 2;
        token[mid] ^= 0xFF;
        assert!(matches!(
            k.decrypt(&token),
            Err(FernetError::BadSignature | FernetError::MalformedToken)
        ));
    }

    #[test]
    fn wrong_key_rejected() {
        let k = FernetKey::from_base64(KEY).unwrap();
        let token = k.encrypt_with(b"x", 1, [3u8; 16]);
        let other = FernetKey::from_base64(b"hX9aQ1m3p5R7tV8yB1nD3fG6jK9mP2sU4wX6zA0cE2g=").unwrap();
        assert!(other.decrypt(token.as_bytes()).is_err());
    }

    #[test]
    fn rejects_bad_key_length() {
        assert!(FernetKey::from_base64(b"too-short").is_err());
    }
}
