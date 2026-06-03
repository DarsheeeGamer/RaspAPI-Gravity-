//! Serialize [`bytes::Bytes`] as standard (padded) base64.
//!
//! Used by [`crate::InputFile`] so binary attachments round-trip through the
//! JSON FFI boundary without storing a second base64 copy in memory.

use base64::{engine::general_purpose::STANDARD, Engine as _};
use bytes::Bytes;
use serde::{Deserialize, Deserializer, Serializer};

/// Serialize raw bytes as a base64 string.
pub fn serialize<S: Serializer>(bytes: &Bytes, s: S) -> Result<S::Ok, S::Error> {
    s.serialize_str(&STANDARD.encode(bytes))
}

/// Deserialize a base64 string into [`Bytes`].
pub fn deserialize<'de, D: Deserializer<'de>>(d: D) -> Result<Bytes, D::Error> {
    let encoded = <&str>::deserialize(d)?;
    STANDARD
        .decode(encoded)
        .map(Bytes::from)
        .map_err(serde::de::Error::custom)
}
