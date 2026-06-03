//! Connect-RPC protobuf codec for Cursor (and the shared Connect framing used
//! by Windsurf/Antigravity).
//!
//! Ports `gravity/cursor/proto.py` — a hand-rolled, dependency-light protobuf
//! encoder/decoder plus the Connect-RPC frame format
//! (`[flag:u8][len:u32be][payload]`) and a `google.protobuf.Struct`/`Value`
//! encoder for tool arguments. Pure byte manipulation, unit-tested for parity.

use flate2::read::GzDecoder;
use serde_json::Value;
use std::io::Read;

// ── primitive encoders ──────────────────────────────────────────────────────

/// Encode an unsigned LEB128 varint.
pub fn varint(mut v: u64) -> Vec<u8> {
    let mut out = Vec::new();
    while v > 0x7F {
        out.push(((v & 0x7F) | 0x80) as u8);
        v >>= 7;
    }
    out.push(v as u8);
    out
}

/// Varint field (`wire type 0`): tag `(field << 3)` + varint value.
pub fn field_varint(fn_: u32, v: u64) -> Vec<u8> {
    let mut out = varint((fn_ as u64) << 3);
    out.extend(varint(v));
    out
}

/// Length-delimited bytes field (`wire type 2`).
pub fn field_bytes(fn_: u32, b: &[u8]) -> Vec<u8> {
    let mut out = varint(((fn_ as u64) << 3) | 2);
    out.extend(varint(b.len() as u64));
    out.extend_from_slice(b);
    out
}

/// String field.
pub fn field_string(fn_: u32, s: &str) -> Vec<u8> {
    field_bytes(fn_, s.as_bytes())
}

/// Embedded-message field (alias of [`field_bytes`]).
pub fn field_message(fn_: u32, b: &[u8]) -> Vec<u8> {
    field_bytes(fn_, b)
}

/// 64-bit double field (`wire type 1`).
pub fn field_double(fn_: u32, v: f64) -> Vec<u8> {
    let mut out = varint(((fn_ as u64) << 3) | 1);
    out.extend_from_slice(&v.to_le_bytes());
    out
}

// ── decoder ───────────────────────────────────────────────────────────────

/// A decoded protobuf field value.
#[derive(Debug, Clone, PartialEq)]
pub enum Field {
    /// Varint (wire type 0).
    Varint(u64),
    /// Raw bytes (wire types 1, 2, 5).
    Bytes(Vec<u8>),
}

impl Field {
    /// Interpret as a UTF-8 string (lossy).
    pub fn as_string(&self) -> String {
        match self {
            Field::Bytes(b) => String::from_utf8_lossy(b).into_owned(),
            Field::Varint(v) => v.to_string(),
        }
    }

    /// Interpret as a bool.
    #[allow(dead_code)]
    pub fn as_bool(&self) -> bool {
        match self {
            Field::Varint(v) => *v != 0,
            Field::Bytes(b) => b.first().map(|&x| x != 0).unwrap_or(false),
        }
    }

    /// Borrow the bytes, if this is a length-delimited field.
    pub fn bytes(&self) -> Option<&[u8]> {
        match self {
            Field::Bytes(b) => Some(b),
            _ => None,
        }
    }
}

/// Parse a protobuf message into `{field_number: [values]}`.
pub fn decode_fields(data: &[u8]) -> std::collections::BTreeMap<u32, Vec<Field>> {
    let mut out: std::collections::BTreeMap<u32, Vec<Field>> = std::collections::BTreeMap::new();
    let mut pos = 0;
    while pos < data.len() {
        let Some((tag, n)) = read_varint(data, pos) else { break };
        pos += n;
        let fn_ = (tag >> 3) as u32;
        let wt = tag & 7;
        match wt {
            0 => {
                let Some((v, n)) = read_varint(data, pos) else { break };
                pos += n;
                out.entry(fn_).or_default().push(Field::Varint(v));
            }
            1 => {
                if pos + 8 > data.len() {
                    break;
                }
                out.entry(fn_).or_default().push(Field::Bytes(data[pos..pos + 8].to_vec()));
                pos += 8;
            }
            2 => {
                let Some((len, n)) = read_varint(data, pos) else { break };
                pos += n;
                let end = pos + len as usize;
                if end > data.len() {
                    break;
                }
                out.entry(fn_).or_default().push(Field::Bytes(data[pos..end].to_vec()));
                pos = end;
            }
            5 => {
                if pos + 4 > data.len() {
                    break;
                }
                out.entry(fn_).or_default().push(Field::Bytes(data[pos..pos + 4].to_vec()));
                pos += 4;
            }
            _ => break,
        }
    }
    out
}

fn read_varint(data: &[u8], mut pos: usize) -> Option<(u64, usize)> {
    let mut v = 0u64;
    let mut shift = 0;
    let start = pos;
    loop {
        let b = *data.get(pos)?;
        pos += 1;
        v |= ((b & 0x7F) as u64) << shift;
        if b & 0x80 == 0 {
            break;
        }
        shift += 7;
    }
    Some((v, pos - start))
}

// ── BulkEmbed codec ─────────────────────────────────────────────────────────
//
// BulkEmbedRequest  { field 1 = repeated string texts }
// BulkEmbedResponse { field 1 = repeated EmbeddingResponse }
// EmbeddingResponse { field 1 = packed repeated double embedding }

/// Encode a BulkEmbedRequest: repeated string field 1.
pub fn build_bulk_embed_request(texts: &[&str]) -> Vec<u8> {
    let mut out = Vec::new();
    for t in texts {
        out.extend(field_string(1, t));
    }
    out
}

/// Decode a BulkEmbedResponse into a Vec of embedding vectors.
pub fn parse_bulk_embed_response(data: &[u8]) -> Vec<Vec<f64>> {
    let mut result = Vec::new();
    let fields = decode_fields(data);
    for emb_bytes in fields.get(&1).into_iter().flatten() {
        let Some(inner_bytes) = emb_bytes.bytes() else { continue };
        let inner = decode_fields(inner_bytes);
        // field 1 = packed repeated double (little-endian f64)
        let embedding: Vec<f64> = inner
            .get(&1)
            .into_iter()
            .flatten()
            .flat_map(|f| {
                // packed doubles arrive as a single Bytes chunk
                if let Some(b) = f.bytes() {
                    let n = b.len() / 8;
                    (0..n)
                        .map(|i| f64::from_le_bytes(b[i * 8..(i + 1) * 8].try_into().unwrap()))
                        .collect::<Vec<_>>()
                } else {
                    vec![]
                }
            })
            .collect();
        if !embedding.is_empty() {
            result.push(embedding);
        }
    }
    result
}

// ── Connect-RPC frame codec ─────────────────────────────────────────────────

/// Encode a Connect frame: `[flag][len:u32be][payload]`.
pub fn encode_frame(payload: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(5 + payload.len());
    out.push(0x00); // uncompressed
    out.extend_from_slice(&(payload.len() as u32).to_be_bytes());
    out.extend_from_slice(payload);
    out
}

/// Decode concatenated Connect frames into `(flag, payload)` pairs, gunzipping
/// frames whose flag has bit 0 set.
pub fn decode_frames(raw: &[u8]) -> Vec<(u8, Vec<u8>)> {
    let mut frames = Vec::new();
    let mut pos = 0;
    while pos + 5 <= raw.len() {
        let flag = raw[pos];
        let len = u32::from_be_bytes([raw[pos + 1], raw[pos + 2], raw[pos + 3], raw[pos + 4]]) as usize;
        let start = pos + 5;
        let end = start + len;
        if end > raw.len() {
            break;
        }
        let mut payload = raw[start..end].to_vec();
        if flag & 0x01 != 0 {
            let mut dec = GzDecoder::new(&payload[..]);
            let mut buf = Vec::new();
            if dec.read_to_end(&mut buf).is_ok() {
                payload = buf;
            }
        }
        frames.push((flag, payload));
        pos = end;
    }
    frames
}

// ── google.protobuf.Struct / Value ──────────────────────────────────────────

/// Encode a JSON object as a `google.protobuf.Struct`.
pub fn encode_struct(map: &serde_json::Map<String, Value>) -> Vec<u8> {
    let mut result = Vec::new();
    for (k, v) in map {
        let mut entry = field_string(1, k);
        entry.extend(field_message(2, &encode_value(v)));
        result.extend(field_message(1, &entry));
    }
    result
}

fn encode_list(list: &[Value]) -> Vec<u8> {
    let mut out = Vec::new();
    for item in list {
        out.extend(field_message(1, &encode_value(item)));
    }
    out
}

/// Encode a JSON value as a `google.protobuf.Value`.
pub fn encode_value(v: &Value) -> Vec<u8> {
    match v {
        Value::Null => field_varint(1, 0),
        Value::Bool(b) => field_varint(4, *b as u64),
        Value::Number(n) => field_double(2, n.as_f64().unwrap_or(0.0)),
        Value::String(s) => field_string(3, s),
        Value::Object(m) => field_message(5, &encode_struct(m)),
        Value::Array(a) => field_message(6, &encode_list(a)),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn varint_roundtrip_via_decode() {
        // field 1 varint = 300
        let buf = field_varint(1, 300);
        let fields = decode_fields(&buf);
        assert_eq!(fields[&1][0], Field::Varint(300));
    }

    #[test]
    fn string_field_decodes() {
        let buf = field_string(2, "hello");
        let fields = decode_fields(&buf);
        assert_eq!(fields[&2][0].as_string(), "hello");
    }

    #[test]
    fn frame_roundtrip_uncompressed() {
        let payload = b"\x08\x96\x01"; // arbitrary proto bytes
        let framed = encode_frame(payload);
        assert_eq!(framed[0], 0x00);
        let frames = decode_frames(&framed);
        assert_eq!(frames.len(), 1);
        assert_eq!(frames[0].1, payload);
    }

    #[test]
    fn struct_encodes_nested() {
        let obj = json!({ "city": "NYC", "n": 5 });
        let bytes = encode_struct(obj.as_object().unwrap());
        // decodes to repeated field 1 (entries)
        let fields = decode_fields(&bytes);
        assert!(fields.contains_key(&1));
        assert_eq!(fields[&1].len(), 2);
    }
}
