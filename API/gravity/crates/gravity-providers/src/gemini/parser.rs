//! Gemini web response frame parser.
//!
//! Ports `gravity/gemini/parser.py`. The `gemini.google.com` batchexecute
//! response is a sequence of **length-prefixed frames** where the length counts
//! **UTF-16 code units** (not bytes or Unicode scalars), preceded by a `)]}'`
//! sentinel. Each frame is a JSON array; the assistant text lives at a fixed
//! nested-index path.

use serde_json::Value;

/// Count how many `chars` of `s[start..]` make up exactly `utf16_units` UTF-16
/// code units (stopping early if the next char would overshoot).
///
/// Returns `(char_count, units_consumed)`. Mirrors
/// `_get_char_count_for_utf16_units`.
fn chars_for_utf16_units(chars: &[char], start: usize, utf16_units: usize) -> (usize, usize) {
    let mut count = 0;
    let mut units = 0;
    while units < utf16_units && start + count < chars.len() {
        let ch = chars[start + count];
        let char_units = if (ch as u32) > 0xFFFF { 2 } else { 1 };
        if units + char_units > utf16_units {
            break;
        }
        units += char_units;
        count += 1;
    }
    (count, units)
}

/// Parse the length-prefixed frames of a response body.
///
/// Returns the flattened JSON frames plus any unconsumed tail (for streaming).
pub fn parse_response_frames(content: &str) -> (Vec<Value>, String) {
    let chars: Vec<char> = content.chars().collect();
    let total = chars.len();
    let mut pos = 0usize;
    let mut frames = Vec::new();

    while pos < total {
        // Skip leading whitespace.
        while pos < total && chars[pos].is_whitespace() {
            pos += 1;
        }
        if pos >= total {
            break;
        }
        // Match the `\d+\n` length marker at `pos`.
        let Some((digits_len, length)) = match_length_marker(&chars, pos) else {
            break;
        };
        // Python: `start_content = match.start() + len(digits)` → points at `\n`.
        // The length counts FROM that `\n` through the frame body AND a trailing
        // `\n`, so `length = 1 (leading \n) + frame_chars + 1 (trailing \n)`.
        // We must start counting at the `\n`, not after it.
        let start_content = pos + digits_len; // position of the `\n` after digits
        let (char_count, units_found) = chars_for_utf16_units(&chars, start_content, length);
        if units_found < length {
            break; // incomplete frame
        }
        let end = start_content + char_count;
        let chunk: String = chars[start_content..end].iter().collect();
        let chunk = chunk.trim(); // strip leading \n and trailing \n
        pos = end;
        if chunk.is_empty() {
            continue;
        }
        match serde_json::from_str::<Value>(chunk) {
            Ok(Value::Array(arr)) => frames.extend(arr),
            Ok(other) => frames.push(other),
            Err(_) => continue,
        }
    }

    let remaining: String = chars[pos.min(total)..].iter().collect();
    (frames, remaining)
}

/// If `chars[pos..]` begins with `\d+` followed by `\n`, return
/// `(digit_count, parsed_length)`.
fn match_length_marker(chars: &[char], pos: usize) -> Option<(usize, usize)> {
    let mut i = pos;
    while i < chars.len() && chars[i].is_ascii_digit() {
        i += 1;
    }
    let digit_count = i - pos;
    if digit_count == 0 || i >= chars.len() || chars[i] != '\n' {
        return None;
    }
    let length: usize = chars[pos..i].iter().collect::<String>().parse().ok()?;
    Some((digit_count, length))
}

/// Strip the `)]}'` sentinel and parse, falling back to plain JSON.
pub fn extract_json_from_response(text: &str) -> Vec<Value> {
    let content = text.strip_prefix(")]}'").unwrap_or(text);
    let content = content.trim_start();
    let (frames, _) = parse_response_frames(content);
    if !frames.is_empty() {
        return frames;
    }
    match serde_json::from_str::<Value>(content) {
        Ok(Value::Array(a)) => a,
        Ok(other) => vec![other],
        Err(_) => Vec::new(),
    }
}

/// Navigate a nested JSON value by an index/key path (`get_nested_value`).
pub fn nested<'a>(mut current: &'a Value, path: &[NestIndex]) -> Option<&'a Value> {
    for key in path {
        current = match key {
            NestIndex::Idx(i) => {
                let arr = current.as_array()?;
                let len = arr.len() as i64;
                let idx = if *i < 0 { len + *i } else { *i };
                if idx < 0 || idx >= len {
                    return None;
                }
                &arr[idx as usize]
            }
            NestIndex::Key(k) => current.get(*k)?,
        };
    }
    (!current.is_null()).then_some(current)
}

/// A path step for [`nested`]: array index (possibly negative) or object key.
#[derive(Debug, Clone, Copy)]
pub enum NestIndex {
    /// Array index (negative counts from the end).
    Idx(i64),
    /// Object key.
    Key(&'static str),
}

/// The text + rcid extracted from a Gemini response.
#[derive(Debug, Default, Clone)]
pub struct GeminiParsed {
    /// Assistant text.
    pub text: String,
    /// Reply candidate id (for threading).
    pub rcid: Option<String>,
    /// Conversation metadata (`[c_id, r_id, ...]`).
    pub metadata: Vec<Option<String>>,
    /// Opaque session continuation state from `part_json[2]["26"]`.
    ///
    /// Must be passed in `inner[2][9]` of the next request to maintain
    /// server-side conversation context. Discovered via HAR analysis.
    pub state_token: Option<String>,
}

/// Extract text/rcid/metadata/state_token from parsed frames.
///
/// Mirrors `parse_gemini_response_parts`. The state token is in `part_json[2]["26"]`
/// (discovered via HAR: required in `inner[2][9]` of the subsequent request
/// for server-side stateful conversation threading).
pub fn parse_parts(parts: &[Value]) -> GeminiParsed {
    let mut out = GeminiParsed::default();
    for part in parts {
        let Some(inner_str) = nested(part, &[NestIndex::Idx(2)]).and_then(Value::as_str) else {
            continue;
        };
        let Ok(part_json) = serde_json::from_str::<Value>(inner_str) else {
            continue;
        };
        // part_json[1] = [c_id, r_id, ...] conversation metadata
        let meta = nested(&part_json, &[NestIndex::Idx(1)]).cloned();
        if let Some(Value::Array(m)) = &meta {
            out.metadata = m
                .iter()
                .map(|v| if v.is_null() { None } else { Some(value_to_string(v)) })
                .collect();
        }
        // part_json[2] = object with keyed fields; "26" = opaque state token
        if let Some(obj) = nested(&part_json, &[NestIndex::Idx(2)]) {
            if let Some(tok) = obj.get("26").and_then(Value::as_str) {
                if !tok.is_empty() {
                    out.state_token = Some(tok.to_owned());
                }
            }
        }
        // part_json[4] = candidates array
        let candidates = nested(&part_json, &[NestIndex::Idx(4)]).cloned();
        let Some(Value::Array(candidates)) = &candidates else {
            continue;
        };
        for cand in candidates {
            if let Some(rcid) = nested(cand, &[NestIndex::Idx(0)]).and_then(Value::as_str) {
                if !rcid.is_empty() {
                    out.rcid = Some(rcid.to_owned());
                }
            }
            if let Some(t) = nested(cand, &[NestIndex::Idx(1), NestIndex::Idx(0)]).and_then(Value::as_str) {
                if !t.is_empty() {
                    out.text = strip_googleusercontent(t);
                }
            }
        }
    }
    out
}

fn value_to_string(v: &Value) -> String {
    match v {
        Value::String(s) => s.clone(),
        other => other.to_string(),
    }
}

/// Remove `http://googleusercontent.com/<word>/<digits>\n*` artifacts.
fn strip_googleusercontent(text: &str) -> String {
    // Lightweight manual scan (avoids a regex dependency).
    let mut out = String::with_capacity(text.len());
    let bytes = text.as_bytes();
    let mut i = 0;
    let marker = "http://googleusercontent.com/";
    while i < text.len() {
        if text[i..].starts_with(marker) {
            let mut j = i + marker.len();
            // skip \w+
            while j < bytes.len() && (bytes[j].is_ascii_alphanumeric() || bytes[j] == b'_') {
                j += 1;
            }
            // require /
            if j < bytes.len() && bytes[j] == b'/' {
                j += 1;
                // skip \d+
                let digits_start = j;
                while j < bytes.len() && bytes[j].is_ascii_digit() {
                    j += 1;
                }
                if j > digits_start {
                    // skip trailing newlines
                    while j < bytes.len() && bytes[j] == b'\n' {
                        j += 1;
                    }
                    i = j;
                    continue;
                }
            }
        }
        let ch = text[i..].chars().next().unwrap();
        out.push(ch);
        i += ch.len_utf8();
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_length_prefixed_frames() {
        // Length counts UTF-16 units from the `\n` (inclusive).
        // `\n` + `[1,2]` = 1 + 5 = 6 units.
        let body = "6\n[1,2]";
        let (frames, rest) = parse_response_frames(body);
        assert_eq!(frames, vec![Value::from(1), Value::from(2)]);
        assert_eq!(rest, "");
    }

    #[test]
    fn utf16_length_counts_surrogate_pairs() {
        // "😀" is 1 char but 2 UTF-16 units; JSON string "😀" = quote+emoji+quote
        // = 1+2+1 = 4 UTF-16 units.  Plus leading `\n` = 5 total.
        let body = "5\n\"😀\"";
        let (frames, _) = parse_response_frames(body);
        assert_eq!(frames, vec![Value::from("😀")]);
    }

    #[test]
    fn incomplete_frame_left_in_tail() {
        let body = "20\n[1,2]"; // claims 20 units; \n+[1,2] = 6 < 20 → incomplete
        let (frames, rest) = parse_response_frames(body);
        assert!(frames.is_empty());
        assert_eq!(rest, body);
    }

    #[test]
    fn strips_sentinel_prefix() {
        // `\n` + `[1,2,3]` = 1 + 7 = 8 UTF-16 units.
        let frames = extract_json_from_response(")]}'\n8\n[1,2,3]");
        assert_eq!(frames, vec![Value::from(1), Value::from(2), Value::from(3)]);
    }

    #[test]
    fn nested_indexing_with_negatives() {
        let v: Value = serde_json::from_str(r#"[["a","b","c"]]"#).unwrap();
        assert_eq!(nested(&v, &[NestIndex::Idx(0), NestIndex::Idx(-1)]).unwrap(), "c");
        assert!(nested(&v, &[NestIndex::Idx(5)]).is_none());
    }

    #[test]
    fn extracts_text_and_rcid_from_parts() {
        let inner = serde_json::json!([null, ["c_id", "r_id"], null, null, [["rc_123", ["Hello from Gemini"]]]]);
        let part = serde_json::json!(["wrb.fr", null, inner.to_string()]);
        let parsed = parse_parts(&[part]);
        assert_eq!(parsed.text, "Hello from Gemini");
        assert_eq!(parsed.rcid.as_deref(), Some("rc_123"));
        assert_eq!(parsed.metadata, vec![Some("c_id".to_string()), Some("r_id".to_string())]);
        assert_eq!(parsed.state_token, None); // no "26" key
    }

    #[test]
    fn extracts_state_token_from_key_26() {
        // HAR-verified: state token lives at part_json[2]["26"] in a separate frame.
        let inner = serde_json::json!([
            null,
            ["c_abc", "r_def"],
            {"26": "AwAAAAAAAAAQwBHO-LzoF6test", "44": true}
        ]);
        let part = serde_json::json!(["wrb.fr", null, inner.to_string()]);
        let parsed = parse_parts(&[part]);
        assert_eq!(parsed.state_token.as_deref(), Some("AwAAAAAAAAAQwBHO-LzoF6test"));
        assert_eq!(parsed.metadata, vec![Some("c_abc".to_string()), Some("r_def".to_string())]);
    }

    #[test]
    fn strips_googleusercontent_artifacts() {
        // The artifact (and its trailing newline) is removed; surrounding text
        // and the preceding space are preserved, matching the Python regex.
        let s = strip_googleusercontent("before http://googleusercontent.com/image/0\nafter");
        assert_eq!(s, "before after");
    }
}
