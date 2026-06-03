//! Extracting and instructing JSON output from free-form model text.
//!
//! Ports `gravity/structured_output.py`: build a schema instruction and pull a
//! standalone JSON value out of a model response (tolerating `<details>` blocks
//! and markdown fences).

use serde_json::Value;

/// Build the instruction injected to request schema-constrained JSON output.
pub fn build_structured_output_instruction(schema: &Value) -> String {
    let schema_text = serde_json::to_string_pretty(schema).unwrap_or_else(|_| "{}".into());
    format!(
        "Return only valid JSON that matches this schema exactly.\n\
Do not wrap the JSON in markdown fences.\n\
Do not include any prose before or after the JSON.\n\
JSON schema:\n{schema_text}"
    )
}

/// Extract a standalone JSON value from model text.
///
/// Strips `<details>` blocks and markdown fences, then scans for the first `{`
/// or `[` that begins a complete JSON value with no trailing content — matching
/// the Python `extract_json_payload` behavior. Returns `None` if none is found.
pub fn extract_json_payload(text: &str) -> Option<Value> {
    let without_details = strip_details(text);
    let mut stripped = without_details.trim();

    // Strip a leading markdown fence (``` or ```json) and trailing ```.
    let unfenced;
    if stripped.starts_with("```") {
        let mut lines: Vec<&str> = stripped.split('\n').collect();
        if !lines.is_empty() {
            lines.remove(0); // ```json / ```
        }
        if lines.last().map(|l| l.trim()) == Some("```") {
            lines.pop();
        }
        unfenced = lines.join("\n");
        stripped = unfenced.trim();
    }

    let bytes = stripped.as_bytes();
    for (idx, &c) in bytes.iter().enumerate() {
        if c != b'{' && c != b'[' {
            continue;
        }
        let slice = &stripped[idx..];
        let mut iter = serde_json::Deserializer::from_str(slice).into_iter::<Value>();
        if let Some(Ok(value)) = iter.next() {
            let consumed = iter.byte_offset();
            if slice[consumed..].trim().is_empty() {
                return Some(value);
            }
        }
    }
    None
}

/// Remove `<details>…</details>` blocks (closed or trailing-unclosed).
fn strip_details(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    let mut rest = text;
    while let Some(start) = rest.find("<details") {
        out.push_str(&rest[..start]);
        let after = &rest[start..];
        match after.find("</details>") {
            Some(end) => rest = &after[end + "</details>".len()..],
            None => return out, // unclosed → drop to end
        }
    }
    out.push_str(rest);
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn extracts_plain_object() {
        assert_eq!(extract_json_payload(r#"{"a":1}"#), Some(json!({"a":1})));
    }

    #[test]
    fn extracts_from_markdown_fence() {
        let t = "```json\n{\"a\": 1, \"b\": [2,3]}\n```";
        assert_eq!(extract_json_payload(t), Some(json!({"a":1,"b":[2,3]})));
    }

    #[test]
    fn rejects_trailing_prose() {
        // standalone-only: object followed by prose is not accepted at that pos,
        // but a later scan won't find a clean one either.
        assert_eq!(extract_json_payload(r#"{"a":1} then chatter"#), None);
    }

    #[test]
    fn strips_details_block() {
        let t = "<details>hidden {\"x\":9}</details>{\"a\":1}";
        assert_eq!(extract_json_payload(t), Some(json!({"a":1})));
    }

    #[test]
    fn finds_object_after_leading_prose_when_standalone() {
        // leading non-bracket chars are skipped; the trailing value is clean.
        assert_eq!(extract_json_payload("answer: [1,2,3]"), Some(json!([1, 2, 3])));
    }
}
