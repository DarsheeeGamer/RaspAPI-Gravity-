//! Kiro's brace-counting event-stream parser.
//!
//! Ports `parse_event_stream_buffer` from `gravity/kiro/client.py`. Kiro's
//! API-Gateway response is a concatenation of JSON objects (not SSE); this
//! scanner finds each object by its leading key, brace-counts to its end while
//! respecting strings/escapes, and classifies it into a [`KiroEvent`].

use serde_json::Value;

/// A decoded Kiro stream event.
#[derive(Debug, Clone, PartialEq)]
pub enum KiroEvent {
    /// Assistant text content.
    Content(String),
    /// A tool-use announcement.
    ToolUse {
        /// Tool name.
        name: String,
        /// Tool-use id.
        tool_use_id: String,
        /// Argument fragment (may be partial JSON).
        input: String,
        /// Whether the tool block is complete.
        stop: bool,
    },
    /// A streamed tool-input fragment.
    ToolUseInput(String),
    /// Tool-use stop marker.
    ToolUseStop(bool),
    /// Context-usage update.
    ContextUsage(f64),
}

/// The object-start markers Kiro emits, matched in priority order.
const MARKERS: &[&str] = &[
    "{\"content\":",
    "{\"name\":",
    "{\"followupPrompt\":",
    "{\"input\":",
    "{\"stop\":",
    "{\"contextUsagePercentage\":",
];

/// Find the earliest marker at or after `from` (byte index), or `None`.
fn next_object_start(buf: &str, from: usize) -> Option<usize> {
    let hay = &buf[from..];
    MARKERS
        .iter()
        .filter_map(|m| hay.find(m).map(|p| p + from))
        .min()
}

/// Find the byte index of the `}` closing the object that begins at
/// `start`, brace-counting while honoring string literals and escapes.
fn object_end(buf: &str, start: usize) -> Option<usize> {
    let mut brace = 0i32;
    let mut in_string = false;
    let mut escape = false;
    for (offset, ch) in buf[start..].char_indices() {
        if escape {
            escape = false;
            continue;
        }
        match ch {
            '\\' => escape = true,
            '"' => in_string = !in_string,
            '{' if !in_string => brace += 1,
            '}' if !in_string => {
                brace -= 1;
                if brace == 0 {
                    return Some(start + offset);
                }
            }
            _ => {}
        }
    }
    None
}

/// Classify a decoded object into a [`KiroEvent`] (Python's branch order).
fn classify(parsed: &Value) -> Option<KiroEvent> {
    let has = |k: &str| parsed.get(k).map(|v| !v.is_null()).unwrap_or(false);
    let followup = parsed.get("followupPrompt").map(|v| !v.is_null()).unwrap_or(false);

    if has("content") && !followup {
        let content = match parsed.get("content") {
            Some(Value::String(s)) => s.clone(),
            Some(other) => other.to_string(),
            None => String::new(),
        };
        return Some(KiroEvent::Content(content));
    }
    let name = parsed.get("name").and_then(Value::as_str);
    let tool_use_id = parsed.get("toolUseId").and_then(Value::as_str);
    if let (Some(name), Some(tid)) = (name, tool_use_id) {
        let input = match parsed.get("input") {
            Some(Value::String(s)) => s.clone(),
            Some(other) => other.to_string(),
            None => String::new(),
        };
        return Some(KiroEvent::ToolUse {
            name: name.to_owned(),
            tool_use_id: tid.to_owned(),
            input,
            stop: parsed.get("stop").and_then(Value::as_bool).unwrap_or(false),
        });
    }
    if has("input") && name.is_none() {
        let input = match parsed.get("input") {
            Some(Value::String(s)) => s.clone(),
            Some(other) => other.to_string(),
            None => String::new(),
        };
        return Some(KiroEvent::ToolUseInput(input));
    }
    if has("stop") && !has("contextUsagePercentage") {
        return Some(KiroEvent::ToolUseStop(
            parsed.get("stop").and_then(Value::as_bool).unwrap_or(false),
        ));
    }
    if let Some(pct) = parsed.get("contextUsagePercentage").and_then(Value::as_f64) {
        return Some(KiroEvent::ContextUsage(pct));
    }
    None
}

/// Parse a complete buffer into events (`parse_event_stream_buffer`).
pub fn parse_event_stream_buffer(buffer: &str) -> Vec<KiroEvent> {
    let (events, _consumed) = parse_stream_buffer(buffer);
    events
}

/// Incremental variant: returns events plus the unconsumed tail (for streaming).
///
/// Mirrors `parse_stream_buffer`, returning `(events, remaining)` where
/// `remaining` is the suffix after the last fully-parsed object.
pub fn parse_stream_buffer(buffer: &str) -> (Vec<KiroEvent>, String) {
    let mut events = Vec::new();
    let mut search_start = 0usize;
    let mut last_consumed = 0usize;

    while let Some(json_start) = next_object_start(buffer, search_start) {
        let Some(json_end) = object_end(buffer, json_start) else {
            break; // incomplete object — keep from json_start as the tail
        };
        let json_str = &buffer[json_start..=json_end];
        if let Ok(parsed) = serde_json::from_str::<Value>(json_str) {
            if let Some(ev) = classify(&parsed) {
                events.push(ev);
            }
        }
        search_start = json_end + 1;
        last_consumed = search_start;
        if search_start >= buffer.len() {
            break;
        }
    }

    let remaining = buffer.get(last_consumed..).unwrap_or("").to_owned();
    (events, remaining)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_concatenated_content_objects() {
        let buf = r#"{"content":"Hello "}{"content":"world"}"#;
        let evs = parse_event_stream_buffer(buf);
        assert_eq!(
            evs,
            vec![
                KiroEvent::Content("Hello ".into()),
                KiroEvent::Content("world".into())
            ]
        );
    }

    #[test]
    fn brace_counting_respects_nested_and_strings() {
        // content holds an escaped quote and a brace inside the string.
        let buf = r#"{"content":"a \"quoted\" { brace"}{"stop":true}"#;
        let evs = parse_event_stream_buffer(buf);
        assert_eq!(evs[0], KiroEvent::Content("a \"quoted\" { brace".into()));
        assert_eq!(evs[1], KiroEvent::ToolUseStop(true));
    }

    #[test]
    fn classifies_tool_use() {
        let buf = r#"{"name":"get_weather","toolUseId":"tu1","input":"{\"city\":\"NYC\"}","stop":false}"#;
        let evs = parse_event_stream_buffer(buf);
        assert_eq!(
            evs[0],
            KiroEvent::ToolUse {
                name: "get_weather".into(),
                tool_use_id: "tu1".into(),
                input: "{\"city\":\"NYC\"}".into(),
                stop: false,
            }
        );
    }

    #[test]
    fn incremental_returns_unconsumed_tail() {
        // second object is incomplete → kept as remaining
        let buf = r#"{"content":"done"}{"content":"par"#;
        let (evs, rest) = parse_stream_buffer(buf);
        assert_eq!(evs, vec![KiroEvent::Content("done".into())]);
        assert_eq!(rest, r#"{"content":"par"#);
    }

    #[test]
    fn context_usage_event() {
        let buf = r#"{"contextUsagePercentage":42.5}"#;
        assert_eq!(parse_event_stream_buffer(buf), vec![KiroEvent::ContextUsage(42.5)]);
    }
}
