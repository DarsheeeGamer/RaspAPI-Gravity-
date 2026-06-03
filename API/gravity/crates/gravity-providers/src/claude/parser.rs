//! Accumulates claude.ai SSE events into a parsed response.
//!
//! Mirrors `accumulate_events` in `gravity/claude/parser.py`.

use serde_json::Value;

/// A tool call assembled from `content_block_start` + `input_json_delta` events.
#[derive(Debug, Clone, Default)]
pub struct ClaudeToolCall {
    /// Tool-use id.
    pub id: String,
    /// Tool name.
    pub name: String,
    /// Fully-formed input object (when sent eagerly).
    pub input: Value,
    /// Streamed partial JSON, concatenated across deltas.
    pub partial_json: String,
}

impl ClaudeToolCall {
    /// Resolve final arguments: prefer `input`, else parse `partial_json`.
    pub fn arguments(&self) -> serde_json::Map<String, Value> {
        if let Value::Object(map) = &self.input {
            if !map.is_empty() {
                return map.clone();
            }
        }
        if !self.partial_json.is_empty() {
            if let Ok(Value::Object(map)) = serde_json::from_str(&self.partial_json) {
                return map;
            }
        }
        serde_json::Map::new()
    }
}

/// The accumulated result of a Claude completion stream.
#[derive(Debug, Clone, Default)]
pub struct ClaudeParsed {
    /// Concatenated assistant text.
    pub text: String,
    /// Assistant message UUID (for checkpoint threading).
    pub assistant_message_uuid: Option<String>,
    /// Stop reason.
    pub stop_reason: Option<String>,
    /// Tool calls in first-seen order.
    pub tool_calls: Vec<ClaudeToolCall>,
}

/// State machine that folds events as they arrive (works for both buffered and
/// streaming consumption).
#[derive(Default)]
pub struct Accumulator {
    parsed: ClaudeParsed,
    // index into parsed.tool_calls, keyed by tool id
    order: Vec<String>,
}

impl Accumulator {
    /// New empty accumulator.
    pub fn new() -> Self {
        Accumulator::default()
    }

    /// Apply one decoded SSE `data:` event. Returns any new text delta so a
    /// streaming caller can forward it immediately.
    pub fn push(&mut self, event: &Value) -> Option<String> {
        let ty = event.get("type").and_then(Value::as_str).unwrap_or("");
        match ty {
            "message_start" => {
                if let Some(uuid) = event
                    .get("message")
                    .and_then(|m| m.get("uuid"))
                    .and_then(Value::as_str)
                {
                    self.parsed.assistant_message_uuid = Some(uuid.to_owned());
                }
                None
            }
            "content_block_start" => {
                let block = event.get("content_block").cloned().unwrap_or(Value::Null);
                if block.get("type").and_then(Value::as_str) == Some("tool_use") {
                    let id = block
                        .get("id")
                        .or_else(|| block.get("tool_use_id"))
                        .and_then(Value::as_str)
                        .unwrap_or("")
                        .to_owned();
                    if !id.is_empty() {
                        self.order.push(id.clone());
                        self.parsed.tool_calls.push(ClaudeToolCall {
                            id,
                            name: block.get("name").and_then(Value::as_str).unwrap_or("").to_owned(),
                            input: block.get("input").cloned().unwrap_or(Value::Null),
                            partial_json: block
                                .get("partial_json")
                                .and_then(Value::as_str)
                                .unwrap_or("")
                                .to_owned(),
                        });
                    }
                }
                None
            }
            "content_block_delta" => {
                let delta = event.get("delta").cloned().unwrap_or(Value::Null);
                match delta.get("type").and_then(Value::as_str) {
                    Some("text_delta") => {
                        let t = delta.get("text").and_then(Value::as_str).unwrap_or("");
                        self.parsed.text.push_str(t);
                        (!t.is_empty()).then(|| t.to_owned())
                    }
                    Some("input_json_delta") | Some("tool_use_delta") => {
                        let tool_id = delta
                            .get("tool_use_id")
                            .or_else(|| delta.get("id"))
                            .and_then(Value::as_str);
                        let pj = delta
                            .get("partial_json")
                            .or_else(|| delta.get("text"))
                            .and_then(Value::as_str);
                        if let (Some(tid), Some(pj)) = (tool_id, pj) {
                            if let Some(tc) = self.parsed.tool_calls.iter_mut().find(|t| t.id == tid) {
                                tc.partial_json.push_str(pj);
                            }
                        }
                        None
                    }
                    _ => None,
                }
            }
            "message_delta" => {
                if let Some(sr) = event
                    .get("delta")
                    .and_then(|d| d.get("stop_reason"))
                    .and_then(Value::as_str)
                {
                    self.parsed.stop_reason = Some(sr.to_owned());
                }
                None
            }
            "message_stop" => {
                if self.parsed.stop_reason.is_none() {
                    if let Some(sr) = event.get("stop_reason").and_then(Value::as_str) {
                        self.parsed.stop_reason = Some(sr.to_owned());
                    }
                }
                None
            }
            _ => None,
        }
    }

    /// Finalize.
    pub fn finish(self) -> ClaudeParsed {
        self.parsed
    }

    /// Borrow the in-progress result.
    #[allow(dead_code)]
    pub fn parsed(&self) -> &ClaudeParsed {
        &self.parsed
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn accumulates_text_and_tool_calls() {
        let events = vec![
            json!({"type":"message_start","message":{"uuid":"m-1"}}),
            json!({"type":"content_block_delta","delta":{"type":"text_delta","text":"Hel"}}),
            json!({"type":"content_block_delta","delta":{"type":"text_delta","text":"lo"}}),
            json!({"type":"content_block_start","content_block":{"type":"tool_use","id":"t1","name":"get_weather"}}),
            json!({"type":"content_block_delta","delta":{"type":"input_json_delta","tool_use_id":"t1","partial_json":"{\"city\":"}}),
            json!({"type":"content_block_delta","delta":{"type":"input_json_delta","tool_use_id":"t1","partial_json":"\"NYC\"}"}}),
            json!({"type":"message_delta","delta":{"stop_reason":"tool_use"}}),
            json!({"type":"message_stop"}),
        ];
        let mut acc = Accumulator::new();
        let mut deltas = String::new();
        for e in &events {
            if let Some(d) = acc.push(e) {
                deltas.push_str(&d);
            }
        }
        let parsed = acc.finish();
        assert_eq!(parsed.text, "Hello");
        assert_eq!(deltas, "Hello");
        assert_eq!(parsed.assistant_message_uuid.as_deref(), Some("m-1"));
        assert_eq!(parsed.stop_reason.as_deref(), Some("tool_use"));
        assert_eq!(parsed.tool_calls.len(), 1);
        let args = parsed.tool_calls[0].arguments();
        assert_eq!(args.get("city").unwrap(), "NYC");
    }
}
