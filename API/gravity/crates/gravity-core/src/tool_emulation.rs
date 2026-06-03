//! JSON tool-call emulation for providers without native function calling.
//!
//! Ports `gravity/tool_emulation.py`: instruct the model to reply with a strict
//! JSON envelope (`{"type":"tool_call",...}` or `{"type":"final",...}`), and
//! parse that envelope back. Used by the client's JSON-emulated AFC loop for
//! providers whose `tool_support` is [`crate::ToolSupportMode::JsonEmulated`].

use crate::structured_output::extract_json_payload;
use crate::tool::ToolDefinition;
use serde_json::{Map, Value};

/// A parsed tool envelope.
#[derive(Debug, Clone, PartialEq)]
pub enum ToolEnvelope {
    /// The model requested a tool call.
    ToolCall {
        /// Tool name.
        name: String,
        /// Arguments object.
        arguments: Map<String, Value>,
    },
    /// The model produced its final answer.
    Final {
        /// Final content (string or structured value).
        content: Value,
    },
}

/// Build the per-turn JSON-tool-mode instruction listing the available tools.
pub fn build_json_tool_instruction(tools: &[ToolDefinition], final_schema: Option<&Value>) -> String {
    let specs: Vec<Value> = tools
        .iter()
        .map(|t| {
            serde_json::json!({
                "name": t.name,
                "description": t.description,
                "input_schema": t.input_schema,
            })
        })
        .collect();
    let specs_text = serde_json::to_string_pretty(&Value::Array(specs)).unwrap_or_else(|_| "[]".into());
    let final_contract = if final_schema.is_some() {
        r#"Return {"type":"final","content": <JSON matching the final schema>}."#
    } else {
        r#"Return {"type":"final","content":"<your final answer>"}."#
    };

    let mut lines = vec![
        "You are operating in Gravity JSON tool mode.".to_string(),
        "Gravity will execute tool calls for you and send the tool result back.".to_string(),
        "You do have access to the tools listed below.".to_string(),
        "Never say that you cannot access tools.".to_string(),
        "Return exactly one JSON object and nothing else.".to_string(),
        "Do not output prose, explanations, analysis, or markdown.".to_string(),
        "In this session, raw JSON is the only correct output format.".to_string(),
        "Available tools:".to_string(),
        specs_text,
        r#"If you need a tool, return {"type":"tool_call","tool":{"name":"<tool name>","arguments":{...}}}."#.to_string(),
        final_contract.to_string(),
        "Only call one tool at a time.".to_string(),
        "Do not use markdown fences.".to_string(),
        r#"Example tool call: {"type":"tool_call","tool":{"name":"example_tool","arguments":{"x":1}}}"#.to_string(),
        r#"Example final: {"type":"final","content":"done"}"#.to_string(),
    ];
    if let Some(schema) = final_schema {
        lines.push("Final content schema:".to_string());
        lines.push(serde_json::to_string_pretty(schema).unwrap_or_default());
    }
    lines.join("\n")
}

/// Parse a model response into a [`ToolEnvelope`], or `None` if it isn't a valid
/// envelope (caller treats that as plain text).
pub fn parse_tool_envelope(text: &str) -> Option<ToolEnvelope> {
    let payload = extract_json_payload(text)?;
    let obj = payload.as_object()?;
    match obj.get("type").and_then(Value::as_str)? {
        "tool_call" => {
            let tool = obj.get("tool")?.as_object()?;
            let name = tool.get("name").and_then(Value::as_str)?.trim().to_string();
            if name.is_empty() {
                return None;
            }
            let arguments = tool.get("arguments").and_then(Value::as_object).cloned().unwrap_or_default();
            Some(ToolEnvelope::ToolCall { name, arguments })
        }
        "final" => {
            let content = obj.get("content")?.clone();
            Some(ToolEnvelope::Final { content })
        }
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn tool() -> ToolDefinition {
        ToolDefinition {
            name: "get_weather".into(),
            description: "weather".into(),
            input_schema: Map::new(),
        }
    }

    #[test]
    fn instruction_lists_tools_and_contract() {
        let i = build_json_tool_instruction(&[tool()], None);
        assert!(i.contains("get_weather"));
        assert!(i.contains(r#"{"type":"final","content":"<your final answer>"}"#));
        assert!(i.contains("JSON tool mode"));
    }

    #[test]
    fn parses_tool_call_envelope() {
        let t = r#"{"type":"tool_call","tool":{"name":"get_weather","arguments":{"city":"NYC"}}}"#;
        match parse_tool_envelope(t).unwrap() {
            ToolEnvelope::ToolCall { name, arguments } => {
                assert_eq!(name, "get_weather");
                assert_eq!(arguments["city"], "NYC");
            }
            _ => panic!("expected tool_call"),
        }
    }

    #[test]
    fn parses_final_envelope() {
        assert_eq!(
            parse_tool_envelope(r#"{"type":"final","content":"done"}"#),
            Some(ToolEnvelope::Final { content: json!("done") })
        );
    }

    #[test]
    fn non_envelope_is_none() {
        assert_eq!(parse_tool_envelope("just some prose"), None);
        assert_eq!(parse_tool_envelope(r#"{"foo":1}"#), None);
    }
}
