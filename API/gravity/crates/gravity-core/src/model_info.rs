//! Model-discovery result type.
//!
//! [`ModelInfo`] is what a provider's model-listing endpoint returns: the
//! request-usable model id plus optional human metadata (display name,
//! description, capability tags). Reverse-engineered web providers (Cursor,
//! ChatGPT, Gemini, GLM) fetch these dynamically from the same endpoints their
//! web UIs use to populate the model picker; API providers read `GET /models`.

use serde::{Deserialize, Serialize};

/// A single model a provider exposes.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ModelInfo {
    /// The id used in a [`crate::ChatRequest`] (e.g. `"gpt-4o"`, `"claude-sonnet-4.6"`).
    pub id: String,
    /// Human-friendly name from the provider, if any (e.g. `"GPT-4o"`).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub display_name: Option<String>,
    /// Provider-supplied description, if any.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    /// Capability/category tags from the provider (e.g. `["thinking"]`).
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub tags: Vec<String>,
}

impl ModelInfo {
    /// A model with just an id and no metadata.
    pub fn bare(id: impl Into<String>) -> Self {
        ModelInfo {
            id: id.into(),
            display_name: None,
            description: None,
            tags: Vec::new(),
        }
    }

    /// A model with an id and a display name.
    pub fn named(id: impl Into<String>, display_name: impl Into<String>) -> Self {
        ModelInfo {
            id: id.into(),
            display_name: Some(display_name.into()),
            description: None,
            tags: Vec::new(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bare_omits_optional_fields_in_json() {
        let m = ModelInfo::bare("gpt-4o");
        let json = serde_json::to_string(&m).unwrap();
        assert_eq!(json, r#"{"id":"gpt-4o"}"#);
    }

    #[test]
    fn named_serializes_display_name() {
        let m = ModelInfo::named("gpt-4o", "GPT-4o");
        let v: serde_json::Value = serde_json::to_value(&m).unwrap();
        assert_eq!(v["display_name"], "GPT-4o");
        assert!(v.get("description").is_none());
    }
}
