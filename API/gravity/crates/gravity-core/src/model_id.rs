//! The `provider/model` identifier.

use crate::capabilities::ProviderName;
use serde::{Deserialize, Serialize};
use std::fmt;
use std::str::FromStr;

/// A parsed `"provider/model"` identifier, e.g. `groq/llama-3.3-70b`.
///
/// Mirrors `ProviderModelId` in `gravity/types.py`. The model component is held
/// as a [`Box<str>`] because it is immutable after construction — this saves the
/// capacity word a `String` would carry.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ModelId {
    /// The provider half.
    pub provider: ProviderName,
    /// The provider-specific model name (never empty).
    pub model: Box<str>,
}

impl ModelId {
    /// Build directly from a provider and model name.
    ///
    /// Returns [`ModelIdParseError::EmptyModel`] if `model` is empty.
    pub fn new(provider: ProviderName, model: impl Into<Box<str>>) -> Result<Self, ModelIdParseError> {
        let model = model.into();
        if model.is_empty() {
            return Err(ModelIdParseError::EmptyModel);
        }
        Ok(ModelId { provider, model })
    }

    /// Parse a `"provider/model"` string.
    ///
    /// ```
    /// use gravity_core::{ModelId, ProviderName};
    /// let id = ModelId::parse("groq/llama-3.3-70b").unwrap();
    /// assert_eq!(id.provider, ProviderName::Groq);
    /// assert_eq!(&*id.model, "llama-3.3-70b");
    /// ```
    pub fn parse(value: &str) -> Result<Self, ModelIdParseError> {
        let (provider, model) = value
            .split_once('/')
            .ok_or(ModelIdParseError::MissingSeparator)?;
        if model.is_empty() {
            return Err(ModelIdParseError::EmptyModel);
        }
        let provider =
            ProviderName::from_str(provider).map_err(|_| ModelIdParseError::UnknownProvider(provider.to_owned()))?;
        Ok(ModelId {
            provider,
            model: model.into(),
        })
    }

    /// The canonical `"provider/model"` string.
    pub fn full_name(&self) -> String {
        format!("{}/{}", self.provider.as_str(), self.model)
    }
}

impl fmt::Display for ModelId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}/{}", self.provider.as_str(), self.model)
    }
}

impl FromStr for ModelId {
    type Err = ModelIdParseError;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        ModelId::parse(s)
    }
}

/// Why a `"provider/model"` string failed to parse.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum ModelIdParseError {
    /// No `/` separator was present.
    #[error("model ids must look like 'provider/model_id'")]
    MissingSeparator,
    /// The model component was empty.
    #[error("model component must not be empty")]
    EmptyModel,
    /// The provider component is not a known provider.
    #[error("unknown provider: {0:?}")]
    UnknownProvider(String),
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_valid_ids() {
        let id = ModelId::parse("openai_api/gpt-4o").unwrap();
        assert_eq!(id.provider, ProviderName::OpenaiApi);
        assert_eq!(&*id.model, "gpt-4o");
        assert_eq!(id.full_name(), "openai_api/gpt-4o");
    }

    #[test]
    fn rejects_malformed_ids() {
        assert_eq!(ModelId::parse("nope").unwrap_err(), ModelIdParseError::MissingSeparator);
        assert_eq!(ModelId::parse("groq/").unwrap_err(), ModelIdParseError::EmptyModel);
        assert!(matches!(
            ModelId::parse("bogus/x").unwrap_err(),
            ModelIdParseError::UnknownProvider(_)
        ));
    }

    #[test]
    fn model_with_slashes_keeps_everything_after_first_slash() {
        let id = ModelId::parse("openrouter/meta-llama/llama-3").unwrap();
        assert_eq!(id.provider, ProviderName::Openrouter);
        assert_eq!(&*id.model, "meta-llama/llama-3");
    }
}
