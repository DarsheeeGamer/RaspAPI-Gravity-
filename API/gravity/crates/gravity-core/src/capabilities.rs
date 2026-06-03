//! Provider identity and capability enums.
//!
//! These mirror `gravity/capabilities.py` exactly — the serialized string
//! values are part of the stable wire/FFI contract and must not change.

use enum_map::Enum;
use serde::{Deserialize, Serialize};

/// The set of every provider Gravity can talk to.
///
/// The `snake_case` serialized form (`"openai_api"`, `"gemini_cli"`, …) is the
/// canonical identifier used in `model` strings (`"groq/llama-3.3-70b"`), the
/// accounts file, and the FFI JSON boundary.
///
/// Marked `#[non_exhaustive]`: downstream `match`es must include a wildcard arm
/// so new providers can be added without breaking callers.
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Enum, Serialize, Deserialize,
)]
#[serde(rename_all = "snake_case")]
#[non_exhaustive]
pub enum ProviderName {
    /// Antigravity language-server proxy (FIPS/boring TLS).
    Antigravity,
    /// ChatGPT web (Auth0 device flow + PoW + Turnstile).
    Chatgpt,
    /// Claude web (claude.ai session cookies).
    Claude,
    /// Google Gemini web (gemini.google.com).
    Gemini,
    /// OpenAI Codex (OAuth).
    Codex,
    /// Perplexity web search.
    Perplexity,
    /// AWS Kiro / Q Developer (OAuth device + social).
    Kiro,
    /// Zhipu GLM web (chat.z.ai HMAC signing).
    Glm,
    /// OpenRouter multi-model gateway.
    Openrouter,
    /// Anthropic public API.
    AnthropicApi,
    /// OpenAI public API.
    OpenaiApi,
    /// Groq fast inference.
    Groq,
    /// Mistral API.
    Mistral,
    /// Together AI.
    Together,
    /// DeepSeek API.
    Deepseek,
    /// xAI Grok API.
    Xai,
    /// Cohere API.
    Cohere,
    /// Local Ollama server.
    Ollama,
    /// Alibaba Qwen API.
    Qwen,
    /// MiniMax API.
    Minimax,
    /// Google Vertex AI.
    Vertex,
    /// Codeium Windsurf (Connect-RPC, Go TLS).
    Windsurf,
    /// Cursor IDE (protobuf).
    Cursor,
    /// Google Gemini CLI (OAuth).
    GeminiCli,
    /// LiteLLM proxy.
    Litellm,
    /// Fireworks AI.
    Fireworks,
    /// Cerebras inference.
    Cerebras,
    /// SambaNova cloud.
    Sambanova,
    /// Hyperbolic Labs.
    Hyperbolic,
    /// NVIDIA NIM microservices.
    NvidiaNim,
    /// Local llama.cpp server.
    LlamaCpp,
    /// HuggingFace inference.
    Huggingface,
    /// Local HuggingFace transformers.
    LocalTransformers,
}

impl ProviderName {
    /// Return the canonical `snake_case` identifier (e.g. `"openai_api"`).
    ///
    /// This is the same string used by serde and the FFI boundary.
    pub const fn as_str(self) -> &'static str {
        match self {
            ProviderName::Antigravity => "antigravity",
            ProviderName::Chatgpt => "chatgpt",
            ProviderName::Claude => "claude",
            ProviderName::Gemini => "gemini",
            ProviderName::Codex => "codex",
            ProviderName::Perplexity => "perplexity",
            ProviderName::Kiro => "kiro",
            ProviderName::Glm => "glm",
            ProviderName::Openrouter => "openrouter",
            ProviderName::AnthropicApi => "anthropic_api",
            ProviderName::OpenaiApi => "openai_api",
            ProviderName::Groq => "groq",
            ProviderName::Mistral => "mistral",
            ProviderName::Together => "together",
            ProviderName::Deepseek => "deepseek",
            ProviderName::Xai => "xai",
            ProviderName::Cohere => "cohere",
            ProviderName::Ollama => "ollama",
            ProviderName::Qwen => "qwen",
            ProviderName::Minimax => "minimax",
            ProviderName::Vertex => "vertex",
            ProviderName::Windsurf => "windsurf",
            ProviderName::Cursor => "cursor",
            ProviderName::GeminiCli => "gemini_cli",
            ProviderName::Litellm => "litellm",
            ProviderName::Fireworks => "fireworks",
            ProviderName::Cerebras => "cerebras",
            ProviderName::Sambanova => "sambanova",
            ProviderName::Hyperbolic => "hyperbolic",
            ProviderName::NvidiaNim => "nvidia_nim",
            ProviderName::LlamaCpp => "llama_cpp",
            ProviderName::Huggingface => "huggingface",
            ProviderName::LocalTransformers => "local_transformers",
        }
    }

    /// Iterate over every provider variant, in declaration order.
    pub fn all() -> impl Iterator<Item = ProviderName> {
        (0..<ProviderName as Enum>::LENGTH).map(<ProviderName as Enum>::from_usize)
    }
}

impl std::fmt::Display for ProviderName {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

impl std::str::FromStr for ProviderName {
    type Err = UnknownProvider;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        ProviderName::all()
            .find(|p| p.as_str() == s)
            .ok_or_else(|| UnknownProvider(s.to_owned()))
    }
}

/// Error returned when a provider identifier doesn't match any known provider.
#[derive(Debug, Clone, thiserror::Error)]
#[error("unknown provider: {0:?}")]
pub struct UnknownProvider(pub String);

/// How a provider exposes tool / function calling.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ToolSupportMode {
    /// Native function-calling API (OpenAI tools, Anthropic tool_use, …).
    Native,
    /// No native support — tools are emulated via a JSON instruction protocol.
    JsonEmulated,
    /// Tools are not supported at all.
    None,
}

/// How conversation history is maintained across turns.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum HistoryMode {
    /// Full history replayed on every request.
    Stateless,
    /// Provider holds remote conversation state (checkpointed).
    Stateful,
    /// Pick automatically from the provider's capabilities.
    Auto,
}

/// Whether a provider keeps a server-side session that must be checkpointed.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RemoteSessionMode {
    /// No remote session.
    None,
    /// A remote session is required for every conversation.
    Required,
    /// A remote session may be used but is not required.
    Optional,
}

/// How a provider produces schema-constrained (structured) output.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum StructuredOutputMode {
    /// Native JSON-schema / response-format support.
    Native,
    /// Emulated by injecting a JSON-schema instruction into the prompt.
    JsonInstruction,
    /// Not supported.
    None,
}

/// Static description of what a provider can do.
///
/// Mirrors `ProviderCapabilities` in `gravity/abcs.py`. Returned by each
/// provider adapter's `capabilities()` method and surfaced through
/// `gravity_list_providers`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ProviderCapabilities {
    /// Which provider this describes.
    pub provider: ProviderName,
    /// Tool-calling support level.
    pub tool_support: ToolSupportMode,
    /// Structured-output support level.
    pub structured_output: StructuredOutputMode,
    /// Remote-session requirement.
    pub remote_session_mode: RemoteSessionMode,
    /// Whether a dedicated system prompt is honored.
    pub supports_system_prompt: bool,
    /// Whether the provider can retain history server-side.
    pub supports_stateful_history: bool,
    /// Whether a remote checkpoint must be created before chatting.
    pub requires_remote_checkpoint: bool,
    /// Default history mode when the request doesn't specify one.
    #[serde(default = "default_history_mode")]
    pub default_history_mode: HistoryMode,
    /// Whether image generation is supported.
    #[serde(default)]
    pub supports_image_generation: bool,
}

fn default_history_mode() -> HistoryMode {
    HistoryMode::Auto
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn provider_name_roundtrips_through_serde() {
        for p in ProviderName::all() {
            let json = serde_json::to_string(&p).unwrap();
            let back: ProviderName = serde_json::from_str(&json).unwrap();
            assert_eq!(p, back);
            // serde form matches as_str()
            assert_eq!(json, format!("{:?}", p.as_str()));
        }
    }

    #[test]
    fn provider_name_from_str_matches_python_values() {
        assert_eq!("openai_api".parse::<ProviderName>().unwrap(), ProviderName::OpenaiApi);
        assert_eq!("gemini_cli".parse::<ProviderName>().unwrap(), ProviderName::GeminiCli);
        assert_eq!("nvidia_nim".parse::<ProviderName>().unwrap(), ProviderName::NvidiaNim);
        assert!("not_a_provider".parse::<ProviderName>().is_err());
    }

    #[test]
    fn all_enumerates_every_variant() {
        assert_eq!(ProviderName::all().count(), 33);
    }
}
