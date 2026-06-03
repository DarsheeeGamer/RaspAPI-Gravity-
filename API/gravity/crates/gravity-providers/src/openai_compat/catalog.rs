//! Concrete OpenAI-compatible provider configurations.
//!
//! Each provider is a unit struct implementing [`OaiConfig`] — base URL,
//! capabilities, default impersonation, and a (best-effort) static model list.
//! Generated with the `oai_providers!` macro to keep each provider a single
//! readable line. Mirrors the per-provider `__init__.py` files in the Python
//! package.

use super::provider::{OaiConfig, OpenAiCompatProvider};
use crate::provider::Provider;
use gravity_core::{
    HistoryMode, ProviderCapabilities, ProviderName, RemoteSessionMode, StructuredOutputMode,
    ToolSupportMode,
};
use std::sync::Arc;

/// Standard stateless capabilities for an OpenAI-compatible provider.
fn standard_caps(provider: ProviderName, tool_support: ToolSupportMode) -> ProviderCapabilities {
    ProviderCapabilities {
        provider,
        tool_support,
        structured_output: StructuredOutputMode::JsonInstruction,
        remote_session_mode: RemoteSessionMode::None,
        supports_system_prompt: true,
        supports_stateful_history: false,
        requires_remote_checkpoint: false,
        default_history_mode: HistoryMode::Stateless,
        supports_image_generation: false,
    }
}

macro_rules! oai_providers {
    ($(
        $variant:ident => $struct:ident {
            base: $base:expr,
            tools: $tools:expr
            $(, profile: $profile:expr)?
            $(, models: [$($m:expr),* $(,)?])?
        }
    ),* $(,)?) => {
        $(
            /// OpenAI-compatible config for this provider.
            pub struct $struct;
            impl OaiConfig for $struct {
                fn provider(&self) -> ProviderName { ProviderName::$variant }
                fn capabilities(&self) -> ProviderCapabilities {
                    standard_caps(ProviderName::$variant, $tools)
                }
                fn default_base_url(&self) -> &str { $base }
                $(fn default_profile(&self) -> &'static str { $profile })?
                fn models(&self) -> Vec<String> {
                    let v: Vec<String> = vec![ $($($m.to_string()),*)? ];
                    v
                }
            }
        )*

        /// Construct every OpenAI-compatible provider adapter.
        pub fn all() -> Vec<Arc<dyn Provider>> {
            vec![
                $( Arc::new(OpenAiCompatProvider::new($struct)) as Arc<dyn Provider>, )*
            ]
        }
    };
}

use ToolSupportMode::{JsonEmulated as Emu, Native};

oai_providers! {
    Groq => Groq {
        base: "https://api.groq.com/openai/v1", tools: Native,
        models: ["llama-3.3-70b-versatile", "llama-3.1-8b-instant", "openai/gpt-oss-120b"]
    },
    OpenaiApi => OpenaiApi {
        base: "https://api.openai.com/v1", tools: Native,
        models: ["gpt-4o", "gpt-4o-mini", "gpt-4.1", "o3", "o4-mini"]
    },
    Mistral => Mistral {
        base: "https://api.mistral.ai/v1", tools: Native,
        models: ["mistral-large-latest", "mistral-small-latest", "codestral-latest"]
    },
    Deepseek => Deepseek {
        base: "https://api.deepseek.com/v1", tools: Native,
        models: ["deepseek-chat", "deepseek-reasoner"]
    },
    Xai => Xai {
        base: "https://api.x.ai/v1", tools: Native,
        models: ["grok-4", "grok-3", "grok-3-mini"]
    },
    Together => Together {
        base: "https://api.together.xyz/v1", tools: Native,
        models: ["meta-llama/Llama-3.3-70B-Instruct-Turbo", "Qwen/Qwen2.5-72B-Instruct-Turbo"]
    },
    Openrouter => Openrouter {
        base: "https://openrouter.ai/api/v1", tools: Native,
        models: ["openai/gpt-4o", "anthropic/claude-sonnet-4", "google/gemini-2.5-pro"]
    },
    Fireworks => Fireworks {
        base: "https://api.fireworks.ai/inference/v1", tools: Native,
        models: ["accounts/fireworks/models/llama-v3p3-70b-instruct"]
    },
    Cerebras => Cerebras {
        base: "https://api.cerebras.ai/v1", tools: Native,
        models: ["llama-3.3-70b", "llama3.1-8b", "qwen-3-235b-a22b-instruct-2507"]
    },
    Sambanova => Sambanova {
        base: "https://api.sambanova.ai/v1", tools: Native,
        models: ["Meta-Llama-3.3-70B-Instruct", "DeepSeek-R1"]
    },
    Hyperbolic => Hyperbolic {
        base: "https://api.hyperbolic.xyz/v1", tools: Native,
        models: ["meta-llama/Llama-3.3-70B-Instruct", "deepseek-ai/DeepSeek-V3"]
    },
    NvidiaNim => NvidiaNim {
        base: "https://integrate.api.nvidia.com/v1", tools: Native,
        models: ["meta/llama-3.3-70b-instruct", "deepseek-ai/deepseek-r1"]
    },
    // Perplexity is handled by the dedicated web provider (crate::perplexity).
    // The OAI-compat API path is available via base_url override in TransportConfig
    // when an api_key account is provided. This entry is intentionally removed to
    // avoid overwriting the web provider registration in the registry.
    Qwen => Qwen {
        base: "https://dashscope-intl.aliyuncs.com/compatible-mode/v1", tools: Native,
        models: ["qwen-max", "qwen-plus", "qwen2.5-72b-instruct"]
    },
    Minimax => Minimax {
        base: "https://api.minimax.io/v1", tools: Native,
        models: ["MiniMax-M2", "abab6.5s-chat"]
    },
    Cohere => Cohere {
        base: "https://api.cohere.ai/compatibility/v1", tools: Native,
        models: ["command-r-plus", "command-r", "command-a-03-2025"]
    },
    Litellm => Litellm {
        base: "http://localhost:4000/v1", tools: Native
    },
    Huggingface => Huggingface {
        base: "https://router.huggingface.co/v1", tools: Emu,
        models: ["meta-llama/Llama-3.3-70B-Instruct"]
    },
    Ollama => Ollama {
        base: "http://localhost:11434/v1", tools: Native, profile: "go"
    },
    LlamaCpp => LlamaCpp {
        base: "http://localhost:8080/v1", tools: Emu, profile: "go"
    },
    LocalTransformers => LocalTransformers {
        base: "http://localhost:8000/v1", tools: Emu, profile: "go"
    },
    // Vertex AI exposes an OpenAI-compatible endpoint; the project/location
    // base URL is supplied per-account via `transport.base_url`.
    Vertex => Vertex {
        base: "https://aiplatform.googleapis.com/v1", tools: Native,
        models: ["google/gemini-2.5-pro", "google/gemini-2.5-flash"]
    },
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn builds_every_provider() {
        let providers = all();
        assert_eq!(providers.len(), 21); // Perplexity moved to dedicated web provider
        // Each reports its own provider identity.
        let groq = providers.iter().find(|p| p.capabilities().provider == ProviderName::Groq);
        assert!(groq.is_some());
    }

    #[test]
    fn groq_config_defaults() {
        let g = Groq;
        assert_eq!(g.default_base_url(), "https://api.groq.com/openai/v1");
        assert_eq!(g.capabilities().tool_support, ToolSupportMode::Native);
        assert!(!g.models().is_empty());
    }

    #[test]
    fn perplexity_web_provider_registered() {
        use crate::perplexity::PerplexityWebProvider;
        let p = PerplexityWebProvider::new();
        assert_eq!(p.capabilities().tool_support, ToolSupportMode::JsonEmulated);
    }
}
