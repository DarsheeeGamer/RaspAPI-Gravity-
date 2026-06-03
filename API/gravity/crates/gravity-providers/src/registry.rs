//! The provider registry.
//!
//! Replaces the Python import-time global-mutation registry with an immutable
//! table built once at startup and indexed by [`ProviderName`] via an
//! [`EnumMap`] (array-backed, O(1), deterministic iteration). The builtin
//! registry is a process-wide `&'static` behind a [`OnceLock`].

use crate::provider::Provider;
use enum_map::EnumMap;
use gravity_core::{Error, ProviderCapabilities, ProviderName, Result};
use std::sync::{Arc, OnceLock};

/// An immutable map from [`ProviderName`] to its adapter.
pub struct Registry {
    providers: EnumMap<ProviderName, Option<Arc<dyn Provider>>>,
}

impl Registry {
    /// Look up a provider, erroring if none is registered.
    pub fn provider(&self, name: ProviderName) -> Result<&Arc<dyn Provider>> {
        self.providers[name]
            .as_ref()
            .ok_or(Error::ProviderNotRegistered { provider: name })
    }

    /// Whether a provider is registered.
    pub fn has(&self, name: ProviderName) -> bool {
        self.providers[name].is_some()
    }

    /// Iterate over every registered `(name, capabilities)` pair.
    pub fn iter_capabilities(&self) -> impl Iterator<Item = (ProviderName, &ProviderCapabilities)> {
        ProviderName::all().filter_map(move |n| {
            self.providers[n]
                .as_ref()
                .map(|p| (n, p.capabilities()))
        })
    }

    /// Number of registered providers.
    pub fn len(&self) -> usize {
        ProviderName::all().filter(|&n| self.has(n)).count()
    }

    /// Whether the registry is empty.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

/// Mutable builder used once to assemble the [`Registry`].
#[derive(Default)]
pub struct RegistryBuilder {
    providers: EnumMap<ProviderName, Option<Arc<dyn Provider>>>,
}

impl RegistryBuilder {
    /// Start an empty builder.
    pub fn new() -> Self {
        RegistryBuilder::default()
    }

    /// Register a provider adapter under its declared [`ProviderName`].
    pub fn add(&mut self, provider: Arc<dyn Provider>) -> &mut Self {
        self.add_arc(provider);
        self
    }

    /// Register an already-`Arc`'d adapter (by value, for iterator pipelines).
    pub fn add_arc(&mut self, provider: Arc<dyn Provider>) {
        let name = provider.capabilities().provider;
        self.providers[name] = Some(provider);
    }

    /// Finalize into an immutable [`Registry`].
    pub fn build(self) -> Registry {
        Registry {
            providers: self.providers,
        }
    }
}

/// The process-wide builtin registry, constructed on first access.
///
/// Providers are added explicitly here (mirroring `_register_providers` in the
/// Python `__init__.py`) so the set is greppable, ordered, and feature-gateable.
pub fn builtin() -> &'static Registry {
    static REG: OnceLock<Registry> = OnceLock::new();
    REG.get_or_init(|| {
        let mut b = RegistryBuilder::new();
        // ── Reverse-engineered web providers (ported first) ──
        b.add(Arc::new(crate::claude::ClaudeProvider::new()));
        b.add(Arc::new(crate::glm::GlmProvider::new()));
        b.add(Arc::new(crate::kiro::KiroProvider::new()));
        b.add(Arc::new(crate::gemini::GeminiProvider::new()));
        b.add(Arc::new(crate::chatgpt::ChatgptProvider::new()));
        b.add(Arc::new(crate::gemini_cli::GeminiCliProvider::new()));
        b.add(Arc::new(crate::codex::CodexProvider::new()));
        b.add(Arc::new(crate::cursor::CursorProvider::new()));
        b.add(Arc::new(crate::windsurf::WindsurfProvider::new()));
        b.add(Arc::new(crate::antigravity::AntigravityProvider::new()));
        // ── Native (non-OpenAI) API providers ──
        b.add(Arc::new(crate::anthropic::AnthropicProvider::new()));
        // ── Perplexity web (anonymous + session auth, replaces OAI-compat entry) ──
        b.add(Arc::new(crate::perplexity::PerplexityWebProvider::new()));
        // ── OpenAI-compatible API-key providers (Perplexity handled above) ──
        for p in crate::openai_compat::all() {
            b.add_arc(p);
        }
        b.build()
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn builtin_contains_claude() {
        let reg = builtin();
        assert!(reg.has(ProviderName::Claude));
        assert!(reg.provider(ProviderName::Claude).is_ok());
        // OpenAI-compatible providers are now registered too.
        assert!(reg.has(ProviderName::Groq));
        assert!(reg.has(ProviderName::OpenaiApi));
        // Every ProviderName variant has a registered adapter — full coverage.
        for p in ProviderName::all() {
            assert!(reg.has(p), "provider {p} is not registered");
        }
        assert_eq!(reg.len(), ProviderName::all().count());
    }

    #[test]
    fn capabilities_iteration_lists_registered() {
        let names: Vec<_> = builtin().iter_capabilities().map(|(n, _)| n).collect();
        assert!(names.contains(&ProviderName::Claude));
    }
}
