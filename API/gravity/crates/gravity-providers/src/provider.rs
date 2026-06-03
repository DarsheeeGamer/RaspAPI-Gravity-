//! The `Provider` and `Authenticator` traits.

use crate::context::ChatCtx;
use async_trait::async_trait;
use futures::stream::BoxStream;
use gravity_core::{
    ChatResponse, Checkpoint, EmbeddingResponse, Error, ImageRequest, ImageResponse, ModelInfo,
    ProviderAccount, ProviderCapabilities, Result, StreamEvent,
};

/// A stream of incremental [`StreamEvent`]s.
///
/// `'static` so it can outlive the borrowed [`ChatCtx`] that produced it —
/// providers move whatever they need out of the context into the stream.
pub type EventStream = BoxStream<'static, Result<StreamEvent>>;

/// A unified, async provider adapter.
///
/// Replaces the parallel `SyncProviderAdapter` / `AsyncProviderAdapter`
/// hierarchies from `gravity/abcs.py` with a single async trait; a synchronous
/// facade is provided by the client via `block_on`.
///
/// Provider instances are immutable configuration shared behind
/// `Arc<dyn Provider>`; all per-call state lives in [`ChatCtx`].
#[async_trait]
pub trait Provider: Send + Sync {
    /// Static capability description.
    fn capabilities(&self) -> &ProviderCapabilities;

    /// Models this provider exposes without a network call (may be empty).
    fn list_models(&self) -> Vec<String> {
        Vec::new()
    }

    /// Discover models from the provider's live endpoint, returning rich
    /// [`ModelInfo`] (id + display name + description + tags).
    ///
    /// The default falls back to [`Self::list_models`] mapped to bare ids (no
    /// network). Providers whose web UI / API exposes a model-listing endpoint
    /// override this to hit it — e.g. Cursor's `GetUsableModels`, ChatGPT's
    /// `/backend-api/models`, Gemini's `otAQ7b` batchexecute RPC, GLM's
    /// `/api/models`, or any OpenAI-compatible `GET /models`. On any network or
    /// auth failure, override impls should fall back to the static list so the
    /// call never hard-errors.
    async fn discover_models(&self, _ctx: &ChatCtx<'_>) -> Result<Vec<ModelInfo>> {
        Ok(self.list_models().into_iter().map(ModelInfo::bare).collect())
    }

    /// Adapt a system prompt for this provider. Default: trim, drop if empty.
    fn normalize_system_prompt(&self, system_prompt: Option<&str>) -> Option<String> {
        system_prompt
            .map(str::trim)
            .filter(|s| !s.is_empty())
            .map(str::to_owned)
    }

    /// Open or refresh a remote session, returning a [`Checkpoint`].
    ///
    /// Default: stateless — no checkpoint. Stateful providers (Claude, …)
    /// override this to create the remote conversation.
    async fn open_checkpoint(&self, _ctx: &ChatCtx<'_>) -> Result<Option<Checkpoint>> {
        Ok(None)
    }

    /// Perform a blocking (buffered) chat turn.
    async fn chat(&self, ctx: &ChatCtx<'_>) -> Result<ChatResponse>;

    /// Perform a streaming chat turn.
    ///
    /// The default buffers [`Self::chat`] into a single terminal
    /// [`StreamEvent::Done`]; providers with real wire streaming override it.
    async fn chat_stream(&self, ctx: &ChatCtx<'_>) -> Result<EventStream> {
        let resp = self.chat(ctx).await?;
        let ev = StreamEvent::Done(Box::new(resp));
        Ok(Box::pin(futures::stream::once(async move { Ok(ev) })))
    }

    /// Generate embeddings. Default: unsupported.
    async fn embeddings(&self, _ctx: &ChatCtx<'_>, _texts: &[String]) -> Result<EmbeddingResponse> {
        Err(Error::unsupported("embeddings"))
    }

    /// Generate images. Default: unsupported.
    async fn generate_image(
        &self,
        _account: &ProviderAccount,
        _request: &ImageRequest,
    ) -> Result<ImageResponse> {
        Err(Error::unsupported("image generation"))
    }

    /// Return the authenticator for this provider, if credential refresh is
    /// supported (e.g. OAuth-based providers like Cursor, Windsurf, Codex).
    ///
    /// Providers that only use static API keys return `None`.
    fn authenticator(&self) -> Option<&dyn Authenticator> {
        None
    }
}

/// Account creation / credential lifecycle for a provider.
///
/// Mirrors `ProviderAuthenticator` in `gravity/abcs.py`.
#[async_trait]
pub trait Authenticator: Send + Sync {
    /// Whether the given account's stored credentials are still usable.
    async fn validate_credentials(&self, account: &ProviderAccount) -> Result<bool>;

    /// Refresh expired credentials in place. Default: unsupported (e.g. API
    /// keys never refresh).
    async fn refresh_credentials(&self, _account: &mut ProviderAccount) -> Result<()> {
        Err(Error::unsupported("credential refresh"))
    }
}
