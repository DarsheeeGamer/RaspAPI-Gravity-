//! The error hierarchy and centralized retry/failover classification.
//!
//! Mirrors `gravity/errors.py` + `gravity/exceptions.py`, and folds the magic
//! status-code sets scattered through `gravity/clients.py` into typed predicates
//! on [`ProviderError`].

use crate::capabilities::ProviderName;
use crate::metadata::Metadata;
use std::borrow::Cow;
use std::time::Duration;

/// Convenient result alias.
pub type Result<T> = std::result::Result<T, Error>;

/// Top-level Gravity error.
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum Error {
    /// Bad library/account configuration.
    #[error("configuration: {0}")]
    Configuration(String),

    /// Invalid request or schema.
    #[error("validation [{code}]: {message}")]
    Validation {
        /// Human-readable message.
        message: String,
        /// Stable machine code (e.g. `provider_model_mismatch`).
        code: &'static str,
    },

    /// No adapter registered for a provider.
    #[error("provider {provider} is not registered")]
    ProviderNotRegistered {
        /// The missing provider.
        provider: ProviderName,
    },

    /// Model routing failed.
    #[error("model routing: {0}")]
    ModelRouting(String),

    /// No usable account could be selected.
    #[error("account selection: {0}")]
    AccountSelection(String),

    /// A conversation/checkpoint problem.
    #[error("conversation: {0}")]
    Conversation(String),

    /// A provider returned an error.
    #[error(transparent)]
    Provider(#[from] ProviderError),

    /// A transport-layer failure (DNS, TLS, timeout, …).
    #[error("network: {0}")]
    Network(String),
}

impl Error {
    /// Build a validation error with a stable code.
    pub fn validation(message: impl Into<String>, code: &'static str) -> Self {
        Error::Validation {
            message: message.into(),
            code,
        }
    }

    /// Build an "unsupported operation" validation error.
    pub fn unsupported(what: impl std::fmt::Display) -> Self {
        Error::Validation {
            message: format!("operation not supported: {what}"),
            code: "unsupported",
        }
    }

    /// Whether the client should rotate to another account and retry.
    pub fn should_failover(&self) -> bool {
        match self {
            Error::Provider(p) => p.should_failover(),
            Error::Network(_) => true,
            _ => false,
        }
    }
}

/// The category of a provider-side error, driving retry policy.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ProviderErrorKind {
    /// HTTP 429 / rate limited.
    RateLimit,
    /// Usage/quota exhausted (not retryable on the same account).
    QuotaExceeded,
    /// 401/403 auth failure.
    Authentication,
    /// 5xx upstream error.
    Server,
    /// 4xx client error (not retryable).
    BadRequest,
    /// Transient upstream blip (overloaded, timeout).
    Transient,
    /// Anything else.
    Other,
}

/// An error returned by a provider.
///
/// Mirrors `ProviderError` (code + details) from `gravity/errors.py` plus the
/// richer rate-limit/quota/auth subclasses from `exceptions.py`.
#[derive(Debug, thiserror::Error)]
#[error("{kind:?} from {provider} [{code}]: {message}")]
pub struct ProviderError {
    /// Provider that produced the error.
    pub provider: ProviderName,
    /// Error category.
    pub kind: ProviderErrorKind,
    /// Stable error code.
    pub code: Cow<'static, str>,
    /// Human-readable message.
    pub message: String,
    /// HTTP status code, if any.
    pub status: Option<u16>,
    /// Suggested retry delay (from `Retry-After`).
    pub retry_after: Option<Duration>,
    /// Extra structured detail (boxed to keep [`Error`] small).
    pub details: Box<Metadata>,
}

impl ProviderError {
    /// Build a provider error with sensible defaults.
    pub fn new(
        provider: ProviderName,
        kind: ProviderErrorKind,
        code: impl Into<Cow<'static, str>>,
        message: impl Into<String>,
    ) -> Self {
        ProviderError {
            provider,
            kind,
            code: code.into(),
            message: message.into(),
            status: None,
            retry_after: None,
            details: Box::new(Metadata::new()),
        }
    }

    /// Set the HTTP status, inferring the [`ProviderErrorKind`] when it is still
    /// [`ProviderErrorKind::Other`].
    pub fn with_status(mut self, status: u16) -> Self {
        self.status = Some(status);
        if self.kind == ProviderErrorKind::Other {
            self.kind = match status {
                429 => ProviderErrorKind::RateLimit,
                401 | 403 => ProviderErrorKind::Authentication,
                500..=599 => ProviderErrorKind::Server,
                400..=499 => ProviderErrorKind::BadRequest,
                _ => ProviderErrorKind::Other,
            };
        }
        self
    }

    /// Attach a retry-after hint.
    pub fn with_retry_after(mut self, after: Duration) -> Self {
        self.retry_after = Some(after);
        self
    }

    /// A transient rate limit that should be retried on the *same* account
    /// after a short delay (vs. a hard usage cap or a server-specified reset
    /// window). Mirrors the distinction in `clients.py`'s rate-limit retry path:
    /// a 429 carrying a `Retry-After`/`resets_at` is a hard window, not transient.
    pub fn is_transient_rate_limit(&self) -> bool {
        self.kind == ProviderErrorKind::RateLimit
            && self.retry_after.is_none()
            && self.code != "usage_limit_reached"
            && self.code != "quota_exceeded"
    }

    /// Whether the client should rotate to another account. Includes
    /// authentication failures: a 401/403 on one account may succeed on another
    /// (different credentials), matching the Python rotate-on-retryable behavior.
    pub fn should_failover(&self) -> bool {
        matches!(
            self.kind,
            ProviderErrorKind::RateLimit
                | ProviderErrorKind::Server
                | ProviderErrorKind::Transient
                | ProviderErrorKind::QuotaExceeded
                | ProviderErrorKind::Authentication
        )
    }

    /// Whether this error invalidates a remote checkpoint.
    ///
    /// Mirrors `CHECKPOINT_INVALIDATION_STATUS_CODES = {401, 403, 404, 409}`.
    pub fn invalidates_checkpoint(&self) -> bool {
        matches!(self.status, Some(401 | 403 | 404 | 409))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn status_infers_kind() {
        let e = ProviderError::new(ProviderName::Claude, ProviderErrorKind::Other, "x", "m")
            .with_status(429);
        assert_eq!(e.kind, ProviderErrorKind::RateLimit);
        assert!(e.should_failover());
        assert!(e.is_transient_rate_limit());
    }

    #[test]
    fn usage_cap_is_not_transient() {
        let e = ProviderError::new(
            ProviderName::Claude,
            ProviderErrorKind::RateLimit,
            "usage_limit_reached",
            "m",
        );
        assert!(!e.is_transient_rate_limit());
    }

    #[test]
    fn checkpoint_invalidation_codes() {
        for code in [401, 403, 404, 409] {
            let e = ProviderError::new(ProviderName::Claude, ProviderErrorKind::Other, "x", "m")
                .with_status(code);
            assert!(e.invalidates_checkpoint(), "status {code}");
        }
        let ok = ProviderError::new(ProviderName::Claude, ProviderErrorKind::Other, "x", "m")
            .with_status(500);
        assert!(!ok.invalidates_checkpoint());
    }
}
