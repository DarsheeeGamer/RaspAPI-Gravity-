//! # gravity-http
//!
//! HTTP transport for Gravity with browser/runtime **TLS impersonation**.
//!
//! Built on [`wreq`](https://docs.rs/wreq) (a BoringSSL-backed `reqwest` fork),
//! this crate is the Rust counterpart of `gravity/http.py`. It provides:
//!
//! - [`Profile`] — the impersonation profiles (`chrome`, `firefox`, `safari`,
//!   `edge`, and the bespoke Go `go_tls`/`go_boring` JA3 configs).
//! - [`HttpClient`] — a cloneable transport that caches one BoringSSL client per
//!   `(profile, proxy, verify_ssl)` and exposes buffered ([`HttpClient::send`])
//!   and streaming ([`HttpClient::stream`]) requests.
//! - [`sse`] — line and Server-Sent-Events decoding over the response stream.
//!
//! BoringSSL gives precise control over the ClientHello (cipher order, curves,
//! extension permutation, GREASE), which is why it — not rustls — backs the
//! Go/FIPS fingerprints.

#![forbid(unsafe_code)]

mod client;
mod profile;
pub mod sse;
mod transport;

pub use client::{ByteStream, HttpClient, HttpRequest, HttpResponse};
pub use profile::Profile;
pub use transport::{MockTransport, Transport};

/// Re-exported `wreq` HTTP primitives so providers don't depend on `wreq` directly.
pub use wreq::header::{HeaderMap, HeaderName, HeaderValue};
pub use wreq::Method;
