//! The impersonating HTTP client.

use crate::profile::{emulation_provider, Profile};
use bytes::Bytes;
use futures::Stream;
use gravity_core::Error;
use std::collections::HashMap;
use std::pin::Pin;
use std::sync::Mutex;
use std::time::Duration;
use wreq::header::{HeaderMap, HeaderName, HeaderValue};
use wreq::{Client, Method};

/// A boxed byte stream of response-body chunks.
pub type ByteStream = Pin<Box<dyn Stream<Item = Result<Bytes, Error>> + Send>>;

/// Identifies a cached `wreq::Client`. Clients are expensive to construct
/// (they configure the BoringSSL emulation), so one is built per distinct
/// `(profile, proxy, verify_ssl, cookie_store)` and reused.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct ClientKey {
    profile: Profile,
    proxy: Option<String>,
    verify_ssl: bool,
    cookie_store: bool,
}

/// An HTTP request to dispatch.
///
/// Built with the fluent setters; `profile` selects the TLS impersonation.
#[derive(Debug, Clone)]
pub struct HttpRequest {
    /// HTTP method.
    pub method: Method,
    /// Absolute URL.
    pub url: String,
    /// Impersonation profile.
    pub profile: Profile,
    /// Header pairs, applied in insertion order.
    pub headers: Vec<(String, String)>,
    /// Optional request body.
    pub body: Option<Bytes>,
    /// Per-request timeout (milliseconds).
    pub timeout_ms: u32,
    /// Optional proxy URL.
    pub proxy: Option<String>,
    /// Whether to verify TLS certificates.
    pub verify_ssl: bool,
    /// Enable cookie store so cookies are persisted across requests in the same
    /// client (needed for web providers that bootstrap a session first).
    pub cookie_store: bool,
}

impl HttpRequest {
    /// Start a request for `method` and `url`.
    pub fn new(method: Method, url: impl Into<String>) -> Self {
        HttpRequest {
            method,
            url: url.into(),
            profile: Profile::Chrome,
            headers: Vec::new(),
            body: None,
            timeout_ms: 60_000,
            proxy: None,
            verify_ssl: true,
            cookie_store: false,
        }
    }

    /// Convenience constructor for a `GET`.
    pub fn get(url: impl Into<String>) -> Self {
        HttpRequest::new(Method::GET, url)
    }

    /// Convenience constructor for a `POST`.
    pub fn post(url: impl Into<String>) -> Self {
        HttpRequest::new(Method::POST, url)
    }

    /// Set the impersonation profile.
    pub fn profile(mut self, profile: Profile) -> Self {
        self.profile = profile;
        self
    }

    /// Add a header.
    pub fn header(mut self, name: impl Into<String>, value: impl Into<String>) -> Self {
        self.headers.push((name.into(), value.into()));
        self
    }

    /// Set the body to raw bytes.
    pub fn body(mut self, body: impl Into<Bytes>) -> Self {
        self.body = Some(body.into());
        self
    }

    /// Set a JSON body and the `content-type` header.
    pub fn json(mut self, value: &serde_json::Value) -> Self {
        self.body = Some(Bytes::from(serde_json::to_vec(value).unwrap_or_default()));
        self.header("content-type", "application/json")
    }

    /// Set the timeout.
    pub fn timeout_ms(mut self, ms: u32) -> Self {
        self.timeout_ms = ms;
        self
    }

    /// Set a proxy.
    pub fn proxy(mut self, proxy: Option<String>) -> Self {
        self.proxy = proxy;
        self
    }

    /// Enable cookie store (persists cookies across requests via the same client).
    pub fn cookie_store(mut self, enable: bool) -> Self {
        self.cookie_store = enable;
        self
    }
}

/// A completed (non-streaming) HTTP response.
#[derive(Debug, Clone)]
pub struct HttpResponse {
    /// HTTP status code.
    pub status: u16,
    /// Response headers.
    pub headers: HeaderMap,
    /// Fully buffered body.
    pub body: Bytes,
}

impl HttpResponse {
    /// `true` for 2xx statuses.
    pub fn is_success(&self) -> bool {
        (200..300).contains(&self.status)
    }

    /// Body as UTF-8 (lossy).
    pub fn text(&self) -> std::borrow::Cow<'_, str> {
        String::from_utf8_lossy(&self.body)
    }

    /// Parse the body as JSON.
    pub fn json(&self) -> Result<serde_json::Value, Error> {
        serde_json::from_slice(&self.body)
            .map_err(|e| Error::Network(format!("invalid JSON response: {e}")))
    }

    /// Read a response header as a string.
    pub fn header(&self, name: &str) -> Option<&str> {
        self.headers.get(name).and_then(|v| v.to_str().ok())
    }
}

/// The shared, cloneable HTTP transport used by every provider.
#[derive(Default)]
pub struct HttpClient {
    cache: Mutex<HashMap<ClientKey, Client>>,
}

impl std::fmt::Debug for HttpClient {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let n = self.cache.lock().map(|c| c.len()).unwrap_or(0);
        f.debug_struct("HttpClient").field("cached_clients", &n).finish()
    }
}

impl HttpClient {
    /// Create an empty client cache.
    pub fn new() -> Self {
        HttpClient::default()
    }

    /// Get or build the `wreq::Client` for a request's `(profile, proxy, verify)`.
    fn client_for(&self, req: &HttpRequest) -> Result<Client, Error> {
        let key = ClientKey {
            profile: req.profile,
            proxy: req.proxy.clone(),
            verify_ssl: req.verify_ssl,
            cookie_store: req.cookie_store,
        };
        let mut cache = self.cache.lock().expect("http client cache poisoned");
        if let Some(c) = cache.get(&key) {
            return Ok(c.clone());
        }
        let mut builder = Client::builder()
            .emulation(emulation_provider(req.profile))
            .cert_verification(req.verify_ssl);
        if req.cookie_store {
            builder = builder.cookie_store(true);
        }
        if let Some(proxy) = &req.proxy {
            let p = wreq::Proxy::all(proxy)
                .map_err(|e| Error::Network(format!("invalid proxy {proxy:?}: {e}")))?;
            builder = builder.proxy(p);
        }
        let client = builder
            .build()
            .map_err(|e| Error::Network(format!("failed to build HTTP client: {e}")))?;
        cache.insert(key, client.clone());
        Ok(client)
    }

    /// Build a `wreq` request builder from an [`HttpRequest`].
    fn prepare(&self, req: &HttpRequest) -> Result<wreq::RequestBuilder, Error> {
        let client = self.client_for(req)?;
        let mut headers = HeaderMap::with_capacity(req.headers.len());
        for (name, value) in &req.headers {
            let hn = HeaderName::from_bytes(name.as_bytes())
                .map_err(|e| Error::Network(format!("invalid header name {name:?}: {e}")))?;
            let hv = HeaderValue::from_str(value)
                .map_err(|e| Error::Network(format!("invalid header value for {name:?}: {e}")))?;
            headers.append(hn, hv);
        }
        let mut rb = client
            .request(req.method.clone(), &req.url)
            .headers(headers)
            .timeout(Duration::from_millis(req.timeout_ms as u64));
        if let Some(body) = &req.body {
            rb = rb.body(body.clone());
        }
        Ok(rb)
    }

    /// Send a request and buffer the full response body.
    pub async fn send(&self, req: HttpRequest) -> Result<HttpResponse, Error> {
        let rb = self.prepare(&req)?;
        let resp = rb
            .send()
            .await
            .map_err(|e| Error::Network(format!("request to {} failed: {e}", req.url)))?;
        let status = resp.status().as_u16();
        let headers = resp.headers().clone();
        let body = resp
            .bytes()
            .await
            .map_err(|e| Error::Network(format!("reading body from {} failed: {e}", req.url)))?;
        Ok(HttpResponse { status, headers, body })
    }

    /// Send a request and return the response as a stream of body chunks.
    ///
    /// The status and headers are returned eagerly; the body streams lazily.
    pub async fn stream(&self, req: HttpRequest) -> Result<(u16, HeaderMap, ByteStream), Error> {
        use futures::StreamExt;
        let rb = self.prepare(&req)?;
        let resp = rb
            .send()
            .await
            .map_err(|e| Error::Network(format!("request to {} failed: {e}", req.url)))?;
        let status = resp.status().as_u16();
        let headers = resp.headers().clone();
        let url = req.url.clone();
        let stream = resp.bytes_stream().map(move |chunk| {
            chunk.map_err(|e| Error::Network(format!("stream from {url} failed: {e}")))
        });
        Ok((status, headers, Box::pin(stream)))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn request_builder_is_fluent() {
        let req = HttpRequest::post("https://example.com/x")
            .profile(Profile::Chrome)
            .header("authorization", "Bearer x")
            .timeout_ms(5000)
            .body(Bytes::from_static(b"hi"));
        assert_eq!(req.method, Method::POST);
        assert_eq!(req.headers.len(), 1);
        assert_eq!(req.timeout_ms, 5000);
        assert!(req.body.is_some());
    }

    #[test]
    fn client_cache_reuses_clients() {
        let http = HttpClient::new();
        let req = HttpRequest::get("https://example.com");
        let _ = http.client_for(&req).unwrap();
        let _ = http.client_for(&req).unwrap();
        assert_eq!(http.cache.lock().unwrap().len(), 1);
    }
}
