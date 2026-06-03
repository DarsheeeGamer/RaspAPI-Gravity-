//! The [`Transport`] abstraction — the seam between providers and the network.
//!
//! Providers issue requests through `&dyn Transport` rather than a concrete
//! client, so tests can inject a [`MockTransport`] that records the exact
//! request a provider builds (for request-byte golden assertions) and returns a
//! scripted response — no network required.

use crate::client::{ByteStream, HttpClient, HttpRequest, HttpResponse};
use async_trait::async_trait;
use bytes::Bytes;
use gravity_core::Error;
use std::sync::Mutex;
use wreq::header::HeaderMap;

/// An HTTP transport: buffered and streaming request dispatch.
#[async_trait]
pub trait Transport: Send + Sync {
    /// Send a request and buffer the full response.
    async fn send(&self, req: HttpRequest) -> Result<HttpResponse, Error>;

    /// Send a request and stream the response body.
    async fn stream(&self, req: HttpRequest) -> Result<(u16, HeaderMap, ByteStream), Error>;
}

#[async_trait]
impl Transport for HttpClient {
    async fn send(&self, req: HttpRequest) -> Result<HttpResponse, Error> {
        HttpClient::send(self, req).await
    }
    async fn stream(&self, req: HttpRequest) -> Result<(u16, HeaderMap, ByteStream), Error> {
        HttpClient::stream(self, req).await
    }
}

/// A recording transport for tests: captures every request and replays a
/// scripted response.
pub struct MockTransport {
    requests: Mutex<Vec<HttpRequest>>,
    /// Status + body returned for every `send`/`stream` call.
    status: u16,
    body: Bytes,
}

impl MockTransport {
    /// A mock that returns `200` with the given body for every request.
    pub fn ok(body: impl Into<Bytes>) -> Self {
        MockTransport {
            requests: Mutex::new(Vec::new()),
            status: 200,
            body: body.into(),
        }
    }

    /// A mock that returns `status` with the given body.
    pub fn with_status(status: u16, body: impl Into<Bytes>) -> Self {
        MockTransport {
            requests: Mutex::new(Vec::new()),
            status,
            body: body.into(),
        }
    }

    /// All requests captured so far, in order.
    pub fn captured(&self) -> Vec<HttpRequest> {
        self.requests.lock().expect("mock transport poisoned").clone()
    }

    /// The most recent captured request, if any.
    pub fn last(&self) -> Option<HttpRequest> {
        self.requests.lock().expect("mock transport poisoned").last().cloned()
    }

    fn record(&self, req: HttpRequest) {
        self.requests.lock().expect("mock transport poisoned").push(req);
    }
}

#[async_trait]
impl Transport for MockTransport {
    async fn send(&self, req: HttpRequest) -> Result<HttpResponse, Error> {
        self.record(req);
        Ok(HttpResponse {
            status: self.status,
            headers: HeaderMap::new(),
            body: self.body.clone(),
        })
    }

    async fn stream(&self, req: HttpRequest) -> Result<(u16, HeaderMap, ByteStream), Error> {
        self.record(req);
        let body = self.body.clone();
        let stream = futures::stream::once(async move { Ok(body) });
        Ok((self.status, HeaderMap::new(), Box::pin(stream)))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Method;

    #[tokio::test]
    async fn mock_records_request_and_replays_body() {
        let mock = MockTransport::ok(Bytes::from_static(b"hello"));
        let req = HttpRequest::new(Method::POST, "https://x/y").header("a", "b").body(Bytes::from_static(b"payload"));
        let resp = Transport::send(&mock, req).await.unwrap();
        assert_eq!(resp.status, 200);
        assert_eq!(&resp.body[..], b"hello");
        let captured = mock.captured();
        assert_eq!(captured.len(), 1);
        assert_eq!(captured[0].url, "https://x/y");
        assert_eq!(captured[0].headers, vec![("a".to_string(), "b".to_string())]);
        assert_eq!(captured[0].body.as_deref(), Some(&b"payload"[..]));
    }
}
