//! # gravity-server
//!
//! An OpenAI-compatible HTTP front end for the Gravity gateway, built on
//! [`axum`]. Ports the relevant endpoints of `gravity/server.py`.
//!
//! Routes:
//! - `GET  /health` — liveness.
//! - `GET  /providers` — registered providers + capabilities.
//! - `GET  /v1/models` — models in OpenAI list form.
//! - `POST /v1/chat/completions` — OpenAI-compatible chat (stream or buffered).
//!
//! The target provider is the prefix of the OpenAI `model` field
//! (`"groq/llama-3.3-70b"`); the credential is the `Authorization: Bearer`
//! header, turned into a transient account.

#![forbid(unsafe_code)]

mod openai;

use axum::routing::{get, post};
use axum::Router;
use gravity_client::AsyncGravityClient;
use std::sync::Arc;

/// Shared application state.
pub struct AppState {
    /// The async orchestrator.
    pub client: AsyncGravityClient,
}

impl AppState {
    /// Build state over the builtin provider registry.
    pub fn new() -> Self {
        AppState {
            client: AsyncGravityClient::new(),
        }
    }
}

impl Default for AppState {
    fn default() -> Self {
        AppState::new()
    }
}

/// Build the router with default state.
pub fn router() -> Router {
    router_with(Arc::new(AppState::new()))
}

/// Build the router over explicit state.
pub fn router_with(state: Arc<AppState>) -> Router {
    Router::new()
        .route("/health", get(openai::health))
        .route("/providers", get(openai::providers))
        .route("/v1/models", get(openai::models))
        .route("/v1/chat/completions", post(openai::chat_completions))
        .with_state(state)
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::body::Body;
    use axum::http::{Request, StatusCode};
    use http_body_util::BodyExt;
    use tower::ServiceExt;

    #[tokio::test]
    async fn health_ok() {
        let app = router();
        let resp = app
            .oneshot(Request::builder().uri("/health").body(Body::empty()).unwrap())
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        let body = resp.into_body().collect().await.unwrap().to_bytes();
        let v: serde_json::Value = serde_json::from_slice(&body).unwrap();
        assert_eq!(v["status"], "ok");
    }

    #[tokio::test]
    async fn providers_lists_registered() {
        let app = router();
        let resp = app
            .oneshot(Request::builder().uri("/providers").body(Body::empty()).unwrap())
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        let body = resp.into_body().collect().await.unwrap().to_bytes();
        let v: serde_json::Value = serde_json::from_slice(&body).unwrap();
        let names: Vec<&str> = v.as_array().unwrap().iter().map(|p| p["name"].as_str().unwrap()).collect();
        assert!(names.contains(&"claude"));
        assert!(names.contains(&"groq"));
        assert!(names.len() >= 24);
    }

    #[tokio::test]
    async fn chat_rejects_unknown_provider() {
        let app = router();
        let body = serde_json::json!({
            "model": "not_a_provider/x",
            "messages": [{"role":"user","content":"hi"}]
        });
        let resp = app
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/v1/chat/completions")
                    .header("content-type", "application/json")
                    .header("authorization", "Bearer sk-x")
                    .body(Body::from(body.to_string()))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
    }
}
