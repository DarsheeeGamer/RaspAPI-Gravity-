//! `gravity-server` binary: start the OpenAI-compatible HTTP gateway.
//!
//! ```text
//! GRAVITY_HOST=0.0.0.0 GRAVITY_PORT=8080 gravity-server
//! ```

use std::net::SocketAddr;

#[tokio::main]
async fn main() {
    tracing_subscriber::fmt()
        .with_env_filter(std::env::var("RUST_LOG").unwrap_or_else(|_| "info".into()))
        .init();

    let host = std::env::var("GRAVITY_HOST").unwrap_or_else(|_| "127.0.0.1".into());
    let port: u16 = std::env::var("GRAVITY_PORT")
        .ok()
        .and_then(|p| p.parse().ok())
        .unwrap_or(8080);
    let addr: SocketAddr = format!("{host}:{port}")
        .parse()
        .expect("invalid GRAVITY_HOST/GRAVITY_PORT");

    let app = gravity_server::router();
    let listener = tokio::net::TcpListener::bind(addr)
        .await
        .unwrap_or_else(|e| panic!("failed to bind {addr}: {e}"));

    tracing::info!("gravity-server listening on http://{addr}");
    axum::serve(listener, app)
        .with_graceful_shutdown(shutdown_signal())
        .await
        .expect("server error");
}

async fn shutdown_signal() {
    let _ = tokio::signal::ctrl_c().await;
    tracing::info!("shutting down");
}
