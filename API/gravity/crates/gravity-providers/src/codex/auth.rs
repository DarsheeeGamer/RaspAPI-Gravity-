//! Codex (OpenAI CLI) OAuth token refresh.
//!
//! Mirrors `gravity/codex/auth.py::_refresh` + `oauth.py::build_refresh_request`.
//! Uses `POST https://auth.openai.com/oauth/token` with grant_type=refresh_token.
//!
//! Access token lives in `account.auth.extra["api_key"]`;
//! refresh token in `account.auth.extra["refresh_token"]`.

use crate::provider::Authenticator;
use async_trait::async_trait;
use gravity_core::{Error, ProviderAccount, ProviderErrorKind, ProviderName, Result};
use gravity_http::{HttpClient, HttpRequest, Profile};

const CLIENT_ID: &str = "app_EMoamEEZ73f0CkXaXp7hrann";
const TOKEN_URL: &str = "https://auth.openai.com/oauth/token";

/// Authenticator for the Codex (OpenAI CLI) provider.
pub struct CodexAuthenticator;

#[async_trait]
impl Authenticator for CodexAuthenticator {
    async fn validate_credentials(&self, account: &ProviderAccount) -> Result<bool> {
        Ok(account.auth.inline_secret("api_key").map(|k| !k.is_empty()).unwrap_or(false))
    }

    async fn refresh_credentials(&self, account: &mut ProviderAccount) -> Result<()> {
        let rt = account.auth.inline_secret("refresh_token")
            .ok_or_else(|| Error::Configuration("Codex: no refresh_token stored".to_owned()))?
            .to_owned();
        let body = serde_json::json!({
            "client_id": CLIENT_ID,
            "refresh_token": rt,
            "grant_type": "refresh_token",
        });
        let proxy = account.transport.proxy_url.as_deref().map(str::to_owned);
        let http = HttpClient::new();
        let req = HttpRequest::post(TOKEN_URL)
            .profile(Profile::Chrome)
            .header("content-type", "application/json")
            .proxy(proxy)
            .body(bytes::Bytes::from(serde_json::to_vec(&body).unwrap()));
        let resp = http.send(req).await.map_err(|e| Error::Network(e.to_string()))?;
        if resp.status != 200 {
            return Err(Error::Provider(gravity_core::ProviderError::new(
                ProviderName::Codex, ProviderErrorKind::Authentication,
                "codex_refresh_failed",
                format!("Codex token refresh failed (status {})", resp.status),
            )));
        }
        let data: serde_json::Value = serde_json::from_slice(&resp.body)
            .map_err(|e| Error::Network(e.to_string()))?;
        let new_access = data.get("access_token").and_then(|v| v.as_str())
            .ok_or_else(|| Error::Configuration("Codex refresh: no access_token in response".to_owned()))?
            .to_owned();
        account.auth.extra.insert("api_key", new_access);
        if let Some(new_rt) = data.get("refresh_token").and_then(|v| v.as_str()) {
            account.auth.extra.insert("refresh_token", new_rt.to_owned());
        }
        Ok(())
    }
}
