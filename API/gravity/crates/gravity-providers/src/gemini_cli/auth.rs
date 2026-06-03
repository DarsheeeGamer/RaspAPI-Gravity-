//! Gemini CLI OAuth token refresh (Google OAuth 2.0).
//!
//! Mirrors `gravity/gemini_cli/authenticator.py::refresh_credentials`.
//! Uses `POST https://oauth2.googleapis.com/token` with grant_type=refresh_token.
//!
//! Access token in `account.auth.extra["access_token"]`;
//! refresh token in `account.auth.extra["refresh_token"]`.
//! Expiry (Unix timestamp) in `account.auth.extra["expires_at"]`.

use crate::provider::Authenticator;
use async_trait::async_trait;
use gravity_core::{Error, ProviderAccount, ProviderErrorKind, ProviderName, Result};
use gravity_http::{HttpClient, HttpRequest, Profile};

// Supply at build time via GRAVITY_GEMINI_CLI_CLIENT_ID / _SECRET env vars.
// Redacted in source (real values were leaked secrets).
const CLIENT_ID: &str = match option_env!("GRAVITY_GEMINI_CLI_CLIENT_ID") {
    Some(v) => v,
    None => "REDACTED",
};
const CLIENT_SECRET: &str = match option_env!("GRAVITY_GEMINI_CLI_CLIENT_SECRET") {
    Some(v) => v,
    None => "REDACTED",
};
const TOKEN_URL: &str = "https://oauth2.googleapis.com/token";

/// Authenticator for the Gemini CLI provider.
pub struct GeminiCliAuthenticator;

#[async_trait]
impl Authenticator for GeminiCliAuthenticator {
    async fn validate_credentials(&self, account: &ProviderAccount) -> Result<bool> {
        Ok(account.auth.inline_secret("access_token").map(|k| !k.is_empty()).unwrap_or(false))
    }

    async fn refresh_credentials(&self, account: &mut ProviderAccount) -> Result<()> {
        let rt = account.auth.inline_secret("refresh_token")
            .ok_or_else(|| Error::Configuration("GeminiCLI: no refresh_token stored".to_owned()))?
            .to_owned();
        // Google token endpoint uses form-encoded body.
        let form = format!(
            "client_id={CLIENT_ID}&client_secret={CLIENT_SECRET}&refresh_token={rt}&grant_type=refresh_token"
        );
        let proxy = account.transport.proxy_url.as_deref().map(str::to_owned);
        let http = HttpClient::new();
        let req = HttpRequest::post(TOKEN_URL)
            .profile(Profile::Chrome)
            .header("content-type", "application/x-www-form-urlencoded")
            .proxy(proxy)
            .body(bytes::Bytes::from(form));
        let resp = http.send(req).await.map_err(|e| Error::Network(e.to_string()))?;
        if resp.status != 200 {
            return Err(Error::Provider(gravity_core::ProviderError::new(
                ProviderName::GeminiCli, ProviderErrorKind::Authentication,
                "gemini_cli_refresh_failed",
                format!("GeminiCLI token refresh failed (status {})", resp.status),
            )));
        }
        let data: serde_json::Value = serde_json::from_slice(&resp.body)
            .map_err(|e| Error::Network(e.to_string()))?;
        let new_access = data.get("access_token").and_then(|v| v.as_str())
            .ok_or_else(|| Error::Configuration("GeminiCLI refresh: no access_token".to_owned()))?
            .to_owned();
        account.auth.extra.insert("access_token", new_access.clone());
        // Also set as the primary inline secret so providers can resolve it normally.
        account.auth.extra.insert("api_key", new_access);
        if let Some(expires_in) = data.get("expires_in").and_then(|v| v.as_i64()) {
            let expires_at = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_secs() as i64)
                .unwrap_or(0) + expires_in;
            account.auth.extra.insert("expires_at", expires_at);
        }
        if let Some(new_rt) = data.get("refresh_token").and_then(|v| v.as_str()) {
            account.auth.extra.insert("refresh_token", new_rt.to_owned());
        }
        Ok(())
    }
}
