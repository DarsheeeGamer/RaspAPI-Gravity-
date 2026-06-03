//! Cursor OAuth token refresh.
//!
//! Mirrors `gravity/cursor/__init__.py::refresh_cursor_token`.
//! POSTs to `https://authentication.cursor.sh/oauth/token` with the stored
//! refresh token; updates `account.auth.extra["api_key"]` with the new access
//! token on success.

use crate::provider::Authenticator;
use async_trait::async_trait;
use gravity_core::{Error, ProviderAccount, Result};

const CLIENT_ID: &str = "KbZUR41cY7W6zRSdpSUJ7I7mLYBKOCmB";
const AUTH_DOMAINS: &[&str] = &[
    "authentication.cursor.sh",
    "prod.authentication.cursor.sh",
];
const USER_AGENT: &str = "Mozilla/5.0 (compatible; CursorAuth/1.0)";

/// Authenticator for the Cursor IDE provider.
///
/// Stores the refresh token in `account.auth.extra["refresh_token"]` and the
/// current access token in `account.auth.extra["api_key"]`. On 401, calls
/// `/oauth/token` to exchange the refresh token for a new access token.
pub struct CursorAuthenticator;

#[async_trait]
impl Authenticator for CursorAuthenticator {
    async fn validate_credentials(&self, account: &ProviderAccount) -> Result<bool> {
        let key = account.auth.inline_secret("api_key");
        Ok(key.map(|k| !k.is_empty()).unwrap_or(false))
    }

    async fn refresh_credentials(&self, account: &mut ProviderAccount) -> Result<()> {
        let refresh_token = account
            .auth
            .inline_secret("refresh_token")
            .ok_or_else(|| Error::Configuration("Cursor: no refresh_token stored".to_owned()))?
            .to_owned();

        let proxy = account.transport.proxy_url.as_deref().map(str::to_owned);

        let body = serde_json::json!({
            "grant_type": "refresh_token",
            "client_id": CLIENT_ID,
            "refresh_token": refresh_token,
        });

        for domain in AUTH_DOMAINS {
            let url = format!("https://{domain}/oauth/token");
            let result = try_refresh(&url, &body, proxy.as_deref()).await;
            if let Ok((new_access, new_refresh)) = result {
                account.auth.extra.insert("api_key", new_access);
                account.auth.extra.insert("refresh_token", new_refresh);
                return Ok(());
            }
        }
        Err(Error::Provider(gravity_core::ProviderError::new(
            gravity_core::ProviderName::Cursor,
            gravity_core::ProviderErrorKind::Authentication,
            "cursor_refresh_failed",
            "Cursor token refresh failed on all endpoints".to_owned(),
        )))
    }
}

async fn try_refresh(
    url: &str,
    body: &serde_json::Value,
    proxy: Option<&str>,
) -> std::result::Result<(String, String), ()> {
    use gravity_http::{HttpRequest, Profile};
    let http = gravity_http::HttpClient::new();
    let req = HttpRequest::post(url)
        .profile(Profile::Chrome)
        .header("content-type", "application/json")
        .header("user-agent", USER_AGENT)
        .proxy(proxy.map(str::to_owned))
        .body(bytes::Bytes::from(serde_json::to_vec(body).unwrap()));
    let resp = http.send(req).await.map_err(|_| ())?;
    if resp.status != 200 {
        return Err(());
    }
    let data: serde_json::Value = serde_json::from_slice(&resp.body).map_err(|_| ())?;
    let access = data
        .get("access_token")
        .or_else(|| data.get("accessToken"))
        .and_then(serde_json::Value::as_str)
        .ok_or(())?
        .to_owned();
    let refresh = data
        .get("refresh_token")
        .or_else(|| data.get("refreshToken"))
        .and_then(serde_json::Value::as_str)
        .unwrap_or(body["refresh_token"].as_str().unwrap_or(""))
        .to_owned();
    Ok((access, refresh))
}
