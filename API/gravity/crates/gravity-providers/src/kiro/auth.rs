//! Kiro OAuth token refresh.
//!
//! Mirrors `gravity/kiro/auth.py::refresh_access_token`.
//!
//! Three auth methods stored in `account.auth.extra["auth_method"]`:
//! - `"desktop"` — Kiro CLI app flow; refresh at `prod.{region}.auth.desktop.kiro.dev/refreshToken`
//! - `"social"` — web/social login; same endpoint
//! - `"idc"` — AWS IDC; refresh at `oidc.{region}.amazonaws.com/token` with client creds
//!
//! The refresh token may be encoded as `{rt}|{method}` or `{rt}|{cid}|{csecret}|idc`.

use crate::provider::Authenticator;
use async_trait::async_trait;
use gravity_core::{Error, ProviderAccount, ProviderErrorKind, ProviderName, Result};
use gravity_http::{HttpClient, HttpRequest, Profile};

const DEFAULT_REGION: &str = "us-east-1";
const KIRO_VERSION: &str = "0.6.4";

/// Authenticator for the Kiro (AWS) provider.
pub struct KiroAuthenticator;

#[async_trait]
impl Authenticator for KiroAuthenticator {
    async fn validate_credentials(&self, account: &ProviderAccount) -> Result<bool> {
        Ok(account.auth.inline_secret("access_token").map(|k| !k.is_empty()).unwrap_or(false))
    }

    async fn refresh_credentials(&self, account: &mut ProviderAccount) -> Result<()> {
        let raw_rt = account.auth.inline_secret("refresh_token")
            .ok_or_else(|| Error::Configuration("Kiro: no refresh_token stored".to_owned()))?
            .to_owned();
        let region = account.auth.inline_secret("region").unwrap_or(DEFAULT_REGION).to_owned();
        let oidc_region = account.auth.inline_secret("oidc_region")
            .or_else(|| account.auth.inline_secret("idc_region"))
            .map(str::to_owned);
        let proxy = account.transport.proxy_url.as_deref().map(str::to_owned);

        let (rt, auth_method, client_id, client_secret) = decode_refresh_token(&raw_rt);
        let auth_method = account.auth.inline_secret("auth_method")
            .map(str::to_owned)
            .unwrap_or(auth_method);

        let (url, payload, ua) = if auth_method == "idc" {
            let cid = client_id.or_else(|| account.auth.inline_secret("client_id").map(str::to_owned))
                .ok_or_else(|| Error::Configuration("Kiro IDC: missing client_id".to_owned()))?;
            let cs = client_secret.or_else(|| account.auth.inline_secret("client_secret").map(str::to_owned))
                .ok_or_else(|| Error::Configuration("Kiro IDC: missing client_secret".to_owned()))?;
            let r = oidc_region.as_deref().unwrap_or(&region);
            let url = format!("https://oidc.{r}.amazonaws.com/token");
            let payload = serde_json::json!({
                "refreshToken": rt, "clientId": cid, "clientSecret": cs, "grantType": "refresh_token"
            });
            let ua = "aws-sdk-js/3.738.0 ua/2.1 os/other lang/js md/browser#unknown_unknown api/sso-oidc#3.738.0 m/E KiroIDE".to_owned();
            (url, payload, ua)
        } else {
            let url = format!("https://prod.{region}.auth.desktop.kiro.dev/refreshToken");
            let payload = serde_json::json!({ "refreshToken": rt });
            let ua = if auth_method == "social" {
                format!("KiroIDE-{KIRO_VERSION}")
            } else {
                "aws-sdk-js/3.0.0 KiroIDE-0.1.0 os/other lang/js md/nodejs/18.0.0".to_owned()
            };
            (url, payload, ua)
        };

        let http = HttpClient::new();
        let req = HttpRequest::post(&url)
            .profile(Profile::Chrome)
            .header("content-type", "application/json")
            .header("accept", "application/json")
            .header("amz-sdk-request", "attempt=1; max=1")
            .header("x-amzn-kiro-agent-mode", "vibe")
            .header("user-agent", &ua)
            .header("connection", "close")
            .proxy(proxy)
            .body(bytes::Bytes::from(serde_json::to_vec(&payload).unwrap()));

        let resp = http.send(req).await.map_err(|e| Error::Network(e.to_string()))?;
        if !resp.is_success() {
            return Err(Error::Provider(gravity_core::ProviderError::new(
                ProviderName::Kiro, ProviderErrorKind::Authentication,
                "kiro_refresh_failed",
                format!("Kiro token refresh failed (status {})", resp.status),
            )));
        }
        let data: serde_json::Value = serde_json::from_slice(&resp.body)
            .map_err(|e| Error::Network(e.to_string()))?;
        let new_access = data.get("access_token").or_else(|| data.get("accessToken"))
            .and_then(|v| v.as_str())
            .ok_or_else(|| Error::Configuration("Kiro refresh: no access_token".to_owned()))?
            .to_owned();

        account.auth.extra.insert("access_token", new_access.clone());
        account.auth.extra.insert("api_key", new_access);

        // Update refresh token and expiry.
        let new_rt_raw = data.get("refresh_token").or_else(|| data.get("refreshToken"))
            .and_then(|v| v.as_str()).unwrap_or(&rt).to_owned();
        let encoded_rt = encode_refresh_token(&new_rt_raw, &auth_method, None, None);
        account.auth.extra.insert("refresh_token", encoded_rt);

        if let Some(exp) = data.get("expires_in").or_else(|| data.get("expiresIn")).and_then(|v| v.as_i64()) {
            let expires_at = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_millis() as i64)
                .unwrap_or(0) + exp * 1000;
            account.auth.extra.insert("expires_at", expires_at);
        }
        Ok(())
    }
}

/// Decode a potentially-encoded refresh token (`rt|method` or `rt|cid|csecret|idc`).
fn decode_refresh_token(encoded: &str) -> (String, String, Option<String>, Option<String>) {
    let parts: Vec<&str> = encoded.split('|').collect();
    if parts.len() < 2 {
        return (encoded.to_owned(), "desktop".to_owned(), None, None);
    }
    let rt = parts[0].to_owned();
    let method = *parts.last().unwrap_or(&"desktop");
    match method {
        "idc" => {
            let cid = parts.get(1).map(|s| s.to_string());
            let cs = parts.get(2).map(|s| s.to_string());
            (rt, "idc".to_owned(), cid, cs)
        }
        "social" => (rt, "social".to_owned(), None, None),
        _ => (rt, "desktop".to_owned(), None, None),
    }
}

fn encode_refresh_token(rt: &str, method: &str, cid: Option<&str>, cs: Option<&str>) -> String {
    match method {
        "idc" if cid.is_some() && cs.is_some() =>
            format!("{rt}|{}|{}|idc", cid.unwrap(), cs.unwrap()),
        "social" => format!("{rt}|social"),
        _ => format!("{rt}|desktop"),
    }
}
