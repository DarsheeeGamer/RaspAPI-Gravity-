//! Claude web auth: resolving credentials and building the cookie/header set.
//!
//! Mirrors `gravity/claude/auth.py`.

use super::constants;
use gravity_core::{Error, ProviderAccount, Result};

/// Resolved Claude web credentials, gathered from the account + resolved secret.
pub struct ClaudeAuth {
    /// `sessionKey` cookie value.
    pub session_key: String,
    /// `anthropic-device-id` (stable per account).
    pub device_id: String,
    /// Organization id, if known.
    pub org_id: Option<String>,
    /// Routing hint (cookie + header).
    pub routing_hint: Option<String>,
    /// Client version header (default `1.0.0`).
    pub client_version: String,
    /// Optional client SHA header.
    pub client_sha: Option<String>,
    /// Optional anonymous id header.
    pub anonymous_id: Option<String>,
    /// Extra cookies carried verbatim.
    pub extra_cookies: Vec<(String, String)>,
    /// Extra headers from transport config.
    pub extra_headers: Vec<(String, String)>,
}

impl ClaudeAuth {
    /// Build from an account and its resolved secret (the session key).
    ///
    /// Accepts `auth.kind` of `session_cookie`/`session`; the session key comes
    /// from the resolved secret, an inline `extra.session_key`, or a
    /// `sk-ant-sid…` `secret_ref`.
    pub fn from_account(account: &ProviderAccount, secret: &str) -> Result<Self> {
        let auth = &account.auth;
        let session_key = if !secret.is_empty() {
            secret.to_owned()
        } else if let Some(k) = auth.inline_secret("session_key") {
            k.to_owned()
        } else if auth.secret_ref.starts_with("sk-ant-sid") {
            auth.secret_ref.to_string()
        } else {
            return Err(Error::Configuration(format!(
                "missing Claude session key for account {}",
                account.account_id
            )));
        };

        // Device ID: use stored value, or auto-generate from account_id for consistency.
        let device_id = auth
            .device_id
            .as_ref()
            .map(|d| d.to_string())
            .or_else(|| auth.inline_secret("device_id").map(str::to_owned))
            .unwrap_or_else(|| {
                // Deterministic from account_id so it's stable across requests.
                format!("gravity-dev-{}", &account.account_id)
            });

        let extra_cookies = auth
            .extra
            .as_map()
            .map(|m| {
                m.iter()
                    .filter_map(|(k, v)| v.as_str().map(|s| (k.clone(), s.to_owned())))
                    .collect()
            })
            .unwrap_or_default();
        let extra_headers = account
            .transport
            .extra_headers
            .iter()
            .map(|(k, v)| (k.clone(), v.clone()))
            .collect();

        Ok(ClaudeAuth {
            session_key,
            device_id,
            org_id: auth.org_id.as_ref().map(|s| s.to_string()),
            routing_hint: auth
                .routing_hint
                .as_ref()
                .map(|s| s.to_string())
                .or_else(|| auth.inline_secret("routing_hint").map(str::to_owned)),
            client_version: auth
                .inline_secret("client_version")
                .unwrap_or("1.0.0")
                .to_owned(),
            client_sha: auth.inline_secret("client_sha").map(str::to_owned),
            anonymous_id: auth.inline_secret("anonymous_id").map(str::to_owned),
            extra_cookies,
            extra_headers,
        })
    }

    /// The resolved org id (explicit or `lastActiveOrg`).
    pub fn resolved_org_id(&self) -> Option<&str> {
        self.org_id.as_deref()
    }

    /// Build the `Cookie` header value.
    pub fn cookie_header(&self) -> String {
        let mut cookies: Vec<(String, String)> = vec![
            ("sessionKey".into(), self.session_key.clone()),
            ("anthropic-device-id".into(), self.device_id.clone()),
            ("CH-prefers-color-scheme".into(), "dark".into()),
            ("cookie_seed_done".into(), "1".into()),
            ("app-shell-mode".into(), "gate-disabled".into()),
        ];
        if let Some(rh) = &self.routing_hint {
            cookies.push(("routingHint".into(), rh.clone()));
        }
        if let Some(org) = self.resolved_org_id() {
            cookies.push(("lastActiveOrg".into(), org.to_owned()));
        }
        for (k, v) in &self.extra_cookies {
            if matches!(k.as_str(), "routingHint" | "lastActiveOrg") {
                cookies.push((k.clone(), v.clone()));
            }
        }
        cookies
            .iter()
            .map(|(k, v)| format!("{k}={v}"))
            .collect::<Vec<_>>()
            .join("; ")
    }

    /// Build the full header list for a request (`stream` selects the `accept`).
    pub fn headers(&self, stream: bool) -> Vec<(String, String)> {
        let mut h: Vec<(String, String)> = vec![
            ("accept".into(), if stream { "text/event-stream".into() } else { "*/*".into() }),
            ("accept-language".into(), "en-US,en;q=0.9".into()),
            ("anthropic-client-platform".into(), "web_claude_ai".into()),
            ("anthropic-client-version".into(), self.client_version.clone()),
            ("content-type".into(), "application/json".into()),
            ("origin".into(), constants::ORIGIN.into()),
            ("referer".into(), constants::CHAT_REFERER.into()),
            ("user-agent".into(), constants::USER_AGENT.into()),
            ("anthropic-device-id".into(), self.device_id.clone()),
            ("cookie".into(), self.cookie_header()),
        ];
        if let Some(rh) = &self.routing_hint {
            h.push(("anthropic-routing-hint".into(), rh.clone()));
        }
        if let Some(sha) = &self.client_sha {
            h.push(("anthropic-client-sha".into(), sha.clone()));
        }
        if let Some(anon) = &self.anonymous_id {
            h.push(("anthropic-anonymous-id".into(), anon.clone()));
        }
        for (k, v) in &self.extra_headers {
            h.push((k.clone(), v.clone()));
        }
        h
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use gravity_core::{
        AuthConfig, HistoryMode, Metadata, ProviderDefaults, ProviderName, RemoteSessionMode,
        StructuredOutputMode, ToolSupportMode, TransportConfig,
    };

    fn account() -> ProviderAccount {
        ProviderAccount {
            account_id: "a".into(),
            provider: ProviderName::Claude,
            enabled: true,
            pool: "default".into(),
            priority: 100,
            weight: 1,
            auth: AuthConfig {
                kind: "session_cookie".into(),
                secret_ref: "ref".into(),
                account_label: None,
                org_id: Some("org-123".into()),
                device_id: Some("dev-abc".into()),
                routing_hint: Some("rh-9".into()),
                chatgpt_account_id: None,
                refresh_secret_ref: None,
                extra: Metadata::new(),
            },
            transport: TransportConfig {
                base_url: constants::BASE_URL.into(),
                timeout_ms: 60000,
                impersonate: None,
                proxy_url: None,
                verify_ssl: true,
                extra_headers: Default::default(),
            },
            rotation_state: Default::default(),
            quota: Default::default(),
            defaults: ProviderDefaults {
                tool_support: ToolSupportMode::Native,
                structured_output: StructuredOutputMode::JsonInstruction,
                remote_session_mode: RemoteSessionMode::Required,
                supports_system_prompt: true,
                default_history_mode: HistoryMode::Auto,
            },
            metadata: Metadata::new(),
        }
    }

    #[test]
    fn builds_cookie_and_headers() {
        let auth = ClaudeAuth::from_account(&account(), "sk-ant-sid-XYZ").unwrap();
        let cookie = auth.cookie_header();
        assert!(cookie.contains("sessionKey=sk-ant-sid-XYZ"));
        assert!(cookie.contains("anthropic-device-id=dev-abc"));
        assert!(cookie.contains("routingHint=rh-9"));
        assert!(cookie.contains("lastActiveOrg=org-123"));

        let headers = auth.headers(true);
        assert!(headers.iter().any(|(k, v)| k == "accept" && v == "text/event-stream"));
        assert!(headers.iter().any(|(k, v)| k == "anthropic-routing-hint" && v == "rh-9"));
    }

    #[test]
    fn auto_generates_device_id_when_missing() {
        let mut acct = account();
        acct.auth.device_id = None;
        // No error — device_id is auto-generated from account_id.
        let auth = ClaudeAuth::from_account(&acct, "sk-ant-sid-x").unwrap();
        assert!(auth.device_id.starts_with("gravity-dev-"));
    }
}
