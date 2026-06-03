"""Account management — provider accounts in DuckDB, indexed by RocksDB.

Handles all 40+ Gravity providers. Account data (the "major data") lives in
DuckDB cells; RocksDB holds the api_key→row pointers so reads are point lookups.
`get_db()` returns the shared RocksDB handle (plyvel-compatible API) used across
the codebase for the hot key-value index.
"""

import os
import json
import time
from typing import Any, Dict, Optional

from gravity.accounts import (
    AccountsFile, EmailBundle, ProviderAccount,
    AuthConfig, TransportConfig, ProviderDefaults, RotationState, QuotaState,
)
from gravity.capabilities import ProviderName, HistoryMode, ToolSupportMode, StructuredOutputMode, RemoteSessionMode

import kvstore
from providers import PROVIDERS, FREE_PROVIDERS

# ── Database ──────────────────────────────────────────────────────────────────
#
# Hot key-value index = RocksDB (kvstore). Bulk/account data = DuckDB (store).

def get_db():
    """Shared RocksDB handle (plyvel-compatible: get/put/delete/iterator)."""
    return kvstore.db()


# ── Provider enum lookup ──────────────────────────────────────────────────────

def _provider_enum(name: str) -> ProviderName:
    name = name.lower().strip()
    for p in ProviderName:
        if p.value == name:
            return p
    raise ValueError(f"Unknown provider: {name!r}. Valid values: {[p.value for p in ProviderName]}")


# ── ProviderAccount builder ───────────────────────────────────────────────────

def _build_defaults(provider: str) -> ProviderDefaults:
    meta = PROVIDERS.get(provider, {})
    ts_map = {"native": ToolSupportMode.NATIVE, "json_emulated": ToolSupportMode.JSON_EMULATED}
    so_map = {"native": StructuredOutputMode.NATIVE, "json_instruction": StructuredOutputMode.JSON_INSTRUCTION}
    rs_map = {"none": RemoteSessionMode.NONE, "required": RemoteSessionMode.REQUIRED, "optional": RemoteSessionMode.OPTIONAL}
    return ProviderDefaults(
        tool_support=ts_map.get(meta.get("tool_support", "native"), ToolSupportMode.NATIVE),
        structured_output=so_map.get(meta.get("structured_output", "json_instruction"), StructuredOutputMode.JSON_INSTRUCTION),
        remote_session_mode=rs_map.get(meta.get("remote_session_mode", "none"), RemoteSessionMode.NONE),
        supports_system_prompt=meta.get("supports_system_prompt", True),
        default_history_mode=HistoryMode.STATELESS,
    )


def build_provider_account(provider: str, account_id: str, payload: Dict[str, Any]) -> ProviderAccount:
    """Build a ProviderAccount for *any* provider from a flat payload dict.

    Payload fields (all optional unless noted):
      api_key           API key / refresh token / session key
      access_token      OAuth access token (oauth providers)
      refresh_token     OAuth refresh token (oauth providers)
      session_key       Session cookie value (claude, chatgpt)
      cookies           Dict of cookie name→value (gemini, chatgpt)
      org_id            Claude org UUID
      device_id         Claude device ID
      routing_hint      Claude routing hint
      project           Project ID (gemini_cli, antigravity)
      region            AWS region (kiro, default us-east-1)
      profile_arn       Kiro profile ARN
      email             OAuth account email
      expires_at        Token expiry epoch
      base_url          Override the default base URL
      pool              "default" | "custom" (default: "custom")
      priority          int (default: 100)
      weight            int (default: 1)
      timeout_ms        int (default: 120000)
      impersonate       TLS impersonation profile
      proxy_url         HTTP proxy URL
    """
    provider = provider.lower().strip()
    meta = PROVIDERS.get(provider, {})
    provider_enum = _provider_enum(provider)

    pool = payload.get("pool", "custom")
    priority = int(payload.get("priority", 100))
    weight = int(payload.get("weight", 1))
    timeout_ms = int(payload.get("timeout_ms", 120000))
    base_url = payload.get("base_url", meta.get("default_base_url", ""))

    # ── Build AuthConfig per-provider ─────────────────────────────────────────

    if provider in ("cursor",):
        raw_token = payload.get("api_key") or payload.get("refresh_token", "")
        access_token = _exchange_cursor_token(raw_token) if raw_token else ""
        auth = AuthConfig(
            kind="api_key",
            secret_ref=f"sec_{account_id}",
            extra={"api_key": access_token, "refresh_token": raw_token},
        )

    elif provider == "windsurf":
        raw_token = payload.get("api_key") or payload.get("access_token", "")
        jwt_token = _exchange_windsurf_token(raw_token)
        auth = AuthConfig(
            kind="api_key",
            secret_ref=f"sec_{account_id}",
            extra={"api_key": jwt_token, "raw_token": raw_token},
        )

    elif provider == "claude":
        session_key = (
            payload.get("api_key")
            or payload.get("session_key")
            or payload.get("access_token", "")
        )
        auth = AuthConfig(
            kind="session_cookie",
            secret_ref=f"sec_{account_id}",
            org_id=payload.get("org_id"),
            device_id=payload.get("device_id"),
            routing_hint=payload.get("routing_hint"),
            extra={"session_key": session_key},
        )

    elif provider == "chatgpt":
        token = payload.get("api_key") or payload.get("access_token", "")
        cookies = payload.get("cookies", {})
        if isinstance(cookies, dict):
            extra = {**cookies, "access_token": token}
        else:
            extra = {"access_token": token}
        auth = AuthConfig(
            kind="session_cookie" if token else "anonymous",
            secret_ref=f"sec_{account_id}",
            chatgpt_account_id=payload.get("chatgpt_account_id"),
            device_id=payload.get("device_id"),
            extra=extra,
        )

    elif provider == "gemini":
        cookies = payload.get("cookies", {})
        at = payload.get("at") or payload.get("api_key", "")
        bl = payload.get("bl", "")
        fsid = payload.get("f_sid") or payload.get("fsid", "")
        extra = {
            **(cookies if isinstance(cookies, dict) else {}),
            "at": at, "bl": bl, "f.sid": fsid,
        }
        auth = AuthConfig(
            kind="session_cookie" if (cookies or at) else "anonymous",
            secret_ref=f"sec_{account_id}",
            extra=extra,
        )

    elif provider == "kiro":
        token = payload.get("api_key") or payload.get("access_token", "")
        auth = AuthConfig(
            kind="api_key",
            secret_ref=f"sec_{account_id}",
            extra={
                "api_key": token,
                "region": payload.get("region", "us-east-1"),
                "profile_arn": payload.get("profile_arn", ""),
            },
        )

    elif provider == "glm":
        token = payload.get("api_key") or payload.get("access_token", "")
        user_id = payload.get("user_id", "")
        auth = AuthConfig(
            kind="api_key" if token else "anonymous",
            secret_ref=f"sec_{account_id}",
            extra={
                "api_key": token,
                "user_id": user_id,
            },
        )

    elif provider in ("antigravity", "gemini_cli"):
        access_token = payload.get("access_token") or payload.get("api_key", "")
        refresh_token = payload.get("refresh_token", "")
        auth = AuthConfig(
            kind="oauth",
            secret_ref=f"sec_{account_id}",
            extra={
                "access_token": access_token,
                "refresh_token": refresh_token,
                "email": payload.get("email", ""),
                "expires_at": int(payload.get("expires_at", 0)),
                "project": payload.get("project") or payload.get("project_id", ""),
            },
        )

    elif provider in ("codex",):
        token = payload.get("api_key") or payload.get("access_token", "")
        auth = AuthConfig(
            kind="api_key",
            secret_ref=f"sec_{account_id}",
            chatgpt_account_id=payload.get("chatgpt_account_id"),
            extra={"api_key": token},
        )

    elif provider == "perplexity":
        token = payload.get("api_key") or payload.get("session_token", "")
        auth = AuthConfig(
            kind="session_cookie" if token else "anonymous",
            secret_ref=f"sec_{account_id}",
            extra={"session_token": token},
        )

    elif provider in FREE_PROVIDERS - {"gemini", "perplexity", "chatgpt", "glm"}:
        # Local servers: ollama, llama_cpp, local_transformers, litellm
        auth = AuthConfig(
            kind="anonymous",
            secret_ref=f"sec_{account_id}",
            extra={},
        )

    else:
        # All OAI-compat API-key providers (groq, openai_api, mistral, etc.)
        api_key = payload.get("api_key") or payload.get("access_token", "")
        auth = AuthConfig(
            kind="api_key",
            secret_ref=f"sec_{account_id}",
            extra={"api_key": api_key},
        )

    transport = TransportConfig(
        base_url=base_url,
        timeout_ms=timeout_ms,
        impersonate=payload.get("impersonate"),
        proxy_url=payload.get("proxy_url"),
        verify_ssl=payload.get("verify_ssl", True),
        extra_headers=payload.get("extra_headers", {}),
    )

    return ProviderAccount(
        account_id=account_id,
        provider=provider_enum,
        enabled=payload.get("enabled", True),
        pool=pool,
        priority=priority,
        weight=weight,
        auth=auth,
        transport=transport,
        rotation_state=RotationState(),
        quota=QuotaState(),
        defaults=_build_defaults(provider),
        metadata={"source": "api", "added_at": int(time.time())},
    )


# ── Token exchange helpers ────────────────────────────────────────────────────

def _exchange_cursor_token(token: str) -> str:
    """Exchange a Cursor refresh token for an access token."""
    if not token or len(token) < 20:
        return token
    try:
        import httpx
        for domain in ["prod.authentication.cursor.sh", "authentication.cursor.sh"]:
            try:
                r = httpx.post(
                    f"https://{domain}/oauth/token",
                    json={
                        "grant_type": "refresh_token",
                        "client_id": "KbZUR41cY7W6zRSdpSUJ7I7mLYBKOCmB",
                        "refresh_token": token,
                    },
                    timeout=15,
                )
                if r.status_code == 200:
                    data = r.json()
                    new = data.get("access_token") or data.get("accessToken")
                    if new:
                        return new
            except Exception:
                continue
    except ImportError:
        pass
    return token


def _exchange_windsurf_token(token: str) -> str:
    """Exchange a Windsurf OTT/activation token for an API server token."""
    if not token or not (token.startswith("ott$") or len(token.split(".")) >= 3):
        return token
    try:
        from gravity.windsurf import register_windsurf_user
        exchanged, _name, _url = register_windsurf_user(token)
        return exchanged or token
    except Exception:
        return token


# ── Account CRUD ──────────────────────────────────────────────────────────────

def add_account_to_db(api_key: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    """Register a provider account linked to an API key."""
    account_id = payload.get("account_id")
    provider = payload.get("provider", "").lower().strip()
    if not account_id:
        raise ValueError("Missing required field: 'account_id'")
    if not provider:
        raise ValueError("Missing required field: 'provider'")
    if provider not in PROVIDERS:
        raise ValueError(
            f"Unknown provider {provider!r}. Supported: {sorted(PROVIDERS.keys())}"
        )

    account_model = build_provider_account(provider, account_id, payload)
    import store
    store.upsert_account(
        api_key, account_id, provider,
        enabled=account_model.enabled, pool=account_model.pool,
        priority=account_model.priority, weight=account_model.weight,
        gravity_account=account_model.model_dump(mode="json"),
    )
    return {
        "status": "success",
        "account_id": account_id,
        "provider": provider,
        "pool": account_model.pool,
    }


def remove_account_from_db(api_key: str, account_id: str, provider: str) -> Dict[str, Any]:
    """Remove a specific account."""
    import store
    removed = store.delete_account(api_key, account_id, provider)
    return {"status": "success" if removed else "not_found", "removed": removed}


def list_accounts_for_key(api_key: str) -> list[Dict[str, Any]]:
    """Account summaries (no secrets) for an API key."""
    import store
    return store.list_account_summaries(api_key)


def load_accounts_for_key(api_key: str) -> AccountsFile:
    """Build an AccountsFile from all accounts linked to an API key."""
    import store
    providers_list: list[ProviderAccount] = []
    for acc_dict in store.get_accounts(api_key):
        try:
            providers_list.append(ProviderAccount.model_validate(acc_dict))
        except Exception as e:
            print(f"[account] skip invalid: {e}")
    return AccountsFile(
        schema_version=4,
        security={"secrets_encrypted": False},
        emails=[EmailBundle(
            email="user@gravity.local", enabled=True,
            display_name="Gateway User", providers=providers_list,
        )],
    )


def get_providers_for_key(api_key: str) -> set[str]:
    """Provider names the key has enabled accounts for (DuckDB DISTINCT)."""
    import store
    return store.providers_for_key(api_key)


# ── Metrics (DuckDB analytics via request log) ─────────────────────────────────

def record_request(model: str, api_key: str = "", ip: str = "", provider: str = "",
                   status: str = "ok", cached: bool = False, latency_ms: float = 0.0,
                   in_tokens: int = 0, out_tokens: int = 0):
    try:
        import store
        if not provider:
            provider = model.split("/", 1)[0] if "/" in model else model
        store.log_request(api_key, ip, provider, model, status, cached,
                          latency_ms, in_tokens, out_tokens)
    except Exception as e:
        print(f"[metrics] record error: {e}")


def get_metrics() -> Dict[str, Any]:
    try:
        import store
        return store.metrics_summary()
    except Exception as e:
        return {"total_requests": 0, "most_used_model": "none", "status": "error", "error": str(e)}
