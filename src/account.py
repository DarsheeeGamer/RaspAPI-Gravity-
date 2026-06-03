"""Account management — CRUD for ProviderAccounts in LevelDB.

Handles all 40+ Gravity providers. Each API key maps to a tenant-isolated
bag of ProviderAccount objects stored as JSON in LevelDB.
"""

import os
import json
import time
import plyvel
from typing import Any, Dict, Optional

from gravity.accounts import (
    AccountsFile, EmailBundle, ProviderAccount,
    AuthConfig, TransportConfig, ProviderDefaults, RotationState, QuotaState,
)
from gravity.capabilities import ProviderName, HistoryMode, ToolSupportMode, StructuredOutputMode, RemoteSessionMode

from providers import PROVIDERS, FREE_PROVIDERS

# ── Database ──────────────────────────────────────────────────────────────────

DB_DIR = os.path.expanduser("~/nodataishere/db")
os.makedirs(DB_DIR, exist_ok=True)
_db: Optional[plyvel.DB] = None

def get_db() -> plyvel.DB:
    global _db
    if _db is None:
        _db = plyvel.DB(DB_DIR, create_if_missing=True)
    return _db


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

    db_conn = get_db()
    user_key = f"user_accounts:{api_key}".encode()
    existing_bytes = db_conn.get(user_key)
    accounts_list = json.loads(existing_bytes.decode()) if existing_bytes else []
    accounts_list = [a for a in accounts_list if not (
        a.get("account_id") == account_id and a.get("provider") == provider
    )]
    accounts_list.append(account_model.model_dump(mode="json"))
    db_conn.put(user_key, json.dumps(accounts_list).encode())

    return {
        "status": "success",
        "account_id": account_id,
        "provider": provider,
        "pool": account_model.pool,
    }


def remove_account_from_db(api_key: str, account_id: str, provider: str) -> Dict[str, Any]:
    """Remove a specific account."""
    db_conn = get_db()
    user_key = f"user_accounts:{api_key}".encode()
    existing_bytes = db_conn.get(user_key)
    if not existing_bytes:
        return {"status": "not_found"}
    accounts_list = json.loads(existing_bytes.decode())
    before = len(accounts_list)
    accounts_list = [a for a in accounts_list if not (
        a.get("account_id") == account_id and a.get("provider") == provider
    )]
    db_conn.put(user_key, json.dumps(accounts_list).encode())
    removed = before - len(accounts_list)
    return {"status": "success", "removed": removed}


def list_accounts_for_key(api_key: str) -> list[Dict[str, Any]]:
    """Return raw account dicts (without secrets) for an API key."""
    db_conn = get_db()
    user_key = f"user_accounts:{api_key}".encode()
    existing_bytes = db_conn.get(user_key)
    if not existing_bytes:
        return []
    accounts = json.loads(existing_bytes.decode())
    # Strip sensitive fields before returning
    safe = []
    for a in accounts:
        entry = {k: v for k, v in a.items() if k != "auth"}
        entry["provider"] = a.get("provider")
        entry["account_id"] = a.get("account_id")
        entry["pool"] = a.get("pool")
        entry["enabled"] = a.get("enabled", True)
        safe.append(entry)
    return safe


def load_accounts_for_key(api_key: str) -> AccountsFile:
    """Build an AccountsFile from all accounts linked to an API key."""
    db_conn = get_db()
    user_key = f"user_accounts:{api_key}".encode()
    existing_bytes = db_conn.get(user_key)
    providers_list: list[ProviderAccount] = []
    if existing_bytes:
        for acc_dict in json.loads(existing_bytes.decode()):
            try:
                providers_list.append(ProviderAccount.model_validate(acc_dict))
            except Exception as e:
                print(f"[account] skip invalid: {e}")

    return AccountsFile(
        schema_version=4,
        security={"secrets_encrypted": False},
        emails=[EmailBundle(
            email="user@gravity.local",
            enabled=True,
            display_name="Gateway User",
            providers=providers_list,
        )],
    )


def get_providers_for_key(api_key: str) -> set[str]:
    """Return the set of provider names the key has accounts for."""
    db_conn = get_db()
    user_key = f"user_accounts:{api_key}".encode()
    existing_bytes = db_conn.get(user_key)
    if not existing_bytes:
        return set()
    return {a.get("provider") for a in json.loads(existing_bytes.decode()) if a.get("enabled", True)}


# ── Metrics ───────────────────────────────────────────────────────────────────

def record_request(model: str, api_key: str = ""):
    try:
        db = get_db()
        for key in (b"metrics:total_requests", f"metrics:key:{api_key}:total".encode()):
            val = db.get(key)
            db.put(key, str(int(val.decode()) + 1 if val else 1).encode())
        mkey = f"metrics:model:{model}".encode()
        val = db.get(mkey)
        db.put(mkey, str(int(val.decode()) + 1 if val else 1).encode())
    except Exception as e:
        print(f"[metrics] record error: {e}")


def get_metrics() -> Dict[str, Any]:
    try:
        db = get_db()
        total_b = db.get(b"metrics:total_requests")
        total = int(total_b.decode()) if total_b else 0
        model_counts: Dict[str, int] = {}
        for key, val in db.iterator(prefix=b"metrics:model:"):
            model_name = key[len(b"metrics:model:"):].decode()
            model_counts[model_name] = int(val.decode())
        top_model = max(model_counts, key=model_counts.get) if model_counts else "none"
        return {
            "total_requests": total,
            "most_used_model": top_model,
            "model_breakdown": model_counts,
            "status": "healthy",
        }
    except Exception as e:
        return {"total_requests": 0, "most_used_model": "none", "status": "error", "error": str(e)}
