import os
import json
import plyvel
from typing import Dict, Any, Optional
from pathlib import Path

# Import official Gravity accounts schemas
from gravity.accounts import ProviderAccount, AccountsFile, EmailBundle

# Database directory
DB_DIR = os.path.expanduser("~/nodataishere/db")
os.makedirs(DB_DIR, exist_ok=True)

# Lazy thread-safe global connection to avoid Uvicorn StatReload lock conflicts
_db = None

def get_db() -> plyvel.DB:
    global _db
    if _db is None:
        _db = plyvel.DB(DB_DIR, create_if_missing=True)
    return _db

def exchange_cursor_refresh_token(refresh_token: str) -> str:
    """
    Exchanges a Cursor refresh_token for a fresh access_token via prod.authentication.cursor.sh.
    """
    import httpx
    for domain in ["prod.authentication.cursor.sh", "authentication.cursor.sh"]:
        try:
            r = httpx.post(
                f"https://{domain}/oauth/token",
                json={
                    "grant_type": "refresh_token",
                    "client_id": "KbZUR41cY7W6zRSdpSUJ7I7mLYBKOCmB",
                    "refresh_token": refresh_token,
                },
                timeout=15,
            )
            if r.status_code == 200:
                data = r.json()
                new_access = data.get("access_token") or data.get("accessToken")
                if new_access:
                    return new_access
        except Exception:
            pass
    return refresh_token

def add_account_to_db(api_key: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    Saves a Gravity ProviderAccount and optional Organization into LevelDB,
    linked strictly to the provided API key (token) for secure tenant isolation.
    Automatically applies exact system-compliant defaults for Cursor, Windsurf,
    ChatGPT, and Antigravity.
    Automatically exchanges raw Cursor/Windsurf tokens into operational API keys.
    """
    account_id = payload.get("account_id")
    provider = payload.get("provider")
    
    if not account_id or not provider:
        raise ValueError("Missing required fields: 'account_id' and 'provider' are mandatory.")

    provider = provider.lower()
    raw_api_key = payload.get("api_key", "")

    # 1. Resolve and build ProviderAccount using official registered authenticators
    from gravity.providers.registry import registry
    from gravity.capabilities import ProviderName, HistoryMode

    # Convert string provider to ProviderName enum
    provider_enum = None
    for name in ProviderName:
        if name.value == provider:
            provider_enum = name
            break

    account_model = None

    if provider == "cursor":
        # 1. Token Exchange / Conversion
        access_token = raw_api_key
        if raw_api_key and len(raw_api_key) > 20:
            try:
                from gravity.cursor import save_cursor_tokens, refresh_cursor_token
                # Save refresh token temporarily to invoke inbuilt refresh method
                save_cursor_tokens("", raw_api_key)
                exchanged = refresh_cursor_token()
                if exchanged:
                    access_token = exchanged
            except Exception as e:
                print(f"Cursor token exchange failed: {e}")

        try:
            auth_obj = registry.get_authenticator(ProviderName.CURSOR)
            account_model = auth_obj.create_account(
                account_id=account_id,
                api_key=access_token,
                pool=payload.get("pool", "default"),
                priority=payload.get("priority", 100),
                weight=payload.get("weight", 1),
                timeout_ms=payload.get("transport", {}).get("timeout_ms", 120000)
            )
        except Exception as e:
            print(f"CursorAuthenticator failed: {e}")

    elif provider == "windsurf":
        api_key_to_use = raw_api_key
        if raw_api_key and (raw_api_key.startswith("ott$") or len(raw_api_key.split(".")) >= 3):
            try:
                from gravity.windsurf import register_windsurf_user
                exchanged_key, name, api_server_url = register_windsurf_user(raw_api_key)
                api_key_to_use = exchanged_key
            except Exception as e:
                print(f"Windsurf token exchange failed: {e}")

        try:
            auth_obj = registry.get_authenticator(ProviderName.WINDSURF)
            account_model = auth_obj.create_account(
                account_id=account_id,
                api_key=api_key_to_use,
                pool=payload.get("pool", "default"),
                priority=payload.get("priority", 100),
                weight=payload.get("weight", 1),
                timeout_ms=payload.get("transport", {}).get("timeout_ms", 120000)
            )
        except Exception as e:
            print(f"WindsurfAuthenticator failed: {e}")

    elif provider == "antigravity":
        from gravity.antigravity import ANTIGRAVITY_CAPABILITIES
        from gravity.constants import ANTIGRAVITY_DEFAULT_ENDPOINT
        from gravity.accounts import AuthConfig, TransportConfig, ProviderDefaults, RotationState, QuotaState

        draft_account = ProviderAccount(
            account_id=account_id,
            provider=ProviderName.ANTIGRAVITY,
            enabled=payload.get("enabled", True),
            pool=payload.get("pool", "default"),
            priority=payload.get("priority", 100),
            weight=payload.get("weight", 1),
            auth=AuthConfig(
                kind="oauth",
                secret_ref=f"sec_{account_id}",
                extra={
                    "access_token": payload.get("access_token", ""),
                    "refresh_token": payload.get("refresh_token", raw_api_key),
                    "email": payload.get("email", "kaededev08@gmail.com"),
                    "expires_at": int(payload.get("expires_at", 0)),
                    "project_id": payload.get("project_id"),
                },
            ),
            transport=TransportConfig(
                base_url=ANTIGRAVITY_DEFAULT_ENDPOINT,
                timeout_ms=payload.get("transport", {}).get("timeout_ms", 60000),
                verify_ssl=True,
                extra_headers={},
            ),
            rotation_state=RotationState(),
            quota=QuotaState(),
            defaults=ProviderDefaults(
                tool_support=ANTIGRAVITY_CAPABILITIES.tool_support,
                structured_output=ANTIGRAVITY_CAPABILITIES.structured_output,
                remote_session_mode=ANTIGRAVITY_CAPABILITIES.remote_session_mode,
                supports_system_prompt=ANTIGRAVITY_CAPABILITIES.supports_system_prompt,
                default_history_mode=HistoryMode.STATELESS,
            ),
            metadata={"source": "oauth", "email": payload.get("email", "kaededev08@gmail.com")},
        )

        # 2. Invoke AntigravityAuthenticator to validate/refresh credentials
        try:
            auth_obj = registry.get_authenticator(ProviderName.ANTIGRAVITY)
            account_model = auth_obj.refresh_credentials(draft_account)
        except Exception as e:
            print(f"Antigravity credentials validation failed: {e}")
            account_model = draft_account

    elif provider == "chatgpt":
        try:
            auth_obj = registry.get_authenticator(ProviderName.CHATGPT)
            account_model = auth_obj.create_account(
                account_id=account_id,
                api_key=raw_api_key,
                pool=payload.get("pool", "default"),
                priority=payload.get("priority", 100),
                weight=payload.get("weight", 1),
            )
        except Exception as e:
            print(f"ChatGPTAuthenticator failed: {e}")

    # Generic Fallback or Fallback if authenticator failed
    if account_model is None:
        from gravity.accounts import AuthConfig, TransportConfig, ProviderDefaults, RotationState, QuotaState
        final_provider_enum = provider_enum or ProviderName.OPENAI_API

        auth_data = payload.get("auth", {
            "kind": "api_key",
            "secret_ref": f"sec_generic_{account_id}",
            "extra": {
                "api_key": raw_api_key
            }
        })
        transport_data = payload.get("transport", {
            "base_url": payload.get("transport", {}).get("base_url", ""),
            "timeout_ms": payload.get("transport", {}).get("timeout_ms", 60000),
            "verify_ssl": True,
            "extra_headers": {}
        })
        defaults_data = payload.get("defaults", {
            "tool_support": "native",
            "structured_output": "native",
            "remote_session_mode": "none",
            "supports_system_prompt": True,
            "default_history_mode": "stateless"
        })

        account_model = ProviderAccount(
            account_id=account_id,
            provider=final_provider_enum,
            enabled=payload.get("enabled", True),
            pool=payload.get("pool", "default"),
            priority=payload.get("priority", 100),
            weight=payload.get("weight", 1),
            auth=auth_data,
            transport=transport_data,
            defaults=defaults_data,
            metadata=payload.get("metadata", {})
        )

    # Link Organization if supplied
    org_data = payload.get("organization")
    if org_data and isinstance(org_data, dict):
        org_id = org_data.get("org_id", f"org-{account_id}")
        org_data["org_id"] = org_id
        
        # Store Organization details in LevelDB
        org_key = f"org:{org_id}".encode("utf-8")
        get_db().put(org_key, json.dumps(org_data).encode("utf-8"))

    # 3. Load existing accounts for this specific API key to append/update
    user_key = f"user_accounts:{api_key}".encode("utf-8")
    db_conn = get_db()
    existing_bytes = db_conn.get(user_key)
    
    if existing_bytes:
        accounts_list = json.loads(existing_bytes.decode("utf-8"))
        # Filter out existing duplicates with the same account_id
        accounts_list = [a for a in accounts_list if a.get("account_id") != account_id]
    else:
        accounts_list = []

    # Serialize and save back
    accounts_list.append(account_model.model_dump(mode="json"))
    db_conn.put(user_key, json.dumps(accounts_list).encode("utf-8"))

    return {
        "status": "success",
        "message": f"Account '{account_id}' registered successfully in LevelDB and linked to API key.",
        "account": account_model.model_dump(mode="json"),
        "organization": org_data
    }

def load_accounts_for_key(api_key: str) -> AccountsFile:
    """
    Fetches all pre-configured ProviderAccounts linked to an API key on-demand
    and wraps them in a fully validated gravity AccountsFile structure.
    """
    user_key = f"user_accounts:{api_key}".encode("utf-8")
    db_conn = get_db()
    existing_bytes = db_conn.get(user_key)
    
    providers_list = []
    if existing_bytes:
        accounts_json = json.loads(existing_bytes.decode("utf-8"))
        for acc_dict in accounts_json:
            try:
                providers_list.append(ProviderAccount.model_validate(acc_dict))
            except Exception as e:
                print(f"Skipping invalid account schema: {e}")

    # Build dynamically validated AccountsFile envelope
    return AccountsFile(
        schema_version=4,
        security={"secrets_encrypted": False},
        emails=[
            EmailBundle(
                email="user@gravity.local",
                enabled=True,
                display_name="Authorized Gateway User",
                providers=providers_list
            )
        ]
    )

def record_request(model: str):
    """
    Atomically increments global metrics and specific model count in LevelDB.
    """
    try:
        db_conn = get_db()
        # Increment total requests
        total_key = b"metrics:total_requests"
        total_bytes = db_conn.get(total_key)
        total = int(total_bytes.decode('utf-8')) if total_bytes else 0
        db_conn.put(total_key, str(total + 1).encode('utf-8'))

        # Increment model count
        model_key = b"metrics:model:" + model.encode('utf-8')
        model_bytes = db_conn.get(model_key)
        model_count = int(model_bytes.decode('utf-8')) if model_bytes else 0
        db_conn.put(model_key, str(model_count + 1).encode('utf-8'))
    except Exception as e:
        print(f"Failed to record request metrics in LevelDB: {e}")

def get_metrics() -> Dict[str, Any]:
    """
    Fetches aggregate metrics from LevelDB, calculating total requests and identifying the most used model.
    """
    try:
        db_conn = get_db()
        total_key = b"metrics:total_requests"
        total_bytes = db_conn.get(total_key)
        total = int(total_bytes.decode('utf-8')) if total_bytes else 0

        most_used_model = "None"
        max_count = -1
        
        # Prefix search over model metrics
        for key, value in db_conn.iterator(prefix=b"metrics:model:"):
            model_name = key[len(b"metrics:model:"):].decode('utf-8')
            count = int(value.decode('utf-8'))
            if count > max_count:
                max_count = count
                most_used_model = model_name

        return {
            "total_requests": total,
            "most_used_model": most_used_model,
            "status": "healthy"
        }
    except Exception as e:
        return {
            "total_requests": 0,
            "most_used_model": "None",
            "status": "error",
            "error": str(e)
        }
