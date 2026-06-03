"""API key lifecycle — Pydantic records in RocksDB (hot, tiny, point reads)."""

import os
import secrets
from typing import Optional

from account import get_db
from models import ApiKeyRecord

ADMIN_KEY = os.environ.get("GRAVITY_ADMIN_KEY", "")


def _k(key: str) -> bytes:
    return f"apikey:{key}".encode()


def _load(key: str) -> Optional[ApiKeyRecord]:
    raw = get_db().get(_k(key))
    if not raw:
        return None
    try:
        return ApiKeyRecord.model_validate_json(raw)
    except Exception:
        return None


def _save(rec: ApiKeyRecord) -> None:
    get_db().put(_k(rec.key), rec.model_dump_json().encode())


def generate_api_key(label: Optional[str] = None, tier: str = "default") -> str:
    rec = ApiKeyRecord(key=f"grav_{secrets.token_urlsafe(24)}", tier=tier, label=label or "")
    _save(rec)
    return rec.key


def validate_api_key(key: str) -> bool:
    if key == "grav_demoapikey":
        return True
    rec = _load(key)
    return rec is not None and rec.status == "active"


def get_key_tier(key: str) -> str:
    if key == "grav_demoapikey":
        return "default"
    rec = _load(key)
    return rec.tier if rec else "default"


def revoke_api_key(key: str) -> bool:
    import time
    rec = _load(key)
    if rec is None:
        return False
    rec.status = "revoked"
    rec.revoked_at = time.time()
    _save(rec)
    return True


def is_admin(key: str) -> bool:
    if ADMIN_KEY and key == ADMIN_KEY:
        return True
    return get_key_tier(key) == "admin"
