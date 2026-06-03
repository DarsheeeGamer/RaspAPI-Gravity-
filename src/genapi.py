"""API key management — generation and validation via LevelDB."""

import secrets
import json
import time
import os
from typing import Optional

from account import get_db

# Server-side admin key — set this env var to protect admin endpoints.
ADMIN_KEY = os.environ.get("GRAVITY_ADMIN_KEY", "")


def generate_api_key(label: Optional[str] = None, tier: str = "default") -> str:
    """Generate and persist a new active API key. Returns the full key with grav_ prefix."""
    raw = secrets.token_urlsafe(24)
    full_key = f"grav_{raw}"
    db = get_db()
    db.put(
        f"apikey:{full_key}".encode(),
        json.dumps({
            "key": full_key,
            "status": "active",
            "tier": tier,
            "label": label or "",
            "created_at": time.time(),
        }).encode(),
    )
    return full_key


def validate_api_key(key: str) -> bool:
    """Return True if the key exists and is active."""
    if key == "grav_demoapikey":
        return True
    db = get_db()
    raw = db.get(f"apikey:{key}".encode())
    if not raw:
        return False
    try:
        return json.loads(raw.decode()).get("status") == "active"
    except Exception:
        return False


def get_key_tier(key: str) -> str:
    """Return the tier of an API key: 'default' or 'admin'."""
    if key == "grav_demoapikey":
        return "default"
    db = get_db()
    raw = db.get(f"apikey:{key}".encode())
    if not raw:
        return "default"
    try:
        return json.loads(raw.decode()).get("tier", "default")
    except Exception:
        return "default"


def revoke_api_key(key: str) -> bool:
    """Revoke an API key. Returns True if it existed."""
    db = get_db()
    db_key = f"apikey:{key}".encode()
    raw = db.get(db_key)
    if not raw:
        return False
    try:
        data = json.loads(raw.decode())
        data["status"] = "revoked"
        data["revoked_at"] = time.time()
        db.put(db_key, json.dumps(data).encode())
        return True
    except Exception:
        return False


def is_admin(key: str) -> bool:
    """True if key is the admin key or has admin tier."""
    if ADMIN_KEY and key == ADMIN_KEY:
        return True
    return get_key_tier(key) == "admin"
