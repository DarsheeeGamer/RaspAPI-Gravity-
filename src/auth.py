"""Authentication: users, passwords, JWT sessions, and request principals.

Two credential types are accepted on ``Authorization: Bearer <token>``:
  - **API key** (``grav_…``)  — long-lived, owned by a user (or the demo key).
  - **JWT**     (``eyJ…``)    — short-lived session token from /auth/login.

Anything else (or no header) resolves to an **anonymous** principal keyed by IP,
which the gate allows only on free providers at a strict quota.

No third-party crypto deps: PBKDF2-HMAC-SHA256 for passwords, a stdlib HS256
JWT. The signing secret comes from ``GRAVITY_JWT_SECRET`` (persisted/generated
in RocksDB if unset).
"""

import os
import json
import time
import hmac
import base64
import secrets
import hashlib
from dataclasses import dataclass
from typing import Optional

import store
import memo
from account import get_db
from genapi import generate_api_key, get_key_tier

# ── Tiers ─────────────────────────────────────────────────────────────────────
# Per-provider hourly quota by principal tier. 0 = unlimited.

TIER_QUOTAS = {
    "anonymous": 5,       # no key — free providers only, tiny budget
    "default":   20,      # registered key, shared pool
    "custom":    2000,    # has own provider account
    "pro":       10000,
    "admin":     0,       # unlimited
}

_PW_ITER = 200_000
_JWT_TTL = 3600          # access token lifetime (s)
_REFRESH_TTL = 30 * 86400

_principal_cache = memo.TTLCache(maxsize=8192, ttl=20.0)


# ── JWT secret ────────────────────────────────────────────────────────────────

def _jwt_secret() -> bytes:
    env = os.environ.get("GRAVITY_JWT_SECRET")
    if env:
        return env.encode()
    db = get_db()
    raw = db.get(b"auth:jwt_secret")
    if raw:
        return raw
    sec = secrets.token_bytes(32)
    db.put(b"auth:jwt_secret", sec)
    return sec


# ── Passwords (PBKDF2) ─────────────────────────────────────────────────────────

def hash_password(password: str) -> str:
    salt = secrets.token_bytes(16)
    dk = hashlib.pbkdf2_hmac("sha256", password.encode(), salt, _PW_ITER)
    return f"pbkdf2_sha256${_PW_ITER}${salt.hex()}${dk.hex()}"


def verify_password(password: str, encoded: str) -> bool:
    try:
        algo, iters, salt_hex, dk_hex = encoded.split("$")
        if algo != "pbkdf2_sha256":
            return False
        dk = hashlib.pbkdf2_hmac("sha256", password.encode(),
                                 bytes.fromhex(salt_hex), int(iters))
        return hmac.compare_digest(dk.hex(), dk_hex)
    except Exception:
        return False


# ── JWT (HS256) ───────────────────────────────────────────────────────────────

def _b64(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode()


def _b64d(s: str) -> bytes:
    return base64.urlsafe_b64decode(s + "=" * (-len(s) % 4))


def issue_jwt(user_id: str, tier: str, ttl: int = _JWT_TTL, kind: str = "access") -> str:
    header = {"alg": "HS256", "typ": "JWT"}
    now = int(time.time())
    payload = {"sub": user_id, "tier": tier, "kind": kind, "iat": now, "exp": now + ttl}
    h = _b64(json.dumps(header, separators=(",", ":")).encode())
    p = _b64(json.dumps(payload, separators=(",", ":")).encode())
    sig = hmac.new(_jwt_secret(), f"{h}.{p}".encode(), hashlib.sha256).digest()
    return f"{h}.{p}.{_b64(sig)}"


def verify_jwt(token: str) -> Optional[dict]:
    try:
        h, p, s = token.split(".")
        expected = hmac.new(_jwt_secret(), f"{h}.{p}".encode(), hashlib.sha256).digest()
        if not hmac.compare_digest(_b64(expected), s):
            return None
        payload = json.loads(_b64d(p))
        if payload.get("exp", 0) < time.time():
            return None
        return payload
    except Exception:
        return None


# ── Users ─────────────────────────────────────────────────────────────────────

class AuthError(Exception):
    pass


def signup(email: str, password: str) -> dict:
    email = email.strip().lower()
    if not email or "@" not in email:
        raise AuthError("invalid email")
    if len(password) < 8:
        raise AuthError("password must be at least 8 characters")
    if store.get_user_by_email(email):
        raise AuthError("email already registered")
    user_id = f"usr_{secrets.token_urlsafe(10)}"
    store.create_user(user_id, email, hash_password(password), tier="user", status="active")
    # Issue a default API key bound to the user.
    api_key = generate_api_key(label=f"default key for {email}")
    get_db().put(f"apikey:{api_key}:user".encode(), user_id.encode())
    _link_key_to_user(user_id, api_key)
    return {"user_id": user_id, "email": email, "api_key": api_key,
            "access_token": issue_jwt(user_id, "default"),
            "refresh_token": issue_jwt(user_id, "default", _REFRESH_TTL, "refresh")}


def login(email: str, password: str) -> dict:
    email = email.strip().lower()
    user = store.get_user_by_email(email)
    if not user or not verify_password(password, user["password_hash"]):
        raise AuthError("invalid credentials")
    if user.get("status") != "active":
        raise AuthError("account disabled")
    tier = user.get("tier", "default")
    return {"user_id": user["user_id"], "email": email,
            "access_token": issue_jwt(user["user_id"], tier),
            "refresh_token": issue_jwt(user["user_id"], tier, _REFRESH_TTL, "refresh")}


def refresh(refresh_token: str) -> dict:
    payload = verify_jwt(refresh_token)
    if not payload or payload.get("kind") != "refresh":
        raise AuthError("invalid refresh token")
    return {"access_token": issue_jwt(payload["sub"], payload.get("tier", "default"))}


def _link_key_to_user(user_id: str, api_key: str) -> None:
    db = get_db()
    raw = db.get(f"user:{user_id}:keys".encode())
    keys = json.loads(raw.decode()) if raw else []
    if api_key not in keys:
        keys.append(api_key)
        db.put(f"user:{user_id}:keys".encode(), json.dumps(keys).encode())


def user_keys(user_id: str) -> list[str]:
    raw = get_db().get(f"user:{user_id}:keys".encode())
    return json.loads(raw.decode()) if raw else []


# ── Principal resolution ───────────────────────────────────────────────────────

@dataclass
class Principal:
    kind: str               # "apikey" | "user" | "anonymous"
    id: str                 # api key, user_id, or ip-derived id
    tier: str               # anonymous | default | custom | pro | admin
    api_key: Optional[str]  # the effective key for account/quota scoping
    user_id: Optional[str] = None

    @property
    def is_authenticated(self) -> bool:
        return self.kind != "anonymous"


def resolve_principal(authorization: Optional[str], ip: str) -> Principal:
    """Map an Authorization header (or its absence) to a Principal."""
    if not authorization or not authorization.lower().startswith("bearer "):
        return Principal("anonymous", f"ip:{ip}", "anonymous", api_key=None)
    token = authorization.split(" ", 1)[1].strip()

    cached = _principal_cache.get(token)
    if cached is not None:
        return cached

    principal: Principal
    # JWT?
    if token.count(".") == 2 and not token.startswith("grav_"):
        payload = verify_jwt(token)
        if payload:
            uid = payload["sub"]
            keys = user_keys(uid)
            principal = Principal("user", uid, payload.get("tier", "default"),
                                  api_key=keys[0] if keys else None, user_id=uid)
            _principal_cache.put(token, principal)
            return principal

    # API key?
    from genapi import validate_api_key
    if validate_api_key(token):
        tier = get_key_tier(token)
        uid_raw = get_db().get(f"apikey:{token}:user".encode())
        principal = Principal("apikey", token, tier, api_key=token,
                              user_id=uid_raw.decode() if uid_raw else None)
        _principal_cache.put(token, principal)
        return principal

    # Unknown token → treat as anonymous (gate decides).
    return Principal("anonymous", f"ip:{ip}", "anonymous", api_key=None)


def quota_limit_for(tier: str) -> int:
    return TIER_QUOTAS.get(tier, TIER_QUOTAS["default"])
