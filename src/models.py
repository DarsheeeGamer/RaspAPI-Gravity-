"""Pydantic models for stored + wire data.

Pydantic v2 (Rust-backed `model_validate_json`/`model_dump_json`) replaces
hand-rolled `json.loads`/`dumps` of bare dicts: one schema, validated on read,
fast serialization, and clean field access instead of `.get("x", default)`
everywhere. Storage modules persist `model_dump_json().encode()` into RocksDB
and rehydrate via `Model.model_validate_json(raw)`.
"""

from __future__ import annotations

import time
from typing import Any, Optional

from pydantic import BaseModel, Field


def _now() -> float:
    return time.time()


# ── API keys ────────────────────────────────────────────────────────────────

class ApiKeyRecord(BaseModel):
    key: str
    status: str = "active"            # active | revoked
    tier: str = "default"             # default | admin
    label: str = ""
    created_at: float = Field(default_factory=_now)
    revoked_at: Optional[float] = None


# ── Accounts ──────────────────────────────────────────────────────────────────

class AccountRecord(BaseModel):
    """A provider account owned by an API key (config + opaque provider payload)."""
    account_id: str
    provider: str
    enabled: bool = True
    pool: str = "custom"
    priority: int = 100
    weight: int = 1
    # Full gravity ProviderAccount JSON (kept opaque; gravity validates it).
    gravity_account: dict[str, Any] = Field(default_factory=dict)
    created_at: float = Field(default_factory=_now)


# ── Balancer health ─────────────────────────────────────────────────────────

class HealthState(BaseModel):
    fails: int = 0
    cooldown_until: float = 0.0
    last_ok: float = 0.0
    last_fail: float = 0.0


# ── Abuse state ─────────────────────────────────────────────────────────────

class AbuseState(BaseModel):
    score: float = 0.0
    updated: float = Field(default_factory=_now)
    req_ts: list[float] = Field(default_factory=list)
    dup: dict[str, list[float]] = Field(default_factory=dict)  # hash -> [count, last_ts]
    err: list[float] = Field(default_factory=list)
    hard_ban: bool = False
    temp_ban_until: float = 0.0


# ── Cache pointer ─────────────────────────────────────────────────────────────

class CachePointer(BaseModel):
    rid: int
    expires_at: float


# ── Wire models (request/response) ─────────────────────────────────────────────

class ChatMessage(BaseModel):
    role: str
    content: Any = None
    name: Optional[str] = None
    tool_calls: Optional[list[dict[str, Any]]] = None
    tool_call_id: Optional[str] = None


class ChatRequest(BaseModel):
    """Stateless chat request — the client owns history (full `messages` each call)."""
    model: str
    messages: list[dict[str, Any]] = Field(default_factory=list)
    stream: bool = False
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None
    seed: Optional[int] = None
    system_prompt: Optional[str] = None
    tools: Optional[list[dict[str, Any]]] = None
    tool_choice: Optional[Any] = None
    frequency_penalty: Optional[float] = None
    presence_penalty: Optional[float] = None


class AddAccountRequest(BaseModel):
    account_id: str
    provider: str
    api_key: Optional[str] = None
    access_token: Optional[str] = None
    refresh_token: Optional[str] = None
    session_key: Optional[str] = None
    cookies: Optional[dict[str, str]] = None
    org_id: Optional[str] = None
    device_id: Optional[str] = None
    routing_hint: Optional[str] = None
    project: Optional[str] = None
    region: Optional[str] = None
    profile_arn: Optional[str] = None
    email: Optional[str] = None
    expires_at: Optional[int] = None
    chatgpt_account_id: Optional[str] = None
    user_id: Optional[str] = None
    at: Optional[str] = None
    bl: Optional[str] = None
    f_sid: Optional[str] = None
    base_url: Optional[str] = None
    pool: Optional[str] = "custom"
    priority: Optional[int] = 100
    weight: Optional[int] = 1
    timeout_ms: Optional[int] = 120000
    impersonate: Optional[str] = None
    proxy_url: Optional[str] = None
    extra_headers: Optional[dict[str, str]] = None
