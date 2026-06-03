"""Response + discovery cache.

Keying and eligibility logic live here; the actual cached bodies live in DuckDB
(bulk) addressed by a RocksDB pointer (see ``store.cache_put``/``cache_get``) so
a hit is a RocksDB TTL check + one DuckDB point read. Hit/miss counters are kept
in RocksDB.

Only deterministic, single-shot, tool-free requests are cached, to preserve
correctness.
"""

import json
import time
import hashlib
from typing import Any, Optional

import store
from account import get_db

RESPONSE_TTL_S = 600
DISCOVERY_TTL_S = 1800
_STATS_KEY = b"cache:stats"


def _hash(payload: dict) -> str:
    blob = json.dumps(payload, sort_keys=True, ensure_ascii=True, separators=(",", ":"))
    return hashlib.sha256(blob.encode()).hexdigest()


def response_hash(provider: str, model: str, messages: list, system_prompt,
                  tools, sampling: dict) -> str:
    return _hash({"p": provider, "m": model, "msgs": messages,
                  "sys": system_prompt, "tools": tools, "s": sampling})


def is_cacheable(stream: bool, tools, temperature, seed) -> bool:
    if stream or tools:
        return False
    if temperature is not None and temperature > 0 and seed is None:
        return False
    return True


def get_response(h: str) -> Optional[Any]:
    val = store.cache_get(h)
    _bump("hit" if val is not None else "miss")
    return val


def put_response(h: str, body: Any, ttl_s: int = RESPONSE_TTL_S) -> None:
    store.cache_put(h, body, time.time() + ttl_s)


def get_discovery(provider: str, secret: str) -> Optional[Any]:
    h = f"disc:{provider}:{fingerprint(secret)}"
    return store.cache_get(h)


def put_discovery(provider: str, secret: str, models: Any, ttl_s: int = DISCOVERY_TTL_S) -> None:
    h = f"disc:{provider}:{fingerprint(secret)}"
    store.cache_put(h, models, time.time() + ttl_s)


def fingerprint(secret: str) -> str:
    if not secret:
        return "anon"
    return hashlib.sha256(secret.encode()).hexdigest()[:12]


# ── Stats (RocksDB) ─────────────────────────────────────────────────────────

def _bump(kind: str) -> None:
    db = get_db()
    raw = db.get(_STATS_KEY)
    stats = json.loads(raw.decode()) if raw else {"hit": 0, "miss": 0}
    stats[kind] = stats.get(kind, 0) + 1
    db.put(_STATS_KEY, json.dumps(stats).encode())


def stats() -> dict:
    raw = get_db().get(_STATS_KEY)
    s = json.loads(raw.decode()) if raw else {"hit": 0, "miss": 0}
    total = s.get("hit", 0) + s.get("miss", 0)
    s["hit_rate"] = round(s.get("hit", 0) / total, 4) if total else 0.0
    return s
