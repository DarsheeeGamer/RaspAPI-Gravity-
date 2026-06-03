"""Per-(api_key, provider) quota — hot counters living in RocksDB cells.

Counters are addressed by the structured cell paths in ``keyspace`` and held
*in-place* in RocksDB (ordered, cheap read-modify-write) rather than DuckDB, so
the per-request hot path never touches the columnar store. Each quota is a
rolling window: usage resets when ``reset_at`` passes.

    <key>:cell:<provider>:quota:currentusage   counter
    <key>:cell:<provider>:quota:limit          ceiling (0 = unlimited)
    <key>:cell:<provider>:quota:reset_at        window end (epoch s)
"""

import time
from typing import Optional

import keyspace
from account import get_db

WINDOW_S = 3600  # rolling hourly window


def _get_int(db, k: bytes, default: int = 0) -> int:
    raw = db.get(k)
    if not raw:
        return default
    try:
        return int(raw.decode())
    except Exception:
        return default


def _roll(db, key: str, provider: str, now: float) -> tuple[int, int, float]:
    """Return (usage, limit, reset_at), rolling the window if it expired."""
    reset_at = float(_get_int(db, keyspace.quota_reset_at(key, provider), 0))
    limit = _get_int(db, keyspace.quota_limit(key, provider), 0)
    if reset_at <= now:
        # New window: zero usage, set next reset.
        reset_at = now + WINDOW_S
        db.put(keyspace.quota_usage(key, provider), b"0")
        db.put(keyspace.quota_reset_at(key, provider), str(int(reset_at)).encode())
        usage = 0
    else:
        usage = _get_int(db, keyspace.quota_usage(key, provider), 0)
    return usage, limit, reset_at


def set_limit(key: str, provider: str, limit: int) -> None:
    get_db().put(keyspace.quota_limit(key, provider), str(int(limit)).encode())


def status(key: str, provider: str) -> dict:
    db = get_db()
    now = time.time()
    usage, limit, reset_at = _roll(db, key, provider, now)
    return {
        "provider": provider, "usage": usage, "limit": limit,
        "remaining": (limit - usage) if limit else None,
        "reset_in_s": max(0, int(reset_at - now)),
        "unlimited": limit == 0,
    }


def try_consume(key: str, provider: str, default_limit: int, cost: int = 1) -> tuple[bool, dict]:
    """Atomically (under the GIL) consume ``cost`` from the quota.

    Returns (allowed, status). If no explicit limit is set, ``default_limit``
    (from the caller's tier) is applied. ``default_limit == 0`` ⇒ unlimited.
    """
    db = get_db()
    now = time.time()
    usage, limit, reset_at = _roll(db, key, provider, now)
    effective = limit if limit else default_limit
    if effective and usage + cost > effective:
        return False, {
            "provider": provider, "usage": usage, "limit": effective,
            "remaining": max(0, effective - usage),
            "reset_in_s": max(0, int(reset_at - now)),
        }
    new_usage = usage + cost
    db.put(keyspace.quota_usage(key, provider), str(new_usage).encode())
    return True, {
        "provider": provider, "usage": new_usage, "limit": effective or 0,
        "remaining": (effective - new_usage) if effective else None,
        "reset_in_s": max(0, int(reset_at - now)),
    }


def snapshot(key: str) -> list[dict]:
    """All provider quotas for a key (prefix scan of its cell namespace)."""
    db = get_db()
    out = {}
    now = time.time()
    prefix = keyspace.quota_prefix(key)
    for k, _v in db.iterator(prefix=prefix):
        ks = k.decode()
        # <key>:cell:<provider>:quota:<field>
        parts = ks.split(":cell:", 1)
        if len(parts) != 2:
            continue
        tail = parts[1]
        if ":quota:" not in tail:
            continue
        provider = tail.split(":quota:", 1)[0]
        if provider not in out:
            out[provider] = status(key, provider)
    return list(out.values())
