"""Unified storage layer: LevelDB index + DuckDB bulk store.

Design
------
**LevelDB** (plyvel) is the hot key-value index — tiny, microsecond reads. It
holds API keys, account configs, balancer health, abuse state, cache TTLs, and
— crucially — **pointers**: a logical key maps to an integer ``rid`` that
addresses an exact row in DuckDB.

**DuckDB** is the bulk + analytical store — conversation transcripts, the
request log (for metrics/aggregation), and cached response bodies. Every bulk
row carries a ``rid BIGINT`` with an index, so a lookup is:

    rid = leveldb.get(logical_key)         # ~µs KV hit
    row = duckdb "WHERE rid = ?"           # point lookup on indexed column

i.e. LevelDB resolves *where* the data is and DuckDB fetches it directly — no
table scan. Aggregations (metrics, top models, usage) run natively in DuckDB.

Concurrency: one process-wide DuckDB connection guarded by a lock (DuckDB
write-serializes anyway); call the blocking helpers via ``asyncio.to_thread``
from async paths.
"""

import os
import json
import threading
from typing import Any, Optional

import duckdb

from account import get_db  # LevelDB handle (shared)

_DATA_DIR = os.path.expanduser("~/nodataishere")
os.makedirs(_DATA_DIR, exist_ok=True)
_DUCK_PATH = os.path.join(_DATA_DIR, "bulk.duckdb")

_conn: Optional[duckdb.DuckDBPyConnection] = None
_lock = threading.RLock()


def duck() -> duckdb.DuckDBPyConnection:
    global _conn
    if _conn is None:
        with _lock:
            if _conn is None:
                _conn = duckdb.connect(_DUCK_PATH)
                _init_schema(_conn)
    return _conn


def _init_schema(c: duckdb.DuckDBPyConnection) -> None:
    # Monotonic rid generator shared by all bulk tables.
    c.execute("CREATE SEQUENCE IF NOT EXISTS rid_seq START 1;")

    c.execute("""
        CREATE TABLE IF NOT EXISTS request_log (
            rid        BIGINT,
            ts         DOUBLE,
            api_key    VARCHAR,
            ip         VARCHAR,
            provider   VARCHAR,
            model      VARCHAR,
            status     VARCHAR,
            cached     BOOLEAN,
            latency_ms DOUBLE,
            in_tokens  BIGINT,
            out_tokens BIGINT
        );
    """)
    c.execute("CREATE INDEX IF NOT EXISTS idx_req_ts ON request_log(ts);")
    c.execute("CREATE INDEX IF NOT EXISTS idx_req_model ON request_log(model);")

    c.execute("""
        CREATE TABLE IF NOT EXISTS response_cache (
            rid        BIGINT,
            hash       VARCHAR,
            body       VARCHAR,
            stored_at  DOUBLE,
            expires_at DOUBLE
        );
    """)
    c.execute("CREATE INDEX IF NOT EXISTS idx_cache_rid ON response_cache(rid);")

    # Account data — the "major data" — lives in DuckDB cells; RocksDB holds the
    # api_key→rid pointers so account fetches are point lookups, never scans.
    c.execute("""
        CREATE TABLE IF NOT EXISTS accounts (
            rid        BIGINT,
            api_key    VARCHAR,
            account_id VARCHAR,
            provider   VARCHAR,
            enabled    BOOLEAN,
            pool       VARCHAR,
            priority   INTEGER,
            weight     INTEGER,
            gravity_account VARCHAR,
            created_at DOUBLE
        );
    """)
    c.execute("CREATE INDEX IF NOT EXISTS idx_acct_rid ON accounts(rid);")
    c.execute("CREATE INDEX IF NOT EXISTS idx_acct_key ON accounts(api_key);")

    # Users (auth) — bulk in DuckDB, email→user_id + user_id→rid pointers in RocksDB.
    c.execute("""
        CREATE TABLE IF NOT EXISTS users (
            rid           BIGINT,
            user_id       VARCHAR,
            email         VARCHAR,
            password_hash VARCHAR,
            tier          VARCHAR,
            status        VARCHAR,
            created_at    DOUBLE
        );
    """)
    c.execute("CREATE INDEX IF NOT EXISTS idx_user_rid ON users(rid);")
    c.execute("CREATE INDEX IF NOT EXISTS idx_user_email ON users(email);")


def _next_rid() -> int:
    with _lock:
        return duck().execute("SELECT nextval('rid_seq')").fetchone()[0]


# ── LevelDB pointer helpers ────────────────────────────────────────────────────

def set_pointer(logical_key: str, rid: int) -> None:
    get_db().put(f"ptr:{logical_key}".encode(), str(rid).encode())


def get_pointer(logical_key: str) -> Optional[int]:
    raw = get_db().get(f"ptr:{logical_key}".encode())
    return int(raw.decode()) if raw else None


def del_pointer(logical_key: str) -> None:
    try:
        get_db().delete(f"ptr:{logical_key}".encode())
    except Exception:
        pass


# ── Response cache bodies (bulk in DuckDB, TTL pointer in LevelDB) ─────────────

def cache_put(hash_: str, body: dict, expires_at: float) -> None:
    import time
    with _lock:
        rid = _next_rid()
        duck().execute("DELETE FROM response_cache WHERE hash = ?", [hash_])
        duck().execute("INSERT INTO response_cache VALUES (?,?,?,?,?)",
                       [rid, hash_, json.dumps(body), time.time(), expires_at])
    # LevelDB holds the pointer + expiry so we can check/evict without touching DuckDB.
    get_db().put(f"cacheptr:{hash_}".encode(),
                 json.dumps({"rid": rid, "expires_at": expires_at}).encode())


def cache_get(hash_: str) -> Optional[dict]:
    import time
    raw = get_db().get(f"cacheptr:{hash_}".encode())
    if not raw:
        return None
    try:
        ptr = json.loads(raw.decode())
    except Exception:
        return None
    if ptr.get("expires_at", 0) < time.time():
        get_db().delete(f"cacheptr:{hash_}".encode())
        return None
    rid = ptr["rid"]
    with _lock:
        row = duck().execute("SELECT body FROM response_cache WHERE rid = ?", [rid]).fetchone()
    if not row:
        return None
    try:
        return json.loads(row[0])
    except Exception:
        return None


# ── Request log + metrics (analytical, native DuckDB aggregation) ──────────────

def log_request(api_key: str, ip: str, provider: str, model: str, status: str,
                cached: bool, latency_ms: float, in_tokens: int, out_tokens: int) -> None:
    import time
    with _lock:
        duck().execute(
            "INSERT INTO request_log VALUES (?,?,?,?,?,?,?,?,?,?,?)",
            [_next_rid(), time.time(), api_key, ip, provider, model, status,
             cached, latency_ms, in_tokens, out_tokens],
        )


# ── Accounts (bulk in DuckDB cells; RocksDB cell-path pointers → rid) ──────────
#
# Pointer scheme (keyspace.cell_account):
#   <api_key>:cell:<provider>:account:<account_id>  ->  DuckDB rid
# A key's accounts are enumerated by prefix-scanning <api_key>:cell:.

import keyspace as _ks


def _ptr_put(rocks_key: bytes, rid: int) -> None:
    get_db().put(rocks_key, str(rid).encode())


def _ptr_get(rocks_key: bytes) -> Optional[int]:
    raw = get_db().get(rocks_key)
    return int(raw.decode()) if raw else None


def upsert_account(api_key: str, account_id: str, provider: str, enabled: bool,
                   pool: str, priority: int, weight: int, gravity_account: dict) -> None:
    import time
    with _lock:
        c = duck()
        c.execute("DELETE FROM accounts WHERE api_key = ? AND account_id = ? AND provider = ?",
                  [api_key, account_id, provider])
        rid = _next_rid()
        c.execute("INSERT INTO accounts VALUES (?,?,?,?,?,?,?,?,?,?)",
                  [rid, api_key, account_id, provider, enabled, pool, priority, weight,
                   json.dumps(gravity_account), time.time()])
    # Hashmap-style cell pointer in RocksDB.
    _ptr_put(_ks.cell_account(api_key, provider, account_id), rid)


def get_account(api_key: str, provider: str, account_id: str) -> Optional[dict]:
    rid = _ptr_get(_ks.cell_account(api_key, provider, account_id))
    if rid is None:
        return None
    with _lock:
        row = duck().execute(
            "SELECT gravity_account FROM accounts WHERE rid = ?", [rid]).fetchone()
    if not row:
        return None
    try:
        return json.loads(row[0])
    except Exception:
        return None


def get_accounts(api_key: str, provider: Optional[str] = None) -> list[dict]:
    """All gravity_account dicts for a key (point lookup by indexed api_key)."""
    with _lock:
        if provider:
            rows = duck().execute(
                "SELECT gravity_account FROM accounts WHERE api_key = ? AND provider = ? AND enabled",
                [api_key, provider]).fetchall()
        else:
            rows = duck().execute(
                "SELECT gravity_account FROM accounts WHERE api_key = ? AND enabled",
                [api_key]).fetchall()
    out = []
    for (g,) in rows:
        try:
            out.append(json.loads(g))
        except Exception:
            continue
    return out


def list_account_summaries(api_key: str) -> list[dict]:
    with _lock:
        rows = duck().execute(
            "SELECT account_id, provider, pool, enabled FROM accounts WHERE api_key = ?",
            [api_key]).fetchall()
    return [{"account_id": a, "provider": p, "pool": pl, "enabled": e}
            for a, p, pl, e in rows]


def providers_for_key(api_key: str) -> set[str]:
    with _lock:
        rows = duck().execute(
            "SELECT DISTINCT provider FROM accounts WHERE api_key = ? AND enabled",
            [api_key]).fetchall()
    return {r[0] for r in rows}


def delete_account(api_key: str, account_id: str, provider: Optional[str] = None) -> int:
    with _lock:
        if provider:
            duck().execute(
                "DELETE FROM accounts WHERE api_key = ? AND account_id = ? AND provider = ?",
                [api_key, account_id, provider])
            get_db().delete(_ks.cell_account(api_key, provider, account_id))
        else:
            # Drop the account across any provider; clear all matching cell pointers.
            rows = duck().execute(
                "SELECT provider FROM accounts WHERE api_key = ? AND account_id = ?",
                [api_key, account_id]).fetchall()
            duck().execute(
                "DELETE FROM accounts WHERE api_key = ? AND account_id = ?",
                [api_key, account_id])
            for (prov,) in rows:
                get_db().delete(_ks.cell_account(api_key, prov, account_id))
    return 1


# ── Users (bulk in DuckDB, email→user_id + user_id→rid pointers in RocksDB) ────

def _user_email_ptr(email: str) -> str:
    return f"useremail:{email.lower()}"


def _user_rid_ptr(user_id: str) -> str:
    return f"userrid:{user_id}"


def create_user(user_id: str, email: str, password_hash: str,
                tier: str = "user", status: str = "active") -> None:
    import time
    with _lock:
        rid = _next_rid()
        duck().execute("INSERT INTO users VALUES (?,?,?,?,?,?,?)",
                       [rid, user_id, email.lower(), password_hash, tier, status, time.time()])
    set_pointer(_user_rid_ptr(user_id), rid)
    # email→user_id is a plain RocksDB mapping (not an rid).
    get_db().put(f"ptr:{_user_email_ptr(email)}".encode(), user_id.encode())


def user_id_by_email(email: str) -> Optional[str]:
    raw = get_db().get(f"ptr:{_user_email_ptr(email)}".encode())
    return raw.decode() if raw else None


def get_user(user_id: str) -> Optional[dict]:
    rid = get_pointer(_user_rid_ptr(user_id))
    if rid is None:
        return None
    with _lock:
        row = duck().execute(
            "SELECT user_id, email, password_hash, tier, status, created_at FROM users WHERE rid = ?",
            [rid]).fetchone()
    if not row:
        return None
    return {"user_id": row[0], "email": row[1], "password_hash": row[2],
            "tier": row[3], "status": row[4], "created_at": row[5]}


def get_user_by_email(email: str) -> Optional[dict]:
    uid = user_id_by_email(email)
    return get_user(uid) if uid else None


def update_user(user_id: str, **fields) -> None:
    cols = [k for k in ("password_hash", "tier", "status") if k in fields]
    if not cols:
        return
    rid = get_pointer(_user_rid_ptr(user_id))
    if rid is None:
        return
    sets = ", ".join(f"{c} = ?" for c in cols)
    with _lock:
        duck().execute(f"UPDATE users SET {sets} WHERE rid = ?",
                       [fields[c] for c in cols] + [rid])


def metrics_summary() -> dict:
    with _lock:
        c = duck()
        total = c.execute("SELECT COUNT(*) FROM request_log").fetchone()[0]
        if not total:
            return {"total_requests": 0, "most_used_model": "none", "status": "healthy"}
        top = c.execute(
            "SELECT model, COUNT(*) n FROM request_log GROUP BY model ORDER BY n DESC LIMIT 1"
        ).fetchone()
        cache_hits = c.execute("SELECT COUNT(*) FROM request_log WHERE cached").fetchone()[0]
        tokens = c.execute(
            "SELECT COALESCE(SUM(in_tokens),0), COALESCE(SUM(out_tokens),0) FROM request_log"
        ).fetchone()
        p50 = c.execute(
            "SELECT quantile_cont(latency_ms, 0.5) FROM request_log WHERE NOT cached"
        ).fetchone()[0]
        p95 = c.execute(
            "SELECT quantile_cont(latency_ms, 0.95) FROM request_log WHERE NOT cached"
        ).fetchone()[0]
        by_model = c.execute(
            "SELECT model, COUNT(*) n FROM request_log GROUP BY model ORDER BY n DESC LIMIT 20"
        ).fetchall()
    return {
        "total_requests": total,
        "most_used_model": top[0] if top else "none",
        "cache_hits": cache_hits,
        "cache_hit_rate": round(cache_hits / total, 4) if total else 0.0,
        "input_tokens": tokens[0],
        "output_tokens": tokens[1],
        "latency_ms_p50": round(p50, 1) if p50 else None,
        "latency_ms_p95": round(p95, 1) if p95 else None,
        "model_breakdown": {m: n for m, n in by_model},
        "status": "healthy",
    }
