"""Canonical RocksDB key scheme — the hashmap-style address space.

RocksDB holds *ordered, key-based* data: API keys, rate-limit counters, quota
usage, and **cell pointers** into DuckDB. Keys are structured paths so a lookup
is a single point read (hashmap-style) and related data is prefix-scannable.

Imagine DuckDB as one big table addressed by an integer ``rid`` (a "cell"). When
bulk data — e.g. the Claude-web account for a key — lands in DuckDB cell 844,
RocksDB stores:

    <apikey>:cell:claude:account:<account_id>   ->  844          (pointer to the cell)
    <apikey>:cell:claude:quota:currentusage     ->  37           (hot counter, in-place)
    <apikey>:cell:claude:quota:limit            ->  2000
    <apikey>:cell:claude:quota:reset_at         ->  1733280000

So hot counters live *in* RocksDB (ordered, cheap RMW) while bulk rows live in
DuckDB, reached in O(1) via the cell pointer. Every module builds keys here so
the scheme stays consistent and greppable.
"""

# ── Top-level records ──────────────────────────────────────────────────────────

def apikey(key: str) -> bytes:
    return f"apikey:{key}".encode()


def apikey_user(key: str) -> bytes:
    """key → owning user_id."""
    return f"apikey:{key}:user".encode()


# ── Cells: pointers from a key's provider namespace into DuckDB rows ───────────

def cell_account(key: str, provider: str, account_id: str) -> bytes:
    """<key>:cell:<provider>:account:<account_id>  → DuckDB rid."""
    return f"{key}:cell:{provider}:account:{account_id}".encode()


def cell_account_prefix(key: str, provider: str = "") -> bytes:
    """Prefix to scan a key's account cells (optionally one provider)."""
    if provider:
        return f"{key}:cell:{provider}:account:".encode()
    return f"{key}:cell:".encode()


def cell_conversation(key: str, conv_id: str) -> bytes:
    """<key>:cell:conv:<conv_id>  → DuckDB transcript address (conv_id)."""
    return f"{key}:cell:conv:{conv_id}".encode()


# ── Quota (hot counters living in-place in RocksDB) ────────────────────────────

def quota_usage(key: str, provider: str) -> bytes:
    """<key>:cell:<provider>:quota:currentusage  → int (in-place counter)."""
    return f"{key}:cell:{provider}:quota:currentusage".encode()


def quota_limit(key: str, provider: str) -> bytes:
    return f"{key}:cell:{provider}:quota:limit".encode()


def quota_reset_at(key: str, provider: str) -> bytes:
    return f"{key}:cell:{provider}:quota:reset_at".encode()


def quota_prefix(key: str) -> bytes:
    return f"{key}:cell:".encode()


# ── Rate limiting (ordered, per identity + window) ─────────────────────────────

def ratelimit(key_or_ip: str, window: int) -> bytes:
    """<id>:rl:<window>  → int requests in this window bucket."""
    return f"{key_or_ip}:rl:{window}".encode()


# ── Abuse / balancer health (already namespaced; kept here for one source) ─────

def abuse_state(identity: str) -> bytes:
    return f"abuse:state:{identity}".encode()


def balancer_health(lb_key: str) -> bytes:
    return f"lb:health:{lb_key}".encode()
