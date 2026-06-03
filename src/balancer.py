"""Account load balancer.

Picks the best provider account for each request across the user's custom
accounts and the shared default pool, with health tracking, weighted
least-connections selection, exponential-backoff cooldown on failure, and
ordered failover candidates.

Health state is persisted in LevelDB (survives restarts); in-flight counts are
per-process (reset on restart). Selection order:

  1. drop accounts whose cooldown hasn't expired
  2. sort by (in_flight / weight) ascending, then by priority, then fewest fails
  3. return the ordered list so the caller can fail over down it
"""

import json
import time
import threading
from dataclasses import dataclass
from typing import List, Optional

from gravity.accounts import (
    AccountsFile, EmailBundle, ProviderAccount,
    AuthConfig, TransportConfig, ProviderDefaults,
)
from gravity.capabilities import (
    HistoryMode, ToolSupportMode, StructuredOutputMode, RemoteSessionMode,
)
from gravity.io import load_accounts

from account import load_accounts_for_key, get_providers_for_key
from providers import FREE_PROVIDERS

# ── Tunables ──────────────────────────────────────────────────────────────────

_BASE_COOLDOWN_S = 15          # first failure cooldown
_MAX_COOLDOWN_S = 900          # cap (15 min)
_FAIL_RESET_AFTER_S = 600      # forget failures after 10 min healthy

# In-flight request counts, keyed "api_key|provider|account_id". Per-process.
_inflight: dict[str, int] = {}
_inflight_lock = threading.Lock()


@dataclass
class Candidate:
    """One selectable account plus the bookkeeping key for health reporting."""
    account: ProviderAccount
    key: str            # health/inflight key
    pool: str           # "custom" | "default"


# ── Health store (LevelDB) ──────────────────────────────────────────────────

def _health_key(lb_key: str) -> bytes:
    return f"lb:health:{lb_key}".encode()


def _get_health(db, lb_key: str) -> dict:
    raw = db.get(_health_key(lb_key))
    if not raw:
        return {"fails": 0, "cooldown_until": 0.0, "last_ok": 0.0, "last_fail": 0.0}
    try:
        return json.loads(raw.decode())
    except Exception:
        return {"fails": 0, "cooldown_until": 0.0, "last_ok": 0.0, "last_fail": 0.0}


def _put_health(db, lb_key: str, h: dict) -> None:
    db.put(_health_key(lb_key), json.dumps(h).encode())


def _is_available(h: dict, now: float) -> bool:
    return h.get("cooldown_until", 0.0) <= now


# ── In-flight tracking ─────────────────────────────────────────────────────────

def _inc_inflight(key: str) -> None:
    with _inflight_lock:
        _inflight[key] = _inflight.get(key, 0) + 1


def _dec_inflight(key: str) -> None:
    with _inflight_lock:
        _inflight[key] = max(0, _inflight.get(key, 0) - 1)


def _get_inflight(key: str) -> int:
    with _inflight_lock:
        return _inflight.get(key, 0)


# ── Candidate assembly ─────────────────────────────────────────────────────────

def _accounts_for(api_key: str, provider: str) -> tuple[list[ProviderAccount], str]:
    """Return (accounts, pool) — custom accounts if the key has any for this
    provider, else the shared default pool."""
    if provider in get_providers_for_key(api_key):
        user = load_accounts_for_key(api_key)
        accts = [
            a for bundle in user.emails for a in bundle.providers
            if a.provider.value == provider and a.enabled
        ]
        if accts:
            return accts, "custom"

    try:
        defaults = load_accounts()
        accts = [
            a for bundle in defaults.emails for a in bundle.providers
            if a.provider.value == provider and a.enabled
        ]
        if accts:
            return accts, "default"
    except Exception:
        pass

    # Free providers with no configured account → synthesize an anonymous one
    # so anonymous/keyless callers can still reach them.
    if provider in FREE_PROVIDERS:
        anon = _anon_account(provider)
        if anon is not None:
            return [anon], "anonymous"
    return [], "default"


def _anon_account(provider: str) -> Optional[ProviderAccount]:
    from gravity.capabilities import ProviderName
    pe = next((p for p in ProviderName if p.value == provider), None)
    if pe is None:
        return None
    try:
        return ProviderAccount(
            account_id=f"anon-{provider}", provider=pe, enabled=True,
            pool="anonymous", priority=-1, weight=1,
            auth=AuthConfig(kind="anonymous", secret_ref="", extra={}),
            transport=TransportConfig(base_url="", timeout_ms=60000,
                                      impersonate="chrome136", verify_ssl=True,
                                      extra_headers={}),
            defaults=ProviderDefaults(
                tool_support=ToolSupportMode.JSON_EMULATED,
                structured_output=StructuredOutputMode.JSON_INSTRUCTION,
                remote_session_mode=RemoteSessionMode.NONE,
                supports_system_prompt=True,
                default_history_mode=HistoryMode.STATELESS),
            metadata={"source": "anonymous"},
        )
    except Exception:
        return None


def select_candidates(db, api_key: str, provider: str) -> List[Candidate]:
    """Return health-ordered failover candidates for (api_key, provider).

    Best account first. Accounts in cooldown are pushed to the end (still
    returned as last resort rather than dropped, so a fully-cooled pool can
    still attempt the least-bad option).
    """
    accts, pool = _accounts_for(api_key, provider)
    if not accts:
        return []

    now = time.time()
    scored = []
    for a in accts:
        lb_key = f"{api_key}|{provider}|{a.account_id}"
        h = _get_health(db, lb_key)
        # Forget stale failures after a healthy quiet period.
        if h.get("fails", 0) and now - h.get("last_fail", 0.0) > _FAIL_RESET_AFTER_S:
            h = {"fails": 0, "cooldown_until": 0.0, "last_ok": h.get("last_ok", 0.0), "last_fail": 0.0}
            _put_health(db, lb_key, h)

        available = _is_available(h, now)
        weight = max(1, getattr(a, "weight", 1) or 1)
        inflight = _get_inflight(lb_key)
        # Lower load_score = preferred. Cooled-down accounts get a huge penalty.
        load_score = inflight / weight
        penalty = 0 if available else 1_000_000
        priority = getattr(a, "priority", 100) or 100
        scored.append((
            penalty + load_score,        # primary: availability + load
            -priority,                   # secondary: higher priority first
            h.get("fails", 0),           # tertiary: fewer past failures
            Candidate(account=a, key=lb_key, pool=pool),
        ))

    scored.sort(key=lambda t: (t[0], t[1], t[2]))
    return [t[3] for t in scored]


# ── Result reporting ───────────────────────────────────────────────────────────

def mark_success(db, key: str) -> None:
    h = _get_health(db, key)
    h["fails"] = 0
    h["cooldown_until"] = 0.0
    h["last_ok"] = time.time()
    _put_health(db, key, h)


def mark_failure(db, key: str) -> None:
    h = _get_health(db, key)
    fails = int(h.get("fails", 0)) + 1
    cooldown = min(_BASE_COOLDOWN_S * (2 ** (fails - 1)), _MAX_COOLDOWN_S)
    now = time.time()
    h.update({"fails": fails, "cooldown_until": now + cooldown, "last_fail": now})
    _put_health(db, key, h)


class Lease:
    """Context manager that tracks an in-flight request and reports health."""

    def __init__(self, db, candidate: Candidate):
        self.db = db
        self.candidate = candidate
        self._ok = False

    def __enter__(self) -> "Lease":
        _inc_inflight(self.candidate.key)
        return self

    def success(self) -> None:
        self._ok = True

    def __exit__(self, exc_type, exc, tb) -> bool:
        _dec_inflight(self.candidate.key)
        if self._ok and exc_type is None:
            mark_success(self.db, self.candidate.key)
        else:
            mark_failure(self.db, self.candidate.key)
        return False  # never suppress


def single_account_file(candidate: Candidate) -> AccountsFile:
    """Wrap one account in an AccountsFile so Gravity uses exactly it."""
    return AccountsFile(
        schema_version=4,
        emails=[EmailBundle(
            email="lb@gravity.local",
            enabled=True,
            display_name=f"LB:{candidate.pool}",
            providers=[candidate.account],
        )],
    )


def health_snapshot(db, api_key: str, provider: Optional[str] = None) -> list[dict]:
    """Read current health for all of a key's accounts (for /api/balancer)."""
    out = []
    prefix = f"lb:health:{api_key}|".encode()
    now = time.time()
    for k, v in db.iterator(prefix=prefix):
        try:
            _, _, rest = k.decode().partition("lb:health:")
            _api, prov, acct = rest.split("|", 2)
            if provider and prov != provider:
                continue
            h = json.loads(v.decode())
            out.append({
                "provider": prov,
                "account_id": acct,
                "fails": h.get("fails", 0),
                "available": _is_available(h, now),
                "cooldown_remaining_s": max(0, round(h.get("cooldown_until", 0) - now, 1)),
                "in_flight": _get_inflight(f"{api_key}|{prov}|{acct}"),
            })
        except Exception:
            continue
    return out
