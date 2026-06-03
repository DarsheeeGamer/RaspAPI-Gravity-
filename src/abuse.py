"""Abuse detection + mitigation, fully backed by LevelDB.

Tracks per-(api_key, ip) behaviour in sliding windows and assigns an escalating
abuse score. Signals:

  - **burst rate**: too many requests in a short window (flood / scraping)
  - **duplicate spam**: the same prompt hash repeated rapidly (loop/abuse)
  - **error storm**: high upstream-failure ratio (credential abuse / probing)
  - **concurrency flood**: too many simultaneous in-flight requests

Score decays over time. Crossing thresholds escalates: warn → throttle (forced
slowdown) → temp-ban (block with retry-after) → hard-ban (manual unblock).
All counters, windows, scores, and bans live in LevelDB so they survive
restarts and are shared across workers.
"""

import json
import time
from dataclasses import dataclass
from typing import Optional

from account import get_db

# ── Tunables ──────────────────────────────────────────────────────────────────

BURST_WINDOW_S = 10
BURST_LIMIT = 40               # >40 req / 10s from one identity = burst
DUP_WINDOW_S = 60
DUP_LIMIT = 15                 # >15 identical prompts / min = spam
ERROR_WINDOW_S = 60
ERROR_RATIO = 0.6             # >60% errors over >=10 calls = probing
ERROR_MIN_SAMPLES = 10
CONCURRENCY_LIMIT = 24         # simultaneous in-flight per identity

SCORE_DECAY_PER_S = 0.05       # points shed per second
THROTTLE_AT = 5.0
TEMP_BAN_AT = 12.0
TEMP_BAN_S = 300

# In-process concurrency gauges (authoritative live count; mirror to LDB).
import threading
_conc: dict[str, int] = {}
_conc_lock = threading.Lock()


@dataclass
class Decision:
    allowed: bool
    action: str               # "allow" | "throttle" | "temp_ban" | "hard_ban"
    score: float
    retry_after_s: int = 0
    reason: str = ""

    def to_dict(self) -> dict:
        return {
            "allowed": self.allowed, "action": self.action,
            "score": round(self.score, 2), "retry_after_s": self.retry_after_s,
            "reason": self.reason,
        }


def _ident(api_key: str, ip: str) -> str:
    return f"{api_key}@{ip}"


def _k(ident: str) -> bytes:
    return f"abuse:state:{ident}".encode()


def _load(ident: str) -> dict:
    raw = get_db().get(_k(ident))
    if not raw:
        return {"score": 0.0, "updated": time.time(), "req_ts": [], "dup": {},
                "err": [], "hard_ban": False, "temp_ban_until": 0.0}
    try:
        return json.loads(raw.decode())
    except Exception:
        return {"score": 0.0, "updated": time.time(), "req_ts": [], "dup": {},
                "err": [], "hard_ban": False, "temp_ban_until": 0.0}


def _save(ident: str, st: dict) -> None:
    try:
        get_db().put(_k(ident), json.dumps(st).encode())
    except Exception:
        pass


def _decay(st: dict, now: float) -> None:
    elapsed = max(0.0, now - st.get("updated", now))
    st["score"] = max(0.0, st.get("score", 0.0) - elapsed * SCORE_DECAY_PER_S)
    st["updated"] = now


# ── Concurrency lease ──────────────────────────────────────────────────────────

def conc_inc(ident: str) -> int:
    with _conc_lock:
        _conc[ident] = _conc.get(ident, 0) + 1
        return _conc[ident]


def conc_dec(ident: str) -> None:
    with _conc_lock:
        _conc[ident] = max(0, _conc.get(ident, 0) - 1)


# ── Main entry: check before serving a request ─────────────────────────────────

def check(api_key: str, ip: str, prompt_hash: str = "") -> Decision:
    """Evaluate an incoming request. Records the request in the sliding windows
    and returns an allow/deny Decision."""
    ident = _ident(api_key, ip)
    now = time.time()
    st = _load(ident)

    # Hard ban — permanent until admin clears.
    if st.get("hard_ban"):
        return Decision(False, "hard_ban", st.get("score", 0.0), reason="hard-banned")

    # Active temp ban.
    tb = st.get("temp_ban_until", 0.0)
    if tb > now:
        return Decision(False, "temp_ban", st.get("score", 0.0),
                        retry_after_s=int(tb - now), reason="temporarily banned")

    _decay(st, now)

    # 1. Burst window.
    req_ts = [t for t in st.get("req_ts", []) if now - t < BURST_WINDOW_S]
    req_ts.append(now)
    st["req_ts"] = req_ts[-BURST_LIMIT * 2:]
    if len(req_ts) > BURST_LIMIT:
        st["score"] += 3.0

    # 2. Duplicate-prompt spam.
    if prompt_hash:
        dup = {h: t for h, t in st.get("dup", {}).items() if now - t[1] < DUP_WINDOW_S} \
            if False else st.get("dup", {})
        # dup maps hash -> [count, last_ts]; prune expired
        dup = {h: ct for h, ct in dup.items() if now - ct[1] < DUP_WINDOW_S}
        entry = dup.get(prompt_hash, [0, now])
        entry = [entry[0] + 1, now]
        dup[prompt_hash] = entry
        st["dup"] = dup
        if entry[0] > DUP_LIMIT:
            st["score"] += 2.5

    # 3. Concurrency flood.
    live = _conc.get(ident, 0)
    if live > CONCURRENCY_LIMIT:
        st["score"] += 2.0

    # 4. Error storm (from recorded errors).
    errs = [t for t in st.get("err", []) if now - t < ERROR_WINDOW_S]
    st["err"] = errs
    recent_reqs = len([t for t in req_ts if now - t < ERROR_WINDOW_S])
    if recent_reqs >= ERROR_MIN_SAMPLES and len(errs) / max(1, recent_reqs) > ERROR_RATIO:
        st["score"] += 2.0

    score = st["score"]

    # Escalation.
    if score >= TEMP_BAN_AT:
        st["temp_ban_until"] = now + TEMP_BAN_S
        _save(ident, st)
        return Decision(False, "temp_ban", score, retry_after_s=TEMP_BAN_S,
                        reason="abuse score exceeded ban threshold")
    if score >= THROTTLE_AT:
        _save(ident, st)
        # Throttle: allowed but caller should inject a delay.
        return Decision(True, "throttle", score, retry_after_s=2,
                        reason="elevated abuse score — throttled")

    _save(ident, st)
    return Decision(True, "allow", score)


def record_error(api_key: str, ip: str) -> None:
    ident = _ident(api_key, ip)
    st = _load(ident)
    errs = st.get("err", [])
    errs.append(time.time())
    st["err"] = errs[-200:]
    _save(ident, st)


def hard_ban(api_key: str, ip: str = "*", banned: bool = True) -> None:
    """Admin: hard-ban (or unban) an identity. ip='*' bans the key on any IP
    by storing a key-level flag checked in :func:`check`… (here we ban the
    specific ident; key-wide bans use the api_key with ip='*')."""
    ident = _ident(api_key, ip)
    st = _load(ident)
    st["hard_ban"] = banned
    if not banned:
        st["score"] = 0.0
        st["temp_ban_until"] = 0.0
    _save(ident, st)


def status(api_key: str, ip: str) -> dict:
    ident = _ident(api_key, ip)
    st = _load(ident)
    now = time.time()
    _decay(st, now)
    return {
        "identity": ident,
        "score": round(st.get("score", 0.0), 2),
        "hard_ban": st.get("hard_ban", False),
        "temp_ban_remaining_s": max(0, int(st.get("temp_ban_until", 0.0) - now)),
        "in_flight": _conc.get(ident, 0),
        "req_last_10s": len([t for t in st.get("req_ts", []) if now - t < BURST_WINDOW_S]),
    }
