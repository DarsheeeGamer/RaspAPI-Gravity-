"""Cron-style background jobs for quota + storage maintenance.

A lightweight asyncio scheduler (no external dep) runs periodic jobs that keep
the hot RocksDB index lean and quota windows honest:

  - **quota_sweep**       roll expired quota windows + drop fully-decayed counters
  - **ratelimit_gc**      delete stale hourly rate-limit buckets
  - **discovery_gc**      evict expired discovery-cache pointers (+ DuckDB rows)
  - **abuse_gc**          clear expired temp-bans and zero-score abuse states
  - **balancer_gc**       forget long-healthy balancer health entries
  - **duckdb_compact**    prune old request_log rows + checkpoint DuckDB

Each job is wrapped so one failure never kills the loop. Started/stopped from
the FastAPI lifespan in main.py.
"""

import time
import asyncio
import contextlib
from dataclasses import dataclass
from typing import Awaitable, Callable, Optional

from account import get_db

# ── Scheduler ─────────────────────────────────────────────────────────────────

@dataclass
class Job:
    name: str
    interval_s: float
    fn: Callable[[], Awaitable[dict] | dict]
    last_run: float = 0.0
    last_result: Optional[dict] = None
    runs: int = 0
    errors: int = 0


class Scheduler:
    def __init__(self):
        self.jobs: list[Job] = []
        self._task: Optional[asyncio.Task] = None
        self._stop = asyncio.Event()

    def add(self, name: str, interval_s: float, fn) -> None:
        self.jobs.append(Job(name=name, interval_s=interval_s, fn=fn))

    async def _run_job(self, job: Job) -> None:
        try:
            res = job.fn()
            if asyncio.iscoroutine(res):
                res = await res
            job.last_result = res if isinstance(res, dict) else {"ok": True}
            job.runs += 1
        except Exception as e:  # noqa: BLE001 — a job error must not kill the loop
            job.errors += 1
            job.last_result = {"error": str(e)}
        finally:
            job.last_run = time.time()

    async def _loop(self) -> None:
        # Stagger first runs slightly so they don't all fire at boot.
        await asyncio.sleep(2)
        while not self._stop.is_set():
            now = time.time()
            for job in self.jobs:
                if now - job.last_run >= job.interval_s:
                    await self._run_job(job)
            with contextlib.suppress(asyncio.TimeoutError):
                await asyncio.wait_for(self._stop.wait(), timeout=5.0)

    def start(self) -> None:
        if self._task is None:
            self._stop.clear()
            self._task = asyncio.create_task(self._loop())

    async def stop(self) -> None:
        self._stop.set()
        if self._task:
            with contextlib.suppress(asyncio.CancelledError):
                await self._task
            self._task = None

    def report(self) -> list[dict]:
        return [{
            "name": j.name, "interval_s": j.interval_s,
            "runs": j.runs, "errors": j.errors,
            "last_run": j.last_run, "last_result": j.last_result,
        } for j in self.jobs]


scheduler = Scheduler()


# ── Jobs ──────────────────────────────────────────────────────────────────────

def quota_sweep() -> dict:
    """Roll expired quota windows: zero usage whose reset_at has passed."""
    db = get_db()
    now = time.time()
    rolled = 0
    for k, v in list(db.iterator(prefix=b"")):
        ks = k.decode("utf-8", "ignore")
        if not ks.endswith(":quota:reset_at"):
            continue
        try:
            reset_at = int(v.decode())
        except Exception:
            continue
        if reset_at <= now:
            base = ks[: -len(":reset_at")]
            db.put(f"{base}:currentusage".encode(), b"0")
            db.put(f"{base}:reset_at".encode(), str(int(now + 3600)).encode())
            rolled += 1
    return {"rolled_windows": rolled}


def ratelimit_gc() -> dict:
    """Delete rate-limit buckets older than 2 hours."""
    db = get_db()
    cur_hour = int(time.time() / 3600)
    removed = 0
    for k, _ in list(db.iterator(prefix=b"")):
        ks = k.decode("utf-8", "ignore")
        if ":rl:" not in ks and not ks.startswith("ratelimit:"):
            continue
        # Trailing component is the hour/window bucket.
        tail = ks.rsplit(":", 1)[-1]
        if tail.isdigit() and int(tail) < cur_hour - 2:
            db.delete(k)
            removed += 1
    return {"removed_buckets": removed}


def discovery_gc() -> dict:
    """Evict expired discovery-cache pointers (DuckDB rows are overwritten on next put)."""
    import json
    db = get_db()
    now = time.time()
    removed = 0
    for k, v in list(db.iterator(prefix=b"cacheptr:")):
        try:
            ptr = json.loads(v.decode())
        except Exception:
            continue
        if ptr.get("expires_at", 0) < now:
            db.delete(k)
            removed += 1
    return {"removed_cache_ptrs": removed}


def abuse_gc() -> dict:
    """Drop abuse states that have fully decayed and aren't banned."""
    import json
    db = get_db()
    now = time.time()
    removed = 0
    for k, v in list(db.iterator(prefix=b"abuse:state:")):
        try:
            st = json.loads(v.decode())
        except Exception:
            continue
        if (not st.get("hard_ban") and st.get("temp_ban_until", 0) < now
                and st.get("score", 0) < 0.1
                and now - st.get("updated", 0) > 1800):
            db.delete(k)
            removed += 1
    return {"removed_abuse_states": removed}


def balancer_gc() -> dict:
    """Forget balancer health that has been healthy + idle for a long time."""
    import json
    db = get_db()
    now = time.time()
    removed = 0
    for k, v in list(db.iterator(prefix=b"lb:health:")):
        try:
            h = json.loads(v.decode())
        except Exception:
            continue
        if (h.get("fails", 0) == 0 and h.get("cooldown_until", 0) < now
                and now - h.get("last_ok", 0) > 3600):
            db.delete(k)
            removed += 1
    return {"removed_health": removed}


def duckdb_compact() -> dict:
    """Prune request-log rows older than 30 days and checkpoint DuckDB."""
    import store
    cutoff = time.time() - 30 * 86400
    with store._lock:
        c = store.duck()
        c.execute("DELETE FROM request_log WHERE ts < ?", [cutoff])
        c.execute("CHECKPOINT")
    return {"compacted": True}


def register_default_jobs() -> None:
    scheduler.add("quota_sweep", 300, quota_sweep)        # 5 min
    scheduler.add("ratelimit_gc", 3600, ratelimit_gc)     # 1 h
    scheduler.add("discovery_gc", 600, discovery_gc)      # 10 min
    scheduler.add("abuse_gc", 1800, abuse_gc)             # 30 min
    scheduler.add("balancer_gc", 3600, balancer_gc)       # 1 h
    scheduler.add("duckdb_compact", 86400, duckdb_compact)  # 1 day
