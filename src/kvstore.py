"""RocksDB key-value index (drop-in for the old LevelDB/plyvel layer).

RocksDB is a faster, more concurrent LSM store than LevelDB (better write
throughput, prefix seeks, column families, tunable caches). This module exposes
a tiny plyvel-compatible surface — ``get`` / ``put`` / ``delete`` /
``iterator(prefix=...)`` — over a single process-wide raw-bytes ``Rdict`` so the
rest of the codebase needs no per-call-site changes.

It is the **hot index**: API keys, pointers into DuckDB, balancer health, abuse
state, cache TTLs. Bulk/analytical data lives in DuckDB (see ``store.py``).
"""

import os
import threading
from typing import Iterator, Optional, Tuple

from rocksdict import Rdict, Options, ReadOptions

_DB_DIR = os.path.expanduser("~/nodataishere/rocks")
os.makedirs(os.path.dirname(_DB_DIR), exist_ok=True)

_db: Optional["RocksKV"] = None
_lock = threading.Lock()


def _tuned_options() -> Options:
    opts = Options(raw_mode=True)
    opts.create_if_missing(True)
    # Throughput/latency tuning for a small home-server workload.
    opts.increase_parallelism(max(2, (os.cpu_count() or 2)))
    opts.optimize_level_style_compaction(64 * 1024 * 1024)  # 64 MB memtable budget
    opts.set_max_background_jobs(4)
    opts.set_bytes_per_sync(1 << 20)
    return opts


class RocksKV:
    """plyvel-compatible wrapper around a raw-bytes RocksDB."""

    def __init__(self, path: str):
        self._rd = Rdict(path, _tuned_options())

    def get(self, key: bytes) -> Optional[bytes]:
        return self._rd.get(key)

    def put(self, key: bytes, value: bytes) -> None:
        self._rd[key] = value

    def delete(self, key: bytes) -> None:
        try:
            del self._rd[key]
        except KeyError:
            pass

    def iterator(self, prefix: bytes = b"") -> Iterator[Tuple[bytes, bytes]]:
        """Yield (key, value) pairs, optionally restricted to a key prefix.

        Uses a RocksDB seek (not a full scan) so prefix queries stay O(matches).
        """
        it = self._rd.iter(ReadOptions())
        if prefix:
            it.seek(prefix)
        else:
            it.seek_to_first()
        while it.valid():
            k = it.key()
            if prefix and not k.startswith(prefix):
                break
            yield k, it.value()
            it.next()

    def close(self) -> None:
        self._rd.close()


def db() -> RocksKV:
    global _db
    if _db is None:
        with _lock:
            if _db is None:
                _db = RocksKV(_DB_DIR)
    return _db
