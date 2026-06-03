"""Tiny thread-safe TTL+LRU memo cache for hot reads.

RocksDB point reads are fast, but validating an API key / resolving a principal
happens on *every* request. A short-TTL in-process cache collapses that to a
dict hit for repeat callers, with bounded memory and automatic expiry. Used for
api-key validation, tier lookups, and JWT verification results.
"""

import time
import threading
from collections import OrderedDict
from typing import Any, Callable, Optional


class TTLCache:
    def __init__(self, maxsize: int = 4096, ttl: float = 30.0):
        self.maxsize = maxsize
        self.ttl = ttl
        self._d: "OrderedDict[Any, tuple[float, Any]]" = OrderedDict()
        self._lock = threading.Lock()

    def get(self, key: Any) -> Optional[Any]:
        now = time.time()
        with self._lock:
            item = self._d.get(key)
            if item is None:
                return None
            exp, val = item
            if exp < now:
                self._d.pop(key, None)
                return None
            self._d.move_to_end(key)
            return val

    def put(self, key: Any, val: Any, ttl: Optional[float] = None) -> None:
        exp = time.time() + (ttl if ttl is not None else self.ttl)
        with self._lock:
            self._d[key] = (exp, val)
            self._d.move_to_end(key)
            while len(self._d) > self.maxsize:
                self._d.popitem(last=False)

    def invalidate(self, key: Any) -> None:
        with self._lock:
            self._d.pop(key, None)

    def clear(self) -> None:
        with self._lock:
            self._d.clear()

    def get_or_set(self, key: Any, factory: Callable[[], Any],
                   ttl: Optional[float] = None) -> Any:
        v = self.get(key)
        if v is not None:
            return v
        v = factory()
        self.put(key, v, ttl)
        return v
