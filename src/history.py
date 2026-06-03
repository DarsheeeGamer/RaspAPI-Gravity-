"""Server-side conversation history.

Split storage (per the LevelDB-index / DuckDB-bulk design):
  - **LevelDB**: conversation metadata (id, title, system_prompt, timestamps)
    + the per-key conversation index. Tiny, hot, microsecond reads.
  - **DuckDB**: the message transcripts (bulk), addressed by ``conv_id``.

Wire usage on /v1/chat/completions: pass ``"conversation_id"`` to continue a
stored thread — prior messages are prepended, new turns appended back. Omit it
for stateless behaviour.
"""

import json
import time
import secrets
from typing import Optional

import store
from account import get_db


def _meta_key(api_key: str, conv_id: str) -> bytes:
    return f"convmeta:{api_key}:{conv_id}".encode()


def _index_key(api_key: str) -> bytes:
    return f"conv_index:{api_key}".encode()


def _now() -> float:
    return time.time()


def _load_meta(api_key: str, conv_id: str) -> Optional[dict]:
    raw = get_db().get(_meta_key(api_key, conv_id))
    if not raw:
        return None
    try:
        return json.loads(raw.decode())
    except Exception:
        return None


def _save_meta(api_key: str, meta: dict) -> None:
    get_db().put(_meta_key(api_key, meta["id"]), json.dumps(meta).encode())


def create_conversation(api_key: str, system_prompt: Optional[str] = None,
                        title: Optional[str] = None, metadata: Optional[dict] = None) -> dict:
    conv_id = f"conv_{secrets.token_urlsafe(12)}"
    meta = {
        "id": conv_id, "title": title or "", "system_prompt": system_prompt,
        "created_at": _now(), "updated_at": _now(), "metadata": metadata or {},
    }
    _save_meta(api_key, meta)
    _index_add(api_key, conv_id)
    return {**meta, "messages": []}


def get_conversation(api_key: str, conv_id: str, with_messages: bool = True) -> Optional[dict]:
    meta = _load_meta(api_key, conv_id)
    if meta is None:
        return None
    if with_messages:
        meta = {**meta, "messages": store.get_messages(conv_id)}
    return meta


def get_or_create(api_key: str, conv_id: Optional[str],
                  system_prompt: Optional[str] = None) -> dict:
    if conv_id:
        existing = _load_meta(api_key, conv_id)
        if existing:
            return existing
        meta = {
            "id": conv_id, "title": "", "system_prompt": system_prompt,
            "created_at": _now(), "updated_at": _now(), "metadata": {},
        }
        _save_meta(api_key, meta)
        _index_add(api_key, conv_id)
        return meta
    rec = create_conversation(api_key, system_prompt=system_prompt)
    return rec


def history_messages(api_key: str, conv_id: str) -> list[dict]:
    """Stored transcript as OpenAI-shape dicts (DuckDB point lookup by conv_id)."""
    if _load_meta(api_key, conv_id) is None:
        return []
    return store.get_messages(conv_id)


def append_messages(api_key: str, conv_id: str, messages: list[dict]) -> None:
    meta = _load_meta(api_key, conv_id)
    if meta is None:
        get_or_create(api_key, conv_id)
        meta = _load_meta(api_key, conv_id)
    store.append_messages(api_key, conv_id, messages)
    meta["updated_at"] = _now()
    _save_meta(api_key, meta)


def set_system_prompt(api_key: str, conv_id: str, system_prompt: Optional[str]) -> None:
    meta = _load_meta(api_key, conv_id)
    if meta is not None:
        meta["system_prompt"] = system_prompt
        meta["updated_at"] = _now()
        _save_meta(api_key, meta)


def delete_conversation(api_key: str, conv_id: str) -> bool:
    if _load_meta(api_key, conv_id) is None:
        return False
    try:
        get_db().delete(_meta_key(api_key, conv_id))
    except Exception:
        pass
    store.delete_conversation_messages(conv_id)
    _index_remove(api_key, conv_id)
    return True


def list_conversations(api_key: str) -> list[dict]:
    out = []
    for conv_id in _index_get(api_key):
        meta = _load_meta(api_key, conv_id)
        if meta:
            out.append({
                "id": meta["id"], "title": meta.get("title", ""),
                "created_at": meta.get("created_at"), "updated_at": meta.get("updated_at"),
            })
    out.sort(key=lambda r: r.get("updated_at", 0), reverse=True)
    return out


# ── Per-key conversation index (LevelDB) ───────────────────────────────────────

def _index_get(api_key: str) -> list[str]:
    raw = get_db().get(_index_key(api_key))
    if not raw:
        return []
    try:
        return json.loads(raw.decode())
    except Exception:
        return []


def _index_add(api_key: str, conv_id: str) -> None:
    ids = _index_get(api_key)
    if conv_id not in ids:
        ids.append(conv_id)
        get_db().put(_index_key(api_key), json.dumps(ids).encode())


def _index_remove(api_key: str, conv_id: str) -> None:
    ids = [c for c in _index_get(api_key) if c != conv_id]
    get_db().put(_index_key(api_key), json.dumps(ids).encode())
