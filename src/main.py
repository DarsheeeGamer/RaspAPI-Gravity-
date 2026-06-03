"""GravityAPI Gateway — OpenAI-compatible multi-provider LLM gateway.

Stack:
  - RocksDB  — hot key-value index: API keys, pointers, balancer health, abuse,
               discovery-list cache TTLs.
  - DuckDB   — bulk + analytical store: accounts, conversation transcripts,
               request log (metrics), discovery-list bodies.
  - Pydantic — validated models for all stored + wire data.
  - Load balancer with health + failover across accounts.
  - Abuse detection (burst / duplicate / error-storm / concurrency).
  - Server-side conversation history (opt-in via conversation_id).
  - HTTP/3 (QUIC) via Hypercorn — see run_http3.py.

Responses are NOT cached (only model-discovery lists are).
"""

import json
import time
import secrets
import hashlib
import asyncio
from typing import Any, Dict, List, Optional

import fastapi as meow
from fastapi import HTTPException, Depends, Request, Query
from fastapi.responses import HTMLResponse, StreamingResponse, JSONResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.middleware.cors import CORSMiddleware

import abuse
import cache
import history
from models import ChatRequest as ChatReq, AddAccountRequest
from genapi import generate_api_key, validate_api_key, is_admin, revoke_api_key
from llm import (
    generate_response, generate_stream, resolve_model, full_model_name,
    list_models_for_provider, NoAccountError,
)
from account import (
    add_account_to_db, remove_account_from_db, list_accounts_for_key,
    record_request, get_metrics, get_providers_for_key,
)
import balancer
from providers import PROVIDERS, FREE_PROVIDERS

app = meow.FastAPI(
    title="GravityAPI Gateway",
    description="Unified multi-provider LLM gateway powered by GravityV2.",
    version="3.0.0",
    docs_url=None,
    redoc_url=None,
)
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_credentials=False,
    allow_methods=["*"], allow_headers=["*"],
)
security = HTTPBearer(auto_error=False)


# ── Auth ──────────────────────────────────────────────────────────────────────

async def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)) -> str:
    if not credentials:
        raise HTTPException(401, "Missing API key. Set Authorization: Bearer grav_...")
    token = credentials.credentials
    if not validate_api_key(token):
        raise HTTPException(401, "Invalid or revoked API key.")
    return token


async def verify_admin(credentials: HTTPAuthorizationCredentials = Depends(security)) -> str:
    if not credentials:
        raise HTTPException(401, "Missing API key.")
    token = credentials.credentials
    if not is_admin(token):
        raise HTTPException(403, "Admin access required.")
    return token


def _client_ip(request: Request) -> str:
    fwd = request.headers.get("x-forwarded-for")
    if fwd:
        return fwd.split(",")[0].strip()
    return request.client.host if request.client else "0.0.0.0"


# ── Gatekeeping: abuse + rate limit ────────────────────────────────────────────

def _gate(request: Request, api_key: str, provider: str, prompt_hash: str = "") -> str:
    """Abuse-check + per-tier rate-limit. Returns the resolved pool, or raises."""
    ip = _client_ip(request)

    decision = abuse.check(api_key, ip, prompt_hash)
    if not decision.allowed:
        raise HTTPException(
            429 if decision.action != "hard_ban" else 403,
            detail=decision.to_dict(),
            headers={"Retry-After": str(decision.retry_after_s)} if decision.retry_after_s else None,
        )

    has_custom = provider in get_providers_for_key(api_key)
    if not has_custom and provider not in FREE_PROVIDERS:
        raise HTTPException(
            400,
            f"Provider '{provider}' requires a registered account. "
            "Add one via POST /api/add . See GET /api/providers.",
        )

    # Hourly rate limit (sliding hour bucket) in RocksDB.
    limit = 2000 if has_custom else (5 if api_key == "grav_demoapikey" else 20)
    from account import get_db
    db = get_db()
    hour = int(time.time() / 3600)
    rl_key = f"ratelimit:{api_key}:{ip}:{hour}".encode()
    raw = db.get(rl_key)
    count = int(raw.decode()) if raw else 0
    if count >= limit:
        raise HTTPException(429, f"Hourly rate limit of {limit} req/hr exceeded.")
    db.put(rl_key, str(count + 1).encode())

    return "custom" if has_custom else "default", decision.action  # type: ignore


# ── Message / tool conversion ──────────────────────────────────────────────────

def _parse_messages(raw: List[Dict[str, Any]]):
    from gravity.types import Message, ToolCall, ToolResult
    messages, system_prompt = [], None
    for m in raw:
        role = m.get("role", "user")
        content = m.get("content") or ""
        if role == "system":
            if not system_prompt:
                system_prompt = content
            messages.append(Message.system(content))
        elif role == "user":
            if isinstance(content, list):
                content = "\n".join(p.get("text", "") for p in content if p.get("type") == "text")
            messages.append(Message.user(content))
        elif role == "assistant":
            tool_calls = []
            for tc in (m.get("tool_calls") or []):
                fn = tc.get("function", {})
                try:
                    args = json.loads(fn.get("arguments", "{}"))
                except Exception:
                    args = {}
                tool_calls.append(ToolCall(
                    id=tc.get("id") or f"call_{secrets.token_hex(4)}",
                    name=fn.get("name", ""), arguments=args))
            messages.append(Message(role="assistant", content=content or "", tool_calls=tool_calls))
        elif role == "tool":
            try:
                cv = json.loads(content) if isinstance(content, str) else content
            except Exception:
                cv = content
            messages.append(Message.tool_result_message(ToolResult(
                tool_call_id=m.get("tool_call_id", ""), name=m.get("name", "tool"), content=cv)))
    return messages, system_prompt


def _parse_tools(raw: Optional[List[Dict[str, Any]]]):
    if not raw:
        return None
    from gravity.tools import Tool, ToolDefinition
    tools = []
    for t in raw:
        if t.get("type") == "function":
            fn = t.get("function", {})
            tools.append(Tool(definition=ToolDefinition(
                name=fn.get("name", ""), description=fn.get("description", ""),
                input_schema=fn.get("parameters", {"type": "object", "properties": {}}))))
    return tools or None


def _format_tool_calls(tool_calls) -> List[Dict]:
    if not tool_calls:
        return []
    return [{"id": str(tc.id), "type": "function",
             "function": {"name": str(tc.name), "arguments": json.dumps(tc.arguments)}}
            for tc in tool_calls]


def _prompt_fingerprint(messages: List[Dict[str, Any]]) -> str:
    last = next((m for m in reversed(messages) if m.get("role") == "user"), None)
    body = json.dumps(last or {}, sort_keys=True, ensure_ascii=True)
    return hashlib.sha256(body.encode()).hexdigest()[:16]


# ── Chat completions ───────────────────────────────────────────────────────────

@app.post("/v1/chat/completions")
async def chat_completions(req: ChatReq, request: Request, token: str = Depends(verify_token)):
    """OpenAI-compatible chat — load-balanced, abuse-gated, optional history.
    Responses are never cached."""
    provider, model_name = resolve_model(req.model)
    full = full_model_name(provider, model_name)
    ip = _client_ip(request)

    pool, action = _gate(request, token, provider, _prompt_fingerprint(req.messages))
    if action == "throttle":
        await asyncio.sleep(2)  # soft back-pressure on elevated abuse score

    # History: prepend stored transcript + system prompt when continuing a thread.
    convo_msgs: List[Dict[str, Any]] = []
    stored_system = None
    if req.conversation_id:
        rec = history.get_or_create(token, req.conversation_id, system_prompt=req.system_prompt)
        stored_system = rec.get("system_prompt")
        convo_msgs = history.history_messages(token, req.conversation_id)

    effective_raw = convo_msgs + req.messages
    messages, sys_in = _parse_messages(effective_raw)
    system_prompt = req.system_prompt or sys_in or stored_system
    tools = _parse_tools(req.tools)

    ident = f"{token}@{ip}"
    started = time.time()

    # Streaming: the generator owns the concurrency lease (inc here, dec in its finally).
    if req.stream:
        abuse.conc_inc(ident)
        return await _stream_response(
            token, ip, req, full, provider, messages, system_prompt, tools, ident)

    # Non-streaming: lease scoped to this call.
    abuse.conc_inc(ident)
    try:
        try:
            resp = await generate_response(
                token, req.model, messages, system_prompt, tools=tools,
                temperature=req.temperature, max_tokens=req.max_tokens, seed=req.seed)
        except NoAccountError as e:
            raise HTTPException(400, str(e))
        except Exception as e:
            abuse.record_error(token, ip)
            record_request(full, token, ip, provider, "error",
                           latency_ms=(time.time() - started) * 1000)
            raise HTTPException(502, f"Generation failed: {e}")

        content_text = _extract_text(resp)
        oai_tcs = _format_tool_calls(resp.message.tool_calls)
        finish = "tool_calls" if oai_tcs else (resp.finish_reason or "stop")

        if req.conversation_id and req.store:
            to_store = list(req.messages)
            to_store.append({"role": "assistant", "content": content_text,
                             "tool_calls": oai_tcs or None})
            history.append_messages(token, req.conversation_id, to_store)

        record_request(full, token, ip, provider, "ok", cached=False,
                       latency_ms=(time.time() - started) * 1000,
                       in_tokens=resp.usage.input_tokens or 0,
                       out_tokens=resp.usage.output_tokens or 0)

        return {
            "id": f"chatcmpl-{secrets.token_hex(12)}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": req.model,
            "conversation_id": req.conversation_id,
            "choices": [{"index": 0, "message": {
                "role": "assistant",
                "content": content_text or None,
                "tool_calls": oai_tcs or None,
            }, "finish_reason": finish}],
            "usage": {
                "prompt_tokens": resp.usage.input_tokens or 0,
                "completion_tokens": resp.usage.output_tokens or 0,
                "total_tokens": resp.usage.total_tokens or (
                    (resp.usage.input_tokens or 0) + (resp.usage.output_tokens or 0)),
            },
        }
    finally:
        abuse.conc_dec(ident)


async def _stream_response(token, ip, req, full, provider, messages, system_prompt, tools, ident):
    chat_id = f"chatcmpl-{secrets.token_hex(12)}"
    created = int(time.time())
    started = time.time()

    async def gen():
        # First chunk announces the role (OpenAI shape).
        yield _sse({"id": chat_id, "object": "chat.completion.chunk", "created": created,
                    "model": req.model, "choices": [{"index": 0,
                    "delta": {"role": "assistant"}, "finish_reason": None}]})
        collected = []
        try:
            async for piece in generate_stream(
                token, req.model, messages, system_prompt, tools=tools,
                temperature=req.temperature, max_tokens=req.max_tokens):
                collected.append(piece)
                yield _sse({"id": chat_id, "object": "chat.completion.chunk", "created": created,
                            "model": req.model, "choices": [{"index": 0,
                            "delta": {"content": piece}, "finish_reason": None}]})
            yield _sse({"id": chat_id, "object": "chat.completion.chunk", "created": created,
                        "model": req.model, "choices": [{"index": 0, "delta": {},
                        "finish_reason": "stop"}]})
            yield "data: [DONE]\n\n"
            if req.conversation_id and req.store:
                txt = "".join(collected)
                history.append_messages(token, req.conversation_id,
                                        list(req.messages) + [{"role": "assistant", "content": txt}])
            record_request(full, token, ip, provider, "ok",
                           latency_ms=(time.time() - started) * 1000)
        except Exception as e:
            abuse.record_error(token, ip)
            record_request(full, token, ip, provider, "error",
                           latency_ms=(time.time() - started) * 1000)
            yield _sse({"id": chat_id, "object": "chat.completion.chunk", "created": created,
                        "model": req.model, "choices": [{"index": 0,
                        "delta": {"content": f"[Error: {e}]"}, "finish_reason": "error"}]})
            yield "data: [DONE]\n\n"
        finally:
            abuse.conc_dec(ident)  # released here for the streaming path

    # conc was incremented by caller; hand ownership to the generator's finally.
    return StreamingResponse(gen(), media_type="text/event-stream",
                             headers={"x-accel-buffering": "no", "cache-control": "no-cache"})


def _sse(obj: dict) -> str:
    return f"data: {json.dumps(obj)}\n\n"


def _extract_text(resp) -> str:
    c = resp.message.content
    if isinstance(c, str):
        return c
    if hasattr(c, "text"):
        return c.text
    return str(c or "")


# ── Models ──────────────────────────────────────────────────────────────────

@app.get("/v1/models")
def list_models(token: str = Depends(verify_token)):
    now = int(time.time())
    data = []
    for pname, meta in PROVIDERS.items():
        for m in meta["models"]:
            data.append({"id": f"{pname}/{m}", "object": "model", "created": now,
                         "owned_by": pname, "provider": pname,
                         "tool_support": meta["tool_support"],
                         "requires_account": meta["requires_custom_account"]})
    return {"object": "list", "data": data}


@app.get("/v1/models/{provider}")
async def list_provider_models(provider: str, live: bool = Query(False),
                               token: str = Depends(verify_token)):
    provider = provider.lower()
    if provider not in PROVIDERS:
        raise HTTPException(404, f"Unknown provider: {provider!r}")
    meta = PROVIDERS[provider]
    now = int(time.time())
    models = await list_models_for_provider(token, provider, live=live)
    return {
        "object": "list", "provider": provider, "display_name": meta["display_name"],
        "requires_account": meta["requires_custom_account"],
        "discovery": "live" if live else "catalog",
        "data": [{"id": m["id"], "object": "model", "created": now, "owned_by": provider,
                  "display_name": m.get("display_name"), "description": m.get("description"),
                  "tags": m.get("tags", [])} for m in models],
    }


# ── Conversations (history management) ─────────────────────────────────────────

@app.post("/v1/conversations")
def create_conversation(payload: Dict[str, Any] = None, token: str = Depends(verify_token)):
    payload = payload or {}
    rec = history.create_conversation(
        token, system_prompt=payload.get("system_prompt"), title=payload.get("title"))
    return rec


@app.get("/v1/conversations")
def list_conversations(token: str = Depends(verify_token)):
    return {"conversations": history.list_conversations(token)}


@app.get("/v1/conversations/{conv_id}")
def get_conversation(conv_id: str, token: str = Depends(verify_token)):
    rec = history.get_conversation(token, conv_id)
    if rec is None:
        raise HTTPException(404, "Conversation not found.")
    return rec


@app.delete("/v1/conversations/{conv_id}")
def delete_conversation(conv_id: str, token: str = Depends(verify_token)):
    if not history.delete_conversation(token, conv_id):
        raise HTTPException(404, "Conversation not found.")
    return {"status": "deleted", "id": conv_id}


# ── Account management ─────────────────────────────────────────────────────────

@app.post("/api/generate-key")
def generate_key_endpoint():
    key = generate_api_key()
    return {"api_key": key, "tier": "default",
            "limits": {"shared_pool": "20 req/hr", "custom_pool": "2000 req/hr"},
            "free_providers": sorted(FREE_PROVIDERS)}


@app.post("/api/add")
def add_account(req: AddAccountRequest, token: str = Depends(verify_token)):
    payload = req.model_dump(exclude_none=True)
    try:
        return add_account_to_db(token, payload)
    except ValueError as e:
        raise HTTPException(400, str(e))
    except Exception as e:
        raise HTTPException(500, f"Account registration failed: {e}")


@app.delete("/api/accounts/{account_id}")
def delete_account(account_id: str, provider: str = Query(...), token: str = Depends(verify_token)):
    result = remove_account_from_db(token, account_id, provider)
    if result["status"] == "not_found":
        raise HTTPException(404, "Account not found.")
    return result


@app.get("/api/accounts")
def get_accounts(token: str = Depends(verify_token)):
    return {"accounts": list_accounts_for_key(token),
            "registered_providers": sorted(get_providers_for_key(token))}


@app.get("/api/providers")
def get_providers():
    result = [{
        "id": n, "display_name": m["display_name"], "category": m["category"],
        "description": m["description"], "auth_kinds": m["auth_kinds"],
        "tool_support": m["tool_support"], "requires_account": m["requires_custom_account"],
        "free": not m["requires_custom_account"], "models": m["models"],
    } for n, m in PROVIDERS.items()]
    return {"total": len(result), "free_providers": sorted(FREE_PROVIDERS), "providers": result}


# ── Ops: metrics, balancer, abuse, cache ───────────────────────────────────────

@app.get("/api/metrics")
def metrics(token: str = Depends(verify_token)):
    m = get_metrics()
    m["cache"] = cache.stats()
    return m


@app.get("/api/balancer")
def balancer_health(token: str = Depends(verify_token)):
    from account import get_db
    return {"accounts": balancer.health_snapshot(get_db(), token)}


@app.get("/api/abuse")
def abuse_status(request: Request, token: str = Depends(verify_token)):
    return abuse.status(token, _client_ip(request))


@app.get("/healthz")
def healthz():
    return {"status": "ok", "version": "3.0.0", "providers": len(PROVIDERS)}


# ── Admin ─────────────────────────────────────────────────────────────────────

@app.post("/admin/generate-key")
def admin_generate_key(payload: Dict[str, Any], token: str = Depends(verify_admin)):
    return {"api_key": generate_api_key(label=payload.get("label", ""),
                                        tier=payload.get("tier", "default")),
            "tier": payload.get("tier", "default")}


@app.post("/admin/revoke-key")
def admin_revoke_key(payload: Dict[str, Any], token: str = Depends(verify_admin)):
    key = payload.get("api_key", "")
    if not key:
        raise HTTPException(400, "Missing api_key")
    return {"status": "revoked" if revoke_api_key(key) else "not_found"}


@app.post("/admin/ban")
def admin_ban(payload: Dict[str, Any], token: str = Depends(verify_admin)):
    api_key = payload.get("api_key", "")
    ip = payload.get("ip", "*")
    banned = payload.get("banned", True)
    if not api_key:
        raise HTTPException(400, "Missing api_key")
    abuse.hard_ban(api_key, ip, banned)
    return {"status": "banned" if banned else "unbanned", "api_key": api_key, "ip": ip}


# ── Root ──────────────────────────────────────────────────────────────────────

@app.get("/")
def index():
    return {
        "service": "GravityAPI Gateway", "version": "3.0.0",
        "providers": len(PROVIDERS), "free_providers": sorted(FREE_PROVIDERS),
        "storage": {"index": "RocksDB", "bulk": "DuckDB"},
        "features": ["load-balancer", "failover", "abuse-detection",
                     "conversation-history", "http3", "streaming"],
        "endpoints": {
            "POST /v1/chat/completions": "chat (stream + non-stream, history, balanced)",
            "GET  /v1/models[/{provider}?live=]": "model catalog / live discovery",
            "POST /v1/conversations": "create conversation",
            "GET/DELETE /v1/conversations/{id}": "fetch / delete conversation",
            "POST /api/add": "register a provider account",
            "GET  /api/accounts": "your accounts",
            "GET  /api/providers": "provider catalog",
            "GET  /api/metrics": "analytics (DuckDB)",
            "GET  /api/balancer": "account health",
            "GET  /api/abuse": "your abuse score",
        },
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=6969, reload=True)
