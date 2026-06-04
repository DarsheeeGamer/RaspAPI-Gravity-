"""GravityAPI Gateway — OpenAI-compatible multi-provider LLM gateway.

Stack:
  - RocksDB  — hot key-value index: API keys, quota counters, pointers,
               balancer health, abuse, discovery-list cache TTLs.
  - DuckDB   — bulk + analytical store: accounts, request log (metrics),
               discovery-list bodies.
  - Pydantic — validated models for all stored + wire data.
  - Load balancer with health + failover across accounts.
  - Abuse detection (burst / duplicate / error-storm / concurrency).
  - Auth (API key / JWT / anonymous) with tiered per-provider quota.
  - HTTP/3 (QUIC) via Hypercorn — see run_http3.py.

**Stateless**: the client owns conversation history (full `messages` array each
call, OpenAI-style). The server never persists transcripts. Responses are NOT
cached either (only model-discovery lists are).
"""

import json
import time
import secrets
import hashlib
import asyncio
import contextlib
from typing import Any, Dict, List, Optional

import fastapi as meow
from fastapi import HTTPException, Depends, Request, Query, Header
from fastapi.responses import HTMLResponse, StreamingResponse, JSONResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.middleware.cors import CORSMiddleware

import abuse
import cache
import auth
import quota
import jobs
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

@contextlib.asynccontextmanager
async def _lifespan(app):
    jobs.register_default_jobs()
    jobs.scheduler.start()
    try:
        yield
    finally:
        await jobs.scheduler.stop()


app = meow.FastAPI(
    title="GravityAPI Gateway",
    description="Unified multi-provider LLM gateway powered by GravityV2.",
    version="3.1.0",
    docs_url=None,
    redoc_url=None,
    lifespan=_lifespan,
)
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_credentials=False,
    allow_methods=["*"], allow_headers=["*"],
)
security = HTTPBearer(auto_error=False)


# ── Auth ──────────────────────────────────────────────────────────────────────

def _client_ip(request: Request) -> str:
    fwd = request.headers.get("x-forwarded-for")
    if fwd:
        return fwd.split(",")[0].strip()
    return request.client.host if request.client else "0.0.0.0"


async def get_principal(request: Request,
                        authorization: Optional[str] = Header(None)) -> auth.Principal:
    """Optional auth — resolves an API key / JWT, or an anonymous IP principal."""
    return auth.resolve_principal(authorization, _client_ip(request))


async def require_principal(request: Request,
                            authorization: Optional[str] = Header(None)) -> auth.Principal:
    """Authenticated-only — rejects anonymous callers."""
    p = auth.resolve_principal(authorization, _client_ip(request))
    if not p.is_authenticated:
        raise HTTPException(401, "Authentication required. Provide Authorization: Bearer <key|jwt>.")
    return p


async def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)) -> str:
    """Legacy API-key dependency (account/conversation management)."""
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


# ── Gatekeeping: dual (auth + anon) abuse + tiered quota ───────────────────────

def _gate(request: Request, principal: auth.Principal, provider: str,
          prompt_hash: str = "") -> str:
    """Abuse-check + tier-quota for a principal (authenticated or anonymous).
    Returns the abuse action ("allow"/"throttle"), or raises 4xx."""
    ip = _client_ip(request)
    # Abuse identity: the key for authed callers, the IP for anonymous.
    ident = principal.api_key or principal.id

    decision = abuse.check(ident, ip, prompt_hash)
    if not decision.allowed:
        raise HTTPException(
            429 if decision.action != "hard_ban" else 403,
            detail=decision.to_dict(),
            headers={"Retry-After": str(decision.retry_after_s)} if decision.retry_after_s else None,
        )

    # Anonymous callers: free providers only, no custom accounts.
    has_custom = bool(principal.api_key) and provider in get_providers_for_key(principal.api_key)
    if not principal.is_authenticated and provider not in FREE_PROVIDERS:
        raise HTTPException(
            401,
            f"Provider '{provider}' requires authentication. "
            "Sign up (POST /auth/signup) or send Authorization: Bearer <key>.",
        )
    if principal.is_authenticated and not has_custom and provider not in FREE_PROVIDERS:
        raise HTTPException(
            400,
            f"Provider '{provider}' requires a registered account. Add one via POST /api/add.",
        )

    # Tiered per-provider quota (hot RocksDB counter at the cell path).
    tier = "custom" if has_custom else principal.tier
    limit = auth.quota_limit_for(tier)
    ok, qstat = quota.try_consume(ident, provider, default_limit=limit)
    if not ok:
        raise HTTPException(
            429, detail={"error": "quota exceeded", "tier": tier, **qstat},
            headers={"Retry-After": str(qstat.get("reset_in_s", 60))},
        )
    return decision.action


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
async def chat_completions(req: ChatReq, request: Request,
                           principal: auth.Principal = Depends(get_principal)):
    """OpenAI-compatible **stateless** chat — works authenticated **or**
    anonymously (free providers only, strict quota). Load-balanced, abuse-gated.
    The client sends the full message history each call; nothing is persisted."""
    provider, model_name = resolve_model(req.model)
    full = full_model_name(provider, model_name)
    ip = _client_ip(request)
    # Scope for accounts / quota: the key for authed, the IP id for anon.
    scope = principal.api_key or principal.id

    action = _gate(request, principal, provider, _prompt_fingerprint(req.messages))
    if action == "throttle":
        await asyncio.sleep(2)  # soft back-pressure on elevated abuse score

    # Stateless: the client owns history — we only ever use the request's messages.
    messages, sys_in = _parse_messages(req.messages)
    system_prompt = req.system_prompt or sys_in
    tools = _parse_tools(req.tools)

    ident = f"{scope}@{ip}"
    started = time.time()

    # Streaming: the generator owns the concurrency lease (inc here, dec in its finally).
    if req.stream:
        abuse.conc_inc(ident)
        return await _stream_response(
            scope, ip, req, full, provider, messages, system_prompt, tools, ident)

    # Non-streaming: lease scoped to this call.
    abuse.conc_inc(ident)
    try:
        try:
            resp = await generate_response(
                scope, req.model, messages, system_prompt, tools=tools,
                temperature=req.temperature, max_tokens=req.max_tokens, seed=req.seed)
        except NoAccountError as e:
            raise HTTPException(400, str(e))
        except Exception as e:
            abuse.record_error(scope, ip)
            record_request(full, scope, ip, provider, "error",
                           latency_ms=(time.time() - started) * 1000)
            raise HTTPException(502, f"Generation failed: {e}")

        content_text = _extract_text(resp)
        oai_tcs = _format_tool_calls(resp.message.tool_calls)
        finish = "tool_calls" if oai_tcs else (resp.finish_reason or "stop")

        record_request(full, scope, ip, provider, "ok", cached=False,
                       latency_ms=(time.time() - started) * 1000,
                       in_tokens=resp.usage.input_tokens or 0,
                       out_tokens=resp.usage.output_tokens or 0)

        return {
            "id": f"chatcmpl-{secrets.token_hex(12)}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": req.model,
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


async def _stream_response(scope, ip, req, full, provider, messages, system_prompt,
                           tools, ident):
    chat_id = f"chatcmpl-{secrets.token_hex(12)}"
    created = int(time.time())
    started = time.time()

    async def gen():
        # First chunk announces the role (OpenAI shape).
        yield _sse({"id": chat_id, "object": "chat.completion.chunk", "created": created,
                    "model": req.model, "choices": [{"index": 0,
                    "delta": {"role": "assistant"}, "finish_reason": None}]})
        try:
            async for piece in generate_stream(
                scope, req.model, messages, system_prompt, tools=tools,
                temperature=req.temperature, max_tokens=req.max_tokens):
                yield _sse({"id": chat_id, "object": "chat.completion.chunk", "created": created,
                            "model": req.model, "choices": [{"index": 0,
                            "delta": {"content": piece}, "finish_reason": None}]})
            yield _sse({"id": chat_id, "object": "chat.completion.chunk", "created": created,
                        "model": req.model, "choices": [{"index": 0, "delta": {},
                        "finish_reason": "stop"}]})
            yield "data: [DONE]\n\n"
            record_request(full, scope, ip, provider, "ok",
                           latency_ms=(time.time() - started) * 1000)
        except Exception as e:
            abuse.record_error(scope, ip)
            record_request(full, scope, ip, provider, "error",
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


# ── Authentication ─────────────────────────────────────────────────────────────

@app.post("/auth/signup")
def auth_signup(payload: Dict[str, Any]):
    try:
        return auth.signup(payload.get("email", ""), payload.get("password", ""))
    except auth.AuthError as e:
        raise HTTPException(400, str(e))


@app.post("/auth/login")
def auth_login(payload: Dict[str, Any]):
    try:
        return auth.login(payload.get("email", ""), payload.get("password", ""))
    except auth.AuthError as e:
        raise HTTPException(401, str(e))


@app.post("/auth/refresh")
def auth_refresh(payload: Dict[str, Any]):
    try:
        return auth.refresh(payload.get("refresh_token", ""))
    except auth.AuthError as e:
        raise HTTPException(401, str(e))


@app.get("/auth/me")
def auth_me(principal: auth.Principal = Depends(require_principal)):
    return {"kind": principal.kind, "tier": principal.tier,
            "user_id": principal.user_id, "api_key": principal.api_key,
            "authenticated": principal.is_authenticated}


@app.get("/auth/keys")
def auth_keys(principal: auth.Principal = Depends(require_principal)):
    if not principal.user_id:
        return {"keys": [principal.api_key] if principal.api_key else []}
    return {"keys": auth.user_keys(principal.user_id)}


@app.post("/auth/keys")
def auth_new_key(payload: Dict[str, Any] = None,
                 principal: auth.Principal = Depends(require_principal)):
    payload = payload or {}
    key = generate_api_key(label=payload.get("label", ""))
    if principal.user_id:
        from account import get_db
        get_db().put(f"apikey:{key}:user".encode(), principal.user_id.encode())
        auth._link_key_to_user(principal.user_id, key)
    return {"api_key": key}


# ── Models ──────────────────────────────────────────────────────────────────

@app.get("/v1/models")
def list_models(principal: auth.Principal = Depends(get_principal)):
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
                               principal: auth.Principal = Depends(get_principal)):
    provider = provider.lower()
    if provider not in PROVIDERS:
        raise HTTPException(404, f"Unknown provider: {provider!r}")
    meta = PROVIDERS[provider]
    now = int(time.time())
    models = await list_models_for_provider(principal.api_key or "", provider, live=live)
    return {
        "object": "list", "provider": provider, "display_name": meta["display_name"],
        "requires_account": meta["requires_custom_account"],
        "discovery": "live" if live else "catalog",
        "data": [{"id": m["id"], "object": "model", "created": now, "owned_by": provider,
                  "display_name": m.get("display_name"), "description": m.get("description"),
                  "tags": m.get("tags", [])} for m in models],
    }


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


@app.get("/api/quota")
def quota_status(principal: auth.Principal = Depends(get_principal), request: Request = None):
    scope = principal.api_key or principal.id
    eff = auth.quota_limit_for(principal.tier)
    snap = quota.snapshot(scope)
    # Surface the effective per-tier limit where no explicit ceiling is set.
    for p in snap:
        if not p.get("limit"):
            p["limit"] = eff
            p["remaining"] = None if eff == 0 else max(0, eff - p["usage"])
            p["unlimited"] = eff == 0
    return {"tier": principal.tier, "scope": scope,
            "default_limit_per_hr": eff, "providers": snap}


@app.get("/api/jobs")
def jobs_report(token: str = Depends(verify_admin)):
    return {"running": jobs.scheduler._task is not None, "jobs": jobs.scheduler.report()}


@app.post("/api/jobs/run")
async def jobs_run(payload: Dict[str, Any], token: str = Depends(verify_admin)):
    """Trigger a maintenance job by name immediately."""
    name = payload.get("name", "")
    job = next((j for j in jobs.scheduler.jobs if j.name == name), None)
    if not job:
        raise HTTPException(404, f"unknown job: {name}. have: {[j.name for j in jobs.scheduler.jobs]}")
    await jobs.scheduler._run_job(job)
    return {"name": name, "last_result": job.last_result}


@app.get("/healthz")
def healthz():
    return {"status": "ok", "version": "3.1.0", "providers": len(PROVIDERS),
            "jobs_running": jobs.scheduler._task is not None}


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
        "service": "GravityAPI Gateway", "version": "3.1.0",
        "stateless": True,
        "providers": len(PROVIDERS), "free_providers": sorted(FREE_PROVIDERS),
        "storage": {"index": "RocksDB", "bulk": "DuckDB"},
        "features": ["load-balancer", "failover", "abuse-detection",
                     "tiered-quota", "auth", "http3", "streaming"],
        "endpoints": {
            "POST /v1/chat/completions": "stateless chat (stream + non-stream, balanced)",
            "GET  /v1/models[/{provider}?live=]": "model catalog / live discovery",
            "POST /auth/signup|login|refresh": "authentication",
            "POST /api/add": "register a provider account",
            "GET  /api/accounts": "your accounts",
            "GET  /api/providers": "provider catalog",
            "GET  /api/quota": "your quota usage",
            "GET  /api/metrics": "analytics (DuckDB)",
            "GET  /api/balancer": "account health",
            "GET  /api/abuse": "your abuse score",
        },
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=6969, reload=True)
