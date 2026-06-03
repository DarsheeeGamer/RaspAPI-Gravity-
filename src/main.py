"""GravityAPI Gateway — OpenAI-compatible multi-provider LLM gateway.

Endpoints:
  GET  /                          service status + directory
  GET  /v1/models                 list all available models
  GET  /v1/models/{provider}      list models for one provider
  POST /v1/chat/completions       OpenAI-compatible chat (streaming + tools)
  POST /v1/embeddings             windsurf/cursor native embeddings
  POST /api/generate-key          create a new API key
  POST /api/add                   register a provider account
  DELETE /api/accounts/{id}       remove an account
  GET  /api/accounts              list your linked accounts
  GET  /api/providers             list all supported providers + capabilities
  GET  /api/metrics               request analytics
  GET  /docs                      interactive documentation
  POST /admin/generate-key        generate key with custom tier (admin only)
  POST /admin/revoke-key          revoke a key (admin only)
"""

import fastapi as meow
from fastapi import HTTPException, Depends, Request, Query
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import json
import time
import secrets

from genapi import generate_api_key, validate_api_key, is_admin, revoke_api_key
from llm import generate_response, generate_stream, resolve_model, full_model_name
from account import (
    add_account_to_db, remove_account_from_db, list_accounts_for_key,
    record_request, get_metrics, get_db,
    get_providers_for_key,
)
from providers import PROVIDERS, FREE_PROVIDERS

# ── App setup ─────────────────────────────────────────────────────────────────

app = meow.FastAPI(
    title="GravityAPI Gateway",
    description="Unified multi-provider LLM gateway powered by GravityV2.",
    version="2.0.0",
    docs_url=None,
    redoc_url=None,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
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


# ── Rate limiting ─────────────────────────────────────────────────────────────

def check_rate_limit(ip: str, api_key: str, provider: str):
    has_custom = provider in get_providers_for_key(api_key)
    if api_key == "grav_demoapikey":
        limit = 5
    elif has_custom:
        limit = 2000
    else:
        limit = 20

    hour = int(time.time() / 3600)
    db = get_db()
    rl_key = f"ratelimit:{api_key}:{ip}:{hour}".encode()
    count_b = db.get(rl_key)
    count = int(count_b.decode()) if count_b else 0
    if count >= limit:
        raise HTTPException(429, f"Hourly rate limit of {limit} req/hr exceeded.")
    db.put(rl_key, str(count + 1).encode())

    # Require custom account for non-free providers
    if not has_custom and provider not in FREE_PROVIDERS:
        raise HTTPException(
            400,
            f"Provider '{provider}' requires a registered account. "
            "Add one via POST /api/add . See GET /api/providers for all providers.",
        )


# ── Message/tool conversion ───────────────────────────────────────────────────

def _parse_messages(raw: List[Dict[str, Any]]):
    from gravity.types import Message, ToolCall, ToolResult

    messages = []
    system_prompt = None

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
            tcs_raw = m.get("tool_calls") or []
            tool_calls = []
            for tc in tcs_raw:
                func = tc.get("function", {})
                try:
                    args = json.loads(func.get("arguments", "{}"))
                except Exception:
                    args = {}
                tool_calls.append(ToolCall(
                    id=tc.get("id") or f"call_{secrets.token_hex(4)}",
                    name=func.get("name", ""),
                    arguments=args,
                ))
            messages.append(Message(role="assistant", content=content or "", tool_calls=tool_calls))

        elif role == "tool":
            try:
                content_val = json.loads(content) if isinstance(content, str) else content
            except Exception:
                content_val = content
            messages.append(Message.tool_result_message(ToolResult(
                tool_call_id=m.get("tool_call_id", ""),
                name=m.get("name", "tool"),
                content=content_val,
            )))

    return messages, system_prompt


def _parse_tools(raw: Optional[List[Dict[str, Any]]]):
    if not raw:
        return None
    from gravity.tools import Tool, ToolDefinition
    tools = []
    for t in raw:
        if t.get("type") == "function":
            func = t.get("function", {})
            tools.append(Tool(definition=ToolDefinition(
                name=func.get("name", ""),
                description=func.get("description", ""),
                input_schema=func.get("parameters", {"type": "object", "properties": {}}),
            )))
    return tools or None


def _format_tool_calls(tool_calls) -> List[Dict]:
    if not tool_calls:
        return []
    return [{
        "id": str(tc.id),
        "type": "function",
        "function": {"name": str(tc.name), "arguments": json.dumps(tc.arguments)},
    } for tc in tool_calls]


# ── Schemas ───────────────────────────────────────────────────────────────────

class ChatRequest(BaseModel):
    model: str
    messages: List[Dict[str, Any]]
    stream: Optional[bool] = False
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None
    seed: Optional[int] = None
    system_prompt: Optional[str] = None
    tools: Optional[List[Dict[str, Any]]] = None
    tool_choice: Optional[Any] = None
    frequency_penalty: Optional[float] = None
    presence_penalty: Optional[float] = None


class AddAccountRequest(BaseModel):
    account_id: str
    provider: str
    # Auth fields — supply what your provider needs
    api_key: Optional[str] = None
    access_token: Optional[str] = None
    refresh_token: Optional[str] = None
    session_key: Optional[str] = None
    cookies: Optional[Dict[str, str]] = None
    # Provider-specific
    org_id: Optional[str] = None
    device_id: Optional[str] = None
    routing_hint: Optional[str] = None
    project: Optional[str] = None
    region: Optional[str] = None
    profile_arn: Optional[str] = None
    email: Optional[str] = None
    expires_at: Optional[int] = None
    chatgpt_account_id: Optional[str] = None
    user_id: Optional[str] = None
    at: Optional[str] = None
    bl: Optional[str] = None
    f_sid: Optional[str] = None
    # Transport
    base_url: Optional[str] = None
    pool: Optional[str] = "custom"
    priority: Optional[int] = 100
    weight: Optional[int] = 1
    timeout_ms: Optional[int] = 120000
    impersonate: Optional[str] = None
    proxy_url: Optional[str] = None
    extra_headers: Optional[Dict[str, str]] = None


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/")
def index():
    return {
        "service": "GravityAPI Gateway",
        "version": "2.0.0",
        "providers": len(PROVIDERS),
        "free_providers": sorted(FREE_PROVIDERS),
        "endpoints": {
            "GET  /v1/models":                 "List all available models",
            "GET  /v1/models/{provider}":      "List models for a specific provider",
            "POST /v1/chat/completions":        "OpenAI-compatible chat completions",
            "POST /v1/embeddings":             "Embeddings (windsurf/cursor)",
            "POST /api/generate-key":          "Generate a new API key",
            "POST /api/add":                   "Register a provider account",
            "DELETE /api/accounts/{id}":       "Remove an account",
            "GET  /api/accounts":              "List your registered accounts",
            "GET  /api/providers":             "All supported providers + auth guide",
            "GET  /api/metrics":               "Request analytics",
            "GET  /docs":                      "Interactive documentation",
        },
    }


@app.get("/v1/models")
def list_models(token: str = Depends(verify_token)):
    now = int(time.time())
    models = []
    for pname, meta in PROVIDERS.items():
        for m in meta["models"]:
            models.append({
                "id": f"{pname}/{m}",
                "object": "model",
                "created": now,
                "owned_by": pname,
                "provider": pname,
                "category": meta["category"],
                "tool_support": meta["tool_support"],
                "requires_account": meta["requires_custom_account"],
            })
    return {"object": "list", "data": models}


@app.get("/v1/models/{provider}")
async def list_provider_models(
    provider: str,
    live: bool = Query(False, description="Discover models live from the provider's own endpoint"),
    token: str = Depends(verify_token),
):
    provider = provider.lower()
    if provider not in PROVIDERS:
        raise HTTPException(404, f"Unknown provider: {provider!r}")
    meta = PROVIDERS[provider]
    now = int(time.time())
    from llm import list_models_for_provider
    models = await list_models_for_provider(token, provider, live=live)
    return {
        "object": "list",
        "provider": provider,
        "display_name": meta["display_name"],
        "tool_support": meta["tool_support"],
        "requires_account": meta["requires_custom_account"],
        "discovery": "live" if live else "catalog",
        "data": [
            {
                "id": m["id"],
                "object": "model",
                "created": now,
                "owned_by": provider,
                "display_name": m.get("display_name"),
                "description": m.get("description"),
                "tags": m.get("tags", []),
            }
            for m in models
        ],
    }


@app.post("/v1/chat/completions")
async def chat_completions(req: ChatRequest, request: Request, token: str = Depends(verify_token)):
    """OpenAI-compatible chat completions — all 40+ providers, streaming, tools, MFC."""
    provider, model_name = resolve_model(req.model)
    full = full_model_name(provider, model_name)
    check_rate_limit(request.client.host, token, provider)
    record_request(full, token)

    messages, sys_prompt = _parse_messages(req.messages)
    system_prompt = req.system_prompt or sys_prompt
    tools = _parse_tools(req.tools)

    if req.stream:
        async def _stream():
            chat_id = f"chatcmpl-{secrets.token_hex(12)}"
            created = int(time.time())
            try:
                async for chunk in generate_stream(
                    token, req.model, messages, system_prompt,
                    tools=tools, temperature=req.temperature, max_tokens=req.max_tokens,
                ):
                    yield f"data: {json.dumps({'id': chat_id, 'object': 'chat.completion.chunk', 'created': created, 'model': req.model, 'choices': [{'index': 0, 'delta': {'content': chunk}, 'finish_reason': None}]})}\n\n"
                yield f"data: {json.dumps({'id': chat_id, 'object': 'chat.completion.chunk', 'created': created, 'model': req.model, 'choices': [{'index': 0, 'delta': {}, 'finish_reason': 'stop'}]})}\n\n"
                yield "data: [DONE]\n\n"
            except Exception as e:
                yield f"data: {json.dumps({'id': chat_id, 'object': 'chat.completion.chunk', 'created': created, 'model': req.model, 'choices': [{'index': 0, 'delta': {'content': f'[Error: {e}]'}, 'finish_reason': 'error'}]})}\n\n"
                yield "data: [DONE]\n\n"
        return StreamingResponse(_stream(), media_type="text/event-stream")

    try:
        resp = await generate_response(
            token, req.model, messages, system_prompt,
            tools=tools, temperature=req.temperature,
            max_tokens=req.max_tokens, seed=req.seed,
        )
    except Exception as e:
        raise HTTPException(500, f"Generation failed: {e}")

    # Extract content
    if isinstance(resp.message.content, str):
        content_text = resp.message.content
    elif hasattr(resp.message.content, "text"):
        content_text = resp.message.content.text
    else:
        content_text = str(resp.message.content or "")

    oai_tcs = _format_tool_calls(resp.message.tool_calls)
    finish = "tool_calls" if oai_tcs else (resp.finish_reason or "stop")

    return {
        "id": f"chatcmpl-{secrets.token_hex(12)}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": req.model,
        "choices": [{
            "index": 0,
            "message": {
                "role": "assistant",
                "content": content_text or None,
                "tool_calls": oai_tcs if oai_tcs else None,
            },
            "finish_reason": finish,
        }],
        "usage": {
            "prompt_tokens": resp.usage.input_tokens or 0,
            "completion_tokens": resp.usage.output_tokens or 0,
            "total_tokens": resp.usage.total_tokens or (
                (resp.usage.input_tokens or 0) + (resp.usage.output_tokens or 0)
            ),
        },
    }


@app.post("/v1/embeddings")
async def embeddings(payload: Dict[str, Any], token: str = Depends(verify_token)):
    """Native embeddings via windsurf or cursor (1024-dim float32)."""
    model = payload.get("model", "windsurf/swe-1-6-fast")
    input_data = payload.get("input", [])
    if isinstance(input_data, str):
        input_data = [input_data]

    provider, _ = resolve_model(model)
    check_rate_limit("127.0.0.1", token, provider)
    record_request(model, token)

    try:
        from gravity import AsyncGravityClient
        from llm import _build_accounts_file
        accounts = _build_accounts_file(token, provider)
        client = AsyncGravityClient(accounts=accounts)
        result = await client.embeddings(model=model, texts=input_data)
        return {
            "object": "list",
            "model": model,
            "data": [
                {"object": "embedding", "index": i, "embedding": emb}
                for i, emb in enumerate(result.embeddings)
            ],
            "usage": {"prompt_tokens": 0, "total_tokens": 0},
        }
    except Exception as e:
        raise HTTPException(500, f"Embeddings failed: {e}")


# ── Account management ────────────────────────────────────────────────────────

@app.post("/api/generate-key")
def generate_key_endpoint():
    key = generate_api_key()
    return {
        "api_key": key,
        "tier": "default",
        "limits": {"shared_pool": "20 req/hr", "custom_pool": "2000 req/hr"},
        "free_providers": sorted(FREE_PROVIDERS),
        "next_steps": "POST /api/add to register provider accounts, GET /api/providers for details.",
    }


@app.post("/api/add")
def add_account(req: AddAccountRequest, token: str = Depends(verify_token)):
    """Register any of the 40+ providers. See GET /api/providers for auth requirements."""
    payload = req.model_dump(exclude_none=True)
    try:
        return add_account_to_db(token, payload)
    except ValueError as e:
        raise HTTPException(400, str(e))
    except Exception as e:
        raise HTTPException(500, f"Account registration failed: {e}")


@app.delete("/api/accounts/{account_id}")
def delete_account(
    account_id: str,
    provider: str = Query(..., description="Provider name, e.g. groq"),
    token: str = Depends(verify_token),
):
    result = remove_account_from_db(token, account_id, provider)
    if result["status"] == "not_found":
        raise HTTPException(404, "Account not found.")
    return result


@app.get("/api/accounts")
def get_accounts(token: str = Depends(verify_token)):
    return {
        "accounts": list_accounts_for_key(token),
        "registered_providers": sorted(get_providers_for_key(token)),
    }


@app.get("/api/providers")
def get_providers():
    """Full provider catalog with auth requirements."""
    result = []
    for name, meta in PROVIDERS.items():
        result.append({
            "id": name,
            "display_name": meta["display_name"],
            "category": meta["category"],
            "description": meta["description"],
            "auth_kinds": meta["auth_kinds"],
            "tool_support": meta["tool_support"],
            "structured_output": meta["structured_output"],
            "supports_system_prompt": meta["supports_system_prompt"],
            "requires_account": meta["requires_custom_account"],
            "free": not meta["requires_custom_account"],
            "default_base_url": meta["default_base_url"] or None,
            "models": meta["models"],
        })
    return {
        "total": len(result),
        "free_providers": sorted(FREE_PROVIDERS),
        "providers": result,
    }


@app.get("/api/metrics")
def metrics(token: str = Depends(verify_token)):
    return get_metrics()


# ── Admin ─────────────────────────────────────────────────────────────────────

@app.post("/admin/generate-key")
def admin_generate_key(payload: Dict[str, Any], token: str = Depends(verify_admin)):
    tier = payload.get("tier", "default")
    label = payload.get("label", "")
    key = generate_api_key(label=label, tier=tier)
    return {"api_key": key, "tier": tier, "label": label}


@app.post("/admin/revoke-key")
def admin_revoke_key(payload: Dict[str, Any], token: str = Depends(verify_admin)):
    key = payload.get("api_key", "")
    if not key:
        raise HTTPException(400, "Missing api_key")
    return {"status": "revoked" if revoke_api_key(key) else "not_found"}


# ── Docs ──────────────────────────────────────────────────────────────────────

@app.get("/docs", response_class=HTMLResponse)
def documentation():
    try:
        with open("./docs/api_guide.md", encoding="utf-8") as f:
            md = f.read()
    except Exception:
        md = ""

    provider_rows = "\n".join(
        f"| `{n}` | {m['display_name']} | {m['category']} | "
        f"{'✓ free' if not m['requires_custom_account'] else 'own key'} | "
        f"{m['tool_support']} |"
        for n, m in PROVIDERS.items()
    )
    providers_section = f"""
## Supported Providers ({len(PROVIDERS)} total, {len(FREE_PROVIDERS)} free)

| ID | Name | Category | Access | Tools |
|----|------|----------|--------|-------|
{provider_rows}

**Free providers** (no account needed): {', '.join(f'`{p}`' for p in sorted(FREE_PROVIDERS))}

### Adding an account

```python
import requests
r = requests.post("https://your-server/api/add",
    headers={{"Authorization": "Bearer grav_your_key"}},
    json={{
        "account_id": "my_groq",
        "provider": "groq",
        "api_key": "gsk_...",
    }})
```

### Chat example

```python
import requests
r = requests.post("https://your-server/v1/chat/completions",
    headers={{"Authorization": "Bearer grav_your_key"}},
    json={{
        "model": "groq/llama-3.3-70b-versatile",
        "messages": [{{"role": "user", "content": "Hello!"}}],
    }})
print(r.json()["choices"][0]["message"]["content"])
```
"""
    full_md = providers_section + "\n\n" + md

    return HTMLResponse(f"""<!DOCTYPE html><html lang="en"><head>
<meta charset="UTF-8"><title>GravityAPI v2 Docs</title>
<link href="https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;700&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet">
<script src="https://cdn.jsdelivr.net/npm/marked/marked.min.js"></script>
<link href="https://cdnjs.cloudflare.com/ajax/libs/prism/1.29.0/themes/prism-tomorrow.min.css" rel="stylesheet"/>
<script src="https://cdnjs.cloudflare.com/ajax/libs/prism/1.29.0/prism.min.js"></script>
<script src="https://cdnjs.cloudflare.com/ajax/libs/prism/1.29.0/components/prism-python.min.js"></script>
<style>:root{{--bg:#080911;--panel:rgba(17,19,36,.75);--border:rgba(255,255,255,.08);--purple:#8b5cf6;--cyan:#06b6d4;--text:#f3f4f6;--muted:#9ca3af}}
*{{box-sizing:border-box;margin:0;padding:0}}body{{font-family:'Outfit',sans-serif;background:var(--bg);color:var(--text);min-height:100vh;background-image:radial-gradient(at 0% 0%,rgba(139,92,246,.12) 0,transparent 50%),radial-gradient(at 100% 100%,rgba(6,182,212,.12) 0,transparent 50%)}}
header{{padding:1.5rem 2rem;display:flex;justify-content:space-between;align-items:center;border-bottom:1px solid var(--border);backdrop-filter:blur(12px);position:sticky;top:0;z-index:100;background:rgba(8,9,17,.8)}}
.logo{{font-weight:700;font-size:1.8rem;background:linear-gradient(135deg,#8b5cf6,#06b6d4);-webkit-background-clip:text;-webkit-text-fill-color:transparent}}
.container{{max-width:1100px;margin:0 auto;padding:2rem}}.panel{{background:var(--panel);border:1px solid var(--border);border-radius:16px;padding:2.5rem;backdrop-filter:blur(16px)}}
#doc h1,#doc h2,#doc h3{{font-weight:700;margin-top:2rem;margin-bottom:1rem;color:#fff;border-bottom:1px solid rgba(255,255,255,.05);padding-bottom:.5rem}}
#doc h1{{font-size:2rem}}#doc h2{{font-size:1.5rem}}#doc h3{{font-size:1.2rem}}#doc p{{line-height:1.7;color:#d1d5db;margin-bottom:1.25rem;font-size:.95rem}}
#doc ul,#doc ol{{margin-bottom:1.5rem;padding-left:1.5rem;color:#d1d5db;font-size:.95rem}}
#doc table{{width:100%;border-collapse:collapse;margin:1.5rem 0;font-size:.83rem}}
#doc th,#doc td{{padding:.55rem .75rem;border-bottom:1px solid rgba(255,255,255,.06);text-align:left}}
#doc th{{color:var(--muted);font-weight:600;background:rgba(255,255,255,.02)}}
#doc code{{font-family:'JetBrains Mono',monospace;background:rgba(255,255,255,.06);padding:.2rem .4rem;border-radius:4px;color:#67e8f9;font-size:.85em}}
#doc pre{{background:#05060b!important;border:1px solid var(--border);border-radius:8px;padding:1.25rem;margin:1.5rem 0;overflow-x:auto}}
#doc pre code{{background:transparent;padding:0;color:inherit}}</style></head><body>
<header><div><div class="logo">🌌 GravityAPI</div><div style="color:var(--muted);font-size:.85rem">v2.0 · {len(PROVIDERS)} providers</div></div>
<a href="/api/providers" style="color:var(--cyan);text-decoration:none;font-size:.85rem">providers JSON →</a></header>
<div class="container"><div class="panel"><div id="doc"></div></div></div>
<textarea id="md" style="display:none">{full_md.replace(chr(96), "&#96;")}</textarea>
<script>
document.getElementById("doc").innerHTML=marked.parse(document.getElementById("md").value.replace(/&#96;/g,"`"));
document.querySelectorAll("#doc pre code").forEach(b=>Prism.highlightElement(b));
</script></body></html>""")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=6969, reload=True)
