import fastapi as meow
from fastapi import Header, HTTPException, Depends, Request
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import json
import time
import secrets
from genapi import generate_api_key, validate_api_key
from llm import generate_response, generate_stream
from account import add_account_to_db, record_request, get_metrics, get_db, load_accounts_for_key

app = meow.FastAPI(
    title="RASPAPI - GravityV2 Gateway",
    description="Unified API gateway powered by GravityV2 for Cursor, Windsurf, ChatGPT, and Gemini.",
    version="1.0.0",
    docs_url=None
)

# Enable CORS for maximum flexibility
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

security = HTTPBearer(auto_error=False)

async def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """Verifies the bearer API token using the SQLite store."""
    if not credentials:
        raise HTTPException(status_code=401, detail="Missing API Key. Provide it in 'Authorization: Bearer grav_...' header.")
    token = credentials.credentials
    if not validate_api_key(token):
        raise HTTPException(status_code=401, detail="Invalid, inactive, or revoked API Key.")
    return token

class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[Dict[str, Any]]
    stream: Optional[bool] = False
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None
    system_prompt: Optional[str] = None
    tools: Optional[List[Dict[str, Any]]] = None
    tool_choice: Optional[Any] = None

@app.post("/v1/chat/completions")
async def chat_completions(req: ChatCompletionRequest, request: Request, token: str = Depends(verify_token)):
    """
    OpenAI-compatible chat completions endpoint.
    Routes requests dynamically to either Custom Pool or Default Pool with IP-based hourly rate limits.
    Supports complete message histories and function tools schemas.
    """
    model_name = req.model
    record_request(model_name)
    
    # 1. Parse provider from model name
    from llm import resolve_model_name
    resolved_model = resolve_model_name(model_name)
    provider = resolved_model.split("/")[0]

    # 2. Resolve pool: custom if the user has their own active credential for this provider, else default
    custom_accounts = load_accounts_for_key(token)
    has_custom = False
    if custom_accounts and custom_accounts.emails:
        for email_bundle in custom_accounts.emails:
            for acc in email_bundle.providers:
                if acc.provider.value == provider and acc.enabled:
                    has_custom = True
                    break
            if has_custom:
                break

    pool = "custom" if has_custom else "default"

    # 3. IP-Based Hourly Rate Limiting
    ip = request.client.host
    current_hour = int(time.time() / 3600)
    db_conn = get_db()
    
    ratelimit_key = f"ratelimit:ip:{ip}:hour:{current_hour}".encode("utf-8")
    current_count_bytes = db_conn.get(ratelimit_key)
    current_count = int(current_count_bytes.decode("utf-8")) if current_count_bytes else 0
    
    limit = 2000 if pool == "custom" else 20
    if current_count >= limit:
        raise HTTPException(
            status_code=429,
            detail=f"IP-based hourly rate limit exceeded for IP {ip} in the default pool ({limit} req/hr)." if pool == "default"
            else f"IP-based hourly rate limit exceeded for IP {ip} in the custom pool ({limit} req/hr)."
        )
        
    db_conn.put(ratelimit_key, str(current_count + 1).encode("utf-8"))

    # 4. Enforce Lifetime Default Allowances for Default Pool
    if pool == "default":
        # Enforce that only windsurf and cursor can fallback to default pool
        if provider not in ["windsurf", "cursor"]:
            raise HTTPException(
                status_code=400,
                detail=f"Provider '{provider}' is not supported in the default pool. Register your own account via /api/add."
            )
            
        # Enforce default models
        if provider == "windsurf" and resolved_model != "windsurf/swe-1-6-slow":
            raise HTTPException(
                status_code=400,
                detail="Only model 'windsurf/swe-1-6-slow' is available in the default pool."
            )
        if provider == "cursor" and resolved_model != "cursor/auto":
            raise HTTPException(
                status_code=400,
                detail="Only model 'cursor/auto' is available in the default pool."
            )
            
        # Lifetime default allowance (20 requests per provider per API key, tracked by IP for grav_demoapikey)
        if token == "grav_demoapikey":
            allowance_key = f"allowance:default:demo:{ip}:{provider}".encode("utf-8")
        else:
            allowance_key = f"allowance:default:{token}:{provider}".encode("utf-8")
            
        used_bytes = db_conn.get(allowance_key)
        used_count = int(used_bytes.decode("utf-8")) if used_bytes else 0
        
        if used_count >= 20:
            raise HTTPException(
                status_code=403,
                detail=f"Lifetime default allowance of 20 requests exceeded for provider '{provider}'. Please link your own provider account to continue."
            )
            
        db_conn.put(allowance_key, str(used_count + 1).encode("utf-8"))

    # 5. Map OpenAI History to Gravity Messages
    from gravity.types import Message, ToolCall, ToolResult
    from gravity.tools import Tool, ToolDefinition

    gravity_messages = []
    system_prompt = req.system_prompt

    for m in req.messages:
        role = m.get("role")
        content = m.get("content")
        
        if role == "system":
            # Set system prompt if not already set, otherwise append as message
            if not system_prompt:
                system_prompt = content
            gravity_messages.append(Message.system(content))
            
        elif role == "user":
            gravity_messages.append(Message.user(content))
            
        elif role == "assistant":
            tool_calls_data = m.get("tool_calls")
            tool_calls = []
            if tool_calls_data:
                for tc in tool_calls_data:
                    func = tc.get("function", {})
                    args_str = func.get("arguments", "{}")
                    try:
                        args = json.loads(args_str)
                    except Exception:
                        args = {}
                    tool_calls.append(
                        ToolCall(
                            id=tc.get("id") or f"call_{secrets.token_hex(4)}",
                            name=func.get("name"),
                            arguments=args
                        )
                    )
            gravity_messages.append(
                Message(
                    role="assistant",
                    content=content or "",
                    tool_calls=tool_calls
                )
            )
            
        elif role == "tool":
            tool_call_id = m.get("tool_call_id")
            tool_content = m.get("content", "")
            try:
                content_val = json.loads(tool_content)
            except Exception:
                content_val = tool_content
                
            tool_name = m.get("name", "tool")
            
            result_obj = ToolResult(
                tool_call_id=tool_call_id,
                name=tool_name,
                content=content_val
            )
            gravity_messages.append(Message.tool_result_message(result_obj))

    # 6. Map OpenAI Tools to Gravity Tools
    gravity_tools = None
    if req.tools:
        gravity_tools = []
        for t in req.tools:
            if t.get("type") == "function":
                func = t.get("function", {})
                name = func.get("name")
                desc = func.get("description")
                params = func.get("parameters", {"type": "object", "properties": {}})
                gravity_tools.append(
                    Tool(
                        definition=ToolDefinition(
                            name=name,
                            description=desc,
                            input_schema=params
                        )
                    )
                )

    if req.stream:
        async def stream_generator():
            chat_id = f"chatcmpl-{secrets.token_hex(12)}"
            created_time = int(time.time())
            try:
                async for token_text in generate_stream(
                    token, 
                    model_name, 
                    gravity_messages, 
                    system_prompt, 
                    tools=gravity_tools, 
                    pool=pool
                ):
                    chunk = {
                        "id": chat_id,
                        "object": "chat.completion.chunk",
                        "created": created_time,
                        "model": model_name,
                        "choices": [
                            {
                                "index": 0,
                                "delta": {"content": token_text},
                                "finish_reason": None
                            }
                        ]
                    }
                    yield f"data: {json.dumps(chunk)}\n\n"
                    
                # Signal stop
                finish_chunk = {
                    "id": chat_id,
                    "object": "chat.completion.chunk",
                    "created": created_time,
                    "model": model_name,
                    "choices": [
                        {
                            "index": 0,
                            "delta": {},
                            "finish_reason": "stop"
                        }
                    ]
                }
                yield f"data: {json.dumps(finish_chunk)}\n\n"
                yield "data: [DONE]\n\n"
            except Exception as e:
                error_chunk = {
                    "id": chat_id,
                    "object": "chat.completion.chunk",
                    "created": created_time,
                    "model": model_name,
                    "choices": [
                        {
                            "index": 0,
                            "delta": {"content": f"\n[Gravity Gate Error: {str(e)}]"},
                            "finish_reason": "error"
                        }
                    ]
                }
                yield f"data: {json.dumps(error_chunk)}\n\n"
                yield "data: [DONE]\n\n"

        return StreamingResponse(stream_generator(), media_type="text/event-stream")
    
    else:
        try:
            resp = await generate_response(
                token, 
                model_name, 
                gravity_messages, 
                system_prompt, 
                tools=gravity_tools, 
                pool=pool
            )
            content_text = resp.message.content if isinstance(resp.message.content, str) else str(resp.message.content)
            
            # Format tool calls back to OpenAI schema
            openai_tool_calls = []
            if resp.message.tool_calls:
                for tc in resp.message.tool_calls:
                    openai_tool_calls.append({
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.name,
                            "arguments": json.dumps(tc.arguments)
                        }
                    })
                    
            return {
                "id": f"chatcmpl-{secrets.token_hex(12)}",
                "object": "chat.completion",
                "created": int(time.time()),
                "model": model_name,
                "choices": [
                    {
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": content_text or None,
                            "tool_calls": openai_tool_calls if openai_tool_calls else None
                        },
                        "finish_reason": "tool_calls" if openai_tool_calls else (resp.finish_reason or "stop")
                    }
                ],
                "usage": {
                    "prompt_tokens": resp.usage.input_tokens or 0,
                    "completion_tokens": resp.usage.output_tokens or 0,
                    "total_tokens": resp.usage.total_tokens or resp.usage.total
                }
            }
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Gravity execution failed: {str(e)}")

@app.post("/api/generate-key")
def generate_key_api() -> dict:
    """JSON API to generate and store a new active API key."""
    apikey = generate_api_key()
    return {"apikey": f"grav_{apikey}", "status": "active", "pool": "default"}
@app.get("/")
def index() -> dict:
    return {
        "message": "Welcome to GravityAPI Gateway!",
        "documentation_url": "/docs",
        "endpoints": {
            "GET /": "Service status and endpoint directory",
            "GET /docs": "Interactive documentation viewer & Python integration templates",
            "GET /api/metrics": "Telemetry health metrics and request volume analytics",
            "POST /v1/chat/completions": "OpenAI-compatible chat completions proxy",
            "POST /api/generate-key": "Generate and persist a new active Bearer token",
            "POST /api/add": "Link custom Cursor/Windsurf provider credentials to elevate limits"
        }
    }
    
@app.post("/api/add")
def add_account(payload: Dict[str, Any], token: str = Depends(verify_token)):
    try:
        result = add_account_to_db(token, payload)
        return result
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/api/metrics")
def metrics():
    return get_metrics()

@app.get("/docs", response_class=HTMLResponse)
def documentation_ui():
    try:
        with open("./docs/api_guide.md", "r", encoding="utf-8") as f:
            md_content = f.read()
    except Exception:
        md_content = "# Gravity AI API documentation\n*(api_guide.md not found)*"

    html_content = r"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Gravity API Gateway Explorer</title>
    <link href="https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet">
    <script src="https://cdn.jsdelivr.net/npm/marked/marked.min.js"></script>
    <link href="https://cdnjs.cloudflare.com/ajax/libs/prism/1.29.0/themes/prism-tomorrow.min.css" rel="stylesheet" />
    <script src="https://cdnjs.cloudflare.com/ajax/libs/prism/1.29.0/prism.min.js"></script>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/prism/1.29.0/components/prism-python.min.js"></script>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/prism/1.29.0/components/prism-json.min.js"></script>
    <style>
        :root {
            --bg-color: #080911;
            --panel-bg: rgba(17, 19, 36, 0.75);
            --border-color: rgba(255, 255, 255, 0.08);
            --accent-purple: #8b5cf6;
            --accent-cyan: #06b6d4;
            --accent-gradient: linear-gradient(135deg, #8b5cf6, #06b6d4);
            --text-primary: #f3f4f6;
            --text-secondary: #9ca3af;
        }

        * {
            box-sizing: border-box;
            margin: 0;
            padding: 0;
        }

        body {
            font-family: 'Outfit', sans-serif;
            background-color: var(--bg-color);
            color: var(--text-primary);
            min-height: 100vh;
            background-image: 
                radial-gradient(at 0% 0%, rgba(139, 92, 246, 0.12) 0px, transparent 50%),
                radial-gradient(at 100% 100%, rgba(6, 182, 212, 0.12) 0px, transparent 50%);
            overflow-x: hidden;
            display: flex;
            flex-direction: column;
        }

        header {
            padding: 1.5rem 2rem;
            display: flex;
            justify-content: space-between;
            align-items: center;
            border-bottom: 1px solid var(--border-color);
            backdrop-filter: blur(12px);
            position: sticky;
            top: 0;
            z-index: 100;
            background: rgba(8, 9, 17, 0.8);
        }

        .logo-section h1 {
            font-weight: 700;
            font-size: 1.8rem;
            background: var(--accent-gradient);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            display: flex;
            align-items: center;
            gap: 0.75rem;
        }

        .logo-section p {
            color: var(--text-secondary);
            font-size: 0.85rem;
            margin-top: 0.2rem;
        }

        .container {
            max-width: 1600px;
            width: 100%;
            margin: 0 auto;
            padding: 2rem;
            display: grid;
            grid-template-columns: 1fr 500px;
            gap: 2rem;
            flex: 1;
        }

        @media (max-width: 1200px) {
            .container {
                grid-template-columns: 1fr;
            }
        }

        .panel {
            background: var(--panel-bg);
            border: 1px solid var(--border-color);
            border-radius: 16px;
            padding: 2.5rem;
            backdrop-filter: blur(16px);
            box-shadow: 0 10px 30px rgba(0, 0, 0, 0.25);
            overflow-y: auto;
        }

        #doc-content h1, #doc-content h2, #doc-content h3 {
            font-weight: 700;
            margin-top: 2rem;
            margin-bottom: 1rem;
            color: #fff;
            border-bottom: 1px solid rgba(255, 255, 255, 0.05);
            padding-bottom: 0.5rem;
        }
        
        #doc-content h1 { font-size: 2rem; }
        #doc-content h2 { font-size: 1.5rem; }
        #doc-content h3 { font-size: 1.2rem; }

        #doc-content p {
            line-height: 1.7;
            color: #d1d5db;
            margin-bottom: 1.25rem;
            font-size: 0.95rem;
        }

        #doc-content ul, #doc-content ol {
            margin-bottom: 1.5rem;
            padding-left: 1.5rem;
            color: #d1d5db;
            font-size: 0.95rem;
        }

        #doc-content li {
            margin-bottom: 0.5rem;
            line-height: 1.6;
        }

        #doc-content table {
            width: 100%;
            border-collapse: collapse;
            margin: 1.5rem 0;
            font-size: 0.9rem;
        }

        #doc-content th, #doc-content td {
            padding: 0.75rem 1rem;
            border-bottom: 1px solid rgba(255, 255, 255, 0.06);
            text-align: left;
        }

        #doc-content th {
            color: var(--text-secondary);
            font-weight: 600;
            background: rgba(255, 255, 255, 0.02);
        }

        #doc-content td code {
            font-family: 'JetBrains Mono', monospace;
            background: rgba(255, 255, 255, 0.06);
            padding: 0.2rem 0.4rem;
            border-radius: 4px;
            color: #67e8f9;
        }

        #doc-content pre {
            background: #05060b !important;
            border: 1px solid var(--border-color);
            border-radius: 8px;
            padding: 1.25rem;
            margin: 1.5rem 0;
            overflow-x: auto;
        }

        .right-panel {
            display: flex;
            flex-direction: column;
            gap: 1.5rem;
            position: sticky;
            top: 6.5rem;
            height: calc(100vh - 9rem);
        }

        .examples-title {
            font-size: 1.2rem;
            font-weight: 600;
            display: flex;
            align-items: center;
            gap: 0.5rem;
            border-bottom: 1px solid rgba(255, 255, 255, 0.06);
            padding-bottom: 0.75rem;
            margin-bottom: 1rem;
        }

        .tab-buttons {
            display: flex;
            gap: 0.5rem;
            background: rgba(255, 255, 255, 0.02);
            padding: 0.4rem;
            border-radius: 10px;
            border: 1px solid var(--border-color);
        }

        .tab-btn {
            flex: 1;
            padding: 0.6rem;
            border-radius: 6px;
            background: transparent;
            color: var(--text-secondary);
            border: none;
            cursor: pointer;
            font-weight: 500;
            font-size: 0.85rem;
            transition: all 0.2s ease;
        }

        .tab-btn.active {
            background: var(--accent-purple);
            color: #fff;
            box-shadow: 0 4px 10px rgba(139, 92, 246, 0.35);
        }

        .code-container {
            flex: 1;
            position: relative;
            background: #05060b;
            border: 1px solid var(--border-color);
            border-radius: 12px;
            overflow: hidden;
            display: flex;
            flex-direction: column;
        }

        .code-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 0.75rem 1rem;
            background: rgba(255, 255, 255, 0.03);
            border-bottom: 1px solid var(--border-color);
            font-size: 0.8rem;
            color: var(--text-secondary);
        }

        .code-body {
            flex: 1;
            overflow: auto;
            margin: 0;
            padding: 1rem;
        }

        .code-body pre {
            margin: 0 !important;
            background: transparent !important;
            padding: 0 !important;
        }

        .copy-btn {
            background: rgba(255, 255, 255, 0.05);
            border: 1px solid rgba(255, 255, 255, 0.1);
            color: var(--text-primary);
            padding: 0.3rem 0.7rem;
            border-radius: 4px;
            font-size: 0.75rem;
            cursor: pointer;
            transition: all 0.2s ease;
        }

        .copy-btn:hover {
            background: rgba(255, 255, 255, 0.1);
            border-color: rgba(255, 255, 255, 0.2);
        }
    </style>
</head>
<body>
    <header>
        <div class="logo-section">
            <h1>🌌 Gravity Gateway</h1>
            <p>API Integration Center</p>
        </div>
        <div>
            <button class="copy-btn" onclick="location.href='/api/metrics'">📊 Telemetry Metrics</button>
        </div>
    </header>

    <div class="container">
        <div class="panel">
            <div id="doc-content"></div>
        </div>

        <div class="panel right-panel">
            <div class="examples-title">🐍 Python Examples</div>
            <div class="tab-buttons">
                <button class="tab-btn active" onclick="switchExample('completions')">Completions</button>
                <button class="tab-btn" onclick="switchExample('tools')">Tool Calling</button>
                <button class="tab-btn" onclick="switchExample('register')">Register Account</button>
            </div>

            <div class="code-container">
                <div class="code-header">
                    <span id="example-filename">example_completions.py</span>
                    <button class="copy-btn" onclick="copyExampleCode()">📋 Copy Code</button>
                </div>
                <div class="code-body">
                    <pre><code id="example-code" class="language-python"># Select a tab above to load example code...</code></pre>
                </div>
            </div>
        </div>
    </div>

    <textarea id="raw-markdown" style="display: none;">{{MD_CONTENT}}</textarea>

    <script>
        const examples = {
            completions: `import urllib.request
import json

url = "https://kaededev.hackclub.app/v1/chat/completions"
headers = {
    "Authorization": "Bearer grav_your_api_key",
    "Content-Type": "application/json"
}
payload = {
    "model": "cursor/auto",
    "messages": [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Write a small haiku abt coding."}
    ]
}

req = urllib.request.Request(
    url, 
    data=json.dumps(payload).encode("utf-8"), 
    headers=headers, 
    method="POST"
)

with urllib.request.urlopen(req) as response:
    result = json.loads(response.read().decode("utf-8"))
    print("🤖 Response:\\n", result["choices"][0]["message"]["content"])`,

            tools: `import urllib.request
import json

url = "https://kaededev.hackclub.app/v1/chat/completions"
headers = {
    "Authorization": "Bearer grav_your_api_key",
    "Content-Type": "application/json"
}

# 1. Ask the model with dynamic tool definitions
messages = [{"role": "user", "content": "Fetch the weather for New York."}]
tools = [{
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get current weather in location",
        "parameters": {
            "type": "object",
            "properties": {"location": {"type": "string"}},
            "required": ["location"]
        }
    }
}]

payload = {"model": "cursor/auto", "messages": messages, "tools": tools}
req = urllib.request.Request(
    url, 
    data=json.dumps(payload).encode("utf-8"), 
    headers=headers, 
    method="POST"
)

with urllib.request.urlopen(req) as r:
    res = json.loads(r.read().decode("utf-8"))

choice = res["choices"][0]
assistant_msg = choice["message"]

# 2. Handle Manual Function Calling (MFC) on the client side
if choice["finish_reason"] == "tool_calls":
    tool_call = assistant_msg["tool_calls"][0]
    tool_call_id = tool_call["id"]
    tool_name = tool_call["function"]["name"]
    arguments = json.loads(tool_call["function"]["arguments"])
    
    print(f"🏃 Executing tool '{tool_name}' for '{arguments['location']}'...")
    # Simulate execution locally
    tool_result = {"temperature": "72°F", "condition": "Partly Cloudy"}
    
    # Update conversation history
    messages.append(assistant_msg)
    messages.append({
        "role": "tool",
        "tool_call_id": tool_call_id,
        "name": tool_name,
        "content": json.dumps(tool_result)
    })
    
    # Send tool result back to gateway
    payload = {"model": "cursor/auto", "messages": messages, "tools": tools}
    req2 = urllib.request.Request(
        url, 
        data=json.dumps(payload).encode("utf-8"), 
        headers=headers, 
        method="POST"
    )
    with urllib.request.urlopen(req2) as r2:
        final_res = json.loads(r2.read().decode("utf-8"))
        print("🤖 Final Response:\\n", final_res["choices"][0]["message"]["content"])`,

            register: `import urllib.request
import json

url = "https://kaededev.hackclub.app/api/add"
headers = {
    "Authorization": "Bearer grav_your_api_key",
    "Content-Type": "application/json"
}

# Payload to link Cursor/Windsurf credentials and elevate limits (2000 req/hr)
payload = {
    "account_id": "personal_account_1",
    "provider": "cursor",
    "api_key": "your_raw_cursor_refresh_token_here",
    "pool": "default",
    "priority": 100,
    "weight": 1,
    "transport": {
        "timeout_ms": 120000
    }
}

req = urllib.request.Request(
    url, 
    data=json.dumps(payload).encode("utf-8"), 
    headers=headers, 
    method="POST"
)

with urllib.request.urlopen(req) as response:
    result = json.loads(response.read().decode("utf-8"))
    print("📋 Registration Response:", result)`
        };

        const mdText = document.getElementById('raw-markdown').value;
        document.getElementById('doc-content').innerHTML = marked.parse(mdText);

        function switchExample(key) {
            document.querySelectorAll('.tab-btn').forEach(btn => btn.classList.remove('active'));
            
            const btn = Array.from(document.querySelectorAll('.tab-btn')).find(b => b.getAttribute('onclick').includes(key));
            if (btn) btn.classList.add('active');

            const codeElem = document.getElementById('example-code');
            codeElem.textContent = examples[key];
            
            document.getElementById('example-filename').textContent = `example_${key === 'completions' ? 'completions' : key === 'tools' ? 'tool_calling' : 'register_account'}.py`;
            
            Prism.highlightElement(codeElem);
        }

        function copyExampleCode() {
            const text = document.getElementById('example-code').textContent;
            navigator.clipboard.writeText(text);
            alert("Example code copied to clipboard!");
        }

        switchExample('completions');
        
        document.querySelectorAll('#doc-content pre code').forEach(block => {
            Prism.highlightElement(block);
        });
    </script>
</body>
</html>"""
    return HTMLResponse(content=html_content.replace("{{MD_CONTENT}}", md_content))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=6969, reload=True)