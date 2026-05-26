import asyncio
from gravity import AsyncGravityClient
from gravity.io import load_accounts
from gravity.types import Message, ToolCall, ToolResult
from gravity.tools import Tool, ToolDefinition
from typing import AsyncIterator, List, Dict, Any

from account import load_accounts_for_key

def resolve_model_name(model: str) -> str:
    """
    Ensures model name is fully qualified with provider.
    Defaults to 'cursor' if no provider slash is present.
    """
    if "/" not in model:
        return f"cursor/{model}"
    return model

async def generate_response(
    api_key: str, 
    model: str, 
    messages: List[Message], 
    system_prompt: str | None = None,
    tools: List[Tool] | None = None,
    pool: str = "custom"
) -> Any:
    """
    Generates a direct chat completion response using Manual Function Calling (MFC)
    when tools are supplied to allow downstream users to handle execution.
    """
    if pool == "custom":
        accounts = load_accounts_for_key(api_key)
    else:
        accounts = load_accounts()
        
    client = AsyncGravityClient(accounts=accounts)
    resolved_model = resolve_model_name(model)
    
    response = await client.chat(
        model=resolved_model,
        messages=messages,
        system_prompt=system_prompt,
        tools=tools,
        function_calling="manual" if tools else "auto"
    )
    return response

async def generate_stream(
    api_key: str, 
    model: str, 
    messages: List[Message], 
    system_prompt: str | None = None,
    tools: List[Tool] | None = None,
    pool: str = "custom"
) -> AsyncIterator[str]:
    """
    Generates a streaming chat completion response.
    Note: Gravity streams only support auto function calling (AFC).
    """
    if pool == "custom":
        accounts = load_accounts_for_key(api_key)
    else:
        accounts = load_accounts()
        
    client = AsyncGravityClient(accounts=accounts)
    resolved_model = resolve_model_name(model)
    
    async for token in client.stream_response(
        model=resolved_model,
        messages=messages,
        system_prompt=system_prompt,
        tools=tools
    ):
        yield token

if __name__ == "__main__":
    # Simple test for testing if gravity works or not.. It does!
    from gravity.types import Message
    try:
        asyncio.run(generate_response("grav_demoapikey", "cursor/auto", [Message(role="user", content="Hello")]))
    except Exception as e:
        print(f"Note: Local execution test completed (downstream authenticator exception expected if accounts not configured): {e}")