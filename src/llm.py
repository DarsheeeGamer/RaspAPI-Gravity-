"""LLM routing layer — model resolution, account selection, generation."""

import asyncio
from typing import AsyncIterator, List, Optional, Any

from gravity import AsyncGravityClient
from gravity.io import load_accounts
from gravity.types import Message
from gravity.tools import Tool

from account import load_accounts_for_key, get_providers_for_key
from providers import PROVIDERS, FREE_PROVIDERS, MODEL_TO_PROVIDER


# ── Model resolution ──────────────────────────────────────────────────────────

def resolve_model(model: str) -> tuple[str, str]:
    """Return (provider, model_name) from a model string.

    Accepts:
      "provider/model"      → (provider, model)
      "provider/"           → (provider, provider's first model or "auto")
      "model"               → infer provider from MODEL_TO_PROVIDER catalog
      "provider"            → (provider, "auto")
    """
    if "/" in model:
        provider, _, model_name = model.partition("/")
        return provider.lower(), model_name or "auto"

    # Bare model name — look up in catalog
    lower = model.lower()
    if lower in MODEL_TO_PROVIDER:
        return MODEL_TO_PROVIDER[lower], lower

    # Bare provider name
    if lower in PROVIDERS:
        first = PROVIDERS[lower]["models"]
        return lower, (first[0] if first else "auto")

    # Last resort: treat as cursor/model (backwards compat)
    return "cursor", model


def full_model_name(provider: str, model: str) -> str:
    return f"{provider}/{model}"


# ── Account selection ─────────────────────────────────────────────────────────

def _build_accounts_file(api_key: str, provider: str):
    """Build an AccountsFile prioritizing custom accounts, falling back to defaults."""
    from gravity.accounts import AccountsFile, EmailBundle

    user_accounts = load_accounts_for_key(api_key)
    user_providers = get_providers_for_key(api_key)

    if provider in user_providers:
        # User has their own account — use only that
        providers_list = [
            acc for bundle in user_accounts.emails
            for acc in bundle.providers
            if acc.provider.value == provider and acc.enabled
        ]
        if providers_list:
            return AccountsFile(
                schema_version=4,
                emails=[EmailBundle(
                    email="user@gravity.local",
                    enabled=True,
                    display_name="Custom",
                    providers=providers_list,
                )],
            )

    # Fall back to default server accounts (from ~/.gravity/accounts.json)
    try:
        defaults = load_accounts()
        default_providers = [
            acc for bundle in defaults.emails
            for acc in bundle.providers
            if acc.provider.value == provider and acc.enabled
        ]
        if default_providers:
            return AccountsFile(
                schema_version=4,
                emails=[EmailBundle(
                    email="server@gravity.local",
                    enabled=True,
                    display_name="Server Default",
                    providers=default_providers,
                )],
            )
    except Exception:
        pass

    # No accounts at all — return user's full file and let Gravity error
    return user_accounts


def _is_custom_pool(api_key: str, provider: str) -> bool:
    return provider in get_providers_for_key(api_key)


# ── Generation ────────────────────────────────────────────────────────────────

async def generate_response(
    api_key: str,
    model: str,
    messages: List[Message],
    system_prompt: Optional[str] = None,
    tools: Optional[List[Tool]] = None,
    temperature: Optional[float] = None,
    max_tokens: Optional[int] = None,
    seed: Optional[int] = None,
) -> Any:
    """Non-streaming chat completion."""
    provider, model_name = resolve_model(model)
    full_model = full_model_name(provider, model_name)
    accounts = _build_accounts_file(api_key, provider)
    client = AsyncGravityClient(accounts=accounts)

    metadata: dict[str, Any] = {}
    if temperature is not None:
        metadata["temperature"] = temperature
    if max_tokens is not None:
        metadata["max_tokens"] = max_tokens
    if seed is not None:
        metadata["seed"] = seed

    return await client.chat(
        model=full_model,
        messages=messages,
        system_prompt=system_prompt,
        tools=tools or [],
        function_calling="manual" if tools else "auto",
        metadata=metadata or None,
    )


async def generate_stream(
    api_key: str,
    model: str,
    messages: List[Message],
    system_prompt: Optional[str] = None,
    tools: Optional[List[Tool]] = None,
    temperature: Optional[float] = None,
    max_tokens: Optional[int] = None,
) -> AsyncIterator[str]:
    """Streaming chat completion — yields text tokens."""
    provider, model_name = resolve_model(model)
    full_model = full_model_name(provider, model_name)
    accounts = _build_accounts_file(api_key, provider)
    client = AsyncGravityClient(accounts=accounts)

    metadata: dict[str, Any] = {}
    if temperature is not None:
        metadata["temperature"] = temperature
    if max_tokens is not None:
        metadata["max_tokens"] = max_tokens

    async for token in client.stream_response(
        model=full_model,
        messages=messages,
        system_prompt=system_prompt,
        tools=tools or [],
        metadata=metadata or None,
    ):
        yield token


async def list_models_for_provider(api_key: str, provider: str) -> List[str]:
    """Try to fetch dynamic model list from a provider; fall back to catalog."""
    catalog_models = PROVIDERS.get(provider, {}).get("models", [])
    try:
        accounts = _build_accounts_file(api_key, provider)
        client = AsyncGravityClient(accounts=accounts)
        dynamic = await asyncio.wait_for(
            client.list_models(provider),
            timeout=5.0,
        )
        if dynamic:
            return dynamic
    except Exception:
        pass
    return catalog_models
