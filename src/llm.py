"""LLM routing layer — model resolution, load-balanced account selection,
streaming + non-streaming generation."""

import asyncio
from typing import AsyncIterator, List, Optional, Any

from gravity import AsyncGravityClient
from gravity.io import load_accounts
from gravity.types import Message
from gravity.tools import Tool

import balancer
from account import load_accounts_for_key, get_providers_for_key, get_db
from providers import PROVIDERS, FREE_PROVIDERS, MODEL_TO_PROVIDER


class NoAccountError(RuntimeError):
    """No usable account/candidate for the requested provider."""


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


# ── Client cache ────────────────────────────────────────────────────────────
#
# AsyncGravityClient holds the impersonating HTTP session pool. Rebuilding it
# per request throws away keep-alive connections (a TLS+TCP handshake on every
# call). Cache one client per single-account AccountsFile, keyed by the
# balancer's account key, so connections are reused across requests.

_client_cache: dict[str, AsyncGravityClient] = {}


def _client_for(candidate) -> AsyncGravityClient:
    cli = _client_cache.get(candidate.key)
    if cli is None:
        cli = AsyncGravityClient(accounts=balancer.single_account_file(candidate))
        _client_cache[candidate.key] = cli
    return cli


def _build_accounts_file(api_key: str, provider: str):
    """Legacy helper retained for discovery paths: full multi-account file."""
    from gravity.accounts import AccountsFile, EmailBundle
    accts, _pool = balancer._accounts_for(api_key, provider)
    return AccountsFile(
        schema_version=4,
        emails=[EmailBundle(email="user@gravity.local", enabled=True,
                            display_name="all", providers=accts)],
    )


def _is_custom_pool(api_key: str, provider: str) -> bool:
    return provider in get_providers_for_key(api_key)


def _sampling_metadata(temperature, max_tokens, seed=None) -> Optional[dict]:
    meta: dict[str, Any] = {}
    if temperature is not None:
        meta["temperature"] = temperature
    if max_tokens is not None:
        meta["max_tokens"] = max_tokens
    if seed is not None:
        meta["seed"] = seed
    return meta or None


# ── Generation (load-balanced, with failover) ──────────────────────────────────

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
    """Non-streaming completion. Tries balancer candidates in order until one
    succeeds; marks health on each attempt."""
    provider, model_name = resolve_model(model)
    full_model = full_model_name(provider, model_name)
    db = get_db()
    candidates = balancer.select_candidates(db, api_key, provider)
    if not candidates:
        raise NoAccountError(f"no usable account for provider '{provider}'")

    metadata = _sampling_metadata(temperature, max_tokens, seed)
    last_err: Optional[Exception] = None
    for cand in candidates:
        client = _client_for(cand)
        with balancer.Lease(db, cand) as lease:
            try:
                resp = await client.chat(
                    model=full_model,
                    messages=messages,
                    system_prompt=system_prompt,
                    tools=tools or [],
                    function_calling="manual" if tools else "auto",
                    metadata=metadata,
                )
                lease.success()
                return resp
            except Exception as e:  # noqa: BLE001 — failover on any provider error
                last_err = e
                continue
    raise last_err or NoAccountError(f"all accounts failed for '{provider}'")


async def generate_stream(
    api_key: str,
    model: str,
    messages: List[Message],
    system_prompt: Optional[str] = None,
    tools: Optional[List[Tool]] = None,
    temperature: Optional[float] = None,
    max_tokens: Optional[int] = None,
) -> AsyncIterator[str]:
    """Streaming completion. Fails over between candidates **only before** the
    first token is emitted (once bytes are sent to the client we can't retry)."""
    provider, model_name = resolve_model(model)
    full_model = full_model_name(provider, model_name)
    db = get_db()
    candidates = balancer.select_candidates(db, api_key, provider)
    if not candidates:
        raise NoAccountError(f"no usable account for provider '{provider}'")

    metadata = _sampling_metadata(temperature, max_tokens)
    last_err: Optional[Exception] = None

    for cand in candidates:
        client = _client_for(cand)
        lease = balancer.Lease(db, cand)
        lease.__enter__()
        started = False
        try:
            async for token in client.stream_response(
                model=full_model,
                messages=messages,
                system_prompt=system_prompt,
                tools=tools or [],
                metadata=metadata,
            ):
                started = True
                yield token
            lease.success()
            lease.__exit__(None, None, None)
            return
        except Exception as e:  # noqa: BLE001
            last_err = e
            lease.__exit__(type(e), e, e.__traceback__)
            if started:
                # Already streamed bytes downstream — cannot fail over silently.
                raise
            continue  # nothing emitted yet → try next account
    raise last_err or NoAccountError(f"all accounts failed for '{provider}'")


async def list_models_for_provider(api_key: str, provider: str, live: bool = False) -> List[dict]:
    """List models for a provider.

    When ``live`` is set, attempt dynamic discovery from the provider's own
    model-list endpoint (Cursor GetUsableModels, ChatGPT /backend-api/models,
    Gemini otAQ7b RPC, GLM /api/models, or OpenAI-compatible /models) via the
    Rust gravity core, falling back to the static catalog on any failure.

    Returns a list of ``{"id","model","display_name","description","tags"}``.
    """
    catalog = [
        {"id": f"{provider}/{m}", "model": m, "display_name": None,
         "description": None, "tags": []}
        for m in PROVIDERS.get(provider, {}).get("models", [])
    ]
    if not live:
        return catalog

    secret = _first_secret_for(api_key, provider)

    # 0. Discovery-list cache (DuckDB body + RocksDB TTL pointer). Model lists
    #    change rarely; caching them is safe (not a response cache).
    try:
        import cache
        hit = cache.get_discovery(provider, secret or "")
        if hit:
            return hit
    except Exception:
        pass

    # 1. Preferred: Rust gravity core (real reverse-engineered discovery).
    try:
        import gravity_rs  # PyO3 wheel from API/gravity (gravity-pyo3)
        discovered = await asyncio.to_thread(gravity_rs.discover_models, provider, secret or "")
        if discovered:
            try:
                import cache
                cache.put_discovery(provider, secret or "", discovered)
            except Exception:
                pass
            return discovered
    except Exception:
        pass

    # 2. Fallback: Python gravity adapter list_models (mostly static).
    try:
        accounts = _build_accounts_file(api_key, provider)
        client = AsyncGravityClient(accounts=accounts)
        dynamic = await asyncio.wait_for(client.list_models(provider), timeout=8.0)
        if dynamic:
            return [
                {"id": f"{provider}/{m}", "model": m, "display_name": None,
                 "description": None, "tags": []}
                for m in dynamic
            ]
    except Exception:
        pass

    return catalog


def _first_secret_for(api_key: str, provider: str) -> str:
    """Resolve the first usable secret for a provider from the user's accounts."""
    try:
        accounts = load_accounts_for_key(api_key)
        for bundle in accounts.emails:
            for acc in bundle.providers:
                if acc.provider.value == provider and acc.enabled:
                    extra = acc.auth.extra or {}
                    for key in ("api_key", "access_token", "session_key", "token"):
                        if extra.get(key):
                            return extra[key]
    except Exception:
        pass
    return ""
