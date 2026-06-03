# Gravity (Rust)

A pure-Rust port of **GravityV2** — a unified multi-provider AI gateway with automatic
failover, account rotation, automatic function calling (AFC), and browser-grade TLS
impersonation. Talks to **33 providers** (clean API-key endpoints *and* reverse-engineered
web UIs) behind one canonical **Gravity Schema**.

## Highlights

- **One async `Provider` trait** (no Python sync/async duplication), with a synchronous
  `block_on` facade.
- **Canonical Gravity Schema** (`ChatRequest`/`ChatResponse`/`Message`/…) that each provider
  converts to/from its own wire format via `SchemaConverter`.
- **Memory-frugal**: `Box<str>` ids, refcounted `bytes::Bytes` payloads, inline `SmallVec`
  collections, `Cow`-borrowed wire requests, null-pointer-cheap empty metadata.
- **BoringSSL TLS impersonation** (`wreq`): Chrome/Firefox/Safari/Edge profiles plus the
  custom Go `go_tls` / `go_boring` (FIPS) JA3 fingerprints.
- **Reverse-engineered kernels verified byte-for-byte against the Python original**: GLM HMAC
  signing, ChatGPT Sentinel proof-of-work (SHA3-512), Kiro brace-counting stream parser,
  Gemini UTF-16 frame parser, Cursor/Windsurf Connect-RPC protobuf codec.
- **Five interface surfaces**: a stable C ABI, native Python (PyO3) and Node (napi) bindings,
  an OpenAI-compatible HTTP server (axum), and a CLI (clap).

## Workspace layout

```
crates/
  gravity-core       schema, traits, errors, clock — no I/O
  gravity-http       wreq TLS impersonation + JA3 profiles + SSE
  gravity-accounts   Fernet-compatible encrypted store + migration
  gravity-providers  Provider trait + registry + all 33 adapters
  gravity-client     failover/retry state machine + AFC + sync facade
  gravity-ffi        C ABI cdylib (libgravity.so) + gravity.h
  gravity-pyo3       Python extension module (gravity_rs)
  gravity-node       Node.js native addon (napi-rs)
  gravity-server     OpenAI-compatible HTTP server (axum)
  gravity-cli        `gravity` command-line tool
```

Dependency flow is strictly downward; `gravity-core` carries no I/O and is the stable contract.

## Providers (33)

**Reverse-engineered web**: claude, chatgpt, gemini, glm, kiro, gemini_cli, codex, cursor,
windsurf, antigravity.
**Native API**: anthropic_api, plus 22 OpenAI-compatible — openai_api, groq, mistral, deepseek,
xai, together, openrouter, fireworks, cerebras, sambanova, hyperbolic, nvidia_nim, perplexity,
qwen, minimax, cohere, litellm, huggingface, ollama, llama_cpp, local_transformers, vertex.

## Build & test

```sh
cargo test --workspace          # unit + integration tests
cargo clippy --workspace        # lints (clean)
cargo doc --workspace --no-deps # API docs

# C ABI
cargo build -p gravity-ffi --release          # → target/release/libgravity.so + crates/gravity-ffi/include/gravity.h
# HTTP server
GRAVITY_PORT=8080 cargo run -p gravity-server  # OpenAI-compatible at /v1/chat/completions
# CLI
cargo run -p gravity-cli -- providers
cargo run -p gravity-cli -- chat --model groq/llama-3.3-70b-versatile --key $GROQ_KEY "Hello"
# Python wheel / Node addon
maturin build -p gravity-pyo3
napi build --release   # in crates/gravity-node
```

## Verification

Reverse-engineered providers are validated with **golden vectors** captured from the Python
library: clocks and entropy are dependency-injected (`Env::fixed`) so request signing, PoW,
and UUIDs are byte-for-byte reproducible. Live integration against real endpoints is gated
behind credentials.

## License

MIT.
