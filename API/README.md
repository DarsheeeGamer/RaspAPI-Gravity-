# API/ — Pure-Rust Gravity Core

This directory contains the **pure-Rust port** of the GravityV2 Python library.

## What's here

```
gravity/          Rust cargo workspace (11 crates)
├── gravity-core        Schema types + Provider trait
├── gravity-http        rquest HTTP/TLS (browser impersonation)
├── gravity-providers   All 40 provider adapters + registry
├── gravity-client      Retry/failover/AFC orchestration
├── gravity-accounts    Auth crypto + LevelDB schema
├── gravity-ffi         Stable C ABI (cdylib)
├── gravity-pyo3        PyO3 Python wheel
├── gravity-node        napi-rs Node.js addon
├── gravity-server      axum OpenAI-compatible HTTP server
├── gravity-cli         clap CLI
└── gravity-testkit     Golden-trace harness
```

## Build

```bash
cd gravity
cargo build --workspace          # debug
cargo build --workspace --release  # release
cargo test --workspace --lib     # 177 tests, all pass

# Python wheel (requires maturin)
maturin build -p gravity-pyo3

# C ABI
cargo build -p gravity-ffi --release
# produces target/release/libgravity.so
```

## Status

| Milestone | Status |
|-----------|--------|
| M0 workspace skeleton + TLS | ✅ |
| M1 walking skeleton + C ABI + PyO3 | ✅ |
| M2 reverse-engineered web providers | ✅ |
| M3 OpenAI-compat base + 21 API providers | ✅ |
| M4 specialized providers (codex/cursor/windsurf/etc) | ✅ |
| M5 hardening + golden suite + full test matrix | ✅ 177 tests |

Gemini live tests: **6/6 pass** (requires credentials).
