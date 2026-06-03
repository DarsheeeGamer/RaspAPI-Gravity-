//! # gravity-accounts
//!
//! Encrypted credential storage, disk IO, and schema migration for Gravity —
//! the Rust counterpart of `gravity/crypto.py`, `gravity/io.py`, and
//! `gravity/account_manager.py`.
//!
//! - [`crypto`] — a [`crypto::FernetKey`] wire-compatible with Python's
//!   `cryptography.fernet.Fernet` (AES-128-CBC + HMAC-SHA256).
//! - [`store`] — machine-key management and the encrypted-file envelope.
//! - [`io`] — loading/saving with legacy-schema migration.
//! - [`AccountManager`] — in-memory account CRUD with secret resolution.
//!
//! Secrets are wrapped in [`secrecy::SecretString`] so they zeroize on drop and
//! never appear in `Debug` output.

#![forbid(unsafe_code)]

pub mod config;
pub mod crypto;
pub mod io;
pub mod manager;
pub mod metrics;
pub mod store;

pub use config::Config;
pub use manager::{expose, AccountManager, ManagerError};
