//! Per-model usage metrics, persisted to `~/.gravity/metrics.json`.
//!
//! Ports `gravity/metrics.py`. The on-disk shape is
//! `{ "<model>": {requests, input_tokens, output_tokens, total_tokens} }`.
//! Access is guarded by a process mutex (read-modify-write), matching Python's
//! `threading.Lock`.

use crate::store::{gravity_dir, StoreError};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::sync::Mutex;

static LOCK: Mutex<()> = Mutex::new(());

/// Usage counters for one model.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct ModelMetrics {
    /// Total successful requests.
    #[serde(default)]
    pub requests: u64,
    /// Cumulative prompt tokens.
    #[serde(default)]
    pub input_tokens: u64,
    /// Cumulative completion tokens.
    #[serde(default)]
    pub output_tokens: u64,
    /// Cumulative total tokens.
    #[serde(default)]
    pub total_tokens: u64,
}

fn metrics_path() -> Result<std::path::PathBuf, StoreError> {
    gravity_dir().map(|d| d.join("metrics.json"))
}

fn load_raw() -> BTreeMap<String, ModelMetrics> {
    metrics_path()
        .ok()
        .and_then(|p| std::fs::read_to_string(p).ok())
        .and_then(|s| serde_json::from_str(&s).ok())
        .unwrap_or_default()
}

fn save_raw(data: &BTreeMap<String, ModelMetrics>) -> Result<(), StoreError> {
    let path = metrics_path()?;
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent).map_err(|e| StoreError::Io(e.to_string()))?;
    }
    let json = serde_json::to_string_pretty(data).map_err(|e| StoreError::Io(e.to_string()))?;
    std::fs::write(&path, json).map_err(|e| StoreError::Io(e.to_string()))
}

/// Record one successful request's token usage against `model`.
pub fn record_usage(model: &str, input_tokens: u64, output_tokens: u64, total_tokens: u64) {
    let _guard = LOCK.lock().expect("metrics lock poisoned");
    let mut data = load_raw();
    let entry = data.entry(model.to_owned()).or_default();
    entry.requests += 1;
    entry.input_tokens += input_tokens;
    entry.output_tokens += output_tokens;
    entry.total_tokens += if total_tokens > 0 {
        total_tokens
    } else {
        input_tokens + output_tokens
    };
    let _ = save_raw(&data);
}

/// Load all recorded metrics.
pub fn load_metrics() -> BTreeMap<String, ModelMetrics> {
    let _guard = LOCK.lock().expect("metrics lock poisoned");
    load_raw()
}

/// Reset metrics for `model`, or all when `None`.
pub fn reset_metrics(model: Option<&str>) {
    let _guard = LOCK.lock().expect("metrics lock poisoned");
    match model {
        Some(m) => {
            let mut data = load_raw();
            data.remove(m);
            let _ = save_raw(&data);
        }
        None => {
            let _ = save_raw(&BTreeMap::new());
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn merge_accumulates() {
        // Pure accumulation logic (no disk): mirror record_usage on a map.
        let mut data: BTreeMap<String, ModelMetrics> = BTreeMap::new();
        for _ in 0..3 {
            let e = data.entry("m".into()).or_default();
            e.requests += 1;
            e.input_tokens += 10;
            e.output_tokens += 5;
            e.total_tokens += 15;
        }
        let e = &data["m"];
        assert_eq!(e.requests, 3);
        assert_eq!(e.input_tokens, 30);
        assert_eq!(e.total_tokens, 45);
    }
}
