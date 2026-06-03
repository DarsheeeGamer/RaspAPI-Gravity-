//! Gravity user configuration, persisted to `~/.gravity/config.json`.
//!
//! Ports the schema of `gravity/config.py`: `autocomplete`, `default_model`,
//! `stream`, `log_level`, `proxy`.

use crate::store::{gravity_dir, StoreError};
use serde::{Deserialize, Serialize};

/// User configuration.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Config {
    /// Enable shell tab-completion.
    #[serde(default)]
    pub autocomplete: bool,
    /// Default model when `--model` is omitted.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub default_model: Option<String>,
    /// Stream responses by default.
    #[serde(default = "default_true")]
    pub stream: bool,
    /// Log level (`DEBUG`/`INFO`/`WARNING`/`ERROR`).
    #[serde(default = "default_log_level")]
    pub log_level: String,
    /// Default proxy URL.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub proxy: Option<String>,
}

fn default_true() -> bool {
    true
}
fn default_log_level() -> String {
    "WARNING".to_string()
}

impl Default for Config {
    fn default() -> Self {
        Config {
            autocomplete: false,
            default_model: None,
            stream: true,
            log_level: default_log_level(),
            proxy: None,
        }
    }
}

fn config_path() -> Result<std::path::PathBuf, StoreError> {
    gravity_dir().map(|d| d.join("config.json"))
}

impl Config {
    /// Load config from disk, returning defaults when absent/invalid.
    pub fn load() -> Config {
        config_path()
            .ok()
            .and_then(|p| std::fs::read_to_string(p).ok())
            .and_then(|s| serde_json::from_str(&s).ok())
            .unwrap_or_default()
    }

    /// Persist config to disk.
    pub fn save(&self) -> Result<(), StoreError> {
        let path = config_path()?;
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent).map_err(|e| StoreError::Io(e.to_string()))?;
        }
        let json = serde_json::to_string_pretty(self).map_err(|e| StoreError::Io(e.to_string()))?;
        std::fs::write(&path, json).map_err(|e| StoreError::Io(e.to_string()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn defaults_match_python() {
        let c = Config::default();
        assert!(!c.autocomplete);
        assert!(c.stream);
        assert_eq!(c.log_level, "WARNING");
        assert!(c.default_model.is_none());
    }

    #[test]
    fn roundtrips_json() {
        let c = Config { default_model: Some("groq/llama".into()), stream: false, ..Default::default() };
        let s = serde_json::to_string(&c).unwrap();
        let back: Config = serde_json::from_str(&s).unwrap();
        assert_eq!(c, back);
    }
}
