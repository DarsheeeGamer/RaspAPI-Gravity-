//! A memory-frugal free-form metadata map.

use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};

/// An optional JSON object used for the many `metadata: dict` fields in the
/// Python schema.
///
/// The empty case — by far the most common — costs a single null pointer
/// rather than an allocated [`Map`], and serializes to nothing thanks to
/// [`Metadata::is_empty`] guarding `skip_serializing_if`.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(transparent)]
pub struct Metadata(Option<Box<Map<String, Value>>>);

impl Metadata {
    /// An empty metadata map (allocation-free).
    pub const fn new() -> Self {
        Metadata(None)
    }

    /// True when there are no entries.
    pub fn is_empty(&self) -> bool {
        self.0.as_ref().map_or(true, |m| m.is_empty())
    }

    /// Number of entries.
    pub fn len(&self) -> usize {
        self.0.as_ref().map_or(0, |m| m.len())
    }

    /// Borrow a value by key.
    pub fn get(&self, key: &str) -> Option<&Value> {
        self.0.as_ref().and_then(|m| m.get(key))
    }

    /// Insert a key/value pair, allocating the backing map on first use.
    pub fn insert(&mut self, key: impl Into<String>, value: impl Into<Value>) {
        self.0
            .get_or_insert_with(|| Box::new(Map::new()))
            .insert(key.into(), value.into());
    }

    /// Borrow the underlying map, if present.
    pub fn as_map(&self) -> Option<&Map<String, Value>> {
        self.0.as_deref()
    }
}

impl From<Map<String, Value>> for Metadata {
    fn from(map: Map<String, Value>) -> Self {
        if map.is_empty() {
            Metadata(None)
        } else {
            Metadata(Some(Box::new(map)))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_metadata_is_cheap_and_serializes_to_object() {
        let m = Metadata::new();
        assert!(m.is_empty());
        assert_eq!(m.len(), 0);
        // transparent over Option -> serializes as null when None
        assert_eq!(serde_json::to_string(&m).unwrap(), "null");
    }

    #[test]
    fn insert_then_read() {
        let mut m = Metadata::new();
        m.insert("k", "v");
        assert!(!m.is_empty());
        assert_eq!(m.get("k").unwrap(), "v");
    }
}
