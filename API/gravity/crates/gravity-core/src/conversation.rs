//! Canonical conversation history and remote-session checkpoints.
//!
//! Mirrors `gravity/history.py`. `history_version` is bumped on every appended
//! turn and is the value a [`Checkpoint`] pins, so a stale checkpoint can be
//! detected and rebuilt.

use crate::capabilities::{HistoryMode, ProviderName};
use crate::message::Message;
use crate::metadata::Metadata;
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

/// One request/response round in a conversation.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CanonicalTurn {
    /// Stable turn id (UUIDv4).
    pub turn_id: Box<str>,
    /// Messages sent in this turn.
    #[serde(default)]
    pub request: Vec<Message>,
    /// Messages received in this turn.
    #[serde(default)]
    pub response: Vec<Message>,
    /// Creation timestamp.
    pub created_at: DateTime<Utc>,
    /// Metadata.
    #[serde(skip_serializing_if = "Metadata::is_empty", default)]
    pub metadata: Metadata,
}

/// The full canonical history of one conversation.
///
/// Mirrors `CanonicalConversation`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Conversation {
    /// Caller-supplied conversation key.
    pub conversation_key: String,
    /// History mode.
    #[serde(default = "default_history_mode")]
    pub history_mode: HistoryMode,
    /// Conversation-level system prompt.
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub system_prompt: Option<String>,
    /// Optional title.
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub title: Option<String>,
    /// Whether this conversation is persisted to disk.
    #[serde(default)]
    pub persistent: bool,
    /// Ordered turns.
    #[serde(default)]
    pub turns: Vec<CanonicalTurn>,
    /// Monotonic version, bumped on every appended turn.
    #[serde(default)]
    pub history_version: u64,
    /// Metadata.
    #[serde(skip_serializing_if = "Metadata::is_empty", default)]
    pub metadata: Metadata,
}

fn default_history_mode() -> HistoryMode {
    HistoryMode::Auto
}

impl Conversation {
    /// A fresh empty conversation under `key`.
    pub fn new(key: impl Into<String>) -> Self {
        Conversation {
            conversation_key: key.into(),
            history_mode: HistoryMode::Auto,
            system_prompt: None,
            title: None,
            persistent: false,
            turns: Vec::new(),
            history_version: 0,
            metadata: Metadata::new(),
        }
    }

    /// Append a turn and bump [`Self::history_version`].
    pub fn append_turn(&mut self, turn: CanonicalTurn) {
        self.turns.push(turn);
        self.history_version += 1;
    }

    /// Flatten all turns into a linear message list, optionally prepending the
    /// system prompt. Mirrors `iter_messages`.
    pub fn iter_messages(&self, include_system_prompt: bool) -> Vec<Message> {
        let mut out = Vec::new();
        if include_system_prompt {
            if let Some(sp) = &self.system_prompt {
                out.push(Message::system(sp.clone()));
            }
        }
        for turn in &self.turns {
            out.extend(turn.request.iter().cloned());
            out.extend(turn.response.iter().cloned());
        }
        out
    }
}

/// A bookmark into a provider's server-side session.
///
/// Mirrors `ProviderCheckpoint`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Checkpoint {
    /// Conversation this checkpoint belongs to.
    pub conversation_key: String,
    /// Provider that owns the remote session.
    pub provider: ProviderName,
    /// Account the session was opened under.
    pub account_id: Box<str>,
    /// Remote conversation id.
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub remote_conversation_id: Option<Box<str>>,
    /// Remote parent/message id for threading.
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub remote_parent_id: Option<Box<str>>,
    /// Opaque remote metadata.
    #[serde(skip_serializing_if = "Metadata::is_empty", default)]
    pub remote_metadata: Metadata,
    /// History version this checkpoint was valid for.
    #[serde(default)]
    pub history_version: u64,
    /// Last update timestamp.
    pub updated_at: DateTime<Utc>,
    /// Whether the checkpoint is still usable.
    #[serde(default = "default_true")]
    pub is_valid: bool,
}

fn default_true() -> bool {
    true
}

/// In-memory runtime state for one conversation: its history plus all
/// per-(provider, account) checkpoints.
///
/// Mirrors `ConversationRuntimeState`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ConversationRuntimeState {
    /// The canonical history.
    pub canonical: Conversation,
    /// Checkpoints keyed implicitly by (provider, account_id).
    #[serde(default)]
    pub checkpoints: Vec<Checkpoint>,
}

impl ConversationRuntimeState {
    /// Wrap a fresh conversation.
    pub fn new(canonical: Conversation) -> Self {
        ConversationRuntimeState {
            canonical,
            checkpoints: Vec::new(),
        }
    }

    /// Mark every checkpoint for `(provider, account_id)` invalid.
    pub fn invalidate(&mut self, provider: ProviderName, account_id: &str) {
        for c in &mut self.checkpoints {
            if c.provider == provider && &*c.account_id == account_id {
                c.is_valid = false;
            }
        }
    }

    /// Return the valid checkpoint for `(provider, account_id)`, if any.
    pub fn get_checkpoint(&self, provider: ProviderName, account_id: &str) -> Option<&Checkpoint> {
        self.checkpoints
            .iter()
            .find(|c| c.provider == provider && &*c.account_id == account_id && c.is_valid)
    }

    /// Insert or replace the checkpoint for its `(provider, account_id)`.
    pub fn upsert_checkpoint(&mut self, checkpoint: Checkpoint) {
        if let Some(slot) = self
            .checkpoints
            .iter_mut()
            .find(|c| c.provider == checkpoint.provider && c.account_id == checkpoint.account_id)
        {
            *slot = checkpoint;
        } else {
            self.checkpoints.push(checkpoint);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn turn(id: &str) -> CanonicalTurn {
        CanonicalTurn {
            turn_id: id.into(),
            request: vec![Message::user("q")],
            response: vec![Message::assistant("a")],
            created_at: DateTime::<Utc>::from_timestamp(0, 0).unwrap(),
            metadata: Metadata::new(),
        }
    }

    #[test]
    fn append_bumps_version_and_flattens() {
        let mut c = Conversation::new("k");
        c.system_prompt = Some("sys".into());
        c.append_turn(turn("t1"));
        c.append_turn(turn("t2"));
        assert_eq!(c.history_version, 2);
        let msgs = c.iter_messages(true);
        // system + 2 turns * (1 req + 1 resp)
        assert_eq!(msgs.len(), 5);
        assert_eq!(c.iter_messages(false).len(), 4);
    }

    #[test]
    fn checkpoint_upsert_and_invalidate() {
        let mut state = ConversationRuntimeState::new(Conversation::new("k"));
        let cp = Checkpoint {
            conversation_key: "k".into(),
            provider: ProviderName::Claude,
            account_id: "a1".into(),
            remote_conversation_id: Some("r1".into()),
            remote_parent_id: None,
            remote_metadata: Metadata::new(),
            history_version: 0,
            updated_at: DateTime::<Utc>::from_timestamp(0, 0).unwrap(),
            is_valid: true,
        };
        state.upsert_checkpoint(cp.clone());
        assert!(state.get_checkpoint(ProviderName::Claude, "a1").is_some());
        state.invalidate(ProviderName::Claude, "a1");
        assert!(state.get_checkpoint(ProviderName::Claude, "a1").is_none());
        // upsert replaces rather than duplicates
        state.upsert_checkpoint(cp);
        assert_eq!(state.checkpoints.len(), 1);
    }
}
