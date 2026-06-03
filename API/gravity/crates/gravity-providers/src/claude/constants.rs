//! Static endpoints and headers for the claude.ai web API.
//!
//! Mirrors `gravity/claude/constants.py`.

/// Base URL for the claude.ai web API.
pub const BASE_URL: &str = "https://claude.ai/api";
/// Origin header value.
pub const ORIGIN: &str = "https://claude.ai";
/// Default referer.
#[allow(dead_code)]
pub const REFERER: &str = "https://claude.ai/";
/// Chat referer (used on completion requests).
pub const CHAT_REFERER: &str = "https://claude.ai/chat";

/// User-Agent string sent by the web client.
pub const USER_AGENT: &str = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) \
AppleWebKit/537.36 (KHTML, like Gecko) Chrome/145.0.0.0 Safari/537.36";

/// Path templates (formatted with `org_id` / `conversation_id`).
pub mod paths {
    /// `GET` — list organizations.
    pub const ORGANIZATIONS: &str = "/organizations";
    /// `POST` — create a chat conversation. `{org_id}`
    pub const CONVERSATIONS: &str = "/organizations/{org_id}/chat_conversations";
    /// `POST` — completion (SSE). `{org_id}`, `{conversation_id}`
    pub const COMPLETION: &str =
        "/organizations/{org_id}/chat_conversations/{conversation_id}/completion";
    /// `POST` — submit a tool result. `{org_id}`, `{conversation_id}`
    pub const TOOL_RESULT: &str =
        "/organizations/{org_id}/chat_conversations/{conversation_id}/tool_result";
}
