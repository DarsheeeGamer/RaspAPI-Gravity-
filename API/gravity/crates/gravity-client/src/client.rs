//! The Gravity orchestrator and its synchronous facade.
//!
//! Ports the request lifecycle from `gravity/clients.py`: account selection,
//! the rate-limit-retry-vs-failover state machine, remote-checkpoint
//! management, and the AFC tool loop — all async, with a `block_on` sync facade
//! on top (eliminating the Python sync/async adapter duplication).

use crate::afc::{execute_tools_parallel, ToolRegistry};
use gravity_core::{
    ChatRequest, ChatResponse, Checkpoint, Conversation, ConversationRuntimeState, Env, Error,
    ExecutedToolCall, Message, ProviderAccount, ProviderName, Result, StreamEvent,
};
use gravity_accounts::{expose, AccountManager};
use gravity_http::HttpClient;
use gravity_core::ProviderErrorKind;
use gravity_providers::{ChatCtx, Provider, Registry};
use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::time::Duration;

/// Max same-account retries on a transient rate limit (Python
/// `RATE_LIMIT_MAX_RETRIES`).
const RATE_LIMIT_MAX_RETRIES: u32 = 5;
/// Max AFC iterations (Python `max_iterations` default).
const MAX_AFC_ITERATIONS: u32 = 20;

/// How a chat turn resolves its model account(s).
pub struct ResolvedAccount {
    /// The account config.
    pub account: ProviderAccount,
    /// The pre-resolved primary secret (API/session key).
    pub secret: String,
}

/// Tuning for the retry/failover loop. Exposed so tests can shrink delays.
#[derive(Debug, Clone)]
pub struct RetryPolicy {
    /// Same-account retries on a transient rate limit.
    pub rate_limit_max_retries: u32,
    /// Delay between rate-limit retries.
    pub rate_limit_delay: Duration,
    /// Max AFC tool-loop iterations.
    pub max_afc_iterations: u32,
}

impl Default for RetryPolicy {
    fn default() -> Self {
        RetryPolicy {
            rate_limit_max_retries: RATE_LIMIT_MAX_RETRIES,
            rate_limit_delay: Duration::from_secs(3),
            max_afc_iterations: MAX_AFC_ITERATIONS,
        }
    }
}

/// The async Gravity client.
///
/// Holds the immutable provider [`Registry`], a shared impersonating
/// [`HttpClient`], injected [`Env`], registered tool handlers, and in-memory
/// conversation state.
pub struct AsyncGravityClient {
    registry: &'static Registry,
    http: Arc<HttpClient>,
    env: Env,
    tools: ToolRegistry,
    policy: RetryPolicy,
    conversations: Mutex<HashMap<String, ConversationRuntimeState>>,
}

impl AsyncGravityClient {
    /// Build a client over the builtin provider registry.
    pub fn new() -> Self {
        AsyncGravityClient::with_registry(gravity_providers::builtin())
    }

    /// Build a client over a specific registry (e.g. for tests).
    pub fn with_registry(registry: &'static Registry) -> Self {
        AsyncGravityClient {
            registry,
            http: Arc::new(HttpClient::new()),
            env: Env::system(),
            tools: ToolRegistry::new(),
            policy: RetryPolicy::default(),
            conversations: Mutex::new(HashMap::new()),
        }
    }

    /// Override the injected environment (clock/entropy) — used by golden tests.
    pub fn with_env(mut self, env: Env) -> Self {
        self.env = env;
        self
    }

    /// Override the retry policy.
    pub fn with_policy(mut self, policy: RetryPolicy) -> Self {
        self.policy = policy;
        self
    }

    /// Mutable access to the tool registry (register AFC handlers here).
    pub fn tools_mut(&mut self) -> &mut ToolRegistry {
        &mut self.tools
    }

    /// The shared HTTP transport.
    pub fn http(&self) -> &Arc<HttpClient> {
        &self.http
    }

    /// Look up a provider adapter.
    fn provider(&self, name: ProviderName) -> Result<&Arc<dyn Provider>> {
        self.registry.provider(name)
    }

    /// Run one chat against an explicit ordered list of candidate accounts,
    /// applying the rate-limit-retry / failover state machine and the AFC loop.
    ///
    /// This is the core entry point; higher-level convenience methods select the
    /// candidate list and build the request.
    pub async fn chat_over(
        &self,
        request: &ChatRequest,
        candidates: &[ResolvedAccount],
        conversation: &Conversation,
    ) -> Result<ChatResponse> {
        request.validate().map_err(|m| Error::validation(m, "invalid_request"))?;
        let provider = self.provider(request.model.provider)?;

        if candidates.is_empty() {
            return Err(Error::AccountSelection(format!(
                "no accounts available for provider {}",
                request.model.provider
            )));
        }

        let mut last_err: Option<Error> = None;
        let mut checkpoint: Option<Checkpoint> = None;

        for (idx, cand) in candidates.iter().enumerate() {
            let is_last = idx + 1 == candidates.len();
            let mut attempt = 0u32;

            loop {
                // Stateful providers: ensure a remote checkpoint exists.
                if provider.capabilities().requires_remote_checkpoint && checkpoint.is_none() {
                    let ctx = self.ctx(request, cand, conversation, None);
                    match provider.open_checkpoint(&ctx).await {
                        Ok(cp) => checkpoint = cp,
                        Err(e) => {
                            last_err = Some(e);
                            break; // try next account
                        }
                    }
                }

                match self
                    .run_afc(provider, request, cand, conversation, checkpoint.as_ref())
                    .await
                {
                    Ok(resp) => return Ok(resp),
                    Err(err) => {
                        if let Error::Provider(pe) = &err {
                            if pe.invalidates_checkpoint() {
                                checkpoint = None;
                            }
                            if pe.is_transient_rate_limit()
                                && attempt < self.policy.rate_limit_max_retries
                            {
                                attempt += 1;
                                tokio::time::sleep(self.policy.rate_limit_delay).await;
                                continue; // retry same account
                            }
                            if pe.should_failover() && !is_last {
                                last_err = Some(err);
                                break; // next account
                            }
                        } else if err.should_failover() && !is_last {
                            last_err = Some(err);
                            break;
                        }
                        return Err(err);
                    }
                }
            }
        }

        Err(last_err.unwrap_or_else(|| {
            Error::AccountSelection("all candidate accounts failed".into())
        }))
    }

    /// Chat using a live [`AccountManager`]: select healthy (non-cooled-down)
    /// accounts, run the failover loop, and **write back** account health —
    /// `mark_success` / `mark_failure` (with a cooldown) — persisting after each
    /// outcome. This is the stateful entry point (vs. [`Self::chat_over`], which
    /// takes a fixed candidate list and mutates nothing).
    pub async fn chat_with_manager(
        &self,
        manager: &Mutex<AccountManager>,
        request: &ChatRequest,
        conversation: &Conversation,
    ) -> Result<ChatResponse> {
        request.validate().map_err(|m| Error::validation(m, "invalid_request"))?;
        let provider_name = request.model.provider;
        let provider = self.provider(provider_name)?;

        // Snapshot candidates + resolve their secrets under one lock.
        let candidates: Vec<(Box<str>, ResolvedAccount)> = {
            let mgr = manager.lock().expect("account manager poisoned");
            mgr.accounts_for(provider_name)
                .into_iter()
                .map(|a| {
                    let secret = mgr
                        .resolve_secret(&a.auth)
                        .map(|s| expose(&s).to_owned())
                        .unwrap_or_default();
                    (a.account_id.clone(), ResolvedAccount { account: a.clone(), secret })
                })
                .collect()
        };
        // Fall back to synthetic anonymous accounts for providers that support
        // unauthenticated access (Perplexity, ChatGPT, GLM).
        let candidates = if candidates.is_empty() {
            match anon_fallback(self.registry, provider_name) {
                Some(anon) => vec![("auto-anon".into(), anon)],
                None => return Err(Error::AccountSelection(format!(
                    "no usable accounts for provider {provider_name}"
                ))),
            }
        } else {
            candidates
        };

        let mut last_err: Option<Error> = None;
        for (idx, (id, cand)) in candidates.iter().enumerate() {
            let is_last = idx + 1 == candidates.len();
            let mut checkpoint = None;
            if provider.capabilities().requires_remote_checkpoint {
                let ctx = self.ctx(request, cand, conversation, None);
                checkpoint = provider.open_checkpoint(&ctx).await.ok().flatten();
            }
            match self.run_afc(provider, request, cand, conversation, checkpoint.as_ref()).await {
                Ok(resp) => {
                    gravity_accounts::metrics::record_usage(
                        &resp.model,
                        resp.usage.input_tokens.unwrap_or(0) as u64,
                        resp.usage.output_tokens.unwrap_or(0) as u64,
                        resp.usage.total_tokens.unwrap_or(0) as u64,
                    );
                    let mut mgr = manager.lock().expect("account manager poisoned");
                    mgr.mark_success(id);
                    let _ = mgr.save();
                    return Ok(resp);
                }
                Err(err) => {
                    // On auth failure, try token refresh once before giving up.
                    let refreshed = if let Error::Provider(ref pe) = err {
                        if pe.kind == ProviderErrorKind::Authentication {
                            if let Some(auth) = provider.authenticator() {
                                let mut account_clone = cand.account.clone();
                                if auth.refresh_credentials(&mut account_clone).await.is_ok() {
                                    let new_secret = {
                                        let mut mgr = manager.lock().expect("account manager poisoned");
                                        mgr.update_auth(id, account_clone.auth.clone());
                                        let _ = mgr.save();
                                        mgr.resolve_secret(&account_clone.auth)
                                            .map(|s| expose(&s).to_owned())
                                            .unwrap_or_default()
                                    };
                                    Some(ResolvedAccount { account: account_clone, secret: new_secret })
                                } else {
                                    None
                                }
                            } else {
                                None
                            }
                        } else {
                            None
                        }
                    } else {
                        None
                    };

                    if let Some(refreshed_cand) = refreshed {
                        match self.run_afc(provider, request, &refreshed_cand, conversation, checkpoint.as_ref()).await {
                            Ok(resp) => {
                                gravity_accounts::metrics::record_usage(
                                    &resp.model,
                                    resp.usage.input_tokens.unwrap_or(0) as u64,
                                    resp.usage.output_tokens.unwrap_or(0) as u64,
                                    resp.usage.total_tokens.unwrap_or(0) as u64,
                                );
                                let mut mgr = manager.lock().expect("account manager poisoned");
                                mgr.mark_success(id);
                                let _ = mgr.save();
                                return Ok(resp);
                            }
                            Err(retry_err) => {
                                let (code, cooldown, failover) = classify_failure(&retry_err);
                                let mut mgr = manager.lock().expect("account manager poisoned");
                                mgr.mark_failure(id, &code, cooldown);
                                let _ = mgr.save();
                                if failover && !is_last {
                                    last_err = Some(retry_err);
                                    continue;
                                }
                                return Err(retry_err);
                            }
                        }
                    }

                    let (code, cooldown, failover) = classify_failure(&err);
                    {
                        let mut mgr = manager.lock().expect("account manager poisoned");
                        mgr.mark_failure(id, &code, cooldown);
                        let _ = mgr.save();
                    }
                    if failover && !is_last {
                        last_err = Some(err);
                        continue;
                    }
                    return Err(err);
                }
            }
        }
        Err(last_err.unwrap_or_else(|| Error::AccountSelection("all accounts failed".into())))
    }

    /// Stream a chat turn from the first candidate account.
    ///
    /// Opens a remote checkpoint first for stateful providers. Returns the
    /// provider's owned event stream; failover is not applied mid-stream.
    pub async fn chat_stream_over(
        &self,
        request: &ChatRequest,
        candidate: &ResolvedAccount,
        conversation: &Conversation,
    ) -> Result<gravity_providers::EventStream> {
        request.validate().map_err(|m| Error::validation(m, "invalid_request"))?;
        let provider = self.provider(request.model.provider)?;
        let mut checkpoint = None;
        if provider.capabilities().requires_remote_checkpoint {
            let ctx = self.ctx(request, candidate, conversation, None);
            checkpoint = provider.open_checkpoint(&ctx).await?;
        }
        let ctx = self.ctx(request, candidate, conversation, checkpoint.as_ref());
        provider.chat_stream(&ctx).await
    }

    /// Discover a provider's available models over a specific resolved account
    /// (hits the live model-listing endpoint; falls back to the static list).
    pub async fn discover_models_over(
        &self,
        provider_name: ProviderName,
        candidate: &ResolvedAccount,
    ) -> Result<Vec<gravity_core::ModelInfo>> {
        let provider = self.provider(provider_name)?;
        // Discovery ignores the request model, but ChatRequest needs a valid
        // ModelId, so use a harmless placeholder.
        let model = gravity_core::ModelId::new(provider_name, "models")
            .unwrap_or_else(|_| gravity_core::ModelId::new(ProviderName::OpenaiApi, "models").unwrap());
        let request = ChatRequest::new(model, Vec::new());
        let conversation = Conversation::new("model-discovery");
        let ctx = self.ctx(&request, candidate, &conversation, None);
        provider.discover_models(&ctx).await
    }

    /// Discover a provider's models, selecting an account from the manager
    /// (or a synthetic anonymous account for providers that allow it).
    pub async fn discover_models_with_manager(
        &self,
        manager: &Mutex<AccountManager>,
        provider_name: ProviderName,
    ) -> Result<Vec<gravity_core::ModelInfo>> {
        // Pick the highest-priority account, resolving its secret under one lock.
        let candidate: Option<ResolvedAccount> = {
            let mgr = manager.lock().expect("account manager poisoned");
            mgr.accounts_for(provider_name).into_iter().next().map(|a| {
                let secret = mgr
                    .resolve_secret(&a.auth)
                    .map(|s| expose(&s).to_owned())
                    .unwrap_or_default();
                ResolvedAccount { account: a.clone(), secret }
            })
        };
        let candidate = candidate
            .or_else(|| anon_fallback(self.registry, provider_name))
            .ok_or_else(|| {
                Error::AccountSelection(format!("no usable accounts for provider {provider_name}"))
            })?;
        self.discover_models_over(provider_name, &candidate).await
    }

    /// Build a [`ChatCtx`] for one provider call.
    fn ctx<'a>(
        &'a self,
        request: &'a ChatRequest,
        cand: &'a ResolvedAccount,
        conversation: &'a Conversation,
        checkpoint: Option<&'a Checkpoint>,
    ) -> ChatCtx<'a> {
        ChatCtx {
            account: &cand.account,
            secret: &cand.secret,
            conversation,
            checkpoint,
            request,
            http: self.http.as_ref(),
            env: &self.env,
        }
    }

    /// Drive the AFC tool loop around `provider.chat`.
    ///
    /// When the model returns tool calls, `function_calling == Auto`, and a
    /// handler exists, the tools are executed and the conversation is extended
    /// for another round (bounded by [`RetryPolicy::max_afc_iterations`]). Calls
    /// without handlers (MFC) are returned to the caller unchanged.
    async fn run_afc(
        &self,
        provider: &Arc<dyn Provider>,
        base_request: &ChatRequest,
        cand: &ResolvedAccount,
        conversation: &Conversation,
        checkpoint: Option<&Checkpoint>,
    ) -> Result<ChatResponse> {
        let auto = matches!(base_request.function_calling, gravity_core::FunctionCalling::Auto);

        // Providers without native tools use the JSON-emulated envelope loop.
        if auto
            && !base_request.tools.is_empty()
            && provider.capabilities().tool_support == gravity_core::ToolSupportMode::JsonEmulated
            && base_request.tools.iter().any(|t| self.tools.contains(&t.name))
        {
            return self
                .run_json_emulated(provider, base_request, cand, conversation, checkpoint)
                .await;
        }

        let mut messages = base_request.messages.clone();
        let mut executed: Vec<ExecutedToolCall> = Vec::new();

        for _ in 0..self.policy.max_afc_iterations {
            let mut request = base_request.clone();
            request.messages = messages.clone();
            let ctx = self.ctx(&request, cand, conversation, checkpoint);
            let mut resp = provider.chat(&ctx).await?;

            let tool_calls = resp.message.tool_calls.clone();
            let runnable = auto
                && !tool_calls.is_empty()
                && tool_calls.iter().all(|c| self.tools.contains(&c.name));

            if !runnable {
                // MFC or terminal: attach any tools executed so far and return.
                if !executed.is_empty() {
                    resp.tool_calls_executed = executed;
                }
                return Ok(resp);
            }

            // Execute tools, fold results into the running transcript.
            let results = execute_tools_parallel(&tool_calls, &self.tools).await;
            messages.push(Message::assistant_with_tools(
                resp.message.content.clone(),
                tool_calls.iter().cloned(),
            ));
            for (call, result) in tool_calls.iter().zip(results.iter()) {
                executed.push(ExecutedToolCall {
                    call: call.clone(),
                    output: gravity_core::ToolOutput {
                        content: result.content.clone(),
                        is_error: result.is_error,
                        structured: result.structured_content.clone(),
                        metadata: gravity_core::Metadata::new(),
                    },
                    duration_ms: None,
                });
                messages.push(Message::tool_result(result.clone()));
            }
        }

        Err(Error::Provider(gravity_core::ProviderError::new(
            base_request.model.provider,
            gravity_core::ProviderErrorKind::Other,
            "afc_loop_exhausted",
            format!("AFC did not converge within {} iterations", self.policy.max_afc_iterations),
        )))
    }

    /// JSON-emulated tool loop for providers without native function calling.
    ///
    /// Injects the JSON tool-mode instruction, then each round parses the
    /// model's reply as a `{"type":"tool_call"|"final"}` envelope: tool calls are
    /// executed and fed back; a `final` (or non-envelope text) ends the loop.
    /// Ports `run_json_emulated_loop` from `gravity/afc.py`.
    async fn run_json_emulated(
        &self,
        provider: &Arc<dyn Provider>,
        base_request: &ChatRequest,
        cand: &ResolvedAccount,
        conversation: &Conversation,
        checkpoint: Option<&Checkpoint>,
    ) -> Result<ChatResponse> {
        let instruction = gravity_core::build_json_tool_instruction(&base_request.tools, None);
        let mut messages = base_request.messages.clone();
        messages.push(Message::user(instruction));
        let mut executed: Vec<ExecutedToolCall> = Vec::new();
        let model_full = base_request.model.full_name();

        for _ in 0..self.policy.max_afc_iterations {
            let mut request = base_request.clone();
            request.messages = messages.clone();
            // Avoid the native branch recursing back here.
            request.function_calling = gravity_core::FunctionCalling::Manual;
            let ctx = self.ctx(&request, cand, conversation, checkpoint);
            let resp = provider.chat(&ctx).await?;
            let text = resp.text().to_owned();

            match gravity_core::parse_tool_envelope(&text) {
                Some(gravity_core::ToolEnvelope::Final { content }) => {
                    let final_text = match content {
                        serde_json::Value::String(s) => s,
                        other => other.to_string(),
                    };
                    return Ok(finalize_emulated(&model_full, final_text, executed));
                }
                Some(gravity_core::ToolEnvelope::ToolCall { name, arguments }) => {
                    if !self.tools.contains(&name) {
                        // Unknown tool — surface the raw model text as the answer.
                        return Ok(finalize_emulated(&model_full, text, executed));
                    }
                    let call = gravity_core::ToolCall {
                        id: format!("emu-{}", executed.len()).into(),
                        name: name.clone().into(),
                        arguments,
                    };
                    let results = execute_tools_parallel(std::slice::from_ref(&call), &self.tools).await;
                    let result = results.into_iter().next().expect("one result");
                    executed.push(ExecutedToolCall {
                        call: call.clone(),
                        output: gravity_core::ToolOutput {
                            content: result.content.clone(),
                            is_error: result.is_error,
                            structured: result.structured_content.clone(),
                            metadata: gravity_core::Metadata::new(),
                        },
                        duration_ms: None,
                    });
                    messages.push(Message::assistant(text));
                    messages.push(Message::user(format!(
                        "Tool result for {name}: {}\nContinue. Reply with one JSON object.",
                        result.content
                    )));
                }
                // Not a valid envelope → treat as the final plain-text answer.
                None => return Ok(finalize_emulated(&model_full, text, executed)),
            }
        }

        Err(Error::Provider(gravity_core::ProviderError::new(
            base_request.model.provider,
            gravity_core::ProviderErrorKind::Other,
            "afc_loop_exhausted",
            format!("AFC did not converge within {} iterations", self.policy.max_afc_iterations),
        )))
    }

    /// Append a completed turn (the request messages + the assistant response)
    /// to the stored conversation so later calls observe prior history.
    ///
    /// Mirrors the `append_user_message`/`append_assistant_message` write-back
    /// the Python client performs after each successful chat.
    pub fn record_turn(&self, key: &str, request_messages: &[Message], response: &Message) {
        let now = chrono::DateTime::from_timestamp_millis(self.env.clock.now_millis() as i64)
            .unwrap_or_else(chrono::Utc::now);
        let mut id = [0u8; 16];
        self.env.entropy.fill(&mut id);
        let turn_id: String = id.iter().map(|b| format!("{b:02x}")).collect();
        let turn = gravity_core::CanonicalTurn {
            turn_id: turn_id.into(),
            request: request_messages.to_vec(),
            response: vec![response.clone()],
            created_at: now,
            metadata: gravity_core::Metadata::new(),
        };
        let mut map = self.conversations.lock().expect("conversation map poisoned");
        let state = map
            .entry(key.to_owned())
            .or_insert_with(|| ConversationRuntimeState::new(Conversation::new(key)));
        state.canonical.append_turn(turn);
        if state.canonical.persistent {
            let _ = save_conversation(key, state);
        }
    }

    /// The accumulated message history for `key` (system prompt + all turns).
    /// Falls back to the on-disk record for persistent conversations.
    pub fn history(&self, key: &str) -> Vec<Message> {
        if let Some(state) = self.conversations.lock().expect("poisoned").get(key) {
            return state.canonical.iter_messages(true);
        }
        load_conversation(key)
            .map(|s| s.canonical.iter_messages(true))
            .unwrap_or_default()
    }

    /// Create (or load) a persistent conversation under `key` that is written to
    /// `~/.gravity/history/<key>.json` after each recorded turn.
    pub fn start_persistent(&self, key: &str, system_prompt: Option<String>) {
        let mut map = self.conversations.lock().expect("poisoned");
        let state = map.entry(key.to_owned()).or_insert_with(|| {
            load_conversation(key).unwrap_or_else(|| {
                let mut conv = Conversation::new(key);
                conv.persistent = true;
                ConversationRuntimeState::new(conv)
            })
        });
        state.canonical.persistent = true;
        if system_prompt.is_some() {
            state.canonical.system_prompt = system_prompt;
        }
    }

    /// Borrow-or-create the conversation runtime state for `key`.
    pub fn conversation(&self, key: &str) -> Conversation {
        let mut map = self.conversations.lock().expect("conversation map poisoned");
        map.entry(key.to_owned())
            .or_insert_with(|| ConversationRuntimeState::new(Conversation::new(key)))
            .canonical
            .clone()
    }
}

impl Default for AsyncGravityClient {
    fn default() -> Self {
        AsyncGravityClient::new()
    }
}

/// Disk path for a conversation key (`~/.gravity/history/<safe-key>.json`).
fn conversation_path(key: &str) -> Option<std::path::PathBuf> {
    let safe: String = key
        .chars()
        .map(|c| if c == '/' || c == ':' || c == '\\' { '_' } else { c })
        .collect();
    gravity_accounts::store::gravity_dir()
        .ok()
        .map(|d| d.join("history").join(format!("{safe}.json")))
}

/// Load a persisted conversation state from disk, if present.
fn load_conversation(key: &str) -> Option<ConversationRuntimeState> {
    let path = conversation_path(key)?;
    let raw = std::fs::read_to_string(path).ok()?;
    serde_json::from_str(&raw).ok()
}

/// Persist a conversation state to disk.
fn save_conversation(key: &str, state: &ConversationRuntimeState) -> std::io::Result<()> {
    let Some(path) = conversation_path(key) else {
        return Ok(());
    };
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let json = serde_json::to_string_pretty(state).unwrap_or_default();
    std::fs::write(path, json)
}

/// Build the terminal [`ChatResponse`] for a JSON-emulated loop.
fn finalize_emulated(model_full: &str, text: String, executed: Vec<ExecutedToolCall>) -> ChatResponse {
    ChatResponse {
        model: model_full.into(),
        message: Message::assistant(text),
        usage: gravity_core::Usage::default(),
        tool_calls_executed: executed,
        provider_response_id: None,
        finish_reason: Some("stop".into()),
        metadata: gravity_core::Metadata::new(),
    }
}

/// Derive `(error_code, cooldown, should_failover)` from a failed attempt.
///
/// Rate-limit / quota errors cool the account down for `Retry-After` (or a
/// 5-minute default); other failures use a zero cooldown (degraded, not parked).
/// Build a synthetic anonymous [`ResolvedAccount`] for providers that allow
/// unauthenticated access (Perplexity, ChatGPT, GLM).  Returns `None` for all
/// other providers.
fn anon_fallback(registry: &Registry, provider: ProviderName) -> Option<ResolvedAccount> {
    use gravity_core::{AuthConfig, Metadata, ProviderAccount, ProviderDefaults, TransportConfig};
    match provider {
        ProviderName::Perplexity | ProviderName::Chatgpt | ProviderName::Glm => {}
        _ => return None,
    }
    let caps = registry.provider(provider).ok()?.capabilities();
    let defaults = ProviderDefaults {
        tool_support: caps.tool_support,
        structured_output: caps.structured_output,
        remote_session_mode: caps.remote_session_mode,
        supports_system_prompt: caps.supports_system_prompt,
        default_history_mode: caps.default_history_mode,
    };
    let auth = AuthConfig {
        kind: "anonymous".into(),
        secret_ref: "".into(),
        account_label: Some("auto-anon".into()),
        org_id: None,
        device_id: None,
        routing_hint: None,
        chatgpt_account_id: None,
        refresh_secret_ref: None,
        extra: Metadata::new(),
    };
    let transport = TransportConfig {
        base_url: String::new(),
        timeout_ms: 60_000,
        impersonate: Some("chrome136".into()),
        proxy_url: None,
        verify_ssl: true,
        extra_headers: Default::default(),
    };
    let account = ProviderAccount {
        account_id: format!("auto-anon-{}", caps.provider.as_str()).into(),
        provider,
        enabled: true,
        pool: "auto-anon".into(),
        priority: -1,
        weight: 1,
        auth,
        transport,
        rotation_state: Default::default(),
        quota: Default::default(),
        defaults,
        metadata: Metadata::new(),
    };
    Some(ResolvedAccount { account, secret: String::new() })
}

fn classify_failure(err: &Error) -> (String, Duration, bool) {
    const DEFAULT_COOLDOWN: Duration = Duration::from_secs(300);
    match err {
        Error::Provider(pe) => {
            let cooldown = match pe.kind {
                gravity_core::ProviderErrorKind::RateLimit
                | gravity_core::ProviderErrorKind::QuotaExceeded => {
                    pe.retry_after.unwrap_or(DEFAULT_COOLDOWN)
                }
                _ => Duration::ZERO,
            };
            (pe.code.to_string(), cooldown, pe.should_failover())
        }
        Error::Network(_) => ("network".into(), Duration::ZERO, true),
        _ => ("error".into(), Duration::ZERO, false),
    }
}

/// Synchronous facade over [`AsyncGravityClient`].
///
/// Owns a multi-thread Tokio runtime and blocks on the async client — one
/// implementation, no parallel sync/async hierarchy.
pub struct GravityClient {
    inner: AsyncGravityClient,
    runtime: tokio::runtime::Runtime,
}

impl GravityClient {
    /// Build a synchronous client over the builtin registry.
    pub fn new() -> Result<Self> {
        let runtime = tokio::runtime::Builder::new_multi_thread()
            .enable_all()
            .build()
            .map_err(|e| Error::Configuration(format!("failed to start runtime: {e}")))?;
        Ok(GravityClient {
            inner: AsyncGravityClient::new(),
            runtime,
        })
    }

    /// Mutable access to the underlying async client (e.g. register tools).
    pub fn inner_mut(&mut self) -> &mut AsyncGravityClient {
        &mut self.inner
    }

    /// Blocking chat over a candidate account list.
    pub fn chat_over(
        &self,
        request: &ChatRequest,
        candidates: &[ResolvedAccount],
        conversation: &Conversation,
    ) -> Result<ChatResponse> {
        self.runtime
            .block_on(self.inner.chat_over(request, candidates, conversation))
    }

    /// Streaming chat: collects chunks into a `Vec<String>` of text deltas.
    ///
    /// Simpler than exposing the async stream — callers get all chunks at once.
    /// For real incremental output use `async_client().chat_stream_over()`.
    pub fn chat_stream_chunks(
        &self,
        request: &ChatRequest,
        candidate: &ResolvedAccount,
        conversation: &Conversation,
    ) -> Result<Vec<String>> {
        use futures::StreamExt;
        self.runtime.block_on(async {
            let mut stream = self.inner.chat_stream_over(request, candidate, conversation).await?;
            let mut chunks = Vec::new();
            while let Some(event) = stream.next().await {
                match event? {
                    StreamEvent::TextDelta(bytes) => {
                        let s = String::from_utf8_lossy(&bytes).into_owned();
                        if !s.is_empty() { chunks.push(s); }
                    }
                    StreamEvent::Done(resp) => {
                        let t = resp.text();
                        if !t.is_empty() && chunks.is_empty() {
                            chunks.push(t.to_owned());
                        }
                        break;
                    }
                    _ => {}
                }
            }
            Ok(chunks)
        })
    }

    /// Blocking model discovery over a resolved account (hits the live
    /// model-listing endpoint, falling back to the static list).
    pub fn discover_models_over(
        &self,
        provider: ProviderName,
        candidate: &ResolvedAccount,
    ) -> Result<Vec<gravity_core::ModelInfo>> {
        self.runtime
            .block_on(self.inner.discover_models_over(provider, candidate))
    }

    /// Access the async client.
    pub fn async_client(&self) -> &AsyncGravityClient {
        &self.inner
    }

    /// Access the runtime handle (for embedding the FFI bridge).
    pub fn runtime(&self) -> &tokio::runtime::Runtime {
        &self.runtime
    }
}
