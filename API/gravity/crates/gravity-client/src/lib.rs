//! # gravity-client
//!
//! The Gravity orchestrator — account selection, the rate-limit/failover state
//! machine, remote checkpoints, the AFC tool loop, and a synchronous facade.
//! Ports `gravity/clients.py`, `gravity/afc.py`, and the conversation glue.
//!
//! - [`AsyncGravityClient`] — the async core.
//! - [`GravityClient`] — a `block_on` synchronous facade (no parallel
//!   sync/async hierarchy).
//! - [`afc`] — tool-handler registration and parallel execution.
//! - [`build_transient`] — synthesize a one-off account for the FFI fast path.

#![forbid(unsafe_code)]

pub mod afc;
mod client;
mod transient;

pub use afc::{ToolHandler, ToolRegistry};
pub use client::{
    AsyncGravityClient, GravityClient, ResolvedAccount, RetryPolicy,
};
pub use transient::{build_transient, TransientOptions};

#[cfg(test)]
mod tests {
    use super::*;
    use async_trait::async_trait;
    use gravity_core::*;
    use gravity_providers::{ChatCtx, Provider, Registry, RegistryBuilder};
    use std::sync::atomic::{AtomicU32, Ordering};
    use std::sync::Arc;

    /// A scripted provider for orchestrator tests: emits a tool call on the
    /// first turn, then a final answer once it sees a tool result.
    struct MockProvider {
        caps: ProviderCapabilities,
        calls: AtomicU32,
    }

    impl MockProvider {
        fn new() -> Self {
            MockProvider {
                caps: ProviderCapabilities {
                    provider: ProviderName::Groq,
                    tool_support: ToolSupportMode::Native,
                    structured_output: StructuredOutputMode::JsonInstruction,
                    remote_session_mode: RemoteSessionMode::None,
                    supports_system_prompt: true,
                    supports_stateful_history: false,
                    requires_remote_checkpoint: false,
                    default_history_mode: HistoryMode::Stateless,
                    supports_image_generation: false,
                },
                calls: AtomicU32::new(0),
            }
        }
    }

    #[async_trait]
    impl Provider for MockProvider {
        fn capabilities(&self) -> &ProviderCapabilities {
            &self.caps
        }
        async fn chat(&self, ctx: &ChatCtx<'_>) -> Result<ChatResponse> {
            let n = self.calls.fetch_add(1, Ordering::SeqCst);
            let saw_tool_result = ctx
                .request
                .messages
                .iter()
                .any(|m| m.role == Role::Tool);
            if n == 0 && !saw_tool_result {
                let call = ToolCall {
                    id: "tc1".into(),
                    name: "double".into(),
                    arguments: serde_json::json!({"n": 21}).as_object().cloned().unwrap(),
                };
                Ok(ChatResponse {
                    model: ctx.request.model.full_name().into(),
                    message: Message::assistant_with_tools(None, [call]),
                    usage: Usage::default(),
                    tool_calls_executed: vec![],
                    provider_response_id: None,
                    finish_reason: Some("tool_use".into()),
                    metadata: Metadata::new(),
                })
            } else {
                Ok(ChatResponse {
                    model: ctx.request.model.full_name().into(),
                    message: Message::assistant("the answer is 42"),
                    usage: Usage { input_tokens: Some(3), output_tokens: Some(5), total_tokens: None },
                    tool_calls_executed: vec![],
                    provider_response_id: None,
                    finish_reason: Some("stop".into()),
                    metadata: Metadata::new(),
                })
            }
        }
    }

    fn mock_registry() -> &'static Registry {
        let mut b = RegistryBuilder::new();
        b.add(Arc::new(MockProvider::new()));
        Box::leak(Box::new(b.build()))
    }

    fn resolved(provider: ProviderName) -> ResolvedAccount {
        build_transient(
            mock_registry(),
            provider,
            TransientOptions { secret: "sk-test".into(), ..Default::default() },
        )
        .unwrap()
    }

    #[tokio::test]
    async fn afc_loop_executes_tools_and_returns_final() {
        let registry = mock_registry();
        let mut client = AsyncGravityClient::with_registry(registry).with_env(Env::fixed(0, 1));
        client
            .tools_mut()
            .register_fn("double", |args| {
                let n = args.get("n").and_then(|v| v.as_i64()).unwrap_or(0);
                ToolOutput::ok(serde_json::json!({"result": n * 2}))
            });

        let req = ChatRequest::new(
            ModelId::parse("groq/mock").unwrap(),
            vec![Message::user("double 21 then answer")],
        );
        let cand = vec![resolved(ProviderName::Groq)];
        let conv = Conversation::new("k");
        let resp = client.chat_over(&req, &cand, &conv).await.unwrap();

        assert_eq!(resp.text(), "the answer is 42");
        assert_eq!(resp.tool_calls_executed.len(), 1);
        assert_eq!(resp.tool_calls_executed[0].output.content["result"], 42);
    }

    #[tokio::test]
    async fn mfc_returns_raw_tool_calls_without_handlers() {
        let registry = mock_registry();
        let client = AsyncGravityClient::with_registry(registry).with_env(Env::fixed(0, 1));
        let mut req = ChatRequest::new(
            ModelId::parse("groq/mock").unwrap(),
            vec![Message::user("hi")],
        );
        req.function_calling = FunctionCalling::Manual;
        let cand = vec![resolved(ProviderName::Groq)];
        let conv = Conversation::new("k");
        let resp = client.chat_over(&req, &cand, &conv).await.unwrap();
        // No handlers + manual mode → raw tool call surfaced to caller.
        assert_eq!(resp.tool_calls().len(), 1);
        assert_eq!(&*resp.tool_calls()[0].name, "double");
    }

    /// A provider that only ever returns plain text (no tools) — exercises the
    /// default `chat_stream` wrapper, which emits a single terminal `Done`.
    struct PlainProvider {
        caps: ProviderCapabilities,
    }
    #[async_trait]
    impl Provider for PlainProvider {
        fn capabilities(&self) -> &ProviderCapabilities {
            &self.caps
        }
        async fn chat(&self, ctx: &ChatCtx<'_>) -> Result<ChatResponse> {
            Ok(ChatResponse {
                model: ctx.request.model.full_name().into(),
                message: Message::assistant("streamed answer"),
                usage: Usage::default(),
                tool_calls_executed: vec![],
                provider_response_id: None,
                finish_reason: Some("stop".into()),
                metadata: Metadata::new(),
            })
        }
    }

    #[tokio::test]
    async fn default_stream_emits_done_with_text() {
        // Confirms the source of the FFI streaming bug: a non-incremental
        // provider's stream carries its text in the terminal Done event, which
        // the FFI/server must surface (verified there separately).
        use futures::StreamExt;
        let caps = ProviderCapabilities {
            provider: ProviderName::Groq,
            tool_support: ToolSupportMode::None,
            structured_output: StructuredOutputMode::None,
            remote_session_mode: RemoteSessionMode::None,
            supports_system_prompt: true,
            supports_stateful_history: false,
            requires_remote_checkpoint: false,
            default_history_mode: HistoryMode::Stateless,
            supports_image_generation: false,
        };
        let registry: &'static Registry = Box::leak(Box::new({
            let mut b = RegistryBuilder::new();
            b.add(Arc::new(PlainProvider { caps }));
            b.build()
        }));
        let client = AsyncGravityClient::with_registry(registry).with_env(Env::fixed(0, 1));
        let req = ChatRequest::new(ModelId::parse("groq/x").unwrap(), vec![Message::user("hi")]);
        let cand = build_transient(registry, ProviderName::Groq, TransientOptions { secret: "k".into(), ..Default::default() }).unwrap();
        let conv = Conversation::new("k");
        let mut stream = client.chat_stream_over(&req, &cand, &conv).await.unwrap();
        let mut deltas = 0;
        let mut done_text = String::new();
        while let Some(ev) = stream.next().await {
            match ev.unwrap() {
                gravity_core::StreamEvent::TextDelta(_) => deltas += 1,
                gravity_core::StreamEvent::Done(resp) => done_text = resp.text().to_owned(),
                _ => {}
            }
        }
        assert_eq!(deltas, 0, "default stream emits no deltas");
        assert_eq!(done_text, "streamed answer", "text must live in Done");
    }

    /// JSON-emulated provider: returns a tool_call envelope until it sees a
    /// tool result, then a final envelope.
    struct EmulatedProvider {
        caps: ProviderCapabilities,
    }
    #[async_trait]
    impl Provider for EmulatedProvider {
        fn capabilities(&self) -> &ProviderCapabilities {
            &self.caps
        }
        async fn chat(&self, ctx: &ChatCtx<'_>) -> Result<ChatResponse> {
            let saw_result = ctx
                .request
                .messages
                .iter()
                .any(|m| m.text().contains("Tool result for"));
            let text = if saw_result {
                r#"{"type":"final","content":"the weather is 22C"}"#
            } else {
                r#"{"type":"tool_call","tool":{"name":"get_weather","arguments":{"city":"NYC"}}}"#
            };
            Ok(ChatResponse {
                model: ctx.request.model.full_name().into(),
                message: Message::assistant(text),
                usage: Usage::default(),
                tool_calls_executed: vec![],
                provider_response_id: None,
                finish_reason: Some("stop".into()),
                metadata: Metadata::new(),
            })
        }
    }

    #[tokio::test]
    async fn json_emulated_tool_loop_executes_and_finalizes() {
        let caps = ProviderCapabilities {
            provider: ProviderName::Glm,
            tool_support: ToolSupportMode::JsonEmulated,
            structured_output: StructuredOutputMode::JsonInstruction,
            remote_session_mode: RemoteSessionMode::None,
            supports_system_prompt: true,
            supports_stateful_history: false,
            requires_remote_checkpoint: false,
            default_history_mode: HistoryMode::Stateless,
            supports_image_generation: false,
        };
        let registry: &'static Registry = Box::leak(Box::new({
            let mut b = RegistryBuilder::new();
            b.add(Arc::new(EmulatedProvider { caps }));
            b.build()
        }));
        let mut client = AsyncGravityClient::with_registry(registry).with_env(Env::fixed(0, 1));
        client.tools_mut().register_fn("get_weather", |args| {
            ToolOutput::ok(serde_json::json!({ "temp_c": 22, "city": args.get("city") }))
        });
        let mut req = ChatRequest::new(ModelId::parse("glm/glm-5").unwrap(), vec![Message::user("weather in NYC?")]);
        req.tools = vec![ToolDefinition {
            name: "get_weather".into(),
            description: "weather".into(),
            input_schema: serde_json::from_str(r#"{"type":"object"}"#).unwrap(),
        }];
        let cand = build_transient(registry, ProviderName::Glm, TransientOptions { secret: "k".into(), ..Default::default() }).unwrap();
        let conv = Conversation::new("k");
        let resp = client.chat_over(&req, std::slice::from_ref(&cand), &conv).await.unwrap();
        assert_eq!(resp.text(), "the weather is 22C");
        assert_eq!(resp.tool_calls_executed.len(), 1);
        assert_eq!(&*resp.tool_calls_executed[0].call.name, "get_weather");
    }

    #[tokio::test]
    async fn record_turn_accumulates_history() {
        let client = AsyncGravityClient::with_registry(mock_registry()).with_env(Env::fixed(0, 1));
        client.record_turn("c1", &[Message::user("hi")], &Message::assistant("hello"));
        client.record_turn("c1", &[Message::user("more")], &Message::assistant("ok"));
        let hist = client.history("c1");
        // 2 turns * (1 req + 1 resp), no system prompt set.
        assert_eq!(hist.len(), 4);
        assert_eq!(hist[0].text(), "hi");
        assert_eq!(hist[3].text(), "ok");
    }

    /// A provider that always rate-limits — drives the failover/cooldown path.
    struct RateLimitedProvider {
        caps: ProviderCapabilities,
    }
    #[async_trait]
    impl Provider for RateLimitedProvider {
        fn capabilities(&self) -> &ProviderCapabilities {
            &self.caps
        }
        async fn chat(&self, _ctx: &ChatCtx<'_>) -> Result<ChatResponse> {
            Err(Error::Provider(ProviderError::new(
                ProviderName::Groq,
                gravity_core::ProviderErrorKind::RateLimit,
                "rate_limited",
                "429",
            )))
        }
    }

    fn manager_with_one_groq() -> gravity_accounts::AccountManager {
        use gravity_core::*;
        let acct = ProviderAccount {
            account_id: "a1".into(),
            provider: ProviderName::Groq,
            enabled: true,
            pool: "default".into(),
            priority: 100,
            weight: 1,
            auth: AuthConfig {
                kind: "api_key".into(),
                secret_ref: "inline".into(),
                account_label: None,
                org_id: None,
                device_id: None,
                routing_hint: None,
                chatgpt_account_id: None,
                refresh_secret_ref: None,
                extra: {
                    let mut m = Metadata::new();
                    m.insert("api_key", "sk-1");
                    m
                },
            },
            transport: TransportConfig {
                base_url: String::new(),
                timeout_ms: 60000,
                impersonate: None,
                proxy_url: None,
                verify_ssl: true,
                extra_headers: Default::default(),
            },
            rotation_state: Default::default(),
            quota: Default::default(),
            defaults: ProviderDefaults {
                tool_support: ToolSupportMode::Native,
                structured_output: StructuredOutputMode::JsonInstruction,
                remote_session_mode: RemoteSessionMode::None,
                supports_system_prompt: true,
                default_history_mode: HistoryMode::Auto,
            },
            metadata: Metadata::new(),
        };
        let file = AccountsFile {
            emails: vec![EmailBundle {
                email: "e".into(),
                enabled: true,
                display_name: None,
                providers: vec![acct],
                model_routes: vec![],
                metadata: Metadata::new(),
            }],
            ..Default::default()
        };
        gravity_accounts::AccountManager::in_memory(Env::fixed(1_000_000, 0), file)
    }

    #[tokio::test]
    async fn chat_with_manager_cools_down_failed_account() {
        let caps = ProviderCapabilities {
            provider: ProviderName::Groq,
            tool_support: ToolSupportMode::None,
            structured_output: StructuredOutputMode::None,
            remote_session_mode: RemoteSessionMode::None,
            supports_system_prompt: true,
            supports_stateful_history: false,
            requires_remote_checkpoint: false,
            default_history_mode: HistoryMode::Stateless,
            supports_image_generation: false,
        };
        let registry: &'static Registry = Box::leak(Box::new({
            let mut b = RegistryBuilder::new();
            b.add(Arc::new(RateLimitedProvider { caps }));
            b.build()
        }));
        let client = AsyncGravityClient::with_registry(registry).with_env(Env::fixed(1_000_000, 0));
        let manager = std::sync::Mutex::new(manager_with_one_groq());
        let req = ChatRequest::new(ModelId::parse("groq/x").unwrap(), vec![Message::user("hi")]);
        let conv = Conversation::new("k");

        let result = client.chat_with_manager(&manager, &req, &conv).await;
        assert!(result.is_err(), "rate-limited provider should error");

        // Account is now cooled down + failure recorded (state written back).
        let mgr = manager.lock().unwrap();
        let acct = mgr.get("a1").unwrap();
        assert_eq!(acct.rotation_state.consecutive_failures, 1);
        assert_eq!(&*acct.rotation_state.state, "cooldown");
        assert!(acct.rotation_state.cooldown_until.is_some());
        // And it is now filtered out of future selection.
        assert!(mgr.accounts_for(ProviderName::Groq).is_empty());
    }

    #[tokio::test]
    async fn empty_candidates_errors() {
        let client = AsyncGravityClient::with_registry(mock_registry());
        let req = ChatRequest::new(ModelId::parse("groq/mock").unwrap(), vec![Message::user("x")]);
        let conv = Conversation::new("k");
        let err = client.chat_over(&req, &[], &conv).await.unwrap_err();
        assert!(matches!(err, Error::AccountSelection(_)));
    }
}
