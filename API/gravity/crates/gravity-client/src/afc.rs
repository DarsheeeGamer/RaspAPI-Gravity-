//! Automatic Function Calling: tool handler registration and parallel exec.
//!
//! Mirrors `execute_tools_parallel` from `gravity/afc.py`. The Python code used
//! a `ThreadPoolExecutor(max_workers=8)`; here we use a [`tokio::task::JoinSet`]
//! bounded by a [`Semaphore`] of 8, preserving call order in the results and
//! capturing handler errors as `is_error` results rather than propagating.

use futures::future::BoxFuture;
use gravity_core::{ToolCall, ToolOutput, ToolResult};
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::Semaphore;

/// Max concurrent tool executions (matches Python `_MAX_TOOL_WORKERS`).
pub const MAX_TOOL_WORKERS: usize = 8;

/// An async tool handler: receives parsed arguments, returns a [`ToolOutput`].
pub type ToolHandler =
    Arc<dyn Fn(serde_json::Map<String, serde_json::Value>) -> BoxFuture<'static, ToolOutput> + Send + Sync>;

/// A registry of callable tool handlers, keyed by tool name.
#[derive(Clone, Default)]
pub struct ToolRegistry {
    handlers: HashMap<String, ToolHandler>,
}

impl ToolRegistry {
    /// An empty registry.
    pub fn new() -> Self {
        ToolRegistry::default()
    }

    /// Register a handler for `name`.
    pub fn register(&mut self, name: impl Into<String>, handler: ToolHandler) -> &mut Self {
        self.handlers.insert(name.into(), handler);
        self
    }

    /// Register a synchronous closure as a handler (run on the async runtime).
    pub fn register_fn<F>(&mut self, name: impl Into<String>, f: F) -> &mut Self
    where
        F: Fn(serde_json::Map<String, serde_json::Value>) -> ToolOutput + Send + Sync + 'static,
    {
        let f = Arc::new(f);
        self.register(
            name,
            Arc::new(move |args| {
                let f = f.clone();
                Box::pin(async move { f(args) })
            }),
        )
    }

    /// Whether `name` has a handler.
    pub fn contains(&self, name: &str) -> bool {
        self.handlers.contains_key(name)
    }

    /// Number of registered handlers.
    pub fn len(&self) -> usize {
        self.handlers.len()
    }

    /// Whether the registry is empty.
    pub fn is_empty(&self) -> bool {
        self.handlers.is_empty()
    }

    fn get(&self, name: &str) -> Option<ToolHandler> {
        self.handlers.get(name).cloned()
    }
}

/// Execute `calls` concurrently (bounded at [`MAX_TOOL_WORKERS`]), returning one
/// [`ToolResult`] per call in the original order. A missing handler or a panic
/// becomes an error result rather than aborting the batch.
pub async fn execute_tools_parallel(
    calls: &[ToolCall],
    registry: &ToolRegistry,
) -> Vec<ToolResult> {
    let sem = Arc::new(Semaphore::new(MAX_TOOL_WORKERS));
    let mut set = tokio::task::JoinSet::new();

    for (idx, call) in calls.iter().enumerate() {
        let call = call.clone();
        let handler = registry.get(&call.name);
        let sem = sem.clone();
        set.spawn(async move {
            let _permit = sem.acquire_owned().await.expect("semaphore closed");
            let output = match handler {
                Some(h) => h(call.arguments.clone()).await,
                None => ToolOutput::error(format!("no handler registered for tool '{}'", call.name)),
            };
            (idx, ToolResult::from_output(&call, output))
        });
    }

    let mut results: Vec<Option<ToolResult>> = vec![None; calls.len()];
    while let Some(joined) = set.join_next().await {
        match joined {
            Ok((idx, res)) => results[idx] = Some(res),
            Err(join_err) => {
                // A panicked handler: synthesize an error in the first empty slot.
                if let Some(slot) = results.iter_mut().find(|s| s.is_none()) {
                    *slot = Some(ToolResult {
                        tool_call_id: "".into(),
                        name: "".into(),
                        content: serde_json::json!({ "error": join_err.to_string() }),
                        is_error: true,
                        structured_content: None,
                    });
                }
            }
        }
    }
    results.into_iter().map(|r| r.expect("every slot filled")).collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn call(id: &str, name: &str, args: serde_json::Value) -> ToolCall {
        ToolCall {
            id: id.into(),
            name: name.into(),
            arguments: args.as_object().cloned().unwrap_or_default(),
        }
    }

    #[tokio::test]
    async fn runs_in_order_and_handles_missing() {
        let mut reg = ToolRegistry::new();
        reg.register_fn("double", |args| {
            let n = args.get("n").and_then(|v| v.as_i64()).unwrap_or(0);
            ToolOutput::ok(json!({ "result": n * 2 }))
        });
        let calls = vec![
            call("c0", "double", json!({"n": 5})),
            call("c1", "missing", json!({})),
            call("c2", "double", json!({"n": 10})),
        ];
        let results = execute_tools_parallel(&calls, &reg).await;
        assert_eq!(results.len(), 3);
        assert_eq!(&*results[0].tool_call_id, "c0");
        assert_eq!(results[0].content["result"], 10);
        assert!(results[1].is_error); // missing handler
        assert_eq!(results[2].content["result"], 20);
    }
}
