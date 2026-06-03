//! `gravity` CLI — list providers/models and run one-shot chats.
//!
//! Ports the core commands of `gravity/cli.py`.
//!
//! ```text
//! gravity providers
//! gravity models groq
//! gravity chat --model groq/llama-3.3-70b-versatile --key $GROQ_KEY "Hello"
//! ```

use clap::{Parser, Subcommand};
use gravity_client::{build_transient, GravityClient, TransientOptions};
use gravity_core::{ChatRequest, Conversation, Message, ModelId, ProviderName};
use std::process::ExitCode;
use std::str::FromStr;

#[derive(Parser)]
#[command(name = "gravity", version, about = "Unified multi-provider AI gateway")]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// List all registered providers and their capabilities.
    Providers,
    /// List the known models for a provider.
    Models {
        /// Provider name, e.g. `groq`.
        provider: String,
    },
    /// Send a single chat message.
    Chat {
        /// Target model as `provider/model`.
        #[arg(short, long)]
        model: String,
        /// API key / session secret (or set GRAVITY_KEY).
        #[arg(short, long)]
        key: Option<String>,
        /// Optional system prompt.
        #[arg(short, long)]
        system: Option<String>,
        /// Stream the response as it arrives.
        #[arg(long)]
        stream: bool,
        /// The user prompt.
        prompt: String,
    },
}

fn main() -> ExitCode {
    let cli = Cli::parse();
    match run(cli.command) {
        Ok(()) => ExitCode::SUCCESS,
        Err(msg) => {
            eprintln!("error: {msg}");
            ExitCode::FAILURE
        }
    }
}

fn run(command: Command) -> Result<(), String> {
    match command {
        Command::Providers => list_providers(),
        Command::Models { provider } => list_models(&provider),
        Command::Chat { model, key, system, stream, prompt } => chat(model, key, system, stream, prompt),
    }
}

fn list_providers() -> Result<(), String> {
    let reg = gravity_providers::builtin();
    let mut rows: Vec<(String, String)> = reg
        .iter_capabilities()
        .map(|(name, caps)| {
            (
                name.as_str().to_owned(),
                format!("tools={:?} system_prompt={}", caps.tool_support, caps.supports_system_prompt),
            )
        })
        .collect();
    rows.sort();
    println!("{} providers registered:\n", rows.len());
    for (name, info) in rows {
        println!("  {name:<20} {info}");
    }
    Ok(())
}

fn list_models(provider: &str) -> Result<(), String> {
    let p = ProviderName::from_str(provider).map_err(|e| e.to_string())?;
    let adapter = gravity_providers::builtin()
        .provider(p)
        .map_err(|e| e.to_string())?;
    let models = adapter.list_models();
    if models.is_empty() {
        println!("{provider}: (models discovered dynamically; none listed statically)");
    } else {
        for m in models {
            println!("{provider}/{m}");
        }
    }
    Ok(())
}

fn chat(
    model: String,
    key: Option<String>,
    system: Option<String>,
    stream: bool,
    prompt: String,
) -> Result<(), String> {
    let model_id = ModelId::parse(&model).map_err(|e| e.to_string())?;
    let secret = key
        .or_else(|| std::env::var("GRAVITY_KEY").ok())
        .unwrap_or_default();
    let kind = if matches!(model_id.provider, ProviderName::Claude | ProviderName::Chatgpt) {
        Some("session_cookie".to_owned())
    } else {
        Some("api_key".to_owned())
    };
    let account = build_transient(
        gravity_providers::builtin(),
        model_id.provider,
        TransientOptions { secret, kind, ..Default::default() },
    )
    .map_err(|e| e.to_string())?;

    let mut request = ChatRequest::new(model_id, vec![Message::user(prompt)]);
    request.system_prompt = system;
    request.stream = stream;

    let client = GravityClient::new().map_err(|e| e.to_string())?;
    let conv = Conversation::new("");

    if stream {
        use std::io::Write;
        let chunks = client
            .chat_stream_chunks(&request, &account, &conv)
            .map_err(|e| e.to_string())?;
        for chunk in chunks {
            print!("{chunk}");
            std::io::stdout().flush().ok();
        }
        println!();
    } else {
        let resp = client
            .chat_over(&request, std::slice::from_ref(&account), &conv)
            .map_err(|e| e.to_string())?;
        println!("{}", resp.text());
    }
    Ok(())
}
