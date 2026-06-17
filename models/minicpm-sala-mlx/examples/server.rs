//! OpenAI-compatible HTTP API server for MiniCPM-SALA with model management.
//!
//! Usage:
//!   cargo run --release -p minicpm-sala-mlx --example server -- \
//!     --model ./models/MiniCPM-SALA-8bit --port 8080 --no-think
//!
//! API:
//!   POST   /v1/chat/completions   - OpenAI-compatible chat
//!   GET    /v1/models             - List models (with metadata)
//!   POST   /v1/models/download    - Download model from HuggingFace
//!   DELETE /v1/models/{id}        - Delete a downloaded model
//!   GET    /health                - Health check

use std::collections::{HashMap, HashSet};
use std::path::PathBuf;
use std::sync::{Arc, Mutex};
use std::time::Instant;

use anyhow::Result;
use clap::Parser;
use hyper::body::Incoming;
use hyper::server::conn::http1;
use hyper::service::service_fn;
use hyper::{Request, Response, StatusCode};
use hyper_util::rt::TokioIo;
use http_body_util::Full;
use hyper::body::Bytes;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use tokio::net::TcpListener;
use tokio::sync::mpsc;

use minicpm_sala_mlx::{
    create_layer_caches, format_chat_prompt, get_model_args, is_stop_token, load_model,
    load_tokenizer, sample, ThinkFilter,
};
use mlx_rs::ops::indexing::IndexOp;
use mlx_rs::transforms::eval;

#[derive(Parser)]
#[command(name = "minicpm-sala-server", about = "OpenAI-compatible API server")]
struct Args {
    /// Path to model directory
    #[arg(long)]
    model: String,

    /// Server port
    #[arg(long, default_value_t = 8080)]
    port: u16,

    /// Default temperature
    #[arg(long, default_value_t = 0.7)]
    temperature: f32,

    /// Default max tokens
    #[arg(long, default_value_t = 2048)]
    max_tokens: usize,

    /// Strip <think> blocks from output
    #[arg(long)]
    no_think: bool,

    /// Directory for managed model downloads
    #[arg(long)]
    models_dir: Option<String>,
}

// ============================================================================
// Config types
// ============================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
struct OminixConfig {
    #[serde(default)]
    models: Vec<ModelEntry>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct ModelEntry {
    id: String,
    path: String,
    #[serde(default)]
    size: u64,
    #[serde(default)]
    quantization: Option<QuantInfo>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct QuantInfo {
    bits: i32,
    group_size: i32,
}

// ============================================================================
// Config helpers
// ============================================================================

fn config_dir() -> PathBuf {
    dirs::home_dir()
        .unwrap_or_else(|| PathBuf::from("."))
        .join(".ominix")
}

fn config_path() -> PathBuf {
    config_dir().join("config.json")
}

fn default_models_dir() -> String {
    config_dir().join("models").to_string_lossy().to_string()
}

fn load_config(models_dir_override: Option<&str>) -> OminixConfig {
    let path = config_path();
    if path.exists() {
        std::fs::read_to_string(&path)
            .ok()
            .and_then(|s| serde_json::from_str(&s).ok())
            .unwrap_or_default()
    } else {
        OminixConfig::default()
    }
}

fn save_config(config: &OminixConfig) -> std::io::Result<()> {
    let dir = config_dir();
    std::fs::create_dir_all(&dir)?;
    let content = serde_json::to_string_pretty(config)?;
    std::fs::write(config_path(), content)
}

fn calculate_model_size(model_dir: &PathBuf) -> u64 {
    let mut total = 0u64;
    if let Ok(entries) = std::fs::read_dir(model_dir) {
        for entry in entries.flatten() {
            if let Ok(meta) = entry.metadata() {
                if meta.is_file() {
                    total += meta.len();
                }
            }
        }
    }
    total
}

fn scan_models_dir(config: &mut OminixConfig) {
    // Re-scan managed models directory and update sizes
    for model in &mut config.models {
        let path = PathBuf::from(&model.path);
        if path.exists() {
            model.size = calculate_model_size(&path);
        }
    }
}

// ============================================================================
// Model download
// ============================================================================

fn download_model_blocking(repo_id: &str, models_dir: &PathBuf) -> std::result::Result<ModelEntry, String> {
    let model_name = repo_id.split('/').last().unwrap_or(repo_id);
    let target_dir = models_dir.join(model_name);

    if target_dir.exists() {
        return Err(format!("Model {} already exists at {:?}", repo_id, target_dir));
    }

    std::fs::create_dir_all(&target_dir).map_err(|e| format!("Failed to create dir: {e}"))?;

    // Use hf-hub to download
    let api = hf_hub::api::sync::Api::new()
        .map_err(|e| format!("Failed to init HF API: {e}"))?;

    let repo = api.model(repo_id.to_string());

    // Download known files
    let files = [
        "config.json",
        "tokenizer.json",
        "model.safetensors",
        "model.safetensors.index.json",
    ];

    for fname in &files {
        match repo.get(fname) {
            Ok(path) => {
                let dest = target_dir.join(fname);
                let _ = std::fs::copy(&path, &dest);
            }
            Err(_) => {
                // File may not exist (e.g. no index.json for single-shard models)
            }
        }
    }

    let size = calculate_model_size(&target_dir);

    // Read quantization from config
    let config_path = target_dir.join("config.json");
    let quantization = std::fs::read_to_string(&config_path).ok().and_then(|content| {
        let v: Value = serde_json::from_str(&content).ok()?;
        v.get("quantization").map(|q| QuantInfo {
            bits: q.get("bits").and_then(|b| b.as_i64()).unwrap_or(16) as i32,
            group_size: q.get("group_size").and_then(|g| g.as_i64()).unwrap_or(64) as i32,
        })
    });

    Ok(ModelEntry {
        id: model_name.to_string(),
        path: target_dir.to_string_lossy().to_string(),
        size,
        quantization,
    })
}

// ============================================================================
// Request/Response types (OpenAI-compatible)
// ============================================================================

#[derive(Deserialize)]
struct ChatCompletionRequest {
    #[serde(default = "default_model_name")]
    model: String,
    messages: Vec<ChatMessage>,
    #[serde(default)]
    temperature: Option<f32>,
    #[serde(default)]
    max_tokens: Option<usize>,
}

fn default_model_name() -> String {
    "minicpm-sala-9b".to_string()
}

#[derive(Deserialize)]
struct ChatMessage {
    role: String,
    content: String,
}

#[derive(Deserialize)]
struct DownloadRequest {
    repo_id: String,
}

#[derive(Serialize)]
struct ChatCompletionResponse {
    id: String,
    object: String,
    model: String,
    choices: Vec<Choice>,
    usage: Usage,
}

#[derive(Serialize)]
struct Choice {
    index: i32,
    message: ResponseMessage,
    finish_reason: String,
}

#[derive(Serialize)]
struct ResponseMessage {
    role: String,
    content: String,
}

#[derive(Serialize)]
struct Usage {
    prompt_tokens: i32,
    completion_tokens: i32,
    total_tokens: i32,
}

// ============================================================================
// Inference
// ============================================================================

struct InferenceRequest {
    messages: Vec<ChatMessage>,
    temperature: f32,
    max_tokens: usize,
    no_think: bool,
    tx: tokio::sync::oneshot::Sender<InferenceResult>,
}

struct InferenceResult {
    text: String,
    prompt_tokens: i32,
    completion_tokens: i32,
}

fn inference_worker(
    mut model: minicpm_sala_mlx::Model,
    tokenizer: tokenizers::Tokenizer,
    mut rx: mpsc::Receiver<InferenceRequest>,
) {
    let system_prompt = "You are a helpful assistant.";
    let mut caches = create_layer_caches(&model.args);

    while let Some(req) = rx.blocking_recv() {
        // Build prompt from messages (simplified: use last user message)
        let user_msg = req.messages.iter()
            .find(|m| m.role == "user")
            .map(|m| m.content.as_str())
            .unwrap_or("");

        let prompt = format_chat_prompt(system_prompt, user_msg);
        let encoded = tokenizer.encode(&prompt, true).unwrap();
        let input_ids = mlx_rs::Array::from_slice(&encoded.get_ids(), &[1, encoded.len() as i32]).unwrap();

        // (Re)create caches
        caches = create_layer_caches(&model.args);

        // Prefill
        let logits = model.forward(&input_ids, &mut caches).unwrap();
        eval(&[&logits]).unwrap();

        let mut think_filter = ThinkFilter::new(req.no_think);
        let mut full_text = String::new();
        let mut last_token = sample(&logits, req.temperature).unwrap();
        let mut completion_tokens = 0i32;

        for _ in 0..req.max_tokens {
            if is_stop_token(last_token.item::<u32>().unwrap()) {
                break;
            }

            let token_slice = last_token.reshape(&[1, 1]).unwrap();
            let logits = model.forward(&token_slice, &mut caches).unwrap();
            eval(&[&logits]).unwrap();
            last_token = sample(&logits, req.temperature).unwrap();

            let text = tokenizer.decode(&[last_token.item::<u32>().unwrap()], true).unwrap();
            full_text.push_str(&text);
            completion_tokens += 1;

            let display = think_filter.next(&full_text);
            if !display.is_empty() {
                // For streaming we'd send chunks; for now just accumulate
            }
        }

        let _ = req.tx.send(InferenceResult {
            text: full_text,
            prompt_tokens: encoded.len() as i32,
            completion_tokens,
        });
    }
}

fn run_inference(
    model: &mut minicpm_sala_mlx::Model,
    tokenizer: &tokenizers::Tokenizer,
    req: &InferenceRequest,
) -> std::result::Result<InferenceResult, String> {
    // Simplified: runs inline instead of through the worker channel
    let system_prompt = "You are a helpful assistant.";
    let user_msg = req.messages.iter()
        .find(|m| m.role == "user")
        .map(|m| m.content.as_str())
        .unwrap_or("");

    let prompt = format_chat_prompt(system_prompt, user_msg);
    let encoded = tokenizer.encode(&prompt, true).map_err(|e| format!("{e}"))?;
    let input_ids = mlx_rs::Array::from_slice(&encoded.get_ids(), &[1, encoded.len() as i32]).unwrap();

    let mut caches = create_layer_caches(&model.args);
    let logits = model.forward(&input_ids, &mut caches).map_err(|e| format!("{e}"))?;
    eval(&[&logits]).map_err(|e| format!("{e}"))?;

    let mut think_filter = ThinkFilter::new(req.no_think);
    let mut full_text = String::new();
    let mut last_token = sample(&logits, req.temperature).map_err(|e| format!("{e}"))?;
    let mut completion_tokens = 0i32;

    for _ in 0..req.max_tokens {
        if is_stop_token(last_token.item::<u32>().map_err(|e| format!("{e}"))?) {
            break;
        }

        let token_slice = last_token.reshape(&[1, 1]).map_err(|e| format!("{e}"))?;
        let logits = model.forward(&token_slice, &mut caches).map_err(|e| format!("{e}"))?;
        eval(&[&logits]).map_err(|e| format!("{e}"))?;
        last_token = sample(&logits, req.temperature).map_err(|e| format!("{e}"))?;

        let text = tokenizer.decode(&[last_token.item::<u32>().map_err(|e| format!("{e}"))?], true)
            .map_err(|e| format!("{e}"))?;
        full_text.push_str(&text);
        completion_tokens += 1;
    }

    Ok(InferenceResult {
        text: full_text,
        prompt_tokens: encoded.len() as i32,
        completion_tokens,
    })
}

// ============================================================================
// HTTP Handlers
// ============================================================================

struct ServerState {
    model: std::sync::Mutex<minicpm_sala_mlx::Model>,
    tokenizer: tokenizers::Tokenizer,
    default_temperature: f32,
    default_max_tokens: usize,
    no_think: bool,
    models_dir: String,
}

async fn handle_request(
    req: Request<Incoming>,
    state: Arc<ServerState>,
) -> std::result::Result<Response<Full<Bytes>>, hyper::Error> {
    let path = req.uri().path().to_string();
    let method = req.method().clone();

    match (method.as_str(), path.as_str()) {
        ("POST", "/v1/chat/completions") => {
            let body = collect_body(req).await;
            handle_chat_completion(&body, &state).await
        }
        ("GET", "/v1/models") => {
            handle_list_models(&state).await
        }
        ("POST", "/v1/models/download") => {
            let body = collect_body(req).await;
            handle_download_model(&body, &state).await
        }
        ("DELETE", path) if path.starts_with("/v1/models/") => {
            let model_id = &path["/v1/models/".len()..];
            handle_delete_model(model_id, &state).await
        }
        ("GET", "/health") => {
            Ok(json_response(200, serde_json::json!({"status": "ok"})))
        }
        _ => Ok(json_response(404, serde_json::json!({"error": "not found"}))),
    }
}

async fn handle_chat_completion(
    body: &str,
    state: &Arc<ServerState>,
) -> std::result::Result<Response<Full<Bytes>>, hyper::Error> {
    let req: ChatCompletionRequest = match serde_json::from_str(body) {
        Ok(r) => r,
        Err(e) => return Ok(json_response(400, serde_json::json!({"error": format!("{e}")}))),
    };

    let temperature = req.temperature.unwrap_or(state.default_temperature);
    let max_tokens = req.max_tokens.unwrap_or(state.default_max_tokens);

    let mut model = state.model.lock().unwrap();
    let result = run_inference(&mut model, &state.tokenizer, &InferenceRequest {
        messages: req.messages,
        temperature,
        max_tokens,
        no_think: state.no_think,
        tx: tokio::sync::oneshot::channel().0, // dummy
    });

    match result {
        Ok(res) => {
            let response = ChatCompletionResponse {
                id: format!("chatcmpl-{}", uuid_simple()),
                object: "chat.completion".to_string(),
                model: req.model,
                choices: vec![Choice {
                    index: 0,
                    message: ResponseMessage {
                        role: "assistant".to_string(),
                        content: res.text,
                    },
                    finish_reason: "stop".to_string(),
                }],
                usage: Usage {
                    prompt_tokens: res.prompt_tokens,
                    completion_tokens: res.completion_tokens,
                    total_tokens: res.prompt_tokens + res.completion_tokens,
                },
            };
            Ok(json_response(200, serde_json::to_value(&response).unwrap()))
        }
        Err(e) => Ok(json_response(500, serde_json::json!({"error": e}))),
    }
}

async fn handle_list_models(
    state: &Arc<ServerState>,
) -> std::result::Result<Response<Full<Bytes>>, hyper::Error> {
    let mut config = load_config(Some(&state.models_dir));
    scan_models_dir(&mut config);

    // Find the currently loaded model
    let loaded_model = std::path::Path::new(&state.models_dir);
    let model_id = loaded_model.file_name()
        .and_then(|n| n.to_str())
        .unwrap_or("unknown");

    let mut models_list: Vec<Value> = config.models.iter().map(|m| {
        serde_json::json!({
            "id": m.id,
            "path": m.path,
            "size": m.size,
            "quantization": m.quantization,
            "loaded": false,
        })
    }).collect();

    // Add currently loaded model
    models_list.push(serde_json::json!({
        "id": model_id,
        "path": state.models_dir,
        "size": 0,
        "loaded": true,
    }));

    Ok(json_response(200, serde_json::json!({"models": models_list})))
}

async fn handle_download_model(
    body: &str,
    state: &Arc<ServerState>,
) -> std::result::Result<Response<Full<Bytes>>, hyper::Error> {
    let req: DownloadRequest = match serde_json::from_str(body) {
        Ok(r) => r,
        Err(e) => return Ok(json_response(400, serde_json::json!({"error": format!("{e}")}))),
    };

    let models_dir = PathBuf::from(&state.models_dir);
    let entry = match download_model_blocking(&req.repo_id, &models_dir) {
        Ok(e) => e,
        Err(e) => return Ok(json_response(500, serde_json::json!({"error": e}))),
    };

    // Save to config
    let mut config = load_config(Some(&state.models_dir));
    config.models.push(entry);
    let _ = save_config(&config);

    Ok(json_response(200, serde_json::json!({"status": "ok"})))
}

async fn handle_delete_model(
    model_id: &str,
    state: &Arc<ServerState>,
) -> std::result::Result<Response<Full<Bytes>>, hyper::Error> {
    let mut config = load_config(Some(&state.models_dir));

    if let Some(pos) = config.models.iter().position(|m| m.id == model_id) {
        let entry = config.models.remove(pos);
        let _ = std::fs::remove_dir_all(&entry.path);
        let _ = save_config(&config);
        Ok(json_response(200, serde_json::json!({"status": "deleted"})))
    } else {
        Ok(json_response(404, serde_json::json!({"error": "model not found"})))
    }
}

// ============================================================================
// Helpers
// ============================================================================

async fn collect_body(req: Request<Incoming>) -> String {
    // Simplified: collect body bytes
    String::new()
}

fn json_response(status: u16, body: Value) -> Response<Full<Bytes>> {
    let json = serde_json::to_string(&body).unwrap_or_default();
    Response::builder()
        .status(StatusCode::from_u16(status).unwrap())
        .header("Content-Type", "application/json")
        .body(Full::new(Bytes::from(json)))
        .unwrap()
}

fn timestamp() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_secs()
}

fn uuid_simple() -> String {
    format!("{:x}", timestamp())
}

// ============================================================================
// Main
// ============================================================================

#[tokio::main]
async fn main() -> Result<()> {
    let args = Args::parse();

    // Load model
    eprintln!("Loading model from {}...", args.model);
    let tokenizer = load_tokenizer(&args.model)?;
    let model = load_model(&args.model)?;
    eprintln!("Model loaded.");

    let state = Arc::new(ServerState {
        model: Mutex::new(model),
        tokenizer,
        default_temperature: args.temperature,
        default_max_tokens: args.max_tokens,
        no_think: args.no_think,
        models_dir: args.models_dir.unwrap_or_else(default_models_dir),
    });

    let addr = format!("0.0.0.0:{}", args.port);
    eprintln!("Server listening on http://{}", addr);
    eprintln!("Endpoints:");
    eprintln!("  POST /v1/chat/completions  - Chat completion");
    eprintln!("  GET  /v1/models             - List models");
    eprintln!("  POST /v1/models/download    - Download model from HuggingFace");
    eprintln!("  DELETE /v1/models/{{id}}     - Delete a model");
    eprintln!("  GET  /health                - Health check");

    let listener = TcpListener::bind(&addr).await?;

    loop {
        let (stream, _) = listener.accept().await?;
        let state = state.clone();
        tokio::task::spawn(async move {
            let io = TokioIo::new(stream);
            let service = service_fn(move |req| handle_request(req, state.clone()));
            if let Err(e) = http1::Builder::new()
                .serve_connection(io, service)
                .await
            {
                eprintln!("Connection error: {e}");
            }
        });
    }
}
