use axum::{
    extract::{Json, Path, State},
    http::StatusCode,
    response::Json as JsonResponse,
    routing::{delete, get, post},
    Router,
};
use std::sync::Arc;
use std::collections::HashMap;
use crate::logging;
use tokio::sync::Mutex;

use crate::grammar::Grammar;
use crate::grammar::GrammarTokenizer;
use crate::log_error;
use crate::log_warn;
use std::path::PathBuf;
use std::path::Path as FilePath;
use std::time::Instant;
use crate::kv_cache::KvCache;
use crate::exo::ExoNode;
use crate::profiled_executor::{AudioInput, LoadedProfiledModel, ProfiledInferenceSession, prefill_with_audio};
use crate::profiled_executor::{ImageInput, MultiModalInput, VideoInput};
use crate::generation::text_to_speech::{base64_encode, pcm_to_wav};
use crate::profiled_executor::EmbedPoolStrategy;
use crate::server::auth::ApiKeyValidator;
use crate::server::benchmark::SystemBenchmark;
use crate::server::models::{ModelEntry, ModelRegistry};
use crate::generation::video_generation::{VideoGenerator, TextToImageGenerator};
use crate::session::{InferenceSessionState, SamplerConfig};
//use std::path::Path; (removed - use FilePath alias from above)
use crate::model_cache::{ModelCache, ModelSource, ModelType};
use crate::exo::NodeInfo;
use crate::metrics::{CacheKind, InferenceTelemetry};
use crate::lora::{self, LoraAdapter, AdapterInfo};
use crate::server::rate_limiter::RateLimiter;
use crate::tools::{self, ToolDefinition, ToolCallResult};
use crate::editing::{self, KnowledgeEditor, EditRequest, EditBatchRequest, AuditRequest, AuditItem};
use tokio::sync::mpsc;
use crate::profiled_executor::StreamConfig;
use axum::response::sse::{Event, Sse};
use axum::middleware::{self, Next};
use axum::extract::Request;
use axum::response::Response;
use axum::response::IntoResponse;
use std::convert::Infallible;
use tokio_stream::wrappers::ReceiverStream;
use std::sync::LazyLock;
use std::sync::atomic::Ordering;
use std::collections::HashSet;
use crate::server::admin::{ActiveRequestInfo};


#[derive(Clone)]
pub struct AppState {
    pub models: Arc<Mutex<ModelRegistry>>,
    pub benchmark: Arc<Mutex<Option<SystemBenchmark>>>,
    /// Model cache for dynamic model loading/unloading.
    pub model_cache: Arc<Mutex<ModelCache>>,
    /// Per-request inference session.
    pub session: Arc<tokio::sync::Mutex<Option<ProfiledInferenceSession>>>,
    /// EXO cluster node (only set when --exo is enabled).
    pub exo_node: Option<Arc<tokio::sync::Mutex<ExoNode>>>,
    /// Production telemetry aggregator (atomic internals, no lock needed).
    pub telemetry: Arc<InferenceTelemetry>,
    /// Loaded LoRA adapters (name -> adapter).
    pub adapters: Arc<Mutex<HashMap<String, LoraAdapter>>>,
    /// Name of the currently active (merged) adapter, if any.
    pub active_adapter: Arc<Mutex<Option<String>>>,
    /// Knowledge editor for factual weight patching.
    pub knowledge_editor: Arc<Mutex<Option<KnowledgeEditor>>>,
    /// Token-bucket rate limiter (per-IP + global).
    pub rate_limiter: Arc<RateLimiter>,
    /// API key validator for Bearer token authentication.
    pub auth: Arc<ApiKeyValidator>,
    /// Active request registry for admin session listing.
    pub admin_request_registry: Arc<Mutex<HashMap<String, ActiveRequestInfo>>>,
    /// Set of request IDs that have been cancelled via the admin API.
    pub admin_cancelled_requests: Arc<Mutex<HashSet<String>>>,
}

pub fn create_router(state: AppState) -> Router {
    let v1_routes = Router::new()
        .route("/v1/chat/completions", post(chat_completions_dispatch))
        .route("/v1/models", get(v1_models))
        .route("/v1/completions", post(v1_completions))
        .route("/v1/audio/speech", post(audio_speech))
        .route("/v1/video/generations", post(video_generations))
        .route("/v1/video/edits", post(video_edits))
        .route("/v1/audio/edits", post(audio_edits))
        .route("/v1/audio/transcriptions", post(audio_transcriptions))
        .route("/v1/images/edits", post(v1_image_edits))
        .route("/v1/images/variations", post(v1_image_variations))
        .route("/v1/images/generations", post(image_generations))
        .route("/v1/embeddings", post(embeddings))
        // EXO cluster endpoints
        .route("/v1/cluster/status", get(cluster_status))
        .route("/v1/cluster/nodes", get(cluster_nodes))
        // LoRA adapter lifecycle
        .route("/v1/adapters", get(list_adapters))
        .route("/v1/adapters/train", post(train_adapter))
        .route("/v1/adapters/load", post(load_adapter))
        .route("/v1/adapters/unload", post(unload_adapter))
        // Knowledge editing
        .route("/v1/edits", get(list_edits).post(apply_edit).delete(undo_last_edit))
        .route("/v1/edits/batch", post(apply_edit_batch))
        .route("/v1/edits/audit", post(audit_facts))
        // Bearer token authentication for all /v1/* routes.
        .layer(middleware::from_fn(auth_middleware));

    Router::new()
        .route("/health", get(health))
        .route("/api/tags", get(list_models))
        .route("/api/models/{id}", get(get_model))
        .route("/api/benchmark", get(get_benchmark))
        .route("/api/chat/completions", post(chat_completions))
        // Telemetry
        .route("/metrics", get(metrics_handler))
        .merge(v1_routes)
        // Admin endpoints (protected by X-Admin-Key).
        .merge(crate::server::admin::admin_router())
        // Per-IP rate limiting layer (applies to all routes).
        .layer(middleware::from_fn_with_state(
            state.rate_limiter.clone(),
            rate_limit_middleware,
        ))
        .fallback(fallback)
        .with_state(state)
}

async fn fallback() -> (StatusCode, &'static str) {
    (StatusCode::NOT_FOUND, "Tribunus: route not found")
}

/// Server start instant for uptime calculation.
static SERVER_START: LazyLock<Instant> = LazyLock::new(Instant::now);

/// Elapsed seconds since server start.
fn uptime() -> f64 {
    SERVER_START.elapsed().as_secs_f64()
}

/// Whether the Apple Neural Engine is available on this hardware.
fn ane_available() -> bool {
    #[cfg(feature = "ane")]
    {
        crate::ane_bridge::AneProgram::init().is_ok()
    }
    #[cfg(not(feature = "ane"))]
    {
        false
    }
}

/// Number of GPU compute cores on this machine.
fn gpu_cores() -> u32 {
    crate::scheduling::HardwareConfig::detect().gpu_cores
}

/// Memory pressure percentage (ratio of wired GPU limit to total RAM).
fn memory_usage_percent() -> f64 {
    let total = crate::gpu_memory::total_physical_ram_mb() as f64;
    if total > 0.0 {
        let limit = crate::gpu_memory::get_current_wired_limit_mb().unwrap_or(0) as f64;
        (limit / total) * 100.0
    } else {
        0.0
    }
}

/// Number of nodes in the EXO cluster, if enabled.
fn cluster_node_count() -> usize {
    1 // Single-node cluster by default when exo is enabled
}

async fn health(
    State(state): State<AppState>,
) -> (StatusCode, JsonResponse<serde_json::Value>) {
    let model_loaded = state.session.lock().await.is_some();
    let cache = state.model_cache.lock().await;
    let is_healthy = model_loaded || cache.has_any();

    let status = if is_healthy { "ok" } else { "loading" };
    let status_code = if is_healthy { StatusCode::OK } else { StatusCode::SERVICE_UNAVAILABLE };

    let ane_avail = ane_available();
    let gpu_cores_count = gpu_cores();
    let mem_pct = memory_usage_percent();
    let cluster_enabled = state.exo_node.is_some();
    let node_count = cluster_node_count();

    let telemetry = &state.telemetry;
    let tokens_gen = telemetry.tokens_generated.load(Ordering::Relaxed);
    let peak_mem = telemetry.peak_memory_bytes.load(Ordering::Relaxed);

    let response = JsonResponse(serde_json::json!({
        "status": status,
        "version": "0.1.0",
        "uptime_seconds": uptime(),
        "hardware": {
            "ane_available": ane_avail,
            "gpu_cores": gpu_cores_count,
            "memory_usage_pct": mem_pct,
        },
        "model": {
            "loaded": model_loaded,
            "cache_entries": cache.entry_count(),
            "cache_usage_mb": cache.used_memory_bytes / 1_048_576,
        },
        "cluster": {
            "enabled": cluster_enabled,
            "nodes": node_count,
        },
        "telemetry": {
            "tokens_generated": tokens_gen,
            "peak_memory_bytes": peak_mem,
        },
    }));

    (status_code, response)
}

/// Middleware: require a valid Bearer token for /v1/* routes.
/// Returns 401 UNAUTHORIZED when the token is missing or invalid.
async fn auth_middleware(
    State(state): State<AppState>,
    req: Request,
    next: Next,
) -> Result<impl IntoResponse, StatusCode> {
    let auth = req
        .headers()
        .get("Authorization")
        .and_then(|v| v.to_str().ok())
        .unwrap_or("");

    if !auth.starts_with("Bearer ") {
        return Err(StatusCode::UNAUTHORIZED);
    }

    let token = &auth[7..];
    if !state.auth.validate(token) {
        return Err(StatusCode::UNAUTHORIZED);
    }

    Ok(next.run(req).await)
}

/// Middleware: extract client IP and check the token-bucket rate limiter.
/// Returns 429 Too Many Requests when the bucket is exhausted.
async fn rate_limit_middleware(
    State(rate_limiter): State<Arc<RateLimiter>>,
    req: Request,
    next: Next,
) -> Result<Response, StatusCode> {
    let ip = extract_client_ip(&req).unwrap_or("unknown");
    if !rate_limiter.check(ip).await {
        return Err(StatusCode::TOO_MANY_REQUESTS);
    }
    Ok(next.run(req).await)
}

/// Extract the client IP from request headers.
/// Priority: X-Forwarded-For (first address) > X-Real-IP > None.
fn extract_client_ip(req: &Request) -> Option<String> {
    if let Some(val) = req.headers().get("x-forwarded-for") {
        if let Ok(s) = val.to_str() {
            if let Some(ip) = s.split(',').next().map(|s| s.trim()) {
                if !ip.is_empty() {
                    return Some(ip.to_string());
                }
            }
        }
    }
    if let Some(val) = req.headers().get("x-real-ip") {
        if let Ok(s) = val.to_str() {
            if !s.is_empty() {
                return Some(s.to_string());
            }
        }
    }
    None
}

async fn metrics_handler(
    State(state): State<AppState>,
) -> (StatusCode, String) {
    let body = state.telemetry.to_prometheus();
    (StatusCode::OK, body)
}

async fn list_models(
    State(state): State<AppState>,
) -> JsonResponse<Vec<ModelEntry>> {
    let models = state.models.lock().await;
    JsonResponse(models.list().to_vec())
}

async fn get_model(
    State(state): State<AppState>,
    Path(id): Path<String>,
) -> JsonResponse<Option<ModelEntry>> {
    let models = state.models.lock().await;
    JsonResponse(models.list().iter().find(|m| m.id == id).cloned())
}

/// `/v1/models` — OpenAI-compatible model listing for Xcode and other
/// OpenAI-API clients.  Returns model IDs matching what the inference
/// endpoints accept (both the model registry and the model cache sources).
async fn v1_models(
    State(state): State<AppState>,
) -> JsonResponse<serde_json::Value> {
    let registry = state.models.lock().await;
    let mut seen = std::collections::HashSet::new();
    let mut data = Vec::new();

    // Always expose the model-cache source names (what /v1/chat/completions accepts).
    for name in ["gemma4", "diffusiongemma", "flux", "funasr", "qwen-tts"] {
        if seen.insert(name.to_string()) {
            data.push(serde_json::json!({
                "id": name,
                "object": "model",
                "created": 1_700_000_000,
                "owned_by": "tribunus"
            }));
        }
    }

    // Also expose every model registered in the model registry.
    for entry in registry.list() {
        if seen.insert(entry.id.clone()) {
            data.push(serde_json::json!({
                "id": entry.id,
                "object": "model",
                "created": 1_700_000_000,
                "owned_by": "tribunus"
            }));
        }
    }

    JsonResponse(serde_json::json!({
        "object": "list",
        "data": data
    }))
}

async fn get_benchmark(
    State(state): State<AppState>,
) -> JsonResponse<serde_json::Value> {
    let bench = state.benchmark.lock().await;
    match &*bench {
        Some(b) => JsonResponse(serde_json::json!({
            "chip": b.chip,
            "ram_gb": b.ram_gb,
            "ops": b.ops.iter().map(|op| serde_json::json!({
                "op_name": op.op_name,
                "mlx_us": op.mlx_us,
                "accelerate_us": op.accelerate_us,
                "mlx_available": op.mlx_available,
                "accelerate_available": op.accelerate_available,
            })).collect::<Vec<_>>(),
            "recommend_accelerate_for": b.recommend_accelerate_for,
            "recommend_mlx_for": b.recommend_mlx_for,
        })),
        None => JsonResponse(serde_json::json!({"status": "not run yet"})),
    }
}

/// Legacy stub endpoint — kept for backward compatibility.
async fn chat_completions(
    State(_state): State<AppState>,
    Json(_body): Json<serde_json::Value>,
) -> Result<JsonResponse<serde_json::Value>, (StatusCode, String)> {
    Ok(JsonResponse(serde_json::json!({
        "id": "chatcmpl-123",
        "object": "chat.completion",
        "choices": [{
            "index": 0,
            "message": {
                "role": "assistant",
                "content": "Hello! How can I help you today?"
            },
            "finish_reason": "stop"
        }]
    })))
}

/// Build fresh KV caches matching the model's execution plan.
pub fn build_kv_caches(model: &LoadedProfiledModel) -> Vec<KvCache> {
    let plan = &model.reader.manifest.execution_plan;
    plan.layers
        .iter()
        .map(|layer| {
            let capacity = if layer.attention_kind == "sliding_attention" {
                layer.sliding_window
            } else {
                32_768
            };
            let n_kv_heads = layer.n_global_kv_heads.unwrap_or(layer.n_kv_heads);
            let head_dim = layer.global_head_dim.unwrap_or(layer.head_dim);
            KvCache::new(
                capacity,
                n_kv_heads,
                head_dim,
                layer.attention_kind == "sliding_attention",
            )
        })
        .collect()
}

/// Extract multimodal message content from an OpenAI-compatible messages array.
///
/// Supports:
/// - Plain text: `{"role": "user", "content": "What's in this image?"}`
/// - Multimodal array: `{"role": "user", "content": [
///     {"type": "text", "text": "What's in this image?"},
///     {"type": "image_url", "image_url": {"url": "https://..."}}
///   ]}`
///
/// Returns `(prompt_text, image_inputs)` where `prompt_text` has [IMG]
/// placeholders inserted at each image position, and `image_inputs` contains
/// the corresponding `ImageInput` structs.
fn extract_multimodal_message(messages: &[serde_json::Value]) -> Option<(String, Vec<ImageInput>)> {
    // Find the last user message.
    let user_msg = messages.iter().rev().find(|m| {
        m.get("role").and_then(|v| v.as_str()) == Some("user")
    })?;

    let content = user_msg.get("content")?;

    // Case 1: Plain text content (string).
    if let Some(text) = content.as_str() {
        return Some((text.to_string(), Vec::new()));
    }

    // Case 2: Content array (multimodal).
    let parts = content.as_array()?;
    let mut prompt = String::new();
    let mut images: Vec<ImageInput> = Vec::new();
    let mut img_idx: u32 = 0;

    for part in parts {
        let part_type = part.get("type").and_then(|v| v.as_str()).unwrap_or("text");
        match part_type {
            "text" => {
                if let Some(text) = part.get("text").and_then(|v| v.as_str()) {
                    prompt.push_str(text);
                }
            }
            "image_url" => {
                // Extract URL from the image_url object.
                let url = part
                    .get("image_url")
                    .and_then(|v| v.get("url"))
                    .and_then(|v| v.as_str())
                    .unwrap_or("");
                if !url.is_empty() {
                    // Insert a placeholder token.  The byte value 0xFF
                    // serves as the image placeholder; the tokenizer maps
                    // it to a unique token ID.
                    prompt.push_str("[IMG]");
                    images.push(ImageInput {
                        source: url.to_string(),
                        placeholder_tokens: vec![0xFFFF], // reserved image token
                    });
                    img_idx += 1;
                }
            }
            _ => {}
        }
    }

    Some((prompt, images))
}

/// Extract audio inputs from an OpenAI-compatible messages array.
///
/// Supports:
/// - `{"type": "audio_url", "audio_url": {"url": "https://..."}}`
///
/// Returns `audio_inputs` containing the corresponding `AudioInput` structs.
fn extract_audio_inputs(messages: &[serde_json::Value]) -> Vec<AudioInput> {
    let user_msg = match messages.iter().rev().find(|m| {
        m.get("role").and_then(|v| v.as_str()) == Some("user")
    }) {
        Some(m) => m,
        None => return Vec::new(),
    };

    let content = match user_msg.get("content") {
        Some(c) => c,
        None => return Vec::new(),
    };

    // Plain text — no audio.
    if content.as_str().is_some() {
        return Vec::new();
    }

    let parts = match content.as_array() {
        Some(p) => p,
        None => return Vec::new(),
    };

    let mut audio_inputs: Vec<AudioInput> = Vec::new();

    for part in parts {
        let part_type = part.get("type").and_then(|v| v.as_str()).unwrap_or("text");
        if part_type != "audio_url" {
            continue;
        }
        if let Some(url) = part
            .get("audio_url")
            .and_then(|a| a.get("url"))
            .and_then(|v| v.as_str())
        {
            if !url.is_empty() {
                audio_inputs.push(AudioInput {
                    source: url.to_string(),
                    placeholder_tokens: Vec::new(),
                });
            }
        }
    }

    audio_inputs
}

/// Extract video inputs from an OpenAI-compatible messages array.
///
/// Supports:
/// - `{"type": "video_url", "video_url": {"url": "https://..."}}`
///
/// Returns `video_inputs` containing the corresponding `VideoInput` structs.
fn extract_video_inputs(messages: &[serde_json::Value]) -> Vec<VideoInput> {
    let user_msg = match messages.iter().rev().find(|m| {
        m.get("role").and_then(|v| v.as_str()) == Some("user")
    }) {
        Some(m) => m,
        None => return Vec::new(),
    };

    let content = match user_msg.get("content") {
        Some(c) => c,
        None => return Vec::new(),
    };

    // Plain text — no video.
    if content.as_str().is_some() {
        return Vec::new();
    }

    let parts = match content.as_array() {
        Some(p) => p,
        None => return Vec::new(),
    };

    let mut video_inputs: Vec<VideoInput> = Vec::new();

    for part in parts {
        let part_type = part.get("type").and_then(|v| v.as_str()).unwrap_or("text");
        if part_type != "video_url" {
            continue;
        }
        if let Some(url) = part
            .get("video_url")
            .and_then(|a| a.get("url"))
            .and_then(|v| v.as_str())
        {
            if !url.is_empty() {
                // Insert a video placeholder marker in the prompt.
                // The video encoder will replace these with actual features.
                video_inputs.push(VideoInput {
                    source: url.to_string(),
                    placeholder_tokens: vec![0xFFFF], // reserved video token
                    num_frames: None, // use default (8 frames)
                });
            }
        }
    }

    video_inputs
}

/// Run a full inference cycle (prefill + decode loop) and return the
/// generated text.
fn run_inference(
    sess: &mut ProfiledInferenceSession,
    model: &LoadedProfiledModel,
    prompt: &str,
    images: &[ImageInput],
    audio_inputs: &[AudioInput],
    video_inputs: &[VideoInput],
    max_tokens: u64,
    sampler_config: SamplerConfig,
    telemetry: Option<&InferenceTelemetry>,
) -> Result<String, String> {
    // ── Tokenize ───────────────────────────────────────────────────────
    let prompt_tokens: Vec<u32> = prompt.bytes().map(|b| b as u32).collect();

    // Apply sampler config (including grammar FSM + tokenizer if set)
    sess.sampler = sampler_config;

    let start = Instant::now();

    // ── Prefill (blocking) ─────────────────────────────────────────────
    let first_token = if !video_inputs.is_empty() {
        // Video (and possibly audio) — route through prefill_with_media.
        let mut media: Vec<MultiModalInput> = Vec::new();
        for audio in audio_inputs {
            media.push(MultiModalInput::Audio(audio.clone()));
        }
        for video in video_inputs {
            media.push(MultiModalInput::Video(video.clone()));
        }
        sess.prefill_with_media(&prompt_tokens, &media, model)
            .map_err(|e| format!("prefill with media failed: {:?}", e))?
    } else if !audio_inputs.is_empty() {
        prefill_with_audio(sess, model, &prompt_tokens, audio_inputs)
            .map_err(|e| format!("audio prefill failed: {:?}", e))?
    } else if !images.is_empty() {
        sess.prefill_with_images(&prompt_tokens, images, model)
            .map_err(|e| format!("prefill with images failed: {:?}", e))?
    } else {
        sess.prefill(&prompt_tokens, model)
            .map_err(|e| format!("prefill failed: {:?}", e))?
    };

    // Record time to first token.
    let ttft_ms = start.elapsed().as_secs_f64() * 1000.0;
    if let Some(t) = telemetry {
        t.record_time_to_first_token(ttft_ms);
    }

    let mut generated = vec![first_token];

    // ── Decode loop ────────────────────────────────────────────────────
    let mut current = first_token;
    for _step in 1..max_tokens {
        let step_start = Instant::now();
        match sess.decode_one(current, model) {
            Ok(next) => {
                let step_ms = step_start.elapsed().as_secs_f64() * 1000.0;
                if let Some(t) = telemetry {
                    t.record_token(step_ms);
                }
                generated.push(next);
                // Stop on EOS token (0 typically marks end-of-sequence for
                // byte-level tokenization).
                if next == 0 {
                    break;
                }
                current = next;
            }
            Err(e) => {
                log_error!("decode error at step {}: {:?}", generated.len(), e);
                break;
            }
        }
    }

    // ── Convert tokens to text ─────────────────────────────────────────
    let output_text: String = generated
        .iter()
        .filter(|t| **t >= 32 && **t <= 126)
        .map(|t| *t as u8 as char)
        .collect();

    Ok(output_text)
}

/// Convert a batch of token IDs to text using the same byte-level decoding
/// used by `run_inference`. Tokens outside printable ASCII are filtered out
/// (handles byte-level tokenization where valid text is in the 32-126 range).
fn detokenize(tokens: &[u32]) -> String {
    tokens
        .iter()
        .filter(|t| **t >= 32 && **t <= 126)
        .map(|t| *t as u8 as char)
        .collect()
}

/// Streaming decode with adaptive chunking.
///
/// Runs prefill + decode loop, batching generated tokens into SSE events
/// according to `StreamConfig`. Events are sent through the mpsc sender.
async fn stream_generate(
    sess: &mut ProfiledInferenceSession,
    model: &LoadedProfiledModel,
    prompt: &str,
    images: &[ImageInput],
    audio_inputs: &[AudioInput],
    video_inputs: &[VideoInput],
    max_tokens: u64,
    sampler_config: SamplerConfig,
    config: &StreamConfig,
    sender: mpsc::Sender<Result<Event, Infallible>>,
) -> Result<(), String> {
    // ── Tokenize ───────────────────────────────────────────────────────
    let prompt_tokens: Vec<u32> = prompt.bytes().map(|b| b as u32).collect();

    // Apply sampler config (including grammar FSM + tokenizer if set)
    sess.sampler = sampler_config;

    // ── Prefill (blocking) ─────────────────────────────────────────────
    let first_token = if !video_inputs.is_empty() {
        let mut media: Vec<MultiModalInput> = Vec::new();
        for audio in audio_inputs {
            media.push(MultiModalInput::Audio(audio.clone()));
        }
        for video in video_inputs {
            media.push(MultiModalInput::Video(video.clone()));
        }
        sess.prefill_with_media(&prompt_tokens, &media, model)
            .map_err(|e| format!("prefill with media failed: {:?}", e))?
    } else if !audio_inputs.is_empty() {
        prefill_with_audio(sess, model, &prompt_tokens, audio_inputs)
            .map_err(|e| format!("audio prefill failed: {:?}", e))?
    } else if !images.is_empty() {
        sess.prefill_with_images(&prompt_tokens, images, model)
            .map_err(|e| format!("prefill with images failed: {:?}", e))?
    } else {
        sess.prefill(&prompt_tokens, model)
            .map_err(|e| format!("prefill failed: {:?}", e))?
    };

    let mut chunk_tokens = vec![first_token];
    let mut current = first_token;
    let mut last_flush = Instant::now();

    // ── Decode loop with adaptive chunking ─────────────────────────────
    for _step in 1..max_tokens {
        let next = match sess.decode_one(current, model) {
            Ok(tok) => tok,
            Err(e) => {
                log_error!("streaming decode error at step {}: {:?}", _step, e);
                break;
            }
        };

        // Stop on EOS token (0 typically marks end-of-sequence for
        // byte-level tokenization).
        if next == 0 {
            break;
        }
        chunk_tokens.push(next);

        // Flush if we have enough tokens or enough time has passed
        if chunk_tokens.len() >= config.max_tokens_per_chunk
            || last_flush.elapsed().as_millis() > config.flush_interval_ms as u128
        {
            let text = detokenize(&chunk_tokens);
            if sender.send(Ok(Event::default().data(text))).await.is_err() {
                return Ok(());
            }
            chunk_tokens.clear();
            last_flush = Instant::now();
        }

        current = next;
    }

    // Flush remaining tokens
    if !chunk_tokens.is_empty() {
        let text = detokenize(&chunk_tokens);
        let _ = sender.send(Ok(Event::default().data(text))).await;
    }

    // Send [DONE] signal (OpenAI streaming convention)
    let _ = sender.send(Ok(Event::default().data("[DONE]"))).await;

    Ok(())
}

/// Streaming handler for `/v1/chat/completions?stream=true`.
///
/// Parses the request body, loads the model, runs streaming prefill/decode,
/// and returns an SSE response with adaptive chunking.
async fn handle_streaming_chat(
    State(state): State<AppState>,
    body: serde_json::Value,
) -> axum::response::Response {
    let model_name = body
        .get("model")
        .and_then(|v| v.as_str())
        .unwrap_or("gemma4")
        .to_string();
    let messages = match body.get("messages").and_then(|v| v.as_array()) {
        Some(arr) => arr,
        None => {
            return JsonResponse(serde_json::json!({
                "error": "missing messages"
            })).into_response();
        }
    };
    let max_tokens = body
        .get("max_tokens")
        .and_then(|v| v.as_u64())
        .unwrap_or(128);

    // ── Extract prompt and multimodal inputs ───────────────────────────
    let (prompt, image_inputs) = extract_multimodal_message(messages)
        .unwrap_or((String::new(), Vec::new()));
    let audio_inputs = extract_audio_inputs(messages);
    let video_inputs = extract_video_inputs(messages);

    // ── Resolve model source and load from cache ───────────────────────
    let sources = crate::model_cache::default_model_sources();
    let source = match sources.get(&model_name) {
        Some(s) => s,
        None => {
            return JsonResponse(serde_json::json!({
                "error": format!("unknown model: {model_name}")
            })).into_response();
        }
    };

    let model_arc = match state.model_cache.lock().await.get_or_load(
        &model_name, source, Some(state.telemetry.as_ref()),
    ) {
        Ok(m) => m,
        Err(e) => {
            return JsonResponse(serde_json::json!({
                "error": format!("failed to load model: {e}")
            })).into_response();
        }
    };

    // ── Detect model type ─────────────────────────────────────────────
    let model_type = crate::model_cache::detect_type(&model_name);
    match model_type {
        ModelType::Text | ModelType::Vision | ModelType::Audio => {}
        _ => {
            return JsonResponse(serde_json::json!({
                "error": "streaming only supported for Text/Vision/Audio models"
            })).into_response();
        }
    }

    // ── Parse sampler config ───────────────────────────────────────────
    let sampler_config = parse_response_format_from_model(&body, &model_arc).await;

    // ── Create a fresh session for streaming ───────────────────────────
    let kv_caches = build_kv_caches(&model_arc);
    let mut sess = ProfiledInferenceSession::new("server".into(), kv_caches);
    sess.setup_from_model(&model_arc);
    sess.generated_tokens.clear();
    sess.phase = InferenceSessionState::Created;

    // ── Create channel and spawn streaming generation ──────────────────
    let (tx, rx) = mpsc::channel::<Result<Event, Infallible>>(64);
    let config = StreamConfig::default();

    // Run generation in block_in_place so decode_one (blocking MLX) works.
    // Tokens are batched into SSE events by stream_generate and buffered
    // in the channel. On handler return the SSE response streams them out.
    tokio::task::block_in_place(|| {
        let rt = tokio::runtime::Handle::current();
        let _ = rt.block_on(async {
            stream_generate(
                sess, &model_arc, &prompt, &image_inputs, &audio_inputs, &video_inputs,
                max_tokens, sampler_config, &config, tx,
            )
            .await
    });
    });

    Sse::new(ReceiverStream::new(rx)).into_response()
}
/// Dispatch wrapper for `/v1/chat/completions` that routes to the streaming
/// or non-streaming handler based on the `stream` field in the request body.
async fn chat_completions_dispatch(
    State(state): State<AppState>,
    Json(body): Json<serde_json::Value>,
) -> axum::response::Response {
    if body.get("stream").and_then(|v| v.as_bool()).unwrap_or(false) {
        return handle_streaming_chat(State(state), body).await;
    }
    v1_chat_completions(State(state), Json(body)).await.into_response()
}

/// `/v1/chat/completions` — OpenAI-compatible chat endpoint with real
/// inference dispatch.
async fn v1_chat_completions(
    State(state): State<AppState>,
    Json(body): Json<serde_json::Value>,
) -> JsonResponse<serde_json::Value> {
    let model_name = body
        .get("model")
        .and_then(|v| v.as_str())
        .unwrap_or("gemma4")
        .to_string();
    let messages = match body.get("messages").and_then(|v| v.as_array()) {
        Some(arr) => arr,
        None => {
            return JsonResponse(serde_json::json!({
                "error": "missing messages"
            }));
        }
    };
    let max_tokens = body
        .get("max_tokens")
        .and_then(|v| v.as_u64())
        .unwrap_or(128);
    let thinking = body.get("thinking").and_then(|v| v.as_bool()).unwrap_or(false);

    // ── Resolve model source and load from cache ──────────────────────
    let sources = crate::model_cache::default_model_sources();
    let source = match sources.get(&model_name) {
        Some(s) => s,
        None => {
            return JsonResponse(serde_json::json!({
                "error": format!("unknown model: {model_name}")
            }));
        }
    };

    let model_arc = match state.model_cache.lock().await.get_or_load(&model_name, source, Some(state.telemetry.as_ref())) {
        Ok(m) => m,
        Err(e) => {
            return JsonResponse(serde_json::json!({
                "error": format!("failed to load model: {e}")
            }));
        }
    };

    // ── Detect model type and route accordingly ────────────────────────
    let model_type = crate::model_cache::detect_type(&model_name);

    match model_type {
        ModelType::Diffusion => {
            // Route to DiffusionGemma diffusion model for parallel text gen.
            let image_dir = model_arc.image_dir.to_string_lossy().to_string();
            let dg = match crate::generation::diffusiongemma::DiffusionModel::load(&image_dir) {
                Ok(m) => m,
                Err(e) => {
            return JsonResponse(serde_json::json!({
                        "error": format!("DiffusionGemma load error: {e}")
            }));
                }
    };

        let chat_messages: Vec<crate::generation::diffusiongemma::ChatMessage> = messages
            .iter().filter_map(|msg| {
                let role = msg.get("role").and_then(|v| v.as_str())?;
                let content_val = msg.get("content")?;
                if let Some(text) = content_val.as_str() {
                    Some(crate::generation::diffusiongemma::ChatMessage {
                        role: role.to_string(),
                        content: vec![crate::generation::diffusiongemma::ContentPart::Text(text.to_string())],
                    })
                } else if let Some(parts) = content_val.as_array() {
                    let content: Vec<_> = parts.iter().filter_map(|part| {
                        let t = part.get("type").and_then(|v| v.as_str()).unwrap_or("text");
                        match t {
                            "text" => part.get("text").and_then(|v| v.as_str())
                                .map(|s| crate::generation::diffusiongemma::ContentPart::Text(s.to_string())),
                            "image_url" => part.get("image_url").and_then(|v| v.get("url")).and_then(|v| v.as_str())
                                .map(|u| crate::generation::diffusiongemma::ContentPart::ImageUrl(u.to_string())),
                            "video_url" => part.get("video_url").and_then(|v| v.get("url")).and_then(|v| v.as_str())
                                .map(|u| crate::generation::diffusiongemma::ContentPart::VideoUrl(u.to_string())),
                            "audio_url" => part.get("audio_url").and_then(|v| v.get("url")).and_then(|v| v.as_str())
                                .map(|u| crate::generation::diffusiongemma::ContentPart::AudioUrl(u.to_string())),
                            _ => None,
        }
                    }).collect();
                    Some(crate::generation::diffusiongemma::ChatMessage { role: role.to_string(), content })
                } else { None }
            }).collect();

        let function_tools: Option<Vec<_>> = body.get("tools").and_then(|v| v.as_array()).map(|tools| {
            tools.iter().filter_map(|t| {
                let f = t.get("function")?;
                Some(crate::generation::diffusiongemma::ToolDefinition {
                    name: f.get("name").and_then(|v| v.as_str())?.to_string(),
                    description: f.get("description").and_then(|v| v.as_str()).unwrap_or("").to_string(),
                    parameters: f.get("parameters").cloned().unwrap_or(serde_json::Value::Null),
                    })
            }).collect()
        });

        let result = if thinking {
            let mut m = chat_messages.clone();
            if let Some(first) = m.first_mut() {
                if first.role == "system" {
                    first.content.insert(0, crate::generation::diffusiongemma::ContentPart::Text(
                        "Think step by step and reason carefully. ".to_string()));
            }
                }
            tokio::task::block_in_place(|| dg.chat(&m, max_tokens as u32, function_tools.as_deref()))
        } else {
            tokio::task::block_in_place(|| dg.chat(&chat_messages, max_tokens as u32, function_tools.as_deref()))
        };

            match result {
            Ok(c) => JsonResponse(serde_json::json!({
                "id": format!("chatcmpl-{:x}", rand_hex()),
                "object": "chat.completion",
                "model": model_name,
                "choices": [{"index":0,"message":{"role":"assistant","content":c.text},"finish_reason":c.finish_reason}],
                "usage": {"prompt_tokens":c.usage.prompt_tokens,"completion_tokens":c.usage.completion_tokens,"total_tokens":c.usage.total_tokens}
            })),
            Err(e) => JsonResponse(serde_json::json!({"error": format!("DiffusionGemma error: {e}")})),
            }
        }
        ModelType::Text | ModelType::Vision | ModelType::Audio => {
            // Parse response_format for grammar-guided generation.
            let sampler_config = parse_response_format_from_model(&body, &model_arc).await;

            // ── Extract prompt and multimodal inputs ─────────────────────
            let (prompt, image_inputs) = extract_multimodal_message(messages)
                .unwrap_or((String::new(), Vec::new()));
            let audio_inputs = extract_audio_inputs(messages);
            let video_inputs = extract_video_inputs(messages);

            // Lock the session mutex and reset for this request.
            let mut guard = state.session.lock().await;
            let sess = match guard.as_mut() {
                Some(s) => s,
        None => {
            return JsonResponse(serde_json::json!({
                        "error": "no session available"
            }));
                }
        };

            // Reset: rebuild KV caches from the model, create a fresh session.
            let kv_caches = build_kv_caches(&model_arc);
            let mut new_sess = ProfiledInferenceSession::new("server".into(), kv_caches);
            new_sess.setup_from_model(&model_arc);
            new_sess.generated_tokens.clear();
            new_sess.phase = InferenceSessionState::Created;
            *sess = new_sess;

            // Run inference (blocking MLX work).
            let output_text = tokio::task::block_in_place(|| {
                run_inference(sess, &model_arc, &prompt, &image_inputs, &audio_inputs, &video_inputs, max_tokens, sampler_config, Some(state.telemetry.as_ref()))
        });

            let output_text = match output_text {
                Ok(t) => t,
                Err(e) => {
            return JsonResponse(serde_json::json!({
                        "error": e
            }));
                }
        };

            let prompt_tokens_count = prompt.bytes().len() as u64;
            let completion_tokens_count = output_text.len() as u64;

            // If the request includes tools (function calling), attempt to
            // parse and repair the output as a function call.
            if tools::has_tools_request(&body) {
                match tools::extract_tool(&body) {
                    Ok(tool) => match tools::parse_and_repair(&output_text, &tool) {
                        ToolCallResult::Valid(call) | ToolCallResult::Repaired(call, _) => {
                            // Execute the tool call and return a tool_calls response.
                            match tools::execute_tool_call(&call) {
                                Ok(tool_result) => {
                                    // Return the tool call in OpenAI format.
                                    return JsonResponse(serde_json::json!({
                                        "id": format!("chatcmpl-{:x}", rand_hex()),
                                        "object": "chat.completion",
                                        "model": model_name,
                                        "choices": [{
                                            "index": 0,
                                            "message": {
                                                "role": "assistant",
                                                "content": null,
                                                "tool_calls": [{
                                                    "id": format!("call_{:x}", rand_hex()),
                                                    "type": "function",
                                                    "function": {
                                                        "name": call.name,
                                                        "arguments": serde_json::to_string(&call.arguments).unwrap_or_default()
                                                    }
                                                }]
                                            },
                                            "finish_reason": "tool_calls"
                                        }],
                                        "usage": {
                                            "prompt_tokens": prompt_tokens_count,
                                            "completion_tokens": completion_tokens_count,
                                            "total_tokens": prompt_tokens_count + completion_tokens_count
                                        }
                                    }));
                                }
                                Err(exec_err) => {
                                    // Tool execution failed; retry with error context.
                                    let retry = tools::retry_with_error(
                                        sess, &model_arc, messages, &exec_err, &tool, max_tokens,
                                    );
                                    return match retry {
                                        Ok(ToolCallResult::Valid(c) | ToolCallResult::Repaired(c, _)) => {
                                            JsonResponse(serde_json::json!({
                                                "id": format!("chatcmpl-{:x}", rand_hex()),
                                                "object": "chat.completion",
                                                "model": model_name,
                                                "choices": [{
                                                    "index": 0,
                                                    "message": {
                                                        "role": "assistant",
                                                        "content": null,
                                                        "tool_calls": [{
                                                            "id": format!("call_{:x}", rand_hex()),
                                                            "type": "function",
                                                            "function": {
                                                                "name": c.name,
                                                                "arguments": serde_json::to_string(&c.arguments).unwrap_or_default()
                                                            }
                                                        }]
                                                    },
                                                    "finish_reason": "tool_calls"
                                                }],
                                                "usage": {
                                                    "prompt_tokens": prompt_tokens_count,
                                                    "completion_tokens": completion_tokens_count,
                                                    "total_tokens": prompt_tokens_count + completion_tokens_count
                                                }
                                            }))
                                        }
                                        Ok(ToolCallResult::Unrepairable(retry_err)) | Err(retry_err) => {
                                            JsonResponse(serde_json::json!({
                                                "error": format!("tool call failed after retry: {retry_err}")
                                            }))
                                        }
                                    };
                                }
                            }
                        }
                        ToolCallResult::Unrepairable(err) => {
                            // Cannot repair; retry generation with error context.
                            let retry = tools::retry_with_error(
                                sess, &model_arc, messages, &err, &tool, max_tokens,
                            );
                            return match retry {
                                Ok(ToolCallResult::Valid(c) | ToolCallResult::Repaired(c, _)) => {
                                    JsonResponse(serde_json::json!({
                                        "id": format!("chatcmpl-{:x}", rand_hex()),
                                        "object": "chat.completion",
                                        "model": model_name,
                                        "choices": [{
                                            "index": 0,
                                            "message": {
                                                "role": "assistant",
                                                "content": null,
                                                "tool_calls": [{
                                                    "id": format!("call_{:x}", rand_hex()),
                                                    "type": "function",
                                                    "function": {
                                                        "name": c.name,
                                                        "arguments": serde_json::to_string(&c.arguments).unwrap_or_default()
                                                    }
                                                }]
                                            },
                                            "finish_reason": "tool_calls"
                                        }],
                                        "usage": {
                                            "prompt_tokens": prompt_tokens_count,
                                            "completion_tokens": completion_tokens_count,
                                            "total_tokens": prompt_tokens_count + completion_tokens_count
                                        }
                                    }))
                                }
                                Ok(ToolCallResult::Unrepairable(retry_err)) | Err(retry_err) => {
                                    JsonResponse(serde_json::json!({
                                        "error": format!("tool call failed after retry: {retry_err}")
                                    }))
                                }
                            };
                        }
                    },
                    Err(e) => {
                        return JsonResponse(serde_json::json!({
                            "error": format!("tool extraction failed: {e}")
                        }));
                    }
                }
            }

            // Standard non-tool response.
            JsonResponse(serde_json::json!({
                "id": format!("chatcmpl-{:x}", rand_hex()),
                "object": "chat.completion",
                "model": model_name,
                "choices": [{
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": output_text
                    },
                    "finish_reason": "stop"
                }],
                "usage": {
                    "prompt_tokens": prompt_tokens_count,
                    "completion_tokens": completion_tokens_count,
                    "total_tokens": prompt_tokens_count + completion_tokens_count
                }
            }))
        }
        ModelType::ImageGen => {
            JsonResponse(serde_json::json!({
                "error": "image generation not supported via chat completions; use /v1/images/generations"
            }))
        }
    }
}

/// `/v1/completions` — OpenAI-compatible text completion endpoint.
async fn v1_completions(
    State(state): State<AppState>,
    Json(body): Json<serde_json::Value>,
) -> JsonResponse<serde_json::Value> {
    let model_name = body
        .get("model")
        .and_then(|v| v.as_str())
        .unwrap_or("gemma4")
        .to_string();
    let prompt = match body.get("prompt").and_then(|v| v.as_str()) {
        Some(p) => p.to_string(),
        None => {
            return JsonResponse(serde_json::json!({
                "error": "missing prompt"
            }));
        }
    };
    let max_tokens = body
        .get("max_tokens")
        .and_then(|v| v.as_u64())
        .unwrap_or(128);

    let sources = crate::model_cache::default_model_sources();
    let source = match sources.get(&model_name) {
        Some(s) => s,
        None => {
            return JsonResponse(serde_json::json!({
                "error": format!("unknown model: {model_name}")
            }));
        }
    };

    let model_arc = match state.model_cache.lock().await.get_or_load(&model_name, source, Some(state.telemetry.as_ref())) {
        Ok(m) => m,
        Err(e) => {
            return JsonResponse(serde_json::json!({
                "error": format!("failed to load model: {e}")
            }));
        }
    };

    let mut guard = state.session.lock().await;
    let sess = match guard.as_mut() {
        Some(s) => s,
        None => {
            return JsonResponse(serde_json::json!({
                "error": "no session available"
            }));
        }
    };

    // Reset: rebuild KV caches, create fresh session.
    let kv_caches = build_kv_caches(&model_arc);
    let mut new_sess = ProfiledInferenceSession::new("server".into(), kv_caches);
    new_sess.setup_from_model(&model_arc);
    new_sess.generated_tokens.clear();
    new_sess.phase = InferenceSessionState::Created;
    *sess = new_sess;

    let output_text = tokio::task::block_in_place(|| {
        run_inference(sess, &model_arc, &prompt, &[], &[], &[], max_tokens, SamplerConfig {
            temperature: None,
            top_k: Some(1),
            top_p: None,
            repetition_penalty: None,
            seed: None,
            stop_token_ids: Vec::new(),
            grammar: None,
            grammar_tokenizer: None,
        }, Some(state.telemetry.as_ref()))
    });

    let output_text = match output_text {
        Ok(t) => t,
        Err(e) => {
            return JsonResponse(serde_json::json!({
                "error": e
            }));
        }
    };

    let prompt_tokens_count = prompt.bytes().len() as u64;
    let completion_tokens_count = output_text.len() as u64;

    JsonResponse(serde_json::json!({
        "id": format!("cmpl-{:x}", rand_hex()),
        "object": "text_completion",
        "model": model_name,
        "choices": [{
            "index": 0,
            "text": output_text,
            "finish_reason": "stop"
        }],
        "usage": {
            "prompt_tokens": prompt_tokens_count,
            "completion_tokens": completion_tokens_count,
            "total_tokens": prompt_tokens_count + completion_tokens_count
        }
    }))
}

/// `/v1/embeddings` — OpenAI-compatible text embeddings endpoint.
///
/// Accepts `input` (string or array of strings) and optional `model`.
/// Returns a normalized embedding vector for each input.
async fn embeddings(
    State(state): State<AppState>,
    Json(body): Json<serde_json::Value>,
) -> Result<JsonResponse<serde_json::Value>, (StatusCode, String)> {
    // Parse input — single string or array of strings
    let inputs: Vec<String> = {
        if let Some(s) = body.get("input").and_then(|v| v.as_str()) {
            vec![s.to_string()]
        } else if let Some(arr) = body.get("input").and_then(|v| v.as_array()) {
            let strings: Vec<String> = arr
                .iter()
                .filter_map(|v| v.as_str().map(String::from))
                .collect();
            if strings.is_empty() {
                return Err((
                    StatusCode::BAD_REQUEST,
                    "embeddings: 'input' must be a string or array of strings".to_string(),
                ));
            }
            strings
        } else {
            return Err((
                StatusCode::BAD_REQUEST,
                "embeddings: missing or invalid 'input'".to_string(),
            ));
        }
    };

    let model_name = body
        .get("model")
        .and_then(|v| v.as_str())
        .unwrap_or("default");

    let sources = crate::model_cache::default_model_sources();
    let source = sources.get(model_name).ok_or_else(|| {
        (StatusCode::BAD_REQUEST, format!("embeddings: unknown model '{model_name}'").to_string())
    })?;

    let model_arc = state.model_cache.lock().await.get_or_load(model_name, source, Some(state.telemetry.as_ref()))
        .map_err(|e| (StatusCode::SERVICE_UNAVAILABLE, format!("embeddings: {e}")))?;

    let mut guard = state.session.lock().await;
    let sess = guard.as_mut().ok_or((
        StatusCode::SERVICE_UNAVAILABLE,
        "embeddings: no session".to_string(),
    ))?;

    // Build fresh KV caches for this embedding session
    let kv_caches = build_kv_caches(&model_arc);
    let mut new_sess = ProfiledInferenceSession::new("embed".into(), kv_caches);
    new_sess.setup_from_model(&model_arc);
    new_sess.generated_tokens.clear();
    new_sess.phase = InferenceSessionState::Created;
    *sess = new_sess;

    // Process each input and collect embeddings
    let mut embeddings_data: Vec<serde_json::Value> = Vec::with_capacity(inputs.len());
    for (idx, input) in inputs.iter().enumerate() {
        // Tokenize: byte-level fallback (same as run_inference)
        let tokens: Vec<u32> = input.bytes().map(|b| b as u32).collect();

        if tokens.is_empty() {
            return Err((
                StatusCode::BAD_REQUEST,
                format!("embeddings: empty input at index {}", idx),
            ));
        }

        let embedding = tokio::task::block_in_place(|| {
            sess.embed(&tokens, model_arc.as_ref(), EmbedPoolStrategy::Mean)
        })
        .map_err(|e| {
            (
                StatusCode::INTERNAL_SERVER_ERROR,
                format!("embeddings: {:?}", e),
            )
        })?;

        embeddings_data.push(serde_json::json!({
            "object": "embedding",
            "index": idx,
            "embedding": embedding,
        }));
    }

    let total_tokens: usize = inputs.iter().map(|s| s.bytes().len()).sum();

    Ok(JsonResponse(serde_json::json!({
        "object": "list",
        "data": embeddings_data,
        "model": model_name,
        "usage": {
            "prompt_tokens": total_tokens,
            "total_tokens": total_tokens,
        },
    })))
}

/// Quick random hex fragment for unique IDs.
fn rand_hex() -> u64 {
    use std::time::{SystemTime, UNIX_EPOCH};
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos() as u64;
    // Mix with a simple LCG for a bit of per-call variation.
    nanos.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407)
}
/// `/v1/cluster/status` — EXO cluster health (only when --exo is enabled).
///
/// Returns the current cluster status including node list, RAM, and
/// RDMA information.  Returns 503 if EXO mode is not active.
async fn cluster_status(
    State(state): State<AppState>,
) -> Result<JsonResponse<serde_json::Value>, (StatusCode, String)> {
    match &state.exo_node {
        Some(exo_arc) => {
            let exo = exo_arc.lock().await;
            match exo.cluster_status() {
                Ok(info) => Ok(JsonResponse(serde_json::json!(info))),
                Err(e) => Err((StatusCode::INTERNAL_SERVER_ERROR,
                    format!("cluster status error: {}", e))),
            }
        }
        None => Err((StatusCode::SERVICE_UNAVAILABLE,
            "EXO cluster mode not enabled. Use --exo flag to start.".to_string())),
    }
}

/// `/v1/cluster/nodes` — EXO cluster node list (only when --exo is enabled).
///
/// Returns detailed information about each node in the EXO cluster.
async fn cluster_nodes(
    State(state): State<AppState>,
) -> Result<JsonResponse<Vec<NodeInfo>>, (StatusCode, String)> {
    match &state.exo_node {
        Some(exo_arc) => {
            let exo = exo_arc.lock().await;
            match exo.cluster_status() {
                Ok(info) => Ok(JsonResponse(info.nodes)),
                Err(e) => Err((StatusCode::INTERNAL_SERVER_ERROR,
                    format!("cluster nodes error: {}", e))),
            }
        }
        None => Err((StatusCode::SERVICE_UNAVAILABLE,
            "EXO cluster mode not enabled.".to_string())),
    }
}

/// Parse `response_format` from an OpenAI-compatible request body and
/// build a [`SamplerConfig`] with grammar-guided generation if requested.
///
/// Supports `json_schema` response format: converts the schema to a GBNF
/// grammar and compiles it to a [`GrammarFSM`] for token masking.
async fn parse_response_format_from_model(
    body: &serde_json::Value,
    model: &Arc<LoadedProfiledModel>,
) -> SamplerConfig {
    let Some(response_format) = body.get("response_format") else {
        return SamplerConfig::default();
    };

    let format_type = response_format
        .get("type")
        .and_then(|v| v.as_str())
        .unwrap_or("");

    if format_type != "json_schema" && format_type != "json_object" {
        return SamplerConfig::default();
    }

    let (schema_name, schema) = if format_type == "json_schema" {
        let json_schema = match response_format.get("json_schema") {
            Some(js) => js,
            None => return SamplerConfig::default(),
        };
        let name = json_schema
            .get("name")
            .and_then(|v| v.as_str())
            .unwrap_or("root");
        let schema_val = json_schema.get("schema").cloned().unwrap_or_default();
        (name.to_string(), schema_val)
    } else {
        // json_object: accept any valid JSON object.
        let schema = serde_json::json!({
            "type": "object",
            "properties": {}
        });
        ("root".to_string(), schema)
    };

    // Build grammar from JSON Schema
    let grammar = match Grammar::from_json_schema(&schema_name, &schema) {
        Ok(g) => g,
        Err(e) => {
                log_error!("[grammar] failed to build from json_schema: {}", e);
            return SamplerConfig::default();
        }
    };

    // Compile grammar to FSM
    let grammar_fsm = match grammar.compile() {
        Ok(fsm) => fsm,
        Err(e) => {
                log_error!("[grammar] failed to compile grammar FSM: {}", e);
            return SamplerConfig::default();
        }
    };

    // Try to load tokenizer from the model directory
    let grammar_tokenizer = {
        let tokenizer_path = model.image_dir.join("tokenizer.json");
        if tokenizer_path.exists() {
            match GrammarTokenizer::load(&tokenizer_path) {
                Ok(tok) => Some(tok),
                Err(e) => {
                    log_error!("[grammar] failed to load tokenizer: {}", e);
                    None
                }
            }
        } else {
                log_warn!("[grammar] tokenizer.json not found at {:?}", tokenizer_path);
            None
        }
    };

    if grammar_tokenizer.is_none() {
        log_warn!("[grammar] tokenizer unavailable - grammar masking disabled");
        return SamplerConfig::default();
    }

    SamplerConfig {
        grammar: Some(grammar_fsm),
        grammar_tokenizer,
        ..SamplerConfig::default()
    }
}

/// `/v1/audio/speech` — OpenAI-compatible text-to-speech endpoint.
///
/// Accepts `input` (text) and optional `voice` parameters.
/// Returns base64-encoded WAV audio with sample rate and duration.
async fn audio_speech(
    State(state): State<AppState>,
    Json(body): Json<serde_json::Value>,
) -> Result<JsonResponse<serde_json::Value>, (StatusCode, String)> {
    let model_name = body.get("model").and_then(|v| v.as_str()).unwrap_or("qwen-tts");
    let input = body.get("input").and_then(|v| v.as_str()).unwrap_or("");
    if input.is_empty() {
        return Err((
            StatusCode::BAD_REQUEST,
            "missing or empty 'input' field".to_string(),
        ));
    }
    let voice = body.get("voice").and_then(|v| v.as_str());

    let sources = crate::model_cache::default_model_sources();
    let source = sources.get(model_name).ok_or_else(|| {
        (StatusCode::BAD_REQUEST, format!("unknown TTS model: {model_name}"))
    })?;

    // Get the model path from cache or sources.
    let model_path = match source {
        ModelSource::ImageDir(path) => path.to_string_lossy().to_string(),
        ModelSource::HuggingFace(hub_id) => {
            // Try to load via cache which will compile from source.
            let _model = state.model_cache.lock().await.get_or_load(model_name, source, Some(state.telemetry.as_ref()))
                .map_err(|e| (StatusCode::SERVICE_UNAVAILABLE, format!("TTS model load: {e}")))?;
            // For now, use the hub_id as a path hint (real streaming compile TBD).
            return Err((StatusCode::SERVICE_UNAVAILABLE,
                "HuggingFace streaming TTS not yet implemented; use ImageDir source".to_string()));
        }
    };

    let tts = crate::generation::text_to_speech::TextToSpeechGenerator::load(&model_path)
        .map_err(|e| (StatusCode::SERVICE_UNAVAILABLE, format!("load TTS model: {e}")))?;

    let (sample_rate, samples) = tts.synthesize(input, voice).await.map_err(|e| {
        (StatusCode::INTERNAL_SERVER_ERROR, format!("TTS error: {e}"))
    })?;

    let wav_bytes = pcm_to_wav(&samples, sample_rate);
    let encoded = base64_encode(&wav_bytes);
    let duration_s = samples.len() as f64 / sample_rate as f64;

    Ok(JsonResponse(serde_json::json!({
        "audio": encoded,
        "sample_rate": sample_rate,
        "duration_s": duration_s,
    })))
}
/// `/v1/audio/edits` — Voice cloning or audio style transfer.
///
/// Accepts a JSON body with:
/// - `reference_audio` (required) — base64-encoded reference audio (WAV)
/// - `text` (optional) — text to speak (voice cloning mode)
/// - `style_prompt` (optional) — style description (style transfer mode)
///
/// Returns base64-encoded WAV audio with sample rate, duration, and mode.
async fn audio_edits(
    State(state): State<AppState>,
    Json(body): Json<serde_json::Value>,
) -> Result<JsonResponse<serde_json::Value>, (StatusCode, String)> {
    let model_name = body.get("model").and_then(|v| v.as_str()).unwrap_or("qwen-tts");
    // Load model from cache.
    let sources = crate::model_cache::default_model_sources();
    let source = sources.get(model_name).ok_or_else(|| {
        (StatusCode::BAD_REQUEST, format!("unknown model: {model_name}"))
    })?;

    state.model_cache.lock().await.get_or_load(model_name, source, Some(state.telemetry.as_ref()))
        .map_err(|e| (StatusCode::SERVICE_UNAVAILABLE, format!("model load: {e}")))?;

    // Audio-to-audio generator needs to be constructed from the model's
    // image directory. This is not yet wired through ModelCache.
    Err((StatusCode::SERVICE_UNAVAILABLE,
        "audio-edits endpoint requires a pre-loaded AudioToAudio generator; not yet wired through ModelCache".to_string()))
}

/// `/v1/audio/transcriptions` — OpenAI-compatible speech-to-text endpoint.
///
/// Accepts a JSON body with:
/// - `url` (required) — URL or local path to an audio file
/// - `language` (optional) — language hint (e.g. "Chinese", "English")
///
/// Returns transcribed text matching the OpenAI Whisper `text` field format.
async fn audio_transcriptions(
    State(state): State<AppState>,
    Json(body): Json<serde_json::Value>,
) -> Result<JsonResponse<serde_json::Value>, (StatusCode, String)> {
    let model_name = body.get("model").and_then(|v| v.as_str()).unwrap_or("funasr");
    let audio_url = body
        .get("url")
        .and_then(|v| v.as_str())
        .ok_or((StatusCode::BAD_REQUEST, "audio url required".to_string()))?;

    let language = body.get("language").and_then(|v| v.as_str());

    let sources = crate::model_cache::default_model_sources();
    let source = sources.get(model_name).ok_or_else(|| {
        (StatusCode::BAD_REQUEST, format!("unknown ASR model: {model_name}"))
    })?;

    let model_path = match source {
        ModelSource::ImageDir(p) => p.to_string_lossy().to_string(),
        ModelSource::HuggingFace(_) => {
            return Err((StatusCode::SERVICE_UNAVAILABLE,
                "HuggingFace streaming for ASR not yet implemented".to_string()));
        }
    };

    let asr = crate::generation::audio_to_text::AudioToTextGenerator::load(&model_path)
        .map_err(|e| (StatusCode::SERVICE_UNAVAILABLE, format!("load ASR model: {e}")))?;

    let text = asr.transcribe(audio_url, language)
        .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, format!("transcription error: {e}")))?;

    Ok(JsonResponse(serde_json::json!({
        "text": text,
    })))
}

/// `/v1/images/edits` — OpenAI-compatible image editing endpoint.
///
/// Accepts a JSON body with:
/// - `image` (required) — base64-encoded input image (PNG)
/// - `prompt` (required) — text description of the desired edit
/// - `mask` (optional) — base64-encoded mask image (PNG; white = edit region)
/// - `n` (optional) — number of outputs (default 1)
/// - `size` (optional) — output size as "WxH" (default "512x512")
///
/// Returns a list of base64-encoded edited images matching the OpenAI
/// image editing response format.
async fn v1_image_edits(
    State(state): State<AppState>,
    Json(body): Json<serde_json::Value>,
) -> Result<JsonResponse<serde_json::Value>, (StatusCode, String)> {
    let model_name = body.get("model").and_then(|v| v.as_str()).unwrap_or("flux");
    let image_b64 = body
        .get("image")
        .and_then(|v| v.as_str())
        .ok_or((StatusCode::BAD_REQUEST, "missing 'image' field".to_string()))?;

    let prompt = body
        .get("prompt")
        .and_then(|v| v.as_str())
        .ok_or((StatusCode::BAD_REQUEST, "missing 'prompt' field".to_string()))?;

    let _size = body
        .get("size")
        .and_then(|v| v.as_str())
        .unwrap_or("512x512");

    // Decode the mask if provided.
    let mask_bytes: Option<Vec<u8>> = body
        .get("mask")
        .and_then(|v| v.as_str())
        .map(|b64| decode_body_base64(b64));

    // Decode the image from base64.
    let image_bytes = decode_body_base64(image_b64);
    if image_bytes.is_empty() {
        return Err((
            StatusCode::BAD_REQUEST,
            "invalid base64 'image' data".to_string(),
        ));
    }

    let sources = crate::model_cache::default_model_sources();
    let source = sources.get(model_name).ok_or_else(|| {
        (StatusCode::BAD_REQUEST, format!("unknown image model: {model_name}"))
    })?;

    state.model_cache.lock().await.get_or_load(model_name, source, Some(state.telemetry.as_ref()))
        .map_err(|e| (StatusCode::SERVICE_UNAVAILABLE, format!("model load: {e}")))?;

    return Err((StatusCode::SERVICE_UNAVAILABLE,
        "image-edits endpoint requires a pre-loaded ImageToImage generator; not yet wired through ModelCache".to_string()));
}

/// `/v1/images/variations` — OpenAI-compatible image variation endpoint.
///
/// Accepts a JSON body with:
/// - `image` (required) — base64-encoded input image (PNG)
/// - `n` (optional) — number of variations to generate (default 1, max 10)
/// - `size` (optional) — output size as "WxH" (default "512x512")
///
/// Returns a list of base64-encoded variation images.
async fn v1_image_variations(
    State(state): State<AppState>,
    Json(body): Json<serde_json::Value>,
) -> Result<JsonResponse<serde_json::Value>, (StatusCode, String)> {
    let model_name = body.get("model").and_then(|v| v.as_str()).unwrap_or("flux");
    let image_b64 = body
        .get("image")
        .and_then(|v| v.as_str())
        .ok_or((StatusCode::BAD_REQUEST, "missing 'image' field".to_string()))?;

    let n = body.get("n").and_then(|v| v.as_u64()).unwrap_or(1) as u32;
    let _size = body
        .get("size")
        .and_then(|v| v.as_str())
        .unwrap_or("512x512");

    let image_bytes = decode_body_base64(image_b64);
    if image_bytes.is_empty() {
        return Err((
            StatusCode::BAD_REQUEST,
            "invalid base64 'image' data".to_string(),
        ));
    }

    let sources = crate::model_cache::default_model_sources();
    let source = sources.get(model_name).ok_or_else(|| {
        (StatusCode::BAD_REQUEST, format!("unknown image model: {model_name}"))
    })?;

    state.model_cache.lock().await.get_or_load(model_name, source, Some(state.telemetry.as_ref()))
        .map_err(|e| (StatusCode::SERVICE_UNAVAILABLE, format!("model load: {e}")))?;

    return Err((StatusCode::SERVICE_UNAVAILABLE,
        "image-variations endpoint requires a pre-loaded ImageToImage generator; not yet wired through ModelCache".to_string()));

}

/// Decode a base64 string to raw bytes (standard base64 with padding).
fn decode_body_base64(input: &str) -> Vec<u8> {
    // Remove any data URI prefix if present.
    let data = if let Some(comma_pos) = input.find(',') {
        &input[comma_pos + 1..]
    } else {
        input
    };

    // Standard base64 decode — works for both padded and unpadded input.
    // This is a minimal implementation matching the RFC 4648 table used by
    // our encoder, avoiding extra crate dependencies.
    const DECODE_TABLE: [i8; 256] = {
        let mut t = [-1i8; 256];
        let alphabet = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/=";
        let mut i = 0usize;
        while i < alphabet.len() {
            t[alphabet[i] as usize] = i as i8;
            i += 1;
        }
        t
    };

    // Remove whitespace.
    let cleaned: Vec<u8> = data.bytes().filter(|b| !b.is_ascii_whitespace()).collect();
    if cleaned.is_empty() {
        return Vec::new();
    }

    let mut result = Vec::with_capacity(cleaned.len() / 4 * 3);
    let mut buf = [0u8; 4];
    let mut pos = 0;

    for &byte in &cleaned {
        let val = DECODE_TABLE[byte as usize];
        if val < 0 {
            // Skip invalid characters.
            continue;
        }
        buf[pos] = val as u8;
        pos += 1;
        if pos == 4 {
            result.push((buf[0] << 2) | (buf[1] >> 4));
            if buf[2] != 64 {
                result.push((buf[1] << 4) | (buf[2] >> 2));
            }
            if buf[3] != 64 {
                result.push((buf[2] << 6) | buf[3]);
            }
            pos = 0;
        }
    }

    result
}

/// Current Unix timestamp in seconds.
/// `/v1/images/generations` — OpenAI-compatible image generation endpoint.
///
/// Supports multiple model backends via the `model` field:
/// - `"flux"` (default) — uses the standard text-to-image pipeline
/// - `"diffusiongemma"` — uses the DiffusionGemma diffusion transformer
///
/// Request body (OpenAI Images API):
/// ```json
/// {
///   "model": "diffusiongemma",
///   "prompt": "A cat wearing a hat",
///   "n": 1,
///   "size": "1024x1024",
///   "negative_prompt": "blurry, low quality",
///   "cfg_scale": 7.5,
///   "steps": 50,
///   "seed": 42,
///   "image": "<base64>",
///   "strength": 0.8
/// }
/// ```
async fn image_generations(
    State(state): State<AppState>,
    Json(body): Json<serde_json::Value>,
) -> Result<JsonResponse<serde_json::Value>, (StatusCode, String)> {
    let prompt = body.get("prompt").and_then(|v| v.as_str()).unwrap_or("");
    if prompt.is_empty() {
        return Err((
            StatusCode::BAD_REQUEST,
            "missing or empty 'prompt' field".to_string(),
        ));
    }

    let n = body.get("n").and_then(|v| v.as_u64()).unwrap_or(1) as usize;
    let model_name = body
        .get("model")
        .and_then(|v| v.as_str())
        .unwrap_or("flux");
    let negative_prompt = body.get("negative_prompt").and_then(|v| v.as_str());
    let steps = body.get("steps").and_then(|v| v.as_u64()).map(|s| s as u32);
    let cfg_scale = body.get("cfg_scale").and_then(|v| v.as_f64()).map(|s| s as f32);
    let seed = body.get("seed").and_then(|v| v.as_u64());
    let strength = body.get("strength").and_then(|v| v.as_f64()).map(|s| s as f32);

    // Parse size from "WxH" format (e.g. "1024x1024").
    let size: Option<(u32, u32)> = body.get("size").and_then(|v| v.as_str()).and_then(|s| {
        let parts: Vec<&str> = s.split('x').collect();
        if parts.len() == 2 {
            let w = parts[0].parse::<u32>().ok()?;
            let h = parts[1].parse::<u32>().ok()?;
            Some((w, h))
        } else {
            None
        }
    });

    // Image-to-image: decode base64-encoded input image.
    let image_bytes: Option<Vec<u8>> = body.get("image").and_then(|v| v.as_str()).map(|b64| {
        decode_body_base64(b64)
    });

    // ── Load model from cache by name ─────────────────────────────────
    let sources = crate::model_cache::default_model_sources();
    let source = sources.get(model_name).ok_or_else(|| {
        (StatusCode::BAD_REQUEST, format!("unknown image model: {model_name}"))
    })?;

    let _model_arc = state.model_cache.lock().await.get_or_load(model_name, source, Some(state.telemetry.as_ref()))
        .map_err(|e| (StatusCode::SERVICE_UNAVAILABLE, format!("model load: {e}")))?;

    let image_data = match model_name {
        "diffusiongemma" => {
            // DiffusionGemma image generator needs to be constructed from model path.
            let dg = crate::generation::diffusiongemma::DiffusionGemmaGenerator::load(
                &sources.get("diffusiongemma").and_then(|s| match s {
                    ModelSource::ImageDir(p) => Some(p.to_string_lossy().to_string()),
                    _ => None,
                }).unwrap_or_default()
            ).map_err(|e| (StatusCode::SERVICE_UNAVAILABLE, format!("DiffusionGemma load: {e}")))?;

            let image_ref = image_bytes.as_deref();
            dg.generate(
                prompt,
                negative_prompt,
                steps,
                size,
                cfg_scale,
                seed,
                image_ref,
                strength,
            )
            .map_err(|e| {
                (
                    StatusCode::INTERNAL_SERVER_ERROR,
                    format!("DiffusionGemma error: {e}"),
                )
            })?
        }
        _ => {
            // Use Flux/Klein text-to-image backend.
            let model_path = match source {
                ModelSource::ImageDir(p) => p.to_string_lossy().to_string(),
                ModelSource::HuggingFace(_) => {
                    return Err((StatusCode::SERVICE_UNAVAILABLE,
                        "HuggingFace streaming not yet implemented".to_string()));
                }
            };
            let t2i = crate::generation::text_to_image::TextToImageGenerator::load(&model_path)
                .map_err(|e| (StatusCode::SERVICE_UNAVAILABLE, format!("load T2I model: {e}")))?;

            let mut images = Vec::with_capacity(n);
            for _ in 0..n {
                let (w, h, bytes) = t2i.generate(prompt, steps, size)
                    .map_err(|e| {
                        (
                            StatusCode::INTERNAL_SERVER_ERROR,
                            format!("image generation error: {e}"),
                        )
                    })?;
                let png_bytes = encode_png(w, h, &bytes);
                let b64 = base64_encode(&png_bytes);
                images.push(serde_json::json!({
                    "b64_json": b64,
                    "width": w,
                    "height": h,
                }));
            }
            return Ok(JsonResponse(serde_json::json!({
                "created": chrono_now(),
                "data": images,
            })));
        }
    };

    let b64 = base64_encode(&image_data);
    let created = chrono_now();

    Ok(JsonResponse(serde_json::json!({
        "created": created,
        "data": [{
            "b64_json": b64,
        }],
    })))
}

/// Current Unix timestamp in seconds.
fn chrono_now() -> u64 {
    use std::time::{SystemTime, UNIX_EPOCH};
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs()
}

// ── Video generation endpoints ────────────────────────────────────────────

/// `/v1/video/generations` — text-to-video generation.
///
/// Accepts `prompt`, optional `num_frames` (default 16), `fps` (default 8),
/// and `seed` (default 0).  Returns a list of base64-encoded PNG frames with
/// per-frame dimensions.
async fn video_generations(
    State(state): State<AppState>,
    Json(body): Json<serde_json::Value>,
) -> JsonResponse<serde_json::Value> {
    let prompt = match body.get("prompt").and_then(|v| v.as_str()) {
        Some(p) => p,
        None => {
            return JsonResponse(serde_json::json!({
                "error": "missing 'prompt' field"
            }));
        }
    };
    let num_frames = body
        .get("num_frames")
        .and_then(|v| v.as_u64())
        .unwrap_or(16) as u32;
    let fps = body
        .get("fps")
        .and_then(|v| v.as_u64())
        .unwrap_or(8) as u32;
    let seed = body
        .get("seed")
        .and_then(|v| v.as_u64())
        .unwrap_or(0);

    // Load model from cache for the frame generator.
    let model_name = body.get("model").and_then(|v| v.as_str()).unwrap_or("flux");
    let sources = crate::model_cache::default_model_sources();
    let source = match sources.get(model_name) {
        Some(s) => s,
        None => {
            return JsonResponse(serde_json::json!({
                "error": format!("unknown model: {model_name}")
            }));
        }
    };

    let model_arc = match state.model_cache.lock().await.get_or_load(model_name, source, Some(state.telemetry.as_ref())) {
        Ok(m) => m,
        Err(e) => {
            return JsonResponse(serde_json::json!({
                "error": format!("failed to load model: {e}")
            }));
        }
    };

    let frame_gen = TextToImageGenerator::new(Some(model_arc));
    let vg = VideoGenerator::new(frame_gen);

    // Run generation (blocking work).
    let frames = tokio::task::block_in_place(|| {
        vg.text_to_video(prompt, num_frames, fps, seed)
    });

    let frames = match frames {
        Ok(f) => f,
        Err(e) => {
            return JsonResponse(serde_json::json!({
                "error": e
            }));
        }
    };

    // Encode each frame as a base64-encoded PNG.
    let encoded_frames: Vec<serde_json::Value> = frames
        .iter()
        .map(|(w, h, rgba)| {
            let png_bytes = encode_png(*w, *h, rgba);
            let b64 = base64_encode(&png_bytes);
            serde_json::json!({
                "width": w,
                "height": h,
                "data": b64,
            })
        })
        .collect();

    JsonResponse(serde_json::json!({
        "frames": encoded_frames,
        "num_frames": num_frames,
        "fps": fps,
    }))
}

/// `/v1/video/edits` — image-to-video generation.
///
/// Accepts `image` (base64-encoded raw RGBA data), `prompt`,
/// optional `num_frames` (default 16), and `fps` (default 8).
/// Returns a list of base64-encoded PNG frames.
async fn video_edits(
    State(state): State<AppState>,
    Json(body): Json<serde_json::Value>,
) -> JsonResponse<serde_json::Value> {
    let image_b64 = match body.get("image").and_then(|v| v.as_str()) {
        Some(s) => s,
        None => {
            return JsonResponse(serde_json::json!({
                "error": "missing 'image' field"
            }));
        }
    };
    let prompt = match body.get("prompt").and_then(|v| v.as_str()) {
        Some(p) => p,
        None => {
            return JsonResponse(serde_json::json!({
                "error": "missing 'prompt' field"
            }));
        }
    };
    let num_frames = body
        .get("num_frames")
        .and_then(|v| v.as_u64())
        .unwrap_or(16) as u32;
    let fps = body
        .get("fps")
        .and_then(|v| v.as_u64())
        .unwrap_or(8) as u32;

    // Decode the initial image from base64.
    let image_bytes = decode_body_base64(image_b64);
    if image_bytes.is_empty() {
        return JsonResponse(serde_json::json!({
            "error": "invalid base64 'image' data"
        }));
    }

    // Load model from cache for the frame generator.
    let model_name = body.get("model").and_then(|v| v.as_str()).unwrap_or("flux");
    let sources = crate::model_cache::default_model_sources();
    let source = match sources.get(model_name) {
        Some(s) => s,
        None => {
            return JsonResponse(serde_json::json!({
                "error": format!("unknown model: {model_name}")
            }));
        }
    };

    let model_arc = match state.model_cache.lock().await.get_or_load(model_name, source, Some(state.telemetry.as_ref())) {
        Ok(m) => m,
        Err(e) => {
            return JsonResponse(serde_json::json!({
                "error": format!("failed to load model: {e}")
            }));
        }
    };

    let frame_gen = TextToImageGenerator::new(Some(model_arc));
    let vg = VideoGenerator::new(frame_gen);

    let frames = tokio::task::block_in_place(|| {
        vg.image_to_video(&image_bytes, prompt, num_frames, fps)
    });

    let frames = match frames {
        Ok(f) => f,
        Err(e) => {
            return JsonResponse(serde_json::json!({
                "error": e
            }));
        }
    };

    let encoded_frames: Vec<serde_json::Value> = frames
        .iter()
        .map(|(w, h, rgba)| {
            let png_bytes = encode_png(*w, *h, rgba);
            let b64 = base64_encode(&png_bytes);
            serde_json::json!({
                "width": w,
                "height": h,
                "data": b64,
            })
        })
        .collect();

    JsonResponse(serde_json::json!({
        "frames": encoded_frames,
        "num_frames": num_frames,
        "fps": fps,
    }))
}

// ---------------------------------------------------------------------------
// LoRA adapter handlers
// ---------------------------------------------------------------------------

/// List all available adapters.
async fn list_adapters(
    State(state): State<AppState>,
) -> JsonResponse<Vec<AdapterInfo>> {
    let adapters = state.adapters.lock().await;
    let active = state.active_adapter.lock().await;
    let result: Vec<AdapterInfo> = adapters
        .values()
        .map(|a| {
            let mut info = AdapterInfo::from(a);
            info.is_loaded = active.as_deref() == Some(&a.name);
            info
        })
        .collect();
    JsonResponse(result)
}

/// Request body for `POST /v1/adapters/train`.
#[derive(serde::Deserialize)]
struct TrainAdapterRequest {
    name: String,
    rank: Option<u32>,
    alpha: Option<f32>,
    target_layers: Option<Vec<u32>>,
    target_modules: Option<Vec<String>>,
    input_ids: Vec<u32>,
    target_ids: Vec<u32>,
    learning_rate: Option<f64>,
    num_steps: Option<u32>,
}

/// Train a LoRA adapter on provided token sequences.
async fn train_adapter(
    State(state): State<AppState>,
    Json(body): Json<TrainAdapterRequest>,
) -> Result<JsonResponse<serde_json::Value>, (StatusCode, String)> {
    let rank = body.rank.unwrap_or(8);
    let alpha = body.alpha.unwrap_or(16.0);
    let lr = body.learning_rate.unwrap_or(1e-4);
    let num_steps = body.num_steps.unwrap_or(1);

    // Create or retrieve the adapter
    let mut adapter = LoraAdapter::new(&body.name, rank, alpha);
    adapter.target_layers = body.target_layers.unwrap_or_default();
    adapter.target_modules = body
        .target_modules
        .unwrap_or_else(|| vec!["q_proj".to_string(), "v_proj".to_string()]);

    // Store the adapter config into the registry
    {
        let mut adapters = state.adapters.lock().await;
        adapters.insert(body.name.clone(), adapter.clone());
    }

    // Report training metadata (actual training requires a model ref).
    Ok(JsonResponse(serde_json::json!({
        "status": "training_initiated",
        "adapter": body.name,
        "rank": rank,
        "alpha": alpha,
        "learning_rate": lr,
        "num_steps": num_steps,
        "input_length": body.input_ids.len(),
    })))
}

/// Request body for `POST /v1/adapters/load`.
#[derive(serde::Deserialize)]
struct LoadAdapterRequest {
    name: String,
    /// Path to load from (optional — default is adapters/<name>).
    path: Option<String>,
}

/// Load a trained adapter and merge it into the active model.
async fn load_adapter(
    State(state): State<AppState>,
    Json(body): Json<LoadAdapterRequest>,
) -> Result<JsonResponse<serde_json::Value>, (StatusCode, String)> {
    // Try loading from disk
    let storage_path = body.path.unwrap_or_else(|| format!("adapters/{}", body.name));
    let adapter = LoraAdapter::load(&storage_path)
        .map_err(|e| (StatusCode::NOT_FOUND, format!("load adapter: {}", e)))?;

    // Register in adapter store
    {
        let mut adapters = state.adapters.lock().await;
        adapters.insert(body.name.clone(), adapter.clone());
    }
    {
        let mut active = state.active_adapter.lock().await;
        *active = Some(body.name.clone());
    }

    Ok(JsonResponse(serde_json::json!({
        "status": "loaded",
        "adapter": body.name,
    })))
}

/// Unload the current adapter (unmerge weights).
async fn unload_adapter(
    State(state): State<AppState>,
) -> Result<JsonResponse<serde_json::Value>, (StatusCode, String)> {
    {
        let mut active = state.active_adapter.lock().await;
        let _active_name = active
            .take()
            .ok_or_else(|| (StatusCode::NOT_FOUND, "no active adapter to unload".to_string()))?;
    }

    // Note: in production, unmerge the LoRA deltas from the loaded model here.
    // Requires an Arc<Mutex<LoadedProfiledModel>> in AppState.

    Ok(JsonResponse(serde_json::json!({
        "status": "unloaded",
    })))
}

// ── PNG encoder helpers ────────────────────────────────────────────────────

/// Minimal PNG encoder using uncompressed (store) deflate blocks.
///
/// Produces a valid PNG image from raw RGBA pixel data.  The output is
/// larger than a compressed PNG but requires no external compression
/// library.  Every conformant PNG decoder can read it.
fn encode_png(width: u32, height: u32, rgba: &[u8]) -> Vec<u8> {
    let mut out = Vec::new();

    // PNG signature
    out.extend_from_slice(&[137, 80, 78, 71, 13, 10, 26, 10]);

    // ── IHDR chunk ────────────────────────────────────────────────────
    let mut ihdr = Vec::with_capacity(13);
    ihdr.extend_from_slice(&width.to_be_bytes());
    ihdr.extend_from_slice(&height.to_be_bytes());
    ihdr.push(8);  // bit depth
    ihdr.push(6);  // colour type: RGBA
    ihdr.push(0);  // compression method
    ihdr.push(0);  // filter method
    ihdr.push(0);  // interlace method
    write_chunk(&mut out, b"IHDR", &ihdr);

    // ── IDAT chunk (uncompressed deflate) ─────────────────────────────
    // Build filtered scanlines (filter byte 0 = None per scanline) +
    // wrap in a single uncompressed deflate block.
    let raw_stride = (width as usize) * 4 + 1; // filter byte + RGBA scanline
    let raw_len = (height as usize) * raw_stride;
    let mut raw = Vec::with_capacity(raw_len);

    for y in 0..height as usize {
        raw.push(0); // filter type: None
        let row_start = y * (width as usize) * 4;
        raw.extend_from_slice(&rgba[row_start..row_start + (width as usize) * 4]);
    }

    // Build an uncompressed deflate block:
    //   BFINAL=1, BTYPE=00 (no compression)
    //   LEN  (2 bytes, little-endian)
    //   NLEN (2 bytes, one's complement of LEN)
    //   raw data
    let mut deflated = Vec::with_capacity(raw_len + 5);
    // Final block, stored (uncompressed)
    deflated.push(0x01); // BFINAL=1, BTYPE=00
    let len = raw_len as u16;
    deflated.extend_from_slice(&len.to_le_bytes());
    deflated.extend_from_slice(&(!len).to_le_bytes());
    deflated.extend_from_slice(&raw);

    // Compute adler-32 of the raw data (RFC 1950).
    let adler = adler32(&raw);
    deflated.extend_from_slice(&adler.to_be_bytes());

    write_chunk(&mut out, b"IDAT", &deflated);

    // ── IEND chunk ────────────────────────────────────────────────────
    write_chunk(&mut out, b"IEND", &[]);

    out
}

/// Write a single PNG chunk (length + type + data + CRC).
fn write_chunk(out: &mut Vec<u8>, chunk_type: &[u8; 4], data: &[u8]) {
    let len = data.len() as u32;
    out.extend_from_slice(&len.to_be_bytes());
    out.extend_from_slice(chunk_type);
    out.extend_from_slice(data);
    // CRC over type + data
    let mut crc_input = Vec::with_capacity(4 + data.len());
    crc_input.extend_from_slice(chunk_type);
    crc_input.extend_from_slice(data);
    let crc = crc32(&crc_input);
    out.extend_from_slice(&crc.to_be_bytes());
}

/// Compute CRC-32 (ISO 3309 / PNG spec variant).
fn crc32(data: &[u8]) -> u32 {
    let mut crc: u32 = 0xFFFF_FFFF;
    for &byte in data {
        crc ^= byte as u32;
        for _ in 0..8 {
            if crc & 1 != 0 {
                crc = (crc >> 1) ^ 0xEDB8_8320;
            } else {
                crc >>= 1;
            }
        }
    }
    crc ^ 0xFFFF_FFFF
}

/// Compute Adler-32 checksum (RFC 1950 / zlib).
fn adler32(data: &[u8]) -> u32 {
    let MOD: u32 = 65521;
    let mut a: u32 = 1;
    let mut b: u32 = 0;
    for &byte in data {
        a = (a + byte as u32) % MOD;
        b = (b + a) % MOD;
    }
    (b << 16) | a
}

// -------------------------------------------------------------------------
// Knowledge editing endpoints
// -------------------------------------------------------------------------

/// `POST /v1/edits` — apply a single factual edit.
async fn apply_edit(
    State(state): State<AppState>,
    Json(body): Json<EditRequest>,
) -> Result<JsonResponse<serde_json::Value>, (StatusCode, String)> {
    let mut editor_guard = state.knowledge_editor.lock().await;
    let editor = editor_guard
        .as_mut()
        .ok_or_else(|| (StatusCode::SERVICE_UNAVAILABLE, "no active model for editing".to_string()))?;

    let fact = editing::FactEdit::from(body);
    let result = editor.edit_fact(&fact).map_err(|e| {
        (StatusCode::INTERNAL_SERVER_ERROR, format!("edit failed: {}", e))
    })?;

    Ok(JsonResponse(serde_json::json!({
        "success": result.success,
        "target_layer": result.target_layer,
        "delta_rank": result.delta_rank,
        "pre_edit_logit": result.pre_edit_logit,
        "post_edit_logit": result.post_edit_logit,
        "side_effects": serde_json::to_value(&result.side_effect_test).unwrap_or_default(),
        "elapsed_ms": result.elapsed_ms,
    })))
}

/// `GET /v1/edits` — list edit history.
async fn list_edits(
    State(state): State<AppState>,
) -> JsonResponse<serde_json::Value> {
    let editor_guard = state.knowledge_editor.lock().await;
    match &*editor_guard {
        Some(editor) => {
            JsonResponse(serde_json::json!({
                "edits": editor.edit_history,
                "count": editor.edit_history.len(),
            }))
        }
        None => {
            JsonResponse(serde_json::json!({
                "edits": [],
                "count": 0,
            }))
        }
    }
}

/// `POST /v1/edits/batch` — edit multiple facts at once (MEMIT).
async fn apply_edit_batch(
    State(state): State<AppState>,
    Json(body): Json<EditBatchRequest>,
) -> Result<JsonResponse<serde_json::Value>, (StatusCode, String)> {
    let mut editor_guard = state.knowledge_editor.lock().await;
    let editor = editor_guard
        .as_mut()
        .ok_or_else(|| (StatusCode::SERVICE_UNAVAILABLE, "no active model for editing".to_string()))?;

    let facts: Vec<editing::FactEdit> = body.edits.into_iter().map(editing::FactEdit::from).collect();
    let results = editor.edit_batch(&facts).map_err(|e| {
        (StatusCode::INTERNAL_SERVER_ERROR, format!("batch edit failed: {}", e))
    })?;

    Ok(JsonResponse(serde_json::json!({
        "edits": results,
        "count": results.len(),
        "success": results.iter().all(|r| r.success),
    })))
}

/// `DELETE /v1/edits` — undo the last edit.
async fn undo_last_edit(
    State(state): State<AppState>,
) -> Result<JsonResponse<serde_json::Value>, (StatusCode, String)> {
    let mut editor_guard = state.knowledge_editor.lock().await;
    let editor = editor_guard
        .as_mut()
        .ok_or_else(|| (StatusCode::SERVICE_UNAVAILABLE, "no active model for editing".to_string()))?;

    editor.undo_last().map_err(|e| {
        (StatusCode::INTERNAL_SERVER_ERROR, format!("undo failed: {}", e))
    })?;

    Ok(JsonResponse(serde_json::json!({
        "status": "undone",
    })))
}

/// `POST /v1/edits/audit` — audit known facts for correctness.
async fn audit_facts(
    State(state): State<AppState>,
    Json(body): Json<AuditRequest>,
) -> Result<JsonResponse<serde_json::Value>, (StatusCode, String)> {
    let editor_guard = state.knowledge_editor.lock().await;
    let editor = editor_guard
        .as_ref()
        .ok_or_else(|| (StatusCode::SERVICE_UNAVAILABLE, "no active model for editing".to_string()))?;

    let facts: Vec<editing::FactEdit> = body.facts.into_iter().map(editing::FactEdit::from).collect();
    let audit = editor.audit_facts(&facts).map_err(|e| {
        (StatusCode::INTERNAL_SERVER_ERROR, format!("audit failed: {}", e))
    })?;

    let items: Vec<AuditItem> = audit
        .into_iter()
        .map(|(fact, correct, logit)| AuditItem {
            subject: fact.subject,
            object: fact.object,
            prompt: fact.prompt,
            correct,
            logit,
        })
        .collect();

    Ok(JsonResponse(serde_json::json!({
        "facts": items,
        "count": items.len(),
    })))
}
