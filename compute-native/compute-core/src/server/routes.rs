use axum::{
    extract::{Json, Path, State},
    response::Json as JsonResponse,
    routing::{get, post},
    Router,
};
use std::sync::Arc;
use tokio::sync::Mutex;

use crate::server::models::{ModelEntry, ModelRegistry};
use crate::server::benchmark::SystemBenchmark;

#[derive(Clone)]
pub struct AppState {
    pub models: Arc<Mutex<ModelRegistry>>,
    pub benchmark: Arc<Mutex<Option<SystemBenchmark>>>,
}

pub fn create_router(state: AppState) -> Router {
    Router::new()
        .route("/health", get(health))
        .route("/api/tags", get(list_models))
        .route("/api/models/{id}", get(get_model))
        .route("/api/benchmark", get(get_benchmark))
        .route("/api/chat/completions", post(chat_completions))
        .fallback(fallback)
        .with_state(state)
}

async fn fallback() -> (axum::http::StatusCode, &'static str) {
    (axum::http::StatusCode::NOT_FOUND, "Tribunus: route not found")
}

async fn health() -> JsonResponse<serde_json::Value> {
    JsonResponse(serde_json::json!({
        "status": "ok",
        "backend": "mlx",
        "accelerate": cfg!(target_os = "macos"),
    }))
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

async fn chat_completions(
    State(_state): State<AppState>,
    Json(body): Json<serde_json::Value>,
) -> Result<JsonResponse<serde_json::Value>, (axum::http::StatusCode, String)> {
    let _model = body.get("model").and_then(|v| v.as_str()).unwrap_or("default");
    let _messages = body.get("messages").and_then(|v| v.as_array());

    // TODO: dispatch to actual inference pipeline
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
