//! Admin API endpoints — system status, session management, config reload.
//!
//! All admin endpoints are protected by the X-Admin-Key header, which must
//! match the `TRIBUNUS_ADMIN_KEY` environment variable.

use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use axum::{
    extract::Request,
    extract::{Path, State},
    http::StatusCode,
    middleware,
    middleware::Next,
    response::IntoResponse,
    response::Json as JsonResponse,
    routing::{get, post},
    Router,
};
use serde::Serialize;
use serde_json::json;
use tokio::sync::Mutex;

use crate::cache::evolkv::{CalibrationSet, EvolKV};
use crate::server::routes::AppState;
use crate::worker_supervisor::WorkerLifecyclePhase;
use crate::worker_supervisor::WorkerSupervisor;

// ---------------------------------------------------------------------------
// Request tracking
// ---------------------------------------------------------------------------

/// Information about an active (or recently completed) request.
#[derive(Debug, Clone, Serialize)]
pub struct ActiveRequestInfo {
    pub id: String,
    pub status: String,
    pub tokens_generated: u64,
    pub max_tokens: u64,
    pub model: String,
    pub elapsed_seconds: f64,
}

/// Shared request registry — maps request IDs to their current info.
pub type RequestRegistry = Arc<Mutex<HashMap<String, ActiveRequestInfo>>>;

/// Shared set of cancelled request IDs.
pub type CancelledSet = Arc<Mutex<HashSet<String>>>;

// ---------------------------------------------------------------------------
// Auth middleware
// ---------------------------------------------------------------------------

/// Admin auth: checks X-Admin-Key against TRIBUNUS_ADMIN_KEY env var.
///
/// Returns 401 Unauthorized if the header is missing or doesn't match.
pub async fn admin_auth(req: Request, next: Next) -> Result<impl IntoResponse, StatusCode> {
    let key = req
        .headers()
        .get("X-Admin-Key")
        .and_then(|v| v.to_str().ok())
        .unwrap_or("");
    if key != std::env::var("TRIBUNUS_ADMIN_KEY").unwrap_or_default() {
        return Err(StatusCode::UNAUTHORIZED);
    }
    Ok(next.run(req).await)
}

// ---------------------------------------------------------------------------
// Router
// ---------------------------------------------------------------------------

/// Build the admin API router with X-Admin-Key auth on all routes.
pub fn admin_router() -> Router<AppState> {
    Router::new()
        .route("/v1/admin/status", get(admin_status))
        .route("/v1/admin/sessions", get(admin_sessions))
        .route("/v1/admin/cancel/{id}", post(admin_cancel))
        .route("/v1/admin/reload", post(admin_reload))
        .route("/v1/admin/evolkv", post(admin_evolkv))
        .route_layer(middleware::from_fn(admin_auth))
}

// ---------------------------------------------------------------------------
// Handler: GET /v1/admin/status
// ---------------------------------------------------------------------------

/// Return a full system state dump.
async fn admin_status(State(state): State<AppState>) -> JsonResponse<serde_json::Value> {
    // ── Hardware ───────────────────────────────────────────────────────
    let hw = crate::scheduling::HardwareConfig::detect();
    let chip = format!("{} cores/{} GB", hw.cpu_cores, hw.total_ram_gb);
    let ram_mb = crate::gpu_memory::total_physical_ram_mb();

    // ── Models ─────────────────────────────────────────────────────────
    let models = state.models.lock().await;
    let model_list: Vec<serde_json::Value> = models
        .list()
        .iter()
        .map(|m| {
            json!({
                "id": m.id,
                "name": m.name,
                "parameter_size": m.parameter_size,
                "quantization": m.quantization,
                "is_loaded": m.is_loaded,
            })
        })
        .collect();
    drop(models);

    // ── Model cache ────────────────────────────────────────────────────
    let cache_info = {
        let cache = state.model_cache.lock().await;
        json!({
            "memory_status": cache.memory_status(),
            "entry_count": cache.entry_count(),
        })
    };

    // ── Benchmark ──────────────────────────────────────────────────────
    let bench_info = {
        let bench = state.benchmark.lock().await;
        match &*bench {
            Some(b) => json!({
                "completed": true,
                "chip": b.chip,
                "ram_gb": b.ram_gb,
                "ops_count": b.ops.len(),
                "recommend_accelerate_for": b.recommend_accelerate_for,
                "recommend_mlx_for": b.recommend_mlx_for,
            }),
            None => json!({"completed": false}),
        }
    };

    // ── Session ────────────────────────────────────────────────────────
    let sess_info = {
        match state.supervisor.as_ref() {
            Some(sup) => json!({
                "worker_pid": sup.process_ctrl.pid(),
                "alive": sup.process_ctrl.is_alive(),
                "active_requests": sup.registry.len(),
                "model_loaded": sup.runtime_state.phase() == WorkerLifecyclePhase::Ready,
                "faulted": sup.runtime_state.is_faulted(),
            }),
            None => json!({"worker": false}),
        }
    };

    // ── Telemetry ──────────────────────────────────────────────────────
    let telemetry = &state.telemetry;
    let telemetry_info = json!({
        "tokens_generated": telemetry.tokens_generated.load(std::sync::atomic::Ordering::Relaxed),
        "model_cache_hits": telemetry.model_cache_hits.load(std::sync::atomic::Ordering::Relaxed),
        "model_cache_misses": telemetry.model_cache_misses.load(std::sync::atomic::Ordering::Relaxed),
        "prefix_cache_hits": telemetry.prefix_cache_hits.load(std::sync::atomic::Ordering::Relaxed),
        "prefix_cache_misses": telemetry.prefix_cache_misses.load(std::sync::atomic::Ordering::Relaxed),
        "mlx_invocations": telemetry.mlx_invocations.load(std::sync::atomic::Ordering::Relaxed),
        "ane_invocations": telemetry.ane_invocations.load(std::sync::atomic::Ordering::Relaxed),
        "accelerate_invocations": telemetry.accelerate_invocations.load(std::sync::atomic::Ordering::Relaxed),
        "peak_memory_bytes": telemetry.peak_memory_bytes.load(std::sync::atomic::Ordering::Relaxed),
        "current_memory_bytes": telemetry.current_memory_bytes.load(std::sync::atomic::Ordering::Relaxed),
    });

    // ── EXO ────────────────────────────────────────────────────────────
    let exo_info = match &state.exo_node {
        Some(_) => json!({"enabled": true}),
        None => json!({"enabled": false}),
    };

    // ─── Adapters ──────────────────────────────────────────────────────
    let adapters = state.adapters.lock().await;
    let adapter_names: Vec<&String> = adapters.keys().collect();
    let active_adapter = state.active_adapter.lock().await;

    // ─── Rate limiter ──────────────────────────────────────────────────
    let rl_info = json!({
        "default_capacity": state.rate_limiter.default_capacity,
        "default_refill_rate": state.rate_limiter.default_refill_rate,
    });

    let response = json!({
        "status": "ok",
        "hardware": {
            "chip": chip,
            "ram_mb": ram_mb,
        },
        "models": {
            "entries": model_list,
            "count": model_list.len(),
        },
        "model_cache": cache_info,
        "benchmark": bench_info,
        "session": sess_info,
        "telemetry": telemetry_info,
        "exo": exo_info,
        "adapters": {
            "loaded": adapter_names,
            "active": active_adapter.as_deref(),
        },
        "rate_limiter": rl_info,
    });

    JsonResponse(response)
}

// ---------------------------------------------------------------------------
// Handler: GET /v1/admin/sessions
// ---------------------------------------------------------------------------

/// List active (and recently completed) requests with ID, status, tokens.
async fn admin_sessions(State(state): State<AppState>) -> JsonResponse<serde_json::Value> {
    // Collect from the request registry.
    let registry = state.admin_request_registry.lock().await;
    let sessions: Vec<serde_json::Value> = registry
        .iter()
        .map(|(id, info)| {
            json!({
                "id": id,
                "request_id": info.id,
                "status": info.status,
                "tokens_generated": info.tokens_generated,
                "max_tokens": info.max_tokens,
                "model": info.model,
                "elapsed_seconds": info.elapsed_seconds,
            })
        })
        .collect();
    drop(registry);
    let cancelled = state.admin_cancelled_requests.lock().await;

    // Report the worker state.
    let (worker_pid, worker_alive, worker_faulted) = state
        .supervisor
        .as_ref()
        .map(|s| {
            (
                s.process_ctrl.pid(),
                s.process_ctrl.is_alive(),
                s.runtime_state.is_faulted(),
            )
        })
        .unwrap_or((0, false, false));

    JsonResponse(json!({
        "sessions": sessions,
        "count": sessions.len(),
        "worker_pid": worker_pid,
        "worker_alive": worker_alive,
        "worker_faulted": worker_faulted,
        "cancelled_count": cancelled.len(),
    }))
}

// ---------------------------------------------------------------------------
// Handler: POST /v1/admin/cancel/{id}
// ---------------------------------------------------------------------------

/// Cancel a running request by adding its ID to the cancelled set.
async fn admin_cancel(
    State(state): State<AppState>,
    Path(id): Path<String>,
) -> JsonResponse<serde_json::Value> {
    // Add the ID to the cancelled set so running handlers can check it.
    {
        let mut cancelled = state.admin_cancelled_requests.lock().await;
        cancelled.insert(id.clone());
    }

    // Update the registry status if the request is tracked.
    {
        let mut registry = state.admin_request_registry.lock().await;
        if let Some(info) = registry.get_mut(&id) {
            info.status = "cancelling".into();
        }
    }

    JsonResponse(json!({
        "ok": true,
        "cancelled": id,
        "message": format!("Cancel signal sent for request '{}'", id),
    }))
}

// ---------------------------------------------------------------------------
// Handler: POST /v1/admin/reload
// ---------------------------------------------------------------------------

/// Hot-reload all watched model segments by processing pending filesystem
/// changes and re-scanning known loaded model image directories.
async fn admin_reload(State(state): State<AppState>) -> JsonResponse<serde_json::Value> {
    let mut cache = state.model_cache.lock().await;

    // Process any pending file-system reloads first.
    cache.process_pending_reloads();

    // Report the cache state after processing reloads.
    let entry_count = cache.entry_count();
    let memory_status = cache.memory_status();

    JsonResponse(json!({
        "ok": true,
        "processed_pending_reloads": true,
        "cache_entry_count": entry_count,
        "memory_status": memory_status,
    }))
}

// ---------------------------------------------------------------------------
// Handler: POST /v1/admin/evolkv
// ---------------------------------------------------------------------------

/// Trigger evolutionary KV budget search.
///
/// Accepts an optional JSON body with `model` to specify which loaded model
/// to use for layer count detection. Defaults to the first loaded model
/// from the cache (or a fallback of 32 layers).
async fn admin_evolkv(
    State(state): State<AppState>,
    body: axum::Json<serde_json::Value>,
) -> JsonResponse<serde_json::Value> {
    // Determine model name and number of layers.
    let model_name = body
        .get("model")
        .and_then(|v| v.as_str())
        .unwrap_or("gemma4");

    let num_layers: usize = {
        let mut cache = state.model_cache.lock().await;
        match cache.get(model_name) {
            Some(cached) => {
                let plan = &cached.model.reader.manifest.execution_plan;
                if plan.layers.is_empty() {
                    0
                } else {
                    plan.layers.len()
                }
            }
            None => 0,
        }
    };

    if num_layers == 0 {
        return JsonResponse(json!({
            "ok": false,
            "error": format!(
                "No loaded model '{}' with known layer count. Load a model first.",
                model_name
            ),
        }));
    }

    // Build a calibration set (short prompts for fast evaluation).
    let calibration = CalibrationSet::new(vec![
        vec![101, 202, 303, 404, 505],
        vec![111, 222, 333],
        vec![11, 22, 33, 44, 55, 66, 77],
        vec![1, 2, 3, 4],
        vec![1001, 1002, 1003, 1004, 1005, 1006],
    ]);

    // Run the evolutionary search.
    let evolkv = EvolKV {
        num_layers,
        population_size: 60,
        generations: 30,
        mutation_rate: 0.15,
        crossover_rate: 0.7,
        elitism_count: 3,
    };

    let total_budget = 4096; // total cache tokens per layer
    let result = evolkv.search(&calibration, total_budget);

    match result {
        Ok(budget) => JsonResponse(json!({
            "ok": true,
            "model": model_name,
            "num_layers": num_layers,
            "population_size": 60,
            "generations": 30,
            "budget": budget.fractions,
            "budget_sum": budget.fractions.iter().sum::<f64>(),
            "total_budget_tokens": total_budget,
        })),
        Err(e) => JsonResponse(json!({
            "ok": false,
            "error": format!("EvolKV search failed: {}", e),
        })),
    }
}
