//! Candle CPU server — minimal inference server for the CPU backend.
//!
//! Unlike the MLX backend which uses a worker subprocess, the Candle CPU
//! backend runs eagerly in-process.  This module provides a minimal axum
//! server with health/readiness endpoints.

use std::sync::Arc;

use axum::{
    extract::State, http::StatusCode, response::Json as JsonResponse, routing::get, Router,
};
use tokio::sync::Mutex;

use crate::readiness_gates::ReadinessGates;
use crate::tokenizer::TribunusTokenizer;

// ---------------------------------------------------------------------------
// CpuAppState
// ---------------------------------------------------------------------------

/// Application state for the candle-cpu server.
#[derive(Clone)]
pub struct CpuAppState {
    /// Readiness gates — determines whether the server is ready.
    pub gates: Arc<Mutex<ReadinessGates>>,
    /// Loaded tokenizer, if available.
    pub tokenizer: Option<Arc<TribunusTokenizer>>,
}

// ---------------------------------------------------------------------------
// Routes
// ---------------------------------------------------------------------------

/// Health check — always responds 200 when the server is running.
async fn health() -> &'static str {
    "ok"
}

/// Readiness check — returns gate status and whether the server is ready.
async fn readiness(
    State(state): State<CpuAppState>,
) -> Result<JsonResponse<serde_json::Value>, StatusCode> {
    let gates = state.gates.lock().await;
    Ok(JsonResponse(serde_json::json!({
        "ready": gates.ready_for_inference(),
        "gates": {
            "summary": gates.summary(),
            "states": gates.gate_states(),
        }
    })))
}

/// Version endpoint.
async fn version() -> JsonResponse<serde_json::Value> {
    JsonResponse(serde_json::json!({
        "version": env!("CARGO_PKG_VERSION"),
        "backend": "candle-cpu",
    }))
}

/// Create the router for the candle-cpu server.
pub fn create_cpu_router(state: CpuAppState) -> Router {
    Router::new()
        .route("/health", get(health))
        .route("/readyz", get(readiness))
        .route("/version", get(version))
        .route("/v1/health", get(health))
        .with_state(state)
}
