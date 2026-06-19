#[cfg(not(all(target_os = "macos", target_arch = "aarch64")))]
#[cfg(not(any(
    feature = "mlx-backend",
    feature = "candle-cpu",
    feature = "stub-backend",
    feature = "storage-adapters"
)))]
compile_error!(
    "Tribunus Compute requires a supported backend: Apple Silicon (macOS arm64), Candle CPU (Linux x86), or a stub/storage backend feature."
);

extern crate self as tribunus_compute_core;

#[cfg(feature = "mlx-backend")]
pub mod analysis;
#[cfg(feature = "mlx-backend")]
pub mod audio;
#[cfg(feature = "mlx-backend")]
pub mod autopsy;
#[cfg(all(target_os = "macos", feature = "mlx-backend"))]
pub mod ane;
#[cfg(all(target_os = "macos", feature = "mlx-backend"))]
pub mod ane_bridge;
#[cfg(all(target_os = "macos", feature = "mlx-backend"))]
pub mod arena;
#[cfg(target_os = "macos")]
pub mod arena_info;
#[cfg(target_os = "macos")]
pub mod arena_lifecycle;
#[cfg(all(target_os = "macos", feature = "mlx-backend"))]
pub mod arena_pool;
#[cfg(feature = "mlx-backend")]
pub mod attention;
pub mod backend;
#[cfg(all(target_os = "macos", feature = "mlx-backend"))]
pub mod bridge;
#[cfg(feature = "mlx-backend")]
pub mod cache;
#[cfg(feature = "mlx-backend")]
pub mod capability;
pub mod cli;
#[cfg(feature = "mlx-backend")]
pub mod compile_pipeline;
#[cfg(feature = "mlx-backend")]
pub mod compile_progress;
#[cfg(feature = "mlx-backend")]
pub mod compile_state;
#[cfg(feature = "mlx-backend")]
pub mod compiler;
#[cfg(feature = "mlx-backend")]
pub mod compute_image;
pub mod compute_image_v0;
pub mod compute_ir;
pub mod compute_lane;
pub mod compute_service;
#[cfg(feature = "mlx-backend")]
pub mod config;
#[cfg(feature = "mlx-backend")]
pub mod contracts;
pub mod crash_breadcrumb;
#[cfg(feature = "mlx-backend")]
pub mod copy_ledger;
#[cfg(all(target_os = "macos", feature = "mlx-backend"))]
pub mod coreml_audit;
#[cfg(all(target_os = "macos", feature = "mlx-backend"))]
pub mod coreml_bridge;
#[cfg(all(target_os = "macos", feature = "mlx-backend"))]
pub mod coreml_pipeline;
#[cfg(all(target_os = "macos", feature = "mlx-backend"))]
pub mod coreml_state;
#[cfg(feature = "mlx-backend")]
pub mod cpu_benchmarks;
#[cfg(feature = "mlx-backend")]
pub mod decode_attribution;
#[cfg(feature = "mlx-backend")]
pub mod engine;
pub mod engine_error;
pub mod engine_policy;
pub mod engine_receipts;
pub mod errors;
#[cfg(feature = "mlx-backend")]
pub mod executor;
pub mod experiment;
#[cfg(feature = "mlx-backend")]
pub mod external_array;
#[cfg(feature = "mlx-backend")]
pub mod generation;
pub mod fusion_region;
#[cfg(feature = "mlx-backend")]
pub mod gemma;
#[cfg(feature = "mlx-backend")]
pub mod grammar;
pub mod gpu_memory;
#[cfg(feature = "mlx-backend")]
pub mod gguf;
#[cfg(feature = "mlx-backend")]
pub mod gpu_worker;
#[cfg(feature = "mlx-backend")]
pub mod heterogeneous;
#[cfg(feature = "mlx-backend")]
pub mod hybrid_profile;
pub mod inference_profile;
#[cfg(feature = "mlx-backend")]
pub mod kv_cache;
#[cfg(feature = "mlx-backend")]
pub mod layout_compiler;
pub mod layout_transform;
#[cfg(feature = "mlx-backend")]
pub mod loader;
#[macro_use] pub mod logging;
#[cfg(feature = "mlx-backend")]
pub mod lora;
#[cfg(feature = "mlx-backend")]
pub mod mapped_image;
#[cfg(feature = "mlx-backend")]
pub mod memory;
pub mod metrics;
#[cfg(all(target_os = "macos", feature = "mlx-backend"))]
pub mod metal_capture;
#[cfg(feature = "mlx-backend")]
pub mod mil_builder;
#[cfg(feature = "mlx-backend")]
pub mod mlpackage;
pub mod plugin;
#[cfg(feature = "mlx-backend")]
pub mod mlx_api_compat;
#[cfg(feature = "mlx-backend")]
pub mod mlx_executor;
#[cfg(feature = "mlx-backend")]
pub mod mlx_inventory;
#[cfg(feature = "mlx-backend")]
pub mod mlx_patch_register;
#[cfg(feature = "mlx-backend")]
pub mod mlx_runtime_probe;
#[cfg(feature = "mlx-backend")]
pub mod model_cache;
#[cfg(feature = "mlx-backend")]
pub mod model;
#[cfg(feature = "mlx-backend")]
pub mod model_runtime;
pub mod model_store;
pub mod native_kernel;
#[cfg(feature = "mlx-backend")]
pub mod operation_catalog;
#[cfg(feature = "mlx-backend")]
pub mod pipeline_parity;
pub mod placement_profile;
#[cfg(feature = "mlx-backend")]
pub mod primitives;
pub mod profile_compiler;
#[cfg(feature = "mlx-backend")]
pub mod profiled_executor;
#[cfg(feature = "mlx-backend")]
pub mod projection_identity;
#[cfg(feature = "mlx-backend")]
pub mod projection_executor;
#[cfg(feature = "mlx-backend")]
pub mod projection_tests;
pub mod quantization;
#[cfg(feature = "mlx-backend")]
pub mod quantized;
pub mod receipt;
pub mod ring;
#[cfg(feature = "mlx-backend")]
pub mod readiness_gates;
pub mod receipts;
#[cfg(feature = "mlx-backend")]
pub mod replay_projection;
pub mod requalification;
#[cfg(feature = "mlx-backend")]
pub mod research_contracts;
#[cfg(feature = "mlx-backend")]
pub mod research_metrics;
#[cfg(feature = "mlx-backend")]
pub mod research_trace;
pub mod residency;
pub mod runtime_contract;
pub mod runtime_orchestration;
#[cfg(feature = "mlx-backend")]
pub mod runtime_trace;
#[cfg(feature = "mlx-backend")]
pub mod scheduling;
#[cfg(feature = "server")]
pub mod server;
#[cfg(feature = "mlx-backend")]
pub mod editing;
#[cfg(feature = "mlx-backend")]
pub mod exo;
#[cfg(feature = "mlx-backend")]
pub mod session;
#[cfg(feature = "mlx-backend")]
pub mod sidecar;
#[cfg(feature = "mlx-backend")]
pub mod speculative;

#[cfg(feature = "storage-adapters")]
pub mod storage_adapters;
pub mod storage_kernel;
pub mod streaming;
pub mod tokenizer;
pub mod toolchain_attest;
#[cfg(feature = "mlx-backend")]
pub mod tools;
pub mod transform_recipe;
#[cfg(feature = "mlx-backend")]
pub mod treatment;
#[cfg(feature = "mlx-backend")]
pub mod validator;
#[cfg(feature = "mlx-backend")]
pub mod video;
#[cfg(feature = "mlx-backend")]
pub mod vision;
pub mod worker_crash_ledger;
#[cfg(feature = "mlx-backend")]
pub mod worker_memory;
pub mod worker_protocol;
#[cfg(feature = "mlx-backend")]
pub mod worker_supervisor;
#[cfg(feature = "candle-cpu")]
pub mod candle_cpu_backend;

#[cfg(feature = "mlx-backend")]
pub use crate::session::{
    ControlSessionState, GenerationControlSession, InferenceSession, InferenceSessionState,
    SamplerConfig,
};
#[cfg(any(feature = "mlx-backend"))]
pub use coreml_proto;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Status {
    InvalidArg,
    GenericFailure,
    InternalError,
    Cancelled,
    Timeout,
}

#[derive(Debug)]
pub struct Error {
    pub status: Status,
    pub reason: String,
}

impl Error {
    pub fn new(status: Status, reason: impl Into<String>) -> Self {
        Self {
            status,
            reason: reason.into(),
        }
    }
    pub fn from_reason(reason: impl Into<String>) -> Self {
        Self {
            status: Status::GenericFailure,
            reason: reason.into(),
        }
    }
}

impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.reason)
    }
}

impl std::error::Error for Error {}

pub type Result<T, E = Error> = std::result::Result<T, E>;

/// Current timestamp as ISO 8601 UTC string.
pub fn now_iso8601() -> String {
    use std::time::{SystemTime, UNIX_EPOCH};
    let secs = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs();
    // Format as ISO 8601 (simple: YYYY-MM-DDTHH:MM:SSZ)
    let days = (secs / 86400) as i64;
    let time_secs = secs % 86400;
    let (year, month, day) = civil_from_days(days);
    let hour = time_secs / 3600;
    let min = (time_secs % 3600) / 60;
    let sec = time_secs % 60;
    format!("{:04}-{:02}-{:02}T{:02}:{:02}:{:02}Z", year, month, day, hour, min, sec)
}

/// Hostname or "unknown" if unavailable.
pub fn hostname_or_default() -> String {
    std::env::var("HOSTNAME")
        .or_else(|_| std::env::var("HOST"))
        .unwrap_or_else(|_| "unknown".to_string())
}

/// Convert a days-from-epoch value to (year, month, day) in the Gregorian
/// civil calendar.
fn civil_from_days(days: i64) -> (i64, i64, i64) {
    // Algorithm from Howard Hinnant
    let z = days + 719468;
    let era = if z >= 0 { z } else { z - 146096 } / 146097;
    let doe = z - era * 146097;
    let yoe = (doe - doe / 1460 + doe / 36524 - doe / 146096) / 365;
    let y = yoe + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let m = if mp < 10 { mp + 3 } else { mp - 9 };
    let y = if m <= 2 { y + 1 } else { y };
    (y, m as i64, d as i64)
}
