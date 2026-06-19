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

pub mod analysis;
pub mod audio;
pub mod autopsy;
#[cfg(target_os = "macos")]
pub mod ane;
#[cfg(target_os = "macos")]
pub mod ane_bridge;
#[cfg(target_os = "macos")]
pub mod arena;
#[cfg(target_os = "macos")]
pub mod arena_info;
#[cfg(target_os = "macos")]
pub mod arena_lifecycle;
#[cfg(target_os = "macos")]
pub mod arena_pool;
pub mod attention;
pub mod backend;
#[cfg(target_os = "macos")]
pub mod bridge;
pub mod cache;
pub mod capability;
pub mod cli;
pub mod compile_pipeline;
pub mod compile_progress;
pub mod compile_state;
pub mod compiler;
pub mod compute_image;
pub mod compute_image_v0;
pub mod compute_ir;
pub mod compute_lane;
pub mod compute_service;
pub mod config;
pub mod contracts;
pub mod crash_breadcrumb;
pub mod copy_ledger;
#[cfg(target_os = "macos")]
pub mod coreml_audit;
#[cfg(target_os = "macos")]
pub mod coreml_bridge;
#[cfg(target_os = "macos")]
pub mod coreml_pipeline;
#[cfg(target_os = "macos")]
pub mod coreml_state;
pub mod cpu_benchmarks;
pub mod decode_attribution;
pub mod engine;
pub mod engine_error;
pub mod engine_policy;
pub mod engine_receipts;
pub mod errors;
pub mod executor;
pub mod experiment;
pub mod external_array;
pub mod generation;
pub mod fusion_region;
pub mod gemma;
pub mod grammar;
pub mod gpu_memory;
pub mod gguf;
pub mod gpu_worker;
pub mod heterogeneous;
pub mod hybrid_profile;
pub mod inference_profile;
pub mod kv_cache;
pub mod layout_compiler;
pub mod layout_transform;
pub mod loader;
#[macro_use] pub mod logging;
pub mod lora;
pub mod mapped_image;
pub mod memory;
pub mod metrics;
#[cfg(target_os = "macos")]
pub mod metal_capture;
pub mod mil_builder;
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
pub mod model_cache;
pub mod model;
pub mod model_runtime;
pub mod model_store;
pub mod native_kernel;
pub mod operation_catalog;
pub mod pipeline_parity;
pub mod placement_profile;
pub mod primitives;
pub mod profile_compiler;
pub mod profiled_executor;
pub mod projection_identity;
#[cfg(feature = "mlx-backend")]
pub mod projection_executor;
pub mod projection_tests;
pub mod quantization;
pub mod quantized;
pub mod receipt;
pub mod ring;
pub mod readiness_gates;
pub mod receipts;
pub mod replay_projection;
pub mod requalification;
pub mod research_contracts;
pub mod research_metrics;
pub mod research_trace;
pub mod residency;
pub mod runtime_contract;
pub mod runtime_orchestration;
pub mod runtime_trace;
pub mod scheduling;
#[cfg(feature = "server")]
pub mod server;
pub mod editing;
pub mod exo;
pub mod session;
pub mod sidecar;
pub mod speculative;

#[cfg(feature = "storage-adapters")]
pub mod storage_adapters;
pub mod storage_kernel;
pub mod streaming;
pub mod tokenizer;
pub mod toolchain_attest;
pub mod tools;
pub mod transform_recipe;
pub mod treatment;
pub mod validator;
pub mod video;
pub mod vision;
pub mod worker_crash_ledger;
pub mod worker_memory;
pub mod worker_protocol;
pub mod worker_supervisor;
#[cfg(feature = "candle-cpu")]
pub mod candle_cpu_backend;

pub use crate::session::{
    ControlSessionState, GenerationControlSession, InferenceSession, InferenceSessionState,
    SamplerConfig,
};
#[cfg(any(target_os = "macos", feature = "ane"))]
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
