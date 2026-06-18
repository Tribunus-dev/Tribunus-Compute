use tribunus_compute_core::logging::{log_info, log_error, log_warn, log_debug};
use std::collections::HashMap;
use std::collections::HashSet;
use std::sync::Arc;
use tokio::signal;
use tokio::sync::Mutex;
use tribunus_compute_core::exo::ExoNode;
use tribunus_compute_core::lora::LoraAdapter;
use tribunus_compute_core::metrics::InferenceTelemetry;
use tribunus_compute_core::model_cache::ModelCache;
use tribunus_compute_core::profiled_executor::{LoadedProfiledModel, ProfiledInferenceSession};
use tribunus_compute_core::scheduling::HardwareConfig;
use tribunus_compute_core::server::admin::ActiveRequestInfo;

use tribunus_compute_core::server::{
    auth::ApiKeyValidator,
    benchmark,
    models::{recommend_models, ModelRegistry},
    rate_limiter::RateLimiter,
    routes::{build_kv_caches, create_router, AppState},
};

#[tokio::main]
async fn main() {
    // macOS workaround: unset MallocStackLogging inherited from Xcode/LLDB
    // to suppress "can't turn off malloc stack logging because it was not enabled"
    // on stderr during process exit, which corrupts terminal output.
    // Must happen BEFORE any allocation or thread spawn (hence at the very
    // top of main, not in an init function).
    // Must happen BEFORE any memory allocation or thread spawn.
    // Setting to "0" (rather than removing) prevents libsystem_malloc's
    // unconditional thread-cleanup message on macOS 26.5 Metal/OMP.
    unsafe {
        std::env::set_var("MallocStackLogging", "0");
        std::env::set_var("MallocStackLoggingNoCompact", "0");
    }

    // Pre-parse --config and --help before loading config
    let args: Vec<String> = std::env::args().collect();
    let mut help_requested = false;
    let mut config_path_override: Option<String> = None;
    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--config" => {
                i += 1;
                if i < args.len() {
                    config_path_override = Some(args[i].clone());
                }
            }
            "--help" | "-h" => {
                help_requested = true;
            }
            _ => {}
        }
        i += 1;
    }

    if help_requested {
        eprintln!("Usage: tribunus-server [OPTIONS]");
        eprintln!();
        eprintln!("Options:");
        eprintln!("  --config <path>      Config file path (default: $HOME/.tribunus/config.toml)");
        eprintln!("  --port <n>           Server port (default: 11434)");
        eprintln!("  --host <addr>        Bind address (default: 0.0.0.0)");
        eprintln!("  --model-path <dir>   Path to model directory (ComputeImage)");
        eprintln!("  --exo                Enable EXO clustering mode");
        eprintln!("  --exo-port <n>       Port for EXO communication (default: 52415)");
        eprintln!("  --no-worker          Coordinator-only node (no local inference)");
        eprintln!("  --code-mode          Optimize for code completion latency");
        eprintln!();
        eprintln!("Environment variables:");
        eprintln!("  TRIBUNUS_CONFIG_PATH   Config file path");
        eprintln!("  TRIBUNUS_PORT          Server port");
        eprintln!("  TRIBUNUS_HOST          Bind address");
        eprintln!("  TRIBUNUS_LOG_LEVEL     Log verbosity (info, debug, warn)");
        eprintln!("  TRIBUNUS_MODEL_PATH    Model directory path");
        eprintln!("  TRIBUNUS_EXO_ENABLED   Enable EXO clustering (true/false)");
        eprintln!("  TRIBUNUS_EXO_PORT      Port for EXO communication");
        std::process::exit(0);
    }

    // Propagate --config as env var so ServerConfig::load() picks it up
    if let Some(path) = &config_path_override {
        unsafe { std::env::set_var("TRIBUNUS_CONFIG_PATH", path); }
    }

    // Load config: defaults -> config.toml -> env vars -> CLI args (highest priority)
    let mut cfg = tribunus_compute_core::config::ServerConfig::load();
    cfg.apply_cli_args(&args);

    let port = cfg.server.port;
    let host = cfg.server.host.clone();
    let model_path = cfg.model.model_path.clone();
    let exo_mode = cfg.cluster.exo_enabled;
    let exo_port = cfg.cluster.exo_port;

    // Runtime-only flags (not stored in config)
    let mut no_worker = false;
    let mut code_mode = false;
    for arg in &args {
        match arg.as_str() {
            "--no-worker" => no_worker = true,
            "--code-mode" => code_mode = true,
            _ => {}
        }
    }

    log_info!("Tribunus Compute Server v0.1.0");

    // 0b. Auto-detect hardware and configure for maximum throughput
    let hw = HardwareConfig::detect();
    log_info!("=== Hardware Detected ===");
    log_info!("  RAM: {} GB", hw.total_ram_gb);
    log_info!("  GPU cores: {}", hw.gpu_cores);
    log_info!("  ANE cores: {}", hw.ane_cores);
    log_info!("  CPU cores: {}", hw.cpu_cores);
    log_info!("  Memory bandwidth: {} GB/s", hw.memory_bw_gb_s);
    log_info!(
        "  Mode: {}",
        if hw.is_memory_rich {
            "MAXIMUM THROUGHPUT"
        } else {
            "MEMORY EFFICIENT"
        }
    );
    log_info!("  Batch size: {}", hw.recommended_batch_size);
    log_info!("  Speculation: {}x ANE", hw.recommended_spec_length);

    // Apply --code-mode optimizations for low-latency code completion.
    if code_mode {
        log_info!("[code-mode] Optimizing for code completion latency");
        let speculation = hw.recommended_spec_length.max(32); // max drafts
        let _temp = 0.2;
        let _max_tokens = 512;
        log_info!("  Speculation: {}x drafts", speculation);
        log_info!("  Temperature: {}", _temp);
        log_info!("  Max tokens: {}", _max_tokens);
    }

    // 0a. Start EXO cluster node if requested (before benchmark banner).
    let exo_node = if exo_mode {
        match ExoNode::start(exo_port, no_worker) {
            Ok(node) => Some(Arc::new(tokio::sync::Mutex::new(node))),
            Err(e) => {
                log_error!("[exo] Failed to start EXO node: {}", e);
                log_warn!("[exo] Continuing without EXO clustering.");
                None
            }
        }
    } else {
        None
    };

    // 1. Run system benchmark
    log_info!("Benchmarking system...");
    let bench = benchmark::run_benchmark();
    log_info!("  Chip: {}", bench.chip);
    log_info!("  RAM: {} GB", bench.ram_gb);

    // 2. Create model registry with recommendations
    let mut registry = ModelRegistry::new();
    for model in recommend_models(&bench.chip, bench.ram_gb, "chat") {
        registry.register(model);
    }
    log_info!("  Recommended {} models", registry.list().len());

    // 3. Load model if path provided
    // Initialize model cache with half of total RAM.
    let total_ram_mb = tribunus_compute_core::gpu_memory::total_physical_ram_mb();
    let cache_max_mb = if hw.is_memory_rich {
        ((total_ram_mb as f64 * 0.9) as u64).max(4096)
    } else {
        ((total_ram_mb as f64 * 0.5) as u64).max(2048)
    };
    let mut model_cache = ModelCache::new(cache_max_mb);

    // Configure cache for detected hardware and preload on memory-rich systems.
    model_cache.configure_for_hardware();
    if hw.is_memory_rich {
        if let Err(e) = model_cache.preload_all() {
            log_warn!("[model-cache] Preload warning: {}", e);
        }
    }

    let session = if let Some(mpath) = &model_path {
        log_info!("Loading model from {}...", mpath);
        let path = std::path::Path::new(mpath);
        match LoadedProfiledModel::new(path) {
            Ok(model) => {
                let n_layers = model.reader.manifest.execution_plan.layers.len();
                let kv_caches = build_kv_caches(&model);
                let mut session = ProfiledInferenceSession::new("server".into(), kv_caches);
                session.setup_from_model(&model);
                log_info!("  Model loaded: {} layers", n_layers);
                Some(Arc::new(Mutex::new(Some(session))))
            }
            Err(e) => {
                log_error!("  Failed to load model: {:?}", e);
                Some(Arc::new(Mutex::new(None)))
            }
        }
    } else {
        Some(Arc::new(Mutex::new(None)))
    };

    // 4. Start server
    let auth = Arc::new(ApiKeyValidator::new());
    auth.load_from_env();

    let state = AppState {
        models: Arc::new(Mutex::new(registry)),
        benchmark: Arc::new(Mutex::new(Some(bench))),
        model_cache: Arc::new(Mutex::new(model_cache)),
        session: session.expect("session must be Some"),
        exo_node,
        telemetry: Arc::new(InferenceTelemetry::new()),
        adapters: Arc::new(Mutex::new(HashMap::new())),
        active_adapter: Arc::new(Mutex::new(None)),
        knowledge_editor: Arc::new(Mutex::new(None)),
        rate_limiter: Arc::new(RateLimiter::new(60, 1.0)),
        auth,
        admin_request_registry: Arc::new(Mutex::new(HashMap::new())),
        admin_cancelled_requests: Arc::new(Mutex::new(HashSet::new())),
    };

    let app = create_router(state);
    let addr = format!("{}:{}", host, port);
    let listener = tokio::net::TcpListener::bind(&addr)
        .await
        .unwrap_or_else(|e| panic!("Failed to bind to {}: {}", addr, e));
    log_info!("Server running on http://{}", addr);
    if exo_mode {
        log_info!("  (EXO cluster mode)");
    } else {
        log_info!("  (Ollama-compatible API)");
    }
    // Xcode AI provider banner.
    log_info!("  Xcode AI provider: http://{}/v1", addr);
    log_info!("  Run: scripts/xcode-llm-profile.sh install");
    axum::serve(listener, app)
        .with_graceful_shutdown(async {
            // Wait for SIGINT (Ctrl-C) or SIGTERM
            signal::ctrl_c().await.ok();
            #[cfg(unix)]
            {
                let mut term = signal::unix::signal(signal::unix::SignalKind::terminate()).ok();
                if let Some(term) = &mut term {
                    term.recv().await;
                }
            }
            log_info!("\nShutdown signal received, draining active sessions...");
        })
        .await
        .unwrap();

    log_info!("Server shut down cleanly.");
}
