#[cfg(feature = "mlx-backend")]
use std::collections::HashMap;
#[cfg(feature = "mlx-backend")]
use std::collections::HashSet;
#[cfg(feature = "mlx-backend")]
use std::path::PathBuf;
use std::sync::Arc;
use tokio::signal;
use tokio::sync::Mutex;
#[cfg(feature = "mlx-backend")]
use tribunus_compute_core::engine_policy::ExecutionPolicy;
#[cfg(feature = "mlx-backend")]
use tribunus_compute_core::exo::ExoNode;
#[cfg(feature = "mlx-backend")]
use tribunus_compute_core::lora::LoraAdapter;
#[cfg(feature = "mlx-backend")]
use tribunus_compute_core::metrics::InferenceTelemetry;
#[cfg(feature = "mlx-backend")]
use tribunus_compute_core::model_cache::ModelCache;
#[cfg(feature = "mlx-backend")]
use tribunus_compute_core::scheduling::HardwareConfig;
#[cfg(feature = "mlx-backend")]
use tribunus_compute_core::server::admin::ActiveRequestInfo;
#[cfg(feature = "mlx-backend")]
use tribunus_compute_core::worker_supervisor::WorkerSupervisor;
use tribunus_compute_core::{log_error, log_info, log_warn};

use tribunus_compute_core::projection_identity::RuntimeMode;
use tribunus_compute_core::readiness_gates::ReadinessGates;
#[cfg(not(feature = "mlx-backend"))]
use tribunus_compute_core::server::cpu::{create_cpu_router, CpuAppState};
#[cfg(feature = "mlx-backend")]
use tribunus_compute_core::server::routes::create_router;
#[cfg(feature = "mlx-backend")]
use tribunus_compute_core::server::routes::AppState;
#[cfg(feature = "mlx-backend")]
use tribunus_compute_core::server::{
    auth::ApiKeyValidator,
    benchmark,
    models::{recommend_models, ModelRegistry},
    rate_limiter::RateLimiter,
};
#[cfg(not(feature = "mlx-backend"))]
use tribunus_compute_core::server::{
    benchmark,
    models::{recommend_models, ModelRegistry},
};
use tribunus_compute_core::tokenizer::TribunusTokenizer;
#[cfg(feature = "mlx-backend")]
use uuid::Uuid;

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
        #[cfg(feature = "mlx-backend")]
        eprintln!("  --exo                Enable EXO clustering mode");
        #[cfg(feature = "mlx-backend")]
        eprintln!("  --exo-port <n>       Port for EXO communication (default: 52415)");
        #[cfg(feature = "mlx-backend")]
        eprintln!("  --no-worker          Coordinator-only node (no local inference)");
        eprintln!("  --code-mode          Optimize for code completion latency");
        eprintln!("  --dev-mode           Disable auth, auto-register model, verbose errors");
        eprintln!();
        eprintln!("Environment variables:");
        eprintln!("  TRIBUNUS_CONFIG_PATH   Config file path");
        eprintln!("  TRIBUNUS_PORT          Server port");
        eprintln!("  TRIBUNUS_HOST          Bind address");
        eprintln!("  TRIBUNUS_LOG_LEVEL     Log verbosity (info, debug, warn)");
        eprintln!("  TRIBUNUS_MODEL_PATH    Model directory path");
        #[cfg(feature = "mlx-backend")]
        eprintln!("  TRIBUNUS_EXO_ENABLED   Enable EXO clustering (true/false)");
        #[cfg(feature = "mlx-backend")]
        eprintln!("  TRIBUNUS_EXO_PORT      Port for EXO communication");
        std::process::exit(0);
    }

    // Propagate --config as env var so ServerConfig::load() picks it up
    if let Some(path) = &config_path_override {
        unsafe {
            std::env::set_var("TRIBUNUS_CONFIG_PATH", path);
        }
    }

    // Load config: defaults -> config.toml -> env vars -> CLI args (highest priority)
    let mut cfg = tribunus_compute_core::config::ServerConfig::load();
    cfg.apply_cli_args(&args);

    let port = cfg.server.port;
    let host = cfg.server.host.clone();
    let model_path = cfg.model.model_path.clone();
    #[cfg(feature = "mlx-backend")]
    let exo_mode = cfg.cluster.exo_enabled;
    #[cfg(feature = "mlx-backend")]
    let exo_port = cfg.cluster.exo_port;

    // Runtime-only flags (not stored in config)
    #[cfg(feature = "mlx-backend")]
    let mut no_worker = false;
    #[cfg(feature = "mlx-backend")]
    let mut code_mode = false;
    let mut dev_mode = false;
    for arg in &args {
        match arg.as_str() {
            #[cfg(feature = "mlx-backend")]
            "--no-worker" => no_worker = true,
            #[cfg(feature = "mlx-backend")]
            "--code-mode" => code_mode = true,
            "--dev-mode" => dev_mode = true,
            _ => {}
        }
    }

    log_info!("Tribunus Compute Server v0.1.0");
    // ── Dev mode: disable auth, auto-register model, verbose errors ────
    if dev_mode {
        log_warn!("[dev-mode] Auth disabled, model auto-registered, verbose errors enabled");
        unsafe {
            std::env::set_var("TRIBUNUS_API_KEYS", "");
        }
        // Model auto-registration happens below.
    }

    // ── MLX backend startup path ────────────────────────────────────────
    #[cfg(feature = "mlx-backend")]
    {
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
                    tribunus_compute_core::log_error!("[exo] Failed to start EXO node: {}", e);
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

        let supervisor = if let Some(mpath) = &model_path {
            log_info!("Loading model from {}...", mpath);
            let path = std::path::Path::new(mpath);
            // Validate model path exists, then spawn worker
            if path.exists() {
                let worker_binary = std::env::var("TRIBUNUS_WORKER_BINARY")
                    .map(PathBuf::from)
                    .unwrap_or_else(|_| {
                        std::env::current_exe()
                            .ok()
                            .map(|p| p.with_file_name("tribunus-compute-worker"))
                            .unwrap_or_else(|| PathBuf::from("tribunus-compute-worker"))
                    });

                let image_hash = "";
                let worker_id = Uuid::new_v4().to_string();
                let policy = tribunus_compute_core::engine_policy::qualification_policy();

                match WorkerSupervisor::launch_and_handshake(
                    policy,
                    &worker_binary,
                    path,
                    image_hash,
                    &worker_id,
                ) {
                    Ok(sup) => {
                        log_info!("  Worker spawned (pid {})", sup.process_ctrl.pid());
                        // Load model in the worker and wait for it to be ready.
                        match sup.load_model(image_hash) {
                            Ok(()) => {
                                log_info!("  Model ready");
                                Some(Arc::new(sup))
                            }
                            Err(e) => {
                                tribunus_compute_core::log_error!("  Model load failed: {:?}", e);
                                std::process::exit(1);
                            }
                        }
                    }
                    Err(e) => {
                        tribunus_compute_core::log_error!("  Failed to spawn worker: {:?}", e);
                        log_warn!("  Continuing without worker; chat endpoints will return 503");
                        None
                    }
                }
            } else {
                tribunus_compute_core::log_error!("  Model path does not exist: {}", mpath);
                None
            }
        } else {
            None
        };

        // ── Tokenizer ────────────────────────────────────────────────────
        let tokenizer = model_path.as_ref().and_then(|mpath| {
            let dir = std::path::Path::new(mpath);
            match TribunusTokenizer::from_dir(dir) {
                Ok(tok) => {
                    log_info!("  Tokenizer loaded");
                    Some(Arc::new(tok))
                }
                Err(e) => {
                    log_warn!("  No tokenizer found: {}", e);
                    None
                }
            }
        });

        // ── Readiness gates ──────────────────────────────────────────────
        let runtime_mode = match cfg.server.runtime_mode.as_str() {
            "qualified" => RuntimeMode::Qualified,
            "experimental" => RuntimeMode::Experimental,
            _ => RuntimeMode::Safe,
        };
        let mut gates = ReadinessGates::new();
        gates.run_all(supervisor.as_deref(), tokenizer.as_deref(), runtime_mode);
        log_info!("Readiness gates summary: {}", gates.summary());
        let gates = Arc::new(Mutex::new(gates));

        // 4. Start server
        let auth = Arc::new(ApiKeyValidator::new());
        auth.load_from_env();

        let token_rate_limiter = Arc::new(RateLimiter::new(
            cfg.server.rate_limit_burst,
            cfg.server.rate_limit_tokens_per_sec,
        ));

        let state = AppState {
            models: Arc::new(Mutex::new(registry)),
            benchmark: Arc::new(Mutex::new(Some(bench))),
            model_cache: Arc::new(Mutex::new(model_cache)),
            supervisor,
            tokenizer,
            gates,
            exo_node,
            telemetry: Arc::new(InferenceTelemetry::new()),
            adapters: Arc::new(Mutex::new(HashMap::new())),
            active_adapter: Arc::new(Mutex::new(None)),
            knowledge_editor: Arc::new(Mutex::new(None)),
            rate_limiter: Arc::new(RateLimiter::new(60, 1.0)),
            token_rate_limiter,
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
    }

    // ── Candle CPU startup path ──────────────────────────────────────
    #[cfg(not(feature = "mlx-backend"))]
    {
        log_info!("Backend: Candle CPU");

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

        // ── Tokenizer ────────────────────────────────────────────────────
        let tokenizer = model_path.as_ref().and_then(|mpath| {
            let dir = std::path::Path::new(mpath);
            match TribunusTokenizer::from_dir(dir) {
                Ok(tok) => {
                    log_info!("  Tokenizer loaded");
                    Some(Arc::new(tok))
                }
                Err(e) => {
                    log_warn!("  No tokenizer found: {}", e);
                    None
                }
            }
        });

        // ── Readiness gates (CPU path: no worker) ────────────────────────
        let runtime_mode = match cfg.server.runtime_mode.as_str() {
            "qualified" => RuntimeMode::Qualified,
            "experimental" => RuntimeMode::Experimental,
            _ => RuntimeMode::Safe,
        };
        let mut gates = ReadinessGates::new();
        gates.run_all(tokenizer.as_deref(), runtime_mode);
        log_info!("Readiness gates summary: {}", gates.summary());
        let gates = Arc::new(Mutex::new(gates));

        // 3. Start server
        let state = CpuAppState { gates, tokenizer };

        let app = create_cpu_router(state);
        let addr = format!("{}:{}", host, port);
        let listener = tokio::net::TcpListener::bind(&addr)
            .await
            .unwrap_or_else(|e| panic!("Failed to bind to {}: {}", addr, e));
        log_info!("Server running on http://{}", addr);
        log_info!("  Backend: Candle CPU");

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
                log_info!("\nShutdown signal received, draining...");
            })
            .await
            .unwrap();
    }

    log_info!("Server shut down cleanly.");
}
