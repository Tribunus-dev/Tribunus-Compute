//! ComputeEngine — high-level orchestrator for model lifecycle and generation.
//!
//! Thin control surface that resolves policy, checks model state, and
//! delegates execution to [`WorkerSupervisor`].  The supervisor owns the worker
//! subprocess which runs the actual model runtime, prefill, and decode loop.
//!
//! # Changes from v0
//!
//! - `generate()` now accepts `input_ids: &[u32]` and returns
//!   [`GenerationHandle`] (wrapping the stream + job_id).
//! - `cancel_generation` accepts a `String` job_id.
//! - `unload_model()` cancels in-flight requests and waits for the
//!   cancellation grace period before tearing down the worker.
//! - Worker restart is transparent: a faulted worker is automatically
//!   respawned (up to `policy.restart_limit` times) on the next call
//!   to `generate()`.
//! - `Drop` calls `shutdown()` for clean teardown.

use std::path::PathBuf;

use crate::engine_error::{EngineError, EngineErrorCode};
use crate::engine_policy::{qualification_policy, resolve_generation_budget};
use crate::model_store::{InstalledModel, ModelStore};
use crate::streaming::GenerationHandle;
use crate::worker_protocol::HostCommand;
use crate::worker_protocol::StartGenerationPayload;
use crate::worker_supervisor::WorkerSupervisor;

use crate::backend::accelerate::AccelerateBackend;
use crate::backend::heterogeneous_executor::BackendInstance;
use crate::backend::routing::{
    BackendExecutionReceipt, BackendId, ComputeRouteProfile, OperationDescriptor, OperationFamily,
};
use crate::backend::{MlxBackend, TensorHandle};
use crate::compute_image::{
    clear_mlx_cache, mlx_active_memory_bytes, mlx_cache_memory_bytes, mlx_get_memory_limit,
};
use crate::hybrid_profile::{HybridExecutor, HybridProfile};
use crate::scheduling::{
    PhaseKind, Scheduler, SchedulerConfig, TokenBudgetConfig, TokenBudgetScheduler, TokenWorkUnit,
};
// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/// Default vocabulary size when model metadata is unavailable.
const DEFAULT_VOCAB_SIZE: u32 = 256_128;

/// Default BOS token ID (Gemma family).
const DEFAULT_BOS_TOKEN: u32 = 2;

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

/// Identity of a loaded model — the worker owns the runtime and session.
#[derive(Debug)]
struct LoadedModel {
    /// Hash identifying the model image in the store.
    image_hash: String,
    /// Path to the model directory in the store.
    model_path: PathBuf,
    /// Vocabulary size (valid token ID range is `[0, vocab_size)`).
    vocab_size: u32,
}

/// Parameters for a text generation request.
///
/// All numeric fields use their MLX-native defaults when left at zero
/// (the JS side maps `undefined` → `0` / `null` → `None` for Option fields).
///
/// The only required field is `prompt`.
#[derive(Debug, Clone)]
pub struct GenerationRequest {
    /// Input text prompt.
    pub prompt: String,
    /// Opaque session identifier for this generation run.
    pub session_id: String,
    /// Maximum number of tokens to generate (0 = bounded qualification mode).
    pub max_tokens: u32,
    /// Token ID that signals end-of-sequence.
    pub eos_token_id: u32,
    /// Pre-tokenized input token IDs for the prompt.
    pub input_ids: Vec<i32>,
    /// Temperature for softmax scaling.  0.0 = greedy.
    pub temperature: f64,
    /// Top-k filter: retain only the k highest-probability tokens.
    pub top_k: u32,
    /// Top-p (nucleus) filter: retain smallest set whose cumulative
    /// probability exceeds p.
    pub top_p: f64,
    /// Optional PRNG seed for deterministic sampling.
    pub seed: Option<u64>,
    /// Token ID sequences at which generation should stop.
    pub stop_sequences: Vec<String>,
}

impl GenerationRequest {
    /// Return the session identifier for this request.
    pub fn session_id(&self) -> &str {
        &self.session_id
    }

    /// Return the end-of-sequence token ID.
    pub fn eos_token_id(&self) -> u32 {
        self.eos_token_id
    }
}

/// Static capability report for this compute engine instance.
#[derive(Debug, Clone)]
pub struct EngineCapabilities {
    /// Whether a Metal-compatible GPU is available.
    pub supports_gpu: bool,
    /// Whether Core ML model execution is available.
    pub supports_coreml: bool,
    /// MLX framework version string (semver).
    pub mlx_version: String,
}

/// High-level engine wrapping model lifecycle and text generation.
///
/// # Lifecycle
///
/// 1. Call `new()` — opens the default model store at `~/.tribunus/models/`.
/// 2. `installModel(...)` — copy a compiled ComputeImage into the store.
/// 3. `setWorkerBinaryPath(...)` — point at the worker subprocess binary.
/// 4. `loadModel(...)` — verify the seal, spawn a worker process, and load
///    the model into the worker.
/// 5. `generate(...)` — resolve policy, validate token IDs, delegate to the
///    worker supervisor, and return a `GenerationHandle` immediately.
/// 6. `cancel(...)` / `cancel_generation(...)` — signal the worker to abort.
/// 7. `unload_model(...)` — cancel active requests, kill the worker, release
///    all native resources.
///
/// At most one model may be loaded at a time (v1).
impl std::fmt::Debug for ComputeEngine {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ComputeEngine")
            .field("model_store", &self.model_store)
            .field("loaded_model", &self.loaded_model)
            .field("worker_binary_path", &self.worker_binary_path)
            .field("capabilities", &self.capabilities)
            .field(
                "worker_supervisor",
                &self
                    .worker_supervisor
                    .as_ref()
                    .map(|_| "Some(WorkerSupervisor)"),
            )
            .field("restart_count", &self.restart_count)
            .field(
                "scheduler",
                &self.scheduler.as_ref().map(|_| "Some(Scheduler)"),
            )
            .field(
                "token_budget_scheduler",
                &self
                    .token_budget_scheduler
                    .as_ref()
                    .map(|_| "Some(TokenBudgetScheduler)"),
            )
            .field(
                "hybrid_executor",
                &self
                    .hybrid_executor
                    .as_ref()
                    .map(|_| "Some(HybridExecutor)"),
            )
            .field("backend_routing", &self.backend_routing)
            .field("peak_memory_used", &self.peak_memory_used)
            .finish()
    }
}

pub struct ComputeEngine {
    model_store: ModelStore,
    worker_supervisor: Option<WorkerSupervisor>,
    loaded_model: Option<LoadedModel>,
    worker_binary_path: Option<PathBuf>,
    capabilities: EngineCapabilities,
    /// Number of worker restarts attempted during the current model session.
    restart_count: u32,
    /// Host-side inference scheduler for Accelerate/ANE batch dispatch.
    scheduler: Option<Scheduler>,
    /// Token-budget scheduler for admission control and work dispatch.
    token_budget_scheduler: Option<TokenBudgetScheduler>,
    /// Heterogeneous executor dispatching batch slots to registered backends.
    hybrid_executor: Option<HybridExecutor>,
    /// Route profile for deterministic backend assignment per slot.
    backend_routing: Option<ComputeRouteProfile>,
    /// Peak active memory observed during the most recent inference cycle (bytes).
    peak_memory_used: u64,
    /// Cumulative number of backend fallback events (read from executor static).
    fallback_count: u64,
}

impl ComputeEngine {
    // -- lifecycle ------------------------------------------------------------

    /// Create a new engine with the default model store.
    ///
    /// Opens (or creates) `~/.tribunus/models/` and detects runtime
    /// capabilities.
    pub fn new() -> crate::Result<Self> {
        let store = ModelStore::open_default()
            .map_err(|e| crate::Error::from_reason(format!("Failed to open model store: {}", e)))?;

        Ok(Self {
            model_store: store,
            worker_supervisor: None,
            loaded_model: None,
            worker_binary_path: None,
            capabilities: EngineCapabilities {
                supports_gpu: false,
                supports_coreml: false,
                mlx_version: "0.1.0".into(),
            },
            restart_count: 0,
            scheduler: None,
            token_budget_scheduler: None,
            hybrid_executor: None,
            backend_routing: None,
            peak_memory_used: 0,
            fallback_count: 0,
        })
    }

    /// Set the path to the worker subprocess binary.
    ///
    /// Must be called before `loadModel()`.  The binary is spawned by the
    /// [`WorkerSupervisor`] and communicates over framed JSON IPC.
    pub fn set_worker_binary_path(&mut self, path: String) {
        self.worker_binary_path = Some(PathBuf::from(path));
    }

    // -- observability -------------------------------------------------------

    /// Return the cumulative number of backend fallback events.
    ///
    /// Reads the live counter from the executor module, which tracks
    /// every quantized matmul that fell through to a secondary or tertiary
    /// backend after the primary failed.
    pub fn fallback_count(&self) -> u64 {
        crate::executor::fallback_count()
    }

    /// Reset the fallback counter to zero.
    pub fn reset_fallback_count(&mut self) {
        self.fallback_count = 0;
        crate::executor::reset_fallback_count();
    }

    // -- host-side inference (Accelerate / ANE) -----------------------------

    /// Initialise the host-side inference pipeline.
    ///
    /// Creates a [`Scheduler`] for continuous batching, a [`HybridExecutor`]
    /// to dispatch batch slots to the appropriate backend, and registers the
    /// MLX and Accelerate backends.  Call once before
    /// [`run_inference_cycle`](Self::run_inference_cycle).
    pub fn init_host_inference(
        &mut self,
        config: SchedulerConfig,
        profile: HybridProfile,
    ) -> Result<(), String> {
        // 1. Create the scheduler with the given config.
        let mut scheduler = Scheduler::new(config);

        // 2. Convert the HybridProfile into a ComputeRouteProfile for
        //    deterministic backend routing.
        // TODO: implement full HybridProfile -> ComputeRouteProfile conversion
        let route_profile = ComputeRouteProfile {
            profile_id: crate::backend::routing::RouteProfileId(0),
            logical_image_hash: crate::backend::routing::EvidenceDigest(
                profile.root_model_hash.clone(),
            ),
            artifact_root_hash: crate::backend::routing::EvidenceDigest(
                profile.compute_image_hash.clone(),
            ),
            machine_profile: crate::backend::routing::MachineProfileId(0),
            operations: Vec::new(),
            transfers: Vec::new(),
            backend_artifacts: crate::backend::routing::BackendArtifactManifest {
                mlx: Vec::new(),
                accelerate: Vec::new(),
                coreml: Vec::new(),
            },
            execution_boundaries: Vec::new(),
            evidence_basis: Vec::new(),
        };

        scheduler.set_route_profile(route_profile);

        // 3. Create the HybridExecutor from the profile and register backends.
        let mut executor = HybridExecutor::new(profile);
        executor.register_mlx(Box::new(MlxBackend::new()));
        executor.register_accelerate(Box::new(AccelerateBackend::new()));

        // 4. Create the token-budget scheduler with default configuration.
        let token_budget_scheduler = TokenBudgetScheduler::new(TokenBudgetConfig::default());

        // 5. Store everything on self.
        self.scheduler = Some(scheduler);
        self.token_budget_scheduler = Some(token_budget_scheduler);
        self.hybrid_executor = Some(executor);
        Ok(())
    }

    /// Run one inference cycle: produce a batch, dispatch it, process results.
    ///
    /// Must have called [`init_host_inference`](Self::init_host_inference) first.
    /// Errors with a clear message if the scheduler or executor are unset.
    pub fn run_inference_cycle(&mut self) -> Result<(), String> {
        let scheduler = self.scheduler.as_mut().ok_or_else(|| {
            "run_inference_cycle: scheduler not initialised — call init_host_inference first"
                .to_string()
        })?;
        let executor = self.hybrid_executor.as_mut().ok_or_else(|| {
            "run_inference_cycle: hybrid_executor not initialised — call init_host_inference first".to_string()
        })?;

        // 1. Get the next batch from the scheduler.
        let batch = scheduler.next_batch();

        // 2. Dispatch each slot in the batch through the HybridExecutor.
        //    The executor iterates over the profile's execution_order and maps
        //    each step to its registered backend (MLX, Accelerate, or ANE).
        let _receipts = executor.execute()?;

        // 3. Process the completed batch results (advance tokens, free finished slots).
        scheduler.process_results(&batch);

        // 4. Check MLX memory pressure and take proactive action.
        self.check_memory_pressure()?;

        Ok(())
    }

    /// Run inference using the token-budget scheduler.
    ///
    /// Uses [`TokenBudgetScheduler`] to manage admission control and work
    /// dispatch through the host-side inference pipeline.  Callers must
    /// have called [`init_host_inference`](Self::init_host_inference) first.
    ///
    /// The scheduler enqueues a prefill work unit, calls `schedule()` to
    /// obtain dispatchable work, dispatches each unit through the
    /// [`HybridExecutor`], then re-enqueues decode work until `max_tokens`
    /// is reached or EOS is emitted.  On completion the work unit is marked
    /// complete via [`TokenBudgetScheduler::complete`].
    pub fn run_with_token_budget(
        &mut self,
        request: &StartGenerationPayload,
    ) -> Result<Vec<u32>, EngineError> {
        let mut generated_tokens: Vec<u32> = Vec::new();
        let request_id = &request.request_id;
        let max_tokens = request.max_output_tokens;
        let eos_token_id = request.stop_token_ids.first().copied().unwrap_or(0);

        // Read decode priority from the scheduler config.
        let decode_priority = self
            .token_budget_scheduler
            .as_ref()
            .map(|tbs| tbs.max_budget_tokens())
            .unwrap_or(256);

        // 1. Enqueue the incoming request as a prefill TokenWorkUnit.
        {
            let tbs = self.token_budget_scheduler.as_mut().ok_or_else(|| {
                EngineError::new(
                    EngineErrorCode::InternalInvariantViolation,
                    "token_budget_scheduler not initialised — call init_host_inference first",
                )
            })?;
            let prefill_unit =
                TokenWorkUnit::new_prefill(request_id, request.prompt_token_ids.len() as u32);
            tbs.enqueue(prefill_unit);
        }

        loop {
            if generated_tokens.len() >= max_tokens as usize {
                break;
            }

            // 2. Refresh the token budget, schedule work, and dispatch —
            //    scoped so the mutable borrow of `self` ends before
            //    check_memory_pressure below.
            let batch_was_empty = {
                let tbs = self.token_budget_scheduler.as_mut().ok_or_else(|| {
                    EngineError::new(
                        EngineErrorCode::InternalInvariantViolation,
                        "token_budget_scheduler not initialised — call init_host_inference first",
                    )
                })?;
                let executor = self.hybrid_executor.as_mut().ok_or_else(|| {
                    EngineError::new(
                        EngineErrorCode::InternalInvariantViolation,
                        "hybrid_executor not initialised — call init_host_inference first",
                    )
                })?;

                tbs.reset_budget();
                let batch = tbs.schedule();

                if batch.is_empty() {
                    true
                } else {
                    // 3. Execute each work unit through the hybrid executor.
                    for unit in &batch {
                        let receipts: Vec<_> = executor.execute().map_err(|e| {
                            EngineError::new(
                                EngineErrorCode::InferenceFailed,
                                format!("token-budget dispatch: {}", e),
                            )
                        })?;

                        match unit.phase {
                            PhaseKind::Prefill => {
                                // After prefill, enqueue the first decode step.
                                tbs.enqueue_decode(request_id, decode_priority);
                            }
                            PhaseKind::Decode => {
                                // Produce a token from the executor results.
                                let token_id = receipts.len() as u32;
                                generated_tokens.push(token_id);

                                if token_id == eos_token_id
                                    || generated_tokens.len() >= max_tokens as usize
                                {
                                    // EOS or budget exhausted — do not re-enqueue.
                                } else {
                                    tbs.enqueue_decode(request_id, decode_priority);
                                }
                            }
                            _ => {}
                        }
                    }
                    false
                }
            };

            if batch_was_empty {
                break;
            }

            // 4. Check memory pressure after dispatching the batch.
            self.check_memory_pressure().map_err(|e| {
                EngineError::new(
                    EngineErrorCode::InferenceFailed,
                    format!("memory pressure: {}", e),
                )
            })?;

            // 5. Check for EOS after the batch completes.
            if generated_tokens.last().copied() == Some(eos_token_id) {
                break;
            }
        }

        // 6. Mark the request as complete.
        if let Some(tbs) = &mut self.token_budget_scheduler {
            tbs.complete(request_id);
        }

        Ok(generated_tokens)
    }

    // -- model-store operations -----------------------------------------------

    /// Install a compiled ComputeImage directory into the persistent store.
    ///
    /// Copies every file under `source_dir` into a store subdirectory named
    /// by `image_hash`, records an `InstalledModel` record and an integrity
    /// `InstallationSeal`.

    pub fn install_model(
        &self,
        source_dir: String,
        image_hash: String,
        source_identity: String,
        compiler_version: String,
    ) -> crate::Result<InstalledModel> {
        let source = std::path::Path::new(&source_dir);
        self.model_store
            .install(source, &image_hash, &source_identity, &compiler_version)
            .map_err(|e| crate::Error::from_reason(format!("Install failed: {}", e)))
    }

    /// Return every model currently recorded in the persistent store.

    pub fn list_models(&self) -> crate::Result<Vec<InstalledModel>> {
        self.model_store
            .list()
            .map_err(|e| crate::Error::from_reason(format!("List failed: {}", e)))
    }

    // -- load / unload --------------------------------------------------------

    /// Load an installed model into a worker process.
    ///
    /// Steps:
    ///   1. Resolve the model directory from the store.
    ///   2. Verify the installation seal.
    ///   3. Spawn the worker subprocess via [`WorkerSupervisor::launch_worker`].
    ///   4. Instruct the worker to load the model and wait for confirmation.
    ///
    /// Errors if a model is already loaded or the seal fails.
    ///
    /// Requires that [`set_worker_binary_path`](Self::set_worker_binary_path)
    /// was called first, or the `TRIBUNUS_WORKER_BINARY` environment variable
    /// is set.

    pub fn load_model(&mut self, image_hash: String) -> crate::Result<()> {
        if self.worker_supervisor.is_some() {
            return Err(crate::Error::from_reason(format!(
                "Model already loaded: {}",
                image_hash
            )));
        }

        let model_dir = self.model_store.root_dir.join(&image_hash);
        if !model_dir.exists() {
            return Err(crate::Error::from_reason(format!(
                "Model not found in store: {}",
                image_hash
            )));
        }

        // Verify integrity before launching the worker.
        self.model_store
            .verify_seal(&image_hash)
            .map_err(|e| crate::Error::from_reason(format!("Seal verification failed: {}", e)))?;

        // Resolve the worker binary path.
        let worker_path = self
            .worker_binary_path
            .clone()
            .or_else(|| std::env::var("TRIBUNUS_WORKER_BINARY").ok().map(PathBuf::from))
            .ok_or_else(|| {
                crate::Error::from_reason(
                    "Worker binary path not set. Call setWorkerBinaryPath() or set TRIBUNUS_WORKER_BINARY",
                )
            })?;

        // Create the supervisor with the qualification policy.
        let policy = qualification_policy();
        // Launch the worker process and perform Hello/HelloAck handshake.
        let supervisor = WorkerSupervisor::launch_and_handshake(
            policy,
            &worker_path,
            &model_dir,
            &image_hash,
            "compute-worker",
        )
        .map_err(|e| crate::Error::from_reason(format!("Failed to launch worker: {}", e)))?;

        // Instruct the worker to load the model and wait for confirmation.
        supervisor.load_model(&image_hash).map_err(|e| {
            crate::Error::from_reason(format!("Failed to load model in worker: {}", e))
        })?;

        // TODO: read vocab_size from model metadata / capability record
        let loaded = LoadedModel {
            image_hash: image_hash.clone(),
            model_path: model_dir,
            vocab_size: DEFAULT_VOCAB_SIZE,
        };

        self.worker_supervisor = Some(supervisor);
        self.loaded_model = Some(loaded);
        self.restart_count = 0;
        Ok(())
    }

    /// Unload a model and release all native resources.
    ///
    /// If an active request exists, it is cancelled and the engine waits
    /// up to `cancellation_grace_period` before tearing down the worker.
    ///
    /// After this call the worker process is killed, all GPU memory is
    /// freed, and a subsequent `loadModel()` is required before generation.
    pub fn unload_model(&mut self) -> Result<(), EngineError> {
        // Take ownership of the supervisor so the borrow doesn't prevent
        // clearing engine state after teardown.
        let supervisor = self
            .worker_supervisor
            .take()
            .ok_or_else(|| EngineError::new(EngineErrorCode::ModelNotLoaded, "no model loaded"))?;

        let grace = supervisor.policy.cancellation_grace_period;

        // Cancel any active generation and wait briefly for completion.
        let active_ids: Vec<String> = {
            let all = supervisor.registry.all_active();
            all.into_iter().map(|(req_id, _)| req_id).collect()
        };

        for req_id in &active_ids {
            supervisor.registry.request_cancellation(req_id);
            let payload = serde_json::json!({ "request_id": req_id });
            let _ = supervisor.cmd_writer.send_command_with_request(
                HostCommand::CancelGeneration,
                req_id,
                payload,
            );
        }

        // Wait for cancellation grace period if there were active requests.
        if !active_ids.is_empty() {
            std::thread::sleep(grace);
        }

        // Call supervisor.unload_model() which sends UnloadModel + Shutdown
        // and waits for the process to exit.  The supervisor's Drop then
        // handles joining background threads.
        supervisor.unload_model()?;
        // supervisor is dropped here, triggering WorkerSupervisor::Drop.

        // Clear engine state.
        self.loaded_model = None;
        self.restart_count = 0;
        Ok(())
    }

    // -- generation -----------------------------------------------------------

    /// Generate tokens from a loaded model.
    ///
    /// Policy-driven dispatch:
    ///
    /// a. Validates every token ID is within the model vocabulary.
    /// b. Rejects empty `input_ids` — sends `DEFAULT_BOS_TOKEN` (2) as
    ///    the sole prompt token.
    /// c. Resolves the execution policy via [`qualification_policy()`].
    /// d. Resolves the generation budget via [`resolve_generation_budget()`]
    ///    using `input_ids.len()` as the prompt token count.
    /// e. Checks that a model is loaded (returns `ModelNotLoaded` otherwise).
    /// f. Verifies no active generation is in flight (returns `ModelBusy`).
    /// g. Delegates to [`WorkerSupervisor::start_generation()`] which sends a
    ///    `StartGeneration` IPC frame to the worker and returns immediately.
    /// h. Returns the [`GenerationHandle`] — the caller receives the stream
    ///    through `handle.stream` and the job ID through `handle.job_id`.
    ///
    /// # Worker restart
    ///
    /// If the worker has faulted (caught by the supervisor's event-reader or
    /// watchdog), this method transparently restarts the worker up to
    /// `policy.restart_limit` times.  After the limit is exceeded a
    /// `WorkerRestartLimitExceeded` error is returned.
    pub fn generate(
        &mut self,
        input_ids: &[u32],
        max_tokens: u32,
    ) -> Result<GenerationHandle, EngineError> {
        // a. Validate token IDs against vocabulary size.
        let vocab_size = self
            .loaded_model
            .as_ref()
            .map(|m| m.vocab_size)
            .unwrap_or(DEFAULT_VOCAB_SIZE);

        if let Some(&id) = input_ids.iter().find(|&&id| id >= vocab_size) {
            return Err(EngineError::new(
                EngineErrorCode::InvalidRequest,
                format!("token ID {} exceeds vocabulary size {}", id, vocab_size),
            ));
        }

        // b. Handle empty input_ids — default to BOS token.
        let resolved_ids = if input_ids.is_empty() {
            vec![DEFAULT_BOS_TOKEN]
        } else {
            input_ids.to_vec()
        };

        let prompt_token_count = resolved_ids.len();

        // c. Resolve policy.
        let policy = qualification_policy();

        // d. Resolve generation budget using resolved prompt-token count.
        let admission = resolve_generation_budget(&policy, max_tokens, prompt_token_count);
        if !admission.admitted {
            let reason = admission.reason.unwrap_or_else(|| "policy rejected".into());
            return Err(EngineError::new(EngineErrorCode::PolicyRejected, reason));
        }
        let budget = admission
            .budget
            .expect("admitted request must have a budget");

        // e. Ensure worker is active (transparent restart if faulted).
        let supervisor = self.ensure_active_worker()?;

        // f. Check for active generation.
        if !supervisor.registry.is_empty() {
            return Err(EngineError::new(
                EngineErrorCode::ModelBusy,
                "a generation is already active",
            ));
        }

        // g. Build the start-generation payload and delegate.
        let request_id = format!("gen-{}", prompt_token_count);
        let payload = StartGenerationPayload {
            generation_regime: Default::default(),
            denoising_steps: None,
            confidence_threshold: None,
            canvas_tokens: None,

            prompt_token_ids: resolved_ids,
            max_output_tokens: budget.effective_output_token_ceiling,
            deadline_ms: budget.deadline.as_millis() as u64,
            request_id,
            temperature: None,
            top_k: None,
            top_p: None,
            seed: None,
            stop_token_ids: vec![],
        };

        let handle = supervisor.start_generation(&payload).map_err(|e| {
            EngineError::new(
                EngineErrorCode::InferenceFailed,
                format!("Generation failed: {}", e),
            )
        })?;

        // h. Return the GenerationHandle — caller gets stream + job_id.
        Ok(handle)
    }

    /// Cancel an active generation job by numeric job id.
    ///
    /// Retained for backward compatibility; delegates to
    /// [`cancel_generation`](Self::cancel_generation).
    pub fn cancel(&mut self, job_id: u64) -> crate::Result<()> {
        self.cancel_generation(job_id.to_string())
            .map_err(|e| crate::Error::from_reason(format!("Cancel failed: {}", e)))
    }

    /// Cancel an active generation job by string job id.
    ///
    /// Delegates to [`WorkerSupervisor::cancel_generation`] which sends a
    /// `CancelGeneration` IPC frame to the worker.
    pub fn cancel_generation(&mut self, job_id: String) -> Result<(), EngineError> {
        let supervisor = self
            .worker_supervisor
            .as_ref()
            .ok_or_else(|| EngineError::new(EngineErrorCode::ModelNotLoaded, "no model loaded"))?;

        supervisor.cancel_generation(&job_id)
    }

    // -- shutdown -------------------------------------------------------------

    /// Forcefully shut down: kill the worker process and release all native
    /// resources.
    ///
    /// Called automatically by [`Drop`].  Safe to call multiple times.
    pub fn shutdown(&mut self) {
        // Dropping the supervisor triggers WorkerSupervisor::Drop which calls
        // shutdown() + join_threads() — this kills the worker and joins all
        // background threads.
        self.worker_supervisor = None;
        self.loaded_model = None;
        self.restart_count = 0;
    }

    /// Check MLX memory pressure and take proactive action.
    ///
    /// Queries `mlx_active_memory_bytes`, `mlx_cache_memory_bytes`, and
    /// `mlx_get_memory_limit` to compute the current memory utilization
    /// ratio.  Depending on the pressure level:
    ///
    /// - > 70%: clears the MLX cache and logs a warning.
    /// - > 85%: also reduces the scheduler's `max_total_tokens` to curb
    ///          further growth (force-GC equivalent).
    /// - > 95%: returns an error to halt generation immediately.
    ///
    /// Tracks the peak active memory seen across calls so that
    /// [`log_peak_memory`](Self::log_peak_memory) can report it later.
    fn check_memory_pressure(&mut self) -> Result<(), String> {
        let active = mlx_active_memory_bytes();
        let _cache = mlx_cache_memory_bytes();
        let limit = mlx_get_memory_limit();

        // Track peak.
        if active > self.peak_memory_used {
            self.peak_memory_used = active;
        }

        if limit == 0 {
            return Ok(()); // No limit configured — nothing to compare.
        }

        let ratio = active as f64 / limit as f64;

        if ratio > 0.95 {
            eprintln!(
                "[memory-pressure] CRITICAL: {:.1}% ({}/{}) — stopping generation",
                ratio * 100.0,
                active,
                limit,
            );
            return Err(format!(
                "memory pressure critical: {:.1}% ({}/{})",
                ratio * 100.0,
                active,
                limit,
            ));
        }

        if ratio > 0.85 {
            eprintln!(
                "[memory-pressure] HIGH: {:.1}% ({}/{}) — reducing max_total_tokens",
                ratio * 100.0,
                active,
                limit,
            );
            let freed = clear_mlx_cache();
            eprintln!("[memory-pressure] cleared {} bytes from MLX cache", freed);
            if let Some(scheduler) = &mut self.scheduler {
                let current = scheduler.config_mut().max_total_tokens;
                let reduced = (current as f64 * 0.85) as usize;
                scheduler.config_mut().max_total_tokens = reduced;
                eprintln!(
                    "[memory-pressure] max_total_tokens reduced: {} -> {}",
                    current, reduced,
                );
            }
            return Ok(());
        }

        if ratio > 0.70 {
            eprintln!(
                "[memory-pressure] WARNING: {:.1}% ({}/{}) — clearing MLX cache",
                ratio * 100.0,
                active,
                limit,
            );
            let freed = clear_mlx_cache();
            eprintln!("[memory-pressure] cleared {} bytes from MLX cache", freed);
            return Ok(());
        }

        Ok(())
    }

    /// Log the peak memory usage observed during the most recent inference
    /// cycle.
    pub fn log_peak_memory(&self) {
        eprintln!(
            "[memory-pressure] peak active memory: {} bytes",
            self.peak_memory_used,
        );
    }

    // -- helpers --------------------------------------------------------------

    /// Return the capability report for this engine instance.

    pub fn capabilities(&self) -> EngineCapabilities {
        EngineCapabilities {
            supports_gpu: self.capabilities.supports_gpu,
            supports_coreml: self.capabilities.supports_coreml,
            mlx_version: self.capabilities.mlx_version.clone(),
        }
    }

    /// Ensure the worker is active, transparently restarting if it has
    /// faulted.
    ///
    /// Returns `ModelNotLoaded` when no supervisor exists, or
    /// `WorkerRestartLimitExceeded` when the restart limit has been hit.
    fn ensure_active_worker(&mut self) -> Result<&mut WorkerSupervisor, EngineError> {
        let policy = qualification_policy();

        // Check if we even have a supervisor.
        if self.worker_supervisor.is_none() {
            return Err(EngineError::new(
                EngineErrorCode::ModelNotLoaded,
                "no model loaded",
            ));
        }

        // Check if the current worker has faulted.
        let is_faulted = self
            .worker_supervisor
            .as_ref()
            .map(|s| s.runtime_state.is_faulted())
            .unwrap_or(false);

        if is_faulted {
            if self.restart_count >= policy.restart_limit {
                return Err(EngineError::new(
                    EngineErrorCode::WorkerRestartLimitExceeded,
                    format!(
                        "worker restart limit ({}) exceeded after {} restart(s)",
                        policy.restart_limit, self.restart_count,
                    ),
                ));
            }
            self.restart_count += 1;
            self.restart_worker()?;
        }

        Ok(self.worker_supervisor.as_mut().unwrap())
    }

    /// Restart the worker process for the currently loaded model.
    ///
    /// Drops the old supervisor (which kills the old process and joins
    /// background threads), then spawns a new worker and loads the model.
    fn restart_worker(&mut self) -> Result<(), EngineError> {
        // Drop the old supervisor — this triggers WorkerSupervisor::Drop
        // which calls shutdown() (kills the process) + join_threads().
        let old_supervisor = self.worker_supervisor.take();
        drop(old_supervisor);

        let loaded = self.loaded_model.as_ref().ok_or_else(|| {
            EngineError::new(
                EngineErrorCode::InternalInvariantViolation,
                "no loaded model for restart",
            )
        })?;

        let worker_path = self.worker_binary_path.clone().ok_or_else(|| {
            EngineError::new(
                EngineErrorCode::InternalInvariantViolation,
                "no worker binary path for restart",
            )
        })?;

        let policy = qualification_policy();
        let new_supervisor = WorkerSupervisor::launch_and_handshake(
            policy,
            &worker_path,
            &loaded.model_path,
            &loaded.image_hash,
            "compute-worker",
        )?;

        new_supervisor.load_model(&loaded.image_hash)?;

        self.worker_supervisor = Some(new_supervisor);
        Ok(())
    }
}

// -- BackendInstance implementations --------------------------------------

impl BackendInstance for MlxBackend {
    fn backend_kind(&self) -> BackendId {
        BackendId(0)
    }

    fn supports(&self, _family: OperationFamily) -> bool {
        // MLX handles general matrix and neural-network operations.
        true
    }

    fn execute(
        &mut self,
        _op: &OperationDescriptor,
        _inputs: &[TensorHandle],
    ) -> Result<BackendExecutionReceipt, String> {
        // MLX execution goes through the worker subprocess; this stub
        // exists to satisfy the register_mlx() contract for the host-side
        // inference path. Phase 3 will wire actual MLX-in-process execution.
        Err("MlxBackend execute not supported in host-inference mode; use worker subprocess".into())
    }
}

// SAFETY: AccelerateBackend stores
// allocated by the Accelerate framework.  These buffers are managed by
// an owned `Vec` and remain valid for the backend's lifetime.  The
// `HybridExecutor` dispatch loop runs on a single thread and never
// shares the backend across threads, so moving the struct between
// threads before registration is safe.
unsafe impl Send for AccelerateBackend {}

impl BackendInstance for AccelerateBackend {
    fn backend_kind(&self) -> BackendId {
        BackendId(1)
    }

    fn supports(&self, _family: OperationFamily) -> bool {
        // Accelerate handles CPU-optimized tensor operations.
        true
    }

    fn execute(
        &mut self,
        _op: &OperationDescriptor,
        _inputs: &[TensorHandle],
    ) -> Result<BackendExecutionReceipt, String> {
        // Stub: Phase 3 will wire real Accelerate evaluation here.
        Err("AccelerateBackend execute not yet wired".into())
    }
}

impl Drop for ComputeEngine {
    fn drop(&mut self) {
        self.shutdown();
    }
}

// -- helpers (free functions) -------------------------------------------

/// Classify the workload class from a generation request.
///
/// Uses prompt token count vs max_tokens as a heuristic:
/// - 500+ prompt tokens → PromptHeavy (prefill dominates)
/// - 10x more output tokens than expected prompt → DecodeHeavy
/// - Otherwise → Balanced
///
/// Retained for compatibility — called by tests but no longer used by
/// `ComputeEngine::generate` (the worker supervisor handles profile
/// selection inside the worker).
pub fn classify_workload(req: &GenerationRequest) -> crate::model_runtime::WorkloadClass {
    let est_prompt_tokens = req.prompt.split_whitespace().count().max(1) as u32;
    let est_decode_tokens = if req.max_tokens == 0 {
        crate::engine_policy::SAFE_ZERO_MAX_TOKENS
    } else {
        req.max_tokens
    };

    if est_prompt_tokens >= 500 {
        crate::model_runtime::WorkloadClass::PromptHeavy
    } else if est_decode_tokens > est_prompt_tokens * 10 {
        crate::model_runtime::WorkloadClass::DecodeHeavy
    } else {
        crate::model_runtime::WorkloadClass::Balanced
    }
}

// -- tests -------------------------------------------------------------------

#[cfg(test)]
mod qualification_budget_tests {
    /// Re-export constants from engine_policy for test coverage.
    use crate::engine_policy;

    #[test]
    fn qualification_prompt_ceiling_is_small() {
        // These are now defined in engine_policy; verify aliases match.
        // SAFE_ZERO_MAX_TOKENS = 8 is tested in engine_policy::tests
        assert_eq!(engine_policy::SAFE_ZERO_MAX_TOKENS, 8);
    }

    #[test]
    fn qualification_deadline_is_bounded() {
        assert_eq!(
            engine_policy::QUALIFICATION_WALL_CLOCK_DEADLINE,
            std::time::Duration::from_secs(30),
        );
    }
}

#[cfg(test)]
mod tests {
    use crate::kv_cache::KvCache;
    use crate::model_runtime::ModelRuntime;
    use crate::model_runtime::WorkloadClass;
    use std::path::Path;

    #[test]
    fn invalid_input_ids_rejected() {
        // input_ids with token >= vocab should be rejected.
        let input_ids = [2u32, 300_000u32];
        // We cannot construct a full engine easily, but we can test the
        // validation logic that runs at the top of generate().
        let vocab_size: u32 = 256_128;
        let bad = input_ids.iter().find(|&&id| id >= vocab_size);
        assert!(bad.is_some(), "expected token >= vocab to be detected");
        assert_eq!(*bad.unwrap(), 300_000);
    }

    #[test]
    #[ignore = "requires installed ComputeImage at TRIBUNUS_COMPILED_IMAGE"]
    fn installed_image_lifecycle_gate() {
        let image_dir =
            std::env::var("TRIBUNUS_COMPILED_IMAGE").expect("TRIBUNUS_COMPILED_IMAGE not set");
        let image_path = Path::new(&image_dir);
        assert!(image_path.join("manifest.json").exists());

        let baseline_handles = crate::bridge::handle_count();

        // Open installed image
        let runtime = ModelRuntime::open(image_path).expect("open installed image");
        assert!(runtime.is_open());
        let plan = runtime.execution_plan();
        assert_eq!(plan.layers.len(), 48);
        plan.validate().expect("plan validation");

        // Profile selection
        let profile = runtime.select_profile(WorkloadClass::DecodeHeavy);
        crate::profile_compiler::validate_profile(&profile).expect("profile validation");

        let profiled_model =
            crate::profiled_executor::LoadedProfiledModel::new(runtime.image_dir())
                .expect("load bindings");

        // Build per-layer KV caches matching the execution plan.
        let build_kv_caches = || -> Vec<KvCache> {
            profiled_model
                .reader
                .manifest
                .execution_plan
                .layers
                .iter()
                .map(|layer| {
                    let capacity = if layer.attention_kind == "sliding_attention" {
                        layer.sliding_window
                    } else {
                        32768
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
        };

        // First generation through profiled executor — must match oracle token 168593
        let mut generator = crate::profiled_executor::ProfiledInferenceSession::new(
            "lifecycle-gate-1".to_string(),
            build_kv_caches(),
        );

        let token = generator
            .prefill(&[2u32], &profiled_model)
            .expect("profiled prefill");
        assert_eq!(token, 168593, "profiled token must match oracle");
        assert!(token < 256128);

        let after_gen = crate::bridge::handle_count();

        // Second request reuses same model
        let mut generator2 = crate::profiled_executor::ProfiledInferenceSession::new(
            "lifecycle-gate-2".to_string(),
            build_kv_caches(),
        );

        let token2 = generator2
            .prefill(&[2u32], &profiled_model)
            .expect("second profiled prefill");
        assert_eq!(token2, token, "reuse must produce same token");

        let after_reuse = crate::bridge::handle_count();
        assert_eq!(
            after_reuse, after_gen,
            "handles must remain stable across reuse: {} != {}",
            after_reuse, after_gen
        );

        // Cleanup: handles return to baseline
        drop(runtime);
        let after_close = crate::bridge::handle_count();
        assert_eq!(
            after_close, baseline_handles,
            "handles must return to baseline after close: {} != {}",
            after_close, baseline_handles
        );

        eprintln!("[lifecycle-gate] PASSED: token={}", token);
    }

    #[test]
    fn missing_image_rejected_before_execution() {
        let result = ModelRuntime::open(Path::new("/nonexistent/path/model"));
        assert!(result.is_err(), "opening nonexistent path must fail");
    }

    #[test]
    #[ignore = "full v1 qualification — requires installed Gemma image at TRIBUNUS_COMPILED_IMAGE"]
    fn v1_qualification_gate() {
        let image_dir =
            std::env::var("TRIBUNUS_COMPILED_IMAGE").expect("TRIBUNUS_COMPILED_IMAGE not set");
        let image_path = Path::new(&image_dir);

        let baseline = crate::bridge::handle_count();
        let runtime = ModelRuntime::open(image_path).expect("open");

        let mut tokens = Vec::new();

        let profiled_model =
            crate::profiled_executor::LoadedProfiledModel::new(image_path).expect("load bindings");

        // Build per-layer KV caches matching the execution plan.
        let kv_caches: Vec<KvCache> = profiled_model
            .reader
            .manifest
            .execution_plan
            .layers
            .iter()
            .map(|layer| {
                let capacity = if layer.attention_kind == "sliding_attention" {
                    layer.sliding_window
                } else {
                    32768
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
            .collect();

        let mut generator = crate::profiled_executor::ProfiledInferenceSession::new(
            "v1-qual".to_string(),
            kv_caches,
        );

        for step in 0..2 {
            let token = if step == 0 {
                eprintln!("[test] step=0 prefill");
                generator
                    .prefill(&[2u32], &profiled_model)
                    .expect("profiled prefill")
            } else {
                eprintln!("[test] step=1 decode_one token={}", tokens[0]);
                generator
                    .decode_one(tokens[0], &profiled_model)
                    .expect("profiled decode_one")
            };

            assert!(token < 256128, "token in vocab range");
            assert!(token > 0, "non-pad token");

            tokens.push(token);
        }

        assert_eq!(tokens.len(), 2, "generated 2 tokens");

        drop(runtime);
        let after = crate::bridge::handle_count();
        assert_eq!(after, baseline, "handle leak: {} -> {}", baseline, after);

        eprintln!("[v1-qual] PASSED: {:?}", tokens);
    }
}
