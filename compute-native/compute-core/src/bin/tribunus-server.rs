use tribunus_compute_core::server::{
    benchmark,
    models::{ModelRegistry, recommend_models},
    routes::{AppState, create_router},
};
use std::sync::Arc;
use tokio::sync::Mutex;

#[tokio::main]
async fn main() {
    println!("Tribunus Compute Server v0.1.0");

    // 1. Run system benchmark
    println!("Benchmarking system...");
    let bench = benchmark::run_benchmark();
    println!("  Chip: {}", bench.chip);
    println!("  RAM: {} GB", bench.ram_gb);

    // 2. Create model registry with recommendations
    let mut registry = ModelRegistry::new();
    for model in recommend_models(&bench.chip, bench.ram_gb, "chat") {
        registry.register(model);
    }
    println!("  Recommended {} models", registry.list().len());

    // 3. Start server
    let state = AppState {
        models: Arc::new(Mutex::new(registry)),
        benchmark: Arc::new(Mutex::new(Some(bench))),
    };

    let app = create_router(state);
    let listener = tokio::net::TcpListener::bind("0.0.0.0:11434")
        .await
        .expect("Failed to bind to port 11434");
    println!("Server running on http://0.0.0.0:11434");
    println!(" (Ollama-compatible API)");

    axum::serve(listener, app).await.unwrap();
}
