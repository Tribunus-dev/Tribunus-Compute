use std::path::Path;
use std::time::Instant;
use tribunus_compute_core::compute_image::{CompiledImageReader, StorageBackend};
use tribunus_compute_core::kv_cache::KvCache;
use tribunus_compute_core::profiled_executor::{LoadedProfiledModel, ProfiledInferenceSession};

fn main() {
    // Need to set this env var because we patched the manifest
    unsafe {
        std::env::set_var("TRIBUNUS_SKIP_MANIFEST_HASH", "1");
    }

    let image_dir = Path::new("compute-native/models/qwen-compiled");
    let model = LoadedProfiledModel::new(image_dir).expect("Failed to load model");

    let n_layers = model.reader.manifest.execution_plan.layers.len();
    let kv_caches: Vec<KvCache> = (0..n_layers)
        .map(|_| KvCache::new(2048, 128, 2, 64))
        .collect();
    let mut session = ProfiledInferenceSession::new("bench".into(), kv_caches);
    session.setup_from_model(&model);

    println!("Model loaded: {} layers", n_layers);

    // Warmup
    let prompt = vec![1u32; 10];
    let mut tokens = prompt.clone();
    for _step in 0..3 {
        let logits = session.step(&model, &tokens, 0).expect("warmup step");
        let next = argmax(logits);
        tokens.push(next);
    }

    // Benchmark
    let prompt = vec![1u32; 10];
    let mut tokens = prompt.clone();
    let n_gen = 50;
    let start = Instant::now();
    for _step in 0..n_gen {
        let logits = session.step(&model, &tokens, 0).expect("bench step");
        let next = argmax(logits);
        tokens.push(next);
    }
    let elapsed = start.elapsed();
    let tok_s = n_gen as f64 / elapsed.as_secs_f64();
    println!(
        "{} tokens in {:.2}s = {:.1} tok/s",
        n_gen,
        elapsed.as_secs_f64(),
        tok_s
    );
}

fn argmax(logits: &[f32]) -> u32 {
    let mut best = 0u32;
    let mut best_val = logits[0];
    for (i, &v) in logits.iter().enumerate().skip(1) {
        if v > best_val {
            best_val = v;
            best = i as u32;
        }
    }
    best
}
