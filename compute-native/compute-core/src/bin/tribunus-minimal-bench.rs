use std::path::Path;
use std::time::Instant;
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
        .map(|_| KvCache::new(2048, 128, 64, false))
        .collect();
    let mut session = ProfiledInferenceSession::new("bench".into(), kv_caches);
    session.setup_from_model(&model);

    println!("Model loaded: {} layers", n_layers);

    // Warmup
    let prompt = vec![1u32; 10];
    let mut next = session.prefill(&prompt, &model).expect("warmup prefill");
    for _step in 0..2 {
        next = session.decode_one(next, &model).expect("warmup decode");
    }

    // Benchmark
    let prompt = vec![1u32; 10];
    let mut next = session.prefill(&prompt, &model).expect("bench prefill");
    let n_gen = 50;
    let start = Instant::now();
    for _step in 0..n_gen {
        next = session.decode_one(next, &model).expect("bench decode");
    }
    let elapsed = start.elapsed();
    let tok_s = (n_gen as f64) / elapsed.as_secs_f64();
    println!(
        "{} tokens in {:.2}s = {:.1} tok/s",
        n_gen,
        elapsed.as_secs_f64(),
        tok_s
    );
}
