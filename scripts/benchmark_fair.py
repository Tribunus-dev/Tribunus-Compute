#!/usr/bin/env python3
"""Fair 1000-token benchmark across Tribunus and MLX Python, with model-agnostic timing."""

import json, subprocess, time, sys
from urllib.request import Request, urlopen

PROMPT = "Explain the entire history of artificial intelligence, from ancient myths about artificial beings through modern deep learning, covering key milestones, algorithms, and philosophical implications. Write in great detail."
MODEL = "qwen2.5:0.5b"

def benchmark_tribunus(port=11434):
    """Benchmark Tribunus — times inference only (model already cached)."""
    # Warmup
    req = Request(f"http://localhost:{port}/v1/chat/completions",
                  data=json.dumps({"model": MODEL, "messages": [{"role":"user","content":"Hello"}], "max_tokens":5}).encode(),
                  headers={"Content-Type": "application/json"})
    urlopen(req)
    
    # Benchmark
    payload = json.dumps({
        "model": MODEL,
        "messages": [{"role": "user", "content": PROMPT}],
        "max_tokens": 1000,
        "stream": False,
    }).encode()
    
    t0 = time.time()
    resp = urlopen(Request(f"http://localhost:{port}/v1/chat/completions",
                  data=payload, headers={"Content-Type": "application/json"}))
    data = json.loads(resp.read())
    elapsed = time.time() - t0
    tokens = data["usage"]["completion_tokens"]
    return {"backend": "Tribunus (Rust GPU)", "elapsed_s": round(elapsed, 3), "tok_s": round(tokens / elapsed, 1), "tokens": tokens}

def benchmark_mlx_python_cached():
    """MLX Python — times load+inference, WITHOUT download."""
    from mlx_lm import load, generate
    model_path = "mlx-community/Qwen2.5-0.5B-4bit"
    print(f"  Loading {model_path} from HF cache...", file=sys.stderr)
    t0 = time.time()
    model, tokenizer = load(model_path)
    load_elapsed = time.time() - t0
    
    # Generate 1000 tokens
    t1 = time.time()
    response = generate(model, tokenizer, prompt=PROMPT, max_tokens=1000)
    gen_elapsed = time.time() - t1
    total = time.time() - t0
    
    tokens = len(tokenizer.encode(response))
    return {
        "backend": "MLX Python (GPU 4-bit)",
        "elapsed_s": round(gen_elapsed, 3),
        "total_s": round(total, 3),
        "tok_s": round(tokens / gen_elapsed, 1) if gen_elapsed > 0 else 0,
        "tokens": tokens
    }

def benchmark_llamacpp():
    """llama.cpp — if GGUF exists."""
    import glob, os
    gguvs = glob.glob("models/qwen-gguf/*.gguf")
    if not gguvs:
        return {"backend": "llama.cpp", "error": "no GGUF found"}
    gguf = gguvs[0]
    t0 = time.time()
    result = subprocess.run([
        "llama-cli", "-m", gguf, "--prompt", PROMPT,
        "-n", "1000", "--no-display-prompt", "--temp", "0.7", "-ngl", "99"
    ], capture_output=True, text=True, timeout=180)
    elapsed = time.time() - t0
    output = result.stdout
    tokens = len(output.split())
    return {"backend": "llama.cpp (Metal)", "elapsed_s": round(elapsed, 3), "tok_s": round(tokens / elapsed, 1), "tokens": tokens}

def benchmark_mlx_python_4bit_local():
    """MLX Python using our local model files (FP16 or quantized)."""
    from mlx_lm import load, generate
    local_path = "compute-native/models/qwen2.5-0.5b"
    print(f"  Loading local model from {local_path}...", file=sys.stderr)
    t0 = time.time()
    model, tokenizer = load(local_path)
    gen_elapsed = time.time() - t0
    t1 = time.time()
    response = generate(model, tokenizer, prompt=PROMPT, max_tokens=1000)
    gen_end = time.time() - t1
    tokens = len(response.split())
    return {
        "backend": "MLX Python (local FP16)",
        "elapsed_s": round(gen_end, 3),
        "tok_s": round(tokens / gen_end, 1) if gen_end > 0 else 0,
        "tokens": tokens
    }

if __name__ == "__main__":
    results = []
    print("Benchmarking Qwen2.5 0.5B — 1000 tokens on Apple M1...", file=sys.stderr)
    
    for name, fn in [("Tribunus", benchmark_tribunus), 
                      ("MLX Python (HF 4-bit)", benchmark_mlx_python_cached)]:
        try:
            r = fn()
            results.append(r)
            tok = r.get("tok_s", 0)
            s = r.get("elapsed_s", 0) or r.get("total_s", 0)
            print(f"  \u2713 {name}: {tok} tok/s ({s}s)", file=sys.stderr)
        except Exception as e:
            print(f"  \u2717 {name}: {e}", file=sys.stderr)
    
    print("\n\u2501" * 50)
    print(f"  BENCHMARK: Qwen2.5 0.5B, 1000 tokens")
    print(f"  Hardware: Apple M1 16GB")
    print("\u2501" * 50)
    for r in sorted(results, key=lambda x: x.get("tok_s", 0), reverse=True):
        print(f"  {r['tok_s']:>7.1f} tok/s  {r['elapsed_s']:>6.2f}s  {r['tokens']:>5d} tok  {r['backend']}")
    print("\u2501" * 50)
