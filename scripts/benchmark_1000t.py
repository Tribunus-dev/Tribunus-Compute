#!/usr/bin/env python3
"""Benchmark 1000-token generation across Tribunus, MLX Python."""

import json, subprocess, time, sys
from urllib.request import Request, urlopen

PROMPT = "Explain the entire history of artificial intelligence, from ancient myths about artificial beings through modern deep learning, covering key milestones, algorithms, and philosophical implications. Write in great detail."

def benchmark_tribunus(port=11434):
    """Benchmark Tribunus server at localhost:port."""
    payload = json.dumps({
        "model": "qwen2.5:0.5b",
        "messages": [{"role": "user", "content": PROMPT}],
        "max_tokens": 1000,
        "stream": False,
    }).encode()
    req = Request(f"http://localhost:{port}/v1/chat/completions",
                  data=payload,
                  headers={"Content-Type": "application/json"})
    t0 = time.time()
    resp = urlopen(req)
    data = json.loads(resp.read())
    elapsed = time.time() - t0
    tokens = data["usage"]["completion_tokens"]
    tok_sec = tokens / elapsed if elapsed > 0 else 0
    return {"backend": "Tribunus (Metal+ANE+Accel)", "tokens": tokens, "elapsed_s": round(elapsed, 3), "tok_s": round(tok_sec, 1)}

def benchmark_mlx():
    """Benchmark MLX Python directly."""
    import mlx.core as mx
    from mlx_lm import load, generate
    t0 = time.time()
    model, tokenizer = load("mlx-community/Qwen2.5-0.5B-4bit")
    response = generate(model, tokenizer, prompt=PROMPT, max_tokens=1000)
    elapsed = time.time() - t0
    tokens = len(tokenizer.encode(response))
    tok_sec = tokens / elapsed if elapsed > 0 else 0
    return {"backend": "MLX Python (GPU)", "tokens": tokens, "elapsed_s": round(elapsed, 3), "tok_s": round(tok_sec, 1)}

def benchmark_llamacpp():
    """Benchmark llama.cpp if available."""
    import shutil
    if not shutil.which("llama-cli"):
        return {"backend": "llama.cpp", "error": "not installed"}
    t0 = time.time()
    result = subprocess.run([
        "llama-cli",
        "-m", "compute-native/models/qwen2.5-0.5b/model.gguf",
        "--prompt", PROMPT,
        "-n", "1000",
        "--no-display-prompt",
        "--temp", "0.7"
    ], capture_output=True, text=True)
    elapsed = time.time() - t0
    output = result.stdout
    tokens = len(output.split())
    tok_sec = tokens / elapsed if elapsed > 0 else 0
    return {"backend": "llama.cpp (Metal)", "tokens": tokens, "elapsed_s": round(elapsed, 3), "tok_s": round(tok_sec, 1)}

if __name__ == "__main__":
    results = []
    # Tribunus
    try:
        r = benchmark_tribunus()
        results.append(r)
        print(f"  ✓ Tribunus: {r['tok_s']} tok/s ({r['elapsed_s']}s for {r['tokens']} tokens)")
    except Exception as e:
        print(f"  ✗ Tribunus: {e}")
    # MLX Python
    try:
        r = benchmark_mlx()
        results.append(r)
        print(f"  ✓ MLX: {r['tok_s']} tok/s ({r['elapsed_s']}s for {r['tokens']} tokens)")
    except Exception as e:
        print(f"  ✗ MLX: {e}")
    # llama.cpp
    try:
        r = benchmark_llamacpp()
        if "error" in r:
            print(f"  - llama.cpp: {r['error']}")
        else:
            results.append(r)
            print(f"  ✓ llama.cpp: {r['tok_s']} tok/s ({r['elapsed_s']}s for {r['tokens']} tokens)")
    except Exception as e:
        print(f"  ✗ llama.cpp: {e}")
    # Summary
    print("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    print("  BENCHMARK: Qwen2.5 0.5B, 1000 tokens")
    print("  Hardware: Apple M1 16GB")
    print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    for r in sorted(results, key=lambda x: x.get("tok_s", 0), reverse=True):
        print(f"  {r['tok_s']:>7.1f} tok/s  {r['elapsed_s']:>6.2f}s  {r['tokens']:>5d} tok  {r['backend']}")
    print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
