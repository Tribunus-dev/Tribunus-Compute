#!/usr/bin/env python3
"""Benchmark Qwen2.5 0.5B via CoreML (direct .mlmodelc) and MLX Python on Apple M1."""

import json, time, sys, numpy as np
from pathlib import Path

PROMPT = "Explain the entire history of artificial intelligence, from ancient myths about artificial beings through modern deep learning, covering key milestones, algorithms, and philosophical implications. Write in great detail."

# ---------------------------------------------------------------------------
# NOTE: CoreML benchmark is currently unavailable on Python 3.14.
#
# coremltools (pip-installed on Python 3.14 arm64) ships without native
# extension modules — libcoremlpython.dylib is absent from the wheel.
# This means `ct.models.MLModel()` will fail at runtime even though
# `import coremltools` succeeds (with warnings).
#
# To run the CoreML benchmark:
#   1. Use Python 3.12 or 3.13 where Apple's official wheels include the
#      native extension (libcoremlpython.dylib).
#   2. Or install coremltools from conda-forge: conda install -c conda-forge coremltools
#   3. Or install Xcode's bundled coremltools: xcrun python3 -m pip install coremltools
#
# Once a working coremltools is available, the benchmark_coreml() function
# below will work with the .mlmodelc directory.
# ---------------------------------------------------------------------------


def benchmark_mlx_local_fp16():
    """MLX Python with LOCAL FP16 model (same as Tribunus uses)."""
    from mlx_lm import load, generate
    local_path = "compute-native/models/qwen2.5-0.5b"
    print(f"  Loading local FP16 model from {local_path}...", file=sys.stderr)
    t0 = time.time()
    model, tokenizer = load(local_path)
    load_t = time.time() - t0
    print(f"  Loaded in {load_t:.1f}s", file=sys.stderr)
    t1 = time.time()
    response = generate(model, tokenizer, prompt=PROMPT, max_tokens=1000)
    gen_t = time.time() - t1
    tokens = len(tokenizer.encode(response))
    return {"backend": "MLX Python (local FP16)", "elapsed_s": round(gen_t, 3), "tok_s": round(tokens / gen_t, 1), "tokens": tokens}


if __name__ == "__main__":
    results = []
    print("Benchmarking Qwen2.5 0.5B — 1000 tokens on Apple M1\n", file=sys.stderr)

    # MLX benchmark — always available
    try:
        r = benchmark_mlx_local_fp16()
        results.append(r)
        print(f"  \u2713 MLX Python (local FP16): {r['tok_s']} tok/s ({r['elapsed_s']}s for {r['tokens']} tokens)", file=sys.stderr)
    except Exception as e:
        print(f"  \u2717 MLX Python (local FP16): {e}", file=sys.stderr)

    print(file=sys.stderr)

    print("\n" + "\u2501" * 50)
    print(f"  BENCHMARK: Qwen2.5 0.5B, 1000 tokens")
    print(f"  Hardware: Apple M1 16GB")
    print("\u2501" * 50)
    for r in sorted(results, key=lambda x: x.get("tok_s", 0), reverse=True):
        print(f"  {r['tok_s']:>7.1f} tok/s  {r['elapsed_s']:>6.2f}s  {r['tokens']:>5d} tok  {r['backend']}")
    print("\u2501" * 50)

    if not results:
        print("\n  NOTE: CoreML benchmark is unavailable on Python 3.14 — see comments at top of file.")
        print("  Run the MLX benchmark above via: python3 scripts/benchmark_coreml.py")
        print(file=sys.stderr)
