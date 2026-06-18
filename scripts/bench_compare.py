#!/usr/bin/env python3
"""Cross-backend benchmark: Ollama (llama.cpp) vs MLX Python vs Tribunus.
Hardware: Apple M1 16GB | Model: Qwen2.5 0.5B | 50 tokens | greedy."""

import sys, time, json, subprocess, urllib.request
from pathlib import Path

PROMPT = "Explain the concept of recursive functions in programming."
MAX_TOKENS = 50

def bench_ollama():
    url = "http://localhost:11434/v1/chat/completions"
    payload = json.dumps({"model":"qwen2.5:0.5b","messages":[{"role":"user","content":PROMPT}],"max_tokens":MAX_TOKENS,"stream":False,"temperature":0}).encode()
    req = urllib.request.Request(url, data=payload, headers={"Content-Type":"application/json"})
    t0 = time.time()
    resp = urllib.request.urlopen(req, timeout=120)
    data = json.loads(resp.read())
    elapsed = time.time() - t0
    n = data["usage"]["completion_tokens"]
    return (n, elapsed)

def bench_mlx():
    from mlx_lm import load, generate
    from mlx_lm.sample_utils import make_sampler
    model, tokenizer = load("Qwen/Qwen2.5-0.5B-Instruct")
    text = tokenizer.apply_chat_template([{"role":"user","content":PROMPT}], tokenize=False, add_generation_prompt=True)
    sampler = make_sampler(temp=0)
    t0 = time.time()
    response = generate(model, tokenizer, prompt=text, max_tokens=MAX_TOKENS, sampler=sampler)
    elapsed = time.time() - t0
    n = len(tokenizer.encode(response))
    return (n, elapsed)

def bench_tribunus(port=11435):
    url = f"http://localhost:{port}/v1/chat/completions"
    payload = json.dumps({"model":"qwen2.5:0.5b","messages":[{"role":"user","content":PROMPT}],"max_tokens":MAX_TOKENS,"stream":False,"temperature":0}).encode()
    req = urllib.request.Request(url, data=payload, headers={"Content-Type":"application/json"})
    t0 = time.time()
    try:
        resp = urllib.request.urlopen(req, timeout=120)
        data = json.loads(resp.read())
        elapsed = time.time() - t0
        n = data["usage"]["completion_tokens"]
        return (n, elapsed)
    except Exception as e:
        return (None, str(e))

if __name__ == "__main__":
    results = []
    for name, fn in [("Ollama (llama.cpp)", bench_ollama), ("MLX Python", bench_mlx)]:
        try:
            n, e = fn()
            results.append((name, n, e, round(n/e, 1)))
            print(f"  {name:25s} {n:>4d} tok  {e:.2f}s  {n/e:.1f} tok/s")
        except Exception as ex:
            print(f"  {name:25s} ERROR: {ex}")

    print(f"\n{'='*55}")
    print(f"  Qwen2.5 0.5B  |  Apple M1 16GB  |  {MAX_TOKENS} tokens")
    print(f"{'='*55}")
    for name, n, e, t in sorted(results, key=lambda x: -x[3]):
        print(f"  {name:25s}  {t:>5.1f} tok/s  ({n} tok in {e:.2f}s)")
    print(f"{'='*55}")
