#!/usr/bin/env python3
"""Benchmark CoreML model directly via pyobjc system framework."""

import time, sys, numpy as np
from pathlib import Path

PROMPT = "Explain AI in 2-3 sentences."
MODEL_PATH = str(Path("models/qwen-coreml/Qwen2.5-0.5B-Instruct-4bit.mlmodelc").resolve())

def benchmark_coreml_pyobjc():
    """Load and run CoreML model via the system CoreML framework using pyobjc."""
    import Foundation
    import CoreML
    
    print(f"  Loading model from {MODEL_PATH}...", file=sys.stderr)
    t0 = time.time()
    
    model_url = Foundation.NSURL.fileURLWithPath_(MODEL_PATH)
    model, error = CoreML.MLModel.modelWithContentsOfURL_error_(model_url, None)
    if error:
        raise RuntimeError(f"Failed to load model: {error}")
    
    load_time = time.time() - t0
    print(f"  Loaded in {load_time:.1f}s", file=sys.stderr)
    
    # Set all compute units
    model.setComputeUnits_(3)  # 3 = CPU+GPU+ANE (all)
    
    # Get model spec for input/output names
    spec = model.modelDescription()
    input_name = spec.inputDescriptions()[0].name()
    output_name = spec.outputDescriptions()[0].name()
    print(f"  Input: {input_name}, Output: {output_name}", file=sys.stderr)
    
    # Run autoregressive generation
    import tokenizers
    tok = tokenizers.Tokenizer.from_file("models/qwen-coreml/tokenizer.json")
    input_ids = tok.encode(PROMPT).ids
    all_tokens = list(input_ids)
    generated = []
    max_tokens = 100  # shorter for CoreML demo
    
    t_gen = time.time()
    for step in range(max_tokens):
        # Prepare input array
        arr = np.array([all_tokens], dtype=np.int32)
        arr_ml = CoreML.MLMultiArray.alloc().initWithDataPointer_shape_dataType_strides_deallocator_(
            arr.tobytes(), [1, arr.shape[1]], CoreML.MLMultiArrayDataType.int32, None, None
        )
        feature_dict = {input_name: arr_ml}
        input_batch = CoreML.MLDictionaryFeatureProvider.dictionaryWithDictionary_error_(feature_dict, None)
        
        # Run prediction
        t_pred = time.time()
        result, error = model.predictionFromFeatures_error_(input_batch, None)
        pred_time = time.time() - t_pred
        
        # Get next token
        output = result.featureValueForFeature_(output_name)
        multi_array = output.multiArrayValue()
        if multi_array is None:
            break
        ptr = multi_array.dataPointer()
        last_logits = np.frombuffer(ptr, dtype=np.float16)
        # Get argmax
        token = int(np.argmax(last_logits))
        
        generated.append(token)
        all_tokens.append(token)
        
        if (step + 1) % 10 == 0:
            elapsed = time.time() - t_gen
            print(f"    {step+1}/{max_tokens} tokens: {step/elapsed:.1f} tok/s", file=sys.stderr)
        
        if token == 151645:  # EOS
            break
    
    elapsed = time.time() - t_gen
    tok_s = len(generated) / elapsed if elapsed > 0 else 0
    print(f"  Generated {len(generated)} tokens in {elapsed:.1f}s ({tok_s:.1f} tok/s)", file=sys.stderr)
    
    # Decode
    text = tok.decode(generated)
    return {
        "backend": "Vanilla CoreML (CPU+GPU+ANE)",
        "elapsed_s": round(elapsed, 3),
        "tok_s": round(tok_s, 1),
        "tokens": len(generated),
    }

if __name__ == "__main__":
    try:
        r = benchmark_coreml_pyobjc()
        print(f"  Result: {r['tok_s']} tok/s, {r['elapsed_s']}s, {r['tokens']} tokens")
    except Exception as e:
        print(f"  Error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
