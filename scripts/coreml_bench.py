#!/usr/bin/env python3.13
"""CoreML benchmark — .mlmodelc with CVPixelBuffer-backed MLState KV cache."""

import sys, time, numpy as np, ctypes
from pathlib import Path

MODEL_PATH = Path("models/qwen-coreml/Qwen2.5-0.5B-Instruct-4bit.mlmodelc")
TOKENIZER_PATH = Path("models/qwen-coreml/tokenizer.json")
PROMPT = "Explain the concept of recursive functions in programming."
MAX_TOKENS = 50
VOCAB = 151936
EOS_ID = 151645

SHAPE = (24, 1, 2, 2048, 64)
TOTAL = 24 * 1 * 2 * 2048 * 64
PB_W, PB_H = 2048 * 64, 48  # flatten 5D → 2D pixel buffer

def load_model(path: str):
    import CoreML, Foundation
    url = Foundation.NSURL.fileURLWithPath_(path)
    config = CoreML.MLModelConfiguration.alloc().init()
    config.setComputeUnits_(3)  # all
    model, err = CoreML.MLModel.modelWithContentsOfURL_configuration_error_(url, config, None)
    if err: raise RuntimeError(f"Model: {err}")
    desc = model.modelDescription()
    inputs = sorted(desc.inputDescriptionsByName().keys())
    outputs = sorted(desc.outputDescriptionsByName().keys())
    states = sorted(desc.stateDescriptionsByName().keys())
    return model, inputs, outputs, states


def make_state(name: str):
    """CVPixelBuffer-backed MLState for KV cache."""
    import Quartz, CoreML, Foundation
    FMT = Quartz.kCVPixelFormatType_OneComponent16Half
    ATTRS = {Quartz.kCVPixelBufferIOSurfacePropertiesKey: {},
             Quartz.kCVPixelBufferMetalCompatibilityKey: True}
    s, pb = Quartz.CVPixelBufferCreate(None, PB_W, PB_H, FMT, ATTRS, None)
    assert s == 0, f"CVPixelBufferCreate: {s}"

    # Lock & zero-fill via numpy view
    Quartz.CVPixelBufferLockBaseAddress(pb, 0)
    ba = Quartz.CVPixelBufferGetBaseAddress(pb)
    buf = np.ctypeslib.as_array(ba, shape=(TOTAL,))
    buf[:] = 0
    Quartz.CVPixelBufferUnlockBaseAddress(pb, 0)

    multi = CoreML.MLMultiArray.alloc().initWithPixelBuffer_shape_(pb, SHAPE)
    k = Foundation.NSString.stringWithUTF8String_(name.encode())
    d = Foundation.NSMutableDictionary.dictionary()
    d.setObject_forKey_(multi, k)
    return CoreML.MLState.alloc().initWithBackings_(d), pb


def make_input_multi(arr, dt_code):
    import CoreML
    itemsize = 2 if dt_code == 65536 else 4
    shape = tuple(arr.shape)
    strides = tuple(s * itemsize for s in arr.strides)
    return CoreML.MLMultiArray.alloc() \
        .initWithBytesNoCopy_shape_dataType_strides_deallocator_mutableShapedBufferProvider_error_(
            ctypes.c_void_p(arr.ctypes.data), shape,
            CoreML.MLMultiArrayDataType(dt_code), strides, None, None, None)


def build_provider(ids_np, mask_np, state_objs, state_pbs):
    import CoreML, Foundation
    multi_ids = make_input_multi(ids_np, 0)
    multi_mask = make_input_multi(mask_np, 65536)
    d = Foundation.NSMutableDictionary.dictionary()
    for key, val in [(b"input_ids", multi_ids), (b"causal_mask", multi_mask),
                     (b"key_cache", state_objs[b"key_cache"]),
                     (b"value_cache", state_objs[b"value_cache"])]:
        d.setObject_forKey_(val, Foundation.NSString.stringWithUTF8String_(key))
    prov, err = CoreML.MLDictionaryFeatureProvider.alloc().initWithDictionary_error_(d, None)
    if prov is None: raise RuntimeError(f"Provider: {err}")
    return prov


def run(model, tok, prompt, max_tokens):
    prompt_ids = tok.encode(prompt).ids
    all_tokens = list(prompt_ids)
    generated = []
    decode_times = []

    sk, pbk = make_state(b"key_cache")
    sv, pbv = make_state(b"value_cache")
    state_objs = {b"key_cache": sk, b"value_cache": sv}
    state_pbs = {b"key_cache": pbk, b"value_cache": pbv}

    # Prefill
    seq = len(prompt_ids)
    ids_np = np.array([all_tokens], dtype=np.int32)
    mask_np = np.ones((1, 1, 1, seq), dtype=np.float16)

    t0 = time.time()
    prov = build_provider(ids_np, mask_np, state_objs, state_pbs)
    result, err = model.predictionFromFeatures_error_(prov, None)
    ttft = time.time() - t0
    if err: raise RuntimeError(f"Prefill: {err}")

    la = result.featureValueForFeature_("logits").multiArrayValue()
    logits = np.frombuffer(ctypes.c_char_p(la.dataPointer()), dtype=np.float16, count=la.count())
    token = int(np.argmax(logits[(seq-1)*VOCAB:seq*VOCAB]))
    generated.append(token)
    all_tokens.append(token)
    if token == EOS_ID: return generated, ttft, decode_times

    for step in range(1, max_tokens):
        seq = len(all_tokens)
        ids_np = np.array([[token]], dtype=np.int32)
        mask_np = np.ones((1, 1, 1, seq), dtype=np.float16)

        t0 = time.time()
        prov = build_provider(ids_np, mask_np, state_objs, state_pbs)
        result, err = model.predictionFromFeatures_error_(prov, None)
        dt = time.time() - t0
        decode_times.append(dt)
        if err: break

        la = result.featureValueForFeature_("logits").multiArrayValue()
        logits = np.frombuffer(ctypes.c_char_p(la.dataPointer()), dtype=np.float16, count=la.count())
        token = int(np.argmax(logits[-VOCAB:]))
        generated.append(token)
        all_tokens.append(token)
        if token == EOS_ID: break
        if (step+1) % 10 == 0: print(f"  {step+1}/{max_tokens-1}", flush=True)

    return generated, ttft, decode_times


if __name__ == "__main__":
    import CoreML, Foundation, Quartz, tokenizers
    import numpy as np

    print(f"Loading model...", end=" ", flush=True)
    t0 = time.time()
    model, inputs, outputs, st_names = load_model(str(MODEL_PATH))
    print(f"{time.time()-t0:.2f}s  |  {type(model).__name__}")
    print(f"  Inputs: {inputs}  States: {st_names}")

    tok = tokenizers.Tokenizer.from_file(str(TOKENIZER_PATH))
    n_prompt = len(tok.encode(PROMPT).ids)
    print(f"Prompt ({n_prompt}t): {PROMPT}")

    print(f"Generating {MAX_TOKENS} tokens...", flush=True)
    gen, ttft, decode_t = run(model, tok, PROMPT, MAX_TOKENS)

    n = len(gen)
    total = ttft + sum(decode_t)
    tok_s = n / total if total else 0
    text = tok.decode(gen)
    med_t = np.median(decode_t) * 1000 if decode_t else 0
    avg_t = np.mean(decode_t) * 1000 if decode_t else 0

    print(f"\n{'='*60}")
    print(f"  CoreML (all devices)  |  Qwen2.5 0.5B 4-bit")
    print(f"  {n:>4d} tok  |  {total:.3f}s  |  {tok_s:.1f} tok/s")
    print(f"  TTFT: {ttft:.3f}s  |  TPOT: {med_t:.1f}ms  |  Avg: {avg_t:.1f}ms")
    print(f"{'='*60}")
    print(f"  {text[:200]}")
    print(f"{'='*60}")
