# Unified Translation Pipeline Analysis

**Date:** 2026-01-30
**Scope:** Leveraging funasr-nano's integrated LLM for correction + translation
**Status:** TESTED - Key Limitations Identified

---

## CRITICAL FINDING

**The Qwen3-0.6B in funasr-nano was fine-tuned EXCLUSIVELY for ASR.** It cannot perform:
- Text translation
- Modified prompt tasks
- Any non-ASR generation

**Test Results:**
```
Input: 
Prompt: {input}
Output: "!!!!!!!!!!"  <-- Model outputs garbage for non-ASR tasks

Audio-to-English prompt: Still outputs Chinese transcription
Custom prompts: Ignored, model always does ASR
```

**Conclusion:** Cannot leverage funasr-nano's LLM for translation. Must use separate translation LLM.

---

## Current Pain Points

1. **ASR Segmentation Errors**: Wrong word boundaries from variable speaker pacing, especially in Chinese/English mixed speech
2. **LLM Correction Latency**: Separate LLM call adds ~200-500ms for error correction
3. **Pipeline Overhead**: ASR -> LLM (correction) -> LLM (translation) -> TTS = multiple serialized calls

---

## funasr-nano Architecture

```
Audio -> SenseVoice Encoder (70 layers) -> Audio Adaptor (2 layers) -> Qwen3-0.6B (28 layers)
                                                                              |
                                                                              v
                                                                       Transcription
```

**Current Prompt** (hardcoded in `src/model.rs:547-575`):
```
<|im_start|>system
You are a helpful assistant.<|im_end|>
<|im_start|>user
{AUDIO}<|im_end|>
<|im_start|>assistant
```

---

## Approach 1: Modified Prompt for Unified Task

**Concept:** Change the prompt to request correction + translation in one pass.

**Proposed Prompt:**
```
<|im_start|>system
You are a speech translation assistant. Transcribe audio accurately, correct any recognition errors, and translate to English.<|im_end|>
<|im_start|>user
请转写并翻译以下语音为英文：<|startofspeech|>{AUDIO}<|endofspeech|><|im_end|>
<|im_start|>assistant
```

**Implementation Changes:**
```rust
// src/model.rs - new method
pub fn translate_to_english(&mut self, audio_path: impl AsRef<Path>) -> Result<String> {
    // ... audio processing same as transcribe()
    self.generate_text_with_prompt(
        &audio_features,
        "请转写并翻译以下语音为英文：",  // Custom prompt
        &SamplingConfig::default(),
    )
}
```

**Pros:**
- Single inference pass
- No additional model loading
- Leverages audio context for better accuracy

**Cons:**
- Qwen3-0.6B (620M params) is small for translation
- Model was fine-tuned for ASR, not translation
- Unknown if training data included translation pairs
- May hallucinate translations

**Verdict:** Requires testing. May work for simple sentences but likely poor for complex translation.

---

## Approach 2: Two-Pass with Shared Model

**Concept:** Use same Qwen3-0.6B twice - once for ASR, once for text correction/translation.

**Pass 1 (ASR):**
```
{AUDIO}
-> "今天天气很好" (with potential errors)
```

**Pass 2 (Text-only correction + translation):**
```
<|im_start|>user
请纠正并翻译以下中文为英文：今天天气很好<|im_end|>
<|im_start|>assistant
The weather is nice today.
```

**Implementation:**
```rust
impl FunASRNano {
    pub fn transcribe_and_translate(&mut self, audio_path: impl AsRef<Path>) -> Result<(String, String)> {
        // Pass 1: ASR
        let transcription = self.transcribe(audio_path)?;

        // Pass 2: Correction + Translation (text only)
        let translation = self.translate_text(&transcription)?;

        Ok((transcription, translation))
    }

    pub fn translate_text(&mut self, text: &str) -> Result<String> {
        // Use LLM directly without audio
        let prompt = format!("请纠正并翻译以下中文为英文：{}", text);
        self.llm.generate(&prompt, &mut vec![], &SamplingConfig::default())
    }
}
```

**Pros:**
- Clear separation of concerns
- Can debug each step independently
- Same model weights, no extra memory

**Cons:**
- Still two inference passes (but faster than loading separate model)
- Small LLM translation quality questionable

**Estimated Latency:**
- ASR pass: ~300-500ms (current)
- Translation pass: ~200-300ms (text-only, shorter sequence)
- Total: ~500-800ms (vs current ~800-1200ms with separate LLM)

---

## Approach 3: Speculative-like Audio Verification

**Concept:** Use ASR output as "draft", have translation LLM verify against audio context.

**Architecture:**
```
Audio -> Encoder -> Adaptor -> Audio Embeddings
                                    |
                                    +---> ASR Decoder (fast draft) -> "今天天气很好"
                                    |
                                    +---> Translation LLM (verify + translate)
                                          Input: Audio Embeddings + ASR Draft
                                          Output: Verified Translation
```

**Key Insight:** The audio embeddings contain rich contextual information that can help the translation model:
1. Detect when ASR made segmentation errors (audio context doesn't match text)
2. Understand prosody for better translation (question marks, emphasis)
3. Handle code-switching (Chinese/English mix) by attending to audio

**Implementation Concept:**
```rust
pub fn translate_with_audio_context(
    &mut self,
    audio_features: &Array,
    asr_draft: &str,
    target_lang: &str,
) -> Result<String> {
    // Build prompt with both audio and ASR draft
    let prompt = format!(
        "语音内容已转写为：{}\n请验证并翻译为{}：",
        asr_draft, target_lang
    );

    // Generate with audio context
    self.generate_text_with_audio_and_prompt(audio_features, &prompt)
}
```

**Pros:**
- Audio context helps verify/correct ASR errors
- Single encoder pass, shared audio embeddings
- Elegant speculative-like verification

**Cons:**
- Requires architectural changes
- Need to modify prompt injection
- More complex implementation

**Verdict:** Most promising for accuracy, but requires significant development.

---

## Approach 4: Concurrent Shared Encoder

**Concept:** Share encoder output, run ASR and translation decoders in parallel.

**Architecture:**
```
                           +---> Qwen3-0.6B (ASR) --> Chinese
Audio -> Encoder -> Adaptor
                           +---> Separate LLM (Translation) --> English
```

**Benefits:**
- Encoder runs once (~50% of total time)
- Decoders run in parallel
- Best latency if translation LLM can use same audio embeddings

**Challenge:** Translation LLM needs to accept audio embeddings from funasr-nano's adaptor.

**Options:**
1. Fine-tune translation LLM with audio adaptor (expensive)
2. Use audio embeddings as prefix for translation LLM (may work with frozen LLM)
3. Train small projection layer to map embeddings (low-cost fine-tuning)

---

## Approach 5: Streaming Pipeline Optimization

**Concept:** Overlap ASR and translation for lower perceived latency.

**Timeline:**
```
Time: 0ms    200ms   400ms   600ms   800ms   1000ms
      |-------|-------|-------|-------|-------|
ASR:  [===ENCODE===][=DECODE=]
                       |
Translation:           [==LLM==]
                            |
TTS:                        [==SYNTH==]
```

**With Overlap:**
```
ASR:  [===ENCODE===][=DECODE=]
Translation:    [==LLM==] (starts when first tokens available)
TTS:               [==SYNTH==] (starts with first translation tokens)
```

**Implementation:**
```rust
pub fn translate_streaming(
    &mut self,
    audio_path: impl AsRef<Path>,
    callback: impl FnMut(&str),
) -> Result<String> {
    // Start ASR
    let asr_stream = self.transcribe_streaming(audio_path)?;

    // Start translation as ASR tokens arrive
    for partial_text in asr_stream {
        if partial_text.ends_with(['。', '，', '？', '！']) {
            // Sentence boundary - translate this chunk
            let translation = self.translate_text(&partial_text)?;
            callback(&translation);
        }
    }
}
```

---

## Recommendations

### Short-term (Low Effort)

1. **Test Approach 1** - Try modified prompts to see if Qwen3-0.6B can do basic translation
2. **Implement Approach 2** - Two-pass with shared model is straightforward

### Medium-term (Medium Effort)

3. **Implement Approach 5** - Streaming pipeline for lower perceived latency
4. **Benchmark Qwen3-0.6B translation quality** - Determine if it's viable

### Long-term (High Effort)

5. **Implement Approach 3** - Audio-conditioned verification for best accuracy
6. **Consider fine-tuning** - If translation quality insufficient, fine-tune on translation pairs

---

## Next Steps

1. Create test script to evaluate Qwen3-0.6B translation capability
2. Implement `translate_text()` method for text-only LLM inference
3. Add configurable prompt support to `FunASRNano`
