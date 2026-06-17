# Translation Capability Development Plan

**Date:** 2026-01-30
**Project:** funasr-nano-mlx Translation Enhancement
**Status:** Phase 1 Complete -> Phase 2B Active

**Update (2026-01-30):** Phase 1 validation failed. Direct Instruct weight swap breaks ASR
due to embedding space mismatch. Proceeding to Phase 2B (LoRA fine-tuning).

---

## Objective

Enable funasr-nano to perform Chinese speech -> English text translation, reducing pipeline latency by consolidating ASR and translation into fewer model calls.

---

## Current State

```
Current Pipeline (High Latency):
Audio -> funasr-nano (ASR) -> Chinese Text -> External LLM -> English Text -> TTS
        ~300ms                              ~300-500ms

Total: ~600-800ms for transcription + translation
```

**Problem:** funasr-nano's Qwen3-0.6B is the BASE model and cannot follow translation prompts.

---

## Target State

```
Option A - Unified Model:
Audio -> funasr-nano (ASR + Translation) -> English Text -> TTS
         ~400-500ms

Option B - Optimized Pipeline:
Audio -> funasr-nano (ASR) -> Chinese -> Small Translation LLM -> English -> TTS
         ~300ms                        ~150-200ms
```

---

## Development Phases

### Phase 1: Quick Validation (1-2 days) - COMPLETED

**Goal:** Test if swapping to Qwen3-0.6B Instruct enables translation.

**Status:** FAILED - Instruct swap breaks ASR

**Test Results (2026-01-30):**

| Model | ASR Output | Time |
|-------|------------|------|
| Original (Base) | 开放时间：早上九点至下午五点。 | 0.32s |
| Instruct-swapped | <think>...</think> 好的，我现在需要... (hallucination) | 3.40s |

**Root Cause Analysis:**
1. Audio adaptor trained with Base model's embedding space
2. Instruct model has shifted embeddings due to instruction-tuning
3. Audio features misinterpreted as text prompts
4. Model enters "thinking mode" instead of transcribing

**Conclusion:** Direct weight swap is NOT viable. Proceed to Phase 2B.

#### Task 1.1: Download and Test Instruct Model

```bash
# Download official Qwen3-0.6B Instruct
huggingface-cli download Qwen/Qwen3-0.6B \
  --local-dir ~/.dora/models/qwen3-0.6b-instruct

# Or MLX-optimized version
huggingface-cli download Qwen/Qwen3-0.6B-MLX-bf16 \
  --local-dir ~/.dora/models/qwen3-0.6b-mlx
```

#### Task 1.2: Create Weight Swap Script

```rust
// examples/test_instruct_swap.rs
// Load funasr-nano but replace LLM weights with Instruct version
```

**Deliverables:**
- [ ] Script to swap LLM weights
- [ ] Test results: ASR quality with Instruct weights
- [ ] Test results: Translation capability with Instruct weights

**Expected Outcome:**
- If ASR quality maintained + translation works -> Proceed to Phase 2A
- If ASR degrades -> Proceed to Phase 2B (LoRA fine-tuning)

---

### Phase 2A: Instruct Model Integration (3-5 days) - SKIPPED

**Prerequisite:** Phase 1 shows Instruct swap works without ASR degradation.

**Status:** SKIPPED - Phase 1 failed, Instruct swap breaks ASR.

#### Task 2A.1: Merge Instruct Weights into Model

Modify `model.safetensors` to include Instruct LLM weights:

```python
# scripts/merge_instruct_weights.py
import safetensors.torch as st

# Load funasr-nano weights (encoder, adaptor)
funasr = st.load_file("funasr-nano/model.safetensors")

# Load Qwen3-0.6B Instruct weights
instruct = st.load_file("qwen3-0.6b-instruct/model.safetensors")

# Replace LLM weights
for key in instruct:
    funasr[f"llm.{key}"] = instruct[key]

# Save merged model
st.save_file(funasr, "funasr-nano-instruct/model.safetensors")
```

#### Task 2A.2: Update Rust Code for Multi-Task Prompts

```rust
// src/model.rs - Add translation mode
pub enum TaskMode {
    Transcribe,           // 
    TranscribeEnglish,    // Transcribe to English
    TranslateToEnglish,   // Transcribe Chinese, translate to English
}

impl FunASRNano {
    pub fn process_audio(&mut self, audio: &[f32], mode: TaskMode) -> Result<String> {
        let prompt = match mode {
            TaskMode::Transcribe => "",
            TaskMode::TranscribeEnglish => "Transcribe the speech to English:",
            TaskMode::TranslateToEnglish => "",
        };
        self.transcribe_with_prompt(audio, prompt)
    }
}
```

#### Task 2A.3: Benchmark and Validate

| Test | Metric | Target |
|------|--------|--------|
| Chinese ASR | WER | < 5% degradation from baseline |
| Translation | BLEU | > 30 on test set |
| Latency | ms | < 500ms for 5s audio |

**Deliverables:**
- [ ] Merged model file
- [ ] Updated Rust API with TaskMode
- [ ] Benchmark results
- [ ] Example: `translate.rs` updated

---

### Phase 2B: LoRA Fine-tuning (5-7 days) - ACTIVE PATH

**Prerequisite:** Phase 1 shows Instruct swap degrades ASR.

**Status:** READY TO START - This is the recommended path forward.

#### Task 2B.1: Prepare Training Environment

```bash
# Install training dependencies
pip install unsloth transformers datasets peft

# Or use ms-swift (Alibaba's framework)
pip install ms-swift
```

#### Task 2B.2: Prepare Dataset

**Dataset Structure:**
```json
[
  {
    "instruction": "",
    "input": "<audio_embedding>",
    "output": "The weather is nice today."
  },
  {
    "instruction": "",
    "input": "<audio_embedding>",
    "output": ""
  }
]
```

**Data Sources:**
- Existing ASR training data (Chinese)
- WMT Chinese-English parallel corpus
- OPUS translation dataset
- Custom recorded samples

**Recommended Mix:**
- 60% ASR (Chinese transcription)
- 30% Translation (Chinese -> English)
- 10% Instruction following (varied prompts)

#### Task 2B.3: LoRA Training Script

```python
# scripts/train_lora.py
from unsloth import FastLanguageModel
from datasets import load_dataset

# Load base model
model, tokenizer = FastLanguageModel.from_pretrained(
    "Qwen/Qwen3-0.6B-Base",
    max_seq_length=2048,
    load_in_4bit=True,
)

# Add LoRA adapters
model = FastLanguageModel.get_peft_model(
    model,
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
)

# Training config
training_args = {
    "learning_rate": 5e-5,
    "num_train_epochs": 2,
    "per_device_train_batch_size": 4,
    "gradient_accumulation_steps": 4,
    "warmup_steps": 100,
    "logging_steps": 10,
    "save_steps": 500,
}

# Train
trainer.train()

# Save LoRA weights
model.save_pretrained("qwen3-0.6b-translation-lora")
```

#### Task 2B.4: Integrate LoRA into Rust

```rust
// src/lora.rs - LoRA weight loading
pub struct LoRAWeights {
    pub r: i32,
    pub alpha: f32,
    pub weights: HashMap<String, (Array, Array)>,  // (A, B) matrices
}

impl QwenModel {
    pub fn load_lora(&mut self, lora_path: &Path) -> Result<()> {
        // Load LoRA weights and merge into model
        // W' = W + (alpha/r) * B @ A
    }
}
```

#### Task 2B.5: Benchmark LoRA Model

| Test | Metric | Target |
|------|--------|--------|
| Chinese ASR | WER | No degradation from baseline |
| English ASR | WER | < 10% (bonus capability) |
| Translation | BLEU | > 35 on test set |
| Latency | ms | < 10% increase from baseline |

**Deliverables:**
- [ ] Training script
- [ ] Trained LoRA weights
- [ ] LoRA loading in Rust
- [ ] Benchmark results

---

### Phase 3: Pipeline Optimization (3-5 days)

**Goal:** Optimize latency regardless of which approach (2A or 2B) succeeded.

#### Task 3.1: Streaming Translation

Start translation before ASR completes:

```rust
pub fn translate_streaming(
    &mut self,
    audio: impl Iterator<Item = Vec<f32>>,
    on_partial: impl FnMut(&str, &str),  // (chinese, english)
) -> Result<(String, String)> {
    let mut chinese_buffer = String::new();
    ...
}
```
