# Adaptor Training Plan: FunASR-Nano -> Qwen3-4B

## Objective

Train a new audio adaptor to project SenseVoice encoder output (512-dim) into Qwen3-4B embedding space (2560-dim), enabling ASR + translation in a single model.

## Architecture Overview

```
+-----------------------------------------------------------------+
|                    SenseVoice Encoder                            |
|                    (FROZEN - 500M params)                        |
|                    Output: [batch, time, 512]                    |
+-------------------------------+---------------------------------+
                                |
                                v
+-----------------------------------------------------------------+
|                    Audio Adaptor (NEW)                           |
|                    (TRAINABLE - ~20M params)                     |
|                                                                  |
|    linear1: 512 -> 2048  (keep from original)                    |
|    linear2: 2048 -> 2560 (new output dim)                        |
|    transformer_blocks: 2-4 layers @ 2560-dim                    |
|                                                                  |
|                    Output: [batch, time, 2560]                   |
+-------------------------------+---------------------------------+
                                |
                                v
+-----------------------------------------------------------------+
|                    Qwen3-4B LLM                                  |
|                    (FROZEN or LoRA - 4B params)                  |
|                    hidden_size: 2560                             |
|                    layers: 36, heads: 32, kv_heads: 8            |
+-----------------------------------------------------------------+
```

## Training Phases

### Phase 1: Audio-Text Alignment

**Goal:** Adaptor learns to produce embeddings that match text embeddings in LLM space.

**Method:** Contrastive learning between audio and text representations.

**Data Required:**
- Audio-text pairs (transcriptions)
- ~1,000-10,000 hours

**Loss Function:**
```python
# InfoNCE contrastive loss
def alignment_loss(audio_embeds, text_embeds, temperature=0.07):
    # audio_embeds: [batch, seq, 2560] -> pool to [batch, 2560]
    # text_embeds: [batch, seq, 2560] -> pool to [batch, 2560]

    audio_pooled = audio_embeds.mean(dim=1)
    text_pooled = text_embeds.mean(dim=1)

    # Normalize
    audio_pooled = F.normalize(audio_pooled, dim=-1)
    text_pooled = F.normalize(text_pooled, dim=-1)

    # Cosine similarity matrix
    logits = audio_pooled @ text_pooled.T / temperature

    # Labels: diagonal is positive
    labels = torch.arange(len(logits), device=logits.device)

    loss = F.cross_entropy(logits, labels)
    return loss
```

**Expected Outcome:**
- Audio embeddings cluster near corresponding text embeddings
- LLM can "understand" audio as pseudo-text

---

### Phase 2: End-to-End ASR Fine-tuning

**Goal:** Full pipeline generates correct transcriptions from audio.

**Method:** Standard autoregressive language modeling loss.

**Data Required:**
- Same audio-text pairs as Phase 1
- Can use same dataset

**Loss Function:**
```python
def generation_loss(model, audio, text_tokens):
    # Encode audio
    audio_features = encoder(audio)           # [B, T, 512]
    adapted = adaptor(audio_features)         # [B, T, 2560]

    # Build input: [audio_embeds, text_embeds[:-1]]
    text_embeds = llm.embed(text_tokens[:, :-1])
    input_embeds = torch.cat([adapted, text_embeds], dim=1)

    # Forward through LLM
    logits = llm(inputs_embeds=input_embeds)

    # Only compute loss on text portion
    text_logits = logits[:, adapted.size(1):, :]

    loss = F.cross_entropy(
        text_logits.reshape(-1, vocab_size),
        text_tokens[:, 1:].reshape(-1)
    )
    return loss
```

**Expected Outcome:**
- Model accurately transcribes audio to text
- CER < 5% on Chinese ASR benchmarks

---

### Phase 3: Translation Fine-tuning (Optional)

**Goal:** Direct audio-to-English translation.

**Method:** Same as Phase 2, but with translation targets.

**Data Required:**
- Audio + English translation pairs
- CoVoST-2: ~500 hours Chinese->English

**Prompt Format:**
```
<|im_start|>system
You are a speech translation assistant.<|im_end|>
<|im_start|>user
Translate the following speech to English:<|startofspeech|>{AUDIO}<|endofspeech|><|im_end|>
<|im_start|>assistant
{ENGLISH_TRANSLATION}<|im_end|>
```

**Expected Outcome:**
- Direct speech-to-English translation
- BLEU > 20 on CoVoST-2

---

## Dataset Preparation

### Dataset Options

| Dataset | Size | Purpose | Download | License |
|---------|------|---------|----------|---------|
| **Emilia** | **50,000h Chinese** | ASR (recommended) | [HuggingFace](https://huggingface.co/datasets/amphion/Emilia-Dataset) | CC BY-NC 4.0 |
| AISHELL-1 | 170h | Chinese ASR | [OpenSLR](http://www.openslr.org/33/) | Apache 2.0 |
| AISHELL-2 | 1000h | Chinese ASR | [Registration](https://www.aishelltech.com/aishell_2) | Research |
| CoVoST-2 | 500h | Translation | `datasets` library | CC0 |

### Recommended: Emilia Dataset

Emilia provides 50,000 hours of Chinese speech - 50x more than AISHELL. Use streaming to avoid downloading all 4.5TB:

```python
from datasets import load_dataset

# Stream Chinese subset only
dataset = load_dataset(
    "amphion/Emilia-Dataset",
    data_files={"train": "Emilia/zh/**/*.tar"},
    split="train",
    streaming=True
)

# Sample 100K utterances (~1000 hours)
train_data = list(dataset.take(100000))
```

### Alternative: AISHELL (Smaller, Faster Start)

| Dataset | Size | Purpose | Download |
|---------|------|---------|----------|
| AISHELL-1 | 170h | Chinese ASR | [OpenSLR](http://www.openslr.org/33/) |
| AISHELL-2 | 1000h | Chinese ASR | [Registration](https://www.aishelltech.com/aishell_2) |
| CoVoST-2 | 500h | Translation | `datasets` library |

### Data Format

```json
{
  "audio_path": "/data/aishell/S0001/001.wav",
  "transcript": "开放时间是早上九点至下午五点",
  "translation": "Opening hours are from 9am to 5pm",
  "duration": 3.2,
  "language": "zh"
}
```

### Preprocessing Script

```python
# training/prepare_data.py

import json
from pathlib import Path
from datasets import load_dataset

def prepare_aishell(data_dir, output_file):
    """Convert AISHELL to training format."""
    samples = []

    transcript_file = data_dir / "transcript" / "aishell_transcript_v0.8.txt"
    with open(transcript_file) as f:
        for line in f:
            parts = line.strip().split()
            utt_id = parts[0]
            text = "".join(parts[1:])

            # Find audio file
            audio_path = find_audio(data_dir, utt_id)

            samples.append({
                "audio_path": str(audio_path),
                "transcript": text,
                "language": "zh"
            })

    with open(output_file, "w") as f:
        for s in samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")

def prepare_covost2(output_file):
    """Download and prepare CoVoST-2."""
    ds = load_dataset("facebook/covost2", "zh-CN_en", split="train")

    samples = []
    for item in ds:
        samples.append({
            "audio_path": item["path"],
            "transcript": item["sentence"],
            "translation": item["translation"],
            "language": "zh"
        })

    with open(output_file, "w") as f:
        for s in samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")
```

---

## Training Scripts

### Main Training Script

```python
# training/train_adaptor.py

import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm

class AudioAdaptorQwen4B(nn.Module):
    """Adaptor for Qwen3-4B (2560 hidden_size)."""

    def __init__(self, encoder_dim=512, ffn_dim=2048, llm_dim=2560, n_layers=2):
        super().__init__()

        self.linear1 = nn.Linear(encoder_dim, ffn_dim)
        self.linear2 = nn.Linear(ffn_dim, llm_dim)
        self.activation = nn.ReLU()

        # Transformer blocks at LLM dimension
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=llm_dim,
            nhead=8,
            dim_feedforward=llm_dim * 4,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

    def forward(self, x):
        # x: [batch, time, 512]
        x = self.linear1(x)
        x = self.activation(x)
        x = self.linear2(x)
        x = self.transformer(x)
        return x  # [batch, time, 2560]


def train_phase1(adaptor, encoder, llm, dataloader, optimizer, epochs=10):
    """Phase 1: Contrastive alignment training."""

    adaptor.train()
    encoder.eval()
    llm.eval()

    for epoch in range(epochs):
        total_loss = 0

        for batch in tqdm(dataloader, desc=f"Epoch {epoch+1}"):
            ...
