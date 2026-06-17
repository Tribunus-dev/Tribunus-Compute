# step-audio2-mlx Fix Dev Plan

Based on critical code review. Ordered by impact and dependency.

---

## Phase 1: Generation Quality (User-Visible)

### 1.1 Add repetition penalty to generation loop
**Files:** `src/model.rs` (generate_text, generate_audio_tokens, generate_with_audio)
**Problem:** Model gets stuck repeating same sentence indefinitely.
**Fix:**
- Add frequency penalty: track token counts, penalize logits of recently generated tokens
- Add n-gram blocking: prevent repeating any 3-gram
- Add to all three generation methods (generate_text, generate_audio_tokens, generate_with_audio)
- Default: frequency_penalty=1.2, no_repeat_ngram_size=3

### 1.2 Fix max_tokens inconsistency
**Files:** `src/model.rs:510, 527`
**Problem:** `transcribe()` and `transcribe_samples()` use max_tokens=512, too low for dense speech.
**Fix:** Change both to 2048 to match `transcribe_long()`.

### 1.3 Make `transcribe()` use chunked processing by default
**Files:** `src/model.rs`
**Problem:** `transcribe()` silently truncates audio >15s.
**Fix:** Have `transcribe()` call `transcribe_long()` internally, so all public transcription methods handle long audio.
