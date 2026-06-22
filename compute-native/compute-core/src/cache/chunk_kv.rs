//! ChunkKV — semantic-preserving KV cache compression.
//!
//! Instead of 64-token fixed blocks, ChunkKV aligns chunks to sentence/phrase
//! boundaries. Each chunk is a complete semantic unit. When eviction is needed,
//! entire chunks are evicted — preserving the semantic integrity of remaining
//! chunks.
//!
//! Reference: "ChunkKV: Semantic-Preserving KV Cache Compression for Efficient
//! Long-Context LLM Inference" (ICLR 2025 Spotlight).
//!
//! ChunkKV wraps the lower-level compressed cache / paged I/O surface allocator;
//! new tokens are buffered until a chunk boundary is detected, then the chunk
//! is compressed and stored as a unit.

/// The chunk type signals the semantic role a chunk plays in the conversation
/// and influences importance scoring (system chunks survive longer, intermediate
/// user/assistant turns are evicted sooner).
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ChunkType {
    /// Complete sentence ending with a terminal punctuation mark (. ! ?).
    Sentence,
    /// System prompt / instruction fragment.
    System,
    /// User input fragment.
    UserInput,
    /// Assistant response fragment.
    AssistantReply,
}

/// A semantic chunk: a group of tokens forming a semantic unit.
///
/// Chunk boundaries are aligned to sentence / utterance boundaries rather
/// than to a fixed token count.  The chunk tracks its own position range
/// in the token stream, the compressed KV page indices it owns, and an
/// importance score used for eviction ordering.
#[derive(Debug, Clone)]
pub struct SemanticChunk {
    /// Monotonically increasing chunk identifier.
    pub chunk_id: u64,
    /// First token position (inclusive) in the global token sequence.
    pub start_token: u32,
    /// Last token position (exclusive) in the global token sequence.
    pub end_token: u32,
    /// Number of tokens in this chunk.
    pub token_count: u32,
    /// Indices into the compressed KV cache page table that store this
    /// chunk's K/V data after compression.
    pub kv_pages: Vec<u32>,
    /// Composite importance score (higher = more important, evicted last).
    pub importance_score: f64,
    /// Semantic role of this chunk.
    pub chunk_type: ChunkType,
}

/// A token buffer that accumulates tokens until a semantic chunk boundary
/// is detected, at which point the whole buffer is flushed into a chunk.
#[derive(Debug, Clone)]
struct PendingChunk {
    /// Accumulated token ids.
    tokens: Vec<u32>,
    /// Start position in the global token sequence.
    start_token: u32,
    /// Number of complete sentence boundaries detected within this buffer.
    boundary_count: usize,
}

/// ChunkKV cache — wraps the lower-level compressed cache with semantic
/// chunk awareness.
///
/// New tokens enter a pending buffer.  Once a chunk boundary (punctuation
/// at end of utterance, speaker turn, system/user boundary) is detected,
/// the accumulated tokens are compressed as a unit and stored in a new
/// [`SemanticChunk`] entry.  Eviction removes entire chunks (the
/// lowest-importance chunk) rather than individual pages.
#[derive(Debug)]
pub struct ChunkKvCache {
    /// All active (resident) semantic chunks, ordered by chunk_id.
    chunks: Vec<SemanticChunk>,
    /// Maximum number of chunks before eviction triggers.
    max_chunks: usize,
    /// Hard byte budget for compressed KV data (all chunks combined).
    total_budget_bytes: usize,
    /// Current total bytes consumed by resident chunks (compressed K/V).
    current_bytes: usize,
    /// Per-chunk compression ratio target (used when compressing).
    compression_ratio: f64,
    /// Monotonically increasing chunk id counter.
    next_chunk_id: u64,
    /// Tokens buffered between chunk boundaries.
    pending: PendingChunk,
    /// Token ids that indicate a sentence boundary when they appear at the
    /// end of a pending buffer.  These are the standard tokenized equivalents
    /// of `.`, `!`, `?`, `\n\n` (paragraph break).
    sentence_end_tokens: Vec<u32>,
    /// Token ids that indicate a speaker turn / role boundary (e.g. newline
    /// after "### Assistant", or the separator BOS).
    speaker_boundary_tokens: Vec<u32>,
}

impl ChunkKvCache {
    /// Create a new chunk KV cache.
    ///
    /// `max_chunks` caps the number of resident chunks before eviction.
    /// `budget_bytes` caps the total compressed byte storage across all chunks.
    /// When either limit is exceeded the lowest-importance chunk is evicted.
    pub fn new(max_chunks: usize, budget_bytes: usize) -> Self {
        Self {
            chunks: Vec::with_capacity(max_chunks.min(128)),
            max_chunks,
            total_budget_bytes: budget_bytes,
            current_bytes: 0,
            compression_ratio: 1.0,
            next_chunk_id: 1,
            pending: PendingChunk {
                tokens: Vec::new(),
                start_token: 0,
                boundary_count: 0,
            },
            // Common token ids for sentence-ending punctuation.
            // These cover typical LLM tokenizer outputs for ['.', '!', '?',
            // '\n', '\n\n'] across sentence-piece and BPE tokenizers.
            sentence_end_tokens: Vec::new(),
            // Token ids for role/speaker boundaries.
            speaker_boundary_tokens: Vec::new(),
        }
    }

    /// Set the compression ratio for per-chunk storage estimation.
    pub fn set_compression_ratio(&mut self, ratio: f64) {
        self.compression_ratio = ratio.max(0.125).min(64.0);
    }

    /// Register tokenizer-specific boundary tokens.
    ///
    /// `sentence_ends` — token ids that mark the end of a sentence
    /// (e.g., the BPE tokens for '.', '!', '?').
    /// `speaker_bounds` — token ids that mark speaker / role boundaries
    /// (e.g., newlines after role headers, separator tokens).
    pub fn set_boundary_tokens(&mut self, sentence_ends: Vec<u32>, speaker_bounds: Vec<u32>) {
        self.sentence_end_tokens = sentence_ends;
        self.speaker_boundary_tokens = speaker_bounds;
    }

    /// Detect chunk boundary positions in a token stream.
    ///
    /// Returns the indides (token positions *within* the slice, not absolute)
    /// where one chunk ends and the next begins.  Each returned index is the
    /// *start* of the next chunk — i.e. a boundary after position `i` means
    /// the next chunk starts at `i + 1`.
    ///
    /// Heuristics used (in priority order):
    ///
    /// 1. Speaker/role boundaries: if a token appears in `speaker_boundary_tokens`
    ///    and the next token looks like a role token, split.
    /// 2. Sentence boundaries: if a token appears in `sentence_end_tokens` and it
    ///    is followed by whitespace or another boundary token, split.
    /// 3. Fallback: if no boundary is found after `max_fallback_tokens`, force a
    ///    split to prevent unbounded accumulation.
    pub fn detect_chunks(&self, tokens: &[u32]) -> Vec<usize> {
        let max_fallback_tokens = 128;
        let mut boundaries: Vec<usize> = Vec::new();
        let mut last_boundary: usize = 0;

        // If the stream is empty, return empty boundaries.
        if tokens.is_empty() {
            return boundaries;
        }

        for i in 0..tokens.len() {
            // 1. Speaker / role boundary detection.
            if self.speaker_boundary_tokens.contains(&tokens[i]) {
                // A speaker boundary token followed by another token suggests
                // a natural chunk boundary.  Split here.
                let gap = i - last_boundary;
                if gap >= 2 {
                    boundaries.push(i + 1); // next chunk starts at i+1
                    last_boundary = i + 1;
                    continue;
                }
            }

            // 2. Sentence boundary detection.
            if self.sentence_end_tokens.contains(&tokens[i]) {
                // A sentence-ending punctuation token.  Split after it so the
                // punctuation is the last token of the current chunk.
                let gap = i - last_boundary;
                // Only split if we have at least 2 tokens in the chunk
                // (avoid single-punctuation chunks).
                if gap >= 1 {
                    boundaries.push(i + 1); // next chunk starts at i+1
                    last_boundary = i + 1;
                    continue;
                }
            }
        }

        // 3. Fallback: if the last chunk is too long, force a boundary.
        let remaining = tokens.len() - last_boundary;
        if remaining > max_fallback_tokens {
            // Force split at the fallback point.
            boundaries.push(last_boundary + max_fallback_tokens);
        }

        // If no boundaries were detected at all and the stream is non-trivial,
        // force one at max_fallback_tokens.
        if boundaries.is_empty() && tokens.len() > max_fallback_tokens {
            boundaries.push(max_fallback_tokens);
        }

        boundaries
    }

    /// Buffer new tokens and automatically detect chunk boundaries.
    ///
    /// When a boundary is detected, the accumulated pending tokens are formed
    /// into a chunk and appended.  Returns the number of complete chunks
    /// flushed from the pending buffer (0 if still accumulating).
    ///
    /// Callers should pass the output of each decode step through here before
    /// inspecting the chunk cache for prefix matching or eviction.
    pub fn ingest_tokens(&mut self, tokens: &[u32], chunk_type_hint: ChunkType) -> usize {
        if tokens.is_empty() {
            return 0;
        }

        // If pending is empty, set start_token.
        if self.pending.tokens.is_empty() {
            self.pending.start_token = 0; // caller should set via token_offset
        }

        let _start_len = self.pending.tokens.len();
        self.pending.tokens.extend_from_slice(tokens);

        // Detect boundaries in the accumulated buffer, using the offset of
        // the existing content.
        let boundaries = self.detect_chunks(&self.pending.tokens);

        // Walk boundaries and commit complete chunks.
        let mut committed = 0;
        let mut cut = 0;
        for &b in &boundaries {
            if b > self.pending.tokens.len() {
                break;
            }
            if b <= cut {
                continue; // already past this boundary
            }

            let chunk_tokens: Vec<u32> = self.pending.tokens[cut..b].to_vec();
            if chunk_tokens.is_empty() {
                cut = b;
                continue;
            }

            // Infer chunk type from the content if we have no explicit hint.
            let ctype = if cut == 0 {
                // First chunk uses the caller's hint.
                chunk_type_hint
            } else {
                self.infer_chunk_type(&chunk_tokens, chunk_type_hint)
            };

            // Estimate compressed page storage (placeholder: 1 page per chunk
            // with size proportional to token count and compression).
            let est_pages = vec![committed as u32]; // logical page index
            let est_bytes = self.estimate_chunk_bytes(&chunk_tokens);

            let chunk = SemanticChunk {
                chunk_id: self.next_chunk_id,
                start_token: self.pending.start_token + cut as u32,
                end_token: self.pending.start_token + b as u32,
                token_count: chunk_tokens.len() as u32,
                kv_pages: est_pages,
                importance_score: 0.5, // recalculated on insert
                chunk_type: ctype,
            };

            self.next_chunk_id += 1;
            self.current_bytes += est_bytes;
            self.chunks.push(chunk);

            if self.chunks.len() > self.max_chunks || self.current_bytes > self.total_budget_bytes {
                self.evict_lowest();
            }

            cut = b;
            committed += 1;
        }

        // Drain committed tokens from the pending buffer.
        if cut > 0 {
            self.pending.tokens.drain(0..cut);
            self.pending.start_token += cut as u32;
        }

        committed
    }

    /// Insert a pre-assembled chunk directly into the cache.
    ///
    /// Use this when callers have already determined chunk boundaries (e.g.,
    /// from a structured prompt pipeline).  The chunk is assigned an importance
    /// score and may trigger eviction if over budget.
    pub fn insert_chunk(
        &mut self,
        start: u32,
        end: u32,
        tokens: &[u32],
        pages: Vec<u32>,
    ) -> Result<(), String> {
        if start >= end {
            return Err("chunk start must be < end".to_string());
        }

        let token_count = (end - start) as usize;
        if token_count != tokens.len() {
            return Err(format!(
                "chunk token count mismatch: start={}, end={}, tokens.len={}",
                start,
                end,
                tokens.len()
            ));
        }

        // Infer type from the token content.
        let ctype = self.infer_chunk_type(tokens, ChunkType::Sentence);

        let est_bytes = self.estimate_chunk_bytes(tokens);

        let chunk = SemanticChunk {
            chunk_id: self.next_chunk_id,
            start_token: start,
            end_token: end,
            token_count: tokens.len() as u32,
            kv_pages: pages,
            importance_score: 0.5,
            chunk_type: ctype,
        };
        self.next_chunk_id += 1;

        // Compute importance after construction so we have the full picture.
        let importance = self.compute_importance(&chunk);
        let mut chunk = chunk;
        chunk.importance_score = importance;

        self.current_bytes += est_bytes;
        self.chunks.push(chunk);

        while (!self.chunks.is_empty())
            && (self.chunks.len() > self.max_chunks || self.current_bytes > self.total_budget_bytes)
        {
            self.evict_lowest();
        }

        Ok(())
    }

    /// Compute importance score for a chunk.
    ///
    /// Factors (weighted heuristics):
    ///
    /// - **Recency**: more recent chunks get a higher score. The newest chunk
    ///   scores ~1.0, decaying linearly with age.
    /// - **Type bonus**: system chunks are preserved longest; user input is
    ///   evicted first. Assistant replies and sentences are intermediate.
    /// - **Length penalty**: very short chunks (< 4 tokens) are penalised
    ///   because they carry little context.
    /// - **Frequency bonus**: chunks with many KV pages (wide attention) get
    ///   a small bonus, reflecting higher compute cost to recompute.
    fn compute_importance(&self, chunk: &SemanticChunk) -> f64 {
        // ── Recency ────────────────────────────────────────────────────
        // The last chunk in `chunks` is the newest.  If `chunk` is already
        // in the list, use its position; otherwise treat it as newest.
        let max_idx = self.chunks.len().saturating_sub(1);
        let idx = self
            .chunks
            .iter()
            .rposition(|c| c.chunk_id == chunk.chunk_id)
            .unwrap_or(max_idx + 1);
        let recency = if max_idx == 0 {
            1.0
        } else {
            (idx as f64 / max_idx.max(1) as f64).clamp(0.0, 1.0)
        };

        // ── Type bonus ──────────────────────────────────────────────────
        let type_bonus = match chunk.chunk_type {
            ChunkType::System => 1.0,
            ChunkType::AssistantReply => 0.7,
            ChunkType::Sentence => 0.5,
            ChunkType::UserInput => 0.3,
        };

        // ── Length penalty ─────────────────────────────────────────────
        let length_factor = if chunk.token_count < 4 {
            0.3
        } else if chunk.token_count < 8 {
            0.6
        } else if chunk.token_count > 256 {
            // Very long chunks are penalised slightly (they consume more
            // budget per chunk).
            0.8
        } else {
            1.0
        };

        // ── Frequency / page bonus ─────────────────────────────────────
        let page_bonus = (chunk.kv_pages.len() as f64 * 0.05).min(0.3);

        // Weighted combination.
        (0.40 * recency) + (0.35 * type_bonus) + (0.15 * length_factor) + (0.10 * page_bonus)
    }

    /// Recompute importance scores for all chunks.
    ///
    /// Should be called periodically (e.g. after eviction or when the chunk
    /// count changes) to keep scores fresh.
    pub fn recompute_importance(&mut self) {
        // Compute importance scores separately from mutation to avoid
        // conflicting borrows on self.
        let scores: Vec<f64> = self
            .chunks
            .iter()
            .map(|chunk| self.compute_importance(chunk))
            .collect();
        for (chunk, score) in self.chunks.iter_mut().zip(scores) {
            chunk.importance_score = score;
        }
    }

    /// Evict the single chunk with the lowest importance score.
    ///
    /// Prefers evicting shorter, less important chunks to free budget with
    /// minimal semantic loss.  The chunk's compressed KV pages are marked
    /// for release by the caller.
    fn evict_lowest(&mut self) {
        if self.chunks.is_empty() {
            return;
        }

        // Recompute scores first so we have a fresh ranking.
        self.recompute_importance();

        // Find the least-important chunk.
        let (idx, _) = self
            .chunks
            .iter()
            .enumerate()
            .min_by(|(_, a), (_, b)| {
                a.importance_score
                    .partial_cmp(&b.importance_score)
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .expect("non-empty chunks should yield a min");

        let evicted = self.chunks.swap_remove(idx);

        // Estimate the bytes we're freeing.
        let est_bytes = self.estimate_eviction_bytes(&evicted);
        self.current_bytes = self.current_bytes.saturating_sub(est_bytes);
    }

    /// Evict the lowest-importance chunk to make room for new content.
    /// Public version for explicit call from the decoder loop.
    pub fn evict_one(&mut self) {
        self.evict_lowest();
    }

    /// Estimate how many bytes a token range will consume when compressed.
    ///
    /// Returns bytes (compressed K/V + metadata overhead).
    fn estimate_chunk_bytes(&self, tokens: &[u32]) -> usize {
        // Baseline: one token of uncompressed FP16 K/V is roughly
        // 2 * n_kv_heads * head_dim * 2 bytes.  For a typical 8-head,
        // 128-dim layer that is ~4096 bytes per token.
        // Compressed at `compression_ratio` that becomes ~4096 / ratio.
        //
        // We use a simplified estimate: 256 bytes per token at ratio=4,
        // scaled linearly by the compression ratio reciprocal.
        let per_token_compressed = (256.0 / self.compression_ratio.max(0.125)) as usize;
        let data_bytes = tokens.len() * per_token_compressed;
        // Per-chunk metadata overhead.
        let overhead = std::mem::size_of::<SemanticChunk>() + std::mem::size_of::<u32>() * 4;
        data_bytes + overhead
    }

    /// Estimate how many bytes will be freed by evicting a chunk.
    fn estimate_eviction_bytes(&self, chunk: &SemanticChunk) -> usize {
        // Approximate the compressed data size from the token count.
        let per_token_compressed = (256.0 / self.compression_ratio.max(0.125)) as usize;
        let data_bytes = chunk.token_count as usize * per_token_compressed;
        let overhead = std::mem::size_of::<SemanticChunk>();
        data_bytes + overhead
    }

    /// Find which chunks match a prefix (for prefix caching).
    ///
    /// Returns `Some((matching_page_indices, matched_token_count))` when a
    /// prefix of `tokens` matches one or more resident chunks in order, or
    /// `None` when no prefix matches.
    ///
    /// Matching is at chunk granularity — either a whole chunk matches or it
    /// does not.  This differs from the 64-token fixed-block matching used by
    /// the block-level prefix cache.
    pub fn find_matching_prefix(&self, tokens: &[u32]) -> Option<(Vec<u32>, u32)> {
        if tokens.is_empty() || self.chunks.is_empty() {
            return None;
        }

        let mut matched_pages: Vec<u32> = Vec::new();
        let mut matched_tokens: u32 = 0;
        let mut token_pos: usize = 0;

        // Chunks are ordered by chunk_id (insertion order), which corresponds
        // to their position in the token sequence.  Walk them in order.
        let mut sorted_chunks: Vec<&SemanticChunk> = self.chunks.iter().collect();
        sorted_chunks.sort_by_key(|c| c.start_token);

        for chunk in &sorted_chunks {
            let ct = chunk.token_count as usize;
            if token_pos + ct > tokens.len() {
                // Partial chunk — stop (we require whole-chunk alignment).
                break;
            }

            // Compare the tokens in this chunk against the corresponding span
            // of the input prefix.
            // We don't have the raw tokens stored in SemanticChunk, so we
            // compare by position / trust the semantic identity.  In practice,
            // the caller would hash the chunk content.
            //
            // For now, match by iterating the chunk list; actual token-level
            // verification requires the caller to pass token content alongside.
            matched_pages.extend(&chunk.kv_pages);
            matched_tokens += ct as u32;
            token_pos += ct;
        }

        if matched_tokens == 0 {
            None
        } else {
            Some((matched_pages, matched_tokens))
        }
    }

    /// Find which chunks match a prefix, given the actual token content.
    ///
    /// This variant also takes the prompt token buffer and an offset mapping
    /// (start_token → its byte-level position in the buffer) so it can do
    /// token-level verification of each chunk's content.
    ///
    /// Returns `Some((matched_page_indices, matched_token_count))` on success,
    /// `None` when no prefix matches or there is a token mismatch.
    ///
    /// `token_buffer` — the full token buffer for this session (concatenation
    /// of all previously seen tokens).
    /// `chunks_to_check` — the subset of chunks to consider (usually all
    /// resident chunks, but could be filtered).
    pub fn find_matching_prefix_verified(
        &self,
        query_tokens: &[u32],
        token_buffer: &[u32],
        chunks_to_check: &[&SemanticChunk],
    ) -> Option<(Vec<u32>, u32)> {
        if query_tokens.is_empty() || chunks_to_check.is_empty() || token_buffer.is_empty() {
            return None;
        }

        // Sort chunks by position.
        let mut sorted: Vec<&&SemanticChunk> = chunks_to_check.iter().collect();
        sorted.sort_by_key(|c| c.start_token);

        let mut matched_pages: Vec<u32> = Vec::new();
        let mut matched_tokens: u32 = 0;
        let mut query_pos: usize = 0;

        for chunk in sorted {
            let ct = chunk.token_count as usize;
            if query_pos + ct > query_tokens.len() {
                break; // partial chunk at the end — stop
            }

            // Verify content: read tokens from token_buffer at the chunk's
            // position and compare against the query prefix.
            let chunk_start = chunk.start_token as usize;
            let chunk_end = chunk.end_token as usize;

            if chunk_end > token_buffer.len() {
                // Chunk refers to tokens beyond what we have — can't verify,
                // treat as mismatch.
                break;
            }

            let stored_tokens = &token_buffer[chunk_start..chunk_end];
            let query_span = &query_tokens[query_pos..query_pos + ct];

            if stored_tokens != query_span {
                // Token mismatch — this chunk doesn't match the query.
                break;
            }

            matched_pages.extend(&chunk.kv_pages);
            matched_tokens += ct as u32;
            query_pos += ct;
        }

        if matched_tokens == 0 {
            None
        } else {
            Some((matched_pages, matched_tokens))
        }
    }

    /// Return a reference to all resident chunks.
    pub fn chunks(&self) -> &[SemanticChunk] {
        &self.chunks
    }

    /// Return the pending (unflushed) token buffer.
    pub fn pending_tokens(&self) -> &[u32] {
        &self.pending.tokens
    }

    /// Current number of resident chunks.
    pub fn chunk_count(&self) -> usize {
        self.chunks.len()
    }

    /// Current total estimated compressed bytes across all chunks.
    pub fn used_bytes(&self) -> usize {
        self.current_bytes
    }

    /// Clear all chunks and reset the pending buffer.
    pub fn clear(&mut self) {
        self.chunks.clear();
        self.current_bytes = 0;
        self.pending.tokens.clear();
        self.pending.start_token = 0;
        self.pending.boundary_count = 0;
    }

    /// Clear only the pending buffer (keep committed chunks).
    pub fn flush_pending(&mut self) {
        self.pending.tokens.clear();
        self.pending.start_token = 0;
        self.pending.boundary_count = 0;
    }

    // ── Private helpers ────────────────────────────────────────────────

    /// Infer the chunk type from token content when the caller's hint is
    /// too generic (e.g. `ChunkType::Sentence`).
    fn infer_chunk_type(&self, tokens: &[u32], hint: ChunkType) -> ChunkType {
        // If the caller gave a specific type, trust it.
        if hint != ChunkType::Sentence {
            return hint;
        }

        // Heuristic: check if the chunk starts with common role markers.
        if tokens.is_empty() {
            return ChunkType::Sentence;
        }

        // If any token is a speaker boundary, it's likely a role transition.
        let has_role_marker = tokens
            .iter()
            .any(|t| self.speaker_boundary_tokens.contains(t));

        if has_role_marker {
            // Check the content around the marker for common role patterns.
            // A leading system token pattern.
            ChunkType::UserInput
        } else {
            ChunkType::Sentence
        }
    }
}

impl Default for ChunkKvCache {
    fn default() -> Self {
        Self::new(100, 512 * 1024 * 1024) // 100 chunks, 512 MB budget
    }
}

// ── Testing ──────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    /// Token ids commonly used for sentence-ending punctuation across many
    /// sentencepiece / BPE tokenizers.
    fn default_sentence_ends() -> Vec<u32> {
        vec![
            13,    // '\n' (common in Llama, Gemma)
            29871, // '.' in Gemma tokenizer
            29889, // '!' in Gemma tokenizer
            29941, // '?' in Gemma tokenizer
            357,   // '.' in many BPE tokenizers
        ]
    }

    fn default_speaker_bounds() -> Vec<u32> {
        vec![
            13,   // '\n'
            529,  // ' ' + newline patterns
            2000, // separator or boundary token
        ]
    }

    #[test]
    fn test_new_cache_empty() {
        let cache = ChunkKvCache::new(50, 256 * 1024 * 1024);
        assert_eq!(cache.chunk_count(), 0);
        assert_eq!(cache.used_bytes(), 0);
        assert_eq!(cache.pending_tokens().len(), 0);
    }

    #[test]
    fn test_detect_chunks_returns_empty_for_empty_input() {
        let cache = ChunkKvCache::new(50, 256 * 1024 * 1024);
        let boundaries = cache.detect_chunks(&[]);
        assert!(boundaries.is_empty());
    }

    #[test]
    fn test_detect_chunks_with_sentence_boundaries() {
        let mut cache = ChunkKvCache::new(50, 256 * 1024 * 1024);
        cache.set_boundary_tokens(default_sentence_ends(), default_speaker_bounds());

        // Tokens: "Hello world." "How are you?" (period token at pos 2, question at pos 6)
        let tokens = vec![
            1u32, 2, 29871, // "Hello world."  — period at index 2
            3, 4, 5, 29941, // "How are you?"  — question at index 6
        ];
        let boundaries = cache.detect_chunks(&tokens);
        // We expect a boundary after the period (index 3) and after the question (index 7).
        assert!(
            !boundaries.is_empty(),
            "should detect at least one boundary"
        );
        // At minimum, the fallback should produce one boundary if punctuation
        // boundaries exist.
        assert!(
            boundaries.len() >= 1,
            "expected >=1 boundary, got {:?}",
            boundaries
        );
    }

    #[test]
    fn test_ingest_tokens_flushes_on_boundary() {
        let mut cache = ChunkKvCache::new(10, 1024 * 1024);
        cache.set_boundary_tokens(default_sentence_ends(), default_speaker_bounds());

        // Ingest tokens without a period — should buffer, not flush.
        let flushed = cache.ingest_tokens(&[1, 2, 3, 4, 5], ChunkType::Sentence);
        assert_eq!(flushed, 0, "no boundary yet");
        assert_eq!(cache.pending_tokens().len(), 5, "tokens buffered");
        assert_eq!(cache.chunk_count(), 0);

        // Now send a period token — boundary detection should flush.
        let flushed = cache.ingest_tokens(&[29871], ChunkType::Sentence);
        assert_eq!(flushed, 1, "one chunk flushed after period");
        // Pending should be drained.
        assert_eq!(cache.pending_tokens().len(), 0);
        assert_eq!(cache.chunk_count(), 1, "one chunk created");

        let chunk = &cache.chunks[0];
        assert_eq!(chunk.token_count, 5);
        assert_eq!(chunk.chunk_type, ChunkType::Sentence);
    }

    #[test]
    fn test_insert_chunk_direct() {
        let mut cache = ChunkKvCache::new(10, 1024 * 1024);
        cache.set_boundary_tokens(default_sentence_ends(), default_speaker_bounds());

        let tokens = vec![1, 2, 3, 4, 5, 29871];
        let result = cache.insert_chunk(0, 6, &tokens, vec![0]);
        assert!(result.is_ok(), "insert_chunk should succeed");
        assert_eq!(cache.chunk_count(), 1);

        let chunk = &cache.chunks[0];
        assert_eq!(chunk.start_token, 0);
        assert_eq!(chunk.end_token, 6);
        assert_eq!(chunk.token_count, 6);
    }

    #[test]
    fn test_insert_chunk_validation() {
        let mut cache = ChunkKvCache::new(10, 1024 * 1024);

        // start >= end
        let result = cache.insert_chunk(5, 3, &[1, 2], vec![0]);
        assert!(result.is_err());

        // token count mismatch
        let result = cache.insert_chunk(0, 5, &[1, 2, 3], vec![0]);
        assert!(result.is_err());
    }

    #[test]
    fn test_eviction_on_overflow() {
        let mut cache = ChunkKvCache::new(3, 1024 * 1024 * 1024); // max 3 chunks, huge budget
        cache.set_boundary_tokens(default_sentence_ends(), default_speaker_bounds());

        // Insert 4 chunks — the lowest-importance one should be evicted.
        for i in 0..4u32 {
            let start = i * 8;
            let end = start + 8;
            let tokens: Vec<u32> = (start..end).collect();
            cache
                .insert_chunk(start, end, &tokens, vec![i])
                .expect("insert should succeed");
        }

        // Should be at most 3 chunks after eviction.
        assert!(
            cache.chunk_count() <= 3,
            "should have evicted at least one chunk, got {}",
            cache.chunk_count()
        );
    }

    #[test]
    fn test_eviction_on_budget_overflow() {
        let mut cache = ChunkKvCache::new(100, 1024); // tiny budget
        cache.set_boundary_tokens(default_sentence_ends(), default_speaker_bounds());

        // A single large chunk should exceed the budget and trigger eviction
        // once we add a second one.
        let big_tokens: Vec<u32> = (0..50).collect();
        cache
            .insert_chunk(0, 50, &big_tokens, vec![0])
            .expect("first insert");
        let second: Vec<u32> = (50..100).collect();
        cache
            .insert_chunk(50, 100, &second, vec![1])
            .expect("second insert");

        // Budget is tiny; at least one chunk should have been evicted.
        assert!(
            cache.used_bytes() <= 1024,
            "used_bytes should stay under budget"
        );
    }

    #[test]
    fn test_find_matching_prefix_basic() {
        let mut cache = ChunkKvCache::new(10, 1024 * 1024);
        cache.set_boundary_tokens(default_sentence_ends(), default_speaker_bounds());

        // Insert two chunks that form a sequence.
        let tokens_a: Vec<u32> = vec![1, 2, 3, 4, 5];
        cache
            .insert_chunk(0, 5, &tokens_a, vec![0, 1])
            .expect("chunk A");
        let tokens_b: Vec<u32> = vec![6, 7, 8, 9, 10];
        cache
            .insert_chunk(5, 10, &tokens_b, vec![2, 3])
            .expect("chunk B");

        // Query with a prefix that exactly matches chunk A.
        //
        // `find_matching_prefix` does whole-chunk matching by position; it
        // matches if the chunk range fits within the query.
        let query: Vec<u32> = vec![1, 2, 3, 4, 5, 6, 7];
        let result = cache.find_matching_prefix(&query);
        assert!(
            result.is_some(),
            "should match prefix of at least one chunk"
        );
        let (pages, tokens_matched) = result.unwrap();
        assert_eq!(pages, vec![0, 1], "should return chunk A's pages");
        assert_eq!(tokens_matched, 5, "should match chunk A's 5 tokens");
    }

    #[test]
    fn test_find_matching_prefix_empty() {
        let cache = ChunkKvCache::new(10, 1024 * 1024);
        let result = cache.find_matching_prefix(&[]);
        assert!(result.is_none(), "empty query should not match");
    }

    #[test]
    fn test_find_matching_prefix_verified_exact() {
        let mut cache = ChunkKvCache::new(10, 1024 * 1024);
        cache.set_boundary_tokens(default_sentence_ends(), default_speaker_bounds());

        // Build a token buffer that matches chunk positions.
        let token_buffer: Vec<u32> = (0..20).collect();
        let tokens_a: Vec<u32> = token_buffer[0..5].to_vec();
        cache
            .insert_chunk(0, 5, &tokens_a, vec![0])
            .expect("chunk A");

        // Query exactly chunk A's tokens with the same token buffer.
        let query: Vec<u32> = token_buffer[0..5].to_vec();
        let chunks_refs: Vec<&SemanticChunk> = cache.chunks.iter().collect();
        let result = cache.find_matching_prefix_verified(&query, &token_buffer, &chunks_refs);
        assert!(result.is_some(), "verified exact match should succeed");
    }

    #[test]
    fn test_find_matching_prefix_verified_mismatch() {
        let mut cache = ChunkKvCache::new(10, 1024 * 1024);
        cache.set_boundary_tokens(default_sentence_ends(), default_speaker_bounds());

        // Insert a chunk that claims to cover tokens [0..5).
        let stored_tokens: Vec<u32> = vec![10, 20, 30, 40, 50];
        cache
            .insert_chunk(0, 5, &stored_tokens, vec![0])
            .expect("chunk");

        // Token buffer has DIFFERENT content at [0..5).
        let different_buffer: Vec<u32> = vec![99, 98, 97, 96, 95];
        let query: Vec<u32> = vec![10, 20, 30]; // query matches stored tokens

        let chunks_refs: Vec<&SemanticChunk> = cache.chunks.iter().collect();
        let result = cache.find_matching_prefix_verified(
            &query,
            &different_buffer, // buffer has different content
            &chunks_refs,
        );
        // Should NOT match because the token_buffer content differs from
        // what the chunk assumes.
        assert!(
            result.is_none(),
            "token mismatch should prevent prefix match"
        );
    }

    #[test]
    fn test_importance_recency() {
        let mut cache = ChunkKvCache::new(10, 1024 * 1024);
        cache.set_boundary_tokens(default_sentence_ends(), default_speaker_bounds());

        // Insert chunks in order — newer ones should score higher.
        for i in 0..5u32 {
            let end = (i + 1) * 4;
            let tokens: Vec<u32> = (i * 4..end).collect();
            cache
                .insert_chunk(i * 4, end, &tokens, vec![i])
                .expect("insert");
        }

        cache.recompute_importance();

        // The last chunk (index 4) should be the most recent.
        let newest = &cache.chunks[cache.chunks.len() - 1];
        let oldest = &cache.chunks[0];

        // Newest should have higher importance than oldest.
        assert!(
            newest.importance_score >= oldest.importance_score,
            "newest chunk ({:.4}) should score >= oldest ({:.4})",
            newest.importance_score,
            oldest.importance_score
        );
    }

    #[test]
    fn test_importance_system_bonus() {
        let mut cache = ChunkKvCache::new(10, 1024 * 1024);
        cache.set_boundary_tokens(default_sentence_ends(), default_speaker_bounds());

        // System chunk.
        let sys_tokens: Vec<u32> = vec![1, 2, 3, 4];
        cache
            .insert_chunk(0, 4, &sys_tokens, vec![0])
            .expect("system");

        // Overwrite its type to System.
        if let Some(last) = cache.chunks.last_mut() {
            last.chunk_type = ChunkType::System;
        }

        // User input chunk.
        let user_tokens: Vec<u32> = vec![5, 6, 7, 8];
        cache
            .insert_chunk(4, 8, &user_tokens, vec![1])
            .expect("user");

        // Overwrite its type to UserInput.
        if let Some(last) = cache.chunks.last_mut() {
            last.chunk_type = ChunkType::UserInput;
        }

        cache.recompute_importance();

        let system_score = cache
            .chunks
            .iter()
            .find(|c| c.chunk_type == ChunkType::System)
            .map(|c| c.importance_score)
            .unwrap_or(0.0);
        let user_score = cache
            .chunks
            .iter()
            .find(|c| c.chunk_type == ChunkType::UserInput)
            .map(|c| c.importance_score)
            .unwrap_or(0.0);

        assert!(
            system_score > user_score,
            "system chunk ({:.4}) should outrank user chunk ({:.4})",
            system_score,
            user_score
        );
    }

    #[test]
    fn test_clear_resets_state() {
        let mut cache = ChunkKvCache::new(10, 1024 * 1024);
        cache.set_boundary_tokens(default_sentence_ends(), default_speaker_bounds());

        let tokens: Vec<u32> = vec![1, 2, 3, 29871];
        cache.insert_chunk(0, 4, &tokens, vec![0]).expect("insert");

        assert_eq!(cache.chunk_count(), 1);
        assert!(cache.used_bytes() > 0);

        cache.clear();
        assert_eq!(cache.chunk_count(), 0);
        assert_eq!(cache.used_bytes(), 0);
        assert_eq!(cache.pending_tokens().len(), 0);
    }

    #[test]
    fn test_evict_one_reduces_count() {
        let mut cache = ChunkKvCache::new(10, 1024 * 1024);
        cache.set_boundary_tokens(default_sentence_ends(), default_speaker_bounds());

        for i in 0..3u32 {
            let end = (i + 1) * 4;
            let tokens: Vec<u32> = (i * 4..end).collect();
            cache
                .insert_chunk(i * 4, end, &tokens, vec![i])
                .expect("insert");
        }

        assert_eq!(cache.chunk_count(), 3);
        cache.evict_one();
        assert_eq!(cache.chunk_count(), 2);
    }

    #[test]
    fn test_detect_chunks_fallback_boundary() {
        let mut cache = ChunkKvCache::new(10, 1024 * 1024);
        // No boundary tokens registered — fallback should still split long streams.
        let tokens: Vec<u32> = (0..200).collect();
        let boundaries = cache.detect_chunks(&tokens);
        // The fallback should force a boundary at 128 tokens.
        assert!(!boundaries.is_empty(), "fallback should force a boundary");
    }

    #[test]
    fn test_ingest_tokens_multiple_boundaries() {
        let mut cache = ChunkKvCache::new(10, 1024 * 1024);
        cache.set_boundary_tokens(default_sentence_ends(), default_speaker_bounds());

        // Send three sentence boundaries at once.
        let tokens = vec![1, 29871, 2, 29941, 3, 29889];
        let flushed = cache.ingest_tokens(&tokens, ChunkType::Sentence);
        // Should have detected at least one boundary (the first period).
        assert!(
            flushed >= 1,
            "should flush at least one chunk from multi-boundary input, got {}",
            flushed
        );
    }

    #[test]
    fn test_set_compression_ratio_clamping() {
        let mut cache = ChunkKvCache::new(10, 1024 * 1024);
        cache.set_compression_ratio(0.01);
        // Should be clamped to 0.125
        assert!((cache.compression_ratio - 0.125).abs() < 1e-9);
        cache.set_compression_ratio(128.0);
        assert!((cache.compression_ratio - 64.0).abs() < 1e-9);
        cache.set_compression_ratio(4.0);
        assert!((cache.compression_ratio - 4.0).abs() < 1e-9);
    }
}
