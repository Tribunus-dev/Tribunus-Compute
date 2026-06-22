use super::Slot;

impl Slot {
    /// Create a new empty slot with the given id and default backend.
    pub fn new(id: usize) -> Self {
        Slot {
            id,
            request_id: None,
            tokens_generated: 0,
            kv_cache_start: 0,
            kv_cache_length: 0,
            backend_id: 0,
            kv_cache_pages: vec![],
        }
    }

    /// Returns true if the slot is not assigned to any request.
    pub fn is_free(&self) -> bool {
        self.request_id.is_none()
    }
}
