use crate::scheduling::RequestState;

impl super::Request {
    /// Create a new request for the given prompt.
    pub fn new(prompt: Vec<u32>, max_tokens: usize) -> Self {
        Self {
            id: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_nanos() as u64,
            prompt,
            max_tokens,
            priority: 0,
            state: RequestState::Queued,
            created_at: std::time::Instant::now(),
            slot: None,
        }
    }

    /// Transition the request to a new state.
    pub fn transition(&mut self, state: RequestState) {
        self.state = state;
    }
}
