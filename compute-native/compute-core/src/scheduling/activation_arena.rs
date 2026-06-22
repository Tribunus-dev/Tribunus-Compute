use crate::scheduling::activation_binding::ArenaBinding;

/// Manages the activation arena — a pre-allocated memory region for
/// activation tensors that persists across phase dispatches.
///
/// The arena avoids repeated allocation/deallocation by recycling
/// slots according to the compiler's EmittedArenaPlan.
pub struct ActivationArena {
    total_bytes: u64,
    allocated_bytes: u64,
}

impl ActivationArena {
    pub fn new(total_bytes: u64) -> Self {
        Self {
            total_bytes,
            allocated_bytes: 0,
        }
    }

    /// Allocate a slot in the arena.
    pub fn allocate(&mut self, byte_size: u64, alignment: u64) -> Result<ArenaBinding, String> {
        let aligned_offset = (self.allocated_bytes + alignment - 1) & !(alignment - 1);
        if aligned_offset + byte_size > self.total_bytes {
            return Err(format!(
                "arena exhausted: need {} bytes at offset {}, have {} total",
                byte_size, aligned_offset, self.total_bytes
            ));
        }
        let binding = ArenaBinding {
            slot_id: String::new(),
            offset: aligned_offset,
            byte_size,
            generation: 0,
        };
        self.allocated_bytes = aligned_offset + byte_size;
        Ok(binding)
    }

    pub fn reset(&mut self) {
        self.allocated_bytes = 0;
    }

    pub fn utilization(&self) -> f64 {
        if self.total_bytes == 0 {
            0.0
        } else {
            self.allocated_bytes as f64 / self.total_bytes as f64
        }
    }
}
