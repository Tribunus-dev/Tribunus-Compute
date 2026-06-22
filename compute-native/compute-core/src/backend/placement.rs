//! PlacementSet and hazard tracking for heterogeneous Apple execution.
//! Each op declares legal placements referencing the same MemoryRegionId.
//! Hazard barriers enforce read/write ordering when crossing lanes.

use crate::backend::unified_arena::ArenaView;

/// Identifies an execution lane.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum ExecutionLane {
    MlxGpu,
    AccelerateCpu,
    CoreMlAne,
    CandleCpu,
    Tensix,
    IntelLevelZero,
}

/// Which lane is this op legally executable on?
#[derive(Clone, Debug)]
pub struct PlacementSet {
    pub op_name: String,
    pub candidates: Vec<ExecutionLane>,
    pub primary: ExecutionLane,
    pub memory_region_id: u64, // references UnifiedExecutionArena region
}

impl PlacementSet {
    pub fn new(op_name: &str, primary: ExecutionLane, memory_region_id: u64) -> Self {
        PlacementSet {
            op_name: op_name.to_string(),
            candidates: vec![primary],
            primary,
            memory_region_id,
        }
    }

    pub fn add_candidate(&mut self, lane: ExecutionLane) {
        if !self.candidates.contains(&lane) {
            self.candidates.push(lane);
        }
    }

    pub fn is_legal(&self, lane: ExecutionLane) -> bool {
        self.candidates.contains(&lane)
    }
}

/// A read-write hazard barrier between two arena views.
#[derive(Clone, Debug)]
pub struct HazardBarrier {
    pub view: ArenaView,
    pub last_writer: Option<ExecutionLane>,
    pub pending_readers: Vec<ExecutionLane>,
}

impl HazardBarrier {
    pub fn new(view: ArenaView) -> Self {
        HazardBarrier {
            view,
            last_writer: None,
            pending_readers: Vec::new(),
        }
    }

    /// Record a write from a lane. Invalidates pending readers if lane changed.
    pub fn write(&mut self, lane: ExecutionLane) {
        if let Some(last) = self.last_writer {
            if last != lane {
                // Cross-lane write hazard — must sync
                self.pending_readers.clear();
            }
        }
        self.last_writer = Some(lane);
    }

    /// Record a read from a lane. Must sync if last writer is a different lane.
    pub fn read(&mut self, lane: ExecutionLane) -> bool {
        if let Some(last) = self.last_writer {
            if last != lane {
                self.pending_readers.push(lane);
                return true; // hazard: need synchronization
            }
        }
        false
    }

    pub fn clear(&mut self) {
        self.last_writer = None;
        self.pending_readers.clear();
    }
}

/// Hazard tracker for the full request execution.
pub struct HazardTracker {
    barriers: std::collections::HashMap<u64, HazardBarrier>,
}

impl HazardTracker {
    pub fn new() -> Self {
        HazardTracker {
            barriers: std::collections::HashMap::new(),
        }
    }

    pub fn register_view(&mut self, view: ArenaView) {
        self.barriers
            .entry(view.0)
            .or_insert_with(|| HazardBarrier::new(view));
    }

    pub fn write(&mut self, view: ArenaView, lane: ExecutionLane) {
        if let Some(barrier) = self.barriers.get_mut(&view.0) {
            barrier.write(lane);
        }
    }

    pub fn needs_sync(&mut self, view: ArenaView, lane: ExecutionLane) -> bool {
        self.barriers
            .get_mut(&view.0)
            .map(|b| b.read(lane))
            .unwrap_or(false)
    }

    pub fn clear(&mut self) {
        self.barriers.clear();
    }
}
