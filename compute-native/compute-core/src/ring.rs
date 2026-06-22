pub struct ArenaPage {
    pub pointer: *mut u8,
    pub byte_length: usize,
    pub iosurface_id: u32,
    pub backend_bits: u8,
    pub lease_count: u32,
}
unsafe impl Send for ArenaPage {}
unsafe impl Sync for ArenaPage {}
pub struct RingSlot {
    pub state: SlotState,
    pub generation: u64,
    pub page_index: Option<usize>,
}
#[derive(Clone, Copy, PartialEq)]
pub enum SlotState {
    Free,
    Reserved,
    Writing,
    Written,
    Readable,
    Verifying,
    Committed,
    Recycled,
    DraftReserved,
    DraftWritten,
    VerifierVisible,
    Accepted,
    Rejected,
}
pub struct Ring {
    pub slots: Vec<RingSlot>,
    pub capacity: u32,
}
pub struct Lease {
    pub page_id: usize,
    pub generation: u64,
}
impl Ring {
    pub fn lease(&mut self, _backend: u8) -> Result<Lease, String> {
        Err("not impl".into())
    }
    pub fn release(&mut self, _lease: &Lease) {}
}
