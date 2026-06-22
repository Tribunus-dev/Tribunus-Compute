//! Arena plan types for serialized phase programs.
//!
//! These types define activation arena regions, ring-buffer
//! requirements, and memory contracts that are pre-compiled and
//! baked into the executable image.

pub type ArenaRegionId = String;
pub type ArenaPlanId = String;

#[derive(Debug, Clone)]
pub struct ArenaPlanDescriptor {
    pub plan_id: ArenaPlanId,
    pub regions: Vec<ArenaRegionDescriptor>,
    pub peak_activation_bytes: u64,
}

#[derive(Debug, Clone)]
pub struct ArenaRegionDescriptor {
    pub region_id: ArenaRegionId,
    pub byte_size: u64,
    pub alignment: u64,
    pub producer_phase: Option<String>,
    pub consumer_phase: Option<String>,
}
