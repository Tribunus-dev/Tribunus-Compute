//! Compiled residency plan types for the SealedComputeImageExecutable.
//!
//! This module contains the compiler-computed memory residency schedule
//! that the runtime executes.  Every type is derived with `Debug`,
//! `Clone`, `Serialize`, and `Deserialize` for inspection, caching,
//! and serialization across process boundaries.
pub(crate) mod arena;
pub mod prefetch;
pub(crate) mod weights;

pub use self::weights::{ResidencyClassifier, WeightObject};
pub mod admission;
pub mod receipts;

pub use self::plan::PeakMemoryAnalyzer;
pub use self::prefetch::PrefetchScheduleBuilder;
pub use plan::{
    ActivationArenaRequirements, CompiledResidencyPlan, EvictableWeightObject, EvictionPolicy,
    KvCacheRequirements, MemoryAdmissionContract, PeakMemoryEstimate, PrefetchAction,
    PrefetchPriority, RequiredWeightObject, RequiredWeightObjectId, ResidencyClass,
    ResidencyPlanId,
};
pub use admission::{ResidencyAdmission, ResidencyAdmissionResult, ResidencyRefusalReason};
pub use receipts::{ResidencyAdmissionReceipt, ResidencyExecutionReceipt};

pub mod plan;
