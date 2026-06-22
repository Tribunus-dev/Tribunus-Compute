//! Tensix fusion region placer.
//! Maps a fusion region (group of related compute_ir ops) to a grid of
//! Tensix cores across multiple cards, with NOC routing within each card
//! and Ethernet routing between cards following a predetermined golden path.

use super::tensix::{CardCoord, GoldenPath, InterconnectType};
use crate::Result;

/// Role of a kernel on a Tensix core.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum KernelRole {
    ReaderUnpack,
    MathCompute,
    WriterPack,
}

/// A dataflow route in the multi-card topology.
/// Routes within a card use `InterconnectType::Noc`;
/// routes between cards use `InterconnectType::Ethernet`.
#[derive(Clone, Debug)]
pub struct CardRoute {
    pub source: CardCoord,
    pub source_role: KernelRole,
    pub dest: CardCoord,
    pub dest_role: KernelRole,
    pub interconnect: InterconnectType,
    pub channel: u32,
    pub bytes_per_tile: u32,
}

/// Core placement on a single Tensix card.
#[derive(Clone, Debug)]
pub struct CardCorePlacement {
    pub card_id: u32,
    /// Cores allocated on this card.
    pub cores: Vec<CardCoord>,
    /// Grid dimensions on this card.
    pub grid_rows: u32,
    pub grid_cols: u32,
}

/// Description of one card's weight shard for a fused region.
///
/// Weights are distributed across cards according to the golden path.
/// Each card holds the shard needed for its portion of the pipeline.
#[derive(Clone, Debug)]
pub struct WeightShard {
    pub card_id: u32,
    pub tensor_name: String,
    pub byte_offset: u64,
    pub byte_size: u64,
}

/// A single fused execution region spanning multiple Tensix cards.
#[derive(Clone, Debug)]
pub struct TensixFusionRegion {
    pub name: String,
    /// Predetermined dataflow path through the card mesh.
    pub golden_path: GoldenPath,
    /// Core placement for each card in the golden path.
    pub card_placements: Vec<CardCorePlacement>,
    /// All routes: NOC within cards, Ethernet between cards.
    pub routes: Vec<CardRoute>,
    /// Weight shards distributed across cards.
    pub weight_shards: Vec<WeightShard>,
}

/// Count of each op type in a fusion region.
#[derive(Clone, Debug, Default)]
pub struct OpCounts {
    pub matmul: u32,
    pub sdpa: u32,
    pub rms_norm: u32,
    pub rope: u32,
    pub silu: u32,
    pub residual_add: u32,
    pub elementwise: u32,
}

impl OpCounts {
    /// Total core slots required for all ops (assuming pipelined execution).
    pub fn total_core_slots(&self) -> u32 {
        self.matmul
            + self.sdpa * 4
            + self.rms_norm
            + self.rope
            + self.silu
            + self.residual_add
            + self.elementwise
    }
}

// ---------------------------------------------------------------------------
// Placement helpers
// ---------------------------------------------------------------------------

/// Build a rectangular grid of `CardCoord` for a single card.
fn build_card_grid(card_id: u32, core_count: u32) -> (Vec<CardCoord>, u32, u32) {
    let cols = (core_count as f64).sqrt().ceil() as u32;
    let rows = (core_count + cols - 1) / cols;

    let mut cores = Vec::with_capacity(core_count as usize);
    for row in 0..rows {
        for col in 0..cols {
            if cores.len() >= core_count as usize {
                break;
            }
            cores.push(CardCoord {
                card_id,
                noc_x: col,
                noc_y: row,
            });
        }
    }

    (cores, rows, cols)
}

/// Distribute `total_ops` across `num_cards` as evenly as possible.
fn distribute_ops(total_ops: u32, num_cards: u32) -> Vec<u32> {
    if num_cards == 0 {
        return vec![];
    }
    let base = total_ops / num_cards;
    let extra = (total_ops % num_cards) as usize;
    let mut dist = vec![base; num_cards as usize];
    for d in dist.iter_mut().take(extra) {
        *d += 1;
    }
    dist
}

// ---------------------------------------------------------------------------
// Public placement entry point
// ---------------------------------------------------------------------------

/// Place a fusion region across multiple Tensix cards following a golden path.
///
/// `golden_path` describes the ordered sequence of cards data flows through.
/// `op_counts` describes the operations to place.
/// `cores_per_card` is the number of Tensix cores available on each card.
///
/// Returns the core placement, NOC routes within cards, Ethernet routes
/// between cards, and weight shard descriptions.
pub fn place_fusion_region(
    name: &str,
    op_counts: &OpCounts,
    golden_path: &GoldenPath,
    cores_per_card: u32,
) -> Result<TensixFusionRegion> {
    let num_cards = golden_path.ordered_cards.len() as u32;
    let matmul_per_card = distribute_ops(op_counts.matmul, num_cards);

    let mut card_placements = Vec::with_capacity(num_cards as usize);
    let mut routes: Vec<CardRoute> = Vec::new();
    let mut weight_shards: Vec<WeightShard> = Vec::new();

    // Per-card last-core tracking for intra-card routing chains
    let mut free_cores: Vec<Vec<CardCoord>> = Vec::new();

    for (idx, &card_id) in golden_path.ordered_cards.iter().enumerate() {
        let mm_ops = matmul_per_card[idx];
        // Reserve cores: one per matmul, plus slack for elementwise
        let needed = mm_ops + 2;
        let card_cores = cores_per_card.min(needed);

        let (coords, rows, cols) = build_card_grid(card_id, card_cores);
        let cores_on_card = coords.len() as u32;

        card_placements.push(CardCorePlacement {
            card_id,
            cores: coords.clone(),
            grid_rows: rows,
            grid_cols: cols,
        });

        // NOC routes within this card: chain matmul cores via MathCompute
        let matmul_count = mm_ops.min(cores_on_card);
        for i in 0..matmul_count.saturating_sub(1) {
            let src = coords[i as usize];
            let dst = coords[(i + 1) as usize];
            routes.push(CardRoute {
                source: src,
                source_role: KernelRole::MathCompute,
                dest: dst,
                dest_role: KernelRole::MathCompute,
                interconnect: InterconnectType::Noc,
                channel: 1,
                bytes_per_tile: 32 * 32 * 2, // bf16 tile
            });
        }

        // Track the last used core on this card for inter-card chaining
        if let Some(&last) = coords.last() {
            free_cores.push(vec![last]);
        }

        // Weight shard for this card (placeholder — actual sizes computed by compiler)
        if matmul_count > 0 {
            weight_shards.push(WeightShard {
                card_id,
                tensor_name: format!("{name}_card{card_id}_weights"),
                byte_offset: 0,
                byte_size: 0, // filled in by weight planner
            });
        }
    }

    // Ethernet routes between cards following the golden path
    for i in 0..golden_path.ordered_cards.len().saturating_sub(1) {
        let src_card = golden_path.ordered_cards[i];
        let dst_card = golden_path.ordered_cards[i + 1];

        // Last core on src card -> first core on dst card
        let src_coord = free_cores[i].last().copied().unwrap_or(CardCoord {
            card_id: src_card,
            noc_x: 0,
            noc_y: 0,
        });
        let dst_coord = card_placements[i + 1]
            .cores
            .first()
            .copied()
            .unwrap_or(CardCoord {
                card_id: dst_card,
                noc_x: 0,
                noc_y: 0,
            });

        routes.push(CardRoute {
            source: src_coord,
            source_role: KernelRole::WriterPack,
            dest: dst_coord,
            dest_role: KernelRole::ReaderUnpack,
            interconnect: InterconnectType::Ethernet,
            channel: 0,
            bytes_per_tile: 32 * 32 * 2,
        });
    }

    Ok(TensixFusionRegion {
        name: name.to_string(),
        golden_path: golden_path.clone(),
        card_placements,
        routes,
        weight_shards,
    })
}
