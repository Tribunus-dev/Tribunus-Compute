//! Tensix layout transform.
//! Converts generic tensor shapes to Tensix-native 32x32 tile format.
//! Handles weight interleaving across DRAM banks and CB buffer sizing.

/// Tensix tile shape (always 32x32).
pub const TENSIX_TILE_ROWS: u32 = 32;
pub const TENSIX_TILE_COLS: u32 = 32;
pub const TENSIX_TILE_ELEMENTS: u32 = TENSIX_TILE_ROWS * TENSIX_TILE_COLS;

/// Convert a shape to tile count (ceil division).
pub fn shape_to_tiles(rows: u32, cols: u32) -> (u32, u32) {
    let tile_rows = (rows + TENSIX_TILE_ROWS - 1) / TENSIX_TILE_ROWS;
    let tile_cols = (cols + TENSIX_TILE_COLS - 1) / TENSIX_TILE_COLS;
    (tile_rows, tile_cols)
}

/// Padded tile-aligned dimensions.
pub fn tile_aligned_dims(rows: u32, cols: u32) -> (u32, u32) {
    let (tiles_r, tiles_c) = shape_to_tiles(rows, cols);
    (tiles_r * TENSIX_TILE_ROWS, tiles_c * TENSIX_TILE_COLS)
}

/// Interleave a weight matrix across N DRAM banks for bandwidth.
/// Returns per-bank byte counts for buffer allocation.
pub struct BankInterleavingPlan {
    pub num_banks: u32,
    pub bank_bytes: Vec<u64>,
    pub tile_rows: u32,
    pub tile_cols: u32,
    pub dtype_bytes: u32,
}

impl BankInterleavingPlan {
    /// Plan how to distribute weight rows across DRAM banks.
    pub fn new(weight_rows: u32, weight_cols: u32, num_banks: u32, dtype_bytes: u32) -> Self {
        let (tile_rows, _) = shape_to_tiles(weight_rows, weight_cols);
        let total_tiles = tile_rows * (weight_cols / TENSIX_TILE_COLS);
        let tiles_per_bank = (total_tiles + num_banks - 1) / num_banks;
        let mut bank_bytes = Vec::new();
        for _ in 0..num_banks {
            bank_bytes
                .push(tiles_per_bank as u64 * TENSIX_TILE_ELEMENTS as u64 * dtype_bytes as u64);
        }
        BankInterleavingPlan {
            num_banks,
            bank_bytes,
            tile_rows: TENSIX_TILE_ROWS,
            tile_cols: TENSIX_TILE_COLS,
            dtype_bytes,
        }
    }

    pub fn total_bytes(&self) -> u64 {
        self.bank_bytes.iter().sum()
    }

    pub fn total_tiles(&self) -> u32 {
        let per_bank = if self.num_banks > 0 {
            self.bank_bytes[0] / (TENSIX_TILE_ELEMENTS as u64 * self.dtype_bytes as u64)
        } else {
            0
        };
        (per_bank as u32) * self.num_banks
    }
}

/// Determine circular buffer (CB) size for a compute kernel.
/// Double-buffered: 2 tiles per input/output.
pub fn cb_buffer_size(
    tile_rows: u32,
    tile_cols: u32,
    dtype_bytes: u32,
    num_buffers: u32, // 2 = double buffer
) -> u64 {
    tile_rows as u64 * tile_cols as u64 * dtype_bytes as u64 * num_buffers as u64
}

/// Standard Tensix CB sizing for matmul.
pub fn matmul_cb_sizes(_k_tiles: u32, _n_tiles: u32, dtype_bytes: u32) -> (u64, u64, u64) {
    let input_cb = cb_buffer_size(TENSIX_TILE_ROWS, TENSIX_TILE_COLS, dtype_bytes, 2);
    let weight_cb = cb_buffer_size(TENSIX_TILE_ROWS, TENSIX_TILE_COLS, dtype_bytes, 2);
    let output_cb = cb_buffer_size(TENSIX_TILE_ROWS, TENSIX_TILE_COLS, dtype_bytes, 2);
    (input_cb, weight_cb, output_cb)
}
