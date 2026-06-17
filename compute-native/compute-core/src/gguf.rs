//! Pure Rust GGUF format parser — reads from [`MappedSegment`] zero-copy.
//!
//! GGUF is the file format used by llama.cpp for quantized model storage.
//! Layout (all multi-byte values are little-endian):
//!
//!   - Magic:     u32 (0x46554747 = "GGUF")
//!   - Version:   u32
//!   - TensorCount:           u64
//!   - MetadataKeyValueCount: u64
//!   - MetadataKeyValue × MetadataKeyValueCount
//!   - TensorInfoEntry × TensorCount
//!   - TensorData (32-byte aligned start)

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use crate::mapped_image::MappedSegment;

// ──────────────────────────────────────────────
// Public types
// ──────────────────────────────────────────────

/// Parsed GGUF model metadata.
#[derive(Debug, Clone, serde::Serialize)]
pub struct GgufModel {
    pub path: PathBuf,
    pub version: u32,
    pub tensor_count: u64,
    pub metadata: HashMap<String, GgufMetadataValue>,
    pub tensors: Vec<GgufTensorInfo>,
}

/// A single metadata key-value entry parsed from the GGUF header.
#[derive(Debug, Clone, serde::Serialize)]
#[serde(untagged)]
pub enum GgufMetadataValue {
    Uint8(u8),
    Int8(i8),
    Uint16(u16),
    Int16(i16),
    Uint32(u32),
    Int32(i32),
    Float32(f32),
    Bool(bool),
    Uint64(u64),
    Int64(i64),
    Float64(f64),
    String(String),
    Array(Vec<GgufMetadataValue>),
}

/// Information about a single tensor in the GGUF file.
#[derive(Debug, Clone, serde::Serialize)]
pub struct GgufTensorInfo {
    pub name: String,
    pub shape: Vec<u64>,
    pub dtype: GgufDtype,
    /// Byte offset from the start of the TensorData section.
    pub offset: u64,
    /// Total size of this tensor's data in bytes (computed from shape × dtype).
    pub size_bytes: u64,
}

/// Tensor element / quantization type.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize)]
pub enum GgufDtype {
    F32,
    F16,
    BF16,
    I8,
    I16,
    I32,
    I64,
    #[allow(non_camel_case_types)]
    Q4_0,
    #[allow(non_camel_case_types)]
    Q4_1,
    #[allow(non_camel_case_types)]
    Q5_0,
    #[allow(non_camel_case_types)]
    Q5_1,
    #[allow(non_camel_case_types)]
    Q8_0,
    #[allow(non_camel_case_types)]
    Q8_1,
    #[allow(non_camel_case_types)]
    Q2_K,
    #[allow(non_camel_case_types)]
    Q3_K,
    #[allow(non_camel_case_types)]
    Q4_K,
    #[allow(non_camel_case_types)]
    Q5_K,
    #[allow(non_camel_case_types)]
    Q6_K,
    #[allow(non_camel_case_types)]
    Q8_K,
}

/// A borrowed view into a single tensor's raw bytes inside a [`MappedSegment`].
#[derive(Debug, Clone)]
pub struct GgufTensorReader {
    segment: Arc<MappedSegment>,
    offset: u64,
    size: u64,
}

impl GgufTensorReader {
    /// Returns a raw pointer to the start of this tensor's data.
    #[inline]
    pub fn data_ptr(&self) -> *const u8 {
        // SAFETY: the segment's mapping pointer is valid for the lifetime of
        // `self.segment` (the Arc), and `offset + size` is bounds-checked at
        // construction in `parse_gguf`.
        unsafe { self.segment.data_ptr().add(self.offset as usize) }
    }

    /// Number of bytes this tensor occupies.
    #[inline]
    pub fn len(&self) -> usize {
        self.size as usize
    }

    /// Whether the tensor has zero bytes.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.size == 0
    }

    /// Copy the tensor data into a freshly allocated `Vec<u8>`.
    pub fn to_vec(&self) -> Vec<u8> {
        // SAFETY: same as data_ptr().
        unsafe {
            std::slice::from_raw_parts(self.data_ptr(), self.len()).to_vec()
        }
    }
}

// ──────────────────────────────────────────────
// Internal helpers
// ──────────────────────────────────────────────

/// Read a `u32` at `offset` in little-endian.
#[inline]
fn read_u32(buf: &[u8], offset: usize) -> u32 {
    let bytes = &buf[offset..offset + 4];
    u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]])
}

/// Read a `u64` at `offset` in little-endian.
#[inline]
fn read_u64(buf: &[u8], offset: usize) -> u64 {
    let bytes = &buf[offset..offset + 8];
    u64::from_le_bytes([
        bytes[0], bytes[1], bytes[2], bytes[3],
        bytes[4], bytes[5], bytes[6], bytes[7],
    ])
}

/// Read a `u16` at `offset` in little-endian.
#[inline]
fn read_u16(buf: &[u8], offset: usize) -> u16 {
    let bytes = &buf[offset..offset + 2];
    u16::from_le_bytes([bytes[0], bytes[1]])
}

/// Read an `i64` at `offset` in little-endian.
#[inline]
fn read_i64(buf: &[u8], offset: usize) -> i64 {
    let bytes = &buf[offset..offset + 8];
    i64::from_le_bytes([
        bytes[0], bytes[1], bytes[2], bytes[3],
        bytes[4], bytes[5], bytes[6], bytes[7],
    ])
}

/// Read a `f32` at `offset` in little-endian.
#[inline]
fn read_f32(buf: &[u8], offset: usize) -> f32 {
    f32::from_bits(read_u32(buf, offset))
}

/// Read a `f64` at `offset` in little-endian.
#[inline]
fn read_f64(buf: &[u8], offset: usize) -> f64 {
    f64::from_bits(read_u64(buf, offset))
}

// ──────────────────────────────────────────────
// Dtype helpers
// ──────────────────────────────────────────────

fn gguf_dtype_from_u32(v: u32) -> Result<GgufDtype, String> {
    Ok(match v {
        0 => GgufDtype::F32,
        1 => GgufDtype::F16,
        2 => GgufDtype::Q4_0,
        3 => GgufDtype::Q4_1,
        4 => GgufDtype::Q5_0,
        5 => GgufDtype::Q5_1,
        6 => GgufDtype::Q8_0,
        7 => GgufDtype::Q8_1,
        8 => GgufDtype::Q2_K,
        9 => GgufDtype::Q3_K,
        10 => GgufDtype::Q4_K,
        11 => GgufDtype::Q5_K,
        12 => GgufDtype::Q6_K,
        13 => GgufDtype::Q8_K,
        14 => GgufDtype::I8,
        15 => GgufDtype::I16,
        16 => GgufDtype::I32,
        17 => GgufDtype::I64,
        24 => GgufDtype::BF16,
        _ => return Err(format!("unknown GGUF tensor dtype {}", v)),
    })
}

/// Returns `(block_elements, block_bytes)` for quantised types, or
/// `(1, element_size)` for unquantised types, so total bytes can be computed
/// as `ceil(num_elements / block_elements) * block_bytes`.
fn dtype_block_info(dtype: GgufDtype) -> (u64, u64) {
    match dtype {
        GgufDtype::F32 => (1, 4),
        GgufDtype::F16 | GgufDtype::BF16 => (1, 2),
        GgufDtype::I8 => (1, 1),
        GgufDtype::I16 => (1, 2),
        GgufDtype::I32 => (1, 4),
        GgufDtype::I64 => (1, 8),
        GgufDtype::Q4_0 => (32, 18),
        GgufDtype::Q4_1 => (32, 20),
        GgufDtype::Q5_0 => (32, 22),
        GgufDtype::Q5_1 => (32, 24),
        GgufDtype::Q8_0 => (32, 34),
        GgufDtype::Q8_1 => (32, 36),
        GgufDtype::Q2_K => (256, 84),
        GgufDtype::Q3_K => (256, 110),
        GgufDtype::Q4_K => (256, 144),
        GgufDtype::Q5_K => (256, 176),
        GgufDtype::Q6_K => (256, 210),
        GgufDtype::Q8_K => (256, 292),
    }
}

/// Compute the on-disk byte size of a tensor given its shape and dtype.
fn tensor_size_bytes(shape: &[u64], dtype: GgufDtype) -> u64 {
    let n_elems: u64 = shape.iter().copied().product();
    let (block_elems, block_bytes) = dtype_block_info(dtype);
    let num_blocks = n_elems.div_ceil(block_elems);
    num_blocks * block_bytes
}

/// Round `offset` up to the next 32-byte alignment boundary.
const fn align_32(offset: u64) -> u64 {
    (offset + 31) & !(31u64)
}

// ──────────────────────────────────────────────
// Metadata value parser
// ──────────────────────────────────────────────

/// Metadata value type tags (GGUF spec).
const GGUF_TYPE_UINT8: u32 = 0;
const GGUF_TYPE_INT8: u32 = 1;
const GGUF_TYPE_UINT16: u32 = 2;
const GGUF_TYPE_INT16: u32 = 3;
const GGUF_TYPE_UINT32: u32 = 4;
const GGUF_TYPE_INT32: u32 = 5;
const GGUF_TYPE_FLOAT32: u32 = 6;
const GGUF_TYPE_BOOL: u32 = 7;
const GGUF_TYPE_STRING: u32 = 8;
const GGUF_TYPE_ARRAY: u32 = 9;
const GGUF_TYPE_UINT64: u32 = 10;
const GGUF_TYPE_INT64: u32 = 11;
const GGUF_TYPE_FLOAT64: u32 = 12;

/// Parse a single metadata key-value pair from `buf` starting at `offset`.
/// Returns `((key, value), new_offset)`.
fn parse_metadata_kv(buf: &[u8], offset: usize) -> Result<((String, GgufMetadataValue), usize), String> {
    let mut pos = offset;

    // KeyLength: u64
    let key_len = read_u64(buf, pos) as usize;
    pos += 8;
    if pos + key_len > buf.len() {
        return Err("truncated metadata key".into());
    }
    let key = std::str::from_utf8(&buf[pos..pos + key_len])
        .map_err(|e| format!("metadata key is not valid UTF-8: {e}"))?
        .to_string();
    pos += key_len;

    // ValueType: u32
    let value_type = read_u32(buf, pos);
    pos += 4;

    let (value, pos) = parse_metadata_value(buf, pos, value_type)?;

    Ok(((key, value), pos))
}

/// Parse a metadata value given its type tag.
fn parse_metadata_value(buf: &[u8], offset: usize, ty: u32) -> Result<(GgufMetadataValue, usize), String> {
    let mut pos = offset;

    let value = match ty {
        GGUF_TYPE_UINT8 => {
            let v = buf[pos];
            pos += 1;
            GgufMetadataValue::Uint8(v)
        }
        GGUF_TYPE_INT8 => {
            let v = buf[pos] as i8;
            pos += 1;
            GgufMetadataValue::Int8(v)
        }
        GGUF_TYPE_UINT16 => {
            let v = read_u16(buf, pos);
            pos += 2;
            GgufMetadataValue::Uint16(v)
        }
        GGUF_TYPE_INT16 => {
            let v = read_u16(buf, pos) as i16;
            pos += 2;
            GgufMetadataValue::Int16(v)
        }
        GGUF_TYPE_UINT32 => {
            let v = read_u32(buf, pos);
            pos += 4;
            GgufMetadataValue::Uint32(v)
        }
        GGUF_TYPE_INT32 => {
            let v = read_u32(buf, pos) as i32;
            pos += 4;
            GgufMetadataValue::Int32(v)
        }
        GGUF_TYPE_FLOAT32 => {
            let v = read_f32(buf, pos);
            pos += 4;
            GgufMetadataValue::Float32(v)
        }
        GGUF_TYPE_BOOL => {
            let v = buf[pos] != 0;
            pos += 1;
            GgufMetadataValue::Bool(v)
        }
        GGUF_TYPE_STRING => {
            // String: Uint64 length + bytes
            let len = read_u64(buf, pos) as usize;
            pos += 8;
            if pos + len > buf.len() {
                return Err("truncated metadata string value".into());
            }
            let s = std::str::from_utf8(&buf[pos..pos + len])
                .map_err(|e| format!("metadata string value is not valid UTF-8: {e}"))?
                .to_string();
            pos += len;
            GgufMetadataValue::String(s)
        }
        GGUF_TYPE_ARRAY => {
            // Array: Uint32 type + Uint64 count + values
            let elem_type = read_u32(buf, pos);
            pos += 4;
            let count = read_u64(buf, pos) as usize;
            pos += 8;
            let mut elems = Vec::with_capacity(count);
            for _ in 0..count {
                let (v, new_pos) = parse_metadata_value(buf, pos, elem_type)?;
                pos = new_pos;
                elems.push(v);
            }
            GgufMetadataValue::Array(elems)
        }
        GGUF_TYPE_UINT64 => {
            let v = read_u64(buf, pos);
            pos += 8;
            GgufMetadataValue::Uint64(v)
        }
        GGUF_TYPE_INT64 => {
            let v = read_i64(buf, pos);
            pos += 8;
            GgufMetadataValue::Int64(v)
        }
        GGUF_TYPE_FLOAT64 => {
            let v = read_f64(buf, pos);
            pos += 8;
            GgufMetadataValue::Float64(v)
        }
        _ => return Err(format!("unknown GGUF metadata value type {}", ty)),
    };

    Ok((value, pos))
}

/// Parse a single tensor info entry from `buf` starting at `offset`.
/// Returns `(GgufTensorInfo, new_offset)`.
fn parse_tensor_info(buf: &[u8], offset: usize) -> Result<(GgufTensorInfo, usize), String> {
    let mut pos = offset;

    // NameLength: u64
    let name_len = read_u64(buf, pos) as usize;
    pos += 8;
    if pos + name_len > buf.len() {
        return Err("truncated tensor name".into());
    }
    let name = std::str::from_utf8(&buf[pos..pos + name_len])
        .map_err(|e| format!("tensor name is not valid UTF-8: {e}"))?
        .trim_end_matches('\0') // GGUF names are often null-terminated
        .to_string();
    pos += name_len;

    // NDims: u32
    let n_dims = read_u32(buf, pos) as usize;
    pos += 4;

    // Dims: [u64; NDims]
    if pos + n_dims * 8 > buf.len() {
        return Err("truncated tensor shape dimensions".into());
    }
    let mut shape = Vec::with_capacity(n_dims);
    for _ in 0..n_dims {
        let dim = read_u64(buf, pos);
        pos += 8;
        shape.push(dim);
    }

    // Dtype: u32
    let dtype_code = read_u32(buf, pos);
    pos += 4;
    let dtype = gguf_dtype_from_u32(dtype_code)?;

    // Offset: u64
    let tensor_offset = read_u64(buf, pos);
    pos += 8;

    let size_bytes = tensor_size_bytes(&shape, dtype);

    Ok((
        GgufTensorInfo {
            name,
            shape,
            dtype,
            offset: tensor_offset,
            size_bytes,
        },
        pos,
    ))
}

// ──────────────────────────────────────────────
// Public API
// ──────────────────────────────────────────────

/// Open and parse a GGUF file from `path`.
///
/// Uses [`MappedSegment`] internally so the entire file is mmap'd read-only.
/// All metadata keys and tensor info entries are parsed eagerly from the
/// header. Tensor data bytes remain on the mapped pages and are accessible
/// via [`GgufTensorReader`].
pub fn parse_gguf(path: &Path) -> Result<GgufModel, String> {
    let segment = MappedSegment::new(path, None)
        .map_err(|e| format!("failed to mmap GGUF file {:?}: {e}", path))?;

    let buf: &[u8] = segment.data_slice();
    let file_len = segment.len();

    if file_len < 4 {
        return Err("file too small to contain GGUF magic".into());
    }

    // ── Magic ──
    let magic = read_u32(buf, 0);
    if magic != 0x4655_4747 {
        return Err(format!(
            "invalid GGUF magic: 0x{magic:08X} (expected 0x46554747)"
        ));
    }

    let mut pos: usize = 4;

    // ── Version ──
    if pos + 4 > file_len {
        return Err("truncated GGUF header: missing version".into());
    }
    let version = read_u32(buf, pos);
    pos += 4;

    // ── TensorCount ──
    if pos + 8 > file_len {
        return Err("truncated GGUF header: missing tensor count".into());
    }
    let tensor_count = read_u64(buf, pos);
    pos += 8;

    // ── MetadataKeyValueCount ──
    if pos + 8 > file_len {
        return Err("truncated GGUF header: missing metadata KV count".into());
    }
    let metadata_kv_count = read_u64(buf, pos);
    pos += 8;

    // ── Metadata key-value pairs ──
    let mut metadata = HashMap::new();
    for _ in 0..metadata_kv_count {
        if pos >= file_len {
            return Err("truncated GGUF metadata section".into());
        }
        let ((key, value), new_pos) = parse_metadata_kv(buf, pos)?;
        metadata.insert(key, value);
        pos = new_pos;
    }

    // ── Tensor info entries ──
    let mut tensors = Vec::with_capacity(tensor_count as usize);
    for _ in 0..tensor_count {
        if pos >= file_len {
            return Err("truncated GGUF tensor info section".into());
        }
        let (info, new_pos) = parse_tensor_info(buf, pos)?;
        tensors.push(info);
        pos = new_pos;
    }

    // ── TensorData section starts here, 32-byte aligned ──
    let tensor_data_start = align_32(pos as u64);

    // Basic sanity: the last tensor's data must fit in the file.
    for info in &tensors {
        let end_offset = info.offset + info.size_bytes;
        if tensor_data_start + end_offset > file_len as u64 {
            return Err(format!(
                "tensor {} data at offset {} size {} exceeds file length {}",
                info.name,
                tensor_data_start + info.offset,
                info.size_bytes,
                file_len,
            ));
        }
    }

    Ok(GgufModel {
        path: path.to_path_buf(),
        version,
        tensor_count,
        metadata,
        tensors,
    })
}

/// Build a [`GgufTensorReader`] for accessing a specific tensor's raw data by
/// name.  Returns `None` if the tensor is not found.
///
/// The returned reader borrows the underlying `segment` via `Arc` so the mmap
/// remains alive for the duration of the reader.
pub fn tensor_reader(
    segment: &Arc<MappedSegment>,
    model: &GgufModel,
    tensor_name: &str,
) -> Option<GgufTensorReader> {
    let info = model.tensors.iter().find(|t| t.name == tensor_name)?;
    let tensor_data_start = compute_tensor_data_offset(model, segment.data_slice())?;
    Some(GgufTensorReader {
        segment: segment.clone(),
        offset: tensor_data_start + info.offset,
        size: info.size_bytes,
    })
}

/// Convenience: build a [`GgufTensorReader`] for every tensor in the model.
/// The returned `Vec` is indexed in the same order as `model.tensors`.
pub fn all_tensor_readers(
    segment: &Arc<MappedSegment>,
    model: &GgufModel,
) -> Vec<GgufTensorReader> {
    let buf: &[u8] = segment.data_slice();
    let Some(tensor_data_start) = compute_tensor_data_offset(model, buf) else {
        return Vec::new();
    };
    model
        .tensors
        .iter()
        .map(|info| GgufTensorReader {
            segment: segment.clone(),
            offset: tensor_data_start + info.offset,
            size: info.size_bytes,
        })
        .collect()
}

/// Re-derive the TensorData section start by walking the metadata KV pairs
/// and tensor info entries in the raw buffer without allocating.
/// Returns `None` on parse failure.
fn compute_tensor_data_offset(model: &GgufModel, buf: &[u8]) -> Option<u64> {
    // Fixed header: magic(4) + version(4) + tensor_count(8) + metadata_kv_count(8)
    let mut pos: usize = 4 + 4 + 8 + 8;

    for _ in 0..model.metadata.len() {
        let key_len = read_u64(buf, pos) as usize;
        pos += 8 + key_len;
        let value_type = read_u32(buf, pos);
        pos += 4;
        let (_, new_pos) = parse_metadata_value(buf, pos, value_type).ok()?;
        pos = new_pos;
    }

    // Skip tensor info entries
    for _info in 0..model.tensors.len() {
        let name_len = read_u64(buf, pos) as usize;
        pos += 8 + name_len;
        let n_dims = read_u32(buf, pos) as usize;
        pos += 4;
        pos += n_dims * 8 + 4 + 8;
    }

    Some(align_32(pos as u64))
}

// ──────────────────────────────────────────────
// Tests
// ──────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    /// Helper: write a minimal valid GGUF file to a temp directory and parse
    /// it.  Contains one metadata key and one tensor.
    fn write_minimal_gguf(path: &Path) {
        let mut f = std::fs::File::create(path).unwrap();
        let mut pos: u64 = 0;

        // Magic
        f.write_all(&0x4655_4747u32.to_le_bytes()).unwrap();
        pos += 4;
        // Version = 3
        f.write_all(&3u32.to_le_bytes()).unwrap();
        pos += 4;
        // TensorCount = 1
        f.write_all(&1u64.to_le_bytes()).unwrap();
        pos += 8;
        // MetadataKeyValueCount = 1
        f.write_all(&1u64.to_le_bytes()).unwrap();
        pos += 8;

        // Metadata KV: key = "test.key", value = (string) "hello"
        let key = b"test.key";
        f.write_all(&(key.len() as u64).to_le_bytes()).unwrap();
        pos += 8;
        f.write_all(key).unwrap();
        pos += key.len() as u64;
        f.write_all(&GGUF_TYPE_STRING.to_le_bytes()).unwrap(); // ValueType = String (8)
        pos += 4;
        let val = b"hello";
        f.write_all(&(val.len() as u64).to_le_bytes()).unwrap();
        pos += 8;
        f.write_all(val).unwrap();
        pos += val.len() as u64;

        // Tensor info
        let tname = b"test_tensor\0";
        f.write_all(&(tname.len() as u64).to_le_bytes()).unwrap();
        pos += 8;
        f.write_all(tname).unwrap();
        pos += tname.len() as u64;
        // NDims = 2
        f.write_all(&2u32.to_le_bytes()).unwrap();
        pos += 4;
        // Dims: [4, 8]
        f.write_all(&4u64.to_le_bytes()).unwrap();
        pos += 8;
        f.write_all(&8u64.to_le_bytes()).unwrap();
        pos += 8;
        // Dtype = F32 (0)
        f.write_all(&0u32.to_le_bytes()).unwrap();
        pos += 4;
        // Offset = 0 (first tensor in the data section)
        f.write_all(&0u64.to_le_bytes()).unwrap();
        pos += 8;

        // Pad to 32-byte alignment for TensorData
        let aligned = align_32(pos);
        for _ in 0..(aligned - pos) {
            f.write_all(&[0u8]).unwrap();
        }

        // Tensor data: 4*8 = 32 floats = 128 bytes
        let data: Vec<f32> = (0..32).map(|i| i as f32).collect();
        for v in &data {
            f.write_all(&v.to_le_bytes()).unwrap();
        }
    }

    #[test]
    fn test_parse_minimal_gguf() {
        let dir = std::env::temp_dir().join("gguf_test_minimal");
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("model.gguf");
        write_minimal_gguf(&path);

        let model = parse_gguf(&path).expect("parse_gguf failed");
        assert_eq!(model.version, 3);
        assert_eq!(model.tensor_count, 1);
        assert_eq!(model.metadata.len(), 1);
        match model.metadata.get("test.key") {
            Some(GgufMetadataValue::String(s)) => assert_eq!(s, "hello"),
            other => panic!("expected String, got {other:?}"),
        }
        assert_eq!(model.tensors.len(), 1);
        assert_eq!(model.tensors[0].name, "test_tensor");
        assert_eq!(model.tensors[0].shape, vec![4, 8]);
        assert_eq!(model.tensors[0].dtype, GgufDtype::F32);
        assert_eq!(model.tensors[0].size_bytes, 32 * 4);

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_invalid_magic() {
        let dir = std::env::temp_dir().join("gguf_test_bad_magic");
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("bad.gguf");
        let mut f = std::fs::File::create(&path).unwrap();
        f.write_all(&0xDEAD_BEEFu32.to_le_bytes()).unwrap();
        f.write_all(&[0u8; 8]).unwrap();
        drop(f);

        let result = parse_gguf(&path);
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("magic"));

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_truncated_file() {
        let dir = std::env::temp_dir().join("gguf_test_truncated");
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("truncated.gguf");
        let mut f = std::fs::File::create(&path).unwrap();
        f.write_all(&0x4655_4747u32.to_le_bytes()).unwrap();
        // Only 2 bytes of version – truncated
        f.write_all(&[0u8; 2]).unwrap();
        drop(f);

        let result = parse_gguf(&path);
        assert!(result.is_err());

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_tensor_reader() {
        let dir = std::env::temp_dir().join("gguf_test_reader");
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("model.gguf");
        write_minimal_gguf(&path);

        let model = parse_gguf(&path).unwrap();
        let segment = MappedSegment::new(&path, None).unwrap();

        let readers = all_tensor_readers(&segment, &model);
        assert_eq!(readers.len(), 1);
        assert_eq!(readers[0].len(), 32 * 4);

        let data = readers[0].to_vec();
        assert_eq!(data.len(), 128);
        // Check first and last float
        let first = f32::from_le_bytes([data[0], data[1], data[2], data[3]]);
        assert!((first - 0.0).abs() < f32::EPSILON);
        let last_offset = 124;
        let last = f32::from_le_bytes([
            data[last_offset],
            data[last_offset + 1],
            data[last_offset + 2],
            data[last_offset + 3],
        ]);
        assert!((last - 31.0).abs() < f32::EPSILON);

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_unknown_dtype_error() {
        let result = gguf_dtype_from_u32(99);
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("unknown GGUF tensor dtype"));
    }

    #[test]
    fn test_dtype_block_info() {
        // Spot-check a few dtypes.
        assert_eq!(dtype_block_info(GgufDtype::F32), (1, 4));
        assert_eq!(dtype_block_info(GgufDtype::I8), (1, 1));
        assert_eq!(dtype_block_info(GgufDtype::Q4_0), (32, 18));
        assert_eq!(dtype_block_info(GgufDtype::Q2_K), (256, 84));
    }

    #[test]
    fn test_size_bytes() {
        // F32 tensor of shape [4, 8] = 32 elements = 128 bytes
        assert_eq!(tensor_size_bytes(&[4, 8], GgufDtype::F32), 128);
        // Q4_0 tensor of shape [64]: 64/32 = 2 blocks * 18 = 36 bytes
        assert_eq!(tensor_size_bytes(&[64], GgufDtype::Q4_0), 36);
        // Single element shapes
        assert_eq!(tensor_size_bytes(&[1], GgufDtype::I8), 1);
    }

    #[test]
    fn test_metadata_value_types() {
        let dir = std::env::temp_dir().join("gguf_test_metatypes");
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("model.gguf");

        // Build a GGUF with several metadata value types
        let mut f = std::fs::File::create(&path).unwrap();
        let mut pos: u64 = 0;
        f.write_all(&0x4655_4747u32.to_le_bytes()).unwrap(); // magic
        pos += 4;
        f.write_all(&3u32.to_le_bytes()).unwrap();          // version
        pos += 4;
        f.write_all(&0u64.to_le_bytes()).unwrap();          // 0 tensors
        pos += 8;
        f.write_all(&3u64.to_le_bytes()).unwrap();          // 3 metadata KVs
        pos += 8;

        // 1) uint32 key "a" = 42
        let key_a = b"a";
        f.write_all(&(key_a.len() as u64).to_le_bytes()).unwrap();
        pos += 8;
        f.write_all(key_a).unwrap();
        pos += 1;
        f.write_all(&GGUF_TYPE_UINT32.to_le_bytes()).unwrap();
        pos += 4;
        f.write_all(&42u32.to_le_bytes()).unwrap();
        pos += 4;

        // 2) float32 key "b" = 3.14
        let key_b = b"b";
        f.write_all(&(key_b.len() as u64).to_le_bytes()).unwrap();
        pos += 8;
        f.write_all(key_b).unwrap();
        pos += 1;
        f.write_all(&GGUF_TYPE_FLOAT32.to_le_bytes()).unwrap();
        pos += 4;
        f.write_all(&3.14f32.to_le_bytes()).unwrap();
        pos += 4;

        // 3) bool key "c" = true
        let key_c = b"c";
        f.write_all(&(key_c.len() as u64).to_le_bytes()).unwrap();
        pos += 8;
        f.write_all(key_c).unwrap();
        pos += 1;
        f.write_all(&GGUF_TYPE_BOOL.to_le_bytes()).unwrap();
        pos += 4;
        f.write_all(&1u8.to_le_bytes()).unwrap();
        pos += 1;

        // Pad to 32 bytes
        for _ in 0..align_32(pos).saturating_sub(pos) {
            f.write_all(&[0u8]).unwrap();
        }
        // No tensor data needed (0 tensors)

        let model = parse_gguf(&path).unwrap();
        assert_eq!(model.metadata.len(), 3);
        match model.metadata.get("a") {
            Some(GgufMetadataValue::Uint32(v)) => assert_eq!(*v, 42),
            other => panic!("expected Uint32, got {other:?}"),
        }
        match model.metadata.get("b") {
            Some(GgufMetadataValue::Float32(v)) => assert!((*v - 3.14).abs() < 0.001),
            other => panic!("expected Float32, got {other:?}"),
        }
        match model.metadata.get("c") {
            Some(GgufMetadataValue::Bool(v)) => assert_eq!(*v, true),
            other => panic!("expected Bool, got {other:?}"),
        }

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_metadata_array_value() {
        let dir = std::env::temp_dir().join("gguf_test_array");
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("model.gguf");

        let mut f = std::fs::File::create(&path).unwrap();
        let mut pos: u64 = 0;
        f.write_all(&0x4655_4747u32.to_le_bytes()).unwrap();
        pos += 4;
        f.write_all(&3u32.to_le_bytes()).unwrap();
        pos += 4;
        f.write_all(&0u64.to_le_bytes()).unwrap();
        pos += 8;
        f.write_all(&1u64.to_le_bytes()).unwrap();
        pos += 8;

        // Array of 3 uint32 values: [10, 20, 30]
        let key = b"arr";
        f.write_all(&(key.len() as u64).to_le_bytes()).unwrap();
        pos += 8;
        f.write_all(key).unwrap();
        pos += 3;
        f.write_all(&GGUF_TYPE_ARRAY.to_le_bytes()).unwrap();
        pos += 4;
        f.write_all(&GGUF_TYPE_UINT32.to_le_bytes()).unwrap(); // element type
        pos += 4;
        f.write_all(&3u64.to_le_bytes()).unwrap();              // count
        pos += 8;
        f.write_all(&10u32.to_le_bytes()).unwrap();
        pos += 4;
        f.write_all(&20u32.to_le_bytes()).unwrap();
        pos += 4;
        f.write_all(&30u32.to_le_bytes()).unwrap();
        pos += 4;

        for _ in 0..align_32(pos).saturating_sub(pos) {
            f.write_all(&[0u8]).unwrap();
        }

        let model = parse_gguf(&path).unwrap();
        match model.metadata.get("arr") {
            Some(GgufMetadataValue::Array(items)) => {
                assert_eq!(items.len(), 3);
                assert!(matches!(items[0], GgufMetadataValue::Uint32(10)));
                assert!(matches!(items[1], GgufMetadataValue::Uint32(20)));
                assert!(matches!(items[2], GgufMetadataValue::Uint32(30)));
            }
            other => panic!("expected Array, got {other:?}"),
        }

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_multiple_tensors() {
        let dir = std::env::temp_dir().join("gguf_test_multi_tensor");
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("model.gguf");

        // Helper to write a tensor info entry
        fn write_tensor_info(
            f: &mut std::fs::File,
            name: &[u8],
            shape: &[u64],
            dtype: u32,
            offset: u64,
        ) {
            f.write_all(&(name.len() as u64).to_le_bytes()).unwrap();
            f.write_all(name).unwrap();
            f.write_all(&(shape.len() as u32).to_le_bytes()).unwrap();
            for &d in shape {
                f.write_all(&d.to_le_bytes()).unwrap();
            }
            f.write_all(&dtype.to_le_bytes()).unwrap();
            f.write_all(&offset.to_le_bytes()).unwrap();
        }

        let mut f = std::fs::File::create(&path).unwrap();
        let mut pos: u64 = 0;
        f.write_all(&0x4655_4747u32.to_le_bytes()).unwrap();
        pos += 4;
        f.write_all(&3u32.to_le_bytes()).unwrap();
        pos += 4;
        f.write_all(&2u64.to_le_bytes()).unwrap(); // 2 tensors
        pos += 8;
        f.write_all(&0u64.to_le_bytes()).unwrap(); // 0 metadata KVs
        pos += 8;

        // Tensor 0: "w1", shape [4,4], F32, offset 0
        write_tensor_info(&mut f, b"w1\0", &[4, 4], 0, 0);
        pos += 8 + 3 + 4 + 16 + 4 + 8; // name_len+name+ndims+dims*2+dtype+offset = 43
        // Tensor 1: "w2", shape [2,2], F16, offset 64 (4*4*4 = 64)
        write_tensor_info(&mut f, b"w2\0", &[2, 2], 1, 64);
        pos += 8 + 3 + 4 + 16 + 4 + 8; // same layout = 43

        // Align to 32
        let aligned = align_32(pos);
        for _ in 0..(aligned - pos) {
            f.write_all(&[0u8]).unwrap();
        }

        // Tensor data: 16 F32 values for w1
        for i in 0..16u32 {
            f.write_all(&i.to_le_bytes()).unwrap();
        }
        // 4 F16 values for w2
        let f16_vals: [u16; 4] = [0x3C00, 0x4000, 0x4400, 0x4800]; // 1,2,3,4 in fp16
        for v in &f16_vals {
            f.write_all(&v.to_le_bytes()).unwrap();
        }

        let model = parse_gguf(&path).unwrap();
        assert_eq!(model.tensors.len(), 2);
        assert_eq!(model.tensors[0].name, "w1");
        assert_eq!(model.tensors[0].dtype, GgufDtype::F32);
        assert_eq!(model.tensors[0].size_bytes, 4 * 4 * 4);
        assert_eq!(model.tensors[0].offset, 0);
        assert_eq!(model.tensors[1].name, "w2");
        assert_eq!(model.tensors[1].dtype, GgufDtype::F16);
        assert_eq!(model.tensors[1].size_bytes, 2 * 2 * 2);
        assert_eq!(model.tensors[1].offset, 64);

        let _ = std::fs::remove_dir_all(&dir);
    }
}
