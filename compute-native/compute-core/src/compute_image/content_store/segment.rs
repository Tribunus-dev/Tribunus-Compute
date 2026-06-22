//! Immutable segment and deterministic segment writer.

use crate::integration::ContentHash;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct SegmentIndex {
    pub entries: Vec<SegmentIndexEntry>,
    pub index_offset: u64,
    pub index_length: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SegmentIndexEntry {
    pub object_id: String,
    pub segment_offset: u64,
    pub payload_bytes: u64,
    pub aligned_bytes: u64,
    pub content_hash: ContentHash,
}

#[derive(Debug, Clone)]
pub struct SegmentWriteReceipt {
    pub object_id: String,
    pub segment_offset: u64,
    pub payload_bytes: u64,
    pub aligned_bytes: u64,
}

#[derive(Debug, Clone)]
pub struct DeterministicSegmentWriter {
    buffer: Vec<u8>,
    alignment: u64,
    entries: Vec<SegmentIndexEntry>,
}

impl DeterministicSegmentWriter {
    pub fn new(alignment: u64) -> Self {
        assert!(alignment > 0, "alignment must be > 0");
        Self {
            buffer: Vec::new(),
            alignment,
            entries: Vec::new(),
        }
    }

    pub fn append_object(
        &mut self,
        data: &[u8],
        object_id: &str,
        content_hash: ContentHash,
    ) -> SegmentWriteReceipt {
        let current_len = self.buffer.len() as u64;
        let misalignment = current_len % self.alignment;
        if misalignment != 0 {
            let pad = self.alignment - misalignment;
            self.buffer.extend(std::iter::repeat(0u8).take(pad as usize));
        }
        let offset = self.buffer.len() as u64;
        let payload_bytes = data.len() as u64;
        self.buffer.extend_from_slice(data);
        let aligned = self.buffer.len() as u64;
        let entry = SegmentIndexEntry {
            object_id: object_id.to_string(),
            segment_offset: offset,
            payload_bytes,
            aligned_bytes: aligned - offset,
            content_hash,
        };
        self.entries.push(entry);
        SegmentWriteReceipt {
            object_id: object_id.to_string(),
            segment_offset: offset,
            payload_bytes,
            aligned_bytes: aligned - offset,
        }
    }

    pub fn pad_to_alignment(&mut self) {
        let current_len = self.buffer.len() as u64;
        let misalignment = current_len % self.alignment;
        if misalignment != 0 {
            let pad = self.alignment - misalignment;
            self.buffer.extend(std::iter::repeat(0u8).take(pad as usize));
        }
    }

    pub fn finalize(self) -> (Vec<u8>, SegmentIndex) {
        let index = SegmentIndex {
            entries: self.entries,
            index_offset: 0,
            index_length: 0,
        };
        (self.buffer, index)
    }

    pub fn current_offset(&self) -> u64 {
        self.buffer.len() as u64
    }

    pub fn entry_count(&self) -> usize {
        self.entries.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::integration::ContentHash;

    #[test]
    fn test_empty_segment() {
        let writer = DeterministicSegmentWriter::new(4096);
        assert_eq!(writer.current_offset(), 0);
        assert_eq!(writer.entry_count(), 0);
    }

    #[test]
    fn test_single_object_no_padding() {
        let mut writer = DeterministicSegmentWriter::new(1);
        let receipt = writer.append_object(&[1, 2, 3], "obj1", ContentHash(1));
        assert_eq!(receipt.segment_offset, 0);
        assert_eq!(receipt.payload_bytes, 3);
    }

    #[test]
    fn test_two_objects_with_padding() {
        let mut writer = DeterministicSegmentWriter::new(16);
        let r1 = writer.append_object(&[1u8; 7], "obj1", ContentHash(1));
        assert_eq!(r1.segment_offset, 0);
        assert_eq!(r1.payload_bytes, 7);
        let r2 = writer.append_object(&[2u8; 8], "obj2", ContentHash(2));
        assert!(r2.segment_offset > 0);
        assert!(r2.segment_offset % 16 == 0);
    }

    #[test]
    fn test_pad_to_alignment_explicit() {
        let mut writer = DeterministicSegmentWriter::new(64);
        writer.append_object(&[1u8; 5], "o1", ContentHash(1));
        writer.pad_to_alignment();
        assert_eq!(writer.current_offset() % 64, 0);
    }

    #[test]
    fn test_deterministic_across_identical_inputs() {
        let mut a = DeterministicSegmentWriter::new(16);
        let mut b = DeterministicSegmentWriter::new(16);
        a.append_object(&[1, 2, 3], "x", ContentHash(1));
        b.append_object(&[1, 2, 3], "x", ContentHash(1));
        let (buf_a, _) = a.finalize();
        let (buf_b, _) = b.finalize();
        assert_eq!(buf_a, buf_b);
    }

    #[test]
    #[should_panic]
    fn test_zero_alignment_panics() {
        DeterministicSegmentWriter::new(0);
    }
}
