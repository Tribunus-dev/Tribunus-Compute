//! Program serialization — JSON and optional MessagePack round-trip.
//!
//! Provides [`ProgramSerializer`] for serializing and deserializing
//! [`SerializedPhaseProgram`] values.  The [`ProgramFormat`] enum
//! selects the encoding format — JSON is always available; MessagePack
//! requires the `msgpack` Cargo feature.

use crate::compute_image::program::phase_program::SerializedPhaseProgram;

/// Serialization format for phase programs.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ProgramFormat {
    /// JSON format (always available).
    Json,
    /// MessagePack format (requires `msgpack` feature).
    #[cfg(feature = "msgpack")]
    MessagePack,
}

/// Stateless program serializer for `SerializedPhaseProgram`.
///
/// Supports round-trip serialization and deserialization in JSON or
/// MessagePack, and pretty-printed JSON output.
pub struct ProgramSerializer;

impl ProgramSerializer {
    /// Create a new `ProgramSerializer`.
    pub fn new() -> Self {
        Self
    }

    /// Serialize a program into bytes in the given format.
    ///
    /// Returns `Err` with a description if serialization fails.
    pub fn serialize(
        program: &SerializedPhaseProgram,
        format: ProgramFormat,
    ) -> Result<Vec<u8>, String> {
        match format {
            ProgramFormat::Json => serde_json::to_vec(program)
                .map_err(|e| format!("serialize program (json): {}", e)),
            #[cfg(feature = "msgpack")]
            ProgramFormat::MessagePack => rmp_serde::to_vec(program)
                .map_err(|e| format!("serialize program (msgpack): {}", e)),
        }
    }

    /// Deserialize a program from bytes in the given format.
    ///
    /// Returns `Err` with a description if deserialization fails.
    pub fn deserialize(
        data: &[u8],
        format: ProgramFormat,
    ) -> Result<SerializedPhaseProgram, String> {
        match format {
            ProgramFormat::Json => serde_json::from_slice(data)
                .map_err(|e| format!("deserialize program (json): {}", e)),
            #[cfg(feature = "msgpack")]
            ProgramFormat::MessagePack => rmp_serde::from_slice(data)
                .map_err(|e| format!("deserialize program (msgpack): {}", e)),
        }
    }

    /// Pretty-print a program as a JSON string.
    ///
    /// Useful for debug output, logging, and human-readable snapshots.
    pub fn serialize_to_string(program: &SerializedPhaseProgram) -> Result<String, String> {
        serde_json::to_string_pretty(program)
            .map_err(|e| format!("serialize program (pretty json): {}", e))
    }
}

impl Default for ProgramSerializer {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::compute_image::execution_shape::ExecutionShapeClass;

    /// Build a minimal `SerializedPhaseProgram` for use in serialization tests.
    fn sample_program() -> SerializedPhaseProgram {
        SerializedPhaseProgram::new(
            1,
            "phase_matmul_0".to_string(),
            ExecutionShapeClass::Decode1,
            vec![0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07],
        )
    }

    #[test]
    fn test_serialization_roundtrip_json() {
        let program = sample_program();

        let bytes = ProgramSerializer::serialize(&program, ProgramFormat::Json)
            .expect("serialize to json");
        let deserialized =
            ProgramSerializer::deserialize(&bytes, ProgramFormat::Json)
                .expect("deserialize from json");

        assert_eq!(
            program, deserialized,
            "JSON round-trip must preserve all fields"
        );
    }

    #[test]
    fn test_serialization_determinism() {
        let program = sample_program();

        let bytes_a = ProgramSerializer::serialize(&program, ProgramFormat::Json)
            .expect("serialize to json (a)");
        let bytes_b = ProgramSerializer::serialize(&program, ProgramFormat::Json)
            .expect("serialize to json (b)");

        assert_eq!(
            bytes_a, bytes_b,
            "serialization must be deterministic for the same struct"
        );
    }

    #[test]
    fn test_serialization_error_handling() {
        let corrupt = b"{\"schema_version\":1,";
        let result = ProgramSerializer::deserialize(corrupt, ProgramFormat::Json);
        assert!(
            result.is_err(),
            "corrupt input must produce an error, got {:?}",
            result
        );
    }

    #[test]
    fn test_serialization_pretty_json() {
        let program = sample_program();

        let pretty = ProgramSerializer::serialize_to_string(&program)
            .expect("pretty-print json");

        assert!(
            pretty.contains('\n'),
            "pretty-printed JSON must contain newlines"
        );

        let deserialized: SerializedPhaseProgram =
            serde_json::from_str(&pretty).expect("deserialize pretty json");
        assert_eq!(
            program, deserialized,
            "pretty-printed JSON must deserialize back to the original"
        );
    }

    #[test]
    fn test_serialization_roundtrip_empty_payload() {
        let empty_program = SerializedPhaseProgram::new(
            0,
            "phase_empty".to_string(),
            ExecutionShapeClass::Decode1,
            vec![],
        );

        let bytes = ProgramSerializer::serialize(&empty_program, ProgramFormat::Json)
            .expect("serialize empty program");
        let deserialized =
            ProgramSerializer::deserialize(&bytes, ProgramFormat::Json)
                .expect("deserialize empty program");

        assert_eq!(empty_program, deserialized);
        assert!(deserialized.program_bytes.is_empty());
    }

    #[test]
    fn test_serialization_maximal_variant() {
        let program = SerializedPhaseProgram::new(
            2,
            "phase_diffusion_attention".to_string(),
            ExecutionShapeClass::DiffusionForward {
                max_canvas_tokens: 1024,
            },
            (0..256).map(|i| (i % 256) as u8).collect(),
        );

        let bytes = ProgramSerializer::serialize(&program, ProgramFormat::Json)
            .expect("serialize diffusion program");
        let deserialized =
            ProgramSerializer::deserialize(&bytes, ProgramFormat::Json)
                .expect("deserialize diffusion program");

        assert_eq!(program, deserialized);
    }

    #[cfg(feature = "msgpack")]
    #[test]
    fn test_serialization_roundtrip_msgpack() {
        let program = sample_program();

        let bytes = ProgramSerializer::serialize(&program, ProgramFormat::MessagePack)
            .expect("serialize to msgpack");
        let deserialized =
            ProgramSerializer::deserialize(&bytes, ProgramFormat::MessagePack)
                .expect("deserialize from msgpack");

        assert_eq!(program, deserialized, "msgpack round-trip must preserve all fields");
    }

    #[cfg(feature = "msgpack")]
    #[test]
    fn test_serialization_msgpack_more_compact_than_json() {
        let program = SerializedPhaseProgram::new(
            42,
            "phase_large".to_string(),
            ExecutionShapeClass::DecodeBatch { max_batch: 32 },
            (0..64).map(|i| i as u8).collect(),
        );

        let json_bytes = ProgramSerializer::serialize(&program, ProgramFormat::Json)
            .expect("serialize to json");
        let msgpack_bytes =
            ProgramSerializer::serialize(&program, ProgramFormat::MessagePack)
                .expect("serialize to msgpack");

        assert!(
            msgpack_bytes.len() < json_bytes.len(),
            "msgpack output ({}) should be smaller than json output ({})",
            msgpack_bytes.len(),
            json_bytes.len(),
        );
    }
}
