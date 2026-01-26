//! Binary persistence for GraphStore.
//!
//! This module provides efficient binary serialization for graph data using bincode.
//! Files are prefixed with magic bytes and version for forward compatibility.
//!
//! # File Format
//!
//! ```text
//! +------------------+
//! | Magic: "NGDB"    | 4 bytes
//! +------------------+
//! | Version: u32     | 4 bytes (little-endian)
//! +------------------+
//! | GraphStore data  | variable (bincode-encoded)
//! +------------------+
//! ```
//!
//! # Example
//!
//! ```ignore
//! use neural_storage::GraphStore;
//!
//! let store = GraphStore::builder()
//!     .add_node(0u64, [("name", "Alice")])
//!     .build();
//!
//! // Save to binary file
//! store.save_binary("graph.ngdb")?;
//!
//! // Load from binary file
//! let loaded = GraphStore::load_binary("graph.ngdb")?;
//! ```

use crate::GraphStore;
use std::fs::File;
use std::io::{BufReader, BufWriter, Read, Write};
use std::path::Path;
use thiserror::Error;

/// Magic bytes identifying NeuralGraphDB files
const MAGIC: &[u8; 4] = b"NGDB";

/// Current file format version
const VERSION: u32 = 1;

/// Errors that can occur during persistence operations.
#[derive(Debug, Error)]
pub enum PersistenceError {
    /// I/O error during file operations
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),

    /// Error during bincode serialization/deserialization
    #[error("Serialization error: {0}")]
    Bincode(#[from] bincode::Error),

    /// File does not have valid NGDB magic bytes
    #[error("Invalid file format: expected NGDB magic bytes")]
    InvalidMagic,

    /// Unsupported file version by this library version
    #[error("Unsupported file version: {0} (current: {VERSION})")]
    UnsupportedVersion(u32),

    /// WAL error during recovery or initialization
    #[error("WAL error: {0}")]
    Wal(#[from] crate::wal::WalError),
}

impl GraphStore {
    /// Saves the graph to a binary file.
    ///
    /// The file format includes a header with magic bytes and version
    /// followed by the bincode-encoded graph data.
    ///
    /// # Arguments
    /// * `path` - Path to the output file (will be created/overwritten)
    ///
    /// # Returns
    /// * `Ok(())` on success
    /// * `Err(PersistenceError)` on I/O or encoding error
    ///
    /// # Example
    /// ```ignore
    /// store.save_binary("graph.ngdb")?;
    /// ```
    pub fn save_binary(&mut self, path: impl AsRef<Path>) -> Result<(), PersistenceError> {
        // Set the path of the database file
        self.path = Some(path.as_ref().to_path_buf());

        let file = File::create(&path)?;
        let mut writer = BufWriter::new(file);

        // Write header
        writer.write_all(MAGIC)?;
        writer.write_all(&VERSION.to_le_bytes())?;

        // Write graph data using bincode
        bincode::serialize_into(&mut writer, &self)?;

        writer.flush()?;

        // Truncate the WAL file after a successful save, as the main DB file
        // is now the source of truth.
        if let Some(db_path) = self.path.as_ref() {
            let wal_path = db_path.with_extension("wal");
            if wal_path.exists() {
                std::fs::File::create(&wal_path)?.set_len(0)?;
            }
        }

        Ok(())
    }

    /// Loads a graph from a binary file.
    ///
    /// Verifies the file header contains valid magic bytes and a supported
    /// version before decoding the graph data.
    ///
    /// # Arguments
    /// * `path` - Path to the input file
    ///
    /// # Returns
    /// * `Ok(GraphStore)` on success
    /// * `Err(PersistenceError)` on I/O, decoding, or format error
    ///
    /// # Note
    /// The vector index is NOT persisted and will be `None` after loading.
    /// You must rebuild it by calling `init_vector_index()` and `add_vector()`
    /// for each node if vector search is needed.
    ///
    /// # Example
    /// ```ignore
    /// let store = GraphStore::load_binary("graph.ngdb")?;
    /// ```
    pub fn load_binary(path: impl AsRef<Path>) -> Result<Self, PersistenceError> {
        let file = File::open(path.as_ref())?;
        let mut reader = BufReader::new(file);

        // Verify magic bytes
        let mut magic = [0u8; 4];
        reader.read_exact(&mut magic)?;
        if &magic != MAGIC {
            return Err(PersistenceError::InvalidMagic);
        }

        // Verify version
        let mut version_bytes = [0u8; 4];
        reader.read_exact(&mut version_bytes)?;
        let version = u32::from_le_bytes(version_bytes);
        if version != VERSION {
            return Err(PersistenceError::UnsupportedVersion(version));
        }

        // Read graph data using bincode
        let mut store: GraphStore = bincode::deserialize_from(&mut reader)?;

        // After loading the main DB file, we need to manually set the path
        // and initialize the WAL writer, as these fields are skipped during serialization.
        store.path = Some(path.as_ref().to_path_buf());
        store.wal = Some(crate::wal::WalWriter::new(path.as_ref().with_extension("wal"))?);

        Ok(store)
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::CsrMatrix;
    use neural_core::{Graph, NodeId, PropertyValue};
    use tempfile::NamedTempFile;

    #[test]
    fn test_binary_roundtrip_empty() {
        let mut store = GraphStore::new_in_memory();

        let file = NamedTempFile::new().unwrap();
        store.save_binary(file.path()).unwrap();
        let loaded = GraphStore::load_binary(file.path()).unwrap();

        assert_eq!(store.node_count(), loaded.node_count());
        assert_eq!(store.edge_count(), loaded.edge_count());
    }

    #[test]
    fn test_binary_roundtrip_with_data() {
        // Create graph with nodes, edges, properties
        let mut store = GraphStore::new_in_memory();

        // Create a node with label and properties
        let node_id = store.create_node(
            Some("Person"),
            [
                ("name".to_string(), PropertyValue::from("Alice")),
                ("age".to_string(), PropertyValue::from(30i64)),
            ]
            .into_iter()
            .collect::<std::collections::HashMap<_, _>>(),
            None,
        ).unwrap();

        // Save and load
        let file = NamedTempFile::new().unwrap();
        store.save_binary(file.path()).unwrap();
        let loaded = GraphStore::load_binary(file.path()).unwrap();

        // Verify integrity
        assert_eq!(store.node_count(), loaded.node_count());
        assert_eq!(store.get_label(node_id), loaded.get_label(node_id));
        assert_eq!(
            store.get_property(node_id, "name"),
            loaded.get_property(node_id, "name")
        );
        assert_eq!(
            store.get_property(node_id, "age"),
            loaded.get_property(node_id, "age")
        );
    }

    #[test]
    fn test_binary_roundtrip_with_edges() {
        let mut store = GraphStore::builder()
            .add_node(0u64, [("name", "Alice")])
            .add_node(1u64, [("name", "Bob")])
            .add_edge(0u64, 1u64)
            .build();

        // Add dynamic edge
        store.create_edge(NodeId::new(0), NodeId::new(1), Some("KNOWS"), None).unwrap();

        let file = NamedTempFile::new().unwrap();
        store.save_binary(file.path()).unwrap();
        let loaded = GraphStore::load_binary(file.path()).unwrap();

        assert_eq!(store.edge_count(), loaded.edge_count());
        assert_eq!(
            store.dynamic_edges().count(),
            loaded.dynamic_edges().count()
        );
    }

    #[test]
    fn test_binary_roundtrip_preserves_indices() {
        let mut store = GraphStore::new_in_memory();

        let file = NamedTempFile::new().unwrap();
        store.save_binary(file.path()).unwrap();
        let loaded = GraphStore::load_binary(file.path()).unwrap();

        // Check node count in properties preserved
        assert_eq!(
            store.versioned_properties().node_count(),
            loaded.versioned_properties().node_count()
        );
    }

    #[test]
    fn test_invalid_magic_bytes() {
        let file = NamedTempFile::new().unwrap();
        std::fs::write(file.path(), b"XXXX").unwrap();

        let result = GraphStore::load_binary(file.path());
        assert!(matches!(result, Err(PersistenceError::InvalidMagic)));
    }

    #[test]
    fn test_unsupported_version() {
        let file = NamedTempFile::new().unwrap();
        let mut data = Vec::new();
        data.extend_from_slice(b"NGDB");
        data.extend_from_slice(&999u32.to_le_bytes()); // Future version
        std::fs::write(file.path(), &data).unwrap();

        let result = GraphStore::load_binary(file.path());
        assert!(matches!(
            result,
            Err(PersistenceError::UnsupportedVersion(999))
        ));
    }

    #[test]
    fn test_binary_serialization_performance() {
        // Create a graph with vector data
        let mut store = GraphStore::new_in_memory();
        for i in 0..50u64 {
            // Generate a 128-dimensional vector (common embedding size)
            let vector: Vec<f32> = (0..128).map(|j| (i * 128 + j) as f32 / 1000.0).collect();
            store.create_node(
                Some("Embedding"),
                [
                    (format!("id_{}", i), PropertyValue::from(i as i64)),
                    ("vector".to_string(), PropertyValue::Vector(vector)),
                ]
                .into_iter()
                .collect::<std::collections::HashMap<_, _>>(),
                None,
            ).unwrap();
        }

        // Verify binary roundtrip works correctly with vector data
        let file = NamedTempFile::new().unwrap();
        store.save_binary(file.path()).unwrap();
        let loaded = GraphStore::load_binary(file.path()).unwrap();

        // Check data integrity
        assert_eq!(store.node_count(), loaded.node_count());

        // Verify a vector property survived the roundtrip
        let node = NodeId::new(50); // First dynamic node
        let original = store.get_property(node, "vector");
        let restored = loaded.get_property(node, "vector");
        assert_eq!(original, restored);
    }
}
