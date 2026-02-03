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
//! // Save to binary file (atomic with fsync)
//! store.save_binary_atomic("graph.ngdb")?;
//!
//! // Save with backup rotation
//! let config = BackupConfig::default();
//! store.save_binary_with_backups("graph.ngdb", &config)?;
//!
//! // Load from binary file
//! let loaded = GraphStore::load_binary("graph.ngdb")?;
//! ```
//!
//! # Non-blocking Saves
//!
//! For applications that can't block during I/O, use snapshots:
//!
//! ```ignore
//! // Take snapshot quickly under read lock
//! let snapshot = {
//!     let store = state.store.read().unwrap();
//!     store.snapshot()
//! };
//! // Lock released - save without blocking
//! snapshot.save_atomic("graph.ngdb")?;
//! ```

use crate::full_text_index::FullTextIndexMetadata;
use crate::wal::TransactionId;
use crate::{
    CsrMatrix, EdgeTypeIndex, GraphStore, LabelIndex, PropertyIndex, TimestampIndex,
    VersionedPropertyStore,
};
use neural_core::{Graph, NodeId};
use serde::{Deserialize, Serialize};
use std::fs::File;
use std::io::{BufReader, BufWriter, Read, Write};
use std::path::{Path, PathBuf};
use thiserror::Error;

/// Magic bytes identifying NeuralGraphDB files
const MAGIC: &[u8; 4] = b"NGDB";

/// Current file format version
const VERSION: u32 = 1;

// =============================================================================
// Error Types
// =============================================================================

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

    /// Atomic rename failed - original file should still be intact
    #[error("Atomic rename failed: {0}")]
    AtomicRenameFailed(std::io::Error),

    /// Backup rotation failed
    #[error("Backup rotation failed: {0}")]
    BackupRotationFailed(String),

    /// Fsync failed - data may not be durable
    #[error("Fsync failed - data may not be durable: {0}")]
    FsyncFailed(std::io::Error),

    /// Temp file cleanup failed (non-fatal)
    #[error("Temp file cleanup failed (non-fatal): {0}")]
    TempFileCleanupFailed(std::io::Error),
}

impl PersistenceError {
    /// Returns true if this error indicates potential data loss.
    pub fn is_critical(&self) -> bool {
        matches!(
            self,
            Self::Io(_) | Self::AtomicRenameFailed(_) | Self::FsyncFailed(_)
        )
    }

    /// Returns true if the operation can be safely retried.
    pub fn is_retryable(&self) -> bool {
        matches!(self, Self::Io(e) if e.kind() == std::io::ErrorKind::Interrupted)
    }
}

// =============================================================================
// Backup Configuration
// =============================================================================

/// Configuration for backup rotation during saves.
///
/// When saving with backups, the system maintains a chain of previous snapshots:
/// - `graph.ngdb` (current)
/// - `graph.backup.1.ngdb` (previous)
/// - `graph.backup.2.ngdb` (older)
/// - ... up to `max_backups`
#[derive(Debug, Clone)]
pub struct BackupConfig {
    /// Number of backup copies to retain (default: 3)
    pub max_backups: usize,
}

impl Default for BackupConfig {
    fn default() -> Self {
        Self { max_backups: 3 }
    }
}

impl BackupConfig {
    /// Creates a new backup configuration.
    pub fn new(max_backups: usize) -> Self {
        Self { max_backups }
    }

    /// Returns the path for backup number `n` (1-indexed).
    fn backup_path(&self, base: &Path, n: usize) -> PathBuf {
        let stem = base
            .file_stem()
            .unwrap_or_default()
            .to_string_lossy()
            .to_string();
        let ext = base.extension().unwrap_or_default().to_string_lossy();
        let backup_name = format!("{}.backup.{}.{}", stem, n, ext);
        base.with_file_name(backup_name)
    }
}

// =============================================================================
// GraphSnapshot - Non-blocking persistence
// =============================================================================

/// A serializable snapshot of the graph state.
///
/// Created quickly under a read lock, can be saved without blocking the main store.
/// This enables non-blocking persistence for high-availability applications.
///
/// # Example
///
/// ```ignore
/// // Take snapshot under read lock (fast - just cloning data)
/// let snapshot = {
///     let store = state.store.read().unwrap();
///     store.snapshot()
/// };
/// // Lock released here - I/O happens without blocking other operations
/// snapshot.save_atomic(&path)?;
/// ```
/// A serializable snapshot of the graph state.
///
/// This struct mirrors the serializable fields of GraphStore to ensure
/// binary compatibility. Files saved by GraphSnapshot can be loaded
/// by GraphStore::load_binary() and vice versa.
#[derive(Serialize, Deserialize)]
pub struct GraphSnapshot {
    /// Graph structure (adjacency) - CSR for outgoing edges
    graph: CsrMatrix,
    /// Node properties (MVCC-versioned)
    versioned_properties: VersionedPropertyStore,
    /// Node labels (MVCC-versioned)
    versioned_labels: VersionedPropertyStore,
    /// Inverted index for O(1) label lookups
    label_index: LabelIndex,
    /// Inverted index for O(1) property value lookups
    property_index: PropertyIndex,
    /// Index for O(1) edge type lookups
    edge_type_index: EdgeTypeIndex,
    /// Number of dynamically created nodes
    dynamic_node_count: usize,
    /// Dynamically created edges
    dynamic_edges: Vec<(NodeId, NodeId, Option<String>)>,
    /// Path to the DB file (always None in snapshots, for format compatibility)
    path: Option<PathBuf>,
    /// Current transaction ID counter
    current_tx_id: TransactionId,
    /// Timestamp index for time-travel queries
    timestamp_index: TimestampIndex,
    /// Full-text index metadata (for rebuild on load)
    full_text_metadata: std::collections::HashMap<String, FullTextIndexMetadata>,
}

impl GraphSnapshot {
    /// Creates a snapshot from a GraphStore.
    ///
    /// This clones all the serializable state from the store.
    /// The `path` field is always set to None in snapshots.
    pub fn from_store(store: &GraphStore) -> Self {
        Self {
            graph: store.graph().clone(),
            versioned_properties: store.versioned_properties().clone(),
            versioned_labels: store.versioned_labels().clone(),
            label_index: store.label_index().clone(),
            property_index: store.property_index().clone(),
            edge_type_index: store.edge_type_index().clone(),
            dynamic_node_count: store.dynamic_node_count(),
            dynamic_edges: store.dynamic_edges().cloned().collect(),
            path: None, // Path is runtime state, not persisted
            current_tx_id: store.current_tx_id(),
            timestamp_index: store.timestamp_index().clone(),
            full_text_metadata: store.full_text_metadata().clone(),
        }
    }

    /// Saves the snapshot to a file atomically.
    ///
    /// Uses write-tmp-rename pattern for crash safety:
    /// 1. Write to temporary file
    /// 2. Flush and fsync for durability
    /// 3. Atomic rename to target path
    pub fn save_atomic(&self, path: impl AsRef<Path>) -> Result<(), PersistenceError> {
        let path = path.as_ref();
        let tmp_path = path.with_extension("ngdb.tmp");

        // Ensure parent directory exists
        if let Some(parent) = path.parent() {
            if !parent.exists() {
                std::fs::create_dir_all(parent)?;
            }
        }

        // Write to temp file
        {
            let file = File::create(&tmp_path)?;
            let mut writer = BufWriter::new(file);

            // Write header
            writer.write_all(MAGIC)?;
            writer.write_all(&VERSION.to_le_bytes())?;

            // Write snapshot data
            bincode::serialize_into(&mut writer, &self)?;

            // Flush buffer
            writer.flush()?;

            // Fsync for durability - ensures data hits disk
            writer
                .get_ref()
                .sync_all()
                .map_err(PersistenceError::FsyncFailed)?;
        }

        // Atomic rename (POSIX guarantees atomicity)
        std::fs::rename(&tmp_path, path).map_err(PersistenceError::AtomicRenameFailed)?;

        // Fsync parent directory (ensures rename is durable on some filesystems)
        if let Some(parent) = path.parent() {
            if let Ok(dir) = File::open(parent) {
                let _ = dir.sync_all(); // Best effort, don't fail if this doesn't work
            }
        }

        Ok(())
    }

    /// Saves the snapshot with backup rotation.
    ///
    /// Before saving, rotates existing backups and moves current file to backup.
    pub fn save_with_backups(
        &self,
        path: impl AsRef<Path>,
        config: &BackupConfig,
    ) -> Result<(), PersistenceError> {
        let path = path.as_ref();

        // Rotate existing backups
        Self::rotate_backups(path, config)?;

        // Move current to .backup.1 (if exists)
        if path.exists() {
            let backup_1 = config.backup_path(path, 1);
            std::fs::rename(path, &backup_1).map_err(|e| {
                PersistenceError::BackupRotationFailed(format!(
                    "Failed to move current to backup: {}",
                    e
                ))
            })?;
        }

        // Write new file atomically
        self.save_atomic(path)
    }

    /// Rotates existing backup files.
    fn rotate_backups(path: &Path, config: &BackupConfig) -> Result<(), PersistenceError> {
        if config.max_backups == 0 {
            return Ok(());
        }

        // Delete oldest if at max
        let oldest = config.backup_path(path, config.max_backups);
        if oldest.exists() {
            std::fs::remove_file(&oldest).map_err(|e| {
                PersistenceError::BackupRotationFailed(format!(
                    "Failed to delete oldest backup: {}",
                    e
                ))
            })?;
        }

        // Rotate N -> N+1 (in reverse order to avoid overwrites)
        for i in (1..config.max_backups).rev() {
            let from = config.backup_path(path, i);
            let to = config.backup_path(path, i + 1);
            if from.exists() {
                std::fs::rename(&from, &to).map_err(|e| {
                    PersistenceError::BackupRotationFailed(format!(
                        "Failed to rotate backup {} -> {}: {}",
                        i,
                        i + 1,
                        e
                    ))
                })?;
            }
        }

        Ok(())
    }

    /// Returns the node count in this snapshot.
    pub fn node_count(&self) -> usize {
        self.graph.node_count() + self.dynamic_node_count
    }

    /// Returns the edge count in this snapshot.
    pub fn edge_count(&self) -> usize {
        self.graph.edge_count() + self.dynamic_edges.len()
    }
}

// =============================================================================
// GraphStore persistence methods
// =============================================================================

impl GraphStore {
    /// Saves the graph to a binary file atomically.
    ///
    /// Uses write-tmp-rename pattern for crash safety:
    /// 1. Write to temporary file (path.ngdb.tmp)
    /// 2. Flush and fsync for durability
    /// 3. Atomic rename to target path
    ///
    /// If the process crashes at any point:
    /// - Before rename: Original file intact, temp file ignored
    /// - After rename: New file complete and valid
    ///
    /// # Arguments
    /// * `path` - Path to the output file
    ///
    /// # Returns
    /// * `Ok(())` on success
    /// * `Err(PersistenceError)` on I/O or encoding error
    pub fn save_binary_atomic(&mut self, path: impl AsRef<Path>) -> Result<(), PersistenceError> {
        let path = path.as_ref();
        let tmp_path = path.with_extension("ngdb.tmp");

        // Set the path of the database file
        self.path = Some(path.to_path_buf());

        // Commit all full-text indexes before saving
        if let Err(e) = self.commit_fulltext_indexes() {
            eprintln!("Warning: Failed to commit full-text indexes: {}", e);
        }

        // Ensure parent directory exists
        if let Some(parent) = path.parent() {
            if !parent.exists() {
                std::fs::create_dir_all(parent)?;
            }
        }

        // Write to temp file
        {
            let file = File::create(&tmp_path)?;
            let mut writer = BufWriter::new(file);

            // Write header
            writer.write_all(MAGIC)?;
            writer.write_all(&VERSION.to_le_bytes())?;

            // Write graph data using bincode
            bincode::serialize_into(&mut writer, &self)?;

            // Flush buffer
            writer.flush()?;

            // Fsync for durability
            writer
                .get_ref()
                .sync_all()
                .map_err(PersistenceError::FsyncFailed)?;
        }

        // Atomic rename
        std::fs::rename(&tmp_path, path).map_err(PersistenceError::AtomicRenameFailed)?;

        // Fsync parent directory
        if let Some(parent) = path.parent() {
            if let Ok(dir) = File::open(parent) {
                let _ = dir.sync_all();
            }
        }

        // Truncate WAL after successful save
        self.truncate_wal()?;

        Ok(())
    }

    /// Saves the graph with backup rotation.
    ///
    /// Before saving:
    /// 1. Rotates existing backups (N -> N+1)
    /// 2. Moves current file to .backup.1
    /// 3. Writes new file atomically
    ///
    /// This ensures you always have N previous versions to recover from.
    ///
    /// # Arguments
    /// * `path` - Path to the output file
    /// * `config` - Backup rotation configuration
    pub fn save_binary_with_backups(
        &mut self,
        path: impl AsRef<Path>,
        config: &BackupConfig,
    ) -> Result<(), PersistenceError> {
        let path = path.as_ref();

        // Rotate existing backups
        GraphSnapshot::rotate_backups(path, config)?;

        // Move current to .backup.1 (if exists)
        if path.exists() {
            let backup_1 = config.backup_path(path, 1);
            std::fs::rename(path, &backup_1).map_err(|e| {
                PersistenceError::BackupRotationFailed(format!(
                    "Failed to move current to backup: {}",
                    e
                ))
            })?;
        }

        // Write new file atomically
        self.save_binary_atomic(path)
    }

    /// Saves the graph to a binary file (legacy method).
    ///
    /// **Note**: Prefer `save_binary_atomic()` for crash safety.
    /// This method exists for backwards compatibility.
    ///
    /// # Arguments
    /// * `path` - Path to the output file (will be created/overwritten)
    pub fn save_binary(&mut self, path: impl AsRef<Path>) -> Result<(), PersistenceError> {
        // Delegate to atomic version for safety
        self.save_binary_atomic(path)
    }

    /// Truncates the WAL file after a successful save.
    fn truncate_wal(&mut self) -> Result<(), PersistenceError> {
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
    /// Full-text indexes are rebuilt automatically from stored metadata.
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

        // After loading, set the path and initialize WAL writer
        store.path = Some(path.as_ref().to_path_buf());
        store.wal = Some(crate::wal::WalWriter::new(
            path.as_ref().with_extension("wal"),
        )?);

        // Rebuild skipped fields (reverse graph, transaction manager, full-text indexes)
        store.post_load_init();

        Ok(store)
    }

    /// Loads a graph from a snapshot file.
    ///
    /// This can load files saved by either `GraphStore::save_binary*()` or
    /// `GraphSnapshot::save*()` methods.
    pub fn load_from_snapshot(path: impl AsRef<Path>) -> Result<Self, PersistenceError> {
        // Snapshots use the same format, so just delegate
        Self::load_binary(path)
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use neural_core::{Graph, PropertyValue};
    use tempfile::{tempdir, NamedTempFile};

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
        let mut store = GraphStore::new_in_memory();

        let node_id = store
            .create_node(
                Some("Person"),
                [
                    ("name".to_string(), PropertyValue::from("Alice")),
                    ("age".to_string(), PropertyValue::from(30i64)),
                ]
                .into_iter()
                .collect::<std::collections::HashMap<_, _>>(),
                None,
            )
            .unwrap();

        let file = NamedTempFile::new().unwrap();
        store.save_binary(file.path()).unwrap();
        let loaded = GraphStore::load_binary(file.path()).unwrap();

        assert_eq!(store.node_count(), loaded.node_count());
        assert_eq!(store.get_label(node_id), loaded.get_label(node_id));
        assert_eq!(
            store.get_property(node_id, "name"),
            loaded.get_property(node_id, "name")
        );
    }

    #[test]
    fn test_binary_roundtrip_with_edges() {
        let mut store = GraphStore::builder()
            .add_node(0u64, [("name", "Alice")])
            .add_node(1u64, [("name", "Bob")])
            .add_edge(0u64, 1u64)
            .build();

        store
            .create_edge(NodeId::new(0), NodeId::new(1), Some("KNOWS"), None)
            .unwrap();

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
        data.extend_from_slice(&999u32.to_le_bytes());
        std::fs::write(file.path(), &data).unwrap();

        let result = GraphStore::load_binary(file.path());
        assert!(matches!(
            result,
            Err(PersistenceError::UnsupportedVersion(999))
        ));
    }

    #[test]
    fn test_atomic_save_creates_no_temp_file_on_success() {
        let mut store = GraphStore::new_in_memory();
        let dir = tempdir().unwrap();
        let path = dir.path().join("test.ngdb");

        store.save_binary_atomic(&path).unwrap();

        // Main file should exist
        assert!(path.exists());

        // Temp file should not exist
        let tmp_path = path.with_extension("ngdb.tmp");
        assert!(!tmp_path.exists());
    }

    #[test]
    fn test_backup_rotation() {
        let mut store = GraphStore::new_in_memory();
        let dir = tempdir().unwrap();
        let path = dir.path().join("test.ngdb");
        let config = BackupConfig::new(3);

        // Save 4 times to test rotation
        for i in 0..4 {
            store
                .create_node(
                    Some("Node"),
                    std::collections::HashMap::from([
                        ("version".to_string(), PropertyValue::from(i as i64))
                    ]),
                    None,
                )
                .unwrap();
            store.save_binary_with_backups(&path, &config).unwrap();
        }

        // Current file should exist
        assert!(path.exists());

        // Backups 1, 2, 3 should exist
        assert!(config.backup_path(&path, 1).exists());
        assert!(config.backup_path(&path, 2).exists());
        assert!(config.backup_path(&path, 3).exists());

        // Backup 4 should NOT exist (max is 3)
        assert!(!config.backup_path(&path, 4).exists());
    }

    #[test]
    fn test_snapshot_save_and_load() {
        let mut store = GraphStore::new_in_memory();
        store
            .create_node(
                Some("Person"),
                std::collections::HashMap::from([
                    ("name".to_string(), PropertyValue::from("Alice"))
                ]),
                None,
            )
            .unwrap();

        // Create snapshot
        let snapshot = store.snapshot();
        assert_eq!(snapshot.node_count(), store.node_count());

        // Save snapshot
        let file = NamedTempFile::new().unwrap();
        snapshot.save_atomic(file.path()).unwrap();

        // Load and verify
        let loaded = GraphStore::load_binary(file.path()).unwrap();
        assert_eq!(loaded.node_count(), store.node_count());
    }

    #[test]
    fn test_snapshot_with_backups() {
        let store = GraphStore::new_in_memory();
        let snapshot = store.snapshot();

        let dir = tempdir().unwrap();
        let path = dir.path().join("test.ngdb");
        let config = BackupConfig::new(2);

        // Save twice
        snapshot.save_with_backups(&path, &config).unwrap();
        snapshot.save_with_backups(&path, &config).unwrap();

        // Should have current + backup.1
        assert!(path.exists());
        assert!(config.backup_path(&path, 1).exists());
    }

    #[test]
    fn test_error_is_critical() {
        let io_err = PersistenceError::Io(std::io::Error::new(
            std::io::ErrorKind::Other,
            "test",
        ));
        assert!(io_err.is_critical());

        let magic_err = PersistenceError::InvalidMagic;
        assert!(!magic_err.is_critical());
    }

    #[test]
    fn test_error_is_retryable() {
        let interrupted = PersistenceError::Io(std::io::Error::new(
            std::io::ErrorKind::Interrupted,
            "interrupted",
        ));
        assert!(interrupted.is_retryable());

        let other = PersistenceError::Io(std::io::Error::new(
            std::io::ErrorKind::Other,
            "other",
        ));
        assert!(!other.is_retryable());
    }

    #[test]
    fn test_backup_config_paths() {
        let config = BackupConfig::new(3);
        let base = Path::new("/data/graph.ngdb");

        assert_eq!(
            config.backup_path(base, 1),
            PathBuf::from("/data/graph.backup.1.ngdb")
        );
        assert_eq!(
            config.backup_path(base, 2),
            PathBuf::from("/data/graph.backup.2.ngdb")
        );
    }

    #[test]
    fn test_binary_serialization_with_vectors() {
        let mut store = GraphStore::new_in_memory();
        for i in 0..10u64 {
            let vector: Vec<f32> = (0..128).map(|j| (i * 128 + j) as f32 / 1000.0).collect();
            store
                .create_node(
                    Some("Embedding"),
                    [
                        (format!("id_{}", i), PropertyValue::from(i as i64)),
                        ("vector".to_string(), PropertyValue::Vector(vector)),
                    ]
                    .into_iter()
                    .collect::<std::collections::HashMap<_, _>>(),
                    None,
                )
                .unwrap();
        }

        let file = NamedTempFile::new().unwrap();
        store.save_binary(file.path()).unwrap();
        let loaded = GraphStore::load_binary(file.path()).unwrap();

        assert_eq!(store.node_count(), loaded.node_count());

        let node = NodeId::new(10); // First dynamic node (CSR has 0 nodes)
        let original = store.get_property(node, "vector");
        let restored = loaded.get_property(node, "vector");
        assert_eq!(original, restored);
    }

    #[test]
    fn test_label_index_persistence() {
        // Create a store and add a labeled node
        let mut store = GraphStore::new_in_memory();

        let node_id = store
            .create_node(
                Some("TestLabel"),
                [("name".to_string(), PropertyValue::from("TestNode"))]
                    .into_iter()
                    .collect::<std::collections::HashMap<_, _>>(),
                None,
            )
            .unwrap();

        // Verify label index works before save
        let nodes_before: Vec<_> = store.nodes_with_label("TestLabel").collect();
        assert_eq!(nodes_before.len(), 1, "Label index should have 1 node before save");
        assert_eq!(nodes_before[0], node_id);

        // Save and reload
        let file = NamedTempFile::new().unwrap();
        store.save_binary(file.path()).unwrap();
        let loaded = GraphStore::load_binary(file.path()).unwrap();

        // Verify label can be retrieved (uses versioned_labels)
        assert_eq!(loaded.get_label(node_id), Some("TestLabel"));

        // Verify label index works after load (this is the bug)
        let nodes_after: Vec<_> = loaded.nodes_with_label("TestLabel").collect();
        assert_eq!(
            nodes_after.len(), 1,
            "Label index should have 1 node after load (currently broken - not rebuilt)"
        );
        assert_eq!(nodes_after[0], node_id);
    }

    #[test]
    fn test_label_index_with_wal_recovery() {
        // Test that label index works correctly when loading from binary + WAL replay
        let dir = tempdir().unwrap();
        let db_path = dir.path().join("test.ngdb");

        // Step 1: Create a store with WAL enabled
        {
            let mut store = GraphStore::new(&db_path).unwrap();

            // Create a node - this goes to WAL and memory
            let _node_id = store
                .create_node(
                    Some("WalLabel"),
                    [("name".to_string(), PropertyValue::from("WalNode"))]
                        .into_iter()
                        .collect::<std::collections::HashMap<_, _>>(),
                    None,
                )
                .unwrap();

            // Save to persist to binary (this should also truncate WAL)
            store.save_binary(&db_path).unwrap();
        }

        // Step 2: Load from binary and verify label index
        {
            let loaded = GraphStore::load_binary(&db_path).unwrap();

            // Verify label can be retrieved
            let node_id = NodeId::new(0);
            assert_eq!(loaded.get_label(node_id), Some("WalLabel"));

            // Verify label index works
            let nodes: Vec<_> = loaded.nodes_with_label("WalLabel").collect();
            assert_eq!(
                nodes.len(), 1,
                "Label index should work after load_binary"
            );
        }
    }
}
