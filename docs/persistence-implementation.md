# Persistence Implementation

This document describes the automatic persistence implementation for NeuralGraphDB, split between the storage library and application layers.

**Version:** 2.0 (Sprint 66 - Database Hardening)

---

## Overview

NeuralGraphDB provides comprehensive persistence with:

- **Write-Ahead Log (WAL)** - Every mutation logged with CRC32 checksums
- **Binary Snapshots** - Full graph snapshots with SHA256 checksums
- **Delta Checkpoints** - Incremental persistence for efficient saves
- **Automatic Recovery** - Rebuild indexes from authoritative sources on load
- **Post-Load Validation** - Verify data integrity after loading

---

## Phase 1: Library Layer (neural-storage)

### Files Modified

- `crates/neural-storage/src/persistence.rs` - Core persistence logic
- `crates/neural-storage/src/graph_store.rs` - Snapshot and getter methods
- `crates/neural-storage/src/lib.rs` - Public exports

---

### 1.1 Enhanced Error Types

**Location**: `persistence.rs`

```rust
#[derive(Debug, Error)]
pub enum PersistenceError {
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),

    #[error("Serialization error: {0}")]
    Bincode(#[from] bincode::Error),

    #[error("Invalid file format: expected NGDB magic bytes")]
    InvalidMagic,

    #[error("Unsupported file version: {0} (current: {VERSION})")]
    UnsupportedVersion(u32),

    #[error("WAL error: {0}")]
    Wal(#[from] crate::wal::WalError),

    #[error("Atomic rename failed: {0}")]
    AtomicRenameFailed(std::io::Error),

    #[error("Backup rotation failed: {0}")]
    BackupRotationFailed(String),

    #[error("Fsync failed - data may not be durable: {0}")]
    FsyncFailed(std::io::Error),

    #[error("Temp file cleanup failed (non-fatal): {0}")]
    TempFileCleanupFailed(std::io::Error),

    #[error("File checksum mismatch - data may be corrupted")]
    ChecksumMismatch,  // NEW in Sprint 66

    #[error("Delta mismatch: expected base tx {expected}, found {found}")]
    DeltaMismatch { expected: u64, found: u64 },  // NEW in Sprint 66
}

impl PersistenceError {
    /// Returns true if this error indicates potential data loss.
    pub fn is_critical(&self) -> bool;

    /// Returns true if the operation can be safely retried.
    pub fn is_retryable(&self) -> bool;
}
```

---

### 1.2 Backup Configuration

**Location**: `persistence.rs:136-172`

```rust
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
    pub fn new(max_backups: usize) -> Self;
    fn backup_path(&self, base: &Path, n: usize) -> PathBuf;
}
```

**Backup file naming**: `graph.backup.1.ngdb`, `graph.backup.2.ngdb`, etc.

---

### 1.3 GraphSnapshot

**Location**: `persistence.rs:178-363`

```rust
#[derive(Serialize, Deserialize)]
pub struct GraphSnapshot {
    graph: CsrMatrix,
    versioned_properties: VersionedPropertyStore,
    versioned_labels: VersionedPropertyStore,
    label_index: LabelIndex,
    property_index: PropertyIndex,
    edge_type_index: EdgeTypeIndex,
    dynamic_node_count: usize,
    dynamic_edges: Vec<(NodeId, NodeId, Option<String>)>,
    path: Option<PathBuf>,  // Always None, for format compatibility
    current_tx_id: TransactionId,
    timestamp_index: TimestampIndex,
    full_text_metadata: HashMap<String, FullTextIndexMetadata>,
}

impl GraphSnapshot {
    /// Creates a snapshot from a GraphStore.
    pub fn from_store(store: &GraphStore) -> Self;

    /// Saves atomically using write-tmp-rename pattern.
    pub fn save_atomic(&self, path: impl AsRef<Path>) -> Result<(), PersistenceError>;

    /// Saves with backup rotation.
    pub fn save_with_backups(&self, path: impl AsRef<Path>, config: &BackupConfig) -> Result<(), PersistenceError>;

    pub fn node_count(&self) -> usize;
    pub fn edge_count(&self) -> usize;
}
```

---

### 1.4 Atomic Save Methods

**Location**: `persistence.rs:369-494`

```rust
impl GraphStore {
    /// Saves atomically with write-tmp-rename + fsync.
    pub fn save_binary_atomic(&mut self, path: impl AsRef<Path>) -> Result<(), PersistenceError> {
        // 1. Write to path.ngdb.tmp
        // 2. Flush buffer
        // 3. fsync for durability
        // 4. Atomic rename tmp -> target
        // 5. fsync parent directory
        // 6. Truncate WAL
    }

    /// Saves with backup rotation.
    pub fn save_binary_with_backups(&mut self, path: impl AsRef<Path>, config: &BackupConfig) -> Result<(), PersistenceError> {
        // 1. Rotate existing backups (N -> N+1)
        // 2. Move current file to .backup.1
        // 3. Call save_binary_atomic()
    }

    /// Legacy method - now delegates to save_binary_atomic().
    pub fn save_binary(&mut self, path: impl AsRef<Path>) -> Result<(), PersistenceError>;
}
```

---

### 1.5 New GraphStore Methods

**Location**: `graph_store.rs:536-582`

```rust
impl GraphStore {
    /// Initializes fields skipped during serialization after loading.
    pub fn post_load_init(&mut self) {
        self.reverse_graph = CscMatrix::from_csr(&self.graph);
        self.transaction_manager = TransactionManager::new();
        self.rebuild_fulltext_indexes();
    }

    /// Creates a snapshot for non-blocking persistence.
    pub fn snapshot(&self) -> GraphSnapshot {
        GraphSnapshot::from_store(self)
    }

    // New getters for snapshot creation:
    pub fn versioned_labels(&self) -> &VersionedPropertyStore;
    pub fn dynamic_node_count(&self) -> usize;
    pub fn current_tx_id(&self) -> TransactionId;
    pub fn full_text_metadata(&self) -> &HashMap<String, FullTextIndexMetadata>;
}
```

---

### 1.6 Public Exports

**Location**: `lib.rs:48-49`

```rust
pub use persistence::{BackupConfig, GraphSnapshot, PersistenceError};
pub use graph_store::{EdgeTypeIndex, LabelIndex, PropertyIndex, TimestampIndex};
```

---

### 1.7 Tests Added

**Location**: `persistence.rs:593-853`

| Test | Description |
|------|-------------|
| `test_atomic_save_creates_no_temp_file_on_success` | Temp file cleaned up after save |
| `test_backup_rotation` | Backups rotate correctly, oldest deleted |
| `test_snapshot_save_and_load` | Snapshot can be saved and loaded as GraphStore |
| `test_snapshot_with_backups` | Snapshot backup rotation works |
| `test_error_is_critical` | Critical error classification |
| `test_error_is_retryable` | Retryable error classification |
| `test_backup_config_paths` | Backup path generation |

---

## Phase 2: Application Layer (neural-cli)

### Files Modified

- `crates/neural-cli/src/server.rs` - HTTP server with persistence

---

### 2.1 PersistenceConfig

**Location**: `server.rs:34-99`

```rust
#[derive(Debug, Clone)]
pub struct PersistenceConfig {
    pub db_path: PathBuf,
    pub save_interval_secs: u64,
    pub mutation_threshold: u64,
    pub backup_count: usize,
    pub shutdown_timeout_secs: u64,
}

impl PersistenceConfig {
    /// Loads from environment variables with defaults.
    pub fn from_env() -> Self;

    /// Creates a BackupConfig from this config.
    pub fn backup_config(&self) -> BackupConfig;
}
```

**Environment Variables**:

| Variable | Default | Description |
|----------|---------|-------------|
| `NGDB_PATH` | `data/graph.ngdb` | Database file path |
| `NGDB_SAVE_INTERVAL` | `60` | Seconds between periodic saves |
| `NGDB_SAVE_THRESHOLD` | `10` | Mutations before auto-save |
| `NGDB_BACKUP_COUNT` | `3` | Number of backups to retain |
| `NGDB_SHUTDOWN_TIMEOUT` | `30` | Graceful shutdown timeout |

---

### 2.2 AppState Changes

**Location**: `server.rs:101-112`

```rust
pub struct AppState {
    /// Async RwLock for non-blocking access
    pub store: Arc<tokio::sync::RwLock<GraphStore>>,
    pub vector_index: VectorIndex,
    pub paper_embeddings: HashMap<String, Vec<f32>>,
    pub papers: Vec<PaperInfo>,
    pub mutation_count: AtomicU64,
    pub config: PersistenceConfig,  // Replaced db_path: PathBuf
}
```

**Key change**: `std::sync::RwLock` → `tokio::sync::RwLock` for async compatibility.

---

### 2.3 Non-blocking Auto-Save

**Location**: `server.rs:516-554`

```rust
async fn maybe_save(state: &Arc<AppState>) {
    let threshold = state.config.mutation_threshold;
    let count = state.mutation_count.load(Ordering::SeqCst);

    if count >= threshold {
        // Atomic reset to prevent duplicate saves
        if state.mutation_count
            .compare_exchange(count, 0, Ordering::SeqCst, Ordering::SeqCst)
            .is_err()
        {
            return;
        }

        // Take snapshot under READ lock (fast)
        let snapshot = {
            let store = state.store.read().await;
            store.snapshot()
        };
        // Lock released - I/O without blocking

        let path = state.config.db_path.clone();
        let backup_config = state.config.backup_config();

        // I/O in spawn_blocking (doesn't block async runtime)
        tokio::task::spawn_blocking(move || {
            snapshot.save_with_backups(&path, &backup_config)
        });
    }
}
```

---

### 2.4 Non-blocking Periodic Save

**Location**: `server.rs:928-960`

```rust
tokio::spawn(async move {
    let mut interval = tokio::time::interval(save_interval);
    loop {
        interval.tick().await;
        let count = save_state.mutation_count.swap(0, Ordering::SeqCst);
        if count > 0 {
            // Snapshot under read lock
            let snapshot = {
                let store = save_state.store.read().await;
                store.snapshot()
            };

            // Blocking I/O in spawn_blocking
            let result = tokio::task::spawn_blocking(move || {
                snapshot.save_with_backups(&path, &backup_config)
            }).await;

            match result {
                Ok(Ok(())) => println!("Periodic save completed"),
                Ok(Err(e)) => eprintln!("Periodic save failed: {}", e),
                Err(e) => eprintln!("Task panicked: {}", e),
            }
        }
    }
});
```

---

### 2.5 Graceful Shutdown

**Location**: `server.rs:963-992`

```rust
let shutdown_signal = async move {
    tokio::signal::ctrl_c().await.ok();
    println!("\nShutdown signal received, draining connections...");

    // Final save with snapshot
    let snapshot = {
        let store = shutdown_state.store.read().await;
        store.snapshot()
    };

    tokio::task::spawn_blocking(move || {
        snapshot.save_with_backups(&path, &backup_config)
    }).await;
};

// Serve with graceful shutdown (drains connections first)
axum::serve(listener, app)
    .with_graceful_shutdown(shutdown_signal)
    .await?;
```

**Key improvement**: No more `std::process::exit(0)`. Axum drains in-flight requests before shutdown.

---

### 2.6 Memory Ordering Fix

All atomic operations changed from `Ordering::Relaxed` to `Ordering::SeqCst`:

| Location | Operation |
|----------|-----------|
| `maybe_save()` | `load`, `compare_exchange` |
| `handle_query()` | `fetch_add` |
| `handle_bulk_load()` | `fetch_add` |
| Periodic save | `swap` |

---

### 2.7 Handler Updates

All handlers updated to use async RwLock:

```rust
// Before (blocking)
let store = state.store.read().unwrap();

// After (async)
let store = state.store.read().await;
```

Affected handlers:
- `similar_papers()`
- `handle_schema()`
- `handle_query()`
- `handle_bulk_load()`

---

---

## Phase 3: Data Integrity (Sprint 66)

### 3.1 WAL Checksums (CRC32)

Every WAL entry now includes a CRC32 checksum for corruption detection.

**New WAL Entry Format:**

```
┌─────────────────────────────────┐
│ Length: u64 (8 bytes, LE)       │  ← Total length including checksum
├─────────────────────────────────┤
│ CRC32: u32 (4 bytes, LE)        │  ← NEW: Checksum of payload
├─────────────────────────────────┤
│ Payload: bincode LogEntry       │
│ (length - 4 bytes)              │
└─────────────────────────────────┘
```

**Backward Compatibility:** Old WAL files (without checksums) are detected by comparing payload length to what a checksum-aware entry would expect. Legacy entries are processed without verification.

**Error Handling:**

```rust
#[error("Checksum mismatch at offset {offset}: expected {expected:#x}, got {computed:#x}")]
ChecksumMismatch { offset: u64, expected: u32, computed: u32 }
```

### 3.2 Binary Snapshot Checksums (SHA256)

Snapshot files now include SHA256 checksums to detect file corruption.

**New File Format (VERSION 2):**

```
┌─────────────────────────────────┐
│ Magic: "NGDB" (4 bytes)         │
├─────────────────────────────────┤
│ Version: u32 (4 bytes, LE) = 2  │
├─────────────────────────────────┤
│ SHA256: [u8; 32] (32 bytes)     │  ← NEW: Checksum of data
├─────────────────────────────────┤
│ Data: bincode GraphStore        │
│ (variable length)               │
└─────────────────────────────────┘
```

**Backward Compatibility:** VERSION 1 files (without checksum) are still supported. The loader detects version and skips checksum verification for V1 files.

### 3.3 Post-Load Validation

After loading a snapshot, the system validates data integrity:

```rust
pub struct ValidationResult {
    pub is_valid: bool,
    pub errors: Vec<ValidationError>,
    pub warnings: Vec<String>,
}

pub enum ValidationError {
    LabelIndexMismatch { node_id: NodeId, expected: Option<String>, found: bool },
    PropertyIndexMismatch { node_id: NodeId, key: String },
    EdgeTypeIndexMismatch { edge_type: String, expected: usize, found: usize },
    CsrIntegrity(String),
}

impl GraphStore {
    /// Validates graph integrity after loading
    pub fn validate_post_load(&self) -> ValidationResult;
}
```

### 3.4 Index Rebuild on Load

All indexes are rebuilt from authoritative sources during `post_load_init()`:

```rust
impl GraphStore {
    pub fn post_load_init(&mut self) {
        self.reverse_graph = CscMatrix::from_csr(&self.graph);
        self.transaction_manager = TransactionManager::new();

        // Rebuild ALL indexes from authoritative sources
        self.rebuild_label_index();      // From versioned_labels
        self.rebuild_property_index();   // From versioned_properties
        self.rebuild_edge_type_index();  // From dynamic_edges

        if let Err(e) = self.rebuild_fulltext_indexes() {
            tracing::warn!("Failed to rebuild full-text indexes: {}", e);
        }
    }
}
```

---

## Phase 4: Delta Persistence (Sprint 66)

### 4.1 Delta Checkpoints

Delta checkpoints store only the changes since the last full snapshot, enabling efficient incremental saves.

```rust
#[derive(Debug, Serialize, Deserialize)]
pub struct DeltaCheckpoint {
    /// Transaction ID when the base snapshot was taken
    pub base_tx_id: TransactionId,
    /// Transaction ID after all changes in this delta
    pub end_tx_id: TransactionId,
    /// The actual changes (WAL entries)
    pub changes: Vec<LogEntry>,
}

impl DeltaCheckpoint {
    pub fn new(base_tx_id: TransactionId, end_tx_id: TransactionId, changes: Vec<LogEntry>) -> Self;
    pub fn save(&self, path: impl AsRef<Path>) -> Result<(), PersistenceError>;
    pub fn load(path: impl AsRef<Path>) -> Result<Self, PersistenceError>;
}
```

### 4.2 GraphStore Delta Methods

```rust
impl GraphStore {
    /// Creates a delta checkpoint with changes since the given transaction
    pub fn create_delta_checkpoint(&self, since_tx: TransactionId) -> DeltaCheckpoint;

    /// Applies a delta checkpoint to bring the store up to date
    pub fn apply_delta(&mut self, delta: &DeltaCheckpoint) -> Result<(), PersistenceError>;

    /// Saves a delta checkpoint to the deltas directory
    pub fn save_delta(&self, base_path: impl AsRef<Path>, since_tx: TransactionId)
        -> Result<PathBuf, PersistenceError>;

    /// Lists available delta files in the deltas directory
    pub fn list_deltas(base_path: impl AsRef<Path>) -> Result<Vec<PathBuf>, PersistenceError>;
}
```

### 4.3 File Structure

```
data/
├── graph.ngdb                    # Full snapshot
└── deltas/
    ├── delta_1000_2000.ngdb      # Changes from tx 1000 to 2000
    ├── delta_2000_3000.ngdb      # Changes from tx 2000 to 3000
    └── delta_3000_3500.ngdb      # Changes from tx 3000 to 3500
```

### 4.4 Usage Example

```rust
// Create a delta since last save
let delta = store.create_delta_checkpoint(last_saved_tx_id);
delta.save("data/deltas/delta_1000_2000.ngdb")?;

// Or use the convenience method
let delta_path = store.save_delta("data/graph.ngdb", last_saved_tx_id)?;

// Apply delta to a loaded snapshot
let delta = DeltaCheckpoint::load("data/deltas/delta_1000_2000.ngdb")?;
store.apply_delta(&delta)?;
```

---

## Summary

### Durability Guarantees

| Layer | Guarantee |
|-------|-----------|
| WAL | Every mutation logged with CRC32 checksum before applying |
| Atomic writes | write-tmp-rename prevents corruption |
| fsync | Data hits disk before success |
| Backup rotation | N previous versions retained |
| Graceful shutdown | Final save before exit |
| **Snapshot checksum** | SHA256 detects file corruption on load |
| **Index rebuild** | All indexes rebuilt from authoritative sources |
| **Delta persistence** | Incremental saves reduce I/O overhead |

### Performance Characteristics

| Operation | Blocking? | Lock held during I/O? |
|-----------|-----------|----------------------|
| Mutation | No | Write lock for mutation only |
| Auto-save | No | Read lock for snapshot only |
| Periodic save | No | Read lock for snapshot only |
| Shutdown save | No | Read lock for snapshot only |

### Files Changed

| File | Lines | Changes |
|------|-------|---------|
| `persistence.rs` | ~1000 | Atomic saves, backups, snapshots, SHA256 checksums, delta persistence |
| `graph_store.rs` | +150 | `snapshot()`, `post_load_init()`, validation, index rebuild |
| `wal.rs` | +50 | CRC32 checksums for WAL entries |
| `wal_reader.rs` | +30 | Checksum verification on recovery |
| `config.rs` | ~150 | NEW: Unified TOML configuration |
| `logging.rs` | ~80 | NEW: Structured logging with tracing |
| `memory.rs` | ~200 | NEW: Memory tracking and limits |
| `constraints.rs` | ~200 | NEW: Unique constraint system |
| `statistics.rs` | ~320 | NEW: Graph statistics collection |
| `lib.rs` | +20 | New module exports |
| `server.rs` | +100 | Config, async RwLock, non-blocking saves, graceful shutdown |
