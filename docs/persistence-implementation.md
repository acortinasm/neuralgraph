# Persistence Implementation

This document describes the automatic persistence implementation for NeuralGraphDB, split between the storage library and application layers.

---

## Phase 1: Library Layer (neural-storage)

### Files Modified

- `crates/neural-storage/src/persistence.rs` - Core persistence logic
- `crates/neural-storage/src/graph_store.rs` - Snapshot and getter methods
- `crates/neural-storage/src/lib.rs` - Public exports

---

### 1.1 Enhanced Error Types

**Location**: `persistence.rs:77-130`

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

## Summary

### Durability Guarantees

| Layer | Guarantee |
|-------|-----------|
| WAL | Every mutation logged before applying |
| Atomic writes | write-tmp-rename prevents corruption |
| fsync | Data hits disk before success |
| Backup rotation | N previous versions retained |
| Graceful shutdown | Final save before exit |

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
| `persistence.rs` | ~850 | Complete rewrite with atomic saves, backups, snapshots |
| `graph_store.rs` | +50 | `snapshot()`, `post_load_init()`, new getters |
| `lib.rs` | +2 | New exports |
| `server.rs` | +100 | Config, async RwLock, non-blocking saves, graceful shutdown |
