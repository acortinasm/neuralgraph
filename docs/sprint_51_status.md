# Sprint 51: MVCC (Snapshot Isolation) - Completed

**Objective:** To refactor the storage engine to support Multi-Version Concurrency Control (MVCC), which is the foundation for enabling non-blocking reads and future time-travel queries.

**Work Completed:**

1.  **Core MVCC Logic:**
    *   Created `crates/neural-storage/src/mvcc.rs`.
    *   Implemented `VersionedValue`, a new data structure that stores a history of property values, each tagged with the `TransactionId` that created it.
    *   Implemented the core visibility rule logic to determine which version of a value is visible to a given transaction snapshot.
    *   Implemented `vacuum()` for garbage collection of old versions.

2.  **VersionedPropertyStore:**
    *   Created `VersionedPropertyStore` in `crates/neural-storage/src/properties.rs`.
    *   Implemented MVCC-aware methods: `set(node, key, value, tx_id)`, `get(node, key, snapshot_id)`, `remove(node, key, tx_id)`, `remove_all(node, tx_id)`, `contains(node, key, snapshot_id)`, `has_properties(node, snapshot_id)`, `get_all(node, snapshot_id)`, `set_many(node, properties, tx_id)`, `vacuum(min_active_tx_id)`.
    *   Added comprehensive tests for version visibility, updates, deletions, and garbage collection.

3.  **GraphStore MVCC Integration:**
    *   Replaced `PropertyStore` with `VersionedPropertyStore` for both properties and labels in `GraphStore`.
    *   Added `MAX_SNAPSHOT_ID` constant for backward-compatible reads (sees all committed data).
    *   Added `current_tx_id` counter for automatic transaction ID assignment.
    *   Updated all internal methods to use versioned storage:
        *   `get_property()` / `get_property_at(node, key, snapshot_id)`
        *   `set_property()` / `set_property_at(node, key, value, tx_id)`
        *   `get_label()` / `get_label_at(node, snapshot_id)`
        *   `set_label()` / `set_label_at(node, label, tx_id)`
        *   `is_deleted()` / `is_deleted_at(node, snapshot_id)`
        *   `get_all_properties_at(node, snapshot_id)`
        *   `current_snapshot_id()` accessor
    *   Updated `apply_log_entry()` to use versioned writes with transaction IDs.
    *   Updated `GraphStoreBuilder::build()` to initialize versioned storage.

4.  **Executor MVCC Support:**
    *   Added `snapshot_id` field to `Executor` struct.
    *   Added `Executor::with_snapshot(store, snapshot_id)` constructor for transactional reads.
    *   Updated all property/label reads to use snapshot-aware methods:
        *   `execute_scan_by_label()` uses `get_label_at()`
        *   `execute_scan_by_property()` uses `get_property_at()`
        *   `execute_project()` uses `get_property_at()`
        *   `execute_filter()` uses `is_true_at()`

5.  **Expression Evaluation MVCC Support:**
    *   Added `evaluate_at(expr, bindings, store, params, snapshot_id)` function.
    *   Added `is_true_at(expr, bindings, store, params, snapshot_id)` function.
    *   Updated all recursive evaluation calls to propagate `snapshot_id`.
    *   Maintained backward compatibility with `evaluate()` and `is_true()` using `MAX_SNAPSHOT_ID`.

**Testing:**
*   All 8 versioned property store tests pass.
*   All 29 executor unit tests pass.
*   All 45 executor integration tests pass (including transaction tests).
*   Full workspace compiles successfully.

**API Usage Examples:**

```rust
// Non-transactional read (sees all committed data)
let executor = Executor::new(&store);
let result = executor.execute(&plan, None)?;

// Transactional read at specific snapshot
let snapshot_id = store.current_snapshot_id();
let executor = Executor::with_snapshot(&store, snapshot_id);
let result = executor.execute(&plan, None)?;

// Direct property access at snapshot
let value = store.get_property_at(node_id, "name", snapshot_id);
let label = store.get_label_at(node_id, snapshot_id);
```
