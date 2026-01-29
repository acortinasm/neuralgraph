# NeuralGraphDB Storage Strategy

This document outlines the physical storage architecture of NeuralGraphDB (nGraph), detailing how graphs, vectors, and metadata are persisted.

## 1. Physical File Formats

NeuralGraphDB uses a custom binary storage engine designed for high-performance Rust interoperability. It does **not** rely on external databases like Postgres or generic formats like Parquet for its core storage.

### Data File (`.ngdb`)
The primary storage is a snapshot-based binary file using **Bincode** serialization.

*   **Magic Bytes:** `NGDB` (4 bytes)
*   **Version:** `u32` (4 bytes, little-endian)
*   **Payload:** Compressed binary representation of the `GraphStore`, including:
    *   **Topology:** Static CSR matrices and dynamic adjacency lists.
    *   **Properties:** Columnar-like storage of node/edge attributes (Strings, Ints, Vectors).
    *   **Indices:** Label and Property inverted indices.

### Write-Ahead Log (`.wal`)
For durability (ACID compliance), an append-only log tracks every mutation between snapshots.

*   **Mechanism:** Every `CREATE`, `SET`, or `DELETE` is serialized and flushed to the `.wal` file before the in-memory state is updated.
*   **Recovery:** On startup, the engine loads the `.ngdb` snapshot and replays the `.wal` to reach the consistent state before the last shutdown.
*   **Truncation:** The WAL is cleared automatically after a successful `save_binary` (checkpoint).

## 2. Component Storage Breakdown

| Component | Storage Type | Persistence Strategy | Detail |
| :--- | :--- | :--- | :--- |
| **Topology** | PCSR / PMA | Snapshot + WAL | Stored as sparse matrices in the `.ngdb` file. |
| **Vectors** | `Vec<f32>` | Snapshot + WAL | Stored as node properties. |
| **Vector Index** | HNSW (RAM) | **Transient** | Rebuilt in memory from raw vector properties on startup. |
| **Metadata** | String Interning | Snapshot | Mappings of property keys/labels to internal IDs. |

## 3. Vector Indexing Details

NeuralGraphDB treats vectors as first-class citizens, but differentiates between the **data** and the **index**:

1.  **Vector Data:** Persistent. Stored as `PropertyValue::Vector(Vec<f32>)`.
2.  **Vector Index (HNSW):** Transient. The Hierarchical Navigable Small World (HNSW) structure is maintained entirely in RAM to ensure sub-millisecond search latency.
3.  **Bootstrap:** During `GraphStore::load_binary()`, the engine scans all nodes, extracts vector properties, and populates the `VectorIndex`.

## 4. Architectural Considerations (Gap Analysis)

Based on recent assessments, the following evolutions are planned for the storage layer:

*   **Persistent HNSW:** Moving from "Rebuild on Load" to "Mapped Disk Storage" to support 1M+ vectors without slow boot times.
*   **LSM-VEC:** Implementing Log-Structured Merge-trees for vectors to allow datasets that exceed available RAM.
*   **Parallel Property Arrays:** Refactoring property storage to run in parallel to topology arrays to improve cache locality during filtered traversals.
