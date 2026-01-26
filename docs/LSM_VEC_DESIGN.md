# LSM-VEC: Disk-Resident Vector Storage Evaluation

**Status:** Draft (Sprint 47)
**Author:** Gemini Agent
**Date:** 2026-01-23

## 1. Problem Definition
The current vector index implementation (`neural-storage::VectorIndex`) uses the `hnsw_rs` crate, which is an **In-Memory** implementation of HNSW.

*   **RAM Usage:**
    *   **Data:** 1 Million vectors (768d, f32) ≈ 3 GB raw data.
    *   **Graph:** HNSW overhead (neighbors, layers) adds ~30-50% overhead.
    *   **Total:** ~4-5 GB RAM per 1M vectors.
*   **Scalability:**
    *   Standard laptop/server (16-32GB RAM) caps at ~3-5 Million vectors.
    *   Startup Time: Loading/Rebuilding the index takes linear time.

**Goal:** Enable storage and search of vector datasets significantly larger than available RAM (e.g., 100M vectors on a 32GB node).

## 2. Solution: LSM-VEC Architecture

We propose an **LSM (Log-Structured Merge-tree)** approach adapted for vector indices. This mirrors the `MemTable` + `SSTable` structure of RocksDB/LevelDB but for approximate nearest neighbor search.

### 2.1 Components

#### **Level 0: MemIndex (RAM)**
*   **Structure:** Standard `hnsw_rs::Hnsw`.
*   **Behavior:**
    *   Receives all new `INSERT` / `UPDATE` operations.
    *   Provides sub-millisecond search.
    *   **Flush:** When `MemIndex` reaches capacity (e.g., 50k vectors), it is frozen and flushed to disk as a `DiskIndex` (Level 1).

#### **Level 1..N: DiskIndex (Disk-Resident)**
*   **Constraint:** Unlike standard LSM keys, vectors cannot be simply "sorted".
*   **Structure:** We need a format that supports **partial reads** (minimizing I/O).
    *   **Option A: Segmented HNSW.** Serialize the HNSW graph but use `mmap` or `pread` to traverse it. (Complex: Random I/O is slow on HDD/SATA SSD, okay on NVMe).
    *   **Option B: IVF-Flat (Inverted File).**
        *   Cluster vectors into $K$ centroids (stored in header).
        *   Store raw vectors in contiguous file blocks corresponding to centroids.
        *   **Search:** Match query against centroids -> Read only relevant blocks from disk.

### 2.2 The "Merge" (Compaction)
*   As with standard LSM, we accumulate many small `DiskIndex` files.
*   **Compaction:** A background thread picks $M$ small indices and merges them into one large index.
    *   For IVF-Flat: Re-assign vectors to new (potentially better) centroids.
    *   Removes deleted vectors (tombstones).

### 2.3 Query Path (Hybrid Search)
To find top-$K$ neighbors:

1.  **Search L0 (RAM):** Query `MemIndex`. Get candidates $C_{mem}$.
2.  **Search L1..N (Disk):** Query all active `DiskIndex` files.
    *   For IVF: Calculate distance to centroids. Read top-probe blocks. Scan vectors in blocks. Get candidates $C_{disk}$.
3.  **Merge:** Combine $C_{mem}$ and $C_{disk}$, deduplicate, sort by similarity, return top-$K$.

## 3. Implementation Strategy for NeuralGraphDB

Given the reliance on `hnsw_rs` (which lacks native disk-support), we recommend a **Hybrid Approach**:

### Phase 1: Partitioned HNSW (Easier)
Instead of a true disk-index, we implement the LSM lifecycle but keep "SSTables" as `bincode` serialized HNSW graphs.
*   **Search:** We must map them into memory.
*   **Benefit:** Fast inserts (only rebuilding small L0). Parallel search over segments.
*   **Drawback:** Doesn't solve RAM limit (must load to search).

### Phase 2: True Disk-IVF (Target)
Implement a custom `DiskVectorIndex`.

*   **Format:**
    ```
    [Header: Magic | Count | Dim | Centroids_Offset]
    [Centroids: K * Dim * f32]
    [Adjacency: Vector_IDs for Cluster 0...N]
    [Data: Raw Vectors for Cluster 0...N]
    ```
*   **Search:**
    1.  Read Centroids (Small, keep in RAM cache).
    2.  Select target clusters.
    3.  `File::seek` + `read` only those clusters.
    4.  Compute distances.

## 4. Evaluation Recommendation (Sprint 47)

1.  **Stick with `hnsw_rs` for L0.**
2.  **Prototype a simple `DiskIndex` using IVF-Flat logic.**
    *   Use `linfa` or simple k-means for clustering.
    *   Store vectors in a flat binary file.
    *   Implement a `DiskSearcher` struct.
3.  **Benchmark:** Compare `HNSW (RAM)` vs `LSM-VEC (RAM + Disk)` on 1M vectors.
    *   Metric: Recall @ 10 vs Latency vs RAM Usage.

## 5. Conclusion
Moving to LSM-VEC is essential for the "Scale" phase. The Hybrid (Mem-HNSW + Disk-IVF) offers the best balance of implementation complexity vs performance for NeuralGraphDB's architecture.
