# Comparison: Neural Graph (nGraph) vs. Standard Graph Databases

**Date:** 2026-01-15  
**Version:** nGraph v0.8.0

## Executive Summary

| Feature | **Neural Graph (nGraph)** | **Neo4j** | **Memgraph** |
| :--- | :--- | :--- | :--- |
| **Primary Use Case** | **AI / GraphRAG / GNNs** | General Purpose Enterprise | Real-time / Streaming |
| **Core Architecture** | **Linear Algebra (CSR Matrices)** | Pointer Chasing (Doubly Linked Lists) | In-Memory (Skip Lists/BST) |
| **Implementation** | **Rust** | Java | C++ / Rust (MAGE) |
| **Vector Search** | **Native Core Feature** (HNSW) | Via Plugin (GDS / Vector Index) | Via Plugin (MAGE) |
| **Query Language** | **NGQL** (Subset of Cypher + AI ops) | Cypher (OpenCypher/GQL) | Cypher (OpenCypher) |
| **Performance** | **<1ns** neighbor access (CPU cache-friendly) | Good, but suffers cache misses | Excellent (In-memory) |
| **Maturity** | **v0.8 (Beta/Prototype)** | Very Mature (v5+) | Mature |

---

## 1. Core Architecture & Performance

### Neural Graph (nGraph)
- **Architecture:** Uses **Compressed Sparse Row (CSR)** matrices for topology. This is the standard data structure for scientific computing and Graph Neural Networks (GNNs).
- **Implication:** Accessing neighbors is a simple array lookup (`O(1)`), highly optimized for CPU caches. It minimizes "pointer chasing" (jumping around memory).
- **Best for:** Global graph algorithms (PageRank, Community Detection), Matrix Multiplications, and traversing dense graphs.

### Neo4j
- **Architecture:** Native Graph Storage using **doubly linked lists**. Every node points to its first relationship, which points to the next, etc.
- **Implication:** "Index-free adjacency" is great for local traversals but can cause high CPU cache misses on large/deep traversals because nodes/edges are scattered in memory.
- **Best for:** Transactional workloads, complex pathfinding, and enterprise data models.

### Memgraph
- **Architecture:** In-memory first, using specialized data structures (Skip Lists).
- **Implication:** Extremely fast for real-time data but limited by RAM (though they have added disk-spilling).

---

## 2. AI & Vector Capabilities

### Neural Graph (nGraph)
- **"AI Native":** Vectors (`Vec<f32>`) are a **first-class data type**, not an afterthought.
- **Integration:** The **HNSW index** is built directly into the storage engine. You can mix scalar filters (`age > 25`) with vector similarity (`vector_similarity(...)`) in a single query plan.
- **GraphRAG:** Explicitly designed for "Graph Retrieval-Augmented Generation," with built-in pipelines to ingest PDFs, chunk text, embed it, and query it.

### Neo4j
- **Plugin Approach:** Vector search was added recently (v5.11+). While powerful, it often feels like a separate index engine bolted onto the graph core.
- **GDS Library:** Advanced algorithms run in the **Graph Data Science (GDS)** library, which projects the graph into an in-memory format (often CSR!) to run algorithms, then writes back. nGraph does this natively without the projection step.

---

## 3. Query Language (NGQL vs. Cypher)

### Neural Graph (nGraph)
- **NGQL:** A dialect of Cypher tailored for AI.
- **Pros:** Adds native AI syntax like `CLUSTER(n)` and `vector_similarity()`.
- **Cons:** Only supports a subset of the full Cypher standard (e.g., currently lacks complex `WITH`, `UNWIND`, or extensive date/time functions).

### Neo4j & Memgraph
- **Standard Cypher:** Full support for the mature Cypher standard.
- **Ecosystem:** Massive library of APOC procedures and user-defined functions.

---

## 4. Storage & Scalability

### Neural Graph (nGraph)
- **Hybrid Storage:** Uses **CSR** for fast reads/analytics and a dynamic **Adjacency List** overlay for mutations (`CREATE`/`UPDATE`).
- **Persistence:** Write-Ahead Log (WAL) + Binary Snapshots (`bincode`).
- **Current State:** Single-node. Distributed consensus is on the roadmap but not implemented.

### Neo4j
- **Fabric / Causal Clustering:** Highly sophisticated sharding and replication for massive scale.
- **ACID:** Full enterprise-grade ACID compliance.

---

## 5. Roadmap: Missing Cypher Capabilities (Prioritized)

To become a viable alternative ("be a player"), nGraph must implement these missing standard Cypher functions and clauses.

### Priority 1: Core Logic & Control Flow (Critical)
*These are structural necessities for complex queries and pipelines.*

1.  **`WITH` Clause**:
    *   **Why**: Essential for pipelining queries, filtering aggregation results (e.g., `... RETURN count(n) as c WITH c WHERE c > 10`), and segmenting query parts to manage variable scope.
    *   **Impact**: Enables multi-stage reasoning, crucial for RAG pipelines (Retrieve -> Filter -> Re-rank).

2.  **`UNWIND` Clause**:
    *   **Why**: Converts a list into individual rows.
    *   **Impact**: Critical for batch ingestion (passing a list of objects as a parameter and unwinding them to creating nodes) and expanding list properties.

3.  **`MERGE` Clause**:
    *   **Why**: "Create if not exists" (Upsert).
    *   **Impact**: Vital for idempotent data ingestion. Without it, developers must write complex `MATCH` + `IF` logic in their application code to avoid duplicates.

4.  **`OPTIONAL MATCH`**:
    *   **Why**: Like a SQL "Left Outer Join".
    *   **Impact**: Allows traversing paths that *might* exist without filtering out the current result if they don't.

### Priority 2: Data Manipulation & Robustness
*These are needed to handle real-world, dirty data.*

5.  **`COALESCE(value, default)`**:
    *   **Why**: Returns the first non-null value.
    *   **Impact**: Handles missing properties in sparse graph data (e.g., `RETURN COALESCE(n.nickname, n.name)`).

6.  **String Functions (`toLower`, `contains`, `split`, `trim`)**:
    *   **Why**: Basic text normalization.
    *   **Impact**: Essential for search features (e.g., case-insensitive matching) and cleaning input data.

7.  **`CASE` Expression**:
    *   **Why**: Conditional logic within `RETURN` or `SET`.
    *   **Impact**: Allows dynamic property setting and formatted output without client-side logic.

### Priority 3: Type Safety & Advanced Features

8.  **Type Conversions (`toInteger`, `toFloat`, `toString`, `toBoolean`)**:
    *   **Why**: Ensures data integrity when importing from loosely typed sources (JSON/CSV).

9.  **List Functions (`range`, `head`, `last`, `size`)**:
    *   **Why**: Analyzing path arrays or property lists.

10. **Date/Time Functions (`date()`, `datetime()`, `duration()`)**:
    *   **Why**: Temporal analysis is key for event graphs (e.g., "Find transactions in the last 24h").

---

## Conclusion

- **Choose Neo4j if:** You need a battle-tested enterprise solution, complex ACID transactions, or have a massive existing team of Cypher developers.
- **Choose Memgraph if:** You need real-time streaming analysis or high-frequency trading speed.
- **Choose Neural Graph (nGraph) if:** Your primary workload is **AI/LLM-driven** (GraphRAG, Agents). You want the performance of an analytical engine (like GDS) merged with the persistence of a database, and you prefer a modern, lightweight Rust architecture over the heavy Java JVM.