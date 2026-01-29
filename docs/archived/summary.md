# NeuralGraphDB (nGraph) - Context for Gemini

**Generated Date:** 2026-01-14
**Current Status:** v0.8.0 Released
**Target Version:** v0.8 (Database Core Release)

---

## 1. Project Overview

**NeuralGraphDB** is a native graph database written in Rust, designed specifically for AI workloads (Autonomous Agents, RAG, GNNs).
It differentiates itself by using **Linear Algebra (Sparse Matrices)** for the core engine instead of pointer chasing, and treats **Vectors/Embeddings** as first-class citizens.

### Core Philosophy
*   **High Performance:** <1ns neighbor access via CSR (Compressed Sparse Row).
*   **AI Native:** Built-in HNSW vector index, Community Detection (Leiden), and LLM integration.
*   **GraphRAG Ready:** "GraphRAG-in-a-box" capabilities (Ingest PDF -> Chunk -> Embed -> Graph -> Query).

---

## 2. Architecture (Workspace Structure)

The project is a Rust Workspace with the following crates:

| Crate | Responsibility | Key Tech |
|-------|----------------|----------|
| `neural-core` | Fundamental types (`NodeId`, `Edge`, `PropertyValue`) and traits (`Graph`). | Newtypes, Enums |
| `neural-storage` | Data storage engine. CSR for topology, Columnar/HashMap for props, Indices. | CSR, `bincode`, WAL |
| `neural-parser` | NGQL (Neural Graph Query Language) parsing. | `logos` (Lexer), `nom` (Parser) |
| `neural-executor` | Query planning and execution. Physical plans. | Iterator-based execution |
| `neural-cli` | REPL, HTTP Server, and Benchmarks. | `rustyline`, `axum` |

---

## 3. Current Development Status (Sprints 1-32)

### Phase 1: Core Engine (Completed)
*   **Storage:** CSR Matrix implementation for immutable topology.
*   **Querying:** Basic `MATCH`, `WHERE`, `RETURN`.
*   **Properties:** Flexible schema (`Int`, `Float`, `String`, `Bool`, `Vector`).
*   **Aggregations:** `COUNT`, `SUM`, `AVG`, `MIN`, `MAX`, `COLLECT`.
*   **Ordering/Limiting:** `ORDER BY`, `LIMIT`, `DISTINCT`, `GROUP BY`.
*   **Optimizations:** Inverted Indices for Labels (`O(1)`), Properties (`O(1)`), and Edge Types.

### Phase 2: GraphRAG & AI (Completed)
*   **Vector Search:** HNSW Index integration. `vector_similarity()` function.
*   **Algorithms:** Leiden Community Detection (`CLUSTER BY`).
*   **Ingestion:** PDF extraction, HuggingFace dataset loader.
*   **Integration:** LLM Client (OpenAI/Ollama/Gemini) + Auto-ETL pipeline.

### Phase 3: Database Infrastructure (Completed)
*   **Mutations (CRUD):**
    *   `CREATE` Nodes & Edges (Dynamic adjacency list overlay on CSR).
    *   `DELETE` Nodes (with `DETACH`).
    *   `SET` Properties (Atomic index updates).
*   **Persistence:**
    *   Binary format (`.ngdb`) using `bincode`.
    *   Write-Ahead Log (`.wal`) for durability.
*   **Advanced Traversals:**
    *   Variable-length paths: `(a)-[*1..5]->(b)`.
    *   Shortest Path: `SHORTEST PATH (a)-[*]->(b)` (BFS implementation).
*   **Debugging & Observability:**
    *   `EXPLAIN` / `PROFILE` support (Query plan inspection & timing).
    *   Parameterized Queries (`$param`) for security and plan reuse.
*   **Performance:**
    *   **Streaming Execution:** Iterator-based processing to handle large result sets efficiently.
    *   **Optimizer:** Filter pushdown and early termination for `LIMIT` / `ORDER BY`.

---

## 4. Roadmap & Next Steps

**Immediate Focus: v1.0 Architecture (Academic Paper)**

| Sprint | Feature | Description |
|--------|---------|-------------|
| **37** | **Packed Memory Array (PMA)** | Implement PMA with density thresholds and rebalance logic for O(log^2 N) inserts. |
| **38** | **Unified PCSR** | Refactor storage to use PCSR (Packed Compressed Sparse Row), replacing dynamic adjacency lists. |
| **39** | **Neural Semirings** | Implement `neural-algebra` crate. Execute HNSW search as SpMV using custom Semirings. |

**Phase 4: Standard Cypher Compliance (Completed)**

| Sprint | Feature | Description |
|--------|---------|-------------|
| **33** | **Query Pipelining** | (Completed) Implement `WITH` (projection/filtering) and `UNWIND` (list expansion) to enable complex RAG pipelines. |
| **34** | **Advanced Patterns** | (Completed) Implement `OPTIONAL MATCH` (Outer Joins) and `MERGE` (Upsert logic) for robust data ingestion. |
| **35** | **Robust Expressions** | (Completed) Add `CASE`, `COALESCE`, String functions (`toLower`, `split`), and Type conversions for data cleaning. |
| **36** | **Temporal Engine** | (Completed) Add `Date`/`DateTime` types and functions for time-series analysis. v0.9 Release. |

**Phase 5: Academic Core (v1.0 Architecture)**

Focus: Optimizing for the "NeuralGraphDB" academic paper. Transition to PCSR and Linear Algebra kernels.

| Sprint | Feature | Description |
|--------|---------|-------------|
| **37** | **Packed Memory Array (PMA)** | (Completed) Implement PMA with density thresholds and rebalance logic for O(log^2 N) inserts. |
| **38** | **Unified PCSR** | Refactor storage to use PCSR (Packed Compressed Sparse Row), replacing dynamic adjacency lists. |
| **39** | **Neural Semirings** | Implement `neural-algebra` crate. Execute HNSW search as SpMV using custom Semirings. |
| **40** | **SIMD Acceleration** | Optimize matrix operations with `faer` and AVX-512/NEON intrinsics. |
| **41** | **Native Leiden** | Implement parallel Leiden community detection directly on the PCSR matrix. |
| **42** | **Context Summary** | Add `SUMMARIZE` clause to NGQL for "Knowledge Subgraph" extraction. |
| **43** | **Python Client & Core Fixes** | (Completed) Enable Python Client pipeline (`MATCH...CREATE`), fix dynamic node IDs, and improve server robustness. |
| **44** | **Validation (LDBC)** | (In Progress) Execute LDBC benchmarks. Current: Faster than Neo4j, aiming for FalkorDB parity. |
| **45** | **Read Latency Optimization** | (Completed) Implemented Arrow Flight query execution, achieving ~0.10ms latency (3x faster than FalkorDB). |
| **46** | **Core Stability & Cypher Standards** | (Completed) Fix parser edge cases (`id()` function, `shortestPath` syntax), ensure standard Cypher compliance, and harden the query planner. |
| **47** | **Vector Scale (1M)** | **(Completed)** Evaluated and prototyped LSM-VEC (Disk-IVF) to support larger-than-RAM vector datasets. Validated with unit tests. |

**Phase 6: Ecosystem & Visualization (New Phase)**

| Sprint | Feature | Description |
|--------|---------|-------------|
| **48** | **SQL Bridge** | Implement Python-based ETL tool (SQLAlchemy -> NeuralGraph) for legacy database synchronization. |
| **49** | **Neural Dashboard** | Develop React-based Web UI for graph visualization ("Reef" view) and system monitoring. |

**Phase 7: Scale & Production (In Progress)**

| Sprint | Feature | Description |
|--------|---------|-------------|
| **50** | **Transaction Manager (ACID)** | (Completed) Implemented `BEGIN`, `COMMIT`, `ROLLBACK` for multi-statement atomicity. |
| **51** | **MVCC (Snapshot Isolation)** | (Completed) Implemented multi-version concurrency control for non-blocking reads. |
| **52** | **Distributed Consensus (Raft)** | (Impl Complete) Multi-node replication and leader election. Currently in Integration Testing. |
| **53** | **Cluster Management** | Node discovery and dynamic membership. |
| **54** | **Time-Travel** | `AT TIME` query support using MVCC history. |

**Phase 8: Advanced GraphRAG (Gap Analysis)**

| Sprint | Feature | Description |
|--------|---------|-------------|
| **55** | **Hybrid Retrieval (wRRF)** | Implement Weighted Reciprocal Rank Fusion to mathematically combine vector similarity scores with graph centrality/topology ranks. |
| **56** | **Semantic Ingestion** | Implement `MERGE ON SIMILARITY` to automatically resolve and fuse duplicate entities using vector thresholds during ingestion. |

*   **Distributed Consensus (Raft):** Multi-node clustering.
*   **ACID Transactions:** Full transactional guarantees.
*   **Time-Travel:** `AT TIME` query support.

---

## 5. Technical Reference

### NGQL Syntax (Supported)

```cypher
-- Basic Match
MATCH (p:Person)-[:KNOWS]->(f:Person)
WHERE p.age > 25 AND f.city = "Madrid"
RETURN p.name, f.name

-- Vector Search (Hybrid)
MATCH (d:Document)
WHERE vector_similarity(d.embedding, $query_vector) > 0.85
RETURN d.title
ORDER BY vector_similarity(d.embedding, $query_vector) DESC
LIMIT 5

-- AI / Clustering
MATCH (n) RETURN CLUSTER(n), COUNT(*)

-- Mutations
CREATE (n:User {name: "Alice", active: true})
MATCH (a:User), (b:User) WHERE a.name="Alice" AND b.name="Bob" CREATE (a)-[:FRIEND]->(b)
MATCH (n:User) WHERE n.id = 1 SET n.active = false
MATCH (n:User) WHERE n.id = 99 DETACH DELETE n

-- Path Algorithms
MATCH p = SHORTEST PATH (a)-[*]->(b) WHERE a.id=1 AND b.id=100 RETURN p
MATCH (a)-[:KNOWS*1..3]->(b) RETURN a, b
```

### Key Data Structures
*   **Topology:** Static CSR + Dynamic Adjacency List (Hybrid approach for mutations).
*   **Indices:**
    *   `LabelIndex`: `HashMap<String, SortedVec<NodeId>>`
    *   `PropertyIndex`: `HashMap<Key, HashMap<Value, SortedVec<NodeId>>>`
    *   `VectorIndex`: HNSW (via crate `hnsw_rs` or similar wrapper)
*   **Persistence:**
    *   Snapshot: `bincode` serialization of GraphStore.
    *   Log: Append-only WAL of mutation events.
*   **Execution:**
    *   `RowStream`: `Box<dyn Iterator<Item = Result<Bindings>>>` for lazy evaluation.

### Development Commands
*   **Run REPL:** `cargo run -p neural-cli --release`
*   **Run Demo:** `cargo run -p neural-cli --release -- --demo`
*   **Run Tests:** `cargo test --workspace`
*   **Docs:** See `@docs/` folder for specific sprint reports.