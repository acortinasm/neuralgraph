# Changelog

All notable changes to NeuralGraphDB.

## [0.9.2] - 2026-01-21

### Added - Scale & Production (Fase 7)

#### Sprint 50: Transaction Manager (ACID Core)
- **Multi-statement Transactions:** Support for `BEGIN`, `COMMIT`, and `ROLLBACK`.
- **Atomic Buffering:** Mutations are buffered in a transaction context and applied atomically on commit.
- **WAL Consistency:** Transaction boundaries are recorded in the Write-Ahead Log to ensure durability and correct recovery.
- **Predictive IDs:** ID generation logic now accounts for pending creations within an active transaction.
- **Parser Update:** Extended NGQL parser to support transaction keywords and improved `CREATE` pattern parsing for complex edge-node combinations.

## [0.9.1] - 2026-01-21

### Added
- **`type()` function:** Support for extracting edge relationship types in queries (e.g., `RETURN type(r)`).
- **Keyword Flexibility:** Keywords like `id`, `type`, `count` can now be used as variable names and property keys.
- **Shortest Path:** Full support for `shortestPath(...)` and `SHORTEST PATH ...` syntax variants.

### Fixed
- **Incoming Edge IDs:** Fixed issue where traversing incoming edges (`<-[r]-`) returned incorrect edge IDs.
- **Plan Formatting:** Improved `EXPLAIN` output formatting to match standard Cypher visual style.

## [0.9.0] - 2026-01-19

### Added - Database Infrastructure (Fase 3)

#### Sprint 21-24: CRUD Operations
- `CREATE` nodes and edges
- `DELETE` nodes with `DETACH` support
- `SET` properties for atomic updates

#### Sprint 25-26: Persistence
- Binary format (`.ngdb`) using `bincode`
- Write-Ahead Log (`.wal`) for durability

#### Sprint 27-28: Advanced Traversals
- Variable-length paths: `(a)-[*1..5]->(b)`
- `SHORTEST PATH` algorithm (BFS implementation)

#### Sprint 29: Query Inspection
- `EXPLAIN` keyword to show physical execution plans
- `PROFILE` keyword to measure execution time and stats

#### Sprint 30: Parameterized Queries
- `$param` syntax support in NGQL
- Resolution of parameters in Executor and eval engine
- CLI support via `:param <name> <value>`

#### Sprint 31: Streaming Execution
- Iterator-based execution engine (`RowStream`) to handle large result sets
- Planner optimization: Push-down of `LIMIT` and `ORDER BY` for early termination
- Refactored `Executor` to avoid eager collection of intermediate results

## [0.5.0-beta] - 2026-01-12

### Added - GraphRAG Suite (Fase 2)

#### Sprint 13: HNSW Vector Index

- HNSW-based approximate nearest neighbor search
- `VectorIndex` with insert, search, and filtered search

#### Sprint 14: Vector ORDER BY

- `OrderBy`, `Limit`, `VectorSearch` plan nodes
- Executor support for ORDER BY expressions

#### Sprint 15: Leiden Community Detection

- `fa-leiden-cd` integration
- `detect_communities()` on GraphStore
- `Communities` struct for node → community mapping

#### Sprint 16: CLUSTER BY Keyword

- `CLUSTER(n)` function in NGQL
- Returns community ID using Leiden algorithm

#### Sprint 17: PDF Ingestion

- `pdf-extract` integration
- `load_pdf()` and `load_pdf_bytes()`

#### Sprint 18: LLM Client

- OpenAI, Ollama, and Gemini support
- `complete()` for chat, `embed()` for embeddings

#### Sprint 19: Auto-ETL Pipeline

- `EtlPipeline` for PDF → LLM → Graph
- Entity/relation extraction from documents

## [0.4.0] - 2026-01-11

### Added - Core Engine (Fase 1)

- CSR matrix storage
- NGQL parser (MATCH, WHERE, RETURN, ORDER BY, LIMIT, GROUP BY)
- Aggregation functions
- Label, Property, and Edge Type indices
- Query planner with filter pushdown
