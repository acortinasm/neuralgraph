# NeuralGraphDB

**High-performance graph database with GraphRAG capabilities**

[![Rust](https://img.shields.io/badge/rust-1.70%2B-orange.svg)](https://www.rust-lang.org)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

## Features

### Core (Fase 1 ✅)

- **CSR Matrix Storage** - O(1) neighbor access, cache-friendly traversals
- **NGQL Query Language** - Cypher-like syntax with MATCH, WHERE, RETURN
- **Aggregations** - COUNT, SUM, AVG, MIN, MAX, COLLECT
- **Indices** - Label, Property, and Edge Type indices

### GraphRAG Suite (Fase 2 ✅)

- **HNSW Vector Index** - Approximate nearest neighbor search
- **Community Detection** - Leiden algorithm via `CLUSTER(n)`
- **PDF Ingestion** - Extract text from PDF documents
- **LLM Client** - OpenAI, Ollama, and Gemini support
- **Auto-ETL** - PDF → LLM → Graph pipeline

### Database Infrastructure (Fase 3 ✅)

- **Mutations** - CREATE, DELETE, SET with atomic updates
- **Persistence** - Binary format and Write-Ahead Log (WAL)
- **Advanced Traversals** - Variable-length paths and Shortest Path (BFS)
- **Observability** - EXPLAIN / PROFILE query plans
- **Performance** - Streaming execution and filter pushdown

### Scale & Production (Fase 7 ✅)

- **ACID Transactions** - `BEGIN`, `COMMIT`, `ROLLBACK` with multi-statement atomicity
- **Distributed Consensus** - Raft-based clustering with automatic leader election
- **Time-Travel** - `AT TIME` temporal queries with MVCC snapshots
- **Database Hardening** - WAL/snapshot checksums, post-load validation, index rebuild
- **Incremental Persistence** - Delta checkpoints for efficient saves
- **Structured Logging** - Configurable logging with `tracing` crate
- **Unified Configuration** - TOML config with environment variable overrides

## Quick Start

```rust
use neural_storage::{GraphStoreBuilder, etl::EtlPipeline, llm::LlmClient};

// Create graph with ETL pipeline
let pipeline = EtlPipeline::new(LlmClient::gemini("API_KEY"), "gemini-pro");
let result = pipeline.process_pdf("document.pdf")?;
let store = pipeline.insert_into_graph(&result, GraphStoreBuilder::new()).build();

// Query with NGQL
let result = execute_query("MATCH (n:Person) RETURN n.name, CLUSTER(n)", &store)?;
```

## Modules

| Crate | Description |
|-------|-------------|
| `neural-core` | Core types (NodeId, Edge, Graph trait) |
| `neural-storage` | CSR storage, indices, vector index, LLM, ETL |
| `neural-parser` | NGQL lexer and parser |
| `neural-executor` | Query execution engine |

## License

MIT
