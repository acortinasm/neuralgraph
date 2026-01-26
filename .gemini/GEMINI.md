# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

NeuralGraphDB (nGraph) is a high-performance native graph database written in Rust for AI workloads (Autonomous Agents, RAG, GNNs). Core differentiators:
- Uses sparse matrices (CSR) for topology instead of pointer-chasing
- Vectors/embeddings as first-class citizens with HNSW indexing
- NGQL query language (Cypher-like syntax)

## Build Commands

```bash
cargo build --release                      # Full optimized build
cargo build -p neural-cli --release        # CLI only

cargo test --workspace                     # All tests
cargo test -p neural-storage               # Specific crate
cargo test vector_index                    # Specific test by name
cargo test -- --nocapture                  # Show stdout

cargo clippy --workspace                   # Lint
cargo fmt                                  # Format

cargo run -p neural-cli --release          # Interactive REPL
cargo run -p neural-cli --release -- --demo      # Demo mode
cargo run -p neural-cli --release -- --benchmark # Benchmarks
```

## Architecture

Five crates in workspace, layered by dependency:

```
neural-cli (REPL, HTTP server, Arrow Flight)
     ↓
neural-executor (query planning + execution)
     ↓
neural-parser (NGQL lexer + parser)
     ↓
neural-storage (CSR topology, properties, indices, persistence, vectors)
     ↓
neural-core (NodeId, EdgeId, Label, PropertyValue, Graph/Vector/Matrix traits)
```

### Key Modules

**neural-core**: Newtypes (`NodeId`, `EdgeId`), `PropertyValue` enum, `Graph`/`Semiring` traits

**neural-storage** (`graph_store.rs`):
- `csr.rs`: Compressed Sparse Row matrix for immutable topology
- `vector_index.rs`: HNSW for approximate nearest neighbor
- `lsm_vec.rs`: LSM-tree for disk-resident vectors (>RAM datasets)
- `persistence.rs`, `wal.rs`: Binary snapshots + write-ahead log
- `transaction.rs`, `mvcc.rs`: ACID transactions with snapshot isolation
- `community.rs`: Leiden algorithm for community detection
- `etl.rs`: PDF → LLM → Graph pipeline

**neural-parser**: `logos`-based lexer, recursive descent parser → AST

**neural-executor**:
- `planner.rs`: AST → LogicalPlan → PhysicalPlan
- `executor.rs`: Iterator-based streaming execution
- `eval.rs`: Expression evaluation for WHERE/RETURN

**neural-cli**: REPL with history, REST API (`server.rs`), Arrow Flight (`flight.rs`)

## Design Patterns

- **CSR + overlay**: Immutable CSR for reads, adjacency lists for mutations
- **Iterator streaming**: `RowStream = Box<dyn Iterator<Item = Result<Bindings>>>` - never materialize full results
- **Transaction buffering**: Mutations buffer until COMMIT, atomic write to WAL
- **Index proliferation**: Label, Property, EdgeType indices - all O(1) HashMap lookups

## Adding Query Features

1. Extend AST in `neural-parser/src/ast.rs`
2. Add parser rules in `neural-parser/src/parser.rs`
3. Implement planner in `neural-executor/src/planner.rs`
4. Implement executor in `neural-executor/src/executor.rs`
5. Add tests, update docs

## Debugging Queries

```sql
EXPLAIN MATCH (n:Person) WHERE n.age > 25 RETURN n.name
PROFILE MATCH (n:Person) RETURN n.name ORDER BY n.age LIMIT 5
```

## Requirements

- Rust 1.85+ (Edition 2024)
- Python 3.10+ for benchmarks/client
