# NeuralGraphDB Benchmarks

Comprehensive benchmark suite comparing **NeuralGraphDB** against **Neo4j** and **FalkorDB**.

## Quick Start

```bash
# 1. Start competitor databases
docker compose -f benchmarks/docker-compose.benchmark.yml up -d

# 2. Start NeuralGraphDB server (in another terminal)
cargo run --release -p neural-cli -- server --http-port 3000

# 3. Run the unified benchmark
python benchmarks/unified_benchmark.py -n 1000

# 4. View results
cat /tmp/benchmark_*/benchmark_report.md
```

## Unified Benchmark

The `unified_benchmark.py` script provides a fair comparison across all three databases.

### Usage

```bash
# Run with 1000 papers (default)
python benchmarks/unified_benchmark.py

# Run with 5000 papers
python benchmarks/unified_benchmark.py -n 5000

# Benchmark specific databases
python benchmarks/unified_benchmark.py --db neuralgraph,neo4j

# Skip complex queries (faster)
python benchmarks/unified_benchmark.py --skip-complex

# Custom output directory
python benchmarks/unified_benchmark.py -o results/my_benchmark/

# Full options
python benchmarks/unified_benchmark.py \
    -n 5000 \
    --db neuralgraph,neo4j,falkordb \
    --warmup 3 \
    --iterations 10 \
    -o benchmarks/results/
```

### What It Measures

| Category | Metrics |
|----------|---------|
| **Data Loading** | Paper nodes, citation edges, author relationships |
| **Simple Queries** | Node counts, 1-hop traversals, filters |
| **Complex Queries** | 2-hop, 3-hop traversals, aggregations, shortest path |
| **Memory** | Peak memory usage during operations (Docker containers) |

### Output

The benchmark generates:
- `benchmark_results.json` - Raw timing data for all databases
- `benchmark_report.md` - Formatted markdown comparison table

## Individual Benchmarks

### ArXiv Dataset Benchmarks

```bash
# NeuralGraphDB
python benchmarks/neuralgraph_benchmark.py -n 5000

# Neo4j
python benchmarks/arxiv_benchmark.py -n 5000

# FalkorDB
python benchmarks/falkordb_benchmark.py -n 5000
```

### LDBC Social Network Benchmark

```bash
# Generate LDBC data
python benchmarks/ldbc/ldbc_datagen.py

# Run benchmarks
python benchmarks/ldbc/ldbc_benchmark.py      # NeuralGraphDB
python benchmarks/ldbc/ldbc_neo4j.py          # Neo4j
python benchmarks/ldbc/ldbc_falkordb.py       # FalkorDB
```

### Performance Tests

```bash
# HTTP API latency profiling
python benchmarks/latency_profile.py

# Arrow Flight latency profiling
python benchmarks/latency_profile_flight.py

# Yelp dataset benchmark
python benchmarks/performance_test.py --load
```

### Vector Benchmarks

```bash
# 1M vector ingestion and search
python benchmarks/vector_1m.py
```

## Prerequisites

### Python Dependencies

```bash
pip install datasets sentence-transformers tqdm requests numpy neo4j falkordb
```

### Docker Services

```bash
# Start all benchmark databases
docker compose -f benchmarks/docker-compose.benchmark.yml up -d

# Check status
docker ps --filter "name=benchmark-"

# View logs
docker logs benchmark-neo4j
docker logs benchmark-falkordb

# Stop services
docker compose -f benchmarks/docker-compose.benchmark.yml down
```

### NeuralGraphDB Server

```bash
# Build release binary
cargo build --release -p neural-cli

# Start server with HTTP API
./target/release/neuralgraph server --http-port 3000 --flight-port 50051

# Or run with cargo
cargo run --release -p neural-cli -- server --http-port 3000
```

## Benchmark Configuration

Edit `unified_benchmark.py` to customize:

```python
CONFIG = {
    "neuralgraph": {
        "http_url": "http://localhost:3000/api/query",
        "container": "benchmark-neuralgraph",
    },
    "neo4j": {
        "uri": "bolt://localhost:17687",
        "auth": ("neo4j", "benchmark123"),
        "container": "benchmark-neo4j",
    },
    "falkordb": {
        "host": "localhost",
        "port": 16379,
        "graph": "unified_bench",
        "container": "benchmark-falkordb",
    },
    "batch_size": 100,
}
```

## Benchmark Queries

### Standard Queries

| Query | Description |
|-------|-------------|
| `count_papers` | `MATCH (p:Paper) RETURN count(p)` |
| `1_hop` | `MATCH (p)-[:CITES]->(c) WHERE p.id = 0 RETURN count(c)` |
| `2_hop` | `MATCH (p)-[:CITES*2]->(c) RETURN count(c)` |
| `filter_category` | `MATCH (p:Paper) WHERE p.category = 'cs.LG' RETURN count(p)` |
| `top_cited` | `MATCH (p)<-[:CITES]-(c) RETURN p.id, count(c) ORDER BY count(c) DESC` |

### Complex Queries

| Query | Description |
|-------|-------------|
| `3_hop` | 3-hop traversal with aggregation |
| `citation_network` | Category-based citation analysis |
| `shortest_path` | BFS shortest path between papers |

## Historical Results

See `docs/benchmarks/` for detailed benchmark reports:

- `benchmark_results.md` - 1K papers comparison
- `benchmark_results_50k.md` - 50K papers at scale
- `benchmark_complex_50k.md` - Complex queries with memory monitoring
- `benchmark_results_ldbc.md` - LDBC SNB results
- `neo4j_vs_neuralgraph.md` - Yelp dataset comparison

## Key Findings

Based on historical benchmarks:

| Metric | NeuralGraphDB | vs Neo4j | vs FalkorDB |
|--------|---------------|----------|-------------|
| **Data Ingestion** | Baseline | 170x faster | 10x faster |
| **1-hop Query** | Baseline | 33x faster | 3x faster |
| **Memory Usage** | Baseline | 100x less | 40x less |

NeuralGraphDB excels at:
- High-throughput data ingestion
- Memory-efficient storage
- Fast traversal queries at scale

## Troubleshooting

### Neo4j Connection Failed
```bash
# Check container is running
docker logs benchmark-neo4j

# Verify port
nc -zv localhost 17687
```

### FalkorDB Connection Failed
```bash
# Check container is running
docker logs benchmark-falkordb

# Test Redis connection
redis-cli -p 16379 ping
```

### NeuralGraphDB Connection Failed
```bash
# Check server is running
curl http://localhost:3000/api/query -d '{"query": "MATCH (n) RETURN count(n)"}'

# Check binary exists
ls -la ./target/release/neuralgraph
```

### Out of Memory
```bash
# Reduce paper count
python benchmarks/unified_benchmark.py -n 500

# Skip complex queries
python benchmarks/unified_benchmark.py --skip-complex
```
