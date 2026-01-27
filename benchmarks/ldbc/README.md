# LDBC-SNB Benchmark Suite for NeuralGraphDB

This directory contains the comprehensive LDBC Social Network Benchmark (SNB) implementation for NeuralGraphDB validation.

## Overview

The LDBC-SNB benchmark measures graph database performance using standardized queries on a social network dataset. This implementation includes:

- **14 LDBC-SNB Interactive queries** (IS1-IS7, IC1-IC7)
- **Configurable scale factors** (SF0.1, SF1, SF10, SF100)
- **Statistical analysis** with p50, p95, p99 percentiles
- **Multi-execution reproducibility** (default: 3 executions)
- **Comparison support** for Neo4j and FalkorDB
- **Paper-ready visualizations**

## Directory Structure

```
benchmarks/ldbc/
├── README.md                  # This file
├── ldbc_queries.py           # 14 LDBC query definitions
├── ldbc_datagen.py           # Dataset generator for scale factors
├── ldbc_benchmark.py         # Main benchmark runner
├── run_benchmark.sh          # Convenience script
├── data/                     # Generated datasets
│   ├── SF0.1/               # 1K persons
│   ├── SF1/                 # 10K persons
│   ├── SF10/                # 100K persons
│   └── SF100/               # 1M persons
└── results/                  # Benchmark results
    └── SF1/
        ├── neuralgraph_results.json
        ├── benchmark_report.md
        ├── latency_comparison.png
        └── percentile_distribution.png
```

## Quick Start

### 1. Generate Data

```bash
# Generate SF1 dataset (10K persons, ~180K edges)
python benchmarks/ldbc/ldbc_datagen.py --sf SF1

# Generate SF10 for larger tests
python benchmarks/ldbc/ldbc_datagen.py --sf SF10
```

### 2. Start NeuralGraphDB Server

```bash
./target/release/neuralgraph server --port 3000
```

### 3. Run Benchmark

```bash
# Basic benchmark (NeuralGraphDB only)
python benchmarks/ldbc/ldbc_benchmark.py --sf SF1

# Full comparison with 3 executions
python benchmarks/ldbc/ldbc_benchmark.py --sf SF1 --db neuralgraph,neo4j,falkordb -e 3

# Custom iterations for higher precision
python benchmarks/ldbc/ldbc_benchmark.py --sf SF1 --warmup 5 --iterations 20
```

## Scale Factors

| Scale Factor | Persons | Avg Edges | Use Case |
|--------------|---------|-----------|----------|
| SF0.1 | 1,000 | ~15K | Quick testing |
| SF1 | 10,000 | ~180K | Standard benchmark |
| SF10 | 100,000 | ~1.8M | Stress testing |
| SF100 | 1,000,000 | ~18M | Production scale |

## LDBC Queries

### Interactive Short (IS1-IS7)

Simple read queries with single-hop traversals:

| ID | Name | Description |
|----|------|-------------|
| IS1 | Profile | Person profile lookup |
| IS2 | Recent Messages | Last 10 messages by person |
| IS3 | Friends | Direct friends list |
| IS4 | Message Content | Message details |
| IS5 | Message Creator | Author of message |
| IS6 | Forum of Message | Container forum |
| IS7 | Message Replies | Replies and authors |

### Interactive Complex (IC1-IC7)

Multi-hop traversals and aggregations:

| ID | Name | Description |
|----|------|-------------|
| IC1 | Friends with Name | 2-hop search by name |
| IC2 | Recent Messages from Friends | Friend posts before date |
| IC3 | Friends in Countries | Geographic filtering |
| IC4 | New Topics | Tag discovery in timeframe |
| IC5 | New Groups | Forum membership analysis |
| IC6 | Tag Co-occurrence | Related tags |
| IC7 | Recent Likers | Engagement analysis |

## Output Format

### JSON Results

```json
{
  "database": "neuralgraph",
  "scale_factor": "SF1",
  "num_executions": 3,
  "aggregated": {
    "IS1": {
      "mean_ms": 2.45,
      "p50_ms": 2.12,
      "p95_ms": 4.56,
      "p99_ms": 6.78
    }
  }
}
```

### Visualizations

- `latency_comparison.png` - Bar chart comparing p50 across databases
- `percentile_distribution.png` - p50/p95/p99 bands per database
- `category_comparison.png` - IS vs IC query categories

## Acceptance Criteria (Sprint 58)

- [x] 14 LDBC-SNB queries executing
- [x] Reproducible results in 3 executions
- [x] Latencies p50, p95, p99 documented
- [x] Graphs ready for paper

## Requirements

```bash
pip install requests tqdm numpy matplotlib
# Optional for comparison:
pip install neo4j falkordb faker
```

## Running with Docker Comparison

```bash
# Start comparison databases
docker run -d --name benchmark-neo4j \
  -p 17687:7687 -p 17474:7474 \
  -e NEO4J_AUTH=neo4j/benchmark123 \
  neo4j:5

docker run -d --name benchmark-falkordb \
  -p 16379:6379 \
  falkordb/falkordb:latest

# Run comparison
python benchmarks/ldbc/ldbc_benchmark.py --sf SF1 --db neuralgraph,neo4j,falkordb
```
