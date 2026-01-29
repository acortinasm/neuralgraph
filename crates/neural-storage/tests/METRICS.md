# Global Metrics Test Suite

Comprehensive performance metrics for NeuralGraphDB features. Run after each sprint to track performance over time.

## Metrics Tracked

| Category | Metrics |
|----------|---------|
| **Vector Index (HNSW)** | Build time, Memory usage, Search latency (p50/p95/p99) for k=10/50/100, Throughput (qps) |
| **Flash Quantization** | Memory savings, Compression ratio (Int8 vs f32) |
| **Distributed Search** | Shard count, Latency, Cache hit rate, Recall vs centralized |
| **Graph Sharding** | Edge cut % (Hash, Range, Community), Routing overhead |
| **Coordinator** | Execution overhead, Vector search routing latency |

## Running the Tests

### Quick Test (Development)

Fast test for development iteration (~2 seconds):

```bash
cargo test -p neural-storage --test global_metrics global_metrics_quick --release -- --nocapture
```

### Standard Test (Comprehensive)

More comprehensive test with larger data sizes (longer runtime):

```bash
cargo test -p neural-storage --test global_metrics global_metrics_standard --release -- --ignored --nocapture
```

### All Metrics Tests

```bash
# Run all non-ignored tests (quick + individual metric tests)
cargo test -p neural-storage --test global_metrics --release -- --nocapture

# Run everything including ignored tests
cargo test -p neural-storage --test global_metrics --release -- --ignored --nocapture
```

### Individual Metric Tests (for CI)

These tests have assertions and are suitable for CI pipelines:

```bash
# Vector search latency (asserts p95 < 50ms)
cargo test -p neural-storage --test global_metrics metrics_vector_search_latency --release -- --nocapture

# Distributed search recall (asserts recall >= 70%)
cargo test -p neural-storage --test global_metrics metrics_distributed_search_recall --release -- --nocapture

# Quantization memory savings (asserts >= 3x compression)
cargo test -p neural-storage --test global_metrics metrics_quantization_memory_savings --release -- --nocapture

# Sharding edge cut (validates edge cut percentages)
cargo test -p neural-storage --test global_metrics metrics_sharding_edge_cut --release -- --nocapture
```

## Test Scales

| Scale | Vectors | Dimension | Graph Nodes | Runtime |
|-------|---------|-----------|-------------|---------|
| Quick (Dev) | 5,000 | 64 | 5,000 | ~2 sec |
| Standard | 20,000 | 128 | 10,000 | ~30 sec |

## Sample Output

```
================================================================================
NEURALGRAPHDB METRICS REPORT
================================================================================
Scale: Quick (Dev)
Timestamp: 2026-01-28T09:16:08.695543+00:00
================================================================================

## Test Configuration
------------------------------------------------------------
  Vector count                   5000
  Vector dimension               64
  Graph nodes                    5000
  Graph avg degree               8
  Query count                    50

## Vector Index (HNSW)
------------------------------------------------------------
  Build time                     1.06s
  Vectors indexed                5000
  Dimension                      64
  Memory (vectors)               1.28 MB (5000 vectors)
  Memory (metadata)              320.00 KB
  Search k=10                    p50=    0.04ms  p95=    0.05ms  p99=    0.05ms  mean=    0.04ms
  Search k=50                    p50=    0.04ms  p95=    0.05ms  p99=    0.05ms  mean=    0.04ms
  Search k=100                   p50=    0.08ms  p95=    0.08ms  p99=    0.08ms  mean=    0.08ms
  Throughput (k=10)              24421 qps

## Flash Quantization
------------------------------------------------------------
  Unquantized memory             128.00 KB
  Int8 quantized memory          32.00 KB
  Memory savings                 75.0%
  Compression ratio              4.0x

## Distributed Vector Search
------------------------------------------------------------
  Shards                         4
  Total vectors                  2500
  Vectors per shard              ~625
  Search latency (k=100)         p50=    0.28ms  p95=    0.29ms  p99=    0.31ms  mean=    0.28ms
  Cache hit rate                 3.8%
  Recall@50 vs centralized       96.4%

## Graph Sharding
------------------------------------------------------------
  Graph nodes                    5000
  Graph edges                    19990
  Avg degree                     8
  Shards                         4
  Edge cut (Hash)                74.7%
  Edge cut (Range)               65.6%
  Routing overhead               p50=    0.00ms  p95=    0.00ms  p99=    0.00ms  mean=    0.00ms

## Shard Coordinator
------------------------------------------------------------
  Execution overhead (4 shards)  p50=    0.00ms  p95=    0.00ms  p99=    0.00ms  mean=    0.00ms
  Vector search routing          p50=    0.00ms  p95=    0.00ms  p99=    0.00ms  mean=    0.00ms

================================================================================
```

## Notes

- Always use `--release` flag for accurate performance measurements
- Use `--nocapture` to see the metrics report in the output
- The `--ignored` flag is required for the standard test (it's marked `#[ignore]` to avoid slow CI runs)
- Individual metric tests include assertions suitable for CI regression detection
