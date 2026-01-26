# Unified Benchmark Report

**Generated:** 2026-01-26 09:17:14

**Dataset:** 1000 papers, ~2460 citations, ~1989 authors

## Data Loading Performance

| Metric | neuralgraph | neo4j | falkordb |
|--------|--------|--------|--------|
| Load Papers | 0.015s | 0.031s | 0.018s |
| Create Citations | 0.029s | 0.100s | 0.091s |
| Create Authors | 0.043s | 0.145s | 0.095s |
| **Total** | **0.087s** | **0.277s** | **0.204s** |

## Query Latency (ms)

| Query | neuralgraph | neo4j | falkordb |
|-------|--------|--------|--------|
| 1_hop | 4.04 | 4.35 | 2.87 |
| 2_hop | 4.89 | 4.32 | 1.72 |
| 3_hop | 5.39 | 11.59 | 10.58 |
| citation_network | 6.14 | 5.59 | 5.05 |
| count_authors | 3.95 | 4.68 | 2.49 |
| count_citations | 4.55 | 3.70 | 3.78 |
| count_papers | 3.89 | 3.80 | 2.11 |
| filter_category | 4.19 | 4.78 | 2.00 |
| filter_with_rel | 6.72 | 7.63 | 7.09 |
| institution_count | 4.65 | 6.34 | 1.99 |
| shortest_path | 4.94 | 3.53 | FAILED |
| top_cited | 5.98 | 6.57 | 4.59 |

## Memory Usage (MB)

| Operation | neuralgraph | neo4j | falkordb |
|-----------|--------|--------|--------|
| Load Papers | 38.0 | 2762.8 | 219.2 |
| Create Citations | 38.1 | 2762.8 | 220.5 |
| Create Authors | 38.2 | 2758.7 | 221.0 |

## Speedup vs NeuralGraphDB

### neo4j

- **Data Loading:** NeuralGraphDB is 3.2x faster
- **count_papers:** NeuralGraphDB is 1.0x faster
- **count_citations:** NeuralGraphDB is 0.8x faster
- **count_authors:** NeuralGraphDB is 1.2x faster
- **1_hop:** NeuralGraphDB is 1.1x faster
- **2_hop:** NeuralGraphDB is 0.9x faster
- **filter_category:** NeuralGraphDB is 1.1x faster
- **filter_with_rel:** NeuralGraphDB is 1.1x faster
- **top_cited:** NeuralGraphDB is 1.1x faster
- **institution_count:** NeuralGraphDB is 1.4x faster
- **3_hop:** NeuralGraphDB is 2.1x faster
- **citation_network:** NeuralGraphDB is 0.9x faster
- **shortest_path:** NeuralGraphDB is 0.7x faster

### falkordb

- **Data Loading:** NeuralGraphDB is 2.3x faster
- **count_papers:** NeuralGraphDB is 0.5x faster
- **count_citations:** NeuralGraphDB is 0.8x faster
- **count_authors:** NeuralGraphDB is 0.6x faster
- **1_hop:** NeuralGraphDB is 0.7x faster
- **2_hop:** NeuralGraphDB is 0.4x faster
- **filter_category:** NeuralGraphDB is 0.5x faster
- **filter_with_rel:** NeuralGraphDB is 1.1x faster
- **top_cited:** NeuralGraphDB is 0.8x faster
- **institution_count:** NeuralGraphDB is 0.4x faster
- **3_hop:** NeuralGraphDB is 2.0x faster
- **citation_network:** NeuralGraphDB is 0.8x faster

