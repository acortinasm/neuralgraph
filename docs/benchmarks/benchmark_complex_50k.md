# ArXiv Benchmark: Complex Queries & Memory (50,000 Papers)

**Date:** 2026-01-20  
**Num Papers:** 50,000

## 1. NeuralGraphDB
| Metric | Time | Max Memory |
| :--- | :--- | :--- |
| **Ingestion (Nodes/Edges)** | 0.46s | 16.59 MB |
| **Q: 3-Hop Traversal** | 0.46s | 16.59 MB |
| **Q: Analytical (Top Cited)** | 3.11s | 16.59 MB |
| **Q: Shortest Path** | 0.37s | 16.59 MB |

---

## 2. Neo4j
| Metric | Time | Max Memory |
| :--- | :--- | :--- |
| **Ingestion (Nodes/Edges)** | 79.57s | 1795.07 MB |
| **Q: 3-Hop Traversal** | 0.22s | 1789.95 MB |
| **Q: Analytical (Top Cited)** | 0.09s | 1794.05 MB |
| **Q: Shortest Path** | 0.07s | 1796.10 MB |

---

## 3. FalkorDB
| Metric | Time | Max Memory |
| :--- | :--- | :--- |
| **Ingestion (Nodes/Edges)** | 138.25s | 642.70 MB |
| **Q: 3-Hop Traversal** | 0.08s | 635.20 MB |
| **Q: Analytical (Top Cited)** | 0.10s | 635.00 MB |
| **Q: Shortest Path** | <0.01s | 635.60 MB |

---

## 📊 Final Scalability Summary (50,000 Papers)

| Metric | **NeuralGraphDB** | Neo4j | FalkorDB |
| :--- | :--- | :--- | :--- |
| **Ingestion Speed** | **0.46s** 🚀 | 79.57s | 138.25s |
| **Peak RAM Usage** | **16.6 MB** 🚀 | 1.8 GB | 642.7 MB |
| **3-Hop Traversal** | 0.46s | **0.22s** | **0.08s** |
| **Shortest Path** | 0.37s | 0.07s | **<0.01s** |
| **Analytical (Top Cited)**| 3.11s | **0.09s** | **0.10s** |

### 🔍 Key Takeaways

1.  **Memory Efficiency (100x Lead):** NeuralGraphDB uses **1%** of the memory required by Neo4j. This is the direct result of the native Rust CSR architecture which avoids the massive overhead of the Java JVM and pointer-heavy object graphs.
2.  **Ingestion Throughput (170x-300x Lead):** While Neo4j and FalkorDB struggle with transactional overhead and index updates during large-scale edge creation, NeuralGraphDB's bulk loading remains nearly instantaneous.
3.  **Query Maturity:** Neo4j and FalkorDB excel in complex analytical queries (aggregations/grouping) due to their mature cost-based query optimizers. NeuralGraphDB is competitive in traversal speed but has clear room for improvement in high-level analytical query optimization.
4.  **Hardware Friendliness:** NeuralGraphDB is the only candidate that could realistically run on resource-constrained Edge devices or embedded systems at this scale.
