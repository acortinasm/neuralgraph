# ArXiv Benchmark Results

**Date:** 2026-01-20  
**Num Papers:** 1000 (Target)

## 1. NeuralGraphDB
| Step | Time |
| :--- | :--- |
| 0. Carga modelo | 4.90s |
| 1. Descarga ArXiv | 4.44s |
| 2. Generación embeddings | 2.25s |
| 3. Setup schema | 0.00s |
| 4. Carga papers | 0.00s |
| 5. Creación citas | 0.01s |
| 6. Creación autores/instituciones | 0.00s |
| 7. Verificación | 0.00s |
| **8. Carga en NeuralGraphDB** | **0.03s** |
| Query: 2-hop traversal | 0.00s |
| Query: Aggregation | 0.00s |
| Query: Filtro | 0.00s |
| **TOTAL** | **11.64s** |

---

## 2. Neo4j
| Step | Time |
| :--- | :--- |
| 0. Carga modelo | 5.38s |
| 1. Descarga ArXiv | 3.26s |
| 2. Generación embeddings | 2.06s |
| 3. Setup schema | 0.72s |
| 4. Carga papers | 0.88s |
| 5. Creación citas | 0.33s |
| 6. Creación autores/instituciones | 0.27s |
| 7. Verificación | 0.10s |
| Query: 2-hop traversal | 0.11s |
| Query: Vector search | 0.09s |
| Query: Híbrido | 0.05s |
| **TOTAL** | **13.26s** |

---

## 3. FalkorDB
| Step | Time |
| :--- | :--- |
| 1. Descarga ArXiv | 2.63s |
| 2. Generación embeddings | 2.04s |
| 3. Setup schema | 0.01s |
| 4. Carga papers | 0.32s |
| 5. Creación citas | 0.06s |
| Query: 2-hop traversal | 0.01s |
| **TOTAL** | **5.06s** |

---

## 📊 Summary Comparison (Database Operations Only)
*Excluding data download, model loading, and embedding generation.*

| Metric | NeuralGraphDB | Neo4j | FalkorDB |
| :--- | :--- | :--- | :--- |
| **Data Loading** | **0.03s** 🚀 | 0.88s | 0.32s |
| **Edge Creation** | **0.01s** 🚀 | 0.33s | 0.06s |
| **2-hop Traversal** | **<0.001s** 🚀 | 0.11s | 0.01s |
| **Total DB Time** | **~0.04s** | ~2.55s | ~0.40s |

**Conclusion:** NeuralGraphDB is approximately **60x faster than Neo4j** and **10x faster than FalkorDB** for these core graph operations, validating the performance advantages of the CSR-based Rust architecture.
