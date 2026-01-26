# **Planificación Detallada: NeuralGraphDB**

Metodología: Agile / Scrum (Sprints de 2 semanas)
Duración Total Estimada: 29 Meses (73 Sprints)
Versión: 7.0
Última Actualización: 2026-01-26

**Criterio de Priorización:** Infraestructura → Performance → AI

---

## **Fase 1: El Motor Algebraico (Meses 1-4)** ✅ COMPLETADA

**Objetivo:** Construir el núcleo de alto rendimiento. Lograr que una consulta MATCH se ejecute mediante multiplicación de matrices.

*(Sprints 1-12 Completados)*

---

## **Fase 2: Suite GraphRAG (Meses 5-8)** ✅ COMPLETADA

**Objetivo:** Diferenciación de mercado. Implementar las "Killer Features" de IA (Clustering y Generación).

*(Sprints 13-20 Completados)*

---

## **Fase 3: Infraestructura de Base de Datos (Meses 9-12)** ✅ COMPLETADA

**Objetivo:** Completar las capacidades fundamentales de base de datos necesarias para uso en producción.

*(Sprints 21-32 Completados)*

---

## **Fase 4: Conformidad Estándar Cypher (Meses 13-14)** ✅ COMPLETADA

**Objetivo:** Alcanzar paridad funcional con Cypher estándar para soportar consultas complejas.

*(Sprints 33-36 Completados)*

---

## **Fase 5: Academic Core (v1.0 Architecture) (Meses 15-18)** 🔄 EN PROGRESO

**Objetivo:** Optimizar para el paper "NeuralGraphDB". Transición a PCSR y Kernels de Álgebra Lineal.

### **Sprints de Fase 5**

| Sprint | Foco Principal | Entregable Clave | Estado |
| :---- | :---- | :---- | :---- |
| **Sprint 37** | Packed Memory Array | Implementación PMA O(log^2 N). | ✅ |
| **Sprint 38** | Unified PCSR | Refactor storage a PCSRMatrix. | ✅ |
| **Sprint 39** | Neural Semirings | Algebra crate & SpMV search. | ✅ |
| **Sprint 40** | SIMD Acceleration | AVX-512/NEON optimizations. | ✅ |
| **Sprint 41** | Native Leiden | Parallel Leiden on PCSR. | ✅ |
| **Sprint 42** | Context Summary | `SUMMARIZE` clause & subgraphs. | ✅ |
| **Sprint 43** | Python Client 2.0 | Pipeline `MATCH...CREATE` & fixes. | ✅ |
| **Sprint 44** | Validation (LDBC) | Benchmarks vs Neo4j/FalkorDB. | 🔄 |
| **Sprint 45** | Read Latency | Arrow Flight implementation. | ✅ |
| **Sprint 46** | Core Stability | Parser fixes & Cypher compliance. | ✅ |
| **Sprint 47** | Vector Scale | 1M Vectors Optimization & LSM-VEC. | ✅ |

---

## **Fase 6: Infraestructura Distribuida (Meses 19-21)** 📅 PLANIFICADA

**Objetivo:** Completar sistema distribuido production-grade con alta disponibilidad.

**Criterio de Priorización:** INFRAESTRUCTURA (fundamentos para escala y producción)

### **Épicas de Fase 6**

#### **Épica 16: Distribución & Replicación**

Alta disponibilidad mediante consenso distribuido.

* **US-16.1:** Como DBA, quiero replicación de datos mediante algoritmo Raft para tolerancia a fallos.
* **US-16.2:** Como Sistema, quiero discovery automático de nodos y routing de queries al líder.
* **US-16.3:** Como Analista, quiero consultar datos históricos (`AT TIME`) mediante time-travel.

#### **Épica 17: Particionamiento Horizontal**

Escalabilidad más allá de un solo nodo.

* **US-17.1:** Como Sistema, quiero particionamiento de grafos (vertex-cut o edge-cut) para sharding horizontal.
* **US-17.2:** Como Sistema, quiero metadatos completos en embeddings: modelo de origen, métrica de distancia, timestamp.
* **US-17.3:** Como Sistema, quiero soporte para multi-aristas paralelas con numeración de puertos.

#### **Épica 18: Transacciones ACID & MVCC** ✅ COMPLETADA
* **US-18.1:** ✅ Como Usuario, quiero transacciones multi-query (`BEGIN`, `COMMIT`, `ROLLBACK`).
* **US-18.2:** ✅ Como Sistema, quiero aislamiento de snapshot (MVCC) para lecturas concurrentes.

### **Sprints de Fase 6**

| Sprint | Foco Principal | Entregable Clave | Estado | Categoría |
| :---- | :---- | :---- | :---- | :---- |
| **Sprint 50** | Transaction Manager | ACID (Begin, Commit, Rollback). | ✅ | Infra |
| **Sprint 51** | MVCC | Snapshot Isolation. | ✅ | Infra |
| **Sprint 52** | **Distributed Raft** | Replicación Multi-nodo con consenso. | 📅 | Infra |
| **Sprint 53** | **Cluster Management** | Node Discovery, Leader Routing, Health Checks. | 📅 | Infra |
| **Sprint 54** | **Time-Travel Queries** | `AT TIME` para consultas históricas sobre MVCC. | 📅 | Infra |
| **Sprint 55** | **Graph Sharding** | Particionamiento vertex-cut/edge-cut para escala horizontal. | 📅 | Infra |
| **Sprint 56** | **Embedding Metadata** | Modelo origen, múltiples métricas (coseno, euclidiana, dot). | 📅 | Infra |
| **Sprint 57** | **Port Numbering** | Identificadores únicos para multi-aristas paralelas. | 📅 | Infra |

---

## **Fase 7: Rendimiento y Escala (Meses 22-23)** 📅 PLANIFICADA

**Objetivo:** Optimización para grafos de billones de nodos con latencia sub-segundo.

**Criterio de Priorización:** PERFORMANCE (optimización y benchmarking)

### **Épicas de Fase 7**

#### **Épica 19: Validación y Benchmarking**

Demostrar rendimiento competitivo vs Neo4j/FalkorDB.

* **US-19.1:** Como Investigador, quiero benchmarks LDBC validados para el paper académico.
* **US-19.2:** Como Sistema, quiero optimización de queries para workloads OLTP y OLAP.

#### **Épica 20: Optimización de Memoria**

Reducción de footprint para grafos masivos.

* **US-20.1:** Como Sistema, quiero cuantización dinámica (Flash Quantization) f32→int8 para reducir memoria 4x.
* **US-20.2:** Como Sistema, quiero búsqueda vectorial distribuida con fusión de resultados paralela.

#### **Épica 21: Algoritmos de Grafos Optimizados**

Primitivas de alto rendimiento para RAG.

* **US-21.1:** Como Usuario, quiero PageRank Personalizado (PPR) optimizado para expansión local desde nodo semilla.

### **Sprints de Fase 7**

| Sprint | Foco Principal | Entregable Clave | Estado | Categoría |
| :---- | :---- | :---- | :---- | :---- |
| **Sprint 58** | **LDBC Validation** | Benchmarks completos vs Neo4j/FalkorDB para paper. | 📅 | Perf |
| **Sprint 59** | **Flash Quantization** | Cuantización dinámica f32→int8/binary, 4x memoria. | 📅 | Perf |
| **Sprint 60** | **Distributed Vector Search** | Búsqueda paralela en múltiples nodos + fusión. | 📅 | Perf |
| **Sprint 61** | **Personalized PageRank** | PPR optimizado con sparse matrix operations. | 📅 | Perf |

---

## **Fase 8: GraphRAG Completo (Meses 24-26)** 📅 PLANIFICADA

**Objetivo:** Capacidades completas de IA nativa para RAG avanzado. Basado en análisis de estado del arte (ver `docs/estado_del_arte.md`).

**Criterio de Priorización:** AI (diferenciación de producto)

### **Épicas de Fase 8**

#### **Épica 22: GraphRAG Global Search**

Habilitar búsqueda global sobre comunidades para consultas temáticas amplias.

* **US-22.1:** Como Sistema, quiero generar resúmenes de comunidades automáticamente con LLM post-Leiden.
* **US-22.2:** Como Sistema, quiero indexar los resúmenes de comunidad en VectorIndex separado.
* **US-22.3:** Como Usuario, quiero ejecutar `CALL neural.globalSearch($query)` para búsqueda sobre comunidades.

#### **Épica 23: Hybrid Retrieval**

Fusionar resultados de búsqueda vectorial y estructural.

* **US-23.1:** Como Usuario, quiero combinar VectorSearch y GraphTraversal con Weighted Reciprocal Rank Fusion (wRRF).
* **US-23.2:** Como Sistema, quiero `MERGE ON SIMILARITY` para deduplicación semántica durante ingesta.
* **US-23.3:** Como Usuario, quiero selección automática de Core Chunks basada en centralidad (degree, betweenness).

#### **Épica 24: Vector Operations Avanzadas**

Operaciones vectoriales especializadas para análisis de grafos.

* **US-24.1:** Como Usuario, quiero Vector Similarity Join para top-k pares con restricciones de patrón.
* **US-24.2:** Como Usuario, quiero embeddings en aristas para búsqueda semántica sobre relaciones.

### **Sprints de Fase 8**

| Sprint | Foco Principal | Entregable Clave | Estado | Categoría |
| :---- | :---- | :---- | :---- | :---- |
| **Sprint 62** | **Community Summaries** | Generación automática de resúmenes con LLM post-Leiden. | 📅 | AI |
| **Sprint 63** | **Community Vector Index** | VectorIndex sobre resúmenes + `globalSearch()` procedure. | 📅 | AI |
| **Sprint 64** | **Core Chunks Selection** | Selección por centralidad + grafo de co-ocurrencia sin LLM. | 📅 | AI |
| **Sprint 65** | **Hybrid Retrieval (wRRF)** | Weighted Reciprocal Rank Fusion de vector + graph. | 📅 | AI |
| **Sprint 66** | **Semantic Ingestion** | `MERGE ON SIMILARITY` para deduplicación automática. | 📅 | AI |
| **Sprint 67** | **Vector Similarity Join** | Operador top-k pairs con restricciones de grafo. | 📅 | AI |
| **Sprint 68** | **Edge Embeddings** | VectorIndex sobre aristas + búsqueda semántica de relaciones. | 📅 | AI |

---

## **Fase 9: Ecosistema y AI Avanzada (Meses 27-29)** 📅 PLANIFICADA

**Objetivo:** UX empresarial, conectividad legacy, y capacidades AI especializadas.

### **Épicas de Fase 9**

#### **Épica 25: Puente Legacy (SQL)**

Permitir la coexistencia con sistemas relacionales.

* **US-25.1:** Como Ingeniero de Datos, quiero sincronizar tablas SQL a Nodos/Aristas automáticamente.
* **US-25.2:** Como Sistema, quiero un conector Python robusto (SQLAlchemy -> NeuralGraph).

#### **Épica 26: Neural Dashboard**

Visualización y gestión para usuarios finales.

* **US-26.1:** Como Analista, quiero visualizar el grafo interactivamente ("Reef view").
* **US-26.2:** Como Admin, quiero ver métricas de salud del sistema en un dashboard web.

#### **Épica 27: GNN Native Adaptations**

Mecanismos para Redes Neuronales de Grafos provablemente potentes.

* **US-27.1:** Como Sistema, quiero Paso de Mensajes Inverso (Reverse Message Passing) para flujos de salida.
* **US-27.2:** Como Sistema, quiero Identificadores de Ego para romper simetrías y detectar ciclos.

#### **Épica 28: Multimodal Support**

Soporte para múltiples modalidades (texto, imagen, audio).

* **US-28.1:** Como Usuario, quiero índices HNSW separados optimizados por modalidad.
* **US-28.2:** Como Sistema, quiero particionamiento consciente de modalidad para evitar sesgos cross-modal.

### **Sprints de Fase 9**

| Sprint | Foco Principal | Entregable Clave | Estado | Categoría |
| :---- | :---- | :---- | :---- | :---- |
| **Sprint 69** | **SQL Bridge** | Herramienta CLI/Python para ETL SQL->Graph. | 📅 | Infra |
| **Sprint 70** | **Neural Dashboard** | UI Web (React/WASM) para visualización. | 📅 | Infra |
| **Sprint 71** | **Reverse Message Passing** | Operador GNN con CSC para flujos entrantes. | 📅 | AI |
| **Sprint 72** | **Ego Identifiers** | Breaking symmetries para detección de ciclos/fraude. | 📅 | AI |
| **Sprint 73** | **Multimodal Indexes** | HNSW separados por modalidad (texto, imagen, audio). | 📅 | AI |

---

## **Resumen de Priorización: Infraestructura → Performance → AI**

### Vista Consolidada por Categoría

| Categoría | Sprints | Fases | Total |
| :---- | :---- | :---- | :---- |
| **Infraestructura** | 52-57, 69-70 | Fase 6, Fase 9 | 10 sprints |
| **Performance** | 58-61 | Fase 7 | 4 sprints |
| **AI** | 62-68, 71-73 | Fase 8, Fase 9 | 10 sprints |

### Dependencias Críticas

```
Fase 6 (Infra)          Fase 7 (Perf)         Fase 8 (AI)           Fase 9 (Ecosystem)
─────────────────────────────────────────────────────────────────────────────────────
Raft (52)
  └─► Cluster (53)
        └─► Time-Travel (54)
              └─► Sharding (55) ──────► Distributed Search (60)
                                              │
Embedding Meta (56) ─────────────────────────►├─► Edge Embeddings (68)
                                              │
Port Numbers (57) ───────────────────────────►│
                                              │
              Flash Quant (59) ──────────────►│
                                              │
              PPR (61) ──────────────────────►├─► Core Chunks (64)
                                              │     └─► wRRF (65)
                                              │
                              Community Sum (62)
                                └─► Community Vec (63)
                                      └─► globalSearch()
                                                        SQL Bridge (69)
                                                        Dashboard (70)
                                                        GNN Ops (71-72)
                                                        Multimodal (73)
```

### Hitos Clave

| Hito | Sprint | Entregable |
| :---- | :---- | :---- |
| **HA Cluster** | 53 | Cluster Raft con failover automático |
| **Billion Scale** | 55 | Sharding horizontal operativo |
| **Paper Ready** | 58 | Benchmarks LDBC validados |
| **GraphRAG v2** | 63 | Global Search sobre comunidades |
| **Enterprise Ready** | 70 | Dashboard + SQL Bridge |