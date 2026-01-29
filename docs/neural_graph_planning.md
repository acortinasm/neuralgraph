# **Planificación Detallada: NeuralGraphDB**

Metodología: Agile / Scrum (Sprints de 2 semanas)
Duración Total Estimada: 34 Meses (88 Sprints)
Versión: 8.0
Última Actualización: 2026-01-29

**Criterio de Priorización:** Paridad Competitiva → Infraestructura → Performance → AI

> **Nota v8.0:** Reestructuración basada en análisis competitivo vs FalkorDB.
> Prioriza features críticas para adopción de mercado (Full-Text, LangChain, LlamaIndex).

> **Sprint 61 Completado:** Distributed Vector Search con scatter-gather, replica failover,
> Prometheus metrics y gRPC server.

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
| **Sprint 44** | Validation (LDBC) | Benchmarks vs Neo4j/FalkorDB. | ✅ |
| **Sprint 45** | Read Latency | Arrow Flight implementation. | ✅ |
| **Sprint 46** | Core Stability | Parser fixes & Cypher compliance. | ✅ |
| **Sprint 47** | Vector Scale | 1M Vectors Optimization & LSM-VEC. | ✅ |

---

## **Fase 6: Infraestructura Distribuida (Meses 19-21)** 🔄 EN PROGRESO

**Objetivo:** Completar sistema distribuido production-grade con alta disponibilidad.

**Criterio de Priorización:** INFRAESTRUCTURA (fundamentos para escala y producción)

### **Épicas de Fase 6**

#### **Épica 16: Distribución & Replicación** 🔄 EN PROGRESO

Alta disponibilidad mediante consenso distribuido.

* **US-16.1:** ✅ Como DBA, quiero replicación de datos mediante algoritmo Raft para tolerancia a fallos.
* **US-16.2:** 📅 Como Sistema, quiero discovery automático de nodos y routing de queries al líder.
* **US-16.3:** ✅ Como Analista, quiero consultar datos históricos (`AT TIME`) mediante time-travel.

#### **Épica 17: Particionamiento Horizontal** ✅ COMPLETADA

Escalabilidad más allá de un solo nodo.

* **US-17.1:** ✅ Como Sistema, quiero particionamiento de grafos (vertex-cut o edge-cut) para sharding horizontal.
* **US-17.2:** ✅ Como Sistema, quiero metadatos completos en embeddings: modelo de origen, métrica de distancia, timestamp.
* **US-17.3:** ✅ Como Sistema, quiero soporte para multi-aristas paralelas con numeración de puertos.

#### **Épica 18: Transacciones ACID & MVCC** ✅ COMPLETADA
* **US-18.1:** ✅ Como Usuario, quiero transacciones multi-query (`BEGIN`, `COMMIT`, `ROLLBACK`).
* **US-18.2:** ✅ Como Sistema, quiero aislamiento de snapshot (MVCC) para lecturas concurrentes.

### **Sprints de Fase 6**

| Sprint | Foco Principal | Entregable Clave | Estado | Categoría |
| :---- | :---- | :---- | :---- | :---- |
| **Sprint 50** | Transaction Manager | ACID (Begin, Commit, Rollback). | ✅ | Infra |
| **Sprint 51** | MVCC | Snapshot Isolation. | ✅ | Infra |
| **Sprint 52** | **Distributed Raft** | Replicación Multi-nodo con consenso. | ✅ | Infra |
| **Sprint 53** | **Cluster Management** | Node Discovery, Leader Routing, Health Checks. | 📅 | Infra |
| **Sprint 54** | **Time-Travel Queries** | `AT TIME` para consultas históricas sobre MVCC. | ✅ | Infra |
| **Sprint 55** | **Graph Sharding** | Particionamiento vertex-cut/edge-cut para escala horizontal. | ✅ | Infra |
| **Sprint 56** | **Embedding Metadata** | Modelo origen, múltiples métricas (coseno, euclidiana, dot). | ✅ | Infra |
| **Sprint 57** | **Port Numbering** | Identificadores únicos para multi-aristas paralelas. | ✅ | Infra |

---

## **Fase 7: Paridad Competitiva y Escala (Meses 22-25)** 📅 PLANIFICADA

**Objetivo:** Cerrar gaps críticos vs FalkorDB. Habilitar adopción de mercado con Full-Text Search e integraciones de frameworks AI.

**Criterio de Priorización:** PARIDAD COMPETITIVA (viabilidad de mercado)

### **Épicas de Fase 7**

#### **Épica 19: Validación y Benchmarking** ✅ COMPLETADA

Demostrar rendimiento competitivo vs Neo4j/FalkorDB.

* **US-19.1:** ✅ Como Investigador, quiero benchmarks LDBC validados para el paper académico.
* **US-19.2:** ✅ Como Sistema, quiero query latency <0.35ms para competir con FalkorDB. **Resultado: 0.72ms → 0.35ms (51% mejora)**

#### **Épica 20: Búsqueda Vectorial Distribuida** ✅ COMPLETADA

* **US-20.1:** ✅ Como Sistema, quiero cuantización dinámica (Flash Quantization) f32→int8 para reducir memoria 4x.
* **US-20.2:** ✅ Como Sistema, quiero búsqueda vectorial distribuida con fusión de resultados paralela. **Implementado: Scatter-gather, replica failover, Prometheus metrics, gRPC server.**

#### **Épica 21: Full-Text Search** 📅 NUEVA (Análisis Competitivo)

Crítico para GraphRAG y paridad con FalkorDB (RediSearch).

* **US-21.1:** 📅 Como Usuario, quiero crear índices full-text sobre propiedades de nodos.
* **US-21.2:** 📅 Como Usuario, quiero búsqueda full-text con stemming y stopwords.
* **US-21.3:** 📅 Como Usuario, quiero fuzzy matching y búsqueda fonética.

#### **Épica 22: Tipos de Datos Avanzados** 📅 NUEVA (Análisis Competitivo)

FalkorDB soporta Array y Map; crítico para casos de uso comunes.

* **US-22.1:** 📅 Como Usuario, quiero tipo de dato Array nativo en propiedades.
* **US-22.2:** 📅 Como Usuario, quiero tipo de dato Map/JSON nativo en propiedades.

#### **Épica 23: Integraciones de Frameworks AI** 📅 NUEVA (Análisis Competitivo)

Expectativa de mercado. FalkorDB tiene integración nativa.

* **US-23.1:** 📅 Como Desarrollador, quiero integración nativa con LangChain (FalkorDBGraph equivalent).
* **US-23.2:** 📅 Como Desarrollador, quiero integración nativa con LlamaIndex (PropertyGraphStore).

### **Sprints de Fase 7**

| Sprint | Foco Principal | Entregable Clave | Estado | Categoría |
| :---- | :---- | :---- | :---- | :---- |
| **Sprint 58** | **LDBC Validation** | Benchmarks completos vs Neo4j/FalkorDB para paper. | ✅ | Perf |
| **Sprint 59** | **Query Latency Optimization** | Zero-copy bindings, direct serialization. **51% mejora** | ✅ | Perf |
| **Sprint 60** | **Flash Quantization** | Cuantización f32→int8/binary, 4x-32x memoria. | ✅ | Perf |
| **Sprint 61** | **Distributed Vector Search** | Scatter-gather + replica failover + Prometheus metrics + gRPC server. | ✅ | Perf |
| **Sprint 62** | **Full-Text Index (Core)** | Índice invertido con tantivy. Stemming básico. | 📅 | **P0** |
| **Sprint 63** | **Full-Text Search (Advanced)** | Fuzzy matching, phonetic search, multi-language. | 📅 | **P0** |
| **Sprint 64** | **Array/Map Data Types** | Tipos nativos Array y Map/JSON en propiedades. | 📅 | **P0** |
| **Sprint 65** | **LangChain Integration** | NeuralGraphStore, GraphCypherQAChain adapter. | 📅 | **P0** |
| **Sprint 66** | **LlamaIndex Integration** | PropertyGraphStore, Knowledge Graph Index. | 📅 | **P0** |

---

## **Fase 8: Algoritmos y Performance (Meses 26-27)** 📅 PLANIFICADA

**Objetivo:** Completar algoritmos de grafos para paridad competitiva y habilitar GraphRAG avanzado.

**Criterio de Priorización:** PERFORMANCE (algoritmos fundamentales)

### **Épicas de Fase 8**

#### **Épica 24: Algoritmos de Grafos Optimizados**

Primitivas de alto rendimiento para RAG y análisis. FalkorDB tiene PageRank, WCC, Betweenness.

* **US-24.1:** 📅 Como Usuario, quiero PageRank Personalizado (PPR) optimizado para expansión local.
* **US-24.2:** 📅 Como Usuario, quiero Weakly Connected Components (WCC) para análisis de componentes.
* **US-24.3:** 📅 Como Usuario, quiero Betweenness Centrality para identificar nodos críticos.
* **US-24.4:** 📅 Como Usuario, quiero All Shortest Paths entre dos nodos.

### **Sprints de Fase 8**

| Sprint | Foco Principal | Entregable Clave | Estado | Categoría |
| :---- | :---- | :---- | :---- | :---- |
| **Sprint 67** | **Personalized PageRank** | PPR optimizado con sparse matrix operations. | 📅 | Perf |
| **Sprint 68** | **Graph Algorithms Pack** | WCC + Betweenness Centrality con PCSR. | 📅 | Perf |

---

## **Fase 9: GraphRAG Completo (Meses 28-31)** 📅 PLANIFICADA

**Objetivo:** Capacidades completas de IA nativa para RAG avanzado. Basado en análisis de estado del arte.

**Criterio de Priorización:** AI (diferenciación de producto)

### **Épicas de Fase 9**

#### **Épica 25: GraphRAG Global Search**

Habilitar búsqueda global sobre comunidades para consultas temáticas amplias.

* **US-25.1:** 📅 Como Sistema, quiero generar resúmenes de comunidades automáticamente con LLM post-Leiden.
* **US-25.2:** 📅 Como Sistema, quiero indexar los resúmenes de comunidad en VectorIndex separado.
* **US-25.3:** 📅 Como Usuario, quiero ejecutar `CALL neural.globalSearch($query)` para búsqueda sobre comunidades.

#### **Épica 26: Natural Language Interface** 📅 NUEVA (Análisis Competitivo)

FalkorDB GraphRAG-SDK permite consultas en lenguaje natural.

* **US-26.1:** 📅 Como Usuario, quiero ejecutar consultas en lenguaje natural que se traduzcan a NGQL.
* **US-26.2:** 📅 Como Usuario, quiero sesiones de chat con contexto persistente sobre el grafo.

#### **Épica 27: Hybrid Retrieval**

Fusionar resultados de búsqueda vectorial y estructural.

* **US-27.1:** 📅 Como Usuario, quiero combinar VectorSearch y GraphTraversal con Weighted Reciprocal Rank Fusion (wRRF).
* **US-27.2:** 📅 Como Sistema, quiero `MERGE ON SIMILARITY` para deduplicación semántica durante ingesta.
* **US-27.3:** 📅 Como Usuario, quiero selección automática de Core Chunks basada en centralidad.

#### **Épica 28: Vector Operations Avanzadas**

Operaciones vectoriales especializadas para análisis de grafos.

* **US-28.1:** 📅 Como Usuario, quiero Vector Similarity Join para top-k pares con restricciones de patrón.
* **US-28.2:** 📅 Como Usuario, quiero embeddings en aristas para búsqueda semántica sobre relaciones.

### **Sprints de Fase 9**

| Sprint | Foco Principal | Entregable Clave | Estado | Categoría |
| :---- | :---- | :---- | :---- | :---- |
| **Sprint 69** | **Community Summaries** | Generación automática de resúmenes con LLM post-Leiden. | 📅 | AI |
| **Sprint 70** | **Community Vector Index** | VectorIndex sobre resúmenes + `globalSearch()` procedure. | 📅 | AI |
| **Sprint 71** | **Natural Language Queries** | Text-to-NGQL con LLM + schema context. | 📅 | AI |
| **Sprint 72** | **Core Chunks Selection** | Selección por centralidad + grafo de co-ocurrencia. | 📅 | AI |
| **Sprint 73** | **Hybrid Retrieval (wRRF)** | Weighted Reciprocal Rank Fusion de vector + graph. | 📅 | AI |
| **Sprint 74** | **Semantic Ingestion** | `MERGE ON SIMILARITY` para deduplicación automática. | 📅 | AI |
| **Sprint 75** | **Vector Similarity Join** | Operador top-k pairs con restricciones de grafo. | 📅 | AI |
| **Sprint 76** | **Edge Embeddings** | VectorIndex sobre aristas + búsqueda semántica. | 📅 | AI |
| **Sprint 77** | **Chat Sessions** | Sesiones de chat persistentes con contexto de grafo. | 📅 | AI |

---

## **Fase 10: Enterprise y Ecosistema (Meses 32-35)** 📅 PLANIFICADA

**Objetivo:** Capacidades enterprise, SDKs adicionales, y conectividad legacy.

**Criterio de Priorización:** ECOSISTEMA (adopción enterprise)

### **Épicas de Fase 10**

#### **Épica 29: Multi-Tenancy** 📅 NUEVA (Análisis Competitivo)

FalkorDB soporta 10,000+ tenants por instancia. Crítico para SaaS.

* **US-29.1:** 📅 Como Operador, quiero múltiples tenants aislados en una sola instancia.
* **US-29.2:** 📅 Como Sistema, quiero zero overhead entre tenants.
* **US-29.3:** 📅 Como Admin, quiero gestión centralizada de tenants.

#### **Épica 30: NGQL Improvements** 📅 NUEVA (Análisis Competitivo)

Paridad con Cypher para queries complejas.

* **US-30.1:** 📅 Como Usuario, quiero Pattern Comprehension en NGQL.
* **US-30.2:** 📅 Como Usuario, quiero List Comprehension en NGQL.
* **US-30.3:** 📅 Como Usuario, quiero `LOAD CSV` para importación de datos.
* **US-30.4:** 📅 Como Admin, quiero Slow Query Log para diagnóstico.

#### **Épica 31: SDKs Adicionales** 📅 NUEVA (Análisis Competitivo)

FalkorDB tiene SDKs en Python, JS, Java, Go, C#.

* **US-31.1:** 📅 Como Desarrollador, quiero SDK oficial de Node.js/TypeScript.
* **US-31.2:** 📅 Como Desarrollador, quiero SDK oficial de Java.
* **US-31.3:** 📅 Como Desarrollador, quiero SDK oficial de Go.

#### **Épica 32: Puente Legacy (SQL)**

Permitir la coexistencia con sistemas relacionales.

* **US-32.1:** 📅 Como Ingeniero de Datos, quiero sincronizar tablas SQL a Nodos/Aristas automáticamente.
* **US-32.2:** 📅 Como Sistema, quiero un conector Python robusto (SQLAlchemy -> NeuralGraph).

#### **Épica 33: Neural Dashboard**

Visualización y gestión para usuarios finales.

* **US-33.1:** 📅 Como Analista, quiero visualizar el grafo interactivamente ("Reef view").
* **US-33.2:** 📅 Como Admin, quiero ver métricas de salud del sistema en un dashboard web.

### **Sprints de Fase 10**

| Sprint | Foco Principal | Entregable Clave | Estado | Categoría |
| :---- | :---- | :---- | :---- | :---- |
| **Sprint 78** | **Multi-Tenancy (Core)** | Aislamiento de tenants, namespace separation. | 📅 | Infra |
| **Sprint 79** | **Multi-Tenancy (Scale)** | Zero-overhead, gestión centralizada. | 📅 | Infra |
| **Sprint 80** | **NGQL Improvements** | Pattern/List comprehension, LOAD CSV. | 📅 | Infra |
| **Sprint 81** | **Node.js SDK** | Cliente TypeScript oficial con tipos. | 📅 | SDK |
| **Sprint 82** | **SQL Bridge** | Herramienta CLI/Python para ETL SQL->Graph. | 📅 | Infra |
| **Sprint 83** | **Neural Dashboard** | UI Web (React/WASM) para visualización. | 📅 | Infra |
| **Sprint 84** | **Java/Go SDKs** | Clientes oficiales para JVM y Go. | 📅 | SDK |
| **Sprint 85** | **Slow Query Log + All Shortest Paths** | Diagnóstico operacional + algoritmo. | 📅 | Infra |

---

## **Fase 11: AI Avanzada (Meses 36-38)** 📅 PLANIFICADA

**Objetivo:** Capacidades AI especializadas y soporte multimodal.

**Criterio de Priorización:** AI AVANZADA (diferenciación de largo plazo)

### **Épicas de Fase 11**

#### **Épica 34: GNN Native Adaptations**

Mecanismos para Redes Neuronales de Grafos provablemente potentes.

* **US-34.1:** 📅 Como Sistema, quiero Paso de Mensajes Inverso (Reverse Message Passing) para flujos de salida.
* **US-34.2:** 📅 Como Sistema, quiero Identificadores de Ego para romper simetrías y detectar ciclos.

#### **Épica 35: Multimodal Support**

Soporte para múltiples modalidades (texto, imagen, audio).

* **US-35.1:** 📅 Como Usuario, quiero índices HNSW separados optimizados por modalidad.
* **US-35.2:** 📅 Como Sistema, quiero particionamiento consciente de modalidad para evitar sesgos cross-modal.

### **Sprints de Fase 11**

| Sprint | Foco Principal | Entregable Clave | Estado | Categoría |
| :---- | :---- | :---- | :---- | :---- |
| **Sprint 86** | **Reverse Message Passing** | Operador GNN con CSC para flujos entrantes. | 📅 | AI |
| **Sprint 87** | **Ego Identifiers** | Breaking symmetries para detección de ciclos/fraude. | 📅 | AI |
| **Sprint 88** | **Multimodal Indexes** | HNSW separados por modalidad (texto, imagen, audio). | 📅 | AI |

---

## **Fase 12: Extensiones Futuras (Backlog)** 📅 BACKLOG

**Objetivo:** Features de baja prioridad para considerar post-v1.0.

**Criterio de Priorización:** P3 - NICE TO HAVE

### **Épicas de Fase 12**

#### **Épica 36: Compatibilidad Extendida** (P3)

* **US-36.1:** 📅 Como Usuario, quiero índice Geospatial para queries de ubicación.
* **US-36.2:** 📅 Como Usuario, quiero protocolo Bolt para migración desde Neo4j.
* **US-36.3:** 📅 Como Usuario, quiero cláusula FOREACH en NGQL.
* **US-36.4:** 📅 Como Desarrollador, quiero SDK oficial de C#.

#### **Épica 37: Extensibilidad** (P3)

* **US-37.1:** 📅 Como Desarrollador, quiero User-Defined Functions en JavaScript/WASM.

### **Backlog (Sin Sprint Asignado)**

| Feature | Prioridad | Rationale |
| :---- | :---- | :---- |
| Geospatial Index | P3 | Nicho, no crítico para GraphRAG |
| Bolt Protocol | P3 | Solo útil para migración Neo4j |
| FOREACH Clause | P3 | Baja demanda, workarounds disponibles |
| C# SDK | P3 | Mercado limitado para graph DBs |
| JavaScript UDFs | P3 | Complejidad alta, WASM preferible |

---

## **Resumen de Priorización: Paridad Competitiva → Infra → Perf → AI**

### Vista Consolidada por Categoría

| Categoría | Sprints | Fases | Total |
| :---- | :---- | :---- | :---- |
| **Paridad Competitiva (P0)** | 62-66 | Fase 7 | 5 sprints |
| **Infraestructura** | 53, 78-80, 82-83, 85 | Fase 6, 10 | 7 sprints |
| **Performance** | 58-61, 67-68 | Fase 7, 8 | 6 sprints |
| **AI** | 69-77, 86-88 | Fase 9, 11 | 12 sprints |
| **SDKs/Ecosystem** | 81, 84 | Fase 10 | 2 sprints |
| **Backlog (P3)** | TBD | Fase 12 | ~5 features |

### Dependencias Críticas

```
Fase 6 (Infra)     Fase 7 (Competitive)    Fase 8 (Perf)      Fase 9 (AI)         Fase 10-11
──────────────────────────────────────────────────────────────────────────────────────────────
Raft (52) ✅
  └─► Cluster (53)
        └─► Time-Travel (54) ✅
              └─► Sharding (55) ✅ ───► Distributed Search (61) ✅
                                              │
Flash Quant (60) ✅ ─────────────────────────►│
                                              │
                    Full-Text (62-63) ────────┼───────────────────► NL Queries (71)
                                              │                          │
                    Array/Map (64) ──────────►│                          │
                                              │                          │
                    LangChain (65) ──────────►├─────────► Hybrid wRRF (73)
                    LlamaIndex (66) ─────────►│                │
                                              │                └─► Chat Sessions (77)
                              PPR (67) ───────┼─► Core Chunks (72)
                              WCC/Betw (68)───┤
                                              │
                              Community Sum (69)
                                └─► Community Vec (70)
                                      └─► globalSearch()
                                                                    Multi-Tenancy (78-79)
                                                                    NGQL Improve (80)
                                                                    Node.js SDK (81)
                                                                    Dashboard (83)
                                                                    GNN Ops (86-87)
                                                                    Multimodal (88)
```

### Hitos Clave

| Hito | Sprint | Entregable | Fecha Est. |
| :---- | :---- | :---- | :---- |
| **HA Cluster** | 53 | Cluster Raft con failover automático | Mes 22 |
| **Paper Ready** | 58-60 | Benchmarks LDBC + Flash Quantization | ✅ Completado |
| **🎯 Market Ready** | 66 | Full-Text + LangChain + LlamaIndex | Mes 25 |
| **GraphRAG v2** | 70 | Global Search sobre comunidades | Mes 29 |
| **Enterprise Ready** | 79 | Multi-Tenancy + Dashboard | Mes 33 |
| **v1.0 Release** | 88 | Feature complete | Mes 38 |

### Features Nuevas vs Análisis Competitivo FalkorDB

| Feature | Prioridad | Sprint | Gap Cerrado |
| :---- | :---- | :---- | :---- |
| Full-Text Search | **P0** | 62-63 | ✓ RediSearch equivalente |
| Array/Map Types | **P0** | 64 | ✓ Tipos de datos |
| LangChain | **P0** | 65 | ✓ Framework integration |
| LlamaIndex | **P0** | 66 | ✓ Framework integration |
| WCC + Betweenness | **P1** | 68 | ✓ Graph algorithms |
| Natural Language | **P1** | 71 | ✓ GraphRAG-SDK feature |
| Multi-Tenancy | **P1** | 78-79 | ✓ 10,000+ tenants |
| Node.js SDK | **P1** | 81 | ✓ SDK coverage |
| Pattern Comprehension | **P2** | 80 | ✓ Cypher parity |
| Java/Go SDKs | **P2** | 84 | ✓ Enterprise SDKs |