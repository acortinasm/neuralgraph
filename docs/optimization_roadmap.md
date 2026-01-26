# NeuralGraphDB: Análisis de Performance y Roadmap de Optimizaciones

**Fecha:** 2026-01-12  
**Basado en:** Benchmark con 100,000 papers ArXiv

---

## Resumen Ejecutivo

NeuralGraphDB v0.1.0 muestra una **ingesta 470x más rápida** que Neo4j pero **queries 3-10x más lentas**. Este documento detalla las optimizaciones necesarias para cerrar la brecha en rendimiento de queries.

---

## Benchmark Actual (100K papers)

| Operación | Neo4j | NeuralGraphDB | Ratio |
|-----------|-------|---------------|-------|
| Carga papers | 51.64s | 0.11s | **Neo4j 470x más lento** |
| Creación citas | 263.67s | 109.30s | **Neo4j 2.4x más lento** |
| Query 2-hop | 0.35s | 0.39s | Neo4j 1.1x más rápido |
| Query Vector | 0.10s | 0.30s | Neo4j 3x más rápido |
| Query Híbrido | 0.03s | 0.30s | Neo4j 10x más rápido |

### Análisis

**Fortalezas actuales:**

- Ingesta extremadamente rápida (generación de CSVs)
- Bajo footprint de memoria
- Estructura CSR eficiente para traversals

**Debilidades actuales:**

- No hay índices secundarios
- No hay caché de queries
- Ejecución de queries no optimizada
- No hay búsqueda vectorial nativa

---

## Roadmap de Optimizaciones

### Fase 1: Índices (Sprint 9-10)

#### 1.1 Índice de Labels (Prioridad: ALTA)

**Problema:** `MATCH (n:Paper)` hace scan completo O(n).

**Solución:** Índice invertido de labels → node IDs.

```rust
// Estructura propuesta
pub struct LabelIndex {
    // label -> sorted Vec<NodeId>
    index: HashMap<Label, Vec<NodeId>>,
}

impl LabelIndex {
    /// O(1) lookup
    fn nodes_with_label(&self, label: &Label) -> &[NodeId];
    
    /// O(log n) check
    fn has_label(&self, node: NodeId, label: &Label) -> bool;
}
```

**Impacto estimado:** 10-100x mejora en queries filtradas por label.

---

#### 1.2 Índice de Propiedades (Prioridad: MEDIA)

**Problema:** `WHERE n.category = "cs.LG"` requiere scan de todas las propiedades.

**Solución:** B-Tree o Hash index por propiedad.

```rust
pub struct PropertyIndex {
    // (property_name, value) -> Vec<NodeId>
    btree: BTreeMap<(String, PropertyValue), Vec<NodeId>>,
}
```

**Impacto estimado:** 50-100x mejora en queries con filtros de igualdad.

---

#### 1.3 Índice de Aristas por Tipo (Prioridad: MEDIA)

**Problema:** `MATCH (a)-[:CITES]->(b)` debe filtrar todas las aristas.

**Solución:** CSR separado por tipo de relación.

```rust
pub struct TypedEdgeStore {
    // edge_type -> CsrMatrix
    matrices: HashMap<Label, CsrMatrix>,
}
```

**Impacto estimado:** 5-20x mejora en traversals filtrados por tipo.

---

### Fase 2: Optimización de Queries (Sprint 11-12)

#### 2.1 Query Planner Optimizado

**Problema actual:** El planner es naive, no considera estadísticas.

**Mejoras:**

1. **Estadísticas de cardinalidad:**

   ```rust
   struct QueryStats {
       node_count: usize,
       edge_count: usize,
       label_cardinality: HashMap<Label, usize>,
       selectivity: HashMap<PropertyKey, f64>,
   }
   ```

2. **Reordenamiento de operaciones:**
   - Pushdown de filtros (aplicar WHERE lo antes posible)
   - Join ordering (empezar por nodos con menor cardinalidad)
   - Index selection (elegir índice más selectivo)

3. **Plan caching:**

   ```rust
   struct PlanCache {
       cache: LruCache<QueryHash, ExecutionPlan>,
   }
   ```

**Impacto estimado:** 2-5x mejora general.

---

#### 2.2 Ejecución Vectorizada

**Problema:** Procesamos fila por fila.

**Solución:** Procesar en batches usando SIMD.

```rust
// Antes (scalar)
for node in nodes {
    if filter(node) {
        results.push(node);
    }
}

// Después (vectorizado)
let mask = filter_batch(&nodes[..BATCH_SIZE]); // SIMD
results.extend(nodes.iter().zip(mask).filter(|(_, m)| *m).map(|(n, _)| n));
```

**Impacto estimado:** 2-4x mejora en scans grandes.

---

#### 2.3 Lazy Evaluation

**Problema:** Materializamos resultados intermedios.

**Solución:** Iteradores lazy con streaming.

```rust
// Pipeline de operadores
let pipeline = ScanOp::new(label)
    .filter(predicate)
    .project(columns)
    .limit(10);

// Solo procesa lo necesario
for row in pipeline.take(10) {
    yield row;
}
```

**Impacto estimado:** Reduce memoria y mejora latencia en queries con LIMIT.

---

### Fase 3: Búsqueda Vectorial (Sprint 13-15)

#### 3.1 Índice HNSW para Embeddings

**Problema:** No hay búsqueda de similitud vectorial.

**Solución:** Implementar o integrar HNSW (Hierarchical Navigable Small World).

```rust
pub struct VectorIndex {
    hnsw: HnswIndex<f32>,
    node_mapping: Vec<NodeId>,
}

impl VectorIndex {
    /// Busca los k vecinos más cercanos
    fn knn(&self, query: &[f32], k: usize) -> Vec<(NodeId, f32)>;
    
    /// Búsqueda con filtro de labels
    fn knn_filtered(
        &self, 
        query: &[f32], 
        k: usize, 
        label: Option<&Label>
    ) -> Vec<(NodeId, f32)>;
}
```

**Opciones de implementación:**

1. **Integrar `hnsw` crate** (más fácil)
2. **Implementar desde cero** (más control)
3. **Bindings a Faiss** (mejor rendimiento)

**Impacto estimado:** Habilita queries vectoriales comparables a Neo4j.

---

#### 3.2 Queries Híbridas (Vector + Graph)

**Problema:** No podemos combinar similitud vectorial con traversals.

**Solución:** Operador híbrido en el planner.

```sql
-- Query híbrida propuesta
MATCH (p:Paper)
WHERE vector_similarity(p.embedding, $query_embedding) > 0.8
  AND (p)-[:CITES]->(cited:Paper)
RETURN p.title, cited.title
ORDER BY vector_similarity(p.embedding, $query_embedding) DESC
LIMIT 10
```

**Impacto estimado:** Diferenciador clave para AI workloads.

---

### Fase 4: Paralelización (Sprint 16+)

#### 4.1 Parallel Scan

```rust
// Usando rayon
nodes.par_iter()
    .filter(|n| predicate(n))
    .collect()
```

#### 4.2 Parallel Aggregation

```rust
// Map-reduce paralelo
let partial_sums: Vec<i64> = chunks.par_iter()
    .map(|chunk| chunk.iter().sum())
    .collect();
let total: i64 = partial_sums.iter().sum();
```

**Impacto estimado:** 2-8x mejora (según cores disponibles).

---

## Priorización Recomendada

| Prioridad | Mejora | Esfuerzo | Impacto |
|-----------|--------|----------|---------|
| 🔴 P0 | Índice de Labels | 1 sprint | 10-100x |
| 🔴 P0 | Filter Pushdown | 1 sprint | 2-5x |
| 🟡 P1 | Índice de Propiedades | 1-2 sprints | 50-100x |
| 🟡 P1 | Lazy Evaluation | 1 sprint | 2-3x |
| 🟢 P2 | Índice HNSW | 2-3 sprints | Habilita Vector |
| 🟢 P2 | Parallel Scan | 1 sprint | 2-8x |
| 🔵 P3 | Query Caching | 1 sprint | 2-5x |
| 🔵 P3 | Vectorización SIMD | 2 sprints | 2-4x |

---

## Métricas Objetivo (Post-Optimización)

| Query | Actual | Objetivo | vs Neo4j |
|-------|--------|----------|----------|
| COUNT(*) | 0.30s | <0.01s | Mejor |
| Label filter | 0.30s | <0.01s | Igual |
| 2-hop traversal | 0.39s | <0.10s | Mejor |
| Vector search | N/A | <0.05s | Igual |
| Híbrido | N/A | <0.10s | Igual |

---

## Siguiente Paso Inmediato

**Sprint 9:** Implementar Índice de Labels

1. Añadir `LabelIndex` a `GraphStore`
2. Modificar `GraphStoreBuilder` para construir índice
3. Actualizar `nodes_with_label()` para usar índice
4. Benchmark antes/después

```rust
// API propuesta
impl GraphStore {
    /// O(1) con índice
    pub fn nodes_with_label(&self, label: &str) -> impl Iterator<Item = NodeId> {
        self.label_index
            .get(label)
            .map(|ids| ids.iter().copied())
            .unwrap_or_default()
    }
}
```

---

## Referencias

- [Neo4j Query Tuning](https://neo4j.com/docs/cypher-manual/current/query-tuning/)
- [GraphBLAS Specification](https://graphblas.org/)
- [HNSW Paper](https://arxiv.org/abs/1603.09320)
- [Vectorized Query Execution](https://www.vldb.org/pvldb/vol11/p2209-kersten.pdf)
