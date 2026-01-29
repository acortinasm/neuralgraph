# NeuralGraphDB - Manual de Usuario

**Versión:** 0.1.0  
**Última actualización:** 2026-01-09

---

## Introducción

NeuralGraphDB es una base de datos de grafos nativa diseñada para cargas de trabajo de Inteligencia Artificial. Utiliza álgebra lineal (matrices dispersas) para operaciones de alta velocidad.

### Características Principales

- **Motor CSR** - Almacenamiento en formato Compressed Sparse Row
- **NGQL** - Lenguaje de consulta similar a Cypher
- **Alto rendimiento** - <1ns para acceso a vecinos
- **Type-safe** - Escrito en Rust con newtypes

---

## Instalación

### Requisitos

- Rust 1.85 o superior
- Cargo

### Compilar desde fuente

```bash
git clone https://github.com/neuralgraphdb/neuralgraph
cd nGraph
cargo build --release
```

### Ejecutar demo

```bash
cargo run --release -p neural-cli
```

---

## Arquitectura

```
crates/
├── neural-core/        # Tipos base (NodeId, Edge, Graph trait, PropertyValue)
├── neural-storage/     # Almacenamiento CSR + PropertyStore + GraphStore
├── neural-parser/      # Parser NGQL
├── neural-executor/    # Planner, Executor, Evaluador de expresiones
└── neural-cli/         # Ejecutable demo
```

---

## Uso Básico

### Crear un Grafo

```rust
use neural_core::{Graph, Label, NodeId};
use neural_storage::GraphBuilder;

// Construir grafo
let mut builder = GraphBuilder::new();
builder.add_labeled_edge(0u64, 1u64, Label::new("KNOWS"));
builder.add_labeled_edge(0u64, 2u64, Label::new("KNOWS"));
builder.add_labeled_edge(1u64, 2u64, Label::new("FOLLOWS"));

let graph = builder.build();

// Consultar
println!("Nodos: {}", graph.node_count());
println!("Aristas: {}", graph.edge_count());

// Obtener vecinos
for neighbor in graph.neighbors(NodeId::new(0)) {
    println!("Vecino: {:?}", neighbor);
}
```

### Ejecutar Consultas NGQL con Propiedades

```rust
use neural_core::PropertyValue;
use neural_storage::GraphStore;
use neural_executor::execute_query;

// Crear grafo con propiedades
let store = GraphStore::builder()
    .add_labeled_node(0u64, "Person", [
        ("name", PropertyValue::from("Alice")),
        ("age", PropertyValue::from(30i64)),
    ])
    .add_labeled_node(1u64, "Person", [
        ("name", PropertyValue::from("Bob")),
        ("age", PropertyValue::from(25i64)),
    ])
    .add_edge(0u64, 1u64)
    .build();

// Query con propiedades
let result = execute_query(&store, "MATCH (n) RETURN n.name").unwrap();
println!("{}", result);
// | n.name  |
// |---------|
// | "Alice" |
// | "Bob"   |

// Query con filtro WHERE
let filtered = execute_query(&store, "MATCH (n) WHERE n.age > 28 RETURN n.name").unwrap();
println!("{}", filtered);
// | n.name  |
// |---------|
// | "Alice" |
```

---

## Lenguaje NGQL

### Sintaxis Básica

```cypher
MATCH <pattern>
[WHERE <expression>]
RETURN <items>
```

### Patterns

```cypher
-- Nodo simple
(n)

-- Nodo con variable
(person)

-- Nodo con label
(n:Person)

-- Nodo con variable y label
(alice:Person)

-- Relación outgoing
(a)-[:KNOWS]->(b)

-- Relación incoming
(a)<-[:KNOWS]-(b)

-- Cadena de relaciones
(a)-[:KNOWS]->(b)-[:WORKS_AT]->(c)

-- Shortest Path (Sprint 28)
SHORTEST PATH (a)-[*]->(b)
```

### WHERE Clause

```cypher
-- Comparación numérica
WHERE n.age > 30

-- Comparación de strings
WHERE n.name = "Alice"

-- Operadores lógicos
WHERE n.age > 20 AND n.age < 40
WHERE n.status = "active" OR n.role = "admin"
WHERE NOT n.deleted
```

### Operadores de Comparación

| Operador | Descripción |
|----------|-------------|
| `=` | Igual |
| `<>` | Distinto |
| `<` | Menor que |
| `>` | Mayor que |
| `<=` | Menor o igual |
| `>=` | Mayor o igual |

### RETURN Clause

```cypher
-- Variable completa
RETURN n

-- Propiedad
RETURN n.name

-- Múltiples items
RETURN n.name, n.age, m.title

-- Con alias
RETURN n.name AS nombre
```

### Funciones de Agregación

```cypher
-- Contar todos los nodos
MATCH (n) RETURN COUNT(*)

-- Contar expresión
MATCH (n) RETURN COUNT(n)

-- Suma
MATCH (n) RETURN SUM(n.age)

-- Promedio
MATCH (n) RETURN AVG(n.age)

-- Mínimo y Máximo
MATCH (n) RETURN MIN(n.age), MAX(n.age)

-- Recolectar valores
MATCH (n) RETURN COLLECT(n.name)

-- Con DISTINCT
MATCH (a)-[]->(b) RETURN COUNT(DISTINCT b)
```

| Función | Descripción |
|---------|-------------|
| `COUNT(*)` | Cuenta todas las filas |
| `COUNT(expr)` | Cuenta valores no-null |
| `SUM(expr)` | Suma valores numéricos |
| `AVG(expr)` | Promedio de valores numéricos |
| `MIN(expr)` | Valor mínimo |
| `MAX(expr)` | Valor máximo |
| `COLLECT(expr)` | Recolecta valores en lista |

---

## API Reference

### neural-core

#### NodeId

```rust
pub struct NodeId(pub u64);

impl NodeId {
    pub const fn new(id: u64) -> Self;
    pub const fn as_u64(self) -> u64;
    pub const fn as_usize(self) -> usize;
}
```

#### Graph Trait

```rust
pub trait Graph {
    fn node_count(&self) -> usize;
    fn edge_count(&self) -> usize;
    fn neighbors(&self, node: NodeId) -> impl Iterator<Item = NodeId>;
    fn out_degree(&self, node: NodeId) -> usize;
    fn contains_node(&self, node: NodeId) -> bool;
    fn has_edge(&self, source: NodeId, target: NodeId) -> bool;
}
```

#### PropertyValue

```rust
pub enum PropertyValue {
    Null,
    Bool(bool),
    Int(i64),
    Float(f64),
    String(String),
    Vector(Vec<f32>),  // Para embeddings
}
```

### neural-storage

#### CsrMatrix

```rust
impl CsrMatrix {
    pub fn empty() -> Self;
    pub fn with_nodes(num_nodes: usize) -> Self;
    pub fn from_edges(edges: &[Edge], num_nodes: usize) -> Self;
    pub fn neighbors_slice(&self, node: usize) -> &[u64];
    pub fn validate(&self) -> Result<()>;
    pub fn stats(&self) -> CsrStats;
}
```

#### GraphBuilder

```rust
impl GraphBuilder {
    pub fn new() -> Self;
    pub fn with_capacity(edge_capacity: usize) -> Self;
    pub fn add_edge(&mut self, source: NodeId, target: NodeId) -> &mut Self;
    pub fn add_labeled_edge(&mut self, source: NodeId, target: NodeId, label: Label) -> &mut Self;
    pub fn build(self) -> CsrMatrix;
}
```

#### GraphStore (con Propiedades)

```rust
impl GraphStore {
    pub fn builder() -> GraphStoreBuilder;
    pub fn get_property(&self, node: NodeId, key: &str) -> Option<&PropertyValue>;
    pub fn set_property(&mut self, node: NodeId, key: &str, value: PropertyValue);
    pub fn get_label(&self, node: NodeId) -> Option<&str>;
    pub fn has_label(&self, node: NodeId, label: &str) -> bool;
    pub fn nodes_with_label(&self, label: &str) -> impl Iterator<Item = NodeId>;
}
```

#### GraphStoreBuilder

```rust
impl GraphStoreBuilder {
    pub fn add_node<I>(self, id: u64, properties: I) -> Self;
    pub fn add_labeled_node<I>(self, id: u64, label: &str, properties: I) -> Self;
    pub fn add_edge(self, source: u64, target: u64) -> Self;
    pub fn build(self) -> GraphStore;
}
```

#### PropertyStore

```rust
impl PropertyStore {
    pub fn get(&self, node: NodeId, key: &str) -> Option<&PropertyValue>;
    pub fn set(&mut self, node: NodeId, key: &str, value: PropertyValue);
    pub fn contains(&self, node: NodeId, key: &str) -> bool;
}
```

### neural-parser

#### parse_query

```rust
pub fn parse_query(input: &str) -> Result<Query, ParseError>;
```

#### Query

```rust
pub struct Query {
    pub match_clause: MatchClause,
    pub where_clause: Option<WhereClause>,
    pub return_clause: ReturnClause,
}
```

---

## Ejemplos

### Grafo Social

```rust
use neural_core::{Graph, Label, NodeId};
use neural_storage::GraphBuilder;

fn main() {
    // Alice (0) knows Bob (1) and Charlie (2)
    // Bob knows Charlie
    let mut builder = GraphBuilder::new();
    builder.add_labeled_edge(0u64, 1u64, Label::new("KNOWS"));
    builder.add_labeled_edge(0u64, 2u64, Label::new("KNOWS"));
    builder.add_labeled_edge(1u64, 2u64, Label::new("KNOWS"));
    
    let graph = builder.build();
    
    // Friends of Alice
    for friend in graph.neighbors(NodeId::new(0)) {
        println!("Alice knows node {}", friend);
    }
    
    // Friends of friends (2-hop)
    for friend in graph.neighbors(NodeId::new(0)) {
        for fof in graph.neighbors(friend) {
            println!("Friend of friend: {}", fof);
        }
    }
}
```

### Parsing y Análisis de Queries

```rust
use neural_parser::{parse_query, Expression, ComparisonOp};

fn main() {
    let query = parse_query(r#"
        MATCH (p:Person)-[:WORKS_AT]->(c:Company)
        WHERE p.salary > 50000
        RETURN p.name, c.name
    "#).unwrap();
    
    // Analizar patterns
    let pattern = &query.match_clause.patterns[0];
    println!("Start: {:?}", pattern.start.label);  // Some("Person")
    
    // Analizar WHERE
    if let Some(where_clause) = &query.where_clause {
        if let Expression::Comparison { op, .. } = &where_clause.expression {
            println!("Comparison: {:?}", op);  // Gt
        }
    }
    
    // Analizar RETURN
    println!("Return items: {}", query.return_clause.items.len());  // 2
}
```

---

## Rendimiento

### Benchmarks (Sprint 1)

| Operación | Tiempo | Notas |
|-----------|--------|-------|
| `neighbors()` | <1ns | Acceso O(1) a slice |
| Build 1M edges | ~10ms | Sorting incluido |
| Memory 100k nodes | 8.8 MB | CSR compacto |

### Optimizaciones

1. **CSR Format** - Cache-friendly, O(1) neighbor access
2. **Zero-copy lexer** - logos evita allocations
3. **Newtypes** - Compilador optimiza a u64 raw

---

## Troubleshooting

### Error: "Unexpected token"

El parser encontró un token inesperado. Verifica la sintaxis:

```rust
// ❌ Incorrecto
MATCH (n Person) RETURN n

// ✅ Correcto  
MATCH (n:Person) RETURN n
```

### Error: "Missing required clause"

Las queries requieren MATCH y RETURN:

```rust
// ❌ Incorrecto
MATCH (n)

// ✅ Correcto
MATCH (n) RETURN n
```

---

## Roadmap

- [x] **Sprint 1:** Core Engine (CSR, Graph trait)
- [x] **Sprint 2:** Parser NGQL (Lexer, AST, Parser)
- [x] **Sprint 3:** Planner y Executor
- [x] **Sprint 4:** Propiedades (PropertyStore, GraphStore, WHERE filtering)
- [x] **Sprint 5:** Agregaciones (COUNT, SUM, AVG, MIN, MAX, COLLECT)
- [x] **Sprint 28:** Shortest Path (BFS, ExpandShortestPath)
- [ ] **Sprint 6:** GROUP BY, ORDER BY, LIMIT
- [ ] **Sprint 7:** Índices vectoriales (HNSW)
- [ ] **Sprint 8:** Algoritmos (Leiden, PageRank)

---

## Licencia

MIT License - Ver LICENSE para detalles.
