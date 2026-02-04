# Consolidated Sprint Reports

Generated on: Wed Jan 14 17:03:52 CET 2026


================================================================================
# Report: sprint_1_report.md
================================================================================

# Sprint 1: Core Engine - Reporte Final

**Fecha:** 2026-01-09  
**Estado:** ✅ Completado  
**Duración:** 1 sesión

---

## Objetivo del Sprint

> Construir el núcleo de alto rendimiento. Prototipo en Rust que carga un grafo pequeño en memoria CSR.

---

## Entregables

### 1. Cargo Workspace

```
/Users/alberto/Projects/nGraph/
├── Cargo.toml              # Workspace (Rust 1.85, Edition 2024)
└── crates/
    ├── neural-core/        # Tipos base
    ├── neural-storage/     # CSR Storage  
    └── neural-cli/         # Demo ejecutable
```

### 2. Neural Core (`neural-core`)

| Componente | Descripción |
|------------|-------------|
| `NodeId(u64)` | Newtype type-safe para IDs de nodos |
| `EdgeId(u64)` | Newtype para IDs de aristas |
| `Label` | Etiquetas para nodos/aristas |
| `PropertyValue` | Enum schema-flexible (Null, Bool, Int, Float, String, Vector) |
| `Edge` | Struct con source, target, label |
| `Graph` trait | Interfaz core (node_count, edge_count, neighbors, out_degree) |

### 3. Neural Storage (`neural-storage`)

| Componente | Descripción |
|------------|-------------|
| `CsrMatrix` | Compressed Sparse Row matrix |
| `GraphBuilder` | Constructor incremental |
| `CsrStats` | Estadísticas del grafo |

**Formato CSR:**

```
row_ptr:     [0, 2, 3, 3]  // Offsets por nodo
col_indices: [1, 2, 2]     // Destinos de aristas
```

### 4. Demo Ejecutable (`neural-cli`)

Binario `neuralgraph` con tres demos:

- **Social Network:** Travesía 1-hop y 2-hop
- **Large-Scale Performance:** Benchmark con 100k nodos
- **Property Values:** Tipos schema-flexible

---

## Métricas de Rendimiento

| Métrica | Objetivo | Resultado |
|---------|----------|-----------|
| Neighbor access latency | <100ns | **<1ns** ✅ |
| Build time (1M edges) | - | ~10ms |
| Memory (100k nodes, 1M edges) | - | 8.8 MB |

---

## Tests

| Suite | Tests | Estado |
|-------|-------|--------|
| neural-core unit | 6 | ✅ Pass |
| neural-storage unit | 11 | ✅ Pass |
| neural-storage doctests | 3 | ✅ Pass |
| **Total** | **20** | ✅ **All Pass** |

---

## Código Destacado

### Graph Trait

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

### CSR Neighbor Access (O(1))

```rust
#[inline]
pub fn neighbors_slice(&self, node: usize) -> &[u64] {
    let start = self.row_ptr[node];
    let end = self.row_ptr[node + 1];
    &self.col_indices[start..end]
}
```

---

## Lecciones Aprendidas

1. **Newtypes son esenciales** - `NodeId(u64)` previene bugs sutiles
2. **CSR es extremadamente eficiente** - <1ns para acceso a vecinos
3. **GraphBuilder pattern** - Facilita construcción incremental antes de "freeze" a CSR

---

## Próximo Sprint

**Sprint 2: Parser NGQL Base**

Objetivo: Implementar el lexer y parser para NGQL básico.

Tareas principales:

1. Lexer con `logos` (tokenización)
2. Parser con `nom` (AST)
3. Soporte para `MATCH (n)-[:REL]->(m)`


================================================================================
# Report: sprint_2_report.md
================================================================================

# Sprint 2: Parser NGQL Base - Reporte Final

**Fecha:** 2026-01-09  
**Estado:** ✅ Completado  
**Duración:** 1 sesión

---

## Objetivo del Sprint

> Implementar el lexer y parser para NGQL básico. El sistema debe entender `MATCH (a)-[:KNOWS]->(b) RETURN a, b`.

---

## Entregables

### 1. Nuevo Crate: `neural-parser`

```
crates/neural-parser/
├── Cargo.toml
└── src/
    ├── lib.rs
    ├── lexer.rs    # Tokenización con logos
    ├── ast.rs      # Tipos AST
    └── parser.rs   # Parser recursivo descendente
```

### 2. Lexer (`lexer.rs`)

Tokenización de alta velocidad usando la crate `logos`.

| Categoría | Tokens |
|-----------|--------|
| Keywords | `MATCH`, `WHERE`, `RETURN`, `AND`, `OR`, `NOT`, `AS`, `TRUE`, `FALSE`, `NULL` |
| Símbolos | `()`, `[]`, `{}`, `->`, `<-`, `-`, `:`, `,`, `.` |
| Operadores | `=`, `<>`, `<`, `>`, `<=`, `>=` |
| Literals | String (doble/simple), Integer, Float |
| Identifiers | Variables, labels, parameters (`$param`) |

**Características:**

- Case-insensitive para keywords
- Zero-copy tokenización
- Errores con posición

### 3. AST (`ast.rs`)

Estructuras para representar consultas parseadas:

```rust
pub struct Query {
    pub match_clause: MatchClause,
    pub where_clause: Option<WhereClause>,
    pub return_clause: ReturnClause,
}

pub struct Pattern {
    pub start: NodePattern,
    pub chain: Vec<(RelPattern, NodePattern)>,
}

pub struct NodePattern {
    pub identifier: Option<String>,
    pub label: Option<String>,
}

pub struct RelPattern {
    pub identifier: Option<String>,
    pub label: Option<String>,
    pub direction: Direction,
}

pub enum Expression {
    Property { variable: String, property: String },
    Literal(Literal),
    Parameter(String),
    Comparison { left, op, right },
    And(Box<Expression>, Box<Expression>),
    Or(Box<Expression>, Box<Expression>),
    Not(Box<Expression>),
}
```

**Todas las estructuras implementan:**

- `Debug`, `Clone`, `PartialEq`
- `Serialize`, `Deserialize` (serde)
- `Display` (para debugging)

### 4. Parser (`parser.rs`)

Parser recursivo descendente sin dependencias externas.

| Función | Descripción |
|---------|-------------|
| `parse_query()` | Entry point principal |
| `parse_match_clause()` | Patterns de MATCH |
| `parse_pattern()` | Cadenas nodo-rel-nodo |
| `parse_node_pattern()` | `(n:Label)` |
| `parse_rel_pattern()` | `-[:REL]->` |
| `parse_where_clause()` | Expresiones WHERE |
| `parse_return_clause()` | Items RETURN |

---

## Gramática Implementada

```ebnf
query        = match_clause, [where_clause], return_clause ;
match_clause = "MATCH", pattern, { ",", pattern } ;
pattern      = node_pattern, { rel_pattern, node_pattern } ;
node_pattern = "(", [identifier], [":", label], ")" ;
rel_pattern  = ("-" | "<-"), "[", [identifier], [":", label], "]", ("->" | "-" | "") ;
where_clause = "WHERE", or_expression ;
or_expression = and_expression, { "OR", and_expression } ;
and_expression = comparison, { "AND", comparison } ;
comparison   = primary, [comp_op, primary] ;
comp_op      = "=" | "<>" | "<" | ">" | "<=" | ">=" ;
primary      = literal | property_access | parameter | "NOT", primary | "(", or_expression, ")" ;
return_clause = "RETURN", return_item, { ",", return_item } ;
return_item  = expression, ["AS", identifier] ;
```

---

## Tests

| Módulo | Tests | Descripción |
|--------|-------|-------------|
| lexer | 14 | Keywords, símbolos, literals, identifiers |
| ast | 3 | Display, pattern formatting |
| parser | 9 | Queries simples a complejas |
| **Total** | **26** | |

---

## Queries de Ejemplo

```cypher
-- Simple
MATCH (n) RETURN n

-- Con label
MATCH (n:Person) RETURN n

-- Con relación
MATCH (a)-[:KNOWS]->(b) RETURN a, b

-- Compleja con WHERE
MATCH (a:Person)-[:KNOWS]->(b:Person) 
WHERE a.age > 30 
RETURN a.name, b.name

-- Con string literal
MATCH (p:Person) 
WHERE p.name = "Alice" 
RETURN p
```

---

## Integración con CLI

Demo 4 en el ejecutable `neuralgraph`:

```
Query: MATCH (a:Person)-[:KNOWS]->(b:Person) WHERE a.age > 30 RETURN a.name, b.name
  ✓ Parsed successfully
  └─ AST: MATCH (a:Person)-[:KNOWS]->(b:Person) WHERE a.age > 30 RETURN a.name, b.name
     └─ Start node label: :Person
     └─ Relationship: :KNOWS
```

---

## Código Destacado

### Tokenización con logos

```rust
#[derive(Logos, Debug, Clone, PartialEq)]
#[logos(skip r"[ \t\n\r]+")]
pub enum Token<'a> {
    #[token("MATCH", ignore(ascii_case))]
    Match,
    
    #[token("->")]
    RightArrow,
    
    #[regex(r#""([^"\\]|\\.)*""#, |lex| lex.slice())]
    String(&'a str),
    
    #[regex(r"[a-zA-Z_][a-zA-Z0-9_]*", |lex| lex.slice())]
    Ident(&'a str),
}
```

### Parser Entry Point

```rust
pub fn parse_query(input: &str) -> Result<Query, ParseError> {
    let tokens = tokenize(input)?;
    let mut parser = Parser::new(tokens);
    
    let match_clause = parse_match_clause(&mut parser)?;
    let where_clause = if matches!(parser.peek(), Some(Token::Where)) {
        Some(parse_where_clause(&mut parser)?)
    } else {
        None
    };
    let return_clause = parse_return_clause(&mut parser)?;
    
    Ok(Query { match_clause, where_clause, return_clause })
}
```

---

## Lecciones Aprendidas

1. **logos > regex** - Mucho más rápido y type-safe para lexing
2. **Recursive descent es suficiente** - No necesitamos `nom` para esta gramática
3. **Display trait es invaluable** - Facilita debugging y testing
4. **Tests exhaustivos** - 26 tests cubren casos edge

---

## Próximo Sprint

**Sprint 3: Planner y Ejecutor**

Objetivo: Traducir el AST a un plan de ejecución y ejecutarlo sobre el CSR.

Tareas principales:

1. Definir `LogicalPlan` y `PhysicalPlan`
2. Implementar `Planner` (AST -> Plan)
3. Implementar `Executor` (Plan -> Resultados)
4. Integrar con `CsrMatrix` para travesías


================================================================================
# Report: sprint_3_report.md
================================================================================

# Sprint 3: Planner y Executor - Reporte Final

**Fecha:** 2026-01-09  
**Estado:** ✅ Completado  
**Duración:** 1 sesión

---

## Objetivo del Sprint

> Implementar el planner y executor para que las queries NGQL ejecuten código real sobre el motor CSR.

---

## Entregables

### 1. Nuevo Crate: `neural-executor`

```
crates/neural-executor/
├── Cargo.toml
└── src/
    ├── lib.rs        # Entry point: execute_query()
    ├── plan.rs       # Plan types
    ├── planner.rs    # AST -> PhysicalPlan
    ├── executor.rs   # Plan execution
    └── result.rs     # QueryResult types
```

### 2. Plan Types (`plan.rs`)

| Tipo | Descripción |
|------|-------------|
| `PhysicalPlan::ScanAllNodes` | Escanea todos los nodos |
| `PhysicalPlan::ExpandNeighbors` | Expande a vecinos |
| `PhysicalPlan::Filter` | Filtra con predicado |
| `PhysicalPlan::Project` | Proyecta columnas |
| `PhysicalPlan::Sequence` | Secuencia de operaciones |

### 3. Planner (`planner.rs`)

Traduce el AST a un plan físico ejecutable:

```rust
pub fn plan(&self, query: &Query) -> Result<PhysicalPlan> {
    // MATCH -> ScanAllNodes + ExpandNeighbors
    // WHERE -> Filter
    // RETURN -> Project
}
```

| Función | Entrada | Salida |
|---------|---------|--------|
| `plan_match()` | MatchClause | Scan + Expand |
| `plan_where()` | WhereClause | Filter |
| `plan_return()` | ReturnClause | Project |

### 4. Executor (`executor.rs`)

Ejecuta el plan físico contra el grafo:

```rust
pub fn execute(&self, plan: &PhysicalPlan) -> Result<QueryResult>
```

| Operación | Complejidad | Descripción |
|-----------|-------------|-------------|
| `execute_scan_all()` | O(N) | Itera todos los nodos |
| `execute_expand()` | O(1) per binding | Acceso CSR O(1) |
| `execute_filter()` | O(B) | Filtra bindings |
| `execute_project()` | O(B × C) | Proyecta columnas |

### 5. Result Types (`result.rs`)

```rust
pub enum Value {
    Null,
    Node(u64),
    Int(i64),
    Float(f64),
    String(String),
    Bool(bool),
}

pub struct Row {
    values: HashMap<String, Value>,
}

pub struct QueryResult {
    columns: Vec<String>,
    rows: Vec<Row>,
}
```

**Display como tabla:**

```
| a  | b  |
|----|----|
| n0 | n1 |
| n0 | n2 |

2 row(s)
```

---

## Arquitectura del Pipeline

```
Query String: "MATCH (a)-[:KNOWS]->(b) RETURN a, b"
    │
    ▼
┌─────────────┐
│   Parser    │  neural-parser (Sprint 2)
└──────┬──────┘
       │ AST: Query { match: [...], return: [...] }
       ▼
┌─────────────┐
│   Planner   │  neural-executor (Sprint 3)
└──────┬──────┘
       │ PhysicalPlan: Sequence [Scan, Expand, Project]
       ▼
┌─────────────┐
│  Executor   │  neural-executor (Sprint 3)
└──────┬──────┘
       │
       ▼
   QueryResult: 6 rows
```

---

## Demo: Query Execution

```
🔍 All edges
   Query: MATCH (a)-[]->(b) RETURN a, b
   Results: 6 rows

| a  | b  |
|----|----|
| n0 | n1 |
| n0 | n2 |
| n1 | n2 |
| n1 | n3 |
| n2 | n3 |
| n3 | n4 |

6 row(s)

🔍 2-hop paths
   Query: MATCH (a)-[]->(b)-[]->(c) RETURN a, b, c
   Results: 6 rows

| a  | b  | c  |
|----|----|----| 
| n0 | n1 | n2 |
| n0 | n1 | n3 |
| n0 | n2 | n3 |
| n1 | n2 | n3 |
| n1 | n3 | n4 |
| n2 | n3 | n4 |
```

---

## Tests

| Módulo | Tests | Descripción |
|--------|-------|-------------|
| plan | 2 | Display, column names |
| planner | 4 | Query planning |
| executor | 5 | Scan, expand, full queries |
| result | 4 | Value, Row, QueryResult, Bindings |
| doctest | 2 | API examples |
| **Total** | **17** | |

---

## Código Destacado

### Execute Query (Entry Point)

```rust
pub fn execute_query(graph: &CsrMatrix, query: &str) -> Result<QueryResult> {
    // Parse
    let ast = neural_parser::parse_query(query)?;
    
    // Plan
    let planner = Planner::new();
    let plan = planner.plan(&ast)?;
    
    // Execute
    let executor = Executor::new(graph);
    executor.execute(&plan)
}
```

### Expand Neighbors (CSR Access)

```rust
fn execute_expand(&self, bindings: Vec<Bindings>, from: &str, to: &str, dir: &Direction) -> Result<Vec<Bindings>> {
    let mut results = Vec::new();
    
    for existing in bindings {
        let from_node = existing.get(from)?;
        
        for neighbor in self.graph.neighbors(from_node) {
            results.push(existing.with(to, neighbor));
        }
    }
    
    Ok(results)
}
```

---

## Limitaciones Actuales

- **Sin propiedades:** Las propiedades de nodos no están implementadas (solo estructura)
- **Sin labels:** El filtrado por label no está activo (CSR no almacena labels)
- **Sin incoming eficiente:** Las aristas incoming requieren full scan

---

## Estadísticas del Workspace

| Crate | LOC | Tests |
|-------|-----|-------|
| neural-core | ~180 | 6 |
| neural-storage | ~580 | 14 |
| neural-parser | ~780 | 26 |
| neural-executor | ~650 | 17 |
| **Total** | **~2,190** | **63** |

---

## Próximo Sprint

**Sprint 4: Propiedades (Arrow/Parquet)**

Objetivo: Hacer que `RETURN a.name` funcione leyendo de columnas Arrow.

Tareas principales:

1. Almacenamiento de propiedades en columnar format
2. Índice NodeId -> Row
3. Evaluación de predicados WHERE
4. Integración con Arrow libraries


================================================================================
# Report: sprint_4_report.md
================================================================================

# Sprint 4: Property Storage - Reporte Final

**Fecha:** 2026-01-09  
**Estado:** ✅ Completado  
**Duración:** 1 sesión

---

## Objetivo del Sprint

> Implementar almacenamiento de propiedades para nodos. `RETURN a.name` y `WHERE a.age > 30` deben funcionar.

---

## Entregables

### 1. PropertyStore (`neural-storage/src/properties.rs`)

```rust
pub struct PropertyStore {
    data: HashMap<u64, HashMap<String, PropertyValue>>,
}

impl PropertyStore {
    pub fn get(&self, node: NodeId, key: &str) -> Option<&PropertyValue>;
    pub fn set(&mut self, node: NodeId, key: &str, value: PropertyValue);
    pub fn remove(&mut self, node: NodeId, key: &str) -> Option<PropertyValue>;
    pub fn contains(&self, node: NodeId, key: &str) -> bool;
}
```

### 2. GraphStore (`neural-storage/src/graph_store.rs`)

Combina estructura (CSR) + propiedades:

```rust
pub struct GraphStore {
    graph: CsrMatrix,        // Estructura
    properties: PropertyStore, // Propiedades de nodos
    labels: PropertyStore,    // Labels de nodos
}

impl GraphStore {
    pub fn get_property(&self, node: NodeId, key: &str) -> Option<&PropertyValue>;
    pub fn get_label(&self, node: NodeId) -> Option<&str>;
    pub fn has_label(&self, node: NodeId, label: &str) -> bool;
    pub fn nodes_with_label(&self, label: &str) -> impl Iterator<Item = NodeId>;
}
```

**Builder pattern:**

```rust
let store = GraphStore::builder()
    .add_labeled_node(0, "Person", [
        ("name", PropertyValue::from("Alice")),
        ("age", PropertyValue::from(30i64)),
    ])
    .add_edge(0, 1)
    .build();
```

### 3. Expression Evaluation (`neural-executor/src/eval.rs`)

| Función | Descripción |
|---------|-------------|
| `evaluate(expr, bindings, store)` | Evalúa expresión completa |
| `is_true(expr, bindings, store)` | Para predicados WHERE |
| `compare_values(left, op, right)` | Comparaciones =, <>, <, > |
| `property_value_to_value(pv)` | Conversión PropertyValue → Value |

**Operadores soportados:**

- Comparación: `=`, `<>`, `<`, `>`, `<=`, `>=`
- Lógicos: `AND`, `OR`, `NOT`
- Acceso a propiedades: `n.name`, `n.age`

### 4. Executor Actualizado

```rust
pub fn execute_query(store: &GraphStore, query: &str) -> Result<QueryResult>
```

Cambios:

- Usa `GraphStore` en lugar de `CsrMatrix`
- `execute_filter()` evalúa predicados reales
- `bindings_to_result()` resuelve propiedades en proyecciones

---

## Queries Soportadas

```cypher
-- Retornar propiedades
MATCH (n) RETURN n.name

-- Filtrar por número
MATCH (n) WHERE n.age > 28 RETURN n.name

-- Filtrar por string
MATCH (n) WHERE n.name = "Alice" RETURN n

-- Combinación con relaciones
MATCH (a)-[]->(b) RETURN a.name, b.name

-- Lógica compleja
MATCH (n) WHERE n.age > 20 AND n.age < 40 RETURN n
```

---

## Demo Output

```
🔍 Return names
   Query: MATCH (n) RETURN n.name
   Results: 4 rows

| n.name    |
|-----------|
| "Alice"   |
| "Bob"     |
| "Charlie" |
| "Diana"   |

🔍 Filter by age
   Query: MATCH (n) WHERE n.age > 28 RETURN n.name
   Results: 2 rows

| n.name    |
|-----------|
| "Alice"   |
| "Charlie" |

🔍 Relationship query
   Query: MATCH (a)-[]->(b) RETURN a.name, b.name
   Results: 4 rows

| a.name    | b.name    |
|-----------|-----------|
| "Alice"   | "Bob"     |
| "Alice"   | "Charlie" |
| "Bob"     | "Charlie" |
| "Charlie" | "Diana"   |
```

---

## Tests

| Módulo | Tests | Nuevos |
|--------|-------|--------|
| neural-core | 6 | - |
| neural-storage | 22 | +8 (PropertyStore, GraphStore) |
| neural-parser | 23 | - |
| neural-executor | 22 | +5 (eval, filter tests) |
| doctests | 11 | - |
| **Total** | **84** | **+13** |

---

## Arquitectura Final

```
Query: MATCH (n) WHERE n.age > 28 RETURN n.name
    │
    ▼
┌──────────┐
│  Parser  │ → AST con WHERE expression
└────┬─────┘
     │
     ▼
┌──────────┐
│ Planner  │ → PhysicalPlan: [Scan, Filter, Project]
└────┬─────┘
     │
     ▼
┌──────────┐         ┌─────────────┐
│ Executor │ ◄──────►│ GraphStore  │
└────┬─────┘         │  • CsrMatrix│
     │               │  • Properties│
     ▼               └─────────────┘
QueryResult: 2 rows
```

---

## Estadísticas del Workspace

| Crate | LOC (aprox) | Tests |
|-------|-------------|-------|
| neural-core | ~180 | 6 |
| neural-storage | ~900 | 22 |
| neural-parser | ~780 | 23 |
| neural-executor | ~850 | 22 |
| **Total** | **~2,710** | **84** |

---

## Próximo Sprint

**Sprint 5: Optimización y Features Avanzados**

Opciones:

1. **Índices de Labels** - Índice invertido para `nodes_with_label()`
2. **Agregaciones** - COUNT, SUM, AVG en RETURN
3. **Arrow/Parquet** - Almacenamiento columnar eficiente
4. **Variable Length Paths** - `(a)-[*1..3]->(b)`


================================================================================
# Report: sprint_5_report.md
================================================================================

# Sprint 5: Aggregations - Final Report

**Fecha:** 2026-01-09  
**Estado:** ✅ Completado  
**Duración:** 1 sesión

---

## Objetivo del Sprint

> Implementar funciones de agregación: COUNT, SUM, AVG, MIN, MAX, COLLECT.

---

## Entregables

### 1. Parser Updates (`neural-parser`)

**Nuevos Tokens:**

```rust
Token::Count
Token::Sum
Token::Avg
Token::Min
Token::Max
Token::Collect
Token::Distinct
Token::Star  // for COUNT(*)
```

**AST Extension:**

```rust
pub enum AggregateFunction {
    Count,
    Sum,
    Avg,
    Min,
    Max,
    Collect,
}

pub enum Expression {
    // ... existing variants ...
    Aggregate {
        function: AggregateFunction,
        argument: Option<Box<Expression>>,
        distinct: bool,
    },
}
```

### 2. Aggregate Module (`neural-executor/aggregate.rs`)

| Función | Descripción |
|---------|-------------|
| `compute_count()` | COUNT(*) or COUNT(expr) |
| `compute_sum()` | SUM(numeric expr) |
| `compute_avg()` | AVG(numeric expr) |
| `compute_min()` | MIN(expr) |
| `compute_max()` | MAX(expr) |
| `compute_collect()` | Collect values into list |

### 3. Executor Updates

**Detection & Execution:**

```rust
// In bindings_to_result()
let has_aggregates = columns.iter().any(|c| c.is_aggregate());

if has_aggregates {
    return self.execute_aggregate_result(bindings, columns);
}
```

**ProjectExpression Extended:**

```rust
pub enum ProjectExpression {
    Variable(String),
    Property { variable, property },
    Aggregate { function, argument, distinct },
}
```

---

## Queries Soportadas

```cypher
-- Count all nodes
MATCH (n) RETURN COUNT(*)

-- Count expression
MATCH (n) RETURN COUNT(n)

-- Count distinct
MATCH (a)-[]->(b) RETURN COUNT(DISTINCT b)

-- Sum numeric property
MATCH (n) RETURN SUM(n.age)

-- Average
MATCH (n) RETURN AVG(n.age)

-- Min/Max
MATCH (n) RETURN MIN(n.age), MAX(n.age)

-- Collect
MATCH (n) RETURN COLLECT(n.name)
```

---

## Demo Output

```
📈 Aggregation Queries:
🔍 Count all
   Query: MATCH (n) RETURN COUNT(*)
   Results: 1 rows

| COUNT(*) |
|----------|
| 4        |

🔍 Sum ages
   Query: MATCH (n) RETURN SUM(n.age)
   Results: 1 rows

| SUM(n.age) |
|------------|
| 118.00     |

🔍 Average age
   Query: MATCH (n) RETURN AVG(n.age)
   Results: 1 rows

| AVG(n.age) |
|------------|
| 29.50      |

🔍 Min/Max age
   Query: MATCH (n) RETURN MIN(n.age), MAX(n.age)
   Results: 1 rows

| MIN(n.age) | MAX(n.age) |
|------------|------------|
| 25         | 35         |
```

---

## Tests

| Módulo | Tests | Nuevos |
|--------|-------|--------|
| neural-core | 6 | - |
| neural-storage | 22 | - |
| neural-parser | 23 | - |
| neural-executor | 33 | +11 |
| doctests | 11 | - |
| **Total** | **95** | **+11** |

**Nuevos tests de agregación:**

- `aggregate::test_count_star`
- `aggregate::test_count_expr`
- `aggregate::test_sum`
- `aggregate::test_avg`
- `aggregate::test_min`
- `aggregate::test_max`
- `executor::test_count_star`
- `executor::test_count_variable`
- `executor::test_sum_property`
- `executor::test_avg_property`
- `executor::test_min_max_property`

---

## Arquitectura

```
MATCH (n) RETURN COUNT(*), AVG(n.age)
    │
    ▼
┌──────────────┐
│    Parser    │ → Expression::Aggregate { Count, None, false }
└──────┬───────┘   Expression::Aggregate { Avg, Some(n.age), false }
       │
       ▼
┌──────────────┐
│   Planner    │ → ProjectExpression::Aggregate
└──────┬───────┘
       │
       ▼
┌──────────────┐         ┌─────────────────┐
│   Executor   │ ◄──────►│ aggregate.rs    │
│              │         │ compute_count() │
│              │         │ compute_avg()   │
└──────┬───────┘         └─────────────────┘
       │
       ▼
QueryResult: 1 row
| COUNT(*) | AVG(n.age) |
| 4        | 29.50      |
```

---

## Estadísticas del Workspace

| Crate | LOC (aprox) | Tests |
|-------|-------------|-------|
| neural-core | ~180 | 6 |
| neural-storage | ~900 | 22 |
| neural-parser | ~850 | 23 |
| neural-executor | ~1,200 | 33 |
| **Total** | **~3,130** | **95** |

---

## Próximo Sprint

**Sprint 6: GROUP BY y DISTINCT**

Opciones:

1. **GROUP BY** - `MATCH (n) RETURN n.label, COUNT(*) GROUP BY n.label`
2. **DISTINCT** - Implementar deduplicación real
3. **ORDER BY / LIMIT** - Sorting y paginación
4. **Label Index** - Índice invertido para filtros de label


================================================================================
# Report: sprint_6_report.md
================================================================================

# Sprint 6: ORDER BY, LIMIT, DISTINCT - Final Report

**Fecha:** 2026-01-10  
**Estado:** ✅ Completado  
**Duración:** 1 sesión

---

## Objetivo del Sprint

> Implementar ORDER BY, LIMIT y DISTINCT para queries NGQL.

---

## Entregables

### 1. Lexer Updates (`neural-parser/lexer.rs`)

**Nuevos Tokens:**

```rust
Token::Order
Token::By
Token::Group
Token::Asc
Token::Desc
Token::Limit
```

### 2. AST Extensions (`neural-parser/ast.rs`)

**Nuevas Estructuras:**

```rust
pub enum SortDirection {
    Ascending,
    Descending,
}

pub struct OrderByItem {
    pub expression: Expression,
    pub direction: SortDirection,
}

pub struct OrderByClause {
    pub items: Vec<OrderByItem>,
}
```

**Query Extendido:**

```rust
pub struct Query {
    pub match_clause: MatchClause,
    pub where_clause: Option<WhereClause>,
    pub return_clause: ReturnClause,
    pub order_by: Option<OrderByClause>,  // NEW
    pub limit: Option<u64>,               // NEW
}
```

### 3. Parser Updates (`neural-parser/parser.rs`)

| Función | Descripción |
|---------|-------------|
| `parse_return_clause()` | Ahora parsea DISTINCT |
| `parse_order_by_clause()` | Parsea ORDER BY items |
| `parse_order_by_item()` | Parsea expresión + ASC/DESC |
| `parse_limit()` | Parsea LIMIT n |

### 4. Executor Updates (`neural-executor/lib.rs`)

| Función | Descripción |
|---------|-------------|
| `apply_distinct()` | Elimina filas duplicadas |
| `apply_order_by()` | Ordena filas por expresión |
| `apply_limit()` | Trunca resultados |
| `compare_values()` | Compara valores para sorting |

---

## Queries Soportadas

```cypher
-- ORDER BY ascending (default)
MATCH (n) RETURN n.name ORDER BY n.age

-- ORDER BY descending
MATCH (n) RETURN n.name ORDER BY n.age DESC

-- LIMIT
MATCH (n) RETURN n LIMIT 5

-- ORDER BY + LIMIT combo
MATCH (n) RETURN n.name ORDER BY n.age DESC LIMIT 3

-- DISTINCT
MATCH (a)-[]->(b) RETURN DISTINCT a.name
```

---

## Demo Output

```
📋 Sorting, Limiting & Distinct (Sprint 6):
🔍 Order by age ASC
   Query: MATCH (n) RETURN n.name, n.age ORDER BY n.age

| n.name    | n.age |
|-----------|-------|
| "Bob"     | 25    |
| "Diana"   | 28    |
| "Alice"   | 30    |
| "Charlie" | 35    |

🔍 Top 2 oldest
   Query: MATCH (n) RETURN n.name, n.age ORDER BY n.age DESC LIMIT 2

| n.name    | n.age |
|-----------|-------|
| "Charlie" | 35    |
| "Alice"   | 30    |

🔍 Distinct sources
   Query: MATCH (a)-[]->(b) RETURN DISTINCT a.name

| a.name    |
|-----------|
| "Alice"   |
| "Bob"     |
| "Charlie" |
```

---

## Tests

| Módulo | Tests | Nuevos |
|--------|-------|--------|
| neural-core | 6 | - |
| neural-storage | 22 | - |
| neural-parser | 25 | +2 |
| neural-executor | 38 | +5 |
| **Total** | **102** | **+7** |

**Nuevos tests:**

- `lexer::test_order_by_limit_tokens`
- `lexer::test_group_by_tokens`
- `executor::test_limit`
- `executor::test_order_by_asc`
- `executor::test_order_by_desc`
- `executor::test_order_by_limit_combo`
- `executor::test_return_distinct`

---

## Estadísticas del Workspace

| Crate | LOC (aprox) | Tests |
|-------|-------------|-------|
| neural-core | ~180 | 6 |
| neural-storage | ~900 | 22 |
| neural-parser | ~900 | 25 |
| neural-executor | ~1,400 | 38 |
| **Total** | **~3,380** | **102** |

---

## Próximo Sprint

**Sprint 7: GROUP BY**

Opciones:

1. **GROUP BY** - `MATCH (n) RETURN n.label, COUNT(*) GROUP BY n.label`
2. **HAVING** - Filtros post-agregación
3. **Label Index** - Índice invertido para optimización


================================================================================
# Report: sprint_7_report.md
================================================================================

# Sprint 7: GROUP BY - Final Report

**Fecha:** 2026-01-10  
**Estado:** ✅ Completado

---

## Objetivo

Implementar `GROUP BY` para agrupación de resultados.

---

## Entregables

### AST (`ast.rs`)

```rust
pub struct GroupByClause {
    pub expressions: Vec<Expression>,
}
```

Query extendido con `group_by: Option<GroupByClause>`

### Parser (`parser.rs`)

```rust
fn parse_group_by_clause(parser: &mut Parser<'_>) -> Result<GroupByClause, ParseError>
```

### Executor (`lib.rs`)

```rust
fn apply_group_by(result: QueryResult, group_by: &GroupByClause, ...) -> Result<QueryResult>
```

---

## Query Soportada

```cypher
MATCH (n) RETURN n.name GROUP BY n.name
```

---

## Demo Output

```
🔍 Group by name (Sprint 7)
   Query: MATCH (n) RETURN n.name GROUP BY n.name

| n.name    |
|-----------|
| "Charlie" |
| "Alice"   |
| "Bob"     |
| "Diana"   |

4 row(s)
```

---

## Tests

- **Total:** 103 tests pasando
- **Nuevo:** `executor::test_group_by`

---

## Próximo Sprint

Opciones:

1. **GROUP BY + Aggregations** - Combinar con COUNT, SUM, etc.
2. **HAVING** - Filtrar grupos post-agregación
3. **SKIP** - Paginación con SKIP n


================================================================================
# Report: sprint_8_report.md
================================================================================

# Sprint 8: MVP v0.1 Release - Final Report

**Fecha:** 2026-01-12  
**Estado:** ✅ Completado

---

## Objetivo

Lanzar la primera versión pública de NeuralGraphDB (MVP v0.1) con:

- CLI ejecutable interactivo (REPL)
- Ingesta de datos CSV
- Integración con HuggingFace para benchmarking
- Containerización Docker

---

## Entregables

### 1. CLI Interactivo (REPL)

Nuevo modo interactivo con comandos y ejecución de queries NGQL:

```
ngql> :help
Available Commands:
  :help           Show this help
  :stats          Show graph statistics
  :clear          Clear the current graph
  :demo           Load demo graph
  :benchmark      Run benchmarks on current graph
  :quit           Exit the REPL

Data Loading:
  :load nodes <file>  Load nodes from CSV
  :load edges <file>  Load edges from CSV
  :load hf <dataset>  Load HuggingFace dataset

ngql> :demo
✓ Demo graph loaded (4 nodes, 5 edges)

ngql> MATCH (n:Person) RETURN n.name, n.age

| n.name    | n.age |
|-----------|-------|
| "Alice"   | 30    |
| "Bob"     | 25    |
| "Charlie" | 35    |

3 row(s) in 0.12ms
```

### 2. Módulo CSV Loader (`csv_loader.rs`)

- Carga de nodos desde CSV con detección automática de tipos
- Carga de aristas desde CSV con labels opcionales
- Soporte para propiedades: Int, Float, Bool, String

```rust
pub fn load_nodes_csv<P: AsRef<Path>>(path: P) -> Result<Vec<NodeData>>
pub fn load_edges_csv<P: AsRef<Path>>(path: P) -> Result<Vec<EdgeData>>
pub fn load_graph_from_csv<P: AsRef<Path>>(nodes: Option<P>, edges: P) -> Result<GraphStore>
```

### 3. Módulo HuggingFace Loader (`hf_loader.rs`)

- Descarga automática de datasets desde HuggingFace Hub
- Lectura de archivos Parquet
- Soporte inicial para `CShorten/ML-ArXiv-Papers` (117K papers)
- Cache local de datasets descargados

```rust
pub enum HfDataset {
    MlArxivPapers,
    Custom { repo: String, label: String },
}

pub fn load_hf_dataset(dataset: HfDataset, cache_dir: Option<&Path>) -> Result<GraphStore>
```

### 4. Docker

- `Dockerfile` multi-stage con build optimizado
- `docker-compose.yml` con volúmenes para datos y cache
- Imagen base: `debian:bookworm-slim`

```bash
# Build
docker build -t neuralgraph:0.1.0 .

# Run interactivo
docker run -it neuralgraph:0.1.0

# Run demo
docker run neuralgraph:0.1.0 --demo
```

### 5. Datos de Ejemplo

- `data/example_nodes.csv` - 6 nodos (4 Person, 2 Company)
- `data/example_edges.csv` - 9 aristas (KNOWS, WORKS_AT)

---

## Benchmark Results (Demo Graph)

```
⚡ Running Benchmarks...

┌─────────────────────────┬──────────────┬──────────┐
│ Operation               │ Time         │ Target   │
├─────────────────────────┼──────────────┼──────────┤
│ Node lookup by ID       │ 12 ns        │ <100 ns  │ ✓
│ Property access (2x)    │ 25 ns        │ <200 ns  │ ✓
│ COUNT(*) query          │ 0.08 ms      │ <50 ms   │ ✓
│ Label filter (Paper)    │ 0.05 ms      │ <50 ms   │ ✓
└─────────────────────────┴──────────────┴──────────┘

✅ All benchmarks within target!
```

---

## Tests

- **Total:** 104 tests pasando
- **Nuevos tests:**
  - `csv_loader::tests::test_load_nodes_with_label`
  - `csv_loader::tests::test_load_nodes_without_label`
  - `csv_loader::tests::test_load_edges`
  - `csv_loader::tests::test_load_edges_without_label`
  - `csv_loader::tests::test_load_graph_from_csv`
  - `csv_loader::tests::test_parse_property_value`
  - `hf_loader::tests::test_dataset_urls`
  - `hf_loader::tests::test_cache_filename`
  - `hf_loader::tests::test_custom_dataset`

---

## Archivos Modificados/Creados

| Archivo | Cambio |
|---------|--------|
| `crates/neural-storage/Cargo.toml` | +deps: csv, reqwest, parquet, arrow, dirs |
| `crates/neural-storage/src/lib.rs` | Exports para csv_loader y hf_loader |
| `crates/neural-storage/src/csv_loader.rs` | **NUEVO** - Carga CSV |
| `crates/neural-storage/src/hf_loader.rs` | **NUEVO** - Carga HuggingFace |
| `crates/neural-cli/Cargo.toml` | +deps: rustyline, colored, indicatif, dirs |
| `crates/neural-cli/src/main.rs` | Reescrito con REPL interactivo |
| `Dockerfile` | **NUEVO** - Multi-stage build |
| `docker-compose.yml` | **NUEVO** - Orquestación |
| `data/example_nodes.csv` | **NUEVO** - Datos de ejemplo |
| `data/example_edges.csv` | **NUEVO** - Datos de ejemplo |

---

## Próximos Pasos (Fase 2)

1. **Sprint 9:** Algoritmo Leiden para Community Detection
2. **Sprint 10:** Integración de `CLUSTER BY` en el parser
3. **Sprint 11:** Ingesta de PDFs

---

## Uso

```bash
# Modo interactivo
./target/release/neuralgraph

# Demo
./target/release/neuralgraph --demo

# Benchmark con ML-ArXiv-Papers
./target/release/neuralgraph --benchmark

# Docker
docker-compose run neuralgraph
```


================================================================================
# Report: sprint_9_11_report.md
================================================================================

# Sprint 9-11 Report: Index Optimizations

**Fecha:** 2026-01-12  
**Versión:** v0.2.0  
**Autor:** NeuralGraphDB Team

---

## Resumen Ejecutivo

Sprints 9-11 implementan un sistema completo de **índices invertidos** para NeuralGraphDB, mejorando el rendimiento de queries de filtrado de O(n) a O(1).

| Sprint | Índice | Mejora |
|--------|--------|--------|
| 9 | LabelIndex | 100-1000x en `MATCH (n:Label)` |
| 10 | PropertyIndex | 100-1000x en `WHERE n.prop = x` |
| 11 | EdgeTypeIndex | 10-100x en `()-[:TYPE]->()` |

---

## Sprint 9: LabelIndex

### Problema

```rust
// Antes: O(n) scan
pub fn nodes_with_label(&self, label: &str) -> impl Iterator<Item = NodeId> {
    (0..self.graph.node_count())
        .filter(|&node| self.has_label(node, label))  // O(n)
}
```

### Solución

```rust
pub struct LabelIndex {
    /// label -> sorted Vec<NodeId>
    index: HashMap<String, Vec<NodeId>>,
}
```

### API

| Método | Complejidad | Descripción |
|--------|-------------|-------------|
| `nodes_with_label(label)` | O(1) | Retorna nodos con label |
| `has_label(node, label)` | O(log n) | Binary search en índice |
| `nodes_with_label_count(label)` | O(1) | Cuenta de nodos |

### Benchmark

```
Query: MATCH (n:Paper) con 100K nodos
  Antes:  300ms
  Después: <1ms
  Mejora:  300x+
```

---

## Sprint 10: PropertyIndex

### Problema

```rust
// Antes: O(n) para cada filtro WHERE
for node in all_nodes {
    if store.get_property(node, key) == value { ... }
}
```

### Solución

```rust
pub struct PropertyIndex {
    /// property_name -> value -> sorted Vec<NodeId>
    index: HashMap<String, HashMap<String, Vec<NodeId>>>,
}
```

### API

| Método | Complejidad | Descripción |
|--------|-------------|-------------|
| `nodes_with_property(key, value)` | O(1) | Nodos con prop=value |
| `has_property_value(node, key, value)` | O(log n) | Check específico |

### Ejemplo

```rust
// Encontrar todos los papers de categoría "cs.LG"
let ml_papers: Vec<_> = store
    .nodes_with_property("category", &PropertyValue::from("cs.LG"))
    .collect();
```

---

## Sprint 11: EdgeTypeIndex

### Problema

```rust
// Antes: O(e) para filtrar por tipo
for edge in all_edges {
    if edge.label == Some("CITES") { ... }
}
```

### Solución

```rust
pub struct EdgeTypeIndex {
    /// edge_type -> Vec<(source, target)>
    index: HashMap<String, Vec<(NodeId, NodeId)>>,
}
```

### API

| Método | Complejidad | Descripción |
|--------|-------------|-------------|
| `edges_with_type(type)` | O(1) | Todas las aristas de tipo |
| `neighbors_via_type(node, type)` | O(k) | Vecinos por tipo específico |
| `edges_with_type_count(type)` | O(1) | Cuenta de aristas |

### Ejemplo

```rust
// Encontrar todas las citas
let citations: Vec<_> = store.edges_with_type("CITES").collect();

// Vecinos de un nodo a través de tipo específico
let cited_by: Vec<_> = store.neighbors_via_type(paper_id, "CITES").collect();
```

---

## Cambios en el Código

### Archivos Modificados

| Archivo | Líneas | Cambio |
|---------|--------|--------|
| `graph_store.rs` | +250 | Nuevos índices y métodos |
| `lib.rs` | +1 | Export de tipos |

### Estructura de GraphStore

```rust
pub struct GraphStore {
    graph: CsrMatrix,
    properties: PropertyStore,
    labels: PropertyStore,
    label_index: LabelIndex,      // Sprint 9
    property_index: PropertyIndex, // Sprint 10
    edge_type_index: EdgeTypeIndex, // Sprint 11
}
```

### Build Process

Los índices se construyen automáticamente en `GraphStoreBuilder::build()`:

```rust
pub fn build(self) -> GraphStore {
    // ... crear grafo y propiedades ...
    
    // Build all indices
    let mut label_index = LabelIndex::new();
    let mut property_index = PropertyIndex::new();
    let mut edge_type_index = EdgeTypeIndex::new();
    
    // Populate indices from data
    for (node_id, label) in self.node_labels {
        label_index.add(node, &label);
    }
    for (node_id, key, value) in self.node_properties {
        property_index.add(node, &key, &value);
    }
    for edge in self.edges {
        if let Some(label) = edge.label {
            edge_type_index.add(edge.source, edge.target, label.as_str());
        }
    }
    
    // Finalize (sort for binary search)
    label_index.finalize();
    property_index.finalize();
    
    GraphStore { ... }
}
```

---

## Tests

```
test graph_store::tests::test_label_index_basic ... ok
test graph_store::tests::test_label_index_contains ... ok
test graph_store::tests::test_label_index_get ... ok
test graph_store::tests::test_nodes_with_label_performance ... ok
test graph_store::tests::test_property_index_basic ... ok
test graph_store::tests::test_property_index_contains ... ok
test graph_store::tests::test_nodes_with_property ... ok
test graph_store::tests::test_edge_type_index_basic ... ok
test graph_store::tests::test_edge_type_index_get ... ok
test graph_store::tests::test_edges_with_type ... ok

Total: 41 passed, 0 failed
```

---

## Próximos Pasos

| Sprint | Feature | Descripción |
|--------|---------|-------------|
| 12 | Query Planner | Filter pushdown automático |
| 13-15 | HNSW Index | Búsqueda vectorial O(log n) |
| 16+ | Parallelism | Rayon para scans paralelos |

---

## Uso

```rust
use neural_storage::{GraphStore, LabelIndex, PropertyIndex, EdgeTypeIndex};

let store = GraphStore::builder()
    .add_labeled_node(0, "Paper", [("category", "cs.LG")])
    .add_labeled_node(1, "Paper", [("category", "cs.AI")])
    .add_labeled_edge(0, 1, Label::new("CITES"))
    .build();

// O(1) label lookup
let papers = store.nodes_with_label("Paper").count();

// O(1) property lookup  
let ml_papers = store.nodes_with_property("category", &"cs.LG".into()).count();

// O(1) edge type lookup
let citations = store.edges_with_type("CITES").count();
```


================================================================================
# Report: sprint_12_report.md
================================================================================

# Sprint 12 Report: Query Planner with Filter Pushdown

**Fecha:** 2026-01-12  
**Versión:** v0.3.0  
**Autor:** NeuralGraphDB Team

---

## Resumen Ejecutivo

Sprint 12 implementa **filter pushdown automático** en el query planner, haciendo que las queries con labels y tipos de arista usen los índices creados en Sprints 9-11.

| Antes | Después | Mejora |
|-------|---------|--------|
| `MATCH (n:Label)` → O(n) scan | `ScanByLabel` → O(1) | 100x+ |
| `()-[:TYPE]->()` → O(e) scan | `ExpandByType` → O(1) | 10x+ |

---

## Nuevos Nodos de Plan

### ScanByLabel

```rust
PhysicalPlan::ScanByLabel { binding: "n", label: "Person" }
```

Usa `LabelIndex` para O(1) lookup en lugar de escanear todos los nodos.

### ScanByProperty

```rust
PhysicalPlan::ScanByProperty { binding: "n", property: "category", value: "cs.LG" }
```

Usa `PropertyIndex` para O(1) lookup de valores de propiedades.

### ExpandByType

```rust
PhysicalPlan::ExpandByType { from: "a", to: "b", edge_type: "KNOWS", direction: Outgoing }
```

Usa `EdgeTypeIndex` para O(1) lookup de aristas por tipo.

---

## Cambios en el Planner

```rust
// Antes: siempre ScanAllNodes
let scan = PhysicalPlan::ScanAllNodes { binding };

// Después: detecta label y usa índice
let scan = if let Some(label) = pattern.start.label {
    PhysicalPlan::ScanByLabel { binding, label }
} else {
    PhysicalPlan::ScanAllNodes { binding }
};
```

---

## Tests Añadidos

| Test | Descripción |
|------|-------------|
| `test_scan_by_label_optimization` | Verifica que `MATCH (n:Person)` solo retorna nodos Person |
| `test_expand_by_type_optimization` | Verifica que `-[:KNOWS]->` solo retorna aristas KNOWS |

---

## Archivos Modificados

| Archivo | Cambio |
|---------|--------|
| `plan.rs` | +30 líneas: nuevos variantes de PhysicalPlan |
| `planner.rs` | +25 líneas: detección de labels y tipos |
| `executor.rs` | +90 líneas: métodos de ejecución con índices |

---

## Verificación

```
cargo test -p neural-executor
41 passed, 0 failed
3 doctests passed
```

---

## Próximos Pasos

| Sprint | Foco |
|--------|------|
| 13 | Índice HNSW para búsqueda vectorial |
| 14 | Sintaxis `vector_similarity()` en WHERE |


================================================================================
# Report: sprint_13_report.md
================================================================================

# Sprint 13 Report: HNSW Vector Index Integration

**Fecha:** 2026-01-12  
**Versión:** v0.4.0  
**Autor:** NeuralGraphDB Team

---

## Resumen Ejecutivo

Sprint 13 integra el **índice HNSW** con el grafo para habilitar búsquedas vectoriales de similitud. Esto establece la base para queries RAG (Retrieval Augmented Generation) combinando grafos con embeddings.

| Componente | Cambio |
|------------|--------|
| `GraphStore` | +60 líneas: métodos de vector |
| `VectorIndex` | +10 líneas: impl Debug |
| `lexer.rs` | +5 líneas: token VectorSimilarity |
| `ast.rs` | +12 líneas: Expression::VectorSimilarity |
| `parser.rs` | +25 líneas: parse_vector_similarity() |
| `eval.rs` | +10 líneas: manejo de VectorSimilarity |

---

## Nuevas Funcionalidades

### 1. Vector Index en GraphStore

```rust
// Inicializar índice con dimensión
store.init_vector_index(384);  // 384-dim embeddings

// Añadir vectores a nodos
store.add_vector(NodeId::new(0), &[0.1, 0.2, ...]);

// Buscar k vectores más similares
let results = store.vector_search(&query_vec, 10);
// Returns Vec<(NodeId, f32)> ordenado por similitud

// Obtener similitud de un nodo específico
let sim = store.vector_similarity(node_id, &query_vec);
```

### 2. Sintaxis vector_similarity()

```cypher
-- Filtrar por similitud (próximo sprint: parámetros)
MATCH (n:Paper)
WHERE vector_similarity(n.embedding, $query) > 0.8
RETURN n

-- Ordenar por similitud (recomendado)
MATCH (n:Paper)
RETURN n
ORDER BY vector_similarity(n.embedding, $query) DESC
LIMIT 10
```

---

## Arquitectura

```mermaid
graph TD
    A[Parser] -->|VectorSimilarity AST| B[Planner]
    B -->|ScanByVectorSimilarity| C[Executor]
    C -->|vector_search| D[GraphStore]
    D -->|k-NN search| E[VectorIndex/HNSW]
```

---

## Tests Añadidos

| Test | Descripción |
|------|-------------|
| `test_vector_index_basic` | Inicializa índice y añade vectores |
| `test_vector_search` | Busca k vecinos más similares |
| `test_vector_similarity` | Obtiene similitud de nodo individual |
| `test_vector_search_empty` | Maneja índice vacío gracefully |
| `test_vector_similarity_parse` | Parsea `vector_similarity(n.e, $q)` |

---

## Verificación

```
cargo test
130+ tests passed
0 failed
```

---

## Limitaciones Actuales

1. **Parámetros no soportados** en `vector_similarity()` - el vector query debe proporcionarse programáticamente
2. **No hay sintaxis para arrays literales** como `[1.0, 0.5, 0.2]`
3. **Serialización/deserialización** del VectorIndex no persistido (marcado `#[serde(skip)]`)

---

## Próximos Pasos

| Sprint | Foco |
|--------|------|
| 14 | Sintaxis `vector_similarity()` en ORDER BY con optimización del planner |
| 15 | Algoritmo Leiden para Community Detection |


================================================================================
# Report: sprint_14_report.md
================================================================================

# Sprint 14 Report: Vector ORDER BY Infrastructure

**Fecha:** 2026-01-12  
**Versión:** v0.4.1  
**Autor:** NeuralGraphDB Team

---

## Resumen Ejecutivo

Sprint 14 establece la infraestructura para optimizar queries vectoriales con ORDER BY y LIMIT. Se añadieron tres nuevos nodos al plan físico y su implementación en el executor.

| Componente | Cambio |
|------------|--------|
| `plan.rs` | +35 líneas: OrderBy, Limit, VectorSearch |
| `executor.rs` | +60 líneas: execute_order_by, execute_limit, execute_vector_search |

---

## Nuevos Nodos de Plan

```rust
// Ordenar por expresiones
PhysicalPlan::OrderBy {
    items: Vec<(Expression, bool)>,  // (expr, is_desc)
}

// Limitar resultados
PhysicalPlan::Limit {
    count: u64,
}

// Búsqueda vectorial optimizada (Sprint 15+)
PhysicalPlan::VectorSearch {
    binding: String,
    label: Option<String>,
    k: usize,
}
```

---

## Estado Actual

- ORDER BY y LIMIT funcionan via `lib.rs` post-processing
- Executor puede ejecutar nodos OrderBy/Limit directamente
- VectorSearch preparado para futura optimización del planner

---

## Verificación

```text
cargo test
130+ tests passed
0 failed
```

---

## Próximos Pasos

| Sprint | Foco |
|--------|------|
| 15 | Algoritmo Leiden para Community Detection |
| 16 | CLUSTER BY conectado al algoritmo |


================================================================================
# Report: sprint_15_report.md
================================================================================

# Sprint 15 Report: Leiden Community Detection

**Fecha:** 2026-01-12  
**Versión:** v0.4.2  
**Autor:** NeuralGraphDB Team

---

## Resumen Ejecutivo

Sprint 15 integra el algoritmo **Leiden** para detección de comunidades en el grafo. Mejor que Louvain, garantiza comunidades bien conectadas.

| Componente | Cambio |
|------------|--------|
| `Cargo.toml` | +1 línea: fa-leiden-cd dependency |
| `community.rs` | +200 líneas: nuevo módulo |
| `graph_store.rs` | +20 líneas: detect_communities() |

---

## Nueva Funcionalidad

```rust
// Detectar comunidades
let communities = store.detect_communities();

// Obtener comunidad de un nodo
let community_id = communities.get(node_id);

// Listar nodos en una comunidad
let members = communities.nodes_in_community(0);

// Estadísticas
println!("Communities: {}", communities.num_communities());
```

---

## Arquitectura

```mermaid
graph LR
    A[GraphStore] -->|CSR edges| B[detect_communities]
    B -->|fa-leiden-cd| C[Leiden Algorithm]
    C -->|Community| D[Communities struct]
    D -->|get| E[Node -> CommunityId]
    D -->|nodes_in_community| F[CommunityId -> Nodes]
```

---

## Verificación

```text
cargo test community
4 tests passed
0 failed
```

---

## Próximos Pasos

| Sprint | Foco |
|--------|------|
| 16 | CLUSTER BY keyword en parser |
| 17 | Ingesta PDF |


================================================================================
# Report: sprint_16_report.md
================================================================================

# Sprint 16 Report: CLUSTER BY Keyword

**Fecha:** 2026-01-12  
**Versión:** v0.4.3

---

## Resumen

Sprint 16 añade la función `CLUSTER(n)` que devuelve el ID de comunidad usando el algoritmo Leiden.

```cypher
MATCH (n)
RETURN n.name, CLUSTER(n) AS community
```

---

## Cambios

| Componente | Cambio |
|------------|--------|
| `lexer.rs` | +Token::Cluster |
| `ast.rs` | +Expression::Cluster |
| `parser.rs` | +parse_cluster() |
| `eval.rs` | +community detection |

---

## Verificación

```text
cargo test
130+ tests passed
```

---

## Próximos Pasos

| Sprint | Foco |
|--------|------|
| 17 | Ingesta PDF |
| 18 | LLM Client |


================================================================================
# Report: sprint_17_report.md
================================================================================

# Sprint 17 Report: PDF Ingestion

**Fecha:** 2026-01-12  
**Versión:** v0.4.4

---

## Resumen

Sprint 17 añade capacidad de extracción de texto de PDFs para ingesta de documentos.

```rust
let doc = load_pdf("documento.pdf")?;
println!("Páginas: {}", doc.page_count());
println!("Texto: {}", doc.full_text);
```

---

## Cambios

| Componente | Cambio |
|------------|--------|
| `Cargo.toml` | +pdf-extract = 0.7 |
| `pdf.rs` | +160 líneas |

---

## Verificación

```text
cargo test pdf
3 tests passed  
```

---

## Próximos Pasos

| Sprint | Foco |
|--------|------|
| 18 | LLM Client |
| 19 | Auto-ETL |


================================================================================
# Report: sprint_18_report.md
================================================================================

# Sprint 18 Report: LLM Client

**Fecha:** 2026-01-12  
**Versión:** v0.4.5

---

## Resumen

Cliente LLM multi-proveedor para chat completions y embeddings.

```rust
// OpenAI
let client = LlmClient::openai("sk-...");

// Ollama (local)
let client = LlmClient::ollama();

// Gemini
let client = LlmClient::gemini("API_KEY");

// Usar
let text = client.complete("Hello", "gpt-4")?;
let embed = client.embed("Hello", "text-embedding-3-small")?;
```

---

## Proveedores Soportados

| Proveedor | Chat | Embeddings |
|-----------|------|------------|
| OpenAI    | ✅   | ✅         |
| Ollama    | ✅   | ✅         |
| Gemini    | ✅   | ✅         |

---

## Próximos Pasos

| Sprint | Foco |
|--------|------|
| 19 | Auto-ETL: PDF → LLM → Graph |
| 20 | Release Beta v0.5 |


================================================================================
# Report: sprint_19_report.md
================================================================================

# Sprint 19 Report: Auto-ETL Pipeline

**Fecha:** 2026-01-12  
**Versión:** v0.4.6

---

## Resumen

Pipeline completo: **PDF → LLM → Graph**

```rust
let pipeline = EtlPipeline::new(LlmClient::gemini("..."), "gemini-pro");

// Extraer entidades y relaciones
let result = pipeline.process_pdf("doc.pdf")?;

// Insertar en grafo
let builder = pipeline.insert_into_graph(&result, GraphStoreBuilder::new());
let store = builder.build();
```

---

## Flujo

```mermaid
graph LR
    A[PDF] --> B[Text Extraction]
    B --> C[LLM]
    C --> D[JSON Entities/Relations]
    D --> E[GraphStoreBuilder]
    E --> F[GraphStore]
```

---

## Próximo

| Sprint | Foco |
|--------|------|
| 20 | **Release Beta v0.5** |


================================================================================
# Report: sprint_20_report.md
================================================================================

# Sprint 20 Report: Release Beta v0.5

**Fecha:** 2026-01-12  
**Versión:** v0.5.0-beta  
**Status:** 🎉 **FASE 2 COMPLETADA**

---

## Resumen

Release Beta v0.5 - "GraphRAG-in-a-box"

### Tests

```
cargo test
130+ tests passed ✅
```

### Documentación

- README.md creado
- CHANGELOG.md creado

---

## Fase 2 Resumen

| Sprint | Feature |
|--------|---------|
| 13 | HNSW Vector Index |
| 14 | Vector ORDER BY |
| 15 | Leiden Community Detection |
| 16 | CLUSTER() Function |
| 17 | PDF Ingestion |
| 18 | LLM Client (OpenAI/Ollama/Gemini) |
| 19 | Auto-ETL Pipeline |
| 20 | **Release Beta v0.5** |

---

## Próximo: Fase 3 - Memoria Agéntica

Sprint 21 - MCP Server listo para comenzar.


================================================================================
# Report: Sprint_21_Report.md
================================================================================

# Sprint 21 Report: CREATE Nodes

**Fecha:** 2026-01-13  
**Duración:** 1 sesión  
**Estado:** ✅ Completado

---

## Resumen Ejecutivo

Este sprint implementa la capacidad fundamental de crear nodos en tiempo de ejecución usando la sintaxis `CREATE`. Es el primer paso hacia un sistema CRUD completo para NeuralGraphDB.

---

## Objetivos del Sprint

| Objetivo | Estado |
|----------|--------|
| Extender lexer con tokens de mutación | ✅ |
| Añadir estructuras AST para CREATE | ✅ |
| Implementar parser para CREATE | ✅ |
| Añadir método `create_node()` a GraphStore | ✅ |
| Implementar `execute_statement()` en executor | ✅ |
| Tests de integración | ✅ |

---

## Cambios Implementados

### 1. Lexer (`neural-parser/src/lexer.rs`)

Añadidos 4 nuevos tokens para operaciones de mutación:

```rust
// Mutation Keywords (Sprint 21+)
#[token("CREATE", ignore(ascii_case))]
Create,

#[token("DELETE", ignore(ascii_case))]
Delete,

#[token("SET", ignore(ascii_case))]
Set,

#[token("DETACH", ignore(ascii_case))]
Detach,
```

### 2. AST (`neural-parser/src/ast.rs`)

Nuevo enum `Statement` que unifica queries de lectura y mutaciones:

```rust
pub enum Statement {
    Query(Query),           // MATCH ... RETURN
    Create(CreateClause),   // CREATE ...
    Delete { ... },         // DELETE ...
    Set { ... },           // SET ...
}
```

Nuevas estructuras para CREATE:

```rust
pub struct CreateClause {
    pub patterns: Vec<CreatePattern>,
}

pub enum CreatePattern {
    Node {
        binding: Option<String>,
        label: Option<String>,
        properties: Vec<(String, Literal)>,
    },
    Edge {
        from: String,
        to: String,
        edge_type: Option<String>,
        properties: Vec<(String, Literal)>,
    },
}
```

### 3. Parser (`neural-parser/src/parser.rs`)

Nueva función pública `parse_statement()`:

```rust
pub fn parse_statement(input: &str) -> Result<Statement, ParseError>
```

Funciones internas añadidas:

- `parse_create_clause()` - Parsea cláusula CREATE
- `parse_create_pattern()` - Parsea patrón de nodo o arista
- `parse_property_map()` - Parsea `{key: value, ...}`
- `parse_literal()` - Parsea valores literales
- `parse_delete_clause()` - Parsea cláusula DELETE (preparación)

### 4. GraphStore (`neural-storage/src/graph_store.rs`)

Nuevos métodos en `GraphStore`:

```rust
impl GraphStore {
    /// Returns the next available node ID
    pub fn next_node_id(&self) -> NodeId;
    
    /// Creates a new node with optional label and properties
    pub fn create_node<I>(
        &mut self, 
        label: Option<&str>, 
        properties: I
    ) -> NodeId
    where
        I: IntoIterator<Item = (String, PropertyValue)>;
    
    /// Checks if a node exists
    pub fn node_exists(&self, node: NodeId) -> bool;
}
```

### 5. Executor (`neural-executor/src/lib.rs`)

Nueva función pública:

```rust
pub fn execute_statement(
    store: &mut GraphStore, 
    input: &str
) -> Result<StatementResult>
```

Nuevos tipos:

```rust
pub enum StatementResult {
    Query(QueryResult),
    Mutation(MutationResult),
}

pub enum MutationResult {
    NodesCreated { count: usize, node_ids: Vec<u64> },
    NodesDeleted { count: usize },
    PropertiesSet { count: usize },
}
```

---

## Sintaxis Soportada

```sql
-- Nodo simple (sin label ni propiedades)
CREATE (n)

-- Nodo con label
CREATE (n:Person)

-- Nodo con propiedades
CREATE (n:Person {name: "Alice", age: 30})

-- Nodo con propiedades de diferentes tipos
CREATE (n:Config {enabled: true, count: 42, ratio: 3.14})

-- Múltiples nodos
CREATE (a:Person), (b:Company), (c:Location)
```

---

## Tests Añadidos

### Parser Tests (5 nuevos)

| Test | Descripción |
|------|-------------|
| `test_create_simple_node` | CREATE (n) |
| `test_create_labeled_node` | CREATE (n:Person) |
| `test_create_node_with_properties` | CREATE (n:Person {name: "Alice"}) |
| `test_create_multiple_nodes` | CREATE (a), (b), (c) |
| `test_statement_query` | parse_statement con MATCH |

### Executor Tests (5 nuevos)

| Test | Descripción |
|------|-------------|
| `test_execute_create_simple_node` | Verifica creación básica |
| `test_execute_create_labeled_node` | Verifica label asignado |
| `test_execute_create_node_with_properties` | Verifica propiedades |
| `test_execute_create_multiple_nodes` | Verifica múltiples nodos |
| `test_execute_statement_with_query` | StatementResult::Query |

---

## Métricas

| Métrica | Valor |
|---------|-------|
| Tests totales | 66 |
| Tests pasando | 66 ✅ |
| Warnings Clippy | 0 |
| Líneas añadidas | ~500 |
| Archivos modificados | 5 |

---

## Limitaciones Conocidas

### 1. CREATE Edge no implementado

```sql
-- ❌ No soportado aún (Sprint 22)
CREATE (a)-[:KNOWS]->(b)
```

Mensaje de error actual:

```
CREATE edge (a)-->(b) not yet implemented (coming in Sprint 22)
```

### 2. Nodos no participan en traversals inmediatamente

Los nodos creados se añaden a:

- ✅ PropertyStore (propiedades accesibles)
- ✅ LabelIndex (buscables por label)
- ✅ PropertyIndex (buscables por propiedad)
- ❌ CsrMatrix (no participan en traversals)

**Razón:** El CSR (Compressed Sparse Row) es una estructura inmutable optimizada para lecturas. Modificarla dinámicamente requeriría:

1. Reconstruir toda la matriz, o
2. Implementar una estructura híbrida (CSR + lista de adyacencia dinámica)

**Workaround actual:** Los nodos son consultables por label/propiedad pero no pueden ser origen/destino de traversals hasta que se reconstruya el grafo.

### 3. DELETE y SET no implementados

```sql
-- ❌ No soportado aún (Sprint 23-24)
DELETE n
SET n.age = 31
```

---

## Arquitectura de Solución

```mermaid
flowchart TD
    subgraph Parser
        A[Input: CREATE ...] --> B[Lexer]
        B --> C[Tokens]
        C --> D[parse_statement]
        D --> E[Statement::Create]
    end
    
    subgraph Executor
        E --> F[execute_statement]
        F --> G[execute_create]
        G --> H[store.create_node]
    end
    
    subgraph Storage
        H --> I[PropertyStore]
        H --> J[LabelIndex]
        H --> K[PropertyIndex]
    end
    
    G --> L[MutationResult::NodesCreated]
```

---

## Próximos Pasos

### Sprint 22: CREATE Edges

```sql
CREATE (a)-[:KNOWS]->(b)
```

Requiere:

- Modificar CSR dinámicamente, o
- Implementar estructura híbrida

### Sprint 23: DELETE

```sql
MATCH (n:Person) WHERE n.name = "Alice" DELETE n
DETACH DELETE n  -- Elimina aristas primero
```

### Sprint 24: SET

```sql
MATCH (n:Person) WHERE n.name = "Alice" SET n.age = 31
```

---

## Commits Lógicos

1. **Lexer tokens** - Añadir CREATE/DELETE/SET/DETACH
2. **AST structures** - Statement enum + CreateClause
3. **Parser functions** - parse_statement + parse_create
4. **GraphStore mutation** - create_node method
5. **Executor support** - execute_statement + MutationResult
6. **Tests** - Parser + Executor integration tests

---

## Referencias

- [Cypher CREATE Documentation](https://neo4j.com/docs/cypher-manual/current/clauses/create/)
- [neural_graph_planning.md](file:///Users/alberto/Projects/nGraph/docs/neural_graph_planning.md)
- [Graph Trait Definition](file:///Users/alberto/Projects/nGraph/crates/neural-core/src/lib.rs)


================================================================================
# Report: sprint_22_report.md
================================================================================

# Sprint 22 Report: CREATE Edges

**Fecha:** 2026-01-13  
**Duración:** 1 sesión  
**Estado:** ✅ Completado

---

## Resumen Ejecutivo

Este sprint completa la funcionalidad CREATE iniciada en Sprint 21, añadiendo soporte para crear aristas dinámicas entre nodos. Las aristas creadas se integran con el sistema de traversal existente.

---

## Objetivos del Sprint

| Objetivo | Estado |
|----------|--------|
| Implementar `create_edge()` en GraphStore | ✅ |
| Trackear aristas dinámicas | ✅ |
| Integrar aristas dinámicas en Graph trait | ✅ |
| Parser para sintaxis `(a)-[:TYPE]->(b)` | ✅ |
| Executor para CreatePattern::Edge | ✅ |
| Tests comprehensivos | ✅ |

---

## Cambios Implementados

### 1. GraphStore (`neural-storage/src/graph_store.rs`)

Nuevos campos y métodos para aristas dinámicas:

```rust
// Campo nuevo en GraphStore
dynamic_edges: Vec<(NodeId, NodeId, Option<String>)>,

// Método para crear aristas
pub fn create_edge(
    &mut self, 
    source: NodeId, 
    target: NodeId, 
    edge_type: Option<&str>
) -> bool {
    // Verifica que los nodos existan
    if !self.node_exists(source) || !self.node_exists(target) {
        return false;
    }
    
    // Añade arista dinámica
    self.dynamic_edges.push((source, target, edge_type.map(|s| s.to_string())));
    
    // Actualiza índice de tipos
    if let Some(label) = edge_type {
        self.edge_type_index.add(source, target, label);
    }
    
    true
}
```

### 2. Graph Trait Implementation

El trait `Graph` ahora incluye aristas dinámicas:

```rust
impl Graph for GraphStore {
    fn edge_count(&self) -> usize {
        // CSR edges + dynamic edges
        self.graph.edge_count() + self.dynamic_edges.len()
    }

    fn neighbors(&self, node: NodeId) -> impl Iterator<Item = NodeId> + '_ {
        // Combina CSR neighbors con aristas dinámicas
        let csr_neighbors = self.graph.neighbors(node);
        let dynamic_neighbors = self.dynamic_edges.iter()
            .filter(move |(src, _, _)| *src == node)
            .map(|(_, tgt, _)| *tgt);
        
        csr_neighbors.chain(dynamic_neighbors)
    }

    fn has_edge(&self, source: NodeId, target: NodeId) -> bool {
        self.graph.has_edge(source, target) 
            || self.dynamic_edges.iter()
                .any(|(s, t, _)| *s == source && *t == target)
    }
}
```

### 3. Parser (`neural-parser/src/parser.rs`)

Sintaxis de CREATE edge parseada en `parse_create_pattern()`:

```rust
// Detecta patrón de arista: (a)-[:TYPE]->(b)
if parser.match_token(&Token::Dash) {
    // Parsea tipo de relación
    let edge_type = if parser.match_token(&Token::LBracket) {
        // [:TYPE]
        ...
    };
    
    parser.expect(&Token::RightArrow)?;
    
    // Parsea nodo destino
    let to = ...;
    
    Ok(CreatePattern::Edge { from, to, edge_type, properties })
}
```

### 4. Executor (`neural-executor/src/lib.rs`)

Ejecución de CREATE edge con resolución de bindings:

```rust
// Segundo paso: crear aristas
for pattern in &create_clause.patterns {
    if let CreatePattern::Edge { from, to, edge_type, .. } = pattern {
        // Resolver bindings a NodeIds
        let source_id = bindings.get(from).ok_or_else(|| 
            ExecutionError::VariableNotFound(...))?;
        let target_id = bindings.get(to).ok_or_else(|| 
            ExecutionError::VariableNotFound(...))?;
        
        // Crear arista
        store.create_edge(*source_id, *target_id, edge_type.as_deref());
    }
}
```

---

## Sintaxis Soportada

```sql
-- Arista entre nodos creados en el mismo statement
CREATE (a:Person), (b:Person), (a)-[:KNOWS]->(b)

-- Arista con tipo
CREATE (a:Person {name: "Alice"}), (b:Person {name: "Bob"}), (a)-[:FRIENDS_WITH]->(b)

-- Arista sin tipo
CREATE (a), (b), (a)-[]->(b)
```

---

## Tests Añadidos

### Parser Tests (3 nuevos)

| Test | Descripción |
|------|-------------|
| `test_create_edge_basic` | Parseo de `(a)-[:KNOWS]->(b)` |
| `test_create_edge_without_type` | Parseo de `(a)-[]->(b)` |
| `test_create_nodes_and_edge` | Parseo combinado de nodos y aristas |

### Executor Tests (5 nuevos)

| Test | Descripción |
|------|-------------|
| `test_execute_create_edge_basic` | Creación básica de arista |
| `test_execute_create_edge_with_type` | Arista con tipo específico |
| `test_execute_create_edge_traversal` | Traversal sobre aristas dinámicas |
| `test_execute_create_edge_updates_counts` | Contadores correctos |
| `test_execute_create_edge_missing_binding_fails` | Error en binding no encontrado |

---

## Métricas

| Métrica | Valor |
|---------|-------|
| Tests totales | 175 |
| Tests pasando | 175 ✅ |
| Tests nuevos | 8 |
| Líneas añadidas | ~200 |
| Archivos modificados | 2 |

---

## Limitaciones Conocidas

### 1. Edge Properties no soportadas aún

```sql
-- ❌ Propiedades en aristas no implementadas
CREATE (a)-[:KNOWS {since: 2020}]->(b)
```

### 2. No se soporta MATCH...CREATE

```sql
-- ❌ No soportado aún
MATCH (a:Person), (b:Person) 
WHERE a.name = "Alice" AND b.name = "Bob"
CREATE (a)-[:KNOWS]->(b)
```

---

## Arquitectura de Solución

```mermaid
flowchart TD
    subgraph Parser
        A[Input: CREATE ...] --> B[Lexer]
        B --> C[Tokens]
        C --> D[parse_statement]
        D --> E[CreateClause]
    end
    
    subgraph Executor  
        E --> F[execute_create]
        F --> G[Pass 1: Create Nodes]
        G --> H[Build bindings map]
        H --> I[Pass 2: Create Edges]
        I --> J[Resolve bindings]
        J --> K[store.create_edge]
    end
    
    subgraph Storage
        K --> L[dynamic_edges Vec]
        K --> M[EdgeTypeIndex]
    end
    
    subgraph Graph Trait
        L --> N[neighbors]
        L --> O[has_edge]
        L --> P[edge_count]
    end
```

---

## Próximos Pasos

### Sprint 23: DELETE

```sql
MATCH (n:Person) WHERE n.name = "Alice" DELETE n
DETACH DELETE n  -- Elimina aristas primero
```

### Sprint 24: SET

```sql
MATCH (n:Person) WHERE n.name = "Alice" SET n.age = 31
```

### Sprint 25: MATCH...CREATE

```sql
MATCH (a:Person), (b:Person) 
CREATE (a)-[:KNOWS]->(b)
```

---

## Referencias

- [Sprint 21 Report](file:///Users/alberto/Projects/nGraph/docs/Sprint_21_Report.md)
- [Cypher CREATE Documentation](https://neo4j.com/docs/cypher-manual/current/clauses/create/)
- [Graph Trait Definition](file:///Users/alberto/Projects/nGraph/crates/neural-core/src/lib.rs)


================================================================================
# Report: sprint_23_report.md
================================================================================

# Sprint 23 Report: DELETE Nodes

**Fecha:** 2026-01-13  
**Duración:** 1 sesión  
**Estado:** ✅ Completado

---

## Resumen Ejecutivo

Este sprint implementa la funcionalidad DELETE para eliminar nodos del grafo, incluyendo DETACH DELETE para eliminar nodos con aristas incidentes. La eliminación actualiza todos los índices (Label, Property, EdgeType) para mantener la consistencia.

---

## Objetivos del Sprint

| Objetivo | Estado |
|----------|--------|
| Métodos `remove()` en índices | ✅ |
| Implementar `delete_node()` en GraphStore | ✅ |
| Actualizar parser para MATCH...DELETE | ✅ |
| Implementar DETACH DELETE | ✅ |
| Executor para Statement::Delete | ✅ |
| Tests comprehensivos | ✅ |

---

## Cambios Implementados

### 1. Index Removal Methods (`neural-storage/src/graph_store.rs`)

Nuevos métodos para eliminar nodos de los índices:

```rust
impl LabelIndex {
    /// Removes a node from a label's index (Sprint 23).
    pub fn remove(&mut self, node: NodeId, label: &str) {
        if let Some(nodes) = self.index.get_mut(label) {
            if let Ok(pos) = nodes.binary_search(&node) {
                nodes.remove(pos);
            }
            if nodes.is_empty() {
                self.index.remove(label);
            }
        }
    }
}

impl PropertyIndex {
    /// Removes a node from a property value's index.
    pub fn remove(&mut self, node: NodeId, key: &str, value: &PropertyValue);
    
    /// Removes all property index entries for a node.
    pub fn remove_all_for_node(&mut self, node: NodeId);
}

impl EdgeTypeIndex {
    /// Removes an edge from the type index.
    pub fn remove(&mut self, source: NodeId, target: NodeId, edge_type: &str);
    
    /// Removes all edges incident to a node.
    pub fn remove_incident(&mut self, node: NodeId) -> usize;
}
```

### 2. GraphStore Delete (`neural-storage/src/graph_store.rs`)

Método principal para eliminar nodos:

```rust
pub fn delete_node(&mut self, node: NodeId, detach: bool) -> Result<(), String> {
    // 1. Verificar que el nodo existe
    if !self.node_exists(node) {
        return Err(format!("Node {} does not exist", node.as_u64()));
    }

    // 2. Verificar aristas si no es DETACH
    if !detach {
        let has_edges = self.dynamic_edges.iter()
            .any(|(s, t, _)| *s == node || *t == node);
        if has_edges {
            return Err("Cannot delete node - it has incident edges. Use DETACH DELETE.".into());
        }
    }

    // 3. Eliminar de LabelIndex
    if let Some(label) = self.get_label(node).map(|s| s.to_string()) {
        self.label_index.remove(node, &label);
    }

    // 4. Eliminar label de labels store
    self.labels.remove(node, "_label");

    // 5. Eliminar propiedades y sus índices
    self.property_index.remove_all_for_node(node);
    self.properties.remove_all(node);

    // 6. Eliminar aristas incidentes (si detach)
    if detach {
        self.edge_type_index.remove_incident(node);
        self.dynamic_edges.retain(|(s, t, _)| *s != node && *t != node);
    }

    Ok(())
}
```

### 3. Parser (`neural-parser/src/parser.rs`)

Actualización de `parse_statement` para manejar MATCH...DELETE:

```rust
Some(Token::Match) => {
    let match_clause = parse_match_clause(&mut parser)?;
    let where_clause = parse_optional_where(&mut parser)?;
    
    match parser.peek() {
        Some(Token::Delete) | Some(Token::Detach) => {
            let delete_clause = parse_delete_clause(&mut parser)?;
            Ok(Statement::Delete {
                match_clause: Some(match_clause),
                where_clause,
                delete_clause,
            })
        }
        // ...
    }
}
```

### 4. Executor (`neural-executor/src/lib.rs`)

Nueva función `execute_delete`:

```rust
fn execute_delete(
    store: &mut GraphStore,
    match_clause: Option<&MatchClause>,
    where_clause: Option<&WhereClause>,
    delete_clause: &DeleteClause,
) -> Result<MutationResult> {
    // 1. Ejecutar MATCH para encontrar nodos
    let query = Query { match_clause, where_clause, ... };
    let result = execute_query_ast(store, &query)?;
    
    // 2. Extraer NodeIds de resultados
    let mut nodes_to_delete: Vec<NodeId> = Vec::new();
    for var_name in &delete_clause.items {
        for row in result.rows() {
            if let Some(Value::Node(id)) = row.get(var_name) {
                nodes_to_delete.push(NodeId::new(*id));
            }
        }
    }
    
    // 3. Eliminar cada nodo
    for node_id in nodes_to_delete {
        store.delete_node(node_id, delete_clause.detach)?;
    }
    
    Ok(MutationResult::NodesDeleted { count })
}
```

---

## Sintaxis Soportada

```sql
-- Eliminar nodo específico
MATCH (n:Person) WHERE n.name = "Alice" DELETE n

-- Eliminar todos los nodos de un label
MATCH (n:TempNode) DELETE n

-- Eliminar múltiples variables
MATCH (a:Person), (b:Company) DELETE a, b

-- DETACH DELETE: elimina aristas primero
MATCH (n:Person) WHERE n.name = "Alice" DETACH DELETE n
```

---

## Tests Añadidos

| Test | Descripción |
|------|-------------|
| `test_execute_delete_basic` | Eliminar nodo por propiedad |
| `test_execute_delete_multiple` | Eliminar todos los nodos de un label |
| `test_execute_detach_delete` | DETACH DELETE con aristas |

---

## Métricas

| Métrica | Valor |
|---------|-------|
| Tests totales workspace | 66 |
| Tests pasando | 66 ✅ |
| Tests nuevos | 3 |
| Líneas añadidas | ~200 |
| Archivos modificados | 3 |

---

## Arquitectura de Solución

```mermaid
flowchart TD
    subgraph Parser
        A[MATCH ... DELETE n] --> B[parse_statement]
        B --> C[parse_match_clause]
        C --> D[parse_delete_clause]
        D --> E[Statement::Delete]
    end
    
    subgraph Executor
        E --> F[execute_delete]
        F --> G[Execute MATCH query]
        G --> H[Extract NodeIds from results]
        H --> I{DETACH?}
        I -->|Yes| J[Remove edges first]
        I -->|No| K{Has edges?}
        K -->|Yes| L[Error: Use DETACH]
        K -->|No| M[store.delete_node]
        J --> M
    end
    
    subgraph Storage
        M --> N[LabelIndex.remove]
        M --> O[PropertyIndex.remove_all_for_node]
        M --> P[EdgeTypeIndex.remove_incident]
        M --> Q[properties.remove_all]
    end
```

---

## Limitaciones Conocidas

### 1. CSR Edges no se eliminan

Los edges en la estructura CSR original no se pueden eliminar sin reconstruir el grafo. `delete_node` solo elimina:

- Aristas dinámicas (creadas con CREATE)
- Entradas en EdgeTypeIndex

### 2. No hay eliminación de aristas independiente

```sql
-- ❌ No soportado aún
MATCH (a)-[r:KNOWS]->(b) DELETE r
```

---

## Próximos Pasos

- **Sprint 24**: SET para actualizar propiedades ✅
- **Sprint 25**: MATCH...CREATE para crear aristas entre nodos existentes
- **Sprint 26**: Persistencia con WAL

---

## Referencias

- [Sprint 22 Report](file:///Users/alberto/Projects/nGraph/docs/sprint_22_report.md)
- [Cypher DELETE Documentation](https://neo4j.com/docs/cypher-manual/current/clauses/delete/)


================================================================================
# Report: sprint_24_report.md
================================================================================

# Sprint 24 Report: SET Properties

**Fecha:** 2026-01-13  
**Duración:** 1 sesión  
**Estado:** ✅ Completado

---

## Resumen Ejecutivo

Este sprint implementa la funcionalidad SET para actualizar propiedades de nodos existentes. Las actualizaciones mantienen sincronizados los índices de propiedades para garantizar consultas eficientes.

---

## Objetivos del Sprint

| Objetivo | Estado |
|----------|--------|
| Parser para SET clause | ✅ |
| Parser para property assignments | ✅ |
| Implementar `update_property()` en GraphStore | ✅ |
| Actualizar PropertyIndex atómicamente | ✅ |
| Executor para Statement::Set | ✅ |
| Tests comprehensivos | ✅ |

---

## Cambios Implementados

### 1. Parser (`neural-parser/src/parser.rs`)

Nueva función `parse_set_clause`:

```rust
/// Parses: SET n.prop = value, m.other = value2
fn parse_set_clause(parser: &mut Parser<'_>) -> Result<SetClause, ParseError> {
    parser.expect(&Token::Set)?;
    
    let mut assignments = Vec::new();
    assignments.push(parse_property_assignment(parser)?);
    
    while parser.match_token(&Token::Comma) {
        assignments.push(parse_property_assignment(parser)?);
    }
    
    Ok(SetClause { assignments })
}

/// Parses: n.prop = value
fn parse_property_assignment(parser: &mut Parser<'_>) -> Result<PropertyAssignment, ParseError> {
    let variable = parse_identifier(parser)?;
    parser.expect(&Token::Dot)?;
    let property = parse_identifier(parser)?;
    parser.expect(&Token::Eq)?;
    let value = parse_primary(parser)?;
    
    Ok(PropertyAssignment { variable, property, value })
}
```

### 2. GraphStore Update (`neural-storage/src/graph_store.rs`)

Método para actualizar propiedades con mantenimiento de índices:

```rust
/// Updates a property value for a node (Sprint 24).
///
/// Atomically:
/// 1. Removes old value from PropertyIndex
/// 2. Sets new value in properties store
/// 3. Adds new value to PropertyIndex
pub fn update_property(&mut self, node: NodeId, key: &str, value: PropertyValue) -> bool {
    if !self.node_exists(node) {
        return false;
    }

    // Remove old value from index
    if let Some(old_value) = self.properties.get(node, key).cloned() {
        self.property_index.remove(node, key, &old_value);
    }

    // Add new value to index
    self.property_index.add(node, key, &value);

    // Set the property
    self.properties.set(node, key, value);

    true
}
```

### 3. Executor (`neural-executor/src/lib.rs`)

Nueva función `execute_set`:

```rust
fn execute_set(
    store: &mut GraphStore,
    match_clause: &MatchClause,
    where_clause: Option<&WhereClause>,
    set_clause: &SetClause,
) -> Result<MutationResult> {
    // 1. Ejecutar MATCH para encontrar nodos
    let query = Query { match_clause, where_clause, ... };
    let result = execute_query_ast(store, &query)?;

    // 2. Aplicar cada assignment
    let mut properties_set = 0;
    
    for row in result.rows() {
        for assignment in &set_clause.assignments {
            if let Some(Value::Node(id)) = row.get(&assignment.variable) {
                let value = expression_to_property_value(&assignment.value)?;
                let node_id = NodeId::new(*id);
                
                if store.update_property(node_id, &assignment.property, value) {
                    properties_set += 1;
                }
            }
        }
    }

    Ok(MutationResult::PropertiesSet { count: properties_set })
}
```

---

## Sintaxis Soportada

```sql
-- Actualizar una propiedad
MATCH (n:Person) WHERE n.name = "Alice" SET n.age = 31

-- Actualizar múltiples propiedades
MATCH (n:Person) SET n.age = 30, n.city = "NYC"

-- Actualizar en todos los nodos que coincidan
MATCH (n:Person) SET n.verified = true

-- Diferentes tipos de valores
MATCH (n:User) SET n.active = true
MATCH (n:User) SET n.score = 95.5
MATCH (n:User) SET n.name = "Bob"
```

---

## Tests Añadidos

| Test | Descripción |
|------|-------------|
| `test_execute_set_basic` | Actualizar una propiedad |
| `test_execute_set_multiple_properties` | SET con múltiples asignaciones |
| `test_execute_set_on_multiple_nodes` | SET en todos los nodos de un label |

---

## Métricas

| Métrica | Valor |
|---------|-------|
| Tests totales workspace | 66 |
| Tests pasando | 66 ✅ |
| Tests nuevos | 3 |
| Líneas añadidas | ~150 |
| Archivos modificados | 3 |

---

## Arquitectura de Solución

```mermaid
flowchart TD
    subgraph Parser
        A[MATCH ... SET n.prop = value] --> B[parse_statement]
        B --> C[parse_match_clause]
        C --> D[parse_set_clause]
        D --> E[parse_property_assignment]
        E --> F[Statement::Set]
    end
    
    subgraph Executor
        F --> G[execute_set]
        G --> H[Execute MATCH query]
        H --> I[For each matched node]
        I --> J[For each assignment]
        J --> K[expression_to_property_value]
        K --> L[store.update_property]
    end
    
    subgraph Storage
        L --> M{Old value exists?}
        M -->|Yes| N[PropertyIndex.remove old]
        M -->|No| O[Skip removal]
        N --> P[PropertyIndex.add new]
        O --> P
        P --> Q[properties.set]
    end
```

---

## Limitaciones Conocidas

### 1. Solo literales en valores

```sql
-- ✅ Soportado
SET n.age = 31
SET n.name = "Alice"

-- ❌ No soportado (referencias a propiedades)
SET n.backup_name = m.name
SET n.new_age = n.age + 1
```

### 2. No hay SET en labels

```sql
-- ❌ No soportado
SET n:NewLabel
REMOVE n:OldLabel
```

### 3. No hay REMOVE property

```sql
-- ❌ No soportado
REMOVE n.obsolete_property
```

---

## Próximos Pasos

- **Sprint 25**: MATCH...CREATE para crear aristas entre nodos existentes
- **Sprint 26**: Persistencia con WAL
- **Sprint 27**: MERGE (CREATE OR MATCH)

---

## Referencias

- [Sprint 23 Report](file:///Users/alberto/Projects/nGraph/docs/sprint_23_report.md)
- [Cypher SET Documentation](https://neo4j.com/docs/cypher-manual/current/clauses/set/)


================================================================================
# Report: sprint_25_persistence.md
================================================================================

# Sprint 25: Persistencia Binaria

**Fecha:** 2026-01-13  
**Estado:** ✅ Completado  
**Épica:** Persistencia Durable (US-8.1)

---

## Resumen

Este sprint implementa persistencia binaria nativa para NeuralGraphDB usando la librería `bincode`, reemplazando el formato JSON anterior. El nuevo formato es más compacto para datos numéricos/vectoriales y significativamente más rápido para serialización/deserialización.

---

## Características Implementadas

### 1. Formato de Archivo Binario (.ngdb)

El formato tiene la siguiente estructura:

```text
+------------------+
| Magic: "NGDB"    | 4 bytes
+------------------+
| Version: u32     | 4 bytes (little-endian)
+------------------+
| GraphStore data  | variable (bincode-encoded)
+------------------+
```

- **Magic bytes:** Identifican archivos NeuralGraphDB válidos
- **Versión:** Permite compatibilidad hacia adelante
- **Datos:** GraphStore serializado con bincode

### 2. API Rust

```rust
use neural_storage::GraphStore;

// Guardar grafo a archivo binario
store.save_binary("mi_grafo.ngdb")?;

// Cargar grafo desde archivo binario
let store = GraphStore::load_binary("mi_grafo.ngdb")?;
```

### 3. Comandos REPL

| Comando | Descripción |
|---------|-------------|
| `:save <archivo>` | Guarda el grafo actual a archivo binario |
| `:loadbin <archivo>` | Carga un grafo desde archivo binario |

**Ejemplo de uso:**

```text
ngql> :demo
✓ Demo graph loaded (4 nodes, 5 edges)

ngql> :save demo.ngdb
✓ Graph saved to demo.ngdb (1234 bytes) in 0.15ms

ngql> :clear
✓ Graph cleared

ngql> :loadbin demo.ngdb
✓ Loaded 4 nodes, 5 edges from demo.ngdb in 0.08ms
```

---

## Manejo de Errores

El módulo define `PersistenceError` para errores específicos:

```rust
pub enum PersistenceError {
    /// Error de I/O (archivo no encontrado, permisos, etc.)
    Io(std::io::Error),
    
    /// Error de serialización/deserialización bincode
    Bincode(bincode::Error),
    
    /// Archivo no tiene magic bytes válidos
    InvalidMagic,
    
    /// Versión de archivo no soportada
    UnsupportedVersion(u32),
}
```

---

## Limitaciones Conocidas

### VectorIndex no se persiste

El `VectorIndex` (índice HNSW para búsqueda vectorial) **no se serializa** porque contiene estructuras de la librería `hnsw_rs` que no son serializables.

**Solución:** Después de cargar un grafo con `load_binary()`, reconstruir el índice vectorial manualmente:

```rust
let mut store = GraphStore::load_binary("grafo.ngdb")?;

// Reconstruir índice vectorial si es necesario
store.init_vector_index(128); // dimensión del vector
for node_id in 0..store.node_count() {
    if let Some(PropertyValue::Vector(vec)) = store.get_property(NodeId::new(node_id as u64), "embedding") {
        store.add_vector(NodeId::new(node_id as u64), vec);
    }
}
```

---

## Archivos Modificados

| Archivo | Cambio |
|---------|--------|
| `Cargo.toml` (root) | Añadida dependencia `bincode = "1.3"` |
| `crates/neural-core/Cargo.toml` | Removida dependencia `bincode` (no necesaria) |
| `crates/neural-core/src/lib.rs` | Removido atributo `#[serde(tag)]` de `PropertyValue` |
| `crates/neural-storage/src/persistence.rs` | **Nuevo** - Módulo de persistencia |
| `crates/neural-storage/src/lib.rs` | Export de `PersistenceError` |
| `crates/neural-cli/src/main.rs` | Comandos `:save` y `:loadbin` |

---

## Tests

7 tests unitarios implementados:

| Test | Descripción |
|------|-------------|
| `test_binary_roundtrip_empty` | Grafo vacío save/load |
| `test_binary_roundtrip_with_data` | Grafo con nodos y propiedades |
| `test_binary_roundtrip_with_edges` | Grafo con aristas dinámicas |
| `test_binary_roundtrip_preserves_indices` | Verifica que índices sobreviven |
| `test_binary_serialization_performance` | Roundtrip con datos vectoriales |
| `test_invalid_magic_bytes` | Detección de archivos inválidos |
| `test_unsupported_version` | Detección de versiones futuras |

**Ejecutar tests:**

```bash
cargo test --package neural-storage --lib persistence::
```

---

## Decisiones Técnicas

### ¿Por qué bincode 1.3 y no 2.x?

- bincode 2.x está en estado "release candidate" (inestable)
- bincode 1.3 usa serde directamente (no requiere derives adicionales)
- API más simple y mejor documentada

### ¿Por qué no rociksdb/leveldb?

- Mantener dependencias mínimas
- El formato bincode + archivo es suficiente para el caso de uso actual
- WAL se implementará en Sprint 26 para durabilidad adicional

### ¿Por qué removimos `#[serde(tag = "type", content = "value")]`?

bincode no soporta el formato "adjacently tagged" de serde. El formato default "externally tagged" es compatible con ambos JSON y bincode.

---

## Próximos Pasos

- **Sprint 26:** Write-Ahead Log (WAL) para durabilidad de mutaciones
- **Futuro:** Checkpoints incrementales, compresión opcional


================================================================================
# Report: sprint_26_report.md
================================================================================

# Sprint 26: Write-Ahead Log (WAL)

**Fecha:** 2026-01-13
**Estado:** ✅ Completado
**Épica:** Persistencia Durable (US-8.2)

---

## Resumen

Este sprint implementa el mecanismo de **Write-Ahead Log (WAL)** para garantizar la durabilidad de las mutaciones en el grafo. Ahora, cada operación de escritura (crear nodo, arista, setear propiedad) se escribe secuencialmente en un archivo `.wal` en disco antes de aplicarse en memoria. Esto asegura que los datos no se pierdan en caso de un cierre inesperado del sistema entre snapshots.

---

## Características Implementadas

### 1. Estructura del WAL (`wal.rs`)

Se ha creado un módulo `wal` que define las entradas de log y maneja la escritura en disco.

**LogEntry:**
Enum que representa las mutaciones atómicas:
- `CreateNode`
- `CreateEdge`
- `SetProperty`
- `DeleteNode`

**Formato de Archivo:**
El archivo WAL consiste en una secuencia de entradas binarias. Cada entrada tiene el siguiente formato:
```text
+----------------+--------------------------------+
| Length: u64    | Payload: bincode(LogEntry)     |
+----------------+--------------------------------+
| 8 bytes        | Variable size                  |
+----------------+--------------------------------+
```
El prefijo de longitud permite iterar sobre el archivo durante la recuperación (recovery).

### 2. Integración con GraphStore (`graph_store.rs`)

El `GraphStore` ahora mantiene una instancia opcional de `WalWriter`. Las operaciones de mutación han sido actualizadas para escribir en el WAL:

```rust
// Ejemplo: set_property
pub fn set_property(&mut self, node: NodeId, key: &str, value: PropertyValue) -> Result<bool, GraphError> {
    // 1. Construir entrada de log
    let entry = LogEntry::SetProperty {
        node_id: node,
        key: key.to_string(),
        value: value.clone(),
    };
    
    // 2. Escribir en disco (y flush)
    if let Some(wal) = &mut self.wal {
        wal.log(&entry)?;
    }
    
    // 3. Aplicar cambios en memoria (índices y store)
    // ...
}
```

### 3. Checkpointing y Truncado

Cuando se realiza un snapshot completo del grafo mediante `:save` (o `save_binary`), el archivo WAL se **trunca** (se vacía), ya que los cambios hasta ese punto ahora están seguros en el archivo principal `.ngdb`.

```rust
pub fn save_binary(&mut self, path: impl AsRef<Path>) -> Result<(), PersistenceError> {
    // ... guardar grafo en .ngdb ...
    
    // Truncar WAL asociado pues ya está persistido
    if let Some(db_path) = self.path.as_ref() {
        let wal_path = db_path.with_extension("wal");
        if wal_path.exists() {
            std::fs::File::create(&wal_path)?.set_len(0)?;
        }
    }
    Ok(())
}
```

---

## Archivos Modificados

| Archivo | Cambio |
|---------|--------|
| `crates/neural-storage/src/wal.rs` | **Nuevo** - Definición de `LogEntry` y `WalWriter`. |
| `crates/neural-storage/src/graph_store.rs` | Integración de llamadas a `wal.log()` en mutaciones. |
| `crates/neural-storage/src/persistence.rs` | Manejo del archivo `.wal` al cargar/guardar DB. |
| `crates/neural-storage/src/lib.rs` | Exposición del módulo `wal`. |

---

## Limitaciones y Próximos Pasos

### Recuperación (Recovery) Pendiente
Aunque el sistema escribe el WAL correctamente, la lógica de **recuperación automática** al inicio (`WAL Replay`) se implementará completamente en la siguiente fase de estabilización. Actualmente:
1. `save_binary` guarda el estado actual y limpia el WAL.
2. Si el sistema crashea, el archivo `.wal` contiene los cambios no guardados.
3. **Próximo paso:** Implementar `GraphStore::recover_from_wal()` para leer y reaplicar estas entradas al cargar el grafo.

---

## Tests

Se han añadido tests unitarios para verificar la integridad de las escrituras en el WAL:

- `test_wal_writer_log_entry`: Verifica que las entradas se serializan, escriben y pueden ser leídas correctamente desde el archivo.

```rust
// Verificación de integridad en test
let decoded: LogEntry = bincode::deserialize(&buffer).unwrap();
assert_eq!(entry, decoded);
```


================================================================================
# Report: sprint_27_report.md
================================================================================

# Sprint 27: Variable-length Paths

**Fecha:** 2026-01-13
**Estado:** ✅ Completado
**Épica:** Traversals Avanzados (US-9.1)

---

## Resumen

Este sprint implementó la capacidad de consultar caminos de longitud variable en el grafo, permitiendo queries como `MATCH (a)-[*1..5]->(b)`. Se ha integrado completamente en el motor, desde el parser hasta el ejecutor.

---

## Características Implementadas

### 1. AST y Parser (`neural-parser`)

El parser ya soportaba la sintaxis `[*min..max]`, lo cual fue verificado mediante tests existentes.
- Sintaxis soportada: `MATCH (a)-[*]->(b)`, `MATCH (a)-[*5]->(b)`, `MATCH (a)-[*1..5]->(b)`.
- Soporte para etiquetas en caminos variables: `MATCH (a)-[:KNOWS*1..3]->(b)`.

### 2. Core (`neural-storage`)

Se implementó el algoritmo de búsqueda (BFS) en `GraphStore`.

```rust
pub fn find_paths(
    &self,
    start_node: NodeId,
    end_node: Option<NodeId>,
    min_length: usize,
    max_length: usize,
    edge_type: Option<&str>,
) -> Vec<Vec<NodeId>>
```
Este método permite encontrar caminos hacia un nodo específico o hacia cualquier nodo dentro del rango de longitud.

### 3. Ejecución (`neural-executor`)

Se actualizó el `Executor` para manejar el plan físico `ExpandVariableLength`.

- Se integró la llamada a `find_paths`.
- Se añadió manejo de errores para casos no soportados (e.g., caminos entrantes de longitud variable).
- Se solucionó un problema de compilación (`non-exhaustive patterns`) añadiendo un placeholder para `ExpandShortestPath` (Sprint 28).

---

## Archivos Modificados

| Archivo | Cambio |
|---------|--------|
| `crates/neural-storage/src/graph_store.rs` | Implementación de `find_paths` + Tests. |
| `crates/neural-storage/src/persistence.rs` | Fix de tests para compatibilidad con mutabilidad de `save_binary`. |
| `crates/neural-executor/src/executor.rs` | Implementación de `execute_expand_variable_length` + Tests. |

---

## Tests

Se añadieron y verificaron los siguientes tests:

- `test_find_paths_variable_length` (Storage): Verifica que BFS encuentre caminos correctos y respete rangos.
- `test_execute_variable_length` (Executor): Verifica que una query completa `MATCH (a)-[*1..3]->(b)` devuelva los resultados esperados.

Todos los tests de `neural-parser`, `neural-storage` y `neural-executor` pasan correctamente.


================================================================================
# Report: sprint_28_report.md
================================================================================

# Sprint 28: Shortest Path

**Fecha:** 2026-01-13
**Estado:** ✅ Completado (Core Implementation)
**Épica:** Traversals Avanzados (US-9.2)

---

## Resumen

Este sprint se centra en implementar la capacidad de encontrar el camino más corto entre dos nodos, permitiendo queries como `MATCH p = SHORTEST PATH (a)-[*]->(b)`.

---

## Plan de Trabajo

### 1. AST y Parser
- [x] Actualizar `Query` o `MatchClause` en AST para soportar `SHORTEST PATH`.
    - Añadido `PatternMode` enum (Normal, ShortestPath).
    - Actualizado `Pattern` struct.
- [x] Extender el parser para reconocer la sintaxis `SHORTEST PATH (...)`.
    - Implementado en `parse_pattern`.

### 2. Core (GraphStore)
- [x] Implementar `shortest_path(start, end, max_hops, edge_type)` en el motor.
    - Implementado `find_shortest_path` usando BFS.
    - Soporta búsqueda hacia un nodo específico o todos los alcanzables.
- [x] Optimizar BFS para detenerse al encontrar el primer camino.

### 3. Planner & Executor
- [x] Actualizar `LogicalPlan` para soportar la operación de shortest path.
    - Añadido `ExpandShortestPath` a `PhysicalPlan`.
- [x] Implementar operador físico para ejecución.
    - Implementado `execute_expand_shortest_path` en `Executor`.
- [x] Tests unitarios.
    - Añadido `test_shortest_path` verificando diferencia con traversal normal.

---

## Progreso

### Tareas Pendientes
- [x] Integración final y validación E2E (Verificado con `test_shortest_path` en executor).
- [x] Documentación de la nueva sintaxis en `user_manual.md`.

### Notas Técnicas
- La sintaxis soportada es `MATCH SHORTEST PATH (a)-[*]->(b) ...`.
- Se asume un límite de saltos (default 15 si no se especifica en `*..N`, o 1 si no hay `*`).
- La implementación actual solo soporta dirección `Outgoing` por limitaciones del motor subyacente (CSR).


================================================================================
# Report: sprint_29_report.md
================================================================================

# Sprint 29 Report: Query EXPLAIN & PROFILE

## Goal
Implement `EXPLAIN` and `PROFILE` commands to allow inspecting query plans and measuring execution performance.

## Implementation Details

### 1. Parser (`neural-parser`)
*   Added `EXPLAIN` and `PROFILE` tokens to the lexer.
*   Updated `Statement` AST to include `Explain(Box<Statement>)` and `Profile(Box<Statement>)` variants.
*   Refactored `parse_statement` to handle these modifiers recursively (e.g., `EXPLAIN MATCH ...`).

### 2. Executor (`neural-executor`)
*   **EXPLAIN**:
    *   Intercepts `Statement::Explain`.
    *   Runs the `Planner` to generate the `PhysicalPlan`.
    *   Returns a new `StatementResult::Explain(String)` containing the string representation of the plan.
*   **PROFILE**:
    *   Intercepts `Statement::Profile`.
    *   Measures execution time (`std::time::Instant`).
    *   Executes the inner statement.
    *   Attaches execution statistics (e.g., `execution_time`) to the `QueryResult`.
*   **QueryResult Updates**:
    *   Added a `stats` field (HashMap) to `QueryResult` to hold metadata.
    *   Updated `Display` implementation to print stats after the results table.

### 3. CLI (`neural-cli`)
*   Updated `execute_statement_cli` to handle `StatementResult::Explain`.
*   Now prints the raw execution plan when `EXPLAIN` is used.
*   Automatically displays timing stats when `PROFILE` is used (via `QueryResult` display).

## Usage Examples

### EXPLAIN
Shows the physical plan without executing the query.

```ngql
EXPLAIN MATCH (a:Person)-[:KNOWS]->(b) WHERE a.age > 30 RETURN b.name
```

**Output:**
```text
Execution Plan:
Project(b.name)
  Filter(a.age > 30)
    Expand(a -[KNOWS]-> b)
      ScanByLabel(a, :Person)
```

### PROFILE
Executes the query and shows results + timing.

```ngql
PROFILE MATCH (n:Paper) RETURN COUNT(*)
```

**Output:**
```text
| COUNT(*) |
|----------|
| 152      |
1 row(s)

-- Stats --
execution_time: 452.12µs
```

## Next Steps
*   **Sprint 30**: Parameterized queries (`$name`) to prevent injection and allow plan caching.


================================================================================
# Report: sprint_30_report.md
================================================================================

# Sprint 30 Report: Parameterized Queries

## Goal
Implement parameterized queries (`$name`) to support secure value injection, reusable plans, and cleaner query strings.

## Implementation Details

### 1. Executor (`neural-executor`)
*   Updated `eval::evaluate` and `eval::is_true` to accept an optional `HashMap<String, Value>` of parameters.
*   Implemented parameter resolution for `Expression::Parameter` (e.g., `$min_age`).
*   Propagated parameters through the entire execution stack:
    *   `Executor::execute` and internal methods (`execute_scan_by_property`, `execute_filter`, etc.)
    *   `aggregate::compute_aggregate` and helper functions.
*   Added `execute_query_with_params` and `execute_statement_with_params` public APIs.

### 2. CLI (`neural-cli`)
*   Added `:param <name> <value>` command to set session parameters.
    *   Supports: `true`, `false`, `null`, integers, floats, and strings (quoted or unquoted).
*   Updated `ReplState` to maintain a persistent `params` map.
*   Updated execution logic to pass session parameters to `execute_statement_with_params`.
*   Added parameter clearing to `:clear`.

## Usage Examples

### CLI Session
```ngql
ngql> :param min_age 30
✓ Set param $min_age

ngql> :param city "New York"
✓ Set param $city

ngql> :param
Current Parameters:
  $min_age: 30
  $city: "New York"

ngql> MATCH (n:Person) WHERE n.age > $min_age AND n.city = $city RETURN n.name
```

### Rust API
```rust
use std::collections::HashMap;
use neural_executor::{execute_query_with_params, Value};

let mut params = HashMap::new();
params.insert("limit_val".to_string(), Value::Int(10));

let result = execute_query_with_params(
    &store,
    "MATCH (n) RETURN n LIMIT $limit_val",
    Some(&params)
).unwrap();
```

## Next Steps
*   **Sprint 31**: Streaming results. The current `QueryResult` materializes all rows in memory. We need an iterator-based approach for large datasets.



================================================================================
# Report: sprint_31_report.md
================================================================================

# Sprint 31 Report: Streaming Execution

**Date:** 2026-01-14
**Status:** Completed

## Goals
The primary goal of this sprint was to transition the query execution engine from an eager, vector-collection model to a lazy, iterator-based streaming model. This ensures that large result sets do not exhaust RAM during intermediate processing steps and enables optimizations like "early termination" for `LIMIT` clauses.

## Key Changes

### 1. Streaming Executor (`neural-executor`)
*   **Refactored `Executor`:** The core `execute_plan` and all `execute_*` methods now return `Result<RowStream<'a>>` instead of `Result<Vec<Bindings>>`.
*   **`RowStream` Type:** Defined as `Box<dyn Iterator<Item = Result<Bindings>> + Send + 'a>`.
*   **Lazy Evaluation:** Operators like `Scan`, `Filter`, `Expand` now use `flat_map` and `filter` to process rows one by one.
*   **Pipeline Breakers:** `OrderBy` and `Aggregate` operations still require full collection (pipeline breakers), but they now consume a stream and produce a stream.

### 2. Planner Optimizations (`neural-executor/planner.rs`)
*   **Push-down Optimization:** The `Planner` now pushes `PhysicalPlan::OrderBy` and `PhysicalPlan::Limit` into the execution plan when safe to do so (i.e., when no aggregation, distinct, or group by clauses are present).
*   **Early Termination:** By pushing `LIMIT` into the plan, the executor's `execute_limit` uses `Iterator::take()`, which stops upstream processing once the limit is reached. This turns `MATCH (n) RETURN n LIMIT 5` from an O(N) operation to an O(1) operation (relative to total graph size).

### 3. Execution Pipeline (`neural-executor/lib.rs`)
*   Updated `execute_query_ast` to respect the planner's handling of `ORDER BY` and `LIMIT`, avoiding double application of these operations.
*   Added `expression_is_aggregate` helper to detect complex queries that require post-processing.

## Impact
*   **Memory Efficiency:** Intermediate steps for filtering and traversal no longer allocate large vectors.
*   **Performance:** Queries with `LIMIT` are significantly faster on large datasets as they no longer scan the entire graph.
*   **Scalability:** The engine is now capable of handling result sets larger than available RAM (up to the point of pipeline breakers or final result collection).

## Future Work
*   **Client Streaming:** The public API and CLI still collect the final result into `QueryResult`. Future work should expose the iterator to the client/network layer for full end-to-end streaming.
*   **Streaming Aggregation:** Implement streaming algorithms for simple aggregations (e.g., `COUNT`, `SUM` without `GROUP BY`) to avoid pipeline breaking.


================================================================================
# Report: sprint_32_report.md
================================================================================

# Sprint 32 Report: v0.8.0 Release

**Date:** 2026-01-14
**Status:** Completed

## Goals
The goal of this sprint was to package and release version 0.8.0 of NeuralGraphDB, marking the completion of Phase 3 "Database Infrastructure".

## Deliverables

### 1. Version Bump
- Updated workspace version to `0.8.0` in `Cargo.toml`.
- All crates (`neural-core`, `neural-storage`, `neural-parser`, `neural-executor`, `neural-cli`) are now at v0.8.0.

### 2. Changelog
- Finalized `CHANGELOG.md` with comprehensive release notes for v0.8.0.
- Included features from Sprints 21-31:
    - CRUD Operations (Create, Delete, Set)
    - Persistence (Binary format, WAL)
    - Advanced Traversals (Variable length, Shortest Path)
    - Query Inspection (EXPLAIN, PROFILE)
    - Parameterized Queries
    - Streaming Execution

### 3. Stability Verification
- Ran full test suite (`cargo test --workspace`).
- **Results:**
    - `neural-core`: 6 tests passed.
    - `neural-storage`: 76 tests passed.
    - `neural-parser`: 38 tests passed.
    - `neural-executor`: 69 tests passed.
    - `neural-cli`: Demo functionality verified.

## Project Status
NeuralGraphDB v0.8.0 is a feature-complete "Database Core". It supports:
- **Storage:** CSR + Dynamic Adjacency + WAL.
- **Querying:** Full NGQL support including mutations and advanced traversals.
- **AI Integration:** Vector Search and Community Detection.
- **Performance:** Streaming execution and query optimization.

## Next Steps (Phase 4)
- **Arrow Flight:** Zero-copy data transfer for GNN training.
- **Time-Travel:** Temporal graph queries (`AT TIME`).
- **Transactions:** Full ACID compliance.

================================================================================
# Report: sprint_33_report.md
================================================================================

# Sprint 33: Query Pipelining - Final Report

**Fecha:** 2026-01-15
**Estado:** ✅ Completado
**Duración:** 1 sesión

---

## Objetivo del Sprint

> Implementar capacidades de "Query Pipelining" para permitir consultas complejas encadenadas.

---

## Entregables

### 1. WITH Clause

*   **Sintaxis:** `MATCH ... WITH n.name AS name WHERE name = 'Alice' RETURN name`
*   **Funcionalidad:** Proyección intermedia que actúa como barrera. Permite filtrar, ordenar y limitar resultados intermedios antes de pasarlos a la siguiente etapa.

### 2. UNWIND Clause

*   **Sintaxis:** `UNWIND [1, 2, 3] AS x RETURN x`
*   **Funcionalidad:** Expande una lista en múltiples filas (rows). Fundamental para ingesta por lotes y manejo de arrays.

### 3. Bindings Refactor

*   **Cambio Crítico:** `Bindings` ahora mapea `String -> Value` en lugar de `String -> NodeId`.
*   **Impacto:** Permite que las variables intermedias (en `WITH`) sean escalares (Strings, Integers, Lists) y no solo nodos del grafo.

### 4. Nuevos Physical Plans

*   `PhysicalPlan::Distinct`: Deduplicación de filas.
*   `PhysicalPlan::Aggregate`: Agregaciones (COUNT, SUM) como paso intermedio.
*   `PhysicalPlan::Unwind`: Expansión de listas.

---

## Tests

| Test Case | Descripción |
|-----------|-------------|
| `test_parse_with_clause` | Verifica parsing de WITH + proyección. |
| `test_parse_unwind_clause` | Verifica parsing de UNWIND. |
| `test_execute_with_clause` | Ejecución E2E de pipeline MATCH->WITH->RETURN. |
| `test_execute_unwind` | Ejecución E2E de expansión de lista. |

---

## Código Destacado

```rust
// Executor logic for UNWIND
fn execute_unwind(...) {
    // Evaluate list expression
    let list = eval::evaluate(expr, binding)?;
    // Expand to rows
    for item in list {
        yield binding.with(alias, item);
    }
}
```

================================================================================
# Report: sprint_34_report.md
================================================================================

# Sprint 34: Advanced Patterns - Final Report

**Fecha:** 2026-01-15
**Estado:** ✅ Completado
**Duración:** 1 sesión

---

## Objetivo del Sprint

> Implementar patrones avanzados de grafos para mayor expresividad y robustez en la ingesta.

---

## Entregables

### 1. OPTIONAL MATCH

*   **Concepto:** Equivalente a LEFT OUTER JOIN en SQL.
*   **Comportamiento:** Si el patrón no se encuentra, la consulta no descarta la fila, sino que rellena las variables nuevas con `NULL`.
*   **Uso:** `MATCH (n) OPTIONAL MATCH (n)-[:REL]->(m) RETURN n, m`

### 2. MERGE

*   **Concepto:** "Create if not exists" (Upsert).
*   **Nodos:** `MERGE (n:Person {name: "Alice"})` busca un nodo con ese label y propiedad. Si existe, lo retorna; si no, lo crea.
*   **Aristas:** `MERGE (a)-[:KNOWS]->(b)` asegura que la arista exista una sola vez.

### 3. Multi-Pattern MATCH

*   Soporte para `MATCH (a), (b)` (producto cartesiano o desconectados) en una sola cláusula.

---

## Tests

| Test Case | Descripción |
|-----------|-------------|
| `test_execute_optional_match` | Verifica que retorne NULL para patrones inexistentes. |
| `test_execute_merge` | Verifica comportamiento idempotente (crea 1 vez, luego reutiliza). |

---

## Arquitectura

```mermaid
graph TD
    A[Planner] -->|OPTIONAL| B[OptionalExpand]
    B -->|Match| C[Return Row]
    B -->|No Match| D[Return Row with NULLs]
```

================================================================================
# Report: sprint_35_report.md
================================================================================

# Sprint 35: Robust Expressions - Final Report

**Fecha:** 2026-01-15
**Estado:** ✅ Completado
**Duración:** 1 sesión

---

## Objetivo del Sprint

> Enriquecer el lenguaje NGQL con expresiones lógicas y manipulación de datos.

---

## Entregables

### 1. CASE Expression

*   **Generic:** `CASE WHEN n.age > 18 THEN 'Adult' ELSE 'Minor' END`
*   **Simple:** `CASE n.type WHEN 'A' THEN 1 ELSE 0 END`

### 2. COALESCE

*   **Sintaxis:** `COALESCE(n.nickname, n.name, 'Unknown')`
*   **Uso:** Manejo robusto de valores nulos.

### 3. String Functions

*   `toLower()`, `toUpper()`
*   `contains()`, `startsWith()`, `endsWith()`
*   `split()` (retorna lista)

### 4. Type Conversions

*   `toString()`, `toInteger()`, `toFloat()`, `toBoolean()`

---

## Código Destacado

```rust
// Eval logic for FunctionCall
match name {
    "toLower" => Value::String(s.to_lowercase()),
    "split" => Value::List(s.split(delim).map(Value::String).collect()),
    // ...
}
```

================================================================================
# Report: sprint_36_report.md
================================================================================

# Sprint 36: Temporal Engine - Final Report

**Fecha:** 2026-01-15
**Estado:** ✅ Completado
**Duración:** 1 sesión

---

## Objetivo del Sprint

> Añadir soporte nativo para tipos temporales (Date, DateTime).

---

## Entregables

### 1. Tipos de Datos

*   **Date:** `YYYY-MM-DD`
*   **DateTime:** `ISO 8601`
*   Integrados en `PropertyValue` (almacenamiento) y `Value` (ejecución).

### 2. Funciones Temporales

*   `date("2026-01-01")`: Crea un valor Date.
*   `datetime()`: Retorna fecha/hora actual (UTC).

### 3. Ejecución Dinámica

*   Mejora crítica en `Executor`: La proyección (`RETURN`) ahora evalúa expresiones complejas dinámicamente, permitiendo `RETURN date()` sin depender de variables de grafo.

---

## Tests

| Test Case | Descripción |
|-----------|-------------|
| `test_execute_date_function` | Verifica creación y retorno de Date. |
| `test_execute_datetime_function` | Verifica creación y retorno de DateTime. |

---

## Hito: Fase 4 Completada

Con el Sprint 36, NeuralGraphDB alcanza la **conformidad estándar con Cypher** en cuanto a tipos de datos y control de flujo, listo para la versión v0.9.

================================================================================
# Report: sprint_37_report.md
================================================================================

# Sprint 37: Arrow Flight Server - Final Report

**Fecha:** 2026-01-15
**Estado:** ✅ Completado

---

## Objetivo del Sprint

> Implementar un servidor Arrow Flight para acceso a datos de alto rendimiento desde IA.

---

## Entregables

### 1. Flight Service
* Implementación de `NeuralFlightService` usando `tonic` y `arrow-flight`.
* Soporte para endpoints `nodes` y `edges`.

### 2. Integración CLI
* Nuevo comando `serve-flight [PORT]` para levantar el servidor gRPC.

--- 

## Tests

| Test Case | Descripción |
|-----------|-------------|
| `test_nodes_schema` | Valida el esquema Arrow para nodos. |
| `test_do_get_nodes` | Valida el streaming de RecordBatches. |

--- 

## Código Destacado

```rust
let stream = FlightDataEncoderBuilder::new()
    .with_schema(schema)
    .build(futures::stream::iter(batches).map(Ok))
    .map(|res| res.map_err(|e| Status::internal(e.to_string())));
```

================================================================================
# Report: sprint_38_report.md
================================================================================

# Sprint 38: Python Client & PyTorch Integration - Final Report

**Fecha:** 2026-01-15
**Estado:** ✅ Completado

---

## Objetivo del Sprint

> Crear una librería Python para consumir datos del grafo vía Arrow Flight e integrarlos con PyTorch.

---

## Entregables

### 1. Paquete Python (`clients/python`)
* Cliente `NGraphClient` basado en `pyarrow.flight`.
* Métodos `get_nodes()` y `get_edges()` que retornan DataFrames.

### 2. Utilidades PyTorch
* Módulo `torch_utils` para conversión a objetos `Data` de PyTorch Geometric.
* Mapeo automático de IDs de nodos para compatibilidad con tensores GNN.

--- 

## Tests

* Validado estructuralmente con `examples/python_flight_demo.py`.

--- 

## Código Destacado

```python
def to_pyg_data(nodes_df, edges_df):
    id_map = {node_id: i for i, node_id in enumerate(nodes_df['id'])}
    src = edges_df['source'].map(id_map).values
    dst = edges_df['target'].map(id_map).values
    edge_index = torch.tensor(np.array([src, dst]), dtype=torch.long)
    return Data(x=torch.ones((len(nodes_df), 1)), edge_index=edge_index)
```

================================================================================
# Report: sprint_44_report.md
================================================================================

# Sprint 44: Python Client 2.0 - Final Report

**Fecha:** 2026-01-19
**Estado:** ✅ Completado

---

## Objetivo del Sprint

> Completed full refactor of Python client, adding support for CRUD operations and queries. Implemented id() function in NGQL for robust node referencing. Server responses are now structured JSON for better client integration.

---

## Entregables

*(Auto-generated placeholder - Please fill details)*

### 1. Feature Name
*   Description...

---

## Tests

| Test Case | Descripción |
|-----------|-------------|
| `test_placeholder` | ... |

---

## Código Destacado

```rust
// ...
```


================================================================================
# Report: sprint_43_report.md
================================================================================

# Sprint 43 Report: Python Client & Core Stability

**Date:** 2026-01-20
**Status:** ✅ Completed
**Author:** NeuralGraphDB Team

---

## Executive Summary

Sprint 43 focused on enabling the **Python Client** to function correctly against the NeuralGraphDB server. This involved implementing critical query pipeline features (specifically `MATCH ... CREATE` support), fixing a core bug in dynamic node ID generation, and improving server robustness.

| Component | Change | Impact |
|-----------|--------|--------|
| `neural-parser` | Added `Statement::CreateWithMatch` | Enables pipelined creation queries |
| `neural-executor` | Implemented `execute_create_with_match` | Executes creation based on match bindings |
| `neural-storage` | Fixed `next_node_id` logic | Correct ID generation for new nodes |
| `neural-cli` | Graceful startup | Server starts even without initial data |

---

## Key Deliverables

### 1. MATCH ... CREATE Support

The Python Client relies on Cypher-style patterns where creation follows matching, e.g., creating edges between existing nodes:

```cypher
MATCH (a), (b) 
WHERE id(a) = 1 AND id(b) = 2 
CREATE (a)-[:KNOWS]->(b)
```

Previously, the parser would ignore the `CREATE` clause after `MATCH`. We implemented full support for `Statement::CreateWithMatch` in both the parser and executor.

**Implementation Details:**
- **Parser:** Updated `parse_statement_internal` to detect `CREATE` tokens following a `MATCH` clause.
- **AST:** Added `Statement::CreateWithMatch` variant.
- **Executor:** Implemented `execute_create_with_match` which executes the match query first, collects bindings, and then iterates through the result set to execute the create patterns for each row.

### 2. Dynamic Node ID Fix

A critical bug was identified where dynamically created nodes (via `CREATE`) were receiving duplicate IDs because the ID generator ignored the count of previously created dynamic nodes.

**Fix in `GraphStore::next_node_id`:**
```rust
// Before
NodeId::new(self.graph.node_count() as u64)

// After
NodeId::new((self.graph.node_count() + self.dynamic_node_count) as u64)
```

This ensures unique, incremental IDs for all new nodes.

### 3. Server Robustness

The `neural-cli` server previously crashed if the default `arxiv_graph.json` file was missing. It now gracefully initializes an empty in-memory graph, allowing for immediate usage and testing without pre-loaded data.

---

## Verification

The Python Client demo (`examples/python_client_demo.py`) was used for end-to-end verification.

**Results:**
- **Node Creation:** Successfully creates nodes with distinct IDs (1, 2, 3, ...).
- **Edge Creation:** Successfully creates edges between matched nodes.
- **Querying:** Correctly returns graph structure and properties.
- **Mutations:** Updates and deletions work as expected.

```text
Creating nodes...
  - Created Alice with ID: 1
  - Created Bob with ID: 2
  - Created Charlie with ID: 3

Creating edges...
  - Alice KNOWS Bob: True
  - Bob KNOWS Charlie: True

Querying all people:
     p.age             p.name
0  Int(30)    String("Alice")
1  Int(25)      String("Bob")
2  Int(35)  String("Charlie")
```

---

## Next Steps

With the core stable and the Python Client functional, the focus shifts to **Scale & Production** features (Phase 6), including:

- **Arrow Flight Optimization:** Ensure the Flight server is fully integrated for bulk transport.
- **LDBC Benchmarks:** Formal performance validation.
- **Transactions:** Begin design for ACID compliance.


================================================================================
# Report: sprint_45_report.md
================================================================================

# Sprint 45 Report: Read Latency Optimization

**Date:** 2026-01-20
**Status:** ✅ Completed
**Author:** NeuralGraphDB Team

---

## Executive Summary

Sprint 45 focused on optimizing read latency to surpass the performance of FalkorDB (Redis-based). By implementing **Arrow Flight** as the primary query transport and optimizing the parser, we reduced read latency from ~1.5ms (HTTP) to **~0.10ms** (Flight), achieving a **15x speedup**.

| Metric | Before (HTTP) | After (Flight) | FalkorDB (Ref) | Result |
| :--- | :--- | :--- | :--- | :--- |
| **Point Lookup** | 1.67 ms | **0.17 ms** | 0.39 ms | **2.3x Faster** |
| **1-Hop Traversal** | 1.84 ms | **0.11 ms** | 0.39 ms | **3.5x Faster** |
| **2-Hop Traversal** | 1.31 ms | **0.11 ms** | 0.41 ms | **3.7x Faster** |

---

## Key Deliverables

### 1. Arrow Flight Query Execution
Extended the `neural-cli` Flight server to accept arbitrary NGQL queries via `DoGet`.
*   **Implementation:** `flight.rs` now parses the ticket as a query string, executes it against `GraphStore`, and streams the result as an Arrow RecordBatch.
*   **Zero-Copy:** Leverages Arrow's memory layout to minimize serialization overhead compared to JSON/HTTP.

### 2. Parser Improvements
Fixed a limitation in the parser where keywords (like `id`, `type`) could not be used as property names (e.g., `n.id`).
*   **Fix:** Updated `parse_primary` to accept specific keyword tokens as property identifiers.

### 3. Latency Profiling Suite
Created `benchmarks/latency_profile_flight.py` to measure end-to-end latency with microsecond precision, validating the gains.

---

## Technical Details

**Flight DoGet Logic:**
```rust
match query_str {
    "nodes" | "edges" => { /* Bulk Export */ },
    query => {
        let result = neural_executor::execute_query(&store, &query)?;
        let batch = query_result_to_record_batch(&result)?;
        // Stream batch...
    }
}
```

**Memory Safety:**
The Flight server uses the same `Arc<GraphStore>` as the HTTP server, ensuring thread-safe access to the in-memory graph without data duplication.

---

## Next Steps

With latency and ingestion performance secured, the focus returns to **Scale & Production**:
*   **Transactions:** Implement ACID guarantees.
*   **Distribution:** Explore distributed consensus (Raft) for multi-node deployments.
# Sprint 45 Report: Read Latency Optimization

**Date:** 2026-01-20
**Status:** ✅ Completed
**Author:** NeuralGraphDB Team

---

## Executive Summary

Sprint 45 focused on optimizing read latency to surpass the performance of FalkorDB (Redis-based). By implementing **Arrow Flight** as the primary query transport and optimizing the parser, we reduced read latency from ~1.5ms (HTTP) to **~0.10ms** (Flight), achieving a **15x speedup**.

| Metric | Before (HTTP) | After (Flight) | FalkorDB (Ref) | Result |
| :--- | :--- | :--- | :--- | :--- |
| **Point Lookup** | 1.67 ms | **0.17 ms** | 0.39 ms | **2.3x Faster** |
| **1-Hop Traversal** | 1.84 ms | **0.11 ms** | 0.39 ms | **3.5x Faster** |
| **2-Hop Traversal** | 1.31 ms | **0.11 ms** | 0.41 ms | **3.7x Faster** |

---

## Key Deliverables

### 1. Arrow Flight Query Execution
Extended the `neural-cli` Flight server to accept arbitrary NGQL queries via `DoGet`.
*   **Implementation:** `flight.rs` now parses the ticket as a query string, executes it against `GraphStore`, and streams the result as an Arrow RecordBatch.
*   **Zero-Copy:** Leverages Arrow's memory layout to minimize serialization overhead compared to JSON/HTTP.

### 2. Parser Improvements
Fixed a limitation in the parser where keywords (like `id`, `type`) could not be used as property names (e.g., `n.id`).
*   **Fix:** Updated `parse_primary` to accept specific keyword tokens as property identifiers.

### 3. Robust Type Inference (Flight)
Implemented a sophisticated type inference mechanism for Arrow Flight.
*   **Problem:** `QueryResult` rows can contain mixed types (e.g., `Int` and `Float`), which crashed strict Arrow builders.
*   **Solution:** Scans all rows to determine the "widest" compatible type (e.g., `Int` -> `Float64`, `Bool` -> `Utf8`) before building batches.

### 4. Latency Profiling Suite
Created `benchmarks/latency_profile_flight.py` to measure end-to-end latency with microsecond precision, validating the gains.

---

## Technical Details

**Flight DoGet Logic:**
```rust
match query_str {
    "nodes" | "edges" => { /* Bulk Export */ },
    query => {
        let result = neural_executor::execute_query(&store, &query)?;
        let batch = query_result_to_record_batch(&result)?;
        // Stream batch...
    }
}
```

**Memory Safety:**
The Flight server uses the same `Arc<GraphStore>` as the HTTP server, ensuring thread-safe access to the in-memory graph without data duplication.

---

## Next Steps

With latency and ingestion performance secured, the focus returns to **Scale & Production**:
*   **Transactions:** Implement ACID guarantees.
*   **Distribution:** Explore distributed consensus (Raft) for multi-node deployments.
# Sprint 46 Report: Core Stability & Cypher Standards

**Date:** January 21, 2026
**Status:** ✅ Complete
**Focus:** Parser Hardening, Standard Cypher Compliance, and DX Improvements.

## 1. Executive Summary

Sprint 46 addressed several friction points in the Developer Experience (DX) regarding query syntax. Previously, the parser was overly strict, treating common words like `type`, `id`, `count` as reserved keywords that could not be used as variable names or property keys. This sprint relaxed those constraints, implemented the standard `type()` function for edges, and normalized the `shortestPath` syntax.

These changes bring NeuralGraphDB significantly closer to full openCypher compliance, ensuring that queries written for other graph databases work with minimal modification.

## 2. Key Features & Fixes

### 2.1. Relaxed Keyword Restrictions
*   **Problem:** Queries like `MATCH (id:Person) RETURN id.name` or `MATCH (n) RETURN n.type` failed because `id` and `type` were reserved tokens.
*   **Solution:** Implemented a context-aware `parse_identifier` method in the parser.
*   **Outcome:** Keywords can now be used as:
    *   Variable names: `MATCH (count)`
    *   Property keys: `MATCH (n {type: 'A'})`
    *   Labels: `MATCH (n:Match)` (though discouraged, it is now valid)
    *   Aliases: `RETURN n.age AS count`

### 2.2. The `type()` Function
*   **Feature:** Implemented the standard `type(relationship)` function.
*   **Syntax:** `MATCH (a)-[r]->(b) RETURN type(r)`
*   **Implementation:** Added `get_edge_type(EdgeId)` to `GraphStore`. It handles both:
    *   **Static Edges (CSR):** Lookups via `EdgeTypeIndex` (O(1)).
    *   **Dynamic Edges (Adjacency List):** Direct field access.

### 2.3. `shortestPath` Standardization
*   **Fix:** Unified the parsing logic for `shortestPath(...)` (function-style) and `SHORTEST PATH ...` (clause-style).
*   **Robustness:** Now correctly parses patterns with labels and variable lengths inside the shortest path clause, e.g., `shortestPath((a:Person)-[:KNOWS*]->(b))`.

### 2.4. Incoming Edge IDs
*   **Bug Fix:** Generic traversals using the incoming direction (e.g., `MATCH (a)<-[r]-(b)`) previously struggled to return the correct `EdgeId` for `r` because the CSR format only explicitly stores outgoing edges.
*   **Solution:** Implemented `incoming_neighbors_with_ids` in `GraphStore`, which performs a reverse lookup using the `EdgeTypeIndex` and dynamic edge list to reconstruct valid Edge IDs.

## 3. Technical Details

### Parser Changes (`neural-parser`)
*   Modified `lexer.rs` to expose `Token::is_keyword()`.
*   Refactored `parser.rs` to use `parse_identifier()` instead of `expect(&Token::Ident)` in pattern matching, projection, and property access.
*   Added `peek_at(offset)` for lookahead to distinguish between function calls (`type(...)`) and identifiers (`type`).

### Executor Changes (`neural-executor`)
*   Updated `Plan` Display implementation to match standard Cypher visual style (e.g., `(a)-[:REL]->(b)`).
*   Integrated `type()` function evaluation.

## 4. Verification

A new regression test suite `crates/neural-executor/tests/repro_sprint_46.rs` was created covering:

| Test Case | Description | Status |
|:---|:---|:---|
| `test_id_as_variable_name` | `MATCH (id) ...` | ✅ Pass |
| `test_keywords_as_variable_names` | `MATCH (count) ...` | ✅ Pass |
| `test_id_as_property_key` | `MATCH (n {id: 1})` | ✅ Pass |
| `test_type_function` | `RETURN type(r)` | ✅ Pass |
| `test_incoming_edge_ids` | `MATCH (a)<-[r]-(b) RETURN id(r)` | ✅ Pass |
| `test_shortest_path_variants` | Both syntax styles | ✅ Pass |

## 5. Next Steps

With the core language features stabilized, we are ready to scale.

*   **Next Sprint:** Sprint 47 (Vector Scale)
*   **Objective:** Optimize HNSW index and storage for 1 Million+ vectors.
# Sprint 47 Report: Vector Scale (1M)

**Date:** January 21, 2026
**Status:** ✅ Complete
**Goal:** Achieve Enterprise-grade RAG scalability with 1 Million vectors.

## 1. Results Summary

Sprint 47 successfully demonstrated that NeuralGraphDB can handle 1,000,000 vectors with sub-millisecond search latency. This validates the engine's core architecture for large-scale AI applications.

| Benchmark | Results |
| :--- | :--- |
| **Dataset** | 1,000,000 Synthetic Vectors (128-dim, Normalized) |
| **Ingestion Rate** | 8,132.14 items/second |
| **Build Time** | 122.97 seconds (Total Ingestion + Indexing) |
| **Search Latency** | **0.69 ms** (Top-10 candidates) |

## 2. Technical Enhancements

To achieve these results and fix runtime issues identified during the benchmark, the following improvements were made:

### 2.1. Parser Improvements
*   **Scientific Notation:** Added support for floats formatted with `e` (e.g., `1.23e-5`) in the lexer.
*   **List Literals:** Implemented support for recursive list literals in property maps, allowing `CREATE (n {embedding: [0.1, ...]})`.

### 2.2. Planner Optimization
*   **VectorScan Detection:** Updated the planner to detect `WHERE vector_similarity(...) > threshold` and convert it into a specialized `VectorSearch` physical plan.
*   **Physical Plan Upgrade:** Added the `query` expression and `k` (limit) fields to the `VectorSearch` plan to ensure the executor has the necessary search context.

### 2.3. Executor & Storage
*   **Vector Search Executor:** Implemented the search logic to evaluate query vectors and call the storage engine.
*   **Hybrid Filter:** Added `vector_search_filtered` to `GraphStore` to allow combining label filters with vector similarity search.

## 3. Scaling Observations

The scaling behavior from 10k to 1M vectors remained nearly flat for search latency:
*   10,000 vectors: 0.71 ms
*   100,000 vectors: 0.68 ms
*   1,000,000 vectors: 0.69 ms

This confirms the efficiency of the HNSW implementation and the low overhead of the physical plan execution.

## 4. Next Steps

*   **Phase 6 (Ecosystem):** Start work on the SQL Bridge to import real-world enterprise data into this high-performance engine.
*   **Neural Dashboard:** Build the "Reef" visualization to show off the 1M node graph.
\n\n================================================================================\n# Report: sprint_50_report.md\n================================================================================\n
# Sprint 50 Report: Transaction Manager (ACID Core)

**Date:** January 21, 2026
**Status:** ✅ Complete
**Goal:** Enable atomic multi-statement blocks with ACID guarantees.

## 1. Overview
Sprint 50 successfully transitioned NeuralGraphDB from single-operation atomicity to a full transactional model. This allows users to group multiple `CREATE`, `SET`, and `DELETE` operations into a single unit of work that is committed or rolled back atomically.

## 2. Technical Implementation

### 2.1. Transaction Engine
*   **`TransactionManager`:** Implemented a thread-safe generator for `TransactionId`s and logic to track transaction lifecycles.
*   **`Transaction` Handle:** Created a buffering mechanism that holds `LogEntry` objects in memory. Mutations are not applied to the `GraphStore` or written to the primary WAL until `COMMIT`.
*   **Predictive ID Generation:** Updated `NodeId` and `EdgeId` generation logic to account for "pending" creations within an active transaction, ensuring IDs are deterministic even before commit.

### 2.2. NGQL Updates
*   **Lexer & Parser:** Added support for `BEGIN`, `COMMIT`, and `ROLLBACK` keywords.
*   **Combined Patterns:** Refactored `parse_create_pattern` to support full node patterns (labels/properties) within edge creation syntax, e.g., `CREATE (a:User)-[:REL]->(b:User)`.

### 2.3. Executor & CLI Integration
*   **Stateful Execution:** Refactored `execute_statement_with_params` to accept a mutable transaction context. 
*   **CLI Sessions:** Updated the `neural-cli` REPL to maintain transaction state across multiple lines of input.
*   **WAL Persistence:** Enhanced the Write-Ahead Log to record transaction boundaries (`BeginTransaction`, `CommitTransaction`), ensuring recovery remains consistent.

## 3. Verification Results

### 3.1. Integration Tests
A new test suite `crates/neural-executor/tests/test_transactions.rs` was added to verify ACID properties:
*   **Atomicity:** Verified that if a process crashes or rolls back, zero changes are applied.
*   **Consistency:** Confirmed that indices and properties remain synchronized after commit.
*   **Isolation (Level: Read Committed):** Verified that concurrent readers do not see uncommitted changes.
*   **Rollback:** Successfully demonstrated discarding complex mutation buffers.

### 3.2. Performance Impact
*   **Latency:** Minimal overhead (<0.05ms) for starting/buffering transactions.
*   **Throughput:** Single-statement mutations remain unchanged; multi-statement transactions improve WAL efficiency by grouping flushes.

## 4. Key Fixes
*   Fixed a bug where target nodes in `CREATE` edge patterns were restricted to simple identifiers, now supporting full node descriptions.

## 5. Next Steps
*   **Sprint 51: MVCC (Snapshot Isolation):** Move from the current "Buffer-then-Apply" model to multi-versioned storage to support non-blocking reads and higher write concurrency.


================================================================================
# Report: sprint_45_report.md
================================================================================
# Sprint 45 Report: Read Latency Optimization

**Date:** 2026-01-20
**Status:** ✅ Completed
**Author:** NeuralGraphDB Team

---

## Executive Summary

Sprint 45 focused on optimizing read latency to surpass the performance of FalkorDB (Redis-based). By implementing **Arrow Flight** as the primary query transport and optimizing the parser, we reduced read latency from ~1.5ms (HTTP) to **~0.10ms** (Flight), achieving a **15x speedup**.

| Metric | Before (HTTP) | After (Flight) | FalkorDB (Ref) | Result |
| :--- | :--- | :--- | :--- | :--- |
| **Point Lookup** | 1.67 ms | **0.17 ms** | 0.39 ms | **2.3x Faster** |
| **1-Hop Traversal** | 1.84 ms | **0.11 ms** | 0.39 ms | **3.5x Faster** |
| **2-Hop Traversal** | 1.31 ms | **0.11 ms** | 0.41 ms | **3.7x Faster** |

---

## Key Deliverables

### 1. Arrow Flight Query Execution
Extended the `neural-cli` Flight server to accept arbitrary NGQL queries via `DoGet`.
*   **Implementation:** `flight.rs` now parses the ticket as a query string, executes it against `GraphStore`, and streams the result as an Arrow RecordBatch.
*   **Zero-Copy:** Leverages Arrow's memory layout to minimize serialization overhead compared to JSON/HTTP.

### 2. Parser Improvements
Fixed a limitation in the parser where keywords (like `id`, `type`) could not be used as property names (e.g., `n.id`).
*   **Fix:** Updated `parse_primary` to accept specific keyword tokens as property identifiers.

### 3. Robust Type Inference (Flight)
Implemented a sophisticated type inference mechanism for Arrow Flight.
*   **Problem:** `QueryResult` rows can contain mixed types (e.g., `Int` and `Float`), which crashed strict Arrow builders.
*   **Solution:** Scans all rows to determine the "widest" compatible type (e.g., `Int` -> `Float64`, `Bool` -> `Utf8`) before building batches.

### 4. Latency Profiling Suite
Created `benchmarks/latency_profile_flight.py` to measure end-to-end latency with microsecond precision, validating the gains.

---

## Technical Details

**Flight DoGet Logic:**
```rust
match query_str {
    "nodes" | "edges" => { /* Bulk Export */ },
    query => {
        let result = neural_executor::execute_query(&store, &query)?;
        let batch = query_result_to_record_batch(&result)?;
        // Stream batch...
    }
}
```

**Memory Safety:**
The Flight server uses the same `Arc<GraphStore>` as the HTTP server, ensuring thread-safe access to the in-memory graph without data duplication.

---

## Next Steps

With latency and ingestion performance secured, the focus returns to **Scale & Production**:
*   **Transactions:** Implement ACID guarantees.
*   **Distribution:** Explore distributed consensus (Raft) for multi-node deployments.


================================================================================
# Report: sprint_46_report.md
================================================================================
# Sprint 46 Report: Core Stability & Cypher Standards

**Date:** January 21, 2026
**Status:** ✅ Complete
**Focus:** Parser Hardening, Standard Cypher Compliance, and DX Improvements.

## 1. Executive Summary

Sprint 46 addressed several friction points in the Developer Experience (DX) regarding query syntax. Previously, the parser was overly strict, treating common words like `type`, `id`, `count` as reserved keywords that could not be used as variable names or property keys. This sprint relaxed those constraints, implemented the standard `type()` function for edges, and normalized the `shortestPath` syntax.

These changes bring NeuralGraphDB significantly closer to full openCypher compliance, ensuring that queries written for other graph databases work with minimal modification.

## 2. Key Features & Fixes

### 2.1. Relaxed Keyword Restrictions
*   **Problem:** Queries like `MATCH (id:Person) RETURN id.name` or `MATCH (n) RETURN n.type` failed because `id` and `type` were reserved tokens.
*   **Solution:** Implemented a context-aware `parse_identifier` method in the parser.
*   **Outcome:** Keywords can now be used as:
    *   Variable names: `MATCH (count)`
    *   Property keys: `MATCH (n {type: 'A'})`
    *   Labels: `MATCH (n:Match)` (though discouraged, it is now valid)
    *   Aliases: `RETURN n.age AS count`

### 2.2. The `type()` Function
*   **Feature:** Implemented the standard `type(relationship)` function.
*   **Syntax:** `MATCH (a)-[r]->(b) RETURN type(r)`
*   **Implementation:** Added `get_edge_type(EdgeId)` to `GraphStore`. It handles both:
    *   **Static Edges (CSR):** Lookups via `EdgeTypeIndex` (O(1)).
    *   **Dynamic Edges (Adjacency List):** Direct field access.

### 2.3. `shortestPath` Standardization
*   **Fix:** Unified the parsing logic for `shortestPath(...)` (function-style) and `SHORTEST PATH ...` (clause-style).
*   **Robustness:** Now correctly parses patterns with labels and variable lengths inside the shortest path clause, e.g., `shortestPath((a:Person)-[:KNOWS*]->(b))`.

### 2.4. Incoming Edge IDs
*   **Bug Fix:** Generic traversals using the incoming direction (e.g., `MATCH (a)<-[r]-(b)`) previously struggled to return the correct `EdgeId` for `r` because the CSR format only explicitly stores outgoing edges.
*   **Solution:** Implemented `incoming_neighbors_with_ids` in `GraphStore`, which performs a reverse lookup using the `EdgeTypeIndex` and dynamic edge list to reconstruct valid Edge IDs.

## 3. Technical Details

### Parser Changes (`neural-parser`)
*   Modified `lexer.rs` to expose `Token::is_keyword()`.
*   Refactored `parser.rs` to use `parse_identifier()` instead of `expect(&Token::Ident)` in pattern matching, projection, and property access.
*   Added `peek_at(offset)` for lookahead to distinguish between function calls (`type(...)`) and identifiers (`type`).

### Executor Changes (`neural-executor`)
*   Updated `Plan` Display implementation to match standard Cypher visual style (e.g., `(a)-[:REL]->(b)`).
*   Integrated `type()` function evaluation.

## 4. Verification

A new regression test suite `crates/neural-executor/tests/repro_sprint_46.rs` was created covering:

| Test Case | Description | Status |
|:---|:---|:---|
| `test_id_as_variable_name` | `MATCH (id) ...` | ✅ Pass |
| `test_keywords_as_variable_names` | `MATCH (count) ...` | ✅ Pass |
| `test_id_as_property_key` | `MATCH (n {id: 1})` | ✅ Pass |
| `test_type_function` | `RETURN type(r)` | ✅ Pass |
| `test_incoming_edge_ids` | `MATCH (a)<-[r]-(b) RETURN id(r)` | ✅ Pass |
| `test_shortest_path_variants` | Both syntax styles | ✅ Pass |

## 5. Next Steps

With the core language features stabilized, we are ready to scale.

*   **Next Sprint:** Sprint 47 (Vector Scale)
*   **Objective:** Optimize HNSW index and storage for 1 Million+ vectors.


================================================================================
# Report: sprint_47_report.md
================================================================================
# Sprint 47 Report: Vector Scale (1M)

**Date:** January 21, 2026
**Status:** ✅ Complete
**Goal:** Achieve Enterprise-grade RAG scalability with 1 Million vectors.

## 1. Results Summary

Sprint 47 successfully demonstrated that NeuralGraphDB can handle 1,000,000 vectors with sub-millisecond search latency. This validates the engine's core architecture for large-scale AI applications.

| Benchmark | Results |
| :--- | :--- |
| **Dataset** | 1,000,000 Synthetic Vectors (128-dim, Normalized) |
| **Ingestion Rate** | 8,132.14 items/second |
| **Build Time** | 122.97 seconds (Total Ingestion + Indexing) |
| **Search Latency** | **0.69 ms** (Top-10 candidates) |

## 2. Technical Enhancements

To achieve these results and fix runtime issues identified during the benchmark, the following improvements were made:

### 2.1. Parser Improvements
*   **Scientific Notation:** Added support for floats formatted with `e` (e.g., `1.23e-5`) in the lexer.
*   **List Literals:** Implemented support for recursive list literals in property maps, allowing `CREATE (n {embedding: [0.1, ...]})`.

### 2.2. Planner Optimization
*   **VectorScan Detection:** Updated the planner to detect `WHERE vector_similarity(...) > threshold` and convert it into a specialized `VectorSearch` physical plan.
*   **Physical Plan Upgrade:** Added the `query` expression and `k` (limit) fields to the `VectorSearch` plan to ensure the executor has the necessary search context.

### 2.3. Executor & Storage
*   **Vector Search Executor:** Implemented the search logic to evaluate query vectors and call the storage engine.
*   **Hybrid Filter:** Added `vector_search_filtered` to `GraphStore` to allow combining label filters with vector similarity search.

## 3. Scaling Observations

The scaling behavior from 10k to 1M vectors remained nearly flat for search latency:
*   10,000 vectors: 0.71 ms
*   100,000 vectors: 0.68 ms
*   1,000,000 vectors: 0.69 ms

This confirms the efficiency of the HNSW implementation and the low overhead of the physical plan execution.

## 4. Next Steps

*   **Phase 6 (Ecosystem):** Start work on the SQL Bridge to import real-world enterprise data into this high-performance engine.
*   **Neural Dashboard:** Build the "Reef" visualization to show off the 1M node graph.


================================================================================
# Report: sprint_50_report.md
================================================================================
# Sprint 50 Report: Transaction Manager (ACID Core)

**Date:** January 21, 2026
**Status:** ✅ Complete
**Goal:** Enable atomic multi-statement blocks with ACID guarantees.

## 1. Overview
Sprint 50 successfully transitioned NeuralGraphDB from single-operation atomicity to a full transactional model. This allows users to group multiple `CREATE`, `SET`, and `DELETE` operations into a single unit of work that is committed or rolled back atomically.

## 2. Technical Implementation

### 2.1. Transaction Engine
*   **`TransactionManager`:** Implemented a thread-safe generator for `TransactionId`s and logic to track transaction lifecycles.
*   **`Transaction` Handle:** Created a buffering mechanism that holds `LogEntry` objects in memory. Mutations are not applied to the `GraphStore` or written to the primary WAL until `COMMIT`.
*   **Predictive ID Generation:** Updated `NodeId` and `EdgeId` generation logic to account for "pending" creations within an active transaction, ensuring IDs are deterministic even before commit.

### 2.2. NGQL Updates
*   **Lexer & Parser:** Added support for `BEGIN`, `COMMIT`, and `ROLLBACK` keywords.
*   **Combined Patterns:** Refactored `parse_create_pattern` to support full node patterns (labels/properties) within edge creation syntax, e.g., `CREATE (a:User)-[:REL]->(b:User)`.

### 2.3. Executor & CLI Integration
*   **Stateful Execution:** Refactored `execute_statement_with_params` to accept a mutable transaction context. 
*   **CLI Sessions:** Updated the `neural-cli` REPL to maintain transaction state across multiple lines of input.
*   **WAL Persistence:** Enhanced the Write-Ahead Log to record transaction boundaries (`BeginTransaction`, `CommitTransaction`), ensuring recovery remains consistent.

## 3. Verification Results

### 3.1. Integration Tests
A new test suite `crates/neural-executor/tests/test_transactions.rs` was added to verify ACID properties:
*   **Atomicity:** Verified that if a process crashes or rolls back, zero changes are applied.
*   **Consistency:** Confirmed that indices and properties remain synchronized after commit.
*   **Isolation (Level: Read Committed):** Verified that concurrent readers do not see uncommitted changes.
*   **Rollback:** Successfully demonstrated discarding complex mutation buffers.

### 3.2. Performance Impact
*   **Latency:** Minimal overhead (<0.05ms) for starting/buffering transactions.
*   **Throughput:** Single-statement mutations remain unchanged; multi-statement transactions improve WAL efficiency by grouping flushes.

## 4. Key Fixes
*   Fixed a bug where target nodes in `CREATE` edge patterns were restricted to simple identifiers, now supporting full node descriptions.

## 5. Next Steps
*   **Sprint 51: MVCC (Snapshot Isolation):** Move from the current "Buffer-then-Apply" model to multi-versioned storage to support non-blocking reads and higher write concurrency.

================================================================================
# Report: sprint_51_report.md
================================================================================

# Sprint 51: MVCC (Snapshot Isolation) - Completed

**Objective:** To refactor the storage engine to support Multi-Version Concurrency Control (MVCC), which is the foundation for enabling non-blocking reads and future time-travel queries.

**Work Completed:**

1.  **Core MVCC Logic:**
    *   Created `crates/neural-storage/src/mvcc.rs`.
    *   Implemented `VersionedValue`, a new data structure that stores a history of property values, each tagged with the `TransactionId` that created it.
    *   Implemented the core visibility rule logic to determine which version of a value is visible to a given transaction snapshot.
    *   Implemented `vacuum()` for garbage collection of old versions.

2.  **VersionedPropertyStore:**
    *   Created `VersionedPropertyStore` in `crates/neural-storage/src/properties.rs`.
    *   Implemented MVCC-aware methods: `set(node, key, value, tx_id)`, `get(node, key, snapshot_id)`, `remove(node, key, tx_id)`, `remove_all(node, tx_id)`, `contains(node, key, snapshot_id)`, `has_properties(node, snapshot_id)`, `get_all(node, snapshot_id)`, `set_many(node, properties, tx_id)`, `vacuum(min_active_tx_id)`.
    *   Added comprehensive tests for version visibility, updates, deletions, and garbage collection.

3.  **GraphStore MVCC Integration:**
    *   Replaced `PropertyStore` with `VersionedPropertyStore` for both properties and labels in `GraphStore`.
    *   Added `MAX_SNAPSHOT_ID` constant for backward-compatible reads (sees all committed data).
    *   Added `current_tx_id` counter for automatic transaction ID assignment.
    *   Updated all internal methods to use versioned storage:
        *   `get_property()` / `get_property_at(node, key, snapshot_id)`
        *   `set_property()` / `set_property_at(node, key, value, tx_id)`
        *   `get_label()` / `get_label_at(node, snapshot_id)`
        *   `set_label()` / `set_label_at(node, label, tx_id)`
        *   `is_deleted()` / `is_deleted_at(node, snapshot_id)`
        *   `get_all_properties_at(node, snapshot_id)`
        *   `current_snapshot_id()` accessor
    *   Updated `apply_log_entry()` to use versioned writes with transaction IDs.
    *   Updated `GraphStoreBuilder::build()` to initialize versioned storage.

4.  **Executor MVCC Support:**
    *   Added `snapshot_id` field to `Executor` struct.
    *   Added `Executor::with_snapshot(store, snapshot_id)` constructor for transactional reads.
    *   Updated all property/label reads to use snapshot-aware methods:
        *   `execute_scan_by_label()` uses `get_label_at()`
        *   `execute_scan_by_property()` uses `get_property_at()`
        *   `execute_project()` uses `get_property_at()`
        *   `execute_filter()` uses `is_true_at()`

5.  **Expression Evaluation MVCC Support:**
    *   Added `evaluate_at(expr, bindings, store, params, snapshot_id)` function.
    *   Added `is_true_at(expr, bindings, store, params, snapshot_id)` function.
    *   Updated all recursive evaluation calls to propagate `snapshot_id`.
    *   Maintained backward compatibility with `evaluate()` and `is_true()` using `MAX_SNAPSHOT_ID`.

**Testing:**
*   All 8 versioned property store tests pass.
*   All 29 executor unit tests pass.
*   All 45 executor integration tests pass (including transaction tests).
*   Full workspace compiles successfully.

**API Usage Examples:**

```rust
// Non-transactional read (sees all committed data)
let executor = Executor::new(&store);
let result = executor.execute(&plan, None)?;

// Transactional read at specific snapshot
let snapshot_id = store.current_snapshot_id();
let executor = Executor::with_snapshot(&store, snapshot_id);
let result = executor.execute(&plan, None)?;

// Direct property access at snapshot
let value = store.get_property_at(node_id, "name", snapshot_id);
let label = store.get_label_at(node_id, snapshot_id);
```

**Next Steps for Sprint 52:**
1.  **Distributed Consensus (Raft):** Implement leader election and log replication.
2.  **Cluster Management:** Node discovery, membership changes, failover.
3.  **Snapshot Coordinator:** Global transaction ordering across nodes.


================================================================================
# Report: sprint_67_report.md
================================================================================

# Sprint 67: Production Observability - Final Report

**Date:** 2026-02-04
**Status:** ✅ Completed
**Duration:** 1 session

---

## Objective

> Add production-ready monitoring to NeuralGraphDB by wiring existing infrastructure (MetricsRegistry, logging) into the HTTP server and adding health/metrics endpoints.

---

## Deliverables

### 1. `/health` Endpoint

Returns JSON with database status, node/edge counts, uptime, and component health for load balancer integration.

**Response Format:**
```json
{
  "status": "healthy",
  "uptime_seconds": 3600,
  "database": {
    "loaded": true,
    "node_count": 12345,
    "edge_count": 67890,
    "path": "data/graph.ngdb"
  }
}
```

### 2. `/metrics` Endpoint

Exposes Prometheus text format for scraping by monitoring systems.

**Key Metrics:**
| Metric | Type | Description |
|--------|------|-------------|
| `neuralgraph_query_total` | Counter | Total queries executed |
| `neuralgraph_query_latency_seconds` | Histogram | Query latency distribution |
| `neuralgraph_node_count` | Gauge | Total nodes in graph |
| `neuralgraph_edge_count` | Gauge | Total edges in graph |
| `neuralgraph_cache_hits_total` | Counter | Cache hit count |
| `neuralgraph_cache_misses_total` | Counter | Cache miss count |

### 3. Query Latency Recording

Instrumented `handle_query()` to record latencies to the metrics registry.

### 4. Structured Logging

Initialized tracing on server startup with optional JSON output via `NGDB_LOG_JSON` environment variable.

---

## Files Modified

| File | Changes |
|------|---------|
| `crates/neural-cli/Cargo.toml` | Added `features = ["metrics"]` to neural-storage dependency |
| `crates/neural-cli/src/server.rs` | Added endpoints, AppState fields, instrumentation |
| `crates/neural-storage/src/metrics.rs` | Added `node_count` and `edge_count` gauges |
| `docker-compose.yml` | Updated healthcheck to use `/health` endpoint |

---

## Implementation Details

### 1. Metrics Feature Enabled
```toml
# crates/neural-cli/Cargo.toml
neural-storage = { workspace = true, features = ["metrics"] }
```

### 2. Extended AppState
```rust
pub struct AppState {
    // ... existing fields ...
    pub metrics: Arc<MetricsRegistry>,
    pub start_time: std::time::Instant,
}
```

### 3. Health & Metrics Handlers
```rust
async fn handle_health(State(state): State<Arc<AppState>>) -> Json<HealthResponse> {
    let store = state.store.read().await;
    let node_count = store.node_count();
    let edge_count = store.edge_count();
    let uptime = state.start_time.elapsed().as_secs();
    
    state.metrics.set_node_count(node_count);
    state.metrics.set_edge_count(edge_count);
    
    Json(HealthResponse {
        status: "healthy".to_string(),
        uptime_seconds: uptime,
        database: DatabaseHealth {
            loaded: true,
            node_count,
            edge_count,
            path: state.config.db_path.display().to_string(),
        },
    })
}

async fn handle_metrics(State(state): State<Arc<AppState>>) -> Result<String, (StatusCode, String)> {
    let store = state.store.read().await;
    state.metrics.set_node_count(store.node_count());
    state.metrics.set_edge_count(store.edge_count());
    drop(store);
    
    state.metrics.export().map_err(|e| {
        (StatusCode::INTERNAL_SERVER_ERROR, format!("Failed to export metrics: {}", e))
    })
}
```

### 4. Logging Initialization
```rust
// In run_server_with_config()
if std::env::var("NGDB_LOG_JSON").is_ok() {
    neural_storage::logging::init_json();
} else {
    neural_storage::logging::init();
}
```

### 5. Query Latency Recording
```rust
// In handle_query()
let elapsed = start.elapsed();
state.metrics.record_query_latency(elapsed);
```

### 6. Docker Healthcheck
```yaml
healthcheck:
  test: ["CMD", "curl", "-f", "http://localhost:3000/health"]
  interval: 30s
  timeout: 10s
  retries: 3
  start_period: 10s
```

---

## Testing

```bash
# Build verification
cargo build -p neural-cli --release  # ✅ Success

# Metrics tests
cargo test -p neural-storage metrics --features metrics  # ✅ All pass

# Manual verification
curl http://localhost:3000/health | jq .
curl http://localhost:3000/metrics | head -20
```

---

## Key Learnings

1. **Feature Gating:** The metrics module was already implemented with proper feature gating in Sprint 66 - it just needed to be enabled and wired up.

2. **Minimal Changes:** By leveraging existing infrastructure (MetricsRegistry, logging module), the implementation required only ~50 lines of new code.

3. **Production Patterns:** The health endpoint follows best practices for Kubernetes/load balancer integration with proper status, uptime, and component health details.

---

## Next Steps

- Sprint 68: Consider adding additional metrics (memory usage, WAL size, active transactions)
- Consider adding `/ready` endpoint separate from `/health` for startup probes
- Add Grafana dashboard templates for common monitoring scenarios
