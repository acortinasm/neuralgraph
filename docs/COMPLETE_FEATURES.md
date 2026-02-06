# NeuralGraphDB v0.9.10 - Documentación Completa de Funcionalidades

> Base de datos de grafos nativa en Rust para cargas de trabajo de IA

**Versión**: 0.9.11
**Fecha**: 2026-02-04
**Estado**: Fase 7 (Paridad Competitiva y Escala) - Sprint 68 (Security & Authentication)

---

## Tabla de Contenidos

1. [Visión General](#1-visión-general)
2. [Arquitectura del Sistema](#2-arquitectura-del-sistema)
3. [Modelo de Datos](#3-modelo-de-datos)
4. [Lenguaje de Consulta NGQL](#4-lenguaje-de-consulta-ngql)
5. [Motor de Almacenamiento](#5-motor-de-almacenamiento)
6. [Índices y Búsqueda](#6-índices-y-búsqueda)
7. [Búsqueda Full-Text](#7-búsqueda-full-text)
8. [Motor Vectorial (HNSW)](#8-motor-vectorial-hnsw)
9. [Transacciones y MVCC](#9-transacciones-y-mvcc)
10. [Persistencia y Durabilidad](#10-persistencia-y-durabilidad)
11. [Sistema Distribuido (Raft + Sharding)](#11-sistema-distribuido-raft--sharding)
12. [GraphRAG y ETL](#12-graphrag-y-etl)
13. [APIs e Interfaces](#13-apis-e-interfaces)
14. [CLI y Comandos](#14-cli-y-comandos)
15. [Rendimiento y Benchmarks](#15-rendimiento-y-benchmarks)
16. [Historial de Sprints](#16-historial-de-sprints)

---

## 1. Visión General

**NeuralGraphDB** es una base de datos de grafos nativa diseñada específicamente para cargas de trabajo de Inteligencia Artificial (Agentes Autónomos, RAG y GNNs). A diferencia de las bases de datos de grafos tradicionales (basadas en navegación de punteros), NeuralGraphDB utiliza un motor de **álgebra lineal (matrices dispersas)** para unificar la estructura del grafo con representaciones vectoriales.

### 1.1 Características Distintivas

| Característica | Descripción |
|----------------|-------------|
| **Motor de Álgebra Lineal** | Usa matrices dispersas (CSR/CSC) en lugar de navegación de punteros |
| **GraphRAG Nativo** | Búsqueda vectorial HNSW integrada con el grafo |
| **Lenguaje NGQL** | Similar a Cypher con extensiones para IA |
| **Alto Rendimiento** | Sub-millisecond queries, 25-50x más eficiente en memoria que alternativas |
| **Distribuido** | Raft consensus + sharding horizontal |

### 1.2 Tecnologías Core

| Componente | Tecnología |
|------------|------------|
| Lenguaje | Rust 1.85+ (Edition 2024) |
| Almacenamiento | CSR/CSC matrices dispersas |
| Búsqueda Vectorial | HNSW (hnsw_rs) |
| Consenso | Raft (OpenRaft) |
| APIs | Arrow Flight, gRPC (Tonic), HTTP (Axum) |
| Serialización | bincode, serde |

### 1.3 Estadísticas del Código

- **~45,750 líneas de Rust** distribuidas en 5 crates
- **40+ módulos** de almacenamiento
- **18 archivos** de documentación
- **14 queries** LDBC-SNB validadas

---

## 2. Arquitectura del Sistema

### 2.1 Estructura de Crates

```
neuralgraph/
├── Cargo.toml                    # Workspace v0.8.0
├── crates/
│   ├── neural-core/              # Tipos fundamentales (~580 líneas)
│   │   └── src/lib.rs            # NodeId, EdgeId, PropertyValue, traits
│   │
│   ├── neural-storage/           # Motor de almacenamiento (~12,000 líneas)
│   │   ├── src/
│   │   │   ├── csr.rs            # Compressed Sparse Row matrix
│   │   │   ├── csc.rs            # Compressed Sparse Column matrix
│   │   │   ├── graph_store.rs    # GraphStore principal
│   │   │   ├── properties.rs     # PropertyStore, VersionedPropertyStore
│   │   │   ├── persistence.rs    # Formato binario .ngdb
│   │   │   ├── wal.rs            # Write-Ahead Log
│   │   │   ├── wal_reader.rs     # Recuperación de WAL
│   │   │   ├── mvcc.rs           # Multi-Version Concurrency Control
│   │   │   ├── transaction.rs    # Transaction Manager
│   │   │   ├── community.rs      # Leiden algorithm
│   │   │   ├── pdf.rs            # PDF extraction
│   │   │   ├── llm.rs            # LLM clients (OpenAI, Ollama, Gemini)
│   │   │   ├── etl.rs            # Auto-ETL pipeline
│   │   │   ├── csv_loader.rs     # CSV loading
│   │   │   ├── hf_loader.rs      # HuggingFace datasets
│   │   │   ├── la.rs             # Linear algebra primitives
│   │   │   ├── lsm_vec.rs        # LSM vector storage
│   │   │   ├── pma.rs            # Packed Memory Arrays
│   │   │   ├── metrics.rs        # Prometheus metrics
│   │   │   ├── config.rs         # Unified TOML configuration (Sprint 66)
│   │   │   ├── logging.rs        # Structured logging with tracing (Sprint 66)
│   │   │   ├── memory.rs         # Memory tracking and limits (Sprint 66)
│   │   │   ├── constraints.rs    # Unique constraint system (Sprint 66)
│   │   │   ├── statistics.rs     # Graph statistics collection (Sprint 66)
│   │   │   ├── auth.rs           # JWT/API key authentication (Sprint 68)
│   │   │   │
│   │   │   ├── full_text_index/  # Búsqueda full-text
│   │   │   │   ├── mod.rs        # FullTextIndex, SearchResult, search_fuzzy
│   │   │   │   ├── config.rs     # FullTextIndexConfig, Language (18), PhoneticAlgorithm
│   │   │   │   ├── schema.rs     # Analyzer, Schema builder, language mapping
│   │   │   │   └── phonetic.rs   # PhoneticTokenFilter (Soundex, Metaphone)
│   │   │   │
│   │   │   ├── vector_index/     # Motor vectorial
│   │   │   │   ├── mod.rs        # Public exports
│   │   │   │   ├── core.rs       # HNSW, quantization, metrics
│   │   │   │   ├── distributed.rs # Distributed vector search
│   │   │   │   ├── client.rs     # VectorShardClient
│   │   │   │   ├── server.rs     # Vector gRPC server
│   │   │   │   ├── cache.rs      # QueryResultCache
│   │   │   │   └── load_balancer.rs # Load balancing strategies
│   │   │   │
│   │   │   ├── raft/             # Consenso distribuido
│   │   │   │   ├── mod.rs        # Main orchestration
│   │   │   │   ├── types.rs      # Raft types and messages
│   │   │   │   ├── state_machine.rs # Graph mutations state machine
│   │   │   │   ├── log_store.rs  # Persistent log storage
│   │   │   │   ├── network.rs    # gRPC network layer
│   │   │   │   ├── cluster.rs    # Cluster membership
│   │   │   │   ├── health.rs     # Node health monitoring
│   │   │   │   └── wrapper.rs    # OpenRaft integration
│   │   │   │
│   │   │   └── sharding/         # Particionamiento horizontal
│   │   │       ├── mod.rs        # Sharding coordination
│   │   │       ├── coordinator.rs # Shard assignment
│   │   │       ├── router.rs     # Query routing
│   │   │       ├── manager.rs    # Shard lifecycle
│   │   │       └── strategy.rs   # Hash/Range/Community partitioning
│   │   │
│   │   └── build.rs              # Protobuf code generation
│   │
│   ├── neural-parser/            # Parser NGQL (~1,500 líneas)
│   │   └── src/
│   │       ├── lexer.rs          # Tokenizer (logos)
│   │       ├── ast.rs            # Abstract Syntax Tree
│   │       ├── parser.rs         # Recursive descent parser
│   │       └── lib.rs            # Public API
│   │
│   ├── neural-executor/          # Planificador y ejecutor (~4,000 líneas)
│   │   └── src/
│   │       ├── planner.rs        # Logical/Physical planning
│   │       ├── plan.rs           # Plan node types
│   │       ├── executor.rs       # Streaming execution
│   │       ├── eval.rs           # Expression evaluation
│   │       ├── aggregate.rs      # Aggregation functions
│   │       ├── result.rs         # QueryResult, Row, Value
│   │       ├── cmp.rs            # Comparison operations
│   │       └── lib.rs            # Public API
│   │
│   └── neural-cli/               # CLI y servidores (~2,200 líneas)
│       └── src/
│           ├── main.rs           # REPL entry point
│           ├── server.rs         # HTTP/REST API (Axum)
│           ├── auth_middleware.rs # Auth middleware (Sprint 68)
│           ├── flight.rs         # Arrow Flight server
│           └── raft_server.rs    # Raft consensus server
│
├── proto/                        # Protocol Buffers
│   ├── raft.proto                # Raft consensus RPC
│   └── vector.proto              # Distributed vector search RPC
│
├── docs/                         # Documentación (18 archivos)
├── benchmarks/                   # Python benchmarking scripts
├── benches/                      # Criterion benchmarks
└── scripts/                      # Utility scripts
```

### 2.2 Capas Funcionales

```
┌─────────────────────────────────────────────────────────────────┐
│                    CAPA DE INTERFAZ                              │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐        │
│  │   REPL   │  │   HTTP   │  │  Arrow   │  │   Raft   │        │
│  │   CLI    │  │   REST   │  │  Flight  │  │   gRPC   │        │
│  └────┬─────┘  └────┬─────┘  └────┬─────┘  └────┬─────┘        │
└───────┼─────────────┼─────────────┼─────────────┼───────────────┘
        │             │             │             │
┌───────▼─────────────▼─────────────▼─────────────▼───────────────┐
│                    CAPA DE PARSING                               │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │                    NGQL Parser                            │   │
│  │  Lexer (logos) → AST → Planner → Physical Plan           │   │
│  └──────────────────────────────────────────────────────────┘   │
└─────────────────────────────┬───────────────────────────────────┘
                              │
┌─────────────────────────────▼───────────────────────────────────┐
│                    CAPA DE INTELIGENCIA                          │
│  ┌────────────┐  ┌────────────┐  ┌────────────┐                 │
│  │   HNSW     │  │   Leiden   │  │    LLM     │                 │
│  │  Vectors   │  │ Community  │  │  Clients   │                 │
│  └────────────┘  └────────────┘  └────────────┘                 │
└─────────────────────────────┬───────────────────────────────────┘
                              │
┌─────────────────────────────▼───────────────────────────────────┐
│                    CAPA DE CÓMPUTO                               │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │              Streaming Executor (Iteradores)              │   │
│  │  Scan → Filter → Expand → Join → Project → Aggregate     │   │
│  └──────────────────────────────────────────────────────────┘   │
└─────────────────────────────┬───────────────────────────────────┘
                              │
┌─────────────────────────────▼───────────────────────────────────┐
│                    CAPA DE ALMACENAMIENTO                        │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐        │
│  │   CSR    │  │   MVCC   │  │   WAL    │  │   Raft   │        │
│  │  Matrix  │  │ Versioned│  │  Logger  │  │ Replicate│        │
│  └──────────┘  └──────────┘  └──────────┘  └──────────┘        │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐        │
│  │  Label   │  │ Property │  │ EdgeType │  │  Vector  │        │
│  │  Index   │  │  Index   │  │  Index   │  │  Index   │        │
│  └──────────┘  └──────────┘  └──────────┘  └──────────┘        │
└─────────────────────────────────────────────────────────────────┘
```

---

## 3. Modelo de Datos

### 3.1 Nodos (Nodes)

```rust
/// Identificador type-safe de 64 bits para nodos
pub struct NodeId(pub u64);

// Características de un nodo:
// - ID único inmutable
// - Labels opcionales (múltiples permitidos): :Person, :Document, :Company
// - Propiedades arbitrarias (key-value)
// - Embeddings vectoriales opcionales
```

**Ejemplo NGQL:**
```cypher
CREATE (n:Person:Employee {
    name: "Alice",
    age: 30,
    email: "alice@example.com",
    embedding: [0.1, 0.2, 0.3, ...]
})
```

### 3.2 Aristas (Edges)

```rust
/// Estructura de una arista
pub struct Edge {
    pub id: EdgeId,           // Identificador único
    pub source: NodeId,       // Nodo origen
    pub target: NodeId,       // Nodo destino
    pub label: Option<Label>, // Tipo de relación
    pub port: u16,            // Puerto para multi-aristas (Sprint 57)
}
```

**Direcciones soportadas:**
| Sintaxis | Dirección | Descripción |
|----------|-----------|-------------|
| `->` | Outgoing | Del nodo origen al destino |
| `<-` | Incoming | Del nodo destino al origen |
| `--` | Bidirectional | Ambas direcciones |

**Ejemplo NGQL:**
```cypher
-- Arista simple
CREATE (a)-[:KNOWS {since: 2020}]->(b)

-- Arista con dirección inversa
MATCH (a)<-[:REPORTS_TO]-(b) RETURN a, b

-- Multi-aristas con puertos
CREATE (a)-[:TRANSFER {port: 0, amount: 100}]->(b)
CREATE (a)-[:TRANSFER {port: 1, amount: 200}]->(b)
```

### 3.3 Propiedades (PropertyValue)

```rust
/// Tipos de valores soportados
pub enum PropertyValue {
    Null,                              // Valor ausente
    Bool(bool),                        // true, false
    Int(i64),                          // Enteros con signo
    Float(f64),                        // Punto flotante
    String(String),                    // Cadenas UTF-8
    Date(String),                      // Formato YYYY-MM-DD
    DateTime(String),                  // ISO 8601
    Vector(Vec<f32>),                  // Embeddings vectoriales
    Array(Vec<PropertyValue>),         // Array heterogéneo (Sprint 64)
    Map(HashMap<String, PropertyValue>), // Map/JSON (Sprint 64)
}
```

#### Arrays y Maps (Sprint 64)

NeuralGraphDB soporta tipos de datos complejos nativos, cerrando la brecha con FalkorDB:

**Arrays:**
```cypher
// Array de strings
CREATE (n:Person {name: "Alice", tags: ["rust", "graph", "database"]})

// Array mixto
CREATE (n:Item {data: ["text", 123, true, null]})

// Arrays numéricos se convierten automáticamente a Vector para embeddings
CREATE (n:Doc {embedding: [0.1, 0.2, 0.3, 0.4]})
```

**Maps:**
```cypher
// Map con propiedades anidadas
CREATE (n:Config {name: "settings", options: {debug: true, level: 5, mode: "fast"}})

// Maps anidados
CREATE (n:Profile {name: "Bob", metadata: {scores: [100, 95, 88], active: true}})

// Map vacío
CREATE (n:Empty {config: {}})
```

**Acceso a propiedades:**
```rust
// Acceso a array
let arr = PropertyValue::Array(vec![...]);
arr.get_index(0)     // Acceso por índice
arr.as_array()       // Slice del array
arr.is_array()       // Verificación de tipo

// Acceso a map
let map = PropertyValue::Map(HashMap::from([...]));
map.get("key")       // Acceso por clave
map.as_map()         // Referencia al HashMap
map.is_map()         // Verificación de tipo
```

**Tabla de tipos:**

| Tipo NGQL | Rust Type | Ejemplo | Bytes |
|-----------|-----------|---------|-------|
| `null` | `Null` | `null` | 0 |
| `true/false` | `bool` | `true` | 1 |
| Integer | `i64` | `42`, `-100` | 8 |
| Float | `f64` | `3.14`, `-0.5` | 8 |
| String | `String` | `"hello"` | variable |
| Date | `String` | `date('2026-01-28')` | ~10 |
| DateTime | `String` | `datetime('2026-01-28T12:00:00Z')` | ~24 |
| Vector | `Vec<f32>` | `[0.1, 0.2, 0.3]` | 4 × dim |

### 3.4 Port Numbering (Multi-aristas) - Sprint 57

Permite múltiples aristas paralelas entre el mismo par de nodos con el mismo tipo:

```cypher
-- Crear múltiples transferencias entre A y B
CREATE (a)-[:TRANSFER {port: 0, amount: 100, date: "2026-01-01"}]->(b)
CREATE (a)-[:TRANSFER {port: 1, amount: 200, date: "2026-01-15"}]->(b)
CREATE (a)-[:TRANSFER {port: 2, amount: 50, date: "2026-01-28"}]->(b)

-- Consultar arista específica por puerto
MATCH (a)-[r:TRANSFER:1]->(b) RETURN r.amount  -- 200

-- Consultar todos los puertos
MATCH (a)-[r:TRANSFER]->(b) RETURN r.port, r.amount
-- Resultado:
-- | port | amount |
-- |------|--------|
-- | 0    | 100    |
-- | 1    | 200    |
-- | 2    | 50     |
```

---

## 4. Lenguaje de Consulta NGQL

NGQL (Neural Graph Query Language) es un lenguaje de consulta tipo Cypher con extensiones para IA y GraphRAG.

### 4.1 Cláusulas de Lectura

#### 4.1.1 MATCH - Pattern Matching

```cypher
-- Todos los nodos
MATCH (n) RETURN n

-- Nodos con label específico
MATCH (n:Person) RETURN n.name

-- Nodos con propiedades
MATCH (n) WHERE n.name = "Alice" RETURN n
MATCH (n:Person) WHERE n.age = 30 RETURN n

-- Patrones con aristas
MATCH (a)-[r]->(b) RETURN a, type(r), b
MATCH (a:Person)-[:KNOWS]->(b:Person) RETURN a.name, b.name

-- Aristas con propiedades
MATCH (a)-[r:WORKS_AT]->(b) WHERE r.role = "Engineer" RETURN a, b

-- Dirección inversa
MATCH (a)<-[:MANAGES]-(b) RETURN a.name AS employee, b.name AS manager

-- Bidireccional
MATCH (a)-[:FRIENDS]-(b) RETURN a, b

-- Caminos de longitud variable
MATCH (a)-[*1..3]->(b) RETURN a, b           -- 1 a 3 hops
MATCH (a)-[:KNOWS*2..5]->(b) RETURN b        -- 2 a 5 hops con tipo
MATCH (a)-[*]->(b) RETURN a, b               -- Cualquier longitud

-- Path binding
MATCH p = (a)-[:KNOWS*]->(b)
WHERE a.name = "Alice" AND b.name = "Bob"
RETURN p, length(p)

-- Múltiples patrones
MATCH (a:Person), (b:Company)
WHERE a.company_id = id(b)
RETURN a.name, b.name
```

#### 4.1.2 WHERE - Filtrado

```cypher
-- Comparaciones numéricas
WHERE n.age > 30
WHERE n.age >= 18 AND n.age <= 65
WHERE n.score <> 0

-- Comparaciones de strings
WHERE n.name = "Alice"
WHERE n.name <> "Bob"

-- Operadores lógicos
WHERE n.active = true AND n.verified = true
WHERE n.category = "A" OR n.category = "B"
WHERE NOT n.deleted

-- Predicados de strings
WHERE n.name STARTS WITH "A"
WHERE n.email CONTAINS "@gmail"
WHERE n.title ENDS WITH "PhD"

-- Null checks
WHERE n.email IS NOT NULL
WHERE n.deleted IS NULL

-- Listas (IN)
WHERE n.status IN ["active", "pending", "review"]

-- Expresiones complejas
WHERE (n.age > 18 AND n.verified) OR n.admin = true
```

#### 4.1.3 RETURN - Proyección

```cypher
-- Nodo completo
RETURN n

-- Propiedades específicas
RETURN n.name, n.age, n.email

-- Alias
RETURN n.name AS nombre, n.age AS edad

-- Valores únicos
RETURN DISTINCT n.category

-- Ordenamiento
RETURN n ORDER BY n.age DESC
RETURN n ORDER BY n.lastName, n.firstName ASC

-- Limitación
RETURN n LIMIT 10
RETURN n ORDER BY n.score DESC LIMIT 5

-- Paginación
RETURN n SKIP 20 LIMIT 10

-- Combinaciones
RETURN n.name, n.age
ORDER BY n.age DESC
SKIP 10
LIMIT 5
```

#### 4.1.4 Agregaciones

```cypher
-- Conteo
RETURN COUNT(*)
RETURN COUNT(n)
RETURN COUNT(DISTINCT n.category)

-- Suma
RETURN SUM(n.amount)
RETURN SUM(n.price * n.quantity) AS total

-- Promedio
RETURN AVG(n.age)
RETURN AVG(n.score) AS average_score

-- Mínimo y máximo
RETURN MIN(n.age), MAX(n.age)
RETURN MIN(n.created_at) AS oldest

-- Colección
RETURN COLLECT(n.name)
RETURN COLLECT(DISTINCT n.tag)

-- Combinaciones
RETURN
    COUNT(*) AS total,
    AVG(n.age) AS avg_age,
    MIN(n.age) AS min_age,
    MAX(n.age) AS max_age
```

#### 4.1.5 GROUP BY

```cypher
-- Agrupación básica
MATCH (n:Person)
RETURN n.country, COUNT(*) AS total
GROUP BY n.country

-- Múltiples columnas
MATCH (n:Employee)
RETURN n.department, n.level, COUNT(*) AS count, AVG(n.salary) AS avg_salary
GROUP BY n.department, n.level
ORDER BY n.department, count DESC

-- Con filtro post-agregación (HAVING simulado con WITH)
MATCH (n:Product)
WITH n.category AS cat, COUNT(*) AS cnt
WHERE cnt > 10
RETURN cat, cnt
```

#### 4.1.6 WITH - Pipeline

```cypher
-- Transformación intermedia
MATCH (n:Person)
WITH n, n.age AS age
WHERE age > 30
RETURN n.name, age

-- Ordenar y limitar antes de continuar
MATCH (n:Person)
WITH n ORDER BY n.score DESC LIMIT 10
MATCH (n)-[:KNOWS]->(friend)
RETURN n.name, COLLECT(friend.name) AS top_friends

-- Agregación intermedia
MATCH (n:Person)-[:PURCHASED]->(p:Product)
WITH n, COUNT(p) AS purchase_count
WHERE purchase_count > 5
RETURN n.name, purchase_count AS loyal_customers

-- Múltiples WITH
MATCH (a:Person)
WITH a WHERE a.active = true
WITH a ORDER BY a.created_at DESC LIMIT 100
RETURN a.name
```

#### 4.1.7 UNWIND - Expansión de Listas

```cypher
-- Lista literal
UNWIND [1, 2, 3] AS x
RETURN x

-- Parámetro de lista
UNWIND $ids AS id
MATCH (n) WHERE id(n) = id
RETURN n

-- Desde COLLECT
MATCH (n:Tag)
WITH COLLECT(n.name) AS tags
UNWIND tags AS tag
RETURN DISTINCT tag
```

#### 4.1.8 OPTIONAL MATCH - Left Join

```cypher
-- Retorna null si no hay match
MATCH (p:Person)
OPTIONAL MATCH (p)-[:OWNS]->(c:Car)
RETURN p.name, c.model  -- c.model puede ser null

-- Múltiples optional
MATCH (p:Person)
OPTIONAL MATCH (p)-[:LIVES_IN]->(city:City)
OPTIONAL MATCH (p)-[:WORKS_AT]->(company:Company)
RETURN p.name, city.name, company.name
```

### 4.2 Cláusulas de Escritura

#### 4.2.1 CREATE

```cypher
-- Crear nodo simple
CREATE (n:Person {name: "Alice"})

-- Crear nodo con múltiples labels
CREATE (n:Person:Employee:Manager {name: "Bob", level: 5})

-- Crear arista entre nodos existentes
MATCH (a:Person), (b:Person)
WHERE a.name = "Alice" AND b.name = "Bob"
CREATE (a)-[:KNOWS {since: 2020}]->(b)

-- Crear patrón completo
CREATE (a:Person {name: "Carol"})-[:WORKS_AT {role: "Engineer"}]->(c:Company {name: "Acme"})

-- Crear múltiples elementos
CREATE (a:Person {name: "Dan"}), (b:Person {name: "Eve"})
CREATE (a)-[:KNOWS]->(b)
```

#### 4.2.2 DELETE

```cypher
-- Eliminar nodo (debe estar desconectado)
MATCH (n:Person)
WHERE n.name = "Alice"
DELETE n

-- Eliminar nodo con todas sus aristas (DETACH)
MATCH (n:Person)
WHERE n.name = "Alice"
DETACH DELETE n

-- Eliminar arista específica
MATCH (a)-[r:KNOWS]->(b)
WHERE a.name = "Alice" AND b.name = "Bob"
DELETE r

-- Eliminar múltiples
MATCH (n:TempNode)
DETACH DELETE n
```

#### 4.2.3 SET - Actualización de Propiedades

```cypher
-- Establecer una propiedad
MATCH (n:Person)
WHERE n.name = "Alice"
SET n.age = 31

-- Establecer múltiples propiedades
MATCH (n:Person)
WHERE n.name = "Alice"
SET n.age = 31, n.updated = true, n.version = 2

-- Incrementar valor
MATCH (n:Counter)
WHERE n.id = 1
SET n.value = n.value + 1

-- Establecer a null (eliminar propiedad)
MATCH (n:Person)
WHERE n.name = "Alice"
SET n.temporary = null
```

#### 4.2.4 MERGE - Upsert

```cypher
-- Crear si no existe
MERGE (n:Person {email: "alice@example.com"})

-- Con propiedades de creación
MERGE (n:Person {email: "alice@example.com"})
ON CREATE SET n.created = datetime(), n.status = "new"

-- Con propiedades de actualización
MERGE (n:Person {email: "alice@example.com"})
ON MATCH SET n.lastSeen = datetime(), n.visits = n.visits + 1

-- Combinado
MERGE (n:Person {email: "alice@example.com"})
ON CREATE SET n.created = datetime()
ON MATCH SET n.updated = datetime()

-- MERGE de aristas
MATCH (a:Person), (b:Person)
WHERE a.name = "Alice" AND b.name = "Bob"
MERGE (a)-[:KNOWS]->(b)
```

### 4.3 Funciones Incorporadas

#### 4.3.1 Funciones Escalares

| Función | Descripción | Ejemplo |
|---------|-------------|---------|
| `id(n)` | ID numérico del nodo | `RETURN id(n)` |
| `type(r)` | Tipo de la arista | `RETURN type(r)` |
| `labels(n)` | Lista de labels del nodo | `RETURN labels(n)` |
| `length(p)` | Longitud del path | `RETURN length(p)` |
| `size(list)` | Tamaño de lista | `RETURN size(COLLECT(n))` |
| `head(list)` | Primer elemento | `RETURN head([1,2,3])` → `1` |
| `last(list)` | Último elemento | `RETURN last([1,2,3])` → `3` |
| `coalesce(a, b, ...)` | Primer valor no-null | `RETURN coalesce(n.nick, n.name)` |
| `CLUSTER(n)` | ID de comunidad Leiden | `RETURN CLUSTER(n)` |

#### 4.3.2 Funciones de String

| Función | Descripción | Ejemplo |
|---------|-------------|---------|
| `toString(x)` | Convertir a string | `toString(42)` → `"42"` |
| `toUpper(s)` | Mayúsculas | `toUpper("hello")` → `"HELLO"` |
| `toLower(s)` | Minúsculas | `toLower("HELLO")` → `"hello"` |
| `trim(s)` | Eliminar espacios | `trim("  hello  ")` → `"hello"` |
| `ltrim(s)` | Trim izquierdo | `ltrim("  hello")` → `"hello"` |
| `rtrim(s)` | Trim derecho | `rtrim("hello  ")` → `"hello"` |
| `replace(s, from, to)` | Reemplazar | `replace("hello", "l", "L")` → `"heLLo"` |
| `substring(s, start, len)` | Subcadena | `substring("hello", 1, 3)` → `"ell"` |
| `left(s, n)` | Primeros n chars | `left("hello", 2)` → `"he"` |
| `right(s, n)` | Últimos n chars | `right("hello", 2)` → `"lo"` |
| `split(s, delim)` | Dividir string | `split("a,b,c", ",")` → `["a","b","c"]` |

#### 4.3.3 Funciones Matemáticas

| Función | Descripción | Ejemplo |
|---------|-------------|---------|
| `abs(x)` | Valor absoluto | `abs(-5)` → `5` |
| `ceil(x)` | Redondeo hacia arriba | `ceil(3.2)` → `4` |
| `floor(x)` | Redondeo hacia abajo | `floor(3.8)` → `3` |
| `round(x)` | Redondeo estándar | `round(3.5)` → `4` |
| `sign(x)` | Signo (-1, 0, 1) | `sign(-5)` → `-1` |
| `sqrt(x)` | Raíz cuadrada | `sqrt(16)` → `4` |
| `log(x)` | Logaritmo natural | `log(2.718)` → `~1` |
| `log10(x)` | Logaritmo base 10 | `log10(100)` → `2` |
| `exp(x)` | Exponencial | `exp(1)` → `2.718...` |
| `rand()` | Aleatorio [0,1) | `rand()` → `0.xyz` |

#### 4.3.4 Funciones Temporales

| Función | Descripción | Ejemplo |
|---------|-------------|---------|
| `date()` | Fecha actual | `date()` → `"2026-01-28"` |
| `datetime()` | DateTime actual | `datetime()` → `"2026-01-28T12:00:00Z"` |
| `date(s)` | Parsear fecha | `date("2026-01-28")` |
| `datetime(s)` | Parsear datetime | `datetime("2026-01-28T12:00:00Z")` |

#### 4.3.5 Expresiones CASE

```cypher
-- CASE simple
RETURN
  CASE n.status
    WHEN "active" THEN "Activo"
    WHEN "pending" THEN "Pendiente"
    WHEN "inactive" THEN "Inactivo"
    ELSE "Desconocido"
  END AS estado

-- CASE con condiciones
RETURN
  CASE
    WHEN n.age < 18 THEN "Menor"
    WHEN n.age < 65 THEN "Adulto"
    ELSE "Senior"
  END AS grupo_edad

-- CASE en WHERE
MATCH (n:Product)
WHERE
  CASE
    WHEN n.category = "electronics" THEN n.price > 100
    ELSE n.price > 50
  END
RETURN n
```

### 4.4 Traversals Avanzados

#### 4.4.1 Camino Más Corto (Shortest Path)

```cypher
-- Sintaxis shortestPath()
MATCH p = shortestPath((a:Person {name: "Alice"})-[*]->(b:Person {name: "Bob"}))
RETURN p, length(p)

-- Sintaxis SHORTEST PATH
MATCH p = SHORTEST PATH (a)-[*1..10]->(b)
WHERE a.name = "Alice" AND b.name = "Bob"
RETURN p

-- Con tipo de arista específico
MATCH p = shortestPath((a)-[:KNOWS*]->(b))
WHERE a.name = "Alice"
RETURN p

-- Retornar nodos del camino
MATCH p = shortestPath((a)-[*]->(b))
RETURN nodes(p), relationships(p), length(p)
```

#### 4.4.2 Caminos de Longitud Variable

```cypher
-- Rango de hops
MATCH (a)-[*1..5]->(b)     -- 1 a 5 hops
MATCH (a)-[*3]->(b)        -- Exactamente 3 hops
MATCH (a)-[*..4]->(b)      -- Hasta 4 hops (min=1)
MATCH (a)-[*2..]->(b)      -- Al menos 2 hops

-- Con tipo de arista
MATCH (a)-[:FOLLOWS*2..3]->(b)
RETURN DISTINCT b

-- Capturar todas las aristas del camino
MATCH p = (a)-[rels*1..3]->(b)
RETURN p, rels

-- Con filtro en camino
MATCH p = (a)-[*1..5]->(b)
WHERE ALL(n IN nodes(p) WHERE n.active = true)
RETURN p
```

### 4.5 Consultas Temporales (Time-Travel) - Sprint 54

```cypher
-- Consultar estado en punto específico del tiempo
MATCH (a:Person) AT TIME '2026-01-15T12:00:00Z'
RETURN a.name, a.status

-- Con patrón completo
MATCH (a:Person)-[:WORKS_AT]->(c:Company) AT TIME '2026-01-01T00:00:00Z'
RETURN a.name, c.name

-- Alternativa con TIMESTAMP
MATCH (a) AT TIMESTAMP '2026-01-15T12:00:00Z'
RETURN a

-- Restaurar base de datos a punto anterior
FLASHBACK TO '2026-01-15T12:00:00Z'
```

### 4.6 Hints de Sharding - Sprint 55

```cypher
-- Consultar shard específico
MATCH (n:Person) USING SHARD 0
RETURN n

-- Consultar múltiples shards
MATCH (n) USING SHARD [0, 1, 2]
RETURN n

-- Combinar con filtros
MATCH (n:Order) USING SHARD [0, 1]
WHERE n.status = "pending"
RETURN n
```

### 4.7 Búsqueda Vectorial - Sprint 56

```cypher
-- Inicializar índice vectorial (requerido antes de almacenar vectores)
CALL neural.vectorInit(768)       -- Para embeddings BERT
CALL neural.vectorInit(1536)      -- Para OpenAI text-embedding-3-small
CALL neural.vectorInit(3584)      -- Para Qwen3-Embedding-8B

-- Crear nodos con propiedades vectoriales (auto-indexados)
CREATE (n:Document {title: "ML Paper", embedding: [0.1, 0.2, ...]})

-- Actualizar propiedad vectorial (auto-indexado)
MATCH (n:Document) WHERE n.title = "ML Paper"
SET n.embedding = [0.3, 0.4, ...]

-- Búsqueda semántica con procedimiento
CALL neural.search($queryVector, 'cosine', 10)
YIELD node, score
RETURN node.title, score

-- Métricas disponibles
CALL neural.search($vec, 'euclidean', 100)    -- Distancia euclidiana
CALL neural.search($vec, 'dot_product', 50)   -- Producto punto

-- Combinar con filtros de grafo
CALL neural.search($query, 'cosine', 20)
YIELD node, score
MATCH (node)-[:AUTHORED_BY]->(author:Person)
RETURN node.title, author.name, score
ORDER BY score DESC
```

**Nota:** El índice vectorial debe inicializarse antes de almacenar propiedades vectoriales. Los vectores se indexan automáticamente al escribirse con CREATE o SET.

### 4.8 Transacciones

```cypher
-- Transacción explícita
BEGIN
CREATE (a:Person {name: "Alice"})
CREATE (b:Person {name: "Bob"})
CREATE (a)-[:KNOWS]->(b)
COMMIT

-- Rollback en caso de error
BEGIN
CREATE (n:Invalid)
-- Error ocurre...
ROLLBACK

-- Transacción con múltiples operaciones
BEGIN
MATCH (n:Counter) SET n.value = n.value + 1
CREATE (log:AuditLog {action: "increment", timestamp: datetime()})
COMMIT
```

### 4.9 Introspección de Queries

```cypher
-- Ver plan de ejecución sin ejecutar
EXPLAIN MATCH (n:Person)-[:KNOWS]->(m)
WHERE n.age > 30
RETURN m.name

-- Salida ejemplo:
-- PhysicalPlan:
--   └─ Project [m.name]
--      └─ Expand (n)-[:KNOWS]->(m)
--         └─ Filter n.age > 30
--            └─ ScanByLabel :Person

-- Ejecutar con métricas de rendimiento
PROFILE MATCH (n) WHERE n.age > 30 RETURN COUNT(*)

-- Salida ejemplo:
-- Rows: 1
-- Execution time: 0.45ms
-- Nodes scanned: 1000
-- Edges traversed: 0
```

---

## 5. Motor de Almacenamiento

### 5.1 Matrices CSR/CSC

#### 5.1.1 CSR (Compressed Sparse Row)

Optimizada para acceso O(1) a vecinos salientes:

```rust
pub struct CsrMatrix {
    row_ptr: Vec<usize>,      // Offset por nodo: [0, 2, 5, 7, ...]
    col_indices: Vec<NodeId>, // IDs de vecinos: [1, 3, 0, 2, 4, ...]
}

// Para obtener vecinos del nodo i:
// neighbors = col_indices[row_ptr[i]..row_ptr[i+1]]
```

**Estadísticas disponibles:**
- `node_count`: Número de nodos
- `edge_count`: Número de aristas
- `max_degree`: Grado máximo
- `avg_degree`: Grado promedio
- `memory_bytes`: Uso de memoria

#### 5.1.2 CSC (Compressed Sparse Column)

Optimizada para acceso O(1) a aristas entrantes (transposición de CSR):

```rust
pub struct CscMatrix {
    col_ptr: Vec<usize>,      // Offset por columna
    row_indices: Vec<NodeId>, // Nodos origen
    edge_ids: Vec<EdgeId>,    // IDs de aristas
}

// Para obtener aristas entrantes al nodo j:
// sources = row_indices[col_ptr[j]..col_ptr[j+1]]
```

### 5.2 GraphStore

Estructura principal que unifica todos los componentes:

```rust
pub struct GraphStore {
    // Estructura del grafo
    graph: CsrMatrix,                           // Adyacencia saliente
    reverse_graph: CscMatrix,                   // Adyacencia entrante

    // Propiedades con versionado
    versioned_properties: VersionedPropertyStore, // Propiedades MVCC
    versioned_labels: VersionedPropertyStore,     // Labels MVCC

    // Índices
    label_index: LabelIndex,                    // Label -> Vec<NodeId>
    property_index: PropertyIndex,              // (prop, val) -> Vec<NodeId>
    edge_type_index: EdgeTypeIndex,             // type -> Vec<(EdgeId, src, tgt, port)>
    timestamp_index: TimestampIndex,            // Para time-travel

    // Búsqueda vectorial
    vector_index: Option<VectorIndex>,          // HNSW

    // Persistencia
    wal: Option<WalWriter>,                     // Write-Ahead Log
    path: Option<PathBuf>,                      // Archivo de BD

    // Transacciones
    current_tx_id: TransactionId,               // ID actual
    snapshot_id: TransactionId,                 // Para lecturas
}
```

### 5.3 Complejidades de Operaciones

| Operación | Complejidad | Estructura |
|-----------|-------------|------------|
| Lookup nodo por ID | O(1) | HashMap |
| Vecinos salientes | O(degree) | CSR row slice |
| Vecinos entrantes | O(degree) | CSC column slice |
| Nodos por label | O(1) | LabelIndex |
| Nodos por propiedad | O(1) | PropertyIndex |
| Aristas por tipo | O(1) | EdgeTypeIndex |
| Búsqueda vectorial k-NN | O(log n) | HNSW |
| Scan completo | O(n) | Streaming iterator |
| Creación de nodo | O(1) amortizado | Append |
| Creación de arista | O(1) amortizado | Append + índices |
| Actualización propiedad | O(1) | MVCC versión |

---

## 6. Índices y Búsqueda

### 6.1 LabelIndex

Índice invertido para búsqueda O(1) por label:

```rust
// Estructura interna
pub struct LabelIndex {
    index: HashMap<Label, Vec<NodeId>>,
    finalized: bool,
}

// Operaciones
impl LabelIndex {
    /// Agregar nodo a un label
    pub fn add(&mut self, node_id: NodeId, label: Label);

    /// Obtener todos los nodos con un label
    pub fn get(&self, label: &Label) -> Option<&Vec<NodeId>>;

    /// Verificar si nodo tiene label (binary search si finalized)
    pub fn contains(&self, node_id: NodeId, label: &Label) -> bool;

    /// Ordenar listas para binary search
    pub fn finalize(&mut self);

    /// Eliminar nodo de label
    pub fn remove(&mut self, node_id: NodeId, label: &Label);
}
```

**Uso en queries:**
```cypher
-- O(1) lookup via LabelIndex
MATCH (n:Person) RETURN n
```

### 6.2 PropertyIndex

Índice invertido para filtros WHERE eficientes:

```rust
// Estructura interna
pub struct PropertyIndex {
    // property_name -> value -> Vec<NodeId>
    index: HashMap<String, HashMap<PropertyValueKey, Vec<NodeId>>>,
}

// Operaciones
impl PropertyIndex {
    pub fn add(&mut self, node_id: NodeId, property: &str, value: &PropertyValue);
    pub fn get(&self, property: &str, value: &PropertyValue) -> Option<&Vec<NodeId>>;
    pub fn remove(&mut self, node_id: NodeId, property: &str, value: &PropertyValue);
    pub fn finalize(&mut self);
}
```

**Uso en queries:**
```cypher
-- O(1) lookup via PropertyIndex
MATCH (n:Paper) WHERE n.category = "cs.LG" RETURN n
```

### 6.3 EdgeTypeIndex

Índice de aristas por tipo con soporte de puertos:

```rust
pub struct EdgeTypeIndex {
    // edge_type -> Vec<(EdgeId, source, target, port)>
    index: HashMap<Label, Vec<(EdgeId, NodeId, NodeId, u16)>>,
}

impl EdgeTypeIndex {
    /// Aristas salientes de un nodo con tipo específico
    pub fn edges_from(&self, source: NodeId, edge_type: &Label)
        -> impl Iterator<Item = (EdgeId, NodeId, u16)>;

    /// Arista específica con puerto
    pub fn edge_with_port(&self, src: NodeId, tgt: NodeId,
                          edge_type: &Label, port: u16) -> Option<EdgeId>;

    /// Todos los puertos entre par de nodos
    pub fn ports_between(&self, src: NodeId, tgt: NodeId,
                         edge_type: &Label) -> Vec<u16>;
}
```

**Uso en queries:**
```cypher
-- O(1) lookup via EdgeTypeIndex
MATCH (a)-[:KNOWS]->(b) RETURN a, b

-- Con puerto específico
MATCH (a)-[:TRANSFER:0]->(b) RETURN a, b
```

### 6.4 TimestampIndex

Para consultas time-travel:

```rust
pub struct TimestampIndex {
    // Ordenado por timestamp
    entries: Vec<(DateTime, TransactionId)>,
}

impl TimestampIndex {
    /// Obtener TransactionId activo en un momento dado
    pub fn get_tx_at_or_before(&self, timestamp: &DateTime) -> Option<TransactionId>;

    /// Registrar nuevo commit
    pub fn record(&mut self, timestamp: DateTime, tx_id: TransactionId);
}
```

**Uso en queries:**
```cypher
-- Consulta el estado al 15 de enero
MATCH (n) AT TIME '2026-01-15T12:00:00Z' RETURN n
```

---

## 7. Búsqueda Full-Text

### 7.1 Arquitectura

NeuralGraphDB incluye búsqueda full-text nativa usando **tantivy** (el equivalente a Lucene en Rust). Esto permite buscar texto en propiedades de nodos con ranking de relevancia.

```
┌─────────────────────────────────────────────────────────────────┐
│                       FullTextIndex                              │
├─────────────────────────────────────────────────────────────────┤
│  ┌─────────────┐  ┌─────────────────┐  ┌──────────────────────┐ │
│  │ Tantivy     │  │  Schema with    │  │  Node ID Mapping     │ │
│  │ Index       │  │  Text Fields    │  │  (u64 -> NodeId)     │ │
│  └─────────────┘  └─────────────────┘  └──────────────────────┘ │
│                                                                  │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │                    Text Analyzer                          │   │
│  │  SimpleTokenizer → LowerCaser → StopWords → Stemmer      │   │
│  └──────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
```

### 7.2 Características

| Característica | Descripción |
|----------------|-------------|
| **Stemming** | Reduce palabras a su raíz (e.g., "learning" → "learn") |
| **Stop Words** | Filtra palabras comunes (the, a, is, etc.) |
| **Phrase Queries** | Búsqueda exacta de frases con comillas |
| **Boolean Queries** | Combinar términos con AND, OR, NOT |
| **Relevance Ranking** | Ordenamiento por score BM25 |
| **Persistence** | Índices persistentes en directorio `.fts/` |
| **Fuzzy Matching** | Tolerancia a errores tipográficos (Levenshtein distance) |
| **Phonetic Search** | Coincidencia por sonido (Soundex, Metaphone, DoubleMetaphone) |
| **Multi-idioma** | 18 idiomas soportados para stemming y stop words |

### 7.3 Procedimientos CALL

#### Crear Índice Full-Text

```cypher
-- Crear índice sobre una o más propiedades
CALL neural.fulltext.createIndex('paper_search', 'Paper', ['title', 'abstract'])

-- Crear índice con idioma específico
CALL neural.fulltext.createIndex('spanish_idx', 'Article', ['titulo', 'contenido'], 'spanish')

-- Crear índice con búsqueda fonética
CALL neural.fulltext.createIndex('name_idx', 'Person', ['name'], 'english', 'soundex')

-- Resultado:
-- | result |
-- | "Created full-text index 'paper_search' on :Paper(['title', 'abstract'])" |
```

#### Buscar en el Índice

```cypher
-- Búsqueda simple
CALL neural.fulltext.query('paper_search', 'machine learning', 10)
YIELD node, score
RETURN node.title, score
ORDER BY score DESC

-- Búsqueda con frase exacta
CALL neural.fulltext.query('paper_search', '"neural networks"', 5)
YIELD node, score
RETURN node.title, score

-- Búsqueda booleana
CALL neural.fulltext.query('paper_search', 'deep AND learning NOT convolutional', 10)
YIELD node, score
RETURN node.title, score
```

#### Búsqueda Fuzzy (Sprint 63)

```cypher
-- Búsqueda con tolerancia a errores tipográficos
-- El 4º argumento es la distancia de Levenshtein máxima (default: 1)
CALL neural.fulltext.fuzzyQuery('paper_search', 'machin lerning', 10, 2)
YIELD node, score
RETURN node.title, score

-- Distancia 1: tolera 1 carácter de diferencia (e.g., "machin" → "machine")
-- Distancia 2: tolera hasta 2 caracteres de diferencia
```

#### Búsqueda Fonética (Sprint 63)

```cypher
-- Si el índice tiene phonetic habilitado, la búsqueda por sonido es automática
-- Crear índice con Soundex
CALL neural.fulltext.createIndex('name_idx', 'Person', ['name'], 'english', 'soundex')

-- "Smith" coincide con "Smyth", "Smithe", etc.
CALL neural.fulltext.query('name_idx', 'Smith', 10)
YIELD node, score
RETURN node.name, score

-- Algoritmos disponibles: soundex, metaphone, double_metaphone
```

#### Listar Índices

```cypher
-- Ver todos los índices full-text
CALL neural.fulltext.indexes()

-- Resultado:
-- | name | label | properties | document_count |
-- | "paper_search" | "Paper" | ["title", "abstract"] | 1500 |
```

#### Eliminar Índice

```cypher
-- Eliminar un índice
CALL neural.fulltext.dropIndex('paper_search')

-- Resultado:
-- | result |
-- | "Dropped full-text index 'paper_search'" |
```

### 7.4 Sintaxis de Búsqueda

| Tipo | Sintaxis | Ejemplo | Descripción |
|------|----------|---------|-------------|
| **Términos simples** | `term1 term2` | `machine learning` | Busca documentos con cualquier término |
| **Frase exacta** | `"term1 term2"` | `"neural network"` | Busca secuencia exacta |
| **AND** | `term1 AND term2` | `deep AND learning` | Ambos términos requeridos |
| **OR** | `term1 OR term2` | `CNN OR RNN` | Cualquier término |
| **NOT** | `-term` o `NOT term` | `learning -deep` | Excluir término |
| **Campo específico** | `field:term` | `title:introduction` | Buscar en campo específico |

### 7.5 Configuración del Analizador

```rust
pub struct AnalyzerConfig {
    /// Idioma para stemming (default: English)
    pub language: Language,
    /// Algoritmo fonético (default: None)
    pub phonetic: PhoneticAlgorithm,
    /// Convertir a minúsculas (default: true)
    pub lowercase: bool,
    /// Eliminar stop words (default: true)
    pub remove_stopwords: bool,
    /// Aplicar stemming (default: true)
    pub stemming: bool,
}

// 18 idiomas soportados (Sprint 63)
pub enum Language {
    English,    // Porter Stemmer
    Spanish,    // Snowball Spanish
    French,     // Snowball French
    German,     // Snowball German
    Italian,    // Snowball Italian
    Portuguese, // Snowball Portuguese
    Dutch,      // Snowball Dutch
    Swedish,    // Snowball Swedish
    Norwegian,  // Snowball Norwegian
    Danish,     // Snowball Danish
    Finnish,    // Snowball Finnish
    Russian,    // Snowball Russian
    Hungarian,  // Snowball Hungarian
    Romanian,   // Snowball Romanian
    Turkish,    // Snowball Turkish
    Arabic,     // Snowball Arabic
    Greek,      // Snowball Greek
    Tamil,      // Snowball Tamil
}

// Algoritmos fonéticos (Sprint 63)
pub enum PhoneticAlgorithm {
    None,            // Sin fonética
    Soundex,         // Clásico, bueno para nombres en inglés
    Metaphone,       // Más preciso que Soundex
    DoubleMetaphone, // Maneja más casos y palabras no-inglesas
}
```

### 7.6 Casos de Uso

| Caso de Uso | Descripción | Query Ejemplo |
|-------------|-------------|---------------|
| **GraphRAG** | Buscar documentos relevantes para contexto LLM | `neural.fulltext.query('docs', $user_query, 5)` |
| **Knowledge Base** | Buscar en base de conocimiento | `neural.fulltext.query('kb', 'error authentication', 10)` |
| **Research Papers** | Buscar papers académicos | `neural.fulltext.query('papers', '"transformer architecture"', 20)` |
| **Log Analysis** | Buscar en logs de errores | `neural.fulltext.query('logs', 'error AND timeout', 100)` |

---

## 8. Motor Vectorial (HNSW)

### 7.1 Configuración del Índice

```rust
pub struct VectorIndexConfig {
    /// Dimensión del vector (768, 1536, etc.)
    dimension: usize,

    /// Links bidireccionales máximos por nodo (16-48)
    m: usize,

    /// Calidad de construcción (100-500)
    ef_construction: usize,

    /// Capacidad máxima
    max_elements: usize,

    /// Método de cuantización
    quantization: QuantizationMethod,

    /// Métrica de distancia
    metric: DistanceMetric,
}
```

**Perfiles predefinidos:**

```rust
// Para datasets pequeños (<10k vectores)
let config = VectorIndexConfig::small(768);
// m=16, ef_construction=200, max_elements=10,000

// Para datasets grandes (1M+ vectores)
let config = VectorIndexConfig::large(768);
// m=24, ef_construction=400, max_elements=1,000,000

// Con cuantización Int8 (4x ahorro de memoria)
let config = VectorIndexConfig::quantized(768);
// m=24, ef_construction=400, max_elements=1,000,000, Int8

// Configuración custom
let config = VectorIndexConfig::new(768)
    .with_m(32)
    .with_ef_construction(300)
    .with_max_elements(500_000)
    .with_metric(DistanceMetric::Cosine)
    .with_quantization(QuantizationMethod::Int8);
```

### 7.2 Métricas de Distancia

```rust
pub enum DistanceMetric {
    Cosine,      // Similitud coseno: dot(a,b) / (|a| * |b|)
    Euclidean,   // Distancia L2: sqrt(sum((a[i]-b[i])^2))
    DotProduct,  // Producto punto: sum(a[i] * b[i])
}
```

| Métrica | Rango | Interpretación | Uso Recomendado |
|---------|-------|----------------|-----------------|
| **Cosine** | -1.0 a 1.0 | 1.0 = idénticos, 0 = ortogonales | Embeddings LLM (OpenAI, Gemini) |
| **Euclidean** | 0 a ∞ | 0 = idénticos | Sentence transformers |
| **DotProduct** | -∞ a ∞ | Mayor = más similar | Matryoshka embeddings |

**Funciones de distancia:**

```rust
// Similitud coseno
pub fn cosine_similarity(a: &[f32], b: &[f32]) -> f32;

// Distancia coseno (1 - similarity)
pub fn cosine_distance(a: &[f32], b: &[f32]) -> f32;

// Distancia euclidiana
pub fn euclidean_distance(a: &[f32], b: &[f32]) -> f32;

// Producto punto
pub fn dot_product(a: &[f32], b: &[f32]) -> f32;
```

### 7.3 Cuantización (Sprint 60)

```rust
pub enum QuantizationMethod {
    None,    // f32 completo (4 bytes/dim)
    Int8,    // Cuantización escalar (1 byte/dim)
    Binary,  // 1 bit por dimensión (0.125 bytes/dim)
}
```

**Comparación de métodos:**

| Método | Bytes/Dim | Ahorro vs f32 | Precisión | Uso |
|--------|-----------|---------------|-----------|-----|
| **None** | 4 | 0% | 100% | Producción, precisión crítica |
| **Int8** | 1 | 75% (4x) | ~99% | Balance memoria/precisión |
| **Binary** | 0.125 | 97% (32x) | ~90% | Datasets masivos |

**Parámetros de cuantización Int8:**

```rust
pub struct QuantizationParams {
    pub scale: f32,   // Factor de escala
    pub offset: f32,  // Offset para dequantización
}

impl QuantizationParams {
    /// Calcular parámetros óptimos para un vector
    pub fn from_vector(vector: &[f32]) -> Self;
}

// Fórmulas:
// Cuantizar: q[i] = clamp(round((v[i] - offset) / scale) - 128, -128, 127)
// Dequantizar: v[i] = (q[i] + 128) * scale + offset
```

**Funciones de cuantización:**

```rust
/// Cuantizar f32 a i8
pub fn quantize_f32_to_i8(vector: &[f32], params: &QuantizationParams) -> Vec<i8>;

/// Dequantizar i8 a f32
pub fn dequantize_i8_to_f32(quantized: &[i8], params: &QuantizationParams) -> Vec<f32>;

/// Cuantización binaria (1 bit por dimensión)
pub fn quantize_f32_to_binary(vector: &[f32]) -> Vec<u8>;

/// Distancia Hamming para vectores binarios
pub fn hamming_distance(a: &[u8], b: &[u8]) -> u32;

/// Distancia asimétrica (query f32, índice cuantizado)
pub fn asymmetric_distance_i8(
    query: &[f32],
    quantized: &[i8],
    params: &QuantizationParams,
    metric: DistanceMetric
) -> f32;
```

### 7.4 Metadata de Embeddings (Sprint 56)

```rust
pub struct EmbeddingMetadata {
    model: String,                        // "text-embedding-3-small"
    metric: DistanceMetric,               // Métrica óptima para este modelo
    created_at: String,                   // ISO 8601 timestamp
    dimension: Option<usize>,             // 768, 1536, etc.
    properties: HashMap<String, String>,  // Propiedades custom
}

impl EmbeddingMetadata {
    pub fn new(model: &str) -> Self;
    pub fn with_metric(self, metric: DistanceMetric) -> Self;
    pub fn with_dimension(self, dim: usize) -> Self;
    pub fn with_property(self, key: &str, value: &str) -> Self;
}
```

**Metadata a nivel de índice:**

```rust
pub struct IndexMetadata {
    pub model: Option<String>,            // Modelo predominante
    pub metric: DistanceMetric,           // Métrica del índice
    pub created_at: String,               // Fecha de creación
    pub embedding_count: usize,           // Total de embeddings
    pub quantization: QuantizationMethod, // Tipo de cuantización
}
```

### 7.5 API de VectorIndex

```rust
impl VectorIndex {
    // ========== Construcción ==========

    /// Crear índice con dimensión
    pub fn new(dimension: usize) -> Self;

    /// Crear con configuración personalizada
    pub fn with_config(config: VectorIndexConfig) -> Self;

    // ========== Operaciones de Datos ==========

    /// Agregar vector
    pub fn add(&mut self, node: NodeId, vector: &[f32]);

    /// Agregar vector con metadata
    pub fn add_with_metadata(
        &mut self,
        node: NodeId,
        vector: &[f32],
        metadata: EmbeddingMetadata
    );

    // ========== Búsqueda ==========

    /// Búsqueda k-NN básica
    pub fn search(&self, query: &[f32], k: usize) -> Vec<(NodeId, f32)>;

    /// Búsqueda con filtro
    pub fn search_filtered<F>(
        &self,
        query: &[f32],
        k: usize,
        filter: F
    ) -> Vec<(NodeId, f32)>
    where F: Fn(NodeId) -> bool;

    // ========== Información ==========

    pub fn len(&self) -> usize;
    pub fn is_empty(&self) -> bool;
    pub fn dimension(&self) -> usize;
    pub fn contains(&self, node: NodeId) -> bool;
    pub fn quantization_method(&self) -> QuantizationMethod;

    /// (vector_bytes, metadata_bytes, count)
    pub fn memory_stats(&self) -> (usize, usize, usize);

    // ========== Metadata ==========

    pub fn get_embedding_metadata(&self, node: NodeId) -> Option<&EmbeddingMetadata>;
    pub fn index_metadata(&self) -> &IndexMetadata;
    pub fn model(&self) -> Option<&str>;
    pub fn metric(&self) -> DistanceMetric;
}
```

---

## 9. Transacciones y MVCC

### 8.1 Transaction Manager

```rust
pub struct Transaction {
    id: TransactionId,
    state: TransactionState,     // Active, Committed, Aborted
    buffer: Vec<LogEntry>,       // Mutaciones pendientes
    pending_node_count: usize,
    pending_edge_count: usize,
}

pub enum TransactionState {
    Active,
    Committed,
    Aborted,
}
```

### 8.2 Flujo de Transacción

```
BEGIN
  ↓
[Transaction Created: Active]
  ↓
CREATE/DELETE/SET mutations
  ↓
[Buffered in Transaction.buffer]
  ↓
COMMIT ─────────────────────────────┐
  ↓                                 │
1. Write BeginTransaction to WAL   │
2. Write all LogEntries to WAL     │
3. Write CommitTransaction to WAL  │
4. Apply entries to GraphStore     │
5. Record timestamp in index       │
  ↓                                 │
[Transaction Complete]              │
                                    │
ROLLBACK ←──────────────────────────┘
  ↓
[Buffer discarded, no WAL write]
```

### 8.3 MVCC (Multi-Version Concurrency Control)

```rust
pub struct VersionedValue {
    /// Versiones ordenadas por TransactionId (más reciente primero)
    versions: Vec<Version>,
}

pub struct Version {
    tx_id: TransactionId,
    value: Option<PropertyValue>,  // None = tombstone (deleted)
}
```

**Regla de visibilidad:**
- Una transacción con `snapshot_id = S` ve versiones donde `tx_id <= S`
- Se selecciona la versión más reciente que cumpla la condición

**Snapshot Isolation:**
- Cada transacción obtiene un `snapshot_id` al comenzar
- Todas las lecturas ven un estado consistente de ese momento
- Lecturas no bloquean escrituras
- Escrituras no bloquean lecturas

### 8.4 Garbage Collection

```rust
impl GraphStore {
    /// Eliminar versiones obsoletas
    /// Mantiene: versiones futuras + versión estable más reciente
    pub fn vacuum(&mut self, min_active_tx_id: TransactionId);
}
```

---

## 10. Persistencia y Durabilidad

### 9.1 Formato Binario (.ngdb)

**VERSION 2 (Sprint 66 - con SHA256 checksum):**

```
┌─────────────────────────────────┐
│ Magic Bytes: "NGDB" (4 bytes)   │
├─────────────────────────────────┤
│ Version: u32 (4 bytes, LE) = 2  │
├─────────────────────────────────┤
│ SHA256: [u8; 32] (32 bytes)     │  ← NEW: Checksum de integridad
├─────────────────────────────────┤
│ Data: bincode-encoded GraphStore│
│ (variable length)               │
└─────────────────────────────────┘
```

**Backward Compatibility:** VERSION 1 (sin checksum) sigue soportado.

**Operaciones:**

```rust
impl GraphStore {
    /// Guardar a archivo binario con checksum SHA256
    pub fn save_binary_atomic(&mut self, path: &str) -> Result<()>;

    /// Cargar desde archivo binario (verifica checksum si V2)
    pub fn load_binary(path: &str) -> Result<Self>;

    /// Validar integridad después de cargar
    pub fn validate_post_load(&self) -> ValidationResult;
}
```

### 9.2 Write-Ahead Log (WAL)

**Formato de entrada (Sprint 66 - con CRC32 checksum):**

```
┌─────────────────────────────────┐
│ Length: u64 (8 bytes, LE)       │  ← Incluye checksum + payload
├─────────────────────────────────┤
│ CRC32: u32 (4 bytes, LE)        │  ← NEW: Checksum de integridad
├─────────────────────────────────┤
│ Payload: bincode LogEntry       │
│ (length - 4 bytes)              │
└─────────────────────────────────┘
```

**Backward Compatibility:** Entradas legacy (sin checksum) se detectan y procesan sin verificación.

**Tipos de LogEntry:**

```rust
pub enum LogEntry {
    // Transacciones
    BeginTransaction { tx_id: TransactionId },
    CommitTransaction { tx_id: TransactionId, timestamp: String },
    RollbackTransaction { tx_id: TransactionId },

    // Mutaciones
    CreateNode { node_id: NodeId, label: Option<Label>, properties: HashMap<String, PropertyValue> },
    CreateEdge { source: NodeId, target: NodeId, edge_type: Option<Label> },
    SetProperty { node_id: NodeId, key: String, value: PropertyValue },
    DeleteNode { node_id: NodeId },

    // Raft
    Blank,                           // Leader commit marker
    Membership { config_bytes: Vec<u8> }, // Cluster changes
}
```

**Recuperación:**

```rust
impl GraphStore {
    /// Recuperar estado desde WAL al iniciar
    pub fn recover_from_wal(&mut self) -> Result<()>;
}
```

### 9.3 Delta Checkpoints (Sprint 66)

Persistencia incremental que guarda solo los cambios desde el último snapshot completo:

```rust
#[derive(Serialize, Deserialize)]
pub struct DeltaCheckpoint {
    pub base_tx_id: TransactionId,    // TX del snapshot base
    pub end_tx_id: TransactionId,     // TX después de aplicar delta
    pub changes: Vec<LogEntry>,       // Cambios a aplicar
}

impl GraphStore {
    /// Crear delta desde una transacción específica
    pub fn create_delta_checkpoint(&self, since_tx: TransactionId) -> DeltaCheckpoint;

    /// Aplicar delta a un snapshot cargado
    pub fn apply_delta(&mut self, delta: &DeltaCheckpoint) -> Result<()>;

    /// Guardar delta a directorio
    pub fn save_delta(&self, base_path: &Path, since_tx: TransactionId) -> Result<PathBuf>;
}
```

**Estructura de archivos:**

```
data/
├── graph.ngdb                    # Snapshot completo
└── deltas/
    ├── delta_1000_2000.ngdb      # Cambios de tx 1000 a 2000
    └── delta_2000_3000.ngdb      # Cambios de tx 2000 a 3000
```

### 9.4 Garantías ACID

| Propiedad | Implementación |
|-----------|----------------|
| **Atomicity** | Buffer de transacciones; commit aplica todo o nada |
| **Consistency** | WAL write con CRC32 antes de modificar memoria |
| **Isolation** | MVCC + Snapshot Isolation |
| **Durability** | Flush explícito + fsync + SHA256 checksum en snapshots |

### 9.5 Integridad de Datos (Sprint 66)

| Feature | Implementación |
|---------|----------------|
| **WAL Checksums** | CRC32 en cada entrada de WAL |
| **Snapshot Checksums** | SHA256 en archivos .ngdb |
| **Index Rebuild** | Reconstrucción de índices al cargar |
| **Post-Load Validation** | Verificación de integridad post-carga |
| **Delta Persistence** | Persistencia incremental eficiente |

### 9.6 Configuración y Observabilidad (Sprint 66)

#### Configuración Unificada

```toml
# neuralgraph.toml
[storage]
path = "data/graph.ngdb"

[persistence]
save_interval_secs = 60
mutation_threshold = 100
backup_count = 3
checksum_enabled = true

[memory]
limit_mb = 4096
warn_percent = 80

[logging]
level = "info"
format = "pretty"
```

**Variables de entorno:** Todas las opciones se pueden sobrescribir con `NGDB__*`:

```bash
export NGDB__PERSISTENCE__CHECKSUM_ENABLED=true
export NGDB__MEMORY__LIMIT_MB=8192
export NGDB_LOG=debug
```

#### Logging Estructurado

```rust
use neural_storage::logging;

// Inicializar con configuración de entorno
logging::init();  // Lee NGDB_LOG

// O con nivel específico
logging::init_with_default("info");

// Formato JSON para producción
logging::init_json();
```

#### Estadísticas del Grafo

```rust
#[derive(Serialize, Deserialize)]
pub struct GraphStatistics {
    pub node_count: usize,
    pub edge_count: usize,
    pub label_cardinalities: HashMap<String, usize>,
    pub property_cardinalities: HashMap<String, usize>,
    pub edge_type_cardinalities: HashMap<String, usize>,
    pub avg_out_degree: f64,
    pub max_out_degree: usize,
    pub last_updated: Option<String>,
}

impl GraphStore {
    pub fn collect_statistics(&mut self);
    pub fn estimate_label_cardinality(&self, label: &str) -> usize;
}
```

### 9.7 Sistema de Constraints (Sprint 66)

```rust
pub enum ConstraintType {
    Unique { property: String, label: Option<String> },
}

impl ConstraintManager {
    /// Crear constraint de unicidad
    pub fn create_unique(&mut self, name: &str, property: &str, label: Option<&str>)
        -> Result<(), ConstraintError>;

    /// Validar antes de insertar
    pub fn validate_insert(&self, node_id: NodeId, label: Option<&str>,
        props: &[(String, PropertyValue)]) -> Result<(), ConstraintError>;
}
```

**Uso en NGQL:**

```cypher
-- Crear constraint de unicidad
CALL neural.constraint.createUnique('person_email', 'email', 'Person')

-- Listar constraints
CALL neural.constraint.list()

-- Eliminar constraint
CALL neural.constraint.drop('person_email')
```

---

## 11. Sistema Distribuido (Raft + Sharding)

### 10.1 Raft Consensus

**Ubicación:** `crates/neural-storage/src/raft/`

**Componentes:**

| Módulo | Responsabilidad |
|--------|-----------------|
| `types.rs` | RaftNodeId, RaftRequest, ClusterConfig |
| `log_store.rs` | Log persistente con WAL |
| `state_machine.rs` | Aplica entradas al GraphStore |
| `network.rs` | Transporte gRPC (Tonic) |
| `cluster.rs` | Gestión de membresía |
| `health.rs` | Monitoreo de nodos |
| `wrapper.rs` | Integración con OpenRaft |

**Configuración del cluster:**

```rust
pub struct ClusterConfig {
    pub node_id: u64,
    pub listen_addr: String,      // "0.0.0.0:50052"
    pub raft_port: u16,
    pub peers: Vec<String>,       // ["127.0.0.1:50053", ...]
}

// Timeouts
const ELECTION_TIMEOUT: Duration = Duration::from_millis(150..300);
const HEARTBEAT_INTERVAL: Duration = Duration::from_millis(50);
```

**Protocolo gRPC (proto/raft.proto):**

```protobuf
service Raft {
    // Replicación de log
    rpc AppendEntries(AppendEntriesRequest) returns (AppendEntriesResponse);

    // Transferencia de snapshot
    rpc InstallSnapshot(InstallSnapshotRequest) returns (InstallSnapshotResponse);

    // Elección de líder
    rpc Vote(VoteRequest) returns (VoteResponse);

    // Gestión de cluster
    rpc Join(JoinRequest) returns (JoinResponse);
    rpc GetClusterInfo(ClusterInfoRequest) returns (ClusterInfoResponse);
    rpc HealthCheck(HealthCheckRequest) returns (HealthCheckResponse);

    // Client mutations (Sprint 53)
    rpc ClientRequest(ClientRequestMessage) returns (ClientResponseMessage);
}
```

**ClusterAwareClient (Sprint 53):**

El cliente cluster-aware enruta automáticamente las mutaciones al líder Raft:

```rust
pub struct ClusterAwareClient {
    cluster: Arc<ClusterManager>,
    clients: Arc<RwLock<BTreeMap<u64, RaftClient>>>,
    max_retries: usize,
}

impl ClusterAwareClient {
    /// Submit mutation to the Raft leader
    pub async fn submit(&self, request: &RaftRequest) -> Result<RaftResponse, ClientError>;

    /// Get cluster information
    pub async fn get_cluster_info(&self) -> Result<ClusterInfoResponse, ClientError>;

    /// Check health of a specific node
    pub async fn health_check(&self, node_id: u64, addr: &str) -> Result<HealthCheckResponse, ClientError>;
}
```

### 10.2 Sharding

**Ubicación:** `crates/neural-storage/src/sharding/`

**Estrategias de particionamiento:**

```rust
pub enum PartitionStrategy {
    /// hash(node_id) % num_shards
    Hash { num_shards: u32 },

    /// Binary search en boundaries ordenados
    Range { boundaries: Vec<NodeId> },

    /// Leiden communities + bin-packing
    Community { graph_edges: Vec<(NodeId, NodeId)> },
}
```

| Estrategia | Algoritmo | Localidad | Uso |
|------------|-----------|-----------|-----|
| **Hash** | `hash(id) % n` | Ninguna | Distribución uniforme |
| **Range** | Binary search | Excelente para rangos | Queries de rango |
| **Community** | Leiden + packing | Minimiza edge-cut | Traversals frecuentes |

**ShardInfo:**

```rust
pub struct ShardInfo {
    pub id: ShardId,
    pub primary_addr: String,
    pub replica_addrs: Vec<String>,
    pub estimated_node_count: u64,
    pub estimated_edge_count: u64,
    pub online: bool,
}
```

**Query Router:**

```rust
impl QueryRouter {
    pub fn plan_node_lookup(&self, node_id: NodeId) -> QueryPlan;
    pub fn plan_multi_node_lookup(&self, node_ids: &[NodeId]) -> QueryPlan;
    pub fn plan_label_scan(&self, label: &Label) -> QueryPlan;
    pub fn plan_full_scan(&self) -> QueryPlan;
    pub fn plan_edge_query(&self, src: NodeId, tgt: NodeId) -> QueryPlan;
    pub fn plan_vector_search(&self, k: usize) -> QueryPlan;
}
```

### 10.3 Búsqueda Vectorial Distribuida

**Configuración:**

```rust
pub struct DistributedVectorConfig {
    pub num_shards: u32,               // 4 default
    pub cache_size: usize,             // 10,000 entries
    pub cache_ttl: Duration,           // 300 seconds
    pub timeout: Duration,             // 100ms per shard
    pub max_concurrent_shards: usize,  // 8
    pub oversampling_factor: f32,      // 1.5x
    pub metric: DistanceMetric,
    pub dimension: usize,
}
```

**Algoritmo Scatter-Gather:**

```
1. Check Cache (SimHash key)
   ↓
2. Calculate per_shard_k = k * oversampling_factor
   ↓
3. Scatter: Send query to all shards (parallel async)
   ↓
4. Gather: Collect top-k from each shard
   ↓
5. Merge: Min-heap merge O(m * log k)
   ↓
6. Cache: Store result with TTL
   ↓
7. Return: Global top-k
```

**Load Balancing:**

```rust
pub trait LoadBalancer: Send + Sync {
    fn select_replica(&self, shard_id: ShardId, replicas: &[String]) -> String;
    fn record_latency(&self, shard_id: ShardId, replica: &str, latency: Duration);
    fn mark_unhealthy(&self, shard_id: ShardId, replica: &str);
    fn mark_healthy(&self, shard_id: ShardId, replica: &str);
}

// Implementaciones disponibles
pub struct RoundRobinBalancer { ... }    // Rotación simple
pub struct LatencyAwareBalancer { ... }  // EMA de latencia
pub struct WeightedBalancer { ... }      // Pesos manuales
```

**Vector Service gRPC (proto/vector.proto):**

```protobuf
service VectorService {
    rpc Search(VectorSearchRequest) returns (VectorSearchResponse);
    rpc Add(VectorAddRequest) returns (VectorAddResponse);
    rpc BatchAdd(VectorBatchAddRequest) returns (VectorBatchAddResponse);
    rpc GetStats(VectorStatsRequest) returns (VectorStatsResponse);
    rpc HealthCheck(VectorHealthCheckRequest) returns (VectorHealthCheckResponse);
}

message VectorSearchRequest {
    bytes query_vector = 1;   // f32[] serialized
    uint32 k = 2;
    uint32 ef_search = 3;
    string metric = 4;        // "cosine", "euclidean", "dot_product"
}

message VectorSearchResponse {
    repeated VectorResult results = 1;
    uint64 execution_time_us = 2;
    uint64 index_size = 3;
}
```

---

## 12. GraphRAG y ETL

### 11.1 Pipeline ETL

```rust
pub struct EtlPipeline {
    llm_client: LlmClient,
    chunking_config: ChunkingConfig,
}

impl EtlPipeline {
    /// Procesar documento completo
    /// Flujo: PDF → Chunks → LLM → Entidades/Relaciones → Grafo
    pub async fn process_document(&self, path: &str) -> Result<()>;

    /// Extraer entidades de texto
    pub async fn extract_entities(&self, text: &str) -> Result<Vec<Entity>>;

    /// Extraer relaciones entre entidades
    pub async fn extract_relations(&self, text: &str, entities: &[Entity])
        -> Result<Vec<Relation>>;
}
```

### 11.2 Ingesta de PDFs

```rust
/// Cargar PDF desde archivo
pub fn load_pdf(path: &str) -> Result<String>;

/// Cargar PDF desde bytes
pub fn load_pdf_bytes(bytes: &[u8]) -> Result<String>;
```

### 11.3 Clientes LLM

```rust
pub struct LlmClient {
    provider: LlmProvider,
    api_key: Option<String>,
    model: String,
}

pub enum LlmProvider {
    OpenAI,    // GPT-4, GPT-3.5, embeddings
    Ollama,    // Modelos locales
    Gemini,    // Google AI
}

impl LlmClient {
    /// Crear cliente OpenAI
    pub fn openai(api_key: &str) -> Result<Self>;

    /// Crear cliente Ollama (local)
    pub fn ollama(host: &str) -> Result<Self>;

    /// Crear cliente Gemini
    pub fn gemini(api_key: &str) -> Result<Self>;

    /// Chat completion
    pub async fn complete(&self, prompt: &str) -> Result<String>;

    /// Generar embedding
    pub async fn embed(&self, text: &str) -> Result<Vec<f32>>;
}
```

### 11.4 Detección de Comunidades

**Algoritmo Leiden:**

```rust
/// Detectar comunidades usando Leiden
pub fn detect_communities_leiden(
    edges: &[(usize, usize)],
    num_nodes: usize
) -> Communities;

pub struct Communities {
    /// Mapeo: node_index -> community_id
    assignments: Vec<CommunityId>,

    /// Número total de comunidades
    num_communities: usize,

    /// Índice inverso: community -> nodes
    community_members: HashMap<CommunityId, Vec<usize>>,
}

impl Communities {
    pub fn get_community(&self, node: usize) -> CommunityId;
    pub fn get_members(&self, community: CommunityId) -> &[usize];
    pub fn num_communities(&self) -> usize;
}
```

**Uso en NGQL:**

```cypher
-- Obtener comunidad de un nodo
MATCH (n:Document)
RETURN n.title, CLUSTER(n) AS community
ORDER BY community

-- Agrupar por comunidad
MATCH (n:Document)
RETURN CLUSTER(n) AS community, COUNT(*) AS size, COLLECT(n.title) AS documents
GROUP BY community
ORDER BY size DESC
```

---

## 13. APIs e Interfaces

### 12.1 HTTP REST API (Axum)

**Puerto por defecto:** 3000

| Endpoint | Método | Descripción | Body |
|----------|--------|-------------|------|
| `/` | GET | Interfaz web HTML | - |
| `/api/query` | POST | Ejecutar NGQL | `{"query": "..."}` |
| `/api/papers` | GET | Listar nodos | - |
| `/api/search` | POST | Búsqueda texto | `{"query": "...", "limit": 10}` |
| `/api/similar/{id}` | GET | Nodos similares | - |
| `/api/bulk-load` | POST | Carga masiva CSV | `{"nodes_path": "...", "edges_path": "..."}` |
| `/api/schema` | GET | Schema del grafo (Sprint 65) | - |

#### Autenticación (Sprint 68)

NeuralGraphDB soporta autenticación opcional mediante JWT tokens o API keys.

**Modelo de Seguridad:**

| Endpoint | Auth Requerida | Roles Permitidos |
|----------|----------------|------------------|
| `/health`, `/metrics`, `/` | No | Público |
| `/api/papers`, `/api/search`, `/api/similar/{id}`, `/api/schema` | Sí* | admin, user, readonly |
| `/api/query` (lectura) | Sí* | admin, user, readonly |
| `/api/query` (mutación) | Sí* | admin, user |
| `/api/bulk-load` | Sí* | admin solamente |

*Cuando la autenticación está habilitada en configuración.

**Métodos de Autenticación:**

```bash
# JWT Token
curl -X POST http://localhost:3000/api/query \
  -H "Authorization: Bearer <jwt-token>" \
  -H "Content-Type: application/json" \
  -d '{"query": "MATCH (n) RETURN n"}'

# API Key
curl -X POST http://localhost:3000/api/query \
  -H "X-API-Key: <your-api-key>" \
  -H "Content-Type: application/json" \
  -d '{"query": "MATCH (n) RETURN n"}'
```

**Configuración:**

```bash
export NGDB__AUTH__ENABLED=true
export NGDB__AUTH__JWT_SECRET="your-32-byte-secret-key-here!!!"
export NGDB__AUTH__JWT_EXPIRATION_SECS=3600
```

**Ejemplo - Ejecutar Query:**

```bash
curl -X POST http://localhost:3000/api/query \
  -H "Content-Type: application/json" \
  -d '{"query": "MATCH (n:Person) RETURN n.name LIMIT 10"}'
```

**Respuesta:**

```json
{
  "success": true,
  "result": {
    "columns": ["name"],
    "rows": [
      {"name": "Alice"},
      {"name": "Bob"}
    ],
    "count": 2
  },
  "execution_time_ms": 1.23
}
```

### 12.2 Arrow Flight (gRPC)

**Puerto por defecto:** 50051

Transferencia de datos columnar de alto rendimiento (zero-copy).

**Python:**

```python
import pyarrow.flight as flight

# Conectar
client = flight.connect("grpc://localhost:50051")

# Ejecutar query y obtener resultados como Arrow Table
reader = client.do_get(flight.Ticket(b"MATCH (n) RETURN n.name, n.age"))
table = reader.read_all()

# Convertir a pandas
df = table.to_pandas()
```

**Schemas disponibles:**

- Nodes: `id (uint64), label (string), properties_json (string)`
- Edges: `source (uint64), target (uint64), type (string)`
- Query results: Dinámico según columnas del RETURN

### 12.3 Raft Cluster (gRPC)

**RPCs disponibles:**

| RPC | Descripción |
|-----|-------------|
| `AppendEntries` | Replicación de log |
| `InstallSnapshot` | Transferencia de estado completo |
| `Vote` | Elección de líder |
| `Join` | Unirse al cluster |
| `GetClusterInfo` | Estado actual del cluster |
| `HealthCheck` | Verificación de salud |

### 12.4 Vector Service (gRPC)

**RPCs disponibles:**

| RPC | Request | Response |
|-----|---------|----------|
| `Search` | query_vector, k, ef_search, metric | results[], execution_time |
| `Add` | node_id, vector | success, error |
| `BatchAdd` | entries[] | added_count, failed_count |
| `GetStats` | - | vector_count, dimension, memory |
| `HealthCheck` | shard_id | healthy, status |

### 13.5 Python LangChain Integration (Sprint 65)

**Instalación:**

```bash
pip install neuralgraph[langchain]
```

**NeuralGraphStore - Graph Store compatible con LangChain:**

```python
from neuralgraph import NeuralGraphStore

# Conectar a NeuralGraphDB
graph = NeuralGraphStore(host="localhost", port=3000)

# Obtener schema del grafo
print(graph.get_schema())
# Node labels: Person, Company
# Relationship types: WORKS_AT, KNOWS

# Ejecutar queries NGQL
results = graph.query("MATCH (n:Person) RETURN n.name, n.age")
# [{'name': 'Alice', 'age': 30}, {'name': 'Bob', 'age': 25}]

# Métodos de conveniencia
node_id = graph.add_node("Person", {"name": "Charlie", "age": 35})
graph.add_edge(node_id, other_id, "KNOWS", {"since": 2024})
```

**GraphCypherQAChain - Natural Language Queries:**

```python
from neuralgraph import NeuralGraphStore, create_qa_chain
from langchain_openai import ChatOpenAI

# Configurar
graph = NeuralGraphStore()
llm = ChatOpenAI(model="gpt-4", temperature=0)

# Crear chain de Q&A
chain = create_qa_chain(llm, graph, verbose=True)

# Consultar en lenguaje natural
result = chain.invoke({"query": "Who works at TechCorp?"})
print(result["result"])
```

**NeuralGraphQAChain - Chain Simplificado:**

```python
from neuralgraph import NeuralGraphStore, NeuralGraphQAChain

graph = NeuralGraphStore()
chain = NeuralGraphQAChain(graph, llm, verbose=True)

# Múltiples interfaces
answer = chain.run("How many people are in the database?")
answer = chain("Find all engineers")
result = chain.invoke({"query": "Who knows Alice?"})
```

---

## 14. CLI y Comandos

### 13.1 Modos de Ejecución

```bash
# REPL interactivo (por defecto)
neuralgraph

# Demo con datos de ejemplo
neuralgraph --demo

# Benchmark con ArXiv dataset
neuralgraph --benchmark

# Servidor HTTP REST
neuralgraph serve [PORT]              # default: 3000

# Servidor Arrow Flight
neuralgraph serve-flight [PORT]       # default: 50051

# Nodo Raft
neuralgraph serve-raft <NODE_ID> <PORT> [--join <SEED_ADDR>]

# Gestión de cluster
neuralgraph cluster info <ADDR>
neuralgraph cluster health <ADDR>
neuralgraph cluster add <ADDR> <NODE_ID> <NODE_ADDR>
neuralgraph cluster remove <ADDR> <NODE_ID>
```

### 13.2 Comandos REPL

| Comando | Descripción |
|---------|-------------|
| `:help`, `:h`, `:?` | Mostrar ayuda |
| `:stats` | Estadísticas del grafo (nodos, aristas, memoria) |
| `:load nodes <file>` | Cargar nodos desde CSV |
| `:load edges <file>` | Cargar aristas desde CSV |
| `:load hf <dataset>` | Cargar desde HuggingFace |
| `:save <file.ngdb>` | Guardar a formato binario |
| `:loadbin <file.ngdb>` | Cargar desde formato binario |
| `:clear` | Limpiar todos los datos |
| `:demo` | Cargar grafo de demostración |
| `:benchmark` | Ejecutar benchmarks |
| `:cluster` | Ejecutar detección de comunidades |
| `:param` | Listar parámetros de sesión |
| `:param <name> <value>` | Establecer parámetro |
| `:quit`, `:exit`, `:q` | Salir |

### 13.3 Formato CSV

**Nodos (nodes.csv):**

```csv
id,label,name,age,embedding
0,Person,Alice,30,"[0.1, 0.2, 0.3, 0.4]"
1,Person,Bob,25,"[0.2, 0.3, 0.4, 0.5]"
2,Company,Acme,,"[]"
```

**Aristas (edges.csv):**

```csv
source,target,label
0,1,KNOWS
0,2,WORKS_AT
1,2,WORKS_AT
```

### 13.4 Datasets HuggingFace

```bash
# Datasets soportados
:load hf ml-arxiv-papers    # ~200K papers con citas
:load hf arxiv              # Alias
:load hf cshorten/ml-arxiv-papers  # Nombre completo
```

### 13.5 Parámetros de Sesión

```bash
# Listar parámetros
:param

# Establecer parámetros
:param name "Alice"           # String
:param age 30                 # Int
:param score 0.95             # Float
:param active true            # Bool
:param data null              # Null

# Usar en queries
MATCH (n:Person) WHERE n.name = $name RETURN n
```

---

## 15. Rendimiento y Benchmarks

### 14.1 Métricas de Rendimiento Verificadas

| Operación | Target | Actual | Estado |
|-----------|--------|--------|--------|
| Node lookup por ID | <100ns | **22ns** | ✅ |
| Filtro por label | <50ms | **<1ms** | ✅ |
| COUNT(*) | <50ms | **<1ms** | ✅ |
| Vector Search (100k) | <50ms | **~10ms** | ✅ |
| Streaming | O(1) mem | ✅ | ✅ |

### 14.2 LDBC-SNB Benchmark (Sprint 59)

**Configuración: SF1 (27K nodos, 192K aristas)**

| Query | Categoría | p50 (ms) | p95 (ms) | p99 (ms) |
|-------|-----------|----------|----------|----------|
| IS1 | Interactive Short | 0.34 | 0.38 | 0.40 |
| IS2 | Interactive Short | 0.36 | 0.41 | 0.43 |
| IS3 | Interactive Short | 0.38 | 0.44 | 0.46 |
| IS4 | Interactive Short | 0.35 | 0.40 | 0.42 |
| IS5 | Interactive Short | 0.40 | 0.48 | 0.51 |
| IS6 | Interactive Short | 0.42 | 0.51 | 0.54 |
| IS7 | Interactive Short | 0.43 | 0.53 | 0.56 |
| IC1 | Interactive Complex | 0.34 | 0.36 | 0.38 |
| IC2 | Interactive Complex | 0.38 | 0.42 | 0.45 |
| IC3 | Interactive Complex | 0.41 | 0.47 | 0.50 |
| IC4 | Interactive Complex | 0.35 | 0.39 | 0.41 |
| IC5 | Interactive Complex | 0.40 | 0.46 | 0.49 |
| IC6 | Interactive Complex | 0.42 | 0.49 | 0.53 |
| IC7 | Interactive Complex | 0.44 | 0.51 | 0.56 |

**Mejoras Sprint 59:**

| Métrica | Antes | Después | Mejora |
|---------|-------|---------|--------|
| Query p50 (promedio) | 0.72ms | 0.35ms | **51% más rápido** |
| Tiempo de ingesta | 0.64s | 0.40s | **38% más rápido** |

### 14.3 Eficiencia de Memoria

| Escala | NeuralGraphDB | FalkorDB | Eficiencia |
|--------|---------------|----------|------------|
| SF1 (27K nodos) | 25.3 MB | 618.8 MB | **25x menos** |
| SF100 (2.1M nodos) | 42.3 MB | ~2 GB | **~50x menos** |

### 14.4 SLAs Objetivo

| Tipo de Consulta | Latencia p95 Target |
|------------------|---------------------|
| Simple (2-hop) | <10ms |
| Híbrida (Vector + 2-hop) | <50ms |
| Agéntica compleja | <140ms |

### 14.5 Capacidad

- **Vertical**: Hasta 100M nodos, 1B aristas en una máquina
- **Concurrencia**: >1000 QPS de lecturas
- **Memoria**: Ejecución streaming evita OOM

---

## 16. Historial de Sprints

| Sprint | Versión | Funcionalidades |
|--------|---------|-----------------|
| 1-12 | 0.4.0 | Core: CSR, Parser NGQL, Agregaciones, Índices |
| 13 | 0.5.0-beta | HNSW Vector Index |
| 14 | 0.5.0-beta | Vector ORDER BY |
| 15 | 0.5.0-beta | Leiden Community Detection |
| 16 | 0.5.0-beta | CLUSTER BY keyword |
| 17 | 0.5.0-beta | PDF Ingestion |
| 18 | 0.5.0-beta | LLM Client (OpenAI, Ollama, Gemini) |
| 19 | 0.5.0-beta | Auto-ETL Pipeline |
| 21-24 | 0.9.0 | CRUD: CREATE, DELETE, SET |
| 25-26 | 0.9.0 | Persistencia: bincode + WAL |
| 27-28 | 0.9.0 | Variable-length paths, Shortest path |
| 29 | 0.9.0 | EXPLAIN, PROFILE |
| 30 | 0.9.0 | Parameterized queries ($param) |
| 31 | 0.9.0 | Streaming execution |
| 32-49 | 0.9.0 | Query pipelining, MERGE, CASE, temporal engine |
| 50 | 0.9.2 | Transaction Manager (ACID) |
| 51 | 0.9.1 | type() function, keyword flexibility |
| 52 | 0.9.2 | Raft Consensus (multi-node replication) |
| 53 | 0.9.5 | Cluster Management (leader routing, health checks, metrics) |
| 54 | 0.9.2 | Time-Travel (AT TIME, FLASHBACK) |
| 55 | 0.9.2 | Shard hints |
| 56 | 0.9.2 | Embedding metadata |
| 57 | 0.9.2 | Port numbering (multi-edges) |
| 58+ | 0.9.2 | Graph sharding |
| 59 | 0.9.5 | Query latency optimization (51% improvement) |
| 60 | 0.9.5 | Flash quantization (4-32x memory), distributed vector search |
| 61 | 0.9.5 | Distributed vector gRPC server, shard coordinator enhancements |
| 62 | 0.9.6 | Full-text search index (tantivy): stemming, stop words, phrase/boolean queries |
| 63 | 0.9.7 | Full-text search avanzado: fuzzy matching (Levenshtein), phonetic search (Soundex, Metaphone, DoubleMetaphone), 18 idiomas |
| 64 | 0.9.8 | Array/Map data types: PropertyValue::Array, PropertyValue::Map, nested structures, JSON-compatible serialization |
| 65 | 0.9.9 | LangChain Integration: NeuralGraphStore class, GraphCypherQAChain adapter, /api/schema endpoint, Python client chains module |
| 66 | 0.9.9 | Database Hardening: WAL CRC32 checksums, SHA256 snapshot checksums, delta checkpoints, unified TOML config, memory tracking, unique constraints |
| 67 | 0.9.10 | Production Observability: /health endpoint, /metrics Prometheus endpoint, query latency instrumentation, structured logging, Docker healthcheck |
| 68 | 0.9.11 | Security & Authentication: JWT token validation, API key auth, role-based access control (Admin/User/Readonly), audit logging, constant-time key comparison |

---

## Apéndice A: Glosario Técnico

| Término | Definición |
|---------|------------|
| **CSR** | Compressed Sparse Row - Estructura que comprime matrices dispersas eliminando ceros |
| **CSC** | Compressed Sparse Column - Transposición de CSR para acceso por columnas |
| **GraphBLAS** | Estándar de API para primitivas de grafos basadas en álgebra lineal |
| **HNSW** | Hierarchical Navigable Small World - Algoritmo de búsqueda aproximada de vecinos más cercanos |
| **MVCC** | Multi-Version Concurrency Control - Técnica que mantiene múltiples versiones de datos |
| **WAL** | Write-Ahead Log - Registro secuencial de cambios para durabilidad |
| **Snapshot Isolation** | Nivel de aislamiento donde cada transacción ve una instantánea consistente |
| **Raft** | Algoritmo de consenso distribuido para replicación y tolerancia a fallos |
| **Sharding** | Particionamiento horizontal del grafo para escala distribuida |
| **Leiden** | Algoritmo de detección de comunidades (mejora de Louvain) |
| **GraphRAG** | Graph + Retrieval-Augmented Generation para LLMs |
| **Port Numbering** | Identificadores únicos para multi-aristas paralelas |
| **Quantization** | Reducción de precisión de vectores para ahorro de memoria |

---

## Apéndice B: Archivos Importantes

```
# Configuración
/Cargo.toml                     # Workspace configuration
/Cargo.lock                     # Dependency lock

# Protocolos
/proto/raft.proto               # Raft consensus RPC
/proto/vector.proto             # Distributed vector search RPC

# Código fuente principal
/crates/neural-core/src/lib.rs              # Core types
/crates/neural-storage/src/graph_store.rs   # Main storage
/crates/neural-storage/src/vector_index/core.rs  # HNSW index
/crates/neural-parser/src/parser.rs         # NGQL parser
/crates/neural-executor/src/executor.rs     # Query executor
/crates/neural-cli/src/main.rs              # CLI entry point

# Documentación
/docs/neural_graph_funcional.md             # Functional spec
/docs/distributed_vector_search.md          # Vector sharding
/docs/consolidated_sprint_reports.md        # Sprint history
```

---

*Documento generado: 2026-02-04*
*Versión de NeuralGraphDB: 0.9.11*
*Sprints completados: 1-68*
