# NeuralGraphDB Quick Reference (v0.9.11)

## Shell Commands

```bash
:help                    # Show help
:stats                   # Database statistics
:demo                    # Load sample data
:save <file.ngdb>        # Save database
:loadbin <file.ngdb>     # Load database
:load nodes <file.csv>   # Import nodes from CSV
:load edges <file.csv>   # Import edges from CSV
:param <name> <value>    # Set query parameter
:quit                    # Exit
```

---

## NGQL Query Language

### Reading Data

```cypher
-- Find nodes
MATCH (n:Person) RETURN n
MATCH (n:Person) WHERE n.name = "Alice" RETURN n

-- Find patterns
MATCH (a:Person)-[:KNOWS]->(b:Person) RETURN a, b
MATCH (a)-[r]->(b) RETURN a, type(r), b

-- Filter
MATCH (n:Person) WHERE n.age > 30 RETURN n
MATCH (n) WHERE n.name STARTS WITH "A" RETURN n
MATCH (n) WHERE n.status IN ["active", "pending"] RETURN n

-- Sort and limit
MATCH (n:Person) RETURN n ORDER BY n.age DESC LIMIT 10
MATCH (n:Person) RETURN n SKIP 20 LIMIT 10

-- Aggregate
MATCH (n:Person) RETURN COUNT(*)
MATCH (n:Person) RETURN n.country, COUNT(*) GROUP BY n.country
MATCH (n:Order) RETURN SUM(n.total), AVG(n.total)

-- Path queries
MATCH (a)-[*1..3]->(b) RETURN a, b                    -- Variable length
MATCH p = shortestPath((a)-[*]->(b)) RETURN p        -- Shortest path
```

### Writing Data

```cypher
-- Create
CREATE (n:Person {name: "Alice", age: 30})
CREATE (a)-[:KNOWS {since: 2020}]->(b)

-- Arrays and Maps (Sprint 64)
CREATE (n:Person {tags: ["dev", "rust"], config: {debug: true, level: 5}})

-- Update
MATCH (n:Person) WHERE n.name = "Alice" SET n.age = 31

-- Delete
MATCH (n:Person) WHERE n.name = "Alice" DELETE n            -- Node only
MATCH (n:Person) WHERE n.name = "Alice" DETACH DELETE n     -- Node + relationships
MATCH (a)-[r:KNOWS]->(b) DELETE r                    -- Relationship

-- Upsert
MERGE (n:Person {email: "alice@example.com"})
ON CREATE SET n.created = datetime()
ON MATCH SET n.updated = datetime()
```

### Transactions

```cypher
BEGIN
  CREATE (a:Account {balance: 1000})
  CREATE (b:Account {balance: 500})
COMMIT

BEGIN
  -- changes...
ROLLBACK   -- Undo all changes
```

---

## Full-Text Search

```cypher
-- Create index on properties
CALL neural.fulltext.createIndex('search_idx', 'Paper', ['title', 'abstract'])

-- Create index with language (18 supported)
CALL neural.fulltext.createIndex('es_idx', 'Article', ['titulo'], 'spanish')

-- Create index with phonetic search
CALL neural.fulltext.createIndex('name_idx', 'Person', ['name'], 'english', 'soundex')

-- Search with relevance ranking
CALL neural.fulltext.query('search_idx', 'machine learning', 10)
YIELD node, score
RETURN node.title, score

-- Phrase search
CALL neural.fulltext.query('search_idx', '"neural network"', 5)

-- Boolean search (AND, OR, NOT/-)
CALL neural.fulltext.query('search_idx', 'deep AND learning -CNN', 10)

-- Fuzzy search (typo tolerance)
CALL neural.fulltext.fuzzyQuery('search_idx', 'machin lerning', 10)      -- distance=1
CALL neural.fulltext.fuzzyQuery('search_idx', 'machin lerning', 10, 2)   -- distance=2

-- List/drop indexes
CALL neural.fulltext.indexes()
CALL neural.fulltext.dropIndex('search_idx')
```

### Languages
English, Spanish, French, German, Italian, Portuguese, Dutch, Swedish, Norwegian, Danish, Finnish, Russian, Hungarian, Romanian, Turkish, Arabic, Greek, Tamil

### Phonetic Algorithms
`soundex`, `metaphone`, `double_metaphone`

---

## Vector Search

```cypher
-- Initialize vector index (required before storing vectors)
CALL neural.vectorInit(768)       -- dimension = 768 (e.g., for BERT embeddings)
CALL neural.vectorInit(3584)      -- dimension = 3584 (e.g., for Qwen3-Embedding-8B)

-- Create nodes with vector properties (auto-indexed)
CREATE (n:Document {title: "ML Paper", embedding: [0.1, 0.2, ...]})

-- Update vector property (auto-indexed)
MATCH (n:Document) WHERE n.title = "ML Paper"
SET n.embedding = [0.3, 0.4, ...]

-- Find similar items
CALL neural.search($embedding, 'cosine', 10)
YIELD node, score
RETURN node.title, score

-- Metrics: 'cosine', 'euclidean', 'dot_product', 'l2'
```

**Note:** Vector index must be initialized before storing vector properties. Vectors are automatically indexed when written via CREATE or SET.

### Quantization (Memory Savings)
| Method | Savings | Precision |
|--------|---------|-----------|
| None   | 0%      | 100%      |
| Int8   | 75% (4x)| ~99%      |
| Binary | 97% (32x)| ~90%     |

---

## Graph Analytics

```cypher
-- Community detection
MATCH (n:Person) RETURN n.name, CLUSTER(n) AS community

-- Degree centrality
MATCH (n:Person)-[r]-() RETURN n.name, COUNT(r) AS connections
ORDER BY connections DESC
```

---

## Functions

### Scalar
| Function | Example |
|----------|---------|
| `id(n)` | Node ID |
| `type(r)` | Relationship type |
| `length(p)` | Path length |
| `coalesce(a, b)` | First non-null |

### Aggregation
| Function | Description |
|----------|-------------|
| `COUNT(*)` | Count rows |
| `SUM(n.x)` | Sum values |
| `AVG(n.x)` | Average |
| `MIN(n.x)` | Minimum |
| `MAX(n.x)` | Maximum |
| `COLLECT(n.x)` | Collect into list |

### String
| Function | Example |
|----------|---------|
| `toUpper(s)` | `"HELLO"` |
| `toLower(s)` | `"hello"` |
| `trim(s)` | Remove whitespace |
| `substring(s, start, len)` | Extract part |

---

## Operators

### Comparison
```
=    Equal
<>   Not equal
<    Less than
>    Greater than
<=   Less or equal
>=   Greater or equal
```

### Logical
```
AND    Both true
OR     Either true
NOT    Negate
```

### String
```
STARTS WITH    Prefix match
ENDS WITH      Suffix match
CONTAINS       Substring match
IN             List membership
```

---

## CSV Format

**nodes.csv:**
```csv
id,label,name,age
1,Person,Alice,30
2,Person,Bob,25
```

**edges.csv:**
```csv
source,target,type
1,2,KNOWS
```

---

## Server Modes

```bash
neuralgraph                      # Interactive shell
neuralgraph serve 3000           # REST API
neuralgraph serve-flight 50051   # Arrow Flight (high-performance)
neuralgraph serve-raft 1 50052   # Raft cluster node
```

---

## Cluster Commands

```bash
neuralgraph cluster info <addr>           # View cluster info
neuralgraph cluster health <addr>         # Check node health
neuralgraph cluster add <addr> <id> <new> # Add node to cluster
neuralgraph cluster remove <addr> <id>    # Remove node from cluster
```

---

## Constraints (Sprint 66)

```cypher
-- Create unique constraint
CALL neural.constraint.createUnique('person_email', 'email', 'Person')

-- List constraints
CALL neural.constraint.list()

-- Drop constraint
CALL neural.constraint.drop('person_email')
```

---

## Configuration (Sprint 66)

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `NGDB__STORAGE__PATH` | `data/graph.ngdb` | Database path |
| `NGDB__PERSISTENCE__SAVE_INTERVAL_SECS` | `60` | Auto-save interval |
| `NGDB__PERSISTENCE__CHECKSUM_ENABLED` | `true` | SHA256 checksums |
| `NGDB__MEMORY__LIMIT_MB` | `0` | Memory limit (0=unlimited) |
| `NGDB__AUTH__ENABLED` | `false` | Enable authentication |
| `NGDB__AUTH__JWT_SECRET` | - | JWT signing secret |
| `NGDB__AUTH__JWT_EXPIRATION_SECS` | `3600` | JWT expiration |
| `NGDB_LOG` | `info` | Log level |

### Logging

```bash
export NGDB_LOG=info                                 # Default
export NGDB_LOG=debug                                # Verbose
export NGDB_LOG=neural_storage::wal=debug,info      # Module-specific
export NGDB_LOG_JSON=1                               # JSON output (Sprint 67)
```

---

## Observability (Sprint 67)

### Health Endpoint
```bash
curl http://localhost:3000/health
```
Returns: status, uptime, node/edge counts, database path

### Metrics Endpoint
```bash
curl http://localhost:3000/metrics
```
Prometheus format with: query_total, query_latency, node_count, edge_count, cache stats

### Key Metrics
| Metric | Type | Description |
|--------|------|-------------|
| `neuralgraph_query_total` | Counter | Total queries |
| `neuralgraph_query_latency_seconds` | Histogram | Latency distribution |
| `neuralgraph_node_count` | Gauge | Total nodes |
| `neuralgraph_edge_count` | Gauge | Total edges |

---

## REST API

```bash
# Execute query
curl -X POST http://localhost:3000/api/query \
  -H "Content-Type: application/json" \
  -d '{"query": "MATCH (n) RETURN n LIMIT 10"}'

# With JWT authentication (Sprint 68)
curl -X POST http://localhost:3000/api/query \
  -H "Authorization: Bearer <jwt-token>" \
  -H "Content-Type: application/json" \
  -d '{"query": "MATCH (n) RETURN n LIMIT 10"}'

# With API key authentication (Sprint 68)
curl -X POST http://localhost:3000/api/query \
  -H "X-API-Key: <your-api-key>" \
  -H "Content-Type: application/json" \
  -d '{"query": "MATCH (n) RETURN n LIMIT 10"}'

# Health check (Sprint 67) - No auth required
curl http://localhost:3000/health

# Prometheus metrics (Sprint 67) - No auth required
curl http://localhost:3000/metrics
```

---

## Authentication (Sprint 68)

### Enable Auth
```bash
export NGDB__AUTH__ENABLED=true
export NGDB__AUTH__JWT_SECRET="your-32-byte-secret-key-here!!!"
```

### Roles
| Role | Read | Write | Admin |
|------|------|-------|-------|
| `admin` | Yes | Yes | Yes |
| `user` | Yes | Yes | No |
| `readonly` | Yes | No | No |

### Public Endpoints (no auth required)
- `GET /health`
- `GET /metrics`
- `GET /`

### Protected Endpoints
| Endpoint | Required Role |
|----------|---------------|
| `/api/query` (read) | admin, user, readonly |
| `/api/query` (mutate) | admin, user |
| `/api/bulk-load` | admin only |

---

## Property Types

| Type | Example |
|------|---------|
| String | `"hello"` |
| Integer | `42` |
| Float | `3.14` |
| Boolean | `true`, `false` |
| Date | `date("2026-01-28")` |
| DateTime | `datetime()` |
| Array | `["a", "b", "c"]` |
| Map | `{key: "value", debug: true}` |
| Vector | `[0.1, 0.2, 0.3]` (numeric arrays) |
| Null | `null` |
