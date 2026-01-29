# NeuralGraphDB Quick Reference (v0.9.6)

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
MATCH (n:Person {name: "Alice"}) RETURN n

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

-- Update
MATCH (n:Person {name: "Alice"}) SET n.age = 31

-- Delete
MATCH (n:Person {name: "Alice"}) DELETE n            -- Node only
MATCH (n:Person {name: "Alice"}) DETACH DELETE n     -- Node + relationships
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

-- Search with relevance ranking
CALL neural.fulltext.query('search_idx', 'machine learning', 10)
YIELD node, score
RETURN node.title, score

-- Phrase search
CALL neural.fulltext.query('search_idx', '"neural network"', 5)

-- Boolean search (AND, OR, NOT/-)
CALL neural.fulltext.query('search_idx', 'deep AND learning -CNN', 10)

-- List/drop indexes
CALL neural.fulltext.indexes()
CALL neural.fulltext.dropIndex('search_idx')
```

---

## Vector Search

```cypher
-- Find similar items
CALL neural.search($embedding, 'cosine', 10)
YIELD node, score
RETURN node.title, score

-- Metrics: 'cosine', 'euclidean', 'dot_product'
```

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

## REST API

```bash
# Execute query
curl -X POST http://localhost:3000/api/query \
  -H "Content-Type: application/json" \
  -d '{"query": "MATCH (n) RETURN n LIMIT 10"}'
```

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
| Null | `null` |
