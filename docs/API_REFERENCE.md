# NeuralGraphDB API Reference

## Overview

NeuralGraphDB provides two APIs for programmatic access:

| API | Protocol | Port | Best For |
|-----|----------|------|----------|
| **REST API** | HTTP/JSON | 3000 | Web apps, simple integrations |
| **Arrow Flight** | gRPC/Arrow | 50051 | Data science, high-performance |

---

## REST API

### Starting the Server

```bash
neuralgraph serve [PORT]
# Default port: 3000
```

### Base URL

```
http://localhost:3000
```

---

### Endpoints

#### Execute Query

Execute an NGQL query and return results.

```
POST /api/query
```

**Request:**
```json
{
  "query": "MATCH (n:Person) WHERE n.age > $min_age RETURN n.name, n.age",
  "params": {
    "min_age": 25
  }
}
```

**Response (Success):**
```json
{
  "success": true,
  "result": {
    "columns": ["name", "age"],
    "rows": [
      {"name": "Alice", "age": 30},
      {"name": "Bob", "age": 28}
    ],
    "count": 2
  },
  "execution_time_ms": 1.23
}
```

**Response (Error):**
```json
{
  "success": false,
  "error": "Parse error: unexpected token at line 1",
  "result": null
}
```

**Example:**
```bash
curl -X POST http://localhost:3000/api/query \
  -H "Content-Type: application/json" \
  -d '{"query": "MATCH (n:Person) RETURN n.name LIMIT 5"}'
```

---

#### Search (Semantic)

Full-text or semantic search across nodes.

```
POST /api/search
```

**Request:**
```json
{
  "query": "machine learning algorithms",
  "limit": 10
}
```

**Response:**
```json
{
  "papers": [
    {
      "id": 123,
      "title": "Introduction to Neural Networks",
      "score": 0.95
    },
    {
      "id": 456,
      "title": "Deep Learning Fundamentals",
      "score": 0.89
    }
  ],
  "query": "machine learning algorithms",
  "total": 2
}
```

---

#### Get Similar Items

Find nodes similar to a given node.

```
GET /api/similar/{node_id}
```

**Response:**
```json
{
  "node_id": 123,
  "similar": [
    {"id": 456, "title": "Related Document", "score": 0.92},
    {"id": 789, "title": "Another Match", "score": 0.87}
  ]
}
```

---

#### Bulk Load

Load data from CSV files.

```
POST /api/bulk-load
```

**Request:**
```json
{
  "nodes_path": "/path/to/nodes.csv",
  "edges_path": "/path/to/edges.csv",
  "clear_existing": false
}
```

**Response:**
```json
{
  "success": true,
  "nodes_loaded": 10000,
  "edges_loaded": 50000,
  "load_time_ms": 1234.56
}
```

---

#### Health Check (Sprint 67)

Returns detailed health status for production monitoring and load balancer integration.

```
GET /health
```

**Response:**
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

**Fields:**
| Field | Type | Description |
|-------|------|-------------|
| `status` | string | `"healthy"` or `"degraded"` |
| `uptime_seconds` | integer | Seconds since server start |
| `database.loaded` | boolean | Whether database is loaded |
| `database.node_count` | integer | Total nodes in graph |
| `database.edge_count` | integer | Total edges in graph |
| `database.path` | string | Database file path |

---

#### Prometheus Metrics (Sprint 67)

Exports metrics in Prometheus text format for scraping by monitoring systems.

```
GET /metrics
```

**Response (text/plain):**
```
# HELP neuralgraph_query_total Total number of queries executed
# TYPE neuralgraph_query_total counter
neuralgraph_query_total 1234

# HELP neuralgraph_query_latency_seconds Query latency in seconds
# TYPE neuralgraph_query_latency_seconds histogram
neuralgraph_query_latency_seconds_bucket{le="0.001"} 500
neuralgraph_query_latency_seconds_bucket{le="0.01"} 1000
neuralgraph_query_latency_seconds_sum 12.34
neuralgraph_query_latency_seconds_count 1234

# HELP neuralgraph_node_count Total number of nodes in the graph
# TYPE neuralgraph_node_count gauge
neuralgraph_node_count 12345

# HELP neuralgraph_edge_count Total number of edges in the graph
# TYPE neuralgraph_edge_count gauge
neuralgraph_edge_count 67890

# HELP neuralgraph_cache_hits_total Total number of cache hits
# TYPE neuralgraph_cache_hits_total counter
neuralgraph_cache_hits_total 5000
```

**Available Metrics:**
| Metric | Type | Description |
|--------|------|-------------|
| `neuralgraph_query_total` | Counter | Total queries executed |
| `neuralgraph_query_latency_seconds` | Histogram | Query latency distribution |
| `neuralgraph_node_count` | Gauge | Total nodes in graph |
| `neuralgraph_edge_count` | Gauge | Total edges in graph |
| `neuralgraph_cache_hits_total` | Counter | Cache hit count |
| `neuralgraph_cache_misses_total` | Counter | Cache miss count |
| `neuralgraph_cache_size` | Gauge | Current cache size |
| `neuralgraph_vectors_total` | Gauge | Total indexed vectors |
| `neuralgraph_active_connections` | Gauge | Active gRPC connections |

---

### Error Codes

| Status | Meaning |
|--------|---------|
| 200 | Success |
| 400 | Bad request (invalid query syntax) |
| 404 | Resource not found |
| 500 | Internal server error |

---

## Arrow Flight API

Arrow Flight provides high-performance, zero-copy data transfer using Apache Arrow format.

### Starting the Server

```bash
neuralgraph serve-flight [PORT]
# Default port: 50051
```

### Connection

```
grpc://localhost:50051
```

---

### Python Client

#### Installation

```bash
pip install pyarrow
```

#### Basic Usage

```python
import pyarrow.flight as flight

# Connect to server
client = flight.connect("grpc://localhost:50051")

# Execute query
query = "MATCH (n:Person) RETURN n.name, n.age LIMIT 100"
ticket = flight.Ticket(query.encode())
reader = client.do_get(ticket)

# Read results as Arrow Table
table = reader.read_all()

# Convert to pandas DataFrame
df = table.to_pandas()
print(df)
```

#### Streaming Large Results

```python
import pyarrow.flight as flight

client = flight.connect("grpc://localhost:50051")

query = "MATCH (n:Person) RETURN n.name, n.age"
reader = client.do_get(flight.Ticket(query.encode()))

# Process in batches (memory efficient)
for batch in reader:
    df_batch = batch.data.to_pandas()
    process(df_batch)
```

#### Error Handling

```python
import pyarrow.flight as flight

try:
    client = flight.connect("grpc://localhost:50051")
    reader = client.do_get(flight.Ticket(b"INVALID QUERY"))
    table = reader.read_all()
except flight.FlightError as e:
    print(f"Query error: {e}")
```

---

### Result Schema

Query results are returned as Arrow tables with the following schema:

**Node queries:**
```
id: uint64
label: string
properties: string (JSON)
```

**Relationship queries:**
```
source: uint64
target: uint64
type: string
properties: string (JSON)
```

**Custom projections:**
Schema matches the RETURN clause columns and types.

---

## Query Parameters

Both APIs support parameterized queries to prevent injection and improve performance.

### REST API

```json
{
  "query": "MATCH (n:Person) WHERE n.age > $min_age RETURN n",
  "params": {
    "min_age": 25,
    "name": "Alice",
    "active": true
  }
}
```

### Supported Parameter Types

| Type | JSON Example |
|------|--------------|
| String | `"Alice"` |
| Integer | `42` |
| Float | `3.14` |
| Boolean | `true` |
| Null | `null` |
| List | `[1, 2, 3]` |

---

## Rate Limits & Best Practices

### Recommendations

1. **Use parameters** for dynamic values (prevents injection, improves caching)
2. **Limit results** with `LIMIT` clause to avoid large transfers
3. **Use Arrow Flight** for data science workloads (10-100x faster for large results)
4. **Batch operations** when loading data

### Connection Pooling

For production use, maintain persistent connections:

```python
# Good: Reuse connection
client = flight.connect("grpc://localhost:50051")
for query in queries:
    result = client.do_get(flight.Ticket(query.encode()))

# Bad: New connection per query
for query in queries:
    client = flight.connect("grpc://localhost:50051")  # Overhead!
    result = client.do_get(flight.Ticket(query.encode()))
```

---

## Examples

### Python - Complete Example

```python
import pyarrow.flight as flight
import json

class NeuralGraphClient:
    def __init__(self, host="localhost", port=50051):
        self.client = flight.connect(f"grpc://{host}:{port}")

    def query(self, ngql: str) -> list[dict]:
        """Execute NGQL query and return results as list of dicts."""
        reader = self.client.do_get(flight.Ticket(ngql.encode()))
        table = reader.read_all()
        return table.to_pylist()

    def query_df(self, ngql: str):
        """Execute NGQL query and return pandas DataFrame."""
        reader = self.client.do_get(flight.Ticket(ngql.encode()))
        return reader.read_all().to_pandas()


# Usage
db = NeuralGraphClient()

# Get all people over 30
people = db.query("MATCH (n:Person) WHERE n.age > 30 RETURN n.name, n.age")
for person in people:
    print(f"{person['name']}: {person['age']}")

# Get as DataFrame
df = db.query_df("MATCH (n:Person) RETURN n.name, n.age")
print(df.describe())
```

### JavaScript/Node.js - REST Example

```javascript
async function query(ngql, params = {}) {
  const response = await fetch('http://localhost:3000/api/query', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ query: ngql, params })
  });

  const data = await response.json();

  if (!data.success) {
    throw new Error(data.error);
  }

  return data.result.rows;
}

// Usage
const people = await query(
  'MATCH (n:Person) WHERE n.age > $min_age RETURN n.name',
  { min_age: 25 }
);

console.log(people);
```

### cURL Examples

```bash
# Simple query
curl -X POST http://localhost:3000/api/query \
  -H "Content-Type: application/json" \
  -d '{"query": "MATCH (n:Person) RETURN n.name LIMIT 5"}'

# With parameters
curl -X POST http://localhost:3000/api/query \
  -H "Content-Type: application/json" \
  -d '{
    "query": "MATCH (n:Person) WHERE n.age > $age RETURN n",
    "params": {"age": 30}
  }'

# Create data
curl -X POST http://localhost:3000/api/query \
  -H "Content-Type: application/json" \
  -d '{"query": "CREATE (n:Person {name: \"Alice\", age: 30})"}'

# Health check
curl http://localhost:3000/api/health
```

---

## Version History

| Version | Changes |
|---------|---------|
| 0.9 | Initial REST and Arrow Flight APIs |
| 0.9.2 | Added parameterized queries |
| 0.9.5 | Performance improvements, batch loading |
| 0.9.9 | /api/schema endpoint for LangChain integration |
| 0.9.10 | /health and /metrics endpoints for production observability |
