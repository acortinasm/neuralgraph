# NeuralGraphDB User Guide

> The Graph Database for AI Applications

**Version**: 0.9.9
**Last Updated**: January 2026

---

## Table of Contents

1. [Introduction](#1-introduction)
2. [Getting Started](#2-getting-started)
3. [Data Model](#3-data-model)
4. [Query Language (NGQL)](#4-query-language-ngql)
5. [Working with Data](#5-working-with-data)
6. [Transactions](#6-transactions)
7. [Full-Text Search](#7-full-text-search)
8. [Vector Search](#8-vector-search)
9. [Graph Analytics](#9-graph-analytics)
10. [APIs & Integrations](#10-apis--integrations)
11. [Administration](#11-administration)
12. [Best Practices](#12-best-practices)

---

## 1. Introduction

### What is NeuralGraphDB?

NeuralGraphDB is a high-performance graph database designed for AI applications. It combines:

- **Graph Storage**: Store entities and their relationships naturally
- **Vector Search**: Find similar items using AI embeddings
- **Graph Analytics**: Discover patterns and communities in your data

### Key Benefits

| Benefit | Description |
|---------|-------------|
| **Fast** | Sub-millisecond query performance |
| **Scalable** | Handle millions of nodes and billions of edges |
| **AI-Ready** | Built-in vector search for semantic queries |
| **Reliable** | ACID transactions with automatic persistence |

### Use Cases

- **Knowledge Graphs**: Connect entities, documents, and concepts
- **Recommendation Systems**: Find similar users, products, or content
- **Fraud Detection**: Discover suspicious patterns in transaction networks
- **Social Networks**: Analyze relationships and communities
- **RAG Applications**: Combine graph context with LLM responses

---

## 2. Getting Started

### Installation

```bash
# Download and install
cargo install neuralgraph

# Or run with Docker

```

### Quick Start

#### Start the Interactive Shell

```bash
neuralgraph
```

#### Create Your First Graph

```cypher
-- Create nodes
CREATE (alice:Person {name: "Alice", age: 30})
CREATE (bob:Person {name: "Bob", age: 25})
CREATE (acme:Company {name: "Acme Corp"})

-- Create relationships
MATCH (a:Person), (b:Person)
WHERE a.name = "Alice" AND b.name = "Bob"
CREATE (a)-[:KNOWS {since: 2020}]->(b)

MATCH (a:Person), (c:Company)
WHERE a.name = "Alice" AND c.name = "Acme Corp"
CREATE (a)-[:WORKS_AT {role: "Engineer"}]->(c)
```

#### Query Your Data

```cypher
-- Find Alice's connections
MATCH (alice:Person)-[r]->(other)
WHERE alice.name = "Alice"
RETURN type(r) AS relationship, other.name AS connected_to
```

**Result:**
| relationship | connected_to |
|--------------|--------------|
| KNOWS | Bob |
| WORKS_AT | Acme Corp |

### Loading Sample Data

```bash
# In the shell
:demo                           # Load demo graph
:load nodes data/nodes.csv      # Load nodes from CSV
:load edges data/edges.csv      # Load edges from CSV
```

---

## 3. Data Model

### Nodes

Nodes represent entities in your graph. Each node can have:

- **Labels**: Categories like `:Person`, `:Product`, `:Document`
- **Properties**: Key-value pairs like `name: "Alice"`, `age: 30`

```cypher
-- Node with label and properties
CREATE (n:Person {name: "Alice", age: 30, email: "alice@example.com"})

-- Node with multiple labels
CREATE (n:Person:Employee {name: "Bob"})
```

### Relationships

Relationships connect nodes and describe how they relate:

```cypher
-- Simple relationship
(alice)-[:KNOWS]->(bob)

-- Relationship with properties
(alice)-[:KNOWS {since: 2020, closeness: "friend"}]->(bob)

-- Direction matters
(alice)-[:FOLLOWS]->(bob)    -- Alice follows Bob
(alice)<-[:FOLLOWS]-(bob)    -- Bob follows Alice
```

### Property Types

| Type | Example | Description |
|------|---------|-------------|
| String | `"Hello"` | Text values |
| Integer | `42` | Whole numbers |
| Float | `3.14` | Decimal numbers |
| Boolean | `true`, `false` | Yes/no values |
| Date | `date("2026-01-28")` | Calendar dates |
| DateTime | `datetime()` | Timestamps |
| List | `["a", "b", "c"]` | Arrays of values |
| Map | `{key: "value"}` | Key-value objects |
| Null | `null` | Missing value |

### Arrays and Maps (Sprint 64)

NeuralGraphDB supports complex data types for flexible data modeling:

#### Arrays

Store heterogeneous lists of values:

```cypher
-- String array
CREATE (n:Person {name: "Alice", tags: ["developer", "rust", "graph"]})

-- Mixed-type array
CREATE (n:Item {data: ["text", 123, true, null]})

-- Numeric arrays become vectors (optimized for embeddings)
CREATE (n:Doc {embedding: [0.1, 0.2, 0.3, 0.4]})
```

#### Maps

Store nested JSON-like structures:

```cypher
-- Simple map
CREATE (n:Config {name: "settings", options: {debug: true, level: 5}})

-- Nested structures
CREATE (n:Profile {
    name: "Bob",
    metadata: {scores: [100, 95, 88], active: true}
})

-- Query nested data
MATCH (n:Config) RETURN n.options AS config
```

---

## 4. Query Language (NGQL)

NGQL is NeuralGraph's query language, similar to Cypher.

### Reading Data

#### MATCH - Find Patterns

```cypher
-- All nodes with a label
MATCH (n:Person) RETURN n

-- Nodes with specific properties
MATCH (n:Person) WHERE n.name = "Alice" RETURN n

-- Patterns with relationships
MATCH (a:Person)-[:KNOWS]->(b:Person) RETURN a.name, b.name

-- Any relationship type
MATCH (a)-[r]->(b) RETURN a, type(r), b
```

#### WHERE - Filter Results

```cypher
-- Comparison operators
MATCH (n:Person) WHERE n.age > 30 RETURN n
MATCH (n:Person) WHERE n.age >= 18 AND n.age <= 65 RETURN n

-- String matching
MATCH (n:Person) WHERE n.name STARTS WITH "A" RETURN n
MATCH (n:Person) WHERE n.email CONTAINS "@gmail" RETURN n

-- Null checks
MATCH (n:Person) WHERE n.email IS NOT NULL RETURN n

-- List membership
MATCH (n:Product) WHERE n.category IN ["Electronics", "Books"] RETURN n
```

#### RETURN - Shape Output

```cypher
-- Select properties
RETURN n.name, n.age

-- Rename columns
RETURN n.name AS person_name

-- Unique values
RETURN DISTINCT n.category

-- Sorting
RETURN n ORDER BY n.age DESC

-- Pagination
RETURN n SKIP 10 LIMIT 5
```

#### Aggregations

```cypher
-- Counting
MATCH (n:Person) RETURN COUNT(*)
MATCH (n:Person) RETURN COUNT(DISTINCT n.country)

-- Math
MATCH (n:Order) RETURN SUM(n.total), AVG(n.total)
MATCH (n:Product) RETURN MIN(n.price), MAX(n.price)

-- Grouping
MATCH (n:Person)
RETURN n.country, COUNT(*) AS population
ORDER BY population DESC
```

#### WITH - Chain Queries

```cypher
-- Filter after aggregation
MATCH (n:Person)-[:PURCHASED]->(p:Product)
WITH n, COUNT(p) AS purchases
WHERE purchases > 10
RETURN n.name, purchases

-- Top results then expand
MATCH (n:Person)
WITH n ORDER BY n.followers DESC LIMIT 10
MATCH (n)-[:POSTED]->(post)
RETURN n.name, COLLECT(post.title) AS top_posts
```

### Path Queries

#### Variable-Length Paths

```cypher
-- 1 to 3 hops
MATCH (a:Person)-[:KNOWS*1..3]->(b:Person)
WHERE a.name = "Alice"
RETURN DISTINCT b.name

-- Any length
MATCH (a)-[*]->(b)
RETURN a, b
```

#### Shortest Path

```cypher
-- Find shortest connection
MATCH p = shortestPath((a:Person)-[*]-(b:Person))
WHERE a.name = "Alice" AND b.name = "Bob"
RETURN p, length(p)
```

### Modifying Data

#### CREATE - Add Data

```cypher
-- Create node
CREATE (n:Person {name: "Carol", age: 28})

-- Create relationship
MATCH (a:Person), (b:Person)
WHERE a.name = "Alice" AND b.name = "Bob"
CREATE (a)-[:KNOWS]->(b)

-- Create everything at once
CREATE (a:Person {name: "Dan"})-[:KNOWS]->(b:Person {name: "Eve"})
```

#### SET - Update Properties

```cypher
MATCH (n:Person)
WHERE n.name = "Alice"
SET n.age = 31

MATCH (n:Person)
WHERE n.name = "Alice"
SET n.verified = true, n.updated_at = datetime()
```

#### DELETE - Remove Data

```cypher
-- Delete relationship
MATCH (a)-[r:KNOWS]->(b)
WHERE a.name = "Alice" AND b.name = "Bob"
DELETE r

-- Delete node (must have no relationships)
MATCH (n:Person)
WHERE n.name = "Alice"
DELETE n

-- Delete node and all its relationships
MATCH (n:Person)
WHERE n.name = "Alice"
DETACH DELETE n
```

#### MERGE - Create If Not Exists

```cypher
-- Create only if doesn't exist
MERGE (n:Person {email: "alice@example.com"})

-- Set properties on create or match
MERGE (n:Person {email: "alice@example.com"})
ON CREATE SET n.created_at = datetime()
ON MATCH SET n.last_seen = datetime()
```

---

## 5. Working with Data

### Loading CSV Data

**nodes.csv:**
```csv
id,label,name,age
1,Person,Alice,30
2,Person,Bob,25
3,Company,Acme,
```

**edges.csv:**
```csv
source,target,type
1,2,KNOWS
1,3,WORKS_AT
```

```bash
# In the shell
:load nodes nodes.csv
:load edges edges.csv
```

### Saving and Loading Databases

```bash
# Save current state
:save my_database.ngdb

# Load saved database
:loadbin my_database.ngdb
```

### Using Parameters

Parameters make queries safer and reusable:

```bash
# Set parameters
:param user_name "Alice"
:param min_age 25
```

```cypher
-- Use parameters in queries
MATCH (n:Person)
WHERE n.name = $user_name AND n.age >= $min_age
RETURN n
```

### Query Analysis

```cypher
-- See execution plan
EXPLAIN MATCH (n:Person)-[:KNOWS]->(m) RETURN m

-- Run with timing info
PROFILE MATCH (n:Person) WHERE n.age > 30 RETURN COUNT(*)
```

---

## 6. Transactions

Transactions ensure your data stays consistent, even if something goes wrong.

### Basic Transactions

```cypher
-- Start transaction
BEGIN

-- Make changes
CREATE (a:Account {id: "A1", balance: 1000})
CREATE (b:Account {id: "A2", balance: 500})

-- Save changes
COMMIT
```

### Rollback on Error

```cypher
BEGIN

-- Make changes
MATCH (a:Account) WHERE a.id = "A1" SET a.balance = a.balance - 100
MATCH (b:Account) WHERE b.id = "A2" SET b.balance = b.balance + 100

-- Oops, something went wrong!
ROLLBACK
-- All changes are undone
```

### Transaction Guarantees

| Property | Meaning |
|----------|---------|
| **Atomic** | All changes succeed or all fail together |
| **Consistent** | Database is always in a valid state |
| **Isolated** | Concurrent transactions don't interfere |
| **Durable** | Committed data survives crashes |

---

## 7. Full-Text Search

Full-text search lets you find nodes by searching text in their properties. It uses relevance ranking to return the most relevant results first.

### Creating a Full-Text Index

Before searching, create an index on the properties you want to search:

```cypher
-- Create an index named 'paper_search' on Paper nodes
-- Index the 'title' and 'abstract' properties
CALL neural.fulltext.createIndex('paper_search', 'Paper', ['title', 'abstract'])

-- Create an index with a specific language (for proper stemming)
CALL neural.fulltext.createIndex('spanish_idx', 'Article', ['titulo'], 'spanish')

-- Create an index with phonetic search enabled
CALL neural.fulltext.createIndex('name_idx', 'Person', ['name'], 'english', 'soundex')
```

**Supported Languages:** English, Spanish, French, German, Italian, Portuguese, Dutch, Swedish, Norwegian, Danish, Finnish, Russian, Hungarian, Romanian, Turkish, Arabic, Greek, Tamil

**Phonetic Algorithms:** `soundex`, `metaphone`, `double_metaphone`

### Searching

```cypher
-- Simple search
CALL neural.fulltext.query('paper_search', 'machine learning', 10)
YIELD node, score
RETURN node.title, score

-- Results are ordered by relevance (higher score = more relevant)
```

### Search Query Syntax

| Syntax | Example | Description |
|--------|---------|-------------|
| Simple terms | `machine learning` | Match any term |
| Phrase | `"neural network"` | Match exact phrase |
| AND | `deep AND learning` | Both terms required |
| NOT | `learning -deep` | Exclude term |
| Field | `title:introduction` | Search specific field |

```cypher
-- Phrase search (exact match)
CALL neural.fulltext.query('paper_search', '"transformer architecture"', 5)
YIELD node, score
RETURN node.title

-- Boolean search
CALL neural.fulltext.query('paper_search', 'deep AND learning NOT CNN', 10)
YIELD node, score
RETURN node.title
```

### Fuzzy Search (Typo Tolerance)

Find matches even with spelling mistakes using Levenshtein distance:

```cypher
-- fuzzyQuery: (index_name, query, limit, max_edit_distance)
-- The 4th argument is the maximum edit distance (default: 1)

-- Distance 1: "machin" matches "machine"
CALL neural.fulltext.fuzzyQuery('paper_search', 'machin lerning', 10)
YIELD node, score
RETURN node.title

-- Distance 2: allows more typos
CALL neural.fulltext.fuzzyQuery('paper_search', 'nueral ntwrks', 10, 2)
YIELD node, score
RETURN node.title
```

### Phonetic Search (Sound-Alike Matching)

Match words that sound similar, useful for names:

```cypher
-- First, create an index with phonetic enabled
CALL neural.fulltext.createIndex('name_idx', 'Person', ['name'], 'english', 'soundex')

-- Now "Smith" matches "Smyth", "Smithe", etc.
CALL neural.fulltext.query('name_idx', 'Smith', 10)
YIELD node, score
RETURN node.name

-- Algorithms:
-- soundex: Classic algorithm, good for English names
-- metaphone: More accurate than Soundex
-- double_metaphone: Handles edge cases and non-English words
```

### Features

- **Stemming**: "learning", "learns", "learned" all match "learn"
- **Stop words**: Common words like "the", "a", "is" are ignored
- **Relevance ranking**: Results sorted by BM25 score
- **Multiple properties**: Search across title, abstract, content simultaneously
- **Fuzzy matching**: Find results even with typos (configurable edit distance)
- **Phonetic search**: Match sound-alike words (Soundex, Metaphone, DoubleMetaphone)
- **18 languages**: Proper stemming for English, Spanish, French, German, and more

### Managing Indexes

```cypher
-- List all full-text indexes
CALL neural.fulltext.indexes()

-- Drop an index
CALL neural.fulltext.dropIndex('paper_search')
```

### Combining with Graph Queries

```cypher
-- Find relevant papers, then get their authors
CALL neural.fulltext.query('paper_search', 'graph neural networks', 10)
YIELD node, score
MATCH (node)-[:AUTHORED_BY]->(author:Person)
RETURN node.title, author.name, score
ORDER BY score DESC
```

---

## 8. Vector Search

Vector search lets you find similar items using AI embeddings.

### What Are Embeddings?

Embeddings are numerical representations of content (text, images, etc.) that capture semantic meaning. Similar items have similar embeddings.

### Performing Vector Search

```cypher
-- Find 10 most similar documents to a query
CALL neural.search($query_embedding, 'cosine', 10)
YIELD node, score
RETURN node.title, score
ORDER BY score DESC
```

### Distance Metrics

| Metric | Best For | Range |
|--------|----------|-------|
| `cosine` | Text embeddings (OpenAI, etc.) | -1 to 1 |
| `euclidean` | Sentence transformers | 0 to infinity |
| `dot_product` | Normalized embeddings | -inf to inf |

### Combining Vector and Graph

```cypher
-- Find similar documents, then get authors
CALL neural.search($query, 'cosine', 5)
YIELD node, score
MATCH (node)-[:AUTHORED_BY]->(author:Person)
RETURN node.title, author.name, score
```

### Adding Vectors

Vectors are typically added during data import:

```csv
id,label,title,embedding
1,Document,Introduction to AI,"[0.1, 0.2, 0.3, ...]"
2,Document,Machine Learning Basics,"[0.15, 0.25, 0.28, ...]"
```

### Vector Quantization

Reduce memory usage with quantization (4x to 32x savings):

| Method | Memory Savings | Precision | Best For |
|--------|----------------|-----------|----------|
| `None` | 0% | 100% | Production, precision critical |
| `Int8` | 75% (4x) | ~99% | Balance of memory and precision |
| `Binary` | 97% (32x) | ~90% | Very large datasets |

### Distributed Vector Search

For large-scale deployments, NeuralGraphDB supports distributed vector search across multiple shards:

- **Scatter-gather algorithm**: Query is sent to all shards in parallel
- **Automatic result merging**: Top-k results are merged efficiently
- **Replica failover**: If a shard is unhealthy, queries route to replicas
- **Result caching**: Frequently-used queries are cached with configurable TTL

The distributed search is transparent - you use the same `neural.search()` API whether running on a single node or a cluster.

---

## 9. Graph Analytics

### Community Detection

Find groups of closely connected nodes:

```cypher
-- Get community for each person
MATCH (n:Person)
RETURN n.name, CLUSTER(n) AS community

-- Count community sizes
MATCH (n:Person)
RETURN CLUSTER(n) AS community, COUNT(*) AS size
ORDER BY size DESC
```

### Path Analysis

```cypher
-- All paths between two nodes (up to 5 hops)
MATCH p = (a:Person {name: "Alice"})-[*1..5]->(b:Person {name: "Bob"})
RETURN p, length(p)

-- Shortest path
MATCH p = shortestPath((a:Person)-[*]-(b:Person))
WHERE a.name = "Alice" AND b.name = "Bob"
RETURN length(p) AS degrees_of_separation
```

### Centrality (Finding Important Nodes)

```cypher
-- Most connected people
MATCH (n:Person)-[r]-()
RETURN n.name, COUNT(r) AS connections
ORDER BY connections DESC
LIMIT 10
```

---

## 10. APIs & Integrations

### REST API

Start the server:
```bash
neuralgraph serve 3000
```

Execute queries:
```bash
curl -X POST http://localhost:3000/api/query \
  -H "Content-Type: application/json" \
  -d '{"query": "MATCH (n:Person) RETURN n.name LIMIT 10"}'
```

Response:
```json
{
  "success": true,
  "result": {
    "columns": ["name"],
    "rows": [{"name": "Alice"}, {"name": "Bob"}],
    "count": 2
  },
  "execution_time_ms": 1.2
}
```

### Python Integration

Using Arrow Flight for high-performance data transfer:

```python
import pyarrow.flight as flight

# Connect
client = flight.connect("grpc://localhost:50051")

# Execute query
query = "MATCH (n:Person) RETURN n.name, n.age"
reader = client.do_get(flight.Ticket(query.encode()))

# Get results as Arrow Table
table = reader.read_all()

# Convert to pandas
df = table.to_pandas()
print(df)
```

Start Arrow Flight server:
```bash
neuralgraph serve-flight 50051
```

### LangChain Integration (New in Sprint 65)

NeuralGraphDB integrates natively with LangChain for building GraphRAG applications.

**Installation:**
```bash
pip install neuralgraph[langchain]
```

**Basic Usage:**
```python
from neuralgraph import NeuralGraphStore

# Connect to NeuralGraphDB
graph = NeuralGraphStore(host="localhost", port=3000)

# View schema
print(graph.get_schema())

# Execute NGQL queries
results = graph.query("MATCH (n:Person) RETURN n.name")
```

**Natural Language Q&A:**
```python
from neuralgraph import NeuralGraphStore, create_qa_chain
from langchain_openai import ChatOpenAI

# Setup
graph = NeuralGraphStore()
llm = ChatOpenAI(model="gpt-4", temperature=0)

# Create Q&A chain
chain = create_qa_chain(llm, graph, verbose=True)

# Ask questions in natural language
result = chain.invoke({"query": "Who are the employees at TechCorp?"})
print(result["result"])
```

**Simplified QA Chain:**
```python
from neuralgraph import NeuralGraphStore, NeuralGraphQAChain

graph = NeuralGraphStore()
chain = NeuralGraphQAChain(graph, llm, verbose=True)

# Multiple calling conventions
answer = chain.run("How many people work here?")
answer = chain("Find all engineers")
result = chain.invoke({"query": "Who knows Alice?"})
```

---

## 11. Administration

### Server Modes

```bash
# Interactive shell (default)
neuralgraph

# REST API server
neuralgraph serve 3000

# Arrow Flight server (high-performance)
neuralgraph serve-flight 50051

# Cluster mode (distributed)
neuralgraph serve-raft 1 50052
```

### Backup and Restore

```bash
# Backup
:save backup_2026-01-28.ngdb

# Restore
:loadbin backup_2026-01-28.ngdb
```

### Monitoring

Check database statistics:
```bash
:stats
```

Output:
```
Nodes: 1,234,567
Edges: 5,678,901
Labels: Person, Company, Product, Document
Memory: 256 MB
```

### Cluster Management

NeuralGraphDB supports distributed deployment with automatic leader election and request routing.

```bash
# Start a Raft cluster node
neuralgraph serve-raft 1 50052              # Bootstrap node
neuralgraph serve-raft 2 50053 --join localhost:50052  # Join existing cluster

# Check cluster health
neuralgraph cluster health localhost:50052

# View cluster info (shows leader, members, term)
neuralgraph cluster info localhost:50052

# Add a new node to the cluster
neuralgraph cluster add localhost:50052 3 localhost:50054

# Remove a node from the cluster
neuralgraph cluster remove localhost:50052 3
```

**Cluster-Aware Writes:**

When connected to a cluster, writes are automatically routed to the current leader:
- If you send a write to a follower, it redirects to the leader
- Reads can be served by any node
- Leader election is automatic if the leader fails

### Prometheus Metrics

NeuralGraphDB exposes metrics for monitoring cluster and vector search performance:

**Cluster Metrics:**
- `neuralgraph_raft_term` - Current Raft term
- `neuralgraph_raft_log_index` - Latest log index
- `neuralgraph_cluster_node_count` - Number of nodes in cluster
- `neuralgraph_cluster_leader` - Current leader node ID

**Vector Search Metrics:**
- `neuralgraph_vector_search_duration_seconds` - Search latency histogram
- `neuralgraph_vector_search_total` - Total searches
- `neuralgraph_vector_cache_hits_total` - Cache hit count
- `neuralgraph_vector_cache_misses_total` - Cache miss count

---

## 12. Best Practices

### Data Modeling

**DO:**
- Use descriptive labels: `:Person`, `:Order`, `:Product`
- Use verb relationships: `:PURCHASED`, `:KNOWS`, `:WORKS_AT`
- Store frequently queried properties on nodes

**DON'T:**
- Create "god nodes" connected to everything
- Store large blobs (files, images) as properties
- Use generic labels like `:Node` or `:Entity`

### Query Performance

**DO:**
```cypher
-- Use labels in MATCH
MATCH (n:Person) WHERE n.age > 30 RETURN n

-- Limit early
MATCH (n:Person)
WITH n ORDER BY n.score DESC LIMIT 100
MATCH (n)-[:PURCHASED]->(p)
RETURN n, COLLECT(p)
```

**DON'T:**
```cypher
-- Avoid unlabeled matches on large graphs
MATCH (n) WHERE n.age > 30 RETURN n  -- Scans everything!

-- Avoid collecting everything then filtering
MATCH (n:Person)-[:PURCHASED]->(p)
RETURN n, COLLECT(p)
LIMIT 10  -- Too late, already collected all!
```

### Transactions

**DO:**
- Use transactions for related changes
- Keep transactions short
- Handle errors with ROLLBACK

**DON'T:**
- Leave transactions open for long periods
- Mix unrelated operations in one transaction

### Vector Search

**DO:**
- Use the same embedding model for queries and data
- Match the distance metric to your embedding model
- Start with k=10-20, increase if needed

**DON'T:**
- Mix embeddings from different models
- Use wrong distance metric (usually cosine for text)

---

## Quick Reference

### Common Commands (Shell)

| Command | Description |
|---------|-------------|
| `:help` | Show help |
| `:stats` | Database statistics |
| `:demo` | Load sample data |
| `:save <file>` | Save database |
| `:loadbin <file>` | Load database |
| `:load nodes <csv>` | Import nodes |
| `:load edges <csv>` | Import edges |
| `:param <name> <value>` | Set parameter |
| `:quit` | Exit |

### NGQL Cheat Sheet

```cypher
-- Find
MATCH (n:Label) RETURN n
MATCH (a)-[:REL]->(b) RETURN a, b

-- Filter
WHERE n.prop = value
WHERE n.age > 30 AND n.active = true

-- Create
CREATE (n:Label {prop: value})
CREATE (a)-[:REL]->(b)

-- Update
SET n.prop = value

-- Delete
DELETE n
DETACH DELETE n

-- Aggregate
COUNT(*), SUM(n.x), AVG(n.x), MIN(n.x), MAX(n.x)

-- Sort & Limit
ORDER BY n.prop DESC
LIMIT 10
SKIP 5

-- Transactions
BEGIN ... COMMIT
BEGIN ... ROLLBACK
```

---

## Getting Help

- **Documentation**: [docs.neuralgraph.io](https://docs.neuralgraph.io)
- **GitHub Issues**: [github.com/neuralgraph/neuralgraph/issues](https://github.com/neuralgraph/neuralgraph/issues)
- **Community**: [discord.gg/neuralgraph](https://discord.gg/neuralgraph)

---

*NeuralGraphDB - The Graph Database for AI Applications*
