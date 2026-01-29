# ArXiv Demo - NeuralGraphDB v0.5

A comprehensive demonstration of NeuralGraphDB's capabilities using real ArXiv research papers.

## Features Showcased

| Feature | Description |
|---------|-------------|
| **PDF Ingestion** | Load and extract text from 1000+ ArXiv PDFs |
| **Graph Storage** | CSR matrix storage with O(1) neighbor access |
| **NGQL Queries** | Cypher-like query language |
| **Embeddings** | Gemini text-embedding-004 (768 dimensions) |
| **Vector Search** | HNSW-based semantic similarity |
| **Community Detection** | Leiden algorithm clustering |
| **LLM Extraction** | Entity/relation extraction with Gemini |

## Quick Start

### Prerequisites

1. **Download ArXiv PDFs** (one-time setup):

```bash
python3 scripts/download_arxiv.py --count 1000
```

1. **Set up Gemini API key** (optional, for embeddings):

```bash
export GEMINI_API_KEY="your-api-key"
```

### Run the Demo

```bash
# Build and run the full demo pipeline
cargo run --example arxiv_demo -p neural-cli --release
```

This will:

1. ✅ Load/build graph from PDFs
2. ✅ Run NGQL queries on paper metadata
3. ✅ Detect communities with Leiden algorithm
4. ✅ Generate embeddings with Gemini
5. ✅ Extract entities/relations with LLM
6. ✅ Build HNSW vector index

### Start the Web Server

```bash
cargo run -p neural-cli --release -- serve
```

Open <http://localhost:8080> for the web UI.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        ArXiv Demo Pipeline                       │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌───────────┐     ┌───────────┐     ┌───────────────────────┐  │
│  │  ArXiv    │     │   PDF     │     │     GraphStore        │  │
│  │   PDFs    │────▶│ Extractor │────▶│  (Nodes + Edges)      │  │
│  │ (1000+)   │     │           │     │                       │  │
│  └───────────┘     └───────────┘     └───────────────────────┘  │
│                                               │                  │
│                                               ▼                  │
│  ┌───────────┐     ┌───────────┐     ┌───────────────────────┐  │
│  │  Gemini   │     │ Embedding │     │    Vector Index       │  │
│  │   API     │◀───▶│ Generator │────▶│      (HNSW)           │  │
│  │           │     │           │     │                       │  │
│  └───────────┘     └───────────┘     └───────────────────────┘  │
│                                               │                  │
│                                               ▼                  │
│                          ┌────────────────────────────────────┐  │
│                          │         Web Server (Axum)          │  │
│                          │   • /api/papers - List all         │  │
│                          │   • /api/search - Text search      │  │
│                          │   • /api/similar/{id} - Vector sim │  │
│                          └────────────────────────────────────┘  │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## Data Files

After running the demo, these files are created in `data/`:

| File | Description |
|------|-------------|
| `arxiv_pdfs/` | Downloaded PDF files |
| `arxiv_graph.json` | Serialized graph with nodes/edges |
| `arxiv_embeddings.json` | Paper embeddings (768-dim vectors) |
| `arxiv_entities.json` | Extracted entities from LLM |
| `arxiv_relations.json` | Extracted relations from LLM |
| `arxiv_processed_pdfs.json` | Tracking for incremental processing |

---

## API Reference

### GET `/api/papers`

Returns all papers in the graph.

**Response:**

```json
[
  {
    "id": "2302.00155",
    "node_id": 0,
    "title": "2302.00155",
    "abstract_text": "First 500 chars of PDF...",
    "word_count": 5432
  }
]
```

### POST `/api/search`

Search papers by text query.

**Request:**

```json
{
  "query": "transformer attention",
  "limit": 10
}
```

**Response:**

```json
{
  "papers": [
    {
      "id": "2302.00155",
      "title": "...",
      "abstract_text": "...",
      "score": 1.5
    }
  ],
  "query": "transformer attention",
  "total": 5
}
```

### GET `/api/similar/{paper_id}`

Find semantically similar papers using embedding cosine similarity.

**Response:**

```json
{
  "papers": [
    {
      "id": "2303.00123",
      "title": "...",
      "abstract_text": "...",
      "score": 0.92,
      "distance": 0.08
    }
  ],
  "query": "Similar to: 2302.00155",
  "total": 10
}
```

The `score` field represents **cosine similarity** (1.0 = identical, 0.0 = orthogonal).
The `distance` field represents **cosine distance** (0.0 = identical, 1.0 = orthogonal).

---

## NGQL Query Examples

```sql
-- Count all papers
MATCH (p:Paper) RETURN COUNT(*) AS total

-- Top 5 papers by word count
MATCH (p:Paper) 
RETURN p.id, p.word_count 
ORDER BY p.word_count DESC 
LIMIT 5

-- Papers in the same community
MATCH (p:Paper)
WHERE CLUSTER(p) = CLUSTER(other)
RETURN p.id, other.id
LIMIT 10
```

---

## Embedding Similarity

The demo uses **Gemini text-embedding-004** to generate 768-dimensional embeddings for each paper's first page text.

### How Similarity Works

1. **Embedding Generation**: Each paper's abstract/first page is converted to a 768-dim vector
2. **HNSW Indexing**: Vectors are indexed using Hierarchical Navigable Small World graphs
3. **Cosine Similarity**: Similarity between papers is computed as:

```
similarity = cos(θ) = (A · B) / (||A|| × ||B||)
distance = 1 - similarity
```

Where:

- **similarity = 1.0**: Papers are identical in semantic content
- **similarity = 0.0**: Papers are completely unrelated
- **distance = 0.0**: Papers are identical
- **distance = 1.0**: Papers are completely unrelated

### Interpreting Scores

| Score Range | Interpretation |
|-------------|----------------|
| 0.90 - 1.00 | Very similar (same topic/approach) |
| 0.75 - 0.90 | Related (similar field/methodology) |
| 0.50 - 0.75 | Somewhat related |
| < 0.50 | Different topics |

---

## Community Detection

The Leiden algorithm identifies clusters of papers that are more densely connected to each other than to papers outside the cluster.

```rust
let communities = store.detect_communities();
println!("Found {} communities", communities.num_communities());
```

---

## Incremental Processing

The demo supports **incremental processing** to handle interruptions:

- **Embeddings**: Saved every 50 papers to `arxiv_embeddings.json`
- **Entities**: Saved every 10 PDFs to `arxiv_entities.json`
- **Tracking**: `arxiv_processed_pdfs.json` tracks which PDFs are done

If the process is interrupted, simply restart and it will resume from where it left off.

---

## Troubleshooting

### "No embeddings file found"

Embeddings require a Gemini API key:

```bash
export GEMINI_API_KEY="your-key-here"
cargo run --example arxiv_demo -p neural-cli --release
```

### PDF extraction failures

Some PDFs may fail to parse. The demo uses `catch_unwind` to handle panics gracefully and continue processing.

### Memory usage

For 1000+ papers with embeddings, expect ~500MB memory usage.

---

## Performance

| Operation | Time (1000 papers) |
|-----------|-------------------|
| PDF loading | ~2 min |
| Embedding generation | ~10 min (rate limited) |
| HNSW index build | < 1 sec |
| Similar paper search | < 10 ms |

---

## License

MIT
