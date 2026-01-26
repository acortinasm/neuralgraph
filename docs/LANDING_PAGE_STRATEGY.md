# nGraph Landing Page Strategy: The Agentic Graph Database

This document outlines the marketing and positioning strategy to differentiate **nGraph** from competitors like **FalkorDB**.

---

## 1. The Core Narrative
**The Pivot:** Move from "High-Performance Graph DB" to **"The AI Memory Engine."**

While FalkorDB focuses on being a high-performance engine for the Redis ecosystem, nGraph positions itself as the **all-in-one substrate for Autonomous Agents and RAG applications.**

---

## 2. Hero Section
*   **Headline:** Stop Gluing Databases Together.
*   **Sub-headline:** The first **Embedded** Graph Database with built-in PDF ingestion, HNSW Vector Search, and LLM Orchestration. Zero dependencies. Pure Rust.
*   **Primary CTA:** [Build your first Agent Memory]
*   **Secondary CTA:** [View RAG Benchmarks]

---

## 3. The Three Pillars of Differentiation

### Pillar 1: "GraphRAG-in-a-Box" (vs. Just a DB)
*   **The Competitor (FalkorDB):** A storage engine. You must build external ETL pipelines to chunk, embed, and link data.
*   **nGraph:** A complete pipeline. Point it at a folder of PDFs; nGraph handles the extraction, LLM-based entity linking, and graph construction internally.
*   **Key Message:** *Don't build an ETL pipeline. nGraph IS the pipeline.*

### Pillar 2: "Zero-Dependency Embedded" (vs. Server-Required)
*   **The Competitor:** Requires a Redis server, GraphBLAS C-libraries, and specific OS environments.
*   **nGraph:** A single Rust crate or binary.
*   **Key Message:** *Runs on the Edge, in a Lambda, or inside your App. No Docker required.*

### Pillar 3: "Hybrid Native Engine" (vs. Modular)
*   **The Competitor:** Often splits Graph logic (Falkor) and Vector logic (RediSearch).
*   **nGraph:** Uses unified Linear Algebra structures for both Graph Topology (CSR) and Vector Proximity (HNSW).
*   **Key Message:** *Sub-millisecond hybrid queries. Traverse the graph and search vectors in a single hop.*

---

## 4. Code Comparison (The "Aha!" Moment)

### The Modular Way (Others)
```python
# Setup Redis, Connect to Falkor, Connect to Vector DB, 
# Run PDF Parser, Map Entities, Manual Inserts...
client = FalkorDB(...)
vector_db = Chroma(...)
# ... 100 lines of glue code later ...
```

### The nGraph Way
```rust
// One engine, one flow.
let pipeline = EtlPipeline::new(GeminiClient::new());

// 1. Ingest (Automatic Entity Extraction & Embedding)
let graph = pipeline.process_pdf("research.pdf")?;

// 2. Hybrid Query
let results = graph.execute(
    "MATCH (d:Document)
     WHERE vector_similarity(d.embed, $query) > 0.85
     RETURN CLUSTER(d)"
)?;
```

---

## 5. Competitive Matrix

| Feature | **nGraph** | **FalkorDB** |
| :--- | :--- | :--- |
| **Primary Focus** | **AI Agents & RAG** | Enterprise Knowledge Graphs |
| **Architecture** | **Embedded / Standalone** | Redis Module / Server |
| **Dependencies** | **Zero (Pure Rust)** | Redis, GraphBLAS (C), OpenMP |
| **Vector Search** | **First-Class Primitive** | Modular / External |
| **Data Ingestion** | **Built-in (PDF/LLM)** | External Scripting Required |

---

## 6. Target Audience
1.  **AI Engineers:** Tired of managing 4 different databases for one RAG app.
2.  **Rust Developers:** Building high-performance, self-contained agents.
3.  **Edge/Local AI:** Building applications that need to run locally (Private LLMs + Local Knowledge).
