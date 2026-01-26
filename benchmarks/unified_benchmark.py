#!/usr/bin/env python3
"""
Unified Benchmark: NeuralGraphDB vs FalkorDB vs Neo4j

A comprehensive benchmark workflow that tests all three graph databases
using the same dataset, queries, and metrics for fair comparison.

Usage:
    python benchmarks/unified_benchmark.py                      # Run all databases (default: 1000 papers)
    python benchmarks/unified_benchmark.py -n 5000              # Custom paper count
    python benchmarks/unified_benchmark.py --db neuralgraph     # Single database
    python benchmarks/unified_benchmark.py --db neo4j,falkordb  # Multiple databases
    python benchmarks/unified_benchmark.py --skip-complex       # Skip complex queries
    python benchmarks/unified_benchmark.py --output results/    # Custom output directory

Requirements:
    - NeuralGraphDB: ./target/release/neuralgraph binary + HTTP server on localhost:3000
    - Neo4j: Docker container 'benchmark-neo4j' on bolt://localhost:17687
    - FalkorDB: Docker container 'benchmark-falkordb' on localhost:16379
"""

import json
import time
import logging
import subprocess
import tempfile
import random
import os
import sys
import csv
from abc import ABC, abstractmethod
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, Dict, List, Any
from datetime import datetime

import requests
from tqdm import tqdm

# Conditional imports for optional dependencies
try:
    from datasets import load_dataset
    HAS_DATASETS = True
except ImportError:
    HAS_DATASETS = False

try:
    from sentence_transformers import SentenceTransformer
    HAS_SENTENCE_TRANSFORMERS = True
except ImportError:
    HAS_SENTENCE_TRANSFORMERS = False

try:
    from neo4j import GraphDatabase
    HAS_NEO4J = True
except ImportError:
    HAS_NEO4J = False

try:
    from falkordb import FalkorDB
    HAS_FALKORDB = True
except ImportError:
    HAS_FALKORDB = False

# Import memory monitor from utils
sys.path.insert(0, os.path.dirname(__file__))
from benchmark_utils import MemoryMonitor, ProcessMemoryMonitor

# =============================================================================
# Configuration
# =============================================================================

CONFIG = {
    "neuralgraph": {
        "binary": "./target/release/neuralgraph",
        "http_url": "http://localhost:3000/api/query",
        "container": "benchmark-neuralgraph",
    },
    "neo4j": {
        "uri": "bolt://localhost:17687",
        "auth": ("neo4j", "benchmark123"),
        "container": "benchmark-neo4j",
    },
    "falkordb": {
        "host": "localhost",
        "port": 16379,
        "graph": "unified_bench",
        "container": "benchmark-falkordb",
    },
    "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
    "batch_size": 100,
}

# Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S"
)
log = logging.getLogger(__name__)


# =============================================================================
# Utility Classes
# =============================================================================

@dataclass
class Timer:
    """Context manager for timing operations."""
    name: str
    start: float = 0
    elapsed: float = 0

    def __enter__(self):
        self.start = time.perf_counter()
        log.info(f"Starting: {self.name}")
        return self

    def __exit__(self, *args):
        self.elapsed = time.perf_counter() - self.start
        log.info(f"Completed: {self.name} in {self.elapsed:.3f}s")


@dataclass
class QueryResult:
    """Result of a benchmark query."""
    name: str
    latency_ms: float
    rows: int = 0
    memory_mb: float = 0.0
    success: bool = True
    error: Optional[str] = None


@dataclass
class BenchmarkResult:
    """Complete benchmark results for a database."""
    database: str
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    config: Dict[str, Any] = field(default_factory=dict)
    data_stats: Dict[str, int] = field(default_factory=dict)
    timings: Dict[str, float] = field(default_factory=dict)
    memory: Dict[str, float] = field(default_factory=dict)
    queries: List[QueryResult] = field(default_factory=list)

    def total_time(self) -> float:
        return sum(self.timings.values())

    def to_dict(self) -> dict:
        return {
            "database": self.database,
            "timestamp": self.timestamp,
            "config": self.config,
            "data_stats": self.data_stats,
            "timings": self.timings,
            "memory_mb": self.memory,
            "queries": [
                {
                    "name": q.name,
                    "latency_ms": q.latency_ms,
                    "rows": q.rows,
                    "memory_mb": q.memory_mb,
                    "success": q.success,
                    "error": q.error,
                }
                for q in self.queries
            ],
            "total_time_s": self.total_time(),
        }


# =============================================================================
# Data Generation
# =============================================================================

class DataGenerator:
    """Generates benchmark dataset (ArXiv papers with citations)."""

    INSTITUTIONS = [
        "Stanford University", "MIT", "Google", "OpenAI", "DeepMind",
        "Meta AI", "Berkeley", "CMU", "Microsoft Research", "Amazon"
    ]

    def __init__(self, num_papers: int = 1000, avg_citations: int = 5, use_embeddings: bool = False):
        self.num_papers = num_papers
        self.avg_citations = avg_citations
        self.use_embeddings = use_embeddings and HAS_SENTENCE_TRANSFORMERS
        self.papers: List[dict] = []
        self.citations: List[tuple] = []
        self.authors: List[dict] = []
        self.model = None

    def generate(self) -> "DataGenerator":
        """Generate the full dataset."""
        log.info(f"Generating dataset: {self.num_papers} papers")

        # Load embedding model if needed
        if self.use_embeddings:
            log.info("Loading embedding model...")
            self.model = SentenceTransformer(CONFIG["embedding_model"])

        # Generate papers
        self._generate_papers()

        # Generate embeddings
        if self.use_embeddings and self.model:
            self._generate_embeddings()

        # Generate citations
        self._generate_citations()

        # Generate authors
        self._generate_authors()

        log.info(f"Dataset ready: {len(self.papers)} papers, {len(self.citations)} citations, {len(self.authors)} authors")
        return self

    def _generate_papers(self):
        """Generate paper nodes from ArXiv or synthetic data."""
        if HAS_DATASETS:
            log.info("Loading papers from ArXiv dataset...")
            dataset = load_dataset("CShorten/ML-ArXiv-Papers", split="train")

            for i, item in enumerate(tqdm(dataset, desc="Loading papers", total=min(self.num_papers, len(dataset)))):
                if i >= self.num_papers:
                    break
                self.papers.append({
                    "id": i,
                    "title": (item.get("title") or f"Paper {i}")[:500].replace("'", "").replace('"', ''),
                    "abstract": (item.get("abstract") or "")[:2000].replace("'", "").replace('"', ''),
                    "category": "cs.LG",
                })
        else:
            log.info("Generating synthetic papers...")
            for i in tqdm(range(self.num_papers), desc="Generating papers"):
                self.papers.append({
                    "id": i,
                    "title": f"Research Paper {i}: Machine Learning Advances",
                    "abstract": f"Abstract for paper {i}. This paper explores novel approaches...",
                    "category": "cs.LG",
                })

    def _generate_embeddings(self):
        """Generate embeddings for paper abstracts."""
        log.info("Generating embeddings...")
        abstracts = [p["abstract"] for p in self.papers]
        embeddings = self.model.encode(abstracts, show_progress_bar=True, batch_size=64)
        for paper, emb in zip(self.papers, embeddings):
            paper["embedding"] = emb.tolist()

    def _generate_citations(self):
        """Generate random citation edges."""
        log.info("Generating citations...")
        prob = self.avg_citations / self.num_papers

        for i in tqdm(range(self.num_papers), desc="Generating citations"):
            for j in range(i + 1, self.num_papers):
                if random.random() < prob:
                    self.citations.append((i, j))

        log.info(f"Generated {len(self.citations)} citations")

    def _generate_authors(self):
        """Generate authors with institutional affiliations."""
        log.info("Generating authors...")
        author_id = 0

        for paper_id in tqdm(range(self.num_papers), desc="Generating authors"):
            num_authors = random.randint(1, 3)
            for a in range(num_authors):
                self.authors.append({
                    "id": author_id,
                    "paper_id": paper_id,
                    "name": f"Author_{paper_id}_{a}",
                    "institution": random.choice(self.INSTITUTIONS),
                })
                author_id += 1

    def export_csv(self, output_dir: Path) -> tuple:
        """Export data to CSV files for bulk loading."""
        output_dir.mkdir(parents=True, exist_ok=True)

        # Papers CSV
        papers_file = output_dir / "papers.csv"
        with open(papers_file, "w", newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["id", "label", "title", "abstract", "category"])
            for p in self.papers:
                writer.writerow([p["id"], "Paper", p["title"], p["abstract"], p["category"]])

        # Citations CSV
        citations_file = output_dir / "citations.csv"
        with open(citations_file, "w", newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["source", "target", "label"])
            for src, tgt in self.citations:
                writer.writerow([src, tgt, "CITES"])

        # Authors CSV
        authors_file = output_dir / "authors.csv"
        with open(authors_file, "w", newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["id", "label", "name", "institution", "paper_id"])
            for a in self.authors:
                writer.writerow([
                    a["id"] + self.num_papers,  # Offset IDs to avoid collision
                    "Author",
                    a["name"],
                    a["institution"],
                    a["paper_id"],
                ])

        return papers_file, citations_file, authors_file


# =============================================================================
# Database Adapters
# =============================================================================

class DatabaseAdapter(ABC):
    """Abstract base class for database adapters."""

    name: str = "base"
    container_name: str = ""

    @abstractmethod
    def connect(self) -> bool:
        """Establish connection to database."""
        pass

    @abstractmethod
    def clear(self):
        """Clear all data from database."""
        pass

    @abstractmethod
    def load_papers(self, papers: List[dict]) -> float:
        """Load paper nodes. Returns time in seconds."""
        pass

    @abstractmethod
    def create_citations(self, citations: List[tuple]) -> float:
        """Create citation edges. Returns time in seconds."""
        pass

    @abstractmethod
    def create_authors(self, authors: List[dict], num_papers: int) -> float:
        """Create author nodes and relationships. Returns time in seconds."""
        pass

    @abstractmethod
    def run_query(self, name: str, query: str) -> QueryResult:
        """Execute a query and return results."""
        pass

    @abstractmethod
    def close(self):
        """Close database connection."""
        pass

    def get_memory_monitor(self) -> Optional[MemoryMonitor]:
        """Get memory monitor for this database's container."""
        if self.container_name:
            return MemoryMonitor(self.container_name)
        return None


class NeuralGraphAdapter(DatabaseAdapter):
    """Adapter for NeuralGraphDB (HTTP API)."""

    name = "NeuralGraphDB"
    container_name = CONFIG["neuralgraph"]["container"]

    def __init__(self):
        self.url = CONFIG["neuralgraph"]["http_url"]
        self.connected = False

    def connect(self) -> bool:
        try:
            response = requests.post(
                self.url,
                json={"query": "MATCH (n) RETURN count(n) LIMIT 1"},
                timeout=5
            )
            self.connected = response.status_code == 200
            return self.connected
        except Exception as e:
            log.warning(f"NeuralGraphDB connection failed: {e}")
            return False

    def clear(self):
        requests.post(self.url, json={"query": "MATCH (n) DETACH DELETE n"}, timeout=60)

    def _query(self, cypher: str, timeout: int = 120) -> dict:
        """Execute a Cypher query via HTTP API."""
        response = requests.post(self.url, json={"query": cypher}, timeout=timeout)
        if response.status_code != 200:
            raise Exception(f"Query failed: {response.text}")
        return response.json()

    def load_papers(self, papers: List[dict]) -> float:
        start = time.perf_counter()
        batch_size = CONFIG["batch_size"]

        # Create index first for faster lookups
        try:
            self._query("CREATE INDEX ON :Paper(id)")
        except:
            pass

        for i in tqdm(range(0, len(papers), batch_size), desc="Loading papers"):
            batch = papers[i:i + batch_size]
            # Build batch data as JSON array string
            batch_data = json.dumps([{
                "id": p["id"],
                "title": p["title"][:200].replace("'", ""),
                "category": p["category"]
            } for p in batch])

            # Use UNWIND for batch insert
            query = f"""
            WITH {batch_data} AS papers
            UNWIND papers AS paper
            CREATE (n:Paper {{
                id: paper.id,
                title: paper.title,
                category: paper.category
            }})
            """
            self._query(query)

        return time.perf_counter() - start

    def create_citations(self, citations: List[tuple]) -> float:
        start = time.perf_counter()
        batch_size = CONFIG["batch_size"]

        for i in tqdm(range(0, len(citations), batch_size), desc="Creating citations"):
            batch = citations[i:i + batch_size]
            # Build batch data as JSON array string
            batch_data = json.dumps([{"src": src, "tgt": tgt} for src, tgt in batch])

            # Use UNWIND for batch edge creation
            query = f"""
            WITH {batch_data} AS edges
            UNWIND edges AS edge
            MATCH (a:Paper), (b:Paper)
            WHERE a.id = edge.src AND b.id = edge.tgt
            CREATE (a)-[:CITES]->(b)
            """
            self._query(query)

        return time.perf_counter() - start

    def create_authors(self, authors: List[dict], num_papers: int) -> float:
        start = time.perf_counter()
        batch_size = CONFIG["batch_size"]

        # Create index for faster lookups
        try:
            self._query("CREATE INDEX ON :Author(id)")
        except:
            pass

        for i in tqdm(range(0, len(authors), batch_size), desc="Creating authors"):
            batch = authors[i:i + batch_size]

            # Build batch data for author nodes
            batch_data = json.dumps([{
                "id": a["id"] + num_papers,
                "name": a["name"],
                "institution": a["institution"],
                "paper_id": a["paper_id"]
            } for a in batch])

            # Create author nodes with UNWIND
            query = f"""
            WITH {batch_data} AS authors
            UNWIND authors AS author
            CREATE (n:Author {{
                id: author.id,
                name: author.name,
                institution: author.institution
            }})
            """
            self._query(query)

            # Create relationships in separate batch
            query = f"""
            WITH {batch_data} AS authors
            UNWIND authors AS author
            MATCH (p:Paper), (a:Author)
            WHERE p.id = author.paper_id AND a.id = author.id
            CREATE (p)-[:AUTHORED_BY]->(a)
            """
            self._query(query)

        return time.perf_counter() - start

    def run_query(self, name: str, query: str) -> QueryResult:
        mem_mon = self.get_memory_monitor()
        if mem_mon:
            mem_mon.start()

        start = time.perf_counter()
        try:
            result = self._query(query, timeout=300)
            elapsed = (time.perf_counter() - start) * 1000

            if mem_mon:
                mem_mon.stop()

            rows = 0
            if isinstance(result, dict):
                if "result" in result:
                    r = result["result"]
                    rows = r.get("count", len(r.get("rows", [])))

            return QueryResult(
                name=name,
                latency_ms=elapsed,
                rows=rows,
                memory_mb=mem_mon.get_max_memory() if mem_mon else 0,
                success=True
            )
        except Exception as e:
            if mem_mon:
                mem_mon.stop()
            return QueryResult(name=name, latency_ms=0, success=False, error=str(e))

    def close(self):
        pass

    def get_memory_monitor(self) -> Optional[ProcessMemoryMonitor]:
        """Use process-based memory monitoring for native binary."""
        return ProcessMemoryMonitor("neuralgraph")


class Neo4jAdapter(DatabaseAdapter):
    """Adapter for Neo4j."""

    name = "Neo4j"
    container_name = CONFIG["neo4j"]["container"]

    def __init__(self):
        self.driver = None

    def connect(self) -> bool:
        if not HAS_NEO4J:
            log.warning("Neo4j driver not installed. Install with: pip install neo4j")
            return False

        try:
            self.driver = GraphDatabase.driver(
                CONFIG["neo4j"]["uri"],
                auth=CONFIG["neo4j"]["auth"]
            )
            self.driver.verify_connectivity()
            return True
        except Exception as e:
            log.warning(f"Neo4j connection failed: {e}")
            return False

    def clear(self):
        with self.driver.session() as session:
            session.run("MATCH (n) DETACH DELETE n")

    def load_papers(self, papers: List[dict]) -> float:
        start = time.perf_counter()
        batch_size = CONFIG["batch_size"]

        with self.driver.session() as session:
            # Create index first
            session.run("CREATE INDEX paper_id IF NOT EXISTS FOR (p:Paper) ON (p.id)")

            for i in tqdm(range(0, len(papers), batch_size), desc="Loading papers"):
                batch = papers[i:i + batch_size]
                session.run("""
                    UNWIND $papers AS paper
                    CREATE (p:Paper {
                        id: paper.id,
                        title: paper.title,
                        category: paper.category
                    })
                """, papers=[{
                    "id": p["id"],
                    "title": p["title"][:200],
                    "category": p["category"]
                } for p in batch])

        return time.perf_counter() - start

    def create_citations(self, citations: List[tuple]) -> float:
        start = time.perf_counter()
        batch_size = CONFIG["batch_size"]

        with self.driver.session() as session:
            for i in tqdm(range(0, len(citations), batch_size), desc="Creating citations"):
                batch = citations[i:i + batch_size]
                session.run("""
                    UNWIND $edges AS edge
                    MATCH (a:Paper {id: edge.src}), (b:Paper {id: edge.tgt})
                    CREATE (a)-[:CITES]->(b)
                """, edges=[{"src": src, "tgt": tgt} for src, tgt in batch])

        return time.perf_counter() - start

    def create_authors(self, authors: List[dict], num_papers: int) -> float:
        start = time.perf_counter()
        batch_size = CONFIG["batch_size"]

        with self.driver.session() as session:
            session.run("CREATE INDEX author_id IF NOT EXISTS FOR (a:Author) ON (a.id)")

            for i in tqdm(range(0, len(authors), batch_size), desc="Creating authors"):
                batch = authors[i:i + batch_size]
                # Create authors
                session.run("""
                    UNWIND $authors AS author
                    CREATE (a:Author {
                        id: author.id,
                        name: author.name,
                        institution: author.institution
                    })
                """, authors=[{
                    "id": a["id"] + num_papers,
                    "name": a["name"],
                    "institution": a["institution"]
                } for a in batch])

                # Link to papers
                session.run("""
                    UNWIND $links AS link
                    MATCH (p:Paper {id: link.paper_id}), (a:Author {id: link.author_id})
                    CREATE (p)-[:AUTHORED_BY]->(a)
                """, links=[{
                    "paper_id": a["paper_id"],
                    "author_id": a["id"] + num_papers
                } for a in batch])

        return time.perf_counter() - start

    def run_query(self, name: str, query: str) -> QueryResult:
        mem_mon = self.get_memory_monitor()
        if mem_mon:
            mem_mon.start()

        start = time.perf_counter()
        try:
            with self.driver.session() as session:
                result = session.run(query)
                records = list(result)

            elapsed = (time.perf_counter() - start) * 1000

            if mem_mon:
                mem_mon.stop()

            return QueryResult(
                name=name,
                latency_ms=elapsed,
                rows=len(records),
                memory_mb=mem_mon.get_max_memory() if mem_mon else 0,
                success=True
            )
        except Exception as e:
            if mem_mon:
                mem_mon.stop()
            return QueryResult(name=name, latency_ms=0, success=False, error=str(e))

    def close(self):
        if self.driver:
            self.driver.close()


class FalkorDBAdapter(DatabaseAdapter):
    """Adapter for FalkorDB."""

    name = "FalkorDB"
    container_name = CONFIG["falkordb"]["container"]

    def __init__(self):
        self.db = None
        self.graph = None

    def connect(self) -> bool:
        if not HAS_FALKORDB:
            log.warning("FalkorDB driver not installed. Install with: pip install falkordb")
            return False

        try:
            self.db = FalkorDB(
                host=CONFIG["falkordb"]["host"],
                port=CONFIG["falkordb"]["port"]
            )
            # Delete existing graph and create new one
            try:
                old_graph = self.db.select_graph(CONFIG["falkordb"]["graph"])
                old_graph.delete()
            except:
                pass
            self.graph = self.db.select_graph(CONFIG["falkordb"]["graph"])
            return True
        except Exception as e:
            log.warning(f"FalkorDB connection failed: {e}")
            return False

    def clear(self):
        try:
            self.graph.delete()
            self.graph = self.db.select_graph(CONFIG["falkordb"]["graph"])
        except:
            pass

    def load_papers(self, papers: List[dict]) -> float:
        start = time.perf_counter()
        batch_size = CONFIG["batch_size"]

        # Create index
        self.graph.query("CREATE INDEX ON :Paper(id)")

        for i in tqdm(range(0, len(papers), batch_size), desc="Loading papers"):
            batch = papers[i:i + batch_size]
            self.graph.query("""
                UNWIND $batch as p
                CREATE (:Paper {
                    id: p.id,
                    title: p.title,
                    category: p.category
                })
            """, {"batch": [{
                "id": p["id"],
                "title": p["title"][:200],
                "category": p["category"]
            } for p in batch]})

        return time.perf_counter() - start

    def create_citations(self, citations: List[tuple]) -> float:
        start = time.perf_counter()
        batch_size = CONFIG["batch_size"]

        for i in tqdm(range(0, len(citations), batch_size), desc="Creating citations"):
            batch = citations[i:i + batch_size]
            self.graph.query("""
                UNWIND $edges AS edge
                MATCH (a:Paper {id: edge.src}), (b:Paper {id: edge.tgt})
                CREATE (a)-[:CITES]->(b)
            """, {"edges": [{"src": src, "tgt": tgt} for src, tgt in batch]})

        return time.perf_counter() - start

    def create_authors(self, authors: List[dict], num_papers: int) -> float:
        start = time.perf_counter()
        batch_size = CONFIG["batch_size"]

        self.graph.query("CREATE INDEX ON :Author(id)")

        for i in tqdm(range(0, len(authors), batch_size), desc="Creating authors"):
            batch = authors[i:i + batch_size]
            # Create authors
            self.graph.query("""
                UNWIND $authors as a
                CREATE (:Author {
                    id: a.id,
                    name: a.name,
                    institution: a.institution
                })
            """, {"authors": [{
                "id": a["id"] + num_papers,
                "name": a["name"],
                "institution": a["institution"]
            } for a in batch]})

            # Link to papers
            self.graph.query("""
                UNWIND $links AS link
                MATCH (p:Paper {id: link.paper_id}), (a:Author {id: link.author_id})
                CREATE (p)-[:AUTHORED_BY]->(a)
            """, {"links": [{
                "paper_id": a["paper_id"],
                "author_id": a["id"] + num_papers
            } for a in batch]})

        return time.perf_counter() - start

    def run_query(self, name: str, query: str) -> QueryResult:
        mem_mon = self.get_memory_monitor()
        if mem_mon:
            mem_mon.start()

        start = time.perf_counter()
        try:
            result = self.graph.query(query)
            elapsed = (time.perf_counter() - start) * 1000

            if mem_mon:
                mem_mon.stop()

            rows = len(result.result_set) if result.result_set else 0

            return QueryResult(
                name=name,
                latency_ms=elapsed,
                rows=rows,
                memory_mb=mem_mon.get_max_memory() if mem_mon else 0,
                success=True
            )
        except Exception as e:
            if mem_mon:
                mem_mon.stop()
            return QueryResult(name=name, latency_ms=0, success=False, error=str(e))

    def close(self):
        pass


# =============================================================================
# Benchmark Runner
# =============================================================================

# Standard queries (compatible with all databases)
BENCHMARK_QUERIES = {
    # Simple queries
    "count_papers": "MATCH (p:Paper) RETURN count(p)",
    "count_citations": "MATCH ()-[r:CITES]->() RETURN count(r)",
    "count_authors": "MATCH (a:Author) RETURN count(a)",

    # Traversal queries
    "1_hop": "MATCH (p:Paper)-[:CITES]->(c:Paper) WHERE p.id = 0 RETURN count(c)",
    "2_hop": "MATCH (p:Paper)-[:CITES]->(c1:Paper)-[:CITES]->(c2:Paper) WHERE p.id = 0 RETURN count(c2)",

    # Filter queries
    "filter_category": "MATCH (p:Paper) WHERE p.category = 'cs.LG' RETURN count(p)",
    "filter_with_rel": "MATCH (p:Paper)-[:CITES]->(c:Paper) WHERE p.category = 'cs.LG' RETURN p.id, count(c) LIMIT 10",

    # Aggregation queries
    "top_cited": "MATCH (p:Paper)<-[:CITES]-(c) RETURN p.id, count(c) AS citations ORDER BY citations DESC LIMIT 10",
    "institution_count": "MATCH (a:Author) RETURN a.institution, count(a) ORDER BY count(a) DESC LIMIT 5",
}

COMPLEX_QUERIES = {
    # Complex traversals
    "3_hop": "MATCH (a:Paper)-[:CITES]->(b:Paper)-[:CITES]->(c:Paper)-[:CITES]->(d:Paper) RETURN count(*)",

    # Analytical queries
    "citation_network": "MATCH (p:Paper)-[:CITES]->(c:Paper) RETURN p.category, count(c) ORDER BY count(c) DESC LIMIT 5",

    # Path queries (may timeout on large datasets)
    "shortest_path": "MATCH (a:Paper {id: 0}), (b:Paper {id: 100}) MATCH path = shortestPath((a)-[:CITES*]->(b)) RETURN path",
}


class BenchmarkRunner:
    """Orchestrates benchmark execution across multiple databases."""

    def __init__(
        self,
        num_papers: int = 1000,
        databases: List[str] = None,
        skip_complex: bool = False,
        output_dir: Optional[str] = None,
        warmup_iterations: int = 2,
        query_iterations: int = 5,
    ):
        self.num_papers = num_papers
        self.databases = databases or ["neuralgraph", "neo4j", "falkordb"]
        self.skip_complex = skip_complex
        self.output_dir = Path(output_dir) if output_dir else Path(tempfile.mkdtemp(prefix="benchmark_"))
        self.warmup_iterations = warmup_iterations
        self.query_iterations = query_iterations

        self.data: Optional[DataGenerator] = None
        self.results: Dict[str, BenchmarkResult] = {}

        self.adapters: Dict[str, type] = {
            "neuralgraph": NeuralGraphAdapter,
            "neo4j": Neo4jAdapter,
            "falkordb": FalkorDBAdapter,
        }

    def run(self) -> Dict[str, BenchmarkResult]:
        """Run the complete benchmark suite."""
        log.info("=" * 70)
        log.info("UNIFIED BENCHMARK: NeuralGraphDB vs FalkorDB vs Neo4j")
        log.info("=" * 70)
        log.info(f"Papers: {self.num_papers}")
        log.info(f"Databases: {', '.join(self.databases)}")
        log.info(f"Output: {self.output_dir}")
        log.info("=" * 70)

        # Generate data
        with Timer("Data Generation"):
            self.data = DataGenerator(num_papers=self.num_papers).generate()

        # Run benchmark for each database
        for db_name in self.databases:
            if db_name not in self.adapters:
                log.warning(f"Unknown database: {db_name}")
                continue

            log.info(f"\n{'='*70}")
            log.info(f"BENCHMARKING: {db_name.upper()}")
            log.info(f"{'='*70}")

            result = self._benchmark_database(db_name)
            if result:
                self.results[db_name] = result

        # Generate report
        self._generate_report()

        return self.results

    def _benchmark_database(self, db_name: str) -> Optional[BenchmarkResult]:
        """Run benchmark for a single database."""
        adapter_class = self.adapters[db_name]
        adapter = adapter_class()

        result = BenchmarkResult(
            database=db_name,
            config={
                "num_papers": self.num_papers,
                "avg_citations": self.data.avg_citations,
            }
        )

        # Connect
        if not adapter.connect():
            log.error(f"Failed to connect to {db_name}")
            return None

        try:
            # Clear existing data
            log.info("Clearing existing data...")
            adapter.clear()

            # Load papers
            mem_mon = adapter.get_memory_monitor()
            if mem_mon:
                mem_mon.start()

            with Timer("Load Papers") as t:
                adapter.load_papers(self.data.papers)
            result.timings["load_papers"] = t.elapsed

            if mem_mon:
                mem_mon.stop()
                result.memory["load_papers"] = mem_mon.get_max_memory()

            # Create citations
            mem_mon = adapter.get_memory_monitor()
            if mem_mon:
                mem_mon.start()

            with Timer("Create Citations") as t:
                adapter.create_citations(self.data.citations)
            result.timings["create_citations"] = t.elapsed

            if mem_mon:
                mem_mon.stop()
                result.memory["create_citations"] = mem_mon.get_max_memory()

            # Create authors
            mem_mon = adapter.get_memory_monitor()
            if mem_mon:
                mem_mon.start()

            with Timer("Create Authors") as t:
                adapter.create_authors(self.data.authors, self.num_papers)
            result.timings["create_authors"] = t.elapsed

            if mem_mon:
                mem_mon.stop()
                result.memory["create_authors"] = mem_mon.get_max_memory()

            # Record data stats
            result.data_stats = {
                "papers": len(self.data.papers),
                "citations": len(self.data.citations),
                "authors": len(self.data.authors),
            }

            # Run queries
            log.info("\nRunning benchmark queries...")

            all_queries = dict(BENCHMARK_QUERIES)
            if not self.skip_complex:
                all_queries.update(COMPLEX_QUERIES)

            for query_name, query in all_queries.items():
                log.info(f"  {query_name}...")

                # Warmup
                for _ in range(self.warmup_iterations):
                    adapter.run_query(query_name, query)

                # Timed runs
                latencies = []
                for _ in range(self.query_iterations):
                    qr = adapter.run_query(query_name, query)
                    if qr.success:
                        latencies.append(qr.latency_ms)

                if latencies:
                    avg_latency = sum(latencies) / len(latencies)
                    qr = adapter.run_query(query_name, query)  # Final run for memory
                    qr.latency_ms = avg_latency
                    result.queries.append(qr)
                    log.info(f"    -> {avg_latency:.2f}ms (rows: {qr.rows})")
                else:
                    log.warning(f"    -> FAILED")

            return result

        finally:
            adapter.close()

    def _generate_report(self):
        """Generate comparison report."""
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Save JSON results
        json_file = self.output_dir / "benchmark_results.json"
        with open(json_file, "w") as f:
            json.dump({
                db: r.to_dict() for db, r in self.results.items()
            }, f, indent=2)
        log.info(f"\nJSON results saved to: {json_file}")

        # Generate markdown report
        md_file = self.output_dir / "benchmark_report.md"
        self._generate_markdown_report(md_file)
        log.info(f"Markdown report saved to: {md_file}")

        # Print summary
        self._print_summary()

    def _generate_markdown_report(self, filepath: Path):
        """Generate a markdown comparison report."""
        with open(filepath, "w") as f:
            f.write("# Unified Benchmark Report\n\n")
            f.write(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            f.write(f"**Dataset:** {self.num_papers} papers, ~{len(self.data.citations)} citations, ~{len(self.data.authors)} authors\n\n")

            # Data Loading Table
            f.write("## Data Loading Performance\n\n")
            f.write("| Metric | " + " | ".join(self.results.keys()) + " |\n")
            f.write("|--------|" + "|".join(["--------"] * len(self.results)) + "|\n")

            for metric in ["load_papers", "create_citations", "create_authors"]:
                row = f"| {metric.replace('_', ' ').title()} |"
                for db, r in self.results.items():
                    val = r.timings.get(metric, 0)
                    row += f" {val:.3f}s |"
                f.write(row + "\n")

            # Total time
            row = "| **Total** |"
            for db, r in self.results.items():
                row += f" **{r.total_time():.3f}s** |"
            f.write(row + "\n\n")

            # Query Latency Table
            f.write("## Query Latency (ms)\n\n")
            f.write("| Query | " + " | ".join(self.results.keys()) + " |\n")
            f.write("|-------|" + "|".join(["--------"] * len(self.results)) + "|\n")

            # Collect all query names
            all_queries = set()
            for r in self.results.values():
                all_queries.update(q.name for q in r.queries)

            for query_name in sorted(all_queries):
                row = f"| {query_name} |"
                for db, r in self.results.items():
                    qr = next((q for q in r.queries if q.name == query_name), None)
                    if qr and qr.success:
                        row += f" {qr.latency_ms:.2f} |"
                    else:
                        row += " FAILED |"
                f.write(row + "\n")

            f.write("\n")

            # Memory Usage Table (if available)
            has_memory = any(r.memory for r in self.results.values())
            if has_memory:
                f.write("## Memory Usage (MB)\n\n")
                f.write("| Operation | " + " | ".join(self.results.keys()) + " |\n")
                f.write("|-----------|" + "|".join(["--------"] * len(self.results)) + "|\n")

                for metric in ["load_papers", "create_citations", "create_authors"]:
                    row = f"| {metric.replace('_', ' ').title()} |"
                    for db, r in self.results.items():
                        val = r.memory.get(metric, 0)
                        row += f" {val:.1f} |" if val > 0 else " N/A |"
                    f.write(row + "\n")

            # Speedup comparison
            if "neuralgraph" in self.results and len(self.results) > 1:
                f.write("\n## Speedup vs NeuralGraphDB\n\n")
                ng = self.results["neuralgraph"]

                for db, r in self.results.items():
                    if db == "neuralgraph":
                        continue

                    f.write(f"### {db}\n\n")

                    # Loading speedup
                    ng_total = ng.total_time()
                    other_total = r.total_time()
                    if other_total > 0:
                        speedup = other_total / ng_total
                        f.write(f"- **Data Loading:** NeuralGraphDB is {speedup:.1f}x faster\n")

                    # Query speedups
                    for qr in r.queries:
                        ng_qr = next((q for q in ng.queries if q.name == qr.name), None)
                        if ng_qr and ng_qr.success and qr.success and qr.latency_ms > 0:
                            speedup = qr.latency_ms / ng_qr.latency_ms
                            f.write(f"- **{qr.name}:** NeuralGraphDB is {speedup:.1f}x faster\n")

                    f.write("\n")

    def _print_summary(self):
        """Print a summary to console."""
        log.info("\n" + "=" * 70)
        log.info("BENCHMARK SUMMARY")
        log.info("=" * 70)

        # Header
        header = f"{'Metric':<25}"
        for db in self.results.keys():
            header += f" | {db:<15}"
        log.info(header)
        log.info("-" * 70)

        # Total load time
        row = f"{'Total Load Time':<25}"
        for db, r in self.results.items():
            row += f" | {r.total_time():<15.3f}"
        log.info(row)

        # Key queries
        key_queries = ["1_hop", "2_hop", "3_hop", "top_cited", "shortest_path"]
        for qname in key_queries:
            row = f"{qname:<25}"
            for db, r in self.results.items():
                qr = next((q for q in r.queries if q.name == qname), None)
                if qr and qr.success:
                    row += f" | {qr.latency_ms:<15.2f}"
                else:
                    row += f" | {'N/A':<15}"
            log.info(row)

        log.info("=" * 70)
        log.info(f"Full results: {self.output_dir}")


# =============================================================================
# Main
# =============================================================================

def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Unified Benchmark: NeuralGraphDB vs FalkorDB vs Neo4j"
    )
    parser.add_argument(
        "-n", "--num-papers",
        type=int,
        default=1000,
        help="Number of papers to generate (default: 1000)"
    )
    parser.add_argument(
        "--db", "--databases",
        type=str,
        default="neuralgraph,neo4j,falkordb",
        help="Comma-separated list of databases to benchmark (default: all)"
    )
    parser.add_argument(
        "--skip-complex",
        action="store_true",
        help="Skip complex queries (3-hop, shortest path)"
    )
    parser.add_argument(
        "-o", "--output",
        type=str,
        default=None,
        help="Output directory for results"
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=2,
        help="Number of warmup iterations per query"
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=5,
        help="Number of timed iterations per query"
    )

    args = parser.parse_args()

    databases = [db.strip().lower() for db in args.db.split(",")]

    runner = BenchmarkRunner(
        num_papers=args.num_papers,
        databases=databases,
        skip_complex=args.skip_complex,
        output_dir=args.output,
        warmup_iterations=args.warmup,
        query_iterations=args.iterations,
    )

    runner.run()


if __name__ == "__main__":
    main()
