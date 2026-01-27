#!/usr/bin/env python3
"""
LDBC-SNB Benchmark Suite for NeuralGraphDB

Comprehensive benchmark following LDBC-SNB Interactive workload specification.
Includes:
- 14 LDBC-SNB queries (IS1-IS7, IC1-IC7)
- Statistical analysis (p50, p95, p99)
- Multi-execution reproducibility
- Comparison with Neo4j and FalkorDB
- Paper-ready visualizations

Usage:
    python benchmarks/ldbc/ldbc_benchmark.py --sf SF1
    python benchmarks/ldbc/ldbc_benchmark.py --sf SF10 --executions 3
    python benchmarks/ldbc/ldbc_benchmark.py --db neuralgraph,neo4j,falkordb
"""

import json
import time
import csv
import statistics
import argparse
import logging
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
from abc import ABC, abstractmethod

import requests
from tqdm import tqdm

# Import query definitions
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ldbc_queries import LDBC_QUERIES, QueryParameterGenerator, LDBCQuery

# Conditional imports
try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False

try:
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

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

# Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S"
)
log = logging.getLogger(__name__)


# =============================================================================
# Configuration
# =============================================================================

CONFIG = {
    "neuralgraph": {
        "http_url": "http://localhost:3000/api/query",
    },
    "neo4j": {
        "uri": "bolt://localhost:7687",
        "auth": ("neo4j", "testpass123"),
    },
    "falkordb": {
        "host": "localhost",
        "port": 16379,
        "graph": "ldbc_bench",
    },
    "warmup_iterations": 3,
    "timed_iterations": 10,
    "executions": 3,  # Number of full benchmark executions
}


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class LatencyStats:
    """Statistical analysis of latencies."""
    raw_latencies: List[float] = field(default_factory=list)

    @property
    def count(self) -> int:
        return len(self.raw_latencies)

    @property
    def mean(self) -> float:
        return statistics.mean(self.raw_latencies) if self.raw_latencies else 0

    @property
    def median(self) -> float:
        return statistics.median(self.raw_latencies) if self.raw_latencies else 0

    @property
    def std(self) -> float:
        return statistics.stdev(self.raw_latencies) if len(self.raw_latencies) > 1 else 0

    @property
    def min(self) -> float:
        return min(self.raw_latencies) if self.raw_latencies else 0

    @property
    def max(self) -> float:
        return max(self.raw_latencies) if self.raw_latencies else 0

    def percentile(self, p: float) -> float:
        """Calculate percentile (p50, p95, p99, etc.)."""
        if not self.raw_latencies:
            return 0
        if HAS_NUMPY:
            return float(np.percentile(self.raw_latencies, p))
        else:
            sorted_data = sorted(self.raw_latencies)
            k = (len(sorted_data) - 1) * p / 100
            f = int(k)
            c = f + 1 if f + 1 < len(sorted_data) else f
            return sorted_data[f] + (k - f) * (sorted_data[c] - sorted_data[f])

    @property
    def p50(self) -> float:
        return self.percentile(50)

    @property
    def p95(self) -> float:
        return self.percentile(95)

    @property
    def p99(self) -> float:
        return self.percentile(99)

    def to_dict(self) -> Dict[str, float]:
        return {
            "count": self.count,
            "mean_ms": round(self.mean, 3),
            "median_ms": round(self.median, 3),
            "std_ms": round(self.std, 3),
            "min_ms": round(self.min, 3),
            "max_ms": round(self.max, 3),
            "p50_ms": round(self.p50, 3),
            "p95_ms": round(self.p95, 3),
            "p99_ms": round(self.p99, 3),
        }


@dataclass
class QueryResult:
    """Result of a single query execution."""
    query_id: str
    query_name: str
    category: str
    latency_ms: float
    rows: int = 0
    success: bool = True
    error: Optional[str] = None


@dataclass
class ExecutionResult:
    """Result of a single benchmark execution."""
    execution_id: int
    timestamp: str
    queries: Dict[str, LatencyStats] = field(default_factory=dict)
    ingestion_time_s: float = 0
    total_nodes: int = 0
    total_edges: int = 0


@dataclass
class BenchmarkReport:
    """Complete benchmark report for a database."""
    database: str
    scale_factor: str
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    executions: List[ExecutionResult] = field(default_factory=list)
    aggregated: Dict[str, LatencyStats] = field(default_factory=dict)

    def aggregate_results(self):
        """Aggregate results across all executions."""
        # Collect all latencies per query
        all_latencies: Dict[str, List[float]] = {}

        for exec_result in self.executions:
            for query_id, stats in exec_result.queries.items():
                if query_id not in all_latencies:
                    all_latencies[query_id] = []
                all_latencies[query_id].extend(stats.raw_latencies)

        # Create aggregated stats
        for query_id, latencies in all_latencies.items():
            self.aggregated[query_id] = LatencyStats(raw_latencies=latencies)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "database": self.database,
            "scale_factor": self.scale_factor,
            "timestamp": self.timestamp,
            "num_executions": len(self.executions),
            "aggregated": {qid: stats.to_dict() for qid, stats in self.aggregated.items()},
            "executions": [
                {
                    "execution_id": e.execution_id,
                    "timestamp": e.timestamp,
                    "ingestion_time_s": e.ingestion_time_s,
                    "total_nodes": e.total_nodes,
                    "total_edges": e.total_edges,
                    "queries": {qid: stats.to_dict() for qid, stats in e.queries.items()}
                }
                for e in self.executions
            ]
        }


# =============================================================================
# Database Adapters
# =============================================================================

class DatabaseAdapter(ABC):
    """Abstract base class for database adapters."""

    name: str = "base"

    @abstractmethod
    def connect(self) -> bool:
        pass

    @abstractmethod
    def clear(self):
        pass

    @abstractmethod
    def load_data(self, data_dir: Path) -> Tuple[float, int, int]:
        """Load data from CSV files. Returns (time_s, nodes, edges)."""
        pass

    @abstractmethod
    def run_query(self, query: str, params: Dict[str, Any]) -> Tuple[float, int, bool, Optional[str]]:
        """Run query. Returns (latency_ms, rows, success, error)."""
        pass

    @abstractmethod
    def close(self):
        pass


class NeuralGraphAdapter(DatabaseAdapter):
    """Adapter for NeuralGraphDB."""

    name = "NeuralGraphDB"

    def __init__(self):
        self.url = CONFIG["neuralgraph"]["http_url"]
        self.connected = False
        self.session = requests.Session()

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
        requests.post(self.url, json={"query": "MATCH (n) DETACH DELETE n"}, timeout=120)

    def _query(self, cypher: str, timeout: int = 300) -> dict:
        response = self.session.post(self.url, json={"query": cypher}, timeout=timeout)
        if response.status_code != 200:
            raise Exception(f"Query failed: {response.text}")
        return response.json()

    def load_data(self, data_dir: Path) -> Tuple[float, int, int]:
        start = time.perf_counter()
        total_nodes = 0
        total_edges = 0

        # Use larger batches for large datasets
        batch_size = 2000

        # Create indexes
        try:
            self._query("CREATE INDEX ON :Person(id)")
            self._query("CREATE INDEX ON :Post(id)")
            self._query("CREATE INDEX ON :Forum(id)")
            self._query("CREATE INDEX ON :Tag(id)")
        except:
            pass

        # Load Persons
        person_file = data_dir / "person.csv"
        if person_file.exists():
            with open(person_file) as f:
                reader = csv.DictReader(f)
                batch = []
                for row in reader:
                    batch.append({
                        "id": int(row["id"]),
                        "firstName": row["firstName"],
                        "lastName": row["lastName"],
                        "gender": row.get("gender", ""),
                        "birthday": row.get("birthday", ""),
                        "creationDate": row.get("creationDate", ""),
                    })
                    if len(batch) >= batch_size:
                        self._batch_create_nodes("Person", batch)
                        total_nodes += len(batch)
                        batch = []
                if batch:
                    self._batch_create_nodes("Person", batch)
                    total_nodes += len(batch)

        # Load KNOWS
        knows_file = data_dir / "person_knows_person.csv"
        if knows_file.exists():
            with open(knows_file) as f:
                reader = csv.DictReader(f)
                batch = []
                for row in reader:
                    batch.append({
                        "src": int(row["Person1Id"]),
                        "tgt": int(row["Person2Id"]),
                        "creationDate": row.get("creationDate", "")
                    })
                    if len(batch) >= batch_size:
                        self._batch_create_edges("Person", "Person", "KNOWS", batch)
                        total_edges += len(batch)
                        batch = []
                if batch:
                    self._batch_create_edges("Person", "Person", "KNOWS", batch)
                    total_edges += len(batch)

        # Load Posts (simplified - just the basic structure)
        post_file = data_dir / "post.csv"
        if post_file.exists():
            with open(post_file) as f:
                reader = csv.DictReader(f)
                batch = []
                for row in reader:
                    batch.append({
                        "id": int(row["id"]),
                        "content": row.get("content", "")[:200],
                        "creationDate": row.get("creationDate", ""),
                    })
                    if len(batch) >= 500:
                        self._batch_create_nodes("Message", batch)
                        total_nodes += len(batch)
                        batch = []
                if batch:
                    self._batch_create_nodes("Message", batch)
                    total_nodes += len(batch)

        # Load Tags
        tag_file = data_dir / "tag.csv"
        if tag_file.exists():
            with open(tag_file) as f:
                reader = csv.DictReader(f)
                batch = []
                for row in reader:
                    batch.append({
                        "id": int(row["id"]),
                        "name": row["name"],
                    })
                    if len(batch) >= 100:
                        self._batch_create_nodes("Tag", batch)
                        total_nodes += len(batch)
                        batch = []
                if batch:
                    self._batch_create_nodes("Tag", batch)
                    total_nodes += len(batch)

        # Load HAS_CREATOR
        creator_file = data_dir / "post_hasCreator_person.csv"
        if creator_file.exists():
            with open(creator_file) as f:
                reader = csv.DictReader(f)
                batch = []
                for row in reader:
                    batch.append({
                        "src": int(row["PostId"]),
                        "tgt": int(row["PersonId"]),
                    })
                    if len(batch) >= 500:
                        self._batch_create_edges("Message", "Person", "HAS_CREATOR", batch)
                        total_edges += len(batch)
                        batch = []
                if batch:
                    self._batch_create_edges("Message", "Person", "HAS_CREATOR", batch)
                    total_edges += len(batch)

        elapsed = time.perf_counter() - start
        return elapsed, total_nodes, total_edges

    def _batch_create_nodes(self, label: str, batch: List[Dict]):
        batch_json = json.dumps(batch)
        props = ", ".join([f"{k}: item.{k}" for k in batch[0].keys()])
        query = f"""
        WITH {batch_json} AS items
        UNWIND items AS item
        CREATE (n:{label} {{{props}}})
        """
        self._query(query)

    def _batch_create_edges(self, src_label: str, tgt_label: str, rel_type: str, batch: List[Dict]):
        batch_json = json.dumps(batch)
        query = f"""
        WITH {batch_json} AS edges
        UNWIND edges AS edge
        MATCH (a:{src_label}), (b:{tgt_label})
        WHERE a.id = edge.src AND b.id = edge.tgt
        CREATE (a)-[:{rel_type}]->(b)
        """
        self._query(query)

    def run_query(self, query: str, params: Dict[str, Any]) -> Tuple[float, int, bool, Optional[str]]:
        # Substitute parameters into query
        for key, value in params.items():
            if isinstance(value, str):
                query = query.replace(f"${key}", f'"{value}"')
            else:
                query = query.replace(f"${key}", str(value))

        start = time.perf_counter()
        try:
            result = self._query(query)
            elapsed = (time.perf_counter() - start) * 1000

            rows = 0
            if isinstance(result, dict) and "result" in result:
                r = result["result"]
                rows = r.get("count", len(r.get("rows", [])))

            return elapsed, rows, True, None
        except Exception as e:
            elapsed = (time.perf_counter() - start) * 1000
            return elapsed, 0, False, str(e)

    def close(self):
        pass


class Neo4jAdapter(DatabaseAdapter):
    """Adapter for Neo4j."""

    name = "Neo4j"

    def __init__(self):
        self.driver = None

    def connect(self) -> bool:
        if not HAS_NEO4J:
            log.warning("Neo4j driver not installed")
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

    def load_data(self, data_dir: Path) -> Tuple[float, int, int]:
        start = time.perf_counter()
        total_nodes = 0
        total_edges = 0

        with self.driver.session() as session:
            # Create indexes
            session.run("CREATE INDEX person_id IF NOT EXISTS FOR (p:Person) ON (p.id)")
            session.run("CREATE INDEX message_id IF NOT EXISTS FOR (m:Message) ON (m.id)")

            # Load persons
            person_file = data_dir / "person.csv"
            if person_file.exists():
                with open(person_file) as f:
                    reader = csv.DictReader(f)
                    batch = []
                    for row in reader:
                        batch.append({
                            "id": int(row["id"]),
                            "firstName": row["firstName"],
                            "lastName": row["lastName"],
                        })
                        if len(batch) >= 500:
                            session.run("""
                                UNWIND $batch AS item
                                CREATE (p:Person {id: item.id, firstName: item.firstName, lastName: item.lastName})
                            """, batch=batch)
                            total_nodes += len(batch)
                            batch = []
                    if batch:
                        session.run("""
                            UNWIND $batch AS item
                            CREATE (p:Person {id: item.id, firstName: item.firstName, lastName: item.lastName})
                        """, batch=batch)
                        total_nodes += len(batch)

            # Load KNOWS
            knows_file = data_dir / "person_knows_person.csv"
            if knows_file.exists():
                with open(knows_file) as f:
                    reader = csv.DictReader(f)
                    batch = []
                    for row in reader:
                        batch.append({
                            "src": int(row["Person1Id"]),
                            "tgt": int(row["Person2Id"]),
                        })
                        if len(batch) >= 500:
                            session.run("""
                                UNWIND $batch AS edge
                                MATCH (a:Person {id: edge.src}), (b:Person {id: edge.tgt})
                                CREATE (a)-[:KNOWS]->(b)
                            """, batch=batch)
                            total_edges += len(batch)
                            batch = []
                    if batch:
                        session.run("""
                            UNWIND $batch AS edge
                            MATCH (a:Person {id: edge.src}), (b:Person {id: edge.tgt})
                            CREATE (a)-[:KNOWS]->(b)
                        """, batch=batch)
                        total_edges += len(batch)

        elapsed = time.perf_counter() - start
        return elapsed, total_nodes, total_edges

    def run_query(self, query: str, params: Dict[str, Any]) -> Tuple[float, int, bool, Optional[str]]:
        start = time.perf_counter()
        try:
            with self.driver.session() as session:
                result = session.run(query, **params)
                records = list(result)
            elapsed = (time.perf_counter() - start) * 1000
            return elapsed, len(records), True, None
        except Exception as e:
            elapsed = (time.perf_counter() - start) * 1000
            return elapsed, 0, False, str(e)

    def close(self):
        if self.driver:
            self.driver.close()


class FalkorDBAdapter(DatabaseAdapter):
    """Adapter for FalkorDB."""

    name = "FalkorDB"

    def __init__(self):
        self.db = None
        self.graph = None

    def connect(self) -> bool:
        if not HAS_FALKORDB:
            log.warning("FalkorDB driver not installed")
            return False
        try:
            self.db = FalkorDB(
                host=CONFIG["falkordb"]["host"],
                port=CONFIG["falkordb"]["port"]
            )
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

    def load_data(self, data_dir: Path) -> Tuple[float, int, int]:
        start = time.perf_counter()
        total_nodes = 0
        total_edges = 0

        # Create index
        self.graph.query("CREATE INDEX ON :Person(id)")

        # Load persons
        person_file = data_dir / "person.csv"
        if person_file.exists():
            with open(person_file) as f:
                reader = csv.DictReader(f)
                batch = []
                for row in reader:
                    batch.append({
                        "id": int(row["id"]),
                        "firstName": row["firstName"],
                        "lastName": row["lastName"],
                    })
                    if len(batch) >= 500:
                        self.graph.query("""
                            UNWIND $batch AS item
                            CREATE (:Person {id: item.id, firstName: item.firstName, lastName: item.lastName})
                        """, {"batch": batch})
                        total_nodes += len(batch)
                        batch = []
                if batch:
                    self.graph.query("""
                        UNWIND $batch AS item
                        CREATE (:Person {id: item.id, firstName: item.firstName, lastName: item.lastName})
                    """, {"batch": batch})
                    total_nodes += len(batch)

        # Load KNOWS
        knows_file = data_dir / "person_knows_person.csv"
        if knows_file.exists():
            with open(knows_file) as f:
                reader = csv.DictReader(f)
                batch = []
                for row in reader:
                    batch.append({
                        "src": int(row["Person1Id"]),
                        "tgt": int(row["Person2Id"]),
                    })
                    if len(batch) >= 500:
                        self.graph.query("""
                            UNWIND $batch AS edge
                            MATCH (a:Person {id: edge.src}), (b:Person {id: edge.tgt})
                            CREATE (a)-[:KNOWS]->(b)
                        """, {"batch": batch})
                        total_edges += len(batch)
                        batch = []
                if batch:
                    self.graph.query("""
                        UNWIND $batch AS edge
                        MATCH (a:Person {id: edge.src}), (b:Person {id: edge.tgt})
                        CREATE (a)-[:KNOWS]->(b)
                    """, {"batch": batch})
                    total_edges += len(batch)

        elapsed = time.perf_counter() - start
        return elapsed, total_nodes, total_edges

    def run_query(self, query: str, params: Dict[str, Any]) -> Tuple[float, int, bool, Optional[str]]:
        start = time.perf_counter()
        try:
            result = self.graph.query(query, params)
            elapsed = (time.perf_counter() - start) * 1000
            rows = len(result.result_set) if result.result_set else 0
            return elapsed, rows, True, None
        except Exception as e:
            elapsed = (time.perf_counter() - start) * 1000
            return elapsed, 0, False, str(e)

    def close(self):
        pass


# =============================================================================
# Benchmark Runner
# =============================================================================

class LDBCBenchmark:
    """LDBC-SNB Benchmark Runner."""

    ADAPTERS = {
        "neuralgraph": NeuralGraphAdapter,
        "neo4j": Neo4jAdapter,
        "falkordb": FalkorDBAdapter,
    }

    def __init__(
        self,
        scale_factor: str = "SF1",
        databases: List[str] = None,
        data_dir: str = None,
        output_dir: str = None,
        num_executions: int = 3,
        warmup: int = 3,
        iterations: int = 10,
    ):
        self.scale_factor = scale_factor
        self.databases = databases or ["neuralgraph"]
        self.data_dir = Path(data_dir) if data_dir else Path(f"benchmarks/ldbc/data/{scale_factor}")
        self.output_dir = Path(output_dir) if output_dir else Path(f"benchmarks/ldbc/results/{scale_factor}")
        self.num_executions = num_executions
        self.warmup = warmup
        self.iterations = iterations

        self.results: Dict[str, BenchmarkReport] = {}

        # Load metadata for parameter generation
        self._load_metadata()

    def _load_metadata(self):
        """Load data metadata for query parameter generation."""
        metadata_file = self.data_dir / "metadata.json"
        if metadata_file.exists():
            with open(metadata_file) as f:
                self.metadata = json.load(f)
        else:
            self.metadata = {
                "num_persons": 1000,
                "first_names": ["Alice", "Bob", "Charlie"],
                "tags": ["AI", "ML", "Data"],
            }

        # Create parameter generator
        self.param_gen = QueryParameterGenerator(
            persons=list(range(self.metadata.get("num_persons", 1000))),
            messages=list(range(self.metadata.get("num_messages", 1000))),
            first_names=self.metadata.get("first_names", ["Alice"]),
            tags=self.metadata.get("tags", ["AI"]),
        )

    def run(self) -> Dict[str, BenchmarkReport]:
        """Run the complete benchmark suite."""
        log.info("=" * 70)
        log.info("LDBC-SNB BENCHMARK SUITE")
        log.info("=" * 70)
        log.info(f"Scale Factor: {self.scale_factor}")
        log.info(f"Databases: {', '.join(self.databases)}")
        log.info(f"Executions: {self.num_executions}")
        log.info(f"Data Directory: {self.data_dir}")
        log.info("=" * 70)

        for db_name in self.databases:
            if db_name not in self.ADAPTERS:
                log.warning(f"Unknown database: {db_name}")
                continue

            log.info(f"\n{'='*70}")
            log.info(f"BENCHMARKING: {db_name.upper()}")
            log.info(f"{'='*70}")

            report = self._benchmark_database(db_name)
            if report:
                report.aggregate_results()
                self.results[db_name] = report

        # Generate outputs
        self._save_results()
        self._generate_report()
        if HAS_MATPLOTLIB:
            self._generate_visualizations()

        return self.results

    def _benchmark_database(self, db_name: str) -> Optional[BenchmarkReport]:
        """Run benchmark for a single database."""
        adapter_class = self.ADAPTERS[db_name]
        adapter = adapter_class()

        if not adapter.connect():
            log.error(f"Failed to connect to {db_name}")
            return None

        report = BenchmarkReport(
            database=db_name,
            scale_factor=self.scale_factor,
        )

        try:
            for exec_num in range(self.num_executions):
                log.info(f"\n--- Execution {exec_num + 1}/{self.num_executions} ---")

                exec_result = ExecutionResult(
                    execution_id=exec_num + 1,
                    timestamp=datetime.now().isoformat(),
                )

                # Clear and reload data
                log.info("Clearing database...")
                adapter.clear()

                log.info("Loading data...")
                ingestion_time, nodes, edges = adapter.load_data(self.data_dir)
                exec_result.ingestion_time_s = ingestion_time
                exec_result.total_nodes = nodes
                exec_result.total_edges = edges
                log.info(f"  Loaded {nodes:,} nodes, {edges:,} edges in {ingestion_time:.2f}s")

                # Run queries
                log.info("Running LDBC queries...")
                for query_id, query_def in LDBC_QUERIES.items():
                    stats = self._run_query(adapter, query_def)
                    exec_result.queries[query_id] = stats

                report.executions.append(exec_result)

            return report

        finally:
            adapter.close()

    def _run_query(self, adapter: DatabaseAdapter, query_def: LDBCQuery) -> LatencyStats:
        """Run a single query multiple times and collect statistics."""
        stats = LatencyStats()

        # Get query template
        query = query_def.cypher_template.strip()

        # Warmup
        for _ in range(self.warmup):
            params = self.param_gen.get_params(query_def.params_generator) if query_def.params_generator else {}
            adapter.run_query(query, params)

        # Timed iterations
        for _ in range(self.iterations):
            params = self.param_gen.get_params(query_def.params_generator) if query_def.params_generator else {}
            latency, rows, success, error = adapter.run_query(query, params)

            if success:
                stats.raw_latencies.append(latency)
            else:
                log.debug(f"  {query_def.id} failed: {error}")

        if stats.raw_latencies:
            log.info(f"  {query_def.id} ({query_def.name}): p50={stats.p50:.2f}ms, p95={stats.p95:.2f}ms, p99={stats.p99:.2f}ms")
        else:
            log.warning(f"  {query_def.id} ({query_def.name}): FAILED")

        return stats

    def _save_results(self):
        """Save results to JSON."""
        self.output_dir.mkdir(parents=True, exist_ok=True)

        for db_name, report in self.results.items():
            filepath = self.output_dir / f"{db_name}_results.json"
            with open(filepath, "w") as f:
                json.dump(report.to_dict(), f, indent=2)
            log.info(f"Saved results: {filepath}")

        # Combined results
        combined = self.output_dir / "combined_results.json"
        with open(combined, "w") as f:
            json.dump({db: r.to_dict() for db, r in self.results.items()}, f, indent=2)

    def _generate_report(self):
        """Generate markdown report."""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        filepath = self.output_dir / "benchmark_report.md"

        with open(filepath, "w") as f:
            f.write("# LDBC-SNB Benchmark Report\n\n")
            f.write(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            f.write(f"**Scale Factor:** {self.scale_factor}\n\n")
            f.write(f"**Executions:** {self.num_executions}\n\n")

            # Summary table
            f.write("## Query Latency Summary (ms)\n\n")
            f.write("| Query | Category |")
            for db in self.results.keys():
                f.write(f" {db} p50 | {db} p95 | {db} p99 |")
            f.write("\n")

            f.write("|-------|----------|")
            for _ in self.results.keys():
                f.write("--------|--------|--------|")
            f.write("\n")

            for query_id in LDBC_QUERIES.keys():
                query = LDBC_QUERIES[query_id]
                f.write(f"| **{query_id}** | {query.category} |")
                for db, report in self.results.items():
                    if query_id in report.aggregated:
                        stats = report.aggregated[query_id]
                        f.write(f" {stats.p50:.2f} | {stats.p95:.2f} | {stats.p99:.2f} |")
                    else:
                        f.write(" N/A | N/A | N/A |")
                f.write("\n")

            f.write("\n")

            # Detailed stats per database
            for db_name, report in self.results.items():
                f.write(f"\n## {db_name} Detailed Statistics\n\n")

                # Ingestion stats
                if report.executions:
                    avg_ingestion = statistics.mean([e.ingestion_time_s for e in report.executions])
                    f.write(f"**Average Ingestion Time:** {avg_ingestion:.2f}s\n\n")

                f.write("| Query | Mean | Median | Std | Min | Max | p50 | p95 | p99 |\n")
                f.write("|-------|------|--------|-----|-----|-----|-----|-----|-----|\n")

                for query_id, stats in report.aggregated.items():
                    f.write(f"| {query_id} | {stats.mean:.2f} | {stats.median:.2f} | "
                            f"{stats.std:.2f} | {stats.min:.2f} | {stats.max:.2f} | "
                            f"{stats.p50:.2f} | {stats.p95:.2f} | {stats.p99:.2f} |\n")

                f.write("\n")

        log.info(f"Generated report: {filepath}")

    def _generate_visualizations(self):
        """Generate paper-ready visualizations."""
        if not HAS_MATPLOTLIB:
            log.warning("Matplotlib not available, skipping visualizations")
            return

        self.output_dir.mkdir(parents=True, exist_ok=True)

        # 1. Latency comparison bar chart
        self._plot_latency_comparison()

        # 2. Percentile distribution
        self._plot_percentile_distribution()

        # 3. Query category comparison
        self._plot_category_comparison()

    def _plot_latency_comparison(self):
        """Plot p50 latency comparison across databases."""
        fig, ax = plt.subplots(figsize=(14, 6))

        query_ids = list(LDBC_QUERIES.keys())
        x = range(len(query_ids))
        width = 0.25
        colors = ['#2ecc71', '#3498db', '#e74c3c']

        for i, (db_name, report) in enumerate(self.results.items()):
            latencies = []
            for qid in query_ids:
                if qid in report.aggregated:
                    latencies.append(report.aggregated[qid].p50)
                else:
                    latencies.append(0)

            offset = (i - len(self.results) / 2 + 0.5) * width
            bars = ax.bar([xi + offset for xi in x], latencies, width,
                         label=db_name, color=colors[i % len(colors)], alpha=0.8)

        ax.set_xlabel('Query', fontsize=12)
        ax.set_ylabel('Latency (ms) - p50', fontsize=12)
        ax.set_title(f'LDBC-SNB Query Latency Comparison ({self.scale_factor})', fontsize=14)
        ax.set_xticks(x)
        ax.set_xticklabels(query_ids, rotation=45, ha='right')
        ax.legend()
        ax.grid(axis='y', alpha=0.3)

        plt.tight_layout()
        filepath = self.output_dir / "latency_comparison.png"
        plt.savefig(filepath, dpi=150)
        plt.close()
        log.info(f"Generated: {filepath}")

    def _plot_percentile_distribution(self):
        """Plot percentile distribution for each database."""
        fig, axes = plt.subplots(1, len(self.results), figsize=(6 * len(self.results), 5))
        if len(self.results) == 1:
            axes = [axes]

        for ax, (db_name, report) in zip(axes, self.results.items()):
            query_ids = list(report.aggregated.keys())
            p50s = [report.aggregated[qid].p50 for qid in query_ids]
            p95s = [report.aggregated[qid].p95 for qid in query_ids]
            p99s = [report.aggregated[qid].p99 for qid in query_ids]

            x = range(len(query_ids))
            ax.fill_between(x, p50s, p99s, alpha=0.2, color='blue', label='p50-p99')
            ax.fill_between(x, p50s, p95s, alpha=0.3, color='blue', label='p50-p95')
            ax.plot(x, p50s, 'b-', linewidth=2, label='p50')
            ax.plot(x, p95s, 'b--', linewidth=1, label='p95')
            ax.plot(x, p99s, 'b:', linewidth=1, label='p99')

            ax.set_xlabel('Query')
            ax.set_ylabel('Latency (ms)')
            ax.set_title(f'{db_name}')
            ax.set_xticks(x)
            ax.set_xticklabels(query_ids, rotation=45, ha='right', fontsize=8)
            ax.legend(loc='upper right')
            ax.grid(alpha=0.3)

        plt.suptitle(f'Latency Percentile Distribution ({self.scale_factor})', fontsize=14)
        plt.tight_layout()
        filepath = self.output_dir / "percentile_distribution.png"
        plt.savefig(filepath, dpi=150)
        plt.close()
        log.info(f"Generated: {filepath}")

    def _plot_category_comparison(self):
        """Plot comparison by query category (IS vs IC)."""
        fig, ax = plt.subplots(figsize=(10, 6))

        categories = ["IS", "IC"]
        db_names = list(self.results.keys())
        x = range(len(categories))
        width = 0.25
        colors = ['#2ecc71', '#3498db', '#e74c3c']

        for i, db_name in enumerate(db_names):
            report = self.results[db_name]
            category_latencies = []

            for cat in categories:
                cat_queries = [qid for qid in report.aggregated.keys() if qid.startswith(cat)]
                if cat_queries:
                    avg_p50 = statistics.mean([report.aggregated[qid].p50 for qid in cat_queries])
                    category_latencies.append(avg_p50)
                else:
                    category_latencies.append(0)

            offset = (i - len(db_names) / 2 + 0.5) * width
            ax.bar([xi + offset for xi in x], category_latencies, width,
                   label=db_name, color=colors[i % len(colors)], alpha=0.8)

        ax.set_xlabel('Query Category', fontsize=12)
        ax.set_ylabel('Average Latency (ms) - p50', fontsize=12)
        ax.set_title(f'Latency by Query Category ({self.scale_factor})', fontsize=14)
        ax.set_xticks(x)
        ax.set_xticklabels(['Interactive Short (IS1-IS7)', 'Interactive Complex (IC1-IC7)'])
        ax.legend()
        ax.grid(axis='y', alpha=0.3)

        plt.tight_layout()
        filepath = self.output_dir / "category_comparison.png"
        plt.savefig(filepath, dpi=150)
        plt.close()
        log.info(f"Generated: {filepath}")


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="LDBC-SNB Benchmark Suite")
    parser.add_argument(
        "--sf", "--scale-factor",
        type=str,
        default="SF1",
        help="Scale factor (SF0.1, SF1, SF10, SF100)"
    )
    parser.add_argument(
        "--db", "--databases",
        type=str,
        default="neuralgraph",
        help="Comma-separated list of databases"
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default=None,
        help="Data directory (default: benchmarks/ldbc/data/{sf})"
    )
    parser.add_argument(
        "-o", "--output",
        type=str,
        default=None,
        help="Output directory"
    )
    parser.add_argument(
        "-e", "--executions",
        type=int,
        default=3,
        help="Number of full benchmark executions"
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=3,
        help="Warmup iterations per query"
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=10,
        help="Timed iterations per query"
    )

    args = parser.parse_args()

    databases = [db.strip().lower() for db in args.db.split(",")]

    benchmark = LDBCBenchmark(
        scale_factor=args.sf,
        databases=databases,
        data_dir=args.data_dir,
        output_dir=args.output,
        num_executions=args.executions,
        warmup=args.warmup,
        iterations=args.iterations,
    )

    benchmark.run()


if __name__ == "__main__":
    main()
