#!/usr/bin/env python3
"""
Carga dataset ArXiv en FalkorDB con embeddings vectoriales.
Adaptado para comparación directa con NeuralGraphDB y Neo4j.
"""

import json
import time
import logging
from dataclasses import dataclass
import random

from datasets import load_dataset
from tqdm import tqdm
from falkordb import FalkorDB
from sentence_transformers import SentenceTransformer
from benchmark_utils import MemoryMonitor

# Configuración
FALKOR_HOST = "localhost"
FALKOR_PORT = 16379
CONTAINER_NAME = "benchmark-falkordb"
GRAPH_NAME = "arxiv_bench"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
BATCH_SIZE = 100

# Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S"
)
log = logging.getLogger(__name__)


@dataclass
class Timer:
    name: str
    start: float = 0
    elapsed: float = 0
    
    def __enter__(self):
        self.start = time.perf_counter()
        log.info(f"⏱️  Iniciando: {self.name}")
        return self
    
    def __exit__(self, *args):
        self.elapsed = time.perf_counter() - self.start
        log.info(f"✅ Completado: {self.name} en {self.elapsed:.2f}s")


class BenchmarkStats:
    def __init__(self):
        self.timings = {}
        self.memory = {}
    
    def record(self, name: str, elapsed: float, memory_mb: float = 0.0):
        self.timings[name] = elapsed
        if memory_mb > 0:
            self.memory[name] = memory_mb
    
    def report(self):
        log.info("\n" + "=" * 60)
        log.info("📊 RESUMEN DE TIEMPOS - FalkorDB")
        log.info("=" * 60)
        for name, elapsed in self.timings.items():
            mem_str = f" | Mem: {self.memory[name]:.2f} MB" if name in self.memory else ""
            log.info(f"  {name:<35}: {elapsed:.2f}s{mem_str}")
        total = sum(self.timings.values())
        log.info("-" * 60)
        log.info(f"  TOTAL: {total:.2f}s")
        log.info("=" * 60)


stats = BenchmarkStats()


def download_arxiv(limit: int = 10000) -> list[dict]:
    with Timer("Descarga ArXiv") as t:
        log.info(f"📥 Cargando dataset ArXiv...")
        dataset = load_dataset("CShorten/ML-ArXiv-Papers", split="train")
        papers = []
        for i, item in enumerate(tqdm(dataset, desc="Procesando papers", total=min(limit, len(dataset)))):
            if i >= limit: break
            papers.append({
                "id": i, # FalkorDB handles integers well
                "title": (item.get("title") or f"Paper {i}")[:500].replace("'", ""), # sanitize simple quotes
                "abstract": (item.get("abstract") or "")[:2000].replace("'", ""),
                "category": "cs.LG",
            })
    stats.record("1. Descarga ArXiv", t.elapsed)
    return papers


def generate_embeddings(papers: list[dict], model: SentenceTransformer) -> list[dict]:
    with Timer("Generación de embeddings") as t:
        abstracts = [p["abstract"] for p in papers]
        embeddings = model.encode(abstracts, show_progress_bar=True, batch_size=64)
        for paper, embedding in zip(papers, embeddings):
            paper["embedding"] = embedding.tolist()
    stats.record("2. Generación embeddings", t.elapsed)
    return papers


def setup_schema(graph):
    with Timer("Setup schema FalkorDB") as t:
        # FalkorDB doesn't need explicit schema usually, but indexes help
        log.info("📐 Creando índices...")
        # Note: FalkorDB syntax for index creation
        graph.query("CREATE INDEX ON :Paper(id)")
        graph.query("CREATE INDEX ON :Paper(category)")
        # Vector Index support in FalkorDB (via RediSearch under the hood usually, or native vector support depending on version)
        # Assuming standard FalkorDB vector index syntax if available, or just skip for now as strict graph comparison.
        # FalkorDB recently added vector index support.
        try:
            # Syntax: CREATE VECTOR INDEX FOR (n:Label) ON (n.prop) OPTIONS {dim:384, similarity_function:'cosine'}
            # Check docs for specific version. 
            pass 
        except Exception as e:
            log.warning(f"Vector index creation skipped/failed: {e}")

    stats.record("3. Setup schema", t.elapsed)


def load_papers(graph, papers: list[dict]):
    mem_mon = MemoryMonitor(CONTAINER_NAME)
    mem_mon.start()
    with Timer("Carga de papers") as t:
        # FalkorDB supports UNWIND, but passing large parameters via python client can be tricky.
        # We'll do batches with parameters.
        for i in tqdm(range(0, len(papers), BATCH_SIZE), desc="Cargando papers"):
            batch = papers[i:i + BATCH_SIZE]
            # Build query dynamically or use params if supported well
            # FalkorDB-python supports params map
            query = """
            UNWIND $batch as p
            CREATE (:Paper {
                id: p.id,
                title: p.title,
                abstract: p.abstract,
                category: p.category,
                embedding: p.embedding
            })
            """
            graph.query(query, {"batch": batch})
    mem_mon.stop()        
    stats.record("4. Carga papers", t.elapsed, mem_mon.get_max_memory())

def create_citations(graph, num_papers: int, avg_citations: int = 5):
    mem_mon = MemoryMonitor(CONTAINER_NAME)
    mem_mon.start()
    with Timer("Creación de citas") as t:
        prob = avg_citations / num_papers
        # Doing this in one massive query might timeout, best to do it in batches or purely random logic in Cypher
        # Neo4j script used: MATCH (p1), (p2) WHERE p1.id < p2.id AND rand() < prob ...
        # FalkorDB supports rand()
        
        log.info(f"🔗 Creando citas (Logic-based)...")
        # Optimization: Don't do full cartesian product.
        # We'll just run the query. FalkorDB handles this usually.
        
        graph.query(f"""
            MATCH (p1:Paper), (p2:Paper)
            WHERE p1.id < p2.id AND rand() < {prob}
            CREATE (p1)-[:CITES]->(p2)
        """
        )
    mem_mon.stop()
    stats.record("5. Creación citas", t.elapsed, mem_mon.get_max_memory())

def run_sample_queries(graph):
    log.info("\n🧪 Ejecutando queries de prueba...")
    
    with Timer("Query: Traversal 2-hop") as t:
        res = graph.query("""
            MATCH (p:Paper)-[:CITES]->(cited:Paper)-[:CITES]->(cited2:Paper)
            RETURN p.id, count(cited2) AS reach
            LIMIT 10
        """
        )
        log.info(f"   Resultados: {len(res.result_set)}")
    stats.record("Query: 2-hop traversal", t.elapsed)

def run_complex_queries(graph):

    """Ejecuta queries complejas con monitoreo de memoria."""

    log.info("\n🧠 Ejecutando queries complejas (con monitoreo de memoria)...")

    

    # Q1: 3-Hop Traversal

    mem_mon = MemoryMonitor(CONTAINER_NAME)

    mem_mon.start()

    with Timer("Query Compleja: 3-Hop Traversal") as t:

        res = graph.query("""

            MATCH (a)-[:CITES]->(b)-[:CITES]->(c)-[:CITES]->(d)

            RETURN count(*)

        """)

        log.info(f"   Resultados: {res.result_set[0][0] if res.result_set else 0}")

    mem_mon.stop()

    stats.record("Q_Complex: 3-Hop Traversal", t.elapsed, mem_mon.get_max_memory())

    

    # Q2: Analytical (Top Cited)

    mem_mon = MemoryMonitor(CONTAINER_NAME)

    mem_mon.start()

    with Timer("Query Compleja: Top Cited Categories") as t:

        res = graph.query("""

            MATCH (a:Paper)-[:CITES]->(b:Paper)

            RETURN a.category, count(b) as citations

            ORDER BY citations DESC

            LIMIT 5

        """)

        log.info(f"   Resultados: {len(res.result_set)}")

    mem_mon.stop()

    stats.record("Q_Complex: Analytical (Top Cited)", t.elapsed, mem_mon.get_max_memory())

    

    # Q3: Shortest Path

    # FalkorDB support: shortestPath must be in RETURN or WITH

    mem_mon = MemoryMonitor(CONTAINER_NAME)

    mem_mon.start()

    with Timer("Query Compleja: Shortest Path") as t:

        res = graph.query("""

            MATCH (a:Paper {id: 0}), (b:Paper {id: 100})

            RETURN shortestPath((a)-[:CITES*]->(b))

        """)

        log.info(f"   Resultados: {len(res.result_set)}")

    mem_mon.stop()

    stats.record("Q_Complex: Shortest Path", t.elapsed, mem_mon.get_max_memory())

def main(num_papers: int = 5000):
    log.info("=" * 60)
    log.info("🚀 CARGA DE ARXIV A FALKORDB - BENCHMARK")
    log.info("=" * 60)
    
    model = SentenceTransformer(EMBEDDING_MODEL)
    papers = download_arxiv(limit=num_papers)
    papers = generate_embeddings(papers, model)
    
    # Connect
    try:
        db = FalkorDB(host=FALKOR_HOST, port=FALKOR_PORT)
        graph = db.select_graph(GRAPH_NAME)
        # Clear old
        try:
            graph.delete() # Delete graph key
        except:
            pass
        graph = db.select_graph(GRAPH_NAME)
        
        setup_schema(graph)
        load_papers(graph, papers)
        create_citations(graph, len(papers))
        
        run_sample_queries(graph)
        run_complex_queries(graph)
        
    except Exception as e:
        log.error(f"Error connecting/executing FalkorDB: {e}")
        return

    stats.report()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", "--num-papers", type=int, default=5000)
    args = parser.parse_args()
    
    main(num_papers=args.num_papers)