#!/usr/bin/env python3
"""
Carga dataset ArXiv en Neo4j con embeddings vectoriales.
Incluye logging de tiempos para benchmark y monitoreo de memoria.
"""

import json
import time
import logging
from pathlib import Path
from dataclasses import dataclass
from typing import Iterator

from datasets import load_dataset
from tqdm import tqdm
from neo4j import GraphDatabase
from sentence_transformers import SentenceTransformer
from benchmark_utils import MemoryMonitor

# Configuración
NEO4J_URI = "bolt://localhost:17687"
NEO4J_AUTH = ("neo4j", "benchmark123")
CONTAINER_NAME = "benchmark-neo4j"
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
    """Context manager para medir tiempos."""
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
    """Acumula estadísticas del benchmark."""
    
    def __init__(self):
        self.timings = {}
        self.memory = {}
    
    def record(self, name: str, elapsed: float, memory_mb: float = 0.0):
        self.timings[name] = elapsed
        if memory_mb > 0:
            self.memory[name] = memory_mb
    
    def report(self):
        log.info("\n" + "=" * 60)
        log.info("📊 RESUMEN DE TIEMPOS - Neo4j")
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
    """Descarga dataset de ArXiv desde HuggingFace."""
    
    with Timer("Descarga ArXiv") as t:
        log.info(f"📥 Cargando dataset ArXiv desde HuggingFace...")
        
        # Usar dataset de papers de AI/ML
        dataset = load_dataset("CShorten/ML-ArXiv-Papers", split="train")
        
        papers = []
        for i, item in enumerate(tqdm(dataset, desc="Procesando papers", total=min(limit, len(dataset)))):
            if i >= limit:
                break
            
            papers.append({
                "id": f"arxiv_{i}",
                "title": (item.get("title") or f"Paper {i}")[:500],
                "abstract": (item.get("abstract") or "")[:2000],
                "category": "cs.LG",  # ML papers
            })
        
        log.info(f"📁 Papers cargados: {len(papers)}")
    
    stats.record("1. Descarga ArXiv", t.elapsed)
    return papers


def generate_embeddings(papers: list[dict], model: SentenceTransformer) -> list[dict]:
    """Genera embeddings para los abstracts."""
    
    with Timer("Generación de embeddings") as t:
        abstracts = [p["abstract"] for p in papers]
        embeddings = model.encode(
            abstracts,
            show_progress_bar=True,
            batch_size=64
        )
        
        for paper, embedding in zip(papers, embeddings):
            paper["embedding"] = embedding.tolist()
    
    stats.record("2. Generación embeddings", t.elapsed)
    return papers


def setup_neo4j_schema(driver):
    """Crea índices y constraints en Neo4j."""
    
    with Timer("Setup schema Neo4j") as t:
        with driver.session() as session:
            # Limpiar datos anteriores
            log.info("🗑️  Limpiando datos anteriores...")
            session.run("MATCH (n) DETACH DELETE n")
            
            # Constraint de unicidad
            log.info("📐 Creando constraints...")
            session.run("""
                CREATE CONSTRAINT paper_id IF NOT EXISTS
                FOR (p:Paper) REQUIRE p.id IS UNIQUE
            """)
            
            # Índice para categoría
            session.run("""
                CREATE INDEX paper_category IF NOT EXISTS
                FOR (p:Paper) ON (p.category)
            """)
            
            # Índice vectorial
            log.info("🧮 Creando índice vectorial...")
            session.run("""
                CREATE VECTOR INDEX paper_embeddings IF NOT EXISTS
                FOR (p:Paper)
                ON (p.embedding)
                OPTIONS {
                    indexConfig: {
                        `vector.dimensions`: 384,
                        `vector.similarity_function`: 'cosine'
                    }
                }
            """)
    
    stats.record("3. Setup schema", t.elapsed)

def load_papers_to_neo4j(driver, papers: list[dict]):
    """Carga papers en Neo4j en batches."""
    
    mem_mon = MemoryMonitor(CONTAINER_NAME)
    mem_mon.start()
    
    with Timer("Carga de papers a Neo4j") as t:
        with driver.session() as session:
            for i in tqdm(range(0, len(papers), BATCH_SIZE), desc="Cargando papers"):
                batch = papers[i:i + BATCH_SIZE]
                session.run("""
                    UNWIND $papers AS paper
                    CREATE (p:Paper {
                        id: paper.id,
                        title: paper.title,
                        abstract: paper.abstract,
                        category: paper.category,
                        embedding: paper.embedding
                    })
                """, papers=batch)
                
    mem_mon.stop()
    stats.record("4. Carga papers", t.elapsed, mem_mon.get_max_memory())

def create_citations(driver, num_papers: int, avg_citations: int = 5):
    """Crea relaciones de citas aleatorias."""
    
    mem_mon = MemoryMonitor(CONTAINER_NAME)
    mem_mon.start()
    
    with Timer("Creación de citas") as t:
        with driver.session() as session:
            # Crear citas aleatorias
            log.info(f"🔗 Creando ~{num_papers * avg_citations} citas aleatorias...")
            
            result = session.run("""
                MATCH (p1:Paper), (p2:Paper)
                WHERE p1.id < p2.id AND rand() < $prob
                CREATE (p1)-[:CITES]->(p2)
                RETURN count(*) as created
            """, prob=avg_citations / num_papers)
            
            created = result.single()["created"]
            log.info(f"🔗 Citas creadas: {created}")
            
    mem_mon.stop()
    stats.record("5. Creación citas", t.elapsed, mem_mon.get_max_memory())

def create_authors_and_institutions(driver):
    """Crea autores e instituciones ficticias para el benchmark."""
    
    institutions = [
        "Stanford University",
        "MIT",
        "Google",
        "OpenAI",
        "DeepMind",
        "Meta AI",
        "Berkeley",
        "CMU",
        "Microsoft Research",
        "Amazon"
    ]
    
    with Timer("Creación de autores e instituciones") as t:
        with driver.session() as session:
            # Crear instituciones
            log.info("🏛️  Creando instituciones...")
            session.run("""
                UNWIND $institutions AS name
                CREATE (i:Institution {name: name})
            """, institutions=institutions)
            
            # Crear autores y asignar a instituciones
            log.info("👤 Creando autores...")
            session.run("""
                MATCH (p:Paper)
                WITH p, toInteger(rand() * 3) + 1 AS num_authors
                UNWIND range(1, num_authors) AS author_num
                CREATE (a:Author {
                    id: p.id + '_author_' + toString(author_num),
                    name: 'Author ' + p.id + '_' + toString(author_num)
                })
                CREATE (p)-[:AUTHORED_BY]->(a)
                WITH a
                MATCH (i:Institution)
                WITH a, i, rand() AS r
                ORDER BY r
                LIMIT 1
                CREATE (a)-[:AFFILIATED_WITH]->(i)
            """)
            
            # Contar
            result = session.run("""
                MATCH (a:Author) RETURN count(a) AS authors
            """)
            authors = result.single()["authors"]
            log.info(f"👤 Autores creados: {authors}")
    
    stats.record("6. Creación autores/instituciones", t.elapsed)

def verify_data(driver):
    """Verifica que los datos se cargaron correctamente."""
    
    with Timer("Verificación de datos") as t:
        with driver.session() as session:
            result = session.run("""
                MATCH (p:Paper) RETURN count(p) AS papers
            """)
            papers = result.single()["papers"]
            
            result = session.run("""
                MATCH ()-[r:CITES]->() RETURN count(r) AS cites
            """)
            cites = result.single()["cites"]
            
            result = session.run("""
                MATCH (a:Author) RETURN count(a) AS authors
            """)
            authors = result.single()["authors"]
            
            result = session.run("""
                MATCH (i:Institution) RETURN count(i) AS institutions
            """)
            institutions = result.single()["institutions"]
            
            # Verificar índice vectorial
            result = session.run("""
                SHOW INDEXES WHERE name = 'paper_embeddings'
            """)
            vector_index = result.single()
            
            log.info("\n" + "-" * 40)
            log.info("📊 DATOS CARGADOS:")
            log.info(f"   Papers: {papers}")
            log.info(f"   Citas: {cites}")
            log.info(f"   Autores: {authors}")
            log.info(f"   Instituciones: {institutions}")
            log.info(f"   Índice vectorial: {vector_index['state'] if vector_index else 'NO EXISTE'}")
            log.info("-" * 40)
    
    stats.record("7. Verificación", t.elapsed)
    
    return {
        "papers": papers,
        "cites": cites,
        "authors": authors,
        "institutions": institutions
    }

def run_sample_queries(driver):
    """Ejecuta queries de prueba para verificar funcionalidad."""
    
    log.info("\n🧪 Ejecutando queries de prueba...")
    
    with driver.session() as session:
        # Query 1: Traversal simple
        with Timer("Query: Traversal 2-hop") as t:
            result = session.run("""
                MATCH (p:Paper)-[:CITES]->(cited:Paper)-[:CITES]->(cited2:Paper)
                RETURN p.id, count(cited2) AS reach
                LIMIT 10
            """)
            records = list(result)
        stats.record("Query: 2-hop traversal", t.elapsed)
        log.info(f"   Resultados: {len(records)}")
        
        # Query 2: Vector search
        with Timer("Query: Vector search") as t:
            # Obtener un embedding de referencia
            ref = session.run("""
                MATCH (p:Paper)
                WHERE p.embedding IS NOT NULL
                RETURN p.embedding AS emb
                LIMIT 1
            """).single()
            
            if ref:
                result = session.run("""
                    CALL db.index.vector.queryNodes('paper_embeddings', 10, $embedding)
                    YIELD node, score
                    RETURN node.title, score
                """, embedding=ref["emb"])
                records = list(result)
        stats.record("Query: Vector search", t.elapsed)
        log.info(f"   Resultados: {len(records)}")
        
        # Query 3: Híbrido (vector + filtro)
        with Timer("Query: Híbrido vector + categoría") as t:
            if ref:
                result = session.run("""
                    CALL db.index.vector.queryNodes('paper_embeddings', 50, $embedding)
                    YIELD node, score
                    WHERE node.category = 'math.AC'
                    RETURN node.title, score
                    LIMIT 10
                """, embedding=ref["emb"])
                records = list(result)
        stats.record("Query: Híbrido", t.elapsed)
        log.info(f"   Resultados: {len(records)}")

def run_complex_queries(driver):
    """Ejecuta queries complejas con monitoreo de memoria."""
    log.info("\n🧠 Ejecutando queries complejas (con monitoreo de memoria)...")
    
    with driver.session() as session:
        
        # Q1: 3-Hop Traversal
        mem_mon = MemoryMonitor(CONTAINER_NAME)
        mem_mon.start()
        with Timer("Query Compleja: 3-Hop Traversal") as t:
            result = session.run("""
                MATCH (a)-[:CITES]->(b)-[:CITES]->(c)-[:CITES]->(d)
                RETURN count(*)
            """)
            records = list(result)
        mem_mon.stop()
        stats.record("Q_Complex: 3-Hop Traversal", t.elapsed, mem_mon.get_max_memory())
        
        # Q2: Analytical (Top Cited)
        mem_mon = MemoryMonitor(CONTAINER_NAME)
        mem_mon.start()
        with Timer("Query Compleja: Top Cited Categories") as t:
            result = session.run("""
                MATCH (a:Paper)-[:CITES]->(b:Paper)
                RETURN a.category, count(b) as citations
                ORDER BY citations DESC
                LIMIT 5
            """)
            records = list(result)
        mem_mon.stop()
        stats.record("Q_Complex: Analytical (Top Cited)", t.elapsed, mem_mon.get_max_memory())
        
        # Q3: Shortest Path
        # Need node IDs. Neo4j string IDs are 'arxiv_0', 'arxiv_100'.
        mem_mon = MemoryMonitor(CONTAINER_NAME)
        mem_mon.start()
        with Timer("Query Compleja: Shortest Path") as t:
            result = session.run("""
                MATCH (a:Paper {id: 'arxiv_0'}), (b:Paper {id: 'arxiv_100'})
                MATCH p = shortestPath((a)-[:CITES*]->(b))
                RETURN p
            """)
            records = list(result)
        mem_mon.stop()
        stats.record("Q_Complex: Shortest Path", t.elapsed, mem_mon.get_max_memory())

def main(num_papers: int = 5000):
    """Pipeline principal."""
    
    log.info("=" * 60)
    log.info("🚀 CARGA DE ARXIV A NEO4J - BENCHMARK")
    log.info(f"   Papers objetivo: {num_papers}")
    log.info("=" * 60)
    
    # Cargar modelo de embeddings
    with Timer("Carga modelo embeddings") as t:
        log.info(f"🤖 Cargando modelo: {EMBEDDING_MODEL}")
        model = SentenceTransformer(EMBEDDING_MODEL)
    stats.record("0. Carga modelo", t.elapsed)
    
    # Descargar/cargar papers
    papers = download_arxiv(limit=num_papers)
    log.info(f"📖 Papers obtenidos: {len(papers)}")
    
    # Generar embeddings
    papers = generate_embeddings(papers, model)
    
    # Conectar a Neo4j
    driver = GraphDatabase.driver(NEO4J_URI, auth=NEO4J_AUTH)
    
    try:
        driver.verify_connectivity()
        log.info("✅ Conexión a Neo4j OK")
        
        # Setup schema
        setup_neo4j_schema(driver)
        
        # Cargar papers
        load_papers_to_neo4j(driver, papers)
        
        # Crear citas
        create_citations(driver, len(papers))
        
        # Crear autores e instituciones
        create_authors_and_institutions(driver)
        
        # Verificar
        verify_data(driver)
        
        # Queries de prueba
        run_sample_queries(driver)
        
        # Queries Complejas
        run_complex_queries(driver)
        
    finally:
        driver.close()
    
    # Reporte final
    stats.report()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Carga ArXiv en Neo4j")
    parser.add_argument(
        "-n", "--num-papers",
        type=int,
        default=5000,
        help="Número de papers a cargar (default: 5000)"
    )
    
    args = parser.parse_args()
    main(num_papers=args.num_papers)
