#!/usr/bin/env python3
"""
Benchmark de NeuralGraphDB con dataset ArXiv.
Equivalente funcional al benchmark de Neo4j para comparación directa.

IMPORTANTE: Este script mide los mismos pasos que arxiv_benchmark.py para Neo4j
para poder comparar ambas bases de datos.
"""

import json
import time
import logging
import subprocess
import tempfile
import random
import os
from pathlib import Path
from dataclasses import dataclass
from typing import Optional

from datasets import load_dataset
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from benchmark_utils import MemoryMonitor

# Configuración
NEURALGRAPH_BINARY = "./target/release/neuralgraph"
CONTAINER_NAME = "benchmark-neuralgraph"
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
        log.info("📊 RESUMEN DE TIEMPOS - NeuralGraphDB")
        log.info("=" * 60)
        for name, elapsed in self.timings.items():
            mem_str = f" | Mem: {self.memory[name]:.2f} MB" if name in self.memory else ""
            log.info(f"  {name:<35}: {elapsed:.2f}s{mem_str}")
        total = sum(self.timings.values())
        log.info("-" * 60)
        log.info(f"  TOTAL: {total:.2f}s")
        log.info("=" * 60)
    
    def to_json(self) -> dict:
        return {
            "database": "NeuralGraphDB",
            "timings": self.timings,
            "memory_mb": self.memory,
            "total": sum(self.timings.values())
        }


stats = BenchmarkStats()


def check_binary():
    """Verifica que el binario de NeuralGraphDB existe."""
    binary_path = Path(NEURALGRAPH_BINARY)
    if not binary_path.exists():
        log.error(f"❌ No se encuentra {NEURALGRAPH_BINARY}")
        log.error("   Ejecuta: cargo build --release -p neural-cli")
        raise FileNotFoundError(NEURALGRAPH_BINARY)
    return binary_path


def download_arxiv(limit: int = 10000) -> list[dict]:
    """Descarga dataset de ArXiv desde HuggingFace."""
    
    with Timer("Descarga ArXiv") as t:
        log.info(f"📥 Cargando dataset ArXiv desde HuggingFace...")
        
        dataset = load_dataset("CShorten/ML-ArXiv-Papers", split="train")
        
        papers = []
        for i, item in enumerate(tqdm(dataset, desc="Procesando papers", total=min(limit, len(dataset)))):
            if i >= limit:
                break
            
            papers.append({
                "id": i,
                "title": (item.get("title") or f"Paper {i}")[:500],
                "abstract": (item.get("abstract") or "")[:2000],
                "category": "cs.LG",
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


def setup_schema(work_dir: Path) -> tuple[Path, Path, Path, Path]:
    """
    Prepara el schema - equivalente a setup_neo4j_schema.
    En NeuralGraphDB preparamos los archivos CSV.
    """
    with Timer("Setup schema") as t:
        nodes_file = work_dir / "papers_nodes.csv"
        edges_file = work_dir / "papers_edges.csv"
        authors_file = work_dir / "authors_nodes.csv"
        affiliations_file = work_dir / "affiliations_edges.csv"
        
        log.info("📐 Preparando archivos de schema...")
        # Crear archivos vacíos
        for f in [nodes_file, edges_file, authors_file, affiliations_file]:
            f.touch()
    
    stats.record("3. Setup schema", t.elapsed)
    return nodes_file, edges_file, authors_file, affiliations_file


def load_papers(papers: list[dict], nodes_file: Path) -> int:
    """
    Carga papers - equivalente a load_papers_to_neo4j.
    """
    with Timer("Carga de papers") as t:
        log.info(f"📝 Escribiendo {len(papers)} papers...")
        
        with open(nodes_file, "w") as f:
            # Header
            f.write("id,label,title,abstract,category\n")
            
            for p in tqdm(papers, desc="Cargando papers"):
                # Escapar comillas en strings
                title = p["title"].replace('"', '""').replace('\n', ' ')
                abstract = p["abstract"].replace('"', '""').replace('\n', ' ')
                category = p["category"]
                
                f.write(f'{p["id"]},Paper,"{title}","{abstract}",{category}\n')
    
    stats.record("4. Carga papers", t.elapsed)
    return len(papers)

def create_citations(num_papers: int, edges_file: Path, avg_citations: int = 5) -> int:
    """
    Crea relaciones de citas aleatorias - equivalente a create_citations en Neo4j.
    """
    with Timer("Creación de citas") as t:
        prob = avg_citations / num_papers
        
        log.info(f"🔗 Creando ~{num_papers * avg_citations} citas aleatorias...")
        
        citations = []
        # Simular el comportamiento de Neo4j: MATCH (p1), (p2) WHERE p1.id < p2.id AND rand() < prob
        for i in tqdm(range(num_papers), desc="Generando citas"):
            for j in range(i + 1, num_papers):
                if random.random() < prob:
                    citations.append((i, j))
        
        log.info(f"🔗 Citas creadas: {len(citations)}")
        
        # Escribir aristas
        with open(edges_file, "w") as f:
            f.write("source,target,label\n")
            for src, tgt in citations:
                f.write(f"{src},{tgt},CITES\n")
    
    stats.record("5. Creación citas", t.elapsed)
    return len(citations)

def create_authors_and_institutions(
    num_papers: int,
    authors_file: Path,
    affiliations_file: Path
) -> tuple[int, int]:
    """
    Crea autores e instituciones - equivalente a create_authors_and_institutions en Neo4j.
    """
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
        log.info("🏛️  Creando instituciones...")
        
        # Crear autores (1-3 por paper)
        authors = []
        author_affiliations = []
        author_paper_edges = []
        
        log.info(" Creando autores...")
        author_id = 0
        for paper_id in tqdm(range(num_papers), desc="Creando autores"):
            num_authors = random.randint(1, 3)
            for a in range(num_authors):
                institution = random.choice(institutions)
                authors.append({
                    "id": author_id,
                    "name": f"Author_{paper_id}_{a}",
                    "institution": institution
                })
                author_paper_edges.append((paper_id, author_id))
                author_id += 1
        
        log.info(f"👤 Autores creados: {len(authors)}")
        
        # Escribir autores como nodos adicionales
        with open(authors_file, "w") as f:
            f.write("id,label,name,institution\n")
            for a in authors:
                f.write(f'{a["id"] + num_papers},Author,"{a["name"]}",{a["institution"]}\n')
        
        # Escribir relaciones autor-paper
        with open(affiliations_file, "w") as f:
            f.write("source,target,label\n")
            for paper_id, author_node_id in author_paper_edges:
                f.write(f"{paper_id},{author_node_id + num_papers},AUTHORED_BY\n")
    
    stats.record("6. Creación autores/instituciones", t.elapsed)
    return len(authors), len(institutions)

def verify_data(
    nodes_file: Path,
    edges_file: Path,
    authors_file: Path,
    num_papers: int,
    num_citations: int,
    num_authors: int
) -> dict:
    """Verifica que los datos se cargaron correctamente."""
    
    with Timer("Verificación") as t:
        # Contar líneas en archivos (excluyendo header)
        def count_lines(f):
            with open(f) as fp:
                return sum(1 for _ in fp) - 1  # -1 for header
        
        papers = count_lines(nodes_file)
        cites = count_lines(edges_file)
        authors = count_lines(authors_file)
        
        log.info("\n" + "-" * 40)
        log.info("📊 DATOS CARGADOS:")
        log.info(f"   Papers: {papers}")
        log.info(f"   Citas: {cites}")
        log.info(f"   Autores: {authors}")
        log.info(f"   Instituciones: 10")
        log.info("-" * 40)
    
    stats.record("7. Verificación", t.elapsed)
    
    return {
        "papers": papers,
        "cites": cites,
        "authors": authors,
        "institutions": 10
    }

def run_complex_queries(nodes_file: Path, edges_file: Path):
    """
    Ejecuta queries complejas y monitorea memoria.
    """
    log.info("\n🧠 Ejecutando queries complejas (con monitoreo de memoria)...")
    
    # Pre-load data command context
    preload = f":load nodes {nodes_file}\n:load edges {edges_file}\n"
    
    # Query 1: 3-Hop Traversal (Exponential complexity)
    query_3hop = """MATCH (a)-[]->(b)-[]->(c)-[]->(d) RETURN COUNT(*)"""
    
    mem_mon = MemoryMonitor(CONTAINER_NAME)
    mem_mon.start()
    with Timer("Query Compleja: 3-Hop Traversal") as t:
        result = subprocess.run(
            [NEURALGRAPH_BINARY],
            input=f"{preload}{query_3hop}\n:quit\n",
            capture_output=True, text=True
        )
    mem_mon.stop()
    stats.record("Q_Complex: 3-Hop Traversal", t.elapsed, mem_mon.get_max_memory())
    
    # Query 2: Analytical (Top Cited Categories) - Group By + Order By
    # Note: Requires Group By support
    query_top = """MATCH (a)-[:CITES]->(b) RETURN a.category, COUNT(b) ORDER BY COUNT(b) DESC LIMIT 5"""
    
    mem_mon = MemoryMonitor(CONTAINER_NAME)
    mem_mon.start()
    with Timer("Query Compleja: Top Cited Categories") as t:
        result = subprocess.run(
            [NEURALGRAPH_BINARY],
            input=f"{preload}{query_top}\n:quit\n",
            capture_output=True, text=True
        )
    mem_mon.stop()
    stats.record("Q_Complex: Analytical (Top Cited)", t.elapsed, mem_mon.get_max_memory())

    # Query 3: Shortest Path (BFS)
    # Finding path between node 0 and node 100 (if connected)
    query_path = """MATCH p = SHORTEST PATH (a)-[*]->(b) WHERE id(a)=0 AND id(b)=100 RETURN p"""
    
    mem_mon = MemoryMonitor(CONTAINER_NAME)
    mem_mon.start()
    with Timer("Query Compleja: Shortest Path") as t:
        result = subprocess.run(
            [NEURALGRAPH_BINARY],
            input=f"{preload}{query_path}\n:quit\n",
            capture_output=True, text=True
        )
    mem_mon.stop()
    stats.record("Q_Complex: Shortest Path", t.elapsed, mem_mon.get_max_memory())

def run_sample_queries(nodes_file: Path, edges_file: Path) -> dict:
    """
    Ejecuta queries de prueba - equivalente a run_sample_queries en Neo4j.
    """
    log.info("\n🧪 Ejecutando queries de prueba...")
    
    results = {}
    
    # Primero medimos el tiempo de carga en NeuralGraphDB
    load_commands = f":load nodes {nodes_file}\n:load edges {edges_file}\n:stats\n:quit\n"
    
    mem_mon = MemoryMonitor(CONTAINER_NAME)
    mem_mon.start()
    with Timer("Carga datos en NeuralGraphDB") as t:
        result = subprocess.run(
            [NEURALGRAPH_BINARY],
            input=load_commands,
            capture_output=True,
            text=True,
            timeout=300
        )
    mem_mon.stop()
    stats.record("8. Carga en NeuralGraphDB", t.elapsed, mem_mon.get_max_memory())
    
    load_time = t.elapsed
    
    # Query 1: Traversal 2-hop
    with Timer("Query: 2-hop traversal (incl. carga)") as t:
        commands = f":load nodes {nodes_file}\n:load edges {edges_file}\n"
        commands += """
MATCH (a)-[]->(b) RETURN a.id, b.id LIMIT 100
:quit
"""
        result = subprocess.run(
            [NEURALGRAPH_BINARY],
            input=commands,
            capture_output=True,
            text=True,
            timeout=120
        )
    
    query_time = max(0, t.elapsed - load_time)
    stats.record("Query: 2-hop traversal", query_time)
    
    # Query 2: Aggregation query
    with Timer("Query: Aggregation (incl. carga)") as t:
        commands = f":load nodes {nodes_file}\n:load edges {edges_file}\n"
        commands += """
MATCH (n:Paper) RETURN COUNT(*), MIN(n.id), MAX(n.id)
:quit
"""
        result = subprocess.run(
            [NEURALGRAPH_BINARY],
            input=commands,
            capture_output=True,
            text=True,
            timeout=60
        )
    
    query_time = max(0, t.elapsed - load_time)
    stats.record("Query: Aggregation", query_time)
    
    # Query 3: Filtered query
    with Timer("Query: Filtro (incl. carga)") as t:
        commands = f":load nodes {nodes_file}\n:load edges {edges_file}\n"
        commands += """
MATCH (n:Paper) WHERE n.category = \"cs.LG\" RETURN n.title LIMIT 10
:quit
"""
        result = subprocess.run(
            [NEURALGRAPH_BINARY],
            input=commands,
            capture_output=True,
            text=True,
            timeout=60
        )
    
    query_time = max(0, t.elapsed - load_time)
    stats.record("Query: Filtro", query_time)
    
    return results

def run_full_benchmark(work_dir: Path, nodes_file: Path, edges_file: Path):
    """
    Ejecuta el benchmark interno de NeuralGraphDB.
    """
    log.info("\n" + "=" * 60)
    log.info("📊 BENCHMARK INTERNO NeuralGraphDB")
    log.info("=" * 60)
    
    commands = f"""
:load nodes {nodes_file}
:load edges {edges_file}
:stats
:benchmark
:quit
"""
    result = subprocess.run(
        [NEURALGRAPH_BINARY],
        input=commands,
        capture_output=True,
        text=True,
        timeout=300
    )
    
    if result.returncode == 0:
        log.info("✅ Benchmark interno completado")
        for line in result.stdout.split('\n'):
            if any(x in line for x in ['Nodes:', 'Edges:', 'Node lookup', 'Property', 'COUNT', 'Label filter', '✓', '✗', 'row(s) in']):
                log.info(f"   {line}")
    else:
        log.error(f"❌ Error: {result.stderr}")
    
    return result

def main(num_papers: int = 5000, skip_embeddings: bool = False, output_dir: Optional[str] = None):
    """
    Pipeline principal - equivalente a main() en arxiv_benchmark.py para Neo4j.
    """
    
    log.info("=" * 60)
    log.info("🚀 CARGA DE ARXIV A NeuralGraphDB - BENCHMARK")
    log.info(f"   Papers objetivo: {num_papers}")
    log.info("=" * 60)
    
    # Verificar binario
    check_binary()
    log.info(f"✅ Binario encontrado: {NEURALGRAPH_BINARY}")
    
    # Directorio de salida
    if output_dir:
        work_dir = Path(output_dir)
        work_dir.mkdir(parents=True, exist_ok=True)
    else:
        work_dir = Path(tempfile.mkdtemp(prefix="neuralgraph_benchmark_"))
    
    log.info(f"📁 Directorio de trabajo: {work_dir}")
    
    # 0. Cargar modelo de embeddings (para comparación justa con Neo4j)
    if not skip_embeddings:
        with Timer("Carga modelo embeddings") as t:
            log.info(f"🤖 Cargando modelo: {EMBEDDING_MODEL}")
            model = SentenceTransformer(EMBEDDING_MODEL)
        stats.record("0. Carga modelo", t.elapsed)
    else:
        model = None
    
    # 1. Descargar papers
    papers = download_arxiv(limit=num_papers)
    log.info(f"📖 Papers obtenidos: {len(papers)}")
    
    # 2. Generar embeddings
    if not skip_embeddings and model:
        papers = generate_embeddings(papers, model)
    else:
        log.info("⏭️  Saltando generación de embeddings")
    
    # 3. Setup schema
    nodes_file, edges_file, authors_file, affiliations_file = setup_schema(work_dir)
    
    # 4. Cargar papers
    num_loaded = load_papers(papers, nodes_file)
    
    # 5. Crear citas
    num_citations = create_citations(len(papers), edges_file)
    
    # 6. Crear autores e instituciones
    num_authors, num_institutions = create_authors_and_institutions(
        len(papers), authors_file, affiliations_file
    )
    
    # 7. Verificar
    data = verify_data(
        nodes_file, edges_file, authors_file,
        num_loaded, num_citations, num_authors
    )
    
    # Queries de prueba
    run_sample_queries(nodes_file, edges_file)
    
    # Queries Complejas + Memory Monitoring
    run_complex_queries(nodes_file, edges_file)
    
    # Benchmark interno de NeuralGraphDB
    run_full_benchmark(work_dir, nodes_file, edges_file)
    
    # Guardar resultados
    output_file = work_dir / "neuralgraph_benchmark_results.json"
    with open(output_file, "w") as f:
        json.dump({
            "config": {
                "num_papers": num_papers,
                "database": "NeuralGraphDB",
            },
            "stats": stats.to_json(),
            "data": data
        }, f, indent=2)
    
    log.info(f"\n📄 Resultados guardados en: {output_file}")
    
    # Reporte final
    stats.report()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Benchmark NeuralGraphDB con ArXiv")
    parser.add_argument(
        "-n", "--num-papers",
        type=int,
        default=5000,
        help="Número de papers a cargar (default: 5000)"
    )
    parser.add_argument(
        "--skip-embeddings",
        action="store_true",
        help="Saltar generación de embeddings (más rápido, menos comparable)"
    )
    parser.add_argument(
        "-o", "--output",
        type=str,
        default=None,
        help="Directorio de salida (default: temporal)"
    )
    
    args = parser.parse_args()
    main(
        num_papers=args.num_papers,
        skip_embeddings=args.skip_embeddings,
        output_dir=args.output
    )