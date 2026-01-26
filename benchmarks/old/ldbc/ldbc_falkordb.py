import time
import csv
import logging
from pathlib import Path
from dataclasses import dataclass
from falkordb import FalkorDB

# Config
FALKOR_HOST = "localhost"
FALKOR_PORT = 16379
GRAPH_NAME = "ldbc_bench"
DATA_DIR = Path("benchmarks/ldbc/data")

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger("ldbc_falkor")

@dataclass
class Stats:
    start_time: float = 0
    end_time: float = 0
    def start(self):
        self.start_time = time.perf_counter()
    def stop(self):
        self.end_time = time.perf_counter()
    def duration(self):
        return self.end_time - self.start_time

def load_data(graph):
    log.info("🚀 Loading LDBC Data into FalkorDB...")
    stats = Stats()
    stats.start()
    
    # 1. Clear DB
    try:
        graph.delete()
    except: pass
    
    # 2. Load Persons
    count = 0
    with open(DATA_DIR / "person.csv") as f:
        reader = csv.DictReader(f)
        batch = []
        for row in reader:
            batch.append(row)
            if len(batch) >= 100:
                # FalkorDB supports UNWIND with params
                graph.query("""
                    UNWIND $batch AS row
                    CREATE (n:Person)
                    SET n = row
                """, {"batch": batch})
                count += len(batch)
                batch = []
                print(f"\r   Loaded {count} Persons...", end="")
        if batch:
            graph.query("""
                UNWIND $batch AS row
                CREATE (n:Person)
                SET n = row
            """, {"batch": batch})
    print()
    
    # Index
    graph.query("CREATE INDEX ON :Person(id)")

    # 3. Load Knows
    count = 0
    with open(DATA_DIR / "person_knows_person.csv") as f:
        reader = csv.DictReader(f)
        batch = []
        for row in reader:
            batch.append(row)
            if len(batch) >= 100:
                graph.query("""
                    UNWIND $batch AS row
                    MATCH (a:Person {id: row.Person1Id}), (b:Person {id: row.Person2Id})
                    CREATE (a)-[:KNOWS {creationDate: row.creationDate}]->(b)
                """, {"batch": batch})
                count += len(batch)
                batch = []
                print(f"\r   Loaded {count} Know edges...", end="")
        if batch:
            graph.query("""
                UNWIND $batch AS row
                MATCH (a:Person {id: row.Person1Id}), (b:Person {id: row.Person2Id})
                CREATE (a)-[:KNOWS {creationDate: row.creationDate}]->(b)
            """, {"batch": batch})
    print()

    stats.stop()
    log.info(f"✅ Data Loading Complete in {stats.duration():.2f}s")

def run_benchmarks(graph):
    log.info("\n🏃 Running Interactive Workload (FalkorDB)...")
    
    # IS1: Profile
    log.info("   [IS1] Person Profile Lookup")
    stats = Stats()
    stats.start()
    for i in range(10):
        pid = str(i * 10)
        # Using string interpolation for params if client is tricky, but params preferred
        graph.query("MATCH (n:Person) WHERE n.id = $pid RETURN n.firstName, n.lastName, n.birthday", {"pid": pid})
    stats.stop()
    log.info(f"     Avg Latency: {stats.duration()/10*1000:.2f} ms")
    
    # IS3: Friends
    log.info("   [IS3] Friends Lookup")
    stats = Stats()
    stats.start()
    for i in range(10):
        pid = str(i * 10)
        graph.query("MATCH (n:Person {id: $pid})-[:KNOWS]->(friend) RETURN friend.firstName, friend.creationDate", {"pid": pid})
    stats.stop()
    log.info(f"     Avg Latency: {stats.duration()/10*1000:.2f} ms")
    
    # IC1: Friends of Friends
    log.info("   [IC1] Friends of Friends")
    stats = Stats()
    stats.start()
    for i in range(5):
        pid = str(i * 20)
        graph.query("MATCH (n:Person {id: $pid})-[:KNOWS]->(f1)-[:KNOWS]->(f2) RETURN count(f2)", {"pid": pid})
    stats.stop()
    log.info(f"     Avg Latency: {stats.duration()/5*1000:.2f} ms")

if __name__ == "__main__":
    db = FalkorDB(host=FALKOR_HOST, port=FALKOR_PORT)
    graph = db.select_graph(GRAPH_NAME)
    load_data(graph)
    run_benchmarks(graph)
