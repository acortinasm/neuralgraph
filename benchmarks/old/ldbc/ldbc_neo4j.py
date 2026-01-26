import time
import csv
import json
import logging
from pathlib import Path
from dataclasses import dataclass
from neo4j import GraphDatabase

# Config
NEO4J_URI = "bolt://localhost:17687"
NEO4J_AUTH = ("neo4j", "benchmark123")
DATA_DIR = Path("benchmarks/ldbc/data")

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger("ldbc_neo4j")

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

def load_data(driver):
    log.info("🚀 Loading LDBC Data into Neo4j...")
    stats = Stats()
    stats.start()
    
    with driver.session() as session:
        # 1. Clear DB
        session.run("MATCH (n) DETACH DELETE n")
        
        # 2. Load Persons
        count = 0
        with open(DATA_DIR / "person.csv") as f:
            reader = csv.DictReader(f)
            # Batching would be faster, but let's keep logic simple and comparable for now
            # Actually, batching is essential for Neo4j speed. I'll do small batches.
            batch = []
            for row in reader:
                batch.append(row)
                if len(batch) >= 100:
                    session.run("""
                        UNWIND $batch AS row
                        CREATE (n:Person)
                        SET n = row
                    """, batch=batch)
                    count += len(batch)
                    batch = []
                    print(f"\r   Loaded {count} Persons...", end="")
            if batch:
                session.run("""
                    UNWIND $batch AS row
                    CREATE (n:Person)
                    SET n = row
                """, batch=batch)
                count += len(batch)
        print()
        
        # Index for lookup speed (Fairness)
        session.run("CREATE INDEX person_id IF NOT EXISTS FOR (n:Person) ON (n.id)")

        # 3. Load Knows
        count = 0
        with open(DATA_DIR / "person_knows_person.csv") as f:
            reader = csv.DictReader(f)
            batch = []
            for row in reader:
                batch.append(row)
                if len(batch) >= 100:
                    session.run("""
                        UNWIND $batch AS row
                        MATCH (a:Person {id: row.Person1Id}), (b:Person {id: row.Person2Id})
                        CREATE (a)-[:KNOWS {creationDate: row.creationDate}]->(b)
                    """, batch=batch)
                    count += len(batch)
                    batch = []
                    print(f"\r   Loaded {count} Know edges...", end="")
            if batch:
                session.run("""
                    UNWIND $batch AS row
                    MATCH (a:Person {id: row.Person1Id}), (b:Person {id: row.Person2Id})
                    CREATE (a)-[:KNOWS {creationDate: row.creationDate}]->(b)
                """, batch=batch)
        print()

    stats.stop()
    log.info(f"✅ Data Loading Complete in {stats.duration():.2f}s")

def run_benchmarks(driver):
    log.info("\n🏃 Running Interactive Workload (Neo4j)...")
    
    with driver.session() as session:
        # IS1: Profile
        log.info("   [IS1] Person Profile Lookup")
        stats = Stats()
        stats.start()
        for i in range(10):
            pid = str(i * 10)
            session.run("MATCH (n:Person {id: $pid}) RETURN n.firstName, n.lastName, n.birthday", pid=pid)
        stats.stop()
        log.info(f"     Avg Latency: {stats.duration()/10*1000:.2f} ms")
        
        # IS3: Friends
        log.info("   [IS3] Friends Lookup")
        stats = Stats()
        stats.start()
        for i in range(10):
            pid = str(i * 10)
            session.run("MATCH (n:Person {id: $pid})-[:KNOWS]->(friend) RETURN friend.firstName, friend.creationDate", pid=pid)
        stats.stop()
        log.info(f"     Avg Latency: {stats.duration()/10*1000:.2f} ms")
        
        # IC1: Friends of Friends
        log.info("   [IC1] Friends of Friends")
        stats = Stats()
        stats.start()
        for i in range(5):
            pid = str(i * 20)
            session.run("MATCH (n:Person {id: $pid})-[:KNOWS]->(f1)-[:KNOWS]->(f2) RETURN count(f2)", pid=pid)
        stats.stop()
        log.info(f"     Avg Latency: {stats.duration()/5*1000:.2f} ms")

if __name__ == "__main__":
    driver = GraphDatabase.driver(NEO4J_URI, auth=NEO4J_AUTH)
    try:
        load_data(driver)
        run_benchmarks(driver)
    finally:
        driver.close()
