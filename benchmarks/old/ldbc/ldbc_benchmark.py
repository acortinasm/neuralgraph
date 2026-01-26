import time
import requests
import json
import csv
import logging
from pathlib import Path
from dataclasses import dataclass

# Setup
SERVER_URL = "http://localhost:3000/api/query"
DATA_DIR = Path("benchmarks/ldbc/data")

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger("ldbc")

@dataclass
class Stats:
    start_time: float = 0
    end_time: float = 0
    
    def start(self): self.start_time = time.perf_counter()
    def stop(self): self.end_time = time.perf_counter()
    def duration(self): return self.end_time - self.start_time

def query(cypher: str):
    res = requests.post(SERVER_URL, json={"query": cypher})
    if res.status_code != 200:
        raise Exception(f"Query failed: {res.text}")
    return res.json()

def load_data():
    log.info("🚀 Loading LDBC Data...")
    stats = Stats()
    stats.start()
    
    # 1. Clear DB
    query("MATCH (n) DETACH DELETE n")
    
    # 2. Load Persons
    count = 0
    with open(DATA_DIR / "person.csv") as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Construct props string safely
            props = ", ".join([f'{k}: {json.dumps(v)}' for k, v in row.items()])
            q = f"CREATE (n:Person {{{props}}})"
            query(q)
            count += 1
            if count % 100 == 0: print(f"\r   Loaded {count} Persons...", end="")
    print()
    
    # 3. Load Knows
    count = 0
    with open(DATA_DIR / "person_knows_person.csv") as f:
        reader = csv.DictReader(f)
        for row in reader:
            q = f"""
            MATCH (a:Person), (b:Person) 
            WHERE a.id = \"{row['Person1Id']}\" AND b.id = \"{row['Person2Id']}\" 
            CREATE (a)-[:KNOWS {{creationDate: \"{row['creationDate']}\"}}]->(b)
            """
            query(q)
            count += 1
            if count % 100 == 0: print(f"\r   Loaded {count} Know edges...", end="")
    print()
    
    # 4. Load Posts (Sample)
    count = 0
    with open(DATA_DIR / "post.csv") as f:
        reader = csv.DictReader(f)
        for row in reader:
            props = ", ".join([f'{k}: {json.dumps(v)}' for k, v in row.items()])
            q = f"CREATE (n:Post {{{props}}})"
            query(q)
            count += 1
            if count % 100 == 0: print(f"\r   Loaded {count} Posts...", end="")
            if count > 500: break # Limit for speed in this demo
    print()
    
    # 5. Link Posts (Creator)
    count = 0
    with open(DATA_DIR / "post_hasCreator_person.csv") as f:
        reader = csv.DictReader(f)
        for row in reader:
            q = f"""
            MATCH (p:Post), (u:Person) 
            WHERE p.id = \"{row['PostId']}\" AND u.id = \"{row['PersonId']}\" 
            CREATE (p)-[:HAS_CREATOR]->(u)
            """
            query(q)
            count += 1
            if count % 100 == 0: print(f"\r   Linked {count} Posts...", end="")
            if count > 500: break
    print()

    stats.stop()
    log.info(f"✅ Data Loading Complete in {stats.duration():.2f}s")

def run_benchmarks():
    log.info("\n🏃 Running Interactive Workload...")
    
    # IS1: Profile
    log.info("   [IS1] Person Profile Lookup (Random ID)")
    stats = Stats()
    stats.start()
    for i in range(10):
        pid = str(i * 10)
        query(f"MATCH (n:Person) WHERE n.id = '{pid}' RETURN n.firstName, n.lastName, n.birthday")
    stats.stop()
    log.info(f"     Avg Latency: {stats.duration()/10*1000:.2f} ms")
    
    # IS3: Friends
    log.info("   [IS3] Friends Lookup")
    stats = Stats()
    stats.start()
    for i in range(10):
        pid = str(i * 10)
        query(f"MATCH (n:Person)-[:KNOWS]->(friend) WHERE n.id = '{pid}' RETURN friend.firstName, friend.creationDate")
    stats.stop()
    log.info(f"     Avg Latency: {stats.duration()/10*1000:.2f} ms")
    
    # IC1: Friends of Friends (Simplified)
    log.info("   [IC1] Friends of Friends")
    stats = Stats()
    stats.start()
    for i in range(5):
        pid = str(i * 20)
        query(f"MATCH (n:Person)-[:KNOWS]->(f1)-[:KNOWS]->(f2) WHERE n.id = '{pid}' RETURN count(f2)")
    stats.stop()
    log.info(f"     Avg Latency: {stats.duration()/5*1000:.2f} ms")

if __name__ == "__main__":
    load_data()
    run_benchmarks()
