import time
import requests
import json
import logging
from dataclasses import dataclass
from typing import List

# Setup
SERVER_URL = "http://localhost:3000/api/query"

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger("profiler")

@dataclass
class LatencyPoint:
    query: str
    client_total_ms: float
    server_exec_ms: float
    network_overhead_ms: float

def profile_query(cypher: str) -> LatencyPoint:
    start = time.perf_counter()
    res = requests.post(SERVER_URL, json={"query": cypher})
    end = time.perf_counter()
    
    if res.status_code != 200:
        raise Exception(f"Query failed: {res.text}")
    
    data = res.json()
    client_total = (end - start) * 1000
    server_exec = data.get("execution_time_ms", 0)
    overhead = client_total - server_exec
    
    return LatencyPoint(cypher, client_total, server_exec, overhead)

def run_profile():
    log.info("🔍 Starting NeuralGraphDB Latency Profiling...")
    log.info("=" * 70)
    log.info(f"{ 'Query Type':<25} | { 'Total':>8} | { 'Server':>8} | { 'Overhead':>8}")
    log.info("-" * 70)
    
    # 1. Warmup
    profile_query("MATCH (n) RETURN count(*)")
    
    queries = [
        ("Point Lookup (ID)", "MATCH (n) WHERE id(n) = 1 RETURN n.firstName"),
        ("1-Hop Traversal", "MATCH (n)-[:KNOWS]->(f) WHERE id(n) = 1 RETURN f.firstName"),
        ("2-Hop Traversal", "MATCH (n)-[:KNOWS]->(f1)-[:KNOWS]->(f2) WHERE id(n) = 1 RETURN count(f2)"),
        ("Aggregation", "MATCH (n:Person) RETURN avg(toInteger(n.id))"),
    ]
    
    for label, cypher in queries:
        # Run 5 times and take average
        samples: List[LatencyPoint] = []
        for _ in range(5):
            samples.append(profile_query(cypher))
        
        avg_total = sum(s.client_total_ms for s in samples) / 5
        avg_server = sum(s.server_exec_ms for s in samples) / 5
        avg_overhead = sum(s.network_overhead_ms for s in samples) / 5
        
        log.info(f"{label:<25} | {avg_total:8.2f} | {avg_server:8.2f} | {avg_overhead:8.2f} ms")

    log.info("=" * 70)
    log.info("💡 Insight: 'Overhead' includes HTTP parsing, JSON serialization, and network RTT.")
    log.info("   FalkorDB achieves <0.5ms by using a binary protocol (Redis) and zero-copy parsing.")

if __name__ == "__main__":
    try:
        run_profile()
    except Exception as e:
        log.error(f"Error: {e}")
        log.info("\nHint: Ensure the server is running with 'cargo run -p neural-cli --release -- serve'")
