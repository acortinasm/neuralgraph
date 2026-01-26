import time
import logging
from dataclasses import dataclass
from typing import List
import pyarrow.flight as flight
import json

# Setup
FLIGHT_URI = "grpc://localhost:50051"

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger("profiler_flight")

@dataclass
class LatencyPoint:
    query: str
    total_ms: float

def profile_query(client: flight.FlightClient, cypher: str) -> LatencyPoint:
    start = time.perf_counter()
    
    # Flight "DoGet" with query as ticket
    ticket = flight.Ticket(cypher.encode('utf-8'))
    reader = client.do_get(ticket)
    table = reader.read_all()
    
    end = time.perf_counter()
    total_ms = (end - start) * 1000
    
    # We can't easily get "Server Exec Time" from Flight metadata yet without custom headers
    # So we compare Total Time directly against HTTP Total Time.
    return LatencyPoint(cypher, total_ms)

def run_profile():
    log.info("🔍 Starting NeuralGraphDB Flight Latency Profiling...")
    log.info("=" * 50)
    log.info(f"{'Query Type':<25} | {'Total':>8}")
    log.info("-" * 50)
    
    client = flight.FlightClient(FLIGHT_URI)
    
    # 1. Warmup
    try:
        profile_query(client, "MATCH (n) RETURN count(*)")
    except Exception as e:
        log.error(f"Failed to connect to Flight: {e}")
        return

    queries = [
        ("Point Lookup (ID)", "MATCH (n) WHERE id(n) = 1 RETURN n.firstName"),
        ("1-Hop Traversal", "MATCH (n)-[:KNOWS]->(f) WHERE id(n) = 1 RETURN f.firstName"),
        ("2-Hop Traversal", "MATCH (n)-[:KNOWS]->(f1)-[:KNOWS]->(f2) WHERE id(n) = 1 RETURN count(f2)"),
        ("Aggregation", "MATCH (n:Person) RETURN avg(toInteger(n.id))"),
    ]
    
    for label, cypher in queries:
        samples = []
        for _ in range(10): # More samples for stability
            samples.append(profile_query(client, cypher))
        
        avg_total = sum(s.total_ms for s in samples) / len(samples)
        log.info(f"{label:<25} | {avg_total:8.2f} ms")

    log.info("=" * 50)

if __name__ == "__main__":
    run_profile()
