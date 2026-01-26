import time
import sys
import os
import csv
import numpy as np
import requests
from tqdm import tqdm

# Ensure we can import the local neuralgraph client
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../clients/python')))

from neuralgraph import NGraphClient

# Connection details
HOST = "localhost"
HTTP_PORT = 3000
FLIGHT_PORT = 50051

DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../misc/neo4j/import'))
NODES_FILE = os.path.join(DATA_DIR, "nodes.csv")
EDGES_FILE = os.path.join(DATA_DIR, "edges.csv")

QUERIES = {
    "1_hop": "MATCH (p:Product {productId: '1'})-[:ALSO_BOUGHT]->(n) RETURN count(n)",
    "3_hop": "MATCH (p:Product {productId: '1'})-[:ALSO_BOUGHT*3]->(n) RETURN count(n)",
    "degree_agg": "MATCH (p:Product)<-[r:ALSO_BOUGHT]-() RETURN p.productId, count(r) ORDER BY count(r) DESC LIMIT 10",
    "shortest_path": "MATCH (p1:Product {productId: '1'}), (p2:Product {productId: '50'}) MATCH p = shortestPath((p1)-[:ALSO_BOUGHT*..10]-(p2)) RETURN length(p)"
}

class BatchedNGraphClient(NGraphClient):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.session = requests.Session()

    def query(self, ngql: str):
        url = f"{self.http_base_url}/query"
        try:
            response = self.session.post(url, json={"query": ngql})
            response.raise_for_status()
            data = response.json()
            if not data.get("success"):
                raise Exception(f"Query failed: {data.get('error')}")
            return data.get("result", {{}})
        except Exception as e:
            raise Exception(f"Batch execution failed: {e}")

def load_data(client, batch_size=500):
    print(f"Loading data from {DATA_DIR}...")
    start_total = time.time()
    
    print("Clearing existing data...")
    client.query("MATCH (n) DETACH DELETE n")
    
    # 1. Load Nodes
    print("  Loading nodes (batched)...")
    nodes_data = []
    with open(NODES_FILE, 'r') as f:
        reader = csv.reader(f)
        next(reader) # Skip header
        for row in reader:
            nodes_data.append(row) # p_id, label

    for i in tqdm(range(0, len(nodes_data), batch_size), desc="Nodes"):
        batch = nodes_data[i:i+batch_size]
        create_clauses = []
        for row in batch:
            p_id, label = row[0], row[1]
            create_clauses.append(f"CREATE (p:{label} {{productId: '{p_id}'}})")
        
        # Combine multiple CREATE statements
        query = " ".join(create_clauses)
        client.query(query)
                
    print(f"  ✓ Loaded {len(nodes_data)} nodes.")

    # 2. Load Edges
    print("  Loading edges (batched)...")
    # Note: We need internal IDs for edges. 
    # In NeuralGraphDB, if we create nodes with specific IDs it's easier,
    # but currently internal IDs are auto-assigned.
    # However, our MATCH uses productId which is indexed.
    
    edges_rows = []
    with open(EDGES_FILE, 'r') as f:
        reader = csv.reader(f)
        next(reader) # Skip header
        for row in reader:
            edges_rows.append(row) # start_id, end_id, type

    for i in tqdm(range(0, len(edges_rows), batch_size), desc="Edges"):
        batch = edges_rows[i:i+batch_size]
        create_clauses = []
        for row in batch:
            u, v, etype = row[0], row[1], row[2]
            # Since we don't have a node_map for internal IDs here easily,
            # we use MATCH and CREATE. This is slightly slower but robust.
            create_clauses.append(f"MATCH (a:Product {{productId: '{u}'}}), (b:Product {{productId: '{v}'}}) CREATE (a)-[:{etype}]->(b)")
        
        # Combine multiple MATCH...CREATE statements
        query = " ".join(create_clauses)
        client.query(query)

    print(f"  ✓ Loaded {len(edges_rows)} edges.")
    print(f"Data loading complete in {time.time() - start_total:.2f}s")

def run_benchmark(client, query_name, ngql, iterations=5):
    times = []
    print(f"\nBenchmarking: {query_name}")
    
    # Warmup
    try:
        client.execute(ngql)
    except Exception as e:
        print(f"  Warmup failed: {e}")
        return None

    for i in range(iterations):
        start_time = time.perf_counter()
        try:
            result = client.execute(ngql)
            # Ensure full consumption
            _ = len(result)
        except Exception as e:
            print(f"  Run {i+1} failed: {e}")
            continue
        latency = (time.perf_counter() - start_time) * 1000
        times.append(latency)
        print(f"  Run {i+1}: {latency:.2f} ms")
            
    return {"mean": np.mean(times), "std": np.std(times)} if times else None

if __name__ == "__main__":
    print(f"Connecting to NeuralGraph at {HOST}...")
    try:
        client = BatchedNGraphClient(HOST, HTTP_PORT, FLIGHT_PORT)
        # Check if data already exists to skip loading
        res = client.query("MATCH (n:Product) RETURN count(n)")
        count = int(res.get('rows', [[0]])[0][0])
        if count == 0:
            load_data(client)
        else:
            print(f"Graph already contains {count} products. Skipping load.")
            print("Hint: Use client.delete_all() if you want to reload.")
    except Exception as e:
        print(f"Initialization error: {e}")
        sys.exit(1)

    results = {}
    for name, ngql in QUERIES.items():
        stats = run_benchmark(client, name, ngql)
        if stats:
            results[name] = stats

    print("\n" + "="*50)
    print(f"{ 'Query Name':<20} | { 'Mean (ms)':<10} | { 'Std Dev':<10}")
    print("-" * 50)
    for name, stats in results.items():
        print(f"{name:<20} | {stats['mean']:<10.2f} | {stats['std']:<10.2f}")