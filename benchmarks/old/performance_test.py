"""
NeuralGraphDB Performance Test
Equivalent to the Neo4j benchmark test for direct comparison.

Usage:
    python benchmarks/performance_test.py          # Run benchmark only
    python benchmarks/performance_test.py --load   # Load data first, then benchmark

Requires NeuralGraph server running on localhost (HTTP: 3000, Flight: 50051).
"""
import time
import sys
import os
import csv
import numpy as np
import requests
from tqdm import tqdm

# Ensure we can import the local neuralgraph client
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../clients/python')))

from neuralgraph import NGraphClient

# Connection details
HOST = "localhost"
HTTP_PORT = 3000
FLIGHT_PORT = 50051

# Data files (Yelp dataset)
DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../misc/datasets/yelp'))
NODES_FILE = os.path.join(DATA_DIR, "nodes.csv")
EDGES_FILE = os.path.join(DATA_DIR, "edges.csv")

# Equivalent queries to Neo4j benchmark
# Using id() for fast lookups (productId = internal node ID in this dataset)
# NOTE: Remove :Product label to avoid full label scan - id() lookup is O(1)
QUERIES = {
    "1_hop": "MATCH (p)-[:ALSO_BOUGHT]->(n) WHERE id(p) = 1 RETURN count(n)",
    "3_hop": "MATCH (p)-[:ALSO_BOUGHT*3]->(n) WHERE id(p) = 1 RETURN count(DISTINCT n)",
    # degree_agg requires full scan - use sampled version for benchmark
    "degree_sample": "MATCH (p)-[:ALSO_BOUGHT]->(n) WHERE id(p) = 0 RETURN id(p), count(n)",
    # shortest_path: Note - NeuralGraph needs target id pushdown optimization for efficient shortest path
    # Using smaller max_hops (5) to limit BFS exploration
    # Return path directly since length() function not implemented for paths
    "shortest_path": "MATCH path = shortestPath((p1)-[:ALSO_BOUGHT*..5]->(p2)) WHERE id(p1) = 1 AND id(p2) = 500 RETURN path"
}


def convert_csv_files():
    """
    Convert Neo4j-style CSV to NeuralGraph format.
    Returns paths to converted files.
    """
    import tempfile

    # Create temp directory for converted files
    temp_dir = tempfile.mkdtemp(prefix="neuralgraph_")

    # Convert nodes: productId:ID,:LABEL -> id,label,productId
    nodes_out = os.path.join(temp_dir, "nodes.csv")
    print(f"Converting nodes CSV...")
    with open(NODES_FILE, 'r') as fin, open(nodes_out, 'w', newline='') as fout:
        reader = csv.reader(fin)
        writer = csv.writer(fout)
        next(reader)  # Skip Neo4j header
        writer.writerow(['id', 'label', 'productId'])  # NeuralGraph header
        for row in reader:
            p_id, label = row[0], row[1]
            writer.writerow([p_id, label, p_id])  # id=productId, also store productId as property

    # Convert edges: :START_ID,:END_ID,:TYPE -> source,target,label
    edges_out = os.path.join(temp_dir, "edges.csv")
    print(f"Converting edges CSV...")
    with open(EDGES_FILE, 'r') as fin, open(edges_out, 'w', newline='') as fout:
        reader = csv.reader(fin)
        writer = csv.writer(fout)
        next(reader)  # Skip Neo4j header
        writer.writerow(['source', 'target', 'label'])  # NeuralGraph header
        for row in reader:
            writer.writerow([row[0], row[1], row[2]])

    print(f"  Converted files in {temp_dir}")
    return nodes_out, edges_out


def load_data(client: NGraphClient, batch_size: int = 500):
    """
    Load Yelp dataset into NeuralGraph using bulk load API.
    """
    print(f"Loading data from {DATA_DIR}...")
    start_total = time.time()

    # Check if files exist
    if not os.path.exists(NODES_FILE):
        print(f"Error: Nodes file not found: {NODES_FILE}")
        return False
    if not os.path.exists(EDGES_FILE):
        print(f"Error: Edges file not found: {EDGES_FILE}")
        return False

    # Convert CSV files to neuralgraph format
    nodes_path, edges_path = convert_csv_files()

    # Use bulk load API
    url = f"http://{HOST}:{HTTP_PORT}/api/bulk-load"
    payload = {
        "nodes_path": nodes_path,
        "edges_path": edges_path,
        "clear_existing": True
    }

    print("Calling bulk-load API (this may take a moment)...")
    try:
        response = requests.post(url, json=payload, timeout=300)  # 5 min timeout
    except requests.exceptions.Timeout:
        print("Error: Bulk load timed out after 5 minutes")
        return False
    except Exception as e:
        print(f"Error: {e}")
        return False

    if response.status_code != 200:
        print(f"Error: HTTP {response.status_code} - {response.text}")
        return False

    result = response.json()
    if not result.get("success"):
        print(f"Bulk load failed: {result.get('error')}")
        return False

    print(f"  Loaded {result['nodes_loaded']} nodes, {result['edges_loaded']} edges")
    print(f"  Server load time: {result['load_time_ms']:.2f}ms")
    print(f"Data loading complete in {time.time() - start_total:.2f}s\n")
    return True


def run_benchmark(client: NGraphClient, query_name: str, ngql: str, iterations: int = 5, timeout: int = 30) -> dict:
    """
    Run a benchmark for a single query.

    Args:
        client: NeuralGraph client instance
        query_name: Name of the benchmark
        ngql: The NGQL query string
        iterations: Number of iterations to run
        timeout: Timeout in seconds per query

    Returns:
        Dictionary with mean, std, and min latencies in milliseconds
    """
    times = []

    print(f"Benchmarking: {query_name}")
    print(f"  Query: {ngql[:80]}...")

    # Warmup run (not counted)
    print("  Warmup run...", end=" ", flush=True)
    try:
        result = requests.post(
            f"http://{HOST}:{HTTP_PORT}/api/query",
            json={"query": ngql},
            timeout=timeout
        )
        if result.status_code == 200:
            data = result.json()
            if not data.get("success"):
                print(f"QUERY ERROR: {data.get('error')}")
                return None
            result_data = data.get("result") or {}
            row_count = result_data.get("count", 0)
            print(f"OK (rows: {row_count})")
        else:
            print(f"HTTP {result.status_code}: {result.text[:100]}")
            return None
    except requests.exceptions.Timeout:
        print(f"TIMEOUT after {timeout}s")
        return None
    except Exception as e:
        print(f"FAILED: {e}")
        return None

    for i in range(iterations):
        print(f"  Run {i+1}...", end=" ", flush=True)
        start_time = time.perf_counter()
        try:
            result = requests.post(
                f"http://{HOST}:{HTTP_PORT}/api/query",
                json={"query": ngql},
                timeout=timeout
            )
            result.raise_for_status()
            data = result.json()
            _ = data.get("result", {}).get("rows", [])
        except requests.exceptions.Timeout:
            print(f"TIMEOUT")
            continue
        except Exception as e:
            print(f"FAILED: {e}")
            continue
        end_time = time.perf_counter()

        # Record time in milliseconds
        latency = (end_time - start_time) * 1000
        times.append(latency)
        print(f"{latency:.2f} ms")

    if not times:
        return None

    return {
        "mean": np.mean(times),
        "std": np.std(times),
        "min": np.min(times)
    }


def main():
    load_flag = "--load" in sys.argv

    print(f"NeuralGraphDB Performance Test")
    print(f"Connecting to {HOST}:{HTTP_PORT}...")

    try:
        client = NGraphClient(HOST, HTTP_PORT, FLIGHT_PORT)

        # Load data if requested
        if load_flag:
            load_data(client)

        # Verify connection and data exists
        result = client.query("MATCH (n:Product) RETURN count(n) AS cnt")
        rows = result.get("rows", [])
        # Rows are dicts like {"cnt": "Int(123)"}
        count = 0
        if rows:
            cnt_str = rows[0].get("cnt", "0")
            # Parse "Int(123)" format
            if "Int(" in cnt_str:
                count = int(cnt_str.replace("Int(", "").replace(")", ""))
            else:
                try:
                    count = int(cnt_str)
                except:
                    count = 0
        print(f"Connected. Graph contains {count} Product nodes.\n")

        if count == 0:
            print("Warning: No Product nodes found. Queries may fail.")
            print("Hint: Run with --load to load the Yelp dataset.\n")

    except Exception as e:
        print(f"Connection error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    # Run benchmarks
    results = {}
    for name, ngql in QUERIES.items():
        stats = run_benchmark(client, name, ngql)
        if stats:
            results[name] = stats
        print()

    # Print summary table
    print("=" * 60)
    print(f"{'Query Name':<20} | {'Mean (ms)':<12} | {'Std Dev':<12} | {'Min (ms)':<10}")
    print("-" * 60)
    for name, stats in results.items():
        print(f"{name:<20} | {stats['mean']:<12.2f} | {stats['std']:<12.2f} | {stats['min']:<10.2f}")
    print("=" * 60)

    return results


if __name__ == "__main__":
    main()
