
import time
import numpy as np
import argparse
import os
from neuralgraph import NGraphClient

# Configuration
DIM = 128
BATCH_SIZE = 500

def generate_vectors(count, dim):
    """Generates synthetic vectors on the unit hypersphere."""
    print(f"Generating {count} vectors of dimension {dim}...")
    vectors = np.random.rand(count, dim).astype(np.float32)
    # Normalize to unit length for Cosine Similarity
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    vectors = vectors / norms
    return vectors

def run_benchmark(count, dim, host="127.0.0.1", http_port=8000):
    client = NGraphClient(host=host, http_port=http_port)
    
    # 1. Reset / Init
    print("Connecting to NeuralGraphDB...")
    # Ideally we should clear the DB here, but client might not support it yet.
    # Assuming fresh start or appended data.
    
    vectors = generate_vectors(count, dim)
    
    # 2. Ingestion
    print("Starting ingestion...")
    start_time = time.time()
    
    for i in range(0, count, BATCH_SIZE):
        batch = vectors[i : i + BATCH_SIZE]
        # Construct CREATE query
        # This is inefficient string building, but tests the parser/network too.
        # Ideally we want a bulk insert API.
        # For 1M vectors, string queries might be the bottleneck, not the index.
        # Let's try to use a more optimized approach if available, otherwise just CREATE.
        
        # We will create nodes with embedding property
        # MATCH (n) is slow for mass insert.
        # We'll generate a massive CREATE statement? No, batch it.
        
        query_parts = []
        for j, vec in enumerate(batch):
            global_id = i + j
            vec_str = str(vec.tolist())
            query_parts.append(f"(:Item {{id: {global_id}, embedding: {vec_str}}})")
        
        query = "CREATE " + ", ".join(query_parts)
        
        try:
            client.execute(query)
        except Exception as e:
            print(f"Error at batch {i}: {e}")
            break
            
        if i % 10000 == 0 and i > 0:
            elapsed = time.time() - start_time
            rate = i / elapsed
            print(f"  Inserted {i}/{count} ({rate:.2f} items/sec)")

    total_time = time.time() - start_time
    print(f"Ingestion complete. Total time: {total_time:.2f}s ({count/total_time:.2f} items/sec)")
    
    # 3. Search Benchmark
    print("Starting search benchmark...")
    query_vec = vectors[0].tolist()
    
    start_search = time.time()
    runs = 100
    for _ in range(runs):
        client.execute(f"MATCH (n:Item) WHERE vector_similarity(n.embedding, {query_vec}) > 0.9 RETURN n.id LIMIT 10")
    
    avg_latency = (time.time() - start_search) / runs * 1000
    print(f"Average Search Latency (Top-10): {avg_latency:.2f} ms")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Vector Scale Benchmark (1M)")
    parser.add_argument("--count", type=int, default=10000, help="Number of vectors (default 10k, aim for 1M)")
    parser.add_argument("--dim", type=int, default=128, help="Vector dimension")
    parser.add_argument("--port", type=int, default=8000, help="HTTP Port")
    args = parser.parse_args()
    
    run_benchmark(args.count, args.dim, http_port=args.port)
