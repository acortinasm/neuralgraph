
⏺ Distributed Vector Search Concepts

  1. Vector Index (HNSW)

  What it is: HNSW (Hierarchical Navigable Small World) is a graph-based algorithm
   for approximate nearest neighbor search. It builds a multi-layer graph where
  each node connects to its closest neighbors.

  How it works:
  - Vectors are organized in hierarchical layers (like an express train system)
  - Top layers have fewer nodes for fast coarse navigation
  - Bottom layers have all nodes for precise search
  - Search starts at top, greedily descends to find nearest neighbors

  Metrics:
  - Build time: Time to construct the index (45s for 100k vectors)
  - Throughput (qps): Queries per second the index can handle
  - Latency p50/p95/p99: Response time percentiles (p95 = 95% of queries are
  faster)

  ---
  2. Scatter-Gather Search

  What it is: A distributed query pattern where a coordinator sends the same query
   to all shards in parallel, then merges the results.

          Query
            │
      ┌─────┼─────┐
      ▼     ▼     ▼
   Shard0 Shard1 Shard2   ← Scatter (parallel)
      │     │     │
      └─────┼─────┘
            ▼
         Merge            ← Gather
            │
         Results

  Why it's needed: When you have 1 billion vectors, no single machine can hold
  them all. You split them across nodes and search all simultaneously.

  ---
  3. Top-K Merge

  What it is: Algorithm to combine results from multiple shards and return only
  the best K results.

  How it works:
  Shard 0: [(A, 0.95), (B, 0.87), (C, 0.82)]
  Shard 1: [(D, 0.91), (E, 0.85), (F, 0.79)]
  Shard 2: [(G, 0.89), (H, 0.84), (I, 0.76)]

  Merged top-5: [(A, 0.95), (D, 0.91), (G, 0.89), (B, 0.87), (E, 0.85)]

  Uses a min-heap data structure for O(n log k) efficiency instead of sorting
  everything.

  ---
  4. Oversampling Factor

  What it is: Request more results from each shard than you actually need, to
  improve final accuracy.

  Why it's needed: If you want top-100 globally but only ask each shard for 100,
  you might miss good results. With 1.5x oversampling, you ask each shard for 150,
   then merge and take top-100.

  Trade-off: Higher oversampling = better recall but more network/computation.

  ---
  5. Query Result Cache (LRU)

  What it is: Stores recent query results to avoid re-computing identical
  searches.

  LRU (Least Recently Used): When cache is full, evict the entry that hasn't been
  accessed longest.

  SimHash: Converts a float vector into a hash for cache key lookup. Similar
  vectors get similar hashes.

  Query [0.5, 0.3, 0.8] → Hash: 0xABCD1234 → Cache lookup
    Hit?  → Return cached results
    Miss? → Execute search, store in cache

  ---
  6. Load Balancer

  What it is: Distributes queries across replica nodes to optimize performance.

  Strategies:
  - Round-robin: Rotate through replicas sequentially (simple, fair)
  - Latency-aware: Prefer replicas with lower response times
  - Weighted: Assign more traffic to stronger nodes

  ---
  7. Flash Quantization

  What it is: Compresses vectors from 32-bit floats to 8-bit integers for 4x
  memory savings.

  Original (f32):  [0.523, -0.871, 0.142, ...]  → 4 bytes per value
  Quantized (i8):  [67, -111, 18, ...]          → 1 byte per value

  Trade-off: Slight accuracy loss for major memory reduction. Essential for
  billion-scale deployments.

  Compression ratio: 4.0x means 1GB of vectors becomes 250MB.

  ---
  8. Graph Sharding

  What it is: Splitting a graph across multiple machines.

  Strategies:
  - Hash partitioning: shard = hash(node_id) % num_shards - simple, even
  distribution
  - Range partitioning: Nodes 0-999 on shard 0, 1000-1999 on shard 1, etc.
  - Community partitioning: Keep densely connected nodes together

  ---
  9. Edge Cut Percentage

  What it is: The percentage of edges that cross shard boundaries.

  Shard 0: [A, B, C]     Shard 1: [D, E, F]

  Edges: A-B (local), B-C (local), C-D (cut!), D-E (local)

  Edge cut = 1/4 = 25%

  Why it matters: Cross-shard edges require network communication. Lower edge cut
  = faster graph traversal.

  - Hash: 75% - Random distribution, many cross-shard edges
  - Range: 65.7% - Better if IDs correlate with relationships
  - Community: ~30-40% - Best, but expensive to compute

  ---
  10. Recall

  What it is: Accuracy measure for approximate search - what percentage of true
  nearest neighbors did we find?

  True top-10:        [A, B, C, D, E, F, G, H, I, J]
  Approximate top-10: [A, B, C, D, E, F, G, K, L, M]  ← Found 7/10

  Recall = 70%

  96.6% recall means the distributed search finds 96.6% of the same results as a
  perfect centralized search.

  ---
  11. Latency Percentiles (p50, p95, p99)

  What they mean:
  - p50 (median): 50% of queries are faster than this
  - p95: 95% of queries are faster (typical SLA target)
  - p99: 99% of queries are faster (catches outliers)

  1000 queries sorted by latency:
    Query #500  → p50 = 0.2ms
    Query #950  → p95 = 0.46ms
    Query #990  → p99 = 1.2ms

  p95 < 100ms was our target, achieved 0.46ms (simulated, no network).

✻ Brewed for 34s
