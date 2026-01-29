//! Integration tests for distributed vector search.
//!
//! Tests correctness and performance of the scatter-gather search
//! and top-k merge algorithms.

use neural_core::NodeId;
use neural_storage::vector_index::{
    DistributedVectorConfig, SimulatedDistributedIndex, VectorIndex, VectorIndexConfig,
};
use std::collections::HashSet;
use std::time::{Duration, Instant};

/// Generate random-ish vectors for testing.
fn generate_test_vectors(count: usize, dim: usize) -> Vec<(NodeId, Vec<f32>)> {
    (0..count)
        .map(|i| {
            // Use deterministic pseudo-random values based on index
            let vec: Vec<f32> = (0..dim)
                .map(|d| {
                    let seed = (i * 31 + d * 17) as f32;
                    ((seed * 0.618033988749895).fract() * 2.0) - 1.0 // Range [-1, 1]
                })
                .collect();
            (NodeId::new(i as u64), vec)
        })
        .collect()
}

/// Normalize a vector to unit length.
fn normalize(v: &mut [f32]) {
    let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > 0.0 {
        for x in v.iter_mut() {
            *x /= norm;
        }
    }
}

#[test]
fn test_distributed_search_correctness() {
    // Test that distributed search returns same results as centralized

    let dim = 128;
    let num_vectors = 10_000;
    let num_shards = 4;
    let k = 100;

    // Generate test vectors
    let vectors = generate_test_vectors(num_vectors, dim);

    // Create centralized index
    let mut centralized = VectorIndex::with_config(VectorIndexConfig::large(dim));
    for (node_id, vec) in &vectors {
        centralized.add(*node_id, vec);
    }

    // Create distributed index (simulated, 4 shards)
    let mut distributed = SimulatedDistributedIndex::new(num_shards, dim);
    for (node_id, vec) in &vectors {
        distributed.add(*node_id, vec);
    }

    // Verify same total count
    assert_eq!(centralized.len(), distributed.len());
    assert_eq!(centralized.len(), num_vectors);

    // Run multiple random queries and compare results
    let num_queries = 100;
    let mut total_recall = 0.0;

    for q in 0..num_queries {
        // Generate query vector
        let mut query: Vec<f32> = (0..dim)
            .map(|d| {
                let seed = (q * 47 + d * 23) as f32;
                ((seed * 0.618033988749895).fract() * 2.0) - 1.0
            })
            .collect();
        normalize(&mut query);

        // Search both indices
        let centralized_results = centralized.search(&query, k);
        let distributed_results = distributed.search(&query, k);

        // Calculate recall
        let centralized_ids: HashSet<_> = centralized_results.iter().map(|(id, _)| id).collect();
        let distributed_ids: HashSet<_> = distributed_results.iter().map(|(id, _)| id).collect();

        let intersection = centralized_ids.intersection(&distributed_ids).count();
        let recall = intersection as f64 / k as f64;
        total_recall += recall;
    }

    let avg_recall = total_recall / num_queries as f64;
    println!(
        "Distributed search recall: {:.2}% (target: >= 80%)",
        avg_recall * 100.0
    );

    // Assert reasonable recall (at least 80%)
    // Note: Distributed HNSW cannot achieve same recall as centralized because
    // each shard builds an independent graph structure. The oversampling factor
    // helps but doesn't fully compensate.
    assert!(
        avg_recall >= 0.80,
        "Recall too low: {:.2}%, expected >= 80%",
        avg_recall * 100.0
    );
}

#[test]
fn test_distributed_search_latency() {
    // Test that distributed search meets latency targets

    let dim = 256;
    let num_vectors = 100_000; // 100k vectors per shard (400k total)
    let num_shards = 4;
    let k = 100;

    // Generate test vectors
    let vectors = generate_test_vectors(num_vectors, dim);

    // Create distributed index
    let mut distributed = SimulatedDistributedIndex::new(num_shards, dim);
    for (node_id, vec) in &vectors {
        distributed.add(*node_id, vec);
    }

    // Warm up (first query includes cache miss)
    let warm_query: Vec<f32> = vec![0.5; dim];
    let _ = distributed.search(&warm_query, k);

    // Run 1000 queries and measure latency
    let num_queries = 1000;
    let mut latencies: Vec<Duration> = Vec::with_capacity(num_queries);

    for q in 0..num_queries {
        let mut query: Vec<f32> = (0..dim)
            .map(|d| {
                let seed = (q * 41 + d * 19) as f32;
                (seed * 0.618033988749895).fract()
            })
            .collect();
        normalize(&mut query);

        let start = Instant::now();
        let _ = distributed.search(&query, k);
        latencies.push(start.elapsed());
    }

    // Sort for percentile calculation
    latencies.sort();

    let p50 = latencies[latencies.len() / 2];
    let p95 = latencies[(latencies.len() as f64 * 0.95) as usize];
    let p99 = latencies[(latencies.len() as f64 * 0.99) as usize];

    println!("Distributed search latency (simulated):");
    println!("  p50: {:?}", p50);
    println!("  p95: {:?}", p95);
    println!("  p99: {:?}", p99);

    // For simulated (in-memory) search, latency should be very low
    // Real distributed search with network would have higher latency
    assert!(
        p95 < Duration::from_millis(50),
        "p95 latency too high: {:?}",
        p95
    );
}

#[test]
fn test_distributed_search_caching() {
    let dim = 64;
    let num_vectors = 1000;

    let mut distributed = SimulatedDistributedIndex::new(4, dim);
    let vectors = generate_test_vectors(num_vectors, dim);
    for (node_id, vec) in &vectors {
        distributed.add(*node_id, vec);
    }

    let query = vec![0.5f32; dim];

    // First query - cache miss
    let start = Instant::now();
    let result1 = distributed.search(&query, 10);
    let first_latency = start.elapsed();

    // Second query - cache hit
    let start = Instant::now();
    let result2 = distributed.search(&query, 10);
    let cached_latency = start.elapsed();

    // Results should be identical
    assert_eq!(result1, result2);

    // Cache stats should show 1 hit, 1 miss
    let stats = distributed.cache_stats();
    assert_eq!(stats.hits, 1);
    assert_eq!(stats.misses, 1);

    // Cached query should be faster (or at least not slower)
    println!(
        "First query: {:?}, Cached query: {:?}",
        first_latency, cached_latency
    );
}

#[test]
fn test_distributed_search_shard_distribution() {
    let dim = 32;
    let num_vectors = 10_000;
    let num_shards = 4;

    let mut distributed = SimulatedDistributedIndex::new(num_shards, dim);
    let vectors = generate_test_vectors(num_vectors, dim);

    for (node_id, vec) in &vectors {
        distributed.add(*node_id, vec);
    }

    // Check that vectors are distributed across shards
    let mut shard_sizes = vec![];
    for shard in 0..num_shards {
        let size = distributed.shard_size(shard);
        shard_sizes.push(size);
    }

    println!("Shard distribution: {:?}", shard_sizes);

    // Each shard should have roughly 25% of vectors
    let expected = num_vectors / num_shards;
    let tolerance = expected / 5; // 20% tolerance

    for (shard, &size) in shard_sizes.iter().enumerate() {
        assert!(
            (size as i32 - expected as i32).unsigned_abs() < tolerance as u32,
            "Shard {} has {} vectors, expected ~{} (tolerance: {})",
            shard,
            size,
            expected,
            tolerance
        );
    }
}

#[test]
fn test_distributed_search_different_k_values() {
    let dim = 64;
    let num_vectors = 5000;

    let mut distributed = SimulatedDistributedIndex::new(4, dim);
    let vectors = generate_test_vectors(num_vectors, dim);
    for (node_id, vec) in &vectors {
        distributed.add(*node_id, vec);
    }

    let query = vec![0.5f32; dim];

    // Test different k values
    for k in [1, 10, 50, 100, 500] {
        let results = distributed.search(&query, k);

        // Should return exactly k results (or all if k > total)
        let expected_count = k.min(num_vectors);
        assert_eq!(
            results.len(),
            expected_count,
            "k={} should return {} results, got {}",
            k,
            expected_count,
            results.len()
        );

        // Results should be sorted by score (descending)
        for i in 1..results.len() {
            assert!(
                results[i - 1].1 >= results[i].1,
                "Results not sorted: {} < {} at position {}",
                results[i - 1].1,
                results[i].1,
                i
            );
        }
    }
}

#[test]
fn test_distributed_search_empty_shards() {
    let dim = 32;

    // Create distributed index with 4 shards but very few vectors
    let mut distributed = SimulatedDistributedIndex::new(4, dim);

    // Add just 2 vectors - some shards will be empty
    distributed.add(NodeId::new(0), &vec![1.0f32; dim]);
    distributed.add(NodeId::new(1), &vec![-1.0f32; dim]);

    let query = vec![0.5f32; dim];
    let results = distributed.search(&query, 10);

    // Should return both vectors even though most shards are empty
    assert_eq!(results.len(), 2);
}

#[test]
fn test_distributed_search_single_shard() {
    let dim = 64;
    let num_vectors = 1000;

    // Single shard - should behave like centralized search
    let mut distributed = SimulatedDistributedIndex::new(1, dim);
    let vectors = generate_test_vectors(num_vectors, dim);

    for (node_id, vec) in &vectors {
        distributed.add(*node_id, vec);
    }

    let query = vec![0.5f32; dim];
    let results = distributed.search(&query, 50);

    assert_eq!(results.len(), 50);
    assert_eq!(distributed.num_shards(), 1);
}

#[test]
fn test_distributed_config_builder() {
    let config = DistributedVectorConfig::new(8, 1536)
        .with_cache_size(50_000)
        .with_cache_ttl(Duration::from_secs(600))
        .with_timeout(Duration::from_millis(50))
        .with_oversampling_factor(2.0);

    assert_eq!(config.num_shards, 8);
    assert_eq!(config.dimension, 1536);
    assert_eq!(config.cache_size, 50_000);
    assert_eq!(config.cache_ttl, Duration::from_secs(600));
    assert_eq!(config.timeout, Duration::from_millis(50));
    assert!((config.oversampling_factor - 2.0).abs() < 0.01);
}
