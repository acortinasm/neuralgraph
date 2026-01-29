//! Benchmarks for distributed vector search.
//!
//! Run with: cargo bench --bench distributed_vector_bench

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use neural_core::NodeId;
use neural_storage::vector_index::{SimulatedDistributedIndex, VectorIndex, VectorIndexConfig};

/// Generate deterministic test vectors.
fn generate_vectors(count: usize, dim: usize) -> Vec<(NodeId, Vec<f32>)> {
    (0..count)
        .map(|i| {
            let vec: Vec<f32> = (0..dim)
                .map(|d| {
                    let seed = (i * 31 + d * 17) as f32;
                    ((seed * 0.618033988749895).fract() * 2.0) - 1.0
                })
                .collect();
            (NodeId::new(i as u64), vec)
        })
        .collect()
}

/// Generate a query vector.
fn generate_query(dim: usize, seed: usize) -> Vec<f32> {
    (0..dim)
        .map(|d| {
            let s = (seed * 41 + d * 19) as f32;
            (s * 0.618033988749895).fract()
        })
        .collect()
}

fn bench_shard_counts(c: &mut Criterion) {
    let mut group = c.benchmark_group("distributed_search_shards");
    group.sample_size(50);

    let dim = 256;
    let total_vectors = 100_000;
    let k = 100;

    for num_shards in [2, 4, 8] {
        // Create and populate index
        let vectors = generate_vectors(total_vectors, dim);
        let mut index = SimulatedDistributedIndex::new(num_shards, dim);
        for (node_id, vec) in &vectors {
            index.add(*node_id, vec);
        }

        group.throughput(Throughput::Elements(1));
        group.bench_with_input(
            BenchmarkId::from_parameter(num_shards),
            &num_shards,
            |b, _| {
                let query = generate_query(dim, 42);
                b.iter(|| {
                    black_box(index.search(black_box(&query), black_box(k)));
                })
            },
        );
    }

    group.finish();
}

fn bench_k_values(c: &mut Criterion) {
    let mut group = c.benchmark_group("distributed_search_k");
    group.sample_size(50);

    let dim = 256;
    let num_shards = 4;
    let total_vectors = 100_000;

    // Create and populate index
    let vectors = generate_vectors(total_vectors, dim);
    let mut index = SimulatedDistributedIndex::new(num_shards, dim);
    for (node_id, vec) in &vectors {
        index.add(*node_id, vec);
    }

    for k in [10, 100, 500] {
        group.throughput(Throughput::Elements(k as u64));
        group.bench_with_input(BenchmarkId::from_parameter(k), &k, |b, &k| {
            let query = generate_query(dim, 42);
            b.iter(|| {
                black_box(index.search(black_box(&query), black_box(k)));
            })
        });
    }

    group.finish();
}

fn bench_dimensions(c: &mut Criterion) {
    let mut group = c.benchmark_group("distributed_search_dim");
    group.sample_size(50);

    let num_shards = 4;
    let total_vectors = 50_000;
    let k = 100;

    for dim in [256, 768, 1536] {
        // Create and populate index
        let vectors = generate_vectors(total_vectors, dim);
        let mut index = SimulatedDistributedIndex::new(num_shards, dim);
        for (node_id, vec) in &vectors {
            index.add(*node_id, vec);
        }

        group.throughput(Throughput::Elements(1));
        group.bench_with_input(BenchmarkId::from_parameter(dim), &dim, |b, &dim| {
            let query = generate_query(dim, 42);
            b.iter(|| {
                black_box(index.search(black_box(&query), black_box(k)));
            })
        });
    }

    group.finish();
}

fn bench_centralized_vs_distributed(c: &mut Criterion) {
    let mut group = c.benchmark_group("centralized_vs_distributed");
    group.sample_size(50);

    let dim = 256;
    let total_vectors = 100_000;
    let k = 100;
    let num_shards = 4;

    let vectors = generate_vectors(total_vectors, dim);
    let query = generate_query(dim, 42);

    // Centralized index
    let mut centralized = VectorIndex::with_config(VectorIndexConfig::large(dim));
    for (node_id, vec) in &vectors {
        centralized.add(*node_id, vec);
    }

    // Distributed index
    let mut distributed = SimulatedDistributedIndex::new(num_shards, dim);
    for (node_id, vec) in &vectors {
        distributed.add(*node_id, vec);
    }

    group.bench_function("centralized", |b| {
        b.iter(|| {
            black_box(centralized.search(black_box(&query), black_box(k)));
        })
    });

    group.bench_function("distributed_4_shards", |b| {
        b.iter(|| {
            black_box(distributed.search(black_box(&query), black_box(k)));
        })
    });

    group.finish();
}

fn bench_cache_performance(c: &mut Criterion) {
    let mut group = c.benchmark_group("distributed_cache");
    group.sample_size(100);

    let dim = 256;
    let num_shards = 4;
    let total_vectors = 50_000;
    let k = 100;

    let vectors = generate_vectors(total_vectors, dim);
    let mut index = SimulatedDistributedIndex::new(num_shards, dim);
    for (node_id, vec) in &vectors {
        index.add(*node_id, vec);
    }

    // Single query that will be cached
    let cached_query = generate_query(dim, 42);

    // Prime the cache
    let _ = index.search(&cached_query, k);

    group.bench_function("cache_hit", |b| {
        b.iter(|| {
            black_box(index.search(black_box(&cached_query), black_box(k)));
        })
    });

    // Different queries each time (cache miss)
    group.bench_function("cache_miss", |b| {
        let mut seed = 0;
        b.iter(|| {
            let query = generate_query(dim, seed);
            seed += 1;
            black_box(index.search(black_box(&query), black_box(k)));
        })
    });

    group.finish();
}

fn bench_throughput(c: &mut Criterion) {
    let mut group = c.benchmark_group("distributed_throughput");
    group.sample_size(20);

    let dim = 256;
    let num_shards = 4;
    let total_vectors = 100_000;
    let k = 100;
    let queries_per_batch = 100;

    let vectors = generate_vectors(total_vectors, dim);
    let mut index = SimulatedDistributedIndex::new(num_shards, dim);
    for (node_id, vec) in &vectors {
        index.add(*node_id, vec);
    }

    // Pre-generate queries
    let queries: Vec<_> = (0..queries_per_batch)
        .map(|i| generate_query(dim, i))
        .collect();

    group.throughput(Throughput::Elements(queries_per_batch as u64));
    group.bench_function("batch_queries", |b| {
        b.iter(|| {
            for query in &queries {
                black_box(index.search(black_box(query), black_box(k)));
            }
        })
    });

    group.finish();
}

criterion_group!(
    benches,
    bench_shard_counts,
    bench_k_values,
    bench_dimensions,
    bench_centralized_vs_distributed,
    bench_cache_performance,
    bench_throughput,
);

criterion_main!(benches);
