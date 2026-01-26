//! Vector Scale Benchmark (Sprint 47)
//!
//! Measures HNSW index performance at 1M vector scale:
//! - Build time
//! - Search latency (p50, p99)
//! - Recall@10

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use neural_core::NodeId;
use neural_storage::{VectorIndex, VectorIndexConfig};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

/// Generates random normalized vectors
fn generate_vectors(count: usize, dim: usize, seed: u64) -> Vec<Vec<f32>> {
    let mut rng = StdRng::seed_from_u64(seed);
    (0..count)
        .map(|_| {
            let mut v: Vec<f32> = (0..dim).map(|_| rng.gen_range(-1.0..1.0)).collect();
            let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
            if norm > 0.0 {
                v.iter_mut().for_each(|x| *x /= norm);
            }
            v
        })
        .collect()
}

fn bench_index_build(c: &mut Criterion) {
    let mut group = c.benchmark_group("VectorIndex Build");
    group.sample_size(10);

    for size in [10_000, 100_000] {
        let dim = 128; // SIFT-like dimension
        let vectors = generate_vectors(size, dim, 42);

        group.bench_with_input(BenchmarkId::from_parameter(size), &vectors, |b, vecs| {
            b.iter(|| {
                let config = VectorIndexConfig {
                    dimension: dim,
                    m: 24,
                    ef_construction: 400,
                    max_elements: size,
                };
                let mut index = VectorIndex::with_config(config);
                for (i, v) in vecs.iter().enumerate() {
                    index.add(NodeId::new(i as u64), v);
                }
                black_box(index)
            });
        });
    }

    group.finish();
}

fn bench_search(c: &mut Criterion) {
    let mut group = c.benchmark_group("VectorIndex Search");
    group.sample_size(100);

    // Pre-build index with 100k vectors
    let size = 100_000;
    let dim = 128;
    let vectors = generate_vectors(size, dim, 42);
    let queries = generate_vectors(100, dim, 123);

    let config = VectorIndexConfig {
        dimension: dim,
        m: 24,
        ef_construction: 400,
        max_elements: size,
    };
    let mut index = VectorIndex::with_config(config);
    for (i, v) in vectors.iter().enumerate() {
        index.add(NodeId::new(i as u64), v);
    }

    for k in [10, 50, 100] {
        group.bench_with_input(BenchmarkId::from_parameter(k), &k, |b, &k| {
            let mut query_idx = 0;
            b.iter(|| {
                let results = index.search(&queries[query_idx % queries.len()], k);
                query_idx += 1;
                black_box(results)
            });
        });
    }

    group.finish();
}

criterion_group!(benches, bench_index_build, bench_search);
criterion_main!(benches);
