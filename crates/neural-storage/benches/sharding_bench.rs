//! Sharding Benchmark (Sprint 55)
//!
//! Measures cross-shard query performance:
//! - Query latency per hop
//! - Edge cut percentage
//! - Coordinator overhead

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use neural_core::{NodeId, Edge};
use neural_storage::sharding::{
    HashPartition, RangePartition, CommunityPartition, PartitionStrategy, ShardManager, ShardConfig,
    ShardCoordinator, CoordinatorConfig, ExecutionStrategy, ShardResult,
};
use std::sync::Arc;
use std::time::Instant;

/// Generates a scale-free graph (power-law degree distribution)
/// Returns (edges, node_count)
fn generate_scale_free_graph(num_nodes: usize, avg_degree: usize, seed: u64) -> Vec<Edge> {
    use rand::rngs::StdRng;
    use rand::{Rng, SeedableRng};

    let mut rng = StdRng::seed_from_u64(seed);
    let mut edges = Vec::new();
    let mut degrees: Vec<usize> = vec![0; num_nodes];

    // Preferential attachment (Barabási–Albert model)
    let m = avg_degree / 2; // edges per new node

    // Initial complete graph on m+1 nodes
    for i in 0..=m {
        for j in (i+1)..=m {
            edges.push(Edge::new(i as u64, j as u64));
            degrees[i] += 1;
            degrees[j] += 1;
        }
    }

    // Add remaining nodes with preferential attachment
    for new_node in (m+1)..num_nodes {
        let total_degree: usize = degrees.iter().sum();
        let mut targets = Vec::new();

        while targets.len() < m {
            let r: f64 = rng.r#gen();
            let mut cumulative = 0.0;

            for (node, &degree) in degrees.iter().enumerate().take(new_node) {
                cumulative += (degree + 1) as f64 / (total_degree + new_node) as f64;
                if r < cumulative && !targets.contains(&node) {
                    targets.push(node);
                    break;
                }
            }
        }

        for target in targets {
            edges.push(Edge::new(new_node as u64, target as u64));
            degrees[new_node] += 1;
            degrees[target] += 1;
        }
    }

    edges
}

/// Calculates edge cut percentage for a partitioning strategy
fn calculate_edge_cut_percentage(
    edges: &[Edge],
    strategy: &dyn PartitionStrategy,
) -> f64 {
    let cross_shard_edges = edges
        .iter()
        .filter(|e| {
            let source_shard = strategy.shard_for_node(e.source);
            let target_shard = strategy.shard_for_node(e.target);
            source_shard != target_shard
        })
        .count();

    (cross_shard_edges as f64 / edges.len() as f64) * 100.0
}

/// Simulates cross-shard hop latency
fn simulate_shard_hop_latency(coordinator: &ShardCoordinator, hops: u32) -> std::time::Duration {
    let start = Instant::now();

    // Simulate query execution across shards
    for _ in 0..hops {
        let router = coordinator.router();
        let plan = router.plan_full_scan();

        // Simulate parallel execution on shards
        let _ = coordinator.execute_plan(&plan, |shard_id, _info| {
            // Simulate local query execution (< 1ms)
            std::thread::sleep(std::time::Duration::from_micros(100));
            Ok(ShardResult::new(shard_id, vec![], 100))
        });
    }

    start.elapsed()
}

fn bench_edge_cut_percentage(c: &mut Criterion) {
    let mut group = c.benchmark_group("Edge Cut Percentage");
    group.sample_size(10);

    // Generate test graphs of different sizes
    for node_count in [10_000, 100_000] {
        let edges = generate_scale_free_graph(node_count, 10, 42);

        // Test hash partitioning with 4 shards
        let hash_strategy = HashPartition::new(4);
        let hash_cut = calculate_edge_cut_percentage(&edges, &hash_strategy);

        group.bench_with_input(
            BenchmarkId::new("Hash/4shards", node_count),
            &(&edges, &hash_strategy),
            |b, (edges, strategy)| {
                b.iter(|| {
                    black_box(calculate_edge_cut_percentage(edges, *strategy))
                });
            },
        );

        println!("Hash partitioning edge cut ({}): {:.2}%", node_count, hash_cut);

        // Test range partitioning
        let range_strategy = RangePartition::uniform(4, node_count as u64);
        let range_cut = calculate_edge_cut_percentage(&edges, &range_strategy);

        group.bench_with_input(
            BenchmarkId::new("Range/4shards", node_count),
            &(&edges, &range_strategy),
            |b, (edges, strategy)| {
                b.iter(|| {
                    black_box(calculate_edge_cut_percentage(edges, *strategy))
                });
            },
        );

        println!("Range partitioning edge cut ({}): {:.2}%", node_count, range_cut);

        // Test community partitioning (graph-aware)
        let community_strategy = CommunityPartition::from_edges(&edges, node_count, 4);
        let community_cut = community_strategy.edge_cut_percentage(&edges);

        group.bench_with_input(
            BenchmarkId::new("Community/4shards", node_count),
            &(&edges, &community_strategy),
            |b, (edges, strategy)| {
                b.iter(|| {
                    black_box(strategy.edge_cut_percentage(edges))
                });
            },
        );

        println!("Community partitioning edge cut ({}): {:.2}%", node_count, community_cut);
    }

    group.finish();
}

fn bench_cross_shard_latency(c: &mut Criterion) {
    let mut group = c.benchmark_group("Cross-Shard Latency");
    group.sample_size(50);

    // Setup coordinator with 4 shards
    let mut manager = ShardManager::new(ShardConfig::hash(4));
    manager.set_local_shard(0, "localhost:9000".to_string());
    manager.update_shard_address(1, "localhost:9001".to_string());
    manager.update_shard_address(2, "localhost:9002".to_string());
    manager.update_shard_address(3, "localhost:9003".to_string());

    let config = CoordinatorConfig {
        strategy: ExecutionStrategy::Parallel,
        shard_timeout_ms: 5000,
        max_retries: 1,
        allow_partial_results: true,
    };

    let coordinator = ShardCoordinator::new(Arc::new(manager), config);

    // Benchmark different hop counts
    for hops in [1, 2, 3, 4] {
        group.bench_with_input(
            BenchmarkId::from_parameter(format!("{}_hops", hops)),
            &hops,
            |b, &hops| {
                b.iter(|| {
                    black_box(simulate_shard_hop_latency(&coordinator, hops))
                });
            },
        );
    }

    group.finish();
}

fn bench_coordinator_overhead(c: &mut Criterion) {
    let mut group = c.benchmark_group("Coordinator Overhead");
    group.sample_size(100);

    let manager = Arc::new(ShardManager::new(ShardConfig::hash(4)));
    let coordinator = ShardCoordinator::with_defaults(manager);

    // Measure pure routing overhead (no actual execution)
    group.bench_function("route_single_node", |b| {
        b.iter(|| {
            let router = coordinator.router();
            black_box(router.plan_node_lookup(NodeId::new(12345)))
        });
    });

    group.bench_function("route_multi_node", |b| {
        let nodes: Vec<_> = (0..100).map(NodeId::new).collect();
        b.iter(|| {
            let router = coordinator.router();
            black_box(router.plan_multi_node_lookup(&nodes))
        });
    });

    group.bench_function("route_full_scan", |b| {
        b.iter(|| {
            let router = coordinator.router();
            black_box(router.plan_full_scan())
        });
    });

    group.finish();
}

criterion_group!(
    benches,
    bench_edge_cut_percentage,
    bench_cross_shard_latency,
    bench_coordinator_overhead
);
criterion_main!(benches);
