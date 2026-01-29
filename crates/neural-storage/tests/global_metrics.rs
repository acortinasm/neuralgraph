//! Global Metrics Test Suite
//!
//! Comprehensive performance metrics for NeuralGraphDB features.
//! Run after each sprint to track performance over time.
//!
//! Run with: cargo test --test global_metrics --release -- --nocapture
//!
//! Metrics tracked:
//! - Query latency (p50, p95, p99)
//! - Memory usage (heap, index size)
//! - Throughput (queries per second)
//! - Recall/accuracy for approximate algorithms
//! - Build/index time

use neural_core::{Edge, NodeId};
use neural_storage::sharding::{
    CommunityPartition, CoordinatorConfig, ExecutionStrategy, HashPartition, PartitionStrategy,
    RangePartition, ShardConfig, ShardCoordinator, ShardManager, ShardResult,
};
use neural_storage::vector_index::{
    QuantizationMethod, SimulatedDistributedIndex, VectorIndex, VectorIndexConfig,
};
use std::collections::HashSet;
use std::sync::Arc;
use std::time::{Duration, Instant};

// =============================================================================
// Configuration
// =============================================================================

/// Test configuration for different scale levels
#[derive(Clone, Copy)]
struct TestScale {
    name: &'static str,
    vector_count: usize,
    vector_dim: usize,
    graph_nodes: usize,
    graph_avg_degree: usize,
    num_queries: usize,
}

const QUICK_SCALE: TestScale = TestScale {
    name: "Quick (Dev)",
    vector_count: 5_000,
    vector_dim: 64,
    graph_nodes: 5_000,
    graph_avg_degree: 8,
    num_queries: 50,
};

const STANDARD_SCALE: TestScale = TestScale {
    name: "Standard",
    vector_count: 100_000,
    vector_dim: 256,
    graph_nodes: 50_000,
    graph_avg_degree: 10,
    num_queries: 500,
};

// Uncomment for large-scale testing
// const LARGE_SCALE: TestScale = TestScale {
//     name: "Large",
//     vector_count: 1_000_000,
//     vector_dim: 768,
//     graph_nodes: 1_000_000,
//     graph_avg_degree: 15,
//     num_queries: 1000,
// };

// =============================================================================
// Metrics Collection
// =============================================================================

#[derive(Debug, Clone)]
struct LatencyMetrics {
    p50: Duration,
    p95: Duration,
    p99: Duration,
    mean: Duration,
    min: Duration,
    max: Duration,
}

impl LatencyMetrics {
    fn from_samples(mut samples: Vec<Duration>) -> Self {
        samples.sort();
        let len = samples.len();
        let sum: Duration = samples.iter().sum();

        Self {
            p50: samples[len / 2],
            p95: samples[(len as f64 * 0.95) as usize],
            p99: samples[(len as f64 * 0.99) as usize],
            mean: sum / len as u32,
            min: samples[0],
            max: samples[len - 1],
        }
    }

    fn format(&self) -> String {
        format!(
            "p50={:>8.2}ms  p95={:>8.2}ms  p99={:>8.2}ms  mean={:>8.2}ms",
            self.p50.as_secs_f64() * 1000.0,
            self.p95.as_secs_f64() * 1000.0,
            self.p99.as_secs_f64() * 1000.0,
            self.mean.as_secs_f64() * 1000.0,
        )
    }
}

#[derive(Debug, Clone)]
struct MemoryMetrics {
    heap_bytes: usize,
    index_bytes: usize,
    metadata_bytes: usize,
}

impl MemoryMetrics {
    fn format(&self) -> String {
        format!(
            "heap={:>10}  index={:>10}  metadata={:>10}",
            format_bytes(self.heap_bytes),
            format_bytes(self.index_bytes),
            format_bytes(self.metadata_bytes),
        )
    }
}

fn format_bytes(bytes: usize) -> String {
    if bytes >= 1_000_000_000 {
        format!("{:.2} GB", bytes as f64 / 1_000_000_000.0)
    } else if bytes >= 1_000_000 {
        format!("{:.2} MB", bytes as f64 / 1_000_000.0)
    } else if bytes >= 1_000 {
        format!("{:.2} KB", bytes as f64 / 1_000.0)
    } else {
        format!("{} B", bytes)
    }
}

#[derive(Debug, Clone)]
struct MetricsReport {
    scale: &'static str,
    timestamp: String,
    sections: Vec<MetricsSection>,
}

#[derive(Debug, Clone)]
struct MetricsSection {
    name: String,
    metrics: Vec<(String, String)>,
}

impl MetricsReport {
    fn new(scale: &'static str) -> Self {
        Self {
            scale,
            timestamp: chrono::Utc::now().to_rfc3339(),
            sections: Vec::new(),
        }
    }

    fn add_section(&mut self, name: impl Into<String>) -> &mut MetricsSection {
        self.sections.push(MetricsSection {
            name: name.into(),
            metrics: Vec::new(),
        });
        self.sections.last_mut().unwrap()
    }

    fn print(&self) {
        println!("\n{}", "=".repeat(80));
        println!("NEURALGRAPHDB METRICS REPORT");
        println!("{}", "=".repeat(80));
        println!("Scale: {}", self.scale);
        println!("Timestamp: {}", self.timestamp);
        println!("{}", "=".repeat(80));

        for section in &self.sections {
            println!("\n## {}", section.name);
            println!("{}", "-".repeat(60));
            for (key, value) in &section.metrics {
                println!("  {:<30} {}", key, value);
            }
        }

        println!("\n{}", "=".repeat(80));
    }
}

impl MetricsSection {
    fn add(&mut self, key: impl Into<String>, value: impl Into<String>) -> &mut Self {
        self.metrics.push((key.into(), value.into()));
        self
    }
}

// =============================================================================
// Test Utilities
// =============================================================================

fn generate_vectors(count: usize, dim: usize, seed: u64) -> Vec<Vec<f32>> {
    (0..count)
        .map(|i| {
            let mut vec: Vec<f32> = (0..dim)
                .map(|d| {
                    let s = ((i as u64).wrapping_mul(31).wrapping_add(d as u64).wrapping_add(seed)) as f32;
                    (s * 0.618033988749895).fract() * 2.0 - 1.0
                })
                .collect();
            // Normalize
            let norm: f32 = vec.iter().map(|x| x * x).sum::<f32>().sqrt();
            if norm > 0.0 {
                for x in &mut vec {
                    *x /= norm;
                }
            }
            vec
        })
        .collect()
}

fn generate_scale_free_graph(num_nodes: usize, avg_degree: usize, seed: u64) -> Vec<Edge> {
    let mut edges = Vec::new();
    let m = avg_degree / 2;

    // Initial clique
    for i in 0..=m {
        for j in (i + 1)..=m {
            edges.push(Edge::new(i as u64, j as u64));
        }
    }

    // Preferential attachment
    let mut degrees = vec![m; m + 1];
    degrees.resize(num_nodes, 0);

    for new_node in (m + 1)..num_nodes {
        let total_degree: usize = degrees.iter().sum();
        let mut targets = Vec::new();
        let mut seed_val = seed.wrapping_add(new_node as u64);

        while targets.len() < m {
            seed_val = seed_val.wrapping_mul(6364136223846793005).wrapping_add(1);
            let r = (seed_val as f64) / (u64::MAX as f64);
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

// =============================================================================
// Vector Index Metrics
// =============================================================================

fn measure_vector_index_metrics(scale: &TestScale, report: &mut MetricsReport) {
    let section = report.add_section("Vector Index (HNSW)");

    // Generate test data
    let vectors = generate_vectors(scale.vector_count, scale.vector_dim, 42);
    let queries = generate_vectors(scale.num_queries, scale.vector_dim, 123);

    // Measure build time
    let build_start = Instant::now();
    let mut index = VectorIndex::with_config(VectorIndexConfig::large(scale.vector_dim));
    for (i, v) in vectors.iter().enumerate() {
        index.add(NodeId::new(i as u64), v);
    }
    let build_time = build_start.elapsed();

    section.add("Build time", format!("{:.2}s", build_time.as_secs_f64()));
    section.add("Vectors indexed", format!("{}", scale.vector_count));
    section.add("Dimension", format!("{}", scale.vector_dim));

    // Memory stats
    let (vec_bytes, meta_bytes, count) = index.memory_stats();
    section.add(
        "Memory (vectors)",
        format!("{} ({} vectors)", format_bytes(vec_bytes), count),
    );
    section.add("Memory (metadata)", format_bytes(meta_bytes));

    // Search latency for different k values
    for k in [10, 50, 100] {
        let mut latencies = Vec::with_capacity(scale.num_queries);
        for query in &queries {
            let start = Instant::now();
            let _ = index.search(query, k);
            latencies.push(start.elapsed());
        }
        let metrics = LatencyMetrics::from_samples(latencies);
        section.add(format!("Search k={}", k), metrics.format());
    }

    // Throughput measurement
    let throughput_start = Instant::now();
    let throughput_queries = 1000.min(scale.num_queries);
    for i in 0..throughput_queries {
        let _ = index.search(&queries[i % queries.len()], 10);
    }
    let throughput_time = throughput_start.elapsed();
    let qps = throughput_queries as f64 / throughput_time.as_secs_f64();
    section.add("Throughput (k=10)", format!("{:.0} qps", qps));
}

// =============================================================================
// Quantization Metrics
// =============================================================================

fn measure_quantization_metrics(scale: &TestScale, report: &mut MetricsReport) {
    let section = report.add_section("Flash Quantization");

    let vectors = generate_vectors(scale.vector_count / 10, scale.vector_dim, 42);

    // Unquantized
    let mut unquantized = VectorIndex::with_config(VectorIndexConfig::new(scale.vector_dim));
    for (i, v) in vectors.iter().enumerate() {
        unquantized.add(NodeId::new(i as u64), v);
    }
    let (unq_vec_bytes, _, _) = unquantized.memory_stats();

    // Int8 quantized
    let mut quantized = VectorIndex::with_config(
        VectorIndexConfig::new(scale.vector_dim).with_quantization(QuantizationMethod::Int8),
    );
    for (i, v) in vectors.iter().enumerate() {
        quantized.add(NodeId::new(i as u64), v);
    }
    let (q_vec_bytes, _, _) = quantized.memory_stats();

    let savings = 1.0 - (q_vec_bytes as f64 / unq_vec_bytes as f64);
    section.add("Unquantized memory", format_bytes(unq_vec_bytes));
    section.add("Int8 quantized memory", format_bytes(q_vec_bytes));
    section.add("Memory savings", format!("{:.1}%", savings * 100.0));
    section.add(
        "Compression ratio",
        format!("{:.1}x", unq_vec_bytes as f64 / q_vec_bytes as f64),
    );
}

// =============================================================================
// Distributed Vector Search Metrics
// =============================================================================

fn measure_distributed_vector_metrics(scale: &TestScale, report: &mut MetricsReport) {
    let section = report.add_section("Distributed Vector Search");

    let num_shards = 4;
    let vectors = generate_vectors(scale.vector_count / 2, scale.vector_dim, 42);
    let queries = generate_vectors(scale.num_queries, scale.vector_dim, 123);

    // Create distributed index
    let mut distributed = SimulatedDistributedIndex::new(num_shards, scale.vector_dim);
    for (i, v) in vectors.iter().enumerate() {
        distributed.add(NodeId::new(i as u64), v);
    }

    section.add("Shards", format!("{}", num_shards));
    section.add("Total vectors", format!("{}", distributed.len()));
    section.add(
        "Vectors per shard",
        format!("~{}", distributed.len() / num_shards),
    );

    // Search latency
    let mut latencies = Vec::with_capacity(scale.num_queries);
    for query in &queries {
        let start = Instant::now();
        let _ = distributed.search(query, 100);
        latencies.push(start.elapsed());
    }
    let metrics = LatencyMetrics::from_samples(latencies);
    section.add("Search latency (k=100)", metrics.format());

    // Cache performance
    let _ = distributed.search(&queries[0], 100); // Prime cache
    let _ = distributed.search(&queries[0], 100); // Cache hit
    let stats = distributed.cache_stats();
    section.add("Cache hit rate", format!("{:.1}%", stats.hit_rate()));

    // Compare with centralized for recall
    let mut centralized = VectorIndex::with_config(VectorIndexConfig::small(scale.vector_dim));
    for (i, v) in vectors.iter().enumerate() {
        centralized.add(NodeId::new(i as u64), v);
    }

    let k = 50;
    let mut total_recall = 0.0;
    let recall_queries = 10.min(queries.len());
    for query in queries.iter().take(recall_queries) {
        let central_results: HashSet<_> = centralized
            .search(query, k)
            .iter()
            .map(|(id, _)| *id)
            .collect();
        let dist_results: HashSet<_> = distributed
            .search(query, k)
            .iter()
            .map(|(id, _)| *id)
            .collect();
        let intersection = central_results.intersection(&dist_results).count();
        total_recall += intersection as f64 / k as f64;
    }
    let avg_recall = total_recall / recall_queries as f64;
    section.add(
        format!("Recall@{} vs centralized", k),
        format!("{:.1}%", avg_recall * 100.0),
    );
}

// =============================================================================
// Sharding Metrics
// =============================================================================

fn measure_sharding_metrics(scale: &TestScale, report: &mut MetricsReport) {
    let section = report.add_section("Graph Sharding");

    let edges = generate_scale_free_graph(scale.graph_nodes, scale.graph_avg_degree, 42);
    let num_shards = 4;

    section.add("Graph nodes", format!("{}", scale.graph_nodes));
    section.add("Graph edges", format!("{}", edges.len()));
    section.add("Avg degree", format!("{}", scale.graph_avg_degree));
    section.add("Shards", format!("{}", num_shards));

    // Edge cut for different strategies
    let hash_strategy = HashPartition::new(num_shards);
    let hash_cut = calculate_edge_cut(&edges, &hash_strategy);
    section.add("Edge cut (Hash)", format!("{:.1}%", hash_cut));

    let range_strategy = RangePartition::uniform(num_shards, scale.graph_nodes as u64);
    let range_cut = calculate_edge_cut(&edges, &range_strategy);
    section.add("Edge cut (Range)", format!("{:.1}%", range_cut));

    // Community detection is expensive, skip in automated tests
    // Uncomment to run manually: takes ~30s for 10k nodes
    if false && scale.graph_nodes >= 10_000 && scale.graph_nodes <= 20_000 {
        let community_strategy =
            CommunityPartition::from_edges(&edges, scale.graph_nodes, num_shards);
        let community_cut = community_strategy.edge_cut_percentage(&edges);
        section.add("Edge cut (Community)", format!("{:.1}%", community_cut));
    }

    // Routing overhead
    let manager = Arc::new(ShardManager::new(ShardConfig::hash(num_shards)));
    let coordinator = ShardCoordinator::with_defaults(manager);

    let mut routing_latencies = Vec::with_capacity(1000);
    for i in 0..1000 {
        let start = Instant::now();
        let router = coordinator.router();
        let _ = router.plan_node_lookup(NodeId::new(i));
        routing_latencies.push(start.elapsed());
    }
    let routing_metrics = LatencyMetrics::from_samples(routing_latencies);
    section.add("Routing overhead", routing_metrics.format());
}

fn calculate_edge_cut(edges: &[Edge], strategy: &dyn PartitionStrategy) -> f64 {
    let cross_shard = edges
        .iter()
        .filter(|e| strategy.shard_for_node(e.source) != strategy.shard_for_node(e.target))
        .count();
    (cross_shard as f64 / edges.len() as f64) * 100.0
}

// =============================================================================
// Coordinator Metrics
// =============================================================================

fn measure_coordinator_metrics(scale: &TestScale, report: &mut MetricsReport) {
    let section = report.add_section("Shard Coordinator");

    let num_shards = 4;
    let mut manager = ShardManager::new(ShardConfig::hash(num_shards));
    for i in 0..num_shards {
        manager.update_shard_address(i, format!("localhost:{}", 9000 + i));
    }
    manager.set_local_shard(0, "localhost:9000".to_string());

    let config = CoordinatorConfig {
        strategy: ExecutionStrategy::Parallel,
        shard_timeout_ms: 5000,
        max_retries: 1,
        allow_partial_results: true,
    };

    let coordinator = ShardCoordinator::new(Arc::new(manager), config);

    // Measure execution overhead (simulated)
    let mut exec_latencies = Vec::with_capacity(100);
    for _ in 0..100 {
        let router = coordinator.router();
        let plan = router.plan_full_scan();

        let start = Instant::now();
        let _ = coordinator.execute_plan(&plan, |shard_id, _info| {
            // Simulate minimal local work
            Ok(ShardResult::new(shard_id, vec![], 10))
        });
        exec_latencies.push(start.elapsed());
    }
    let exec_metrics = LatencyMetrics::from_samples(exec_latencies);
    section.add("Execution overhead (4 shards)", exec_metrics.format());

    // Vector search routing
    let mut vec_routing_latencies = Vec::with_capacity(1000);
    for _ in 0..1000 {
        let start = Instant::now();
        let router = coordinator.router();
        let _ = router.plan_vector_search();
        vec_routing_latencies.push(start.elapsed());
    }
    let vec_routing_metrics = LatencyMetrics::from_samples(vec_routing_latencies);
    section.add("Vector search routing", vec_routing_metrics.format());
}

// =============================================================================
// Main Test
// =============================================================================

#[test]
fn global_metrics_quick() {
    run_metrics_suite(&QUICK_SCALE);
}

#[test]
#[ignore] // Run with: cargo test --test global_metrics global_metrics_standard --release -- --ignored --nocapture
fn global_metrics_standard() {
    run_metrics_suite(&STANDARD_SCALE);
}

fn run_metrics_suite(scale: &TestScale) {
    let mut report = MetricsReport::new(scale.name);

    // Add scale info
    let scale_section = report.add_section("Test Configuration");
    scale_section.add("Vector count", format!("{}", scale.vector_count));
    scale_section.add("Vector dimension", format!("{}", scale.vector_dim));
    scale_section.add("Graph nodes", format!("{}", scale.graph_nodes));
    scale_section.add("Graph avg degree", format!("{}", scale.graph_avg_degree));
    scale_section.add("Query count", format!("{}", scale.num_queries));

    // Run all metric collections
    measure_vector_index_metrics(scale, &mut report);
    measure_quantization_metrics(scale, &mut report);
    measure_distributed_vector_metrics(scale, &mut report);
    measure_sharding_metrics(scale, &mut report);
    measure_coordinator_metrics(scale, &mut report);

    // Print report
    report.print();

    // Basic assertions to ensure metrics are reasonable
    assert!(report.sections.len() >= 5, "Should have at least 5 sections");
}

// =============================================================================
// Individual Feature Tests (for CI)
// =============================================================================

#[test]
fn metrics_vector_search_latency() {
    let scale = QUICK_SCALE;
    let vectors = generate_vectors(scale.vector_count, scale.vector_dim, 42);
    let queries = generate_vectors(100, scale.vector_dim, 123);

    let mut index = VectorIndex::with_config(VectorIndexConfig::large(scale.vector_dim));
    for (i, v) in vectors.iter().enumerate() {
        index.add(NodeId::new(i as u64), v);
    }

    let mut latencies = Vec::new();
    for query in &queries {
        let start = Instant::now();
        let _ = index.search(query, 10);
        latencies.push(start.elapsed());
    }

    let metrics = LatencyMetrics::from_samples(latencies);
    println!("Vector search k=10: {}", metrics.format());

    // Assert reasonable latency (should be under 10ms for 10k vectors)
    assert!(
        metrics.p95 < Duration::from_millis(50),
        "p95 latency too high: {:?}",
        metrics.p95
    );
}

#[test]
fn metrics_distributed_search_recall() {
    let dim = 64;
    let count = 5000;
    let k = 50;

    let vectors = generate_vectors(count, dim, 42);
    let queries = generate_vectors(10, dim, 123);

    let mut centralized = VectorIndex::with_config(VectorIndexConfig::small(dim));
    let mut distributed = SimulatedDistributedIndex::new(4, dim);

    for (i, v) in vectors.iter().enumerate() {
        centralized.add(NodeId::new(i as u64), v);
        distributed.add(NodeId::new(i as u64), v);
    }

    let mut total_recall = 0.0;
    for query in &queries {
        let central: HashSet<_> = centralized.search(query, k).iter().map(|(id, _)| *id).collect();
        let dist: HashSet<_> = distributed.search(query, k).iter().map(|(id, _)| *id).collect();
        let intersection = central.intersection(&dist).count();
        total_recall += intersection as f64 / k as f64;
    }

    let avg_recall = total_recall / queries.len() as f64;
    println!("Distributed search recall@{}: {:.1}%", k, avg_recall * 100.0);

    assert!(
        avg_recall >= 0.70,
        "Recall too low: {:.1}%",
        avg_recall * 100.0
    );
}

#[test]
fn metrics_quantization_memory_savings() {
    let dim = 256;
    let count = 1000;

    let vectors = generate_vectors(count, dim, 42);

    let mut unquantized = VectorIndex::with_config(VectorIndexConfig::new(dim));
    let mut quantized = VectorIndex::with_config(
        VectorIndexConfig::new(dim).with_quantization(QuantizationMethod::Int8),
    );

    for (i, v) in vectors.iter().enumerate() {
        unquantized.add(NodeId::new(i as u64), v);
        quantized.add(NodeId::new(i as u64), v);
    }

    let (unq_bytes, _, _) = unquantized.memory_stats();
    let (q_bytes, _, _) = quantized.memory_stats();
    let ratio = unq_bytes as f64 / q_bytes as f64;

    println!(
        "Quantization: {} -> {} ({:.1}x compression)",
        format_bytes(unq_bytes),
        format_bytes(q_bytes),
        ratio
    );

    // Int8 should provide at least 3x compression
    assert!(ratio >= 3.0, "Compression ratio too low: {:.1}x", ratio);
}

#[test]
fn metrics_sharding_edge_cut() {
    let nodes = 10_000;
    let edges = generate_scale_free_graph(nodes, 10, 42);
    let num_shards = 4;

    let hash_strategy = HashPartition::new(num_shards);
    let hash_cut = calculate_edge_cut(&edges, &hash_strategy);

    let range_strategy = RangePartition::uniform(num_shards, nodes as u64);
    let range_cut = calculate_edge_cut(&edges, &range_strategy);

    println!("Hash edge cut: {:.1}%", hash_cut);
    println!("Range edge cut: {:.1}%", range_cut);

    // Hash partitioning should have roughly (1 - 1/num_shards) edge cut
    // For 4 shards, expect ~75% edge cut
    assert!(
        hash_cut > 50.0 && hash_cut < 90.0,
        "Hash edge cut unexpected: {:.1}%",
        hash_cut
    );
}
