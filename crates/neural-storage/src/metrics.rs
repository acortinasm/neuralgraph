//! Prometheus metrics export for NeuralGraphDB.
//!
//! Provides metrics collection and export for monitoring query performance,
//! cache efficiency, and system health.
//!
//! # Feature Flag
//!
//! Metrics are only available when the `metrics` feature is enabled:
//!
//! ```toml
//! [dependencies]
//! neural-storage = { version = "*", features = ["metrics"] }
//! ```
//!
//! # Metrics Exported
//!
//! | Metric | Type | Description |
//! |--------|------|-------------|
//! | `neuralgraph_query_latency_seconds` | Histogram | Query latency distribution |
//! | `neuralgraph_query_total` | Counter | Total number of queries |
//! | `neuralgraph_cache_hits_total` | Counter | Cache hit count |
//! | `neuralgraph_cache_misses_total` | Counter | Cache miss count |
//! | `neuralgraph_cache_size` | Gauge | Current cache size |
//! | `neuralgraph_active_connections` | Gauge | Active gRPC connections |
//! | `neuralgraph_vectors_total` | Gauge | Total indexed vectors |
//!
//! # Example
//!
//! ```ignore
//! use neural_storage::metrics::MetricsRegistry;
//! use std::time::Duration;
//!
//! let metrics = MetricsRegistry::new().unwrap();
//!
//! // Record query latency
//! metrics.record_query_latency(Duration::from_millis(5));
//!
//! // Record cache hit/miss
//! metrics.record_cache_hit();
//!
//! // Export metrics in Prometheus format
//! let output = metrics.export().unwrap();
//! println!("{}", output);
//! ```

use std::time::Duration;

#[cfg(feature = "metrics")]
use prometheus::{
    Encoder, Histogram, HistogramOpts, IntCounter, IntCounterVec, IntGauge, Opts, Registry, TextEncoder,
};

/// Error type for metrics operations.
#[derive(Debug)]
pub struct MetricsError(String);

impl std::fmt::Display for MetricsError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "MetricsError: {}", self.0)
    }
}

impl std::error::Error for MetricsError {}

#[cfg(feature = "metrics")]
impl From<prometheus::Error> for MetricsError {
    fn from(e: prometheus::Error) -> Self {
        MetricsError(e.to_string())
    }
}

/// Metrics registry for NeuralGraphDB.
///
/// Collects and exports Prometheus-compatible metrics for monitoring
/// query performance, cache efficiency, and system health.
#[cfg(feature = "metrics")]
pub struct MetricsRegistry {
    /// Prometheus registry.
    registry: Registry,
    /// Query latency histogram.
    query_latency: Histogram,
    /// Total query count.
    query_count: IntCounter,
    /// Cache hit counter.
    cache_hits: IntCounter,
    /// Cache miss counter.
    cache_misses: IntCounter,
    /// Current cache size.
    cache_size: IntGauge,
    /// Active connections gauge.
    active_connections: IntGauge,
    /// Total vector count.
    vector_count: IntGauge,

    // =========================================================================
    // Raft Cluster Metrics (Sprint 53)
    // =========================================================================
    /// Current Raft term.
    raft_term: IntGauge,
    /// Total leader elections/changes.
    raft_leader_changes: IntCounter,
    /// Current committed log index.
    raft_log_index: IntGauge,
    /// Raft commit latency histogram.
    raft_commit_latency: Histogram,
    /// Total nodes in the cluster.
    cluster_node_count: IntGauge,
    /// Healthy nodes in the cluster.
    cluster_healthy_nodes: IntGauge,
    /// Client requests by status (success/failure).
    raft_client_requests: IntCounterVec,

    // =========================================================================
    // Graph Statistics Gauges (Sprint 67)
    // =========================================================================
    /// Total nodes in the graph.
    node_count: IntGauge,
    /// Total edges in the graph.
    edge_count: IntGauge,
}

#[cfg(feature = "metrics")]
impl MetricsRegistry {
    /// Creates a new metrics registry with all metrics registered.
    pub fn new() -> Result<Self, MetricsError> {
        let registry = Registry::new();

        // Query latency histogram with buckets for microseconds to seconds
        let query_latency = Histogram::with_opts(
            HistogramOpts::new(
                "neuralgraph_query_latency_seconds",
                "Query latency in seconds",
            )
            .buckets(vec![
                0.0001, // 100μs
                0.0005, // 500μs
                0.001,  // 1ms
                0.005,  // 5ms
                0.01,   // 10ms
                0.025,  // 25ms
                0.05,   // 50ms
                0.1,    // 100ms
                0.25,   // 250ms
                0.5,    // 500ms
                1.0,    // 1s
            ]),
        )?;

        let query_count = IntCounter::new(
            "neuralgraph_query_total",
            "Total number of queries executed",
        )?;

        let cache_hits = IntCounter::new(
            "neuralgraph_cache_hits_total",
            "Total number of cache hits",
        )?;

        let cache_misses = IntCounter::new(
            "neuralgraph_cache_misses_total",
            "Total number of cache misses",
        )?;

        let cache_size = IntGauge::new(
            "neuralgraph_cache_size",
            "Current number of entries in the cache",
        )?;

        let active_connections = IntGauge::new(
            "neuralgraph_active_connections",
            "Number of active gRPC connections",
        )?;

        let vector_count = IntGauge::new(
            "neuralgraph_vectors_total",
            "Total number of indexed vectors",
        )?;

        // Raft cluster metrics
        let raft_term = IntGauge::new(
            "neuralgraph_raft_term",
            "Current Raft term",
        )?;

        let raft_leader_changes = IntCounter::new(
            "neuralgraph_raft_leader_changes_total",
            "Total number of leader elections/changes",
        )?;

        let raft_log_index = IntGauge::new(
            "neuralgraph_raft_log_index",
            "Current committed log index",
        )?;

        let raft_commit_latency = Histogram::with_opts(
            HistogramOpts::new(
                "neuralgraph_raft_commit_latency_seconds",
                "Raft commit latency in seconds",
            )
            .buckets(vec![
                0.001,  // 1ms
                0.005,  // 5ms
                0.01,   // 10ms
                0.025,  // 25ms
                0.05,   // 50ms
                0.1,    // 100ms
                0.25,   // 250ms
                0.5,    // 500ms
                1.0,    // 1s
                2.5,    // 2.5s
            ]),
        )?;

        let cluster_node_count = IntGauge::new(
            "neuralgraph_cluster_node_count",
            "Total number of nodes in the cluster",
        )?;

        let cluster_healthy_nodes = IntGauge::new(
            "neuralgraph_cluster_healthy_nodes",
            "Number of healthy nodes in the cluster",
        )?;

        let raft_client_requests = IntCounterVec::new(
            Opts::new(
                "neuralgraph_raft_client_requests_total",
                "Total Raft client requests by status",
            ),
            &["status"], // "success" or "failure"
        )?;

        // Graph statistics gauges
        let node_count = IntGauge::new(
            "neuralgraph_node_count",
            "Total number of nodes in the graph",
        )?;

        let edge_count = IntGauge::new(
            "neuralgraph_edge_count",
            "Total number of edges in the graph",
        )?;

        // Register all metrics
        registry.register(Box::new(query_latency.clone()))?;
        registry.register(Box::new(query_count.clone()))?;
        registry.register(Box::new(cache_hits.clone()))?;
        registry.register(Box::new(cache_misses.clone()))?;
        registry.register(Box::new(cache_size.clone()))?;
        registry.register(Box::new(active_connections.clone()))?;
        registry.register(Box::new(vector_count.clone()))?;
        registry.register(Box::new(raft_term.clone()))?;
        registry.register(Box::new(raft_leader_changes.clone()))?;
        registry.register(Box::new(raft_log_index.clone()))?;
        registry.register(Box::new(raft_commit_latency.clone()))?;
        registry.register(Box::new(cluster_node_count.clone()))?;
        registry.register(Box::new(cluster_healthy_nodes.clone()))?;
        registry.register(Box::new(raft_client_requests.clone()))?;
        registry.register(Box::new(node_count.clone()))?;
        registry.register(Box::new(edge_count.clone()))?;

        Ok(Self {
            registry,
            query_latency,
            query_count,
            cache_hits,
            cache_misses,
            cache_size,
            active_connections,
            vector_count,
            raft_term,
            raft_leader_changes,
            raft_log_index,
            raft_commit_latency,
            cluster_node_count,
            cluster_healthy_nodes,
            raft_client_requests,
            node_count,
            edge_count,
        })
    }

    /// Records the latency of a query.
    pub fn record_query_latency(&self, duration: Duration) {
        self.query_latency.observe(duration.as_secs_f64());
        self.query_count.inc();
    }

    /// Records a cache hit.
    pub fn record_cache_hit(&self) {
        self.cache_hits.inc();
    }

    /// Records a cache miss.
    pub fn record_cache_miss(&self) {
        self.cache_misses.inc();
    }

    /// Sets the current cache size.
    pub fn set_cache_size(&self, size: i64) {
        self.cache_size.set(size);
    }

    /// Sets the number of active connections.
    pub fn set_active_connections(&self, count: i64) {
        self.active_connections.set(count);
    }

    /// Sets the total vector count.
    pub fn set_vector_count(&self, count: i64) {
        self.vector_count.set(count);
    }

    /// Increments the active connection count.
    pub fn inc_active_connections(&self) {
        self.active_connections.inc();
    }

    /// Decrements the active connection count.
    pub fn dec_active_connections(&self) {
        self.active_connections.dec();
    }

    /// Exports all metrics in Prometheus text format.
    pub fn export(&self) -> Result<String, MetricsError> {
        let encoder = TextEncoder::new();
        let metric_families = self.registry.gather();
        let mut buffer = Vec::new();
        encoder.encode(&metric_families, &mut buffer)?;
        Ok(String::from_utf8(buffer).unwrap_or_default())
    }

    /// Returns the cache hit rate (0.0 to 1.0).
    pub fn cache_hit_rate(&self) -> f64 {
        let hits = self.cache_hits.get() as f64;
        let misses = self.cache_misses.get() as f64;
        let total = hits + misses;
        if total > 0.0 {
            hits / total
        } else {
            0.0
        }
    }

    // =========================================================================
    // Raft Cluster Metrics (Sprint 53)
    // =========================================================================

    /// Sets the current Raft term.
    pub fn set_raft_term(&self, term: u64) {
        self.raft_term.set(term as i64);
    }

    /// Increments the leader change counter.
    pub fn record_leader_change(&self) {
        self.raft_leader_changes.inc();
    }

    /// Sets the current committed log index.
    pub fn set_raft_log_index(&self, index: u64) {
        self.raft_log_index.set(index as i64);
    }

    /// Records Raft commit latency.
    pub fn record_raft_commit_latency(&self, duration: Duration) {
        self.raft_commit_latency.observe(duration.as_secs_f64());
    }

    /// Sets the total number of nodes in the cluster.
    pub fn set_cluster_node_count(&self, count: usize) {
        self.cluster_node_count.set(count as i64);
    }

    /// Sets the number of healthy nodes in the cluster.
    pub fn set_cluster_healthy_nodes(&self, count: usize) {
        self.cluster_healthy_nodes.set(count as i64);
    }

    /// Records a successful client request.
    pub fn record_client_request_success(&self) {
        self.raft_client_requests.with_label_values(&["success"]).inc();
    }

    /// Records a failed client request.
    pub fn record_client_request_failure(&self) {
        self.raft_client_requests.with_label_values(&["failure"]).inc();
    }

    // =========================================================================
    // Graph Statistics Gauges (Sprint 67)
    // =========================================================================

    /// Sets the total node count.
    pub fn set_node_count(&self, count: usize) {
        self.node_count.set(count as i64);
    }

    /// Sets the total edge count.
    pub fn set_edge_count(&self, count: usize) {
        self.edge_count.set(count as i64);
    }
}

#[cfg(feature = "metrics")]
impl std::fmt::Debug for MetricsRegistry {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MetricsRegistry")
            .field("query_count", &self.query_count.get())
            .field("cache_hits", &self.cache_hits.get())
            .field("cache_misses", &self.cache_misses.get())
            .field("cache_size", &self.cache_size.get())
            .field("vector_count", &self.vector_count.get())
            .field("raft_term", &self.raft_term.get())
            .field("raft_log_index", &self.raft_log_index.get())
            .field("cluster_node_count", &self.cluster_node_count.get())
            .field("cluster_healthy_nodes", &self.cluster_healthy_nodes.get())
            .finish()
    }
}

// =============================================================================
// No-op Implementation (when metrics feature is disabled)
// =============================================================================

/// No-op metrics registry when the `metrics` feature is disabled.
///
/// All methods are no-ops, allowing code to call metrics methods without
/// conditional compilation in application code.
#[cfg(not(feature = "metrics"))]
#[derive(Debug, Clone, Default)]
pub struct MetricsRegistry;

#[cfg(not(feature = "metrics"))]
impl MetricsRegistry {
    /// Creates a new no-op metrics registry.
    pub fn new() -> Result<Self, MetricsError> {
        Ok(Self)
    }

    /// No-op: Records query latency.
    pub fn record_query_latency(&self, _duration: Duration) {}

    /// No-op: Records a cache hit.
    pub fn record_cache_hit(&self) {}

    /// No-op: Records a cache miss.
    pub fn record_cache_miss(&self) {}

    /// No-op: Sets the cache size.
    pub fn set_cache_size(&self, _size: i64) {}

    /// No-op: Sets active connections.
    pub fn set_active_connections(&self, _count: i64) {}

    /// No-op: Sets vector count.
    pub fn set_vector_count(&self, _count: i64) {}

    /// No-op: Increments active connections.
    pub fn inc_active_connections(&self) {}

    /// No-op: Decrements active connections.
    pub fn dec_active_connections(&self) {}

    /// Returns an empty string (no metrics to export).
    pub fn export(&self) -> Result<String, MetricsError> {
        Ok(String::new())
    }

    /// Returns 0.0 (no data).
    pub fn cache_hit_rate(&self) -> f64 {
        0.0
    }

    // Raft cluster metrics (no-op)
    /// No-op: Sets the Raft term.
    pub fn set_raft_term(&self, _term: u64) {}
    /// No-op: Records a leader change.
    pub fn record_leader_change(&self) {}
    /// No-op: Sets the Raft log index.
    pub fn set_raft_log_index(&self, _index: u64) {}
    /// No-op: Records Raft commit latency.
    pub fn record_raft_commit_latency(&self, _duration: Duration) {}
    /// No-op: Sets the cluster node count.
    pub fn set_cluster_node_count(&self, _count: usize) {}
    /// No-op: Sets the healthy node count.
    pub fn set_cluster_healthy_nodes(&self, _count: usize) {}
    /// No-op: Records a successful client request.
    pub fn record_client_request_success(&self) {}
    /// No-op: Records a failed client request.
    pub fn record_client_request_failure(&self) {}
    /// No-op: Sets the node count.
    pub fn set_node_count(&self, _count: usize) {}
    /// No-op: Sets the edge count.
    pub fn set_edge_count(&self, _count: usize) {}
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_metrics_registry_creation() {
        let metrics = MetricsRegistry::new().unwrap();
        // Should not panic
        metrics.record_query_latency(Duration::from_millis(5));
        metrics.record_cache_hit();
        metrics.record_cache_miss();
        metrics.set_cache_size(100);
        metrics.set_vector_count(1000);
    }

    #[cfg(feature = "metrics")]
    #[test]
    fn test_metrics_export() {
        let metrics = MetricsRegistry::new().unwrap();

        // Record some metrics
        metrics.record_query_latency(Duration::from_millis(5));
        metrics.record_query_latency(Duration::from_millis(10));
        metrics.record_cache_hit();
        metrics.record_cache_miss();
        metrics.set_cache_size(42);
        metrics.set_vector_count(1000);

        let output = metrics.export().unwrap();

        // Verify key metrics are present
        assert!(output.contains("neuralgraph_query_total"));
        assert!(output.contains("neuralgraph_cache_hits_total"));
        assert!(output.contains("neuralgraph_cache_misses_total"));
        assert!(output.contains("neuralgraph_cache_size"));
        assert!(output.contains("neuralgraph_vectors_total"));
    }

    #[cfg(feature = "metrics")]
    #[test]
    fn test_cache_hit_rate() {
        let metrics = MetricsRegistry::new().unwrap();

        // No data yet
        assert_eq!(metrics.cache_hit_rate(), 0.0);

        // 3 hits, 1 miss = 75% hit rate
        metrics.record_cache_hit();
        metrics.record_cache_hit();
        metrics.record_cache_hit();
        metrics.record_cache_miss();

        assert!((metrics.cache_hit_rate() - 0.75).abs() < 0.01);
    }

    #[cfg(feature = "metrics")]
    #[test]
    fn test_metrics_debug() {
        let metrics = MetricsRegistry::new().unwrap();
        metrics.record_query_latency(Duration::from_millis(1));
        metrics.record_cache_hit();

        let debug_str = format!("{:?}", metrics);
        assert!(debug_str.contains("MetricsRegistry"));
        assert!(debug_str.contains("query_count"));
    }
}
