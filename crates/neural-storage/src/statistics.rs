//! Statistics collection for NeuralGraphDB.
//!
//! This module provides graph statistics collection for query planning
//! and monitoring.
//!
//! # Example
//!
//! ```ignore
//! use neural_storage::statistics::GraphStatistics;
//!
//! // Collect statistics from a graph store
//! let stats = store.collect_statistics();
//!
//! println!("Nodes: {}", stats.node_count);
//! println!("Edges: {}", stats.edge_count);
//! println!("Labels: {:?}", stats.label_cardinalities);
//! ```

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Statistics about a graph store.
///
/// These statistics are used for:
/// - Query planning and optimization
/// - Monitoring and observability
/// - Capacity planning
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct GraphStatistics {
    /// Total number of nodes
    pub node_count: usize,
    /// Total number of edges
    pub edge_count: usize,
    /// Number of nodes per label
    pub label_cardinalities: HashMap<String, usize>,
    /// Number of nodes with each property
    pub property_cardinalities: HashMap<String, usize>,
    /// Number of edges per edge type
    pub edge_type_cardinalities: HashMap<String, usize>,
    /// Average out-degree (edges per node)
    pub avg_out_degree: f64,
    /// Maximum out-degree
    pub max_out_degree: usize,
    /// Number of dynamic (non-CSR) nodes
    pub dynamic_node_count: usize,
    /// Number of dynamic (non-CSR) edges
    pub dynamic_edge_count: usize,
    /// Current transaction ID
    pub current_tx_id: u64,
    /// Number of recorded timestamps (for time-travel)
    pub timestamp_count: usize,
    /// Number of full-text indexes
    pub fulltext_index_count: usize,
    /// ISO 8601 timestamp of when statistics were collected
    pub last_updated: Option<String>,
}

impl GraphStatistics {
    /// Creates a new empty statistics struct.
    pub fn new() -> Self {
        Self::default()
    }

    /// Estimates the cardinality (number of results) for a label query.
    ///
    /// Returns the exact count if known, or a default estimate.
    pub fn estimate_label_cardinality(&self, label: &str) -> usize {
        self.label_cardinalities
            .get(label)
            .copied()
            .unwrap_or_else(|| {
                // Default estimate: 10% of nodes if label is unknown
                (self.node_count / 10).max(1)
            })
    }

    /// Estimates the cardinality for a property filter.
    ///
    /// This is a rough estimate based on property presence, not values.
    pub fn estimate_property_cardinality(&self, property: &str) -> usize {
        self.property_cardinalities
            .get(property)
            .copied()
            .unwrap_or_else(|| {
                // Default estimate: 50% of nodes have any given property
                self.node_count / 2
            })
    }

    /// Estimates the cardinality for an edge type traversal.
    pub fn estimate_edge_type_cardinality(&self, edge_type: &str) -> usize {
        self.edge_type_cardinalities
            .get(edge_type)
            .copied()
            .unwrap_or_else(|| {
                // Default: average edge count per type
                let type_count = self.edge_type_cardinalities.len().max(1);
                self.edge_count / type_count
            })
    }

    /// Estimates the selectivity of a label (0.0 - 1.0).
    ///
    /// Lower selectivity = fewer results = more selective.
    pub fn label_selectivity(&self, label: &str) -> f64 {
        if self.node_count == 0 {
            return 1.0;
        }
        let cardinality = self.estimate_label_cardinality(label);
        cardinality as f64 / self.node_count as f64
    }

    /// Returns a summary string for display.
    pub fn summary(&self) -> String {
        format!(
            "Nodes: {}, Edges: {}, Labels: {}, Edge Types: {}, Avg Degree: {:.2}",
            self.node_count,
            self.edge_count,
            self.label_cardinalities.len(),
            self.edge_type_cardinalities.len(),
            self.avg_out_degree
        )
    }
}

/// Builder for GraphStatistics.
///
/// Allows incremental construction of statistics.
#[derive(Debug, Default)]
pub struct StatisticsBuilder {
    stats: GraphStatistics,
}

impl StatisticsBuilder {
    /// Creates a new statistics builder.
    pub fn new() -> Self {
        Self::default()
    }

    /// Sets the node count.
    pub fn node_count(mut self, count: usize) -> Self {
        self.stats.node_count = count;
        self
    }

    /// Sets the edge count.
    pub fn edge_count(mut self, count: usize) -> Self {
        self.stats.edge_count = count;
        self
    }

    /// Adds a label cardinality.
    pub fn add_label(mut self, label: &str, count: usize) -> Self {
        self.stats.label_cardinalities.insert(label.to_string(), count);
        self
    }

    /// Adds a property cardinality.
    pub fn add_property(mut self, property: &str, count: usize) -> Self {
        self.stats.property_cardinalities.insert(property.to_string(), count);
        self
    }

    /// Adds an edge type cardinality.
    pub fn add_edge_type(mut self, edge_type: &str, count: usize) -> Self {
        self.stats.edge_type_cardinalities.insert(edge_type.to_string(), count);
        self
    }

    /// Sets the average out-degree.
    pub fn avg_out_degree(mut self, degree: f64) -> Self {
        self.stats.avg_out_degree = degree;
        self
    }

    /// Sets the maximum out-degree.
    pub fn max_out_degree(mut self, degree: usize) -> Self {
        self.stats.max_out_degree = degree;
        self
    }

    /// Sets the dynamic node count.
    pub fn dynamic_node_count(mut self, count: usize) -> Self {
        self.stats.dynamic_node_count = count;
        self
    }

    /// Sets the dynamic edge count.
    pub fn dynamic_edge_count(mut self, count: usize) -> Self {
        self.stats.dynamic_edge_count = count;
        self
    }

    /// Sets the current transaction ID.
    pub fn current_tx_id(mut self, tx_id: u64) -> Self {
        self.stats.current_tx_id = tx_id;
        self
    }

    /// Sets the timestamp count.
    pub fn timestamp_count(mut self, count: usize) -> Self {
        self.stats.timestamp_count = count;
        self
    }

    /// Sets the full-text index count.
    pub fn fulltext_index_count(mut self, count: usize) -> Self {
        self.stats.fulltext_index_count = count;
        self
    }

    /// Sets the last updated timestamp.
    pub fn last_updated(mut self, timestamp: &str) -> Self {
        self.stats.last_updated = Some(timestamp.to_string());
        self
    }

    /// Builds the statistics struct.
    pub fn build(self) -> GraphStatistics {
        self.stats
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_statistics() {
        let stats = GraphStatistics::default();
        assert_eq!(stats.node_count, 0);
        assert_eq!(stats.edge_count, 0);
        assert!(stats.label_cardinalities.is_empty());
    }

    #[test]
    fn test_statistics_builder() {
        let stats = StatisticsBuilder::new()
            .node_count(1000)
            .edge_count(5000)
            .add_label("Person", 500)
            .add_label("Company", 100)
            .add_edge_type("KNOWS", 3000)
            .avg_out_degree(5.0)
            .build();

        assert_eq!(stats.node_count, 1000);
        assert_eq!(stats.edge_count, 5000);
        assert_eq!(stats.label_cardinalities.get("Person"), Some(&500));
        assert_eq!(stats.edge_type_cardinalities.get("KNOWS"), Some(&3000));
        assert_eq!(stats.avg_out_degree, 5.0);
    }

    #[test]
    fn test_estimate_label_cardinality() {
        let stats = StatisticsBuilder::new()
            .node_count(1000)
            .add_label("Person", 500)
            .build();

        // Known label
        assert_eq!(stats.estimate_label_cardinality("Person"), 500);

        // Unknown label - defaults to 10% of nodes
        assert_eq!(stats.estimate_label_cardinality("Unknown"), 100);
    }

    #[test]
    fn test_label_selectivity() {
        let stats = StatisticsBuilder::new()
            .node_count(1000)
            .add_label("Rare", 10)
            .add_label("Common", 900)
            .build();

        // Rare label is more selective
        assert!(stats.label_selectivity("Rare") < stats.label_selectivity("Common"));
        assert!((stats.label_selectivity("Rare") - 0.01).abs() < 0.001);
        assert!((stats.label_selectivity("Common") - 0.9).abs() < 0.001);
    }

    #[test]
    fn test_summary() {
        let stats = StatisticsBuilder::new()
            .node_count(1000)
            .edge_count(5000)
            .add_label("Person", 500)
            .add_edge_type("KNOWS", 3000)
            .avg_out_degree(5.0)
            .build();

        let summary = stats.summary();
        assert!(summary.contains("1000"));
        assert!(summary.contains("5000"));
        assert!(summary.contains("5.00"));
    }

    #[test]
    fn test_statistics_serialization() {
        let stats = StatisticsBuilder::new()
            .node_count(100)
            .edge_count(200)
            .add_label("Test", 50)
            .last_updated("2026-01-15T12:00:00Z")
            .build();

        let serialized = bincode::serialize(&stats).unwrap();
        let deserialized: GraphStatistics = bincode::deserialize(&serialized).unwrap();

        assert_eq!(deserialized.node_count, 100);
        assert_eq!(deserialized.label_cardinalities.get("Test"), Some(&50));
        assert_eq!(deserialized.last_updated, Some("2026-01-15T12:00:00Z".to_string()));
    }
}
