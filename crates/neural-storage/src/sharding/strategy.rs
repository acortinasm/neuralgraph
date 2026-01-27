//! Partition strategies for graph sharding.
//!
//! Defines how nodes and edges are distributed across shards.

use neural_core::NodeId;
use serde::{Deserialize, Serialize};
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};

/// Unique identifier for a shard.
pub type ShardId = u32;

/// Trait for partition strategies that determine shard assignment.
///
/// Implementations must be deterministic - the same node ID must always
/// map to the same shard.
pub trait PartitionStrategy: Send + Sync {
    /// Returns the shard ID for a given node.
    fn shard_for_node(&self, node_id: NodeId) -> ShardId;

    /// Returns the shard ID for an edge (based on source node by default).
    ///
    /// This implements "vertex-cut" semantics where edges follow their source.
    /// Override for edge-cut semantics.
    fn shard_for_edge(&self, source: NodeId, _target: NodeId) -> ShardId {
        self.shard_for_node(source)
    }

    /// Returns the total number of shards.
    fn num_shards(&self) -> u32;

    /// Returns all shard IDs.
    fn all_shards(&self) -> Vec<ShardId> {
        (0..self.num_shards()).collect()
    }

    /// Returns shards that might contain nodes matching a label.
    ///
    /// Default: all shards (no label-based partitioning).
    /// Override for label-aware partitioning.
    fn shards_for_label(&self, _label: &str) -> Vec<ShardId> {
        self.all_shards()
    }

    /// Returns a description of the strategy for debugging.
    fn describe(&self) -> String;
}

// =============================================================================
// Hash-based Partitioning
// =============================================================================

/// Hash-based partition strategy using consistent hashing.
///
/// Distributes nodes evenly across shards using hash(node_id) % num_shards.
/// Good for uniform distribution but doesn't preserve locality.
///
/// # Example
///
/// ```
/// use neural_storage::sharding::{HashPartition, PartitionStrategy};
/// use neural_core::NodeId;
///
/// let partition = HashPartition::new(4);
/// let shard = partition.shard_for_node(NodeId::new(42));
/// assert!(shard < 4);
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HashPartition {
    num_shards: u32,
}

impl HashPartition {
    /// Creates a new hash partition with the specified number of shards.
    pub fn new(num_shards: u32) -> Self {
        assert!(num_shards > 0, "num_shards must be > 0");
        Self { num_shards }
    }

    fn hash_node(&self, node_id: NodeId) -> u64 {
        let mut hasher = DefaultHasher::new();
        node_id.as_u64().hash(&mut hasher);
        hasher.finish()
    }
}

impl PartitionStrategy for HashPartition {
    fn shard_for_node(&self, node_id: NodeId) -> ShardId {
        (self.hash_node(node_id) % self.num_shards as u64) as ShardId
    }

    fn num_shards(&self) -> u32 {
        self.num_shards
    }

    fn describe(&self) -> String {
        format!("HashPartition(shards={})", self.num_shards)
    }
}

// =============================================================================
// Range-based Partitioning
// =============================================================================

/// Range-based partition strategy for locality-preserving distribution.
///
/// Assigns contiguous ranges of node IDs to each shard.
/// Good for range queries and locality but may cause hotspots.
///
/// # Example
///
/// ```
/// use neural_storage::sharding::{RangePartition, PartitionStrategy};
/// use neural_core::NodeId;
///
/// // 4 shards with boundaries at 1000, 2000, 3000
/// let partition = RangePartition::uniform(4, 4000);
/// assert_eq!(partition.shard_for_node(NodeId::new(500)), 0);
/// assert_eq!(partition.shard_for_node(NodeId::new(1500)), 1);
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RangePartition {
    /// Shard boundaries: shard i owns nodes in [boundaries[i], boundaries[i+1])
    /// Last shard owns nodes >= boundaries[last]
    boundaries: Vec<u64>,
}

impl RangePartition {
    /// Creates a range partition with explicit boundaries.
    ///
    /// `boundaries` should be sorted and have length = num_shards - 1.
    /// Shard 0 owns [0, boundaries[0])
    /// Shard i owns [boundaries[i-1], boundaries[i])
    /// Last shard owns [boundaries[last], infinity)
    pub fn new(boundaries: Vec<u64>) -> Self {
        assert!(!boundaries.is_empty(), "boundaries must not be empty");
        // Verify sorted
        for i in 1..boundaries.len() {
            assert!(
                boundaries[i] > boundaries[i - 1],
                "boundaries must be strictly increasing"
            );
        }
        Self { boundaries }
    }

    /// Creates a uniform range partition dividing max_node_id evenly.
    pub fn uniform(num_shards: u32, max_node_id: u64) -> Self {
        assert!(num_shards > 0, "num_shards must be > 0");
        let range_size = max_node_id / num_shards as u64;
        let boundaries: Vec<u64> = (1..num_shards as u64)
            .map(|i| i * range_size)
            .collect();
        Self { boundaries }
    }

    fn find_shard(&self, node_id: u64) -> ShardId {
        // Binary search for the appropriate shard
        match self.boundaries.binary_search(&node_id) {
            Ok(pos) => (pos + 1) as ShardId, // Exact match: belongs to next shard
            Err(pos) => pos as ShardId,       // Insert position is the shard
        }
    }
}

impl PartitionStrategy for RangePartition {
    fn shard_for_node(&self, node_id: NodeId) -> ShardId {
        self.find_shard(node_id.as_u64())
    }

    fn num_shards(&self) -> u32 {
        (self.boundaries.len() + 1) as u32
    }

    fn describe(&self) -> String {
        format!(
            "RangePartition(shards={}, boundaries={:?})",
            self.num_shards(),
            self.boundaries
        )
    }
}

// =============================================================================
// Community-based Partitioning (Graph-Aware)
// =============================================================================

/// Community-based partition strategy for minimizing edge cuts.
///
/// Uses the Leiden algorithm to detect communities and assigns entire
/// communities to shards. This keeps connected nodes together, minimizing
/// the number of edges that cross shard boundaries.
///
/// # Example
///
/// ```ignore
/// use neural_storage::sharding::{CommunityPartition, PartitionStrategy};
/// use neural_core::{NodeId, Edge};
///
/// let edges = vec![
///     Edge::new(0, 1), Edge::new(1, 2), // community 1
///     Edge::new(3, 4), Edge::new(4, 5), // community 2
/// ];
/// let partition = CommunityPartition::from_edges(&edges, 6, 4);
/// // Nodes in same community will be on same shard
/// ```
#[derive(Debug, Clone)]
pub struct CommunityPartition {
    /// Number of shards
    num_shards: u32,
    /// Node ID -> Shard ID assignment
    node_to_shard: Vec<ShardId>,
}

impl CommunityPartition {
    /// Creates a community partition from edges.
    ///
    /// Uses Leiden algorithm to detect communities, then assigns communities
    /// to shards using greedy bin-packing to balance load.
    pub fn from_edges(edges: &[neural_core::Edge], num_nodes: usize, num_shards: u32) -> Self {
        use crate::community::detect_communities_leiden;

        // Convert Edge to (usize, usize) format for Leiden
        let edge_pairs: Vec<(usize, usize)> = edges
            .iter()
            .map(|e| (e.source.as_usize(), e.target.as_usize()))
            .collect();

        // Detect communities
        let communities = detect_communities_leiden(&edge_pairs, num_nodes);

        // Assign communities to shards using greedy bin-packing
        let node_to_shard = Self::assign_communities_to_shards(
            communities.assignments(),
            communities.num_communities(),
            num_shards,
        );

        Self {
            num_shards,
            node_to_shard,
        }
    }

    /// Creates a partition with pre-computed assignments.
    pub fn with_assignments(assignments: Vec<ShardId>, num_shards: u32) -> Self {
        Self {
            num_shards,
            node_to_shard: assignments,
        }
    }

    /// Assigns communities to shards using greedy bin-packing.
    ///
    /// Sorts communities by size (descending) and assigns each to the
    /// shard with the smallest current load.
    fn assign_communities_to_shards(
        node_communities: &[usize],
        num_communities: usize,
        num_shards: u32,
    ) -> Vec<ShardId> {
        if node_communities.is_empty() {
            return vec![];
        }

        // Count nodes per community
        let mut community_sizes: Vec<(usize, usize)> = (0..num_communities)
            .map(|c| {
                let count = node_communities.iter().filter(|&&x| x == c).count();
                (c, count)
            })
            .collect();

        // Sort by size descending (largest first for better bin-packing)
        community_sizes.sort_by(|a, b| b.1.cmp(&a.1));

        // Track shard loads
        let mut shard_loads: Vec<usize> = vec![0; num_shards as usize];

        // Assign communities to shards
        let mut community_to_shard: Vec<ShardId> = vec![0; num_communities];
        for (community_id, size) in community_sizes {
            // Find shard with minimum load
            let min_shard = shard_loads
                .iter()
                .enumerate()
                .min_by_key(|&(_, load)| *load)
                .map(|(idx, _)| idx)
                .unwrap_or(0);

            community_to_shard[community_id] = min_shard as ShardId;
            shard_loads[min_shard] += size;
        }

        // Map nodes to shards via their communities
        node_communities
            .iter()
            .map(|&community| community_to_shard[community])
            .collect()
    }

    /// Returns the percentage of edges that cross shard boundaries.
    pub fn edge_cut_percentage(&self, edges: &[neural_core::Edge]) -> f64 {
        if edges.is_empty() {
            return 0.0;
        }

        let cross_shard = edges
            .iter()
            .filter(|e| {
                let s1 = self.shard_for_node(e.source);
                let s2 = self.shard_for_node(e.target);
                s1 != s2
            })
            .count();

        (cross_shard as f64 / edges.len() as f64) * 100.0
    }
}

impl PartitionStrategy for CommunityPartition {
    fn shard_for_node(&self, node_id: NodeId) -> ShardId {
        self.node_to_shard
            .get(node_id.as_usize())
            .copied()
            .unwrap_or(0)
    }

    fn num_shards(&self) -> u32 {
        self.num_shards
    }

    fn describe(&self) -> String {
        format!(
            "CommunityPartition(shards={}, nodes={})",
            self.num_shards,
            self.node_to_shard.len()
        )
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hash_partition_basic() {
        let partition = HashPartition::new(4);

        // Should distribute to shards 0-3
        for i in 0..100 {
            let shard = partition.shard_for_node(NodeId::new(i));
            assert!(shard < 4, "shard {} out of range", shard);
        }
    }

    #[test]
    fn test_hash_partition_deterministic() {
        let partition = HashPartition::new(4);

        // Same node should always get same shard
        let shard1 = partition.shard_for_node(NodeId::new(42));
        let shard2 = partition.shard_for_node(NodeId::new(42));
        assert_eq!(shard1, shard2);
    }

    #[test]
    fn test_hash_partition_distribution() {
        let partition = HashPartition::new(4);

        // Count distribution across 1000 nodes
        let mut counts = [0u32; 4];
        for i in 0..1000 {
            let shard = partition.shard_for_node(NodeId::new(i));
            counts[shard as usize] += 1;
        }

        // Each shard should have roughly 250 nodes (allow 20% variance)
        for count in counts {
            assert!(count > 150 && count < 350, "uneven distribution: {:?}", counts);
        }
    }

    #[test]
    fn test_range_partition_uniform() {
        let partition = RangePartition::uniform(4, 4000);

        // Check boundary behavior
        assert_eq!(partition.shard_for_node(NodeId::new(0)), 0);
        assert_eq!(partition.shard_for_node(NodeId::new(999)), 0);
        assert_eq!(partition.shard_for_node(NodeId::new(1000)), 1);
        assert_eq!(partition.shard_for_node(NodeId::new(1999)), 1);
        assert_eq!(partition.shard_for_node(NodeId::new(2000)), 2);
        assert_eq!(partition.shard_for_node(NodeId::new(3000)), 3);
        assert_eq!(partition.shard_for_node(NodeId::new(10000)), 3); // Beyond max
    }

    #[test]
    fn test_range_partition_custom() {
        // Custom boundaries: [100, 500, 1000]
        // Shard 0: [0, 100), Shard 1: [100, 500), Shard 2: [500, 1000), Shard 3: [1000, ∞)
        let partition = RangePartition::new(vec![100, 500, 1000]);

        assert_eq!(partition.shard_for_node(NodeId::new(50)), 0);
        assert_eq!(partition.shard_for_node(NodeId::new(100)), 1);
        assert_eq!(partition.shard_for_node(NodeId::new(499)), 1);
        assert_eq!(partition.shard_for_node(NodeId::new(500)), 2);
        assert_eq!(partition.shard_for_node(NodeId::new(1000)), 3);
        assert_eq!(partition.shard_for_node(NodeId::new(5000)), 3);
    }

    #[test]
    fn test_edge_sharding_follows_source() {
        let partition = HashPartition::new(4);

        let source = NodeId::new(100);
        let target = NodeId::new(200);

        // Edge should be on same shard as source
        let edge_shard = partition.shard_for_edge(source, target);
        let source_shard = partition.shard_for_node(source);
        assert_eq!(edge_shard, source_shard);
    }

    #[test]
    fn test_all_shards() {
        let partition = HashPartition::new(3);
        assert_eq!(partition.all_shards(), vec![0, 1, 2]);
    }

    #[test]
    fn test_community_partition_basic() {
        use neural_core::Edge;

        // Two clear communities: {0,1,2} and {3,4,5}
        // Connected internally, with one bridge edge
        let edges = vec![
            // Community 1
            Edge::new(0u64, 1u64), Edge::new(1u64, 0u64),
            Edge::new(1u64, 2u64), Edge::new(2u64, 1u64),
            Edge::new(0u64, 2u64), Edge::new(2u64, 0u64),
            // Community 2
            Edge::new(3u64, 4u64), Edge::new(4u64, 3u64),
            Edge::new(4u64, 5u64), Edge::new(5u64, 4u64),
            Edge::new(3u64, 5u64), Edge::new(5u64, 3u64),
            // Bridge
            Edge::new(2u64, 3u64), Edge::new(3u64, 2u64),
        ];

        let partition = CommunityPartition::from_edges(&edges, 6, 2);

        // Should have 2 shards
        assert_eq!(partition.num_shards(), 2);

        // Nodes in same community should be on same shard
        let shard_0 = partition.shard_for_node(NodeId::new(0));
        let shard_1 = partition.shard_for_node(NodeId::new(1));
        let shard_2 = partition.shard_for_node(NodeId::new(2));
        assert_eq!(shard_0, shard_1);
        assert_eq!(shard_1, shard_2);

        let shard_3 = partition.shard_for_node(NodeId::new(3));
        let shard_4 = partition.shard_for_node(NodeId::new(4));
        let shard_5 = partition.shard_for_node(NodeId::new(5));
        assert_eq!(shard_3, shard_4);
        assert_eq!(shard_4, shard_5);

        // Edge cut should be low (only the bridge edge)
        let cut = partition.edge_cut_percentage(&edges);
        assert!(cut < 20.0, "Edge cut too high: {}%", cut);
    }

    #[test]
    fn test_community_partition_edge_cut() {
        use neural_core::Edge;

        // 4 dense cliques (complete subgraphs) with sparse bridges
        // Cliques have clear community structure that Leiden can detect
        let mut edges = Vec::new();

        // Helper to create a complete clique
        fn add_clique(edges: &mut Vec<Edge>, start: u64, size: u64) {
            for i in start..(start + size) {
                for j in (i + 1)..(start + size) {
                    edges.push(Edge::new(i, j));
                    edges.push(Edge::new(j, i)); // bidirectional
                }
            }
        }

        // 4 cliques of 5 nodes each (nodes 0-4, 5-9, 10-14, 15-19)
        add_clique(&mut edges, 0, 5);   // Community 0
        add_clique(&mut edges, 5, 5);   // Community 1
        add_clique(&mut edges, 10, 5);  // Community 2
        add_clique(&mut edges, 15, 5);  // Community 3

        // Sparse bridges (1 edge between each adjacent community)
        edges.push(Edge::new(4u64, 5u64));  // Community 0 -> 1
        edges.push(Edge::new(9u64, 10u64)); // Community 1 -> 2
        edges.push(Edge::new(14u64, 15u64)); // Community 2 -> 3

        let partition = CommunityPartition::from_edges(&edges, 20, 4);
        let cut = partition.edge_cut_percentage(&edges);

        // With 4 well-separated cliques and 4 shards,
        // edge cut should be low (3 bridge edges out of 43 total)
        // Expected: 3/43 = ~7%
        assert!(cut < 15.0, "Edge cut should be low with clear cliques: {}%", cut);
    }
}
