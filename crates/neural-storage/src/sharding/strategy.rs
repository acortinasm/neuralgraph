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
}
