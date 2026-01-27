//! Shard manager for tracking shard metadata and assignments.

use super::strategy::{PartitionStrategy, ShardId, HashPartition};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;

/// Configuration for a sharded cluster.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ShardConfig {
    /// Total number of shards.
    pub num_shards: u32,
    /// Partition strategy type.
    pub strategy: StrategyType,
    /// Replication factor (number of replicas per shard).
    pub replication_factor: u32,
}

/// Supported partition strategy types.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum StrategyType {
    /// Hash-based partitioning.
    Hash,
    /// Range-based partitioning with uniform distribution.
    RangeUniform { max_node_id: u64 },
    /// Range-based partitioning with custom boundaries.
    RangeCustom { boundaries: Vec<u64> },
}

impl Default for ShardConfig {
    fn default() -> Self {
        Self {
            num_shards: 1,
            strategy: StrategyType::Hash,
            replication_factor: 1,
        }
    }
}

impl ShardConfig {
    /// Creates a new shard config with hash partitioning.
    pub fn hash(num_shards: u32) -> Self {
        Self {
            num_shards,
            strategy: StrategyType::Hash,
            replication_factor: 1,
        }
    }

    /// Creates a config with range partitioning.
    pub fn range_uniform(num_shards: u32, max_node_id: u64) -> Self {
        Self {
            num_shards,
            strategy: StrategyType::RangeUniform { max_node_id },
            replication_factor: 1,
        }
    }

    /// Sets the replication factor.
    pub fn with_replication(mut self, factor: u32) -> Self {
        self.replication_factor = factor;
        self
    }

    /// Builds the partition strategy from the config.
    pub fn build_strategy(&self) -> Arc<dyn PartitionStrategy> {
        match &self.strategy {
            StrategyType::Hash => Arc::new(HashPartition::new(self.num_shards)),
            StrategyType::RangeUniform { max_node_id } => {
                Arc::new(super::strategy::RangePartition::uniform(
                    self.num_shards,
                    *max_node_id,
                ))
            }
            StrategyType::RangeCustom { boundaries } => {
                Arc::new(super::strategy::RangePartition::new(boundaries.clone()))
            }
        }
    }
}

/// Information about a shard.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ShardInfo {
    /// Shard identifier.
    pub id: ShardId,
    /// Node address hosting this shard (primary).
    pub primary_addr: String,
    /// Addresses of replica nodes.
    pub replica_addrs: Vec<String>,
    /// Estimated number of nodes in this shard.
    pub estimated_node_count: u64,
    /// Estimated number of edges in this shard.
    pub estimated_edge_count: u64,
    /// Whether this shard is online and accepting requests.
    pub online: bool,
}

impl ShardInfo {
    /// Creates a new shard info.
    pub fn new(id: ShardId, primary_addr: String) -> Self {
        Self {
            id,
            primary_addr,
            replica_addrs: Vec::new(),
            estimated_node_count: 0,
            estimated_edge_count: 0,
            online: true,
        }
    }
}

/// Manages shard metadata, assignments, and routing information.
///
/// The ShardManager tracks:
/// - Which nodes host which shards
/// - Shard health and statistics
/// - Partition strategy for routing
pub struct ShardManager {
    /// Configuration.
    config: ShardConfig,
    /// Partition strategy.
    strategy: Arc<dyn PartitionStrategy>,
    /// Shard metadata indexed by shard ID.
    shards: HashMap<ShardId, ShardInfo>,
    /// This node's shard ID (if hosting a shard).
    local_shard_id: Option<ShardId>,
}

impl std::fmt::Debug for ShardManager {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ShardManager")
            .field("config", &self.config)
            .field("strategy", &self.strategy.describe())
            .field("shards", &self.shards)
            .field("local_shard_id", &self.local_shard_id)
            .finish()
    }
}

impl ShardManager {
    /// Creates a new shard manager with the given configuration.
    pub fn new(config: ShardConfig) -> Self {
        let strategy = config.build_strategy();
        let num_shards = config.num_shards;

        // Initialize shard info (addresses will be populated later)
        let mut shards = HashMap::new();
        for id in 0..num_shards {
            shards.insert(id, ShardInfo::new(id, String::new()));
        }

        Self {
            config,
            strategy,
            shards,
            local_shard_id: None,
        }
    }

    /// Creates a single-shard manager (no sharding).
    pub fn single_shard() -> Self {
        Self::new(ShardConfig::default())
    }

    /// Sets this node as hosting a specific shard.
    pub fn set_local_shard(&mut self, shard_id: ShardId, addr: String) {
        self.local_shard_id = Some(shard_id);
        if let Some(info) = self.shards.get_mut(&shard_id) {
            info.primary_addr = addr;
        }
    }

    /// Returns the local shard ID, if any.
    pub fn local_shard_id(&self) -> Option<ShardId> {
        self.local_shard_id
    }

    /// Returns whether this node hosts the given shard.
    pub fn is_local_shard(&self, shard_id: ShardId) -> bool {
        self.local_shard_id == Some(shard_id)
    }

    /// Returns the shard ID for a node.
    pub fn shard_for_node(&self, node_id: neural_core::NodeId) -> ShardId {
        self.strategy.shard_for_node(node_id)
    }

    /// Returns the shard ID for an edge.
    pub fn shard_for_edge(
        &self,
        source: neural_core::NodeId,
        target: neural_core::NodeId,
    ) -> ShardId {
        self.strategy.shard_for_edge(source, target)
    }

    /// Returns whether a node belongs to the local shard.
    pub fn is_local_node(&self, node_id: neural_core::NodeId) -> bool {
        self.local_shard_id
            .map(|local| self.shard_for_node(node_id) == local)
            .unwrap_or(false)
    }

    /// Returns shard info by ID.
    pub fn get_shard(&self, shard_id: ShardId) -> Option<&ShardInfo> {
        self.shards.get(&shard_id)
    }

    /// Returns mutable shard info by ID.
    pub fn get_shard_mut(&mut self, shard_id: ShardId) -> Option<&mut ShardInfo> {
        self.shards.get_mut(&shard_id)
    }

    /// Returns all shard infos.
    pub fn all_shards(&self) -> impl Iterator<Item = &ShardInfo> {
        self.shards.values()
    }

    /// Returns the total number of shards.
    pub fn num_shards(&self) -> u32 {
        self.config.num_shards
    }

    /// Returns the partition strategy.
    pub fn strategy(&self) -> &dyn PartitionStrategy {
        self.strategy.as_ref()
    }

    /// Updates shard address.
    pub fn update_shard_address(&mut self, shard_id: ShardId, addr: String) {
        if let Some(info) = self.shards.get_mut(&shard_id) {
            info.primary_addr = addr;
        }
    }

    /// Updates shard statistics.
    pub fn update_shard_stats(
        &mut self,
        shard_id: ShardId,
        node_count: u64,
        edge_count: u64,
    ) {
        if let Some(info) = self.shards.get_mut(&shard_id) {
            info.estimated_node_count = node_count;
            info.estimated_edge_count = edge_count;
        }
    }

    /// Marks a shard as online or offline.
    pub fn set_shard_online(&mut self, shard_id: ShardId, online: bool) {
        if let Some(info) = self.shards.get_mut(&shard_id) {
            info.online = online;
        }
    }

    /// Returns shards that might contain nodes with a label.
    pub fn shards_for_label(&self, label: &str) -> Vec<ShardId> {
        self.strategy.shards_for_label(label)
    }

    /// Returns the configuration.
    pub fn config(&self) -> &ShardConfig {
        &self.config
    }

    /// Returns addresses for all online shards.
    pub fn online_shard_addresses(&self) -> Vec<(ShardId, String)> {
        self.shards
            .values()
            .filter(|s| s.online && !s.primary_addr.is_empty())
            .map(|s| (s.id, s.primary_addr.clone()))
            .collect()
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use neural_core::NodeId;

    #[test]
    fn test_shard_manager_creation() {
        let config = ShardConfig::hash(4);
        let manager = ShardManager::new(config);

        assert_eq!(manager.num_shards(), 4);
        assert!(manager.local_shard_id().is_none());
    }

    #[test]
    fn test_shard_manager_local_shard() {
        let config = ShardConfig::hash(4);
        let mut manager = ShardManager::new(config);

        manager.set_local_shard(2, "localhost:9000".to_string());

        assert_eq!(manager.local_shard_id(), Some(2));
        assert!(manager.is_local_shard(2));
        assert!(!manager.is_local_shard(0));
    }

    #[test]
    fn test_shard_manager_routing() {
        let config = ShardConfig::hash(4);
        let manager = ShardManager::new(config);

        // Different nodes should route to potentially different shards
        let shard1 = manager.shard_for_node(NodeId::new(1));
        let shard2 = manager.shard_for_node(NodeId::new(2));

        assert!(shard1 < 4);
        assert!(shard2 < 4);
    }

    #[test]
    fn test_shard_info_update() {
        let config = ShardConfig::hash(2);
        let mut manager = ShardManager::new(config);

        manager.update_shard_address(0, "node1:9000".to_string());
        manager.update_shard_stats(0, 1000, 5000);

        let shard = manager.get_shard(0).unwrap();
        assert_eq!(shard.primary_addr, "node1:9000");
        assert_eq!(shard.estimated_node_count, 1000);
        assert_eq!(shard.estimated_edge_count, 5000);
    }

    #[test]
    fn test_single_shard_mode() {
        let manager = ShardManager::single_shard();

        assert_eq!(manager.num_shards(), 1);

        // All nodes should go to shard 0
        for i in 0..100 {
            assert_eq!(manager.shard_for_node(NodeId::new(i)), 0);
        }
    }
}
