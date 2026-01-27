//! Query routing for sharded graphs.
//!
//! Handles routing queries to appropriate shards and merging results.

use super::manager::ShardManager;
use super::strategy::ShardId;
use neural_core::NodeId;
use std::collections::HashSet;

/// A query plan indicating which shards to query.
#[derive(Debug, Clone)]
pub struct QueryPlan {
    /// Shards that must be queried.
    pub target_shards: Vec<ShardId>,
    /// Whether results need to be merged from multiple shards.
    pub requires_merge: bool,
    /// Whether the query can be executed locally only.
    pub local_only: bool,
    /// Estimated cost (lower is better).
    pub estimated_cost: u64,
}

impl QueryPlan {
    /// Creates a plan targeting a single shard.
    pub fn single_shard(shard_id: ShardId, is_local: bool) -> Self {
        Self {
            target_shards: vec![shard_id],
            requires_merge: false,
            local_only: is_local,
            estimated_cost: 1,
        }
    }

    /// Creates a plan targeting all shards.
    pub fn all_shards(num_shards: u32, local_shard: Option<ShardId>) -> Self {
        let target_shards: Vec<_> = (0..num_shards).collect();
        let local_only = num_shards == 1 && local_shard == Some(0);
        Self {
            target_shards,
            requires_merge: num_shards > 1,
            local_only,
            estimated_cost: num_shards as u64,
        }
    }

    /// Creates a plan for specific shards.
    pub fn specific_shards(shards: Vec<ShardId>, local_shard: Option<ShardId>) -> Self {
        let requires_merge = shards.len() > 1;
        let local_only = shards.len() == 1 && local_shard == Some(shards[0]);
        let estimated_cost = shards.len() as u64;
        Self {
            target_shards: shards,
            requires_merge,
            local_only,
            estimated_cost,
        }
    }
}

/// Represents a sharded query with routing information.
#[derive(Debug)]
pub struct ShardedQuery {
    /// The original query string (for forwarding).
    pub query: String,
    /// The query plan.
    pub plan: QueryPlan,
    /// Node IDs explicitly referenced in the query (for targeted routing).
    pub referenced_nodes: Vec<NodeId>,
}

impl ShardedQuery {
    /// Creates a new sharded query.
    pub fn new(query: String, plan: QueryPlan) -> Self {
        Self {
            query,
            plan,
            referenced_nodes: Vec::new(),
        }
    }

    /// Adds referenced node IDs for routing.
    pub fn with_referenced_nodes(mut self, nodes: Vec<NodeId>) -> Self {
        self.referenced_nodes = nodes;
        self
    }
}

/// Routes queries to appropriate shards based on query analysis.
pub struct ShardRouter<'a> {
    manager: &'a ShardManager,
}

impl<'a> ShardRouter<'a> {
    /// Creates a new shard router.
    pub fn new(manager: &'a ShardManager) -> Self {
        Self { manager }
    }

    /// Plans a query that targets a specific node by ID.
    pub fn plan_node_lookup(&self, node_id: NodeId) -> QueryPlan {
        let shard = self.manager.shard_for_node(node_id);
        let is_local = self.manager.is_local_shard(shard);
        QueryPlan::single_shard(shard, is_local)
    }

    /// Plans a query that targets multiple specific nodes.
    pub fn plan_multi_node_lookup(&self, node_ids: &[NodeId]) -> QueryPlan {
        // Find unique shards needed
        let shards: HashSet<_> = node_ids
            .iter()
            .map(|id| self.manager.shard_for_node(*id))
            .collect();

        let shards: Vec<_> = shards.into_iter().collect();
        QueryPlan::specific_shards(shards, self.manager.local_shard_id())
    }

    /// Plans a query for a label scan (no specific nodes).
    pub fn plan_label_scan(&self, label: &str) -> QueryPlan {
        let shards = self.manager.shards_for_label(label);

        if shards.len() == 1 {
            QueryPlan::single_shard(shards[0], self.manager.is_local_shard(shards[0]))
        } else {
            QueryPlan::specific_shards(shards, self.manager.local_shard_id())
        }
    }

    /// Plans a full scan query (must hit all shards).
    pub fn plan_full_scan(&self) -> QueryPlan {
        QueryPlan::all_shards(self.manager.num_shards(), self.manager.local_shard_id())
    }

    /// Plans a traversal starting from a specific node.
    pub fn plan_traversal(&self, start_node: NodeId, hops: u32) -> QueryPlan {
        // For single-hop, we might stay on one shard (if using vertex-cut)
        // For multi-hop, we likely need all shards
        if hops <= 1 {
            // Single hop: start shard + potentially adjacent shards
            // For simplicity, assume we need all shards for traversals
            // (optimization: track edge distribution for better routing)
            QueryPlan::all_shards(self.manager.num_shards(), self.manager.local_shard_id())
        } else {
            // Multi-hop: definitely need all shards
            QueryPlan::all_shards(self.manager.num_shards(), self.manager.local_shard_id())
        }
    }

    /// Plans an edge query between two specific nodes.
    pub fn plan_edge_query(&self, source: NodeId, target: NodeId) -> QueryPlan {
        let edge_shard = self.manager.shard_for_edge(source, target);
        let is_local = self.manager.is_local_shard(edge_shard);
        QueryPlan::single_shard(edge_shard, is_local)
    }

    /// Returns shards that need to be queried for a CREATE operation.
    pub fn plan_create_node(&self, node_id: NodeId) -> QueryPlan {
        let shard = self.manager.shard_for_node(node_id);
        let is_local = self.manager.is_local_shard(shard);
        QueryPlan::single_shard(shard, is_local)
    }

    /// Returns shards that need to be queried for a CREATE edge operation.
    pub fn plan_create_edge(&self, source: NodeId, target: NodeId) -> QueryPlan {
        let shard = self.manager.shard_for_edge(source, target);
        let is_local = self.manager.is_local_shard(shard);
        QueryPlan::single_shard(shard, is_local)
    }

    /// Analyzes a query string and determines routing.
    ///
    /// This is a simplified analysis - a full implementation would parse
    /// the query and extract node IDs, labels, and patterns.
    pub fn analyze_query(&self, query: &str) -> QueryPlan {
        let query_lower = query.to_lowercase();

        // Simple heuristics for query routing
        // (In production, this would parse the AST)

        // Check for node ID lookups (WHERE id(n) = X)
        if query_lower.contains("id(") && query_lower.contains("=") {
            // Would need to extract the actual ID - for now, assume full scan
            return self.plan_full_scan();
        }

        // Check for label scans
        if let Some(label_start) = query_lower.find(':') {
            // Extract label (simplified)
            let after_colon = &query[label_start + 1..];
            if let Some(end) = after_colon.find(|c: char| !c.is_alphanumeric() && c != '_') {
                let label = &after_colon[..end];
                return self.plan_label_scan(label);
            }
        }

        // Default: full scan
        self.plan_full_scan()
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sharding::ShardConfig;

    fn make_manager(num_shards: u32, local_shard: Option<ShardId>) -> ShardManager {
        let config = ShardConfig::hash(num_shards);
        let mut manager = ShardManager::new(config);
        if let Some(shard) = local_shard {
            manager.set_local_shard(shard, "localhost:9000".to_string());
        }
        manager
    }

    #[test]
    fn test_plan_node_lookup() {
        let manager = make_manager(4, Some(0));
        let router = ShardRouter::new(&manager);

        let plan = router.plan_node_lookup(NodeId::new(42));

        assert_eq!(plan.target_shards.len(), 1);
        assert!(!plan.requires_merge);
    }

    #[test]
    fn test_plan_multi_node_lookup() {
        let manager = make_manager(4, Some(0));
        let router = ShardRouter::new(&manager);

        // Multiple nodes might span multiple shards
        let nodes: Vec<_> = (0..100).map(NodeId::new).collect();
        let plan = router.plan_multi_node_lookup(&nodes);

        // Should have some shards (likely all 4 with enough nodes)
        assert!(!plan.target_shards.is_empty());
    }

    #[test]
    fn test_plan_full_scan() {
        let manager = make_manager(4, Some(0));
        let router = ShardRouter::new(&manager);

        let plan = router.plan_full_scan();

        assert_eq!(plan.target_shards.len(), 4);
        assert!(plan.requires_merge);
        assert!(!plan.local_only);
    }

    #[test]
    fn test_single_shard_no_merge() {
        let manager = make_manager(1, Some(0));
        let router = ShardRouter::new(&manager);

        let plan = router.plan_full_scan();

        assert_eq!(plan.target_shards.len(), 1);
        assert!(!plan.requires_merge);
        assert!(plan.local_only);
    }

    #[test]
    fn test_plan_edge_query() {
        let manager = make_manager(4, Some(0));
        let router = ShardRouter::new(&manager);

        let source = NodeId::new(10);
        let target = NodeId::new(20);
        let plan = router.plan_edge_query(source, target);

        // Edge should be on one shard (following source)
        assert_eq!(plan.target_shards.len(), 1);
        assert!(!plan.requires_merge);
    }

    #[test]
    fn test_query_plan_estimated_cost() {
        let plan1 = QueryPlan::single_shard(0, true);
        let plan4 = QueryPlan::all_shards(4, None);

        assert_eq!(plan1.estimated_cost, 1);
        assert_eq!(plan4.estimated_cost, 4);
    }
}
