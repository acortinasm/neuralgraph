//! Cross-shard query coordinator.
//!
//! Handles executing queries across multiple shards and merging results.

use super::manager::{ShardInfo, ShardManager};
use super::router::{QueryPlan, ShardRouter};
use super::strategy::ShardId;
use std::collections::HashMap;
use std::sync::Arc;

/// Result from a single shard query.
#[derive(Debug, Clone)]
pub struct ShardResult {
    /// The shard that produced this result.
    pub shard_id: ShardId,
    /// Number of rows/records returned.
    pub row_count: usize,
    /// Raw result data (serialized for transport).
    pub data: Vec<u8>,
    /// Execution time in milliseconds.
    pub execution_time_ms: u64,
    /// Whether this result is partial (shard was unavailable).
    pub is_partial: bool,
}

impl ShardResult {
    /// Creates a new shard result.
    pub fn new(shard_id: ShardId, data: Vec<u8>, row_count: usize) -> Self {
        Self {
            shard_id,
            row_count,
            data,
            execution_time_ms: 0,
            is_partial: false,
        }
    }

    /// Creates an empty result for an unavailable shard.
    pub fn unavailable(shard_id: ShardId) -> Self {
        Self {
            shard_id,
            row_count: 0,
            data: Vec::new(),
            execution_time_ms: 0,
            is_partial: true,
        }
    }

    /// Sets execution time.
    pub fn with_execution_time(mut self, ms: u64) -> Self {
        self.execution_time_ms = ms;
        self
    }
}

/// Aggregated result from multiple shards.
#[derive(Debug)]
pub struct AggregatedResult {
    /// Results from each shard.
    pub shard_results: Vec<ShardResult>,
    /// Total row count across all shards.
    pub total_row_count: usize,
    /// Whether any shard returned partial results.
    pub has_partial_results: bool,
    /// Shards that failed or timed out.
    pub failed_shards: Vec<ShardId>,
    /// Total execution time (max across shards for parallel execution).
    pub total_time_ms: u64,
}

impl AggregatedResult {
    /// Creates a new aggregated result from shard results.
    pub fn from_results(results: Vec<ShardResult>) -> Self {
        let total_row_count = results.iter().map(|r| r.row_count).sum();
        let has_partial_results = results.iter().any(|r| r.is_partial);
        let failed_shards: Vec<_> = results
            .iter()
            .filter(|r| r.is_partial)
            .map(|r| r.shard_id)
            .collect();
        let total_time_ms = results.iter().map(|r| r.execution_time_ms).max().unwrap_or(0);

        Self {
            shard_results: results,
            total_row_count,
            has_partial_results,
            failed_shards,
            total_time_ms,
        }
    }

    /// Returns true if all shards responded successfully.
    pub fn is_complete(&self) -> bool {
        !self.has_partial_results && self.failed_shards.is_empty()
    }

    /// Returns combined data from all shards.
    pub fn combined_data(&self) -> Vec<u8> {
        let total_len: usize = self.shard_results.iter().map(|r| r.data.len()).sum();
        let mut combined = Vec::with_capacity(total_len);
        for result in &self.shard_results {
            combined.extend_from_slice(&result.data);
        }
        combined
    }
}

/// Strategy for executing queries across shards.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExecutionStrategy {
    /// Execute on shards sequentially.
    Sequential,
    /// Execute on all shards in parallel.
    Parallel,
    /// Execute locally first, then remote if needed.
    LocalFirst,
}

/// Configuration for cross-shard query execution.
#[derive(Debug, Clone)]
pub struct CoordinatorConfig {
    /// Execution strategy.
    pub strategy: ExecutionStrategy,
    /// Timeout for individual shard queries (ms).
    pub shard_timeout_ms: u64,
    /// Maximum number of retries per shard.
    pub max_retries: u32,
    /// Whether to continue on partial failures.
    pub allow_partial_results: bool,
}

impl Default for CoordinatorConfig {
    fn default() -> Self {
        Self {
            strategy: ExecutionStrategy::Parallel,
            shard_timeout_ms: 30_000,
            max_retries: 2,
            allow_partial_results: false,
        }
    }
}

/// Coordinates query execution across multiple shards.
///
/// The coordinator is responsible for:
/// - Determining which shards need to be queried
/// - Dispatching queries to shards (local or remote)
/// - Collecting and merging results
/// - Handling failures and retries
pub struct ShardCoordinator {
    /// Configuration.
    config: CoordinatorConfig,
    /// Shard manager reference.
    manager: Arc<ShardManager>,
}

impl ShardCoordinator {
    /// Creates a new coordinator.
    pub fn new(manager: Arc<ShardManager>, config: CoordinatorConfig) -> Self {
        Self { config, manager }
    }

    /// Creates a coordinator with default configuration.
    pub fn with_defaults(manager: Arc<ShardManager>) -> Self {
        Self::new(manager, CoordinatorConfig::default())
    }

    /// Returns the coordinator configuration.
    pub fn config(&self) -> &CoordinatorConfig {
        &self.config
    }

    /// Returns the shard manager.
    pub fn manager(&self) -> &ShardManager {
        &self.manager
    }

    /// Creates a router for query planning.
    pub fn router(&self) -> ShardRouter<'_> {
        ShardRouter::new(&self.manager)
    }

    /// Determines which shards need to be queried for a plan.
    pub fn target_shards(&self, plan: &QueryPlan) -> Vec<&ShardInfo> {
        plan.target_shards
            .iter()
            .filter_map(|&id| self.manager.get_shard(id))
            .collect()
    }

    /// Returns online shards from a query plan.
    pub fn online_targets(&self, plan: &QueryPlan) -> Vec<&ShardInfo> {
        self.target_shards(plan)
            .into_iter()
            .filter(|s| s.online)
            .collect()
    }

    /// Checks if a query can be executed locally only.
    pub fn is_local_only(&self, plan: &QueryPlan) -> bool {
        plan.local_only
    }

    /// Checks if all required shards are online.
    pub fn all_shards_available(&self, plan: &QueryPlan) -> bool {
        plan.target_shards.iter().all(|&id| {
            self.manager
                .get_shard(id)
                .map(|s| s.online)
                .unwrap_or(false)
        })
    }

    /// Returns unavailable shards from a query plan.
    pub fn unavailable_shards(&self, plan: &QueryPlan) -> Vec<ShardId> {
        plan.target_shards
            .iter()
            .filter(|&&id| {
                self.manager
                    .get_shard(id)
                    .map(|s| !s.online)
                    .unwrap_or(true)
            })
            .copied()
            .collect()
    }

    /// Executes a query plan and returns aggregated results.
    ///
    /// This is a synchronous stub - actual implementation would use
    /// async/await for parallel execution.
    pub fn execute_plan<F>(&self, plan: &QueryPlan, executor: F) -> Result<AggregatedResult, String>
    where
        F: Fn(ShardId, &ShardInfo) -> Result<ShardResult, String>,
    {
        // Check availability
        let unavailable = self.unavailable_shards(plan);
        if !unavailable.is_empty() && !self.config.allow_partial_results {
            return Err(format!(
                "Shards unavailable: {:?}",
                unavailable
            ));
        }

        let mut results = Vec::new();

        // Execute on each shard
        for &shard_id in &plan.target_shards {
            match self.manager.get_shard(shard_id) {
                Some(info) if info.online => {
                    match executor(shard_id, info) {
                        Ok(result) => results.push(result),
                        Err(_) if self.config.allow_partial_results => {
                            results.push(ShardResult::unavailable(shard_id));
                        }
                        Err(e) => return Err(e),
                    }
                }
                Some(_) | None if self.config.allow_partial_results => {
                    results.push(ShardResult::unavailable(shard_id));
                }
                _ => {
                    return Err(format!("Shard {} unavailable", shard_id));
                }
            }
        }

        Ok(AggregatedResult::from_results(results))
    }

    /// Merges results using a custom merge function.
    pub fn merge_results<T, F>(
        &self,
        aggregated: &AggregatedResult,
        merge_fn: F,
    ) -> Result<T, String>
    where
        F: FnOnce(&[ShardResult]) -> Result<T, String>,
    {
        if !self.config.allow_partial_results && aggregated.has_partial_results {
            return Err(format!(
                "Partial results from shards: {:?}",
                aggregated.failed_shards
            ));
        }
        merge_fn(&aggregated.shard_results)
    }
}

impl std::fmt::Debug for ShardCoordinator {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ShardCoordinator")
            .field("config", &self.config)
            .field("manager", &self.manager)
            .finish()
    }
}

// =============================================================================
// Result Merging Utilities
// =============================================================================

/// Utilities for merging shard results.
pub mod merge {
    use super::*;
    use neural_core::NodeId;
    use ordered_float::OrderedFloat;
    use std::cmp::Reverse;
    use std::collections::BinaryHeap;

    /// Concatenates raw data from all shards.
    pub fn concat_data(results: &[ShardResult]) -> Vec<u8> {
        let total: usize = results.iter().map(|r| r.data.len()).sum();
        let mut data = Vec::with_capacity(total);
        for r in results {
            data.extend_from_slice(&r.data);
        }
        data
    }

    /// Sums row counts across shards.
    pub fn sum_counts(results: &[ShardResult]) -> usize {
        results.iter().map(|r| r.row_count).sum()
    }

    /// Returns the maximum execution time.
    pub fn max_time(results: &[ShardResult]) -> u64 {
        results.iter().map(|r| r.execution_time_ms).max().unwrap_or(0)
    }

    /// Checks if all results are complete (non-partial).
    pub fn all_complete(results: &[ShardResult]) -> bool {
        results.iter().all(|r| !r.is_partial)
    }

    /// Merges vector search results from multiple shards.
    ///
    /// Uses a min-heap to efficiently find the top-k highest scoring results
    /// across all shards.
    ///
    /// # Arguments
    ///
    /// * `results` - Shard results containing serialized (NodeId, score) pairs
    /// * `k` - Number of top results to return
    ///
    /// # Returns
    ///
    /// Top-k results sorted by score (highest first).
    ///
    /// # Data Format
    ///
    /// Each shard result's data field should contain:
    /// - Repeated entries of: 8 bytes (u64 node_id) + 4 bytes (f32 score)
    pub fn merge_vector_results(
        results: &[ShardResult],
        k: usize,
    ) -> Result<Vec<(NodeId, f32)>, String> {
        // Min-heap: store (score, node_id), smallest score at top
        let mut heap: BinaryHeap<Reverse<(OrderedFloat<f32>, NodeId)>> =
            BinaryHeap::with_capacity(k + 1);

        for result in results {
            if result.is_partial {
                continue; // Skip failed shards
            }

            // Parse serialized results
            let entries = deserialize_vector_results(&result.data)?;

            for (node_id, score) in entries {
                heap.push(Reverse((OrderedFloat(score), node_id)));
                if heap.len() > k {
                    heap.pop(); // Remove smallest
                }
            }
        }

        // Extract from heap and sort by score descending
        let mut merged: Vec<_> = heap
            .into_iter()
            .map(|Reverse((score, node_id))| (node_id, score.0))
            .collect();

        merged.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        Ok(merged)
    }

    /// Serializes vector results for transport.
    ///
    /// Format: Repeated [u64 node_id (LE), f32 score (LE)]
    pub fn serialize_vector_results(results: &[(NodeId, f32)]) -> Vec<u8> {
        let mut data = Vec::with_capacity(results.len() * 12);
        for (node_id, score) in results {
            data.extend_from_slice(&node_id.as_u64().to_le_bytes());
            data.extend_from_slice(&score.to_le_bytes());
        }
        data
    }

    /// Deserializes vector results from transport format.
    fn deserialize_vector_results(data: &[u8]) -> Result<Vec<(NodeId, f32)>, String> {
        if data.len() % 12 != 0 {
            return Err(format!(
                "Invalid data length: {} bytes (expected multiple of 12)",
                data.len()
            ));
        }

        let mut results = Vec::with_capacity(data.len() / 12);
        for chunk in data.chunks_exact(12) {
            let node_id = u64::from_le_bytes([
                chunk[0], chunk[1], chunk[2], chunk[3],
                chunk[4], chunk[5], chunk[6], chunk[7],
            ]);
            let score = f32::from_le_bytes([chunk[8], chunk[9], chunk[10], chunk[11]]);
            results.push((NodeId::new(node_id), score));
        }

        Ok(results)
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sharding::ShardConfig;

    fn make_coordinator(num_shards: u32) -> ShardCoordinator {
        let config = ShardConfig::hash(num_shards);
        let manager = Arc::new(ShardManager::new(config));
        ShardCoordinator::with_defaults(manager)
    }

    #[test]
    fn test_coordinator_creation() {
        let coord = make_coordinator(4);
        assert_eq!(coord.manager().num_shards(), 4);
    }

    #[test]
    fn test_query_plan_targets() {
        let coord = make_coordinator(4);
        let router = coord.router();
        let plan = router.plan_full_scan();

        assert_eq!(plan.target_shards.len(), 4);
        assert!(plan.requires_merge);
    }

    #[test]
    fn test_shard_result_creation() {
        let result = ShardResult::new(0, vec![1, 2, 3], 10);
        assert_eq!(result.shard_id, 0);
        assert_eq!(result.row_count, 10);
        assert!(!result.is_partial);

        let unavail = ShardResult::unavailable(1);
        assert_eq!(unavail.shard_id, 1);
        assert!(unavail.is_partial);
    }

    #[test]
    fn test_aggregated_result() {
        let results = vec![
            ShardResult::new(0, vec![1, 2], 5),
            ShardResult::new(1, vec![3, 4], 3),
        ];
        let agg = AggregatedResult::from_results(results);

        assert_eq!(agg.total_row_count, 8);
        assert!(!agg.has_partial_results);
        assert!(agg.is_complete());
        assert_eq!(agg.combined_data(), vec![1, 2, 3, 4]);
    }

    #[test]
    fn test_aggregated_with_partial() {
        let results = vec![
            ShardResult::new(0, vec![1, 2], 5),
            ShardResult::unavailable(1),
        ];
        let agg = AggregatedResult::from_results(results);

        assert_eq!(agg.total_row_count, 5);
        assert!(agg.has_partial_results);
        assert!(!agg.is_complete());
        assert_eq!(agg.failed_shards, vec![1]);
    }

    #[test]
    fn test_execute_plan_success() {
        let coord = make_coordinator(2);

        // Set shards online
        let config = ShardConfig::hash(2);
        let mut manager = ShardManager::new(config);
        manager.set_local_shard(0, "localhost:9000".to_string());
        manager.update_shard_address(1, "localhost:9001".to_string());

        let coord = ShardCoordinator::with_defaults(Arc::new(manager));
        let router = coord.router();
        let plan = router.plan_full_scan();

        let result = coord.execute_plan(&plan, |shard_id, _info| {
            Ok(ShardResult::new(shard_id, vec![shard_id as u8], 1))
        });

        assert!(result.is_ok());
        let agg = result.unwrap();
        assert_eq!(agg.shard_results.len(), 2);
    }

    #[test]
    fn test_merge_utilities() {
        let results = vec![
            ShardResult::new(0, vec![1, 2], 5).with_execution_time(100),
            ShardResult::new(1, vec![3, 4], 3).with_execution_time(200),
        ];

        assert_eq!(merge::concat_data(&results), vec![1, 2, 3, 4]);
        assert_eq!(merge::sum_counts(&results), 8);
        assert_eq!(merge::max_time(&results), 200);
        assert!(merge::all_complete(&results));
    }

    #[test]
    fn test_merge_vector_results() {
        use neural_core::NodeId;

        // Create shard results with serialized vector data
        let shard0_results = vec![
            (NodeId::new(1), 0.9f32),
            (NodeId::new(2), 0.7f32),
        ];
        let shard1_results = vec![
            (NodeId::new(3), 0.95f32),
            (NodeId::new(4), 0.6f32),
        ];

        let results = vec![
            ShardResult::new(0, merge::serialize_vector_results(&shard0_results), 2),
            ShardResult::new(1, merge::serialize_vector_results(&shard1_results), 2),
        ];

        let merged = merge::merge_vector_results(&results, 3).unwrap();

        assert_eq!(merged.len(), 3);
        // Highest scores first
        assert_eq!(merged[0].0, NodeId::new(3)); // 0.95
        assert_eq!(merged[1].0, NodeId::new(1)); // 0.9
        assert_eq!(merged[2].0, NodeId::new(2)); // 0.7
    }

    #[test]
    fn test_merge_vector_results_with_partial() {
        use neural_core::NodeId;

        let shard0_results = vec![(NodeId::new(1), 0.9f32)];

        let results = vec![
            ShardResult::new(0, merge::serialize_vector_results(&shard0_results), 1),
            ShardResult::unavailable(1), // Partial result
        ];

        let merged = merge::merge_vector_results(&results, 5).unwrap();

        // Should only have results from shard 0
        assert_eq!(merged.len(), 1);
        assert_eq!(merged[0].0, NodeId::new(1));
    }

    #[test]
    fn test_serialize_deserialize_vector_results() {
        use neural_core::NodeId;

        let original = vec![
            (NodeId::new(100), 0.95f32),
            (NodeId::new(200), 0.85f32),
            (NodeId::new(300), 0.75f32),
        ];

        let serialized = merge::serialize_vector_results(&original);
        assert_eq!(serialized.len(), 36); // 3 * 12 bytes

        // Create a shard result with the serialized data
        let results = vec![ShardResult::new(0, serialized, 3)];
        let restored = merge::merge_vector_results(&results, 10).unwrap();

        // Should match original (sorted by score desc)
        assert_eq!(restored.len(), 3);
        assert_eq!(restored[0], (NodeId::new(100), 0.95));
        assert_eq!(restored[1], (NodeId::new(200), 0.85));
        assert_eq!(restored[2], (NodeId::new(300), 0.75));
    }

    #[test]
    fn test_coordinator_config() {
        let config = CoordinatorConfig {
            strategy: ExecutionStrategy::Sequential,
            shard_timeout_ms: 5000,
            max_retries: 1,
            allow_partial_results: true,
        };

        let manager = Arc::new(ShardManager::new(ShardConfig::hash(2)));
        let coord = ShardCoordinator::new(manager, config);

        assert_eq!(coord.config().strategy, ExecutionStrategy::Sequential);
        assert!(coord.config().allow_partial_results);
    }
}
