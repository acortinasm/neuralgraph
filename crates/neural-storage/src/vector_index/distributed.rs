//! Distributed vector index with scatter-gather search.
//!
//! Provides horizontal scaling of vector search across multiple nodes,
//! with features including:
//! - Scatter-gather parallel search
//! - Top-k result fusion using min-heap
//! - Query result caching
//! - Load balancing across replicas
//!
//! # Architecture
//!
//! The distributed index coordinates searches across multiple shards:
//!
//! 1. **Scatter**: Query is sent to all shards in parallel
//! 2. **Local Search**: Each shard searches its local HNSW index
//! 3. **Gather**: Results are collected from all shards
//! 4. **Merge**: Top-k merge algorithm combines results
//! 5. **Cache**: Final results are cached for future queries

use super::cache::{CacheStats, QueryResultCache};
use super::client::{VectorClientError, VectorClientPool};
use super::load_balancer::LoadBalancer;
use super::{DistanceMetric, VectorIndex, VectorIndexConfig};
use crate::sharding::{ShardId, ShardManager};
use neural_core::NodeId;
use ordered_float::OrderedFloat;
use std::cmp::Reverse;
use std::collections::BinaryHeap;
use std::sync::{Arc, RwLock};
use std::time::{Duration, Instant};

/// Result from searching a single shard.
pub type ShardSearchResult = Result<Vec<(NodeId, f32)>, VectorClientError>;

/// Configuration for distributed vector index.
#[derive(Debug, Clone)]
pub struct DistributedVectorConfig {
    /// Number of shards to distribute vectors across.
    pub num_shards: u32,
    /// Maximum number of entries to cache.
    pub cache_size: usize,
    /// Time-to-live for cache entries.
    pub cache_ttl: Duration,
    /// Timeout for individual shard requests.
    pub timeout: Duration,
    /// Maximum number of shards to query concurrently.
    pub max_concurrent_shards: usize,
    /// Oversampling factor: request k * factor from each shard.
    /// Higher values improve recall but increase network traffic.
    pub oversampling_factor: f32,
    /// Distance metric for similarity computation.
    pub metric: DistanceMetric,
    /// Vector dimension.
    pub dimension: usize,
}

impl Default for DistributedVectorConfig {
    fn default() -> Self {
        Self {
            num_shards: 4,
            cache_size: 10_000,
            cache_ttl: Duration::from_secs(300), // 5 minutes
            timeout: Duration::from_millis(100), // 100ms target latency
            max_concurrent_shards: 8,
            oversampling_factor: 1.5, // Request 50% more than needed
            metric: DistanceMetric::Cosine,
            dimension: 768,
        }
    }
}

impl DistributedVectorConfig {
    /// Creates a new configuration with the given number of shards and dimension.
    pub fn new(num_shards: u32, dimension: usize) -> Self {
        Self {
            num_shards,
            dimension,
            ..Default::default()
        }
    }

    /// Sets the cache size.
    pub fn with_cache_size(mut self, size: usize) -> Self {
        self.cache_size = size;
        self
    }

    /// Sets the cache TTL.
    pub fn with_cache_ttl(mut self, ttl: Duration) -> Self {
        self.cache_ttl = ttl;
        self
    }

    /// Sets the request timeout.
    pub fn with_timeout(mut self, timeout: Duration) -> Self {
        self.timeout = timeout;
        self
    }

    /// Sets the oversampling factor.
    pub fn with_oversampling_factor(mut self, factor: f32) -> Self {
        self.oversampling_factor = factor.max(1.0);
        self
    }

    /// Sets the distance metric.
    pub fn with_metric(mut self, metric: DistanceMetric) -> Self {
        self.metric = metric;
        self
    }
}

/// Distributed vector index with scatter-gather search.
///
/// Coordinates vector search across multiple shards, providing:
/// - Parallel scatter-gather search
/// - Top-k result merging
/// - Query result caching
/// - Load balancing
///
/// # Example
///
/// ```ignore
/// use neural_storage::vector_index::{
///     DistributedVectorIndex, DistributedVectorConfig,
///     RoundRobinBalancer, VectorClientPool,
/// };
/// use std::sync::Arc;
///
/// let config = DistributedVectorConfig::new(4, 768);
/// let balancer = Arc::new(RoundRobinBalancer::new());
/// let pool = Arc::new(VectorClientPool::new(Duration::from_secs(5)));
///
/// let index = DistributedVectorIndex::new(
///     shard_manager,
///     pool,
///     balancer,
///     config,
/// );
///
/// // Search for 100 nearest neighbors
/// let results = index.search(&query_vector, 100).await?;
/// ```
pub struct DistributedVectorIndex {
    /// Shard manager for routing.
    shard_manager: Arc<ShardManager>,
    /// Pool of clients for remote shards.
    client_pool: Arc<VectorClientPool>,
    /// Query result cache.
    cache: Arc<RwLock<QueryResultCache>>,
    /// Load balancer for replica selection.
    load_balancer: Arc<dyn LoadBalancer>,
    /// Configuration.
    config: DistributedVectorConfig,
    /// Local shard index (if this node hosts a shard).
    local_index: Option<VectorIndex>,
    /// Local shard ID.
    local_shard_id: Option<ShardId>,
}

impl DistributedVectorIndex {
    /// Creates a new distributed vector index.
    pub fn new(
        shard_manager: Arc<ShardManager>,
        client_pool: Arc<VectorClientPool>,
        load_balancer: Arc<dyn LoadBalancer>,
        config: DistributedVectorConfig,
    ) -> Self {
        let cache = QueryResultCache::new(config.cache_size, config.cache_ttl);

        Self {
            shard_manager,
            client_pool,
            cache: Arc::new(RwLock::new(cache)),
            load_balancer,
            config,
            local_index: None,
            local_shard_id: None,
        }
    }

    /// Creates a distributed index with a local shard.
    pub fn with_local_index(
        shard_manager: Arc<ShardManager>,
        client_pool: Arc<VectorClientPool>,
        load_balancer: Arc<dyn LoadBalancer>,
        config: DistributedVectorConfig,
        local_index: VectorIndex,
        local_shard_id: ShardId,
    ) -> Self {
        let cache = QueryResultCache::new(config.cache_size, config.cache_ttl);

        Self {
            shard_manager,
            client_pool,
            cache: Arc::new(RwLock::new(cache)),
            load_balancer,
            config,
            local_index: Some(local_index),
            local_shard_id: Some(local_shard_id),
        }
    }

    /// Returns the configuration.
    pub fn config(&self) -> &DistributedVectorConfig {
        &self.config
    }

    /// Returns cache statistics.
    pub fn cache_stats(&self) -> CacheStats {
        self.cache.read().unwrap().stats().clone()
    }

    /// Returns the local shard ID if one is configured.
    pub fn local_shard_id(&self) -> Option<ShardId> {
        self.local_shard_id
    }

    /// Adds a vector to the local shard.
    ///
    /// Returns an error if no local shard is configured.
    pub fn add_local(&mut self, node_id: NodeId, vector: &[f32]) -> Result<(), String> {
        match &mut self.local_index {
            Some(index) => {
                index.add(node_id, vector);
                Ok(())
            }
            None => Err("No local shard configured".to_string()),
        }
    }

    /// Searches for k nearest neighbors using scatter-gather.
    ///
    /// # Algorithm
    ///
    /// 1. Check cache for existing results
    /// 2. Calculate per-shard k with oversampling
    /// 3. Scatter: parallel search on all shards
    /// 4. Gather: collect results
    /// 5. Merge: top-k merge using min-heap
    /// 6. Cache result
    pub async fn search(&self, query: &[f32], k: usize) -> Result<Vec<(NodeId, f32)>, String> {
        // 1. Check cache
        {
            let mut cache = self.cache.write().unwrap();
            if let Some(cached) = cache.get(query, k) {
                return Ok(cached);
            }
        }

        // 2. Calculate per-shard k with oversampling
        let per_shard_k = ((k as f32) * self.config.oversampling_factor).ceil() as usize;
        let per_shard_k = per_shard_k.max(k);

        // 3. Scatter: parallel search on all shards
        let shard_results = self.scatter_search(query, per_shard_k).await;

        // 4-5. Gather & Merge: top-k merge using min-heap
        let merged = self.merge_top_k(shard_results, k);

        // 6. Cache result
        {
            let mut cache = self.cache.write().unwrap();
            cache.put(query, k, merged.clone());
        }

        Ok(merged)
    }

    /// Scatter search to all shards in parallel.
    async fn scatter_search(&self, query: &[f32], k: usize) -> Vec<ShardSearchResult> {
        let num_shards = self.config.num_shards;
        let mut futures = Vec::with_capacity(num_shards as usize);

        for shard_id in 0..num_shards {
            let future = self.search_shard(shard_id, query, k);
            futures.push(future);
        }

        // Execute all searches in parallel
        futures::future::join_all(futures).await
    }

    /// Search a single shard (local or remote).
    async fn search_shard(
        &self,
        shard_id: ShardId,
        query: &[f32],
        k: usize,
    ) -> ShardSearchResult {
        let start = Instant::now();

        let result = if Some(shard_id) == self.local_shard_id {
            // Local search
            self.search_local(query, k)
        } else {
            // Remote search
            self.search_remote(shard_id, query, k).await
        };

        // Record latency for load balancer
        let latency = start.elapsed();
        if let Some(info) = self.shard_manager.get_shard(shard_id) {
            self.load_balancer
                .record_latency(shard_id, &info.primary_addr, latency);
        }

        result
    }

    /// Search the local shard.
    fn search_local(&self, query: &[f32], k: usize) -> ShardSearchResult {
        match &self.local_index {
            Some(index) => Ok(index.search(query, k)),
            None => Err(VectorClientError::ServerError(
                "No local index available".to_string(),
            )),
        }
    }

    /// Search a remote shard with replica failover.
    ///
    /// Attempts to search the primary replica first, then falls back to
    /// other replicas on failure. Uses exponential backoff between retries.
    async fn search_remote(
        &self,
        shard_id: ShardId,
        query: &[f32],
        k: usize,
    ) -> ShardSearchResult {
        let info = match self.shard_manager.get_shard(shard_id) {
            Some(info) => info.clone(),
            None => {
                return Err(VectorClientError::ServerError(format!(
                    "Shard {} not found",
                    shard_id
                )));
            }
        };

        // Build replica list: primary + replicas
        let mut replicas = vec![info.primary_addr.clone()];
        replicas.extend(info.replica_addrs.iter().cloned());

        let mut last_error = None;
        let mut backoff = Duration::from_millis(10);
        const MAX_RETRIES: usize = 3;

        for attempt in 0..MAX_RETRIES {
            // Select replica using load balancer
            let selected = self.load_balancer.select_replica(shard_id, &replicas);

            match self.try_search_replica(shard_id, &selected, query, k).await {
                Ok(results) => {
                    // Mark as healthy on success
                    self.load_balancer.mark_healthy(shard_id, &selected);
                    return Ok(results);
                }
                Err(e) => {
                    // Mark as unhealthy and retry
                    self.load_balancer.mark_unhealthy(shard_id, &selected);
                    tracing_error(shard_id, &e);
                    last_error = Some(e);

                    if attempt < MAX_RETRIES - 1 {
                        tokio::time::sleep(backoff).await;
                        backoff *= 2; // Exponential backoff
                    }
                }
            }
        }

        Err(last_error.unwrap_or_else(|| {
            VectorClientError::ServerError("All replicas failed".to_string())
        }))
    }

    /// Attempt to search a specific replica address.
    async fn try_search_replica(
        &self,
        shard_id: ShardId,
        addr: &str,
        query: &[f32],
        k: usize,
    ) -> ShardSearchResult {
        let timeout = self.config.timeout;

        match tokio::time::timeout(
            timeout,
            self.client_pool.search_at_addr(shard_id, addr, query, k),
        )
        .await
        {
            Ok(result) => result,
            Err(_) => Err(VectorClientError::Timeout {
                timeout_ms: timeout.as_millis() as u64,
            }),
        }
    }

    /// Efficient top-k merge using a min-heap.
    ///
    /// Time complexity: O(n * log k) where n = total results across shards.
    ///
    /// # Algorithm
    ///
    /// Uses a min-heap (reversed binary heap) to maintain the top-k highest
    /// scoring results. For each result:
    /// 1. Push to heap
    /// 2. If heap size > k, pop the minimum (lowest score)
    ///
    /// Final heap contains top-k, extracted in sorted order.
    fn merge_top_k(&self, shard_results: Vec<ShardSearchResult>, k: usize) -> Vec<(NodeId, f32)> {
        // Min-heap: Reverse to get min-heap behavior (pop smallest)
        // We store (score, node_id) for ordering by score
        let mut heap: BinaryHeap<Reverse<(OrderedFloat<f32>, NodeId)>> =
            BinaryHeap::with_capacity(k + 1);

        let mut successful_shards = 0;
        let mut total_results = 0;

        for result in shard_results {
            match result {
                Ok(results) => {
                    successful_shards += 1;
                    total_results += results.len();

                    for (node_id, score) in results {
                        // Push to min-heap
                        heap.push(Reverse((OrderedFloat(score), node_id)));

                        // If we have more than k, remove the smallest
                        if heap.len() > k {
                            heap.pop();
                        }
                    }
                }
                Err(_) => {
                    // Shard failed, continue with partial results
                    // In production, might want to retry or log
                }
            }
        }

        // Log if we had partial results
        if successful_shards < self.config.num_shards as usize {
            tracing::warn!(
                successful_shards = successful_shards,
                total_shards = self.config.num_shards,
                total_results = total_results,
                "Distributed search had partial results"
            );
        }

        // Extract results from heap (min-heap, so we need to collect and sort)
        let mut merged: Vec<_> = heap
            .into_iter()
            .map(|Reverse((score, node_id))| (node_id, score.0))
            .collect();

        // Sort by score descending (highest first)
        merged.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        merged
    }

    /// Clears the query cache.
    pub fn clear_cache(&self) {
        self.cache.write().unwrap().clear();
    }

    /// Returns the number of shards.
    pub fn num_shards(&self) -> u32 {
        self.config.num_shards
    }
}

impl std::fmt::Debug for DistributedVectorIndex {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("DistributedVectorIndex")
            .field("num_shards", &self.config.num_shards)
            .field("dimension", &self.config.dimension)
            .field("local_shard", &self.local_shard_id)
            .field("cache_stats", &self.cache_stats())
            .finish()
    }
}

/// Log search errors using structured tracing.
fn tracing_error(shard_id: ShardId, error: &VectorClientError) {
    tracing::error!(shard_id = shard_id, error = %error, "Shard search error");
}

// =============================================================================
// Synchronous API for testing without async runtime
// =============================================================================

/// Simulated distributed search for testing without network.
///
/// Uses in-memory shard indices for local testing.
pub struct SimulatedDistributedIndex {
    /// Shard indices.
    shards: Vec<VectorIndex>,
    /// Configuration.
    config: DistributedVectorConfig,
    /// Query cache.
    cache: RwLock<QueryResultCache>,
}

impl SimulatedDistributedIndex {
    /// Creates a new simulated distributed index with the given shard count.
    pub fn new(num_shards: usize, dimension: usize) -> Self {
        let config = DistributedVectorConfig::new(num_shards as u32, dimension);
        let shards = (0..num_shards)
            .map(|_| VectorIndex::with_config(VectorIndexConfig::small(dimension)))
            .collect();
        let cache = QueryResultCache::new(config.cache_size, config.cache_ttl);

        Self {
            shards,
            config,
            cache: RwLock::new(cache),
        }
    }

    /// Adds a vector to the appropriate shard based on hash partitioning.
    pub fn add(&mut self, node_id: NodeId, vector: &[f32]) {
        let shard_id = (node_id.as_u64() as usize) % self.shards.len();
        self.shards[shard_id].add(node_id, vector);
    }

    /// Returns the total number of vectors across all shards.
    pub fn len(&self) -> usize {
        self.shards.iter().map(|s| s.len()).sum()
    }

    /// Returns true if the index is empty.
    pub fn is_empty(&self) -> bool {
        self.shards.iter().all(|s| s.is_empty())
    }

    /// Searches for k nearest neighbors across all shards.
    pub fn search(&self, query: &[f32], k: usize) -> Vec<(NodeId, f32)> {
        // Check cache
        {
            let mut cache = self.cache.write().unwrap();
            if let Some(cached) = cache.get(query, k) {
                return cached;
            }
        }

        // Calculate per-shard k with oversampling
        let per_shard_k = ((k as f32) * self.config.oversampling_factor).ceil() as usize;

        // Search each shard
        let shard_results: Vec<_> = self
            .shards
            .iter()
            .map(|shard| Ok(shard.search(query, per_shard_k)))
            .collect();

        // Merge results
        let merged = merge_top_k_sync(&shard_results, k);

        // Cache result
        {
            let mut cache = self.cache.write().unwrap();
            cache.put(query, k, merged.clone());
        }

        merged
    }

    /// Returns cache statistics.
    pub fn cache_stats(&self) -> CacheStats {
        self.cache.read().unwrap().stats().clone()
    }

    /// Returns the number of shards.
    pub fn num_shards(&self) -> usize {
        self.shards.len()
    }

    /// Returns the number of vectors in a specific shard.
    pub fn shard_size(&self, shard_id: usize) -> usize {
        self.shards.get(shard_id).map(|s| s.len()).unwrap_or(0)
    }
}

/// Synchronous top-k merge for testing.
fn merge_top_k_sync(shard_results: &[ShardSearchResult], k: usize) -> Vec<(NodeId, f32)> {
    let mut heap: BinaryHeap<Reverse<(OrderedFloat<f32>, NodeId)>> =
        BinaryHeap::with_capacity(k + 1);

    for result in shard_results {
        if let Ok(results) = result {
            for (node_id, score) in results {
                heap.push(Reverse((OrderedFloat(*score), *node_id)));
                if heap.len() > k {
                    heap.pop();
                }
            }
        }
    }

    // Extract from heap and sort by score descending
    let mut merged: Vec<_> = heap
        .into_iter()
        .map(|Reverse((score, node_id))| (node_id, score.0))
        .collect();

    merged.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    merged
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_merge_top_k_basic() {
        let results: Vec<ShardSearchResult> = vec![
            Ok(vec![
                (NodeId::new(1), 0.9),
                (NodeId::new(2), 0.7),
            ]),
            Ok(vec![
                (NodeId::new(3), 0.95),
                (NodeId::new(4), 0.6),
            ]),
        ];

        let merged = merge_top_k_sync(&results, 3);

        assert_eq!(merged.len(), 3);
        // Highest scores first
        assert_eq!(merged[0].0, NodeId::new(3)); // 0.95
        assert_eq!(merged[1].0, NodeId::new(1)); // 0.9
        assert_eq!(merged[2].0, NodeId::new(2)); // 0.7
    }

    #[test]
    fn test_merge_top_k_empty() {
        let results: Vec<ShardSearchResult> = vec![Ok(vec![]), Ok(vec![])];

        let merged = merge_top_k_sync(&results, 10);
        assert!(merged.is_empty());
    }

    #[test]
    fn test_merge_top_k_with_errors() {
        let results: Vec<ShardSearchResult> = vec![
            Ok(vec![(NodeId::new(1), 0.9)]),
            Err(VectorClientError::Timeout { timeout_ms: 100 }),
            Ok(vec![(NodeId::new(2), 0.8)]),
        ];

        let merged = merge_top_k_sync(&results, 5);

        // Should still get results from successful shards
        assert_eq!(merged.len(), 2);
    }

    #[test]
    fn test_merge_top_k_k_larger_than_results() {
        let results: Vec<ShardSearchResult> = vec![
            Ok(vec![(NodeId::new(1), 0.9)]),
            Ok(vec![(NodeId::new(2), 0.8)]),
        ];

        let merged = merge_top_k_sync(&results, 100);

        // Should return all available results
        assert_eq!(merged.len(), 2);
    }

    #[test]
    fn test_simulated_distributed_index() {
        let mut index = SimulatedDistributedIndex::new(4, 3);

        // Add vectors
        for i in 0..100 {
            let vec = vec![(i as f32) / 100.0, 1.0 - (i as f32) / 100.0, 0.5];
            index.add(NodeId::new(i), &vec);
        }

        assert_eq!(index.len(), 100);

        // Search
        let query = vec![0.5, 0.5, 0.5];
        let results = index.search(&query, 10);

        assert_eq!(results.len(), 10);

        // Results should be sorted by score (descending)
        for i in 1..results.len() {
            assert!(results[i - 1].1 >= results[i].1);
        }
    }

    #[test]
    fn test_simulated_distributed_index_caching() {
        let mut index = SimulatedDistributedIndex::new(2, 3);

        for i in 0..20 {
            let vec = vec![i as f32, 0.0, 0.0];
            index.add(NodeId::new(i), &vec);
        }

        let query = vec![10.0, 0.0, 0.0];

        // First search - cache miss
        let _ = index.search(&query, 5);
        assert_eq!(index.cache_stats().misses, 1);
        assert_eq!(index.cache_stats().hits, 0);

        // Second search - cache hit
        let _ = index.search(&query, 5);
        assert_eq!(index.cache_stats().hits, 1);
    }

    #[test]
    fn test_simulated_index_shard_distribution() {
        let mut index = SimulatedDistributedIndex::new(4, 3);

        // Add 100 vectors
        for i in 0..100u64 {
            index.add(NodeId::new(i), &[0.0, 0.0, 0.0]);
        }

        // Check distribution - should be roughly even
        for shard in 0..4 {
            let size = index.shard_size(shard);
            assert!(size >= 15 && size <= 35, "Shard {} has {} vectors", shard, size);
        }
    }

    #[test]
    fn test_distributed_config() {
        let config = DistributedVectorConfig::new(8, 1536)
            .with_cache_size(50_000)
            .with_cache_ttl(Duration::from_secs(600))
            .with_timeout(Duration::from_millis(50))
            .with_oversampling_factor(2.0)
            .with_metric(DistanceMetric::DotProduct);

        assert_eq!(config.num_shards, 8);
        assert_eq!(config.dimension, 1536);
        assert_eq!(config.cache_size, 50_000);
        assert_eq!(config.cache_ttl, Duration::from_secs(600));
        assert_eq!(config.timeout, Duration::from_millis(50));
        assert!((config.oversampling_factor - 2.0).abs() < 0.01);
        assert_eq!(config.metric, DistanceMetric::DotProduct);
    }

    #[test]
    fn test_oversampling_factor_minimum() {
        let config = DistributedVectorConfig::default()
            .with_oversampling_factor(0.5); // Should be clamped to 1.0

        assert!((config.oversampling_factor - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_recall_vs_centralized() {
        // Create a centralized index with higher dimensions for more realistic test
        let dim = 64;
        let mut centralized = VectorIndex::with_config(VectorIndexConfig::small(dim));

        // Create a distributed index with 4 shards
        let mut distributed = SimulatedDistributedIndex::new(4, dim);

        // Add same vectors to both - use normalized random-ish vectors
        let vectors: Vec<(NodeId, Vec<f32>)> = (0..1000)
            .map(|i| {
                let mut vec: Vec<f32> = (0..dim)
                    .map(|d| {
                        let seed = (i * 31 + d * 17) as f32;
                        (seed * 0.618033988749895).fract() * 2.0 - 1.0
                    })
                    .collect();
                // Normalize
                let norm: f32 = vec.iter().map(|x| x * x).sum::<f32>().sqrt();
                if norm > 0.0 {
                    for x in &mut vec {
                        *x /= norm;
                    }
                }
                (NodeId::new(i as u64), vec)
            })
            .collect();

        for (node_id, vec) in &vectors {
            centralized.add(*node_id, vec);
            distributed.add(*node_id, vec);
        }

        // Run multiple queries and compare results
        let queries: Vec<Vec<f32>> = (0..10)
            .map(|q| {
                let mut vec: Vec<f32> = (0..dim)
                    .map(|d| {
                        let seed = (q * 47 + d * 23) as f32;
                        (seed * 0.618033988749895).fract() * 2.0 - 1.0
                    })
                    .collect();
                let norm: f32 = vec.iter().map(|x| x * x).sum::<f32>().sqrt();
                if norm > 0.0 {
                    for x in &mut vec {
                        *x /= norm;
                    }
                }
                vec
            })
            .collect();

        let mut total_recall = 0.0;
        let k = 50; // Smaller k for more reliable recall

        for query in &queries {
            let centralized_results = centralized.search(query, k);
            let distributed_results = distributed.search(query, k);

            // Calculate recall
            let centralized_ids: std::collections::HashSet<_> =
                centralized_results.iter().map(|(id, _)| id).collect();
            let distributed_ids: std::collections::HashSet<_> =
                distributed_results.iter().map(|(id, _)| id).collect();

            let intersection = centralized_ids.intersection(&distributed_ids).count();
            let recall = if centralized_ids.is_empty() {
                1.0
            } else {
                intersection as f64 / centralized_ids.len() as f64
            };
            total_recall += recall;
        }

        let avg_recall = total_recall / queries.len() as f64;

        // With oversampling and proper merging, we should get good recall
        // Using 80% threshold to account for HNSW approximation differences
        assert!(
            avg_recall >= 0.80,
            "Average recall too low: {:.2}%",
            avg_recall * 100.0
        );
    }
}
