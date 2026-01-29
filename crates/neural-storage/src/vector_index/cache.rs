//! Query result cache for distributed vector search.
//!
//! Uses LRU eviction with TTL-based expiration to cache search results.
//! Cache keys are based on SimHash of query vectors for approximate matching.

use lru::LruCache;
use neural_core::NodeId;
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};
use std::num::NonZeroUsize;
use std::time::{Duration, Instant};

/// Statistics about cache usage.
#[derive(Debug, Clone, Default)]
pub struct CacheStats {
    /// Number of cache hits.
    pub hits: u64,
    /// Number of cache misses.
    pub misses: u64,
    /// Number of entries evicted.
    pub evictions: u64,
    /// Number of entries expired.
    pub expirations: u64,
    /// Current number of entries.
    pub size: usize,
}

impl CacheStats {
    /// Returns the hit rate as a percentage.
    pub fn hit_rate(&self) -> f64 {
        let total = self.hits + self.misses;
        if total == 0 {
            0.0
        } else {
            (self.hits as f64 / total as f64) * 100.0
        }
    }
}

/// Cache key based on query vector hash.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct QueryCacheKey {
    /// SimHash of the query vector for approximate matching.
    query_hash: u64,
    /// Number of neighbors requested.
    k: usize,
}

impl QueryCacheKey {
    /// Creates a new cache key from a query vector and k.
    fn new(query: &[f32], k: usize) -> Self {
        Self {
            query_hash: simhash_f32(query),
            k,
        }
    }
}

/// Cached search result with timestamp for TTL.
#[derive(Debug, Clone)]
struct CachedResult {
    /// Search results (node_id, score).
    results: Vec<(NodeId, f32)>,
    /// When this entry was created.
    created_at: Instant,
}

impl CachedResult {
    /// Creates a new cached result.
    fn new(results: Vec<(NodeId, f32)>) -> Self {
        Self {
            results,
            created_at: Instant::now(),
        }
    }

    /// Checks if this entry has expired.
    fn is_expired(&self, ttl: Duration) -> bool {
        self.created_at.elapsed() > ttl
    }
}

/// LRU cache for vector search results.
///
/// Features:
/// - SimHash-based keys for approximate query matching
/// - TTL-based expiration
/// - LRU eviction when capacity is reached
///
/// # Example
///
/// ```ignore
/// use neural_storage::vector_index::QueryResultCache;
/// use std::time::Duration;
///
/// let mut cache = QueryResultCache::new(1000, Duration::from_secs(60));
///
/// // Cache a result
/// let results = vec![(NodeId::new(1), 0.95), (NodeId::new(2), 0.85)];
/// cache.put(&query_vector, 10, results.clone());
///
/// // Retrieve from cache
/// if let Some(cached) = cache.get(&query_vector, 10) {
///     // Use cached results
/// }
/// ```
pub struct QueryResultCache {
    /// LRU cache storing results.
    cache: LruCache<QueryCacheKey, CachedResult>,
    /// Time-to-live for cache entries.
    ttl: Duration,
    /// Cache statistics.
    stats: CacheStats,
}

impl QueryResultCache {
    /// Creates a new cache with the given capacity and TTL.
    ///
    /// # Arguments
    ///
    /// * `max_size` - Maximum number of entries to cache
    /// * `ttl` - Time-to-live for cache entries
    pub fn new(max_size: usize, ttl: Duration) -> Self {
        let capacity = NonZeroUsize::new(max_size.max(1)).unwrap();
        Self {
            cache: LruCache::new(capacity),
            ttl,
            stats: CacheStats::default(),
        }
    }

    /// Gets a cached result for the given query and k.
    ///
    /// Returns `None` if:
    /// - The entry doesn't exist
    /// - The entry has expired (will also remove it)
    pub fn get(&mut self, query: &[f32], k: usize) -> Option<Vec<(NodeId, f32)>> {
        let key = QueryCacheKey::new(query, k);

        match self.cache.get(&key) {
            Some(cached) => {
                if cached.is_expired(self.ttl) {
                    // Entry expired, remove it
                    self.cache.pop(&key);
                    self.stats.expirations += 1;
                    self.stats.misses += 1;
                    self.stats.size = self.cache.len();
                    None
                } else {
                    self.stats.hits += 1;
                    Some(cached.results.clone())
                }
            }
            None => {
                self.stats.misses += 1;
                None
            }
        }
    }

    /// Caches a search result.
    ///
    /// If the cache is full, the least recently used entry is evicted.
    pub fn put(&mut self, query: &[f32], k: usize, results: Vec<(NodeId, f32)>) {
        let key = QueryCacheKey::new(query, k);
        let was_full = self.cache.len() >= self.cache.cap().get();

        self.cache.put(key, CachedResult::new(results));

        if was_full {
            self.stats.evictions += 1;
        }
        self.stats.size = self.cache.len();
    }

    /// Returns the current cache statistics.
    pub fn stats(&self) -> &CacheStats {
        &self.stats
    }

    /// Returns the number of entries in the cache.
    pub fn len(&self) -> usize {
        self.cache.len()
    }

    /// Returns true if the cache is empty.
    pub fn is_empty(&self) -> bool {
        self.cache.is_empty()
    }

    /// Clears all entries from the cache.
    pub fn clear(&mut self) {
        self.cache.clear();
        self.stats.size = 0;
    }

    /// Removes expired entries from the cache.
    ///
    /// This is an O(n) operation and should be called periodically
    /// in a background task rather than on every operation.
    pub fn evict_expired(&mut self) {
        let ttl = self.ttl;
        let expired_keys: Vec<_> = self
            .cache
            .iter()
            .filter(|(_, v)| v.is_expired(ttl))
            .map(|(k, _)| k.clone())
            .collect();

        for key in expired_keys {
            self.cache.pop(&key);
            self.stats.expirations += 1;
        }
        self.stats.size = self.cache.len();
    }
}

impl std::fmt::Debug for QueryResultCache {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("QueryResultCache")
            .field("size", &self.cache.len())
            .field("capacity", &self.cache.cap())
            .field("ttl", &self.ttl)
            .field("stats", &self.stats)
            .finish()
    }
}

/// Computes a SimHash of a f32 vector for approximate matching.
///
/// SimHash produces similar hashes for similar vectors, making it useful
/// for cache key generation where we want to match approximately equal queries.
fn simhash_f32(vector: &[f32]) -> u64 {
    let mut hasher = DefaultHasher::new();

    // Quantize floats to reduce sensitivity to small differences
    // Use 3 decimal places of precision
    for &v in vector {
        let quantized = (v * 1000.0).round() as i32;
        quantized.hash(&mut hasher);
    }

    hasher.finish()
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cache_basic() {
        let mut cache = QueryResultCache::new(100, Duration::from_secs(60));

        let query = vec![1.0, 0.0, 0.0];
        let results = vec![(NodeId::new(1), 0.95), (NodeId::new(2), 0.85)];

        // Initially empty
        assert!(cache.get(&query, 10).is_none());

        // Add to cache
        cache.put(&query, 10, results.clone());

        // Should be retrievable
        let cached = cache.get(&query, 10);
        assert!(cached.is_some());
        assert_eq!(cached.unwrap(), results);
    }

    #[test]
    fn test_cache_different_k() {
        let mut cache = QueryResultCache::new(100, Duration::from_secs(60));

        let query = vec![1.0, 0.0, 0.0];
        let results_10 = vec![(NodeId::new(1), 0.95)];
        let results_20 = vec![(NodeId::new(1), 0.95), (NodeId::new(2), 0.85)];

        cache.put(&query, 10, results_10.clone());
        cache.put(&query, 20, results_20.clone());

        // Different k values should have different cache entries
        assert_eq!(cache.get(&query, 10), Some(results_10));
        assert_eq!(cache.get(&query, 20), Some(results_20));
    }

    #[test]
    fn test_cache_lru_eviction() {
        let mut cache = QueryResultCache::new(2, Duration::from_secs(60));

        let query1 = vec![1.0, 0.0, 0.0];
        let query2 = vec![0.0, 1.0, 0.0];
        let query3 = vec![0.0, 0.0, 1.0];

        cache.put(&query1, 10, vec![(NodeId::new(1), 0.9)]);
        cache.put(&query2, 10, vec![(NodeId::new(2), 0.8)]);

        // Access query1 to make it recently used
        let _ = cache.get(&query1, 10);

        // Add query3, should evict query2 (least recently used)
        cache.put(&query3, 10, vec![(NodeId::new(3), 0.7)]);

        assert!(cache.get(&query1, 10).is_some());
        assert!(cache.get(&query2, 10).is_none()); // Evicted
        assert!(cache.get(&query3, 10).is_some());
    }

    #[test]
    fn test_cache_ttl_expiration() {
        let mut cache = QueryResultCache::new(100, Duration::from_millis(1));

        let query = vec![1.0, 0.0, 0.0];
        cache.put(&query, 10, vec![(NodeId::new(1), 0.95)]);

        // Sleep to let it expire
        std::thread::sleep(Duration::from_millis(10));

        // Should return None due to expiration
        assert!(cache.get(&query, 10).is_none());
        assert_eq!(cache.stats().expirations, 1);
    }

    #[test]
    fn test_cache_stats() {
        let mut cache = QueryResultCache::new(100, Duration::from_secs(60));

        let query = vec![1.0, 0.0, 0.0];

        // Miss
        let _ = cache.get(&query, 10);
        assert_eq!(cache.stats().misses, 1);
        assert_eq!(cache.stats().hits, 0);

        // Add
        cache.put(&query, 10, vec![(NodeId::new(1), 0.95)]);

        // Hit
        let _ = cache.get(&query, 10);
        assert_eq!(cache.stats().hits, 1);
        assert_eq!(cache.stats().misses, 1);

        // Hit rate should be 50%
        assert!((cache.stats().hit_rate() - 50.0).abs() < 0.1);
    }

    #[test]
    fn test_simhash_similar_vectors() {
        let v1 = vec![1.0, 0.0, 0.0];
        let v2 = vec![1.0001, 0.0001, 0.0001]; // Very similar
        let v3 = vec![0.0, 1.0, 0.0]; // Different

        let h1 = simhash_f32(&v1);
        let h2 = simhash_f32(&v2);
        let h3 = simhash_f32(&v3);

        // Similar vectors should have same hash (due to quantization)
        assert_eq!(h1, h2);
        // Different vectors should have different hashes
        assert_ne!(h1, h3);
    }

    #[test]
    fn test_cache_clear() {
        let mut cache = QueryResultCache::new(100, Duration::from_secs(60));

        let query = vec![1.0, 0.0, 0.0];
        cache.put(&query, 10, vec![(NodeId::new(1), 0.95)]);
        assert_eq!(cache.len(), 1);

        cache.clear();
        assert_eq!(cache.len(), 0);
        assert!(cache.is_empty());
    }

    #[test]
    fn test_evict_expired() {
        let mut cache = QueryResultCache::new(100, Duration::from_millis(1));

        cache.put(&vec![1.0, 0.0], 10, vec![(NodeId::new(1), 0.9)]);
        cache.put(&vec![0.0, 1.0], 10, vec![(NodeId::new(2), 0.8)]);

        std::thread::sleep(Duration::from_millis(10));

        cache.evict_expired();
        assert_eq!(cache.len(), 0);
    }
}
