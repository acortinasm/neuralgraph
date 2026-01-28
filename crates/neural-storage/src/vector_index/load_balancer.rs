//! Load balancing strategies for distributed vector search.
//!
//! Provides multiple load balancing strategies for selecting replicas:
//! - Round-robin: Simple rotation through replicas
//! - Latency-aware: Prefer replicas with lower latency
//! - Weighted: Assign weights based on capacity/performance

use crate::sharding::ShardId;
use dashmap::DashMap;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::time::{Duration, Instant};

/// Trait for load balancing across replicas.
pub trait LoadBalancer: Send + Sync {
    /// Selects a replica for the given shard.
    ///
    /// # Arguments
    ///
    /// * `shard_id` - The shard to select a replica for
    /// * `replicas` - List of replica addresses
    ///
    /// # Returns
    ///
    /// The selected replica address.
    fn select_replica(&self, shard_id: ShardId, replicas: &[String]) -> String;

    /// Records the latency of a request to a replica.
    ///
    /// Used by latency-aware balancers to optimize selection.
    fn record_latency(&self, shard_id: ShardId, replica: &str, latency: Duration);

    /// Marks a replica as unhealthy.
    ///
    /// Unhealthy replicas should be avoided until they recover.
    fn mark_unhealthy(&self, shard_id: ShardId, replica: &str);

    /// Marks a replica as healthy again.
    fn mark_healthy(&self, shard_id: ShardId, replica: &str);
}

// =============================================================================
// Round-Robin Load Balancer
// =============================================================================

/// Simple round-robin load balancer.
///
/// Rotates through replicas in order, ensuring even distribution.
/// Thread-safe using atomic counters.
pub struct RoundRobinBalancer {
    /// Counter per shard for round-robin rotation.
    counters: DashMap<ShardId, AtomicUsize>,
    /// Unhealthy replicas to skip.
    unhealthy: DashMap<(ShardId, String), Instant>,
    /// How long to consider a replica unhealthy.
    unhealthy_duration: Duration,
}

impl RoundRobinBalancer {
    /// Creates a new round-robin balancer.
    pub fn new() -> Self {
        Self {
            counters: DashMap::new(),
            unhealthy: DashMap::new(),
            unhealthy_duration: Duration::from_secs(30),
        }
    }

    /// Sets how long a replica stays marked as unhealthy.
    pub fn with_unhealthy_duration(mut self, duration: Duration) -> Self {
        self.unhealthy_duration = duration;
        self
    }

    fn is_healthy(&self, shard_id: ShardId, replica: &str) -> bool {
        let key = (shard_id, replica.to_string());
        match self.unhealthy.get(&key) {
            Some(marked_at) => marked_at.elapsed() > self.unhealthy_duration,
            None => true,
        }
    }
}

impl Default for RoundRobinBalancer {
    fn default() -> Self {
        Self::new()
    }
}

impl LoadBalancer for RoundRobinBalancer {
    fn select_replica(&self, shard_id: ShardId, replicas: &[String]) -> String {
        if replicas.is_empty() {
            panic!("No replicas available for shard {}", shard_id);
        }

        if replicas.len() == 1 {
            return replicas[0].clone();
        }

        // Get or create counter for this shard
        let counter = self
            .counters
            .entry(shard_id)
            .or_insert_with(|| AtomicUsize::new(0));

        // Filter healthy replicas
        let healthy: Vec<_> = replicas
            .iter()
            .filter(|r| self.is_healthy(shard_id, r))
            .collect();

        // If all unhealthy, try all replicas anyway
        let candidates = if healthy.is_empty() {
            replicas.iter().collect()
        } else {
            healthy
        };

        let idx = counter.fetch_add(1, Ordering::Relaxed) % candidates.len();
        candidates[idx].clone()
    }

    fn record_latency(&self, _shard_id: ShardId, _replica: &str, _latency: Duration) {
        // Round-robin doesn't use latency information
    }

    fn mark_unhealthy(&self, shard_id: ShardId, replica: &str) {
        let key = (shard_id, replica.to_string());
        self.unhealthy.insert(key, Instant::now());
    }

    fn mark_healthy(&self, shard_id: ShardId, replica: &str) {
        let key = (shard_id, replica.to_string());
        self.unhealthy.remove(&key);
    }
}

// =============================================================================
// Latency-Aware Load Balancer
// =============================================================================

/// Latency information for a replica.
struct LatencyInfo {
    /// Exponential moving average of latency in microseconds.
    ema_latency_us: AtomicU64,
    /// Last update time.
    last_update: std::sync::RwLock<Instant>,
    /// Whether the replica is healthy.
    healthy: std::sync::atomic::AtomicBool,
}

impl LatencyInfo {
    fn new() -> Self {
        Self {
            ema_latency_us: AtomicU64::new(1000), // Start at 1ms
            last_update: std::sync::RwLock::new(Instant::now()),
            healthy: std::sync::atomic::AtomicBool::new(true),
        }
    }

    fn update_latency(&self, latency: Duration, alpha: f64) {
        let latency_us = latency.as_micros() as u64;
        let old = self.ema_latency_us.load(Ordering::Relaxed);
        let new = (alpha * latency_us as f64 + (1.0 - alpha) * old as f64) as u64;
        self.ema_latency_us.store(new, Ordering::Relaxed);
        *self.last_update.write().unwrap() = Instant::now();
    }

    fn get_latency(&self) -> u64 {
        self.ema_latency_us.load(Ordering::Relaxed)
    }
}

/// Latency-aware load balancer.
///
/// Tracks request latencies and prefers replicas with lower latency.
/// Uses exponential moving average (EMA) to smooth latency measurements.
pub struct LatencyAwareBalancer {
    /// Latency info per (shard, replica).
    latencies: DashMap<(ShardId, String), LatencyInfo>,
    /// EMA smoothing factor (0-1). Higher = more weight to recent samples.
    alpha: f64,
    /// Probability of exploring a random replica (0-1).
    exploration_rate: f64,
}

impl LatencyAwareBalancer {
    /// Creates a new latency-aware balancer.
    pub fn new() -> Self {
        Self {
            latencies: DashMap::new(),
            alpha: 0.3,            // 30% weight to new samples
            exploration_rate: 0.1, // 10% random exploration
        }
    }

    /// Sets the EMA smoothing factor.
    pub fn with_alpha(mut self, alpha: f64) -> Self {
        self.alpha = alpha.clamp(0.0, 1.0);
        self
    }

    /// Sets the exploration rate for discovering faster replicas.
    pub fn with_exploration_rate(mut self, rate: f64) -> Self {
        self.exploration_rate = rate.clamp(0.0, 1.0);
        self
    }

    fn get_or_create_info(&self, shard_id: ShardId, replica: &str) -> dashmap::mapref::one::Ref<'_, (ShardId, String), LatencyInfo> {
        let key = (shard_id, replica.to_string());
        self.latencies.entry(key.clone()).or_insert_with(LatencyInfo::new);
        self.latencies.get(&key).unwrap()
    }
}

impl Default for LatencyAwareBalancer {
    fn default() -> Self {
        Self::new()
    }
}

impl LoadBalancer for LatencyAwareBalancer {
    fn select_replica(&self, shard_id: ShardId, replicas: &[String]) -> String {
        if replicas.is_empty() {
            panic!("No replicas available for shard {}", shard_id);
        }

        if replicas.len() == 1 {
            return replicas[0].clone();
        }

        // Random exploration to discover potentially faster replicas
        let random_val: f64 = rand::random();
        if random_val < self.exploration_rate {
            let idx = (rand::random::<f64>() * replicas.len() as f64) as usize % replicas.len();
            return replicas[idx].clone();
        }

        // Select replica with lowest latency
        let mut best_replica = &replicas[0];
        let mut best_latency = u64::MAX;

        for replica in replicas {
            let info = self.get_or_create_info(shard_id, replica);
            if !info.healthy.load(Ordering::Relaxed) {
                continue;
            }
            let latency = info.get_latency();
            if latency < best_latency {
                best_latency = latency;
                best_replica = replica;
            }
        }

        best_replica.clone()
    }

    fn record_latency(&self, shard_id: ShardId, replica: &str, latency: Duration) {
        let info = self.get_or_create_info(shard_id, replica);
        info.update_latency(latency, self.alpha);
    }

    fn mark_unhealthy(&self, shard_id: ShardId, replica: &str) {
        let info = self.get_or_create_info(shard_id, replica);
        info.healthy.store(false, Ordering::Relaxed);
    }

    fn mark_healthy(&self, shard_id: ShardId, replica: &str) {
        let info = self.get_or_create_info(shard_id, replica);
        info.healthy.store(true, Ordering::Relaxed);
    }
}

// =============================================================================
// Weighted Load Balancer
// =============================================================================

/// Weighted load balancer.
///
/// Assigns weights to replicas and distributes load proportionally.
/// Useful when replicas have different capacities.
pub struct WeightedBalancer {
    /// Weights per (shard, replica).
    weights: DashMap<(ShardId, String), u32>,
    /// Default weight for new replicas.
    default_weight: u32,
    /// Counter for weighted selection.
    counters: DashMap<ShardId, AtomicU64>,
    /// Unhealthy replicas.
    unhealthy: DashMap<(ShardId, String), ()>,
}

impl WeightedBalancer {
    /// Creates a new weighted balancer.
    pub fn new() -> Self {
        Self {
            weights: DashMap::new(),
            default_weight: 100,
            counters: DashMap::new(),
            unhealthy: DashMap::new(),
        }
    }

    /// Sets the default weight for new replicas.
    pub fn with_default_weight(mut self, weight: u32) -> Self {
        self.default_weight = weight;
        self
    }

    /// Sets the weight for a specific replica.
    pub fn set_weight(&self, shard_id: ShardId, replica: &str, weight: u32) {
        let key = (shard_id, replica.to_string());
        self.weights.insert(key, weight);
    }

    fn get_weight(&self, shard_id: ShardId, replica: &str) -> u32 {
        let key = (shard_id, replica.to_string());
        self.weights.get(&key).map(|w| *w).unwrap_or(self.default_weight)
    }
}

impl Default for WeightedBalancer {
    fn default() -> Self {
        Self::new()
    }
}

impl LoadBalancer for WeightedBalancer {
    fn select_replica(&self, shard_id: ShardId, replicas: &[String]) -> String {
        if replicas.is_empty() {
            panic!("No replicas available for shard {}", shard_id);
        }

        if replicas.len() == 1 {
            return replicas[0].clone();
        }

        // Filter healthy replicas and compute total weight
        let healthy_with_weights: Vec<_> = replicas
            .iter()
            .filter(|r| !self.unhealthy.contains_key(&(shard_id, r.to_string())))
            .map(|r| (r, self.get_weight(shard_id, r)))
            .collect();

        if healthy_with_weights.is_empty() {
            // All unhealthy, pick first
            return replicas[0].clone();
        }

        let total_weight: u32 = healthy_with_weights.iter().map(|(_, w)| w).sum();
        if total_weight == 0 {
            return healthy_with_weights[0].0.clone();
        }

        // Get counter and select based on weighted distribution
        let counter = self
            .counters
            .entry(shard_id)
            .or_insert_with(|| AtomicU64::new(0));
        let val = counter.fetch_add(1, Ordering::Relaxed);
        let target = (val % total_weight as u64) as u32;

        let mut cumulative = 0u32;
        for (replica, weight) in &healthy_with_weights {
            cumulative += weight;
            if target < cumulative {
                return (*replica).clone();
            }
        }

        healthy_with_weights.last().unwrap().0.clone()
    }

    fn record_latency(&self, _shard_id: ShardId, _replica: &str, _latency: Duration) {
        // Weighted balancer doesn't automatically adjust weights based on latency
        // Use set_weight() to manually adjust
    }

    fn mark_unhealthy(&self, shard_id: ShardId, replica: &str) {
        let key = (shard_id, replica.to_string());
        self.unhealthy.insert(key, ());
    }

    fn mark_healthy(&self, shard_id: ShardId, replica: &str) {
        let key = (shard_id, replica.to_string());
        self.unhealthy.remove(&key);
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_round_robin_basic() {
        let balancer = RoundRobinBalancer::new();
        let replicas = vec!["a".to_string(), "b".to_string(), "c".to_string()];

        let selections: Vec<_> = (0..6)
            .map(|_| balancer.select_replica(0, &replicas))
            .collect();

        // Should rotate through replicas
        assert_eq!(selections[0], "a");
        assert_eq!(selections[1], "b");
        assert_eq!(selections[2], "c");
        assert_eq!(selections[3], "a");
        assert_eq!(selections[4], "b");
        assert_eq!(selections[5], "c");
    }

    #[test]
    fn test_round_robin_single_replica() {
        let balancer = RoundRobinBalancer::new();
        let replicas = vec!["only".to_string()];

        let selection = balancer.select_replica(0, &replicas);
        assert_eq!(selection, "only");
    }

    #[test]
    fn test_round_robin_unhealthy() {
        let balancer = RoundRobinBalancer::new()
            .with_unhealthy_duration(Duration::from_secs(60));
        let replicas = vec!["a".to_string(), "b".to_string()];

        // Mark 'a' as unhealthy
        balancer.mark_unhealthy(0, "a");

        // Should only select 'b'
        let selections: Vec<_> = (0..3)
            .map(|_| balancer.select_replica(0, &replicas))
            .collect();

        assert!(selections.iter().all(|s| s == "b"));

        // Mark 'a' as healthy again
        balancer.mark_healthy(0, "a");

        // Should now include 'a'
        let mut saw_a = false;
        for _ in 0..10 {
            if balancer.select_replica(0, &replicas) == "a" {
                saw_a = true;
                break;
            }
        }
        assert!(saw_a);
    }

    #[test]
    fn test_latency_aware_basic() {
        let balancer = LatencyAwareBalancer::new()
            .with_exploration_rate(0.0); // Disable random exploration for test

        let replicas = vec!["fast".to_string(), "slow".to_string()];

        // Record latencies
        balancer.record_latency(0, "fast", Duration::from_micros(100));
        balancer.record_latency(0, "slow", Duration::from_millis(10));

        // Should prefer 'fast'
        let selections: Vec<_> = (0..5)
            .map(|_| balancer.select_replica(0, &replicas))
            .collect();

        assert!(selections.iter().filter(|s| *s == "fast").count() >= 4);
    }

    #[test]
    fn test_latency_aware_exploration() {
        let balancer = LatencyAwareBalancer::new()
            .with_exploration_rate(1.0); // Always explore

        let replicas = vec!["a".to_string(), "b".to_string()];

        // Should randomly select both
        let mut saw_a = false;
        let mut saw_b = false;
        for _ in 0..100 {
            let selection = balancer.select_replica(0, &replicas);
            if selection == "a" {
                saw_a = true;
            }
            if selection == "b" {
                saw_b = true;
            }
        }
        assert!(saw_a && saw_b);
    }

    #[test]
    fn test_weighted_basic() {
        let balancer = WeightedBalancer::new();
        let replicas = vec!["heavy".to_string(), "light".to_string()];

        // Set weights: heavy=90, light=10
        balancer.set_weight(0, "heavy", 90);
        balancer.set_weight(0, "light", 10);

        // Count selections
        let mut heavy_count = 0;
        let mut light_count = 0;
        for _ in 0..100 {
            match balancer.select_replica(0, &replicas).as_str() {
                "heavy" => heavy_count += 1,
                "light" => light_count += 1,
                _ => {}
            }
        }

        // Should be roughly 90/10 split
        assert!(heavy_count > light_count * 5);
    }

    #[test]
    fn test_weighted_unhealthy() {
        let balancer = WeightedBalancer::new();
        let replicas = vec!["a".to_string(), "b".to_string()];

        balancer.mark_unhealthy(0, "a");

        // Should only select 'b'
        let selections: Vec<_> = (0..5)
            .map(|_| balancer.select_replica(0, &replicas))
            .collect();

        assert!(selections.iter().all(|s| s == "b"));
    }

    #[test]
    fn test_different_shards() {
        let balancer = RoundRobinBalancer::new();
        let replicas = vec!["a".to_string(), "b".to_string()];

        // Different shards should have independent counters
        let s0 = balancer.select_replica(0, &replicas);
        let s1 = balancer.select_replica(1, &replicas);

        // Both should start at first replica
        assert_eq!(s0, "a");
        assert_eq!(s1, "a");
    }
}
