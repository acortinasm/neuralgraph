//! Memory tracking and limits for NeuralGraphDB.
//!
//! This module provides memory usage monitoring with configurable limits
//! and warning thresholds.
//!
//! # Example
//!
//! ```ignore
//! use neural_storage::memory::MemoryTracker;
//!
//! // Create a tracker with 4GB limit and 80% warning threshold
//! let tracker = MemoryTracker::new(4096, 80.0);
//!
//! // Try to allocate memory
//! tracker.try_allocate(1024 * 1024)?; // 1MB
//!
//! // Check usage
//! println!("Usage: {:.1}%", tracker.usage_percent());
//!
//! // Deallocate when done
//! tracker.deallocate(1024 * 1024);
//! ```

use std::sync::atomic::{AtomicUsize, Ordering};
use thiserror::Error;

/// Memory tracking errors.
#[derive(Debug, Error)]
pub enum MemoryError {
    #[error("Memory limit exceeded: requested {requested} bytes, available {available} bytes (limit: {limit} bytes)")]
    LimitExceeded {
        requested: usize,
        available: usize,
        limit: usize,
    },
}

/// Tracks memory usage with configurable limits and warnings.
///
/// Thread-safe using atomic operations for concurrent access.
#[derive(Debug)]
pub struct MemoryTracker {
    /// Current allocated bytes
    current_bytes: AtomicUsize,
    /// Maximum allowed bytes (0 = unlimited)
    limit_bytes: usize,
    /// Warning threshold as a percentage (0.0 - 100.0)
    warn_threshold: f64,
    /// Whether a warning has been issued (to avoid spam)
    warning_issued: AtomicUsize, // 0 = no, 1 = yes
}

impl MemoryTracker {
    /// Creates a new memory tracker.
    ///
    /// # Arguments
    /// * `limit_mb` - Memory limit in megabytes (0 = unlimited)
    /// * `warn_percent` - Warning threshold as percentage (0-100)
    pub fn new(limit_mb: usize, warn_percent: f64) -> Self {
        Self {
            current_bytes: AtomicUsize::new(0),
            limit_bytes: limit_mb * 1024 * 1024,
            warn_threshold: warn_percent.clamp(0.0, 100.0),
            warning_issued: AtomicUsize::new(0),
        }
    }

    /// Creates an unlimited memory tracker (no limit enforcement).
    pub fn unlimited() -> Self {
        Self {
            current_bytes: AtomicUsize::new(0),
            limit_bytes: 0,
            warn_threshold: 100.0,
            warning_issued: AtomicUsize::new(0),
        }
    }

    /// Attempts to allocate memory, checking against the limit.
    ///
    /// Returns `Ok(())` if allocation succeeds, `Err` if limit would be exceeded.
    /// Also logs a warning if usage exceeds the warning threshold.
    pub fn try_allocate(&self, bytes: usize) -> Result<(), MemoryError> {
        if self.limit_bytes == 0 {
            // Unlimited - just track
            self.current_bytes.fetch_add(bytes, Ordering::Relaxed);
            return Ok(());
        }

        let current = self.current_bytes.load(Ordering::Relaxed);
        let new_total = current.saturating_add(bytes);

        if new_total > self.limit_bytes {
            return Err(MemoryError::LimitExceeded {
                requested: bytes,
                available: self.limit_bytes.saturating_sub(current),
                limit: self.limit_bytes,
            });
        }

        self.current_bytes.fetch_add(bytes, Ordering::Relaxed);

        // Check warning threshold
        let usage = self.usage_percent();
        if usage > self.warn_threshold {
            // Only warn once per threshold crossing
            if self.warning_issued.swap(1, Ordering::Relaxed) == 0 {
                tracing::warn!(
                    usage_percent = format!("{:.1}", usage),
                    threshold = format!("{:.1}", self.warn_threshold),
                    current_mb = self.current_bytes.load(Ordering::Relaxed) / (1024 * 1024),
                    limit_mb = self.limit_bytes / (1024 * 1024),
                    "Memory usage exceeds warning threshold"
                );
            }
        }

        Ok(())
    }

    /// Records a deallocation, reducing tracked usage.
    pub fn deallocate(&self, bytes: usize) {
        let prev = self.current_bytes.fetch_sub(bytes, Ordering::Relaxed);

        // Reset warning if we've gone back below threshold
        if self.limit_bytes > 0 {
            let new_usage = (prev.saturating_sub(bytes) as f64 / self.limit_bytes as f64) * 100.0;
            if new_usage < self.warn_threshold {
                self.warning_issued.store(0, Ordering::Relaxed);
            }
        }
    }

    /// Returns current memory usage in bytes.
    pub fn current_bytes(&self) -> usize {
        self.current_bytes.load(Ordering::Relaxed)
    }

    /// Returns memory limit in bytes (0 = unlimited).
    pub fn limit_bytes(&self) -> usize {
        self.limit_bytes
    }

    /// Returns current usage as a percentage of the limit.
    ///
    /// Returns 0.0 if unlimited.
    pub fn usage_percent(&self) -> f64 {
        if self.limit_bytes == 0 {
            return 0.0;
        }
        let current = self.current_bytes.load(Ordering::Relaxed) as f64;
        (current / self.limit_bytes as f64) * 100.0
    }

    /// Returns available bytes before hitting the limit.
    ///
    /// Returns `usize::MAX` if unlimited.
    pub fn available_bytes(&self) -> usize {
        if self.limit_bytes == 0 {
            return usize::MAX;
        }
        self.limit_bytes.saturating_sub(self.current_bytes.load(Ordering::Relaxed))
    }

    /// Resets the tracker to zero usage.
    pub fn reset(&self) {
        self.current_bytes.store(0, Ordering::Relaxed);
        self.warning_issued.store(0, Ordering::Relaxed);
    }
}

impl Default for MemoryTracker {
    fn default() -> Self {
        Self::unlimited()
    }
}

/// Estimates the memory size of common graph operations.
pub mod estimates {
    use neural_core::PropertyValue;

    /// Estimates bytes needed for a node with properties.
    pub fn node_size(num_properties: usize, avg_property_size: usize) -> usize {
        // Base node overhead (NodeId, label pointer, property map overhead)
        let base = 64;
        // Properties: key string + value
        let props = num_properties * (32 + avg_property_size);
        base + props
    }

    /// Estimates bytes needed for an edge.
    pub fn edge_size(has_type: bool) -> usize {
        // Base edge: source NodeId + target NodeId + edge ID
        let base = 24;
        // Edge type string (if present)
        let type_size = if has_type { 32 } else { 0 };
        base + type_size
    }

    /// Estimates bytes for a property value.
    pub fn property_value_size(value: &PropertyValue) -> usize {
        match value {
            PropertyValue::Null => 1,
            PropertyValue::Bool(_) => 1,
            PropertyValue::Int(_) => 8,
            PropertyValue::Float(_) => 8,
            PropertyValue::String(s) => 24 + s.len(),
            PropertyValue::Date(s) | PropertyValue::DateTime(s) => 24 + s.len(),
            PropertyValue::Vector(v) => 24 + v.len() * 4,
            PropertyValue::Array(a) => 24 + a.iter().map(property_value_size).sum::<usize>(),
            PropertyValue::Map(m) => {
                24 + m.iter()
                    .map(|(k, v)| 24 + k.len() + property_value_size(v))
                    .sum::<usize>()
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_unlimited_tracker() {
        let tracker = MemoryTracker::unlimited();
        assert_eq!(tracker.limit_bytes(), 0);
        assert_eq!(tracker.usage_percent(), 0.0);

        // Should always succeed
        tracker.try_allocate(1_000_000_000).unwrap();
        assert_eq!(tracker.current_bytes(), 1_000_000_000);

        tracker.deallocate(1_000_000_000);
        assert_eq!(tracker.current_bytes(), 0);
    }

    #[test]
    fn test_limited_tracker() {
        // 1MB limit, 50% warning
        let tracker = MemoryTracker::new(1, 50.0);
        assert_eq!(tracker.limit_bytes(), 1024 * 1024);

        // Allocate 256KB - should succeed
        tracker.try_allocate(256 * 1024).unwrap();
        assert!(tracker.usage_percent() < 50.0);

        // Allocate another 512KB - should succeed but trigger warning
        tracker.try_allocate(512 * 1024).unwrap();
        assert!(tracker.usage_percent() > 50.0);

        // Try to allocate more than available - should fail
        let result = tracker.try_allocate(512 * 1024);
        assert!(result.is_err());
    }

    #[test]
    fn test_limit_exceeded_error() {
        let tracker = MemoryTracker::new(1, 80.0); // 1MB limit

        // Try to allocate 2MB
        let result = tracker.try_allocate(2 * 1024 * 1024);
        match result {
            Err(MemoryError::LimitExceeded { requested, available, limit }) => {
                assert_eq!(requested, 2 * 1024 * 1024);
                assert_eq!(available, 1024 * 1024);
                assert_eq!(limit, 1024 * 1024);
            }
            _ => panic!("Expected LimitExceeded error"),
        }
    }

    #[test]
    fn test_available_bytes() {
        let tracker = MemoryTracker::new(1, 80.0); // 1MB
        assert_eq!(tracker.available_bytes(), 1024 * 1024);

        tracker.try_allocate(256 * 1024).unwrap();
        assert_eq!(tracker.available_bytes(), 768 * 1024);

        // Unlimited tracker
        let unlimited = MemoryTracker::unlimited();
        assert_eq!(unlimited.available_bytes(), usize::MAX);
    }

    #[test]
    fn test_reset() {
        let tracker = MemoryTracker::new(1, 80.0);
        tracker.try_allocate(512 * 1024).unwrap();
        assert!(tracker.current_bytes() > 0);

        tracker.reset();
        assert_eq!(tracker.current_bytes(), 0);
    }

    #[test]
    fn test_estimates() {
        use neural_core::PropertyValue;

        // Node estimate
        let node_size = estimates::node_size(3, 50);
        assert!(node_size > 64); // At least base size

        // Edge estimate
        let edge_with_type = estimates::edge_size(true);
        let edge_without_type = estimates::edge_size(false);
        assert!(edge_with_type > edge_without_type);

        // Property value estimates
        assert!(estimates::property_value_size(&PropertyValue::Null) < 10);
        assert!(estimates::property_value_size(&PropertyValue::Int(42)) == 8);

        let string_val = PropertyValue::String("hello world".to_string());
        assert!(estimates::property_value_size(&string_val) > 11);
    }
}
