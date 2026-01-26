//! Multi-Version Concurrency Control (MVCC) core types.
//!
//! This module implements versioned storage for Snapshot Isolation.
//! Instead of overwriting values in place, we store a chain of versions,
//! allowing readers to see a consistent snapshot of the database at a
//! specific point in time (Transaction ID).

use crate::wal::TransactionId;
use neural_core::PropertyValue;
use serde::{Deserialize, Serialize};

/// A single version of a value.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct Version {
    /// The transaction that created this version.
    pub tx_id: TransactionId,
    /// The value (None indicates deletion/tombstone).
    pub value: Option<PropertyValue>,
}

/// A versioned container for a property.
///
/// Stores a history of values, ordered by transaction ID (descending).
/// Most recent version is first.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct VersionedValue {
    versions: Vec<Version>,
}

impl VersionedValue {
    /// Creates a new versioned value with an initial value.
    pub fn new(tx_id: TransactionId, value: PropertyValue) -> Self {
        Self {
            versions: vec![Version {
                tx_id,
                value: Some(value),
            }],
        }
    }

    /// Adds a new version (update).
    pub fn update(&mut self, tx_id: TransactionId, value: PropertyValue) {
        // Insert at the beginning (newest first)
        // In a real optimized system, we might use a linked list or similar,
        // but Vec is fine for MVP.
        self.versions.insert(0, Version {
            tx_id,
            value: Some(value),
        });
    }

    /// Adds a tombstone (deletion).
    pub fn delete(&mut self, tx_id: TransactionId) {
        self.versions.insert(0, Version {
            tx_id,
            value: None,
        });
    }

    /// Gets the visible value for a given snapshot transaction ID.
    ///
    /// Rules:
    /// 1. A transaction can always see its own writes (if we passed writing tx_id, but here we pass snapshot_id).
    /// 2. A transaction sees versions committed before it started (tx_id <= snapshot_id).
    ///
    /// Returns `None` if the value is deleted or no version is visible.
    pub fn get(&self, snapshot_id: TransactionId) -> Option<&PropertyValue> {
        // Iterate through versions (newest to oldest)
        for version in &self.versions {
            if version.tx_id <= snapshot_id {
                // This is the newest version visible to the snapshot
                return version.value.as_ref();
            }
        }
        None
    }

    /// Prunes versions older than the oldest active transaction (Vacuum).
    ///
    /// `min_active_tx_id`: The Transaction ID of the oldest active transaction.
    /// Any version older than this that is NOT the newest visible version for that ID
    /// can be discarded.
    pub fn vacuum(&mut self, min_active_tx_id: TransactionId) {
        // We need to keep:
        // 1. All versions newer than min_active_tx_id (future/concurrent)
        // 2. The newest version <= min_active_tx_id (the stable version)
        
        let mut split_idx = None;
        
        for (i, version) in self.versions.iter().enumerate() {
            if version.tx_id <= min_active_tx_id {
                // Found the stable version. Keep this one, drop the rest (older ones).
                // Actually, we keep this one because it's what min_active_tx_id sees.
                // Any older version is shadowed by this one for everyone >= min_active_tx_id.
                split_idx = Some(i + 1); 
                break;
            }
        }

        if let Some(idx) = split_idx {
            self.versions.truncate(idx);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_version_visibility() {
        let mut v = VersionedValue::new(1, PropertyValue::from(10)); // Tx 1 sets 10
        v.update(3, PropertyValue::from(30)); // Tx 3 updates to 30
        v.update(5, PropertyValue::from(50)); // Tx 5 updates to 50

        // Snapshot at Tx 0 (before anything)
        assert_eq!(v.get(0), None);

        // Snapshot at Tx 2 (should see Tx 1)
        assert_eq!(v.get(2), Some(&PropertyValue::from(10)));

        // Snapshot at Tx 4 (should see Tx 3)
        assert_eq!(v.get(4), Some(&PropertyValue::from(30)));

        // Snapshot at Tx 6 (should see Tx 5)
        assert_eq!(v.get(6), Some(&PropertyValue::from(50)));
    }

    #[test]
    fn test_deletion_visibility() {
        let mut v = VersionedValue::new(1, PropertyValue::from("alive"));
        v.delete(3); // Deleted at Tx 3

        assert_eq!(v.get(2), Some(&PropertyValue::from("alive")));
        assert_eq!(v.get(4), None); // Should be effectively None
    }

    #[test]
    fn test_vacuum() {
        let mut v = VersionedValue::new(1, PropertyValue::from(1));
        v.update(2, PropertyValue::from(2));
        v.update(3, PropertyValue::from(3));
        v.update(4, PropertyValue::from(4));

        // State: [v4(4), v3(3), v2(2), v1(1)]
        assert_eq!(v.versions.len(), 4);

        // Oldest active tx is 3. 
        // Everyone >= 3 sees v3 (or v4).
        // No one is < 3 (implied by "active").
        // So v2 and v1 are unreachable.
        // We keep v3 as the base.
        
        v.vacuum(3);
        
        // Should keep v4 (future) and v3 (current stable).
        // [v4(4), v3(3)]
        assert_eq!(v.versions.len(), 2);
        assert_eq!(v.versions[0].tx_id, 4);
        assert_eq!(v.versions[1].tx_id, 3);
        
        // Verify visibility still works for new txs
        assert_eq!(v.get(5), Some(&PropertyValue::from(4)));
        assert_eq!(v.get(3), Some(&PropertyValue::from(3)));
    }
}
