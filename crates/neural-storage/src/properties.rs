//! Property storage for nodes and edges.
//!
//! This module provides key-value property storage for graph elements.
//! Includes both non-versioned (`PropertyStore`) and MVCC-versioned
//! (`VersionedPropertyStore`) implementations.

use crate::mvcc::VersionedValue;
use crate::wal::TransactionId;
use neural_core::{NodeId, PropertyValue};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

// =============================================================================
// PropertyStore
// =============================================================================

/// Storage for node or edge properties.
///
/// Properties are stored as a hash map of hash maps, providing O(1) average
/// access time for property lookups.
///
/// ## Example
///
/// ```
/// use neural_core::{NodeId, PropertyValue};
/// use neural_storage::PropertyStore;
///
/// let mut store = PropertyStore::new();
///
/// // Set properties for node 0
/// store.set(NodeId::new(0), "name", PropertyValue::from("Alice"));
/// store.set(NodeId::new(0), "age", PropertyValue::from(30i64));
///
/// // Get properties
/// assert_eq!(store.get(NodeId::new(0), "name"), Some(&PropertyValue::from("Alice")));
/// assert_eq!(store.get(NodeId::new(0), "age"), Some(&PropertyValue::from(30i64)));
/// ```
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct PropertyStore {
    /// Map from NodeId to property map
    data: HashMap<u64, HashMap<String, PropertyValue>>,
}

impl PropertyStore {
    /// Creates an empty property store.
    pub fn new() -> Self {
        Self::default()
    }

    /// Creates a property store with initial capacity.
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            data: HashMap::with_capacity(capacity),
        }
    }

    /// Sets a property for a node.
    pub fn set(&mut self, node: NodeId, key: impl Into<String>, value: PropertyValue) {
        self.data
            .entry(node.as_u64())
            .or_default()
            .insert(key.into(), value);
    }

    /// Gets a property for a node.
    pub fn get(&self, node: NodeId, key: &str) -> Option<&PropertyValue> {
        self.data
            .get(&node.as_u64())
            .and_then(|props| props.get(key))
    }

    /// Gets all properties for a node.
    pub fn get_all(&self, node: NodeId) -> Option<&HashMap<String, PropertyValue>> {
        self.data.get(&node.as_u64())
    }

    /// Removes a property for a node.
    pub fn remove(&mut self, node: NodeId, key: &str) -> Option<PropertyValue> {
        self.data
            .get_mut(&node.as_u64())
            .and_then(|props| props.remove(key))
    }

    /// Removes all properties for a node.
    pub fn remove_all(&mut self, node: NodeId) -> Option<HashMap<String, PropertyValue>> {
        self.data.remove(&node.as_u64())
    }

    /// Checks if a node has a specific property.
    pub fn contains(&self, node: NodeId, key: &str) -> bool {
        self.data
            .get(&node.as_u64())
            .is_some_and(|props| props.contains_key(key))
    }

    /// Checks if a node has any properties.
    pub fn has_properties(&self, node: NodeId) -> bool {
        self.data
            .get(&node.as_u64())
            .is_some_and(|props| !props.is_empty())
    }

    /// Returns the number of nodes with properties.
    pub fn node_count(&self) -> usize {
        self.data.len()
    }

    /// Returns the total number of properties across all nodes.
    pub fn property_count(&self) -> usize {
        self.data.values().map(|props| props.len()).sum()
    }

    /// Sets multiple properties for a node at once.
    pub fn set_many<I, K>(&mut self, node: NodeId, properties: I)
    where
        I: IntoIterator<Item = (K, PropertyValue)>,
        K: Into<String>,
    {
        let entry = self.data.entry(node.as_u64()).or_default();
        for (key, value) in properties {
            entry.insert(key.into(), value);
        }
    }

    /// Returns an iterator over all nodes with properties.
    pub fn nodes(&self) -> impl Iterator<Item = NodeId> + '_ {
        self.data.keys().map(|&id| NodeId::new(id))
    }

    /// Returns an iterator over all (node, key, value) tuples.
    pub fn iter(&self) -> impl Iterator<Item = (NodeId, &str, &PropertyValue)> + '_ {
        self.data.iter().flat_map(|(&node_id, props)| {
            props
                .iter()
                .map(move |(key, value)| (NodeId::new(node_id), key.as_str(), value))
        })
    }
}

// =============================================================================
// VersionedPropertyStore (MVCC)
// =============================================================================

/// MVCC-enabled property storage for Snapshot Isolation.
///
/// Instead of storing single values, this store maintains a version chain
/// for each property, allowing concurrent readers to see consistent snapshots
/// while writers add new versions.
///
/// ## Example
///
/// ```
/// use neural_core::{NodeId, PropertyValue};
/// use neural_storage::VersionedPropertyStore;
///
/// let mut store = VersionedPropertyStore::new();
///
/// // Transaction 1 sets properties
/// store.set(NodeId::new(0), "name", PropertyValue::from("Alice"), 1);
/// store.set(NodeId::new(0), "age", PropertyValue::from(30i64), 1);
///
/// // Transaction 5 updates name
/// store.set(NodeId::new(0), "name", PropertyValue::from("Alicia"), 5);
///
/// // Snapshot at tx 3 sees old value
/// assert_eq!(
///     store.get(NodeId::new(0), "name", 3),
///     Some(&PropertyValue::from("Alice"))
/// );
///
/// // Snapshot at tx 6 sees new value
/// assert_eq!(
///     store.get(NodeId::new(0), "name", 6),
///     Some(&PropertyValue::from("Alicia"))
/// );
/// ```
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct VersionedPropertyStore {
    /// Map from NodeId to property map, where each property is versioned.
    data: HashMap<u64, HashMap<String, VersionedValue>>,
}

impl VersionedPropertyStore {
    /// Creates an empty versioned property store.
    pub fn new() -> Self {
        Self::default()
    }

    /// Creates a versioned property store with initial capacity.
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            data: HashMap::with_capacity(capacity),
        }
    }

    /// Sets a property for a node at a given transaction ID.
    ///
    /// This creates a new version visible to transactions >= tx_id.
    pub fn set(&mut self, node: NodeId, key: impl Into<String>, value: PropertyValue, tx_id: TransactionId) {
        let key = key.into();
        let props = self.data.entry(node.as_u64()).or_default();

        if let Some(versioned) = props.get_mut(&key) {
            versioned.update(tx_id, value);
        } else {
            props.insert(key, VersionedValue::new(tx_id, value));
        }
    }

    /// Gets a property for a node as seen by a given snapshot.
    ///
    /// Returns the most recent version with tx_id <= snapshot_id.
    pub fn get(&self, node: NodeId, key: &str, snapshot_id: TransactionId) -> Option<&PropertyValue> {
        self.data
            .get(&node.as_u64())
            .and_then(|props| props.get(key))
            .and_then(|versioned| versioned.get(snapshot_id))
    }

    /// Gets all visible properties for a node at a given snapshot.
    ///
    /// Returns a HashMap of property key -> value for properties visible at snapshot_id.
    pub fn get_all(&self, node: NodeId, snapshot_id: TransactionId) -> HashMap<String, PropertyValue> {
        let mut result = HashMap::new();
        if let Some(props) = self.data.get(&node.as_u64()) {
            for (key, versioned) in props {
                if let Some(value) = versioned.get(snapshot_id) {
                    result.insert(key.clone(), value.clone());
                }
            }
        }
        result
    }

    /// Removes (tombstones) a property for a node at a given transaction ID.
    ///
    /// The property will appear as deleted for transactions >= tx_id.
    pub fn remove(&mut self, node: NodeId, key: &str, tx_id: TransactionId) {
        if let Some(props) = self.data.get_mut(&node.as_u64()) {
            if let Some(versioned) = props.get_mut(key) {
                versioned.delete(tx_id);
            }
        }
    }

    /// Removes (tombstones) all properties for a node at a given transaction ID.
    pub fn remove_all(&mut self, node: NodeId, tx_id: TransactionId) {
        if let Some(props) = self.data.get_mut(&node.as_u64()) {
            for versioned in props.values_mut() {
                versioned.delete(tx_id);
            }
        }
    }

    /// Checks if a node has a specific property visible at a snapshot.
    pub fn contains(&self, node: NodeId, key: &str, snapshot_id: TransactionId) -> bool {
        self.get(node, key, snapshot_id).is_some()
    }

    /// Checks if a node has any visible properties at a snapshot.
    pub fn has_properties(&self, node: NodeId, snapshot_id: TransactionId) -> bool {
        self.data
            .get(&node.as_u64())
            .is_some_and(|props| props.values().any(|v| v.get(snapshot_id).is_some()))
    }

    /// Returns the number of nodes with any property data.
    pub fn node_count(&self) -> usize {
        self.data.len()
    }

    /// Sets multiple properties for a node at once.
    pub fn set_many<I, K>(&mut self, node: NodeId, properties: I, tx_id: TransactionId)
    where
        I: IntoIterator<Item = (K, PropertyValue)>,
        K: Into<String>,
    {
        for (key, value) in properties {
            self.set(node, key, value, tx_id);
        }
    }

    /// Runs garbage collection, removing versions older than min_active_tx_id.
    ///
    /// Call this periodically with the oldest active transaction ID to reclaim space.
    pub fn vacuum(&mut self, min_active_tx_id: TransactionId) {
        for props in self.data.values_mut() {
            for versioned in props.values_mut() {
                versioned.vacuum(min_active_tx_id);
            }
        }
    }

    /// Returns an iterator over all nodes with property data.
    pub fn nodes(&self) -> impl Iterator<Item = NodeId> + '_ {
        self.data.keys().map(|&id| NodeId::new(id))
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_property_store_basic() {
        let mut store = PropertyStore::new();

        store.set(NodeId::new(0), "name", PropertyValue::from("Alice"));
        store.set(NodeId::new(0), "age", PropertyValue::from(30i64));
        store.set(NodeId::new(1), "name", PropertyValue::from("Bob"));

        assert_eq!(
            store.get(NodeId::new(0), "name"),
            Some(&PropertyValue::from("Alice"))
        );
        assert_eq!(
            store.get(NodeId::new(0), "age"),
            Some(&PropertyValue::from(30i64))
        );
        assert_eq!(
            store.get(NodeId::new(1), "name"),
            Some(&PropertyValue::from("Bob"))
        );
        assert_eq!(store.get(NodeId::new(2), "name"), None);
    }

    #[test]
    fn test_property_store_overwrite() {
        let mut store = PropertyStore::new();

        store.set(NodeId::new(0), "score", PropertyValue::from(10i64));
        store.set(NodeId::new(0), "score", PropertyValue::from(20i64));

        assert_eq!(
            store.get(NodeId::new(0), "score"),
            Some(&PropertyValue::from(20i64))
        );
    }

    #[test]
    fn test_property_store_remove() {
        let mut store = PropertyStore::new();

        store.set(NodeId::new(0), "name", PropertyValue::from("Alice"));
        store.set(NodeId::new(0), "age", PropertyValue::from(30i64));

        let removed = store.remove(NodeId::new(0), "age");
        assert_eq!(removed, Some(PropertyValue::from(30i64)));
        assert_eq!(store.get(NodeId::new(0), "age"), None);
        assert!(store.contains(NodeId::new(0), "name"));
    }

    #[test]
    fn test_property_store_contains() {
        let mut store = PropertyStore::new();

        store.set(NodeId::new(0), "name", PropertyValue::from("Alice"));

        assert!(store.contains(NodeId::new(0), "name"));
        assert!(!store.contains(NodeId::new(0), "age"));
        assert!(!store.contains(NodeId::new(1), "name"));
    }

    #[test]
    fn test_property_store_set_many() {
        let mut store = PropertyStore::new();

        store.set_many(
            NodeId::new(0),
            [
                ("name", PropertyValue::from("Alice")),
                ("age", PropertyValue::from(30i64)),
                ("active", PropertyValue::from(true)),
            ],
        );

        assert_eq!(store.get_all(NodeId::new(0)).map(|p| p.len()), Some(3));
    }

    #[test]
    fn test_property_store_counts() {
        let mut store = PropertyStore::new();

        store.set(NodeId::new(0), "a", PropertyValue::from(1i64));
        store.set(NodeId::new(0), "b", PropertyValue::from(2i64));
        store.set(NodeId::new(1), "c", PropertyValue::from(3i64));

        assert_eq!(store.node_count(), 2);
        assert_eq!(store.property_count(), 3);
    }

    #[test]
    fn test_property_store_iter() {
        let mut store = PropertyStore::new();

        store.set(NodeId::new(0), "x", PropertyValue::from(1i64));
        store.set(NodeId::new(1), "y", PropertyValue::from(2i64));

        let items: Vec<_> = store.iter().collect();
        assert_eq!(items.len(), 2);
    }

    // =========================================================================
    // VersionedPropertyStore Tests (Sprint 51 - MVCC)
    // =========================================================================

    #[test]
    fn test_versioned_store_basic() {
        let mut store = VersionedPropertyStore::new();

        // Tx 1 sets properties
        store.set(NodeId::new(0), "name", PropertyValue::from("Alice"), 1);
        store.set(NodeId::new(0), "age", PropertyValue::from(30i64), 1);

        // Snapshot at tx 1 sees values
        assert_eq!(
            store.get(NodeId::new(0), "name", 1),
            Some(&PropertyValue::from("Alice"))
        );
        assert_eq!(
            store.get(NodeId::new(0), "age", 1),
            Some(&PropertyValue::from(30i64))
        );

        // Snapshot at tx 0 sees nothing
        assert_eq!(store.get(NodeId::new(0), "name", 0), None);
    }

    #[test]
    fn test_versioned_store_update() {
        let mut store = VersionedPropertyStore::new();

        // Tx 1 sets name
        store.set(NodeId::new(0), "name", PropertyValue::from("Alice"), 1);

        // Tx 5 updates name
        store.set(NodeId::new(0), "name", PropertyValue::from("Alicia"), 5);

        // Snapshot at tx 3 sees old value
        assert_eq!(
            store.get(NodeId::new(0), "name", 3),
            Some(&PropertyValue::from("Alice"))
        );

        // Snapshot at tx 5 sees new value
        assert_eq!(
            store.get(NodeId::new(0), "name", 5),
            Some(&PropertyValue::from("Alicia"))
        );

        // Snapshot at tx 10 also sees new value
        assert_eq!(
            store.get(NodeId::new(0), "name", 10),
            Some(&PropertyValue::from("Alicia"))
        );
    }

    #[test]
    fn test_versioned_store_delete() {
        let mut store = VersionedPropertyStore::new();

        // Tx 1 sets property
        store.set(NodeId::new(0), "name", PropertyValue::from("Alice"), 1);

        // Tx 5 deletes property
        store.remove(NodeId::new(0), "name", 5);

        // Snapshot at tx 3 sees the value
        assert_eq!(
            store.get(NodeId::new(0), "name", 3),
            Some(&PropertyValue::from("Alice"))
        );

        // Snapshot at tx 5+ sees deletion
        assert_eq!(store.get(NodeId::new(0), "name", 5), None);
        assert_eq!(store.get(NodeId::new(0), "name", 10), None);
    }

    #[test]
    fn test_versioned_store_get_all() {
        let mut store = VersionedPropertyStore::new();

        // Tx 1 sets multiple properties
        store.set(NodeId::new(0), "name", PropertyValue::from("Alice"), 1);
        store.set(NodeId::new(0), "age", PropertyValue::from(30i64), 1);

        // Tx 3 updates age
        store.set(NodeId::new(0), "age", PropertyValue::from(31i64), 3);

        // Snapshot at tx 2 sees old age
        let props = store.get_all(NodeId::new(0), 2);
        assert_eq!(props.get("name"), Some(&PropertyValue::from("Alice")));
        assert_eq!(props.get("age"), Some(&PropertyValue::from(30i64)));

        // Snapshot at tx 5 sees new age
        let props = store.get_all(NodeId::new(0), 5);
        assert_eq!(props.get("age"), Some(&PropertyValue::from(31i64)));
    }

    #[test]
    fn test_versioned_store_contains() {
        let mut store = VersionedPropertyStore::new();

        store.set(NodeId::new(0), "name", PropertyValue::from("Alice"), 1);
        store.remove(NodeId::new(0), "name", 5);

        assert!(!store.contains(NodeId::new(0), "name", 0));
        assert!(store.contains(NodeId::new(0), "name", 3));
        assert!(!store.contains(NodeId::new(0), "name", 5));
    }

    #[test]
    fn test_versioned_store_has_properties() {
        let mut store = VersionedPropertyStore::new();

        store.set(NodeId::new(0), "name", PropertyValue::from("Alice"), 1);

        assert!(!store.has_properties(NodeId::new(0), 0));
        assert!(store.has_properties(NodeId::new(0), 1));
        assert!(!store.has_properties(NodeId::new(1), 1)); // Different node
    }

    #[test]
    fn test_versioned_store_set_many() {
        let mut store = VersionedPropertyStore::new();

        store.set_many(
            NodeId::new(0),
            [
                ("name", PropertyValue::from("Alice")),
                ("age", PropertyValue::from(30i64)),
            ],
            1,
        );

        let props = store.get_all(NodeId::new(0), 1);
        assert_eq!(props.len(), 2);
    }

    #[test]
    fn test_versioned_store_remove_all() {
        let mut store = VersionedPropertyStore::new();

        store.set(NodeId::new(0), "name", PropertyValue::from("Alice"), 1);
        store.set(NodeId::new(0), "age", PropertyValue::from(30i64), 1);

        // Delete all at tx 5
        store.remove_all(NodeId::new(0), 5);

        // Snapshot at tx 3 sees both
        assert!(store.has_properties(NodeId::new(0), 3));

        // Snapshot at tx 5 sees nothing
        assert!(!store.has_properties(NodeId::new(0), 5));
    }
}
