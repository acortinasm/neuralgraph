//! Graph store combining structure and properties.
//!
//! This module provides a unified interface for accessing both
//! graph structure (edges) and node properties.

use crate::{CsrMatrix, CsrStats, VersionedPropertyStore, VectorIndex};
use crate::csc::CscMatrix;
use crate::wal::{LogEntry, TransactionId, WalWriter};
use crate::transaction::TransactionManager;
use neural_core::{Edge, Graph, Label, NodeId, PropertyValue};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::{Path, PathBuf};

// =============================================================================
// LabelIndex - Inverted index for fast label lookups
// =============================================================================

/// Inverted index for O(1) label lookups.
///
/// Instead of scanning all nodes to find those with a specific label,
/// we maintain a mapping from label -> sorted Vec<NodeId>.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct LabelIndex {
    /// label -> sorted Vec<NodeId>
    index: HashMap<String, Vec<NodeId>>,
}

impl LabelIndex {
    /// Creates a new empty label index.
    pub fn new() -> Self {
        Self::default()
    }

    /// Adds a node to a label's index.
    pub fn add(&mut self, node: NodeId, label: &str) {
        self.index.entry(label.to_string()).or_default().push(node);
    }

    /// Gets all nodes with a label - O(1) lookup.
    pub fn get(&self, label: &str) -> Option<&[NodeId]> {
        self.index.get(label).map(|v| v.as_slice())
    }

    /// Checks if a node has a label - O(log n) with binary search.
    pub fn contains(&self, node: NodeId, label: &str) -> bool {
        self.index
            .get(label)
            .is_some_and(|nodes| nodes.binary_search(&node).is_ok())
    }

    /// Sorts all index vectors for binary search.
    /// Must be called after all nodes are added.
    pub fn finalize(&mut self) {
        for nodes in self.index.values_mut() {
            nodes.sort_unstable();
        }
    }

    /// Returns the count of unique labels.
    pub fn label_count(&self) -> usize {
        self.index.len()
    }

    /// Returns the count of nodes with a specific label.
    pub fn nodes_with_label_count(&self, label: &str) -> usize {
        self.index.get(label).map(|v| v.len()).unwrap_or(0)
    }

    /// Returns all label names in the index.
    pub fn labels(&self) -> impl Iterator<Item = &str> {
        self.index.keys().map(|s| s.as_str())
    }

    /// Removes a node from a label's index (Sprint 23).
    ///
    /// This removes the node from the sorted vector for the given label.
    /// Note: After removal, the vector remains sorted.
    pub fn remove(&mut self, node: NodeId, label: &str) {
        if let Some(nodes) = self.index.get_mut(label) {
            if let Ok(pos) = nodes.binary_search(&node) {
                nodes.remove(pos);
            }
            // Clean up empty label entries
            if nodes.is_empty() {
                self.index.remove(label);
            }
        }
    }
}

// =============================================================================
// PropertyIndex - Inverted index for property value lookups
// =============================================================================

/// Inverted index for O(1) property value lookups.
///
/// Enables fast queries like `WHERE n.category = "cs.LG"` by mapping
/// (property_name, value) -> sorted Vec<NodeId>.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct PropertyIndex {
    /// (property_name, value_hash) -> sorted Vec<NodeId>
    /// We use String representation of values for simplicity
    index: HashMap<String, HashMap<String, Vec<NodeId>>>,
}

impl PropertyIndex {
    /// Creates a new empty property index.
    pub fn new() -> Self {
        Self::default()
    }

    /// Adds a node's property to the index.
    pub fn add(&mut self, node: NodeId, key: &str, value: &PropertyValue) {
        let value_key = value_to_key(value);
        self.index
            .entry(key.to_string())
            .or_default()
            .entry(value_key)
            .or_default()
            .push(node);
    }

    /// Gets all nodes with a specific property value - O(1) lookup.
    pub fn get(&self, key: &str, value: &PropertyValue) -> Option<&[NodeId]> {
        let value_key = value_to_key(value);
        self.index
            .get(key)
            .and_then(|values| values.get(&value_key))
            .map(|v| v.as_slice())
    }

    /// Checks if a node has a specific property value - O(log n).
    pub fn contains(&self, node: NodeId, key: &str, value: &PropertyValue) -> bool {
        self.get(key, value)
            .is_some_and(|nodes| nodes.binary_search(&node).is_ok())
    }

    /// Sorts all index vectors for binary search.
    pub fn finalize(&mut self) {
        for values in self.index.values_mut() {
            for nodes in values.values_mut() {
                nodes.sort_unstable();
            }
        }
    }

    /// Returns the count of indexed properties.
    pub fn property_count(&self) -> usize {
        self.index.len()
    }

    /// Returns all indexed property names.
    pub fn properties(&self) -> impl Iterator<Item = &str> {
        self.index.keys().map(|s| s.as_str())
    }

    /// Removes a node from a property value's index (Sprint 23).
    pub fn remove(&mut self, node: NodeId, key: &str, value: &PropertyValue) {
        let value_key = value_to_key(value);
        if let Some(values) = self.index.get_mut(key) {
            if let Some(nodes) = values.get_mut(&value_key) {
                if let Ok(pos) = nodes.binary_search(&node) {
                    nodes.remove(pos);
                }
                // Clean up empty value entries
                if nodes.is_empty() {
                    values.remove(&value_key);
                }
            }
            // Clean up empty property entries
            if values.is_empty() {
                self.index.remove(key);
            }
        }
    }

    /// Removes all property index entries for a node (Sprint 23).
    ///
    /// This is used when deleting a node - we need to remove it from
    /// all property indices. This is O(properties * values) in worst case.
    pub fn remove_all_for_node(&mut self, node: NodeId) {
        for values in self.index.values_mut() {
            for nodes in values.values_mut() {
                if let Ok(pos) = nodes.binary_search(&node) {
                    nodes.remove(pos);
                }
            }
            // Clean up empty value entries
            values.retain(|_, nodes| !nodes.is_empty());
        }
        // Clean up empty property entries
        self.index.retain(|_, values| !values.is_empty());
    }
}

/// Converts a PropertyValue to a string key for indexing.
fn value_to_key(value: &PropertyValue) -> String {
    match value {
        PropertyValue::Null => "null".to_string(),
        PropertyValue::Bool(b) => format!("b:{}", b),
        PropertyValue::Int(i) => format!("i:{}", i),
        PropertyValue::Float(f) => format!("f:{:.10}", f),
        PropertyValue::String(s) => format!("\"{}\"", s),
        PropertyValue::Date(s) => s.clone(),
        PropertyValue::DateTime(s) => s.clone(),
        PropertyValue::Vector(v) => format!("v:[{}]", v.len()),
    }
}

// =============================================================================
// EdgeTypeIndex - Index for edge types (labels)
// =============================================================================

/// Index for O(1) edge type lookups.
///
/// Enables fast queries like `MATCH ()-[:CITES]->()` by storing
/// separate edge lists per type.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct EdgeTypeIndex {
    /// edge_type -> Vec<(edge_id, source, target)>
    index: HashMap<String, Vec<(neural_core::EdgeId, NodeId, NodeId)>>,
    /// Total edges indexed
    total_edges: usize,
}

impl EdgeTypeIndex {
    /// Creates a new empty edge type index.
    pub fn new() -> Self {
        Self::default()
    }

    /// Adds an edge with a type to the index.
    pub fn add(&mut self, edge_id: neural_core::EdgeId, source: NodeId, target: NodeId, edge_type: &str) {
        self.index
            .entry(edge_type.to_string())
            .or_default()
            .push((edge_id, source, target));
        self.total_edges += 1;
    }

    /// Gets all edges of a specific type - O(1) lookup.
    pub fn get(&self, edge_type: &str) -> Option<&[(neural_core::EdgeId, NodeId, NodeId)]> {
        self.index.get(edge_type).map(|v| v.as_slice())
    }

    /// Gets edges from a specific source node with a type.
    pub fn edges_from(&self, source: NodeId, edge_type: &str) -> impl Iterator<Item = (neural_core::EdgeId, NodeId)> + '_ {
        self.index
            .get(edge_type)
            .into_iter()
            .flatten()
            .filter(move |(_, s, _)| *s == source)
            .map(|(eid, _, t)| (*eid, *t))
    }

    /// Returns the count of edges with a specific type.
    pub fn edges_with_type_count(&self, edge_type: &str) -> usize {
        self.index.get(edge_type).map(|v| v.len()).unwrap_or(0)
    }

    /// Returns all edge type names.
    pub fn edge_types(&self) -> impl Iterator<Item = &str> {
        self.index.keys().map(|s| s.as_str())
    }

    /// Returns the count of unique edge types.
    pub fn type_count(&self) -> usize {
        self.index.len()
    }

    /// Returns total edges indexed.
    pub fn total_edges(&self) -> usize {
        self.total_edges
    }

    /// Finds the type name for a given edge ID.
    pub fn get_type_by_id(&self, edge_id: neural_core::EdgeId) -> Option<&str> {
        for (etype, edges) in &self.index {
            if edges.iter().any(|(id, _, _)| *id == edge_id) {
                return Some(etype);
            }
        }
        None
    }

    /// Removes an edge from the type index (Sprint 23).
    pub fn remove(&mut self, source: NodeId, target: NodeId, edge_type: &str) {
        if let Some(edges) = self.index.get_mut(edge_type) {
            if let Some(pos) = edges.iter().position(|(_, s, t)| *s == source && *t == target) {
                edges.remove(pos);
                self.total_edges = self.total_edges.saturating_sub(1);
            }
            // Clean up empty type entries
            if edges.is_empty() {
                self.index.remove(edge_type);
            }
        }
    }

    /// Removes all edges incident to a node (Sprint 23).
    ///
    /// Removes all edges where the node is source OR target.
    /// Returns the count of edges removed.
    pub fn remove_incident(&mut self, node: NodeId) -> usize {
        let mut removed = 0;
        for edges in self.index.values_mut() {
            let original_len = edges.len();
            edges.retain(|(_, s, t)| *s != node && *t != node);
            removed += original_len - edges.len();
        }
        self.total_edges = self.total_edges.saturating_sub(removed);
        // Clean up empty type entries
        self.index.retain(|_, edges| !edges.is_empty());
        removed
    }
}

// =============================================================================
// GraphStore
// =============================================================================

/// Unified storage for graph structure and properties.
///
/// `GraphStore` combines:
/// - Graph structure (adjacency) in CSR format
/// - Node properties in a hash map
///
/// ## Example
///
/// ```
/// use neural_core::{NodeId, PropertyValue, Graph};
/// use neural_storage::GraphStore;
///
/// let store = GraphStore::builder()
///     .add_node(0u64, [("name", "Alice"), ("age", "30")])
///     .add_node(1u64, [("name", "Bob")])
///     .add_edge(0u64, 1u64)
///     .build();
///
/// // Access properties
/// assert_eq!(
///     store.get_property(NodeId::new(0), "name"),
///     Some(&PropertyValue::from("Alice"))
/// );
///
/// // Access structure
/// let neighbors: Vec<_> = store.neighbors(NodeId::new(0)).collect();
/// assert_eq!(neighbors.len(), 1);
/// ```
/// The maximum transaction ID, used as "current" snapshot for non-transactional reads.
pub const MAX_SNAPSHOT_ID: TransactionId = u64::MAX;

#[derive(Debug, Serialize, Deserialize)]
pub struct GraphStore {
    /// Graph structure (adjacency) - CSR for outgoing edges
    graph: CsrMatrix,
    /// Reverse graph structure (CSC) for O(1) incoming edge lookups
    /// This is built from CSR and enables fast incoming neighbor queries.
    #[serde(skip, default)]
    reverse_graph: CscMatrix,
    /// Node properties (MVCC-versioned for snapshot isolation)
    versioned_properties: VersionedPropertyStore,
    /// Node labels (MVCC-versioned)
    versioned_labels: VersionedPropertyStore,
    /// Inverted index for O(1) label lookups
    label_index: LabelIndex,
    /// Inverted index for O(1) property value lookups
    property_index: PropertyIndex,
    /// Index for O(1) edge type lookups
    edge_type_index: EdgeTypeIndex,
    /// Vector index for similarity search (Sprint 13)
    /// Skipped during serialization - must be rebuilt after load
    #[serde(skip)]
    vector_index: Option<VectorIndex>,
    /// Number of nodes created dynamically via CREATE (Sprint 21)
    /// These nodes exist in indices but not in the CSR structure
    #[serde(default)]
    dynamic_node_count: usize,
    /// Edges created dynamically via CREATE (Sprint 22)
    /// Format: (source, target, optional_label)
    #[serde(default)]
    dynamic_edges: Vec<(NodeId, NodeId, Option<String>)>,
    /// Path to the main DB file (for WAL)
    pub path: Option<PathBuf>,
    /// Write-Ahead Log for durability (Sprint 26)
    #[serde(skip)]
    pub wal: Option<WalWriter>,
    /// Transaction Manager for ACID support (Sprint 50)
    #[serde(skip, default)]
    pub transaction_manager: TransactionManager,
    /// Current transaction ID counter (monotonically increasing)
    #[serde(default = "default_tx_counter")]
    current_tx_id: TransactionId,
}

fn default_tx_counter() -> TransactionId {
    1 // Start at 1 so that 0 means "nothing visible"
}

impl GraphStore {
    /// Returns statistics about the graph store.
    pub fn stats(&self) -> CsrStats {
        self.graph.stats()
    }

    /// Validates the internal consistency of the graph store.
    pub fn validate(&self) -> Result<(), String> {
        self.graph.validate()
    }

    /// Creates a new empty graph store.
    pub fn new_in_memory() -> Self {
        Self {
            graph: CsrMatrix::empty(),
            reverse_graph: CscMatrix::empty(),
            versioned_properties: VersionedPropertyStore::new(),
            versioned_labels: VersionedPropertyStore::new(),
            label_index: LabelIndex::new(),
            property_index: PropertyIndex::new(),
            edge_type_index: EdgeTypeIndex::new(),
            vector_index: None,
            dynamic_node_count: 0,
            dynamic_edges: Vec::new(),
            path: None,
            wal: None,
            transaction_manager: TransactionManager::new(),
            current_tx_id: 1,
        }
    }

    /// Creates a new graph store from a CSR matrix, for in-memory use without a WAL.
    pub fn new_from_csr(graph: CsrMatrix) -> Self {
        let reverse_graph = CscMatrix::from_csr(&graph);
        Self {
            graph,
            reverse_graph,
            versioned_properties: VersionedPropertyStore::new(),
            versioned_labels: VersionedPropertyStore::new(),
            label_index: LabelIndex::new(),
            property_index: PropertyIndex::new(),
            edge_type_index: EdgeTypeIndex::new(),
            vector_index: None,
            dynamic_node_count: 0,
            dynamic_edges: Vec::new(),
            path: None,
            wal: None,
            transaction_manager: TransactionManager::new(),
            current_tx_id: 1,
        }
    }

    /// Opens a graph store from a path, enabling the WAL.
    pub fn new(path: impl AsRef<Path>) -> Result<Self, crate::wal::WalError> {
        let path = path.as_ref();
        let wal_path = path.with_extension("wal");

        // Create an empty store and then load from disk
        let mut store = Self {
            graph: CsrMatrix::empty(),
            reverse_graph: CscMatrix::empty(),
            versioned_properties: VersionedPropertyStore::new(),
            versioned_labels: VersionedPropertyStore::new(),
            label_index: LabelIndex::new(),
            property_index: PropertyIndex::new(),
            edge_type_index: EdgeTypeIndex::new(),
            vector_index: None,
            dynamic_node_count: 0,
            dynamic_edges: Vec::new(),
            path: Some(path.to_path_buf()),
            wal: None, // Will be initialized after recovery
            transaction_manager: TransactionManager::new(),
            current_tx_id: 1,
        };
        
        // Attempt to open WAL for recovery
        if wal_path.exists() {
            store.recover_from_wal(&wal_path)?;
        }

        // After recovery, initialize a new WAL writer for ongoing operations
        store.wal = Some(WalWriter::new(wal_path)?);

        Ok(store)
    }

    /// Recovers the graph state by replaying operations from the WAL file.
    /// This should be called on startup, before any new mutations.
    fn recover_from_wal(&mut self, wal_path: &Path) -> Result<(), crate::wal::WalError> {
        use std::io::{BufReader, Read};
        use bincode;

        let file = std::fs::File::open(wal_path)?;
        let mut reader = BufReader::new(file);

        loop {
            let mut len_bytes = [0u8; 8];
            match reader.read_exact(&mut len_bytes) {
                Ok(_) => {
                    let len = u64::from_le_bytes(len_bytes);
                    let mut buffer = vec![0; len as usize];
                    reader.read_exact(&mut buffer)?;
                    let entry: LogEntry = bincode::deserialize(&buffer)?;

                    // Replay the log entry
                    if let Err(e) = self.apply_log_entry(&entry) {
                        // In recovery, an error is critical.
                        // However, for backward compatibility or corruption tolerance, 
                        // we might log and continue. For now, panic/fail.
                        return Err(std::io::Error::new(std::io::ErrorKind::InvalidData, e).into());
                    }
                }
                Err(e) if e.kind() == std::io::ErrorKind::UnexpectedEof => {
                    // Reached end of file cleanly
                    break;
                }
                Err(e) => return Err(e.into()), // Other I/O errors
            }
        }

        // WAL has been successfully replayed. We can truncate it now.
        // This means the recovered state is the new baseline.
        std::fs::File::create(wal_path)?.set_len(0)?;
        
        Ok(())
    }

    /// Applies a log entry to the in-memory graph state.
    ///
    /// This is used by:
    /// 1. WAL Recovery (replaying history)
    /// 2. Active Mutations (applying new changes)
    /// 3. Transactions (applying buffered changes on commit)
    ///
    /// For MVCC, writes use `current_tx_id` which is incremented after each mutation.
    pub fn apply_log_entry(&mut self, entry: &LogEntry) -> Result<(), String> {
        match entry {
            LogEntry::BeginTransaction { tx_id } => {
                // Update current_tx_id to track the highest seen transaction
                self.current_tx_id = self.current_tx_id.max(*tx_id);
                Ok(())
            }
            LogEntry::CommitTransaction { tx_id } => {
                // Committed transaction - ensure tx_id is visible
                self.current_tx_id = self.current_tx_id.max(*tx_id + 1);
                Ok(())
            }
            LogEntry::RollbackTransaction { .. } => {
                // Rollback doesn't affect committed state
                Ok(())
            }
            LogEntry::CreateNode {
                node_id,
                label,
                properties,
            } => {
                let tx_id = self.current_tx_id;

                // Ensure dynamic_node_count reflects the highest node_id
                let new_count = node_id.as_usize() + 1;
                if new_count > self.node_count() {
                     let base_count = self.graph.node_count();
                     if new_count > base_count {
                         self.dynamic_node_count = self.dynamic_node_count.max(new_count - base_count);
                     }
                }

                if let Some(label_str) = label {
                    self.versioned_labels.set(*node_id, "_label", PropertyValue::from(label_str.clone()), tx_id);
                    self.label_index.add(*node_id, label_str);
                }
                for (key, value) in properties {
                    self.property_index.add(*node_id, key, value);
                    self.versioned_properties.set(*node_id, key, value.clone(), tx_id);
                }

                // Increment tx_id for next operation
                self.current_tx_id += 1;
                Ok(())
            }
            LogEntry::CreateEdge {
                source,
                target,
                edge_type,
            } => {
                // For recovery/application, we just append to dynamic_edges.
                self.dynamic_edges.push((*source, *target, edge_type.clone()));

                // We need the edge ID for the index.
                // The ID of the newly added edge is (total_edges - 1).
                let total_edges = self.graph.edge_count() + self.dynamic_edges.len();
                let edge_id = neural_core::EdgeId::new((total_edges - 1) as u64);

                if let Some(label_str) = edge_type {
                    self.edge_type_index.add(edge_id, *source, *target, label_str);
                }
                Ok(())
            }
            LogEntry::SetProperty {
                node_id,
                key,
                value,
            } => {
                let tx_id = self.current_tx_id;

                // Remove old value from property index (using current snapshot)
                if let Some(old_value) = self.versioned_properties.get(*node_id, key, MAX_SNAPSHOT_ID).cloned() {
                    self.property_index.remove(*node_id, key, &old_value);
                }
                // Add new value to index and set versioned property
                self.property_index.add(*node_id, key, value);
                self.versioned_properties.set(*node_id, key, value.clone(), tx_id);

                self.current_tx_id += 1;
                Ok(())
            }
            LogEntry::DeleteNode { node_id } => {
                let tx_id = self.current_tx_id;

                // Get label using current snapshot for index removal
                if let Some(label) = self.get_label_at(*node_id, MAX_SNAPSHOT_ID).map(|s| s.to_string()) {
                    self.label_index.remove(*node_id, &label);
                }
                self.versioned_labels.remove(*node_id, "_label", tx_id);
                self.property_index.remove_all_for_node(*node_id);
                self.versioned_properties.remove_all(*node_id, tx_id);
                self.edge_type_index.remove_incident(*node_id);
                self.dynamic_edges.retain(|(s, t, _)| *s != *node_id && *t != *node_id);

                self.current_tx_id += 1;
                Ok(())
            }
            LogEntry::Blank => Ok(()),
            LogEntry::Membership { .. } => Ok(()),
        }
    }
    
    /// Creates a new graph store builder.
    pub fn builder() -> GraphStoreBuilder {
        GraphStoreBuilder::new()
    }

    /// Returns a reference to the underlying graph structure.
    pub fn graph(&self) -> &CsrMatrix {
        &self.graph
    }

    /// Returns a reference to the reverse graph (CSC) for incoming edge lookups.
    pub fn reverse_graph(&self) -> &CscMatrix {
        &self.reverse_graph
    }

    /// Returns the in-degree of a node - O(1) using CSC index.
    pub fn in_degree(&self, node: NodeId) -> usize {
        // CSC provides O(1) in-degree lookup
        let csc_degree = self.reverse_graph.in_degree(node);
        // Add dynamic incoming edges
        let dynamic_degree = self.dynamic_edges.iter()
            .filter(|(_, target, _)| *target == node)
            .count();
        csc_degree + dynamic_degree
    }

    /// Returns a reference to the versioned property store.
    pub fn versioned_properties(&self) -> &VersionedPropertyStore {
        &self.versioned_properties
    }

    /// Returns a reference to the label index.
    pub fn label_index(&self) -> &LabelIndex {
        &self.label_index
    }

    /// Returns the current transaction ID (for snapshot reads).
    pub fn current_snapshot_id(&self) -> TransactionId {
        self.current_tx_id
    }

    /// Gets a property for a node (using current snapshot - sees all committed data).
    pub fn get_property(&self, node: NodeId, key: &str) -> Option<&PropertyValue> {
        self.versioned_properties.get(node, key, MAX_SNAPSHOT_ID)
    }

    /// Gets a property for a node at a specific snapshot.
    pub fn get_property_at(&self, node: NodeId, key: &str, snapshot_id: TransactionId) -> Option<&PropertyValue> {
        self.versioned_properties.get(node, key, snapshot_id)
    }

    /// Gets all properties for a node at a specific snapshot.
    pub fn get_all_properties_at(&self, node: NodeId, snapshot_id: TransactionId) -> HashMap<String, PropertyValue> {
        self.versioned_properties.get_all(node, snapshot_id)
    }

    /// Sets a property for a node (auto-commit with new tx_id).
    pub fn set_property(&mut self, node: NodeId, key: &str, value: PropertyValue) {
        let tx_id = self.current_tx_id;
        self.versioned_properties.set(node, key, value, tx_id);
        self.current_tx_id += 1;
    }

    /// Sets a property for a node with a specific transaction ID.
    pub fn set_property_at(&mut self, node: NodeId, key: &str, value: PropertyValue, tx_id: TransactionId) {
        self.versioned_properties.set(node, key, value, tx_id);
        self.current_tx_id = self.current_tx_id.max(tx_id + 1);
    }

    /// Gets the label for a node (using current snapshot).
    pub fn get_label(&self, node: NodeId) -> Option<&str> {
        self.versioned_labels.get(node, "_label", MAX_SNAPSHOT_ID).and_then(|v| v.as_str())
    }

    /// Gets the label for a node at a specific snapshot.
    pub fn get_label_at(&self, node: NodeId, snapshot_id: TransactionId) -> Option<&str> {
        self.versioned_labels.get(node, "_label", snapshot_id).and_then(|v| v.as_str())
    }

    /// Sets the label for a node (auto-commit with new tx_id).
    /// Note: Also updates the label index.
    pub fn set_label(&mut self, node: NodeId, label: &str) {
        let tx_id = self.current_tx_id;
        self.versioned_labels.set(node, "_label", PropertyValue::from(label), tx_id);
        self.label_index.add(node, label);
        self.current_tx_id += 1;
    }

    /// Sets the label for a node with a specific transaction ID.
    pub fn set_label_at(&mut self, node: NodeId, label: &str, tx_id: TransactionId) {
        self.versioned_labels.set(node, "_label", PropertyValue::from(label), tx_id);
        self.label_index.add(node, label);
        self.current_tx_id = self.current_tx_id.max(tx_id + 1);
    }

    /// Checks if a node has a specific label.
    /// Uses the label index for O(log n) lookup.
    pub fn has_label(&self, node: NodeId, label: &str) -> bool {
        self.label_index.contains(node, label)
    }

    /// Returns nodes with a specific label.
    /// Uses the label index for O(1) lookup instead of O(n) scan.
    pub fn nodes_with_label<'a>(&'a self, label: &'a str) -> impl Iterator<Item = NodeId> + 'a {
        self.label_index.get(label).into_iter().flatten().copied()
    }

    /// Returns the count of nodes with a specific label.
    pub fn nodes_with_label_count(&self, label: &str) -> usize {
        self.label_index.nodes_with_label_count(label)
    }

    /// Returns all unique labels in the graph.
    pub fn labels(&self) -> impl Iterator<Item = &str> {
        self.label_index.labels()
    }

    // =========================================================================
    // Property Index Methods (Sprint 10)
    // =========================================================================

    /// Returns a reference to the property index.
    pub fn property_index(&self) -> &PropertyIndex {
        &self.property_index
    }

    /// Returns nodes with a specific property value - O(1) lookup.
    pub fn nodes_with_property<'a>(
        &'a self,
        key: &'a str,
        value: &'a PropertyValue,
    ) -> impl Iterator<Item = NodeId> + 'a {
        self.property_index
            .get(key, value)
            .into_iter()
            .flatten()
            .copied()
    }

    /// Checks if a node has a specific property value - O(log n).
    pub fn has_property_value(&self, node: NodeId, key: &str, value: &PropertyValue) -> bool {
        self.property_index.contains(node, key, value)
    }

    // =========================================================================
    // Edge Type Index Methods (Sprint 11)
    // =========================================================================

    /// Returns a reference to the edge type index.
    pub fn edge_type_index(&self) -> &EdgeTypeIndex {
        &self.edge_type_index
    }

    /// Returns all edges of a specific type - O(1) lookup.
    pub fn edges_with_type(&self, edge_type: &str) -> impl Iterator<Item = (NodeId, NodeId)> + '_ {
        self.edge_type_index
            .get(edge_type)
            .into_iter()
            .flatten()
            .map(|(_, s, t)| (*s, *t))
    }

    /// Returns neighbors through edges of a specific type.
    pub fn neighbors_via_type(
        &self,
        node: NodeId,
        edge_type: &str,
    ) -> impl Iterator<Item = NodeId> + '_ {
        self.edge_type_index.edges_from(node, edge_type).map(|(_, t)| t)
    }

    /// Returns neighbors (with IDs) through edges of a specific type.
    pub fn neighbors_via_type_with_ids(
        &self,
        node: NodeId,
        edge_type: &str,
    ) -> impl Iterator<Item = (neural_core::EdgeId, NodeId)> + '_ {
        self.edge_type_index.edges_from(node, edge_type)
    }

    /// Returns the count of edges with a specific type.
    pub fn edges_with_type_count(&self, edge_type: &str) -> usize {
        self.edge_type_index.edges_with_type_count(edge_type)
    }

    /// Returns all unique edge types in the graph.
    pub fn edge_types(&self) -> impl Iterator<Item = &str> {
        self.edge_type_index.edge_types()
    }

    // =========================================================================
    // Vector Index Methods (Sprint 13)
    // =========================================================================

    /// Returns a reference to the vector index, if initialized.
    pub fn vector_index(&self) -> Option<&VectorIndex> {
        self.vector_index.as_ref()
    }

    /// Initializes the vector index with the given dimension.
    /// Must be called before adding vectors.
    pub fn init_vector_index(&mut self, dimension: usize) {
        self.vector_index = Some(VectorIndex::new(dimension));
    }

    /// Adds a vector embedding for a node.
    /// Panics if vector index is not initialized.
    pub fn add_vector(&mut self, node: NodeId, vector: &[f32]) {
        let index = self
            .vector_index
            .as_mut()
            .expect("Vector index not initialized. Call init_vector_index first.");
        index.add(node, vector);
    }

    /// Searches for the k most similar vectors.
    /// Returns (NodeId, similarity) pairs sorted by decreasing similarity.
    pub fn vector_search(&self, query: &[f32], k: usize) -> Vec<(NodeId, f32)> {
        self.vector_index
            .as_ref()
            .map(|idx| idx.search(query, k))
            .unwrap_or_default()
    }

    /// Searches for the k most similar vectors, filtered by a label.
    pub fn vector_search_filtered(&self, query: &[f32], k: usize, label: &str) -> Vec<(NodeId, f32)> {
        if let Some(idx) = &self.vector_index {
            // Get nodes with label
            if let Some(nodes) = self.label_index.get(label) {
                // Use filtered search on HNSW
                idx.search_filtered(query, k, |node| nodes.contains(&node))
            } else {
                Vec::new()
            }
        } else {
            Vec::new()
        }
    }

    /// Gets the similarity between a node's vector and a query vector.
    /// Returns None if the node has no vector or vector index is not initialized.
    pub fn vector_similarity(&self, node: NodeId, query: &[f32]) -> Option<f32> {
        let index = self.vector_index.as_ref()?;
        if !index.contains(node) {
            return None;
        }
        // Search for just this node and get its similarity
        // This is a workaround since HNSW doesn't have direct similarity lookup
        let results = index.search_filtered(query, 1, |n| n == node);
        results.first().map(|(_, sim)| *sim)
    }

    /// Returns true if a node has a vector in the index.
    pub fn has_vector(&self, node: NodeId) -> bool {
        self.vector_index
            .as_ref()
            .is_some_and(|idx| idx.contains(node))
    }

    /// Returns the dimension of vectors in the index, if initialized.
    pub fn vector_dimension(&self) -> Option<usize> {
        self.vector_index.as_ref().map(|idx| idx.dimension())
    }

    // =========================================================================
    // Sprint 15: Community Detection
    // =========================================================================

    /// Detects communities in the graph using the Leiden algorithm.
    ///
    /// The Leiden algorithm is an improvement over Louvain that guarantees
    /// well-connected communities and runs in O(n log n) time.
    ///
    /// # Returns
    /// A `Communities` struct mapping each node to its community ID.
    pub fn detect_communities(&self) -> crate::community::Communities {
        // Collect all edges from the CSR matrix
        let mut edges = Vec::new();
        let num_nodes = self.graph.node_count();

        for node in 0..num_nodes {
            for &neighbor in self.graph.neighbors_slice(node) {
                edges.push((node, neighbor.as_usize()));
            }
        }

        crate::community::detect_communities_leiden(&edges, num_nodes)
    }

    // =========================================================================
    // Sprint 21: Mutations (CREATE/DELETE/SET)
    // =========================================================================

    /// Returns the next available node ID.
    ///
    /// This is the current node count, which would be the ID of the next
    /// node to be created.
    pub fn next_node_id(&self) -> NodeId {
        NodeId::new((self.graph.node_count() + self.dynamic_node_count) as u64)
    }

    /// Creates a new node with an optional label and properties.
    ///
    /// If a transaction is provided, the mutation is buffered and not applied until commit.
    pub fn create_node<I>(
        &mut self,
        label: Option<&str>,
        properties: I,
        tx: Option<&mut crate::transaction::Transaction>,
    ) -> Result<NodeId, crate::wal::WalError>
    where
        I: IntoIterator<Item = (String, PropertyValue)>,
    {
        // Calculate ID based on store + pending transaction nodes
        let offset = tx.as_ref().map(|t| t.pending_node_count).unwrap_or(0);
        let node_id = NodeId::new(self.next_node_id().as_u64() + offset as u64);
        
        let props: Vec<_> = properties.into_iter().collect();

        // Construct LogEntry
        let entry = LogEntry::CreateNode {
            node_id,
            label: label.map(|s| s.to_string()),
            properties: props,
        };
        
        if let Some(t) = tx {
            // Buffer
            t.buffer_entry(entry).expect("Failed to buffer entry");
            t.pending_node_count += 1;
        } else {
            // Auto-commit
            // 1. Write to WAL
            if let Some(wal) = &mut self.wal {
                wal.log(&entry)?;
            }
            // 2. Apply to Memory
            self.apply_log_entry(&entry).expect("Failed to apply CreateNode entry");
        }

        Ok(node_id)
    }

    /// Returns an iterator over (EdgeId, NodeId) pairs for outgoing neighbors.
    pub fn neighbors_with_ids(&self, node: NodeId) -> impl Iterator<Item = (neural_core::EdgeId, NodeId)> + '_ {
        let u = node.as_usize();
        let csr_edges = if u < self.graph.num_nodes {
            let start = self.graph.row_ptr[u];
            let end = self.graph.row_ptr[u + 1];
            (start..end).map(|i| {
                (neural_core::EdgeId::new(i as u64), self.graph.col_indices[i])
            }).collect::<Vec<_>>()
        } else {
            Vec::new()
        };

        let csr_count = self.graph.edge_count();
        let dynamic = self.dynamic_edges.iter().enumerate()
            .filter(move |(_, (src, _, _))| *src == node)
            .map(move |(i, (_, tgt, _))| {
                (neural_core::EdgeId::new((csr_count + i) as u64), *tgt)
            });

        csr_edges.into_iter().chain(dynamic)
    }

    /// Returns an iterator over (EdgeId, NodeId) pairs for incoming neighbors.
    ///
    /// Uses CSC matrix for O(degree) lookups on static edges (vs O(E) full scan).
    /// Dynamic edges are still scanned linearly but typically small in number.
    pub fn incoming_neighbors_with_ids(&self, node: NodeId) -> impl Iterator<Item = (neural_core::EdgeId, NodeId)> + '_ {
        // O(degree) lookup using CSC reverse graph for static edges
        let csc_incoming = self.reverse_graph.incoming_neighbors_with_ids(node)
            .map(|(eid, source)| (neural_core::EdgeId::new(eid), source));

        // Dynamic edges still need linear scan (typically small)
        let csr_count = self.graph.edge_count();
        let dynamic_incoming = self.dynamic_edges.iter().enumerate()
            .filter(move |(_, (_, target, _))| *target == node)
            .map(move |(i, (source, _, _))| {
                (neural_core::EdgeId::new((csr_count + i) as u64), *source)
            });

        csc_incoming.chain(dynamic_incoming)
    }

    /// Checks if the graph contains a node with the given ID.
    pub fn node_exists(&self, node: NodeId) -> bool {
        node.as_usize() < self.node_count()
    }

    /// Returns the next available edge ID.
    pub fn next_edge_id(&self) -> neural_core::EdgeId {
        neural_core::EdgeId::new((self.graph.edge_count() + self.dynamic_edges.len()) as u64)
    }

    /// Returns the type (label) of an edge by its ID.
    pub fn get_edge_type(&self, id: neural_core::EdgeId) -> Option<String> {
        let id_val = id.as_u64() as usize;
        let csr_count = self.graph.edge_count();

        if id_val < csr_count {
            // Static edges must be in the type index to have a type
            self.edge_type_index
                .get_type_by_id(id)
                .map(|s| s.to_string())
        } else {
            // Dynamic edges store their type directly
            let dyn_idx = id_val - csr_count;
            self.dynamic_edges.get(dyn_idx).and_then(|(_, _, t)| t.clone())
        }
    }

    /// Creates a new edge with an optional label.
    pub fn create_edge(
        &mut self,
        source: NodeId,
        target: NodeId,
        label: Option<&str>,
        tx: Option<&mut crate::transaction::Transaction>,
    ) -> Result<neural_core::EdgeId, crate::wal::WalError> {
        let offset = tx.as_ref().map(|t| t.pending_edge_count).unwrap_or(0);
        let edge_id = neural_core::EdgeId::new(self.next_edge_id().as_u64() + offset as u64);
        
        // Construct LogEntry
        let entry = LogEntry::CreateEdge {
            source,
            target,
            edge_type: label.map(|s| s.to_string()),
        };
        
        if let Some(t) = tx {
            // Buffer
            t.buffer_entry(entry).expect("Failed to buffer entry");
            t.pending_edge_count += 1;
        } else {
            // Auto-commit
            // 1. Write to WAL
            if let Some(wal) = &mut self.wal {
                wal.log(&entry)?;
            }

            // 2. Apply to Memory
            self.apply_log_entry(&entry).expect("Failed to apply CreateEdge entry");
        }

        Ok(edge_id)
    }

    /// Returns an iterator over dynamic edges (source, target, label).
    pub fn dynamic_edges(&self) -> impl Iterator<Item = &(NodeId, NodeId, Option<String>)> {
        self.dynamic_edges.iter()
    }

    // =========================================================================
    // Sprint 23: DELETE
    // =========================================================================

    /// Deletes a node and all its associated data (Sprint 23).
    ///
    /// This removes:
    /// - Node label from labels store and LabelIndex
    /// - All node properties from properties store and PropertyIndex
    /// - All incident dynamic edges (with DETACH)
    /// - Edge type index entries for deleted edges
    ///
    /// **Note:** CSR edges cannot be removed without full rebuild. This method
    /// only removes dynamic edges. Nodes in CSR are marked as deleted by removing
    /// their label/properties, making them effectively "zombie" nodes.
    ///
    /// # Arguments
    /// * `node` - Node ID to delete
    /// * `detach` - If true, delete incident edges. If false, fail if node has dynamic edges.
    ///
    /// # Returns
    /// `Ok(())` if deleted successfully, `Err` if node doesn't exist or has edges (when !detach)
    pub fn delete_node(&mut self, node: NodeId, detach: bool, tx: Option<&mut crate::transaction::Transaction>) -> Result<(), String> {
        // Check if node exists
        if !self.node_exists(node) {
            return Err(format!("Node {} does not exist", node.as_u64()));
        }

        // Check for incident dynamic edges if not detaching
        if !detach {
            let has_edges = self
                .dynamic_edges
                .iter()
                .any(|(s, t, _)| *s == node || *t == node);
            if has_edges {
                return Err(format!(
                    "Cannot delete node {} - it has incident edges. Use DETACH DELETE.",
                    node.as_u64()
                ));
            }
        }

        // Construct LogEntry
        let entry = LogEntry::DeleteNode { node_id: node };
        
        if let Some(t) = tx {
            t.buffer_entry(entry).map_err(|e| format!("{}", e))?;
        } else {
            // 1. Write to WAL
            if let Some(wal) = &mut self.wal {
                wal.log(&entry).expect("WAL write failed during delete");
            }
            // 2. Apply to Memory
            self.apply_log_entry(&entry)?;
        }
        Ok(())
    }

    /// Checks if a dynamic node was deleted (Sprint 23).
    ///
    /// A node is considered deleted if it was in the dynamic range but
    /// no longer has a label or properties (using current snapshot).
    pub fn is_deleted(&self, node: NodeId) -> bool {
        self.is_deleted_at(node, MAX_SNAPSHOT_ID)
    }

    /// Checks if a dynamic node was deleted at a specific snapshot.
    pub fn is_deleted_at(&self, node: NodeId, snapshot_id: TransactionId) -> bool {
        let in_csr = node.as_usize() < self.graph.node_count();
        if in_csr {
            // CSR nodes are never fully deleted, just emptied
            return false;
        }

        // Dynamic node - check if it has any data at this snapshot
        let has_label = self.versioned_labels.get(node, "_label", snapshot_id).is_some();
        let has_properties = self.versioned_properties.has_properties(node, snapshot_id);

        !has_label && !has_properties
    }

    // =========================================================================
    // Sprint 24: SET
    // =========================================================================

    /// Updates a property value for a node (Sprint 24).
    ///
    /// This atomically:
    /// 1. Removes the old value from PropertyIndex (if exists)
    /// 2. Sets the new value in properties store
    /// 3. Adds the new value to PropertyIndex
    ///
    /// # Arguments
    /// * `node` - Node ID to update
    /// * `key` - Property name
    /// * `value` - New property value
    ///
    /// # Returns
    /// `Ok(true)` if updated, `Ok(false)` if node doesn't exist.
    pub fn update_property(
        &mut self,
        node: NodeId,
        key: &str,
        value: PropertyValue,
        tx: Option<&mut crate::transaction::Transaction>,
    ) -> Result<bool, crate::wal::WalError> {
        if !self.node_exists(node) {
            return Ok(false);
        }

        // Construct LogEntry
        let entry = LogEntry::SetProperty {
            node_id: node,
            key: key.to_string(),
            value: value.clone(),
        };
        
        if let Some(t) = tx {
            t.buffer_entry(entry).expect("Failed to buffer entry");
        } else {
            // 1. Write to WAL
            if let Some(wal) = &mut self.wal {
                wal.log(&entry)?;
            }
            // 2. Apply to Memory
            self.apply_log_entry(&entry).expect("Failed to apply SetProperty entry");
        }

        Ok(true)
    }

    // =========================================================================
    // Sprint 27: Variable-length Paths
    // =========================================================================

    /// Finds all paths from a start node to an optional end node (or any node)
    /// within a given length range.
    ///
    /// This performs a BFS (Breadth-First Search) up to `max_length`.
    ///
    /// # Arguments
    /// * `start_node` - The starting node ID.
    /// * `end_node` - Optional target node ID. If None, returns paths to any node.
    /// * `min_length` - Minimum path length (number of edges).
    /// * `max_length` - Maximum path length (number of edges).
    /// * `edge_type` - Optional edge label to follow. If None, follows any edge.
    ///
    /// # Returns
    /// A vector of paths, where each path is a vector of NodeId.
    pub fn find_paths(
        &self,
        start_node: NodeId,
        end_node: Option<NodeId>,
        min_length: usize,
        max_length: usize,
        edge_type: Option<&str>,
    ) -> Vec<Vec<NodeId>> {
        let mut results = Vec::new();
        let mut queue = std::collections::VecDeque::new();

        // Path: (current_node, history_of_nodes)
        queue.push_back((start_node, vec![start_node]));

        while let Some((current_node, path)) = queue.pop_front() {
            let current_len = path.len() - 1; // Length in edges

            // Check if we reached the max length (shouldn't happen if we check before pushing, but safe guard)
            if current_len >= max_length {
                continue;
            }

            // Get neighbors
            // We use Box<dyn Iterator> to handle both cases uniformly
            let neighbors: Box<dyn Iterator<Item = NodeId>> = if let Some(etype) = edge_type {
                Box::new(self.neighbors_via_type(current_node, etype))
            } else {
                Box::new(self.neighbors(current_node))
            };

            for neighbor in neighbors {
                // Prevent cycles: check if neighbor is already in path
                if path.contains(&neighbor) {
                    continue;
                }

                let mut new_path = path.clone();
                new_path.push(neighbor);
                let new_len = new_path.len() - 1;

                // Check if valid result
                if new_len >= min_length {
                    if let Some(target) = end_node {
                        if neighbor == target {
                            results.push(new_path.clone());
                        }
                    } else {
                        // Any node is a valid target if end_node is None
                        results.push(new_path.clone());
                    }
                }

                // Continue BFS if we haven't reached max length
                if new_len < max_length {
                    queue.push_back((neighbor, new_path));
                }
            }
        }

        results
    }

    /// Finds the shortest path between two nodes (or all nodes if end_node is None) using BFS.
    ///
    /// # Arguments
    /// * `start_node` - The starting node ID.
    /// * `end_node` - The target node ID (optional).
    /// * `max_hops` - Maximum number of hops to search.
    /// * `edge_type` - Optional edge label to follow.
    ///
    /// # Returns
    /// A vector of paths. If end_node is Some, contains at most 1 path.
    pub fn find_shortest_path(
        &self,
        start_node: NodeId,
        end_node: Option<NodeId>,
        max_hops: usize,
        edge_type: Option<&str>,
    ) -> Vec<Vec<NodeId>> {
        let mut queue = std::collections::VecDeque::new();
        let mut visited = std::collections::HashSet::new();
        let mut results = Vec::new();

        queue.push_back((start_node, vec![start_node]));
        visited.insert(start_node);

        while let Some((current_node, path)) = queue.pop_front() {
            if let Some(target) = end_node {
                if current_node == target {
                    return vec![path];
                }
            } else if path.len() > 1 {
                 // If searching for all, record path to this node
                 results.push(path.clone());
            }

            // Check if we reached the max hops
            // path.len() is number of nodes. edges = nodes - 1.
            if path.len() > max_hops {
                continue;
            }

            let neighbors: Box<dyn Iterator<Item = NodeId>> = if let Some(etype) = edge_type {
                Box::new(self.neighbors_via_type(current_node, etype))
            } else {
                Box::new(self.neighbors(current_node))
            };

            for neighbor in neighbors {
                if !visited.contains(&neighbor) {
                    visited.insert(neighbor);
                    let mut new_path = path.clone();
                    new_path.push(neighbor);
                    queue.push_back((neighbor, new_path));
                }
            }
        }

        results
    }
}

// Delegate Graph trait to inner CsrMatrix
impl Graph for GraphStore {
    fn node_count(&self) -> usize {
        // Include both CSR nodes and dynamically created nodes
        self.graph.node_count() + self.dynamic_node_count
    }

    fn edge_count(&self) -> usize {
        // Include both CSR edges and dynamically created edges
        self.graph.edge_count() + self.dynamic_edges.len()
    }

    #[allow(refining_impl_trait)]
    fn neighbors(&self, node: NodeId) -> impl Iterator<Item = NodeId> + '_ {
        // Combine CSR neighbors with dynamic edges from this node
        let csr_neighbors = self.graph.neighbors(node);
        let dynamic_neighbors = self
            .dynamic_edges
            .iter()
            .filter(move |(src, _, _)| *src == node)
            .map(|(_, tgt, _)| *tgt);

        csr_neighbors.chain(dynamic_neighbors)
    }

    fn out_degree(&self, node: NodeId) -> usize {
        // CSR out degree + dynamic edges from this node
        let csr_degree = self.graph.out_degree(node);
        let dynamic_degree = self
            .dynamic_edges
            .iter()
            .filter(|(src, _, _)| *src == node)
            .count();

        csr_degree + dynamic_degree
    }

    fn contains_node(&self, node: NodeId) -> bool {
        // Check CSR or if it's a dynamic node
        self.graph.contains_node(node) || node.as_usize() < self.node_count()
    }

    fn has_edge(&self, source: NodeId, target: NodeId) -> bool {
        // Check CSR first, then dynamic edges
        self.graph.has_edge(source, target)
            || self
                .dynamic_edges
                .iter()
                .any(|(s, t, _)| *s == source && *t == target)
    }
}

// =============================================================================
// GraphStoreBuilder
// =============================================================================

/// Builder for constructing GraphStore instances.
#[derive(Debug, Default)]
pub struct GraphStoreBuilder {
    edges: Vec<Edge>,
    node_properties: Vec<(u64, String, PropertyValue)>,
    node_labels: Vec<(u64, String)>,
    max_node_id: u64,
}

impl GraphStoreBuilder {
    /// Creates a new builder.
    pub fn new() -> Self {
        Self::default()
    }

    /// Creates a new builder with pre-allocated capacity for edges.
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            edges: Vec::with_capacity(capacity),
            node_properties: Vec::new(),
            node_labels: Vec::new(),
            max_node_id: 0,
        }
    }

    /// Adds a node with properties.
    pub fn add_node<I, K, V>(mut self, id: impl Into<u64>, properties: I) -> Self
    where
        I: IntoIterator<Item = (K, V)>,
        K: Into<String>,
        V: Into<PropertyValue>,
    {
        let node_id = id.into();
        self.max_node_id = self.max_node_id.max(node_id);

        for (key, value) in properties {
            self.node_properties
                .push((node_id, key.into(), value.into()));
        }

        self
    }

    /// Adds a node with a label.
    pub fn add_labeled_node<I, K, V>(
        mut self,
        id: impl Into<u64>,
        label: impl Into<String>,
        properties: I,
    ) -> Self
    where
        I: IntoIterator<Item = (K, V)>,
        K: Into<String>,
        V: Into<PropertyValue>,
    {
        let node_id = id.into();
        self.max_node_id = self.max_node_id.max(node_id);
        self.node_labels.push((node_id, label.into()));

        for (key, value) in properties {
            self.node_properties
                .push((node_id, key.into(), value.into()));
        }

        self
    }

    /// Adds an edge.
    pub fn add_edge(mut self, source: impl Into<u64>, target: impl Into<u64>) -> Self {
        let src = source.into();
        let tgt = target.into();
        self.max_node_id = self.max_node_id.max(src).max(tgt);
        self.edges.push(Edge::new(src, tgt));
        self
    }

    /// Adds a labeled edge.
    pub fn add_labeled_edge(
        mut self,
        source: impl Into<u64>,
        target: impl Into<u64>,
        label: Label,
    ) -> Self {
        let src = source.into();
        let tgt = target.into();
        self.max_node_id = self.max_node_id.max(src).max(tgt);
        self.edges.push(Edge::with_label(src, tgt, label));
        self
    }

    /// Builds the GraphStore.
    pub fn build(self) -> GraphStore {
        let num_nodes = (self.max_node_id + 1) as usize;
        let graph = CsrMatrix::from_edges(&self.edges, num_nodes);

        // Initial transaction ID for builder data (all data visible from tx 1)
        let initial_tx_id: TransactionId = 1;

        // Build property index and versioned properties
        let mut property_index = PropertyIndex::new();
        let mut versioned_properties = VersionedPropertyStore::with_capacity(num_nodes);
        for (node_id, key, value) in self.node_properties {
            versioned_properties.set(NodeId::new(node_id), &key, value.clone(), initial_tx_id);
            property_index.add(NodeId::new(node_id), &key, &value);
        }

        // Build label index and versioned labels
        let mut versioned_labels = VersionedPropertyStore::new();
        let mut label_index = LabelIndex::new();

        for (node_id, label) in self.node_labels {
            versioned_labels.set(
                NodeId::new(node_id),
                "_label",
                PropertyValue::from(label.clone()),
                initial_tx_id,
            );
            label_index.add(NodeId::new(node_id), &label);
        }

        // Build edge type index
        let mut edge_type_index = EdgeTypeIndex::new();
        // Keep track of how many times we've seen a (src, tgt) pair to handle multi-edges
        let mut edge_counts: HashMap<(NodeId, NodeId), usize> = HashMap::new();

        for edge in &self.edges {
            if let Some(ref label) = edge.label {
                // Find the edge ID in the CSR matrix
                let u = edge.source.as_usize();
                if u < graph.num_nodes {
                    let start = graph.row_ptr[u];
                    let end = graph.row_ptr[u + 1];
                    let count = edge_counts.entry((edge.source, edge.target)).or_insert(0);

                    // Find the n-th occurrence of target in this row
                    let mut found_count = 0;
                    let mut edge_id = None;
                    for i in start..end {
                        if graph.col_indices[i] == edge.target {
                            if found_count == *count {
                                edge_id = Some(i as u64);
                                break;
                            }
                            found_count += 1;
                        }
                    }

                    if let Some(eid) = edge_id {
                        edge_type_index.add(neural_core::EdgeId::new(eid), edge.source, edge.target, label.as_str());
                        *count += 1;
                    }
                }
            }
        }

        // Finalize all indices (sort for binary search)
        label_index.finalize();
        property_index.finalize();

        // Build CSC reverse graph for O(1) incoming edge lookups
        let reverse_graph = CscMatrix::from_csr(&graph);

        GraphStore {
            graph,
            reverse_graph,
            versioned_properties,
            versioned_labels,
            label_index,
            property_index,
            edge_type_index,
            vector_index: None,
            dynamic_node_count: 0,
            dynamic_edges: Vec::new(),
            path: None,
            wal: None,
            transaction_manager: TransactionManager::new(),
            current_tx_id: initial_tx_id + 1, // Next tx is 2
        }
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_graph_store_basic() {
        let store = GraphStore::builder()
            .add_node(0u64, [("name", "Alice")])
            .add_node(1u64, [("name", "Bob")])
            .add_edge(0u64, 1u64)
            .build();

        assert_eq!(store.node_count(), 2);
        assert_eq!(store.edge_count(), 1);
        assert_eq!(
            store.get_property(NodeId::new(0), "name"),
            Some(&PropertyValue::from("Alice"))
        );
    }

    #[test]
    fn test_graph_store_with_labels() {
        let store = GraphStore::builder()
            .add_labeled_node(0u64, "Person", [("name", "Alice")])
            .add_labeled_node(1u64, "Person", [("name", "Bob")])
            .add_labeled_node(2u64, "Company", [("name", "Acme")])
            .add_edge(0u64, 2u64)
            .build();

        assert!(store.has_label(NodeId::new(0), "Person"));
        assert!(store.has_label(NodeId::new(2), "Company"));
        assert!(!store.has_label(NodeId::new(0), "Company"));

        let people: Vec<_> = store.nodes_with_label("Person").collect();
        assert_eq!(people.len(), 2);
    }

    #[test]
    fn test_graph_store_neighbors() {
        let store = GraphStore::builder()
            .add_node(0u64, [("name", "Alice")])
            .add_node(1u64, [("name", "Bob")])
            .add_node(2u64, [("name", "Charlie")])
            .add_edge(0u64, 1u64)
            .add_edge(0u64, 2u64)
            .build();

        let neighbors: Vec<_> = store.neighbors(NodeId::new(0)).collect();
        assert_eq!(neighbors.len(), 2);
    }

    #[test]
    fn test_graph_store_set_property() {
        let mut store = GraphStore::builder()
            .add_node(0u64, [("name", "Alice")])
            .build();

        store.set_property(NodeId::new(0), "age", PropertyValue::from(30i64));

        assert_eq!(
            store.get_property(NodeId::new(0), "age"),
            Some(&PropertyValue::from(30i64))
        );
    }

    #[test]
    fn test_label_index_basic() {
        let mut index = LabelIndex::new();

        index.add(NodeId::new(0), "Person");
        index.add(NodeId::new(1), "Person");
        index.add(NodeId::new(2), "Company");
        index.finalize();

        assert_eq!(index.label_count(), 2);
        assert_eq!(index.nodes_with_label_count("Person"), 2);
        assert_eq!(index.nodes_with_label_count("Company"), 1);
        assert_eq!(index.nodes_with_label_count("Unknown"), 0);
    }

    #[test]
    fn test_label_index_contains() {
        let mut index = LabelIndex::new();

        index.add(NodeId::new(0), "Person");
        index.add(NodeId::new(5), "Person");
        index.add(NodeId::new(10), "Company");
        index.finalize();

        assert!(index.contains(NodeId::new(0), "Person"));
        assert!(index.contains(NodeId::new(5), "Person"));
        assert!(!index.contains(NodeId::new(10), "Person"));
        assert!(index.contains(NodeId::new(10), "Company"));
    }

    #[test]
    fn test_label_index_get() {
        let mut index = LabelIndex::new();

        index.add(NodeId::new(3), "Person");
        index.add(NodeId::new(1), "Person");
        index.add(NodeId::new(5), "Person");
        index.finalize();

        let nodes = index.get("Person").unwrap();
        // Should be sorted after finalize
        assert_eq!(nodes, &[NodeId::new(1), NodeId::new(3), NodeId::new(5)]);
    }

    #[test]
    fn test_nodes_with_label_performance() {
        // Create a large graph to test performance
        let mut builder = GraphStore::builder();

        for i in 0..10000u64 {
            if i % 10 == 0 {
                builder =
                    builder.add_labeled_node(i, "Rare", [("id", PropertyValue::from(i as i64))]);
            } else {
                builder =
                    builder.add_labeled_node(i, "Common", [("id", PropertyValue::from(i as i64))]);
            }
        }

        let store = builder.build();

        // This should be O(1) now, not O(n)
        let rare_count = store.nodes_with_label("Rare").count();
        assert_eq!(rare_count, 1000);

        let common_count = store.nodes_with_label("Common").count();
        assert_eq!(common_count, 9000);

        // Direct count should also work
        assert_eq!(store.nodes_with_label_count("Rare"), 1000);
        assert_eq!(store.nodes_with_label_count("Common"), 9000);
    }

    // =========================================================================
    // PropertyIndex Tests (Sprint 10)
    // =========================================================================

    #[test]
    fn test_property_index_basic() {
        let mut index = PropertyIndex::new();

        index.add(NodeId::new(0), "category", &PropertyValue::from("A"));
        index.add(NodeId::new(1), "category", &PropertyValue::from("A"));
        index.add(NodeId::new(2), "category", &PropertyValue::from("B"));
        index.finalize();

        assert_eq!(index.property_count(), 1);

        let a_nodes = index.get("category", &PropertyValue::from("A")).unwrap();
        assert_eq!(a_nodes.len(), 2);

        let b_nodes = index.get("category", &PropertyValue::from("B")).unwrap();
        assert_eq!(b_nodes.len(), 1);
    }

    #[test]
    fn test_property_index_contains() {
        let mut index = PropertyIndex::new();

        index.add(NodeId::new(0), "age", &PropertyValue::from(30i64));
        index.add(NodeId::new(1), "age", &PropertyValue::from(25i64));
        index.finalize();

        assert!(index.contains(NodeId::new(0), "age", &PropertyValue::from(30i64)));
        assert!(!index.contains(NodeId::new(0), "age", &PropertyValue::from(25i64)));
        assert!(index.contains(NodeId::new(1), "age", &PropertyValue::from(25i64)));
    }

    #[test]
    fn test_nodes_with_property() {
        let store = GraphStore::builder()
            .add_labeled_node(0u64, "Person", [("category", "engineer")])
            .add_labeled_node(1u64, "Person", [("category", "engineer")])
            .add_labeled_node(2u64, "Person", [("category", "designer")])
            .build();

        let engineers: Vec<_> = store
            .nodes_with_property("category", &PropertyValue::from("engineer"))
            .collect();
        assert_eq!(engineers.len(), 2);

        let designers: Vec<_> = store
            .nodes_with_property("category", &PropertyValue::from("designer"))
            .collect();
        assert_eq!(designers.len(), 1);
    }

    // =========================================================================
    // EdgeTypeIndex Tests (Sprint 11)
    // =========================================================================

    #[test]
    fn test_edge_type_index_basic() {
        let mut index = EdgeTypeIndex::new();

        index.add(neural_core::EdgeId::new(0), NodeId::new(0), NodeId::new(1), "KNOWS");
        index.add(neural_core::EdgeId::new(1), NodeId::new(0), NodeId::new(2), "KNOWS");
        index.add(neural_core::EdgeId::new(2), NodeId::new(1), NodeId::new(2), "WORKS_WITH");

        assert_eq!(index.type_count(), 2);
        assert_eq!(index.edges_with_type_count("KNOWS"), 2);
        assert_eq!(index.edges_with_type_count("WORKS_WITH"), 1);
        assert_eq!(index.total_edges(), 3);
    }

    #[test]
    fn test_edge_type_index_get() {
        let mut index = EdgeTypeIndex::new();

        index.add(neural_core::EdgeId::new(0), NodeId::new(0), NodeId::new(1), "CITES");
        index.add(neural_core::EdgeId::new(1), NodeId::new(0), NodeId::new(2), "CITES");

        let cites = index.get("CITES").unwrap();
        assert_eq!(cites.len(), 2);
        assert!(cites.contains(&(neural_core::EdgeId::new(0), NodeId::new(0), NodeId::new(1))));
        assert!(cites.contains(&(neural_core::EdgeId::new(1), NodeId::new(0), NodeId::new(2))));
    }

    #[test]
    fn test_edges_with_type() {
        let store = GraphStore::builder()
            .add_labeled_node(0u64, "Person", [("name", "Alice")])
            .add_labeled_node(1u64, "Person", [("name", "Bob")])
            .add_labeled_node(2u64, "Company", [("name", "Acme")])
            .add_labeled_edge(0u64, 1u64, Label::new("KNOWS"))
            .add_labeled_edge(0u64, 2u64, Label::new("WORKS_AT"))
            .add_labeled_edge(1u64, 2u64, Label::new("WORKS_AT"))
            .build();

        let knows: Vec<_> = store.edges_with_type("KNOWS").collect();
        assert_eq!(knows.len(), 1);
        assert_eq!(knows[0], (NodeId::new(0), NodeId::new(1)));

        let works_at: Vec<_> = store.edges_with_type("WORKS_AT").collect();
        assert_eq!(works_at.len(), 2);

        // Test neighbors via type
        let alice_coworkers: Vec<_> = store.neighbors_via_type(NodeId::new(0), "KNOWS").collect();
        assert_eq!(alice_coworkers.len(), 1);
        assert_eq!(alice_coworkers[0], NodeId::new(1));
    }

    // =========================================================================
    // VectorIndex Tests (Sprint 13)
    // =========================================================================

    #[test]
    fn test_vector_index_basic() {
        let mut store = GraphStore::builder()
            .add_labeled_node(0u64, "Paper", [("title", "Paper A")])
            .add_labeled_node(1u64, "Paper", [("title", "Paper B")])
            .add_labeled_node(2u64, "Paper", [("title", "Paper C")])
            .build();

        // Initialize vector index
        store.init_vector_index(3);
        assert_eq!(store.vector_dimension(), Some(3));

        // Add vectors
        store.add_vector(NodeId::new(0), &[1.0, 0.0, 0.0]);
        store.add_vector(NodeId::new(1), &[0.0, 1.0, 0.0]);
        store.add_vector(NodeId::new(2), &[0.0, 0.0, 1.0]);

        assert!(store.has_vector(NodeId::new(0)));
        assert!(store.has_vector(NodeId::new(1)));
        assert!(store.has_vector(NodeId::new(2)));
    }

    #[test]
    fn test_vector_search() {
        let mut store = GraphStore::builder()
            .add_labeled_node(0u64, "Paper", [("title", "ML Paper")])
            .add_labeled_node(1u64, "Paper", [("title", "AI Paper")])
            .add_labeled_node(2u64, "Paper", [("title", "DB Paper")])
            .build();

        store.init_vector_index(3);
        store.add_vector(NodeId::new(0), &[1.0, 0.0, 0.0]);
        store.add_vector(NodeId::new(1), &[0.9, 0.1, 0.0]);
        store.add_vector(NodeId::new(2), &[0.0, 0.0, 1.0]);

        // Search for vectors similar to [1, 0, 0]
        let results = store.vector_search(&[1.0, 0.0, 0.0], 2);

        assert_eq!(results.len(), 2);
        // First result should be node 0 (exact match)
        assert_eq!(results[0].0, NodeId::new(0));
        // Second should be node 1 (close to query)
        assert_eq!(results[1].0, NodeId::new(1));
    }

    #[test]
    fn test_vector_similarity() {
        let mut store = GraphStore::builder()
            .add_labeled_node(0u64, "Paper", [("title", "Paper A")])
            .build();

        store.init_vector_index(3);
        store.add_vector(NodeId::new(0), &[1.0, 0.0, 0.0]);

        // Similarity with exact same vector should be ~1.0
        let sim = store.vector_similarity(NodeId::new(0), &[1.0, 0.0, 0.0]);
        assert!(sim.is_some());
        assert!(sim.unwrap() > 0.99);

        // Node without vector returns None
        let no_sim = store.vector_similarity(NodeId::new(99), &[1.0, 0.0, 0.0]);
        assert!(no_sim.is_none());
    }

    #[test]
    fn test_vector_search_empty() {
        let store = GraphStore::builder()
            .add_labeled_node(0u64, "Paper", [("title", "Paper A")])
            .build();

        // Search without initialized index returns empty
        let results = store.vector_search(&[1.0, 0.0, 0.0], 5);
        assert!(results.is_empty());
    }

    #[test]
    fn test_find_paths_variable_length() {
        let store = GraphStore::builder()
            // Chain: 0 -> 1 -> 2 -> 3 -> 4
            .add_edge(0u64, 1u64)
            .add_edge(1u64, 2u64)
            .add_edge(2u64, 3u64)
            .add_edge(3u64, 4u64)
            // Branch: 1 -> 5
            .add_edge(1u64, 5u64)
            .build();

        // Path from 0 to 2 (length 2)
        // Range 1..3
        let paths = store.find_paths(NodeId::new(0), Some(NodeId::new(2)), 1, 3, None);
        assert_eq!(paths.len(), 1);
        assert_eq!(paths[0], vec![NodeId::new(0), NodeId::new(1), NodeId::new(2)]);

        // Path from 0 to 4 (length 4)
        // Range 1..3 - should not find it
        let paths = store.find_paths(NodeId::new(0), Some(NodeId::new(4)), 1, 3, None);
        assert!(paths.is_empty());

        // Range 1..5 - should find it
        let paths = store.find_paths(NodeId::new(0), Some(NodeId::new(4)), 1, 5, None);
        assert_eq!(paths.len(), 1);
        assert_eq!(paths[0].len(), 5); // 0,1,2,3,4

        // Variable length to ANY node from 0
        // Range 1..2
        // Should find:
        // 0->1 (len 1)
        // 0->1->2 (len 2)
        // 0->1->5 (len 2)
        let paths = store.find_paths(NodeId::new(0), None, 1, 2, None);
        assert_eq!(paths.len(), 3);
    }
}
