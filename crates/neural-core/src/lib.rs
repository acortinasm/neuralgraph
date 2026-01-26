//! # Neural Core
//!
//! Core types and traits for NeuralGraphDB.
//!
//! This crate provides the fundamental building blocks:
//! - [`NodeId`] and [`EdgeId`] - Type-safe identifiers
//! - [`PropertyValue`] - Schema-flexible property storage
//! - [`Label`] - Node and edge labels
//! - [`Graph`] - Core trait for graph implementations

use serde::{Deserialize, Serialize};
use std::fmt;
use thiserror::Error;

// =============================================================================
// Identifiers (Newtypes for type safety)
// =============================================================================

/// A unique identifier for a node in the graph.
///
/// Uses a newtype pattern to prevent mixing up node IDs with other integer types.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
#[repr(transparent)]
pub struct NodeId(pub u64);

impl NodeId {
    /// Creates a new NodeId from a u64.
    #[inline]
    pub const fn new(id: u64) -> Self {
        Self(id)
    }

    /// Returns the raw u64 value.
    #[inline]
    pub const fn as_u64(self) -> u64 {
        self.0
    }

    /// Returns the ID as a usize for indexing.
    #[inline]
    pub const fn as_usize(self) -> usize {
        self.0 as usize
    }
}

impl fmt::Display for NodeId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "n{}", self.0)
    }
}

impl From<u64> for NodeId {
    #[inline]
    fn from(id: u64) -> Self {
        Self(id)
    }
}

impl From<usize> for NodeId {
    #[inline]
    fn from(id: usize) -> Self {
        Self(id as u64)
    }
}

/// A unique identifier for an edge in the graph.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
#[repr(transparent)]
pub struct EdgeId(pub u64);

impl EdgeId {
    /// Creates a new EdgeId from a u64.
    #[inline]
    pub const fn new(id: u64) -> Self {
        Self(id)
    }

    /// Returns the raw u64 value.
    #[inline]
    pub const fn as_u64(self) -> u64 {
        self.0
    }
}

impl fmt::Display for EdgeId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "e{}", self.0)
    }
}

impl From<u64> for EdgeId {
    #[inline]
    fn from(id: u64) -> Self {
        Self(id)
    }
}

// =============================================================================
// Labels
// =============================================================================

/// A label for nodes or edges (e.g., `:Person`, `:KNOWS`).
///
/// Labels are interned strings for efficient comparison.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct Label(String);

impl Label {
    /// Creates a new label.
    pub fn new(name: impl Into<String>) -> Self {
        Self(name.into())
    }

    /// Returns the label as a string slice.
    #[inline]
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl fmt::Display for Label {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, ":{}", self.0)
    }
}

impl From<&str> for Label {
    fn from(s: &str) -> Self {
        Self::new(s)
    }
}

impl From<String> for Label {
    fn from(s: String) -> Self {
        Self(s)
    }
}

// =============================================================================
// Property Values
// =============================================================================

/// A property value that can be stored on nodes or edges.
///
/// Supports the types defined in the NGQL specification:
/// - Primitives: Bool, Int, Float, String
/// - Complex: Vector (for embeddings)
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Default)]
pub enum PropertyValue {
    /// Null/missing value
    #[default]
    Null,
    /// Boolean value
    Bool(bool),
    /// 64-bit signed integer
    Int(i64),
    /// 64-bit floating point
    Float(f64),
    /// UTF-8 string
    String(String),
    /// Date (YYYY-MM-DD) stored as days since epoch or string for simplicity for now
    Date(String),
    /// DateTime (ISO 8601)
    DateTime(String),
    /// Vector of f32 (for embeddings)
    Vector(Vec<f32>),
}

impl PropertyValue {
    /// Returns true if the value is null.
    #[inline]
    pub fn is_null(&self) -> bool {
        matches!(self, PropertyValue::Null)
    }

    /// Attempts to get the value as a bool.
    pub fn as_bool(&self) -> Option<bool> {
        match self {
            PropertyValue::Bool(b) => Some(*b),
            _ => None,
        }
    }

    /// Attempts to get the value as an i64.
    pub fn as_int(&self) -> Option<i64> {
        match self {
            PropertyValue::Int(i) => Some(*i),
            _ => None,
        }
    }

    /// Attempts to get the value as an f64.
    pub fn as_float(&self) -> Option<f64> {
        match self {
            PropertyValue::Float(f) => Some(*f),
            PropertyValue::Int(i) => Some(*i as f64),
            _ => None,
        }
    }

    /// Attempts to get the value as a string slice.
    pub fn as_str(&self) -> Option<&str> {
        match self {
            PropertyValue::String(s) => Some(s),
            _ => None,
        }
    }

    /// Attempts to get the value as a vector slice.
    pub fn as_vector(&self) -> Option<&[f32]> {
        match self {
            PropertyValue::Vector(v) => Some(v),
            _ => None,
        }
    }
}

// Convenient From implementations
impl From<bool> for PropertyValue {
    fn from(v: bool) -> Self {
        PropertyValue::Bool(v)
    }
}

impl From<i64> for PropertyValue {
    fn from(v: i64) -> Self {
        PropertyValue::Int(v)
    }
}

impl From<i32> for PropertyValue {
    fn from(v: i32) -> Self {
        PropertyValue::Int(v as i64)
    }
}

impl From<f64> for PropertyValue {
    fn from(v: f64) -> Self {
        PropertyValue::Float(v)
    }
}

impl From<f32> for PropertyValue {
    fn from(v: f32) -> Self {
        PropertyValue::Float(v as f64)
    }
}

impl From<String> for PropertyValue {
    fn from(v: String) -> Self {
        PropertyValue::String(v)
    }
}

impl From<&str> for PropertyValue {
    fn from(v: &str) -> Self {
        PropertyValue::String(v.to_string())
    }
}

impl From<Vec<f32>> for PropertyValue {
    fn from(v: Vec<f32>) -> Self {
        PropertyValue::Vector(v)
    }
}

// =============================================================================
// Edge Definition
// =============================================================================

/// An edge in the graph, connecting a source node to a target node.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct Edge {
    /// Source node ID
    pub source: NodeId,
    /// Target node ID
    pub target: NodeId,
    /// Edge label/type (e.g., :KNOWS, :OWNS)
    pub label: Option<Label>,
}

impl Edge {
    /// Creates a new edge from source to target.
    pub fn new(source: impl Into<NodeId>, target: impl Into<NodeId>) -> Self {
        Self {
            source: source.into(),
            target: target.into(),
            label: None,
        }
    }

    /// Creates a new labeled edge.
    pub fn with_label(
        source: impl Into<NodeId>,
        target: impl Into<NodeId>,
        label: impl Into<Label>,
    ) -> Self {
        Self {
            source: source.into(),
            target: target.into(),
            label: Some(label.into()),
        }
    }
}

// =============================================================================
// Graph Trait
// =============================================================================

/// Core trait for graph implementations.
///
/// This trait defines the minimal interface that any graph storage must implement.
/// It is designed to be compatible with both CSR matrices and other representations.
pub trait Graph {
    /// Returns the number of nodes in the graph.
    fn node_count(&self) -> usize;

    /// Returns the number of edges in the graph.
    fn edge_count(&self) -> usize;

    /// Returns an iterator over the neighbor node IDs of a given node.
    ///
    /// For directed graphs, this returns the outgoing neighbors.
    fn neighbors(&self, node: NodeId) -> impl Iterator<Item = NodeId>;

    /// Returns the out-degree of a node (number of outgoing edges).
    fn out_degree(&self, node: NodeId) -> usize;

    /// Checks if the graph contains a node with the given ID.
    fn contains_node(&self, node: NodeId) -> bool {
        node.as_usize() < self.node_count()
    }

    /// Checks if there is an edge from source to target.
    fn has_edge(&self, source: NodeId, target: NodeId) -> bool {
        self.neighbors(source).any(|n| n == target)
    }
}

// =============================================================================
// Linear Algebra Traits (GraphBLAS Primitives)
// =============================================================================

/// A semiring defines the addition and multiplication operations for matrix/vector operations.
///
/// Common semirings:
/// - Arithmetic: (+, *) over reals. For standard linear algebra.
/// - Tropical: (min, +) over reals. For shortest paths.
/// - Boolean: (OR, AND) over booleans. For connectivity/BFS.
pub trait Semiring<T> {
    /// Addition operation (e.g., + or min or OR)
    fn add(a: T, b: T) -> T;
    /// Multiplication operation (e.g., * or + or AND)
    fn mul(a: T, b: T) -> T;
    /// Additive identity (e.g., 0 or inf or false)
    fn zero() -> T;
    /// Multiplicative identity (e.g., 1 or 0 or true)
    fn one() -> T;
}

/// A Vector trait for GraphBLAS operations.
pub trait Vector<T> {
    /// Returns the dimension of the vector.
    fn len(&self) -> usize;
    
    /// Returns true if the vector is empty.
    fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Get value at index. Returns None if value is not present (sparse) or zero.
    fn get(&self, index: usize) -> Option<T>;
    
    /// Set value at index.
    fn set(&mut self, index: usize, value: T);
    
    /// Iterate over non-zero elements (index, value).
    fn iter_active(&self) -> impl Iterator<Item = (usize, T)>;
}

/// A Matrix trait for GraphBLAS operations.
pub trait Matrix<T> {
    /// Returns number of rows.
    fn nrows(&self) -> usize;
    
    /// Returns number of columns.
    fn ncols(&self) -> usize;
    
    /// Matrix-Vector multiplication (mxv).
    /// y = A * x using the given Semiring.
    ///
    /// The default implementation iterates over active elements of x.
    /// Ideally specialized implementations (like CSR) override this.
    fn mxv<S, VIn, VOut>(&self, x: &VIn, semiring: S) -> VOut
    where
        S: Semiring<T>,
        VIn: Vector<T>,
        VOut: Vector<T> + Default,
        T: Copy + Default;
}

// =============================================================================
// Errors
// =============================================================================

/// Errors that can occur in graph operations.
#[derive(Debug, Error)]
pub enum GraphError {
    /// Node not found in the graph
    #[error("Node {0} not found")]
    NodeNotFound(NodeId),

    /// Edge not found in the graph
    #[error("Edge from {0} to {1} not found")]
    EdgeNotFound(NodeId, NodeId),

    /// Invalid graph structure
    #[error("Invalid graph structure: {0}")]
    InvalidStructure(String),

    /// Storage error
    #[error("Storage error: {0}")]
    StorageError(String),
}

/// Result type for graph operations.
pub type Result<T> = std::result::Result<T, GraphError>;

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_node_id() {
        let id = NodeId::new(42);
        assert_eq!(id.as_u64(), 42);
        assert_eq!(id.as_usize(), 42);
        assert_eq!(format!("{}", id), "n42");

        // From conversions
        let id2: NodeId = 100u64.into();
        assert_eq!(id2.as_u64(), 100);

        let id3: NodeId = 50usize.into();
        assert_eq!(id3.as_u64(), 50);
    }

    #[test]
    fn test_edge_id() {
        let id = EdgeId::new(123);
        assert_eq!(id.as_u64(), 123);
        assert_eq!(format!("{}", id), "e123");
    }

    #[test]
    fn test_label() {
        let label = Label::new("Person");
        assert_eq!(label.as_str(), "Person");
        assert_eq!(format!("{}", label), ":Person");

        // From conversions
        let label2: Label = "Company".into();
        assert_eq!(label2.as_str(), "Company");
    }

    #[test]
    fn test_property_value_types() {
        assert!(PropertyValue::Null.is_null());

        let bool_val = PropertyValue::from(true);
        assert_eq!(bool_val.as_bool(), Some(true));

        let int_val = PropertyValue::from(42i64);
        assert_eq!(int_val.as_int(), Some(42));
        assert_eq!(int_val.as_float(), Some(42.0)); // Int can be read as float

        let float_val = PropertyValue::from(2.5f64);
        assert_eq!(float_val.as_float(), Some(2.5));
        assert_eq!(float_val.as_int(), None); // Float cannot be read as int

        let str_val = PropertyValue::from("hello");
        assert_eq!(str_val.as_str(), Some("hello"));

        let vec_val = PropertyValue::from(vec![1.0f32, 2.0, 3.0]);
        assert_eq!(vec_val.as_vector(), Some(&[1.0f32, 2.0, 3.0][..]));
    }

    #[test]
    fn test_property_value_serialization() {
        let val = PropertyValue::from(42i64);
        let json = serde_json::to_string(&val).unwrap();
        let parsed: PropertyValue = serde_json::from_str(&json).unwrap();
        assert_eq!(val, parsed);

        let vec_val = PropertyValue::from(vec![1.0f32, 2.0, 3.0]);
        let json = serde_json::to_string(&vec_val).unwrap();
        let parsed: PropertyValue = serde_json::from_str(&json).unwrap();
        assert_eq!(vec_val, parsed);
    }

    #[test]
    fn test_edge_creation() {
        let edge = Edge::new(0u64, 1u64);
        assert_eq!(edge.source, NodeId(0));
        assert_eq!(edge.target, NodeId(1));
        assert!(edge.label.is_none());

        let labeled_edge = Edge::with_label(0u64, 1u64, "KNOWS");
        assert!(labeled_edge.label.is_some());
        assert_eq!(labeled_edge.label.unwrap().as_str(), "KNOWS");
    }
}