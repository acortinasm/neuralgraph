//! HNSW-based vector index for approximate nearest neighbor search.
//!
//! This module provides O(log n) vector similarity search using the
//! Hierarchical Navigable Small World (HNSW) algorithm.

use hnsw_rs::anndists::dist::DistCosine;
use hnsw_rs::hnsw::Hnsw;
use neural_core::NodeId;
use std::collections::HashMap;

// =============================================================================
// Configuration
// =============================================================================

/// Configuration for the HNSW vector index.
///
/// # Parameters
///
/// - `dimension`: Vector dimensionality (e.g., 768 for Gemini embeddings)
/// - `m`: Max graph edges per node (16-48). Higher = better recall, more memory.
/// - `ef_construction`: Build quality (100-500). Higher = better graph, slower build.
/// - `max_elements`: Pre-allocated capacity for vectors.
#[derive(Debug, Clone)]
pub struct VectorIndexConfig {
    /// Vector dimensionality
    pub dimension: usize,
    /// Maximum number of bidirectional links per node (M parameter)
    pub m: usize,
    /// Size of the dynamic candidate list during construction
    pub ef_construction: usize,
    /// Initial capacity for the index
    pub max_elements: usize,
}

impl VectorIndexConfig {
    /// Creates a new configuration with the given dimension.
    pub fn new(dimension: usize) -> Self {
        Self {
            dimension,
            ..Default::default()
        }
    }

    /// Creates a configuration optimized for small datasets (<10k vectors).
    pub fn small(dimension: usize) -> Self {
        Self {
            dimension,
            m: 16,
            ef_construction: 200,
            max_elements: 10_000,
        }
    }

    /// Creates a configuration optimized for large datasets (1M+ vectors).
    pub fn large(dimension: usize) -> Self {
        Self {
            dimension,
            m: 24,
            ef_construction: 400,
            max_elements: 1_000_000,
        }
    }
}

impl Default for VectorIndexConfig {
    fn default() -> Self {
        // Default optimized for 1M scale (Sprint 47)
        Self {
            dimension: 768, // Gemini text-embedding-004
            m: 24,
            ef_construction: 400,
            max_elements: 1_000_000,
        }
    }
}

/// Computes the cosine similarity between two vectors.
///
/// Returns a value between -1.0 and 1.0:
/// - 1.0 means identical direction
/// - 0.0 means orthogonal (unrelated)
/// - -1.0 means opposite direction
///
/// # Panics
///
/// Panics if vectors have different lengths.
pub fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(
        a.len(),
        b.len(),
        "Vector dimension mismatch: {} vs {}",
        a.len(),
        b.len()
    );

    let dot_product: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();

    if norm_a == 0.0 || norm_b == 0.0 {
        return 0.0;
    }

    dot_product / (norm_a * norm_b)
}

/// Computes the cosine distance between two vectors.
///
/// Returns a value between 0.0 and 2.0:
/// - 0.0 means identical vectors
/// - 1.0 means orthogonal (unrelated)
/// - 2.0 means opposite direction
///
/// This is computed as: distance = 1.0 - similarity
pub fn cosine_distance(a: &[f32], b: &[f32]) -> f32 {
    1.0 - cosine_similarity(a, b)
}

/// A vector index using HNSW for fast approximate nearest neighbor search.
///
/// # Example
///
/// ```ignore
/// use neural_storage::VectorIndex;
/// use neural_core::NodeId;
///
/// let mut index = VectorIndex::new(3);  // 3-dimensional vectors
/// index.add(NodeId::new(0), &[1.0, 0.0, 0.0]);
/// index.add(NodeId::new(1), &[0.0, 1.0, 0.0]);
///
/// let results = index.search(&[0.9, 0.1, 0.0], 1);
/// assert_eq!(results[0].0, NodeId::new(0));  // Most similar
/// ```
pub struct VectorIndex {
    /// The HNSW index structure
    hnsw: Hnsw<'static, f32, DistCosine>,
    /// Maps HNSW internal IDs to NodeIds
    id_to_node: HashMap<usize, NodeId>,
    /// Maps NodeIds to HNSW internal IDs
    node_to_id: HashMap<NodeId, usize>,
    /// Vector dimension
    dimension: usize,
    /// Counter for internal IDs
    next_id: usize,
}

impl VectorIndex {
    /// Creates a new vector index for the given dimension using default config.
    ///
    /// The default configuration is optimized for large datasets (1M+ vectors):
    /// - M = 24 (graph connectivity)
    /// - ef_construction = 400 (build quality)
    /// - max_elements = 1,000,000
    ///
    /// For smaller datasets, use `VectorIndex::with_config(VectorIndexConfig::small(dim))`.
    ///
    /// # Arguments
    ///
    /// * `dimension` - The dimensionality of vectors to be indexed
    pub fn new(dimension: usize) -> Self {
        Self::with_config(VectorIndexConfig::new(dimension))
    }

    /// Creates a new vector index with custom configuration.
    ///
    /// # Example
    ///
    /// ```ignore
    /// use neural_storage::{VectorIndex, VectorIndexConfig};
    ///
    /// // For small datasets
    /// let small_index = VectorIndex::with_config(VectorIndexConfig::small(128));
    ///
    /// // For large datasets (1M+ vectors)
    /// let large_index = VectorIndex::with_config(VectorIndexConfig::large(768));
    ///
    /// // Custom configuration
    /// let custom = VectorIndexConfig {
    ///     dimension: 512,
    ///     m: 32,
    ///     ef_construction: 500,
    ///     max_elements: 500_000,
    /// };
    /// let custom_index = VectorIndex::with_config(custom);
    /// ```
    pub fn with_config(config: VectorIndexConfig) -> Self {
        Self {
            hnsw: Hnsw::new(
                config.m,
                config.max_elements,
                16, // nb_layer - auto-determined by hnsw_rs
                config.ef_construction,
                DistCosine,
            ),
            id_to_node: HashMap::new(),
            node_to_id: HashMap::new(),
            dimension: config.dimension,
            next_id: 0,
        }
    }

    /// Adds a vector to the index associated with a node.
    ///
    /// # Arguments
    ///
    /// * `node` - The NodeId to associate with this vector
    /// * `vector` - The vector data (must match the dimension)
    ///
    /// # Panics
    ///
    /// Panics if the vector dimension doesn't match the index dimension.
    pub fn add(&mut self, node: NodeId, vector: &[f32]) {
        assert_eq!(
            vector.len(),
            self.dimension,
            "Vector dimension mismatch: expected {}, got {}",
            self.dimension,
            vector.len()
        );

        let internal_id = self.next_id;
        self.next_id += 1;

        // Store mappings
        self.id_to_node.insert(internal_id, node);
        self.node_to_id.insert(node, internal_id);

        // Insert into HNSW
        self.hnsw.insert((vector, internal_id));
    }

    /// Searches for the k nearest neighbors to the query vector.
    ///
    /// # Arguments
    ///
    /// * `query` - The query vector
    /// * `k` - Number of neighbors to return
    ///
    /// # Returns
    ///
    /// A vector of (NodeId, similarity_score) pairs, sorted by decreasing similarity.
    /// The similarity score is cosine similarity (1.0 = identical, 0.0 = orthogonal).
    pub fn search(&self, query: &[f32], k: usize) -> Vec<(NodeId, f32)> {
        if self.id_to_node.is_empty() {
            return Vec::new();
        }

        // Dynamic ef_search: scales with k for better recall at larger k values
        // Rule: ef_search = max(k * 2, 100) for good recall/speed tradeoff
        let ef_search = (k * 2).max(100);

        let results = self.hnsw.search(query, k, ef_search);

        results
            .into_iter()
            .filter_map(|neighbor| {
                let internal_id = neighbor.d_id;
                let distance = neighbor.distance;
                // Convert distance to similarity (cosine distance -> cosine similarity)
                let similarity = 1.0 - distance;

                self.id_to_node
                    .get(&internal_id)
                    .map(|&node| (node, similarity))
            })
            .collect()
    }

    /// Searches with a filter function to exclude certain nodes.
    ///
    /// # Arguments
    ///
    /// * `query` - The query vector
    /// * `k` - Number of neighbors to return
    /// * `filter` - A function that returns true for nodes to include
    ///
    /// # Returns
    ///
    /// A vector of (NodeId, similarity_score) pairs that pass the filter.
    pub fn search_filtered<F>(&self, query: &[f32], k: usize, filter: F) -> Vec<(NodeId, f32)>
    where
        F: Fn(NodeId) -> bool,
    {
        // For filtered search, we request more candidates and filter after
        let candidates = k * 4;
        let results = self.search(query, candidates);

        results
            .into_iter()
            .filter(|(node, _)| filter(*node))
            .take(k)
            .collect()
    }

    /// Returns the number of vectors in the index.
    pub fn len(&self) -> usize {
        self.id_to_node.len()
    }

    /// Returns true if the index is empty.
    pub fn is_empty(&self) -> bool {
        self.id_to_node.is_empty()
    }

    /// Returns the dimension of vectors in this index.
    pub fn dimension(&self) -> usize {
        self.dimension
    }

    /// Checks if a node has a vector in the index.
    pub fn contains(&self, node: NodeId) -> bool {
        self.node_to_id.contains_key(&node)
    }
}

impl std::fmt::Debug for VectorIndex {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("VectorIndex")
            .field("dimension", &self.dimension)
            .field("vector_count", &self.id_to_node.len())
            .finish_non_exhaustive()
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_vector_index_basic() {
        let mut index = VectorIndex::new(3);

        index.add(NodeId::new(0), &[1.0, 0.0, 0.0]);
        index.add(NodeId::new(1), &[0.0, 1.0, 0.0]);
        index.add(NodeId::new(2), &[0.0, 0.0, 1.0]);

        assert_eq!(index.len(), 3);
        assert!(!index.is_empty());
        assert!(index.contains(NodeId::new(0)));
    }

    #[test]
    fn test_vector_search() {
        let mut index = VectorIndex::new(3);

        // Add unit vectors
        index.add(NodeId::new(0), &[1.0, 0.0, 0.0]);
        index.add(NodeId::new(1), &[0.0, 1.0, 0.0]);
        index.add(NodeId::new(2), &[0.0, 0.0, 1.0]);

        // Search for vector close to [1, 0, 0]
        let results = index.search(&[0.9, 0.1, 0.0], 1);

        assert_eq!(results.len(), 1);
        assert_eq!(results[0].0, NodeId::new(0));
        assert!(results[0].1 > 0.9); // High similarity
    }

    #[test]
    fn test_vector_search_k_neighbors() {
        let mut index = VectorIndex::new(2);

        index.add(NodeId::new(0), &[1.0, 0.0]);
        index.add(NodeId::new(1), &[0.9, 0.1]);
        index.add(NodeId::new(2), &[0.0, 1.0]);

        let results = index.search(&[1.0, 0.0], 2);

        assert_eq!(results.len(), 2);
        // First result should be node 0 (exact match)
        assert_eq!(results[0].0, NodeId::new(0));
        // Second should be node 1 (close)
        assert_eq!(results[1].0, NodeId::new(1));
    }

    #[test]
    fn test_vector_search_filtered() {
        let mut index = VectorIndex::new(2);

        index.add(NodeId::new(0), &[1.0, 0.0]);
        index.add(NodeId::new(1), &[0.9, 0.1]);
        index.add(NodeId::new(2), &[0.8, 0.2]);

        // Filter to only include node 2
        let results = index.search_filtered(&[1.0, 0.0], 1, |node| node == NodeId::new(2));

        assert_eq!(results.len(), 1);
        assert_eq!(results[0].0, NodeId::new(2));
    }

    #[test]
    fn test_empty_search() {
        let index = VectorIndex::new(3);
        let results = index.search(&[1.0, 0.0, 0.0], 5);
        assert!(results.is_empty());
    }

    #[test]
    #[should_panic(expected = "Vector dimension mismatch")]
    fn test_dimension_mismatch() {
        let mut index = VectorIndex::new(3);
        index.add(NodeId::new(0), &[1.0, 0.0]); // Wrong dimension
    }

    #[test]
    fn test_cosine_similarity_identical() {
        let a = [1.0, 0.0, 0.0];
        let b = [1.0, 0.0, 0.0];
        let sim = super::cosine_similarity(&a, &b);
        assert!((sim - 1.0).abs() < 1e-6, "Expected 1.0, got {}", sim);
    }

    #[test]
    fn test_cosine_similarity_orthogonal() {
        let a = [1.0, 0.0, 0.0];
        let b = [0.0, 1.0, 0.0];
        let sim = super::cosine_similarity(&a, &b);
        assert!(sim.abs() < 1e-6, "Expected 0.0, got {}", sim);
    }

    #[test]
    fn test_cosine_similarity_opposite() {
        let a = [1.0, 0.0, 0.0];
        let b = [-1.0, 0.0, 0.0];
        let sim = super::cosine_similarity(&a, &b);
        assert!((sim - (-1.0)).abs() < 1e-6, "Expected -1.0, got {}", sim);
    }

    #[test]
    fn test_cosine_distance() {
        let a = [1.0, 0.0, 0.0];
        let b = [1.0, 0.0, 0.0];
        let dist = super::cosine_distance(&a, &b);
        assert!(dist.abs() < 1e-6, "Expected 0.0, got {}", dist);

        let b2 = [0.0, 1.0, 0.0];
        let dist2 = super::cosine_distance(&a, &b2);
        assert!((dist2 - 1.0).abs() < 1e-6, "Expected 1.0, got {}", dist2);
    }
}
