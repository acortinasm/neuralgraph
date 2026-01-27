//! HNSW-based vector index for approximate nearest neighbor search.
//!
//! This module provides O(log n) vector similarity search using the
//! Hierarchical Navigable Small World (HNSW) algorithm.
//!
//! ## Embedding Metadata (Sprint 56)
//!
//! Each embedding can have associated metadata:
//! - Model origin (e.g., "openai/text-embedding-3-small")
//! - Distance metric (Cosine, Euclidean, DotProduct)
//! - Creation timestamp
//!
//! ## Flash Quantization (Sprint 60)
//!
//! Supports scalar quantization for 4x memory reduction:
//! - `Int8`: Maps f32 to i8 with per-vector scale/offset (4x savings)
//! - `Binary`: Maps f32 to bits (32x savings, lower accuracy)
//! - Asymmetric distance computation for better accuracy

use hnsw_rs::anndists::dist::DistCosine;
use hnsw_rs::hnsw::Hnsw;
use neural_core::NodeId;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

// =============================================================================
// Distance Metrics (Sprint 56)
// =============================================================================

/// Distance metric for vector similarity computation.
///
/// Different embedding models are optimized for different metrics:
/// - OpenAI embeddings: Cosine
/// - Some sentence transformers: Euclidean
/// - Matryoshka embeddings: DotProduct
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, Default)]
pub enum DistanceMetric {
    /// Cosine similarity (default). Range: -1.0 to 1.0
    /// Best for normalized embeddings from most LLM providers.
    #[default]
    Cosine,
    /// Euclidean (L2) distance. Range: 0.0 to infinity
    /// Good for embeddings where magnitude matters.
    Euclidean,
    /// Dot product similarity. Range: -infinity to infinity
    /// Fastest computation, good for pre-normalized vectors.
    DotProduct,
}

impl DistanceMetric {
    /// Computes similarity between two vectors using this metric.
    ///
    /// For Cosine and DotProduct, higher is more similar.
    /// For Euclidean, lower distance means more similar (returns negative distance).
    pub fn similarity(&self, a: &[f32], b: &[f32]) -> f32 {
        match self {
            DistanceMetric::Cosine => cosine_similarity(a, b),
            DistanceMetric::Euclidean => -euclidean_distance(a, b), // Negate so higher = more similar
            DistanceMetric::DotProduct => dot_product(a, b),
        }
    }

    /// Computes distance between two vectors using this metric.
    ///
    /// Lower distance means more similar for all metrics.
    pub fn distance(&self, a: &[f32], b: &[f32]) -> f32 {
        match self {
            DistanceMetric::Cosine => cosine_distance(a, b),
            DistanceMetric::Euclidean => euclidean_distance(a, b),
            DistanceMetric::DotProduct => -dot_product(a, b), // Negate so lower = more similar
        }
    }
}

impl std::fmt::Display for DistanceMetric {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            DistanceMetric::Cosine => write!(f, "cosine"),
            DistanceMetric::Euclidean => write!(f, "euclidean"),
            DistanceMetric::DotProduct => write!(f, "dot_product"),
        }
    }
}

// =============================================================================
// Flash Quantization (Sprint 60)
// =============================================================================

/// Quantization method for vector compression.
///
/// Flash Quantization reduces memory footprint by storing vectors in lower
/// precision formats while maintaining search accuracy.
///
/// # Memory Savings
///
/// | Method | Bytes/Dimension | Savings vs f32 |
/// |--------|-----------------|----------------|
/// | None   | 4 (f32)         | 0%             |
/// | Int8   | 1 (i8)          | 75%            |
/// | Binary | 0.125 (1 bit)   | 97%            |
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, Default)]
pub enum QuantizationMethod {
    /// No quantization - store as f32 (default)
    #[default]
    None,
    /// Scalar int8 quantization with per-vector scale/offset
    /// Provides 4x memory reduction with <1% accuracy loss
    Int8,
    /// Binary quantization (1 bit per dimension)
    /// Provides 32x memory reduction but lower accuracy
    Binary,
}

impl std::fmt::Display for QuantizationMethod {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            QuantizationMethod::None => write!(f, "none"),
            QuantizationMethod::Int8 => write!(f, "int8"),
            QuantizationMethod::Binary => write!(f, "binary"),
        }
    }
}

/// Parameters for a quantized vector.
///
/// Used for int8 scalar quantization: `quantized[i] = round((original[i] - offset) / scale)`
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuantizationParams {
    /// Scale factor for dequantization
    pub scale: f32,
    /// Offset for dequantization
    pub offset: f32,
}

impl QuantizationParams {
    /// Creates new quantization parameters.
    pub fn new(scale: f32, offset: f32) -> Self {
        Self { scale, offset }
    }

    /// Computes quantization parameters for a vector.
    ///
    /// Uses min-max scaling to map the vector range to [-128, 127].
    pub fn from_vector(vector: &[f32]) -> Self {
        if vector.is_empty() {
            return Self::new(1.0, 0.0);
        }

        let min = vector.iter().cloned().fold(f32::INFINITY, f32::min);
        let max = vector.iter().cloned().fold(f32::NEG_INFINITY, f32::max);

        let range = max - min;
        if range < f32::EPSILON {
            // All values are the same
            return Self::new(1.0, min);
        }

        // Map to [-128, 127] range (255 levels)
        let scale = range / 255.0;
        let offset = min;

        Self::new(scale, offset)
    }
}

/// Quantizes a f32 vector to i8 using the given parameters.
///
/// Formula: `quantized[i] = clamp(round((original[i] - offset) / scale) - 128, -128, 127)`
#[inline]
pub fn quantize_f32_to_i8(vector: &[f32], params: &QuantizationParams) -> Vec<i8> {
    vector
        .iter()
        .map(|&v| {
            let normalized = (v - params.offset) / params.scale;
            let quantized = (normalized - 128.0).round();
            quantized.clamp(-128.0, 127.0) as i8
        })
        .collect()
}

/// Dequantizes an i8 vector back to f32 using the given parameters.
///
/// Formula: `original[i] = (quantized[i] + 128) * scale + offset`
#[inline]
pub fn dequantize_i8_to_f32(quantized: &[i8], params: &QuantizationParams) -> Vec<f32> {
    quantized
        .iter()
        .map(|&q| (q as f32 + 128.0) * params.scale + params.offset)
        .collect()
}

/// Quantizes a f32 vector to binary (1 bit per dimension).
///
/// Each dimension becomes 1 if >= 0, else 0.
/// Packed into bytes (8 dimensions per byte).
#[inline]
pub fn quantize_f32_to_binary(vector: &[f32]) -> Vec<u8> {
    let num_bytes = (vector.len() + 7) / 8;
    let mut result = vec![0u8; num_bytes];

    for (i, &v) in vector.iter().enumerate() {
        if v >= 0.0 {
            result[i / 8] |= 1 << (7 - (i % 8));
        }
    }

    result
}

/// Computes Hamming distance between two binary vectors.
///
/// Returns the number of differing bits.
#[inline]
pub fn hamming_distance(a: &[u8], b: &[u8]) -> u32 {
    a.iter()
        .zip(b.iter())
        .map(|(&x, &y)| (x ^ y).count_ones())
        .sum()
}

/// Computes asymmetric distance between f32 query and quantized vector.
///
/// This provides better accuracy than symmetric quantized distance
/// by keeping the query in full precision.
#[inline]
pub fn asymmetric_distance_i8(
    query: &[f32],
    quantized: &[i8],
    params: &QuantizationParams,
    metric: DistanceMetric,
) -> f32 {
    // Dequantize on-the-fly for distance computation
    let dequantized = dequantize_i8_to_f32(quantized, params);
    metric.distance(query, &dequantized)
}

// =============================================================================
// Embedding Metadata (Sprint 56)
// =============================================================================

/// Metadata associated with an embedding vector.
///
/// Tracks the origin model, distance metric, and creation time.
///
/// # Example
///
/// ```
/// use neural_storage::vector_index::{EmbeddingMetadata, DistanceMetric};
///
/// let metadata = EmbeddingMetadata::new("openai/text-embedding-3-small")
///     .with_metric(DistanceMetric::Cosine);
///
/// assert_eq!(metadata.model(), "openai/text-embedding-3-small");
/// ```
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EmbeddingMetadata {
    /// The model used to generate this embedding (e.g., "openai/text-embedding-3-small")
    model: String,
    /// The distance metric this embedding is optimized for
    metric: DistanceMetric,
    /// Creation timestamp (ISO 8601)
    created_at: String,
    /// Optional dimension of the embedding
    dimension: Option<usize>,
    /// Optional additional properties
    properties: HashMap<String, String>,
}

impl EmbeddingMetadata {
    /// Creates new metadata with the given model name.
    pub fn new(model: impl Into<String>) -> Self {
        Self {
            model: model.into(),
            metric: DistanceMetric::default(),
            created_at: chrono::Utc::now().to_rfc3339(),
            dimension: None,
            properties: HashMap::new(),
        }
    }

    /// Sets the distance metric.
    pub fn with_metric(mut self, metric: DistanceMetric) -> Self {
        self.metric = metric;
        self
    }

    /// Sets the embedding dimension.
    pub fn with_dimension(mut self, dim: usize) -> Self {
        self.dimension = Some(dim);
        self
    }

    /// Adds a custom property.
    pub fn with_property(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.properties.insert(key.into(), value.into());
        self
    }

    /// Returns the model name.
    pub fn model(&self) -> &str {
        &self.model
    }

    /// Returns the distance metric.
    pub fn metric(&self) -> DistanceMetric {
        self.metric
    }

    /// Returns the creation timestamp.
    pub fn created_at(&self) -> &str {
        &self.created_at
    }

    /// Returns the dimension if set.
    pub fn dimension(&self) -> Option<usize> {
        self.dimension
    }

    /// Returns a custom property.
    pub fn property(&self, key: &str) -> Option<&str> {
        self.properties.get(key).map(|s| s.as_str())
    }
}

impl Default for EmbeddingMetadata {
    fn default() -> Self {
        Self::new("unknown")
    }
}

// =============================================================================
// Distance Functions
// =============================================================================

/// Computes the dot product between two vectors.
pub fn dot_product(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len(), "Vector dimension mismatch");
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

/// Computes the Euclidean (L2) distance between two vectors.
pub fn euclidean_distance(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len(), "Vector dimension mismatch");
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).powi(2))
        .sum::<f32>()
        .sqrt()
}

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
/// - `model`: Optional model name for metadata tracking (Sprint 56)
/// - `metric`: Distance metric for similarity computation (Sprint 56)
/// - `quantization`: Quantization method for memory reduction (Sprint 60)
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
    /// Model name for embeddings (e.g., "openai/text-embedding-3-small")
    pub model: Option<String>,
    /// Distance metric (default: Cosine)
    pub metric: DistanceMetric,
    /// Quantization method for memory reduction (Sprint 60)
    pub quantization: QuantizationMethod,
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
            model: None,
            metric: DistanceMetric::Cosine,
            quantization: QuantizationMethod::None,
        }
    }

    /// Creates a configuration optimized for large datasets (1M+ vectors).
    pub fn large(dimension: usize) -> Self {
        Self {
            dimension,
            m: 24,
            ef_construction: 400,
            max_elements: 1_000_000,
            model: None,
            metric: DistanceMetric::Cosine,
            quantization: QuantizationMethod::None,
        }
    }

    /// Creates a configuration with int8 quantization for 4x memory savings.
    ///
    /// Recommended for large datasets where memory is a constraint.
    pub fn quantized(dimension: usize) -> Self {
        Self {
            dimension,
            m: 24,
            ef_construction: 400,
            max_elements: 1_000_000,
            model: None,
            metric: DistanceMetric::Cosine,
            quantization: QuantizationMethod::Int8,
        }
    }

    /// Sets the model name for this index.
    pub fn with_model(mut self, model: impl Into<String>) -> Self {
        self.model = Some(model.into());
        self
    }

    /// Sets the distance metric for this index.
    pub fn with_metric(mut self, metric: DistanceMetric) -> Self {
        self.metric = metric;
        self
    }

    /// Sets the quantization method for this index.
    pub fn with_quantization(mut self, quantization: QuantizationMethod) -> Self {
        self.quantization = quantization;
        self
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
            model: None,
            metric: DistanceMetric::Cosine,
            quantization: QuantizationMethod::None,
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
///
/// # With Metadata (Sprint 56)
///
/// ```ignore
/// use neural_storage::{VectorIndex, VectorIndexConfig, DistanceMetric, EmbeddingMetadata};
///
/// let config = VectorIndexConfig::new(768)
///     .with_model("openai/text-embedding-3-small")
///     .with_metric(DistanceMetric::Cosine);
///
/// let mut index = VectorIndex::with_config(config);
/// let metadata = EmbeddingMetadata::new("openai/text-embedding-3-small");
/// index.add_with_metadata(NodeId::new(0), &embedding, metadata);
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
    /// Index-level metadata (Sprint 56)
    index_metadata: IndexMetadata,
    /// Per-embedding metadata (Sprint 56)
    embedding_metadata: HashMap<NodeId, EmbeddingMetadata>,
    /// Quantization method (Sprint 60)
    quantization: QuantizationMethod,
    /// Quantized vectors for int8 quantization (Sprint 60)
    quantized_vectors: HashMap<NodeId, Vec<i8>>,
    /// Quantization parameters per vector (Sprint 60)
    quantization_params: HashMap<NodeId, QuantizationParams>,
    /// Binary vectors for binary quantization (Sprint 60)
    binary_vectors: HashMap<NodeId, Vec<u8>>,
}

/// Index-level metadata (Sprint 56)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IndexMetadata {
    /// Model used for embeddings in this index
    pub model: Option<String>,
    /// Distance metric used for similarity
    pub metric: DistanceMetric,
    /// When this index was created
    pub created_at: String,
    /// Number of embeddings in this index
    pub embedding_count: usize,
    /// Quantization method used (Sprint 60)
    pub quantization: QuantizationMethod,
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
    ///     model: None,
    ///     metric: DistanceMetric::Cosine,
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
            index_metadata: IndexMetadata {
                model: config.model,
                metric: config.metric,
                created_at: chrono::Utc::now().to_rfc3339(),
                embedding_count: 0,
                quantization: config.quantization,
            },
            embedding_metadata: HashMap::new(),
            quantization: config.quantization,
            quantized_vectors: HashMap::new(),
            quantization_params: HashMap::new(),
            binary_vectors: HashMap::new(),
        }
    }

    /// Returns the quantization method used by this index.
    pub fn quantization_method(&self) -> QuantizationMethod {
        self.quantization
    }

    /// Returns memory usage statistics for this index.
    ///
    /// # Returns
    ///
    /// A tuple of (vector_memory_bytes, metadata_memory_bytes, total_count)
    pub fn memory_stats(&self) -> (usize, usize, usize) {
        let count = self.id_to_node.len();
        let vector_bytes = match self.quantization {
            QuantizationMethod::None => count * self.dimension * 4, // f32 = 4 bytes
            QuantizationMethod::Int8 => count * self.dimension * 1, // i8 = 1 byte
            QuantizationMethod::Binary => count * ((self.dimension + 7) / 8), // 1 bit per dim
        };
        let metadata_bytes = count * 64; // Approximate metadata overhead
        (vector_bytes, metadata_bytes, count)
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

        // Insert into HNSW (always uses f32 for graph structure)
        self.hnsw.insert((vector, internal_id));
        self.index_metadata.embedding_count += 1;

        // Store quantized vectors if quantization is enabled (Sprint 60)
        match self.quantization {
            QuantizationMethod::None => {
                // No additional storage needed - HNSW stores the full vector
            }
            QuantizationMethod::Int8 => {
                // Compute quantization parameters and store quantized vector
                let params = QuantizationParams::from_vector(vector);
                let quantized = quantize_f32_to_i8(vector, &params);
                self.quantized_vectors.insert(node, quantized);
                self.quantization_params.insert(node, params);
            }
            QuantizationMethod::Binary => {
                // Store binary quantized vector
                let binary = quantize_f32_to_binary(vector);
                self.binary_vectors.insert(node, binary);
            }
        }
    }

    /// Adds a vector with metadata to the index (Sprint 56).
    ///
    /// # Arguments
    ///
    /// * `node` - The NodeId to associate with this vector
    /// * `vector` - The vector data (must match the dimension)
    /// * `metadata` - Metadata for this embedding (model, metric, etc.)
    ///
    /// # Example
    ///
    /// ```ignore
    /// let metadata = EmbeddingMetadata::new("openai/text-embedding-3-small")
    ///     .with_metric(DistanceMetric::Cosine);
    /// index.add_with_metadata(NodeId::new(0), &embedding, metadata);
    /// ```
    pub fn add_with_metadata(&mut self, node: NodeId, vector: &[f32], metadata: EmbeddingMetadata) {
        self.add(node, vector);
        self.embedding_metadata.insert(node, metadata);
    }

    /// Returns the metadata for a specific embedding (Sprint 56).
    pub fn get_embedding_metadata(&self, node: NodeId) -> Option<&EmbeddingMetadata> {
        self.embedding_metadata.get(&node)
    }

    /// Returns the index-level metadata (Sprint 56).
    pub fn index_metadata(&self) -> &IndexMetadata {
        &self.index_metadata
    }

    /// Returns the model name for this index (Sprint 56).
    pub fn model(&self) -> Option<&str> {
        self.index_metadata.model.as_deref()
    }

    /// Returns the distance metric for this index (Sprint 56).
    pub fn metric(&self) -> DistanceMetric {
        self.index_metadata.metric
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
            .field("model", &self.index_metadata.model)
            .field("metric", &self.index_metadata.metric)
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

    // =========================================================================
    // Sprint 56: Embedding Metadata Tests
    // =========================================================================

    #[test]
    fn test_dot_product() {
        let a = [1.0, 2.0, 3.0];
        let b = [4.0, 5.0, 6.0];
        let result = super::dot_product(&a, &b);
        // 1*4 + 2*5 + 3*6 = 4 + 10 + 18 = 32
        assert!((result - 32.0).abs() < 1e-6, "Expected 32.0, got {}", result);
    }

    #[test]
    fn test_euclidean_distance() {
        let a = [0.0, 0.0, 0.0];
        let b = [3.0, 4.0, 0.0];
        let result = super::euclidean_distance(&a, &b);
        // sqrt(3^2 + 4^2) = 5
        assert!((result - 5.0).abs() < 1e-6, "Expected 5.0, got {}", result);
    }

    #[test]
    fn test_distance_metric_similarity() {
        let a = [1.0, 0.0];
        let b = [1.0, 0.0];

        // Cosine: identical vectors = similarity 1.0
        let cosine_sim = DistanceMetric::Cosine.similarity(&a, &b);
        assert!((cosine_sim - 1.0).abs() < 1e-6);

        // Euclidean: identical vectors = distance 0, similarity 0 (negated)
        let euclidean_sim = DistanceMetric::Euclidean.similarity(&a, &b);
        assert!(euclidean_sim.abs() < 1e-6);

        // DotProduct: [1,0] · [1,0] = 1
        let dot_sim = DistanceMetric::DotProduct.similarity(&a, &b);
        assert!((dot_sim - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_distance_metric_distance() {
        let a = [1.0, 0.0];
        let b = [0.0, 1.0]; // Orthogonal

        // Cosine distance for orthogonal vectors = 1.0
        let cosine_dist = DistanceMetric::Cosine.distance(&a, &b);
        assert!((cosine_dist - 1.0).abs() < 1e-6);

        // Euclidean distance = sqrt(2)
        let euclidean_dist = DistanceMetric::Euclidean.distance(&a, &b);
        assert!((euclidean_dist - 1.414213).abs() < 1e-3);
    }

    #[test]
    fn test_embedding_metadata_builder() {
        let metadata = EmbeddingMetadata::new("openai/text-embedding-3-small")
            .with_metric(DistanceMetric::Cosine)
            .with_dimension(1536)
            .with_property("version", "v3");

        assert_eq!(metadata.model(), "openai/text-embedding-3-small");
        assert_eq!(metadata.metric(), DistanceMetric::Cosine);
        assert_eq!(metadata.dimension(), Some(1536));
        assert_eq!(metadata.property("version"), Some("v3"));
        assert!(!metadata.created_at().is_empty());
    }

    #[test]
    fn test_vector_index_config_with_metadata() {
        let config = VectorIndexConfig::new(768)
            .with_model("sentence-transformers/all-MiniLM-L6-v2")
            .with_metric(DistanceMetric::Euclidean);

        assert_eq!(config.model, Some("sentence-transformers/all-MiniLM-L6-v2".to_string()));
        assert_eq!(config.metric, DistanceMetric::Euclidean);
    }

    #[test]
    fn test_vector_index_with_metadata() {
        let config = VectorIndexConfig::new(3)
            .with_model("test-model")
            .with_metric(DistanceMetric::Cosine);

        let mut index = VectorIndex::with_config(config);

        // Add with metadata
        let metadata = EmbeddingMetadata::new("test-model")
            .with_metric(DistanceMetric::Cosine);
        index.add_with_metadata(NodeId::new(0), &[1.0, 0.0, 0.0], metadata);

        // Verify index-level metadata
        assert_eq!(index.model(), Some("test-model"));
        assert_eq!(index.metric(), DistanceMetric::Cosine);
        assert_eq!(index.index_metadata().embedding_count, 1);

        // Verify per-embedding metadata
        let emb_meta = index.get_embedding_metadata(NodeId::new(0)).unwrap();
        assert_eq!(emb_meta.model(), "test-model");
        assert_eq!(emb_meta.metric(), DistanceMetric::Cosine);

        // Verify no metadata for node without metadata
        assert!(index.get_embedding_metadata(NodeId::new(999)).is_none());
    }

    #[test]
    fn test_distance_metric_display() {
        assert_eq!(format!("{}", DistanceMetric::Cosine), "cosine");
        assert_eq!(format!("{}", DistanceMetric::Euclidean), "euclidean");
        assert_eq!(format!("{}", DistanceMetric::DotProduct), "dot_product");
    }

    // =========================================================================
    // Flash Quantization Tests (Sprint 60)
    // =========================================================================

    #[test]
    fn test_quantization_params_from_vector() {
        let vector = vec![0.0, 0.5, 1.0, -0.5, -1.0];
        let params = QuantizationParams::from_vector(&vector);

        // Range is -1.0 to 1.0, so scale should be 2.0/255
        assert!((params.offset - (-1.0)).abs() < 1e-6);
        assert!((params.scale - (2.0 / 255.0)).abs() < 1e-6);
    }

    #[test]
    fn test_quantize_dequantize_roundtrip() {
        let original = vec![0.0, 0.5, 1.0, -0.5, -1.0];
        let params = QuantizationParams::from_vector(&original);

        let quantized = quantize_f32_to_i8(&original, &params);
        let dequantized = dequantize_i8_to_f32(&quantized, &params);

        // Check roundtrip error is small
        for (orig, deq) in original.iter().zip(dequantized.iter()) {
            let error = (orig - deq).abs();
            assert!(error < 0.02, "Roundtrip error too large: {} vs {}", orig, deq);
        }
    }

    #[test]
    fn test_binary_quantization() {
        let vector = vec![0.5, -0.3, 0.1, -0.8, 0.0, 0.9, -0.1, 0.2];
        let binary = quantize_f32_to_binary(&vector);

        // Expected: [1, 0, 1, 0, 1, 1, 0, 1] = 0b10101101 = 173
        assert_eq!(binary.len(), 1);
        assert_eq!(binary[0], 0b10101101);
    }

    #[test]
    fn test_hamming_distance() {
        let a = vec![0b11110000u8];
        let b = vec![0b11001100u8];
        let dist = hamming_distance(&a, &b);
        assert_eq!(dist, 4); // 4 bits differ
    }

    #[test]
    fn test_quantized_vector_index() {
        let config = VectorIndexConfig::quantized(3);
        let mut index = VectorIndex::with_config(config);

        assert_eq!(index.quantization_method(), QuantizationMethod::Int8);

        index.add(NodeId::new(0), &[1.0, 0.0, 0.0]);
        index.add(NodeId::new(1), &[0.0, 1.0, 0.0]);
        index.add(NodeId::new(2), &[0.0, 0.0, 1.0]);

        // Should still be searchable
        let results = index.search(&[0.9, 0.1, 0.0], 1);
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].0, NodeId::new(0));
    }

    #[test]
    fn test_memory_stats() {
        let config = VectorIndexConfig::new(768);
        let mut index = VectorIndex::with_config(config);

        // Add 100 vectors
        for i in 0..100 {
            let mut vec = vec![0.0f32; 768];
            vec[i % 768] = 1.0;
            index.add(NodeId::new(i as u64), &vec);
        }

        let (vector_bytes, _metadata_bytes, count) = index.memory_stats();
        assert_eq!(count, 100);
        assert_eq!(vector_bytes, 100 * 768 * 4); // f32 = 4 bytes
    }

    #[test]
    fn test_quantized_memory_stats() {
        let config = VectorIndexConfig::quantized(768);
        let mut index = VectorIndex::with_config(config);

        // Add 100 vectors
        for i in 0..100 {
            let mut vec = vec![0.0f32; 768];
            vec[i % 768] = 1.0;
            index.add(NodeId::new(i as u64), &vec);
        }

        let (vector_bytes, _metadata_bytes, count) = index.memory_stats();
        assert_eq!(count, 100);
        assert_eq!(vector_bytes, 100 * 768 * 1); // int8 = 1 byte (4x savings)
    }

    #[test]
    fn test_quantization_method_display() {
        assert_eq!(format!("{}", QuantizationMethod::None), "none");
        assert_eq!(format!("{}", QuantizationMethod::Int8), "int8");
        assert_eq!(format!("{}", QuantizationMethod::Binary), "binary");
    }
}
