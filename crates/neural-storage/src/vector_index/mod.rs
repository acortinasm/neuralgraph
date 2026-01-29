//! Vector index module for approximate nearest neighbor search.
//!
//! This module provides:
//! - HNSW-based vector index for O(log n) similarity search
//! - Distributed vector search across multiple shards (Sprint 60)
//! - Query result caching with LRU eviction
//! - Load balancing across replicas
//!
//! # Architecture
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────────┐
//! │                     DistributedVectorIndex                       │
//! ├─────────────────────────────────────────────────────────────────┤
//! │  ┌─────────────┐  ┌─────────────────┐  ┌──────────────────────┐ │
//! │  │ QueryCache  │  │  LoadBalancer   │  │ VectorSearchCoord    │ │
//! │  │   (LRU)     │  │ (Round-Robin+)  │  │ (Scatter-Gather)     │ │
//! │  └─────────────┘  └─────────────────┘  └──────────────────────┘ │
//! │                                                                  │
//! │  ┌──────────────────────────────────────────────────────────┐   │
//! │  │                    ShardClients                           │   │
//! │  │  ┌────────┐  ┌────────┐  ┌────────┐  ┌────────┐         │   │
//! │  │  │Shard 0 │  │Shard 1 │  │Shard 2 │  │Shard 3 │         │   │
//! │  │  │(gRPC)  │  │(gRPC)  │  │(gRPC)  │  │(gRPC)  │         │   │
//! │  │  └────────┘  └────────┘  └────────┘  └────────┘         │   │
//! │  └──────────────────────────────────────────────────────────┘   │
//! └─────────────────────────────────────────────────────────────────┘
//! ```

mod core;
mod cache;
mod client;
mod distributed;
mod load_balancer;
mod server;

// Re-export core types
pub use core::{
    // Distance metrics
    DistanceMetric,
    cosine_similarity,
    cosine_distance,
    dot_product,
    euclidean_distance,
    // Quantization (Sprint 60)
    QuantizationMethod,
    QuantizationParams,
    quantize_f32_to_i8,
    dequantize_i8_to_f32,
    quantize_f32_to_binary,
    hamming_distance,
    asymmetric_distance_i8,
    // Metadata (Sprint 56)
    EmbeddingMetadata,
    IndexMetadata,
    // Index
    VectorIndex,
    VectorIndexConfig,
};

// Re-export distributed types (Sprint 60)
pub use cache::{QueryResultCache, CacheStats};
pub use client::{VectorShardClient, VectorClientPool, VectorClientError};
pub use distributed::{
    DistributedVectorIndex,
    DistributedVectorConfig,
    ShardSearchResult,
    SimulatedDistributedIndex,
};
pub use load_balancer::{
    LoadBalancer,
    RoundRobinBalancer,
    LatencyAwareBalancer,
    WeightedBalancer,
};

// Re-export gRPC server types (Sprint 61)
pub use server::VectorGrpcServer;
pub use server::proto::vector_service_server::VectorServiceServer;
