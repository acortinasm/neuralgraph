//! gRPC server for vector operations on a single shard.
//!
//! Provides a gRPC interface for distributed vector search, allowing remote
//! nodes to perform vector operations via RPC.
//!
//! # Services
//!
//! - `Search`: k-nearest neighbor search
//! - `Add`: Add a single vector
//! - `BatchAdd`: Add multiple vectors
//! - `GetStats`: Get index statistics
//! - `HealthCheck`: Check shard health status

use super::core::VectorIndex;
use crate::sharding::ShardId;
use neural_core::NodeId;
use std::sync::{Arc, RwLock};
use std::time::Instant;
use tonic::{Request, Response, Status};

/// Include the generated protobuf code for vector service.
pub mod proto {
    tonic::include_proto!("vector");
}

use proto::vector_service_server::VectorService;
use proto::*;

/// gRPC server for vector operations on a single shard.
///
/// Wraps a `VectorIndex` and exposes its functionality via gRPC.
/// Thread-safe through interior mutability (RwLock).
///
/// # Example
///
/// ```ignore
/// use neural_storage::vector_index::{VectorGrpcServer, VectorIndex};
/// use std::sync::{Arc, RwLock};
///
/// let index = VectorIndex::new(768);
/// let server = VectorGrpcServer::new(
///     Arc::new(RwLock::new(index)),
///     0, // shard_id
/// );
///
/// // Start tonic server with VectorServiceServer::new(server)
/// ```
#[derive(Clone)]
pub struct VectorGrpcServer {
    /// The underlying vector index.
    index: Arc<RwLock<VectorIndex>>,
    /// Shard ID this server is responsible for.
    shard_id: ShardId,
}

impl VectorGrpcServer {
    /// Creates a new gRPC server for vector operations.
    ///
    /// # Arguments
    ///
    /// * `index` - The vector index to wrap (thread-safe via RwLock)
    /// * `shard_id` - The shard ID this server is responsible for
    pub fn new(index: Arc<RwLock<VectorIndex>>, shard_id: ShardId) -> Self {
        Self { index, shard_id }
    }

    /// Returns the shard ID this server is responsible for.
    pub fn shard_id(&self) -> ShardId {
        self.shard_id
    }
}

#[tonic::async_trait]
impl VectorService for VectorGrpcServer {
    /// Searches for k nearest neighbors to a query vector.
    async fn search(
        &self,
        request: Request<VectorSearchRequest>,
    ) -> Result<Response<VectorSearchResponse>, Status> {
        let start = Instant::now();
        let req = request.into_inner();

        // Deserialize query vector
        let query = deserialize_f32_vec(&req.query_vector);
        if query.is_empty() {
            return Err(Status::invalid_argument("Empty query vector"));
        }

        let k = req.k as usize;
        if k == 0 {
            return Err(Status::invalid_argument("k must be greater than 0"));
        }

        // Perform search
        let results = {
            let index = self.index.read().map_err(|e| {
                Status::internal(format!("Failed to acquire read lock: {}", e))
            })?;

            // Validate dimension
            if query.len() != index.dimension() {
                return Err(Status::invalid_argument(format!(
                    "Query dimension mismatch: expected {}, got {}",
                    index.dimension(),
                    query.len()
                )));
            }

            index.search(&query, k)
        };

        // Convert results to proto format
        let proto_results: Vec<VectorResult> = results
            .into_iter()
            .map(|(node_id, score)| VectorResult {
                node_id: node_id.as_u64(),
                score,
            })
            .collect();

        let execution_time_us = start.elapsed().as_micros() as u64;
        let index_size = {
            let index = self.index.read().map_err(|e| {
                Status::internal(format!("Failed to acquire read lock: {}", e))
            })?;
            index.len() as u64
        };

        Ok(Response::new(VectorSearchResponse {
            results: proto_results,
            execution_time_us,
            index_size,
        }))
    }

    /// Adds a single vector to the index.
    async fn add(
        &self,
        request: Request<VectorAddRequest>,
    ) -> Result<Response<VectorAddResponse>, Status> {
        let req = request.into_inner();
        let node_id = NodeId::new(req.node_id);
        let vector = deserialize_f32_vec(&req.vector);

        if vector.is_empty() {
            return Err(Status::invalid_argument("Empty vector"));
        }

        let result = {
            let mut index = self.index.write().map_err(|e| {
                Status::internal(format!("Failed to acquire write lock: {}", e))
            })?;

            // Validate dimension
            if vector.len() != index.dimension() {
                return Ok(Response::new(VectorAddResponse {
                    success: false,
                    error: format!(
                        "Dimension mismatch: expected {}, got {}",
                        index.dimension(),
                        vector.len()
                    ),
                }));
            }

            index.add(node_id, &vector);
            Ok(())
        };

        match result {
            Ok(()) => Ok(Response::new(VectorAddResponse {
                success: true,
                error: String::new(),
            })),
            Err(e) => Ok(Response::new(VectorAddResponse {
                success: false,
                error: e,
            })),
        }
    }

    /// Adds multiple vectors in batch.
    async fn batch_add(
        &self,
        request: Request<VectorBatchAddRequest>,
    ) -> Result<Response<VectorBatchAddResponse>, Status> {
        let req = request.into_inner();

        let mut added_count = 0u32;
        let mut failed_count = 0u32;
        let mut errors = std::collections::HashMap::new();

        let mut index = self.index.write().map_err(|e| {
            Status::internal(format!("Failed to acquire write lock: {}", e))
        })?;

        for entry in req.entries {
            let node_id = NodeId::new(entry.node_id);
            let vector = deserialize_f32_vec(&entry.vector);

            if vector.is_empty() {
                failed_count += 1;
                errors.insert(entry.node_id, "Empty vector".to_string());
                continue;
            }

            if vector.len() != index.dimension() {
                failed_count += 1;
                errors.insert(
                    entry.node_id,
                    format!(
                        "Dimension mismatch: expected {}, got {}",
                        index.dimension(),
                        vector.len()
                    ),
                );
                continue;
            }

            index.add(node_id, &vector);
            added_count += 1;
        }

        Ok(Response::new(VectorBatchAddResponse {
            added_count,
            failed_count,
            errors,
        }))
    }

    /// Gets index statistics.
    async fn get_stats(
        &self,
        _request: Request<VectorStatsRequest>,
    ) -> Result<Response<VectorStatsResponse>, Status> {
        let index = self.index.read().map_err(|e| {
            Status::internal(format!("Failed to acquire read lock: {}", e))
        })?;

        let (vector_memory_bytes, metadata_memory_bytes, vector_count) = index.memory_stats();

        Ok(Response::new(VectorStatsResponse {
            vector_count: vector_count as u64,
            dimension: index.dimension() as u32,
            vector_memory_bytes: vector_memory_bytes as u64,
            metadata_memory_bytes: metadata_memory_bytes as u64,
            quantization: index.quantization_method().to_string(),
            metric: index.metric().to_string(),
        }))
    }

    /// Health check for this shard.
    async fn health_check(
        &self,
        request: Request<VectorHealthCheckRequest>,
    ) -> Result<Response<VectorHealthCheckResponse>, Status> {
        let req = request.into_inner();

        // Verify shard ID matches
        if req.shard_id != self.shard_id {
            return Ok(Response::new(VectorHealthCheckResponse {
                healthy: false,
                vector_count: 0,
                status: format!(
                    "Shard ID mismatch: expected {}, got {}",
                    self.shard_id, req.shard_id
                ),
            }));
        }

        // Check if we can access the index
        match self.index.read() {
            Ok(index) => Ok(Response::new(VectorHealthCheckResponse {
                healthy: true,
                vector_count: index.len() as u64,
                status: "ready".to_string(),
            })),
            Err(_) => Ok(Response::new(VectorHealthCheckResponse {
                healthy: false,
                vector_count: 0,
                status: "error: lock poisoned".to_string(),
            })),
        }
    }
}

impl std::fmt::Debug for VectorGrpcServer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("VectorGrpcServer")
            .field("shard_id", &self.shard_id)
            .finish_non_exhaustive()
    }
}

// =============================================================================
// Helper Functions
// =============================================================================

/// Deserializes bytes to a f32 vector (little-endian).
fn deserialize_f32_vec(bytes: &[u8]) -> Vec<f32> {
    bytes
        .chunks_exact(4)
        .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
        .collect()
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::vector_index::VectorIndexConfig;

    fn create_test_server() -> VectorGrpcServer {
        let index = VectorIndex::with_config(VectorIndexConfig::small(3));
        VectorGrpcServer::new(Arc::new(RwLock::new(index)), 0)
    }

    fn serialize_f32_vec(vector: &[f32]) -> Vec<u8> {
        let mut bytes = Vec::with_capacity(vector.len() * 4);
        for &v in vector {
            bytes.extend_from_slice(&v.to_le_bytes());
        }
        bytes
    }

    #[tokio::test]
    async fn test_add_and_search() {
        let server = create_test_server();

        // Add a vector
        let add_request = Request::new(VectorAddRequest {
            node_id: 1,
            vector: serialize_f32_vec(&[1.0, 0.0, 0.0]),
        });
        let add_response = server.add(add_request).await.unwrap();
        assert!(add_response.get_ref().success);

        // Search for it
        let search_request = Request::new(VectorSearchRequest {
            query_vector: serialize_f32_vec(&[1.0, 0.0, 0.0]),
            k: 1,
            ef_search: 0,
            metric: "cosine".to_string(),
        });
        let search_response = server.search(search_request).await.unwrap();
        let results = &search_response.get_ref().results;

        assert_eq!(results.len(), 1);
        assert_eq!(results[0].node_id, 1);
        assert!(results[0].score > 0.99);
    }

    #[tokio::test]
    async fn test_batch_add() {
        let server = create_test_server();

        let batch_request = Request::new(VectorBatchAddRequest {
            entries: vec![
                VectorEntry {
                    node_id: 1,
                    vector: serialize_f32_vec(&[1.0, 0.0, 0.0]),
                },
                VectorEntry {
                    node_id: 2,
                    vector: serialize_f32_vec(&[0.0, 1.0, 0.0]),
                },
            ],
        });

        let response = server.batch_add(batch_request).await.unwrap();
        assert_eq!(response.get_ref().added_count, 2);
        assert_eq!(response.get_ref().failed_count, 0);
    }

    #[tokio::test]
    async fn test_get_stats() {
        let server = create_test_server();

        // Add some vectors
        {
            let mut index = server.index.write().unwrap();
            index.add(NodeId::new(1), &[1.0, 0.0, 0.0]);
            index.add(NodeId::new(2), &[0.0, 1.0, 0.0]);
        }

        let stats_request = Request::new(VectorStatsRequest {});
        let response = server.get_stats(stats_request).await.unwrap();

        assert_eq!(response.get_ref().vector_count, 2);
        assert_eq!(response.get_ref().dimension, 3);
    }

    #[tokio::test]
    async fn test_health_check() {
        let server = create_test_server();

        // Correct shard ID
        let request = Request::new(VectorHealthCheckRequest { shard_id: 0 });
        let response = server.health_check(request).await.unwrap();
        assert!(response.get_ref().healthy);
        assert_eq!(response.get_ref().status, "ready");

        // Wrong shard ID
        let request = Request::new(VectorHealthCheckRequest { shard_id: 99 });
        let response = server.health_check(request).await.unwrap();
        assert!(!response.get_ref().healthy);
    }

    #[tokio::test]
    async fn test_dimension_mismatch() {
        let server = create_test_server();

        // Try to add a 2D vector to a 3D index
        let add_request = Request::new(VectorAddRequest {
            node_id: 1,
            vector: serialize_f32_vec(&[1.0, 0.0]), // Wrong dimension
        });
        let add_response = server.add(add_request).await.unwrap();
        assert!(!add_response.get_ref().success);
        assert!(add_response.get_ref().error.contains("mismatch"));
    }

    #[tokio::test]
    async fn test_empty_vector_error() {
        let server = create_test_server();

        let add_request = Request::new(VectorAddRequest {
            node_id: 1,
            vector: vec![],
        });
        let result = server.add(add_request).await;
        assert!(result.is_err() || !result.unwrap().get_ref().success);
    }

    #[test]
    fn test_deserialize_f32_vec() {
        let original = vec![1.0f32, -2.5, 0.0, 3.14159];
        let bytes = serialize_f32_vec(&original);
        let restored = deserialize_f32_vec(&bytes);

        assert_eq!(original.len(), restored.len());
        for (a, b) in original.iter().zip(restored.iter()) {
            assert!((a - b).abs() < 1e-6);
        }
    }
}
