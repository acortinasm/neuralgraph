//! gRPC client for remote vector index operations.
//!
//! Provides connection pooling and client management for distributed search.

use crate::sharding::ShardId;
use crate::vector_index::DistanceMetric;
use dashmap::DashMap;
use neural_core::NodeId;
use std::sync::Arc;
use std::time::Duration;
use thiserror::Error;
use tonic::transport::{Channel, Endpoint};

/// Include the generated protobuf code for vector service.
pub mod proto {
    tonic::include_proto!("vector");
}

/// Errors that can occur during vector client operations.
#[derive(Error, Debug)]
pub enum VectorClientError {
    /// Connection failed.
    #[error("Connection failed to {addr}: {message}")]
    ConnectionFailed { addr: String, message: String },

    /// Request timed out.
    #[error("Request timed out after {timeout_ms}ms")]
    Timeout { timeout_ms: u64 },

    /// gRPC error.
    #[error("gRPC error: {0}")]
    Grpc(#[from] tonic::Status),

    /// Transport error.
    #[error("Transport error: {0}")]
    Transport(#[from] tonic::transport::Error),

    /// Serialization error.
    #[error("Serialization error: {0}")]
    Serialization(String),

    /// Server returned an error.
    #[error("Server error: {0}")]
    ServerError(String),
}

/// Client for remote vector index operations on a single shard.
pub struct VectorShardClient {
    /// gRPC client.
    client: proto::vector_service_client::VectorServiceClient<Channel>,
    /// Shard ID this client is connected to.
    shard_id: ShardId,
    /// Server address.
    addr: String,
    /// Request timeout.
    timeout: Duration,
    /// Distance metric to use.
    metric: DistanceMetric,
}

impl VectorShardClient {
    /// Creates a new client connected to the given address.
    pub async fn connect(
        shard_id: ShardId,
        addr: &str,
        timeout: Duration,
    ) -> Result<Self, VectorClientError> {
        let endpoint = Endpoint::from_shared(format!("http://{}", addr))
            .map_err(|e| VectorClientError::ConnectionFailed {
                addr: addr.to_string(),
                message: e.to_string(),
            })?
            .timeout(timeout)
            .connect_timeout(Duration::from_secs(5));

        let channel = endpoint.connect().await?;
        let client = proto::vector_service_client::VectorServiceClient::new(channel);

        Ok(Self {
            client,
            shard_id,
            addr: addr.to_string(),
            timeout,
            metric: DistanceMetric::Cosine,
        })
    }

    /// Sets the distance metric for search requests.
    pub fn with_metric(mut self, metric: DistanceMetric) -> Self {
        self.metric = metric;
        self
    }

    /// Returns the shard ID this client is connected to.
    pub fn shard_id(&self) -> ShardId {
        self.shard_id
    }

    /// Returns the server address.
    pub fn addr(&self) -> &str {
        &self.addr
    }

    /// Searches for k nearest neighbors on this shard.
    pub async fn search(
        &mut self,
        query: &[f32],
        k: usize,
    ) -> Result<Vec<(NodeId, f32)>, VectorClientError> {
        self.search_with_ef(query, k, None).await
    }

    /// Searches with custom ef_search parameter.
    pub async fn search_with_ef(
        &mut self,
        query: &[f32],
        k: usize,
        ef_search: Option<u32>,
    ) -> Result<Vec<(NodeId, f32)>, VectorClientError> {
        let request = proto::VectorSearchRequest {
            query_vector: serialize_f32_vec(query),
            k: k as u32,
            ef_search: ef_search.unwrap_or(0),
            metric: self.metric.to_string(),
        };

        let response = self.client.search(request).await?.into_inner();

        let results = response
            .results
            .into_iter()
            .map(|r| (NodeId::new(r.node_id), r.score))
            .collect();

        Ok(results)
    }

    /// Adds a single vector to the remote index.
    pub async fn add(
        &mut self,
        node_id: NodeId,
        vector: &[f32],
    ) -> Result<(), VectorClientError> {
        let request = proto::VectorAddRequest {
            node_id: node_id.as_u64(),
            vector: serialize_f32_vec(vector),
        };

        let response = self.client.add(request).await?.into_inner();

        if response.success {
            Ok(())
        } else {
            Err(VectorClientError::ServerError(response.error))
        }
    }

    /// Adds multiple vectors in batch.
    pub async fn batch_add(
        &mut self,
        entries: Vec<(NodeId, Vec<f32>)>,
    ) -> Result<(u32, u32), VectorClientError> {
        let request = proto::VectorBatchAddRequest {
            entries: entries
                .into_iter()
                .map(|(node_id, vector)| proto::VectorEntry {
                    node_id: node_id.as_u64(),
                    vector: serialize_f32_vec(&vector),
                })
                .collect(),
        };

        let response = self.client.batch_add(request).await?.into_inner();

        Ok((response.added_count, response.failed_count))
    }

    /// Gets index statistics from the remote shard.
    pub async fn get_stats(&mut self) -> Result<proto::VectorStatsResponse, VectorClientError> {
        let request = proto::VectorStatsRequest {};
        let response = self.client.get_stats(request).await?.into_inner();
        Ok(response)
    }
}

impl std::fmt::Debug for VectorShardClient {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("VectorShardClient")
            .field("shard_id", &self.shard_id)
            .field("addr", &self.addr)
            .field("timeout", &self.timeout)
            .finish()
    }
}

/// Pool of vector shard clients for connection reuse.
///
/// Provides thread-safe access to clients for multiple shards,
/// with automatic connection management.
pub struct VectorClientPool {
    /// Clients keyed by shard ID.
    clients: DashMap<ShardId, Arc<tokio::sync::Mutex<VectorShardClient>>>,
    /// Shard addresses for lazy connection.
    addresses: DashMap<ShardId, String>,
    /// Default timeout for connections.
    timeout: Duration,
    /// Distance metric.
    metric: DistanceMetric,
}

impl VectorClientPool {
    /// Creates a new empty client pool.
    pub fn new(timeout: Duration) -> Self {
        Self {
            clients: DashMap::new(),
            addresses: DashMap::new(),
            timeout,
            metric: DistanceMetric::Cosine,
        }
    }

    /// Sets the distance metric for all clients.
    pub fn with_metric(mut self, metric: DistanceMetric) -> Self {
        self.metric = metric;
        self
    }

    /// Registers a shard address for lazy connection.
    pub fn register_shard(&self, shard_id: ShardId, addr: String) {
        self.addresses.insert(shard_id, addr);
    }

    /// Gets or creates a client for the given shard.
    pub async fn get_client(
        &self,
        shard_id: ShardId,
    ) -> Result<Arc<tokio::sync::Mutex<VectorShardClient>>, VectorClientError> {
        // Check if client already exists
        if let Some(client) = self.clients.get(&shard_id) {
            return Ok(Arc::clone(&client));
        }

        // Create new client
        let addr = self
            .addresses
            .get(&shard_id)
            .ok_or_else(|| VectorClientError::ConnectionFailed {
                addr: format!("shard_{}", shard_id),
                message: "No address registered for shard".to_string(),
            })?
            .clone();

        let client = VectorShardClient::connect(shard_id, &addr, self.timeout)
            .await?
            .with_metric(self.metric);

        let client = Arc::new(tokio::sync::Mutex::new(client));
        self.clients.insert(shard_id, Arc::clone(&client));

        Ok(client)
    }

    /// Searches on a specific shard.
    pub async fn search(
        &self,
        shard_id: ShardId,
        query: &[f32],
        k: usize,
    ) -> Result<Vec<(NodeId, f32)>, VectorClientError> {
        let client = self.get_client(shard_id).await?;
        let mut client = client.lock().await;
        client.search(query, k).await
    }

    /// Searches at a specific replica address.
    ///
    /// Creates a new connection to the specified address if one doesn't exist.
    /// Used for replica failover when the primary is unavailable.
    ///
    /// # Arguments
    ///
    /// * `shard_id` - The shard ID (for context)
    /// * `addr` - The replica address to connect to
    /// * `query` - The query vector
    /// * `k` - Number of neighbors to return
    pub async fn search_at_addr(
        &self,
        shard_id: ShardId,
        addr: &str,
        query: &[f32],
        k: usize,
    ) -> Result<Vec<(NodeId, f32)>, VectorClientError> {
        // Check if we have this address registered for this shard
        let is_registered = self
            .addresses
            .get(&shard_id)
            .map(|a| a.value() == addr)
            .unwrap_or(false);

        if !is_registered {
            // This is a different address (replica), create a direct connection
            let mut client = VectorShardClient::connect(shard_id, addr, self.timeout)
                .await?
                .with_metric(self.metric);
            return client.search(query, k).await;
        }

        // Use existing connection pool
        self.search(shard_id, query, k).await
    }

    /// Returns the number of connected clients.
    pub fn connected_count(&self) -> usize {
        self.clients.len()
    }

    /// Returns the number of registered shards.
    pub fn registered_count(&self) -> usize {
        self.addresses.len()
    }

    /// Removes a client connection (e.g., when shard becomes unhealthy).
    pub fn disconnect(&self, shard_id: ShardId) {
        self.clients.remove(&shard_id);
    }

    /// Clears all connections.
    pub fn clear(&self) {
        self.clients.clear();
    }
}

impl std::fmt::Debug for VectorClientPool {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("VectorClientPool")
            .field("connected", &self.clients.len())
            .field("registered", &self.addresses.len())
            .field("timeout", &self.timeout)
            .finish()
    }
}

/// Serializes a f32 vector to bytes (little-endian).
fn serialize_f32_vec(vector: &[f32]) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(vector.len() * 4);
    for &v in vector {
        bytes.extend_from_slice(&v.to_le_bytes());
    }
    bytes
}

/// Deserializes bytes to a f32 vector (little-endian).
#[allow(dead_code)]
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

    #[test]
    fn test_serialize_deserialize_f32() {
        let original = vec![1.0f32, -2.5, 0.0, 3.14159];
        let bytes = serialize_f32_vec(&original);
        let restored = deserialize_f32_vec(&bytes);

        assert_eq!(original.len(), restored.len());
        for (a, b) in original.iter().zip(restored.iter()) {
            assert!((a - b).abs() < 1e-6);
        }
    }

    #[test]
    fn test_client_pool_registration() {
        let pool = VectorClientPool::new(Duration::from_secs(5));

        pool.register_shard(0, "localhost:50051".to_string());
        pool.register_shard(1, "localhost:50052".to_string());

        assert_eq!(pool.registered_count(), 2);
        assert_eq!(pool.connected_count(), 0);
    }

    #[test]
    fn test_vector_client_error_display() {
        let err = VectorClientError::ConnectionFailed {
            addr: "localhost:50051".to_string(),
            message: "Connection refused".to_string(),
        };
        assert!(err.to_string().contains("localhost:50051"));

        let err = VectorClientError::Timeout { timeout_ms: 5000 };
        assert!(err.to_string().contains("5000ms"));
    }
}
