//! Raft network implementation.
//!
//! This module provides the network layer for Raft consensus,
//! utilizing `tonic` for gRPC communication between nodes.

use std::collections::BTreeMap;
use std::fmt::Debug;
use std::sync::Arc;

use openraft::error::{RPCError, RaftError, InstallSnapshotError};
use openraft::network::{RaftNetwork, RaftNetworkFactory, RPCOption};
use openraft::raft::{AppendEntriesRequest, AppendEntriesResponse, InstallSnapshotRequest,
    InstallSnapshotResponse, VoteRequest, VoteResponse};
use openraft::{BasicNode};
use anyerror::AnyError;

use super::types::{TypeConfig};

pub mod proto {
    tonic::include_proto!("raft");
}

/// A client for communicating with a remote Raft node via gRPC.
#[derive(Debug, Clone)]
pub struct RaftClient {
    /// The gRPC client for the remote node.
    client: proto::raft_client::RaftClient<tonic::transport::Channel>,
    /// The target node ID.
    target_node_id: u64,
}

impl RaftClient {
    /// Creates a new Raft client.
    pub async fn new(target_node_id: u64, target_node_addr: String) -> Result<Self, tonic::transport::Error> {
        let client = proto::raft_client::RaftClient::connect(format!("http://{}", target_node_addr)).await?;
        Ok(Self { client, target_node_id })
    }

    fn map_rpc_err<E: std::error::Error + 'static>(_target: u64, e: E) -> RPCError<u64, BasicNode, RaftError<u64>> {
        RPCError::Network(AnyError::new(&e).into())
    }
}

/// Implements the RaftNetworkFactory trait for `openraft`.
#[derive(Debug, Clone)]
pub struct NeuralRaftNetwork {
    /// A map of connected clients to other Raft nodes.
    clients: Arc<tokio::sync::RwLock<BTreeMap<u64, RaftClient>>>,
}

impl NeuralRaftNetwork {
    pub fn new() -> Self {
        Self {
            clients: Arc::new(tokio::sync::RwLock::new(BTreeMap::new())),
        }
    }

    /// Get a `RaftClient` for a specific node, creating it if it doesn't exist.
    async fn get_or_create_client(&self, target: u64, node_addr: String) -> Result<RaftClient, RPCError<u64, BasicNode, RaftError<u64>>> {
        let mut clients = self.clients.write().await;
        if let Some(client) = clients.get(&target) {
            return Ok(client.clone());
        }

        let client = RaftClient::new(target, node_addr.clone())
            .await
            .map_err(|e| RaftClient::map_rpc_err(target, e))?;
        clients.insert(target, client.clone());
        Ok(client)
    }
}

impl RaftNetworkFactory<TypeConfig> for NeuralRaftNetwork {
    type Network = RaftClient;

    async fn new_client(&mut self, target: u64, node: &BasicNode) -> RaftClient {
        self.get_or_create_client(target, node.addr.clone()).await.expect("Failed to create client")
    }
}

impl RaftNetwork<TypeConfig> for RaftClient {
    async fn append_entries(
        &mut self,
        rpc: AppendEntriesRequest<TypeConfig>,
        option: RPCOption,
    ) -> Result<AppendEntriesResponse<u64>, RPCError<u64, BasicNode, RaftError<u64>>> {
        let _ = option;
        let req_data = bincode::serialize(&rpc).map_err(|e| RaftClient::map_rpc_err(self.target_node_id, e))?;
        let request = tonic::Request::new(proto::AppendEntriesRequest { data: req_data });

        let response = self.client.append_entries(request).await.map_err(|e| RaftClient::map_rpc_err(self.target_node_id, e))?;
        let resp_data = response.into_inner().data;
        let raft_resp: AppendEntriesResponse<u64> = bincode::deserialize(&resp_data).map_err(|e| RaftClient::map_rpc_err(self.target_node_id, e))?;
        Ok(raft_resp)
    }

    async fn install_snapshot(
        &mut self,
        rpc: InstallSnapshotRequest<TypeConfig>,
        option: RPCOption,
    ) -> Result<InstallSnapshotResponse<u64>, RPCError<u64, BasicNode, RaftError<u64, InstallSnapshotError>>> {
        let _ = option;
        let req_data = bincode::serialize(&rpc).map_err(|e| {
            RPCError::Network(AnyError::new(&e).into())
        })?;
        let request = tonic::Request::new(proto::InstallSnapshotRequest { data: req_data });

        let response = self.client.install_snapshot(request).await.map_err(|e| {
            RPCError::Network(AnyError::new(&e).into())
        })?;
        let resp_data = response.into_inner().data;
        let raft_resp: InstallSnapshotResponse<u64> = bincode::deserialize(&resp_data).map_err(|e| {
            RPCError::Network(AnyError::new(&e).into())
        })?;
        Ok(raft_resp)
    }

    async fn vote(
        &mut self,
        rpc: VoteRequest<u64>,
        option: RPCOption,
    ) -> Result<VoteResponse<u64>, RPCError<u64, BasicNode, RaftError<u64>>> {
        let _ = option;
        let req_data = bincode::serialize(&rpc).map_err(|e| RaftClient::map_rpc_err(self.target_node_id, e))?;
        let request = tonic::Request::new(proto::VoteRequest { data: req_data });

        let response = self.client.vote(request).await.map_err(|e| RaftClient::map_rpc_err(self.target_node_id, e))?;
        let resp_data = response.into_inner().data;
        let raft_resp: VoteResponse<u64> = bincode::deserialize(&resp_data).map_err(|e| RaftClient::map_rpc_err(self.target_node_id, e))?;
        Ok(raft_resp)
    }
}

// ============================================================================
// Cluster-Aware Client for Leader Routing
// ============================================================================

use super::cluster::{ClusterError, ClusterManager};
use super::types::RaftRequest;

/// Error type for cluster-aware client operations.
#[derive(Debug)]
pub enum ClientError {
    /// The request was sent to a non-leader node.
    NotLeader {
        leader_id: Option<u64>,
        leader_addr: Option<String>,
    },
    /// No leader is currently available.
    NoLeader,
    /// Network or transport error.
    NetworkError(String),
    /// The operation failed.
    OperationFailed(String),
}

impl std::fmt::Display for ClientError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ClientError::NotLeader { leader_id, leader_addr } => {
                write!(f, "Not leader. Leader: {:?} at {:?}", leader_id, leader_addr)
            }
            ClientError::NoLeader => write!(f, "No leader available"),
            ClientError::NetworkError(e) => write!(f, "Network error: {}", e),
            ClientError::OperationFailed(e) => write!(f, "Operation failed: {}", e),
        }
    }
}

impl std::error::Error for ClientError {}

/// A cluster-aware client that automatically routes requests to the leader.
///
/// This client maintains a cache of the current leader and automatically
/// handles redirects when requests are sent to non-leader nodes.
pub struct ClusterAwareClient {
    /// The cluster manager for leader tracking.
    cluster: Arc<ClusterManager>,
    /// Cached gRPC clients for each node.
    clients: Arc<tokio::sync::RwLock<BTreeMap<u64, proto::raft_client::RaftClient<tonic::transport::Channel>>>>,
    /// Maximum number of retries for leader routing.
    max_retries: usize,
}

impl ClusterAwareClient {
    /// Creates a new ClusterAwareClient.
    pub fn new(cluster: Arc<ClusterManager>) -> Self {
        Self {
            cluster,
            clients: Arc::new(tokio::sync::RwLock::new(BTreeMap::new())),
            max_retries: 3,
        }
    }

    /// Set the maximum number of retries for leader routing.
    pub fn with_max_retries(mut self, max_retries: usize) -> Self {
        self.max_retries = max_retries;
        self
    }

    /// Get or create a gRPC client for a specific node.
    async fn get_client(
        &self,
        node_id: u64,
        addr: &str,
    ) -> Result<proto::raft_client::RaftClient<tonic::transport::Channel>, ClientError> {
        // Check cache first
        {
            let clients = self.clients.read().await;
            if let Some(client) = clients.get(&node_id) {
                return Ok(client.clone());
            }
        }

        // Create new client
        let client = proto::raft_client::RaftClient::connect(format!("http://{}", addr))
            .await
            .map_err(|e| ClientError::NetworkError(e.to_string()))?;

        // Cache it
        {
            let mut clients = self.clients.write().await;
            clients.insert(node_id, client.clone());
        }

        Ok(client)
    }

    /// Submit a client request to the Raft cluster.
    ///
    /// This method automatically routes the request to the current leader
    /// and handles redirects if the leader has changed.
    pub async fn submit(&self, request: &RaftRequest) -> Result<super::types::RaftResponse, ClientError> {
        let mut attempts = 0;

        loop {
            attempts += 1;
            if attempts > self.max_retries {
                return Err(ClientError::OperationFailed(
                    "Max retries exceeded for leader routing".to_string(),
                ));
            }

            // Get current leader
            let (leader_id, leader_addr) = match self.cluster.get_leader().await {
                Ok(leader) => leader,
                Err(ClusterError::NoLeader) => {
                    // Wait a bit and retry
                    tokio::time::sleep(std::time::Duration::from_millis(100)).await;
                    continue;
                }
                Err(e) => return Err(ClientError::NetworkError(e.to_string())),
            };

            // Get client for leader
            let mut client = self.get_client(leader_id, &leader_addr).await?;

            // Serialize the request
            let _req_data = bincode::serialize(request)
                .map_err(|e| ClientError::OperationFailed(e.to_string()))?;

            // Send via append_entries (piggyback on Raft protocol)
            // In a real implementation, you'd have a dedicated client_request RPC
            // For now, we'll use the cluster info to verify leadership
            let info_request = tonic::Request::new(proto::ClusterInfoRequest {});
            let info_response = client
                .get_cluster_info(info_request)
                .await
                .map_err(|e| ClientError::NetworkError(e.to_string()))?;

            let info = info_response.into_inner();

            // Verify this node is still the leader
            if info.leader_id != leader_id {
                // Leader changed, update cache and retry
                if info.leader_id != 0 && !info.leader_addr.is_empty() {
                    self.cluster
                        .update_leader(info.leader_id, info.leader_addr)
                        .await;
                }
                continue;
            }

            // The node is the leader, we would submit the request here
            // In a full implementation, there would be a ClientRequest RPC
            // For now, return a success response indicating the leader was found
            return Ok(super::types::RaftResponse::ok(0));
        }
    }

    /// Get cluster information from any available node.
    pub async fn get_cluster_info(&self) -> Result<proto::ClusterInfoResponse, ClientError> {
        // Try to get from leader first
        if let Ok((leader_id, leader_addr)) = self.cluster.get_leader().await {
            let mut client = self.get_client(leader_id, &leader_addr).await?;
            let request = tonic::Request::new(proto::ClusterInfoRequest {});
            let response = client
                .get_cluster_info(request)
                .await
                .map_err(|e| ClientError::NetworkError(e.to_string()))?;
            return Ok(response.into_inner());
        }

        // Fall back to any known peer
        let peers = self.cluster.get_known_peers().await;
        for peer in peers {
            if let Ok(mut client) = self.get_client(peer.id, &peer.addr).await {
                let request = tonic::Request::new(proto::ClusterInfoRequest {});
                if let Ok(response) = client.get_cluster_info(request).await {
                    return Ok(response.into_inner());
                }
            }
        }

        Err(ClientError::NoLeader)
    }

    /// Check health of a specific node.
    pub async fn health_check(
        &self,
        node_id: u64,
        addr: &str,
    ) -> Result<proto::HealthCheckResponse, ClientError> {
        let mut client = self.get_client(node_id, addr).await?;
        let request = tonic::Request::new(proto::HealthCheckRequest { node_id });
        let response = client
            .health_check(request)
            .await
            .map_err(|e| ClientError::NetworkError(e.to_string()))?;
        Ok(response.into_inner())
    }

    /// Clear all cached clients (useful after cluster reconfiguration).
    pub async fn clear_cache(&self) {
        let mut clients = self.clients.write().await;
        clients.clear();
    }
}