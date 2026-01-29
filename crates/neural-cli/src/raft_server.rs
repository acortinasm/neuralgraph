//! Raft gRPC server implementation.
//!
//! This module provides the gRPC server for Raft consensus,
//! allowing other nodes to send RPCs (AppendEntries, InstallSnapshot, Vote)
//! as well as cluster management RPCs (Join, GetClusterInfo, HealthCheck).

use std::collections::BTreeMap;
use std::sync::Arc;

use openraft::raft::{AppendEntriesRequest, AppendEntriesResponse, InstallSnapshotRequest,
    InstallSnapshotResponse, VoteRequest, VoteResponse};
use openraft::{BasicNode, ChangeMembers, Raft};
use tonic::{Request, Response, Status};

use neural_storage::raft::network::proto::raft_server::Raft as RaftService;
use neural_storage::raft::network::proto::{
    AppendEntriesRequest as ProtoAppendEntriesRequest,
    AppendEntriesResponse as ProtoAppendEntriesResponse,
    InstallSnapshotRequest as ProtoInstallSnapshotRequest,
    InstallSnapshotResponse as ProtoInstallSnapshotResponse,
    VoteRequest as ProtoVoteRequest,
    VoteResponse as ProtoVoteResponse,
    JoinRequest, JoinResponse, NodeInfo,
    ClusterInfoRequest, ClusterInfoResponse,
    HealthCheckRequest, HealthCheckResponse,
    ClientRequestMessage, ClientResponseMessage,
};
use neural_storage::raft::types::{TypeConfig, RaftRequest};
use neural_storage::raft::ClusterManager;

/// Implements the `RaftService` trait for handling Raft gRPC calls.
#[derive(Clone)]
pub struct RaftGrpcServer {
    /// The `openraft` instance.
    raft: Raft<TypeConfig>,
    /// The cluster manager for membership operations.
    cluster: Option<Arc<ClusterManager>>,
}

impl RaftGrpcServer {
    /// Creates a new Raft gRPC server.
    #[allow(dead_code)]
    pub fn new(raft: Raft<TypeConfig>) -> Self {
        Self { raft, cluster: None }
    }

    /// Creates a new Raft gRPC server with cluster management support.
    pub fn with_cluster(raft: Raft<TypeConfig>, cluster: Arc<ClusterManager>) -> Self {
        Self {
            raft,
            cluster: Some(cluster),
        }
    }

    /// Get the current node state as a string.
    fn get_node_state(&self) -> &'static str {
        let metrics_watch = self.raft.metrics();
        let metrics = metrics_watch.borrow();
        if let Some(leader_id) = metrics.current_leader {
            if let Some(ref cluster) = self.cluster {
                if cluster.node_id() == leader_id {
                    return "leader";
                }
            }
        }
        // Check if we're a candidate based on vote state
        if metrics.vote.committed {
            "follower"
        } else {
            "candidate"
        }
    }
}

#[tonic::async_trait]
impl RaftService for RaftGrpcServer
{
    /// Handles `AppendEntries` RPCs.
    async fn append_entries(
        &self,
        request: Request<ProtoAppendEntriesRequest>,
    ) -> Result<Response<ProtoAppendEntriesResponse>, Status> {
        let proto_req = request.into_inner();
        let req: AppendEntriesRequest<TypeConfig> = bincode::deserialize(&proto_req.data)
            .map_err(|e| Status::internal(format!("Failed to deserialize AppendEntriesRequest: {}", e)))?;

        let resp: AppendEntriesResponse<u64> = self.raft.append_entries(req).await
            .map_err(|e| Status::internal(format!("Failed to handle AppendEntries: {}", e)))?;
        
        let proto_resp_data = bincode::serialize(&resp)
            .map_err(|e| Status::internal(format!("Failed to serialize AppendEntriesResponse: {}", e)))?;
        
        Ok(Response::new(ProtoAppendEntriesResponse { data: proto_resp_data }))
    }

    /// Handles `InstallSnapshot` RPCs.
    async fn install_snapshot(
        &self,
        request: Request<ProtoInstallSnapshotRequest>,
    ) -> Result<Response<ProtoInstallSnapshotResponse>, Status> {
        let proto_req = request.into_inner();
        let req: InstallSnapshotRequest<TypeConfig> = bincode::deserialize(&proto_req.data)
            .map_err(|e| Status::internal(format!("Failed to deserialize InstallSnapshotRequest: {}", e)))?;

        let resp: InstallSnapshotResponse<u64> = self.raft.install_snapshot(req).await
            .map_err(|e| Status::internal(format!("Failed to handle InstallSnapshot: {}", e)))?;

        let proto_resp_data = bincode::serialize(&resp)
            .map_err(|e| Status::internal(format!("Failed to serialize InstallSnapshotResponse: {}", e)))?;

        Ok(Response::new(ProtoInstallSnapshotResponse { data: proto_resp_data }))
    }

    /// Handles `Vote` RPCs.
    async fn vote(
        &self,
        request: Request<ProtoVoteRequest>,
    ) -> Result<Response<ProtoVoteResponse>, Status> {
        let proto_req = request.into_inner();
        let req: VoteRequest<u64> = bincode::deserialize(&proto_req.data)
            .map_err(|e| Status::internal(format!("Failed to deserialize VoteRequest: {}", e)))?;

        let resp: VoteResponse<u64> = self.raft.vote(req).await
            .map_err(|e| Status::internal(format!("Failed to handle Vote: {}", e)))?;

        let proto_resp_data = bincode::serialize(&resp)
            .map_err(|e| Status::internal(format!("Failed to serialize VoteResponse: {}", e)))?;

        Ok(Response::new(ProtoVoteResponse { data: proto_resp_data }))
    }

    /// Handles `Join` RPCs for cluster membership.
    async fn join(
        &self,
        request: Request<JoinRequest>,
    ) -> Result<Response<JoinResponse>, Status> {
        let req = request.into_inner();
        let node_id = req.node_id;
        let node_addr = req.addr;

        // Check if we have cluster management
        let cluster = self.cluster.as_ref().ok_or_else(|| {
            Status::unavailable("Cluster management not enabled")
        })?;

        // Check if we're the leader
        let metrics = self.raft.metrics().borrow().clone();
        let current_leader = metrics.current_leader;

        if current_leader != Some(cluster.node_id()) {
            // Not the leader, redirect to leader
            let leader_addr = if let Some(leader_id) = current_leader {
                // Try to find leader address from membership
                let membership = metrics.membership_config.membership();
                membership
                    .nodes()
                    .find(|(id, _)| **id == leader_id)
                    .map(|(_, node)| node.addr.clone())
                    .unwrap_or_default()
            } else {
                String::new()
            };

            return Ok(Response::new(JoinResponse {
                success: false,
                leader_id: current_leader.unwrap_or(0),
                leader_addr,
                members: vec![],
                error: "Not the leader".to_string(),
            }));
        }

        // We're the leader, add the new member
        let node = BasicNode { addr: node_addr.clone() };
        let mut new_nodes = BTreeMap::new();
        new_nodes.insert(node_id, node);

        match self.raft.change_membership(ChangeMembers::AddNodes(new_nodes), false).await {
            Ok(_) => {
                // Get current membership for response
                let metrics = self.raft.metrics().borrow().clone();
                let mut members = Vec::new();

                let membership = metrics.membership_config.membership();
                for (id, node) in membership.nodes() {
                    members.push(NodeInfo {
                        id: *id,
                        addr: node.addr.clone(),
                        is_leader: Some(*id) == metrics.current_leader,
                    });
                }

                Ok(Response::new(JoinResponse {
                    success: true,
                    leader_id: cluster.node_id(),
                    leader_addr: cluster.node_addr().to_string(),
                    members,
                    error: String::new(),
                }))
            }
            Err(e) => Ok(Response::new(JoinResponse {
                success: false,
                leader_id: cluster.node_id(),
                leader_addr: cluster.node_addr().to_string(),
                members: vec![],
                error: format!("Failed to add member: {}", e),
            })),
        }
    }

    /// Handles `GetClusterInfo` RPCs.
    async fn get_cluster_info(
        &self,
        _request: Request<ClusterInfoRequest>,
    ) -> Result<Response<ClusterInfoResponse>, Status> {
        let metrics = self.raft.metrics().borrow().clone();

        let leader_id = metrics.current_leader.unwrap_or(0);
        let term = metrics.current_term;

        // Get membership
        let mut members = Vec::new();
        let mut leader_addr = String::new();

        let membership = metrics.membership_config.membership();
        for (id, node) in membership.nodes() {
            let is_leader = Some(*id) == metrics.current_leader;
            if is_leader {
                leader_addr = node.addr.clone();
            }
            members.push(NodeInfo {
                id: *id,
                addr: node.addr.clone(),
                is_leader,
            });
        }

        Ok(Response::new(ClusterInfoResponse {
            leader_id,
            leader_addr,
            members,
            term,
        }))
    }

    /// Handles `HealthCheck` RPCs.
    async fn health_check(
        &self,
        request: Request<HealthCheckRequest>,
    ) -> Result<Response<HealthCheckResponse>, Status> {
        let _req = request.into_inner();
        let metrics = self.raft.metrics().borrow().clone();

        // Determine node state
        let state = self.get_node_state().to_string();

        // Get last log index
        let last_log_index = metrics.last_log_index.unwrap_or(0);

        Ok(Response::new(HealthCheckResponse {
            healthy: true,
            term: metrics.current_term,
            state,
            last_log_index,
        }))
    }

    /// Handles `ClientRequest` RPCs for submitting graph mutations.
    async fn client_request(
        &self,
        request: Request<ClientRequestMessage>,
    ) -> Result<Response<ClientResponseMessage>, Status> {
        let req = request.into_inner();

        // Check if we have cluster management
        let cluster = self.cluster.as_ref().ok_or_else(|| {
            Status::unavailable("Cluster management not enabled")
        })?;

        // Check if we're the leader
        let metrics = self.raft.metrics().borrow().clone();
        let current_leader = metrics.current_leader;

        if current_leader != Some(cluster.node_id()) {
            // Not the leader, redirect to leader
            let leader_addr = if let Some(leader_id) = current_leader {
                let membership = metrics.membership_config.membership();
                membership
                    .nodes()
                    .find(|(id, _)| **id == leader_id)
                    .map(|(_, node)| node.addr.clone())
                    .unwrap_or_default()
            } else {
                String::new()
            };

            return Ok(Response::new(ClientResponseMessage {
                success: false,
                response: vec![],
                error: "Not the leader".to_string(),
                leader_id: current_leader.unwrap_or(0),
                leader_addr,
                log_index: 0,
            }));
        }

        // Deserialize the request
        let raft_request: RaftRequest = bincode::deserialize(&req.request)
            .map_err(|e| Status::invalid_argument(format!("Failed to deserialize request: {}", e)))?;

        // Submit to Raft - this will replicate to followers and apply to state machine
        match self.raft.client_write(raft_request).await {
            Ok(client_write_response) => {
                // Get the response from the state machine
                let log_index = client_write_response.log_id.index;
                let raft_response = client_write_response.data;

                let response_bytes = bincode::serialize(&raft_response)
                    .map_err(|e| Status::internal(format!("Failed to serialize response: {}", e)))?;

                Ok(Response::new(ClientResponseMessage {
                    success: true,
                    response: response_bytes,
                    error: String::new(),
                    leader_id: cluster.node_id(),
                    leader_addr: cluster.node_addr().to_string(),
                    log_index,
                }))
            }
            Err(e) => {
                // Check if this is a forward-to-leader error
                let error_str = e.to_string();
                Ok(Response::new(ClientResponseMessage {
                    success: false,
                    response: vec![],
                    error: error_str,
                    leader_id: current_leader.unwrap_or(0),
                    leader_addr: String::new(),
                    log_index: 0,
                }))
            }
        }
    }
}