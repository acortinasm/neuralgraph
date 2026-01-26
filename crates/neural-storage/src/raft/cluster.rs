//! Cluster management for NeuralGraphDB Raft consensus.
//!
//! This module provides the `ClusterManager` for handling node discovery,
//! membership changes, and leader tracking in a Raft cluster.

use std::collections::{BTreeMap, BTreeSet};
use std::sync::Arc;

use openraft::{BasicNode, ChangeMembers, Raft, RaftMetrics};
use tokio::sync::RwLock;

use super::network::proto::raft_client::RaftClient;
use super::network::proto::{JoinRequest, NodeInfo as ProtoNodeInfo};
use super::types::{RaftNodeId, TypeConfig};

/// Error type for cluster operations.
#[derive(Debug, thiserror::Error)]
pub enum ClusterError {
    #[error("Not the leader. Leader is node {leader_id:?} at {leader_addr:?}")]
    NotLeader {
        leader_id: Option<u64>,
        leader_addr: Option<String>,
    },
    #[error("Failed to connect to node: {0}")]
    ConnectionFailed(String),
    #[error("Node {0} not found in cluster")]
    NodeNotFound(u64),
    #[error("Cluster operation failed: {0}")]
    OperationFailed(String),
    #[error("No leader available")]
    NoLeader,
    #[error("gRPC error: {0}")]
    GrpcError(#[from] tonic::Status),
    #[error("Transport error: {0}")]
    TransportError(#[from] tonic::transport::Error),
    #[error("Raft error: {0}")]
    RaftError(String),
}

/// Information about a node in the cluster.
#[derive(Debug, Clone)]
pub struct NodeInfo {
    pub id: RaftNodeId,
    pub addr: String,
    pub is_leader: bool,
}

impl From<ProtoNodeInfo> for NodeInfo {
    fn from(proto: ProtoNodeInfo) -> Self {
        Self {
            id: proto.id,
            addr: proto.addr,
            is_leader: proto.is_leader,
        }
    }
}

impl From<NodeInfo> for ProtoNodeInfo {
    fn from(info: NodeInfo) -> Self {
        Self {
            id: info.id,
            addr: info.addr,
            is_leader: info.is_leader,
        }
    }
}

/// Current cluster state information.
#[derive(Debug, Clone)]
pub struct ClusterInfo {
    pub leader_id: Option<RaftNodeId>,
    pub leader_addr: Option<String>,
    pub members: Vec<NodeInfo>,
    pub term: u64,
}

impl Default for ClusterInfo {
    fn default() -> Self {
        Self {
            leader_id: None,
            leader_addr: None,
            members: Vec::new(),
            term: 0,
        }
    }
}

/// Manages cluster membership and leader tracking.
///
/// The `ClusterManager` handles:
/// - Joining an existing cluster via a seed node
/// - Tracking current cluster membership
/// - Caching the current leader for efficient routing
/// - Adding/removing nodes from the cluster (leader only)
pub struct ClusterManager {
    /// This node's ID.
    node_id: RaftNodeId,
    /// This node's address.
    node_addr: String,
    /// The Raft instance.
    raft: Arc<Raft<TypeConfig>>,
    /// Known peers in the cluster (id -> NodeInfo).
    known_peers: Arc<RwLock<BTreeMap<RaftNodeId, NodeInfo>>>,
    /// Cached leader information (id, addr).
    leader_cache: Arc<RwLock<Option<(RaftNodeId, String)>>>,
}

impl ClusterManager {
    /// Creates a new ClusterManager.
    pub fn new(node_id: RaftNodeId, node_addr: String, raft: Arc<Raft<TypeConfig>>) -> Self {
        Self {
            node_id,
            node_addr,
            raft,
            known_peers: Arc::new(RwLock::new(BTreeMap::new())),
            leader_cache: Arc::new(RwLock::new(None)),
        }
    }

    /// Returns this node's ID.
    pub fn node_id(&self) -> RaftNodeId {
        self.node_id
    }

    /// Returns this node's address.
    pub fn node_addr(&self) -> &str {
        &self.node_addr
    }

    /// Returns a reference to the Raft instance.
    pub fn raft(&self) -> &Raft<TypeConfig> {
        &self.raft
    }

    /// Join an existing cluster by contacting a seed node.
    ///
    /// This will:
    /// 1. Connect to the seed node
    /// 2. Send a join request
    /// 3. If redirected to leader, retry with leader
    /// 4. Update local membership once joined
    pub async fn join_cluster(&self, seed_addr: &str) -> Result<ClusterInfo, ClusterError> {
        let mut target_addr = seed_addr.to_string();
        let max_redirects = 5;

        for _ in 0..max_redirects {
            let mut client = RaftClient::connect(format!("http://{}", target_addr))
                .await
                .map_err(|e| ClusterError::ConnectionFailed(e.to_string()))?;

            let request = tonic::Request::new(JoinRequest {
                node_id: self.node_id,
                addr: self.node_addr.clone(),
            });

            let response = client.join(request).await?;
            let join_response = response.into_inner();

            if join_response.success {
                // Successfully joined, update local state
                let mut peers = self.known_peers.write().await;
                for member in &join_response.members {
                    peers.insert(
                        member.id,
                        NodeInfo {
                            id: member.id,
                            addr: member.addr.clone(),
                            is_leader: member.is_leader,
                        },
                    );
                }

                // Update leader cache
                if join_response.leader_id != 0 {
                    let mut leader = self.leader_cache.write().await;
                    *leader = Some((join_response.leader_id, join_response.leader_addr.clone()));
                }

                return Ok(ClusterInfo {
                    leader_id: if join_response.leader_id != 0 {
                        Some(join_response.leader_id)
                    } else {
                        None
                    },
                    leader_addr: if join_response.leader_addr.is_empty() {
                        None
                    } else {
                        Some(join_response.leader_addr)
                    },
                    members: join_response.members.into_iter().map(Into::into).collect(),
                    term: 0,
                });
            }

            // Not successful - check if we need to redirect to leader
            if join_response.leader_id != 0 && !join_response.leader_addr.is_empty() {
                target_addr = join_response.leader_addr;
                continue;
            }

            // Failed without redirect
            let error_msg = if join_response.error.is_empty() {
                "Unknown error".to_string()
            } else {
                join_response.error
            };
            return Err(ClusterError::OperationFailed(error_msg));
        }

        Err(ClusterError::OperationFailed(
            "Too many redirects while joining cluster".to_string(),
        ))
    }

    /// Get current cluster information from the local Raft state.
    pub async fn get_cluster_info(&self) -> ClusterInfo {
        let metrics = self.raft.metrics().borrow().clone();

        let leader_id = metrics.current_leader;
        let term = metrics.current_term;

        // Get membership from Raft
        let members = self.get_members_from_metrics(&metrics).await;

        // Get leader address
        let leader_addr = if let Some(lid) = leader_id {
            members
                .iter()
                .find(|m| m.id == lid)
                .map(|m| m.addr.clone())
        } else {
            None
        };

        ClusterInfo {
            leader_id,
            leader_addr,
            members,
            term,
        }
    }

    /// Extract member list from Raft metrics.
    async fn get_members_from_metrics(&self, metrics: &RaftMetrics<RaftNodeId, BasicNode>) -> Vec<NodeInfo> {
        let mut members = Vec::new();
        let leader_id = metrics.current_leader;

        // Get membership from Raft's membership config
        let membership = metrics.membership_config.membership();
        for (node_id, node) in membership.nodes() {
            members.push(NodeInfo {
                id: *node_id,
                addr: node.addr.clone(),
                is_leader: Some(*node_id) == leader_id,
            });
        }

        // Also include known peers
        let peers = self.known_peers.read().await;
        for (id, info) in peers.iter() {
            if !members.iter().any(|m| m.id == *id) {
                members.push(info.clone());
            }
        }

        members
    }

    /// Update the cached leader information.
    pub async fn update_leader(&self, leader_id: RaftNodeId, leader_addr: String) {
        let mut cache = self.leader_cache.write().await;
        *cache = Some((leader_id, leader_addr.clone()));

        // Also update in known_peers
        let mut peers = self.known_peers.write().await;
        // Clear old leader flag
        for info in peers.values_mut() {
            info.is_leader = false;
        }
        // Set new leader
        if let Some(info) = peers.get_mut(&leader_id) {
            info.is_leader = true;
        } else {
            peers.insert(
                leader_id,
                NodeInfo {
                    id: leader_id,
                    addr: leader_addr,
                    is_leader: true,
                },
            );
        }
    }

    /// Get the current leader (id, addr) from cache or Raft state.
    pub async fn get_leader(&self) -> Result<(RaftNodeId, String), ClusterError> {
        // First check cache
        {
            let cache = self.leader_cache.read().await;
            if let Some((id, addr)) = cache.as_ref() {
                return Ok((*id, addr.clone()));
            }
        }

        // Fall back to Raft metrics
        let metrics = self.raft.metrics().borrow().clone();
        if let Some(leader_id) = metrics.current_leader {
            // Try to find leader address
            let info = self.get_cluster_info().await;
            if let Some(addr) = info.leader_addr {
                // Update cache
                self.update_leader(leader_id, addr.clone()).await;
                return Ok((leader_id, addr));
            }
        }

        Err(ClusterError::NoLeader)
    }

    /// Check if this node is the leader.
    pub fn is_leader(&self) -> bool {
        let metrics_watch = self.raft.metrics();
        let metrics = metrics_watch.borrow();
        metrics.current_leader == Some(self.node_id)
    }

    /// Add a new node to the cluster (must be called on leader).
    pub async fn add_member(&self, node_id: RaftNodeId, addr: String) -> Result<(), ClusterError> {
        if !self.is_leader() {
            let info = self.get_cluster_info().await;
            return Err(ClusterError::NotLeader {
                leader_id: info.leader_id,
                leader_addr: info.leader_addr,
            });
        }

        // Add the node to Raft membership using ChangeMembers::AddNodes
        let node = BasicNode { addr: addr.clone() };
        let mut new_nodes = BTreeMap::new();
        new_nodes.insert(node_id, node);

        self.raft
            .change_membership(ChangeMembers::AddNodes(new_nodes), false)
            .await
            .map_err(|e| ClusterError::RaftError(e.to_string()))?;

        // Update local state
        let mut peers = self.known_peers.write().await;
        peers.insert(
            node_id,
            NodeInfo {
                id: node_id,
                addr,
                is_leader: false,
            },
        );

        Ok(())
    }

    /// Remove a node from the cluster (must be called on leader).
    pub async fn remove_member(&self, node_id: RaftNodeId) -> Result<(), ClusterError> {
        if !self.is_leader() {
            let info = self.get_cluster_info().await;
            return Err(ClusterError::NotLeader {
                leader_id: info.leader_id,
                leader_addr: info.leader_addr,
            });
        }

        // Remove the node from Raft membership using ChangeMembers::RemoveNodes
        let mut nodes_to_remove = BTreeSet::new();
        nodes_to_remove.insert(node_id);

        self.raft
            .change_membership(ChangeMembers::RemoveNodes(nodes_to_remove), false)
            .await
            .map_err(|e| ClusterError::RaftError(e.to_string()))?;

        // Update local state
        let mut peers = self.known_peers.write().await;
        peers.remove(&node_id);

        Ok(())
    }

    /// Add a peer to the known peers list (without changing Raft membership).
    pub async fn add_known_peer(&self, node_id: RaftNodeId, addr: String) {
        let mut peers = self.known_peers.write().await;
        peers.insert(
            node_id,
            NodeInfo {
                id: node_id,
                addr,
                is_leader: false,
            },
        );
    }

    /// Get all known peers.
    pub async fn get_known_peers(&self) -> Vec<NodeInfo> {
        let peers = self.known_peers.read().await;
        peers.values().cloned().collect()
    }

    /// Clear the leader cache (useful when leader changes are detected).
    pub async fn clear_leader_cache(&self) {
        let mut cache = self.leader_cache.write().await;
        *cache = None;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_node_info_conversion() {
        let proto = ProtoNodeInfo {
            id: 1,
            addr: "127.0.0.1:50052".to_string(),
            is_leader: true,
        };

        let info: NodeInfo = proto.into();
        assert_eq!(info.id, 1);
        assert_eq!(info.addr, "127.0.0.1:50052");
        assert!(info.is_leader);

        let back: ProtoNodeInfo = info.into();
        assert_eq!(back.id, 1);
    }

    #[test]
    fn test_cluster_info_default() {
        let info = ClusterInfo::default();
        assert!(info.leader_id.is_none());
        assert!(info.members.is_empty());
        assert_eq!(info.term, 0);
    }
}
