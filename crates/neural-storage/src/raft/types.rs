//! Type definitions for Raft consensus.
//!
//! Defines the core types used by OpenRaft for NeuralGraphDB clustering.

use openraft::BasicNode;
use serde::{Deserialize, Serialize};
use std::io::Cursor;

use crate::wal::LogEntry;

/// Raft node ID type (u64 for simplicity).
pub type RaftNodeId = u64;

/// Raft node information.
pub type RaftNode = BasicNode;

openraft::declare_raft_types!(
    /// Type configuration for NeuralGraphDB Raft.
    pub TypeConfig:
        D = RaftRequest,
        R = RaftResponse,
        Node = RaftNode,
        NodeId = RaftNodeId,
        Entry = openraft::Entry<TypeConfig>,
        SnapshotData = Cursor<Vec<u8>>,
);

/// A request to the Raft state machine.
///
/// Wraps the existing WAL LogEntry for seamless integration.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct RaftRequest {
    /// The graph mutation to apply.
    pub entry: LogEntry,
}

impl RaftRequest {
    /// Creates a new Raft request from a log entry.
    pub fn new(entry: LogEntry) -> Self {
        Self { entry }
    }
}

/// Response from the Raft state machine after applying a request.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct RaftResponse {
    /// Whether the operation was successful.
    pub success: bool,
    /// Optional error message.
    pub error: Option<String>,
    /// The log index at which this was applied.
    pub applied_index: u64,
}

impl RaftResponse {
    /// Creates a successful response.
    pub fn ok(applied_index: u64) -> Self {
        Self {
            success: true,
            error: None,
            applied_index,
        }
    }

    /// Creates an error response.
    pub fn err(message: impl Into<String>, applied_index: u64) -> Self {
        Self {
            success: false,
            error: Some(message.into()),
            applied_index,
        }
    }
}

/// Cluster configuration for a Raft node.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClusterConfig {
    /// This node's ID.
    pub node_id: RaftNodeId,
    /// Address this node listens on (host:port).
    pub listen_addr: String,
    /// Port for the Raft gRPC server.
    pub raft_port: u16,
    /// Peer nodes in the cluster (id -> address).
    pub peers: Vec<(RaftNodeId, String)>,
    /// Election timeout range in milliseconds.
    pub election_timeout_ms: (u64, u64),
    /// Heartbeat interval in milliseconds.
    pub heartbeat_interval_ms: u64,
    /// Path to store Raft log and snapshots.
    pub data_dir: String,
}

impl Default for ClusterConfig {
    fn default() -> Self {
        Self {
            node_id: 1,
            listen_addr: "127.0.0.1:9000".to_string(),
            raft_port: 50052,
            peers: vec![],
            election_timeout_ms: (150, 300),
            heartbeat_interval_ms: 50,
            data_dir: "./raft-data".to_string(),
        }
    }
}

impl ClusterConfig {
    /// Creates a single-node cluster configuration.
    pub fn single_node(node_id: RaftNodeId, listen_addr: impl Into<String>, raft_port: u16) -> Self {
        Self {
            node_id,
            listen_addr: listen_addr.into(),
            raft_port,
            peers: vec![],
            ..Default::default()
        }
    }

    /// Creates a multi-node cluster configuration.
    pub fn cluster(
        node_id: RaftNodeId,
        listen_addr: impl Into<String>,
        raft_port: u16,
        peers: Vec<(RaftNodeId, String)>,
    ) -> Self {
        Self {
            node_id,
            listen_addr: listen_addr.into(),
            raft_port,
            peers,
            ..Default::default()
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::wal::LogEntry;
    use neural_core::{NodeId, PropertyValue};

    #[test]
    fn test_raft_request_serialization() {
        let entry = LogEntry::CreateNode {
            node_id: NodeId::new(1),
            label: Some("Person".to_string()),
            properties: vec![("name".to_string(), PropertyValue::from("Alice"))],
        };
        let request = RaftRequest::new(entry);

        let serialized = bincode::serialize(&request).unwrap();
        let deserialized: RaftRequest = bincode::deserialize(&serialized).unwrap();

        assert_eq!(request, deserialized);
    }

    #[test]
    fn test_raft_response() {
        let ok = RaftResponse::ok(42);
        assert!(ok.success);
        assert_eq!(ok.applied_index, 42);

        let err = RaftResponse::err("failed", 10);
        assert!(!err.success);
        assert_eq!(err.error, Some("failed".to_string()));
    }

    #[test]
    fn test_cluster_config() {
        let config = ClusterConfig::cluster(
            1,
            "127.0.0.1:9000",
            50052,
            vec![(2, "127.0.0.1:9001".to_string()), (3, "127.0.0.1:9002".to_string())],
        );

        assert_eq!(config.node_id, 1);
        assert_eq!(config.peers.len(), 2);
    }
}
