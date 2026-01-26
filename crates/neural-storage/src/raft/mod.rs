//! Raft consensus module for distributed NeuralGraphDB.
//!
//! This module implements the Raft consensus algorithm for multi-node replication,
//! enabling high availability and fault tolerance.
//!
//! # Architecture
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────────┐
//! │                        Raft Cluster                              │
//! │  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐         │
//! │  │   Node 1    │    │   Node 2    │    │   Node 3    │         │
//! │  │  (Leader)   │◄──►│ (Follower)  │◄──►│ (Follower)  │         │
//! │  └─────────────┘    └─────────────┘    └─────────────┘         │
//! │         │                  │                  │                 │
//! │         ▼                  ▼                  ▼                 │
//! │  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐         │
//! │  │ GraphStore  │    │ GraphStore  │    │ GraphStore  │         │
//! │  │  (Primary)  │    │  (Replica)  │    │  (Replica)  │         │
//! │  └─────────────┘    └─────────────┘    └─────────────┘         │
//! └─────────────────────────────────────────────────────────────────┘
//! ```
//!
//! # Components
//!
//! - **TypeConfig**: Type configuration for OpenRaft
//! - **RaftRequest/RaftResponse**: Request/response types wrapping LogEntry
//! - **LogStore**: Persistent log storage for Raft entries
//! - **GraphStateMachine**: State machine that applies entries to GraphStore
//! - **ClusterConfig**: Configuration for cluster nodes
//!
//! # Example
//!
//! ```ignore
//! use neural_storage::raft::{ClusterConfig, GraphStateMachine, LogStore};
//!
//! // Create storage components
//! let log_store = LogStore::new_arc();
//! let state_machine = GraphStateMachine::new_arc();
//!
//! // Apply entries directly to the state machine (for testing)
//! use neural_storage::raft::RaftRequest;
//! use neural_storage::wal::LogEntry;
//! use openraft::{Entry, EntryPayload, LogId};
//!
//! let request = RaftRequest::new(LogEntry::CreateNode {
//!     node_id: neural_core::NodeId::new(0),
//!     label: Some("Person".to_string()),
//!     properties: vec![],
//! });
//!
//! // In a real cluster, entries would be replicated via Raft
//! ```
//!
//! # Sprint 52 Status
//!
//! This module is under active development as part of Sprint 52.
//! Current status:
//! - [x] Type definitions
//! - [x] In-memory log store
//! - [x] State machine (GraphStore wrapper)
//! - [ ] Network layer (gRPC)
//! - [ ] Full Raft node integration
//! - [ ] Cluster management

pub mod log_store;
pub mod state_machine;
pub mod types;
pub mod network;
pub mod cluster;
pub mod health;

// Re-exports for convenience
pub use log_store::LogStore;
pub use state_machine::GraphStateMachine;
pub use types::{ClusterConfig, RaftNodeId, RaftRequest, RaftResponse, TypeConfig};
pub use wrapper::NeuralRaftStorage;
pub use network::{NeuralRaftNetwork, ClusterAwareClient};
pub use cluster::{ClusterManager, ClusterInfo, ClusterError, NodeInfo};
pub use health::{HealthMonitor, HealthConfig, NodeHealth, NodeState};

use openraft::Config;

/// Default Raft configuration for NeuralGraphDB.
pub fn default_raft_config() -> Config {
    Config {
        cluster_name: "neuralgraph".to_string(),
        heartbeat_interval: 50,
        election_timeout_min: 150,
        election_timeout_max: 300,
        ..Default::default()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::wal::LogEntry;
    use neural_core::NodeId;

    #[test]
    fn test_raft_request_from_log_entry() {
        let entry = LogEntry::CreateNode {
            node_id: NodeId::new(1),
            label: Some("Test".to_string()),
            properties: vec![],
        };

        let request = RaftRequest::new(entry.clone());
        assert_eq!(request.entry, entry);
    }

    #[test]
    fn test_cluster_config() {
        let config = ClusterConfig::cluster(
            1,
            "127.0.0.1:9000",
            50052,
            vec![
                (2, "127.0.0.1:9001".to_string()),
                (3, "127.0.0.1:9002".to_string()),
            ],
        );

        assert_eq!(config.node_id, 1);
        assert_eq!(config.listen_addr, "127.0.0.1:9000");
        assert_eq!(config.peers.len(), 2);
    }

    #[test]
    fn test_default_raft_config() {
        let config = default_raft_config();
        assert_eq!(config.cluster_name, "neuralgraph");
        assert_eq!(config.heartbeat_interval, 50);
    }
}
mod wrapper;
