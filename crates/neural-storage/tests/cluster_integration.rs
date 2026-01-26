//! Integration tests for NeuralGraphDB Raft cluster management.
//!
//! These tests verify:
//! - Cluster formation with multiple nodes
//! - Leader routing
//! - Health monitoring
//! - Node join/leave operations

use std::collections::BTreeMap;
use std::sync::Arc;
use std::time::Duration;

use neural_storage::raft::{
    ClusterInfo, ClusterManager, HealthConfig, HealthMonitor, NodeHealth, NodeState,
    default_raft_config, LogStore, GraphStateMachine, NeuralRaftStorage, NeuralRaftNetwork,
};
use openraft::{BasicNode, Raft};

/// Helper to create a test Raft node.
async fn create_test_node(
    node_id: u64,
    port: u16,
) -> (Raft<neural_storage::raft::TypeConfig>, Arc<ClusterManager>) {
    let data_dir = format!("/tmp/raft-test-{}-{}", node_id, std::process::id());
    let node_addr = format!("127.0.0.1:{}", port);

    let log_store = LogStore::new_arc(&data_dir);
    let state_machine = GraphStateMachine::new_arc();
    let storage = NeuralRaftStorage::new(log_store, state_machine);
    let network = NeuralRaftNetwork::new();

    let (ls, sm) = openraft::storage::Adaptor::new(storage);

    let config = default_raft_config();
    let raft_config = Arc::new(config.validate().unwrap());

    let raft = Raft::new(node_id, raft_config, network, ls, sm)
        .await
        .expect("Failed to create Raft node");

    let cluster = Arc::new(ClusterManager::new(
        node_id,
        node_addr,
        Arc::new(raft.clone()),
    ));

    (raft, cluster)
}

#[cfg(test)]
mod cluster_manager_tests {
    use super::*;

    #[test]
    fn test_cluster_info_default() {
        let info = ClusterInfo::default();
        assert!(info.leader_id.is_none());
        assert!(info.leader_addr.is_none());
        assert!(info.members.is_empty());
        assert_eq!(info.term, 0);
    }

    #[tokio::test]
    async fn test_cluster_manager_creation() {
        let (raft, cluster) = create_test_node(1, 59001).await;

        assert_eq!(cluster.node_id(), 1);
        assert_eq!(cluster.node_addr(), "127.0.0.1:59001");

        // Clean up
        drop(cluster);
        drop(raft);
    }

    #[tokio::test]
    async fn test_leader_cache_update() {
        let (raft, cluster) = create_test_node(1, 59002).await;

        // Initially no leader cached
        assert!(cluster.get_leader().await.is_err());

        // Update leader cache
        cluster.update_leader(2, "127.0.0.1:59003".to_string()).await;

        // Now should have leader
        let (leader_id, leader_addr) = cluster.get_leader().await.unwrap();
        assert_eq!(leader_id, 2);
        assert_eq!(leader_addr, "127.0.0.1:59003");

        // Clear cache
        cluster.clear_leader_cache().await;
        assert!(cluster.get_leader().await.is_err());

        drop(cluster);
        drop(raft);
    }

    #[tokio::test]
    async fn test_known_peers_management() {
        let (raft, cluster) = create_test_node(1, 59004).await;

        // Add peers
        cluster.add_known_peer(2, "127.0.0.1:59005".to_string()).await;
        cluster.add_known_peer(3, "127.0.0.1:59006".to_string()).await;

        let peers = cluster.get_known_peers().await;
        assert_eq!(peers.len(), 2);

        // Check peer info
        let peer2 = peers.iter().find(|p| p.id == 2).unwrap();
        assert_eq!(peer2.addr, "127.0.0.1:59005");
        assert!(!peer2.is_leader);

        drop(cluster);
        drop(raft);
    }
}

#[cfg(test)]
mod health_monitor_tests {
    use super::*;

    #[test]
    fn test_node_state_parsing() {
        assert_eq!(NodeState::from_str("leader"), NodeState::Leader);
        assert_eq!(NodeState::from_str("FOLLOWER"), NodeState::Follower);
        assert_eq!(NodeState::from_str("Candidate"), NodeState::Candidate);
        assert_eq!(NodeState::from_str("learner"), NodeState::Learner);
        assert_eq!(NodeState::from_str("unknown"), NodeState::Unknown);
        assert_eq!(NodeState::from_str("garbage"), NodeState::Unknown);
    }

    #[test]
    fn test_node_state_display() {
        assert_eq!(NodeState::Leader.as_str(), "leader");
        assert_eq!(NodeState::Follower.as_str(), "follower");
        assert_eq!(NodeState::Candidate.as_str(), "candidate");
        assert_eq!(NodeState::Learner.as_str(), "learner");
        assert_eq!(NodeState::Unknown.as_str(), "unknown");
    }

    #[test]
    fn test_node_health_creation() {
        let health = NodeHealth::new(1, "127.0.0.1:50052".to_string());

        assert_eq!(health.node_id, 1);
        assert_eq!(health.addr, "127.0.0.1:50052");
        assert!(!health.healthy);
        assert_eq!(health.consecutive_failures, 0);
        assert_eq!(health.state, NodeState::Unknown);
        assert_eq!(health.term, 0);
        assert_eq!(health.last_log_index, 0);
        assert!(health.latency_ms.is_none());
    }

    #[test]
    fn test_node_health_mark_healthy() {
        let mut health = NodeHealth::new(1, "127.0.0.1:50052".to_string());

        health.mark_unhealthy();
        health.mark_unhealthy();
        assert_eq!(health.consecutive_failures, 2);
        assert!(!health.healthy);

        health.mark_healthy(NodeState::Follower, 5, 100, 15);

        assert!(health.healthy);
        assert_eq!(health.consecutive_failures, 0);
        assert_eq!(health.state, NodeState::Follower);
        assert_eq!(health.term, 5);
        assert_eq!(health.last_log_index, 100);
        assert_eq!(health.latency_ms, Some(15));
    }

    #[test]
    fn test_node_health_is_dead() {
        let mut health = NodeHealth::new(1, "127.0.0.1:50052".to_string());

        // Not dead initially
        assert!(!health.is_dead(3));

        // One failure - not dead
        health.mark_unhealthy();
        assert!(!health.is_dead(3));

        // Two failures - not dead
        health.mark_unhealthy();
        assert!(!health.is_dead(3));

        // Three failures - dead
        health.mark_unhealthy();
        assert!(health.is_dead(3));

        // Recovery resets counter
        health.mark_healthy(NodeState::Follower, 1, 1, 10);
        assert!(!health.is_dead(3));
    }

    #[test]
    fn test_health_config_default() {
        let config = HealthConfig::default();

        assert_eq!(config.check_interval, Duration::from_millis(500));
        assert_eq!(config.timeout, Duration::from_millis(200));
        assert_eq!(config.failure_threshold, 3);
    }

    #[tokio::test]
    async fn test_health_monitor_add_remove_peer() {
        let config = HealthConfig {
            check_interval: Duration::from_millis(100),
            timeout: Duration::from_millis(50),
            failure_threshold: 3,
        };

        let monitor = HealthMonitor::new(1, config);

        // Add peers
        monitor.add_peer(2, "127.0.0.1:50053".to_string()).await;
        monitor.add_peer(3, "127.0.0.1:50054".to_string()).await;

        let health = monitor.get_cluster_health().await;
        assert_eq!(health.len(), 2);

        // Remove peer
        monitor.remove_peer(2).await;

        let health = monitor.get_cluster_health().await;
        assert_eq!(health.len(), 1);
        assert_eq!(health[0].node_id, 3);
    }

    #[tokio::test]
    async fn test_health_monitor_healthy_unhealthy_count() {
        let config = HealthConfig::default();
        let monitor = HealthMonitor::new(1, config);

        monitor.add_peer(2, "127.0.0.1:50053".to_string()).await;
        monitor.add_peer(3, "127.0.0.1:50054".to_string()).await;

        // Initially all unhealthy (not checked yet)
        assert_eq!(monitor.healthy_count().await, 0);
        assert_eq!(monitor.unhealthy_count().await, 2);
    }

    #[tokio::test]
    async fn test_health_monitor_quorum() {
        let config = HealthConfig::default();
        let monitor = HealthMonitor::new(1, config);

        // 3 node cluster needs 2 for quorum
        // With 0 healthy, no quorum
        assert!(!monitor.has_quorum(3).await);

        // 5 node cluster needs 3 for quorum
        assert!(!monitor.has_quorum(5).await);
    }
}

#[cfg(test)]
mod cluster_aware_client_tests {
    use super::*;
    use neural_storage::raft::ClusterAwareClient;

    #[tokio::test]
    async fn test_cluster_aware_client_creation() {
        let (raft, cluster) = create_test_node(1, 59010).await;

        let client = ClusterAwareClient::new(cluster.clone());
        let client = client.with_max_retries(5);

        // Should be able to create client
        drop(client);
        drop(cluster);
        drop(raft);
    }
}

/// Integration test for cluster formation (requires multiple processes in real deployment).
/// This test validates the core components work together.
#[tokio::test]
async fn test_cluster_components_integration() {
    // Create a single node
    let (raft, cluster) = create_test_node(1, 59020).await;

    // Initialize as single-node cluster
    let mut members = BTreeMap::new();
    members.insert(1u64, BasicNode { addr: "127.0.0.1:59020".to_string() });
    raft.initialize(members).await.expect("Failed to initialize");

    // Wait for election
    tokio::time::sleep(Duration::from_millis(500)).await;

    // Get cluster info
    let info = cluster.get_cluster_info().await;

    // Should be leader of single-node cluster
    assert!(info.leader_id.is_some());
    assert!(cluster.is_leader());

    // Set up health monitor
    let health_config = HealthConfig {
        check_interval: Duration::from_millis(100),
        timeout: Duration::from_millis(50),
        failure_threshold: 3,
    };
    let monitor = HealthMonitor::new(1, health_config);

    // No other nodes to monitor yet
    assert_eq!(monitor.get_cluster_health().await.len(), 0);

    drop(monitor);
    drop(cluster);
    drop(raft);
}
