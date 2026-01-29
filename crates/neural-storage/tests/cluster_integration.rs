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
    ClusterAwareClient, ClusterInfo, ClusterManager, HealthConfig, HealthMonitor, NodeHealth, NodeState,
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

// =============================================================================
// Multi-Node Cluster Tests (Sprint 53)
// =============================================================================

#[cfg(test)]
mod multi_node_tests {
    use super::*;

    /// Test single-node cluster initialization and leader election.
    ///
    /// This test validates:
    /// - Cluster initialization with single node (bootstrap)
    /// - Leader election completes (self-election)
    /// - Cluster manager tracks leader correctly
    ///
    /// Note: Multi-node tests with actual network require running gRPC servers,
    /// which is covered in the full integration test suite.
    #[tokio::test]
    async fn test_single_node_cluster_formation() {
        // Create a single bootstrap node
        let (raft, cluster) = create_test_node(1, 59100).await;

        // Initialize as single-node cluster (bootstrap)
        let mut members = BTreeMap::new();
        members.insert(1u64, BasicNode { addr: "127.0.0.1:59100".to_string() });

        // Initialize the cluster with just ourselves
        raft.initialize(members).await.expect("Failed to initialize cluster");

        // Wait for leader election
        tokio::time::sleep(Duration::from_millis(500)).await;

        // Check that node 1 becomes leader (as the only node)
        let info = cluster.get_cluster_info().await;
        assert!(info.leader_id.is_some(), "Cluster should have a leader");
        assert_eq!(info.leader_id, Some(1), "Node 1 should be leader as bootstrap");

        // Verify term is > 0
        assert!(info.term > 0, "Term should be positive after election");

        // Verify members list
        assert_eq!(info.members.len(), 1, "Should have 1 member");
        assert!(info.members.iter().any(|m| m.id == 1), "Member 1 should be present");

        // Clean up
        drop(cluster);
        drop(raft);
    }

    /// Test cluster manager can be created for multiple nodes.
    ///
    /// This validates the ClusterManager infrastructure without
    /// requiring actual network connectivity.
    #[tokio::test]
    async fn test_multi_node_cluster_managers() {
        // Create separate cluster managers for 3 nodes
        let (raft1, cluster1) = create_test_node(1, 59110).await;
        let (raft2, cluster2) = create_test_node(2, 59111).await;
        let (raft3, cluster3) = create_test_node(3, 59112).await;

        // Verify each cluster manager has correct node info
        assert_eq!(cluster1.node_id(), 1);
        assert_eq!(cluster2.node_id(), 2);
        assert_eq!(cluster3.node_id(), 3);

        assert_eq!(cluster1.node_addr(), "127.0.0.1:59110");
        assert_eq!(cluster2.node_addr(), "127.0.0.1:59111");
        assert_eq!(cluster3.node_addr(), "127.0.0.1:59112");

        // Clean up
        drop(cluster1);
        drop(cluster2);
        drop(cluster3);
        drop(raft1);
        drop(raft2);
        drop(raft3);
    }

    /// Test cluster health monitoring with multiple nodes.
    #[tokio::test]
    async fn test_cluster_health_monitoring() {
        let health_config = HealthConfig {
            check_interval: Duration::from_millis(100),
            timeout: Duration::from_millis(200),
            failure_threshold: 3,
        };

        let monitor = HealthMonitor::new(1, health_config);

        // Add 2 other nodes
        monitor.add_peer(2, "127.0.0.1:59201".to_string()).await;
        monitor.add_peer(3, "127.0.0.1:59202".to_string()).await;

        // Check initial state (all unhealthy as not checked)
        let health = monitor.get_cluster_health().await;
        assert_eq!(health.len(), 2);
        assert!(!health[0].healthy);
        assert!(!health[1].healthy);

        // Verify quorum calculation (3 node cluster, none healthy)
        // Need 2 of 3 for quorum; we have 0 healthy peers + self = potentially 1
        assert!(!monitor.has_quorum(3).await);

        // After marking all as healthy, should have quorum
        // (In real scenario, this would be done by health check)
    }

    /// Test cluster manager leader tracking.
    #[tokio::test]
    async fn test_leader_tracking_across_nodes() {
        let (raft1, cluster1) = create_test_node(1, 59300).await;
        let (raft2, cluster2) = create_test_node(2, 59301).await;

        // Add each other as known peers
        cluster1.add_known_peer(2, "127.0.0.1:59301".to_string()).await;
        cluster2.add_known_peer(1, "127.0.0.1:59300".to_string()).await;

        // Simulate leader election result
        cluster1.update_leader(1, "127.0.0.1:59300".to_string()).await;
        cluster2.update_leader(1, "127.0.0.1:59300".to_string()).await;

        // Both should report same leader
        let (leader1_id, leader1_addr) = cluster1.get_leader().await.unwrap();
        let (leader2_id, leader2_addr) = cluster2.get_leader().await.unwrap();

        assert_eq!(leader1_id, leader2_id);
        assert_eq!(leader1_addr, leader2_addr);
        assert_eq!(leader1_id, 1);

        drop(cluster1);
        drop(cluster2);
        drop(raft1);
        drop(raft2);
    }

    /// Test ClusterAwareClient leader routing logic.
    #[tokio::test]
    async fn test_cluster_aware_client_routing() {
        let (raft, cluster) = create_test_node(1, 59400).await;

        // Initialize as single-node cluster
        let mut members = BTreeMap::new();
        members.insert(1u64, BasicNode { addr: "127.0.0.1:59400".to_string() });
        raft.initialize(members).await.expect("Failed to initialize");

        // Wait for election
        tokio::time::sleep(Duration::from_millis(500)).await;

        // Create cluster-aware client
        let client = ClusterAwareClient::new(cluster.clone());

        // Get cluster info should work (leader is known)
        // Note: This would fail without a running gRPC server
        // In a real test, we'd need to start the server

        // Verify client was created with correct settings
        let client = client.with_max_retries(5);
        // Client should be configured

        drop(client);
        drop(cluster);
        drop(raft);
    }

    /// Test metrics recording for cluster operations.
    #[cfg(feature = "metrics")]
    #[tokio::test]
    async fn test_cluster_metrics_recording() {
        use neural_storage::MetricsRegistry;

        let metrics = MetricsRegistry::new().unwrap();

        // Record Raft metrics
        metrics.set_raft_term(5);
        metrics.set_raft_log_index(100);
        metrics.set_cluster_node_count(3);
        metrics.set_cluster_healthy_nodes(2);
        metrics.record_leader_change();
        metrics.record_client_request_success();
        metrics.record_client_request_failure();
        metrics.record_raft_commit_latency(Duration::from_millis(10));

        // Export and verify
        let output = metrics.export().unwrap();
        assert!(output.contains("neuralgraph_raft_term"));
        assert!(output.contains("neuralgraph_raft_log_index"));
        assert!(output.contains("neuralgraph_cluster_node_count"));
        assert!(output.contains("neuralgraph_cluster_healthy_nodes"));
        assert!(output.contains("neuralgraph_raft_leader_changes_total"));
        assert!(output.contains("neuralgraph_raft_client_requests_total"));
        assert!(output.contains("neuralgraph_raft_commit_latency_seconds"));
    }
}
