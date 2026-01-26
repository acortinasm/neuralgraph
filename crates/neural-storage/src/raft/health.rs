//! Health monitoring for NeuralGraphDB Raft cluster.
//!
//! This module provides the `HealthMonitor` for periodic health checks
//! and failure detection of cluster nodes.

use std::collections::BTreeMap;
use std::sync::Arc;
use std::time::{Duration, Instant};

use tokio::sync::RwLock;
use tokio::task::JoinHandle;

use super::network::proto::raft_client::RaftClient;
use super::network::proto::HealthCheckRequest;
use super::types::RaftNodeId;

/// State of a node in the cluster.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NodeState {
    Leader,
    Follower,
    Candidate,
    Learner,
    Unknown,
}

impl NodeState {
    /// Parse from string representation.
    pub fn from_str(s: &str) -> Self {
        match s.to_lowercase().as_str() {
            "leader" => NodeState::Leader,
            "follower" => NodeState::Follower,
            "candidate" => NodeState::Candidate,
            "learner" => NodeState::Learner,
            _ => NodeState::Unknown,
        }
    }

    /// Convert to string representation.
    pub fn as_str(&self) -> &'static str {
        match self {
            NodeState::Leader => "leader",
            NodeState::Follower => "follower",
            NodeState::Candidate => "candidate",
            NodeState::Learner => "learner",
            NodeState::Unknown => "unknown",
        }
    }
}

impl std::fmt::Display for NodeState {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.as_str())
    }
}

/// Health status of a single node.
#[derive(Debug, Clone)]
pub struct NodeHealth {
    pub node_id: RaftNodeId,
    pub addr: String,
    pub healthy: bool,
    pub last_check: Instant,
    pub consecutive_failures: u32,
    pub state: NodeState,
    pub term: u64,
    pub last_log_index: u64,
    pub latency_ms: Option<u64>,
}

impl NodeHealth {
    /// Create a new NodeHealth in unknown state.
    pub fn new(node_id: RaftNodeId, addr: String) -> Self {
        Self {
            node_id,
            addr,
            healthy: false,
            last_check: Instant::now(),
            consecutive_failures: 0,
            state: NodeState::Unknown,
            term: 0,
            last_log_index: 0,
            latency_ms: None,
        }
    }

    /// Mark this node as healthy with updated info.
    pub fn mark_healthy(&mut self, state: NodeState, term: u64, last_log_index: u64, latency_ms: u64) {
        self.healthy = true;
        self.last_check = Instant::now();
        self.consecutive_failures = 0;
        self.state = state;
        self.term = term;
        self.last_log_index = last_log_index;
        self.latency_ms = Some(latency_ms);
    }

    /// Mark this node as unhealthy.
    pub fn mark_unhealthy(&mut self) {
        self.healthy = false;
        self.last_check = Instant::now();
        self.consecutive_failures += 1;
        self.latency_ms = None;
    }

    /// Check if the node is considered dead (too many consecutive failures).
    pub fn is_dead(&self, threshold: u32) -> bool {
        self.consecutive_failures >= threshold
    }
}

/// Configuration for health monitoring.
#[derive(Debug, Clone)]
pub struct HealthConfig {
    /// Interval between health checks.
    pub check_interval: Duration,
    /// Timeout for individual health check RPCs.
    pub timeout: Duration,
    /// Number of consecutive failures before considering a node dead.
    pub failure_threshold: u32,
}

impl Default for HealthConfig {
    fn default() -> Self {
        Self {
            check_interval: Duration::from_millis(500),
            timeout: Duration::from_millis(200),
            failure_threshold: 3,
        }
    }
}

/// Callback type for node state changes.
pub type OnNodeUnhealthy = Arc<dyn Fn(RaftNodeId) + Send + Sync>;
pub type OnNodeHealthy = Arc<dyn Fn(RaftNodeId) + Send + Sync>;

/// Monitors health of cluster nodes via periodic health checks.
pub struct HealthMonitor {
    /// This node's ID.
    node_id: RaftNodeId,
    /// Health check configuration.
    config: HealthConfig,
    /// Health status of all known peers.
    peers: Arc<RwLock<BTreeMap<RaftNodeId, NodeHealth>>>,
    /// Running flag for the background task.
    running: Arc<RwLock<bool>>,
    /// Callback when a node becomes unhealthy.
    on_unhealthy: Option<OnNodeUnhealthy>,
    /// Callback when a node becomes healthy.
    on_healthy: Option<OnNodeHealthy>,
}

impl HealthMonitor {
    /// Creates a new HealthMonitor.
    pub fn new(node_id: RaftNodeId, config: HealthConfig) -> Self {
        Self {
            node_id,
            config,
            peers: Arc::new(RwLock::new(BTreeMap::new())),
            running: Arc::new(RwLock::new(false)),
            on_unhealthy: None,
            on_healthy: None,
        }
    }

    /// Set callback for when a node becomes unhealthy.
    pub fn on_node_unhealthy(&mut self, callback: impl Fn(RaftNodeId) + Send + Sync + 'static) {
        self.on_unhealthy = Some(Arc::new(callback));
    }

    /// Set callback for when a node becomes healthy.
    pub fn on_node_healthy(&mut self, callback: impl Fn(RaftNodeId) + Send + Sync + 'static) {
        self.on_healthy = Some(Arc::new(callback));
    }

    /// Add a peer to monitor.
    pub async fn add_peer(&self, node_id: RaftNodeId, addr: String) {
        let mut peers = self.peers.write().await;
        peers.insert(node_id, NodeHealth::new(node_id, addr));
    }

    /// Remove a peer from monitoring.
    pub async fn remove_peer(&self, node_id: RaftNodeId) {
        let mut peers = self.peers.write().await;
        peers.remove(&node_id);
    }

    /// Start the background health check loop.
    pub fn start(&self) -> JoinHandle<()> {
        let peers = self.peers.clone();
        let running = self.running.clone();
        let config = self.config.clone();
        let node_id = self.node_id;
        let on_unhealthy = self.on_unhealthy.clone();
        let on_healthy = self.on_healthy.clone();

        tokio::spawn(async move {
            {
                let mut r = running.write().await;
                *r = true;
            }

            loop {
                {
                    let r = running.read().await;
                    if !*r {
                        break;
                    }
                }

                // Get list of peers to check
                let peer_list: Vec<(RaftNodeId, String)> = {
                    let peers = peers.read().await;
                    peers
                        .values()
                        .map(|h| (h.node_id, h.addr.clone()))
                        .collect()
                };

                // Check each peer
                for (peer_id, peer_addr) in peer_list {
                    if peer_id == node_id {
                        continue; // Don't check ourselves
                    }

                    let result =
                        Self::check_node_health(peer_id, &peer_addr, config.timeout).await;

                    // Update health status
                    let mut peers = peers.write().await;
                    if let Some(health) = peers.get_mut(&peer_id) {
                        let was_healthy = health.healthy;

                        match result {
                            Ok((state, term, last_log_index, latency_ms)) => {
                                health.mark_healthy(state, term, last_log_index, latency_ms);
                                if !was_healthy {
                                    if let Some(ref callback) = on_healthy {
                                        callback(peer_id);
                                    }
                                }
                            }
                            Err(_) => {
                                health.mark_unhealthy();
                                if was_healthy {
                                    if let Some(ref callback) = on_unhealthy {
                                        callback(peer_id);
                                    }
                                }
                            }
                        }
                    }
                }

                tokio::time::sleep(config.check_interval).await;
            }
        })
    }

    /// Stop the background health check loop.
    pub async fn stop(&self) {
        let mut running = self.running.write().await;
        *running = false;
    }

    /// Check health of a single node.
    pub async fn check_node(
        &self,
        node_id: RaftNodeId,
        addr: &str,
    ) -> Result<NodeHealth, String> {
        let start = Instant::now();
        let result = Self::check_node_health(node_id, addr, self.config.timeout).await;

        match result {
            Ok((state, term, last_log_index, latency_ms)) => {
                let mut health = NodeHealth::new(node_id, addr.to_string());
                health.mark_healthy(state, term, last_log_index, latency_ms);
                Ok(health)
            }
            Err(e) => {
                let mut health = NodeHealth::new(node_id, addr.to_string());
                health.mark_unhealthy();
                Err(e)
            }
        }
    }

    /// Internal function to perform a health check RPC.
    async fn check_node_health(
        node_id: RaftNodeId,
        addr: &str,
        timeout: Duration,
    ) -> Result<(NodeState, u64, u64, u64), String> {
        let start = Instant::now();

        let connect_result = tokio::time::timeout(
            timeout,
            RaftClient::connect(format!("http://{}", addr)),
        )
        .await;

        let mut client = match connect_result {
            Ok(Ok(client)) => client,
            Ok(Err(e)) => return Err(format!("Connection failed: {}", e)),
            Err(_) => return Err("Connection timeout".to_string()),
        };

        let request = tonic::Request::new(HealthCheckRequest { node_id });

        let response_result = tokio::time::timeout(timeout, client.health_check(request)).await;

        match response_result {
            Ok(Ok(response)) => {
                let latency_ms = start.elapsed().as_millis() as u64;
                let resp = response.into_inner();

                if resp.healthy {
                    Ok((
                        NodeState::from_str(&resp.state),
                        resp.term,
                        resp.last_log_index,
                        latency_ms,
                    ))
                } else {
                    Err("Node reported unhealthy".to_string())
                }
            }
            Ok(Err(e)) => Err(format!("RPC failed: {}", e)),
            Err(_) => Err("RPC timeout".to_string()),
        }
    }

    /// Get health status of all known nodes.
    pub async fn get_cluster_health(&self) -> Vec<NodeHealth> {
        let peers = self.peers.read().await;
        peers.values().cloned().collect()
    }

    /// Get health status of a specific node.
    pub async fn get_node_health(&self, node_id: RaftNodeId) -> Option<NodeHealth> {
        let peers = self.peers.read().await;
        peers.get(&node_id).cloned()
    }

    /// Get count of healthy nodes.
    pub async fn healthy_count(&self) -> usize {
        let peers = self.peers.read().await;
        peers.values().filter(|h| h.healthy).count()
    }

    /// Get count of unhealthy nodes.
    pub async fn unhealthy_count(&self) -> usize {
        let peers = self.peers.read().await;
        peers.values().filter(|h| !h.healthy).count()
    }

    /// Check if we have a quorum of healthy nodes.
    pub async fn has_quorum(&self, cluster_size: usize) -> bool {
        let healthy = self.healthy_count().await;
        // Quorum requires majority: (n/2) + 1
        healthy >= (cluster_size / 2) + 1
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_node_state_parsing() {
        assert_eq!(NodeState::from_str("leader"), NodeState::Leader);
        assert_eq!(NodeState::from_str("FOLLOWER"), NodeState::Follower);
        assert_eq!(NodeState::from_str("candidate"), NodeState::Candidate);
        assert_eq!(NodeState::from_str("learner"), NodeState::Learner);
        assert_eq!(NodeState::from_str("garbage"), NodeState::Unknown);
    }

    #[test]
    fn test_node_health() {
        let mut health = NodeHealth::new(1, "127.0.0.1:50052".to_string());
        assert!(!health.healthy);
        assert_eq!(health.consecutive_failures, 0);

        health.mark_unhealthy();
        assert!(!health.healthy);
        assert_eq!(health.consecutive_failures, 1);

        health.mark_unhealthy();
        assert_eq!(health.consecutive_failures, 2);

        health.mark_healthy(NodeState::Follower, 5, 100, 10);
        assert!(health.healthy);
        assert_eq!(health.consecutive_failures, 0);
        assert_eq!(health.state, NodeState::Follower);
    }

    #[test]
    fn test_is_dead() {
        let mut health = NodeHealth::new(1, "127.0.0.1:50052".to_string());
        assert!(!health.is_dead(3));

        health.mark_unhealthy();
        health.mark_unhealthy();
        assert!(!health.is_dead(3));

        health.mark_unhealthy();
        assert!(health.is_dead(3));
    }

    #[test]
    fn test_health_config_default() {
        let config = HealthConfig::default();
        assert_eq!(config.check_interval, Duration::from_millis(500));
        assert_eq!(config.timeout, Duration::from_millis(200));
        assert_eq!(config.failure_threshold, 3);
    }
}
