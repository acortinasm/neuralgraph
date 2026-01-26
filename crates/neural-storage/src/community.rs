//! Community detection using the Leiden algorithm.
//!
//! This module provides community detection functionality using the Leiden algorithm,
//! which is an improvement over the Louvain algorithm for finding community structure
//! in large networks.

use neural_core::NodeId;
use std::collections::HashMap;

/// Type alias for community identifiers.
pub type CommunityId = usize;

/// Community detection results.
///
/// Stores the assignment of each node to a community.
#[derive(Debug, Clone)]
pub struct Communities {
    /// Node index -> Community ID
    assignments: Vec<CommunityId>,
    /// Number of communities detected
    num_communities: usize,
    /// Community ID -> list of node indices (cached for fast lookup)
    community_members: HashMap<CommunityId, Vec<usize>>,
}

impl Communities {
    /// Creates a new Communities instance from node assignments.
    pub fn new(assignments: Vec<CommunityId>) -> Self {
        let num_communities = assignments.iter().max().map(|m| m + 1).unwrap_or(0);

        // Build reverse mapping
        let mut community_members: HashMap<CommunityId, Vec<usize>> = HashMap::new();
        for (node_idx, &community) in assignments.iter().enumerate() {
            community_members
                .entry(community)
                .or_default()
                .push(node_idx);
        }

        Self {
            assignments,
            num_communities,
            community_members,
        }
    }

    /// Gets the community ID for a given node.
    pub fn get(&self, node: NodeId) -> Option<CommunityId> {
        self.assignments.get(node.as_usize()).copied()
    }

    /// Returns all nodes in a given community.
    pub fn nodes_in_community(&self, community: CommunityId) -> Vec<NodeId> {
        self.community_members
            .get(&community)
            .map(|members| members.iter().map(|&idx| NodeId::new(idx as u64)).collect())
            .unwrap_or_default()
    }

    /// Returns the number of communities detected.
    pub fn num_communities(&self) -> usize {
        self.num_communities
    }

    /// Returns the total number of nodes.
    pub fn num_nodes(&self) -> usize {
        self.assignments.len()
    }

    /// Returns an iterator over (CommunityId, node count) pairs.
    pub fn community_sizes(&self) -> impl Iterator<Item = (CommunityId, usize)> + '_ {
        self.community_members
            .iter()
            .map(|(&id, members)| (id, members.len()))
    }

    /// Returns the raw assignments vector.
    pub fn assignments(&self) -> &[CommunityId] {
        &self.assignments
    }
}

/// Detects communities in a graph using the Leiden algorithm.
///
/// Takes edges as (source, target) pairs and returns community assignments.
pub fn detect_communities_leiden(edges: &[(usize, usize)], num_nodes: usize) -> Communities {
    use fa_leiden_cd::{Graph, TrivialModularityOptimizer};

    if edges.is_empty() || num_nodes == 0 {
        return Communities::new(vec![]);
    }

    // Build fa-leiden-cd Graph
    let mut graph: Graph<(), ()> = Graph::default();

    // Add all nodes
    for _ in 0..num_nodes {
        graph.add_node(());
    }

    // Add edges with unit weight
    for &(src, dst) in edges {
        graph.add_edge(src, dst, (), 1.0);
    }

    // Run Leiden algorithm
    let mut optimizer = TrivialModularityOptimizer {
        parallel_scale: 1000,
        tol: 0.0001,
    };
    let result_graph = graph.leiden(None, &mut optimizer);

    // Extract community assignments from result
    // The result graph contains Community nodes - we need to flatten to node assignments
    let mut assignments = vec![0usize; num_nodes];

    // Get node data slice which contains Community enum for each cluster
    let communities_data = result_graph.node_data_slice();

    // Flatten community structure to get per-node assignments
    for (community_id, community) in communities_data.iter().enumerate() {
        collect_nodes_from_community(community, community_id, &mut assignments);
    }

    Communities::new(assignments)
}

/// Recursively collects node IDs from a Community structure.
fn collect_nodes_from_community(
    community: &fa_leiden_cd::Community,
    community_id: usize,
    assignments: &mut [usize],
) {
    match community {
        fa_leiden_cd::Community::L1Community(nodes) => {
            for &node_id in nodes {
                if node_id < assignments.len() {
                    assignments[node_id] = community_id;
                }
            }
        }
        fa_leiden_cd::Community::LNCommunity(sub_communities) => {
            for sub in sub_communities {
                collect_nodes_from_community(sub, community_id, assignments);
            }
        }
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_communities_basic() {
        let assignments = vec![0, 0, 1, 1, 2];
        let communities = Communities::new(assignments);

        assert_eq!(communities.num_communities(), 3);
        assert_eq!(communities.num_nodes(), 5);
        assert_eq!(communities.get(NodeId::new(0)), Some(0));
        assert_eq!(communities.get(NodeId::new(2)), Some(1));
        assert_eq!(communities.get(NodeId::new(4)), Some(2));
    }

    #[test]
    fn test_nodes_in_community() {
        let assignments = vec![0, 0, 1, 1, 0];
        let communities = Communities::new(assignments);

        let community_0 = communities.nodes_in_community(0);
        assert_eq!(community_0.len(), 3);

        let community_1 = communities.nodes_in_community(1);
        assert_eq!(community_1.len(), 2);
    }

    #[test]
    fn test_leiden_basic() {
        // Create a graph with two clear communities
        // Community 1: nodes 0, 1, 2 (fully connected)
        // Community 2: nodes 3, 4, 5 (fully connected)
        // One edge between communities: 2 -> 3
        let edges = vec![
            (0, 1),
            (1, 0),
            (0, 2),
            (2, 0),
            (1, 2),
            (2, 1),
            (3, 4),
            (4, 3),
            (3, 5),
            (5, 3),
            (4, 5),
            (5, 4),
            (2, 3),
            (3, 2), // Bridge
        ];

        let communities = detect_communities_leiden(&edges, 6);

        // Should detect communities
        assert!(communities.num_communities() >= 1);
        assert_eq!(communities.num_nodes(), 6);
    }

    #[test]
    fn test_leiden_empty() {
        let communities = detect_communities_leiden(&[], 0);
        assert_eq!(communities.num_communities(), 0);
        assert_eq!(communities.num_nodes(), 0);
    }
}
