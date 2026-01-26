//! Compressed Sparse Column (CSC) matrix for O(1) incoming edge lookups.
//!
//! This module provides a CSC representation of the graph adjacency matrix,
//! which is the transpose of CSR. While CSR provides O(1) outgoing neighbor
//! lookups, CSC provides O(1) incoming neighbor lookups.
//!
//! Memory trade-off: ~2x edge storage for O(degree) reverse lookups vs O(E) scans.

use neural_core::NodeId;
use serde::{Deserialize, Serialize};

/// Compressed Sparse Column matrix for efficient incoming edge lookups.
///
/// CSC stores the transpose of the adjacency matrix:
/// - `col_ptr[i]` to `col_ptr[i+1]` gives the range of incoming edges for node i
/// - `row_indices[j]` gives the source node for edge j
///
/// This allows O(degree) incoming neighbor lookups instead of O(E) full scans.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct CscMatrix {
    /// Column pointers - col_ptr[i] to col_ptr[i+1] gives range for node i's incoming edges
    pub col_ptr: Vec<usize>,
    /// Row indices - source nodes for each edge
    pub row_indices: Vec<NodeId>,
    /// Edge IDs corresponding to each entry (for returning edge metadata)
    pub edge_ids: Vec<u64>,
    /// Number of nodes
    pub num_nodes: usize,
}

impl CscMatrix {
    /// Creates an empty CSC matrix.
    pub fn empty() -> Self {
        Self {
            col_ptr: vec![0],
            row_indices: Vec::new(),
            edge_ids: Vec::new(),
            num_nodes: 0,
        }
    }

    /// Creates a CSC matrix from a CSR matrix by transposing.
    ///
    /// This is an O(E + V) operation where E is edges and V is nodes.
    pub fn from_csr(csr: &super::CsrMatrix) -> Self {
        let num_nodes = csr.num_nodes;
        let num_edges = csr.col_indices.len();

        if num_nodes == 0 || num_edges == 0 {
            return Self {
                col_ptr: vec![0; num_nodes + 1],
                row_indices: Vec::new(),
                edge_ids: Vec::new(),
                num_nodes,
            };
        }

        // Step 1: Count incoming edges for each node (column counts in CSC)
        let mut col_counts = vec![0usize; num_nodes];
        for &target in &csr.col_indices {
            let target_idx = target.as_usize();
            if target_idx < num_nodes {
                col_counts[target_idx] += 1;
            }
        }

        // Step 2: Build col_ptr from counts (cumulative sum)
        let mut col_ptr = Vec::with_capacity(num_nodes + 1);
        col_ptr.push(0);
        for count in &col_counts {
            col_ptr.push(col_ptr.last().unwrap() + count);
        }

        // Step 3: Fill row_indices and edge_ids
        // We need to iterate through CSR and place each edge in the correct position
        let mut row_indices = vec![NodeId::new(0); num_edges];
        let mut edge_ids = vec![0u64; num_edges];
        let mut col_positions = vec![0usize; num_nodes]; // Current write position for each column

        for source in 0..num_nodes {
            let start = csr.row_ptr[source];
            let end = csr.row_ptr[source + 1];

            for edge_idx in start..end {
                let target = csr.col_indices[edge_idx];
                let target_idx = target.as_usize();
                if target_idx < num_nodes {
                    // Calculate position in CSC arrays
                    let pos = col_ptr[target_idx] + col_positions[target_idx];
                    row_indices[pos] = NodeId::new(source as u64);
                    edge_ids[pos] = edge_idx as u64;
                    col_positions[target_idx] += 1;
                }
            }
        }

        Self {
            col_ptr,
            row_indices,
            edge_ids,
            num_nodes,
        }
    }

    /// Returns the number of nodes.
    pub fn node_count(&self) -> usize {
        self.num_nodes
    }

    /// Returns the number of edges.
    pub fn edge_count(&self) -> usize {
        self.row_indices.len()
    }

    /// Returns the incoming neighbors of a node - O(degree) operation.
    ///
    /// This is the main optimization: instead of scanning all edges O(E),
    /// we directly access the incoming edges for a specific node.
    pub fn incoming_neighbors(&self, node: NodeId) -> impl Iterator<Item = NodeId> + '_ {
        let idx = node.as_usize();
        if idx < self.num_nodes {
            let start = self.col_ptr[idx];
            let end = self.col_ptr[idx + 1];
            self.row_indices[start..end].iter().copied()
        } else {
            [].iter().copied()
        }
    }

    /// Returns the incoming neighbors with their edge IDs - O(degree) operation.
    pub fn incoming_neighbors_with_ids(&self, node: NodeId) -> IncomingNeighborsWithIds<'_> {
        let idx = node.as_usize();
        let (start, end) = if idx < self.num_nodes {
            (self.col_ptr[idx], self.col_ptr[idx + 1])
        } else {
            (0, 0)
        };
        IncomingNeighborsWithIds {
            edge_ids: &self.edge_ids[start..end],
            row_indices: &self.row_indices[start..end],
            pos: 0,
        }
    }
}

/// Iterator over incoming neighbors with edge IDs.
pub struct IncomingNeighborsWithIds<'a> {
    edge_ids: &'a [u64],
    row_indices: &'a [NodeId],
    pos: usize,
}

impl<'a> Iterator for IncomingNeighborsWithIds<'a> {
    type Item = (u64, NodeId);

    fn next(&mut self) -> Option<Self::Item> {
        if self.pos < self.edge_ids.len() {
            let result = (self.edge_ids[self.pos], self.row_indices[self.pos]);
            self.pos += 1;
            Some(result)
        } else {
            None
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let remaining = self.edge_ids.len() - self.pos;
        (remaining, Some(remaining))
    }
}

impl CscMatrix {
    /// Returns the in-degree of a node - O(1) operation.
    pub fn in_degree(&self, node: NodeId) -> usize {
        let idx = node.as_usize();
        if idx < self.num_nodes {
            self.col_ptr[idx + 1] - self.col_ptr[idx]
        } else {
            0
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use neural_core::Edge;

    #[test]
    fn test_csc_from_csr() {
        // Graph: 0 -> 1, 0 -> 2, 1 -> 2
        // CSR for outgoing: row 0 has [1, 2], row 1 has [2], row 2 has []
        // CSC for incoming: col 0 has [], col 1 has [0], col 2 has [0, 1]
        let edges = vec![
            Edge::new(0u64, 1u64),
            Edge::new(0u64, 2u64),
            Edge::new(1u64, 2u64),
        ];
        let csr = super::super::CsrMatrix::from_edges(&edges, 3);
        let csc = CscMatrix::from_csr(&csr);

        assert_eq!(csc.node_count(), 3);
        assert_eq!(csc.edge_count(), 3);

        // Node 0 has no incoming edges
        assert_eq!(csc.in_degree(NodeId::new(0)), 0);
        let incoming_0: Vec<_> = csc.incoming_neighbors(NodeId::new(0)).collect();
        assert!(incoming_0.is_empty());

        // Node 1 has 1 incoming edge from node 0
        assert_eq!(csc.in_degree(NodeId::new(1)), 1);
        let incoming_1: Vec<_> = csc.incoming_neighbors(NodeId::new(1)).collect();
        assert_eq!(incoming_1, vec![NodeId::new(0)]);

        // Node 2 has 2 incoming edges from nodes 0 and 1
        assert_eq!(csc.in_degree(NodeId::new(2)), 2);
        let mut incoming_2: Vec<_> = csc.incoming_neighbors(NodeId::new(2)).collect();
        incoming_2.sort();
        assert_eq!(incoming_2, vec![NodeId::new(0), NodeId::new(1)]);
    }

    #[test]
    fn test_csc_empty() {
        let csc = CscMatrix::empty();
        assert_eq!(csc.node_count(), 0);
        assert_eq!(csc.edge_count(), 0);
    }

    #[test]
    fn test_csc_no_edges() {
        let edges: Vec<Edge> = vec![];
        let csr = super::super::CsrMatrix::from_edges(&edges, 3);
        let csc = CscMatrix::from_csr(&csr);

        assert_eq!(csc.node_count(), 3);
        assert_eq!(csc.edge_count(), 0);

        for i in 0..3 {
            assert_eq!(csc.in_degree(NodeId::new(i)), 0);
        }
    }

    #[test]
    fn test_csc_incoming_with_ids() {
        let edges = vec![
            Edge::new(0u64, 2u64),
            Edge::new(1u64, 2u64),
        ];
        let csr = super::super::CsrMatrix::from_edges(&edges, 3);
        let csc = CscMatrix::from_csr(&csr);

        let incoming_with_ids: Vec<_> = csc.incoming_neighbors_with_ids(NodeId::new(2)).collect();
        assert_eq!(incoming_with_ids.len(), 2);

        // Verify we get back the source nodes
        let sources: Vec<_> = incoming_with_ids.iter().map(|(_, src)| *src).collect();
        assert!(sources.contains(&NodeId::new(0)));
        assert!(sources.contains(&NodeId::new(1)));
    }
}
