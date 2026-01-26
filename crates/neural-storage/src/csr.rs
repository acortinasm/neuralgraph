use neural_core::{Edge, Graph, NodeId};
use serde::{Deserialize, Serialize};

/// Compressed Sparse Row (CSR) matrix representation of a graph.
///
/// Efficient for read-heavy workloads.
/// O(1) neighbor access.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct CsrMatrix {
    /// Offsets for each row (node). Size = num_nodes + 1.
    pub row_ptr: Vec<usize>,
    /// Column indices (neighbors). Size = num_edges.
    pub col_indices: Vec<NodeId>,
    /// Number of nodes.
    pub num_nodes: usize,
}

/// Statistics about a CSR matrix.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CsrStats {
    /// Number of nodes
    pub node_count: usize,
    /// Number of edges
    pub edge_count: usize,
    /// Maximum out-degree
    pub max_degree: usize,
    /// Minimum out-degree
    pub min_degree: usize,
    /// Average out-degree
    pub avg_degree: f64,
    /// Estimated memory usage in bytes
    pub memory_bytes: usize,
}

impl CsrMatrix {
    pub fn new(row_ptr: Vec<usize>, col_indices: Vec<NodeId>, num_nodes: usize) -> Self {
        Self {
            row_ptr,
            col_indices,
            num_nodes,
        }
    }

    pub fn empty() -> Self {
        Self {
            row_ptr: vec![0],
            col_indices: vec![],
            num_nodes: 0,
        }
    }

    /// Validates the internal consistency of the CSR matrix.
    pub fn validate(&self) -> Result<(), String> {
        if self.row_ptr.is_empty() {
            return Err("row_ptr is empty".to_string());
        }

        if self.row_ptr[0] != 0 {
            return Err("row_ptr[0] must be 0".to_string());
        }

        if *self.row_ptr.last().unwrap() != self.col_indices.len() {
            return Err("last element of row_ptr must equal col_indices.len()".to_string());
        }

        for i in 0..self.row_ptr.len() - 1 {
            if self.row_ptr[i] > self.row_ptr[i + 1] {
                return Err(format!("row_ptr[{}] > row_ptr[{}]", i, i + 1));
            }
        }

        for &col in &self.col_indices {
            if col.as_usize() >= self.num_nodes {
                return Err(format!(
                    "column index {} out of bounds (max {})",
                    col.as_usize(),
                    self.num_nodes - 1
                ));
            }
        }

        Ok(())
    }

    /// Returns statistics about the matrix.
    pub fn stats(&self) -> CsrStats {
        let node_count = self.num_nodes;
        let edge_count = self.col_indices.len();

        let mut max_degree = 0;
        let mut min_degree = if node_count > 0 { usize::MAX } else { 0 };
        let mut total_degree = 0;

        for i in 0..node_count {
            let degree = self.row_ptr[i + 1] - self.row_ptr[i];
            max_degree = max_degree.max(degree);
            min_degree = min_degree.min(degree);
            total_degree += degree;
        }

        if node_count == 0 {
            min_degree = 0;
        }

        let avg_degree = if node_count > 0 {
            total_degree as f64 / node_count as f64
        } else {
            0.0
        };

        let memory_bytes = std::mem::size_of::<Self>()
            + self.row_ptr.len() * std::mem::size_of::<usize>()
            + self.col_indices.len() * std::mem::size_of::<NodeId>();

        CsrStats {
            node_count,
            edge_count,
            max_degree,
            min_degree,
            avg_degree,
            memory_bytes,
        }
    }

    /// Constructs a CSR matrix from a list of edges.
    /// Assumes edges are directed.
    pub fn from_edges(edges: &[Edge], num_nodes: usize) -> Self {
        let mut degree = vec![0; num_nodes];
        for edge in edges {
            if edge.source.as_usize() < num_nodes {
                degree[edge.source.as_usize()] += 1;
            }
        }

        let mut row_ptr = vec![0; num_nodes + 1];
        let mut total_edges = 0;
        for i in 0..num_nodes {
            row_ptr[i] = total_edges;
            total_edges += degree[i];
        }
        row_ptr[num_nodes] = total_edges;

        let mut col_indices = vec![NodeId(0); total_edges];
        let mut current_pos = row_ptr.clone();

        for edge in edges {
            if edge.source.as_usize() < num_nodes {
                let pos = current_pos[edge.source.as_usize()];
                col_indices[pos] = edge.target;
                current_pos[edge.source.as_usize()] += 1;
            }
        }
        
        // Sort neighbors for faster intersection/lookup
        for i in 0..num_nodes {
            let start = row_ptr[i];
            let end = row_ptr[i+1];
            col_indices[start..end].sort();
        }

        Self {
            row_ptr,
            col_indices,
            num_nodes,
        }
    }
    
    // Direct slice access for performance (internal use)
    pub fn neighbors_slice(&self, node: usize) -> &[NodeId] {
        if node >= self.num_nodes {
            return &[];
        }
        let start = self.row_ptr[node];
        let end = self.row_ptr[node + 1];
        &self.col_indices[start..end]
    }
    
    // Matrix-Vector multiplication primitive (mxv)
    // y = A * x
    pub fn mxv<S, VIn, VOut>(&self, _x: &VIn, _semiring: S) -> VOut
    where
        S: neural_core::Semiring<bool>, // Simplified for now, should be generic
        VIn: neural_core::Vector<bool>,
        VOut: neural_core::Vector<bool> + Default,
    {
        // Placeholder for now as we fix the build
        VOut::default()
    }
}

impl Graph for CsrMatrix {
    fn node_count(&self) -> usize {
        self.num_nodes
    }

    fn edge_count(&self) -> usize {
        self.col_indices.len()
    }

    fn neighbors(&self, node: NodeId) -> impl Iterator<Item = NodeId> {
        let start = self.row_ptr[node.as_usize()];
        let end = self.row_ptr[node.as_usize() + 1];
        self.col_indices[start..end].iter().copied()
    }

    fn out_degree(&self, node: NodeId) -> usize {
        let u = node.as_usize();
        if u >= self.num_nodes {
            return 0;
        }
        self.row_ptr[u + 1] - self.row_ptr[u]
    }
}
