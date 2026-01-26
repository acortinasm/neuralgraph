//! CSV data loading for NeuralGraphDB
//!
//! This module provides functionality to load graph data from CSV files.
//!
//! ## Node CSV Format
//!
//! ```csv
//! id,label,prop1,prop2,...
//! 0,Person,Alice,30
//! 1,Person,Bob,25
//! ```
//!
//! - First column must be `id` (numeric node ID)
//! - Second column is `label` (optional, can be empty)
//! - Remaining columns are properties
//!
//! ## Edge CSV Format
//!
//! ```csv
//! source,target,label
//! 0,1,KNOWS
//! 1,2,WORKS_AT
//! ```

use crate::{GraphStore, GraphStoreBuilder};
use neural_core::PropertyValue;
use std::collections::HashMap;
use std::path::Path;
use thiserror::Error;

/// Errors that can occur during CSV loading
#[derive(Error, Debug)]
pub enum CsvError {
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    #[error("CSV parsing error: {0}")]
    Csv(#[from] csv::Error),

    #[error("Missing required column: {0}")]
    MissingColumn(String),

    #[error("Invalid node ID at row {row}: {value}")]
    InvalidNodeId { row: usize, value: String },

    #[error("Invalid edge at row {row}: source={source_id}, target={target_id}")]
    InvalidEdge {
        row: usize,
        source_id: String,
        target_id: String,
    },
}

/// Result type for CSV operations
pub type Result<T> = std::result::Result<T, CsvError>;

/// Parsed node data from CSV
#[derive(Debug, Clone)]
pub struct NodeData {
    pub id: u64,
    pub label: Option<String>,
    pub properties: HashMap<String, PropertyValue>,
}

/// Parsed edge data from CSV
#[derive(Debug, Clone)]
pub struct EdgeData {
    pub source: u64,
    pub target: u64,
    pub label: Option<String>,
}

/// Loads nodes from a CSV file.
///
/// Expected format:
/// - First column: `id` (required, numeric)
/// - Second column: `label` (optional, can be empty)
/// - Remaining columns: properties (column name = property key)
///
/// # Example
///
/// ```csv
/// id,label,name,age
/// 0,Person,Alice,30
/// 1,Person,Bob,25
/// 2,Company,Acme,
/// ```
pub fn load_nodes_csv<P: AsRef<Path>>(path: P) -> Result<Vec<NodeData>> {
    let mut reader = csv::Reader::from_path(path)?;
    let headers: Vec<String> = reader.headers()?.iter().map(|s| s.to_string()).collect();

    // Validate required columns
    if headers.is_empty() || headers[0].to_lowercase() != "id" {
        return Err(CsvError::MissingColumn("id".to_string()));
    }

    let has_label = headers.len() > 1 && headers[1].to_lowercase() == "label";
    let prop_start = if has_label { 2 } else { 1 };

    let mut nodes = Vec::new();

    for (row_idx, result) in reader.records().enumerate() {
        let record = result?;

        // Parse node ID
        let id_str = record.get(0).unwrap_or("");
        let id: u64 = id_str.parse().map_err(|_| CsvError::InvalidNodeId {
            row: row_idx + 2, // +2 for header and 1-indexing
            value: id_str.to_string(),
        })?;

        // Parse label (if present)
        let label = if has_label {
            let label_str = record.get(1).unwrap_or("").trim();
            if label_str.is_empty() {
                None
            } else {
                Some(label_str.to_string())
            }
        } else {
            None
        };

        // Parse properties
        let mut properties = HashMap::new();
        for (i, header) in headers.iter().enumerate().skip(prop_start) {
            if let Some(value) = record.get(i) {
                let value = value.trim();
                if !value.is_empty() {
                    properties.insert(header.clone(), parse_property_value(value));
                }
            }
        }

        nodes.push(NodeData {
            id,
            label,
            properties,
        });
    }

    Ok(nodes)
}

/// Loads edges from a CSV file.
///
/// Expected format:
/// - First column: `source` (required, numeric node ID)
/// - Second column: `target` (required, numeric node ID)
/// - Third column: `label` (optional, edge type)
///
/// # Example
///
/// ```csv
/// source,target,label
/// 0,1,KNOWS
/// 1,2,WORKS_AT
/// ```
pub fn load_edges_csv<P: AsRef<Path>>(path: P) -> Result<Vec<EdgeData>> {
    let mut reader = csv::Reader::from_path(path)?;
    let headers: Vec<String> = reader.headers()?.iter().map(|s| s.to_lowercase()).collect();

    // Validate required columns
    if !headers.contains(&"source".to_string()) {
        return Err(CsvError::MissingColumn("source".to_string()));
    }
    if !headers.contains(&"target".to_string()) {
        return Err(CsvError::MissingColumn("target".to_string()));
    }

    let source_idx = headers.iter().position(|h| h == "source").unwrap();
    let target_idx = headers.iter().position(|h| h == "target").unwrap();
    let label_idx = headers.iter().position(|h| h == "label");

    let mut edges = Vec::new();

    for (row_idx, result) in reader.records().enumerate() {
        let record = result?;

        let source_str = record.get(source_idx).unwrap_or("");
        let target_str = record.get(target_idx).unwrap_or("");

        let source: u64 = source_str.parse().map_err(|_| CsvError::InvalidEdge {
            row: row_idx + 2,
            source_id: source_str.to_string(),
            target_id: target_str.to_string(),
        })?;

        let target: u64 = target_str.parse().map_err(|_| CsvError::InvalidEdge {
            row: row_idx + 2,
            source_id: source_str.to_string(),
            target_id: target_str.to_string(),
        })?;

        let label = label_idx.and_then(|idx| {
            record.get(idx).and_then(|s| {
                let s = s.trim();
                if s.is_empty() {
                    None
                } else {
                    Some(s.to_string())
                }
            })
        });

        edges.push(EdgeData {
            source,
            target,
            label,
        });
    }

    Ok(edges)
}

/// Loads a complete graph from CSV files.
///
/// # Arguments
///
/// * `nodes_path` - Optional path to nodes CSV file
/// * `edges_path` - Path to edges CSV file
///
/// If `nodes_path` is None, nodes will be inferred from edges.
pub fn load_graph_from_csv<P: AsRef<Path>>(
    nodes_path: Option<P>,
    edges_path: P,
) -> Result<GraphStore> {
    let mut builder = GraphStoreBuilder::new();

    // Load nodes if provided
    if let Some(path) = nodes_path {
        let nodes = load_nodes_csv(path)?;
        for node in nodes {
                            if let Some(label) = node.label {
                                builder = builder.add_labeled_node(node.id, label, node.properties);
                            } else {
                                builder = builder.add_node(node.id, node.properties);
                            }
                        }
                    }
            
                // Load edges
                let edges = load_edges_csv(edges_path)?;
                for edge in edges {
                    if let Some(label) = edge.label {
                        builder = builder.add_labeled_edge(edge.source, edge.target, neural_core::Label::new(label));
                    } else {
                        builder = builder.add_edge(edge.source, edge.target);
                    }
                }
    Ok(builder.build())
}

/// Parses a string value into a PropertyValue.
///
/// Attempts to parse as:
/// 1. Integer (i64)
/// 2. Float (f64)
/// 3. Boolean (true/false)
/// 4. String (fallback)
fn parse_property_value(s: &str) -> PropertyValue {
    // Try integer
    if let Ok(i) = s.parse::<i64>() {
        return PropertyValue::Int(i);
    }

    // Try float
    if let Ok(f) = s.parse::<f64>() {
        return PropertyValue::Float(f);
    }

    // Try boolean
    match s.to_lowercase().as_str() {
        "true" => return PropertyValue::Bool(true),
        "false" => return PropertyValue::Bool(false),
        _ => {}
    }

    // Default to string
    PropertyValue::String(s.to_string())
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use neural_core::Graph;
    use std::io::Write;
    use tempfile::NamedTempFile;

    fn create_temp_csv(content: &str) -> NamedTempFile {
        let mut file = NamedTempFile::new().unwrap();
        file.write_all(content.as_bytes()).unwrap();
        file
    }

    #[test]
    fn test_load_nodes_with_label() {
        let csv = "id,label,name,age\n0,Person,Alice,30\n1,Person,Bob,25\n";
        let file = create_temp_csv(csv);

        let nodes = load_nodes_csv(file.path()).unwrap();

        assert_eq!(nodes.len(), 2);
        assert_eq!(nodes[0].id, 0);
        assert_eq!(nodes[0].label, Some("Person".to_string()));
        assert_eq!(
            nodes[0].properties.get("name"),
            Some(&PropertyValue::String("Alice".to_string()))
        );
        assert_eq!(
            nodes[0].properties.get("age"),
            Some(&PropertyValue::Int(30))
        );
    }

    #[test]
    fn test_load_nodes_without_label() {
        let csv = "id,name,score\n0,Alice,95.5\n1,Bob,87.0\n";
        let file = create_temp_csv(csv);

        let nodes = load_nodes_csv(file.path()).unwrap();

        assert_eq!(nodes.len(), 2);
        assert_eq!(nodes[0].label, None);
        assert_eq!(
            nodes[0].properties.get("score"),
            Some(&PropertyValue::Float(95.5))
        );
    }

    #[test]
    fn test_load_edges() {
        let csv = "source,target,label\n0,1,KNOWS\n1,2,WORKS_AT\n";
        let file = create_temp_csv(csv);

        let edges = load_edges_csv(file.path()).unwrap();

        assert_eq!(edges.len(), 2);
        assert_eq!(edges[0].source, 0);
        assert_eq!(edges[0].target, 1);
        assert_eq!(edges[0].label, Some("KNOWS".to_string()));
    }

    #[test]
    fn test_load_edges_without_label() {
        let csv = "source,target\n0,1\n1,2\n";
        let file = create_temp_csv(csv);

        let edges = load_edges_csv(file.path()).unwrap();

        assert_eq!(edges.len(), 2);
        assert_eq!(edges[0].label, None);
    }

    #[test]
    fn test_load_graph_from_csv() {
        let nodes_csv = "id,label,name\n0,Person,Alice\n1,Person,Bob\n";
        let edges_csv = "source,target,label\n0,1,KNOWS\n";

        let nodes_file = create_temp_csv(nodes_csv);
        let edges_file = create_temp_csv(edges_csv);

        let store = load_graph_from_csv(Some(nodes_file.path()), edges_file.path()).unwrap();

        assert_eq!(store.node_count(), 2);
        assert_eq!(store.edge_count(), 1);
    }

    #[test]
    fn test_parse_property_value() {
        assert_eq!(parse_property_value("42"), PropertyValue::Int(42));
        assert_eq!(parse_property_value("2.5"), PropertyValue::Float(2.5));
        assert_eq!(parse_property_value("true"), PropertyValue::Bool(true));
        assert_eq!(parse_property_value("false"), PropertyValue::Bool(false));
        assert_eq!(
            parse_property_value("hello"),
            PropertyValue::String("hello".to_string())
        );
    }
}
