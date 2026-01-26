//! Execution plan types.
//!
//! Defines both logical and physical plans for query execution.

use neural_parser::{Direction, Expression};
use serde::{Deserialize, Serialize};

// =============================================================================
// Physical Plan
// =============================================================================

/// A physical execution plan.
///
/// Physical plans are directly executable against the graph storage.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum PhysicalPlan {
    /// Scan all nodes in the graph (O(n))
    ScanAllNodes {
        /// Variable binding for the scanned nodes
        binding: String,
    },

    /// Scan only nodes with a specific label (O(1) using LabelIndex)
    ScanByLabel {
        /// Variable binding for the scanned nodes
        binding: String,
        /// Label to filter by
        label: String,
    },

    /// Scan nodes with a specific property value (O(1) using PropertyIndex)
    ScanByProperty {
        /// Variable binding for the scanned nodes
        binding: String,
        /// Property key
        property: String,
        /// Property value to match
        value: Expression,
    },

    /// Direct node lookup by ID (O(1))
    ScanNodeById {
        /// Variable binding for the node
        binding: String,
        /// Node ID to fetch
        node_id: u64,
    },

    /// Expand from current nodes to their neighbors (all types)
    ExpandNeighbors {
        /// Source variable binding
        from_binding: String,
        /// Target variable binding
        to_binding: String,
        /// Optional edge variable binding
        edge_binding: Option<String>,
        /// Optional relationship label filter
        rel_label: Option<String>,
        /// Direction of traversal
        direction: Direction,
    },

    /// Expand from current nodes to their neighbors (all types), returning NULL if no match
    OptionalExpandNeighbors {
        /// Source variable binding
        from_binding: String,
        /// Target variable binding
        to_binding: String,
        /// Optional edge variable binding
        edge_binding: Option<String>,
        /// Optional relationship label filter
        rel_label: Option<String>,
        /// Direction of traversal
        direction: Direction,
    },

    /// Expand using edge type index (O(1) using EdgeTypeIndex)
    ExpandByType {
        /// Source variable binding
        from_binding: String,
        /// Target variable binding
        to_binding: String,
        /// Optional edge variable binding
        edge_binding: Option<String>,
        /// Required edge type
        edge_type: String,
        /// Direction of traversal
        direction: Direction,
    },

    /// Expand using edge type index, returning NULL if no match
    OptionalExpandByType {
        /// Source variable binding
        from_binding: String,
        /// Target variable binding
        to_binding: String,
        /// Optional edge variable binding
        edge_binding: Option<String>,
        /// Required edge type
        edge_type: String,
        /// Direction of traversal
        direction: Direction,
    },

    /// Upsert nodes/edges
    Merge {
        /// The pattern to merge
        pattern: neural_parser::Pattern,
    },

    /// Expand using variable length path (BFS)
    ExpandVariableLength {
        /// Source variable binding
        from_binding: String,
        /// Target variable binding
        to_binding: String,
        /// Optional path variable binding (p =)
        path_binding: Option<String>,
        /// Optional relationship label filter
        edge_type: Option<String>,
        /// Direction of traversal
        direction: Direction,
        /// Minimum hops
        min_hops: u64,
        /// Maximum hops (None = Infinity)
        max_hops: Option<u64>,
    },

    /// Find shortest path (BFS)
    ExpandShortestPath {
        /// Source variable binding
        from_binding: String,
        /// Target variable binding
        to_binding: String,
        /// Optional path variable binding (p =)
        path_binding: Option<String>,
        /// Optional relationship label filter
        edge_type: Option<String>,
        /// Direction of traversal
        direction: Direction,
        /// Maximum hops
        max_hops: u64,
    },

    /// Filter rows based on a predicate
    Filter {
        /// The predicate expression
        predicate: Expression,
    },

    /// Project specific columns for output
    Project {
        /// Columns to project
        columns: Vec<ProjectColumn>,
    },

    /// Sequential execution of multiple plans
    Sequence {
        /// Plans to execute in order
        steps: Vec<PhysicalPlan>,
    },

    // =========================================================================
    // Sprint 14: ORDER BY / LIMIT / Vector Search
    // =========================================================================
    /// Order results by expressions
    OrderBy {
        /// Items to order by: (expression, is_descending)
        items: Vec<(Expression, bool)>,
    },

    /// Limit the number of results
    Limit {
        /// Maximum number of results
        count: u64,
    },

    /// Remove duplicate rows (DISTINCT)
    Distinct,

    /// Group by and Aggregate
    Aggregate {
        /// Expressions to group by (if empty, aggregate all)
        group_by: Vec<Expression>,
        /// Aggregations to compute: (function, argument, distinct, alias)
        aggregations: Vec<(neural_parser::AggregateFunction, Option<Expression>, bool, String)>,
    },

    /// Unwind a list into rows
    Unwind {
        /// The list expression
        expression: Expression,
        /// The alias for the unwound items
        alias: String,
    },

    /// Optimized vector similarity search using HNSW index (Sprint 14)
    ///
    /// This plan is generated when the planner detects:
    /// `ORDER BY vector_similarity(...) DESC LIMIT k`
    VectorSearch {
        /// Variable binding for resulting nodes
        binding: String,
        /// The query vector expression
        query: Expression,
        /// Optional label filter (combines with HNSW search)
        label: Option<String>,
        /// Number of results to return
        k: usize,
    },

    // =========================================================================
    // Sprint 53: COUNT Optimization (O(1) instead of O(n))
    // =========================================================================

    /// Optimized node count - O(1) using index
    /// Generated for: MATCH (n) RETURN count(n) or MATCH (n:Label) RETURN count(n)
    CountNodes {
        /// Optional label filter
        label: Option<String>,
        /// Alias for the count result column
        alias: String,
    },

    /// Optimized edge count - O(1) using index
    /// Generated for: MATCH ()-[r]->() RETURN count(r) or MATCH ()-[r:TYPE]->() RETURN count(r)
    CountEdges {
        /// Optional edge type filter
        edge_type: Option<String>,
        /// Alias for the count result column
        alias: String,
    },
}

/// A column in the projection.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ProjectColumn {
    /// The expression to project
    pub expression: ProjectExpression,
    /// Optional alias for the column
    pub alias: Option<String>,
}

/// Expression types for projection.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum ProjectExpression {
    /// A variable binding (returns the node ID)
    Variable(String),
    /// A property access (variable.property)
    Property { variable: String, property: String },
    /// A complex expression
    Expression(neural_parser::Expression),
    /// An aggregate function
    Aggregate {
        function: neural_parser::AggregateFunction,
        argument: Option<Box<neural_parser::Expression>>,
        distinct: bool,
    },
}

impl ProjectColumn {
    /// Creates a column projecting a variable.
    pub fn variable(name: impl Into<String>) -> Self {
        Self {
            expression: ProjectExpression::Variable(name.into()),
            alias: None,
        }
    }

    /// Creates a column projecting a property.
    pub fn property(variable: impl Into<String>, property: impl Into<String>) -> Self {
        Self {
            expression: ProjectExpression::Property {
                variable: variable.into(),
                property: property.into(),
            },
            alias: None,
        }
    }

    /// Creates a column projecting an aggregate.
    pub fn aggregate(
        function: neural_parser::AggregateFunction,
        argument: Option<neural_parser::Expression>,
        distinct: bool,
    ) -> Self {
        Self {
            expression: ProjectExpression::Aggregate {
                function,
                argument: argument.map(Box::new),
                distinct,
            },
            alias: None,
        }
    }

    /// Returns the output name for this column.
    pub fn output_name(&self) -> String {
        if let Some(ref alias) = self.alias {
            alias.clone()
        } else {
            match &self.expression {
                ProjectExpression::Variable(name) => name.clone(),
                ProjectExpression::Property { variable, property } => {
                    format!("{}.{}", variable, property)
                }
                ProjectExpression::Expression(expr) => {
                    format!("{}", expr)
                }
                ProjectExpression::Aggregate {
                    function,
                    argument,
                    distinct,
                } => {
                    let arg_str = match argument {
                        Some(expr) => format!("{}", expr),
                        None => "*".to_string(),
                    };
                    if *distinct {
                        format!("{}(DISTINCT {})", function, arg_str)
                    } else {
                        format!("{}({})", function, arg_str)
                    }
                }
            }
        }
    }

    /// Returns true if this projection contains an aggregate.
    pub fn is_aggregate(&self) -> bool {
        matches!(self.expression, ProjectExpression::Aggregate { .. })
    }
}

// =============================================================================
// Logical Plan (for future optimization)
// =============================================================================

/// A logical execution plan.
///
/// Logical plans represent the query intent before optimization.
/// Currently not used directly, but prepared for future query optimizer.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum LogicalPlan {
    /// Scan nodes with optional label filter
    Scan {
        binding: String,
        label: Option<String>,
    },

    /// Expand relationships
    Expand {
        from: String,
        to: String,
        rel_label: Option<String>,
        direction: Direction,
    },

    /// Filter with predicate
    Filter { predicate: Expression },

    /// Project columns
    Project { columns: Vec<ProjectColumn> },

    /// Join two plans
    Join {
        left: Box<LogicalPlan>,
        right: Box<LogicalPlan>,
    },

    /// Sequential operations
    Sequence { steps: Vec<LogicalPlan> },
}

// =============================================================================
// Display
// =============================================================================

impl std::fmt::Display for PhysicalPlan {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            PhysicalPlan::ScanAllNodes { binding } => {
                write!(f, "ScanAllNodes({})", binding)
            }
            PhysicalPlan::ScanByLabel { binding, label } => {
                write!(f, "ScanByLabel({}, :{})", binding, label)
            }
            PhysicalPlan::ScanByProperty {
                binding,
                property,
                value,
            } => {
                write!(f, "ScanByProperty({}, {} = {})", binding, property, value)
            }
            PhysicalPlan::ScanNodeById { binding, node_id } => {
                write!(f, "ScanNodeById({}, id={})", binding, node_id)
            }
            PhysicalPlan::ExpandNeighbors {
                from_binding,
                to_binding,
                edge_binding,
                rel_label,
                direction,
            } => {
                let rel = rel_label.as_deref().unwrap_or("*");
                let dir = match direction {
                    Direction::Outgoing => "->",
                    Direction::Incoming => "<-",
                    Direction::Both => "--",
                };
                let edge = edge_binding.as_deref().unwrap_or("");
                write!(
                    f,
                    "Expand({} -[{}:{}]{}{})",
                    from_binding, edge, rel, dir, to_binding
                )
            }
            PhysicalPlan::OptionalExpandNeighbors {
                from_binding,
                to_binding,
                edge_binding,
                rel_label,
                direction,
            } => {
                let rel = rel_label.as_deref().unwrap_or("*");
                let dir = match direction {
                    Direction::Outgoing => "->",
                    Direction::Incoming => "<-",
                    Direction::Both => "--",
                };
                let edge = edge_binding.as_deref().unwrap_or("");
                write!(
                    f,
                    "OptionalExpand({} -[{}:{}]{}{})",
                    from_binding, edge, rel, dir, to_binding
                )
            }
            PhysicalPlan::ExpandByType {
                from_binding,
                to_binding,
                edge_binding,
                edge_type,
                direction,
            } => {
                let dir = match direction {
                    Direction::Outgoing => "->",
                    Direction::Incoming => "<-",
                    Direction::Both => "--",
                };
                let edge = edge_binding.as_deref().unwrap_or("");
                write!(
                    f,
                    "ExpandByType({} -[{}:{}]{}{})",
                    from_binding, edge, edge_type, dir, to_binding
                )
            }
            PhysicalPlan::OptionalExpandByType {
                from_binding,
                to_binding,
                edge_binding,
                edge_type,
                direction,
            } => {
                let dir = match direction {
                    Direction::Outgoing => "->",
                    Direction::Incoming => "<-",
                    Direction::Both => "--",
                };
                let edge = edge_binding.as_deref().unwrap_or("");
                write!(
                    f,
                    "OptionalExpandByType({} -[{}:{}]{}{})",
                    from_binding, edge, edge_type, dir, to_binding
                )
            }
            PhysicalPlan::Merge { pattern } => {
                write!(f, "Merge({})", pattern)
            }
            PhysicalPlan::ExpandVariableLength {
                from_binding,
                to_binding,
                path_binding,
                edge_type,
                direction,
                min_hops,
                max_hops,
            } => {
                let rel = edge_type.as_deref().unwrap_or("");
                let dir = match direction {
                    Direction::Outgoing => "->",
                    Direction::Incoming => "<-",
                    Direction::Both => "--",
                };
                let max_str = match max_hops {
                    Some(m) => format!("{}", m),
                    None => "".to_string(),
                };
                let path_str = path_binding.as_ref().map(|p| format!("{} = ", p)).unwrap_or_default();
                write!(
                    f,
                    "{}ExpandVariableLength({} -[:{}*{}..{}]{}{})",
                    path_str, from_binding, rel, min_hops, max_str, dir, to_binding
                )
            }
            PhysicalPlan::ExpandShortestPath {
                from_binding,
                to_binding,
                path_binding,
                edge_type,
                direction,
                max_hops,
            } => {
                let rel = edge_type.as_deref().unwrap_or("*");
                let dir = match direction {
                    Direction::Outgoing => "->",
                    Direction::Incoming => "<-",
                    Direction::Both => "--",
                };
                let path_str = path_binding.as_ref().map(|p| format!("{} = ", p)).unwrap_or_default();
                write!(
                    f,
                    "{}ShortestPath({} -[:{}*..{}]{}{})",
                    path_str, from_binding, rel, max_hops, dir, to_binding
                )
            }
            PhysicalPlan::Filter { predicate } => {
                write!(f, "Filter({})", predicate)
            }
            PhysicalPlan::Project { columns } => {
                let cols: Vec<_> = columns.iter().map(|c| c.output_name()).collect();
                write!(f, "Project({})", cols.join(", "))
            }
            PhysicalPlan::Sequence { steps } => {
                writeln!(f, "Sequence [")?;
                for (i, step) in steps.iter().enumerate() {
                    writeln!(f, "  {}: {}", i, step)?;
                }
                write!(f, "]")
            }
            PhysicalPlan::OrderBy { items } => {
                let items_str: Vec<_> = items
                    .iter()
                    .map(|(expr, desc)| {
                        if *desc {
                            format!("{} DESC", expr)
                        } else {
                            format!("{} ASC", expr)
                        }
                    })
                    .collect();
                write!(f, "OrderBy({})", items_str.join(", "))
            }
            PhysicalPlan::Limit { count } => {
                write!(f, "Limit({})", count)
            }
            PhysicalPlan::Distinct => {
                write!(f, "Distinct")
            }
            PhysicalPlan::Aggregate { group_by, aggregations } => {
                write!(f, "Aggregate(GroupBy: {:?}, Aggs: {:?})", group_by, aggregations)
            }
            PhysicalPlan::Unwind { expression, alias } => {
                write!(f, "Unwind({}, AS {})", expression, alias)
            }
            PhysicalPlan::VectorSearch { binding, query, label, k } => {
                if let Some(lbl) = label {
                    write!(f, "VectorSearch({}: {}, query={}, k={})", binding, lbl, query, k)
                } else {
                    write!(f, "VectorSearch({}, query={}, k={})", binding, query, k)
                }
            }
            PhysicalPlan::CountNodes { label, alias } => {
                if let Some(lbl) = label {
                    write!(f, "CountNodes(:{}, AS {})", lbl, alias)
                } else {
                    write!(f, "CountNodes(*, AS {})", alias)
                }
            }
            PhysicalPlan::CountEdges { edge_type, alias } => {
                if let Some(et) = edge_type {
                    write!(f, "CountEdges(:{}, AS {})", et, alias)
                } else {
                    write!(f, "CountEdges(*, AS {})", alias)
                }
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
    fn test_project_column_output_name() {
        let col = ProjectColumn::variable("n");
        assert_eq!(col.output_name(), "n");

        let col = ProjectColumn::property("n", "name");
        assert_eq!(col.output_name(), "n.name");

        let col = ProjectColumn {
            expression: ProjectExpression::Variable("n".into()),
            alias: Some("node".into()),
        };
        assert_eq!(col.output_name(), "node");
    }

    #[test]
    fn test_plan_display() {
        let plan = PhysicalPlan::ScanAllNodes {
            binding: "n".into(),
        };
        assert_eq!(format!("{}", plan), "ScanAllNodes(n)");

        let plan = PhysicalPlan::ExpandNeighbors {
            from_binding: "a".into(),
            to_binding: "b".into(),
            edge_binding: None,
            rel_label: Some("KNOWS".into()),
            direction: Direction::Outgoing,
        };
        assert_eq!(format!("{}", plan), "Expand(a -[:KNOWS]->b)");
    }
}
