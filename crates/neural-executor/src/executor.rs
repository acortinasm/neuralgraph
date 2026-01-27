//! Query executor.
//!
//! Executes physical plans against the graph storage.
//!
//! Supports MVCC snapshot isolation via `snapshot_id`. When executing
//! within a transaction, use `with_snapshot()` to read from a consistent
//! point in time.

use crate::eval;
use crate::plan::{PhysicalPlan, ProjectColumn, ProjectExpression};
use crate::result::{Bindings, QueryResult, Row, Value};
use crate::{ExecutionError, Result};
use neural_core::{Graph, NodeId};
use neural_parser::Direction;
use neural_storage::{GraphStore, graph_store::MAX_SNAPSHOT_ID};
use neural_storage::wal::TransactionId;

/// A stream of query results (bindings).
pub type RowStream<'a> = Box<dyn Iterator<Item = Result<Bindings>> + Send + 'a>;

/// Query executor that runs physical plans against the graph.
///
/// Supports MVCC snapshot isolation. By default, the executor sees all
/// committed data (MAX_SNAPSHOT_ID). For transactional reads, use
/// `with_snapshot()` to read from a specific point in time.
pub struct Executor<'a> {
    /// The graph store to execute against
    store: &'a GraphStore,
    /// Snapshot ID for MVCC reads (default: MAX_SNAPSHOT_ID sees all data)
    snapshot_id: TransactionId,
}

impl<'a> Executor<'a> {
    /// Creates a new executor for the given graph store.
    ///
    /// Uses MAX_SNAPSHOT_ID so all committed data is visible.
    pub fn new(store: &'a GraphStore) -> Self {
        Self {
            store,
            snapshot_id: MAX_SNAPSHOT_ID,
        }
    }

    /// Creates a new executor with a specific snapshot ID for MVCC reads.
    ///
    /// The executor will only see data committed before `snapshot_id`.
    pub fn with_snapshot(store: &'a GraphStore, snapshot_id: TransactionId) -> Self {
        Self { store, snapshot_id }
    }

    /// Returns the current snapshot ID.
    pub fn snapshot_id(&self) -> TransactionId {
        self.snapshot_id
    }

    /// Executes a physical plan and returns the query result.
    pub fn execute(
        &self,
        plan: &PhysicalPlan,
        params: Option<&eval::Parameters>,
    ) -> Result<QueryResult> {
        let initial_bindings: RowStream = Box::new(std::iter::once(Ok(Bindings::new())));
        let (final_stream, columns) = self.execute_plan_internal(plan, initial_bindings, params)?;
        let final_bindings: Vec<Bindings> = final_stream.collect::<Result<Vec<_>>>()?;
        self.bindings_to_result(&final_bindings, &columns, params)
    }

    /// Executes a plan step.
    pub fn execute_plan_internal(
        &self,
        plan: &PhysicalPlan,
        input: RowStream<'a>,
        params: Option<&eval::Parameters>,
    ) -> Result<(RowStream<'a>, Vec<ProjectColumn>)> {
        match plan {
            PhysicalPlan::ScanAllNodes { binding } => {
                let result = self.execute_scan_all(input, binding)?;
                Ok((result, Vec::new()))
            }
            PhysicalPlan::ScanByLabel { binding, label } => {
                let result = self.execute_scan_by_label(input, binding, label)?;
                Ok((result, Vec::new()))
            }
            PhysicalPlan::ScanByProperty { binding, property, value } => {
                let result = self.execute_scan_by_property(input, binding, property, value, params)?;
                Ok((result, Vec::new()))
            }
            PhysicalPlan::ScanNodeById { binding, node_id } => {
                let result = self.execute_scan_node_by_id(input, binding, *node_id)?;
                Ok((result, Vec::new()))
            }
            PhysicalPlan::ExpandNeighbors {
                from_binding,
                to_binding,
                edge_binding,
                rel_label,
                direction,
            } => {
                let result = if let Some(label) = rel_label {
                    self.execute_expand_by_type(input, from_binding, to_binding, edge_binding.as_deref(), label, direction)?
                } else {
                    self.execute_expand_neighbors(input, from_binding, to_binding, edge_binding.as_deref(), direction)?
                };
                Ok((result, Vec::new()))
            }
            PhysicalPlan::OptionalExpandNeighbors {
                from_binding,
                to_binding,
                edge_binding,
                rel_label,
                direction,
            } => {
                let result = if let Some(label) = rel_label {
                    self.execute_optional_expand_by_type(input, from_binding, to_binding, edge_binding.as_deref(), label, direction)?
                } else {
                    self.execute_optional_expand_neighbors(input, from_binding, to_binding, edge_binding.as_deref(), direction)?
                };
                Ok((result, Vec::new()))
            }
            PhysicalPlan::ExpandByType {
                from_binding,
                to_binding,
                edge_binding,
                edge_type,
                direction,
            } => {
                let result = self.execute_expand_by_type(input, from_binding, to_binding, edge_binding.as_deref(), edge_type, direction)?;
                Ok((result, Vec::new()))
            }
            PhysicalPlan::OptionalExpandByType {
                from_binding,
                to_binding,
                edge_binding,
                edge_type,
                direction,
            } => {
                let result = self.execute_optional_expand_by_type(input, from_binding, to_binding, edge_binding.as_deref(), edge_type, direction)?;
                Ok((result, Vec::new()))
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
                let result = self.execute_expand_variable_length(
                    input,
                    from_binding,
                    to_binding,
                    path_binding.as_deref(),
                    edge_type.as_deref(),
                    direction,
                    *min_hops,
                    *max_hops,
                )?;
                Ok((result, Vec::new()))
            }
            PhysicalPlan::ExpandShortestPath {
                from_binding,
                to_binding,
                path_binding,
                edge_type,
                direction,
                max_hops,
            } => {
                let result = self.execute_expand_shortest_path(
                    input,
                    from_binding,
                    to_binding,
                    path_binding.as_deref(),
                    edge_type.as_deref(),
                    direction,
                    *max_hops,
                )?;
                Ok((result, Vec::new()))
            }
            PhysicalPlan::Filter { predicate } => {
                let result = self.execute_filter(input, predicate, params)?;
                Ok((result, Vec::new()))
            }
            PhysicalPlan::Project { columns } => {
                let result = self.execute_project(input, columns, params)?;
                Ok((result, columns.clone()))
            }
            PhysicalPlan::Sequence { steps } => {
                let mut current_stream = input;
                let mut final_columns = Vec::new();
                for step in steps {
                    let (new_stream, columns) = self.execute_plan_internal(step, current_stream, params)?;
                    current_stream = new_stream;
                    if !columns.is_empty() {
                        final_columns = columns;
                    }
                }
                Ok((current_stream, final_columns))
            }
            PhysicalPlan::OrderBy { items } => {
                let result = self.execute_order_by(input, items, params)?;
                Ok((result, Vec::new()))
            }
            PhysicalPlan::Limit { count } => {
                let result = self.execute_limit(input, *count)?;
                Ok((result, Vec::new()))
            }
            PhysicalPlan::Unwind { expression, alias } => {
                let result = self.execute_unwind(input, expression, alias, params)?;
                Ok((result, Vec::new()))
            }
            PhysicalPlan::Distinct => {
                let result = self.execute_distinct(input)?;
                Ok((result, Vec::new()))
            }
            PhysicalPlan::Aggregate { group_by, aggregations } => {
                let result = self.execute_aggregate(input, group_by, aggregations, params)?;
                Ok((result, Vec::new()))
            }
            PhysicalPlan::Merge { pattern: _ } => {
                // MERGE is complex because it might mutate.
                // For now, nGraph is mostly read-only in execution pipeline.
                // Mutation steps like CREATE/DELETE are handled in lib.rs separately.
                // We will handle MERGE as a special case later.
                Err(ExecutionError::ExecutionError("MERGE not yet supported in physical plan".into()))
            }
            PhysicalPlan::VectorSearch { binding, query, label, k } => {
                let result = self.execute_vector_search(binding, query, label.as_deref(), *k, params)?;
                Ok((result, Vec::new()))
            }
            PhysicalPlan::CountNodes { label, alias } => {
                let result = self.execute_count_nodes(label.as_deref(), alias)?;
                Ok((result, Vec::new()))
            }
            PhysicalPlan::CountEdges { edge_type, alias } => {
                let result = self.execute_count_edges(edge_type.as_deref(), alias)?;
                Ok((result, Vec::new()))
            }
        }
    }

    fn execute_scan_all(&self, input: RowStream<'a>, binding: &str) -> Result<RowStream<'a>> {
        let binding = binding.to_string();
        let store = self.store;
        let stream = input.flat_map(move |res| -> RowStream<'a> {
            match res {
                Ok(existing) => {
                    // OPTIMIZATION: If already bound, we don't scan, we filter?
                    // Actually Cypher: MATCH (a) MATCH (a) ... the second (a) must match previous (a).
                    if existing.get(&binding).is_some() {
                         // Variable already bound. We return the row as is if it matches?
                         // For ScanAllNodes, any bound node "matches" because it exists.
                         // But we should probably check if it actually exists in the graph.
                         return Box::new(std::iter::once(Ok(existing)));
                    }
                    
                    let binding = binding.clone();
                    let iter = (0..store.node_count()).map(move |node_id| {
                        let node = NodeId::new(node_id as u64);
                        Ok(existing.with(&binding, node))
                    });
                    Box::new(iter)
                }
                Err(e) => Box::new(std::iter::once(Err(e))),
            }
        });
        Ok(Box::new(stream))
    }

    fn execute_scan_by_label(&self, input: RowStream<'a>, binding: &str, label: &str) -> Result<RowStream<'a>> {
        let binding = binding.to_string();
        let label = label.to_string();
        let store = self.store;
        let snapshot_id = self.snapshot_id;
        let stream = input.flat_map(move |res| -> RowStream<'a> {
            match res {
                Ok(existing) => {
                    if let Some(val) = existing.get(&binding) {
                         if let Some(node_id) = val.as_node().map(NodeId::new) {
                              // Use snapshot-aware label read for MVCC
                              if store.get_label_at(node_id, snapshot_id) == Some(&label) {
                                  return Box::new(std::iter::once(Ok(existing)));
                              } else {
                                  return Box::new(std::iter::empty());
                              }
                         }
                    }

                    let binding = binding.clone();
                    let nodes: Vec<_> = store.nodes_with_label(&label).collect();
                    let iter = nodes.into_iter().map(move |node| {
                        Ok(existing.with(&binding, node))
                    });
                    Box::new(iter)
                }
                Err(e) => Box::new(std::iter::once(Err(e))),
            }
        });
        Ok(Box::new(stream))
    }

    fn execute_scan_by_property(&self, input: RowStream<'a>, binding: &str, property: &str, value: &neural_parser::Expression, params: Option<&eval::Parameters>) -> Result<RowStream<'a>> {
        use neural_core::PropertyValue;
        let empty_binding = Bindings::new();
        let evaluated_value = eval::evaluate(value, &empty_binding, self.store, params)?;
        let prop_value = match evaluated_value {
             Value::String(s) => PropertyValue::from(s),
             Value::Int(i) => PropertyValue::from(i),
             Value::Float(f) => PropertyValue::from(f),
             Value::Bool(b) => PropertyValue::from(b),
             Value::Date(s) => PropertyValue::Date(s),
             Value::DateTime(s) => PropertyValue::DateTime(s),
             Value::Null => PropertyValue::Null,
             Value::Node(_) | Value::Edge(_) => return Err(ExecutionError::ExecutionError("Cannot scan by ID".into())),
             Value::List(_) => return Err(ExecutionError::ExecutionError("Cannot scan by list".into())),
        };
        let binding = binding.to_string();
        let property = property.to_string();
        let store = self.store;
        let snapshot_id = self.snapshot_id;
        let stream = input.flat_map(move |res| -> RowStream<'a> {
            match res {
                Ok(existing) => {
                    if let Some(val) = existing.get(&binding) {
                         if let Some(node_id) = val.as_node().map(NodeId::new) {
                              // Use snapshot-aware property read for MVCC
                              if let Some(v) = store.get_property_at(node_id, &property, snapshot_id) {
                                  if v == &prop_value { return Box::new(std::iter::once(Ok(existing))); }
                              }
                              return Box::new(std::iter::empty());
                         }
                    }

                    let binding = binding.clone();
                    let nodes: Vec<_> = store.nodes_with_property(&property, &prop_value).collect();
                    let iter = nodes.into_iter().map(move |node| {
                        Ok(existing.with(&binding, node))
                    });
                    Box::new(iter)
                }
                Err(e) => Box::new(std::iter::once(Err(e))),
            }
        });
        Ok(Box::new(stream))
    }

    /// O(1) direct node lookup by ID
    fn execute_scan_node_by_id(&self, input: RowStream<'a>, binding: &str, node_id: u64) -> Result<RowStream<'a>> {
        let binding = binding.to_string();
        let store = self.store;
        let nid = NodeId::new(node_id);

        // Check if node exists
        if node_id as usize >= store.node_count() {
            return Ok(Box::new(std::iter::empty()));
        }

        let stream = input.flat_map(move |res| -> RowStream<'a> {
            match res {
                Ok(existing) => {
                    // If already bound, check if it matches
                    if let Some(val) = existing.get(&binding) {
                        if val.as_node() == Some(node_id) {
                            return Box::new(std::iter::once(Ok(existing)));
                        } else {
                            return Box::new(std::iter::empty());
                        }
                    }
                    // Bind the node directly - O(1)
                    Box::new(std::iter::once(Ok(existing.with(&binding, nid))))
                }
                Err(e) => Box::new(std::iter::once(Err(e))),
            }
        });
        Ok(Box::new(stream))
    }

    fn execute_expand_neighbors(&self, input: RowStream<'a>, from_binding: &str, to_binding: &str, edge_binding: Option<&str>, direction: &Direction) -> Result<RowStream<'a>> {
        self.execute_expand_neighbors_internal(input, from_binding, to_binding, edge_binding, direction, false)
    }

        fn execute_optional_expand_neighbors(&self, input: RowStream<'a>, from_binding: &str, to_binding: &str, edge_binding: Option<&str>, direction: &Direction) -> Result<RowStream<'a>> {

            self.execute_expand_neighbors_internal(input, from_binding, to_binding, edge_binding, direction, true)

        }

    

        fn execute_expand_neighbors_internal(&self, input: RowStream<'a>, from_binding: &str, to_binding: &str, edge_binding: Option<&str>, direction: &Direction, optional: bool) -> Result<RowStream<'a>> {
            let from_binding = from_binding.to_string();
            let to_binding = to_binding.to_string();
            let edge_binding = edge_binding.map(|s| s.to_string());
            let direction = *direction;
            let store = self.store;

            let stream = input.flat_map(move |res| -> RowStream<'a> {
                match res {
                    Ok(existing) => {
                        if let Some(val) = existing.get(&from_binding) {
                            if let Some(node_id) = val.as_node().map(NodeId::new) {
                                // Use degree check for O(1) emptiness test, then collect
                                // This avoids allocation when degree is 0
                                let degree = match direction {
                                    Direction::Outgoing | Direction::Both => store.out_degree(node_id),
                                    Direction::Incoming => store.in_degree(node_id),
                                };

                                if degree == 0 {
                                    if optional {
                                        let mut bound = existing.with(&to_binding, Value::Null);
                                        if let Some(ref eb) = edge_binding {
                                            bound = bound.with(eb, Value::Null);
                                        }
                                        return Box::new(std::iter::once(Ok(bound)));
                                    } else {
                                        return Box::new(std::iter::empty());
                                    }
                                }

                                // Only collect when we know there are neighbors
                                let neighbors: Vec<_> = match direction {
                                    Direction::Outgoing | Direction::Both => {
                                        store.neighbors_with_ids(node_id).collect()
                                    }
                                    Direction::Incoming => {
                                        store.incoming_neighbors_with_ids(node_id).collect()
                                    }
                                };

                                let to_binding = to_binding.clone();
                                let edge_binding = edge_binding.clone();
                                let iter = neighbors.into_iter().map(move |(edge_id, neighbor)| {
                                    let mut bound = existing.with(&to_binding, neighbor);
                                    if let Some(ref eb) = edge_binding {
                                        bound = bound.with(eb, Value::from_edge(edge_id));
                                    }
                                    Ok(bound)
                                });
                                Box::new(iter)
                            } else {
                                Box::new(std::iter::once(Err(ExecutionError::ExecutionError(
                                    format!("Variable '{}' is not a node", from_binding)
                                ))))
                            }
                        } else {
                            Box::new(std::iter::once(Err(ExecutionError::VariableNotFound(from_binding.clone()))))
                        }
                    }
                    Err(e) => Box::new(std::iter::once(Err(e))),
                }
            });

            Ok(Box::new(stream))
        }

    

    fn execute_expand_by_type(&self, input: RowStream<'a>, from_binding: &str, to_binding: &str, edge_binding: Option<&str>, edge_type: &str, direction: &Direction) -> Result<RowStream<'a>> {
        self.execute_expand_by_type_internal(input, from_binding, to_binding, edge_binding, edge_type, direction, false)
    }

    fn execute_optional_expand_by_type(&self, input: RowStream<'a>, from_binding: &str, to_binding: &str, edge_binding: Option<&str>, edge_type: &str, direction: &Direction) -> Result<RowStream<'a>> {
        self.execute_expand_by_type_internal(input, from_binding, to_binding, edge_binding, edge_type, direction, true)
    }

    fn execute_expand_by_type_internal(&self, input: RowStream<'a>, from_binding: &str, to_binding: &str, edge_binding: Option<&str>, edge_type: &str, direction: &Direction, optional: bool) -> Result<RowStream<'a>> {
        let from_binding = from_binding.to_string();
        let to_binding = to_binding.to_string();
        let edge_binding = edge_binding.map(|s| s.to_string());
        let edge_type = edge_type.to_string();
        let direction = *direction;
        let store = self.store;

        let stream = input.flat_map(move |res| -> RowStream<'a> {
            match res {
                Ok(existing) => {
                    if let Some(val) = existing.get(&from_binding) {
                        if let Some(node_id) = val.as_node().map(NodeId::new) {
                            // Collect neighbors for this type
                            let neighbors: Vec<(neural_core::EdgeId, NodeId)> = match direction {
                                Direction::Outgoing | Direction::Both => {
                                    store.neighbors_via_type_with_ids(node_id, &edge_type).collect()
                                }
                                Direction::Incoming => {
                                    store.edge_type_index().get(&edge_type).into_iter().flatten()
                                        .filter(|(_, _, target, _)| *target == node_id)
                                        .map(|(eid, source, _, _)| (*eid, *source))
                                        .collect()
                                }
                            };

                            if neighbors.is_empty() {
                                if optional {
                                    let mut bound = existing.with(&to_binding, Value::Null);
                                    if let Some(ref eb) = edge_binding {
                                        bound = bound.with(eb, Value::Null);
                                    }
                                    return Box::new(std::iter::once(Ok(bound)));
                                } else {
                                    return Box::new(std::iter::empty());
                                }
                            }

                            let to_binding = to_binding.clone();
                            let edge_binding = edge_binding.clone();
                            let iter = neighbors.into_iter().map(move |(edge_id, neighbor)| {
                                let mut bound = existing.with(&to_binding, neighbor);
                                if let Some(ref eb) = edge_binding {
                                    bound = bound.with(eb, Value::from_edge(edge_id));
                                }
                                Ok(bound)
                            });
                            Box::new(iter)
                        } else {
                            Box::new(std::iter::once(Err(ExecutionError::ExecutionError(
                                format!("Variable '{}' is not a node", from_binding)
                            ))))
                        }
                    } else {
                        Box::new(std::iter::once(Err(ExecutionError::VariableNotFound(from_binding.clone()))))
                    }
                }
                Err(e) => Box::new(std::iter::once(Err(e))),
            }
        });

        Ok(Box::new(stream))
    }

    fn execute_expand_variable_length(&self, input: RowStream<'a>, from_binding: &str, to_binding: &str, path_binding: Option<&str>, edge_type: Option<&str>, direction: &Direction, min_hops: u64, max_hops: Option<u64>) -> Result<RowStream<'a>> {
        let from_binding = from_binding.to_string();
        let to_binding = to_binding.to_string();
        let path_binding = path_binding.map(|s| s.to_string());
        let edge_type = edge_type.map(|s| s.to_string());
        let direction = *direction;
        let store = self.store;
        let max_len = max_hops.unwrap_or(50) as usize;
        let min_len = min_hops as usize;
        let stream = input.flat_map(move |res| -> RowStream<'a> {
             match res {
                Ok(existing) => {
                     if matches!(direction, Direction::Incoming) { return Box::new(std::iter::once(Err(ExecutionError::ExecutionError("Incoming variable length paths not supported".into())))); }
                     if let Some(val) = existing.get(&from_binding) {
                        if let Some(node_id) = val.as_node().map(NodeId::new) {
                            let paths = store.find_paths(node_id, None, min_len, max_len, edge_type.as_deref());
                            let to_binding = to_binding.clone();
                            let path_binding = path_binding.clone();
                            let iter = paths.into_iter().filter_map(move |path| {
                                path.last().map(|end_node| {
                                    let mut bound = existing.with(&to_binding, *end_node);
                                    if let Some(ref pb) = path_binding {
                                        let path_values: Vec<Value> = path.iter().map(|n| Value::from_node(*n)).collect();
                                        bound = bound.with(pb, Value::List(path_values));
                                    }
                                    Ok(bound)
                                })
                            });
                            Box::new(iter)
                        } else { Box::new(std::iter::once(Err(ExecutionError::ExecutionError(format!("Variable '{}' is not a node", from_binding))))) }
                     } else { Box::new(std::iter::once(Err(ExecutionError::VariableNotFound(from_binding.clone())))) }
                }
                Err(e) => Box::new(std::iter::once(Err(e))),
            }
        });
        Ok(Box::new(stream))
    }

    fn execute_expand_shortest_path(&self, input: RowStream<'a>, from_binding: &str, to_binding: &str, path_binding: Option<&str>, edge_type: Option<&str>, direction: &Direction, max_hops: u64) -> Result<RowStream<'a>> {
        let from_binding = from_binding.to_string();
        let to_binding = to_binding.to_string();
        let path_binding = path_binding.map(|s| s.to_string());
        let edge_type = edge_type.map(|s| s.to_string());
        let direction = *direction;
        let store = self.store;
        let max_len = max_hops as usize;
        let stream = input.flat_map(move |res| -> RowStream<'a> {
            match res {
                Ok(existing) => {
                     if matches!(direction, Direction::Incoming) { return Box::new(std::iter::once(Err(ExecutionError::ExecutionError("Incoming shortest paths not supported".into())))); }
                     if let Some(val) = existing.get(&from_binding) {
                        if let Some(node_id) = val.as_node().map(NodeId::new) {
                            let end_node_opt = existing.get(&to_binding).and_then(|v| v.as_node()).map(NodeId::new);
                            let paths = store.find_shortest_path(node_id, end_node_opt, max_len, edge_type.as_deref());
                            let to_binding = to_binding.clone();
                            let path_binding = path_binding.clone();
                            let iter = paths.into_iter().filter_map(move |path| {
                                path.last().map(|final_node| {
                                    let mut bound = existing.with(&to_binding, *final_node);
                                    if let Some(ref pb) = path_binding {
                                        let path_values: Vec<Value> = path.iter().map(|n| Value::from_node(*n)).collect();
                                        bound = bound.with(pb, Value::List(path_values));
                                    }
                                    Ok(bound)
                                })
                            });
                            Box::new(iter)
                        } else { Box::new(std::iter::once(Err(ExecutionError::ExecutionError(format!("Variable '{}' is not a node", from_binding))))) }
                    } else { Box::new(std::iter::once(Err(ExecutionError::VariableNotFound(from_binding.clone())))) }
                }
                Err(e) => Box::new(std::iter::once(Err(e))),
            }
        });
        Ok(Box::new(stream))
    }

    fn execute_filter(&self, input: RowStream<'a>, predicate: &neural_parser::Expression, params: Option<&eval::Parameters>) -> Result<RowStream<'a>> {
        let predicate = predicate.clone();
        let params = params.cloned();
        let store = self.store;
        let snapshot_id = self.snapshot_id;
        let stream = input.flat_map(move |res| -> RowStream<'a> {
             match res {
                Ok(binding) => {
                     // Use snapshot-aware filter evaluation for MVCC
                     match eval::is_true_at(&predicate, &binding, store, params.as_ref(), snapshot_id) {
                        Ok(true) => Box::new(std::iter::once(Ok(binding))),
                        Ok(false) => Box::new(std::iter::empty()),
                        Err(e) => Box::new(std::iter::once(Err(e))),
                    }
                }
                Err(e) => Box::new(std::iter::once(Err(e))),
            }
        });
        Ok(Box::new(stream))
    }
    
    fn execute_project(&self, input: RowStream<'a>, columns: &[ProjectColumn], params: Option<&eval::Parameters>) -> Result<RowStream<'a>> {
        let columns = columns.to_vec();
        let params = params.cloned();
        let store = self.store;
        let snapshot_id = self.snapshot_id;
        let stream = input.flat_map(move |res| -> RowStream<'a> {
            match res {
                Ok(existing) => {
                    let mut new_binding = Bindings::new();
                    for col in &columns {
                        let val_result = match &col.expression {
                            ProjectExpression::Variable(var) => {
                                if let Some(val) = existing.get(var) {
                                    Ok(val.clone())
                                } else {
                                    Err(ExecutionError::VariableNotFound(var.clone()))
                                }
                            }
                            ProjectExpression::Property { variable, property } => {
                                if let Some(val) = existing.get(variable) {
                                    if let Some(node_id) = val.as_node().map(NodeId::new) {
                                        // Use snapshot-aware property read for MVCC
                                        if let Some(prop_value) = store.get_property_at(node_id, property, snapshot_id) {
                                            Ok(eval::property_value_to_value(prop_value))
                                        } else if property == "id" {
                                            Ok(Value::Int(node_id.as_u64() as i64))
                                        } else { Ok(Value::Null) }
                                    } else { Err(ExecutionError::ExecutionError(format!("Variable '{}' is not a node", variable))) }
                                } else { Err(ExecutionError::VariableNotFound(variable.clone())) }
                            }
                            ProjectExpression::Expression(expr) => {
                                eval::evaluate(expr, &existing, store, params.as_ref())
                            }
                            ProjectExpression::Aggregate { .. } => Ok(Value::Null),
                        };
                        match val_result {
                            Ok(val) => { new_binding.bind(col.output_name(), val); }
                            Err(e) => return Box::new(std::iter::once(Err(e))),
                        }
                    }
                    Box::new(std::iter::once(Ok(new_binding)))
                }
                Err(e) => Box::new(std::iter::once(Err(e))),
            }
        });
        Ok(Box::new(stream))
    }
    
    fn execute_unwind(&self, input: RowStream<'a>, expression: &neural_parser::Expression, alias: &str, params: Option<&eval::Parameters>) -> Result<RowStream<'a>> {
        let expression = expression.clone();
        let alias = alias.to_string();
        let params = params.cloned();
        let store = self.store;
        let stream = input.flat_map(move |res| -> RowStream<'a> {
            match res {
                Ok(binding) => {
                    let val_result = eval::evaluate(&expression, &binding, store, params.as_ref());
                    match val_result {
                        Ok(Value::List(list)) => {
                            let alias = alias.clone();
                            let binding = binding.clone();
                            let iter = list.into_iter().map(move |item| Ok(binding.with(&alias, item)));
                            Box::new(iter)
                        }
                        Ok(Value::Null) => Box::new(std::iter::empty()),
                        Ok(other) => {
                             let alias = alias.clone();
                             Box::new(std::iter::once(Ok(binding.with(&alias, other))))
                        }
                        Err(e) => Box::new(std::iter::once(Err(e))),
                    }
                }
                Err(e) => Box::new(std::iter::once(Err(e))),
            }
        });
        Ok(Box::new(stream))
    }

    fn execute_vector_search(&self, binding: &str, query: &neural_parser::Expression, label: Option<&str>, k: usize, params: Option<&eval::Parameters>) -> Result<RowStream<'a>> {
        let binding = binding.to_string();
        let store = self.store;
        let k = k;
        let label = label.map(|s| s.to_string());
        
        // Evaluate the query vector (it should be a parameter or a literal list)
        let query_val = eval::evaluate(query, &Bindings::new(), store, params)?;
        
        let query_vec = match query_val {
            Value::List(l) => {
                let mut vec = Vec::new();
                for item in l {
                    match item {
                        Value::Int(i) => vec.push(i as f32),
                        Value::Float(f) => vec.push(f as f32),
                        _ => return Err(ExecutionError::ExecutionError("Vector elements must be numbers".into())),
                    }
                }
                vec
            },
            _ => return Err(ExecutionError::ExecutionError("Query vector must be a list".into())),
        };

        // Perform vector search
        let results = if let Some(lbl) = label {
            store.vector_search_filtered(&query_vec, k, &lbl)
        } else {
            store.vector_search(&query_vec, k)
        };

        let iter = results.into_iter().map(move |(node_id, _similarity)| {
            let mut b = Bindings::new();
            b.bind(&binding, Value::from_node(node_id));
            Ok(b)
        });
        
        Ok(Box::new(iter))
    }

    fn execute_distinct(&self, input: RowStream<'a>) -> Result<RowStream<'a>> {
        use std::collections::HashSet;
        let rows: Vec<Bindings> = input.collect::<Result<Vec<_>>>()?;
        let mut seen = HashSet::new();
        let mut distinct_rows = Vec::new();
        for row in rows {
            let mut items: Vec<_> = row.iter().collect();
            items.sort_by_key(|(k, _)| *k);
            let key = format!("{:?}", items); 
            if seen.insert(key) { distinct_rows.push(row); }
        }
        Ok(Box::new(distinct_rows.into_iter().map(Ok)))
    }

    fn execute_aggregate(&self, input: RowStream<'a>, group_by: &[neural_parser::Expression], aggregations: &[(neural_parser::AggregateFunction, Option<neural_parser::Expression>, bool, String)], params: Option<&eval::Parameters>) -> Result<RowStream<'a>> {
        use std::collections::HashMap;
        use crate::aggregate;
        let rows: Vec<Bindings> = input.collect::<Result<Vec<_>>>()?;
        let store = self.store;
        let params = params.cloned();
        let mut groups: HashMap<Vec<String>, Vec<Bindings>> = HashMap::new();
        for row in rows {
            let mut key = Vec::new();
            for expr in group_by {
                let val = eval::evaluate(expr, &row, store, params.as_ref())?;
                key.push(format!("{:?}", val));
            }
            groups.entry(key).or_default().push(row);
        }
        let mut result_rows = Vec::new();
        for (_, group_rows) in groups {
            let mut new_binding = Bindings::new();
            if let Some(first_row) = group_rows.first() {
                for expr in group_by {
                    let val = eval::evaluate(expr, first_row, store, params.as_ref())?;
                    new_binding.bind(format!("{}", expr), val);
                }
            }
            for (func, arg, distinct, alias) in aggregations {
                let val = aggregate::compute_aggregate(func, &arg.clone().map(Box::new), *distinct, &group_rows, store, params.as_ref())?;
                new_binding.bind(alias, val);
            }
            result_rows.push(new_binding);
        }
        Ok(Box::new(result_rows.into_iter().map(Ok)))
    }

    fn execute_order_by(&self, input: RowStream<'a>, items: &[(neural_parser::Expression, bool)], params: Option<&eval::Parameters>) -> Result<RowStream<'a>> {
        let mut bindings: Vec<Bindings> = input.collect::<Result<Vec<_>>>()?;
        let items = items.to_vec();
        let params = params.cloned();
        let store = self.store;
        bindings.sort_by(|a, b| {
            for (expr, is_desc) in &items {
                let val_a = eval::evaluate(expr, a, store, params.as_ref()).unwrap_or(Value::Null);
                let val_b = eval::evaluate(expr, b, store, params.as_ref()).unwrap_or(Value::Null);
                let cmp = crate::cmp::compare_for_ordering(&val_a, &val_b);
                if cmp != std::cmp::Ordering::Equal {
                    return if *is_desc { cmp.reverse() } else { cmp };
                }
            }
            std::cmp::Ordering::Equal
        });
        Ok(Box::new(bindings.into_iter().map(Ok)))
    }

    fn execute_limit(&self, input: RowStream<'a>, count: u64) -> Result<RowStream<'a>> {
        Ok(Box::new(input.take(count as usize)))
    }

    /// O(1) node count using index - Sprint 53 optimization
    fn execute_count_nodes(&self, label: Option<&str>, alias: &str) -> Result<RowStream<'a>> {
        use neural_core::Graph;

        let count = if let Some(label) = label {
            // O(1) count using label index
            self.store.nodes_with_label_count(label)
        } else {
            // O(1) total node count
            self.store.node_count()
        };

        let mut binding = Bindings::new();
        binding.bind(alias, Value::Int(count as i64));
        Ok(Box::new(std::iter::once(Ok(binding))))
    }

    /// O(1) edge count using index - Sprint 53 optimization
    fn execute_count_edges(&self, edge_type: Option<&str>, alias: &str) -> Result<RowStream<'a>> {
        use neural_core::Graph;

        let count = if let Some(edge_type) = edge_type {
            // O(1) count using edge type index
            self.store.edges_with_type_count(edge_type)
        } else {
            // O(1) total edge count
            self.store.edge_count()
        };

        let mut binding = Bindings::new();
        binding.bind(alias, Value::Int(count as i64));
        Ok(Box::new(std::iter::once(Ok(binding))))
    }

    /// Converts bindings to a QueryResult.
    ///
    /// ## Sprint 59 Optimization: Pre-allocation
    ///
    /// Uses `with_capacity()` to pre-allocate the exact number of rows needed,
    /// avoiding reallocations during result building.
    pub fn bindings_to_result(&self, bindings: &[Bindings], columns: &[ProjectColumn], _params: Option<&eval::Parameters>) -> Result<QueryResult> {
        let column_names: Vec<String> = if columns.is_empty() {
            if let Some(first) = bindings.first() {
                first.iter().map(|(k, _)| k.to_string()).collect()
            } else { Vec::new() }
        } else {
            columns.iter().map(|c| c.output_name()).collect()
        };

        // Pre-allocate with exact capacity needed
        let mut result = QueryResult::with_capacity(column_names.clone(), bindings.len());
        for binding in bindings {
            let mut row = Row::with_columns(column_names.clone());
            for col_name in &column_names {
                if let Some(val) = binding.get(col_name) {
                    row.set(col_name, val.clone());
                } else {
                    row.set(col_name, Value::Null);
                }
            }
            result.add_row(row);
        }
        Ok(result)
    }
}

impl From<neural_parser::Literal> for Value {
    fn from(lit: neural_parser::Literal) -> Self {
        match lit {
            neural_parser::Literal::Null => Value::Null,
            neural_parser::Literal::Bool(b) => Value::Bool(b),
            neural_parser::Literal::Int(i) => Value::Int(i),
            neural_parser::Literal::Float(f) => Value::Float(f),
            neural_parser::Literal::String(s) => Value::String(s),
            neural_parser::Literal::List(items) => {
                let values = items.into_iter().map(Value::from).collect();
                Value::List(values)
            }
        }
    }
}
