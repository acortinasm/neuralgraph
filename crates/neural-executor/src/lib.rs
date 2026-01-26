//! # Neural Executor
//!
//! Query planner and executor for NeuralGraphDB.

pub mod aggregate;
pub mod cmp;
pub mod eval;
pub mod executor;
pub mod plan;
pub mod planner;
pub mod result;

pub use executor::Executor;
pub use plan::{LogicalPlan, PhysicalPlan};
pub use planner::Planner;
pub use result::{QueryResult, Row, Value};
use result::Bindings;

use neural_core::Graph;
use neural_storage::{CsrMatrix, GraphStore};
use thiserror::Error;

/// Errors that can occur during query execution.
#[derive(Debug, Error)]
pub enum ExecutionError {
    /// Parse error
    #[error("Parse error: {0}")]
    ParseError(#[from] neural_parser::ParseError),

    /// WAL error
    #[error("WAL error: {0}")]
    WalError(#[from] neural_storage::wal::WalError),

    /// Planning error
    #[error("Planning error: {0}")]
    PlanningError(String),

    /// Execution error
    #[error("Execution error: {0}")]
    ExecutionError(String),

    /// Variable not found
    #[error("Variable not found: {0}")]
    VariableNotFound(String),
}

/// Result type for execution operations.
pub type Result<T> = std::result::Result<T, ExecutionError>;

/// Execute an NGQL query against a GraphStore (with properties).
pub fn execute_query(store: &GraphStore, query: &str) -> Result<QueryResult> {
    execute_query_with_params(store, query, None)
}

/// Execute an NGQL query against a GraphStore with parameters.
pub fn execute_query_with_params(
    store: &GraphStore,
    query: &str,
    params: Option<&eval::Parameters>,
) -> Result<QueryResult> {
    // Parse
    let ast = neural_parser::parse_query(query)?;
    execute_query_ast(store, &ast, params)
}

// =============================================================================
// Statement Execution (Sprint 21+)
// =============================================================================

/// Result of a mutation statement.
#[derive(Debug, Clone)]
pub enum MutationResult {
    NodesCreated { count: usize, node_ids: Vec<u64> },
    EdgesCreated { count: usize },
    NodesDeleted { count: usize },
    PropertiesSet { count: usize },
}

impl std::fmt::Display for MutationResult {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            MutationResult::NodesCreated { count, node_ids } => {
                write!(f, "Created {} node(s): {:?}", count, node_ids)
            }
            MutationResult::EdgesCreated { count } => {
                write!(f, "Created {} edge(s)", count)
            }
            MutationResult::NodesDeleted { count } => {
                write!(f, "Deleted {} node(s)", count)
            }
            MutationResult::PropertiesSet { count } => {
                write!(f, "Set {} propert(ies)", count)
            }
        }
    }
}

/// Result of executing a statement - either a query result or mutation result.
#[derive(Debug)]
pub enum StatementResult {
    Query(QueryResult),
    Mutation(MutationResult),
    Explain(String),
    TransactionStarted,
    TransactionCommitted,
    TransactionRolledBack,
}

/// Execute an NGQL statement (query or mutation) against a mutable GraphStore.
pub fn execute_statement(store: &mut GraphStore, input: &str) -> Result<StatementResult> {
    execute_statement_with_params(store, input, None, &mut None)
}

/// Execute an NGQL statement with parameters.
pub fn execute_statement_with_params(
    store: &mut GraphStore,
    input: &str,
    params: Option<&eval::Parameters>,
    tx: &mut Option<neural_storage::transaction::Transaction>,
) -> Result<StatementResult> {
    use neural_parser::parse_statement;
    let stmt = parse_statement(input)?;
    execute_statement_struct(store, stmt, params, tx)
}

/// Execute a read-only query from a pre-parsed AST.
/// This avoids double-parsing when the caller already has parsed the query.
pub fn execute_query_from_ast(
    store: &GraphStore,
    ast: &neural_parser::Query,
    params: Option<&eval::Parameters>,
) -> Result<QueryResult> {
    execute_query_ast(store, ast, params)
}

/// Execute a statement from a pre-parsed AST.
/// This avoids double-parsing when the caller already has parsed the statement.
pub fn execute_statement_from_ast(
    store: &mut GraphStore,
    stmt: neural_parser::Statement,
    params: Option<&eval::Parameters>,
    tx: &mut Option<neural_storage::transaction::Transaction>,
) -> Result<StatementResult> {
    execute_statement_struct(store, stmt, params, tx)
}

/// Internal execution logic for a parsed statement.
pub fn execute_statement_struct(
    store: &mut GraphStore,
    stmt: neural_parser::Statement,
    params: Option<&eval::Parameters>,
    tx: &mut Option<neural_storage::transaction::Transaction>,
) -> Result<StatementResult> {
    use neural_parser::Statement;

    match stmt {
        Statement::Begin => {
            if tx.is_some() {
                 return Err(ExecutionError::ExecutionError("Transaction already active".into()));
            }
            *tx = Some(store.transaction_manager.begin());
            Ok(StatementResult::TransactionStarted)
        }
        Statement::Commit => {
            if let Some(mut t) = tx.take() {
                t.commit(store).map_err(|e| ExecutionError::WalError(neural_storage::wal::WalError::Io(std::io::Error::new(std::io::ErrorKind::Other, format!("{:?}", e)))))?;
                Ok(StatementResult::TransactionCommitted)
            } else {
                Err(ExecutionError::ExecutionError("No active transaction to commit".into()))
            }
        }
        Statement::Rollback => {
            if let Some(mut t) = tx.take() {
                t.rollback().map_err(|e| ExecutionError::ExecutionError(format!("{:?}", e)))?;
                Ok(StatementResult::TransactionRolledBack)
            } else {
                Err(ExecutionError::ExecutionError("No active transaction to rollback".into()))
            }
        }
        Statement::Query(query) => {
            // Check if query contains a MERGE clause
             let mut has_merge = false;
             for clause in &query.clauses {
                 if let neural_parser::Clause::Merge(_) = clause {
                     has_merge = true;
                     break;
                 }
             }

             if has_merge {
                 let res = execute_merge_query(store, &query, params, tx.as_mut())?;
                 Ok(StatementResult::Query(res))
             } else {
                 let result = execute_query_ast(store, &query, params)?;
                 Ok(StatementResult::Query(result))
             }
        }
        Statement::Create(create_clause) => {
            let mutation_result = execute_create(store, &create_clause, tx.as_mut())?;
            Ok(StatementResult::Mutation(mutation_result))
        }
        Statement::Delete {
            match_clause,
            where_clause,
            delete_clause,
        } => {
            let mutation_result = execute_delete(
                store,
                match_clause.as_ref(),
                where_clause.as_ref(),
                &delete_clause,
                params,
                tx.as_mut()
            )?;
            Ok(StatementResult::Mutation(mutation_result))
        }
        Statement::Set {
            match_clause,
            where_clause,
            set_clause,
        } => {
            let mutation_result =
                execute_set(store, &match_clause, where_clause.as_ref(), &set_clause, params, tx.as_mut())?;
            Ok(StatementResult::Mutation(mutation_result))
        }
        Statement::CreateWithMatch {
            match_clause,
            where_clause,
            create_clause,
        } => {
            let mutation_result =
                execute_create_with_match(store, &match_clause, where_clause.as_ref(), &create_clause, params, tx.as_mut())?;
            Ok(StatementResult::Mutation(mutation_result))
        }
        Statement::Explain(inner) => {
            match *inner {
                Statement::Query(query) => {
                    let planner = Planner::new();
                    let plan = planner.plan(&query)?;
                    let plan_str = format!("{}", plan);
                    Ok(StatementResult::Explain(plan_str))
                }
                _ => {
                    Ok(StatementResult::Explain("EXPLAIN only supported for MATCH queries currently".to_string()))
                }
            }
        }
        Statement::Profile(inner) => {
            let start = std::time::Instant::now();
            let result = execute_statement_struct(store, *inner, params, tx)?;
            let duration = start.elapsed();
            
            match result {
                StatementResult::Query(mut query_result) => {
                    query_result.add_stat("execution_time", format!("{:?}", duration));
                    Ok(StatementResult::Query(query_result))
                }
                StatementResult::Mutation(mutation_result) => {
                    Ok(StatementResult::Mutation(mutation_result))
                }
                StatementResult::Explain(plan) => {
                    Ok(StatementResult::Explain(plan))
                }
                other => Ok(other),
            }
        }
    }
}

/// Execute a read query from an already-parsed AST.
pub fn execute_query_ast(
    store: &GraphStore,
    ast: &neural_parser::Query,
    params: Option<&eval::Parameters>,
) -> Result<QueryResult> {
    let planner = Planner::new();
    let plan = planner.plan(ast)?;
    let executor = Executor::new(store);
    executor.execute(&plan, params)
}

/// Execute a query that contains MERGE clauses.
fn execute_merge_query(
    store: &mut GraphStore,
    query: &neural_parser::Query,
    params: Option<&eval::Parameters>,
    mut tx: Option<&mut neural_storage::transaction::Transaction>,
) -> Result<QueryResult> {
    use neural_parser::Clause;

    // We execute clauses one by one.
    // Read clauses are executed normally.
    // MERGE clauses are executed against the current state of bindings.
    
    let mut current_bindings: Vec<Bindings> = vec![Bindings::new()];
    let mut last_columns = Vec::new();

    for clause in &query.clauses {
        match clause {
            Clause::Match(c) => {
                let plan = Planner::new().plan_match(c)?;
                let executor = Executor::new(store);
                // We need a stream-based execution that can take existing bindings
                let initial_stream: executor::RowStream = Box::new(current_bindings.into_iter().map(Ok));
                let (new_stream, _) = executor.execute_plan_internal(&plan, initial_stream, params)?;
                current_bindings = new_stream.collect::<Result<Vec<_>>>()?;
            }
            Clause::OptionalMatch(c) => {
                let plan = Planner::new().plan_optional_match(c)?;
                let executor = Executor::new(store);
                let initial_stream: executor::RowStream = Box::new(current_bindings.into_iter().map(Ok));
                let (new_stream, _) = executor.execute_plan_internal(&plan, initial_stream, params)?;
                current_bindings = new_stream.collect::<Result<Vec<_>>>()?;
            }
            Clause::Merge(c) => {
                current_bindings = execute_merge_clause(store, c, current_bindings, params, tx.as_mut().map(|t| &mut **t))?;
            }
            Clause::With(c) => {
                let plan = Planner::new().plan_with(c)?;
                let executor = Executor::new(store);
                let initial_stream: executor::RowStream = Box::new(current_bindings.into_iter().map(Ok));
                let (new_stream, cols) = executor.execute_plan_internal(&plan, initial_stream, params)?;
                current_bindings = new_stream.collect::<Result<Vec<_>>>()?;
                last_columns = cols;
            }
            Clause::Unwind(c) => {
                let plan = Planner::new().plan_unwind(c)?;
                let executor = Executor::new(store);
                let initial_stream: executor::RowStream = Box::new(current_bindings.into_iter().map(Ok));
                let (new_stream, _) = executor.execute_plan_internal(&plan, initial_stream, params)?;
                current_bindings = new_stream.collect::<Result<Vec<_>>>()?;
            }
            Clause::Return(c) => {
                let plan = Planner::new().plan_return(c)?;
                let executor = Executor::new(store);
                let initial_stream: executor::RowStream = Box::new(current_bindings.into_iter().map(Ok));
                let (new_stream, cols) = executor.execute_plan_internal(&plan, initial_stream, params)?;
                current_bindings = new_stream.collect::<Result<Vec<_>>>()?;
                last_columns = cols;
            }
        }
        
        if current_bindings.is_empty() {
            break;
        }
    }

    let executor = Executor::new(store);
    executor.bindings_to_result(&current_bindings, &last_columns, params)
}

/// Execute a MERGE clause.
fn execute_merge_clause(
    store: &mut GraphStore,
    merge: &neural_parser::MergeClause,
    input_bindings: Vec<Bindings>,
    _params: Option<&eval::Parameters>,
    mut tx: Option<&mut neural_storage::transaction::Transaction>,
) -> Result<Vec<Bindings>> {
    use neural_core::NodeId;
    

    let mut output_bindings = Vec::new();

    for mut binding in input_bindings {
        // For simplicity, we only support simple node MERGE for now: MERGE (n:Label {prop: val})
        let pattern = &merge.pattern;
        if pattern.chain.is_empty() {
             let node_pat = &pattern.start;
             let var_name = node_pat.identifier.as_ref();
             
             // Check if already matches
             // 1. If var is bound, check if it matches label/props.
             // 2. If not bound, search for matching node.
             
             let mut matched_node = None;
             
             if let Some(name) = var_name {
                 if let Some(val) = binding.get(name) {
                      if let Some(node_id) = val.as_node().map(NodeId::new) {
                           // Verify match
                           let mut matches = true;
                           if let Some(lbl) = &node_pat.label {
                               if store.get_label(node_id) != Some(lbl) { matches = false; }
                           }
                           if matches {
                               if let Some(props) = &node_pat.properties {
                                   for (k, v) in props {
                                       let prop_val = literal_to_property_value(v);
                                       if store.get_property(node_id, k) != Some(&prop_val) {
                                           matches = false; break;
                                       }
                                   }
                               }
                           }
                           
                           if matches { matched_node = Some(node_id); }
                      }
                 }
             }
             
             if matched_node.is_none() {
                 // Try searching by label + props
                 if let Some(lbl) = &node_pat.label {
                      for node_id in store.nodes_with_label(lbl) {
                          let mut matches = true;
                          if let Some(props) = &node_pat.properties {
                               for (k, v) in props {
                                   let prop_val = literal_to_property_value(v);
                                   if store.get_property(node_id, k) != Some(&prop_val) {
                                       matches = false; break;
                                   }
                               }
                          }
                          if matches { matched_node = Some(node_id); break; }
                      }
                 } else if let Some(props) = &node_pat.properties {
                      // Search by first property (inefficient)
                      if let Some((k, v)) = props.iter().next() {
                           let prop_val = literal_to_property_value(v);
                           for node_id in store.nodes_with_property(k, &prop_val) {
                                let mut matches = true;
                                for (pk, pv) in props {
                                     let ppv = literal_to_property_value(pv);
                                     if store.get_property(node_id, pk) != Some(&ppv) {
                                         matches = false; break;
                                     }
                                }
                                if matches { matched_node = Some(node_id); break; }
                           }
                      }
                 }
             }
             
             if let Some(node_id) = matched_node {
                 if let Some(name) = var_name {
                     binding.bind(name, node_id);
                 }
                 output_bindings.push(binding);
             } else {
                 // CREATE
                 let mut props = Vec::new();
                 if let Some(p_map) = &node_pat.properties {
                      for (k, v) in p_map {
                          props.push((k.clone(), literal_to_property_value(v)));
                      }
                 }
                 let node_id = store.create_node(node_pat.label.as_deref(), props, tx.as_mut().map(|t| &mut **t))?;
                 if let Some(name) = var_name {
                     binding.bind(name, node_id);
                 }
                 output_bindings.push(binding);
             }
        } else {
            // Edge MERGE (a)-[:TYPE]->(b)
            // 1. Resolve 'a' and 'b' from bindings or error.
            // 2. Check if edge exists.
            // 3. Create if not.
            
            let from_pat = &pattern.start;
            let (rel_pat, to_pat) = &pattern.chain[0];
            
            let from_name = from_pat.identifier.as_ref().ok_or_else(|| ExecutionError::ExecutionError("MERGE edge requires bound source node".into()))?;
            let to_name = to_pat.identifier.as_ref().ok_or_else(|| ExecutionError::ExecutionError("MERGE edge requires bound target node".into()))?;
            
            let from_node = binding.get_node(from_name).ok_or_else(|| ExecutionError::ExecutionError(format!("MERGE source '{}' not bound", from_name)))?;
            let to_node = binding.get_node(to_name).ok_or_else(|| ExecutionError::ExecutionError(format!("MERGE target '{}' not bound", to_name)))?;
            
            let rel_type = rel_pat.label.as_deref();
            
            let exists = if let Some(t) = rel_type {
                store.neighbors_via_type(from_node, t).any(|n| n == to_node)
            } else {
                store.neighbors(from_node).any(|n| n == to_node)
            };
            
            if !exists {
                store.create_edge(from_node, to_node, rel_type, tx.as_mut().map(|t| &mut **t))?;
            }
            
            output_bindings.push(binding);
        }
    }

    Ok(output_bindings)
}

/// Execute a CREATE statement.
fn execute_create(
    store: &mut GraphStore,
    create_clause: &neural_parser::CreateClause,
    mut tx: Option<&mut neural_storage::transaction::Transaction>,
) -> Result<MutationResult> {
    use neural_core::{NodeId, PropertyValue};
    use neural_parser::CreatePattern;
    use std::collections::HashMap;

    let mut created_node_ids = Vec::new();
    let mut created_edge_count = 0usize;
    let mut bindings: HashMap<String, NodeId> = HashMap::new();

    for pattern in &create_clause.patterns {
        if let CreatePattern::Node {
            binding,
            label,
            properties,
        } = pattern
        {
            let props: Vec<(String, PropertyValue)> = properties
                .iter()
                .map(|(k, lit)| {
                    let value = literal_to_property_value(lit);
                    (k.clone(), value)
                })
                .collect();

            // We must re-borrow tx for each call if it's mutable reference
            // Option<&mut T> is tricky with loops.
            // We can re-borrow from `tx` using `as_deref_mut` logic or just `tx.as_deref_mut()` if it was Option<Box>..
            // But `tx` is `Option<&mut Transaction>`.
            // We can't copy it.
            // We need to pass `tx.as_deref_mut()` but `tx` is not a Box.
            // We can pass `tx.as_mut().map(|t| *t)`? No.
            
            // Rust: Option<&mut T>
            // We need to pass `Option<&mut T>` to `create_node`.
            // But `tx` is consumed? No, it's `Option<&mut T>`.
            // We can reborrow inside the loop?
            // `store.create_node(..., tx.as_mut().map(|t| *t))`?
            // `*t` is `&mut Transaction`.
            // `as_mut()` on `Option<&mut T>` gives `Option<&mut &mut T>`.
            
            // The issue is standard "cannot borrow `*tx` as mutable more than once at a time".
            // Since `create_node` takes `Option<&mut Transaction>`, we need to pass a reborrow.
            
            // Solution: Use `as_deref_mut()`? No, that's for `Option<Box<T>>` or `Option<String>`.
            // For `Option<&mut T>`, we can use `tx.as_mut().map(|t| &mut **t)`.
            
            let node_id = store.create_node(label.as_deref(), props, tx.as_mut().map(|t| &mut **t))?;
            created_node_ids.push(node_id.as_u64());

            if let Some(var_name) = binding {
                bindings.insert(var_name.clone(), node_id);
            }
        }
    }

    for pattern in &create_clause.patterns {
        if let CreatePattern::Edge {
            from,
            to,
            edge_type,
            properties: _,
        } = pattern
        {
            let source_id = bindings.get(from).ok_or_else(|| {
                ExecutionError::VariableNotFound(format!(
                    "Node binding '{}' not found in CREATE statement",
                    from
                ))
            })?;

            let target_id = bindings.get(to).ok_or_else(|| {
                ExecutionError::VariableNotFound(format!(
                    "Node binding '{}' not found in CREATE statement",
                    to
                ))
            })?;

            let _edge_id = store.create_edge(*source_id, *target_id, edge_type.as_deref(), tx.as_mut().map(|t| &mut **t))?;

            created_edge_count += 1;
        }
    }

    if created_edge_count > 0 && !created_node_ids.is_empty() {
        Ok(MutationResult::NodesCreated {
            count: created_node_ids.len(),
            node_ids: created_node_ids,
        })
    } else if created_edge_count > 0 {
        Ok(MutationResult::EdgesCreated {
            count: created_edge_count,
        })
    } else {
        Ok(MutationResult::NodesCreated {
            count: created_node_ids.len(),
            node_ids: created_node_ids,
        })
    }
}

fn literal_to_property_value(lit: &neural_parser::Literal) -> neural_core::PropertyValue {
    use neural_core::PropertyValue;
    use neural_parser::Literal;

    match lit {
        Literal::String(s) => PropertyValue::String(s.clone()),
        Literal::Int(i) => PropertyValue::Int(*i),
        Literal::Float(f) => PropertyValue::Float(*f),
        Literal::Bool(b) => PropertyValue::Bool(*b),
        Literal::Null => PropertyValue::Null,
        Literal::List(items) => {
            // Try to convert to Vector (Vec<f32>) if all items are numeric
            let mut float_vec = Vec::new();
            let mut is_numeric_vector = true;

            for item in items {
                match item {
                    Literal::Int(i) => float_vec.push(*i as f32),
                    Literal::Float(f) => float_vec.push(*f as f32),
                    _ => {
                        is_numeric_vector = false;
                        break;
                    }
                }
            }

            if is_numeric_vector && !items.is_empty() {
                PropertyValue::Vector(float_vec)
            } else {
                // Fallback: Store as string representation for now, or handle general lists later.
                // NeuralGraph core properties currently support primitives + Vector.
                // For now, let's format it as a string to avoid crashing.
                // Ideal would be PropertyValue::List(...) but that requires core changes.
                PropertyValue::String(format!("{:?}", items))
            }
        }
    }
}

/// Execute a DELETE statement.
fn execute_delete(
    store: &mut GraphStore,
    match_clause: Option<&neural_parser::MatchClause>,
    where_clause: Option<&neural_parser::WhereClause>,
    delete_clause: &neural_parser::DeleteClause,
    params: Option<&eval::Parameters>,
    mut tx: Option<&mut neural_storage::transaction::Transaction>,
) -> Result<MutationResult> {
    use neural_core::NodeId;
    use neural_parser::{Clause, Query, ReturnClause, ReturnItem, Expression};

    let match_cl = match_clause.ok_or_else(|| {
        ExecutionError::ExecutionError("DELETE requires a MATCH clause to find nodes".to_string())
    })?;

    let mut nodes_to_delete: Vec<NodeId> = Vec::new();

    for var_name in &delete_clause.items {
        // Build query using new Query struct
        let query = Query {
            clauses: vec![
                Clause::Match(neural_parser::MatchClause {
                    patterns: match_cl.patterns.clone(),
                    where_clause: match_cl.where_clause.clone().or(where_clause.cloned()),
                }),
                Clause::Return(ReturnClause {
                    items: vec![ReturnItem {
                        expression: Expression::Property {
                            variable: var_name.clone(),
                            property: String::new(),
                        },
                        alias: None,
                    }],
                    distinct: false,
                    order_by: None,
                    limit: None,
                    group_by: None,
                })
            ]
        };

        let result = execute_query_ast(store, &query, params)?;

        for row in result.rows() {
            if let Some(Value::Node(id)) = row.get(var_name) {
                nodes_to_delete.push(NodeId::new(*id));
            }
        }
    }

    nodes_to_delete.sort_unstable();
    nodes_to_delete.dedup();

    let mut deleted_count = 0;
    for node_id in nodes_to_delete {
        match store.delete_node(node_id, delete_clause.detach, tx.as_mut().map(|t| &mut **t)) {
            Ok(()) => deleted_count += 1,
            Err(e) => return Err(ExecutionError::ExecutionError(e)),
        }
    }

    Ok(MutationResult::NodesDeleted {
        count: deleted_count,
    })
}

/// Execute a SET statement.
fn execute_set(
    store: &mut GraphStore,
    match_clause: &neural_parser::MatchClause,
    where_clause: Option<&neural_parser::WhereClause>,
    set_clause: &neural_parser::SetClause,
    params: Option<&eval::Parameters>,
    mut tx: Option<&mut neural_storage::transaction::Transaction>,
) -> Result<MutationResult> {
    use neural_core::NodeId;
    use neural_parser::{Clause, Query, ReturnClause, ReturnItem, Expression};

    let var_names: Vec<_> = set_clause
        .assignments
        .iter()
        .map(|a| a.variable.clone())
        .collect();

    let query = Query {
        clauses: vec![
            Clause::Match(neural_parser::MatchClause {
                patterns: match_clause.patterns.clone(),
                where_clause: match_clause.where_clause.clone().or(where_clause.cloned()),
            }),
            Clause::Return(ReturnClause {
                items: var_names
                    .iter()
                    .map(|v| ReturnItem {
                        expression: Expression::Property {
                            variable: v.clone(),
                            property: String::new(),
                        },
                        alias: None,
                    })
                    .collect(),
                distinct: false,
                order_by: None,
                limit: None,
                group_by: None,
            })
        ]
    };

    let result = execute_query_ast(store, &query, params)?;

    let mut properties_set = 0;

    for row in result.rows() {
        for assignment in &set_clause.assignments {
            if let Some(Value::Node(id)) = row.get(&assignment.variable) {
                let empty_bindings = Bindings::new();
                let value_res = eval::evaluate(&assignment.value, &empty_bindings, store, params)?;
                let prop_value = value_to_property_value(value_res);

                let node_id = NodeId::new(*id);
                if store.update_property(node_id, &assignment.property, prop_value, tx.as_mut().map(|t| &mut **t))? {
                    properties_set += 1;
                }
            }
        }
    }

    Ok(MutationResult::PropertiesSet {
        count: properties_set,
    })
}

/// Execute a CREATE statement with a preceding MATCH.
fn execute_create_with_match(
    store: &mut GraphStore,
    match_clause: &neural_parser::MatchClause,
    where_clause: Option<&neural_parser::WhereClause>,
    create_clause: &neural_parser::CreateClause,
    params: Option<&eval::Parameters>,
    mut tx: Option<&mut neural_storage::transaction::Transaction>,
) -> Result<MutationResult> {
    use neural_core::{NodeId, PropertyValue};
    use neural_parser::{Clause, Query, ReturnClause, ReturnItem, Expression, CreatePattern};
    use std::collections::HashMap;

    // 1. Identify variables needed for CREATE
    let mut needed_vars = Vec::new();
    for pattern in &create_clause.patterns {
        match pattern {
            CreatePattern::Edge { from, to, .. } => {
                needed_vars.push(from.clone());
                needed_vars.push(to.clone());
            }
            CreatePattern::Node { .. } => {} // Nodes usually define new vars or don't need existing ones (unless copying?)
        }
    }
    needed_vars.sort();
    needed_vars.dedup();

    // 2. Execute MATCH query
    let query = Query {
        clauses: vec![
            Clause::Match(neural_parser::MatchClause {
                patterns: match_clause.patterns.clone(),
                where_clause: match_clause.where_clause.clone().or(where_clause.cloned()),
            }),
            Clause::Return(ReturnClause {
                items: needed_vars
                    .iter()
                    .map(|v| ReturnItem {
                        expression: Expression::Property {
                            variable: v.clone(),
                            property: String::new(),
                        },
                        alias: None,
                    })
                    .collect(),
                distinct: false,
                order_by: None,
                limit: None,
                group_by: None,
            })
        ]
    };

    // Use read-only query execution first
    let result = execute_query_ast(store, &query, params)?;
    
    // 3. Iterate results and perform creations
    // Note: We collect rows first to avoid borrowing issues if we need to mutate store
    let rows: Vec<_> = result.rows().to_vec();
    
    let mut created_nodes = 0;
    let mut created_edges = 0;
    let mut new_node_ids = Vec::new();

    for row in rows {
        // Build a local binding map for this row
        let mut row_bindings: HashMap<String, NodeId> = HashMap::new();
        for var in &needed_vars {
            if let Some(Value::Node(id)) = row.get(var) {
                row_bindings.insert(var.clone(), NodeId::new(*id));
            }
        }

        // Execute CREATE patterns for this row
        for pattern in &create_clause.patterns {
            match pattern {
                CreatePattern::Node { binding, label, properties } => {
                    let props: Vec<(String, PropertyValue)> = properties
                        .iter()
                        .map(|(k, lit)| (k.clone(), literal_to_property_value(lit)))
                        .collect();
                    
                    let node_id = store.create_node(label.as_deref(), props, tx.as_mut().map(|t| &mut **t))?;
                    created_nodes += 1;
                    new_node_ids.push(node_id.as_u64());
                    
                    if let Some(name) = binding {
                        row_bindings.insert(name.clone(), node_id);
                    }
                }
                CreatePattern::Edge { from, to, edge_type, properties: _ } => {
                    // For edges, from/to must be bound either from MATCH or previous CREATE in this row
                    if let (Some(src), Some(tgt)) = (row_bindings.get(from), row_bindings.get(to)) {
                         store.create_edge(*src, *tgt, edge_type.as_deref(), tx.as_mut().map(|t| &mut **t))?;
                         created_edges += 1;
                    } else {
                        // If variables are missing, we skip (or should error?)
                        // Cypher usually errors if unbound variables are used in CREATE
                    }
                }
            }
        }
    }

    if created_edges > 0 {
        Ok(MutationResult::EdgesCreated { count: created_edges })
    } else {
        Ok(MutationResult::NodesCreated { count: created_nodes, node_ids: new_node_ids })
    }
}

fn value_to_property_value(val: Value) -> neural_core::PropertyValue {
    use neural_core::PropertyValue;
    match val {
        Value::Null => PropertyValue::Null,
        Value::Bool(b) => PropertyValue::Bool(b),
        Value::Int(i) => PropertyValue::Int(i),
        Value::Float(f) => PropertyValue::Float(f),
        Value::String(s) => PropertyValue::String(s),
        Value::Date(s) => PropertyValue::Date(s),
        Value::DateTime(s) => PropertyValue::DateTime(s),
        Value::Node(n) => PropertyValue::Int(n as i64),
        Value::Edge(e) => PropertyValue::Int(e as i64),
        Value::List(_) => PropertyValue::Null,
    }
}

pub fn execute_query_on_csr(graph: &CsrMatrix, query: &str) -> Result<QueryResult> {
    let store = GraphStore::new_from_csr(graph.clone());
    execute_query(&store, query)
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use neural_core::Graph;

    #[test]
    fn test_execute_create_simple_node() {
        let mut store = GraphStore::builder().build();
        let initial_count = store.node_count(); // 1
        
        let result = execute_statement(&mut store, "CREATE (n)").unwrap();

        match result {
            StatementResult::Mutation(MutationResult::NodesCreated { count, node_ids }) => {
                assert_eq!(count, 1);
                assert_eq!(node_ids.len(), 1);
                // The new node gets the next available ID
                assert_eq!(node_ids[0] as usize, initial_count);
            }
            _ => panic!("Expected NodesCreated result"),
        }
    }

    #[test]
    fn test_execute_create_labeled_node() {
        let mut store = GraphStore::builder().build();
        let result = execute_statement(&mut store, "CREATE (n:Person)").unwrap();

        match result {
            StatementResult::Mutation(MutationResult::NodesCreated { count, node_ids }) => {
                assert_eq!(count, 1);
                let node_id = neural_core::NodeId::new(node_ids[0]);
                assert_eq!(store.get_label(node_id), Some("Person"));
            }
            _ => panic!("Expected NodesCreated result"),
        }
    }

    #[test]
    fn test_execute_create_node_with_properties() {
        let mut store = GraphStore::builder().build();
        let result = execute_statement(&mut store, r#"CREATE (n:Person {name: "Alice", age: 30})"#).unwrap();

        match result {
            StatementResult::Mutation(MutationResult::NodesCreated { count, node_ids }) => {
                assert_eq!(count, 1);
                let node_id = neural_core::NodeId::new(node_ids[0]);
                assert_eq!(store.get_label(node_id), Some("Person"));
                
                let name = store.get_property(node_id, "name");
                assert!(matches!(name, Some(neural_core::PropertyValue::String(s)) if s == "Alice"));
            }
            _ => panic!("Expected NodesCreated result"),
        }
    }

    #[test]
    fn test_execute_statement_with_query() {
        let store = GraphStore::builder().add_node(0u64, [("name", "Alice")]).build();
        let mut store_mut = store;
        let result = execute_statement(&mut store_mut, "MATCH (n) RETURN n").unwrap();

        match result {
            StatementResult::Query(query_result) => {
                assert!(query_result.row_count() > 0);
            }
            _ => panic!("Expected Query result"),
        }
    }
    
    #[test]
    fn test_execute_with_clause() {
        let mut store = GraphStore::builder()
            .add_labeled_node(0u64, "Person", [("name", "Alice"), ("age", "30")])
            .build();

        // MATCH (n) WITH n.name AS name RETURN name
        let result = execute_statement(
            &mut store, 
            "MATCH (n:Person) WITH n.name AS name RETURN name"
        ).unwrap();

        match result {
            StatementResult::Query(res) => {
                assert_eq!(res.row_count(), 1);
                assert_eq!(res.columns(), &["name"]);
                let row = &res.rows()[0];
                assert_eq!(row.get("name"), Some(&Value::String("Alice".to_string())));
            }
            _ => panic!("Expected Query result"),
        }
    }

    #[test]
    fn test_execute_unwind() {
        let mut store = GraphStore::builder().build();
        
        use std::collections::HashMap;
        let mut params = HashMap::new();
        params.insert("list".to_string(), Value::List(vec![
            Value::Int(1), Value::Int(2), Value::Int(3)
        ]));

        let result = execute_statement_with_params(
            &mut store,
            "UNWIND $list AS x RETURN x",
            Some(&params),
            &mut None,
        ).unwrap();

        match result {
            StatementResult::Query(res) => {
                assert_eq!(res.row_count(), 3);
                let rows = res.rows();
                assert_eq!(rows[0].get("x"), Some(&Value::Int(1)));
                assert_eq!(rows[1].get("x"), Some(&Value::Int(2)));
                assert_eq!(rows[2].get("x"), Some(&Value::Int(3)));
            }
            _ => panic!("Expected Query result"),
        }
    }

    #[test]
    fn test_execute_optional_match() {
        let mut store = GraphStore::builder()
            .add_labeled_node(0u64, "Person", [("name", "Alice")])
            .build();

        let query = "MATCH (a:Person) OPTIONAL MATCH (a)-[:KNOWS]->(b) RETURN a.name, b";
        let result = execute_statement(&mut store, query).unwrap();

        match result {
            StatementResult::Query(res) => {
                assert_eq!(res.row_count(), 1);
                let row = &res.rows()[0];
                assert_eq!(row.get("a.name"), Some(&Value::String("Alice".to_string())));
                assert_eq!(row.get("b"), Some(&Value::Null));
            }
            _ => panic!("Expected Query result"),
        }
    }

    #[test]
    fn test_execute_merge() {
        let mut store = GraphStore::builder().build();
        
        // 1. MERGE new node
        let query1 = "MERGE (n:Person {name: 'Alice'}) RETURN n.name";
        let result1 = execute_statement(&mut store, query1).unwrap();
        match result1 {
            StatementResult::Query(res) => {
                assert_eq!(res.row_count(), 1);
                assert_eq!(res.rows()[0].get("n.name"), Some(&Value::String("Alice".to_string())));
            }
            _ => panic!("Expected Query result"),
        }
        
        // 2. MERGE existing node (should not create duplicate)
        let node_count_before = store.node_count();
        let result2 = execute_statement(&mut store, query1).unwrap();
        assert_eq!(store.node_count(), node_count_before);
        match result2 {
            StatementResult::Query(res) => {
                assert_eq!(res.row_count(), 1);
                assert_eq!(res.rows()[0].get("n.name"), Some(&Value::String("Alice".to_string())));
            }
            _ => panic!("Expected Query result"),
        }
    }
}
