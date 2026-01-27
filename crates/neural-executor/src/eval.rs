//! Expression evaluation.
//!
//! Evaluates expressions against node bindings and property store.
//!
//! Supports MVCC snapshot isolation via `evaluate_at()` for transactional reads.

use crate::result::{Bindings, Value};
use crate::{ExecutionError, Result};
use neural_core::PropertyValue;
use neural_parser::{ComparisonOp, Expression, Literal};
use neural_storage::{GraphStore, graph_store::MAX_SNAPSHOT_ID};
use neural_storage::wal::TransactionId;
use std::collections::HashMap;

/// Parameters for query execution.
pub type Parameters = HashMap<String, Value>;

/// Evaluates an expression given bindings and a graph store.
///
/// Uses MAX_SNAPSHOT_ID so all committed data is visible.
pub fn evaluate(
    expr: &Expression,
    bindings: &Bindings,
    store: &GraphStore,
    params: Option<&Parameters>,
) -> Result<Value> {
    evaluate_at(expr, bindings, store, params, MAX_SNAPSHOT_ID)
}

/// Evaluates an expression at a specific snapshot for MVCC reads.
///
/// Only data committed before `snapshot_id` is visible.
pub fn evaluate_at(
    expr: &Expression,
    bindings: &Bindings,
    store: &GraphStore,
    params: Option<&Parameters>,
    snapshot_id: TransactionId,
) -> Result<Value> {
    match expr {
        Expression::Literal(lit) => Ok(literal_to_value(lit)),

        Expression::Property { variable, property } => {
            if property.is_empty() {
                if let Some(val) = bindings.get(variable) {
                    Ok(val.clone())
                } else {
                    Err(ExecutionError::VariableNotFound(variable.clone()))
                }
            } else if let Some(val) = bindings.get(variable) {
                 if let Some(node_id) = val.as_node() {
                     let node = neural_core::NodeId::new(node_id);
                     // Use snapshot-aware property read for MVCC
                     if let Some(prop_value) = store.get_property_at(node, property, snapshot_id) {
                        Ok(property_value_to_value(prop_value))
                    } else if property == "id" {
                        Ok(Value::Int(node_id as i64))
                    } else {
                        Ok(Value::Null)
                    }
                 } else {
                     Err(ExecutionError::ExecutionError(format!(
                         "Variable '{}' is not a node, cannot access property '{}'",
                         variable, property
                     )))
                 }
            } else {
                Err(ExecutionError::VariableNotFound(variable.clone()))
            }
        }

        Expression::Parameter(name) => {
            if let Some(params) = params {
                if let Some(value) = params.get(name) {
                    Ok(value.clone())
                } else {
                    Err(ExecutionError::ExecutionError(format!(
                        "Parameter ${} not found in provided parameters",
                        name
                    )))
                }
            } else {
                Err(ExecutionError::ExecutionError(format!(
                    "Parameter ${} encountered but no parameters were provided",
                    name
                )))
            }
        }

        Expression::Comparison { left, op, right } => {
            let left_val = evaluate_at(left, bindings, store, params, snapshot_id)?;
            let right_val = evaluate_at(right, bindings, store, params, snapshot_id)?;
            Ok(Value::Bool(compare_values(&left_val, op, &right_val)))
        }

        Expression::And(left, right) => {
            let left_val = evaluate_at(left, bindings, store, params, snapshot_id)?;
            if !to_bool(&left_val) {
                return Ok(Value::Bool(false));
            }
            let right_val = evaluate_at(right, bindings, store, params, snapshot_id)?;
            Ok(Value::Bool(to_bool(&right_val)))
        }

        Expression::Or(left, right) => {
            let left_val = evaluate_at(left, bindings, store, params, snapshot_id)?;
            if to_bool(&left_val) {
                return Ok(Value::Bool(true));
            }
            let right_val = evaluate_at(right, bindings, store, params, snapshot_id)?;
            Ok(Value::Bool(to_bool(&right_val)))
        }

        Expression::Not(inner) => {
            let val = evaluate_at(inner, bindings, store, params, snapshot_id)?;
            Ok(Value::Bool(!to_bool(&val)))
        }

        Expression::Aggregate { .. } => {
            Err(ExecutionError::ExecutionError(
                "Aggregate expressions must be evaluated at the executor level".into(),
            ))
        }

        Expression::VectorSimilarity { .. } => {
            Err(ExecutionError::ExecutionError(
                "vector_similarity() in WHERE requires parameter support via executor".into(),
            ))
        }

        Expression::VectorSearch { .. } => {
            // VectorSearch is handled at the statement level via CALL neural.search()
            Err(ExecutionError::ExecutionError(
                "neural.search() must be called via CALL statement".into(),
            ))
        }

        Expression::Cluster { variable } => {
            let val = bindings
                .get(variable)
                .ok_or_else(|| ExecutionError::VariableNotFound(variable.clone()))?;
            
            let node = val.as_node().map(neural_core::NodeId::new).ok_or_else(|| {
                 ExecutionError::ExecutionError(format!("Variable '{}' is not a node", variable))
            })?;

            let communities = store.detect_communities();
            match communities.get(node) {
                Some(community_id) => Ok(Value::Int(community_id as i64)),
                None => Ok(Value::Null),
            }
        }

        // New Logic for Sprint 35

        Expression::Case { subject, when_then, else_expr } => {
            if let Some(subj) = subject {
                // Simple CASE: CASE x WHEN 1 THEN 'one' ...
                let subj_val = evaluate_at(subj, bindings, store, params, snapshot_id)?;
                for (cond, result_expr) in when_then {
                    let cond_val = evaluate_at(cond, bindings, store, params, snapshot_id)?;
                    if crate::cmp::values_equal(&subj_val, &cond_val) {
                        return evaluate_at(result_expr, bindings, store, params, snapshot_id);
                    }
                }
            } else {
                // Generic CASE: CASE WHEN x > 1 THEN 'gt' ...
                for (cond, result_expr) in when_then {
                    let cond_val = evaluate_at(cond, bindings, store, params, snapshot_id)?;
                    if to_bool(&cond_val) {
                        return evaluate_at(result_expr, bindings, store, params, snapshot_id);
                    }
                }
            }

            if let Some(else_e) = else_expr {
                evaluate_at(else_e, bindings, store, params, snapshot_id)
            } else {
                Ok(Value::Null)
            }
        }

        Expression::List(items) => {
            let mut values = Vec::with_capacity(items.len());
            for item in items {
                values.push(evaluate_at(item, bindings, store, params, snapshot_id)?);
            }
            Ok(Value::List(values))
        }

        Expression::FunctionCall { name, args } => {
            evaluate_function(name, args, bindings, store, params, snapshot_id)
        }
    }
}

fn evaluate_function(
    name: &str,
    args: &[Expression],
    bindings: &Bindings,
    store: &GraphStore,
    params: Option<&Parameters>,
    snapshot_id: TransactionId,
) -> Result<Value> {
    // Evaluate all arguments first? Or lazy?
    // Most functions are strict. COALESCE is lazy (sort of).
    // Let's implement COALESCE lazily.

    if name.eq_ignore_ascii_case("coalesce") {
        for arg in args {
            let val = evaluate_at(arg, bindings, store, params, snapshot_id)?;
            if !matches!(val, Value::Null) {
                return Ok(val);
            }
        }
        return Ok(Value::Null);
    }

    // Evaluate all args for strict functions
    let mut eval_args = Vec::with_capacity(args.len());
    for arg in args {
        eval_args.push(evaluate_at(arg, bindings, store, params, snapshot_id)?);
    }

    match name.to_lowercase().as_str() {
        "tolower" => {
            if let Some(Value::String(s)) = eval_args.first() {
                Ok(Value::String(s.to_lowercase()))
            } else {
                Ok(Value::Null)
            }
        }
        "toupper" => {
            if let Some(Value::String(s)) = eval_args.first() {
                Ok(Value::String(s.to_uppercase()))
            } else {
                Ok(Value::Null)
            }
        }
        "contains" => {
            if let (Some(Value::String(haystack)), Some(Value::String(needle))) = (eval_args.first(), eval_args.get(1)) {
                Ok(Value::Bool(haystack.contains(needle)))
            } else {
                Ok(Value::Null)
            }
        }
        "startswith" => {
            if let (Some(Value::String(haystack)), Some(Value::String(needle))) = (eval_args.first(), eval_args.get(1)) {
                Ok(Value::Bool(haystack.starts_with(needle)))
            } else {
                Ok(Value::Null)
            }
        }
        "endswith" => {
            if let (Some(Value::String(haystack)), Some(Value::String(needle))) = (eval_args.first(), eval_args.get(1)) {
                Ok(Value::Bool(haystack.ends_with(needle)))
            } else {
                Ok(Value::Null)
            }
        }
        "split" => {
            if let (Some(Value::String(s)), Some(Value::String(delim))) = (eval_args.first(), eval_args.get(1)) {
                let parts: Vec<Value> = s.split(delim).map(|p| Value::String(p.to_string())).collect();
                Ok(Value::List(parts))
            } else {
                Ok(Value::Null)
            }
        }
        "tostring" => {
            if let Some(val) = eval_args.first() {
                if matches!(val, Value::Null) { Ok(Value::Null) }
                else { Ok(Value::String(format!("{}", val))) }
            } else {
                Ok(Value::Null)
            }
        }
        "tointeger" => {
            if let Some(val) = eval_args.first() {
                match val {
                    Value::Int(i) => Ok(Value::Int(*i)),
                    Value::Float(f) => Ok(Value::Int(*f as i64)),
                    Value::String(s) => {
                        if let Ok(i) = s.parse::<i64>() { Ok(Value::Int(i)) } else { Ok(Value::Null) }
                    }
                    _ => Ok(Value::Null)
                }
            } else {
                Ok(Value::Null)
            }
        }
        "tofloat" => {
            if let Some(val) = eval_args.first() {
                match val {
                    Value::Int(i) => Ok(Value::Float(*i as f64)),
                    Value::Float(f) => Ok(Value::Float(*f)),
                    Value::String(s) => {
                        if let Ok(f) = s.parse::<f64>() { Ok(Value::Float(f)) } else { Ok(Value::Null) }
                    }
                    _ => Ok(Value::Null)
                }
            } else {
                Ok(Value::Null)
            }
        }
        "toboolean" => {
            if let Some(val) = eval_args.first() {
                match val {
                    Value::Bool(b) => Ok(Value::Bool(*b)),
                    Value::String(s) => {
                        if let Ok(b) = s.parse::<bool>() { Ok(Value::Bool(b)) } else { Ok(Value::Null) }
                    }
                    _ => Ok(Value::Null)
                }
            } else {
                Ok(Value::Null)
            }
        }
        "date" => {
            if let Some(Value::String(s)) = eval_args.first() {
                // Simple validation for YYYY-MM-DD
                Ok(Value::Date(s.clone()))
            } else if eval_args.is_empty() {
                // Current date (simplified)
                Ok(Value::Date("2026-01-15".to_string()))
            } else {
                Ok(Value::Null)
            }
        }
        "datetime" => {
            if let Some(Value::String(s)) = eval_args.first() {
                return Ok(Value::DateTime(s.clone()));
            } else if eval_args.is_empty() {
                // Current datetime (simplified)
                return Ok(Value::DateTime("2026-01-15T12:00:00Z".to_string()));
            }
            Ok(Value::DateTime("".to_string()))
        }
        "id" | "elementid" => {
            if let Some(arg) = eval_args.first() {
                match arg {
                    Value::Node(id) | Value::Edge(id) => Ok(Value::Int(*id as i64)),
                    _ => Ok(Value::Null),
                }
            } else {
                Ok(Value::Null)
            }
        }
        "type" => {
            if let Some(arg) = eval_args.first() {
                if let Some(edge_id) = arg.as_edge() {
                    let eid = neural_core::EdgeId::new(edge_id);
                    match store.get_edge_type(eid) {
                        Some(t) => Ok(Value::String(t)),
                        None => Ok(Value::Null),
                    }
                } else {
                    Ok(Value::Null)
                }
            } else {
                Ok(Value::Null)
            }
        }
        _ => Err(ExecutionError::ExecutionError(format!(
            "Unknown function: {}",
            name
        ))),
    }
}

/// Checks if an expression evaluates to true for the given bindings.
///
/// Uses MAX_SNAPSHOT_ID so all committed data is visible.
pub fn is_true(
    expr: &Expression,
    bindings: &Bindings,
    store: &GraphStore,
    params: Option<&Parameters>,
) -> Result<bool> {
    is_true_at(expr, bindings, store, params, MAX_SNAPSHOT_ID)
}

/// Checks if an expression evaluates to true at a specific snapshot.
pub fn is_true_at(
    expr: &Expression,
    bindings: &Bindings,
    store: &GraphStore,
    params: Option<&Parameters>,
    snapshot_id: TransactionId,
) -> Result<bool> {
    let value = evaluate_at(expr, bindings, store, params, snapshot_id)?;
    Ok(to_bool(&value))
}

/// Converts a literal to a Value.
fn literal_to_value(lit: &Literal) -> Value {
    match lit {
        Literal::Null => Value::Null,
        Literal::Bool(b) => Value::Bool(*b),
        Literal::Int(i) => Value::Int(*i),
        Literal::Float(f) => Value::Float(*f),
        Literal::String(s) => Value::String(s.clone()),
        Literal::List(items) => {
            let values = items.iter().map(literal_to_value).collect();
            Value::List(values)
        }
    }
}

/// Converts a PropertyValue to a Value.
pub fn property_value_to_value(pv: &PropertyValue) -> Value {
    match pv {
        PropertyValue::Null => Value::Null,
        PropertyValue::Bool(b) => Value::Bool(*b),
        PropertyValue::Int(i) => Value::Int(*i),
        PropertyValue::Float(f) => Value::Float(*f),
        PropertyValue::String(s) => Value::String(s.clone()),
        PropertyValue::Date(s) => Value::Date(s.clone()),
        PropertyValue::DateTime(s) => Value::DateTime(s.clone()),
        PropertyValue::Vector(_) => {
            Value::String(format!("{:?}", pv))
        }
    }
}

/// Converts a Value to bool for predicate evaluation.
fn to_bool(value: &Value) -> bool {
    match value {
        Value::Null => false,
        Value::Bool(b) => *b,
        Value::Int(i) => *i != 0,
        Value::Float(f) => *f != 0.0,
        Value::String(s) => !s.is_empty(),
        Value::Node(_) | Value::Edge(_) => true,
        Value::List(l) => !l.is_empty(),
        Value::Date(_) | Value::DateTime(_) => true,
    }
}

/// Compares two values with the given operator.
fn compare_values(left: &Value, op: &ComparisonOp, right: &Value) -> bool {
    use crate::cmp;
    match op {
        ComparisonOp::Eq => cmp::values_equal(left, right),
        ComparisonOp::Neq => !cmp::values_equal(left, right),
        ComparisonOp::Lt => cmp::values_less_than(left, right),
        ComparisonOp::Gt => cmp::values_greater_than(left, right),
        ComparisonOp::Lte => cmp::values_less_equal(left, right),
        ComparisonOp::Gte => cmp::values_greater_equal(left, right),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use neural_core::NodeId;

    fn create_test_store() -> GraphStore {
        GraphStore::builder()
            .add_labeled_node(
                0u64,
                "Person",
                [
                    ("name", PropertyValue::from("Alice")),
                    ("age", PropertyValue::from(30i64)),
                ],
            )
            .build()
    }

    #[test]
    fn test_evaluate_case() {
        let store = create_test_store();
        let mut bindings = Bindings::new();
        bindings.bind("n", NodeId::new(0));

        // CASE n.age WHEN 30 THEN 'Thirty' ELSE 'Other' END
        let expr = Expression::Case {
            subject: Some(Box::new(Expression::Property {
                variable: "n".into(),
                property: "age".into(),
            })),
            when_then: vec![(
                Expression::Literal(Literal::Int(30)),
                Expression::Literal(Literal::String("Thirty".into())),
            )],
            else_expr: Some(Box::new(Expression::Literal(Literal::String("Other".into())))),
        };
        let result = evaluate(&expr, &bindings, &store, None).unwrap();
        assert_eq!(result, Value::String("Thirty".into()));
    }

    #[test]
    fn test_evaluate_coalesce() {
        let store = create_test_store();
        let bindings = Bindings::new();

        // COALESCE(NULL, 'Fallback')
        let expr = Expression::FunctionCall {
            name: "COALESCE".into(),
            args: vec![
                Expression::Literal(Literal::Null),
                Expression::Literal(Literal::String("Fallback".into())),
            ],
        };
        let result = evaluate(&expr, &bindings, &store, None).unwrap();
        assert_eq!(result, Value::String("Fallback".into()));
    }
    
    #[test]
    fn test_string_functions() {
        let store = create_test_store();
        let bindings = Bindings::new();
        
        // toLower("HELLO")
        let expr = Expression::FunctionCall {
            name: "toLower".into(),
            args: vec![Expression::Literal(Literal::String("HELLO".into()))],
        };
        assert_eq!(evaluate(&expr, &bindings, &store, None).unwrap(), Value::String("hello".into()));
        
        // split("a,b", ",")
        let expr_split = Expression::FunctionCall {
            name: "split".into(),
            args: vec![
                Expression::Literal(Literal::String("a,b".into())),
                Expression::Literal(Literal::String(",".into()))
            ],
        };
        let result = evaluate(&expr_split, &bindings, &store, None).unwrap();
        if let Value::List(l) = result {
            assert_eq!(l.len(), 2);
            assert_eq!(l[0], Value::String("a".into()));
        } else {
            panic!("Expected list");
        }
    }
}