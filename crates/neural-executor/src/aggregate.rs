//! Aggregation execution.
//!
//! Implements COUNT, SUM, AVG, MIN, MAX, COLLECT aggregation functions.

use crate::result::{Bindings, Value};
use crate::{ExecutionError, Result, eval};
use neural_parser::{AggregateFunction, Expression};
use neural_storage::GraphStore;
use std::collections::HashSet;

/// Computes an aggregate value from a list of bindings.
pub fn compute_aggregate(
    function: &AggregateFunction,
    argument: &Option<Box<Expression>>,
    distinct: bool,
    bindings: &[Bindings],
    store: &GraphStore,
    params: Option<&eval::Parameters>,
) -> Result<Value> {
    match function {
        AggregateFunction::Count => compute_count(argument, distinct, bindings, store, params),
        AggregateFunction::Sum => compute_sum(argument, distinct, bindings, store, params),
        AggregateFunction::Avg => compute_avg(argument, distinct, bindings, store, params),
        AggregateFunction::Min => compute_min(argument, bindings, store, params),
        AggregateFunction::Max => compute_max(argument, bindings, store, params),
        AggregateFunction::Collect => compute_collect(argument, distinct, bindings, store, params),
    }
}

/// COUNT(*) or COUNT(expr)
fn compute_count(
    argument: &Option<Box<Expression>>,
    distinct: bool,
    bindings: &[Bindings],
    store: &GraphStore,
    params: Option<&eval::Parameters>,
) -> Result<Value> {
    match argument {
        None => {
            // COUNT(*) - count all rows
            Ok(Value::Int(bindings.len() as i64))
        }
        Some(expr) => {
            if distinct {
                // COUNT(DISTINCT expr)
                let mut seen: HashSet<String> = HashSet::new();
                for binding in bindings {
                    if let Ok(val) = eval::evaluate(expr, binding, store, params) {
                        if !matches!(val, Value::Null) {
                            seen.insert(format!("{:?}", val));
                        }
                    }
                }
                Ok(Value::Int(seen.len() as i64))
            } else {
                // COUNT(expr) - count non-null values
                let mut count = 0i64;
                for binding in bindings {
                    if let Ok(val) = eval::evaluate(expr, binding, store, params) {
                        if !matches!(val, Value::Null) {
                            count += 1;
                        }
                    }
                }
                Ok(Value::Int(count))
            }
        }
    }
}

/// SUM(expr)
fn compute_sum(
    argument: &Option<Box<Expression>>,
    _distinct: bool,
    bindings: &[Bindings],
    store: &GraphStore,
    params: Option<&eval::Parameters>,
) -> Result<Value> {
    let expr = argument
        .as_ref()
        .ok_or_else(|| ExecutionError::ExecutionError("SUM requires an argument".into()))?;

    let mut sum = 0.0f64;
    let mut has_values = false;

    for binding in bindings {
        if let Ok(val) = eval::evaluate(expr, binding, store, params) {
            match val {
                Value::Int(i) => {
                    sum += i as f64;
                    has_values = true;
                }
                Value::Float(f) => {
                    sum += f;
                    has_values = true;
                }
                _ => {}
            }
        }
    }

    if has_values {
        Ok(Value::Float(sum))
    } else {
        Ok(Value::Null)
    }
}

/// AVG(expr)
fn compute_avg(
    argument: &Option<Box<Expression>>,
    _distinct: bool,
    bindings: &[Bindings],
    store: &GraphStore,
    params: Option<&eval::Parameters>,
) -> Result<Value> {
    let expr = argument
        .as_ref()
        .ok_or_else(|| ExecutionError::ExecutionError("AVG requires an argument".into()))?;

    let mut sum = 0.0f64;
    let mut count = 0i64;

    for binding in bindings {
        if let Ok(val) = eval::evaluate(expr, binding, store, params) {
            match val {
                Value::Int(i) => {
                    sum += i as f64;
                    count += 1;
                }
                Value::Float(f) => {
                    sum += f;
                    count += 1;
                }
                _ => {}
            }
        }
    }

    if count > 0 {
        Ok(Value::Float(sum / count as f64))
    } else {
        Ok(Value::Null)
    }
}

/// MIN(expr)
fn compute_min(
    argument: &Option<Box<Expression>>,
    bindings: &[Bindings],
    store: &GraphStore,
    params: Option<&eval::Parameters>,
) -> Result<Value> {
    let expr = argument
        .as_ref()
        .ok_or_else(|| ExecutionError::ExecutionError("MIN requires an argument".into()))?;

    let mut min_val: Option<Value> = None;

    for binding in bindings {
        if let Ok(val) = eval::evaluate(expr, binding, store, params) {
            if matches!(val, Value::Null) {
                continue;
            }
            min_val = Some(match min_val {
                None => val,
                Some(current) => {
                    if crate::cmp::values_less_than(&val, &current) {
                        val
                    } else {
                        current
                    }
                }
            });
        }
    }

    Ok(min_val.unwrap_or(Value::Null))
}

/// MAX(expr)
fn compute_max(
    argument: &Option<Box<Expression>>,
    bindings: &[Bindings],
    store: &GraphStore,
    params: Option<&eval::Parameters>,
) -> Result<Value> {
    let expr = argument
        .as_ref()
        .ok_or_else(|| ExecutionError::ExecutionError("MAX requires an argument".into()))?;

    let mut max_val: Option<Value> = None;

    for binding in bindings {
        if let Ok(val) = eval::evaluate(expr, binding, store, params) {
            if matches!(val, Value::Null) {
                continue;
            }
            max_val = Some(match max_val {
                None => val,
                Some(current) => {
                    if crate::cmp::values_greater_than(&val, &current) {
                        val
                    } else {
                        current
                    }
                }
            });
        }
    }

    Ok(max_val.unwrap_or(Value::Null))
}

/// COLLECT(expr) - collect values into a list
fn compute_collect(
    argument: &Option<Box<Expression>>,
    distinct: bool,
    bindings: &[Bindings],
    store: &GraphStore,
    params: Option<&eval::Parameters>,
) -> Result<Value> {
    let expr = argument
        .as_ref()
        .ok_or_else(|| ExecutionError::ExecutionError("COLLECT requires an argument".into()))?;

    let mut values = Vec::new();
    let mut seen: HashSet<String> = HashSet::new(); // Use string repr for deduplication for now

    for binding in bindings {
        if let Ok(val) = eval::evaluate(expr, binding, store, params) {
            if matches!(val, Value::Null) {
                continue;
            }
            if distinct {
                let key = format!("{:?}", val);
                if !seen.contains(&key) {
                    seen.insert(key);
                    values.push(val);
                }
            } else {
                values.push(val);
            }
        }
    }

    Ok(Value::List(values))
}

// Comparison functions are now centralized in crate::cmp

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use neural_core::PropertyValue;

    fn create_test_store() -> GraphStore {
        GraphStore::builder()
            .add_node(0u64, [("age", PropertyValue::from(30i64))])
            .add_node(1u64, [("age", PropertyValue::from(25i64))])
            .add_node(2u64, [("age", PropertyValue::from(35i64))])
            .build()
    }

    fn create_bindings_for_nodes() -> Vec<Bindings> {
        vec![
            {
                let mut b = Bindings::new();
                b.bind("n", neural_core::NodeId::new(0));
                b
            },
            {
                let mut b = Bindings::new();
                b.bind("n", neural_core::NodeId::new(1));
                b
            },
            {
                let mut b = Bindings::new();
                b.bind("n", neural_core::NodeId::new(2));
                b
            },
        ]
    }

    #[test]
    fn test_count_star() {
        let store = create_test_store();
        let bindings = create_bindings_for_nodes();

        let result =
            compute_aggregate(&AggregateFunction::Count, &None, false, &bindings, &store, None).unwrap();

        assert_eq!(result, Value::Int(3));
    }

    #[test]
    fn test_count_expr() {
        let store = create_test_store();
        let bindings = create_bindings_for_nodes();

        let expr = Expression::Property {
            variable: "n".into(),
            property: "age".into(),
        };

        let result = compute_aggregate(
            &AggregateFunction::Count,
            &Some(Box::new(expr)),
            false,
            &bindings,
            &store,
            None,
        )
        .unwrap();

        assert_eq!(result, Value::Int(3));
    }

    #[test]
    fn test_sum() {
        let store = create_test_store();
        let bindings = create_bindings_for_nodes();

        let expr = Expression::Property {
            variable: "n".into(),
            property: "age".into(),
        };

        let result = compute_aggregate(
            &AggregateFunction::Sum,
            &Some(Box::new(expr)),
            false,
            &bindings,
            &store,
            None,
        )
        .unwrap();

        assert_eq!(result, Value::Float(90.0)); // 30 + 25 + 35
    }

    #[test]
    fn test_avg() {
        let store = create_test_store();
        let bindings = create_bindings_for_nodes();

        let expr = Expression::Property {
            variable: "n".into(),
            property: "age".into(),
        };

        let result = compute_aggregate(
            &AggregateFunction::Avg,
            &Some(Box::new(expr)),
            false,
            &bindings,
            &store,
            None,
        )
        .unwrap();

        assert_eq!(result, Value::Float(30.0)); // (30 + 25 + 35) / 3
    }

    #[test]
    fn test_min() {
        let store = create_test_store();
        let bindings = create_bindings_for_nodes();

        let expr = Expression::Property {
            variable: "n".into(),
            property: "age".into(),
        };

        let result = compute_aggregate(
            &AggregateFunction::Min,
            &Some(Box::new(expr)),
            false,
            &bindings,
            &store,
            None,
        )
        .unwrap();

        assert_eq!(result, Value::Int(25));
    }

    #[test]
    fn test_max() {
        let store = create_test_store();
        let bindings = create_bindings_for_nodes();

        let expr = Expression::Property {
            variable: "n".into(),
            property: "age".into(),
        };

        let result = compute_aggregate(
            &AggregateFunction::Max,
            &Some(Box::new(expr)),
            false,
            &bindings,
            &store,
            None,
        )
        .unwrap();

        assert_eq!(result, Value::Int(35));
    }
}
