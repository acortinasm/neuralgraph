//! Value comparison utilities.
//!
//! This module provides centralized value comparison functions used throughout
//! the executor for WHERE predicates, ORDER BY, and aggregations (MIN/MAX).

use crate::result::Value;
use std::cmp::Ordering;

// =============================================================================
// Ordering Comparison (for sorting)
// =============================================================================

/// Compares two values for ordering purposes (ORDER BY, sorting).
///
/// Returns `Ordering::Equal` for incompatible types.
/// Null values sort before non-null values.
pub fn compare_for_ordering(a: &Value, b: &Value) -> Ordering {
    match (a, b) {
        // Null handling
        (Value::Null, Value::Null) => Ordering::Equal,
        (Value::Null, _) => Ordering::Less, // Nulls sort first
        (_, Value::Null) => Ordering::Greater,

        // Same-type comparisons
        (Value::Bool(a), Value::Bool(b)) => a.cmp(b),
        (Value::Int(a), Value::Int(b)) => a.cmp(b),
        (Value::Float(a), Value::Float(b)) => a.partial_cmp(b).unwrap_or(Ordering::Equal),
        (Value::String(a), Value::String(b)) => a.cmp(b),
        (Value::Node(a), Value::Node(b)) => a.cmp(b),

        // Cross-type numeric comparisons
        (Value::Int(a), Value::Float(b)) => (*a as f64).partial_cmp(b).unwrap_or(Ordering::Equal),
        (Value::Float(a), Value::Int(b)) => a.partial_cmp(&(*b as f64)).unwrap_or(Ordering::Equal),

        // Incompatible types are equal
        _ => Ordering::Equal,
    }
}

/// Compares two optional values for ordering.
///
/// None values are treated as less than Some values.
pub fn compare_optional_for_ordering(a: Option<&Value>, b: Option<&Value>) -> Ordering {
    match (a, b) {
        (None, None) => Ordering::Equal,
        (None, Some(_)) => Ordering::Less,
        (Some(_), None) => Ordering::Greater,
        (Some(a), Some(b)) => compare_for_ordering(a, b),
    }
}

// =============================================================================
// Equality Comparison
// =============================================================================

/// Checks if two values are equal.
///
/// Handles cross-type numeric comparisons (Int vs Float).
pub fn values_equal(left: &Value, right: &Value) -> bool {
    match (left, right) {
        (Value::Null, Value::Null) => true,
        (Value::Bool(a), Value::Bool(b)) => a == b,
        (Value::Int(a), Value::Int(b)) => a == b,
        (Value::Float(a), Value::Float(b)) => (a - b).abs() < f64::EPSILON,
        (Value::Int(a), Value::Float(b)) | (Value::Float(b), Value::Int(a)) => {
            ((*a as f64) - b).abs() < f64::EPSILON
        }
        (Value::String(a), Value::String(b)) => a == b,
        (Value::Node(a), Value::Node(b)) => a == b,
        _ => false,
    }
}

// =============================================================================
// Relational Comparisons
// =============================================================================

/// Checks if left < right.
pub fn values_less_than(left: &Value, right: &Value) -> bool {
    compare_for_ordering(left, right) == Ordering::Less
}

/// Checks if left > right.
pub fn values_greater_than(left: &Value, right: &Value) -> bool {
    compare_for_ordering(left, right) == Ordering::Greater
}

/// Checks if left <= right.
pub fn values_less_equal(left: &Value, right: &Value) -> bool {
    !values_greater_than(left, right)
}

/// Checks if left >= right.
pub fn values_greater_equal(left: &Value, right: &Value) -> bool {
    !values_less_than(left, right)
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_compare_ints() {
        assert_eq!(
            compare_for_ordering(&Value::Int(5), &Value::Int(10)),
            Ordering::Less
        );
        assert_eq!(
            compare_for_ordering(&Value::Int(10), &Value::Int(5)),
            Ordering::Greater
        );
        assert_eq!(
            compare_for_ordering(&Value::Int(5), &Value::Int(5)),
            Ordering::Equal
        );
    }

    #[test]
    fn test_compare_floats() {
        assert_eq!(
            compare_for_ordering(&Value::Float(1.5), &Value::Float(2.5)),
            Ordering::Less
        );
    }

    #[test]
    fn test_compare_mixed_numeric() {
        assert_eq!(
            compare_for_ordering(&Value::Int(5), &Value::Float(5.0)),
            Ordering::Equal
        );
        assert!(values_less_than(&Value::Int(4), &Value::Float(5.0)));
    }

    #[test]
    fn test_compare_nulls() {
        assert_eq!(
            compare_for_ordering(&Value::Null, &Value::Null),
            Ordering::Equal
        );
        assert_eq!(
            compare_for_ordering(&Value::Null, &Value::Int(5)),
            Ordering::Less
        );
        assert_eq!(
            compare_for_ordering(&Value::Int(5), &Value::Null),
            Ordering::Greater
        );
    }

    #[test]
    fn test_values_equal() {
        assert!(values_equal(&Value::Int(42), &Value::Int(42)));
        assert!(values_equal(&Value::Int(42), &Value::Float(42.0)));
        assert!(values_equal(
            &Value::String("hello".into()),
            &Value::String("hello".into())
        ));
        assert!(!values_equal(&Value::Int(42), &Value::Int(43)));
    }

    #[test]
    fn test_relational() {
        assert!(values_less_than(&Value::Int(1), &Value::Int(2)));
        assert!(values_greater_than(&Value::Int(2), &Value::Int(1)));
        assert!(values_less_equal(&Value::Int(1), &Value::Int(1)));
        assert!(values_greater_equal(&Value::Int(1), &Value::Int(1)));
    }
}
