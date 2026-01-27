//! Tests for time-travel query parsing (Sprint 54)

use neural_parser::{parse_statement, parse_query};
use neural_parser::{Statement, Expression, Literal};

#[test]
fn test_parse_at_time_clause() {
    let query = "MATCH (n:Person) RETURN n AT TIME '2026-01-15T12:00:00Z'";
    let result = parse_query(query).unwrap();

    assert!(result.temporal.is_some());
    let temporal = result.temporal.unwrap();

    // Check that timestamp is a string literal
    match temporal.timestamp {
        Expression::Literal(Literal::String(s)) => {
            assert_eq!(s, "2026-01-15T12:00:00Z");
        }
        _ => panic!("Expected string literal for timestamp"),
    }
}

#[test]
fn test_parse_at_timestamp_clause() {
    let query = "MATCH (n) RETURN n.name AT TIMESTAMP '2026-01-14'";
    let result = parse_query(query).unwrap();

    assert!(result.temporal.is_some());
}

#[test]
fn test_parse_query_without_temporal() {
    let query = "MATCH (n:Person) RETURN n";
    let result = parse_query(query).unwrap();

    assert!(result.temporal.is_none());
}

#[test]
fn test_parse_flashback_statement() {
    let stmt = "FLASHBACK TO '2026-01-15T00:00:00Z'";
    let result = parse_statement(stmt).unwrap();

    match result {
        Statement::Flashback { timestamp } => {
            match timestamp {
                Expression::Literal(Literal::String(s)) => {
                    assert_eq!(s, "2026-01-15T00:00:00Z");
                }
                _ => panic!("Expected string literal for flashback timestamp"),
            }
        }
        _ => panic!("Expected Flashback statement"),
    }
}

#[test]
fn test_parse_at_time_with_datetime_function() {
    // Also support datetime() function in AT TIME clause
    let query = "MATCH (n) RETURN n AT TIME datetime('2026-01-15T12:00:00Z')";
    let result = parse_query(query).unwrap();

    assert!(result.temporal.is_some());
    let temporal = result.temporal.unwrap();

    // Check that it's a function call
    match temporal.timestamp {
        Expression::FunctionCall { name, args } => {
            assert_eq!(name.to_lowercase(), "datetime");
            assert_eq!(args.len(), 1);
        }
        _ => panic!("Expected function call for timestamp"),
    }
}
