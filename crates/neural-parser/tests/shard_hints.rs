//! Tests for NGQL shard hint syntax (Sprint 55).

use neural_parser::{parse_query, parse_statement};
use neural_parser::ast::Statement;

#[test]
fn test_parse_single_shard_hint() {
    let query = parse_query("MATCH (n) RETURN n USING SHARD 0").unwrap();
    assert!(query.shard_hint.is_some());
    let hint = query.shard_hint.unwrap();
    assert_eq!(hint.shards, vec![0]);
}

#[test]
fn test_parse_multi_shard_hint() {
    let query = parse_query("MATCH (n) RETURN n USING SHARD [0, 1, 2]").unwrap();
    assert!(query.shard_hint.is_some());
    let hint = query.shard_hint.unwrap();
    assert_eq!(hint.shards, vec![0, 1, 2]);
}

#[test]
fn test_parse_shard_hint_with_temporal() {
    // Shard hints can be combined with temporal clauses
    let query = parse_query("MATCH (n) RETURN n AT TIME '2026-01-15T12:00:00Z' USING SHARD 1").unwrap();
    assert!(query.temporal.is_some());
    assert!(query.shard_hint.is_some());
    assert_eq!(query.shard_hint.unwrap().shards, vec![1]);
}

#[test]
fn test_parse_query_without_shard_hint() {
    let query = parse_query("MATCH (n) RETURN n").unwrap();
    assert!(query.shard_hint.is_none());
}

#[test]
fn test_parse_statement_with_shard_hint() {
    let stmt = parse_statement("MATCH (n:Person) RETURN n.name USING SHARD [0, 3]").unwrap();
    match stmt {
        Statement::Query(query) => {
            assert!(query.shard_hint.is_some());
            assert_eq!(query.shard_hint.unwrap().shards, vec![0, 3]);
        }
        _ => panic!("Expected Query statement"),
    }
}

#[test]
fn test_shard_hint_display() {
    use neural_parser::ast::ShardHint;

    let single = ShardHint { shards: vec![5] };
    assert_eq!(format!("{}", single), "USING SHARD 5");

    let multi = ShardHint { shards: vec![1, 2, 3] };
    assert_eq!(format!("{}", multi), "USING SHARD [1, 2, 3]");
}
