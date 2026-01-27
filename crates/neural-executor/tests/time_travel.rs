//! Tests for time-travel query execution (Sprint 54)

use neural_executor::{execute_query, Value};
use neural_storage::GraphStore;

#[test]
fn test_time_travel_query_parsing() {
    // Test that AT TIME queries parse correctly
    let store = GraphStore::builder()
        .add_labeled_node(0u64, "Person", [("name", "Alice")])
        .build();

    // This should parse but may fail execution if no timestamps recorded
    let query = "MATCH (n:Person) RETURN n.name AT TIME '2026-01-15T12:00:00Z'";
    let result = execute_query(&store, query);

    // Will error because no timestamps exist in the index
    assert!(result.is_err());

    // Error message should mention timestamp
    let err = result.unwrap_err().to_string();
    assert!(err.contains("timestamp") || err.contains("No transactions"));
}

#[test]
fn test_time_travel_with_recorded_timestamp() {
    let mut store = GraphStore::builder()
        .add_labeled_node(0u64, "Person", [("name", "Alice")])
        .build();

    // Manually record a timestamp for transaction 1 (the builder's tx)
    store.record_commit_timestamp("2026-01-15T10:00:00Z".to_string(), 1);

    // Now query at that timestamp
    let query = "MATCH (n:Person) RETURN n.name AT TIME '2026-01-15T12:00:00Z'";
    let result = execute_query(&store, query);

    // Should succeed since we have a timestamp at or before the query time
    assert!(result.is_ok());
    let result = result.unwrap();
    assert_eq!(result.row_count(), 1);

    let name = result.rows()[0].get("n.name").unwrap();
    assert!(matches!(name, Value::String(s) if s == "Alice"));
}

#[test]
fn test_query_without_temporal_clause() {
    let store = GraphStore::builder()
        .add_labeled_node(0u64, "Person", [("name", "Alice")])
        .add_labeled_node(1u64, "Person", [("name", "Bob")])
        .build();

    // Regular query without AT TIME
    let query = "MATCH (n:Person) RETURN n.name ORDER BY n.name";
    let result = execute_query(&store, query).unwrap();

    assert_eq!(result.row_count(), 2);
}
