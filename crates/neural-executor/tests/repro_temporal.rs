use neural_executor::execute_query;
use neural_storage::GraphStore;
use neural_executor::Value;

#[test]
fn test_execute_date_function() {
    let store = GraphStore::builder().add_node(0u64, [("name", "Alice")]).build();
    let query = "MATCH (n) RETURN date('2026-01-15') AS d";
    let result = execute_query(&store, query).unwrap();
    
    assert_eq!(result.row_count(), 1);
    let val = result.rows()[0].get("d").unwrap();
    assert!(matches!(val, Value::Date(s) if s == "2026-01-15"));
}

#[test]
fn test_execute_datetime_function() {
    let store = GraphStore::builder().add_node(0u64, [("name", "Alice")]).build();
    let query = "MATCH (n) RETURN datetime('2026-01-15T12:00:00Z') AS dt";
    let result = execute_query(&store, query).unwrap();
    
    assert_eq!(result.row_count(), 1);
    let val = result.rows()[0].get("dt").unwrap();
    assert!(matches!(val, Value::DateTime(s) if s == "2026-01-15T12:00:00Z"));
}
