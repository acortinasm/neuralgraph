use neural_executor::{execute_statement, StatementResult};
use neural_storage::GraphStore;
use neural_core::PropertyValue;
use neural_executor::result::Value;

#[test]
fn test_node_dot_id() {
    let mut store = GraphStore::builder()
        .add_labeled_node(42u64, "Person", vec![("name".to_string(), PropertyValue::from("Alice"))])
        .build();

    let query = "MATCH (p:Person) RETURN p.id AS person_id";
    let result = execute_statement(&mut store, query).unwrap();

    if let StatementResult::Query(qr) = result {
        assert_eq!(qr.row_count(), 1);
        let row = qr.rows().get(0).unwrap();
        let id_val = row.get("person_id").unwrap();
        assert_eq!(*id_val, Value::Int(42));
    } else {
        panic!("Expected a query result");
    }
}

#[test]
fn test_node_dot_id_in_where() {
    let mut store = GraphStore::builder()
        .add_labeled_node(42u64, "Person", Vec::<(String, PropertyValue)>::new())
        .build();

    let query = "MATCH (p:Person) WHERE p.id = 42 RETURN p.id";
    let result = execute_statement(&mut store, query).unwrap();

    if let StatementResult::Query(qr) = result {
        assert_eq!(qr.row_count(), 1);
    } else {
        panic!("Expected a query result");
    }
}
