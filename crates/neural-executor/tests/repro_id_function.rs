use neural_core::PropertyValue;
use neural_executor::{execute_statement, StatementResult};
use neural_storage::GraphStore;
use neural_executor::result::Value;

#[test]
fn test_id_function_in_return() {
    let mut store = GraphStore::builder()
        .add_labeled_node(0u64, "Person", vec![("name".to_string(), PropertyValue::from("Alice"))])
        .build();

    let query = "MATCH (p:Person) WHERE p.name = 'Alice' RETURN id(p) AS person_id";
    let result = execute_statement(&mut store, query).unwrap();

    match result {
        StatementResult::Query(qr) => {
            assert_eq!(qr.row_count(), 1);
            let row = qr.rows().get(0).unwrap();
            let id_val = row.get("person_id").unwrap();
            assert_eq!(*id_val, Value::Int(0));
        }
        _ => panic!("Expected a query result"),
    }
}

#[test]fn test_id_function_in_where() {
    let mut store = GraphStore::builder()
        .add_labeled_node(0u64, "Person", vec![("name".to_string(), PropertyValue::from("Alice"))])
        .add_labeled_node(1u64, "Person", vec![("name".to_string(), PropertyValue::from("Bob"))])
        .build();

    let query = "MATCH (p:Person) WHERE id(p) = 1 RETURN p.name AS person_name";
    let result = execute_statement(&mut store, query).unwrap();

    match result {
        StatementResult::Query(qr) => {
            assert_eq!(qr.row_count(), 1);
            let row = qr.rows().get(0).unwrap();
            let name_val = row.get("person_name").unwrap();
            assert_eq!(*name_val, Value::String("Bob".to_string()));
        }
        _ => panic!("Expected a query result"),
    }
}
