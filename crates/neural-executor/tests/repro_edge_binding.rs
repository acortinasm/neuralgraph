use neural_executor::{execute_statement, StatementResult};
use neural_storage::GraphStore;
use neural_core::PropertyValue;
use neural_executor::result::Value;

#[test]
fn test_edge_binding_and_id() {
    let mut store = GraphStore::builder()
        .add_node(0u64, vec![("name".to_string(), PropertyValue::from("A"))])
        .add_node(1u64, vec![("name".to_string(), PropertyValue::from("B"))])
        .add_labeled_edge(0u64, 1u64, neural_core::Label::new("KNOWS"))
        .build();

    // The first edge in CSR should have ID 0
    let query = "MATCH (a)-[r:KNOWS]->(b) RETURN id(r) AS edge_id";
    let result = execute_statement(&mut store, query).unwrap();

    if let StatementResult::Query(qr) = result {
        assert_eq!(qr.row_count(), 1);
        let row = qr.rows().get(0).unwrap();
        let id_val = row.get("edge_id").unwrap();
        assert_eq!(*id_val, Value::Int(0));
    } else {
        panic!("Expected a query result");
    }
}

#[test]
fn test_edge_binding_multiple() {
     let mut store = GraphStore::builder()
        .add_node(0u64, Vec::<(String, PropertyValue)>::new())
        .add_node(1u64, Vec::<(String, PropertyValue)>::new())
        .add_labeled_edge(0u64, 1u64, neural_core::Label::new("KNOWS"))
        .add_labeled_edge(0u64, 1u64, neural_core::Label::new("LIKES"))
        .build();

    let query = "MATCH (a)-[r]->(b) RETURN id(r) AS eid ORDER BY eid";
    let result = execute_statement(&mut store, query).unwrap();

    if let StatementResult::Query(qr) = result {
        assert_eq!(qr.row_count(), 2);
        assert_eq!(*qr.rows()[0].get("eid").unwrap(), Value::Int(0));
        assert_eq!(*qr.rows()[1].get("eid").unwrap(), Value::Int(1));
    } else {
        panic!("Expected a query result");
    }
}
