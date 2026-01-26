use neural_executor::{execute_statement};
use neural_storage::GraphStore;
use neural_core::PropertyValue;
use neural_executor::result::Value;

#[test]
fn test_shortest_path_one_word() {
    let mut store = GraphStore::builder().build();
    // Test one word syntax
    let query = "MATCH p = shortestPath((a)-[*]->(b)) RETURN p";
    let result = execute_statement(&mut store, query);
    assert!(result.is_ok(), "Should support shortestPath (one word)");
}

#[test]
fn test_shortest_path_binding() {
    let mut store = GraphStore::builder()
        .add_node(0u64, vec![("name".to_string(), PropertyValue::from("A"))])
        .add_node(1u64, vec![("name".to_string(), PropertyValue::from("B"))])
        .add_node(2u64, vec![("name".to_string(), PropertyValue::from("C"))])
        .add_labeled_edge(0u64, 1u64, neural_core::Label::new("KNOWS"))
        .add_labeled_edge(1u64, 2u64, neural_core::Label::new("KNOWS"))
        .build();

    let query = "MATCH p = shortestPath((a)-[:KNOWS*]->(c)) WHERE id(a)=0 AND id(c)=2 RETURN p";
    let result = neural_executor::execute_statement(&mut store, query).unwrap();

    if let neural_executor::StatementResult::Query(qr) = result {
        assert_eq!(qr.row_count(), 1);
        let row = qr.rows().get(0).unwrap();
        let p_val = row.get("p").unwrap();
        if let neural_executor::result::Value::List(l) = p_val {
            assert_eq!(l.len(), 3); // A, B, C
            assert_eq!(l[0], neural_executor::result::Value::Node(0));
            assert_eq!(l[1], neural_executor::result::Value::Node(1));
            assert_eq!(l[2], neural_executor::result::Value::Node(2));
        } else {
            panic!("Expected p to be a List, found {:?}", p_val);
        }
    } else {
        panic!("Expected a query result");
    }
}
