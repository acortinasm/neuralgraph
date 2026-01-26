use neural_core::PropertyValue;
use neural_executor::{execute_statement, StatementResult};
use neural_storage::GraphStore;
use neural_executor::result::Value;

#[test]
fn test_id_as_variable_name() {
    let mut store = GraphStore::builder()
        .add_node(0u64, vec![("name".to_string(), PropertyValue::from("Alice"))])
        .build();

    // Using 'id' as a variable name should work
    let query = "MATCH (id) RETURN id.name AS name";
    let result = execute_statement(&mut store, query);
    
    assert!(result.is_ok(), "Query with 'id' as variable should be valid, but got: {:?}", result.err());
    
    if let Ok(StatementResult::Query(qr)) = result {
        assert_eq!(qr.row_count(), 1);
        let row = qr.rows().get(0).unwrap();
        assert_eq!(row.get("name").unwrap(), &Value::String("Alice".to_string()));
    }
}

#[test]
fn test_keywords_as_variable_names() {
    let mut store = GraphStore::builder()
        .add_node(0u64, vec![("name".to_string(), PropertyValue::from("Alice"))])
        .build();

    // 'count' is a keyword, but should be allowed as a variable
    let query = "MATCH (count) RETURN count.name AS name";
    let result = execute_statement(&mut store, query);
    assert!(result.is_ok(), "Query with 'count' as variable should be valid, but got: {:?}", result.err());
}

#[test]
fn test_shortest_path_variants() {
    let mut store = GraphStore::builder()
        .add_labeled_node(0u64, "Person", vec![("name".to_string(), PropertyValue::from("Alice"))])
        .add_labeled_node(1u64, "Person", vec![("name".to_string(), PropertyValue::from("Bob"))])
        .add_labeled_edge(0u64, 1u64, neural_core::Label::new("KNOWS"))
        .build();

    // 1. shortestPath((a)-[*]->(b))
    let query1 = "MATCH p = shortestPath((a)-[*]->(b)) RETURN p";
    assert!(execute_statement(&mut store, query1).is_ok());

    // 2. SHORTEST PATH (a)-[*]->(b)
    let query2 = "MATCH SHORTEST PATH (a)-[*]->(b) RETURN a";
    assert!(execute_statement(&mut store, query2).is_ok(), "SHORTEST PATH (spaced) should be valid");
}

#[test]
fn test_type_function() {
    let mut store = GraphStore::builder()
        .add_node(0u64, Vec::<(String, PropertyValue)>::new())
        .add_node(1u64, Vec::<(String, PropertyValue)>::new())
        .add_labeled_edge(0u64, 1u64, neural_core::Label::new("FRIEND"))
        .build();

    let query = "MATCH (a)-[r]->(b) RETURN type(r) AS rel_type";
    let result = execute_statement(&mut store, query).unwrap();

    if let StatementResult::Query(qr) = result {
        assert_eq!(qr.row_count(), 1);
        let row = qr.rows().get(0).unwrap();
        assert_eq!(row.get("rel_type").unwrap(), &Value::String("FRIEND".to_string()));
    } else {
        panic!("Expected query result");
    }
}

#[test]
fn test_incoming_edge_ids() {
    let mut store = GraphStore::builder()
        .add_node(0u64, Vec::<(String, PropertyValue)>::new())
        .add_node(1u64, Vec::<(String, PropertyValue)>::new())
        .add_labeled_edge(0u64, 1u64, neural_core::Label::new("KNOWS"))
        .build();

    // MATCH (a)<-[r]-(b)
    let query = "MATCH (a)<-[r]-(b) WHERE id(a) = 1 RETURN id(r) AS eid";
    let result = execute_statement(&mut store, query).unwrap();

    if let StatementResult::Query(qr) = result {
        assert_eq!(qr.row_count(), 1);
        let row = qr.rows().get(0).unwrap();
        // Edge ID for the first edge should be 0 (CSR)
        assert_eq!(row.get("eid").unwrap(), &Value::Int(0));
    } else {
        panic!("Expected query result");
    }
}

#[test]
fn test_id_as_property_key() {
    let mut store = GraphStore::builder()
        .add_node(0u64, vec![("id".to_string(), PropertyValue::from(100i64))])
        .build();

    // id: 100 in the pattern
    let query = "MATCH (n {id: 100}) RETURN n.id AS val";
    let result = execute_statement(&mut store, query).unwrap();

    if let StatementResult::Query(qr) = result {
        assert_eq!(qr.row_count(), 1);
        let row = qr.rows().get(0).unwrap();
        assert_eq!(row.get("val").unwrap(), &Value::Int(100));
    } else {
        panic!("Expected query result");
    }
}
