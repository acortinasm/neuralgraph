//! Integration tests for Array/Map data types (Sprint 64)

use neural_core::PropertyValue;
use neural_executor::{execute_statement, StatementResult};
use neural_storage::GraphStore;
use neural_executor::result::Value;
use std::collections::HashMap;

#[test]
fn test_create_node_with_array_property() {
    let mut store = GraphStore::builder().build();

    // Create a node with an array property
    let query = r#"CREATE (n:Person {name: "Alice", tags: ["rust", "graph", "database"]})"#;
    let result = execute_statement(&mut store, query);
    assert!(result.is_ok(), "CREATE with array property failed: {:?}", result.err());

    // Verify the node was created with array property
    let query = "MATCH (n:Person) RETURN n.name AS name, n.tags AS tags";
    let result = execute_statement(&mut store, query);

    if let Ok(StatementResult::Query(qr)) = result {
        assert_eq!(qr.row_count(), 1);
        let row = qr.rows().get(0).unwrap();
        assert_eq!(row.get("name").unwrap(), &Value::String("Alice".to_string()));

        // Tags should be a list
        if let Value::List(tags) = row.get("tags").unwrap() {
            assert_eq!(tags.len(), 3);
            assert_eq!(tags[0], Value::String("rust".to_string()));
            assert_eq!(tags[1], Value::String("graph".to_string()));
            assert_eq!(tags[2], Value::String("database".to_string()));
        } else {
            panic!("Expected tags to be a List, got: {:?}", row.get("tags"));
        }
    } else {
        panic!("Expected Query result");
    }
}

#[test]
fn test_create_node_with_numeric_array() {
    let mut store = GraphStore::builder().build();

    // Numeric arrays become Vectors (for embeddings)
    let query = r#"CREATE (n:Doc {embedding: [0.1, 0.2, 0.3, 0.4]})"#;
    let result = execute_statement(&mut store, query);
    assert!(result.is_ok(), "CREATE with numeric array failed: {:?}", result.err());

    // Verify the node was created
    let query = "MATCH (n:Doc) RETURN n.embedding AS embedding";
    let result = execute_statement(&mut store, query);

    if let Ok(StatementResult::Query(qr)) = result {
        assert_eq!(qr.row_count(), 1);
        let row = qr.rows().get(0).unwrap();

        // Numeric arrays become Vector, which becomes List when returned
        if let Value::List(vec) = row.get("embedding").unwrap() {
            assert_eq!(vec.len(), 4);
        } else {
            panic!("Expected embedding to be a List, got: {:?}", row.get("embedding"));
        }
    } else {
        panic!("Expected Query result");
    }
}

#[test]
fn test_create_node_with_map_property() {
    let mut store = GraphStore::builder().build();

    // Create a node with a map property
    let query = r#"CREATE (n:Config {name: "settings", options: {debug: true, level: 5, mode: "fast"}})"#;
    let result = execute_statement(&mut store, query);
    assert!(result.is_ok(), "CREATE with map property failed: {:?}", result.err());

    // Verify the node was created with map property
    let query = "MATCH (n:Config) RETURN n.name AS name, n.options AS options";
    let result = execute_statement(&mut store, query);

    if let Ok(StatementResult::Query(qr)) = result {
        assert_eq!(qr.row_count(), 1);
        let row = qr.rows().get(0).unwrap();
        assert_eq!(row.get("name").unwrap(), &Value::String("settings".to_string()));

        // Options should be a map
        if let Value::Map(opts) = row.get("options").unwrap() {
            assert_eq!(opts.len(), 3);
            assert_eq!(opts.get("debug").unwrap(), &Value::Bool(true));
            assert_eq!(opts.get("level").unwrap(), &Value::Int(5));
            assert_eq!(opts.get("mode").unwrap(), &Value::String("fast".to_string()));
        } else {
            panic!("Expected options to be a Map, got: {:?}", row.get("options"));
        }
    } else {
        panic!("Expected Query result");
    }
}

#[test]
fn test_create_node_with_nested_structures() {
    let mut store = GraphStore::builder().build();

    // Create a node with nested array in map
    // Note: Using Person label instead of Profile to avoid potential keyword conflicts
    let query = r#"CREATE (n:Person {name: "Bob", metadata: {scores: [100, 95, 88], active: true}})"#;
    let result = execute_statement(&mut store, query);
    assert!(result.is_ok(), "CREATE with nested structure failed: {:?}", result.err());

    // Verify
    let query = "MATCH (n:Person) WHERE n.name = 'Bob' RETURN n.metadata AS metadata";
    let result = execute_statement(&mut store, query);

    if let Ok(StatementResult::Query(qr)) = result {
        assert_eq!(qr.row_count(), 1);
        let row = qr.rows().get(0).unwrap();

        if let Value::Map(meta) = row.get("metadata").unwrap() {
            assert_eq!(meta.get("active").unwrap(), &Value::Bool(true));
            // scores array within the map - numeric arrays become vectors
            if let Value::List(scores) = meta.get("scores").unwrap() {
                assert_eq!(scores.len(), 3);
            } else {
                panic!("Expected scores to be a List");
            }
        } else {
            panic!("Expected metadata to be a Map");
        }
    } else {
        panic!("Expected Query result");
    }
}

#[test]
fn test_create_node_with_mixed_array() {
    let mut store = GraphStore::builder().build();

    // Mixed arrays (non-numeric) should stay as arrays
    let query = r#"CREATE (n:Item {data: ["text", 123, true, null]})"#;
    let result = execute_statement(&mut store, query);
    assert!(result.is_ok(), "CREATE with mixed array failed: {:?}", result.err());

    // Verify
    let query = "MATCH (n:Item) RETURN n.data AS data";
    let result = execute_statement(&mut store, query);

    if let Ok(StatementResult::Query(qr)) = result {
        assert_eq!(qr.row_count(), 1);
        let row = qr.rows().get(0).unwrap();

        if let Value::List(data) = row.get("data").unwrap() {
            assert_eq!(data.len(), 4);
            assert_eq!(data[0], Value::String("text".to_string()));
            assert_eq!(data[1], Value::Int(123));
            assert_eq!(data[2], Value::Bool(true));
            assert_eq!(data[3], Value::Null);
        } else {
            panic!("Expected data to be a List, got: {:?}", row.get("data"));
        }
    } else {
        panic!("Expected Query result");
    }
}

#[test]
fn test_empty_array() {
    let mut store = GraphStore::builder().build();

    let query = r#"CREATE (n:Empty {items: []})"#;
    let result = execute_statement(&mut store, query);
    assert!(result.is_ok(), "CREATE with empty array failed: {:?}", result.err());

    let query = "MATCH (n:Empty) RETURN n.items AS items";
    let result = execute_statement(&mut store, query);

    if let Ok(StatementResult::Query(qr)) = result {
        assert_eq!(qr.row_count(), 1);
        let row = qr.rows().get(0).unwrap();

        if let Value::List(items) = row.get("items").unwrap() {
            assert!(items.is_empty());
        } else {
            panic!("Expected items to be an empty List");
        }
    } else {
        panic!("Expected Query result");
    }
}

#[test]
fn test_empty_map() {
    let mut store = GraphStore::builder().build();

    let query = r#"CREATE (n:Empty {config: {}})"#;
    let result = execute_statement(&mut store, query);
    assert!(result.is_ok(), "CREATE with empty map failed: {:?}", result.err());

    let query = "MATCH (n:Empty) RETURN n.config AS config";
    let result = execute_statement(&mut store, query);

    if let Ok(StatementResult::Query(qr)) = result {
        assert_eq!(qr.row_count(), 1);
        let row = qr.rows().get(0).unwrap();

        if let Value::Map(config) = row.get("config").unwrap() {
            assert!(config.is_empty());
        } else {
            panic!("Expected config to be an empty Map");
        }
    } else {
        panic!("Expected Query result");
    }
}

#[test]
fn test_property_value_array_direct() {
    // Test PropertyValue::Array directly
    let arr = PropertyValue::Array(vec![
        PropertyValue::String("a".to_string()),
        PropertyValue::Int(1),
        PropertyValue::Bool(true),
    ]);

    assert!(arr.is_array());
    assert!(!arr.is_map());

    let items = arr.as_array().unwrap();
    assert_eq!(items.len(), 3);
    assert_eq!(items[0].as_str(), Some("a"));
    assert_eq!(items[1].as_int(), Some(1));
    assert_eq!(items[2].as_bool(), Some(true));

    // Test index access
    assert_eq!(arr.get_index(0).unwrap().as_str(), Some("a"));
    assert!(arr.get_index(10).is_none());
}

#[test]
fn test_property_value_map_direct() {
    // Test PropertyValue::Map directly
    let mut map = HashMap::new();
    map.insert("name".to_string(), PropertyValue::String("test".to_string()));
    map.insert("count".to_string(), PropertyValue::Int(42));

    let map_val = PropertyValue::Map(map);

    assert!(map_val.is_map());
    assert!(!map_val.is_array());

    let m = map_val.as_map().unwrap();
    assert_eq!(m.len(), 2);

    // Test key access
    assert_eq!(map_val.get("name").unwrap().as_str(), Some("test"));
    assert_eq!(map_val.get("count").unwrap().as_int(), Some(42));
    assert!(map_val.get("nonexistent").is_none());
}
