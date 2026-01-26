use neural_parser::parse_query;

#[test]
fn test_parse_with_clause() {
    let query = "MATCH (n:Person) WITH n.name AS name RETURN name";
    let result = parse_query(query);
    assert!(result.is_ok(), "Should parse WITH clause");
}

#[test]
fn test_parse_unwind_clause() {
    let query = "UNWIND $list AS x RETURN x";
    let result = parse_query(query);
    assert!(result.is_ok(), "Should parse UNWIND clause");
}

#[test]
fn test_parse_optional_match() {
    let query = "MATCH (a) OPTIONAL MATCH (a)-[:KNOWS]->(b) RETURN a, b";
    let result = parse_query(query);
    assert!(result.is_ok(), "Should parse OPTIONAL MATCH clause");
}

#[test]
fn test_parse_merge() {
    let query = "MERGE (n:Person {name: 'Alice'}) RETURN n";
    let result = parse_query(query);
    assert!(result.is_ok(), "Should parse MERGE clause");
}
