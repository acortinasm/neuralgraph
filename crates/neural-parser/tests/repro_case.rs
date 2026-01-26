use neural_parser::parse_query;

#[test]
fn test_parse_case() {
    let query = "MATCH (n) RETURN CASE WHEN n.age > 18 THEN 'Adult' ELSE 'Minor' END";
    let result = parse_query(query);
    assert!(result.is_ok(), "Should parse CASE expression");
}

#[test]
fn test_parse_coalesce() {
    let query = "MATCH (n) RETURN COALESCE(n.nickname, n.name) AS name";
    let result = parse_query(query);
    assert!(result.is_ok(), "Should parse COALESCE expression");
}

#[test]
fn test_parse_string_functions() {
    let query = "MATCH (n) WHERE toLower(n.name) = 'alice' RETURN split(n.email, '@')";
    let result = parse_query(query);
    assert!(result.is_ok(), "Should parse string functions");
}
