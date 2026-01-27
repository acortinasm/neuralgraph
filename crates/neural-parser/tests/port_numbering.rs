//! Tests for port numbering syntax (Sprint 57)

use neural_parser::{parse_query, parse_statement, Clause, Statement};

#[test]
fn test_parse_rel_pattern_with_port() {
    let query = parse_query("MATCH (a)-[:DEPENDS_ON:1]->(b) RETURN a, b").unwrap();

    if let Clause::Match(m) = &query.clauses[0] {
        let pattern = &m.patterns[0];
        let (rel, _) = &pattern.chain[0];
        assert_eq!(rel.label.as_deref(), Some("DEPENDS_ON"));
        assert_eq!(rel.port, Some(1));
    } else {
        panic!("Expected Match clause");
    }
}

#[test]
fn test_parse_rel_pattern_with_port_and_binding() {
    let query = parse_query("MATCH (a)-[r:TRANSFER:2]->(b) RETURN r").unwrap();

    if let Clause::Match(m) = &query.clauses[0] {
        let pattern = &m.patterns[0];
        let (rel, _) = &pattern.chain[0];
        assert_eq!(rel.identifier.as_deref(), Some("r"));
        assert_eq!(rel.label.as_deref(), Some("TRANSFER"));
        assert_eq!(rel.port, Some(2));
    } else {
        panic!("Expected Match clause");
    }
}

#[test]
fn test_parse_rel_pattern_without_port() {
    let query = parse_query("MATCH (a)-[:KNOWS]->(b) RETURN a").unwrap();

    if let Clause::Match(m) = &query.clauses[0] {
        let pattern = &m.patterns[0];
        let (rel, _) = &pattern.chain[0];
        assert_eq!(rel.label.as_deref(), Some("KNOWS"));
        assert_eq!(rel.port, None);
    } else {
        panic!("Expected Match clause");
    }
}

#[test]
fn test_parse_multiple_ports_in_pattern() {
    let query = parse_query("MATCH (a)-[:DEP:0]->(b)-[:DEP:1]->(c) RETURN a, b, c").unwrap();

    if let Clause::Match(m) = &query.clauses[0] {
        let pattern = &m.patterns[0];

        let (rel1, _) = &pattern.chain[0];
        assert_eq!(rel1.label.as_deref(), Some("DEP"));
        assert_eq!(rel1.port, Some(0));

        let (rel2, _) = &pattern.chain[1];
        assert_eq!(rel2.label.as_deref(), Some("DEP"));
        assert_eq!(rel2.port, Some(1));
    } else {
        panic!("Expected Match clause");
    }
}

#[test]
fn test_rel_pattern_display_with_port() {
    use neural_parser::RelPattern;

    let rel = RelPattern::outgoing_with_label_and_port("TRANSFER", 5);
    let display = format!("{}", rel);
    assert_eq!(display, "-[:TRANSFER:5]->");
}
