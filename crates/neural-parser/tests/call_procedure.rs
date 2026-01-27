//! Tests for CALL procedure syntax (Sprint 56)

use neural_parser::{parse_statement, Statement, CallClause, Expression, Literal};

#[test]
fn test_parse_call_neural_search_cosine() {
    let stmt = parse_statement("CALL neural.search($vec, 'cosine', 10)").unwrap();

    match stmt {
        Statement::Call(CallClause { namespace, name, args }) => {
            assert_eq!(namespace, "neural");
            assert_eq!(name, "search");
            assert_eq!(args.len(), 3);

            // First arg: parameter $vec ($ is stripped during parsing)
            assert!(matches!(&args[0], Expression::Parameter(p) if p == "vec"));

            // Second arg: string literal 'cosine'
            assert!(matches!(&args[1], Expression::Literal(Literal::String(s)) if s == "cosine"));

            // Third arg: integer 10
            assert!(matches!(&args[2], Expression::Literal(Literal::Int(10))));
        }
        _ => panic!("Expected Statement::Call, got {:?}", stmt),
    }
}

#[test]
fn test_parse_call_neural_search_euclidean() {
    let stmt = parse_statement("CALL neural.search($queryVector, 'euclidean', 5)").unwrap();

    match stmt {
        Statement::Call(CallClause { namespace, name, args }) => {
            assert_eq!(namespace, "neural");
            assert_eq!(name, "search");
            assert_eq!(args.len(), 3);

            // Second arg: string literal 'euclidean'
            assert!(matches!(&args[1], Expression::Literal(Literal::String(s)) if s == "euclidean"));
        }
        _ => panic!("Expected Statement::Call"),
    }
}

#[test]
fn test_parse_call_neural_search_dot_product() {
    let stmt = parse_statement("CALL neural.search($vec, 'dot_product', 20)").unwrap();

    match stmt {
        Statement::Call(CallClause { namespace, name, args }) => {
            assert_eq!(namespace, "neural");
            assert_eq!(name, "search");
            assert!(matches!(&args[1], Expression::Literal(Literal::String(s)) if s == "dot_product"));
        }
        _ => panic!("Expected Statement::Call"),
    }
}

#[test]
fn test_parse_call_neural_search_l2() {
    // L2 is alias for euclidean
    let stmt = parse_statement("CALL neural.search($embedding, 'l2', 100)").unwrap();

    match stmt {
        Statement::Call(CallClause { namespace, name, args }) => {
            assert_eq!(namespace, "neural");
            assert_eq!(name, "search");
            assert!(matches!(&args[1], Expression::Literal(Literal::String(s)) if s == "l2"));
        }
        _ => panic!("Expected Statement::Call"),
    }
}

#[test]
fn test_call_clause_display() {
    // Test that CallClause can be created and accessed
    let call = CallClause {
        namespace: "neural".to_string(),
        name: "search".to_string(),
        args: vec![
            Expression::Parameter("$vec".to_string()),
            Expression::Literal(Literal::String("cosine".to_string())),
            Expression::Literal(Literal::Int(10)),
        ],
    };

    assert_eq!(call.namespace, "neural");
    assert_eq!(call.name, "search");
    assert_eq!(call.args.len(), 3);
}
