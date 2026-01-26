use neural_core::Graph;
use neural_storage::GraphStore;
use neural_executor::{execute_statement_with_params, StatementResult};

#[test]
fn test_transaction_commit_atomicity() {
    let mut store = GraphStore::new_in_memory();
    let mut tx = None;
    
    // 1. Begin
    execute_statement_with_params(&mut store, "BEGIN", None, &mut tx).unwrap();
    
    // 2. Create nodes in transaction
    execute_statement_with_params(&mut store, "CREATE (:Person {name: 'Alice'})", None, &mut tx).unwrap();
    execute_statement_with_params(&mut store, "CREATE (:Person {name: 'Bob'})", None, &mut tx).unwrap();
    
    // 3. Verify nodes NOT in store yet (Isolation)
    // We use a separate query without the transaction context
    let res = neural_executor::execute_query(&store, "MATCH (n:Person) RETURN n.name").unwrap();
    assert_eq!(res.row_count(), 0);
    
    // 4. Commit
    execute_statement_with_params(&mut store, "COMMIT", None, &mut tx).unwrap();
    
    // 5. Verify nodes exist
    let res = neural_executor::execute_query(&store, "MATCH (n:Person) RETURN n.name").unwrap();
    assert_eq!(res.row_count(), 2);
}

#[test]
fn test_transaction_rollback() {
    let mut store = GraphStore::new_in_memory();
    let mut tx = None;
    
    execute_statement_with_params(&mut store, "BEGIN", None, &mut tx).unwrap();
    execute_statement_with_params(&mut store, "CREATE (:Person {name: 'Alice'})", None, &mut tx).unwrap();
    
    // Rollback
    execute_statement_with_params(&mut store, "ROLLBACK", None, &mut tx).unwrap();
    
    // Verify node does not exist
    let res = neural_executor::execute_query(&store, "MATCH (n:Person) RETURN n.name").unwrap();
    assert_eq!(res.row_count(), 0);
    assert!(tx.is_none());
}

#[test]
fn test_transaction_id_generation() {
    let mut store = GraphStore::new_in_memory();
    let mut tx = None;
    
    execute_statement_with_params(&mut store, "BEGIN", None, &mut tx).unwrap();
    
    // Create first node
    let res1 = execute_statement_with_params(&mut store, "CREATE (a:Person {name: 'Alice'})", None, &mut tx).unwrap();
    // Create second node
    let res2 = execute_statement_with_params(&mut store, "CREATE (b:Person {name: 'Bob'})", None, &mut tx).unwrap();
    
    if let StatementResult::Mutation(neural_executor::MutationResult::NodesCreated { node_ids: ids1, .. }) = res1 {
        if let StatementResult::Mutation(neural_executor::MutationResult::NodesCreated { node_ids: ids2, .. }) = res2 {
            assert_eq!(ids1[0], 0);
            assert_eq!(ids2[0], 1);
        } else { panic!("Expected second node created"); }
    } else { panic!("Expected first node created"); }
    
    execute_statement_with_params(&mut store, "COMMIT", None, &mut tx).unwrap();
    assert_eq!(store.node_count(), 2);
}

#[test]
fn test_transaction_edge_creation() {
    let mut store = GraphStore::new_in_memory();
    let mut tx = None;
    
    execute_statement_with_params(&mut store, "BEGIN", None, &mut tx).unwrap();
    
    // Use separate patterns to avoid parser limitations for combined edge-node creation
    execute_statement_with_params(&mut store, "CREATE (u1:User {name: 'U1'}), (u2:User {name: 'U2'}), (u1)-[:FRIEND]->(u2)", None, &mut tx).unwrap();
    
    execute_statement_with_params(&mut store, "COMMIT", None, &mut tx).unwrap();
    
    assert_eq!(store.node_count(), 2);
    assert_eq!(store.edge_count(), 1);
}

#[test]
fn test_transaction_error_handling() {
    let mut store = GraphStore::new_in_memory();
    let mut tx = None;
    
    // Commit without begin
    let res = execute_statement_with_params(&mut store, "COMMIT", None, &mut tx);
    assert!(res.is_err());
    
    // Double begin
    execute_statement_with_params(&mut store, "BEGIN", None, &mut tx).unwrap();
    let res = execute_statement_with_params(&mut store, "BEGIN", None, &mut tx);
    assert!(res.is_err());
}