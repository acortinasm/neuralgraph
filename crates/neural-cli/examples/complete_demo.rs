//! Complete NeuralGraphDB v0.5 Demo
//!
//! This example demonstrates all major features:
//! - Graph construction with labeled nodes/edges
//! - NGQL queries with aggregations
//! - Community detection (Leiden)
//! - Vector similarity search (HNSW)
//! - ETL pipeline (simulated)
//!
//! Run with: cargo run --example complete_demo -p neural-cli --release

use neural_core::{Graph, Label, NodeId};
use neural_executor::execute_query;
use neural_storage::{
    GraphStore, GraphStoreBuilder,
    etl::{Entity, EtlPipeline, ExtractionResult, Relation},
    llm::LlmClient,
    vector_index::VectorIndex,
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║          NeuralGraphDB v0.5.0-beta - Complete Demo           ║");
    println!("╚══════════════════════════════════════════════════════════════╝\n");

    // =========================================================================
    // 1. BUILD CITATION GRAPH
    // =========================================================================
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("1. Building Citation Graph (1000 papers, 5000+ citations)...");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");

    let store = create_citation_graph();
    println!(
        "✓ Built graph: {} nodes, {} edges\n",
        store.node_count(),
        store.edge_count()
    );

    // =========================================================================
    // 2. NGQL QUERIES
    // =========================================================================
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("2. Running NGQL Queries...");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");

    run_query("MATCH (n) RETURN COUNT(*) AS total", &store);
    run_query(
        "MATCH (n:paper) RETURN n.category, COUNT(*) AS cnt ORDER BY cnt DESC LIMIT 5",
        &store,
    );

    // =========================================================================
    // 3. COMMUNITY DETECTION
    // =========================================================================
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("3. Running Leiden Community Detection...");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");

    let communities = store.detect_communities();
    println!("✓ Detected {} communities\n", communities.num_communities());

    // =========================================================================
    // 4. VECTOR SEARCH
    // =========================================================================
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("4. Vector Similarity Search (HNSW)...");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");

    demo_vector_search();

    // =========================================================================
    // 5. ETL PIPELINE (Simulated)
    // =========================================================================
    println!("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("5. ETL Pipeline Demo (PDF → LLM → Graph)...");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");

    demo_etl_pipeline();

    // =========================================================================
    // SUMMARY
    // =========================================================================
    println!("\n╔══════════════════════════════════════════════════════════════╗");
    println!("║                     Demo Complete! 🎉                        ║");
    println!("╠══════════════════════════════════════════════════════════════╣");
    println!("║ ✓ Graph construction with labels and properties              ║");
    println!("║ ✓ NGQL queries (MATCH, WHERE, RETURN, ORDER BY, LIMIT)       ║");
    println!("║ ✓ Aggregations (COUNT, GROUP BY)                             ║");
    println!("║ ✓ Community detection (Leiden algorithm)                     ║");
    println!("║ ✓ Vector similarity search (HNSW index)                      ║");
    println!("║ ✓ ETL pipeline (PDF → LLM → Graph)                           ║");
    println!("╚══════════════════════════════════════════════════════════════╝");

    Ok(())
}

fn run_query(query: &str, store: &GraphStore) {
    println!("Query: {}", query);
    match execute_query(store, query) {
        Ok(result) => {
            let rows = result.rows();
            println!("Result: {} rows", rows.len());
            for row in rows.iter().take(3) {
                println!("  {:?}", row);
            }
        }
        Err(e) => println!("Error: {}", e),
    }
    println!();
}

fn create_citation_graph() -> GraphStore {
    let mut builder = GraphStoreBuilder::new();
    let categories = ["cs.LG", "cs.AI", "cs.CV", "stat.ML", "cs.CL"];

    for i in 0..1000u64 {
        let cat = categories[i as usize % categories.len()];
        builder = builder.add_labeled_node(
            i,
            "paper",
            [
                ("title".to_string(), format!("Paper #{}", i)),
                ("category".to_string(), cat.to_string()),
            ],
        );
    }

    for i in 0..1000u64 {
        for j in 1..=((i % 10) + 1) {
            let target = (i + j * 7) % 1000;
            if target != i {
                builder = builder.add_labeled_edge(i, target, Label::from("CITES"));
            }
        }
    }

    builder.build()
}

fn demo_vector_search() {
    let dim = 64;
    let mut index = VectorIndex::new(dim);

    for i in 0..50u64 {
        let vec: Vec<f32> = (0..dim)
            .map(|j| ((i + j as u64) % 100) as f32 / 100.0)
            .collect();
        index.add(NodeId::new(i), &vec);
    }

    let query: Vec<f32> = (0..dim).map(|j| (j % 10) as f32 / 10.0).collect();
    let results = index.search(&query, 5);

    println!("Top 5 nearest neighbors:");
    for (node_id, similarity) in results {
        println!("  {:?}: similarity = {:.4}", node_id, similarity);
    }
}

fn demo_etl_pipeline() {
    let extraction = ExtractionResult {
        entities: vec![
            Entity {
                name: "Einstein".into(),
                label: "Person".into(),
                properties: Default::default(),
            },
            Entity {
                name: "Relativity".into(),
                label: "Theory".into(),
                properties: Default::default(),
            },
        ],
        relations: vec![Relation {
            from: "Einstein".into(),
            to: "Relativity".into(),
            relation_type: "DEVELOPED".into(),
        }],
    };

    println!(
        "Simulated extraction: {} entities, {} relations",
        extraction.entities.len(),
        extraction.relations.len()
    );

    let pipeline = EtlPipeline::new(LlmClient::ollama(), "llama2");
    let store = pipeline
        .insert_into_graph(&extraction, GraphStoreBuilder::new())
        .build();

    println!(
        "Built graph: {} nodes, {} edges",
        store.node_count(),
        store.edge_count()
    );
}
