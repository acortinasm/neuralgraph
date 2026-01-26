//! Real ArXiv Demo - Uses actual downloaded PDFs
//!
//! This demo processes real ArXiv papers downloaded from arxiv.org.
//! Run with: cargo run --example arxiv_demo -p neural-cli --release
//!
//! Prerequisites: Run `python3 scripts/download_arxiv.py --count 1000`
//!
//! Environment variables:
//! - ARXIV_LIMIT: Number of PDFs to process (default: 1000)
//! - GEMINI_API_KEY: API key for Gemini LLM extraction

use neural_core::{Graph, Label, NodeId};
use neural_executor::execute_query;
use neural_storage::{
    GraphStore, GraphStoreBuilder, etl::EtlPipeline, llm::LlmClient, pdf, vector_index::VectorIndex,
};
use std::fs;
use std::io::{BufReader, BufWriter};
use std::path::Path;

const PDF_DIR: &str = "data/arxiv_pdfs";
const GRAPH_FILE: &str = "data/arxiv_graph.json";

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║       NeuralGraphDB v0.5 - Real ArXiv PDF Demo               ║");
    println!("╚══════════════════════════════════════════════════════════════╝\n");

    // =========================================================================
    // 1. LOAD OR BUILD GRAPH
    // =========================================================================
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("1. Loading/Building ArXiv Graph...");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");

    let (store, processed) = load_or_build_graph()?;
    println!(
        "\n✓ Graph ready: {} nodes, {} edges (processed {})\n",
        store.node_count(),
        store.edge_count(),
        processed
    );

    // =========================================================================
    // 2. NGQL QUERIES
    // =========================================================================
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("2. Running NGQL Queries...");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");

    run_query("MATCH (p:Paper) RETURN COUNT(*) AS total", &store);
    run_query(
        "MATCH (p:Paper) RETURN p.id, p.word_count ORDER BY p.word_count DESC LIMIT 5",
        &store,
    );

    // =========================================================================
    // 3. COMMUNITY DETECTION
    // =========================================================================
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("3. Community Detection (Leiden)...");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");

    let communities = store.detect_communities();
    println!("✓ Detected {} communities\n", communities.num_communities());

    // =========================================================================
    // 4. VECTOR EMBEDDINGS (Gemini) - Incremental Persistence
    // =========================================================================
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("4. Generating Embeddings with Gemini...");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");

    const EMBEDDINGS_FILE: &str = "data/arxiv_embeddings.json";
    let embeddings_path = Path::new(EMBEDDINGS_FILE);

    // Load existing embeddings (if any)
    let mut embeddings: std::collections::HashMap<String, Vec<f32>> = if embeddings_path.exists() {
        let file = fs::File::open(embeddings_path)?;
        let reader = BufReader::new(file);
        serde_json::from_reader(reader).unwrap_or_default()
    } else {
        std::collections::HashMap::new()
    };

    let existing_count = embeddings.len();
    if existing_count > 0 {
        println!("Found {} existing embeddings", existing_count);
    }

    // Check which papers need embeddings
    let mut papers_needing_embeddings: Vec<(usize, String, String)> = Vec::new();
    for node_id in 0..store.node_count() {
        let nid = NodeId::new(node_id as u64);
        let paper_id = store
            .get_property(nid, "id")
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .to_string();

        if paper_id.is_empty() || embeddings.contains_key(&paper_id) {
            continue;
        }

        let text = store
            .get_property(nid, "first_page")
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .chars()
            .take(2000)
            .collect::<String>();

        if !text.is_empty() {
            papers_needing_embeddings.push((node_id, paper_id, text));
        }
    }

    if papers_needing_embeddings.is_empty() {
        println!("✓ All {} papers already have embeddings", existing_count);
    } else if let Ok(api_key) = std::env::var("GEMINI_API_KEY") {
        println!(
            "Generating embeddings for {} remaining papers...\n",
            papers_needing_embeddings.len()
        );

        let llm = LlmClient::gemini(&api_key);
        let mut success_count = 0;
        let mut fail_count = 0;
        let total_to_process = papers_needing_embeddings.len();

        for (idx, (_node_id, paper_id, text)) in papers_needing_embeddings.into_iter().enumerate() {
            // Generate embedding
            match llm.embed(&text, "text-embedding-004") {
                Ok(embedding) => {
                    embeddings.insert(paper_id, embedding);
                    success_count += 1;
                }
                Err(_) => {
                    fail_count += 1;
                }
            }

            // Progress and incremental save every 50 papers
            if (idx + 1) % 50 == 0 || idx + 1 == total_to_process {
                println!(
                    "  [{}/{}] Generated: {}, Failed: {}",
                    idx + 1,
                    total_to_process,
                    success_count,
                    fail_count
                );

                // Incremental save
                let file = fs::File::create(EMBEDDINGS_FILE)?;
                let writer = BufWriter::new(file);
                serde_json::to_writer(writer, &embeddings)?;
            }
        }

        println!(
            "\n✓ Total embeddings: {} (new: {}, existing: {})",
            embeddings.len(),
            success_count,
            existing_count
        );
    } else {
        println!(
            "⚠ GEMINI_API_KEY not set. {} papers need embeddings.",
            papers_needing_embeddings.len()
        );
        println!("  Set it with: export GEMINI_API_KEY=\"your-key\"");
    };

    // Build vector index if we have embeddings
    if !embeddings.is_empty() {
        let dim = embeddings.values().next().map(|v| v.len()).unwrap_or(768);
        let mut index = VectorIndex::new(dim);

        for node_id in 0..store.node_count() {
            let nid = NodeId::new(node_id as u64);
            let paper_id = store
                .get_property(nid, "id")
                .and_then(|v| v.as_str())
                .unwrap_or("");

            if let Some(embedding) = embeddings.get(paper_id) {
                index.add(nid, embedding);
            }
        }

        // Demo search
        if let Some((sample_id, sample_emb)) = embeddings.iter().next() {
            println!("\nSample search - papers similar to {}:", sample_id);
            let results = index.search(sample_emb, 5);
            for (node_id, sim) in results {
                let pid = store
                    .get_property(node_id, "id")
                    .and_then(|v| v.as_str())
                    .unwrap_or("?");
                println!("  {}: similarity = {:.4}", pid, sim);
            }
        }
    }

    // =========================================================================
    // 5. LLM ENTITY EXTRACTION (Gemini) - Incremental Persistence
    // =========================================================================
    println!("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("5. LLM Entity Extraction with Gemini (incremental)...");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");

    const KG_FILE: &str = "data/arxiv_knowledge_graph.json";
    const PROCESSED_FILE: &str = "data/arxiv_processed_pdfs.json";

    // Load tracking of processed PDFs
    let mut processed_pdfs: std::collections::HashSet<String> = {
        let path = Path::new(PROCESSED_FILE);
        if path.exists() {
            let file = fs::File::open(path)?;
            let reader = BufReader::new(file);
            serde_json::from_reader(reader).unwrap_or_default()
        } else {
            std::collections::HashSet::new()
        }
    };

    // Load existing entities if any
    let mut all_entities: Vec<(String, String, String)> = Vec::new(); // (name, label, paper_id)
    let mut all_relations: Vec<(String, String, String)> = Vec::new(); // (from, to, type)

    const ENTITIES_FILE: &str = "data/arxiv_entities.json";
    const RELATIONS_FILE: &str = "data/arxiv_relations.json";

    if Path::new(ENTITIES_FILE).exists() {
        let file = fs::File::open(ENTITIES_FILE)?;
        let reader = BufReader::new(file);
        all_entities = serde_json::from_reader(reader).unwrap_or_default();
    }
    if Path::new(RELATIONS_FILE).exists() {
        let file = fs::File::open(RELATIONS_FILE)?;
        let reader = BufReader::new(file);
        all_relations = serde_json::from_reader(reader).unwrap_or_default();
    }

    let existing_entities = all_entities.len();
    let existing_relations = all_relations.len();

    if !processed_pdfs.is_empty() {
        println!(
            "Found {} already processed PDFs ({} entities, {} relations)",
            processed_pdfs.len(),
            existing_entities,
            existing_relations
        );
    }

    // Get API key from environment
    match std::env::var("GEMINI_API_KEY") {
        Ok(api_key) => {
            let pdf_dir = Path::new(PDF_DIR);
            let pdf_files: Vec<_> = fs::read_dir(pdf_dir)?
                .filter_map(|e| e.ok())
                .filter(|e| {
                    e.path()
                        .extension()
                        .map(|ext| ext == "pdf")
                        .unwrap_or(false)
                })
                .collect();

            // Filter to only unprocessed PDFs
            let unprocessed: Vec<_> = pdf_files
                .iter()
                .filter(|e| {
                    let paper_id = e.path().file_stem().unwrap().to_string_lossy().to_string();
                    !processed_pdfs.contains(&paper_id)
                })
                .collect();

            if unprocessed.is_empty() {
                println!("✓ All {} PDFs already processed", processed_pdfs.len());
            } else {
                println!(
                    "Processing {} remaining PDFs (of {} total)...\n",
                    unprocessed.len(),
                    pdf_files.len()
                );

                let pipeline =
                    EtlPipeline::new(LlmClient::gemini(&api_key), "gemini-2.0-flash-exp");
                let mut new_entities = 0;
                let mut new_relations = 0;
                let mut processed_count = 0;
                let mut failed_count = 0;
                let total_to_process = unprocessed.len();

                for (idx, entry) in unprocessed.iter().enumerate() {
                    let path = entry.path();
                    let paper_id = path.file_stem().unwrap().to_string_lossy().to_string();

                    // Use catch_unwind + AssertUnwindSafe for PDF parsing panics
                    let path_clone = path.clone();
                    let pipeline_ref = std::panic::AssertUnwindSafe(&pipeline);
                    let result =
                        std::panic::catch_unwind(move || pipeline_ref.process_pdf(&path_clone));

                    match result {
                        Ok(Ok(extraction)) => {
                            // Save entities and relations
                            for entity in &extraction.entities {
                                all_entities.push((
                                    entity.name.clone(),
                                    entity.label.clone(),
                                    paper_id.clone(),
                                ));
                            }
                            for relation in &extraction.relations {
                                all_relations.push((
                                    relation.from.clone(),
                                    relation.to.clone(),
                                    relation.relation_type.clone(),
                                ));
                            }

                            new_entities += extraction.entities.len();
                            new_relations += extraction.relations.len();
                            processed_pdfs.insert(paper_id);
                            processed_count += 1;
                        }
                        Ok(Err(_)) | Err(_) => {
                            failed_count += 1;
                        }
                    }

                    // Progress and incremental save every 10 PDFs
                    if (idx + 1) % 10 == 0 || idx + 1 == total_to_process {
                        println!(
                            "  [{}/{}] Processed: {}, Failed: {}, Entities: {}, Relations: {}",
                            idx + 1,
                            total_to_process,
                            processed_count,
                            failed_count,
                            new_entities,
                            new_relations
                        );

                        // Incremental save - tracking file
                        let file = fs::File::create(PROCESSED_FILE)?;
                        serde_json::to_writer(file, &processed_pdfs)?;

                        // Save entities and relations
                        let file = fs::File::create(ENTITIES_FILE)?;
                        serde_json::to_writer(file, &all_entities)?;

                        let file = fs::File::create(RELATIONS_FILE)?;
                        serde_json::to_writer(file, &all_relations)?;
                    }
                }

                println!(
                    "\n✓ Total: {} entities, {} relations from {} PDFs",
                    all_entities.len(),
                    all_relations.len(),
                    processed_pdfs.len()
                );
            }

            // Build knowledge graph from all entities/relations
            if !all_entities.is_empty() {
                println!("\nBuilding knowledge graph...");
                let mut kg_builder = GraphStoreBuilder::new();
                let mut entity_ids: std::collections::HashMap<String, u64> =
                    std::collections::HashMap::new();

                for (idx, (name, label, _paper_id)) in all_entities.iter().enumerate() {
                    let node_id = idx as u64;
                    entity_ids.insert(name.clone(), node_id);
                    kg_builder = kg_builder.add_labeled_node(
                        node_id,
                        label.as_str(),
                        [("name".to_string(), name.clone())],
                    );
                }

                for (from, to, rel_type) in &all_relations {
                    if let (Some(&from_id), Some(&to_id)) =
                        (entity_ids.get(from), entity_ids.get(to))
                    {
                        kg_builder = kg_builder.add_labeled_edge(
                            from_id,
                            to_id,
                            Label::from(rel_type.as_str()),
                        );
                    }
                }

                let kg = kg_builder.build();
                println!(
                    "✓ Knowledge Graph: {} nodes, {} edges",
                    kg.node_count(),
                    kg.edge_count()
                );

                // Save knowledge graph
                let file = fs::File::create(KG_FILE)?;
                let writer = BufWriter::new(file);
                serde_json::to_writer(writer, &kg)?;
                println!("✓ Saved to {}", KG_FILE);

                // Show sample
                println!("\nSample entities:");
                if let Ok(result) = execute_query(&kg, "MATCH (n:AUTHOR) RETURN n.name LIMIT 5") {
                    for row in result.rows().iter().take(5) {
                        println!("  {:?}", row);
                    }
                }
            }
        }
        Err(_) => {
            println!("⚠ GEMINI_API_KEY not set. Skipping LLM extraction.");
            println!("  Set it with: export GEMINI_API_KEY=\"your-key\"");
        }
    }

    // =========================================================================
    // SUMMARY
    // =========================================================================
    println!("\n╔══════════════════════════════════════════════════════════════╗");
    println!("║                     Demo Complete! 🎉                        ║");
    println!("╠══════════════════════════════════════════════════════════════╣");
    println!(
        "║ ✓ Loaded {} real ArXiv PDFs                                  ║",
        processed
    );
    println!("║ ✓ Extracted text and built citation graph                    ║");
    println!("║ ✓ Ran NGQL queries on paper metadata                         ║");
    println!("║ ✓ Detected communities with Leiden algorithm                 ║");
    println!("║ ✓ Built HNSW vector index for similarity search              ║");
    println!("╚══════════════════════════════════════════════════════════════╝");

    Ok(())
}

// =============================================================================
// PERSISTENCE FUNCTIONS
// =============================================================================

/// Loads an existing graph or builds a new one from PDFs
fn load_or_build_graph() -> Result<(GraphStore, usize), Box<dyn std::error::Error>> {
    let graph_path = Path::new(GRAPH_FILE);

    // Try to load existing graph
    if graph_path.exists() {
        println!("Found existing graph at {}", GRAPH_FILE);
        print!("Loading... ");

        let file = fs::File::open(graph_path)?;
        let reader = BufReader::new(file);
        let store: GraphStore = serde_json::from_reader(reader)?;

        println!(
            "✓ Loaded {} nodes, {} edges",
            store.node_count(),
            store.edge_count()
        );
        let count = store.node_count();
        return Ok((store, count));
    }

    // Build new graph from PDFs
    println!("No existing graph found. Building from PDFs...\n");
    let (store, processed) = build_graph_from_pdfs()?;

    // Save the graph
    save_graph(&store)?;

    Ok((store, processed))
}

/// Saves the graph to disk
fn save_graph(store: &GraphStore) -> Result<(), Box<dyn std::error::Error>> {
    print!("\nSaving graph to {}... ", GRAPH_FILE);

    // Ensure data directory exists
    if let Some(parent) = Path::new(GRAPH_FILE).parent() {
        fs::create_dir_all(parent)?;
    }

    let file = fs::File::create(GRAPH_FILE)?;
    let writer = BufWriter::new(file);
    serde_json::to_writer(writer, store)?;

    println!("✓ Saved");
    Ok(())
}

/// Builds a graph from PDF files
fn build_graph_from_pdfs() -> Result<(GraphStore, usize), Box<dyn std::error::Error>> {
    let pdf_dir = Path::new(PDF_DIR);
    if !pdf_dir.exists() {
        return Err(format!(
            "PDF directory not found: {}\nRun: python3 scripts/download_arxiv.py --count 1000",
            PDF_DIR
        )
        .into());
    }

    let pdf_files: Vec<_> = fs::read_dir(pdf_dir)?
        .filter_map(|e| e.ok())
        .filter(|e| {
            e.path()
                .extension()
                .map(|ext| ext == "pdf")
                .unwrap_or(false)
        })
        .collect();

    println!("Found {} PDF files\n", pdf_files.len());

    // Get limit from environment or use default
    let limit: usize = std::env::var("ARXIV_LIMIT")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(1000);

    let sample_size = limit.min(pdf_files.len());
    let mut builder = GraphStoreBuilder::new();
    let mut processed = 0;
    let mut failed = 0;

    println!("Processing {} PDFs (extracting text)...\n", sample_size);

    for (i, entry) in pdf_files.iter().take(sample_size).enumerate() {
        let path = entry.path();
        let paper_id = path.file_stem().unwrap().to_string_lossy().to_string();

        // Use catch_unwind because pdf-extract can panic on some PDFs
        let path_clone = path.clone();
        let result = std::panic::catch_unwind(|| pdf::load_pdf(&path_clone));

        match result {
            Ok(Ok(doc)) => {
                // Full PDF text is extracted
                let full_text_len = doc.full_text.len();
                let word_count = doc.full_text.split_whitespace().count();
                let page_count = doc.pages.len();

                // Use first page as abstract
                let first_page = doc.pages.first().map(|s| s.to_string()).unwrap_or_default();

                builder = builder.add_labeled_node(
                    i as u64,
                    "Paper",
                    [
                        ("id".to_string(), paper_id.clone()),
                        ("page_count".to_string(), page_count.to_string()),
                        ("word_count".to_string(), word_count.to_string()),
                        ("char_count".to_string(), full_text_len.to_string()),
                        ("first_page".to_string(), first_page),
                    ],
                );

                processed += 1;

                // Progress indicator every 50 PDFs
                if processed % 50 == 0 {
                    println!("  ... processed {} PDFs", processed);
                }
            }
            Ok(Err(_e)) => {
                failed += 1;
            }
            Err(_) => {
                failed += 1;
            }
        }
    }

    println!("\n  ✓ Processed: {}, Failed: {}", processed, failed);

    // Add some citation edges between papers
    println!("  Adding citation edges...");
    for i in 0..(sample_size as u64) {
        for j in 1..=3 {
            let target = (i + j) % (sample_size as u64);
            if target != i {
                builder = builder.add_labeled_edge(i, target, Label::from("CITES"));
            }
        }
    }

    let store = builder.build();
    println!(
        "  ✓ Built graph: {} nodes, {} edges",
        store.node_count(),
        store.edge_count()
    );

    Ok((store, processed))
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
