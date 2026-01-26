//! ArXiv Search Web Server
//!
//! Provides semantic search over ArXiv papers.
//!
//! Run with: cargo run -p neural-cli --release -- serve

use axum::{
    Json, Router,
    extract::{Path, State},
    http::StatusCode,
    response::Html,
    routing::{get, post},
};
use neural_storage::csv_loader::{load_nodes_csv, load_edges_csv};
use neural_core::{Graph, Label, NodeId};
use neural_executor::{execute_statement_from_ast, execute_query_from_ast, StatementResult};
use neural_parser::Statement;
use neural_storage::{GraphStore, VectorIndex};
use serde::{Deserialize, Serialize};
use serde_json::json;
use std::collections::HashMap;
use std::fs;
use std::io::BufReader;
use std::sync::{Arc, RwLock};
use tower_http::cors::CorsLayer;

const GRAPH_FILE: &str = "data/arxiv_graph.json";
const EMBEDDINGS_FILE: &str = "data/arxiv_embeddings.json";

/// Application state shared across handlers
pub struct AppState {
    pub store: Arc<RwLock<GraphStore>>,
    pub vector_index: VectorIndex,
    pub paper_embeddings: HashMap<String, Vec<f32>>,
    pub papers: Vec<PaperInfo>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PaperInfo {
    pub id: String,
    pub node_id: u64,
    pub title: String,
    pub abstract_text: String,
    pub word_count: usize,
}

#[derive(Debug, Deserialize)]
pub struct SearchRequest {
    pub query: String,
    #[serde(default = "default_limit")]
    pub limit: usize,
}

fn default_limit() -> usize {
    10
}

#[derive(Debug, Serialize)]
pub struct SearchResult {
    pub papers: Vec<PaperWithScore>,
    pub query: String,
    pub total: usize,
}

#[derive(Debug, Clone, Serialize)]
pub struct PaperWithScore {
    pub id: String,
    pub title: String,
    pub abstract_text: String,
    /// Cosine similarity (1.0 = identical, 0.0 = unrelated)
    pub score: f32,
    /// Cosine distance (0.0 = identical, 1.0 = unrelated)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub distance: Option<f32>,
}

/// Loads the graph and embeddings, returns AppState
pub fn load_data() -> Result<AppState, Box<dyn std::error::Error>> {
    println!("Loading graph from {}...", GRAPH_FILE);

    let graph_path = std::path::Path::new(GRAPH_FILE);
    if !graph_path.exists() {
        println!("  ⚠ Graph file not found. Starting with empty graph.");
        return Ok(AppState {
            store: Arc::new(RwLock::new(GraphStore::builder().build())),
            vector_index: VectorIndex::new(768),
            paper_embeddings: HashMap::new(),
            papers: Vec::new(),
        });
    }

    let file = fs::File::open(graph_path)?;
    let reader = BufReader::new(file);
    let store: GraphStore = serde_json::from_reader(reader)?;
    println!(
        "  ✓ Loaded {} nodes, {} edges",
        store.node_count(),
        store.edge_count()
    );

    // Extract paper info
    let mut papers = Vec::new();
    for node_id in 0..store.node_count() {
        let nid = NodeId::new(node_id as u64);

        let id = store
            .get_property(nid, "id")
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .to_string();

        let title = store
            .get_property(nid, "id") // Using ID as title for now
            .and_then(|v| v.as_str())
            .unwrap_or("Unknown")
            .to_string();

        let abstract_text = store
            .get_property(nid, "first_page")
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .chars()
            .take(500)
            .collect();

        let word_count: usize = store
            .get_property(nid, "word_count")
            .and_then(|v| v.as_str())
            .and_then(|s| s.parse().ok())
            .unwrap_or(0);

        if !id.is_empty() {
            papers.push(PaperInfo {
                id,
                node_id: node_id as u64,
                title,
                abstract_text,
                word_count,
            });
        }
    }
    println!("  ✓ Extracted {} papers", papers.len());

    // Load embeddings if available
    let mut paper_embeddings = HashMap::new();
    let mut vector_index = VectorIndex::new(768); // Gemini text-embedding-004 dimension

    let embeddings_path = std::path::Path::new(EMBEDDINGS_FILE);
    if embeddings_path.exists() {
        println!("Loading embeddings from {}...", EMBEDDINGS_FILE);
        let file = fs::File::open(embeddings_path)?;
        let reader = BufReader::new(file);
        let embeddings: HashMap<String, Vec<f32>> = serde_json::from_reader(reader)?;

        for (paper_id, embedding) in &embeddings {
            // Find node_id for this paper
            if let Some(paper) = papers.iter().find(|p| &p.id == paper_id) {
                vector_index.add(NodeId::new(paper.node_id), embedding);
            }
        }
        paper_embeddings = embeddings;
        println!("  ✓ Loaded {} embeddings", paper_embeddings.len());
    } else {
        println!("  ⚠ No embeddings file found. Semantic search will use basic matching.");
    }

    Ok(AppState {
        store: Arc::new(RwLock::new(store)),
        vector_index,
        paper_embeddings,
        papers,
    })
}

/// Creates the Axum router
pub fn create_router(state: Arc<AppState>) -> Router {
    Router::new()
        .route("/", get(index_handler))
        .route("/api/papers", get(list_papers))
        .route("/api/search", post(search_papers))
        .route("/api/similar/{id}", get(similar_papers))
        .route("/api/query", post(handle_query))
        .route("/api/bulk-load", post(handle_bulk_load))
        .layer(CorsLayer::permissive())
        .with_state(state)
}

/// Serves the main HTML page
async fn index_handler() -> Html<&'static str> {
    Html(include_str!("../static/index.html"))
}

/// Lists all papers
async fn list_papers(State(state): State<Arc<AppState>>) -> Json<Vec<PaperInfo>> {
    Json(state.papers.clone())
}

/// Searches papers by query text
async fn search_papers(
    State(state): State<Arc<AppState>>,
    Json(req): Json<SearchRequest>,
) -> Json<SearchResult> {
    let query_lower = req.query.to_lowercase();
    let limit = req.limit.min(50);

    // Simple text matching for now (until embeddings are loaded)
    let mut results: Vec<PaperWithScore> = state
        .papers
        .iter()
        .filter_map(|paper| {
            let title_lower = paper.title.to_lowercase();
            let abstract_lower = paper.abstract_text.to_lowercase();

            // Simple scoring based on text matches
            let mut score = 0.0;

            if title_lower.contains(&query_lower) {
                score += 1.0;
            }
            if abstract_lower.contains(&query_lower) {
                score += 0.5;
            }

            // Check for individual words
            for word in query_lower.split_whitespace() {
                if title_lower.contains(word) {
                    score += 0.3;
                }
                if abstract_lower.contains(word) {
                    score += 0.1;
                }
            }

            if score > 0.0 {
                Some(PaperWithScore {
                    id: paper.id.clone(),
                    title: paper.title.clone(),
                    abstract_text: paper.abstract_text.clone(),
                    score,
                    distance: None, // Text search doesn't use distances
                })
            } else {
                None
            }
        })
        .collect();

    // Sort by score descending
    results.sort_by(|a, b| {
        b.score
            .partial_cmp(&a.score)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    results.truncate(limit);

    let total = results.len();

    Json(SearchResult {
        papers: results,
        query: req.query,
        total,
    })
}

/// Finds similar papers to a given paper ID
async fn similar_papers(
    State(state): State<Arc<AppState>>,
    Path(paper_id): Path<String>,
) -> Result<Json<SearchResult>, (StatusCode, String)> {
    // Find the paper
    let paper = state.papers.iter().find(|p| p.id == paper_id).ok_or((
        StatusCode::NOT_FOUND,
        format!("Paper not found: {}", paper_id),
    ))?;

    // Check if we have embeddings for vector search
    if let Some(embedding) = state.paper_embeddings.get(&paper_id) {
        // Vector similarity search
        let results = state.vector_index.search(embedding, 11); // +1 for self

        let papers: Vec<PaperWithScore> = results
            .into_iter()
            .filter(|(nid, _)| nid.as_u64() != paper.node_id) // Exclude self
            .filter_map(|(nid, similarity)| {
                state
                    .papers
                    .iter()
                    .find(|p| p.node_id == nid.as_u64())
                    .map(|p| PaperWithScore {
                        id: p.id.clone(),
                        title: p.title.clone(),
                        abstract_text: p.abstract_text.clone(),
                        score: similarity,
                        distance: Some(1.0 - similarity), // Cosine distance
                    })
            })
            .take(10)
            .collect();

        let total = papers.len();
        return Ok(Json(SearchResult {
            papers,
            query: format!("Similar to: {}", paper_id),
            total,
        }));
    }

    // Fallback: Use graph neighbors (papers that cite or are cited by this paper)
    let nid = NodeId::new(paper.node_id);
    let store = state.store.read().unwrap();
    let neighbors: Vec<_> = store.neighbors(nid).collect();

    let papers: Vec<PaperWithScore> = neighbors
        .into_iter()
        .filter_map(|neighbor_id| {
            state
                .papers
                .iter()
                .find(|p| p.node_id == neighbor_id.as_u64())
                .map(|p| {
                    PaperWithScore {
                        id: p.id.clone(),
                        title: p.title.clone(),
                        abstract_text: p.abstract_text.clone(),
                        score: 1.0,          // Graph-based similarity
                        distance: Some(0.0), // Connected nodes have 0 distance
                    }
                })
        })
        .take(10)
        .collect();

    let total = papers.len();
    Ok(Json(SearchResult {
        papers,
        query: format!("Connected to: {}", paper_id),
        total,
    }))
}

#[derive(Debug, Deserialize)]
pub struct QueryRequest {
    pub query: String,
}

#[derive(Debug, Deserialize)]
pub struct BulkLoadRequest {
    pub nodes_path: Option<String>,
    pub edges_path: Option<String>,
    #[serde(default)]
    pub clear_existing: bool,
}

#[derive(Debug, Serialize)]
pub struct BulkLoadResponse {
    pub success: bool,
    pub nodes_loaded: usize,
    pub edges_loaded: usize,
    pub error: Option<String>,
    pub load_time_ms: f64,
}

#[derive(Debug, Serialize)]
pub struct QueryResponse {
    pub success: bool,
    pub result: Option<serde_json::Value>,
    pub error: Option<String>,
    pub execution_time_ms: f64,
}

async fn handle_query(
    State(state): State<Arc<AppState>>,
    Json(req): Json<QueryRequest>,
) -> Json<QueryResponse> {
    let start = std::time::Instant::now();
    println!("Executing query: {}", req.query);

    // 1. Parse to check type
    let stmt = match neural_parser::parse_statement(&req.query) {
        Ok(s) => s,
        Err(e) => return Json(QueryResponse {
            success: false,
            result: None,
            error: Some(format!("Parse Error: {}", e)),
            execution_time_ms: start.elapsed().as_secs_f64() * 1000.0,
        }),
    };

    // 2. Execute (using pre-parsed AST to avoid double parsing)
    let res = match stmt {
        Statement::Query(ref q) => {
            // Read-only lock - pass AST directly
            let store = state.store.read().unwrap();
            match execute_query_from_ast(&store, q, None) {
                Ok(r) => Ok(StatementResult::Query(r)),
                Err(e) => Err(e),
            }
        },
        _ => {
            // Write lock - pass AST directly
            let mut store = state.store.write().unwrap();
            execute_statement_from_ast(&mut store, stmt, None, &mut None)
        }
    };

    let elapsed_ms = start.elapsed().as_secs_f64() * 1000.0;

    // 3. Format response
    match res {
        Ok(StatementResult::Query(q_res)) => {
            // Convert rows to JSON
            let mut rows = Vec::new();
            let columns = q_res.columns().to_vec();
            for row in q_res.rows() {
                let mut row_map = HashMap::new();
                for col_name in columns.iter() {
                    let val = row.get(col_name).unwrap();
                    row_map.insert(col_name.clone(), format!("{:?}", val)); // Simple string repr for now
                }
                rows.push(row_map);
            }
            Json(QueryResponse {
                success: true,
                result: Some(serde_json::json!({
                    "type": "query",
                    "columns": columns,
                    "rows": rows,
                    "count": q_res.row_count()
                })),
                error: None,
                execution_time_ms: elapsed_ms,
            })
        },
        Ok(StatementResult::Mutation(m_res)) => {
            let result_json = match m_res {
                neural_executor::MutationResult::NodesCreated { count, node_ids } => json!({
                    "type": "mutation",
                    "operation": "create_nodes",
                    "count": count,
                    "node_ids": node_ids
                }),
                neural_executor::MutationResult::EdgesCreated { count } => json!({
                    "type": "mutation",
                    "operation": "create_edges",
                    "count": count
                }),
                neural_executor::MutationResult::NodesDeleted { count } => json!({
                    "type": "mutation",
                    "operation": "delete_nodes",
                    "count": count
                }),
                neural_executor::MutationResult::PropertiesSet { count } => json!({
                    "type": "mutation",
                    "operation": "set_properties",
                    "count": count
                }),
            };
             Json(QueryResponse {
                success: true,
                result: Some(result_json),
                error: None,
                execution_time_ms: elapsed_ms,
            })
        },
        Ok(StatementResult::Explain(plan)) => {
             Json(QueryResponse {
                success: true,
                result: Some(serde_json::json!({
                    "type": "explain",
                    "plan": plan
                })),
                error: None,
                execution_time_ms: elapsed_ms,
            })
        },
        Ok(StatementResult::TransactionStarted) => {
             Json(QueryResponse {
                success: true,
                result: Some(serde_json::json!({
                    "type": "transaction",
                    "status": "started",
                    "message": "Transaction started (Note: HTTP is stateless, tx scoped to request)"
                })),
                error: None,
                execution_time_ms: elapsed_ms,
            })
        },
        Ok(StatementResult::TransactionCommitted) => {
             Json(QueryResponse {
                success: true,
                result: Some(serde_json::json!({
                    "type": "transaction",
                    "status": "committed"
                })),
                error: None,
                execution_time_ms: elapsed_ms,
            })
        },
        Ok(StatementResult::TransactionRolledBack) => {
             Json(QueryResponse {
                success: true,
                result: Some(serde_json::json!({
                    "type": "transaction",
                    "status": "rolled_back"
                })),
                error: None,
                execution_time_ms: elapsed_ms,
            })
        },
        Err(e) => Json(QueryResponse {
            success: false,
            result: None,
            error: Some(format!("Execution Error: {}", e)),
            execution_time_ms: elapsed_ms,
        }),
    }
}

/// Handles bulk loading of CSV data using the fast builder pattern
async fn handle_bulk_load(
    State(state): State<Arc<AppState>>,
    Json(req): Json<BulkLoadRequest>,
) -> Json<BulkLoadResponse> {
    let start = std::time::Instant::now();
    println!("Bulk load request: nodes={:?}, edges={:?}, clear={}",
             req.nodes_path, req.edges_path, req.clear_existing);

    let mut nodes_loaded = 0usize;
    let mut edges_loaded = 0usize;

    // Use builder pattern for fast bulk loading (like REPL rebuild)
    let mut builder = GraphStore::builder();

    // Load nodes if path provided
    if let Some(path) = &req.nodes_path {
        match load_nodes_csv(path) {
            Ok(nodes) => {
                nodes_loaded = nodes.len();
                for node in nodes {
                    if let Some(label) = node.label {
                        builder = builder.add_labeled_node(node.id, label, node.properties);
                    } else {
                        builder = builder.add_node(node.id, node.properties);
                    }
                }
                println!("  Parsed {} nodes", nodes_loaded);
            }
            Err(e) => {
                return Json(BulkLoadResponse {
                    success: false,
                    nodes_loaded: 0,
                    edges_loaded: 0,
                    error: Some(format!("Failed to load nodes: {}", e)),
                    load_time_ms: start.elapsed().as_secs_f64() * 1000.0,
                });
            }
        }
    }

    // Load edges if path provided
    if let Some(path) = &req.edges_path {
        match load_edges_csv(path) {
            Ok(edges) => {
                edges_loaded = edges.len();
                for edge in edges {
                    if let Some(label) = edge.label {
                        builder = builder.add_labeled_edge(
                            edge.source,
                            edge.target,
                            Label::new(label),
                        );
                    } else {
                        builder = builder.add_edge(edge.source, edge.target);
                    }
                }
                println!("  Parsed {} edges", edges_loaded);
            }
            Err(e) => {
                return Json(BulkLoadResponse {
                    success: false,
                    nodes_loaded,
                    edges_loaded: 0,
                    error: Some(format!("Failed to load edges: {}", e)),
                    load_time_ms: start.elapsed().as_secs_f64() * 1000.0,
                });
            }
        }
    }

    // Build and replace the store
    println!("  Building graph store...");
    let new_store = builder.build();

    {
        let mut store = state.store.write().unwrap();
        *store = new_store;
    }

    let elapsed = start.elapsed().as_secs_f64() * 1000.0;
    println!("  Bulk load completed in {:.2}ms", elapsed);

    Json(BulkLoadResponse {
        success: true,
        nodes_loaded,
        edges_loaded,
        error: None,
        load_time_ms: elapsed,
    })
}

/// Starts the web server
pub async fn run_server(port: u16) -> Result<(), Box<dyn std::error::Error>> {
    let state = Arc::new(load_data()?);
    let app = create_router(state);

    let addr = format!("0.0.0.0:{}", port);
    println!("\n🚀 Server running at http://localhost:{}", port);
    println!("   Press Ctrl+C to stop\n");

    let listener = tokio::net::TcpListener::bind(&addr).await?;
    axum::serve(listener, app).await?;

    Ok(())
}
