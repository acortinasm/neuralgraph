//! ArXiv Search Web Server
//!
//! Provides semantic search over ArXiv papers.
//!
//! Run with: cargo run -p neural-cli --release -- serve

use axum::{
    Extension, Json, Router,
    extract::{Path, State},
    http::StatusCode,
    middleware,
    response::Html,
    routing::{get, post},
};
use neural_storage::csv_loader::{load_nodes_csv, load_edges_csv};
use neural_core::{Graph, Label, NodeId};
use neural_executor::{execute_statement_from_ast, execute_query_from_ast, StatementResult};
use neural_parser::Statement;
use neural_storage::{AuthConfig, AuthUser, BackupConfig, GraphStore, MetricsRegistry, VectorIndex};
use crate::auth_middleware::{auth_middleware, check_permission, is_mutation_query};
use serde::{Deserialize, Serialize};
use serde_json::json;
use std::collections::HashMap;
use std::fs;
use std::io::BufReader;
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::RwLock;
use tower_http::cors::CorsLayer;

const GRAPH_FILE: &str = "data/arxiv_graph.json";
const EMBEDDINGS_FILE: &str = "data/arxiv_embeddings.json";

// =============================================================================
// Persistence Configuration
// =============================================================================

/// Persistence configuration loaded from environment variables.
///
/// Environment variables:
/// - `NGDB_PATH`: Database file path (default: "data/graph.ngdb")
/// - `NGDB_SAVE_INTERVAL`: Seconds between periodic saves (default: 60)
/// - `NGDB_SAVE_THRESHOLD`: Mutations before auto-save (default: 10)
/// - `NGDB_BACKUP_COUNT`: Number of backups to retain (default: 3)
/// - `NGDB_SHUTDOWN_TIMEOUT`: Graceful shutdown timeout in seconds (default: 30)
#[derive(Debug, Clone)]
pub struct PersistenceConfig {
    /// Database file path
    pub db_path: PathBuf,
    /// Seconds between periodic saves
    pub save_interval_secs: u64,
    /// Mutations before triggering save
    pub mutation_threshold: u64,
    /// Number of backups to retain
    pub backup_count: usize,
    /// Graceful shutdown timeout in seconds
    pub shutdown_timeout_secs: u64,
}

impl Default for PersistenceConfig {
    fn default() -> Self {
        Self {
            db_path: PathBuf::from("data/graph.ngdb"),
            save_interval_secs: 60,
            mutation_threshold: 10,
            backup_count: 3,
            shutdown_timeout_secs: 30,
        }
    }
}

impl PersistenceConfig {
    /// Loads configuration from environment variables with defaults.
    pub fn from_env() -> Self {
        Self {
            db_path: std::env::var("NGDB_PATH")
                .map(PathBuf::from)
                .unwrap_or_else(|_| PathBuf::from("data/graph.ngdb")),

            save_interval_secs: std::env::var("NGDB_SAVE_INTERVAL")
                .ok()
                .and_then(|s| s.parse().ok())
                .unwrap_or(60),

            mutation_threshold: std::env::var("NGDB_SAVE_THRESHOLD")
                .ok()
                .and_then(|s| s.parse().ok())
                .unwrap_or(10),

            backup_count: std::env::var("NGDB_BACKUP_COUNT")
                .ok()
                .and_then(|s| s.parse().ok())
                .unwrap_or(3),

            shutdown_timeout_secs: std::env::var("NGDB_SHUTDOWN_TIMEOUT")
                .ok()
                .and_then(|s| s.parse().ok())
                .unwrap_or(30),
        }
    }

    /// Creates a BackupConfig from this persistence config.
    pub fn backup_config(&self) -> BackupConfig {
        BackupConfig::new(self.backup_count)
    }
}

/// Loads auth configuration from environment variables (Sprint 68).
///
/// Environment variables:
/// - `NGDB__AUTH__ENABLED`: Enable authentication (default: false)
/// - `NGDB__AUTH__JWT_SECRET`: JWT signing secret (required if enabled)
/// - `NGDB__AUTH__JWT_EXPIRATION_SECS`: JWT expiration in seconds (default: 3600)
fn load_auth_config_from_env() -> AuthConfig {
    let enabled = std::env::var("NGDB__AUTH__ENABLED")
        .map(|v| v.to_lowercase() == "true" || v == "1")
        .unwrap_or(false);

    let jwt_secret = std::env::var("NGDB__AUTH__JWT_SECRET").unwrap_or_default();

    let jwt_expiration_secs = std::env::var("NGDB__AUTH__JWT_EXPIRATION_SECS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(3600);

    AuthConfig {
        enabled,
        jwt_secret,
        jwt_expiration_secs,
        api_keys: Vec::new(), // API keys loaded from config file only
    }
}

/// Application state shared across handlers
pub struct AppState {
    /// The graph store (async RwLock for non-blocking access)
    pub store: Arc<RwLock<GraphStore>>,
    pub vector_index: VectorIndex,
    pub paper_embeddings: HashMap<String, Vec<f32>>,
    pub papers: Vec<PaperInfo>,
    /// Mutation counter for auto-save threshold
    pub mutation_count: AtomicU64,
    /// Persistence configuration
    pub config: PersistenceConfig,
    /// Metrics registry for observability (Sprint 67)
    pub metrics: Arc<MetricsRegistry>,
    /// Server start time for uptime calculation (Sprint 67)
    pub start_time: std::time::Instant,
    /// Authentication configuration (Sprint 68)
    pub auth_config: Arc<AuthConfig>,
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
pub fn load_data(config: PersistenceConfig) -> Result<AppState, Box<dyn std::error::Error>> {
    load_data_with_auth(config, AuthConfig::default())
}

/// Loads the graph and embeddings with auth configuration, returns AppState
pub fn load_data_with_auth(config: PersistenceConfig, auth_config: AuthConfig) -> Result<AppState, Box<dyn std::error::Error>> {
    let path = &config.db_path;

    // Try to load from persistent binary format first
    let store = if path.exists() {
        println!("Loading graph from {}...", path.display());
        match GraphStore::load_binary(path) {
            Ok(s) => {
                println!(
                    "  ✓ Loaded {} nodes, {} edges from binary format",
                    s.node_count(),
                    s.edge_count()
                );
                s
            }
            Err(e) => {
                println!("  ⚠ Failed to load binary: {}. Trying JSON fallback...", e);
                load_from_json_fallback()?
            }
        }
    } else {
        // Check for legacy JSON graph file
        let json_path = std::path::Path::new(GRAPH_FILE);
        if json_path.exists() {
            println!("Loading graph from legacy JSON {}...", GRAPH_FILE);
            load_from_json_fallback()?
        } else {
            // Create new empty store with WAL enabled
            println!("Creating new graph at {}...", path.display());
            if let Some(parent) = path.parent() {
                std::fs::create_dir_all(parent)?;
            }
            match GraphStore::new(path) {
                Ok(s) => {
                    println!("  ✓ Created new graph with WAL enabled");
                    s
                }
                Err(e) => {
                    println!("  ⚠ Failed to create with WAL: {}. Using in-memory.", e);
                    GraphStore::builder().build()
                }
            }
        }
    };

    // Extract paper info (for arxiv search compatibility)
    let papers = extract_paper_info(&store);
    println!("  ✓ Extracted {} papers", papers.len());

    // Load embeddings if available
    let (vector_index, paper_embeddings) = load_embeddings(&papers);

    // Initialize metrics registry (Sprint 67)
    let metrics = Arc::new(MetricsRegistry::new().expect("Failed to create metrics registry"));
    metrics.set_node_count(store.node_count());
    metrics.set_edge_count(store.edge_count());

    Ok(AppState {
        store: Arc::new(RwLock::new(store)),
        vector_index,
        paper_embeddings,
        papers,
        mutation_count: AtomicU64::new(0),
        config,
        metrics,
        start_time: std::time::Instant::now(),
        auth_config: Arc::new(auth_config),
    })
}

/// Loads graph from legacy JSON format
fn load_from_json_fallback() -> Result<GraphStore, Box<dyn std::error::Error>> {
    let graph_path = std::path::Path::new(GRAPH_FILE);
    if !graph_path.exists() {
        return Ok(GraphStore::builder().build());
    }

    let file = fs::File::open(graph_path)?;
    let reader = BufReader::new(file);
    let store: GraphStore = serde_json::from_reader(reader)?;
    println!(
        "  ✓ Loaded {} nodes, {} edges from JSON",
        store.node_count(),
        store.edge_count()
    );
    Ok(store)
}

/// Extracts paper info from the graph store (for arxiv search)
fn extract_paper_info(store: &GraphStore) -> Vec<PaperInfo> {
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
    papers
}

/// Loads embeddings if available
fn load_embeddings(papers: &[PaperInfo]) -> (VectorIndex, HashMap<String, Vec<f32>>) {
    let mut paper_embeddings = HashMap::new();
    let mut vector_index = VectorIndex::new(768); // Gemini text-embedding-004 dimension

    let embeddings_path = std::path::Path::new(EMBEDDINGS_FILE);
    if embeddings_path.exists() {
        println!("Loading embeddings from {}...", EMBEDDINGS_FILE);
        if let Ok(file) = fs::File::open(embeddings_path) {
            let reader = BufReader::new(file);
            if let Ok(embeddings) = serde_json::from_reader::<_, HashMap<String, Vec<f32>>>(reader) {
                for (paper_id, embedding) in &embeddings {
                    // Find node_id for this paper
                    if let Some(paper) = papers.iter().find(|p| &p.id == paper_id) {
                        vector_index.add(NodeId::new(paper.node_id), embedding);
                    }
                }
                println!("  ✓ Loaded {} embeddings", embeddings.len());
                paper_embeddings = embeddings;
            }
        }
    } else {
        println!("  ⚠ No embeddings file found. Semantic search will use basic matching.");
    }

    (vector_index, paper_embeddings)
}

/// Creates the Axum router with authentication (Sprint 68)
pub fn create_router(state: Arc<AppState>) -> Router {
    // Public routes - no authentication required
    let public = Router::new()
        .route("/", get(index_handler))
        .route("/health", get(handle_health))
        .route("/metrics", get(handle_metrics));

    // Protected routes - require authentication when enabled
    let protected = Router::new()
        .route("/api/papers", get(list_papers))
        .route("/api/search", post(search_papers))
        .route("/api/similar/{id}", get(similar_papers))
        .route("/api/query", post(handle_query))
        .route("/api/bulk-load", post(handle_bulk_load))
        .route("/api/schema", get(handle_schema))
        .layer(middleware::from_fn(auth_middleware))
        .layer(Extension(state.auth_config.clone()));

    Router::new()
        .merge(public)
        .merge(protected)
        .layer(CorsLayer::permissive())
        .with_state(state)
}

/// Serves the main HTML page
async fn index_handler() -> Html<&'static str> {
    Html(include_str!("../static/index.html"))
}

// =============================================================================
// Health & Metrics Handlers (Sprint 67)
// =============================================================================

/// Returns health status for production monitoring and load balancer health checks.
async fn handle_health(State(state): State<Arc<AppState>>) -> Json<HealthResponse> {
    let store = state.store.read().await;
    let node_count = store.node_count();
    let edge_count = store.edge_count();
    let uptime = state.start_time.elapsed().as_secs();

    // Update metrics with current counts
    state.metrics.set_node_count(node_count);
    state.metrics.set_edge_count(edge_count);

    Json(HealthResponse {
        status: "healthy".to_string(),
        uptime_seconds: uptime,
        database: DatabaseHealth {
            loaded: true,
            node_count,
            edge_count,
            path: state.config.db_path.display().to_string(),
        },
    })
}

/// Exports metrics in Prometheus text format for scraping.
async fn handle_metrics(State(state): State<Arc<AppState>>) -> Result<String, (StatusCode, String)> {
    // Update graph statistics before export
    let store = state.store.read().await;
    state.metrics.set_node_count(store.node_count());
    state.metrics.set_edge_count(store.edge_count());
    drop(store);

    state.metrics.export().map_err(|e| {
        (StatusCode::INTERNAL_SERVER_ERROR, format!("Failed to export metrics: {}", e))
    })
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
    let store = state.store.read().await;
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

// =============================================================================
// Health & Metrics Response Types (Sprint 67)
// =============================================================================

/// Health check response for production monitoring.
#[derive(Debug, Serialize)]
pub struct HealthResponse {
    /// Overall health status: "healthy" or "degraded"
    pub status: String,
    /// Server uptime in seconds
    pub uptime_seconds: u64,
    /// Database health details
    pub database: DatabaseHealth,
}

/// Database health details.
#[derive(Debug, Serialize)]
pub struct DatabaseHealth {
    /// Whether the database is loaded and operational
    pub loaded: bool,
    /// Total number of nodes
    pub node_count: usize,
    /// Total number of edges
    pub edge_count: usize,
    /// Database file path
    pub path: String,
}

/// Schema response for LangChain integration (Sprint 65)
#[derive(Debug, Serialize)]
pub struct SchemaResponse {
    /// Human-readable schema description for LLM context
    pub schema: String,
    /// Structured schema for programmatic use
    pub structured_schema: StructuredSchema,
}

/// Structured schema representation
#[derive(Debug, Serialize)]
pub struct StructuredSchema {
    /// Node labels and their properties
    pub node_props: HashMap<String, Vec<String>>,
    /// Relationship types
    pub rel_types: Vec<String>,
    /// All property keys used in the graph
    pub property_keys: Vec<String>,
}

/// Triggers non-blocking auto-save if mutation threshold is reached.
///
/// Uses compare-exchange to atomically check and reset the counter,
/// preventing multiple concurrent saves.
async fn maybe_save(state: &Arc<AppState>) {
    let threshold = state.config.mutation_threshold;
    let count = state.mutation_count.load(Ordering::SeqCst);

    if count >= threshold {
        // Atomically reset counter to prevent duplicate saves
        if state.mutation_count
            .compare_exchange(count, 0, Ordering::SeqCst, Ordering::SeqCst)
            .is_err()
        {
            return; // Another task already handled it
        }

        // Take snapshot under read lock (fast - just cloning)
        let snapshot = {
            let store = state.store.read().await;
            store.snapshot()
        };
        // Lock released here - I/O happens without blocking

        let path = state.config.db_path.clone();
        let backup_config = state.config.backup_config();

        // Save in background using spawn_blocking for I/O
        tokio::task::spawn_blocking(move || {
            match snapshot.save_with_backups(&path, &backup_config) {
                Ok(()) => println!("Auto-saved graph to {:?}", path),
                Err(e) => {
                    eprintln!("Auto-save failed: {}", e);
                    if e.is_critical() {
                        eprintln!("CRITICAL: Persistence failure may cause data loss!");
                    }
                }
            }
        });
    }
}

/// Returns the graph schema for LangChain integration (Sprint 65)
///
/// Provides both human-readable and structured schema representations
/// for use with GraphCypherQAChain and other LLM-based query generators.
async fn handle_schema(
    State(state): State<Arc<AppState>>,
) -> Json<SchemaResponse> {
    let store = state.store.read().await;

    // Collect node labels
    let labels: Vec<String> = store.labels().map(|s| s.to_string()).collect();

    // Collect relationship types
    let rel_types: Vec<String> = store.edge_types().map(|s| s.to_string()).collect();

    // Collect property keys
    let property_keys: Vec<String> = store.property_index().properties().map(|s| s.to_string()).collect();

    // Build node_props map (label -> properties)
    // For now, we associate all properties with all labels since we don't track per-label properties
    let mut node_props: HashMap<String, Vec<String>> = HashMap::new();
    for label in &labels {
        node_props.insert(label.clone(), property_keys.clone());
    }

    // Build human-readable schema string for LLM context
    let mut schema_parts = Vec::new();

    if !labels.is_empty() {
        schema_parts.push(format!("Node labels: {}", labels.join(", ")));
    }

    if !rel_types.is_empty() {
        schema_parts.push(format!("Relationship types: {}", rel_types.join(", ")));
    }

    if !property_keys.is_empty() {
        schema_parts.push(format!("Node properties: {}", property_keys.join(", ")));
    }

    let schema = if schema_parts.is_empty() {
        "Empty graph - no schema information available.".to_string()
    } else {
        schema_parts.join("\n")
    };

    Json(SchemaResponse {
        schema,
        structured_schema: StructuredSchema {
            node_props,
            rel_types,
            property_keys,
        },
    })
}

async fn handle_query(
    State(state): State<Arc<AppState>>,
    Extension(user): Extension<AuthUser>,
    Json(req): Json<QueryRequest>,
) -> Result<Json<QueryResponse>, (StatusCode, Json<QueryResponse>)> {
    let start = std::time::Instant::now();

    // Check if this is a mutation query and verify write permission (Sprint 68)
    let is_mutation = is_mutation_query(&req.query);
    if is_mutation {
        if let Err(response) = check_permission(&user, true, false) {
            return Err((StatusCode::FORBIDDEN, Json(QueryResponse {
                success: false,
                result: None,
                error: Some(format!("Insufficient permissions: write access required for mutation queries")),
                execution_time_ms: start.elapsed().as_secs_f64() * 1000.0,
            })));
        }
    }

    // Audit log (Sprint 68)
    tracing::info!(
        target: "audit",
        user = %user.name,
        role = %user.role,
        query_type = if is_mutation { "mutation" } else { "read" },
        "Query executed"
    );

    println!("Executing query: {}", req.query);

    // 1. Parse to check type
    let stmt = match neural_parser::parse_statement(&req.query) {
        Ok(s) => s,
        Err(e) => return Ok(Json(QueryResponse {
            success: false,
            result: None,
            error: Some(format!("Parse Error: {}", e)),
            execution_time_ms: start.elapsed().as_secs_f64() * 1000.0,
        })),
    };

    // 2. Execute (using pre-parsed AST to avoid double parsing)
    let res = match stmt {
        Statement::Query(ref q) => {
            // Read-only lock - pass AST directly
            let store = state.store.read().await;
            match execute_query_from_ast(&store, q, None) {
                Ok(r) => Ok(StatementResult::Query(r)),
                Err(e) => Err(e),
            }
        },
        _ => {
            // Write lock - pass AST directly
            let mut store = state.store.write().await;
            execute_statement_from_ast(&mut store, stmt, None, &mut None)
        }
    };

    let elapsed = start.elapsed();
    let elapsed_ms = elapsed.as_secs_f64() * 1000.0;

    // Record query latency to metrics (Sprint 67)
    state.metrics.record_query_latency(elapsed);

    // 3. Format response
    match res {
        Ok(StatementResult::Query(q_res)) => {
            // Sprint 59 Optimization: Direct JSON serialization instead of format!("{:?}")
            // This reduces serialization overhead by ~5x
            let columns = q_res.columns().to_vec();
            let rows: Vec<HashMap<String, serde_json::Value>> = q_res.rows()
                .iter()
                .map(|row| {
                    columns.iter()
                        .map(|col_name| {
                            let val = row.get(col_name)
                                .map(|v| v.to_json())
                                .unwrap_or(serde_json::Value::Null);
                            (col_name.clone(), val)
                        })
                        .collect()
                })
                .collect();
            Ok(Json(QueryResponse {
                success: true,
                result: Some(serde_json::json!({
                    "type": "query",
                    "columns": columns,
                    "rows": rows,
                    "count": q_res.row_count()
                })),
                error: None,
                execution_time_ms: elapsed_ms,
            }))
        },
        Ok(StatementResult::Mutation(m_res)) => {
            // Increment mutation counter and trigger auto-save if threshold reached
            state.mutation_count.fetch_add(1, Ordering::SeqCst);
            maybe_save(&state).await;

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
            Ok(Json(QueryResponse {
                success: true,
                result: Some(result_json),
                error: None,
                execution_time_ms: elapsed_ms,
            }))
        },
        Ok(StatementResult::Explain(plan)) => {
            Ok(Json(QueryResponse {
                success: true,
                result: Some(serde_json::json!({
                    "type": "explain",
                    "plan": plan
                })),
                error: None,
                execution_time_ms: elapsed_ms,
            }))
        },
        Ok(StatementResult::TransactionStarted) => {
            Ok(Json(QueryResponse {
                success: true,
                result: Some(serde_json::json!({
                    "type": "transaction",
                    "status": "started",
                    "message": "Transaction started (Note: HTTP is stateless, tx scoped to request)"
                })),
                error: None,
                execution_time_ms: elapsed_ms,
            }))
        },
        Ok(StatementResult::TransactionCommitted) => {
            Ok(Json(QueryResponse {
                success: true,
                result: Some(serde_json::json!({
                    "type": "transaction",
                    "status": "committed"
                })),
                error: None,
                execution_time_ms: elapsed_ms,
            }))
        },
        Ok(StatementResult::TransactionRolledBack) => {
            Ok(Json(QueryResponse {
                success: true,
                result: Some(serde_json::json!({
                    "type": "transaction",
                    "status": "rolled_back"
                })),
                error: None,
                execution_time_ms: elapsed_ms,
            }))
        },
        Ok(StatementResult::Flashback { timestamp, tx_id }) => {
            Ok(Json(QueryResponse {
                success: true,
                result: Some(serde_json::json!({
                    "type": "flashback",
                    "timestamp": timestamp,
                    "tx_id": tx_id,
                    "message": "Database flashback completed"
                })),
                error: None,
                execution_time_ms: elapsed_ms,
            }))
        },
        Ok(StatementResult::Call { procedure, result }) => {
            Ok(Json(QueryResponse {
                success: true,
                result: Some(serde_json::json!({
                    "type": "call",
                    "procedure": procedure,
                    "result": result,
                })),
                error: None,
                execution_time_ms: elapsed_ms,
            }))
        },
        Err(e) => Ok(Json(QueryResponse {
            success: false,
            result: None,
            error: Some(format!("Execution Error: {}", e)),
            execution_time_ms: elapsed_ms,
        })),
    }
}

/// Handles bulk loading of CSV data using the fast builder pattern
/// Requires admin role (Sprint 68)
async fn handle_bulk_load(
    State(state): State<Arc<AppState>>,
    Extension(user): Extension<AuthUser>,
    Json(req): Json<BulkLoadRequest>,
) -> Result<Json<BulkLoadResponse>, (StatusCode, Json<BulkLoadResponse>)> {
    let start = std::time::Instant::now();

    // Require admin permission for bulk load (Sprint 68)
    if let Err(_) = check_permission(&user, false, true) {
        return Err((StatusCode::FORBIDDEN, Json(BulkLoadResponse {
            success: false,
            nodes_loaded: 0,
            edges_loaded: 0,
            error: Some("Admin role required for bulk load operations".to_string()),
            load_time_ms: start.elapsed().as_secs_f64() * 1000.0,
        })));
    }

    // Audit log (Sprint 68)
    tracing::info!(
        target: "audit",
        user = %user.name,
        role = %user.role,
        nodes_path = ?req.nodes_path,
        edges_path = ?req.edges_path,
        "Bulk load initiated"
    );

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
                return Ok(Json(BulkLoadResponse {
                    success: false,
                    nodes_loaded: 0,
                    edges_loaded: 0,
                    error: Some(format!("Failed to load nodes: {}", e)),
                    load_time_ms: start.elapsed().as_secs_f64() * 1000.0,
                }));
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
                return Ok(Json(BulkLoadResponse {
                    success: false,
                    nodes_loaded,
                    edges_loaded: 0,
                    error: Some(format!("Failed to load edges: {}", e)),
                    load_time_ms: start.elapsed().as_secs_f64() * 1000.0,
                }));
            }
        }
    }

    // Build and replace the store
    println!("  Building graph store...");
    let new_store = builder.build();

    {
        let mut store = state.store.write().await;
        *store = new_store;
    }

    // Trigger immediate save after bulk load (significant mutation)
    state.mutation_count.fetch_add(state.config.mutation_threshold, Ordering::SeqCst);
    maybe_save(&state).await;

    let elapsed = start.elapsed().as_secs_f64() * 1000.0;
    println!("  Bulk load completed in {:.2}ms", elapsed);

    Ok(Json(BulkLoadResponse {
        success: true,
        nodes_loaded,
        edges_loaded,
        error: None,
        load_time_ms: elapsed,
    }))
}

/// Starts the web server with automatic persistence using default config.
pub async fn run_server(port: u16) -> Result<(), Box<dyn std::error::Error>> {
    let config = PersistenceConfig::from_env();
    run_server_with_config(port, config).await
}

/// Starts the web server with a custom database path.
pub async fn run_server_with_path(port: u16, db_path: &str) -> Result<(), Box<dyn std::error::Error>> {
    let mut config = PersistenceConfig::from_env();
    config.db_path = PathBuf::from(db_path);
    run_server_with_config(port, config).await
}

/// Starts the web server with full configuration control.
pub async fn run_server_with_config(port: u16, config: PersistenceConfig) -> Result<(), Box<dyn std::error::Error>> {
    run_server_with_auth(port, config, AuthConfig::default()).await
}

/// Starts the web server with full configuration and auth control.
pub async fn run_server_with_auth(port: u16, config: PersistenceConfig, auth_config: AuthConfig) -> Result<(), Box<dyn std::error::Error>> {
    // Initialize structured logging (Sprint 67)
    if std::env::var("NGDB_LOG_JSON").is_ok() {
        neural_storage::logging::init_json();
    } else {
        neural_storage::logging::init();
    }

    // Load auth config from environment if not provided (Sprint 68)
    let auth_config = if auth_config.jwt_secret.is_empty() {
        load_auth_config_from_env()
    } else {
        auth_config
    };

    let state = Arc::new(load_data_with_auth(config.clone(), auth_config.clone())?);
    let app = create_router(state.clone());

    // Background periodic save task (non-blocking)
    let save_state = state.clone();
    let save_interval = Duration::from_secs(config.save_interval_secs);
    tokio::spawn(async move {
        let mut interval = tokio::time::interval(save_interval);
        loop {
            interval.tick().await;
            let count = save_state.mutation_count.swap(0, Ordering::SeqCst);
            if count > 0 {
                // Take snapshot under read lock (fast)
                let snapshot = {
                    let store = save_state.store.read().await;
                    store.snapshot()
                };
                // Lock released - save without blocking

                let path = save_state.config.db_path.clone();
                let backup_config = save_state.config.backup_config();

                // Blocking I/O in spawn_blocking
                let result = tokio::task::spawn_blocking(move || {
                    snapshot.save_with_backups(&path, &backup_config)
                }).await;

                match result {
                    Ok(Ok(())) => println!("Periodic save completed ({} mutations)", count),
                    Ok(Err(e)) => {
                        eprintln!("Periodic save failed: {}", e);
                        if e.is_critical() {
                            eprintln!("CRITICAL: Persistence failure may cause data loss!");
                        }
                    }
                    Err(e) => eprintln!("Periodic save task panicked: {}", e),
                }
            }
        }
    });

    // Create shutdown signal
    let shutdown_state = state.clone();

    let shutdown_signal = async move {
        tokio::signal::ctrl_c().await.ok();
        println!("\nShutdown signal received, draining connections...");

        // Perform final save
        println!("Performing final save...");
        let snapshot = {
            let store = shutdown_state.store.read().await;
            store.snapshot()
        };

        let path = shutdown_state.config.db_path.clone();
        let backup_config = shutdown_state.config.backup_config();

        match tokio::task::spawn_blocking(move || {
            snapshot.save_with_backups(&path, &backup_config)
        }).await {
            Ok(Ok(())) => println!("Final save completed successfully"),
            Ok(Err(e)) => eprintln!("Final save failed: {}", e),
            Err(e) => eprintln!("Final save task panicked: {}", e),
        }
    };

    let addr = format!("0.0.0.0:{}", port);
    println!("\n🚀 Server running at http://localhost:{}", port);
    println!("   Database: {}", config.db_path.display());
    println!("   Auto-save: every {}s or {} mutations", config.save_interval_secs, config.mutation_threshold);
    println!("   Backups: {} retained", config.backup_count);
    println!("   Shutdown timeout: {}s", config.shutdown_timeout_secs);
    // Sprint 68: Auth status
    if auth_config.enabled {
        println!("   Auth: enabled (JWT + {} API keys)", auth_config.api_keys.len());
    } else {
        println!("   Auth: disabled (all endpoints public)");
    }
    println!("   Press Ctrl+C to stop\n");

    let listener = tokio::net::TcpListener::bind(&addr).await?;

    // Serve with graceful shutdown
    axum::serve(listener, app)
        .with_graceful_shutdown(shutdown_signal)
        .await?;

    println!("Server stopped gracefully");
    Ok(())
}
