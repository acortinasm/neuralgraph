//! # NeuralGraphDB CLI
//!
//! Interactive command-line interface for NeuralGraphDB.
//!
//! ## Usage
//!
//! ```bash
//! # Interactive REPL mode (default)
//! neuralgraph
//!
//! # Run demo
//! neuralgraph --demo
//!
//! # Run benchmarks
//! neuralgraph --benchmark
//! ```

mod server;
mod flight;
mod raft_server;

use colored::Colorize;
use indicatif::{ProgressBar, ProgressStyle};
use neural_core::{Graph, NodeId, PropertyValue};
use neural_storage::{
    GraphStore, GraphStoreBuilder as GraphBuilder,
    csv_loader::{EdgeData, NodeData},
    hf_loader::{HfDataset, download_dataset, load_hf_dataset},
    load_edges_csv, load_nodes_csv,
};
use neural_executor::Value;
use arrow_flight::flight_service_server::FlightServiceServer;
use tonic::transport::Server;
use rustyline::DefaultEditor;
use rustyline::error::ReadlineError;
use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Instant;

// =============================================================================
// REPL State
// =============================================================================

/// State maintained during the REPL session.
/// Keeps track of loaded node/edge data separately so incremental loading works.
#[derive(Default)]
struct ReplState {
    /// The built graph store (for queries)
    store: Option<GraphStore>,
    /// Loaded node data (preserved across loads)
    nodes: Vec<NodeData>,
    /// Loaded edge data (preserved across loads)
    edges: Vec<EdgeData>,
    /// Session parameters
    params: HashMap<String, Value>,
    /// Active Transaction (if any)
    tx: Option<neural_storage::transaction::Transaction>,
}

impl ReplState {
    /// Rebuilds the GraphStore from the current node and edge data.
    fn rebuild(&mut self) {
        let mut builder = GraphBuilder::new();

        // Add all nodes
        for node in &self.nodes {
            if let Some(label) = &node.label {
                builder = builder.add_labeled_node(node.id, label.clone(), node.properties.clone());
            } else {
                builder = builder.add_node(node.id, node.properties.clone());
            }
        }

        for edge in &self.edges {
            if let Some(label) = &edge.label {
                builder = builder.add_labeled_edge(
                    edge.source,
                    edge.target,
                    neural_core::Label::new(label.clone()),
                );
            } else {
                builder = builder.add_edge(edge.source, edge.target);
            }
        }

        self.store = Some(builder.build());
    }

    /// Clears all data.
    fn clear(&mut self) {
        self.store = None;
        self.nodes.clear();
        self.edges.clear();
        self.params.clear();
        self.tx = None;
    }

    /// Returns the number of nodes loaded.
    fn node_count(&self) -> usize {
        self.nodes.len()
    }

    /// Returns the number of edges loaded.
    fn edge_count(&self) -> usize {
        self.edges.len()
    }
}

// =============================================================================
// Main Entry Point
// =============================================================================

fn main() {
    let args: Vec<String> = std::env::args().collect();

    match args.get(1).map(|s| s.as_str()) {
        Some("--demo") => run_demo(),
        Some("--benchmark") => run_benchmark(),
        Some("serve") | Some("--serve") => run_server(&args),
        Some("serve-flight") => run_flight_server_cli(&args),
        Some("serve-raft") => run_raft_server_cli(&args),
        Some("cluster") => run_cluster_cli(&args),
        Some("--help") | Some("-h") => print_help(),
        _ => run_repl(),
    }
}

fn print_help() {
    println!("{}", "NeuralGraphDB v0.8.0".bold().cyan());
    println!("The Database for AI Agents\n");
    println!("USAGE:");
    println!("    neuralgraph [OPTIONS]\n");
    println!("OPTIONS:");
    println!("    --demo         Run the demonstration with sample data");
    println!("    --benchmark    Download ML-ArXiv-Papers and run benchmarks");
    println!("    serve [PORT]   Start web server for ArXiv search (default port: 3000)");
    println!("    serve-flight   Start Arrow Flight gRPC server (default port: 50051)");
    println!("    serve-raft <NODE_ID> <PORT> [--join <SEED_ADDR>]");
    println!("                   Start Raft gRPC server");
    println!("    cluster info <ADDR>     Get cluster information");
    println!("    cluster health <ADDR>   Check cluster health");
    println!("    cluster add <ADDR> <NODE_ID> <NODE_ADDR>");
    println!("                            Add a node to the cluster");
    println!("    cluster remove <ADDR> <NODE_ID>");
    println!("                            Remove a node from the cluster");
    println!("    --help, -h     Show this help message\n");
    println!("Without options, NeuralGraphDB starts in interactive REPL mode.");
}

fn run_server(args: &[String]) {
    // Parse port from args (e.g., "serve 8080" or just "serve")
    let port: u16 = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(3000);

    print_banner();
    println!(
        "{}",
        format!("🌐 Starting ArXiv Search Server on port {}...", port).bold()
    );
    println!();

    // Create tokio runtime and run the server
    let rt = tokio::runtime::Runtime::new().expect("Failed to create Tokio runtime");
    if let Err(e) = rt.block_on(server::run_server(port)) {
        eprintln!("{}: {}", "Error".red(), e);
        std::process::exit(1);
    }
}

fn run_flight_server_cli(args: &[String]) {
    let port: u16 = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(50051);
    
    print_banner();
    println!(
        "{}",
        format!("✈️  Starting Arrow Flight Server on port {}...", port).bold()
    );
    println!("   Loading data...");

    // For now, load ArXiv data as default for Flight
    let cache_dir = dirs::cache_dir()
        .unwrap_or_else(|| PathBuf::from("."))
        .join("neuralgraph");
    
    let store = match load_hf_dataset(HfDataset::MlArxivPapers, Some(&cache_dir)) {
        Ok(s) => Arc::new(s),
        Err(_) => {
            println!("   {} No dataset found. Starting with empty graph.", "⚠".yellow());
            Arc::new(GraphStore::builder().build())
        }
    };

    println!("   {} Nodes: {}, Edges: {}", "✓".green(), store.node_count(), store.edge_count());

    let service = flight::NeuralFlightService::new(store);
    let svc = FlightServiceServer::new(service);

    let addr = format!("0.0.0.0:{}", port).parse().unwrap();
    println!("\n🚀 Flight Server listening on {}\n", addr);

    let rt = tokio::runtime::Runtime::new().expect("Failed to create Tokio runtime");
    rt.block_on(async {
        Server::builder()
            .add_service(svc)
            .serve(addr)
            .await
            .expect("Flight server failed");
    });
}

fn run_raft_server_cli(args: &[String]) {
    let node_id: u64 = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(1);
    let port: u16 = args.get(3).and_then(|s| s.parse().ok()).unwrap_or(50052);

    // Check for --join option
    let join_addr = args.iter().position(|s| s == "--join").and_then(|i| args.get(i + 1).cloned());

    print_banner();
    println!(
        "{}",
        format!("⚡️ Starting Raft Node {} on port {}...", node_id, port).bold()
    );
    if let Some(ref addr) = join_addr {
        println!("   Joining cluster via {}", addr.cyan());
    }
    println!();

    let rt = tokio::runtime::Runtime::new().expect("Failed to create Tokio runtime");
    rt.block_on(async {
        use neural_storage::raft::{default_raft_config, LogStore, GraphStateMachine, NeuralRaftStorage, NeuralRaftNetwork, ClusterManager};
        use openraft::{BasicNode, Raft};
        use crate::raft_server::RaftGrpcServer;
        use tonic::transport::Server;
        use std::collections::BTreeMap;

        // Create data directory for this node
        let data_dir = format!("raft-data-{}", node_id);
        let node_addr = format!("127.0.0.1:{}", port);

        // Create storage components
        let log_store = LogStore::new_arc(&data_dir);
        let state_machine = GraphStateMachine::new_arc();
        let storage = NeuralRaftStorage::new(log_store.clone(), state_machine.clone());
        let network = NeuralRaftNetwork::new();

        let (ls, sm) = openraft::storage::Adaptor::new(storage);

        // Create OpenRaft config
        let config = default_raft_config();
        let raft_config = Arc::new(config.validate().unwrap());

        // Create Raft node
        let raft = Raft::new(
            node_id,
            raft_config.clone(),
            network.clone(),
            ls,
            sm,
        ).await.expect("Failed to create Raft node");

        // Create cluster manager
        let cluster = Arc::new(ClusterManager::new(node_id, node_addr.clone(), Arc::new(raft.clone())));

        if let Some(seed_addr) = join_addr {
            // Join existing cluster
            println!("   Attempting to join cluster...");
            match cluster.join_cluster(&seed_addr).await {
                Ok(info) => {
                    println!("   {} Joined cluster successfully!", "✓".green());
                    if let Some(leader_id) = info.leader_id {
                        println!("   Leader: Node {} at {}", leader_id, info.leader_addr.unwrap_or_default());
                    }
                    println!("   Members: {}", info.members.len());
                }
                Err(e) => {
                    eprintln!("   {} Failed to join cluster: {}", "✗".red(), e);
                    eprintln!("   Starting as standalone node instead...");

                    // Initialize as single-node cluster
                    let mut members = BTreeMap::new();
                    members.insert(node_id, BasicNode { addr: node_addr.clone() });
                    raft.initialize(members).await.expect("Failed to initialize Raft node");
                }
            }
        } else {
            // Initialize as single-node cluster (leader)
            let mut members = BTreeMap::new();
            members.insert(node_id, BasicNode { addr: node_addr.clone() });
            raft.initialize(members).await.expect("Failed to initialize Raft node");
            println!("   {} Initialized as cluster leader", "✓".green());
        }

        // Start gRPC server
        let addr = format!("0.0.0.0:{}", port).parse().unwrap();
        let raft_service = RaftGrpcServer::with_cluster(raft.clone(), cluster.clone());
        println!("\n🚀 Raft Server listening on {}\n", addr);

        Server::builder()
            .add_service(neural_storage::raft::network::proto::raft_server::RaftServer::new(raft_service))
            .serve(addr)
            .await
            .expect("Raft server failed");
    });
}

/// Run cluster management CLI commands.
fn run_cluster_cli(args: &[String]) {
    let subcommand = args.get(2).map(|s| s.as_str()).unwrap_or("");

    match subcommand {
        "info" => {
            let addr = args.get(3).unwrap_or(&"127.0.0.1:50052".to_string()).clone();
            run_cluster_info(&addr);
        }
        "health" => {
            let addr = args.get(3).unwrap_or(&"127.0.0.1:50052".to_string()).clone();
            run_cluster_health(&addr);
        }
        "add" => {
            if args.len() < 6 {
                eprintln!("{}: Usage: cluster add <CLUSTER_ADDR> <NODE_ID> <NODE_ADDR>", "Error".red());
                std::process::exit(1);
            }
            let cluster_addr = &args[3];
            let node_id: u64 = args[4].parse().expect("Invalid node ID");
            let node_addr = &args[5];
            run_cluster_add(cluster_addr, node_id, node_addr);
        }
        "remove" => {
            if args.len() < 5 {
                eprintln!("{}: Usage: cluster remove <CLUSTER_ADDR> <NODE_ID>", "Error".red());
                std::process::exit(1);
            }
            let cluster_addr = &args[3];
            let node_id: u64 = args[4].parse().expect("Invalid node ID");
            run_cluster_remove(cluster_addr, node_id);
        }
        _ => {
            eprintln!("{}: Unknown cluster command '{}'. Use info, health, add, or remove.", "Error".red(), subcommand);
            std::process::exit(1);
        }
    }
}

fn run_cluster_info(addr: &str) {
    let rt = tokio::runtime::Runtime::new().expect("Failed to create Tokio runtime");
    rt.block_on(async {
        use neural_storage::raft::network::proto::raft_client::RaftClient;
        use neural_storage::raft::network::proto::ClusterInfoRequest;

        println!("\n{}", "Cluster Information".bold());
        println!("{}", "─".repeat(50));

        match RaftClient::connect(format!("http://{}", addr)).await {
            Ok(mut client) => {
                let request = tonic::Request::new(ClusterInfoRequest {});
                match client.get_cluster_info(request).await {
                    Ok(response) => {
                        let info = response.into_inner();
                        println!("  Term: {}", info.term.to_string().cyan());
                        println!("  Leader: Node {} at {}",
                            info.leader_id.to_string().green(),
                            info.leader_addr.cyan());
                        println!("\n  {}", "Members:".bold());
                        for member in &info.members {
                            let role = if member.is_leader { " (leader)" } else { "" };
                            println!("    Node {}: {}{}",
                                member.id,
                                member.addr,
                                role.green());
                        }
                        println!();
                    }
                    Err(e) => {
                        eprintln!("{}: Failed to get cluster info: {}", "Error".red(), e);
                    }
                }
            }
            Err(e) => {
                eprintln!("{}: Failed to connect to {}: {}", "Error".red(), addr, e);
            }
        }
    });
}

fn run_cluster_health(addr: &str) {
    let rt = tokio::runtime::Runtime::new().expect("Failed to create Tokio runtime");
    rt.block_on(async {
        use neural_storage::raft::network::proto::raft_client::RaftClient;
        use neural_storage::raft::network::proto::{ClusterInfoRequest, HealthCheckRequest};

        println!("\n{}", "Cluster Health".bold());
        println!("{}", "─".repeat(50));

        match RaftClient::connect(format!("http://{}", addr)).await {
            Ok(mut client) => {
                // First get cluster info to know all members
                let request = tonic::Request::new(ClusterInfoRequest {});
                match client.get_cluster_info(request).await {
                    Ok(response) => {
                        let info = response.into_inner();

                        for member in &info.members {
                            print!("  Node {}: ", member.id);

                            // Try to health check each member
                            match RaftClient::connect(format!("http://{}", member.addr)).await {
                                Ok(mut member_client) => {
                                    let health_req = tonic::Request::new(HealthCheckRequest {
                                        node_id: member.id
                                    });
                                    match member_client.health_check(health_req).await {
                                        Ok(health_resp) => {
                                            let health = health_resp.into_inner();
                                            let status = if health.healthy {
                                                "healthy".green()
                                            } else {
                                                "unhealthy".red()
                                            };
                                            println!("{} | state: {} | term: {} | log: {}",
                                                status,
                                                health.state,
                                                health.term,
                                                health.last_log_index);
                                        }
                                        Err(_) => {
                                            println!("{}", "unreachable".red());
                                        }
                                    }
                                }
                                Err(_) => {
                                    println!("{}", "unreachable".red());
                                }
                            }
                        }
                        println!();
                    }
                    Err(e) => {
                        eprintln!("{}: Failed to get cluster info: {}", "Error".red(), e);
                    }
                }
            }
            Err(e) => {
                eprintln!("{}: Failed to connect to {}: {}", "Error".red(), addr, e);
            }
        }
    });
}

fn run_cluster_add(cluster_addr: &str, node_id: u64, node_addr: &str) {
    let rt = tokio::runtime::Runtime::new().expect("Failed to create Tokio runtime");
    let node_addr = node_addr.to_string();
    rt.block_on(async {
        use neural_storage::raft::network::proto::raft_client::RaftClient;
        use neural_storage::raft::network::proto::JoinRequest;

        println!("\n{}", "Adding Node to Cluster".bold());
        println!("{}", "─".repeat(50));
        println!("  Target: {} -> {}", node_id, node_addr);

        match RaftClient::connect(format!("http://{}", cluster_addr)).await {
            Ok(mut client) => {
                let request = tonic::Request::new(JoinRequest {
                    node_id,
                    addr: node_addr,
                });
                match client.join(request).await {
                    Ok(response) => {
                        let resp = response.into_inner();
                        if resp.success {
                            println!("  {} Node {} added successfully!", "✓".green(), node_id);
                            println!("  Current members: {}", resp.members.len());
                        } else {
                            if resp.leader_id != 0 {
                                eprintln!("  {} Not the leader. Try: {}", "✗".red(), resp.leader_addr);
                            } else {
                                eprintln!("  {} Failed: {}", "✗".red(), resp.error);
                            }
                        }
                    }
                    Err(e) => {
                        eprintln!("{}: Failed to add node: {}", "Error".red(), e);
                    }
                }
            }
            Err(e) => {
                eprintln!("{}: Failed to connect to {}: {}", "Error".red(), cluster_addr, e);
            }
        }
        println!();
    });
}

fn run_cluster_remove(cluster_addr: &str, node_id: u64) {
    let rt = tokio::runtime::Runtime::new().expect("Failed to create Tokio runtime");
    rt.block_on(async {
        use neural_storage::raft::network::proto::raft_client::RaftClient;
        use neural_storage::raft::network::proto::ClusterInfoRequest;

        println!("\n{}", "Removing Node from Cluster".bold());
        println!("{}", "─".repeat(50));
        println!("  Target: Node {}", node_id);

        // Note: For a full implementation, we'd need a RemoveMember RPC
        // For now, we show cluster info and note the limitation
        match RaftClient::connect(format!("http://{}", cluster_addr)).await {
            Ok(mut client) => {
                let request = tonic::Request::new(ClusterInfoRequest {});
                match client.get_cluster_info(request).await {
                    Ok(response) => {
                        let info = response.into_inner();
                        if info.members.iter().any(|m| m.id == node_id) {
                            println!("  {} Node {} found in cluster", "!".yellow(), node_id);
                            println!("  Note: Use leader's Raft API to remove membership");
                            // In a full implementation, this would call a RemoveMember RPC
                        } else {
                            println!("  {} Node {} not found in cluster", "?".yellow(), node_id);
                        }
                    }
                    Err(e) => {
                        eprintln!("{}: Failed to get cluster info: {}", "Error".red(), e);
                    }
                }
            }
            Err(e) => {
                eprintln!("{}: Failed to connect to {}: {}", "Error".red(), cluster_addr, e);
            }
        }
        println!();
    });
}

// =============================================================================
// Interactive REPL
// =============================================================================

fn run_repl() {
    print_banner();

    let mut rl = DefaultEditor::new().expect("Failed to initialize readline");
    let mut state = ReplState::default();

    println!(
        "{}",
        "Type :help for commands, or enter NGQL queries.".dimmed()
    );
    println!();

    loop {
        let prompt = "ngql> ".green().bold().to_string();
        match rl.readline(&prompt) {
            Ok(line) => {
                let line = line.trim();
                if line.is_empty() {
                    continue;
                }

                let _ = rl.add_history_entry(line);

                if line.starts_with(':') {
                    if !handle_command(line, &mut state) {
                        break;
                    }
                } else {
                    execute_statement_cli(line, &mut state);
                }
            }
            Err(ReadlineError::Interrupted) => {
                println!("{}", "Use :quit to exit".dimmed());
            }
            Err(ReadlineError::Eof) => {
                println!("\n{}", "Goodbye! 👋".cyan());
                break;
            }
            Err(err) => {
                eprintln!("{}: {:?}", "Error".red(), err);
                break;
            }
        }
    }
}

fn print_banner() {
    println!();
    println!(
        "{}",
        "╔══════════════════════════════════════════════════════════════╗".cyan()
    );
    println!(
        "{}",
        "║             🧠 NeuralGraphDB v0.1.0                          ║".cyan()
    );
    println!(
        "{}",
        "║          The Database for AI Agents                          ║".cyan()
    );
    println!(
        "{}",
        "╚══════════════════════════════════════════════════════════════╝".cyan()
    );
    println!();
}

/// Handles REPL commands (starting with :)
/// Returns false if the REPL should exit
fn handle_command(line: &str, state: &mut ReplState) -> bool {
    let parts: Vec<&str> = line.split_whitespace().collect();
    let cmd = parts.first().map(|s| s.to_lowercase()).unwrap_or_default();

    match cmd.as_str() {
        ":quit" | ":exit" | ":q" => {
            println!("{}", "Goodbye! 👋".cyan());
            return false;
        }
        ":help" | ":h" | ":?" => {
            print_repl_help();
        }
        ":stats" => {
            print_stats(&state.store);
        }
        ":clear" => {
            state.clear();
            println!("{}", "✓ Graph cleared".green());
        }
        ":load" => {
            if parts.len() < 3 {
                println!(
                    "{}: Usage: :load <nodes|edges|hf> <path/dataset>",
                    "Error".red()
                );
            } else {
                handle_load(&parts[1..], state);
            }
        }
        ":save" => {
            if parts.len() < 2 {
                println!("{}: Usage: :save <file.ngdb>", "Error".red());
            } else {
                handle_save(parts[1], state);
            }
        }
        ":loadbin" => {
            if parts.len() < 2 {
                println!("{}: Usage: :loadbin <file.ngdb>", "Error".red());
            } else {
                handle_loadbin(parts[1], state);
            }
        }
        ":benchmark" => {
            run_benchmark_on_store(&state.store);
        }
        ":demo" => {
            state.clear();
            state.store = Some(create_demo_graph());
            println!("{}", "✓ Demo graph loaded (4 nodes, 5 edges)".green());
        }
        ":cluster" => {
            handle_cluster(state);
        }
        ":param" => {
            if parts.len() < 2 {
                // List params
                println!("{}", "Current Parameters:".bold());
                if state.params.is_empty() {
                    println!("  (none)");
                } else {
                    for (k, v) in &state.params {
                        println!("  ${}: {}", k, v);
                    }
                }
            } else if parts.len() < 3 {
                println!("{}: Usage: :param <name> <value>", "Error".red());
            } else {
                let name = parts[1];
                let val_str = parts[2..].join(" "); // handle strings with spaces? simplified for now
                
                // Parse value
                let val = if val_str == "true" {
                    Value::Bool(true)
                } else if val_str == "false" {
                    Value::Bool(false)
                } else if val_str == "null" {
                    Value::Null
                } else if let Ok(i) = val_str.parse::<i64>() {
                    Value::Int(i)
                } else if let Ok(f) = val_str.parse::<f64>() {
                    Value::Float(f)
                } else {
                    // Treat as string, remove quotes if present
                    let s = val_str.trim_matches('"').trim_matches('‘').to_string();
                    Value::String(s)
                };
                
                state.params.insert(name.to_string(), val);
                println!("{} Set param ${}", "✓".green(), name);
            }
        }
        _ => {
            println!(
                "{}: Unknown command '{}'. Type :help for available commands.",
                "Error".red(),
                cmd
            );
        }
    }
    true
}

/// Saves the current graph to a binary file
fn handle_save(path: &str, state: &mut ReplState) {
    match &mut state.store {
        Some(store) => {
            let start = std::time::Instant::now();
            match store.save_binary(path) {
                Ok(()) => {
                    let elapsed = start.elapsed();
                    let file_size = std::fs::metadata(path).map(|m| m.len()).unwrap_or(0);
                    println!(
                        "{} Graph saved to {} ({} bytes) in {:.2}ms",
                        "✓".green(),
                        path.cyan(),
                        file_size,
                        elapsed.as_secs_f64() * 1000.0
                    );
                }
                Err(e) => {
                    println!("{}: Failed to save graph: {}", "Error".red(), e);
                }
            }
        }
        None => {
            println!(
                "{}",
                "No graph loaded. Use :load or :demo to load data first.".yellow()
            );
        }
    }
}

/// Loads a graph from a binary file
fn handle_loadbin(path: &str, state: &mut ReplState) {
    let start = std::time::Instant::now();
    match GraphStore::load_binary(path) {
        Ok(store) => {
            let elapsed = start.elapsed();
            let node_count = store.node_count();
            let edge_count = store.edge_count();
            state.store = Some(store);
            state.nodes.clear();
            state.edges.clear();
            println!(
                "{} Loaded {} nodes, {} edges from {} in {:.2}ms",
                "✓".green(),
                node_count.to_string().cyan(),
                edge_count.to_string().cyan(),
                path.cyan(),
                elapsed.as_secs_f64() * 1000.0
            );
        }
        Err(e) => {
            println!("{}: Failed to load graph: {}", "Error".red(), e);
        }
    }
}

/// Runs Leiden community detection on the graph
fn handle_cluster(state: &mut ReplState) {
    match &state.store {
        Some(store) => {
            let node_count = store.node_count();
            let edge_count = store.edge_count();

            // Warn for large graphs
            if node_count > 100_000 {
                println!(
                    "{}: Large graph ({} nodes, {} edges). This may take several minutes...",
                    "Warning".yellow(),
                    node_count.to_string().cyan(),
                    edge_count.to_string().cyan()
                );
            }

            println!("\n{}", "🔍 Leiden Community Detection".bold());
            println!("{}", "─".repeat(50));

            // Phase 1: Building graph structure
            println!("  [1/3] {} Building graph structure...", "▶".cyan());
            let start = std::time::Instant::now();
            std::io::Write::flush(&mut std::io::stdout()).ok();

            // Show graph stats
            println!(
                "        Nodes: {}, Edges: {}",
                node_count.to_string().green(),
                edge_count.to_string().green()
            );
            std::io::Write::flush(&mut std::io::stdout()).ok();

            // Phase 2: Running algorithm
            println!("  [2/3] {} Running Leiden algorithm...", "▶".cyan());
            println!("        (optimizing modularity, may take a while)");
            std::io::Write::flush(&mut std::io::stdout()).ok();

            let communities = store.detect_communities();
            let elapsed = start.elapsed();

            // Phase 3: Collecting results
            println!("  [3/3] {} Collecting community assignments...", "▶".cyan());
            std::io::Write::flush(&mut std::io::stdout()).ok();

            // Get community statistics
            let num_communities = communities.num_communities();
            let sizes: Vec<_> = communities.community_sizes().collect();

            println!(
                "\n{} Completed in {:.2}s\n",
                "✓".green(),
                elapsed.as_secs_f64()
            );

            // Show top 10 communities by size
            let mut sorted_sizes: Vec<_> = sizes.clone();
            sorted_sizes.sort_by(|a, b| b.1.cmp(&a.1));

            println!(
                "{} {} communities detected",
                "📊".cyan(),
                num_communities.to_string().bold()
            );
            println!();
            println!("{}", "Top Communities by Size:".bold());
            println!("{}", "─".repeat(50));
            for (community_id, size) in sorted_sizes.iter().take(10) {
                let pct = (*size as f64 / store.node_count() as f64) * 100.0;
                let bar_len = ((pct / 100.0) * 30.0).min(30.0) as usize;
                let bar = "█".repeat(bar_len);
                println!(
                    "  C{:>4}: {:>8} nodes ({:>5.1}%) {}",
                    community_id,
                    size.to_string().green(),
                    pct,
                    bar.cyan()
                );
            }

            if sorted_sizes.len() > 10 {
                println!("  ... and {} more communities", sorted_sizes.len() - 10);
            }
            println!();
        }
        None => {
            println!(
                "{}",
                "No graph loaded. Use :load or :demo to load data first.".yellow()
            );
        }
    }
}

fn print_repl_help() {
    println!("\n{}", "Available Commands:".bold());
    println!("  {}           Show this help", ":help".cyan());
    println!("  {}          Show graph statistics", ":stats".cyan());
    println!("  {}          Clear the current graph", ":clear".cyan());
    println!("  {}           Load demo graph", ":demo".cyan());
    println!("  {} Run benchmarks on current graph", ":benchmark".cyan());
    println!("  {}           Exit the REPL", ":quit".cyan());
    println!();
    println!("{}", "Data Loading:".bold());
    println!("  {} Load nodes from CSV", ":load nodes <file>".cyan());
    println!("  {} Load edges from CSV", ":load edges <file>".cyan());
    println!(
        "  {}    Load HuggingFace dataset",
        ":load hf <dataset>".cyan()
    );
    println!();
    println!("{}", "Persistence (Binary):".bold());
    println!("  {}      Save graph to binary file", ":save <file>".cyan());
    println!(
        "  {}   Load graph from binary file",
        ":loadbin <file>".cyan()
    );
    println!();
    println!("{}", "GraphRAG Features:".bold());
    println!(
        "  {}        Run Leiden community detection",
        ":cluster".cyan()
    );
    println!();
    println!("{}", "Example Queries:".bold());
    println!("  MATCH (n) RETURN n");
    println!("  MATCH (n:Person) RETURN n.name, n.age");
    println!("  MATCH (a)-[]->(b) RETURN a.name, b.name");
    println!("  MATCH (n) RETURN COUNT(*)");
    println!();
}

fn print_stats(store: &Option<GraphStore>) {
    match store {
        Some(s) => {
            println!("\n{}", "Graph Statistics:".bold());
            println!("  Nodes: {}", s.node_count().to_string().green());
            println!("  Edges: {}", s.edge_count().to_string().green());
            // TODO: Add label counts, memory usage
            println!();
        }
        None => {
            println!(
                "{}",
                "No graph loaded. Use :load or :demo to load data.".yellow()
            );
        }
    }
}

fn handle_load(args: &[&str], state: &mut ReplState) {
    let load_type = args[0].to_lowercase();
    let path = args[1];

    match load_type.as_str() {
        "nodes" => match load_nodes_csv(path) {
            Ok(nodes) => {
                let count = nodes.len();
                // Add to existing nodes (preserving edges)
                state.nodes.extend(nodes);
                // Rebuild the store with all data
                state.rebuild();
                println!(
                    "{} Loaded {} nodes (total: {} nodes, {} edges)",
                    "✓".green(),
                    count.to_string().cyan(),
                    state.node_count().to_string().cyan(),
                    state.edge_count().to_string().cyan()
                );
            }
            Err(e) => {
                println!("{}: Failed to load nodes: {}", "Error".red(), e);
            }
        },
        "edges" => {
            match load_edges_csv(path) {
                Ok(edges) => {
                    let count = edges.len();
                    // Add to existing edges (preserving nodes)
                    state.edges.extend(edges);
                    // Rebuild the store with all data
                    state.rebuild();
                    println!(
                        "{} Loaded {} edges (total: {} nodes, {} edges)",
                        "✓".green(),
                        count.to_string().cyan(),
                        state.node_count().to_string().cyan(),
                        state.edge_count().to_string().cyan()
                    );
                }
                Err(e) => {
                    println!("{}: Failed to load edges: {}", "Error".red(), e);
                }
            }
        }
        "hf" => {
            load_huggingface_dataset(path, &mut state.store);
        }
        _ => {
            println!(
                "{}: Unknown load type '{}'. Use nodes, edges, or hf.",
                "Error".red(),
                load_type
            );
        }
    }
}

fn load_huggingface_dataset(dataset_name: &str, store: &mut Option<GraphStore>) {
    let dataset = match dataset_name.to_lowercase().as_str() {
        "arxiv" | "ml-arxiv" | "ml-arxiv-papers" | "cshorten/ml-arxiv-papers" => {
            HfDataset::MlArxivPapers
        }
        _ => {
            println!(
                "{}: Unknown dataset '{}'. Supported: arxiv, ml-arxiv-papers",
                "Error".red(),
                dataset_name
            );
            return;
        }
    };

    println!("{}", "⠋ Downloading from HuggingFace...".dimmed());

    let pb = ProgressBar::new_spinner();
    pb.set_style(
        ProgressStyle::default_spinner()
            .template("{spinner:.cyan} {msg}")
            .unwrap(),
    );
    pb.set_message("Downloading dataset...");
    pb.enable_steady_tick(std::time::Duration::from_millis(100));

    let cache_dir = dirs::cache_dir()
        .unwrap_or_else(|| PathBuf::from("."))
        .join("neuralgraph");

    // Download with progress
    match download_dataset(&dataset, &cache_dir, None) {
        Ok(_path) => {
            pb.set_message("Loading Parquet file...");

            match load_hf_dataset(dataset, Some(&cache_dir)) {
                Ok(graph) => {
                    pb.finish_and_clear();
                    let node_count = graph.node_count();
                    *store = Some(graph);
                    println!(
                        "{} Loaded {} papers from HuggingFace",
                        "✓".green(),
                        node_count.to_string().cyan()
                    );
                }
                Err(e) => {
                    pb.finish_and_clear();
                    println!("{}: Failed to parse dataset: {}", "Error".red(), e);
                }
            }
        }
        Err(e) => {
            pb.finish_and_clear();
            println!("{}: Failed to download dataset: {}", "Error".red(), e);
        }
    }
}

fn execute_statement_cli(input: &str, state: &mut ReplState) {
    // Ensure we have a store (create empty one if needed for CREATE)
    if state.store.is_none() {
        // Check if this is a mutation (doesn't need existing data)
        let upper = input.trim().to_uppercase();
        if upper.starts_with("CREATE") || upper.starts_with("BEGIN") {
            state.store = Some(GraphStore::builder().build());
        } else {
            println!(
                "{}",
                "No graph loaded. Use :load or :demo to load data first.".yellow()
            );
            return;
        }
    }

    let store = state.store.as_mut().unwrap();
    let start = Instant::now();

    // Use parameterized execution
    let params = if state.params.is_empty() {
        None
    } else {
        Some(&state.params)
    };

    match neural_executor::execute_statement_with_params(store, input, params, &mut state.tx) {
        Ok(result) => {
            let elapsed = start.elapsed();
            println!();
            match result {
                neural_executor::StatementResult::Query(query_result) => {
                    print!("{}", query_result);
                    println!(
                        "{} row(s) in {:.2}ms\n",
                        query_result.row_count().to_string().green(),
                        elapsed.as_secs_f64() * 1000.0
                    );
                }
                neural_executor::StatementResult::Mutation(mutation_result) => {
                    println!("{}", format!("{}", mutation_result).green());
                    println!("in {:.2}ms\n", elapsed.as_secs_f64() * 1000.0);
                }
                neural_executor::StatementResult::Explain(plan) => {
                    println!("{}", "Execution Plan:".bold());
                    println!("{}", plan);
                    println!();
                }
                neural_executor::StatementResult::TransactionStarted => {
                    println!("{} Transaction started", "✓".green());
                }
                neural_executor::StatementResult::TransactionCommitted => {
                    println!("{} Transaction committed", "✓".green());
                }
                neural_executor::StatementResult::TransactionRolledBack => {
                    println!("{} Transaction rolled back", "✓".yellow());
                }
            }
        }
        Err(e) => {
            println!("{}: {}", "Query Error".red(), e);
        }
    }
}

fn create_demo_graph() -> GraphStore {
    GraphStore::builder()
        .add_labeled_node(
            0u64,
            "Person",
            [
                ("name", PropertyValue::from("Alice")),
                ("age", PropertyValue::from(30i64)),
            ],
        )
        .add_labeled_node(
            1u64,
            "Person",
            [
                ("name", PropertyValue::from("Bob")),
                ("age", PropertyValue::from(25i64)),
            ],
        )
        .add_labeled_node(
            2u64,
            "Person",
            [
                ("name", PropertyValue::from("Charlie")),
                ("age", PropertyValue::from(35i64)),
            ],
        )
        .add_labeled_node(3u64, "Company", [("name", PropertyValue::from("Acme"))])
        .add_edge(0u64, 1u64) // Alice -> Bob
        .add_edge(0u64, 2u64) // Alice -> Charlie
        .add_edge(1u64, 2u64) // Bob -> Charlie
        .add_edge(0u64, 3u64) // Alice -> Acme
        .add_edge(1u64, 3u64) // Bob -> Acme
        .build()
}

// =============================================================================
// Benchmark Mode
// =============================================================================

fn run_benchmark() {
    print_banner();
    println!(
        "{}",
        "📦 Benchmark Mode: Loading ML-ArXiv-Papers from HuggingFace...".bold()
    );
    println!();

    let mut store: Option<GraphStore> = None;
    load_huggingface_dataset("ml-arxiv-papers", &mut store);

    if store.is_some() {
        run_benchmark_on_store(&store);
    }
}

fn run_benchmark_on_store(store: &Option<GraphStore>) {
    let Some(s) = store else {
        println!("{}", "No graph loaded. Load a dataset first.".yellow());
        return;
    };

    println!("\n{}", "⚡ Running Benchmarks...".bold());
    println!();

    let warmup = 10;
    let iterations = 100;

    // Benchmark 1: Node lookup by ID
    let mut total_ns = 0u128;
    for i in 0..warmup + iterations {
        let node_id = NodeId::new((i as u64) % (s.node_count() as u64));
        let start = Instant::now();
        let _ = s.get_property(node_id, "title");
        if i >= warmup {
            total_ns += start.elapsed().as_nanos();
        }
    }
    let node_lookup_ns = total_ns / iterations as u128;

    // Benchmark 2: Property access
    let mut total_ns = 0u128;
    for i in 0..warmup + iterations {
        let node_id = NodeId::new((i as u64) % (s.node_count() as u64));
        let start = Instant::now();
        let _ = s.get_property(node_id, "title");
        let _ = s.get_property(node_id, "abstract");
        if i >= warmup {
            total_ns += start.elapsed().as_nanos();
        }
    }
    let property_access_ns = total_ns / iterations as u128;

    // Benchmark 3: COUNT(*) query
    let start = Instant::now();
    let _ = neural_executor::execute_query(s, "MATCH (n) RETURN COUNT(*)");
    let count_query_ms = start.elapsed().as_secs_f64() * 1000.0;

    // Benchmark 4: Label filter query
    let start = Instant::now();
    let _ = neural_executor::execute_query(s, "MATCH (n:Paper) RETURN COUNT(*)");
    let label_filter_ms = start.elapsed().as_secs_f64() * 1000.0;

    // Print results
    println!(
        "{}",
        "┌─────────────────────────┬──────────────┬──────────┐".dimmed()
    );
    println!(
        "{}",
        "│ Operation               │ Time         │ Target   │".bold()
    );
    println!(
        "{}",
        "├─────────────────────────┼──────────────┼──────────┤".dimmed()
    );

    print_benchmark_row(
        "Node lookup by ID",
        &format!("{} ns", node_lookup_ns),
        "<100 ns",
        node_lookup_ns < 100,
    );
    print_benchmark_row(
        "Property access (2x)",
        &format!("{} ns", property_access_ns),
        "<200 ns",
        property_access_ns < 200,
    );
    print_benchmark_row(
        "COUNT(*) query",
        &format!("{:.2} ms", count_query_ms),
        "<50 ms",
        count_query_ms < 50.0,
    );
    print_benchmark_row(
        "Label filter (Paper)",
        &format!("{:.2} ms", label_filter_ms),
        "<50 ms",
        label_filter_ms < 50.0,
    );

    println!(
        "{}",
        "└─────────────────────────┴──────────────┴──────────┘".dimmed()
    );
    println!();

    let all_pass = node_lookup_ns < 100
        && property_access_ns < 200
        && count_query_ms < 50.0
        && label_filter_ms < 50.0;

    if all_pass {
        println!("{}", "✅ All benchmarks within target!".green().bold());
    } else {
        println!("{}", "⚠️  Some benchmarks above target".yellow());
    }
    println!();
}

fn print_benchmark_row(name: &str, time: &str, target: &str, passed: bool) {
    let status = if passed { "✓".green() } else { "✗".red() };
    println!("│ {:<23} │ {:>12} │ {:>8} │ {}", name, time, target, status);
}

// =============================================================================
// Demo Mode (Legacy)
// =============================================================================

fn run_demo() {
    print_banner();
    println!("{}", "Running demonstration...".bold());
    println!();

    // Demo 1: Simple social network graph
    demo_social_network();

    // Demo 2: Large-scale graph
    demo_large_graph();

    // Demo 3: Property values
    demo_property_values();

    // Demo 4: NGQL Parser
    demo_ngql_parser();

    // Demo 5: Query Execution
    demo_query_execution();

    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!(" {} All demos completed successfully!", "✅".green());
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
}

/// Demo 1: Create a simple social network graph
fn demo_social_network() {
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!(" 📊 Demo 1: Social Network Graph");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!();

    let names = ["Alice", "Bob", "Charlie", "Diana", "Eve"];
    let mut builder = GraphBuilder::new();

    builder = builder.add_labeled_edge(0u64, 1u64, neural_core::Label::new("KNOWS"));
    builder = builder.add_labeled_edge(0u64, 2u64, neural_core::Label::new("KNOWS"));
    builder = builder.add_labeled_edge(1u64, 2u64, neural_core::Label::new("KNOWS"));
    builder = builder.add_labeled_edge(1u64, 3u64, neural_core::Label::new("KNOWS"));
    builder = builder.add_labeled_edge(2u64, 3u64, neural_core::Label::new("KNOWS"));
    builder = builder.add_labeled_edge(3u64, 4u64, neural_core::Label::new("KNOWS"));
    builder = builder.add_labeled_edge(4u64, 0u64, neural_core::Label::new("KNOWS"));

    let graph = builder.build();

    println!("Graph created:");
    println!("  • Nodes: {}", graph.node_count());
    println!("  • Edges: {}", graph.edge_count());
    println!();

    println!("Adjacency list:");
    for i in 0..graph.node_count() {
        let node_id = NodeId::new(i as u64);
        let neighbors: Vec<_> = graph.neighbors(node_id).collect();
        let neighbor_names: Vec<_> = neighbors.iter().map(|n| names[n.as_usize()]).collect();
        println!(
            "  {} ({}) → {:?}",
            names[i],
            graph.out_degree(node_id),
            neighbor_names
        );
    }
    println!();

    let stats = graph.stats();
    println!("Graph Statistics:");
    println!("  • Nodes: {}", stats.node_count);
    println!("  • Edges: {}", stats.edge_count);
    println!("  • Max out-degree: {}", stats.max_degree);
    println!("  • Min out-degree: {}", stats.min_degree);
    println!("  • Avg out-degree: {:.2}", stats.avg_degree);
    println!("  • Memory usage: {} bytes", stats.memory_bytes);
    println!();
}

/// Demo 2: Large-scale graph performance
fn demo_large_graph() {
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!(" ⚡ Demo 2: Large-Scale Graph Performance");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!();

    let num_nodes = 100_000;
    let edges_per_node = 10;
    let total_edges = num_nodes * edges_per_node;

    println!(
        "Creating graph with {} nodes and {} edges...",
        num_nodes,
        total_edges
    );

    let start = Instant::now();
    let mut builder = GraphBuilder::with_capacity(total_edges);

    for i in 0..num_nodes as u64 {
        for j in 1..=edges_per_node as u64 {
            let target = (i + j) % num_nodes as u64;
            builder = builder.add_edge(i, target);
        }
    }

    let graph = builder.build();
    let build_time = start.elapsed();

    println!("  ✓ Build time: {:?}", build_time);
    println!();

    assert!(graph.validate().is_ok());
    println!("  ✓ Graph validation passed");

    let iterations = 100_000;
    let start = Instant::now();

    let mut total_neighbors = 0usize;
    for i in 0..iterations {
        let node = NodeId::new((i % num_nodes) as u64);
        total_neighbors += graph.neighbors(node).count();
    }

    let elapsed = start.elapsed();
    let avg_ns = elapsed.as_nanos() / iterations as u128;

    println!();
    println!("Neighbor access benchmark ({} iterations):", iterations);
    println!("  • Total time: {:?}", elapsed);
    println!("  • Avg per call: {} ns", avg_ns);
    println!("  • Target: <100 ns");
    println!(
        "  • Status: {}",
        if avg_ns < 100 {
            "✅ PASS".green()
        } else {
            "⚠️  Above target".yellow()
        }
    );
    println!();

    let stats = graph.stats();
    println!("Graph Statistics:");
    println!("  • Nodes: {}", stats.node_count);
    println!("  • Edges: {}", stats.edge_count);
    println!(
        "  • Memory: {:.2} MB",
        stats.memory_bytes as f64 / 1_000_000.0
    );
    println!();

    assert!(total_neighbors > 0);
}

/// Demo 3: Property values
fn demo_property_values() {
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!(" 🏷️  Demo 3: Property Values (Schema-Flexible)");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!();

    let properties: Vec<(&str, PropertyValue)> = vec![
        ("name", PropertyValue::from("Alice")),
        ("age", PropertyValue::from(30i64)),
        ("score", PropertyValue::from(0.95f64)),
        ("active", PropertyValue::from(true)),
        (
            "embedding",
            PropertyValue::from(vec![0.1f32, 0.2, 0.3, 0.4]),
        ),
        ("optional", PropertyValue::Null),
    ];

    println!("Node properties example:");
    for (key, value) in &properties {
        let type_name = match value {
            PropertyValue::Null => "Null",
            PropertyValue::Bool(_) => "Bool",
            PropertyValue::Int(_) => "Int",
            PropertyValue::Float(_) => "Float",
            PropertyValue::String(_) => "String",
            PropertyValue::Date(_) => "Date",
            PropertyValue::DateTime(_) => "DateTime",
            PropertyValue::Vector(_) => "Vector",
        };
        println!("  {:12} : {:8} = {:?}", key, type_name, value);
    }
    println!();

    println!("Type-safe access:");
    if let Some(name) = properties[0].1.as_str() {
        println!("  name.as_str() = \"{}\"", name);
    }
    if let Some(age) = properties[1].1.as_int() {
        println!("  age.as_int() = {}", age);
    }
    if let Some(embedding) = properties[4].1.as_vector() {
        println!(
            "  embedding.as_vector() = {:?} (len={})",
            embedding,
            embedding.len()
        );
    }
    println!();

    let json = serde_json::to_string_pretty(&properties[4].1).unwrap();
    println!("JSON serialization of embedding:");
    println!("{}", json);
    println!();
}

/// Demo 4: NGQL Parser
fn demo_ngql_parser() {
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!(" 🔤 Demo 4: NGQL Parser");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!();

    let queries = [
        "MATCH (n) RETURN n",
        "MATCH (n:Person) RETURN n",
        "MATCH (a)-[:KNOWS]->(b) RETURN a, b",
        "MATCH (a:Person)-[:KNOWS]->(b:Person) WHERE a.age > 30 RETURN a.name, b.name",
        r###"MATCH (p:Person) WHERE p.name = "Alice" RETURN p"###,
    ];

    for query in queries {
        println!("Query: {}", query);

        match neural_parser::parse_query(query) {
            Ok(ast) => {
                println!("  ✓ Parsed successfully");
                println!("  └─ AST: {}", ast);

                use neural_parser::Clause;
                if let Some(Clause::Match(match_clause)) = ast.clauses.iter().find(|c| matches!(c, Clause::Match(_))) {
                    let pattern = &match_clause.patterns[0];
                    if let Some(ref label) = pattern.start.label {
                        println!("     └─ Start node label: :{}", label);
                    }
                    if !pattern.chain.is_empty() {
                        let (rel, _) = &pattern.chain[0];
                        if let Some(ref label) = rel.label {
                            println!("     └─ Relationship: :{}", label);
                        }
                    }
                }
            }
            Err(e) => {
                println!("  ✗ Parse error: {}", e);
            }
        }
        println!();
    }

    println!("Lexer tokens for 'MATCH (n:Person) RETURN n':");
    let tokens = neural_parser::lexer::tokenize("MATCH (n:Person) RETURN n").unwrap();
    for (i, token) in tokens.iter().enumerate() {
        println!("  [{}] {:?}", i, token);
    }
    println!();
}

/// Demo 5: Query Execution with Properties
fn demo_query_execution() {
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!(" ⚡ Demo 5: Query Execution with Properties");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!();

    let store = create_demo_graph();

    println!("Graph created:");
    println!(
        "  • Nodes: {} (with properties: name, age)",
        store.node_count()
    );
    println!("  • Edges: {}", store.edge_count());
    println!();

    println!("📊 Basic Queries:");
    let basic_queries = [
        ("All nodes", "MATCH (n) RETURN n"),
        ("Return names", "MATCH (n) RETURN n.name"),
        ("Filter by age", "MATCH (n) WHERE n.age > 28 RETURN n.name"),
    ];

    for (name, query) in basic_queries {
        println!("🔍 {}", name);
        println!("   Query: {}", query);

        match neural_executor::execute_query(&store, query) {
            Ok(result) => {
                println!("   Results: {} rows", result.row_count());
                println!();
                print!("{}", result);
            }
            Err(e) => {
                println!("   Error: {}", e);
            }
        }
        println!();
    }

    println!("📈 Aggregation Queries:");
    let agg_queries = [
        ("Count all", "MATCH (n) RETURN COUNT(*)"),
        ("Sum ages", "MATCH (n) RETURN SUM(n.age)"),
        ("Average age", "MATCH (n) RETURN AVG(n.age)"),
        ("Min/Max age", "MATCH (n) RETURN MIN(n.age), MAX(n.age)"),
    ];

    for (name, query) in agg_queries {
        println!("🔍 {}", name);
        println!("   Query: {}", query);

        match neural_executor::execute_query(&store, query) {
            Ok(result) => {
                println!("   Results: {} rows", result.row_count());
                println!();
                print!("{}", result);
            }
            Err(e) => {
                println!("   Error: {}", e);
            }
        }
        println!();
    }
}