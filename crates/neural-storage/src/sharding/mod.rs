//! Graph Sharding for horizontal scalability (Sprint 55).
//!
//! This module provides partitioning strategies and shard management for
//! distributing graph data across multiple nodes.
//!
//! # Partitioning Strategies
//!
//! - **Hash-based**: Consistent hashing on node IDs for even distribution
//! - **Range-based**: Node ID ranges assigned to shards for locality
//!
//! # Architecture
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────────┐
//! │                      ShardedGraphStore                          │
//! │  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐         │
//! │  │   Shard 0   │    │   Shard 1   │    │   Shard 2   │         │
//! │  │ Nodes 0-99  │    │ Nodes 100-  │    │ Nodes 200-  │         │
//! │  │             │    │    199      │    │    299      │         │
//! │  └─────────────┘    └─────────────┘    └─────────────┘         │
//! │         │                  │                  │                 │
//! │         └──────────────────┼──────────────────┘                 │
//! │                            │                                    │
//! │                    ┌───────▼───────┐                           │
//! │                    │ ShardManager  │                           │
//! │                    │  (Routing)    │                           │
//! │                    └───────────────┘                           │
//! └─────────────────────────────────────────────────────────────────┘
//! ```

mod strategy;
mod manager;
mod router;

pub use strategy::{PartitionStrategy, HashPartition, RangePartition, ShardId};
pub use manager::{ShardManager, ShardConfig, ShardInfo};
pub use router::{ShardRouter, ShardedQuery, QueryPlan};
