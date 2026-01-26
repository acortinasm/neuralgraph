# Phase 7: Scale & Production Roadmap

**Status:** Planning
**Target:** v1.0.0 (Enterprise Release)
**Focus:** Reliability, Horizontal Scalability, and Temporal Analysis.

---

## 1. Context & Goals

**Current State (v0.8.0):**
*   **Architecture:** Single-node, In-memory (with disk persistence).
*   **Durability:** Write-Ahead Log (WAL) ensures crash recovery.
*   **Consistency:** Atomic single-operation updates. No multi-statement transactions.
*   **Concurrency:** `RwLock` based (Multiple readers, Single writer).

**Phase 7 Goals:**
1.  **ACID Transactions:** Support multi-statement operations (`BEGIN`...`COMMIT`) with Snapshot Isolation.
2.  **Distributed Consensus:** Replicate data across multiple nodes using the Raft protocol.
3.  **High Availability:** Automatic failover and leader election.
4.  **Time-Travel:** Query the database state as it existed at a specific point in time.

---

## 2. Technical Strategy

### 2.1 Storage Engine Evolution: MVCC
To support both ACID transactions and Time-Travel without locking the entire database for readers, we will move from "Update-in-place" to **Multi-Version Concurrency Control (MVCC)**.

*   **Current:** `PropertyStore` stores `Value`.
*   **Target:** `PropertyStore` stores `Vec<(TransactionId, Value)>` or a delta chain.
*   **Benefit:** Readers see a consistent snapshot defined by their `TransactionId`, independent of active writers.

### 2.2 Distributed Consensus: Raft
We will implement/integrate a Raft layer to manage the WAL.
*   **Leader:** Handles all writes. Appends to local WAL and replicates to Followers.
*   **Followers:** Apply WAL entries to their local state machine (GraphStore) only after "Commit".
*   **Library Choice:** Evaluate `openraft` (Async Rust) vs `tikv/raft-rs`.

---

## 3. Sprint Breakdown

### **Sprint 50: Transaction Manager (ACID Core)**
**Goal:** Enable atomic multi-statement blocks.
*   **Deliverables:**
    *   `TransactionManager` struct to track active transaction IDs (`TxId`).
    *   `WalEntry` enhancement to support `Begin`, `Commit`, `Abort` markers.
    *   Updated `Executor` to buffer mutations in a `TransactionContext` before flushing to WAL/Store.
    *   Basic locking (2PL) for conflict detection.

### **Sprint 51: MVCC (Snapshot Isolation)** - ✅ Completed
**Goal:** Non-blocking reads and Time-Travel foundation.
*   **Deliverables:**
    *   Refactor `PropertyStore` and `Topology` to versioned storage.
    *   Implement `Snapshot` visibility logic: "Can Tx A see data written by Tx B?"
    *   Garbage Collection (Vacuum process) for old versions.
    *   **Metric:** concurrent_read_throughput > single_threaded_read_throughput.

### **Sprint 52: Raft Consensus (Replication)** - 🚧 Implementation Complete (Testing)
**Goal:** Replicate the WAL across a 3-node cluster.
*   **Deliverables:**
    *   Integrate Raft library.
    *   Define `RaftStateMachine` that wraps `GraphStore`.
    *   Network layer (gRPC/TCP) for node-to-node communication.
    *   Basic Leader Election tests.

### **Sprint 53: Cluster Management**
**Goal:** Dynamic cluster membership and client routing.
*   **Deliverables:**
    *   Node Discovery / Configuration.
    *   `Join` / `Leave` cluster commands.
    *   Smart Client / Router: Redirect writes to Leader, load balance reads to Followers.
    *   Chaos Testing (kill nodes and verify recovery).

### **Sprint 54: Time-Travel & Auditing**
**Goal:** Query past states.
*   **Deliverables:**
    *   `AT TIME <timestamp>` syntax in NGQL.
    *   Map `timestamp` to `TxId`.
    *   Planner update to inject version constraints into scans.
    *   "Flashback" feature to revert database state.

---

## 4. Architecture Diagram (Conceptual)

```mermaid
graph TD
    Client -->|Write| RaftLeader
    Client -->|Read| LoadBalancer
    LoadBalancer -->|Read| RaftLeader
    LoadBalancer -->|Read| RaftFollower1
    LoadBalancer -->|Read| RaftFollower2
    
    subgraph Cluster
        RaftLeader[Node 1 (Leader)] --Replicate WAL--> RaftFollower1[Node 2]
        RaftLeader --Replicate WAL--> RaftFollower2[Node 3]
    end
    
    subgraph Storage Node
        Log[Raft Log / WAL] --> StateMachine[GraphStore (MVCC)]
        StateMachine --> Index[Vector Index]
    end
```
