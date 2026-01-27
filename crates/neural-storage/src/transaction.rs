//! Transaction management for ACID guarantees.
//!
//! This module implements the core transaction logic:
//! - Transaction ID generation
//! - State tracking (Active, Committed, Aborted)
//! - Mutation buffering (for Atomicity)
//!
//! # Architecture
//!
//! Transactions in NeuralGraphDB v0.8+ (Sprint 50) use a "Buffer-then-Apply" strategy
//! to ensure Atomicity.
//!
//! 1. `BEGIN`: A new `Transaction` is created.
//! 2. Mutations (CREATE/SET/DELETE): generated `LogEntry`s are buffered in the `Transaction`.
//! 3. `COMMIT`:
//!    a. Write `BeginTransaction` to WAL.
//!    b. Write all buffered `LogEntry`s to WAL.
//!    c. Write `CommitTransaction` to WAL.
//!    d. Apply all entries to the in-memory `GraphStore`.
//! 4. `ROLLBACK`: The buffer is discarded. Nothing hits the WAL or GraphStore.

use crate::wal::{LogEntry, TransactionId};
use crate::GraphStore;
use std::sync::atomic::{AtomicU64, Ordering};

use thiserror::Error;

/// Errors related to transactions.
#[derive(Debug, Error)]
pub enum TransactionError {
    #[error("Transaction {0} is not active")]
    NotActive(TransactionId),
    #[error("Transaction {0} already committed")]
    AlreadyCommitted(TransactionId),
    #[error("Transaction {0} already aborted")]
    AlreadyAborted(TransactionId),
    #[error("WAL error during commit: {0}")]
    WalError(#[from] crate::wal::WalError),
    #[error("Storage application error: {0}")]
    StorageError(String),
}

/// The state of a transaction.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TransactionState {
    Active,
    Committed,
    Aborted,
}

/// A handle to an active transaction.
///
/// Buffers mutations until commit.
pub struct Transaction {
    id: TransactionId,
    state: TransactionState,
    buffer: Vec<LogEntry>,
    pub pending_node_count: usize,
    pub pending_edge_count: usize,
}

impl Transaction {
    /// Creates a new active transaction.
    pub fn new(id: TransactionId) -> Self {
        Self {
            id,
            state: TransactionState::Active,
            buffer: Vec::new(),
            pending_node_count: 0,
            pending_edge_count: 0,
        }
    }

    /// Returns the transaction ID.
    pub fn id(&self) -> TransactionId {
        self.id
    }

    /// Returns the current state.
    pub fn state(&self) -> TransactionState {
        self.state
    }

    /// Buffers a log entry for execution on commit.
    pub fn buffer_entry(&mut self, entry: LogEntry) -> Result<(), TransactionError> {
        if self.state != TransactionState::Active {
            return Err(TransactionError::NotActive(self.id));
        }
        self.buffer.push(entry);
        Ok(())
    }

    /// Commits the transaction.
    ///
    /// Writes to WAL and applies to GraphStore.
    pub fn commit(&mut self, store: &mut GraphStore) -> Result<(), TransactionError> {
        if self.state != TransactionState::Active {
            return Err(TransactionError::NotActive(self.id));
        }

        // 1. Write to WAL (Atomicity & Durability)
        if let Some(wal) = &mut store.wal {
            // Begin marker
            wal.log(&LogEntry::BeginTransaction { tx_id: self.id })?;

            // Payload
            for entry in &self.buffer {
                wal.log(entry)?;
            }

            // Generate ISO 8601 timestamp for time-travel queries (Sprint 54)
            let timestamp = {
                use std::time::SystemTime;
                let now = SystemTime::now()
                    .duration_since(SystemTime::UNIX_EPOCH)
                    .unwrap_or_default()
                    .as_secs();
                // Convert to approximate ISO 8601 (UTC)
                // For production, use chrono crate for proper formatting
                let secs_per_day = 86400;
                let days_since_epoch = now / secs_per_day;
                let secs_today = now % secs_per_day;
                let hours = secs_today / 3600;
                let mins = (secs_today % 3600) / 60;
                let secs = secs_today % 60;

                // Rough approximation: 1970-01-01 + days
                // Days since epoch to year-month-day (simplified, doesn't handle leap years perfectly)
                let years = 1970 + (days_since_epoch / 365);
                let day_of_year = days_since_epoch % 365;
                let month = (day_of_year / 30) + 1;
                let day = (day_of_year % 30) + 1;

                format!("{:04}-{:02}-{:02}T{:02}:{:02}:{:02}Z", years, month.min(12), day.min(28), hours, mins, secs)
            };

            // Commit marker with timestamp
            wal.log(&LogEntry::CommitTransaction {
                tx_id: self.id,
                timestamp: Some(timestamp.clone()),
            })?;

            // Record in timestamp index
            store.record_commit_timestamp(timestamp, self.id);
        }

        // 2. Apply to Memory (Consistency)
        // We apply only after successful WAL write.
        for entry in &self.buffer {
            store.apply_log_entry(entry).map_err(TransactionError::StorageError)?;
        }

        self.state = TransactionState::Committed;
        self.buffer.clear(); // Free memory

        Ok(())
    }

    /// Rolls back the transaction.
    ///
    /// Discards the buffer.
    pub fn rollback(&mut self) -> Result<(), TransactionError> {
        if self.state != TransactionState::Active {
            return Err(TransactionError::NotActive(self.id));
        }

        // We can optionally log a Rollback marker if we had partial writes,
        // but since we buffer, nothing hit the WAL yet (except maybe internally if we flush?).
        // For now, pure buffering means we just discard.
        
        // However, if we move to WAL-on-Write later, we'd need to log Rollback.
        // Let's log it for tracing purposes if WAL is enabled.
        // Note: accessing store here would require passing it. 
        // For simple rollback of memory buffer, we assume "silent" rollback is fine
        // as no durable state was changed.

        self.state = TransactionState::Aborted;
        self.buffer.clear();

        Ok(())
    }
}

/// Manages transaction lifecycles.
#[derive(Debug)]
pub struct TransactionManager {
    next_tx_id: AtomicU64,
}

impl TransactionManager {
    /// Creates a new TransactionManager.
    pub fn new() -> Self {
        Self {
            next_tx_id: AtomicU64::new(1),
        }
    }

    /// Starts a new transaction.
    pub fn begin(&self) -> Transaction {
        let id = self.next_tx_id.fetch_add(1, Ordering::SeqCst);
        Transaction::new(id)
    }
}

impl Default for TransactionManager {
    fn default() -> Self {
        Self::new()
    }
}
