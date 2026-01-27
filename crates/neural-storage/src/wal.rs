//! Write-Ahead Log for ensuring mutation durability.
// Sprint 26

use std::fs::{File, OpenOptions};
use std::io::{Write, BufWriter};
use std::path::{Path, PathBuf};
use serde::{Serialize, Deserialize};
use thiserror::Error;
use bincode;

use neural_core::{NodeId, PropertyValue};
use openraft::{EntryPayload};
use crate::raft::types::TypeConfig;

/// Unique identifier for a transaction.
pub type TransactionId = u64;

/// Errors related to the Write-Ahead Log.
#[derive(Error, Debug)]
pub enum WalError {
    #[error("Failed to open WAL file: {0}")]
    Io(#[from] std::io::Error),
    #[error("Failed to serialize log entry: {0}")]
    Serialization(#[from] bincode::Error),
    #[error("Failed to serialize membership config: {0}")]
    MembershipSerialization(String),
}

/// Represents a single, atomic mutation to the graph.
/// These entries are written to the WAL before being applied in memory.
#[derive(Serialize, Deserialize, Debug, PartialEq, Clone)]
pub enum LogEntry {
    /// Start of a transaction.
    BeginTransaction {
        tx_id: TransactionId,
    },
    /// Commit a transaction.
    CommitTransaction {
        tx_id: TransactionId,
        /// Timestamp of commit (Sprint 54 - Time-Travel)
        /// ISO 8601 format: "2026-01-15T12:00:00Z"
        #[serde(default)]
        timestamp: Option<String>,
    },
    /// Rollback/Abort a transaction.
    RollbackTransaction {
        tx_id: TransactionId,
    },
    /// A new node was created.
    CreateNode {
        node_id: NodeId,
        label: Option<String>,
        properties: Vec<(String, PropertyValue)>,
    },
    /// A new edge was created.
    CreateEdge {
        source: NodeId,
        target: NodeId,
        edge_type: Option<String>,
    },
    /// A property was set or updated on a node.
    SetProperty {
        node_id: NodeId,
        key: String,
        value: PropertyValue,
    },
    /// A node and its incident edges were deleted.
    DeleteNode {
        node_id: NodeId,
    },
    /// A blank entry (used for leader commit).
    Blank,
    /// A cluster membership change.
    Membership {
        config_bytes: Vec<u8>,
    },
}

impl From<EntryPayload<TypeConfig>> for LogEntry {
    fn from(payload: EntryPayload<TypeConfig>) -> Self {
        match payload {
            EntryPayload::Normal(req) => req.entry,
            EntryPayload::Blank => LogEntry::Blank,
            EntryPayload::Membership(membership) => {
                let config_bytes = bincode::serialize(&membership)
                    .expect("Failed to serialize membership config");
                LogEntry::Membership { config_bytes }
            }
        }
    }
}

/// Manages writing entries to the WAL file.
#[derive(Debug)]
pub struct WalWriter {
    _path: PathBuf,
    writer: BufWriter<File>,
}

impl WalWriter {
    /// Creates a new WalWriter, opening or creating the file at `path`.
    pub fn new(path: impl AsRef<Path>) -> Result<Self, WalError> {
        let file = OpenOptions::new()
            .append(true)
            .create(true)
            .open(path.as_ref())?;

        Ok(Self {
            _path: path.as_ref().to_path_buf(),
            writer: BufWriter::new(file),
        })
    }

    /// Logs a single mutation.
    ///
    /// Serializes the entry, writes it to the buffer, and flushes to disk
    /// to guarantee durability.
    pub fn log(&mut self, entry: &LogEntry) -> Result<(), WalError> {
        // We must serialize with a size header to know how many bytes to read
        // during recovery.
        let encoded: Vec<u8> = bincode::serialize(entry)?;
        let len = encoded.len() as u64;

        // Write length prefix
        self.writer.write_all(&len.to_le_bytes())?;
        // Write payload
        self.writer.write_all(&encoded)?;
        // IMPORTANT: Flush to disk to ensure durability
        self.writer.flush()?;

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;
    use std::fs;
    use std::collections::{BTreeMap, BTreeSet};
    use std::io::{Read, Seek, SeekFrom};
    use neural_core::PropertyValue;
    use openraft::{EntryPayload, CommittedLeaderId, StoredMembership, LogId, Membership};
    use crate::raft::types::{RaftRequest, TypeConfig, RaftNodeId};

    #[test]
    fn test_log_entry_from_payload() {
        // Test Normal payload
        let node_id = NodeId::new(10);
        let original_entry = LogEntry::CreateNode {
            node_id,
            label: Some("TestNode".to_string()),
            properties: vec![],
        };
        let raft_request = RaftRequest::new(original_entry.clone());
        let normal_payload = EntryPayload::Normal(raft_request);
        let converted_log_entry: LogEntry = normal_payload.into();
        assert_eq!(converted_log_entry, original_entry);

        // Test Blank payload
        let blank_payload = EntryPayload::Blank;
        let converted_blank_entry: LogEntry = blank_payload.into();
        assert_eq!(converted_blank_entry, LogEntry::Blank);

        // Test Membership payload
        let membership = Membership::new(
            vec![[1, 2, 3].into_iter().collect::<BTreeSet<_>>()], 
            ()
        );
        let membership_payload = EntryPayload::Membership(membership.clone());
        let converted_membership_entry: LogEntry = membership_payload.into();
        match converted_membership_entry {
            LogEntry::Membership { config_bytes } => {
                let deserialized_membership: Membership<RaftNodeId, openraft::BasicNode> = bincode::deserialize(&config_bytes).unwrap();
                assert_eq!(deserialized_membership, membership);
            },
            _ => panic!("Expected Membership LogEntry"),
        }
    }
}
