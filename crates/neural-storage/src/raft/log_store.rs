//! Raft log storage implementation.
//!
//! Provides persistent storage for Raft log entries using the Write-Ahead Log (WAL).
//!

use std::collections::BTreeMap;
use std::fmt::Debug;
use std::ops::RangeBounds;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use openraft::{StorageError, StorageIOError};
use openraft::{Entry, LogId, OptionalSend, Vote, ErrorSubject, ErrorVerb};
use openraft::Membership;
use tokio::sync::RwLock;
use anyerror::AnyError;

use super::types::{RaftNodeId, TypeConfig, RaftRequest};
use crate::wal::{WalError, WalWriter, LogEntry};
use crate::wal_reader::WalReader;

/// Persistent log store for Raft.
#[derive(Debug)]
pub struct LogStore {
    /// The path to the directory where WAL files are stored.
    data_dir: PathBuf,
    /// Writer for the current WAL segment.
    wal_writer: RwLock<WalWriter>,
    /// Reader for the current WAL segment (for recovery and reads).
    wal_reader: RwLock<WalReader>,
    /// Cached log entries, mapping Raft log index to entry.
    log: RwLock<BTreeMap<u64, Entry<TypeConfig>>>,
    /// Last known committed log ID.
    committed: RwLock<Option<LogId<RaftNodeId>>>,
    /// Last known vote (persisted separately).
    vote: RwLock<Option<Vote<RaftNodeId>>>,
    /// Last purged log ID (for compaction).
    last_purged_log_id: RwLock<Option<LogId<RaftNodeId>>>,
}

impl LogStore {
    /// Creates a new persistent log store.
    pub fn new(data_dir: impl AsRef<Path>) -> Result<Self, WalError> {
        let data_dir = data_dir.as_ref().to_path_buf();
        std::fs::create_dir_all(&data_dir)?;

        let wal_path = data_dir.join("raft.wal");
        let wal_writer = WalWriter::new(&wal_path)?;
        let wal_reader = WalReader::new(&wal_path)?;

        let mut store = Self {
            data_dir,
            wal_writer: RwLock::new(wal_writer),
            wal_reader: RwLock::new(wal_reader),
            log: RwLock::new(BTreeMap::new()),
            committed: RwLock::new(None),
            vote: RwLock::new(None),
            last_purged_log_id: RwLock::new(None),
        };

        // Recover state from WAL on startup
        store.recover()?;

        Ok(store)
    }

    /// Creates a new log store wrapped in Arc for sharing.
    pub fn new_arc(data_dir: impl AsRef<Path>) -> Arc<Self> {
        Arc::new(Self::new(data_dir).expect("Failed to create persistent LogStore"))
    }

    /// Recovers the log store state from the WAL.
    fn recover(&mut self) -> Result<(), WalError> {
        let entries = self.wal_reader.get_mut().read_entries()?;
        let log = self.log.get_mut();

        for entry_from_wal in entries {
            let payload = match entry_from_wal {
                LogEntry::Blank => openraft::EntryPayload::Blank,
                LogEntry::Membership { config_bytes } => {
                    let membership: Membership<RaftNodeId, openraft::BasicNode> = bincode::deserialize(&config_bytes)
                        .map_err(|e| WalError::MembershipSerialization(e.to_string()))?;
                    openraft::EntryPayload::Membership(membership)
                }
                _ => openraft::EntryPayload::Normal(RaftRequest::new(entry_from_wal)),
            };

            let raft_entry = Entry {
                // The log_id.index will be sequentially assigned during recovery
                // term and leader_id are placeholders for now, actual values would come from snapshots or external metadata
                log_id: LogId::new(openraft::CommittedLeaderId::new(0, 0), log.len() as u64 + 1),
                payload,
            };
            log.insert(raft_entry.log_id.index, raft_entry);
        }

        // Recover vote
        let vote_path = self.data_dir.join("vote");
        if vote_path.exists() {
            let vote_bytes = std::fs::read(&vote_path)?;
            let vote: Vote<RaftNodeId> = bincode::deserialize(&vote_bytes)
                .map_err(|e| WalError::Serialization(e))?;
            *self.vote.get_mut() = Some(vote);
        }

        // Recover committed index (if stored separately)
        let committed_path = self.data_dir.join("committed");
        if committed_path.exists() {
            let committed_bytes = std::fs::read(&committed_path)?;
            let committed: LogId<RaftNodeId> = bincode::deserialize(&committed_bytes)
                .map_err(|e| WalError::Serialization(e))?;
            *self.committed.get_mut() = Some(committed);
        }

        Ok(())
    }

    pub async fn get_log_state(&self) -> Result<openraft::storage::LogState<TypeConfig>, StorageError<RaftNodeId>> {
        let log = self.log.read().await;
        let last_purged = *self.last_purged_log_id.read().await;
        let last_log_id = log.iter().next_back().map(|(_, entry)| entry.log_id);

        Ok(openraft::storage::LogState {
            last_purged_log_id: last_purged,
            last_log_id,
        })
    }

    pub async fn save_vote(&self, vote: &Vote<RaftNodeId>) -> Result<(), StorageError<RaftNodeId>> {
        let vote_path = self.data_dir.join("vote");
        let vote_bytes = bincode::serialize(vote).map_err(|e| StorageError::IO {
            source: StorageIOError::new(ErrorSubject::Vote, ErrorVerb::Write, AnyError::new(&e))
        })?;
        std::fs::write(&vote_path, vote_bytes).map_err(|e| StorageError::IO {
            source: StorageIOError::new(ErrorSubject::Vote, ErrorVerb::Write, AnyError::new(&e))
        })?;
        *self.vote.write().await = Some(*vote);
        Ok(())
    }

    pub async fn read_vote(&self) -> Result<Option<Vote<RaftNodeId>>, StorageError<RaftNodeId>> {
        let vote_path = self.data_dir.join("vote");
        if vote_path.exists() {
            let vote_bytes = std::fs::read(&vote_path).map_err(|e| StorageError::IO {
                source: StorageIOError::new(ErrorSubject::Vote, ErrorVerb::Read, AnyError::new(&e))
            })?;
            let vote: Vote<RaftNodeId> = bincode::deserialize(&vote_bytes).map_err(|e| StorageError::IO {
                source: StorageIOError::new(ErrorSubject::Vote, ErrorVerb::Read, AnyError::new(&e))
            })?;
            *self.vote.write().await = Some(vote);
        }
        Ok(*self.vote.read().await)
    }

    pub async fn save_committed(
        &self,
        committed: Option<LogId<RaftNodeId>>,
    ) -> Result<(), StorageError<RaftNodeId>> {
        let committed_path = self.data_dir.join("committed");
        if let Some(log_id) = committed {
            let committed_bytes = bincode::serialize(&log_id).map_err(|e| StorageError::IO {
                source: StorageIOError::new(ErrorSubject::Vote, ErrorVerb::Write, AnyError::new(&e))
            })?;
            std::fs::write(&committed_path, committed_bytes).map_err(|e| StorageError::IO {
                source: StorageIOError::new(ErrorSubject::Vote, ErrorVerb::Write, AnyError::new(&e))
            })?;
        } else if committed_path.exists() {
            std::fs::remove_file(&committed_path).map_err(|e| StorageError::IO {
                source: StorageIOError::new(ErrorSubject::Vote, ErrorVerb::Write, AnyError::new(&e))
            })?;
        }
        *self.committed.write().await = committed;
        Ok(())
    }

    pub async fn read_committed(
        &self,
    ) -> Result<Option<LogId<RaftNodeId>>, StorageError<RaftNodeId>> {
        let committed_path = self.data_dir.join("committed");
        if committed_path.exists() {
            let committed_bytes = std::fs::read(&committed_path).map_err(|e| StorageError::IO {
                source: StorageIOError::new(ErrorSubject::Vote, ErrorVerb::Read, AnyError::new(&e))
            })?;
            let committed: LogId<RaftNodeId> = bincode::deserialize(&committed_bytes).map_err(|e| StorageError::IO {
                source: StorageIOError::new(ErrorSubject::Vote, ErrorVerb::Read, AnyError::new(&e))
            })?;
            *self.committed.write().await = Some(committed);
        }
        Ok(*self.committed.read().await)
    }

    pub async fn append<I>(
        &self,
        entries: I,
    ) -> Result<(), StorageError<RaftNodeId>>
    where
        I: IntoIterator<Item = Entry<TypeConfig>> + OptionalSend,
    {
        let mut wal_writer = self.wal_writer.write().await;
        let mut log = self.log.write().await;

        for entry in entries {
            // Write to WAL
            let log_entry = LogEntry::from(entry.payload.clone()); 
            wal_writer.log(&log_entry).map_err(|e| StorageError::IO {
                source: StorageIOError::new(ErrorSubject::Logs, ErrorVerb::Write, AnyError::new(&e))
            })?;
            
            log.insert(entry.log_id.index, entry);
        }
        Ok(())
    }

    pub async fn truncate(&self, log_id: LogId<RaftNodeId>) -> Result<(), StorageError<RaftNodeId>> {
        let mut log = self.log.write().await;
        let keys_to_remove: Vec<_> = log.range(log_id.index..).map(|(k, _)| *k).collect();
        for key in keys_to_remove {
            log.remove(&key);
        }
        // TODO: Implement actual WAL truncation/compaction
        Ok(())
    }

    pub async fn purge(&self, log_id: LogId<RaftNodeId>) -> Result<(), StorageError<RaftNodeId>> {
        {
            let mut last_purged = self.last_purged_log_id.write().await;
            *last_purged = Some(log_id);
        }

        {
            let mut log = self.log.write().await;
            let keys_to_remove: Vec<_> = log.range(..=log_id.index).map(|(k, _)| *k).collect();
            for key in keys_to_remove {
                log.remove(&key);
            }
        }
        // TODO: Implement actual WAL purging/compaction
        Ok(())
    }
}

impl openraft::storage::RaftLogReader<TypeConfig> for Arc<LogStore> {
    async fn try_get_log_entries<RB: RangeBounds<u64> + Clone + Debug + OptionalSend>(
        &mut self,
        range: RB,
    ) -> Result<Vec<Entry<TypeConfig>>, StorageError<RaftNodeId>> {
        let log = self.log.read().await;
        let entries: Vec<_> = log.range(range).map(|(_, entry)| entry.clone()).collect();
        Ok(entries)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;
    use openraft::{Entry, CommittedLeaderId};
    use openraft::storage::RaftLogReader;

    #[tokio::test]
    async fn test_log_store_basic() {
        let dir = tempdir().unwrap();
        let store = LogStore::new_arc(dir.path());
        let store_ref = store.as_ref();

        // Get initial state
        let state = store_ref.get_log_state().await.unwrap();
        assert!(state.last_log_id.is_none());
        assert!(state.last_purged_log_id.is_none());
    }

    #[tokio::test]
    async fn test_log_store_vote() {
        let dir = tempdir().unwrap();
        let store = LogStore::new_arc(dir.path());
        let store_ref = store.as_ref();

        // Initially no vote
        let vote = store_ref.read_vote().await.unwrap();
        assert!(vote.is_none());

        // Save a vote
        let new_vote = Vote::new(1, 1);
        store_ref.save_vote(&new_vote).await.unwrap();

        // Read it back
        let vote = store_ref.read_vote().await.unwrap();
        assert_eq!(vote, Some(new_vote));

        // Overwrite vote
        let new_vote_2 = Vote::new(2, 2);
        store_ref.save_vote(&new_vote_2).await.unwrap();
        let vote_2 = store_ref.read_vote().await.unwrap();
        assert_eq!(vote_2, Some(new_vote_2));
    }

    #[tokio::test]
    async fn test_log_store_append_and_read() {
        let dir = tempdir().unwrap();
        let mut store = LogStore::new_arc(dir.path());
        
        let entry1 = Entry {
            log_id: LogId::new(CommittedLeaderId::new(1, 1), 1),
            payload: openraft::EntryPayload::Normal(super::RaftRequest::new(crate::wal::LogEntry::CreateNode {
                node_id: neural_core::NodeId::new(1),
                label: Some("Test".to_string()),
                properties: vec![],
            })),
        };
        let entry2 = Entry {
            log_id: LogId::new(CommittedLeaderId::new(1, 1), 2),
            payload: openraft::EntryPayload::Blank,
        };

        store.append(vec![entry1.clone(), entry2.clone()]).await.unwrap();

        let entries = store.try_get_log_entries(1..=2).await.unwrap();
        assert_eq!(entries.len(), 2);
        assert_eq!(entries[0].log_id, entry1.log_id);
        assert_eq!(entries[1].log_id, entry2.log_id);

        // Verify log state
        let state = store.get_log_state().await.unwrap();
        assert_eq!(state.last_log_id, Some(entry2.log_id));
    }

    #[tokio::test]
    async fn test_log_store_committed() {
        let dir = tempdir().unwrap();
        let store = LogStore::new_arc(dir.path());
        let store_ref = store.as_ref();

        // Initially no committed
        let committed = store_ref.read_committed().await.unwrap();
        assert!(committed.is_none());

        let log_id = LogId::new(CommittedLeaderId::new(1, 1), 5);
        store_ref.save_committed(Some(log_id)).await.unwrap();

        let committed = store_ref.read_committed().await.unwrap();
        assert_eq!(committed, Some(log_id));

        // Clear committed
        store_ref.save_committed(None).await.unwrap();
        let committed = store_ref.read_committed().await.unwrap();
        assert!(committed.is_none());
    }
}
