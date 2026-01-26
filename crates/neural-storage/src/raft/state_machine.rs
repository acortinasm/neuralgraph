//! Raft state machine implementation.
//!
//! Wraps GraphStore to provide Raft state machine semantics.
//! Applies committed log entries to the graph and handles snapshots.

use std::io::Cursor;
use std::sync::Arc;

use openraft::{
    BasicNode, EntryPayload, LogId, OptionalSend, Snapshot, SnapshotMeta,
    StorageError, StoredMembership,
};
use tokio::sync::RwLock;

use super::types::{RaftNodeId, RaftResponse, TypeConfig};
use crate::GraphStore;

/// Raft state machine wrapping GraphStore.
///
/// This is the core component that applies Raft log entries to the graph database.
/// It maintains consistency by applying entries in order after they are committed.
#[derive(Debug)]
pub struct GraphStateMachine {
    /// The underlying graph store.
    store: RwLock<GraphStore>,
    /// Last applied log ID.
    last_applied_log: RwLock<Option<LogId<RaftNodeId>>>,
    /// Current cluster membership.
    last_membership: RwLock<StoredMembership<RaftNodeId, BasicNode>>,
    /// Snapshot index for compaction.
    snapshot_idx: RwLock<u64>,
}

impl GraphStateMachine {
    /// Creates a new state machine with an empty graph store.
    pub fn new() -> Self {
        Self {
            store: RwLock::new(GraphStore::new_in_memory()),
            last_applied_log: RwLock::new(None),
            last_membership: RwLock::new(StoredMembership::default()),
            snapshot_idx: RwLock::new(0),
        }
    }

    /// Creates a new state machine with an existing graph store.
    pub fn with_store(store: GraphStore) -> Self {
        Self {
            store: RwLock::new(store),
            last_applied_log: RwLock::new(None),
            last_membership: RwLock::new(StoredMembership::default()),
            snapshot_idx: RwLock::new(0),
        }
    }

    /// Creates a new state machine wrapped in Arc for sharing.
    pub fn new_arc() -> Arc<Self> {
        Arc::new(Self::new())
    }

    /// Gets a read reference to the underlying store.
    pub async fn read_store(&self) -> tokio::sync::RwLockReadGuard<'_, GraphStore> {
        self.store.read().await
    }

    /// Gets a write reference to the underlying store.
    pub async fn write_store(&self) -> tokio::sync::RwLockWriteGuard<'_, GraphStore> {
        self.store.write().await
    }

    pub async fn applied_state(
        &self,
    ) -> Result<
        (
            Option<LogId<RaftNodeId>>,
            StoredMembership<RaftNodeId, BasicNode>,
        ),
        StorageError<RaftNodeId>,
    > {
        let last_applied = *self.last_applied_log.read().await;
        let membership = self.last_membership.read().await.clone();
        Ok((last_applied, membership))
    }

    pub async fn apply<I>(&self, entries: I) -> Result<Vec<RaftResponse>, StorageError<RaftNodeId>>
    where
        I: IntoIterator<Item = openraft::Entry<TypeConfig>> + OptionalSend,
    {
        let mut responses = Vec::new();
        let mut store = self.store.write().await;

        for entry in entries {
            let log_id = entry.log_id;

            // Update last applied
            *self.last_applied_log.write().await = Some(log_id);

            match entry.payload {
                EntryPayload::Blank => {
                    // No-op entry (used for leader commit)
                    responses.push(RaftResponse::ok(log_id.index));
                }
                EntryPayload::Normal(req) => {
                    // Apply the graph mutation
                    match store.apply_log_entry(&req.entry) {
                        Ok(()) => {
                            responses.push(RaftResponse::ok(log_id.index));
                        }
                        Err(e) => {
                            responses.push(RaftResponse::err(e, log_id.index));
                        }
                    }
                }
                EntryPayload::Membership(membership) => {
                    // Update cluster membership
                    let new_membership = StoredMembership::new(Some(log_id), membership);
                    *self.last_membership.write().await = new_membership;
                    responses.push(RaftResponse::ok(log_id.index));
                }
            }
        }

        Ok(responses)
    }

    pub async fn begin_receiving_snapshot(
        &self,
    ) -> Result<Box<Cursor<Vec<u8>>>, StorageError<RaftNodeId>> {
        Ok(Box::new(Cursor::new(Vec::new())))
    }

    pub async fn install_snapshot(
        &self,
        meta: &SnapshotMeta<RaftNodeId, BasicNode>,
        snapshot: Box<Cursor<Vec<u8>>>,
    ) -> Result<(), StorageError<RaftNodeId>> {
        // Deserialize the snapshot data into a GraphStore
        let data = snapshot.into_inner();
        let new_store: GraphStore = bincode::deserialize(&data).map_err(|e| StorageError::IO {
            source: openraft::StorageIOError::read_snapshot(None, &e),
        })?;

        // Replace the store
        *self.store.write().await = new_store;

        // Update metadata
        *self.last_applied_log.write().await = meta.last_log_id;
        *self.last_membership.write().await = meta.last_membership.clone();

        Ok(())
    }

    pub async fn get_current_snapshot(
        &self,
    ) -> Result<Option<Snapshot<TypeConfig>>, StorageError<RaftNodeId>> {
        let last_applied = *self.last_applied_log.read().await;
        let membership = self.last_membership.read().await.clone();

        match last_applied {
            Some(last_log_id) => {
                // Serialize the current store
                let store = self.store.read().await;
                let data = bincode::serialize(&*store).map_err(|e| StorageError::IO {
                    source: openraft::StorageIOError::write_snapshot(None, &e),
                })?;

                let snapshot_idx = {
                    let mut idx = self.snapshot_idx.write().await;
                    *idx += 1;
                    *idx
                };

                let snapshot_id = format!("snapshot-{}-{}", last_log_id.index, snapshot_idx);

                let meta = SnapshotMeta {
                    last_log_id: Some(last_log_id),
                    last_membership: membership,
                    snapshot_id,
                };

                Ok(Some(Snapshot {
                    meta,
                    snapshot: Box::new(Cursor::new(data)),
                }))
            }
            None => Ok(None),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::raft::types::RaftRequest;
    use crate::wal::LogEntry;
    use neural_core::{NodeId, PropertyValue, Graph};
    use openraft::{Entry, LogId, CommittedLeaderId};

    #[tokio::test]
    async fn test_state_machine_apply() {
        let sm = GraphStateMachine::new_arc();
        let sm_ref = sm.as_ref();

        // Create a log entry to apply
        let request = RaftRequest::new(LogEntry::CreateNode {
            node_id: NodeId::new(0),
            label: Some("Person".to_string()),
            properties: vec![("name".to_string(), PropertyValue::from("Alice"))],
        });

        let entry = Entry {
            log_id: LogId::new(CommittedLeaderId::new(1, 1), 1),
            payload: EntryPayload::Normal(request),
        };

        // Apply the entry
        let responses = sm_ref.apply(vec![entry]).await.unwrap();
        assert_eq!(responses.len(), 1);
        assert!(responses[0].success);

        // Verify the node was created
        let store = sm.read_store().await;
        assert_eq!(store.node_count(), 1);
        assert_eq!(
            store.get_property(NodeId::new(0), "name"),
            Some(&PropertyValue::from("Alice"))
        );
    }

    #[tokio::test]
    async fn test_state_machine_snapshot() {
        let sm = GraphStateMachine::new_arc();
        let sm_ref = sm.as_ref();

        // Apply some entries
        let request = RaftRequest::new(LogEntry::CreateNode {
            node_id: NodeId::new(0),
            label: Some("Test".to_string()),
            properties: vec![],
        });

        let entry = Entry {
            log_id: LogId::new(CommittedLeaderId::new(1, 1), 1),
            payload: EntryPayload::Normal(request),
        };

        sm_ref.apply(vec![entry]).await.unwrap();

        // Get snapshot
        let snapshot = sm_ref.get_current_snapshot().await.unwrap();
        assert!(snapshot.is_some());

        let snapshot = snapshot.unwrap();
        assert_eq!(snapshot.meta.last_log_id, Some(LogId::new(CommittedLeaderId::new(1, 1), 1)));
    }
}