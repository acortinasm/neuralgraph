
use std::fmt::Debug;
use std::ops::RangeBounds;
use std::sync::Arc;
use std::io::Cursor;

use openraft::storage::{LogState, RaftStorage, Snapshot, RaftSnapshotBuilder};
use openraft::{Entry, LogId, RaftLogReader, SnapshotMeta, StorageError, Vote, OptionalSend};

use super::{LogStore, GraphStateMachine, TypeConfig, RaftNodeId, RaftResponse};

/// A wrapper that implements the monolithic RaftStorage trait
/// by delegating to LogStore and GraphStateMachine.
#[derive(Debug, Clone)]
pub struct NeuralRaftStorage {
    log_store: Arc<LogStore>,
    state_machine: Arc<GraphStateMachine>,
}

impl NeuralRaftStorage {
    pub fn new(log_store: Arc<LogStore>, state_machine: Arc<GraphStateMachine>) -> Self {
        Self {
            log_store,
            state_machine,
        }
    }
}

impl RaftSnapshotBuilder<TypeConfig> for NeuralRaftStorage {
    async fn build_snapshot(&mut self) -> Result<Snapshot<TypeConfig>, StorageError<RaftNodeId>> {
        let snap = self.state_machine.get_current_snapshot().await?;
        snap.ok_or_else(|| StorageError::IO {
            source: openraft::StorageIOError::read_snapshot(None, &std::io::Error::new(std::io::ErrorKind::Other, "No snapshot available")),
        })
    }
}

impl RaftStorage<TypeConfig> for NeuralRaftStorage {
    type LogReader = Self;
    type SnapshotBuilder = Self;

    async fn save_vote(&mut self, vote: &Vote<RaftNodeId>) -> Result<(), StorageError<RaftNodeId>> {
        self.log_store.save_vote(vote).await
    }

    async fn read_vote(&mut self) -> Result<Option<Vote<RaftNodeId>>, StorageError<RaftNodeId>> {
        self.log_store.read_vote().await
    }

    async fn get_log_state(&mut self) -> Result<LogState<TypeConfig>, StorageError<RaftNodeId>> {
        self.log_store.get_log_state().await
    }

    async fn get_log_reader(&mut self) -> Self::LogReader {
        self.clone()
    }

    async fn append_to_log<I>(
        &mut self,
        entries: I,
    ) -> Result<(), StorageError<RaftNodeId>>
    where
        I: IntoIterator<Item = Entry<TypeConfig>> + OptionalSend,
    {
        self.log_store.append(entries).await
    }

    async fn delete_conflict_logs_since(
        &mut self,
        log_id: LogId<RaftNodeId>,
    ) -> Result<(), StorageError<RaftNodeId>> {
        self.log_store.truncate(log_id).await
    }

    async fn purge_logs_upto(
        &mut self,
        log_id: LogId<RaftNodeId>,
    ) -> Result<(), StorageError<RaftNodeId>> {
        self.log_store.purge(log_id).await
    }

    async fn last_applied_state(
        &mut self,
    ) -> Result<
        (Option<LogId<RaftNodeId>>, openraft::StoredMembership<RaftNodeId, openraft::BasicNode>),
        StorageError<RaftNodeId>,
    > {
        self.state_machine.applied_state().await
    }

    async fn apply_to_state_machine(
        &mut self,
        entries: &[Entry<TypeConfig>],
    ) -> Result<Vec<RaftResponse>, StorageError<RaftNodeId>> {
        self.state_machine.apply(entries.to_vec()).await
    }

    async fn get_snapshot_builder(&mut self) -> Self::SnapshotBuilder {
        self.clone()
    }

    async fn begin_receiving_snapshot(
        &mut self,
    ) -> Result<Box<Cursor<Vec<u8>>>, StorageError<RaftNodeId>> {
        self.state_machine.begin_receiving_snapshot().await
    }

    async fn install_snapshot(
        &mut self,
        meta: &SnapshotMeta<RaftNodeId, openraft::BasicNode>,
        snapshot: Box<Cursor<Vec<u8>>>,
    ) -> Result<(), StorageError<RaftNodeId>> {
        self.state_machine.install_snapshot(meta, snapshot).await
    }

    async fn get_current_snapshot(
        &mut self,
    ) -> Result<Option<Snapshot<TypeConfig>>, StorageError<RaftNodeId>> {
        self.state_machine.get_current_snapshot().await
    }
}

impl RaftLogReader<TypeConfig> for NeuralRaftStorage {
    async fn try_get_log_entries<RB: RangeBounds<u64> + Clone + Debug + OptionalSend>(
        &mut self,
        range: RB,
    ) -> Result<Vec<Entry<TypeConfig>>, StorageError<RaftNodeId>> {
        self.log_store.clone().try_get_log_entries(range).await
    }
}
