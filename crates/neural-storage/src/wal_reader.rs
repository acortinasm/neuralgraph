//! WAL Reader for recovering log entries.

use std::fs::File;
use std::io::{Read, BufReader, Seek, SeekFrom};
use std::path::{Path, PathBuf};
use bincode;

use super::wal::{LogEntry, WalError};

/// Manages reading entries from the WAL file.
#[derive(Debug)]
pub struct WalReader {
    _path: PathBuf,
    reader: BufReader<File>,
    current_offset: u64,
}

impl WalReader {
    /// Creates a new WalReader, opening the file at `path`.
    pub fn new(path: impl AsRef<Path>) -> Result<Self, WalError> {
        let file = File::open(path.as_ref())?;
        Ok(Self {
            _path: path.as_ref().to_path_buf(),
            reader: BufReader::new(file),
            current_offset: 0,
        })
    }

    /// Reads all log entries from the current offset to the end of the WAL.
    pub fn read_entries(&mut self) -> Result<Vec<LogEntry>, WalError> {
        let mut entries = Vec::new();
        self.reader.seek(SeekFrom::Start(self.current_offset))?;

        loop {
            // Read length prefix
            let mut len_bytes = [0u8; 8];
            match self.reader.read_exact(&mut len_bytes) {
                Ok(_) => {},
                Err(e) if e.kind() == std::io::ErrorKind::UnexpectedEof => break,
                Err(e) => return Err(e.into()),
            }
            let len = u64::from_le_bytes(len_bytes);

            // Read payload
            let mut buffer = vec![0; len as usize];
            self.reader.read_exact(&mut buffer)?;

            let decoded: LogEntry = bincode::deserialize(&buffer)?;
            entries.push(decoded);

            // Update current offset
            self.current_offset += 8 + len;
        }

        Ok(entries)
    }

    /// Seeks to a specific offset in the WAL.
    pub fn seek(&mut self, offset: u64) -> Result<(), WalError> {
        self.reader.seek(SeekFrom::Start(offset))?;
        self.current_offset = offset;
        Ok(())
    }

    /// Returns the current offset in the WAL.
    pub fn current_offset(&self) -> u64 {
        self.current_offset
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;
    use crate::wal::WalWriter;
    use neural_core::{NodeId, PropertyValue};

    #[test]
    fn test_wal_reader_read_entry() {
        let dir = tempdir().unwrap();
        let wal_path = dir.path().join("test_read.wal");

        // Write some entries
        let mut writer = WalWriter::new(&wal_path).unwrap();
        let entry1 = LogEntry::CreateNode {
            node_id: NodeId::new(1),
            label: Some("Person".to_string()),
            properties: vec![],
        };
        let entry2 = LogEntry::CreateEdge {
            source: NodeId::new(1),
            target: NodeId::new(2),
            edge_type: Some("KNOWS".to_string()),
        };
        writer.log(&entry1).unwrap();
        writer.log(&entry2).unwrap();

        // Read entries
        let mut reader = WalReader::new(&wal_path).unwrap();
        let entries = reader.read_entries().unwrap();

        assert_eq!(entries.len(), 2);
        assert_eq!(entries[0], entry1);
        assert_eq!(entries[1], entry2);
    }

    #[test]
    fn test_wal_reader_seek() {
        let dir = tempdir().unwrap();
        let wal_path = dir.path().join("test_seek.wal");

        let mut writer = WalWriter::new(&wal_path).unwrap();
        let entry1 = LogEntry::CreateNode {
            node_id: NodeId::new(1),
            label: Some("Person".to_string()),
            properties: vec![],
        };
        let entry2 = LogEntry::CreateNode {
            node_id: NodeId::new(2),
            label: Some("Company".to_string()),
            properties: vec![],
        };
        writer.log(&entry1).unwrap();
        writer.log(&entry2).unwrap();

        // Read first entry, then seek back and read second
        let mut reader = WalReader::new(&wal_path).unwrap();
        let first_entry_len = bincode::serialize(&entry1).unwrap().len() as u64;

        // Read all entries first
        let entries = reader.read_entries().unwrap();
        assert_eq!(entries.len(), 2);
        assert_eq!(entries[0], entry1);
        assert_eq!(entries[1], entry2);

        // Seek to the beginning of the second entry and read it
        reader.seek(8 + first_entry_len).unwrap();
        let entries2 = reader.read_entries().unwrap();
        assert_eq!(entries2.len(), 1);
        assert_eq!(entries2[0], entry2);
    }
}
