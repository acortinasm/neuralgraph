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
    ///
    /// ## Entry Format (V2 with checksums)
    /// ```text
    /// [8 bytes: length] [4 bytes: CRC32] [N bytes: bincode payload]
    /// ```
    /// The length field includes the checksum (4 bytes) + payload (N bytes).
    ///
    /// For backward compatibility, entries without checksums (length < 4 or
    /// old format) are detected and read without verification.
    pub fn read_entries(&mut self) -> Result<Vec<LogEntry>, WalError> {
        let mut entries = Vec::new();
        self.reader.seek(SeekFrom::Start(self.current_offset))?;

        loop {
            let entry_start_offset = self.current_offset;

            // Read length prefix
            let mut len_bytes = [0u8; 8];
            match self.reader.read_exact(&mut len_bytes) {
                Ok(_) => {},
                Err(e) if e.kind() == std::io::ErrorKind::UnexpectedEof => break,
                Err(e) => return Err(e.into()),
            }
            let len = u64::from_le_bytes(len_bytes);

            if len < 4 {
                // Old format without checksum (shouldn't happen with proper entries)
                // Read the entire buffer as payload
                let mut buffer = vec![0; len as usize];
                self.reader.read_exact(&mut buffer)?;
                let decoded: LogEntry = bincode::deserialize(&buffer)?;
                entries.push(decoded);
                self.current_offset += 8 + len;
                continue;
            }

            // Read checksum (4 bytes)
            let mut checksum_bytes = [0u8; 4];
            self.reader.read_exact(&mut checksum_bytes)?;
            let expected_checksum = u32::from_le_bytes(checksum_bytes);

            // Read payload (len - 4 bytes)
            let payload_len = (len - 4) as usize;
            let mut buffer = vec![0; payload_len];
            self.reader.read_exact(&mut buffer)?;

            // Verify checksum
            let computed_checksum = crc32fast::hash(&buffer);
            if computed_checksum != expected_checksum {
                return Err(WalError::ChecksumMismatch {
                    offset: entry_start_offset,
                    expected: expected_checksum,
                    computed: computed_checksum,
                });
            }

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

        // New format: length (8) + checksum (4) + payload
        let first_entry_payload_len = bincode::serialize(&entry1).unwrap().len() as u64;
        let first_entry_total_len = 8 + 4 + first_entry_payload_len;

        // Read all entries first
        let entries = reader.read_entries().unwrap();
        assert_eq!(entries.len(), 2);
        assert_eq!(entries[0], entry1);
        assert_eq!(entries[1], entry2);

        // Seek to the beginning of the second entry and read it
        reader.seek(first_entry_total_len).unwrap();
        let entries2 = reader.read_entries().unwrap();
        assert_eq!(entries2.len(), 1);
        assert_eq!(entries2[0], entry2);
    }

    #[test]
    fn test_wal_checksum_roundtrip() {
        let dir = tempdir().unwrap();
        let wal_path = dir.path().join("test_checksum.wal");

        // Write entries with checksums
        let mut writer = WalWriter::new(&wal_path).unwrap();
        let entries = vec![
            LogEntry::CreateNode {
                node_id: NodeId::new(1),
                label: Some("Person".to_string()),
                properties: vec![("name".to_string(), PropertyValue::from("Alice"))],
            },
            LogEntry::CreateEdge {
                source: NodeId::new(1),
                target: NodeId::new(2),
                edge_type: Some("KNOWS".to_string()),
            },
            LogEntry::SetProperty {
                node_id: NodeId::new(1),
                key: "age".to_string(),
                value: PropertyValue::from(30i64),
            },
        ];

        for entry in &entries {
            writer.log(entry).unwrap();
        }
        drop(writer);

        // Read and verify checksums pass
        let mut reader = WalReader::new(&wal_path).unwrap();
        let read_entries = reader.read_entries().unwrap();

        assert_eq!(read_entries.len(), 3);
        assert_eq!(read_entries[0], entries[0]);
        assert_eq!(read_entries[1], entries[1]);
        assert_eq!(read_entries[2], entries[2]);
    }

    #[test]
    fn test_wal_checksum_detects_corruption() {
        use std::io::{Seek, Write};

        let dir = tempdir().unwrap();
        let wal_path = dir.path().join("test_corrupt.wal");

        // Write a valid entry
        let mut writer = WalWriter::new(&wal_path).unwrap();
        let entry = LogEntry::CreateNode {
            node_id: NodeId::new(1),
            label: Some("Person".to_string()),
            properties: vec![],
        };
        writer.log(&entry).unwrap();
        drop(writer);

        // Corrupt the payload (byte after length and checksum)
        {
            let mut file = std::fs::OpenOptions::new()
                .read(true)
                .write(true)
                .open(&wal_path)
                .unwrap();

            // Skip length (8) + checksum (4) to get to payload
            file.seek(std::io::SeekFrom::Start(12)).unwrap();
            // Write garbage byte
            file.write_all(&[0xFF]).unwrap();
        }

        // Try to read - should fail with checksum mismatch
        let mut reader = WalReader::new(&wal_path).unwrap();
        let result = reader.read_entries();

        assert!(result.is_err());
        match result {
            Err(WalError::ChecksumMismatch { offset, .. }) => {
                assert_eq!(offset, 0, "Corruption should be detected at offset 0");
            }
            Err(e) => panic!("Expected ChecksumMismatch, got: {:?}", e),
            Ok(_) => panic!("Expected error but got success"),
        }
    }
}
