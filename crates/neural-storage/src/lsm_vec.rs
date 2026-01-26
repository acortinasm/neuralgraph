use std::fs::File;
use std::io::{self, BufReader, BufWriter, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};

use neural_core::NodeId;
use thiserror::Error;
use bincode;
use rand::prelude::*;

use crate::vector_index::{VectorIndex, cosine_similarity};

// =============================================================================
// Error Handling
// =============================================================================

#[derive(Debug, Error)]
pub enum LsmError {
    #[error("I/O error: {0}")]
    Io(#[from] io::Error),
    #[error("Serialization error: {0}")]
    Serialization(#[from] bincode::Error),
    #[error("Dimension mismatch: expected {0}, got {1}")]
    DimensionMismatch(usize, usize),
    #[error("Index empty, cannot build")]
    IndexEmpty,
    #[error("Buffer error: {0}")]
    IntoInner(String),
}

impl<W> From<std::io::IntoInnerError<W>> for LsmError {
    fn from(err: std::io::IntoInnerError<W>) -> Self {
        Self::IntoInner(err.to_string())
    }
}

// =============================================================================
// Disk Index (SSTable Equivalent)
// =============================================================================

/// Header for the Disk-based Vector Index (IVF-Flat format).
#[derive(Debug, serde::Serialize, serde::Deserialize)]
struct DiskIndexHeader {
    magic: u32,       // "IVF1"
    version: u32,     // 1
    count: usize,     // Total vectors
    dimension: usize, // Vector dimension
    num_centroids: usize, // Number of clusters (IVF lists)
}

/// A read-only, disk-resident vector index using IVF-Flat.
///
/// Layout:
/// [Header]
/// [Centroids (K * Dim * f32)]
/// [Cluster Offsets (K * u64)] -> Point to start of each cluster's data
/// [Cluster Sizes (K * u64)]   -> Number of vectors in each cluster
/// [Data Region]
///    [Cluster 0: (NodeId, Vector)...]
///    [Cluster 1: (NodeId, Vector)...]
pub struct DiskIndex {
    path: PathBuf,
    header: DiskIndexHeader,
    centroids: Vec<Vec<f32>>,
    cluster_offsets: Vec<u64>,
    cluster_sizes: Vec<u64>,
}

impl DiskIndex {
    const MAGIC: u32 = 0x49564631; // "IVF1"

    /// Opens an existing DiskIndex.
    pub fn open(path: impl AsRef<Path>) -> Result<Self, LsmError> {
        let path = path.as_ref().to_path_buf();
        let file = File::open(&path)?;
        let mut reader = BufReader::new(&file);

        // 1. Read Header
        let header: DiskIndexHeader = bincode::deserialize_from(&mut reader)?;
        if header.magic != Self::MAGIC {
            return Err(io::Error::new(io::ErrorKind::InvalidData, "Invalid magic bytes").into());
        }

        // 2. Read Centroids
        let centroids: Vec<Vec<f32>> = bincode::deserialize_from(&mut reader)?;

        // 3. Read Offsets and Sizes
        let cluster_offsets: Vec<u64> = bincode::deserialize_from(&mut reader)?;
        let cluster_sizes: Vec<u64> = bincode::deserialize_from(&mut reader)?;

        Ok(Self {
            path,
            header,
            centroids,
            cluster_offsets,
            cluster_sizes,
        })
    }

    /// Builds a DiskIndex from a set of vectors and writes it to disk.
    ///
    /// This is a simplified "Compaction" or "Flush" operation.
    /// It uses random centroid initialization for the prototype.
    pub fn build(
        path: impl AsRef<Path>,
        vectors: &[(NodeId, Vec<f32>)],
        dimension: usize,
        num_centroids: usize,
    ) -> Result<(), LsmError> {
        if vectors.is_empty() {
            return Err(LsmError::IndexEmpty);
        }

        let mut file = BufWriter::new(File::create(path)?);
        let mut rng = rand::thread_rng();

        // 1. Train Centroids (Random Selection for Prototype)
        // In prod, use K-Means.
        let mut centroids = Vec::new();
        for _ in 0..num_centroids {
             if let Some((_, v)) = vectors.choose(&mut rng) {
                 centroids.push(v.clone());
             }
        }
        
        // 2. Assign Vectors to Clusters
        let mut clusters: Vec<Vec<(NodeId, Vec<f32>)>> = vec![Vec::new(); num_centroids];
        
        for (id, vec) in vectors {
            if vec.len() != dimension {
                return Err(LsmError::DimensionMismatch(dimension, vec.len()));
            }

            // Find nearest centroid
            let mut best_sim = -1.0;
            let mut best_idx = 0;
            
            for (i, centroid) in centroids.iter().enumerate() {
                let sim = cosine_similarity(vec, centroid);
                if sim > best_sim {
                    best_sim = sim;
                    best_idx = i;
                }
            }
            clusters[best_idx].push((*id, vec.clone()));
        }

        // 3. Write Header
        let header = DiskIndexHeader {
            magic: Self::MAGIC,
            version: 1,
            count: vectors.len(),
            dimension,
            num_centroids: centroids.len(),
        };
        bincode::serialize_into(&mut file, &header)?;
        bincode::serialize_into(&mut file, &centroids)?;

        // Placeholder for offsets - we will overwrite later
        let offset_pos = file.stream_position()?;
        // Write dummy offsets/sizes
        let dummy_vec = vec![0u64; num_centroids];
        bincode::serialize_into(&mut file, &dummy_vec)?; // Offsets
        bincode::serialize_into(&mut file, &dummy_vec)?; // Sizes

        // 4. Write Clusters and Record Offsets
        let mut real_offsets = Vec::new();
        let mut real_sizes = Vec::new();

        for cluster in &clusters {
            let start_pos = file.stream_position()?;
            real_offsets.push(start_pos);
            real_sizes.push(cluster.len() as u64);

            // Write vectors in this cluster
            for item in cluster {
                bincode::serialize_into(&mut file, item)?;
            }
        }

        // 5. Overwrite Offsets/Sizes
        file.flush()?;
        let mut file = file.into_inner()?; // Get underlying File
        file.seek(SeekFrom::Start(offset_pos))?;
        bincode::serialize_into(&mut file, &real_offsets)?;
        bincode::serialize_into(&mut file, &real_sizes)?;

        Ok(())
    }

    /// Searches the disk index.
    ///
    /// 1. Finds the nearest `n_probes` centroids to the query.
    /// 2. Reads only those clusters from disk.
    /// 3. Scans vectors in those clusters.
    pub fn search(&self, query: &[f32], k: usize, n_probes: usize) -> Result<Vec<(NodeId, f32)>, LsmError> {
        if query.len() != self.header.dimension {
             return Err(LsmError::DimensionMismatch(self.header.dimension, query.len()));
        }

        // 1. Find nearest centroids
        let mut centroid_scores: Vec<(usize, f32)> = self.centroids
            .iter()
            .enumerate()
            .map(|(i, c)| (i, cosine_similarity(query, c)))
            .collect();
        
        // Sort DESC by similarity
        centroid_scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        let probes = n_probes.min(centroid_scores.len());
        let target_clusters: Vec<usize> = centroid_scores.iter().take(probes).map(|(i, _)| *i).collect();

        // 2. Scan Target Clusters
        let mut candidates: Vec<(NodeId, f32)> = Vec::new();
        let mut file = BufReader::new(File::open(&self.path)?);

        for &cluster_idx in &target_clusters {
            let offset = self.cluster_offsets[cluster_idx];
            let size = self.cluster_sizes[cluster_idx];
            
            if size == 0 { continue; }

            // Seek to cluster start
            file.seek(SeekFrom::Start(offset))?;

            // Read all vectors in cluster
            for _ in 0..size {
                 let (id, vec): (NodeId, Vec<f32>) = bincode::deserialize_from(&mut file)?;
                 let sim = cosine_similarity(query, &vec);
                 candidates.push((id, sim));
            }
        }

        // 3. Sort and Top-K
        candidates.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        candidates.truncate(k);

        Ok(candidates)
    }
}

// =============================================================================
// LSM Orchestrator
// =============================================================================

/// The main LSM-VEC structure managing MemTable and SSTables.
#[allow(dead_code)]
pub struct LsmVectorIndex {
    /// In-memory HNSW (Mutable)
    mem_index: VectorIndex,
    /// On-disk IVF Indices (Immutable)
    disk_indices: Vec<DiskIndex>,
    /// Flush threshold (number of vectors)
    threshold: usize,
    /// Directory for storing segment files
    base_path: PathBuf,
}

impl LsmVectorIndex {
    pub fn new(base_path: impl AsRef<Path>, dimension: usize, threshold: usize) -> Self {
        // Ensure directory exists
        std::fs::create_dir_all(&base_path).unwrap_or_default();

        Self {
            mem_index: VectorIndex::new(dimension),
            disk_indices: Vec::new(),
            threshold,
            base_path: base_path.as_ref().to_path_buf(),
        }
    }

    /// Adds a vector. Flushes to disk if threshold is reached.
    /// NOTE: In a real implementation, we need to track raw vectors in RAM 
    /// separate from HNSW to support flushing (since HNSW is graph-only).
    /// For this prototype, we assume we can't extract from HNSW efficiently
    /// without keeping a copy, so we'll skip the "Flush from HNSW" logic 
    /// and just support "Build Disk Index from provided list" for now.
    pub fn insert(&mut self, node: NodeId, vector: &[f32]) {
        self.mem_index.add(node, vector);
        // TODO: Track size and trigger flush
    }

    /// Manually loads a disk segment.
    pub fn load_segment(&mut self, path: PathBuf) -> Result<(), LsmError> {
        let index = DiskIndex::open(path)?;
        self.disk_indices.push(index);
        Ok(())
    }

    /// Unified Search (Mem + Disk)
    pub fn search(&self, query: &[f32], k: usize) -> Result<Vec<(NodeId, f32)>, LsmError> {
        // 1. Search Memory
        let mut results = self.mem_index.search(query, k);

        // 2. Search Disk
        for disk_idx in &self.disk_indices {
            // Use n_probes = 3 for demo
            let disk_results = disk_idx.search(query, k, 3)?;
            results.extend(disk_results);
        }

        // 3. Merge and Sort
        results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        
        // Dedup (Keep first/best occurrence of a NodeId)
        let mut unique = Vec::new();
        let mut seen = std::collections::HashSet::new();
        for (node, score) in results {
            if seen.insert(node) {
                unique.push((node, score));
            }
            if unique.len() >= k { break; }
        }

        Ok(unique)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::NamedTempFile;

    #[test]
    fn test_disk_index_build_and_search() {
        let file = NamedTempFile::new().unwrap();
        let path = file.path().to_path_buf();

        // Create 3 vectors (dim=2)
        // Cluster 1: [1.0, 0.0]
        // Cluster 2: [0.0, 1.0]
        let vectors = vec![
            (NodeId::new(1), vec![1.0, 0.0]),
            (NodeId::new(2), vec![0.9, 0.1]), // Close to 1
            (NodeId::new(3), vec![0.0, 1.0]),
            (NodeId::new(4), vec![0.1, 0.9]), // Close to 3
        ];

        // Build Index with 2 centroids
        DiskIndex::build(&path, &vectors, 2, 2).unwrap();

        // Open Index
        let index = DiskIndex::open(&path).unwrap();
        assert_eq!(index.header.count, 4);
        assert_eq!(index.header.num_centroids, 2);

        // Search for [1, 0] -> Should find Node 1 and 2
        let results = index.search(&[1.0, 0.0], 2, 1).unwrap();
        assert_eq!(results.len(), 2);
        assert!(results.iter().any(|(id, _)| id.0 == 1));
        assert!(results.iter().any(|(id, _)| id.0 == 2));
        
        // Ensure scores are high
        assert!(results[0].1 > 0.9);
    }

    #[test]
    fn test_lsm_integration() {
        let dir = tempfile::tempdir().unwrap();
        let mut lsm = LsmVectorIndex::new(dir.path(), 2, 100);

        // Add to Mem
        lsm.insert(NodeId::new(10), &[0.5, 0.5]);

        // Create a disk segment manually
        let segment_path = dir.path().join("segment_1.ivf");
        let vectors = vec![
            (NodeId::new(20), vec![0.9, 0.0]),
        ];
        DiskIndex::build(&segment_path, &vectors, 2, 1).unwrap();
        lsm.load_segment(segment_path).unwrap();

        // Search matches both Mem and Disk
        let results = lsm.search(&[0.5, 0.5], 2).unwrap();
        
        // Should find node 10 (Mem) and maybe node 20 (Disk) if k allows
        assert!(results.iter().any(|(id, _)| id.0 == 10));
        assert!(results.iter().any(|(id, _)| id.0 == 20)); // 0.9,0.0 is somewhat similar to 0.5,0.5
    }
}
