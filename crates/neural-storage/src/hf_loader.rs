//! HuggingFace dataset loading for NeuralGraphDB
//!
//! This module provides functionality to download and import datasets
//! from HuggingFace Hub for benchmarking and testing.
//!
//! ## Supported Datasets
//!
//! - `CShorten/ML-ArXiv-Papers` - 117K ML papers with title/abstract
//!
//! ## Usage
//!
//! ```rust,ignore
//! use neural_storage::hf_loader::{HfDataset, load_hf_dataset};
//!
//! let store = load_hf_dataset(HfDataset::MlArxivPapers, None)?;
//! println!("Loaded {} papers", store.node_count());
//! ```

use crate::{GraphStore, GraphStoreBuilder};
use arrow::array::{Array, StringArray};
use arrow::record_batch::RecordBatch;
use neural_core::{Graph, PropertyValue};
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
use std::fs::{self, File};
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use thiserror::Error;

/// Errors that can occur during HuggingFace loading
#[derive(Error, Debug)]
pub enum HfError {
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    #[error("HTTP error: {0}")]
    Http(#[from] reqwest::Error),

    #[error("Parquet error: {0}")]
    Parquet(#[from] parquet::errors::ParquetError),

    #[error("Arrow error: {0}")]
    Arrow(#[from] arrow::error::ArrowError),

    #[error("Dataset not found: {0}")]
    DatasetNotFound(String),

    #[error("Missing column in dataset: {0}")]
    MissingColumn(String),

    #[error("Cache directory error: {0}")]
    CacheError(String),
}

/// Result type for HuggingFace operations
pub type Result<T> = std::result::Result<T, HfError>;

/// Supported HuggingFace datasets
#[derive(Debug, Clone)]
pub enum HfDataset {
    /// ML ArXiv Papers - 117K papers with title/abstract
    /// Dataset: CShorten/ML-ArXiv-Papers
    MlArxivPapers,

    /// Custom dataset by repository name
    Custom {
        repo: String,
        /// Column to use as node label
        label: String,
    },
}

impl HfDataset {
    /// Returns the HuggingFace repository path
    pub fn repo(&self) -> &str {
        match self {
            HfDataset::MlArxivPapers => "CShorten/ML-ArXiv-Papers",
            HfDataset::Custom { repo, .. } => repo,
        }
    }

    /// Returns the default label for nodes
    pub fn default_label(&self) -> &str {
        match self {
            HfDataset::MlArxivPapers => "Paper",
            HfDataset::Custom { label, .. } => label,
        }
    }

    /// Returns the Parquet file URL for auto-converted datasets
    pub fn parquet_url(&self) -> String {
        // HuggingFace auto-converts datasets to Parquet
        // URL format: https://huggingface.co/datasets/{repo}/resolve/refs%2Fconvert%2Fparquet/{config}/{split}/0000.parquet
        let repo = self.repo();
        format!(
            "https://huggingface.co/datasets/{}/resolve/refs%2Fconvert%2Fparquet/default/train/0000.parquet",
            repo
        )
    }

    /// Returns the cache filename
    pub fn cache_filename(&self) -> String {
        let repo = self.repo().replace('/', "_");
        format!("{}.parquet", repo)
    }
}

/// Progress callback for download operations
pub type ProgressCallback = Box<dyn Fn(u64, u64)>;

/// Downloads a dataset from HuggingFace Hub
///
/// # Arguments
///
/// * `dataset` - The dataset to download
/// * `cache_dir` - Directory to cache downloaded files
/// * `progress` - Optional progress callback (bytes_downloaded, total_bytes)
///
/// # Returns
///
/// Path to the downloaded Parquet file
pub fn download_dataset(
    dataset: &HfDataset,
    cache_dir: &Path,
    progress: Option<ProgressCallback>,
) -> Result<PathBuf> {
    // Ensure cache directory exists
    fs::create_dir_all(cache_dir).map_err(|e| HfError::CacheError(e.to_string()))?;

    let cache_path = cache_dir.join(dataset.cache_filename());

    // Check if already cached
    if cache_path.exists() {
        return Ok(cache_path);
    }

    let url = dataset.parquet_url();

    // Download the file
    let client = reqwest::blocking::Client::builder()
        .user_agent("NeuralGraphDB/0.1.0")
        .build()?;

    let mut response = client.get(&url).send()?;

    if !response.status().is_success() {
        return Err(HfError::DatasetNotFound(format!(
            "Failed to download {}: HTTP {}",
            dataset.repo(),
            response.status()
        )));
    }

    let total_size = response.content_length().unwrap_or(0);
    let mut downloaded: u64 = 0;

    let mut file = File::create(&cache_path)?;
    let mut buffer = [0u8; 8192];

    loop {
        let bytes_read = response.read(&mut buffer)?;
        if bytes_read == 0 {
            break;
        }

        file.write_all(&buffer[..bytes_read])?;
        downloaded += bytes_read as u64;

        if let Some(ref cb) = progress {
            cb(downloaded, total_size);
        }
    }

    Ok(cache_path)
}

/// Loads a HuggingFace dataset as a graph
///
/// # Arguments
///
/// * `dataset` - The dataset to load
/// * `cache_dir` - Optional cache directory (defaults to ~/.cache/neuralgraph)
///
/// # Returns
///
/// A GraphStore with the dataset loaded
pub fn load_hf_dataset(dataset: HfDataset, cache_dir: Option<&Path>) -> Result<GraphStore> {
    let default_cache = dirs_cache_dir();
    let cache = cache_dir.unwrap_or(&default_cache);

    let parquet_path = download_dataset(&dataset, cache, None)?;

    match dataset {
        HfDataset::MlArxivPapers => load_arxiv_papers(&parquet_path),
        HfDataset::Custom { label, .. } => load_generic_dataset(&parquet_path, &label),
    }
}

/// Loads ML-ArXiv-Papers dataset as a graph
///
/// Each paper becomes a node with:
/// - Label: "Paper"
/// - Properties: title, abstract
fn load_arxiv_papers(parquet_path: &Path) -> Result<GraphStore> {
    let file = File::open(parquet_path)?;
    let builder = ParquetRecordBatchReaderBuilder::try_new(file)?;
    let reader = builder.build()?;

    let mut graph_builder = GraphStoreBuilder::new();
    let mut node_id: u64 = 0;

    for batch_result in reader {
        let batch: RecordBatch = batch_result?;

        // Get column arrays
        let title_col = get_string_column(&batch, "title")?;
        let abstract_col = get_string_column(&batch, "abstract")?;

        for i in 0..batch.num_rows() {
            let title = title_col.value(i);
            let abstract_text = abstract_col.value(i);

            let properties: Vec<(&str, PropertyValue)> = vec![
                ("title", PropertyValue::String(title.to_string())),
                ("abstract", PropertyValue::String(abstract_text.to_string())),
            ];

            graph_builder = graph_builder.add_labeled_node(node_id, "Paper", properties);
            node_id += 1;
        }
    }

    Ok(graph_builder.build())
}

/// Loads a generic dataset with each row as a node
fn load_generic_dataset(parquet_path: &Path, label: &str) -> Result<GraphStore> {
    let file = File::open(parquet_path)?;
    let builder = ParquetRecordBatchReaderBuilder::try_new(file)?;
    let reader = builder.build()?;

    let mut graph_builder = GraphStoreBuilder::new();
    let mut node_id: u64 = 0;

    for batch_result in reader {
        let batch: RecordBatch = batch_result?;
        let schema = batch.schema();

        for i in 0..batch.num_rows() {
            let mut properties: Vec<(String, PropertyValue)> = Vec::new();

            for (col_idx, field) in schema.fields().iter().enumerate() {
                let col = batch.column(col_idx);
                if let Some(string_array) = col.as_any().downcast_ref::<StringArray>() {
                    if !string_array.is_null(i) {
                        properties.push((
                            field.name().clone(),
                            PropertyValue::String(string_array.value(i).to_string()),
                        ));
                    }
                }
            }

            graph_builder = graph_builder.add_labeled_node(node_id, label, properties);
            node_id += 1;
        }
    }

    Ok(graph_builder.build())
}

/// Gets a string column from a RecordBatch
fn get_string_column<'a>(batch: &'a RecordBatch, name: &str) -> Result<&'a StringArray> {
    let col_idx = batch
        .schema()
        .index_of(name)
        .map_err(|_| HfError::MissingColumn(name.to_string()))?;

    batch
        .column(col_idx)
        .as_any()
        .downcast_ref::<StringArray>()
        .ok_or_else(|| HfError::MissingColumn(format!("{} is not a string column", name)))
}

/// Returns the default cache directory
fn dirs_cache_dir() -> PathBuf {
    dirs::cache_dir()
        .unwrap_or_else(|| PathBuf::from("."))
        .join("neuralgraph")
}

/// Statistics about a loaded dataset
#[derive(Debug, Clone)]
pub struct DatasetStats {
    pub num_nodes: usize,
    pub num_edges: usize,
    pub labels: Vec<String>,
    pub memory_bytes: usize,
}

impl DatasetStats {
    /// Creates stats from a GraphStore
    pub fn from_store(store: &GraphStore) -> Self {
        Self {
            num_nodes: store.node_count(),
            num_edges: store.edge_count(),
            labels: vec![],  // TODO: count unique labels
            memory_bytes: 0, // TODO: calculate memory
        }
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dataset_urls() {
        let dataset = HfDataset::MlArxivPapers;
        assert_eq!(dataset.repo(), "CShorten/ML-ArXiv-Papers");
        assert!(dataset.parquet_url().contains("huggingface.co"));
        assert!(dataset.parquet_url().contains("CShorten"));
    }

    #[test]
    fn test_cache_filename() {
        let dataset = HfDataset::MlArxivPapers;
        assert_eq!(dataset.cache_filename(), "CShorten_ML-ArXiv-Papers.parquet");
    }

    #[test]
    fn test_custom_dataset() {
        let dataset = HfDataset::Custom {
            repo: "user/dataset".to_string(),
            label: "Document".to_string(),
        };
        assert_eq!(dataset.repo(), "user/dataset");
        assert_eq!(dataset.default_label(), "Document");
    }

    // Integration test - only run manually
    #[test]
    #[ignore]
    fn test_download_arxiv_papers() {
        let dataset = HfDataset::MlArxivPapers;
        let cache_dir = PathBuf::from("/tmp/neuralgraph_test");

        let result = download_dataset(&dataset, &cache_dir, None);
        assert!(result.is_ok());

        let path = result.unwrap();
        assert!(path.exists());
    }
}
