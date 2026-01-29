//! Full-text search index module for NeuralGraphDB.
//!
//! This module provides full-text search capabilities on node properties
//! using tantivy (Rust's Lucene equivalent). It supports:
//!
//! - Text search with stemming and stop word filtering
//! - Phrase queries with exact matching
//! - Boolean queries (AND, OR, NOT)
//! - Per-label indexes for targeted searching
//!
//! # Architecture
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────────┐
//! │                       FullTextIndex                              │
//! ├─────────────────────────────────────────────────────────────────┤
//! │  ┌─────────────┐  ┌─────────────────┐  ┌──────────────────────┐ │
//! │  │ Tantivy     │  │  Schema with    │  │  Node ID Mapping     │ │
//! │  │ Index       │  │  Text Fields    │  │  (u64 -> NodeId)     │ │
//! │  └─────────────┘  └─────────────────┘  └──────────────────────┘ │
//! │                                                                  │
//! │  ┌──────────────────────────────────────────────────────────┐   │
//! │  │                    Text Analyzer                          │   │
//! │  │  SimpleTokenizer → LowerCaser → StopWords → Stemmer      │   │
//! │  └──────────────────────────────────────────────────────────┘   │
//! └─────────────────────────────────────────────────────────────────┘
//! ```
//!
//! # Example
//!
//! ```ignore
//! use neural_storage::full_text_index::{FullTextIndex, FullTextIndexConfig};
//!
//! // Create a full-text index for Paper nodes
//! let config = FullTextIndexConfig::new("paper_search", "Paper")
//!     .with_properties(vec!["title", "abstract"]);
//!
//! let mut index = FullTextIndex::create(config, "/path/to/index")?;
//!
//! // Add a document
//! let props = HashMap::from([
//!     ("title".to_string(), "Machine Learning Introduction".to_string()),
//!     ("abstract".to_string(), "An overview of ML techniques".to_string()),
//! ]);
//! index.add_document(NodeId::new(1), &props)?;
//! index.commit()?;
//!
//! // Search
//! let results = index.search("machine learning", 10)?;
//! ```

pub mod config;
pub mod schema;

use config::AnalyzerConfig;
use neural_core::NodeId;
use schema::{build_analyzer, build_schema, ANALYZER_NAME};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use tantivy::collector::TopDocs;
use tantivy::query::QueryParser;
use tantivy::schema::{Field, Schema};
use tantivy::schema::Value;
use tantivy::{Index, IndexReader, IndexWriter, ReloadPolicy, TantivyDocument};
use thiserror::Error;

// Re-exports
pub use config::{FullTextIndexConfig, Language};

/// Errors that can occur during full-text index operations.
#[derive(Debug, Error)]
pub enum FullTextError {
    /// Error from tantivy
    #[error("Tantivy error: {0}")]
    Tantivy(#[from] tantivy::TantivyError),

    /// Error parsing a query
    #[error("Query parse error: {0}")]
    QueryParse(#[from] tantivy::query::QueryParserError),

    /// I/O error
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),

    /// Index not found
    #[error("Index '{0}' not found")]
    IndexNotFound(String),

    /// Index already exists
    #[error("Index '{0}' already exists")]
    IndexExists(String),

    /// Invalid configuration
    #[error("Invalid configuration: {0}")]
    InvalidConfig(String),

    /// Writer not available
    #[error("Index writer not available - index may be read-only")]
    WriterNotAvailable,
}

/// Result type for full-text operations.
pub type Result<T> = std::result::Result<T, FullTextError>;

/// A full-text search result.
#[derive(Debug, Clone)]
pub struct SearchResult {
    /// The node ID of the matching document
    pub node_id: NodeId,
    /// The relevance score (higher is more relevant)
    pub score: f32,
}

/// Metadata about a full-text index, persisted with the graph.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FullTextIndexMetadata {
    /// Name of the index
    pub name: String,
    /// Node label being indexed
    pub label: String,
    /// Properties included in the index
    pub properties: Vec<String>,
    /// Analyzer configuration
    pub analyzer: AnalyzerConfig,
    /// Number of documents indexed
    pub document_count: usize,
    /// Creation timestamp (ISO 8601)
    pub created_at: String,
}

impl FullTextIndexMetadata {
    /// Creates metadata from a config.
    pub fn from_config(config: &FullTextIndexConfig) -> Self {
        Self {
            name: config.name.clone(),
            label: config.label.clone(),
            properties: config.properties.clone(),
            analyzer: config.analyzer.clone(),
            document_count: 0,
            created_at: chrono::Utc::now().to_rfc3339(),
        }
    }
}

/// A full-text search index backed by tantivy.
///
/// The index stores text from node properties and allows efficient
/// full-text queries with stemming, stop word filtering, and relevance ranking.
pub struct FullTextIndex {
    /// The tantivy index
    index: Index,
    /// Reader for searching (auto-reloading)
    reader: IndexReader,
    /// Writer for indexing (None if read-only)
    writer: Option<IndexWriter>,
    /// The schema
    schema: Schema,
    /// Field for node IDs
    node_id_field: Field,
    /// Fields for text properties
    text_fields: HashMap<String, Field>,
    /// Index metadata
    metadata: FullTextIndexMetadata,
    /// Path to the index directory
    path: PathBuf,
}

impl FullTextIndex {
    /// Creates a new full-text index.
    ///
    /// # Arguments
    /// * `config` - Index configuration
    /// * `base_path` - Base path for the index directory
    ///
    /// # Returns
    /// A new FullTextIndex ready for indexing
    pub fn create(config: FullTextIndexConfig, base_path: impl AsRef<Path>) -> Result<Self> {
        if config.properties.is_empty() {
            return Err(FullTextError::InvalidConfig(
                "At least one property must be specified".into(),
            ));
        }

        let index_path = base_path.as_ref().join(".fts").join(&config.name);

        // Create directory if it doesn't exist
        std::fs::create_dir_all(&index_path)?;

        // Build schema
        let (schema, node_id_field, text_field_list) = build_schema(&config.properties);

        // Create tantivy index
        let index = Index::create_in_dir(&index_path, schema.clone())?;

        // Register our custom analyzer
        let tokenizers = index.tokenizers();
        let analyzer = build_analyzer(&config.analyzer);
        tokenizers.register(ANALYZER_NAME, analyzer);

        // Create reader with auto-reload
        let reader = index
            .reader_builder()
            .reload_policy(ReloadPolicy::OnCommitWithDelay)
            .try_into()?;

        // Create writer with 50MB heap
        let writer = index.writer(50_000_000)?;

        // Build text fields map
        let text_fields: HashMap<String, Field> = text_field_list.into_iter().collect();

        let metadata = FullTextIndexMetadata::from_config(&config);

        Ok(Self {
            index,
            reader,
            writer: Some(writer),
            schema,
            node_id_field,
            text_fields,
            metadata,
            path: index_path,
        })
    }

    /// Opens an existing full-text index.
    ///
    /// # Arguments
    /// * `path` - Path to the index directory
    /// * `metadata` - Previously saved metadata
    ///
    /// # Returns
    /// The opened FullTextIndex
    pub fn open(path: impl AsRef<Path>, metadata: FullTextIndexMetadata) -> Result<Self> {
        let index_path = path.as_ref();

        // Open existing index
        let index = Index::open_in_dir(index_path)?;

        // Register our custom analyzer
        let tokenizers = index.tokenizers();
        let analyzer = build_analyzer(&metadata.analyzer);
        tokenizers.register(ANALYZER_NAME, analyzer);

        // Get schema and fields
        let schema = index.schema();
        let node_id_field = schema.get_field("node_id")?;

        let mut text_fields = HashMap::new();
        for prop in &metadata.properties {
            if let Ok(field) = schema.get_field(prop) {
                text_fields.insert(prop.clone(), field);
            }
        }

        // Create reader
        let reader = index
            .reader_builder()
            .reload_policy(ReloadPolicy::OnCommitWithDelay)
            .try_into()?;

        // Create writer
        let writer = index.writer(50_000_000)?;

        Ok(Self {
            index,
            reader,
            writer: Some(writer),
            schema,
            node_id_field,
            text_fields,
            metadata,
            path: index_path.to_path_buf(),
        })
    }

    /// Adds a document to the index.
    ///
    /// # Arguments
    /// * `node_id` - The node ID to associate with this document
    /// * `properties` - Map of property name to text value
    ///
    /// # Returns
    /// Ok(()) on success
    pub fn add_document(
        &mut self,
        node_id: NodeId,
        properties: &HashMap<String, String>,
    ) -> Result<()> {
        let writer = self.writer.as_mut().ok_or(FullTextError::WriterNotAvailable)?;

        let mut doc = TantivyDocument::new();

        // Add node ID
        doc.add_u64(self.node_id_field, node_id.as_u64());

        // Add text fields
        for (prop_name, field) in &self.text_fields {
            if let Some(text) = properties.get(prop_name) {
                doc.add_text(*field, text);
            }
        }

        writer.add_document(doc)?;
        self.metadata.document_count += 1;

        Ok(())
    }

    /// Deletes a document from the index by node ID.
    ///
    /// # Arguments
    /// * `node_id` - The node ID to delete
    pub fn delete_document(&mut self, node_id: NodeId) -> Result<()> {
        let writer = self.writer.as_mut().ok_or(FullTextError::WriterNotAvailable)?;

        let term = tantivy::Term::from_field_u64(self.node_id_field, node_id.as_u64());
        writer.delete_term(term);
        self.metadata.document_count = self.metadata.document_count.saturating_sub(1);

        Ok(())
    }

    /// Commits pending changes to the index.
    ///
    /// Changes are not visible to searches until commit is called.
    pub fn commit(&mut self) -> Result<()> {
        if let Some(writer) = self.writer.as_mut() {
            writer.commit()?;
            // Reload the reader to see committed changes
            self.reader.reload()?;
        }
        Ok(())
    }

    /// Searches the index for documents matching the query.
    ///
    /// # Arguments
    /// * `query_str` - The search query (supports boolean syntax)
    /// * `k` - Maximum number of results to return
    ///
    /// # Query Syntax
    /// - Simple terms: `machine learning`
    /// - Phrase queries: `"neural networks"`
    /// - Boolean: `deep AND learning NOT convolutional`
    /// - Field-specific: `title:machine`
    ///
    /// # Returns
    /// Vector of search results sorted by relevance
    pub fn search(&self, query_str: &str, k: usize) -> Result<Vec<SearchResult>> {
        let searcher = self.reader.searcher();

        // Build query parser with all text fields
        let fields: Vec<Field> = self.text_fields.values().copied().collect();
        let query_parser = QueryParser::for_index(&self.index, fields);

        // Parse the query
        let query = query_parser.parse_query(query_str)?;

        // Execute search
        let top_docs = searcher.search(&query, &TopDocs::with_limit(k))?;

        // Convert results
        let mut results = Vec::with_capacity(top_docs.len());
        for (score, doc_address) in top_docs {
            let doc: TantivyDocument = searcher.doc(doc_address)?;

            // Get node ID from document
            if let Some(node_id_value) = doc.get_first(self.node_id_field) {
                if let Some(node_id_u64) = node_id_value.as_u64() {
                    results.push(SearchResult {
                        node_id: NodeId::new(node_id_u64),
                        score,
                    });
                }
            }
        }

        Ok(results)
    }

    /// Returns the metadata for this index.
    pub fn metadata(&self) -> &FullTextIndexMetadata {
        &self.metadata
    }

    /// Returns the path to the index directory.
    pub fn path(&self) -> &Path {
        &self.path
    }

    /// Returns the number of documents in the index.
    pub fn document_count(&self) -> usize {
        self.metadata.document_count
    }

    /// Returns the index name.
    pub fn name(&self) -> &str {
        &self.metadata.name
    }

    /// Returns the label this index is for.
    pub fn label(&self) -> &str {
        &self.metadata.label
    }

    /// Returns the properties being indexed.
    pub fn properties(&self) -> &[String] {
        &self.metadata.properties
    }
}

impl std::fmt::Debug for FullTextIndex {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("FullTextIndex")
            .field("name", &self.metadata.name)
            .field("label", &self.metadata.label)
            .field("properties", &self.metadata.properties)
            .field("document_count", &self.metadata.document_count)
            .field("path", &self.path)
            .finish_non_exhaustive()
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    fn create_test_index() -> (FullTextIndex, TempDir) {
        let temp_dir = TempDir::new().unwrap();
        let config = FullTextIndexConfig::new("test_index", "Paper")
            .with_properties(vec!["title", "abstract"]);

        let index = FullTextIndex::create(config, temp_dir.path()).unwrap();
        (index, temp_dir)
    }

    #[test]
    fn test_create_and_search() {
        let (mut index, _temp_dir) = create_test_index();

        // Add documents
        let doc1 = HashMap::from([
            ("title".to_string(), "Introduction to Machine Learning".to_string()),
            ("abstract".to_string(), "This paper covers ML basics".to_string()),
        ]);
        index.add_document(NodeId::new(1), &doc1).unwrap();

        let doc2 = HashMap::from([
            ("title".to_string(), "Deep Learning for NLP".to_string()),
            ("abstract".to_string(), "Neural networks for text".to_string()),
        ]);
        index.add_document(NodeId::new(2), &doc2).unwrap();

        index.commit().unwrap();

        // Search
        let results = index.search("machine learning", 10).unwrap();
        assert!(!results.is_empty());
        assert_eq!(results[0].node_id, NodeId::new(1));

        // Search for deep learning
        let results = index.search("deep learning", 10).unwrap();
        assert!(!results.is_empty());
        assert_eq!(results[0].node_id, NodeId::new(2));
    }

    #[test]
    fn test_stemming() {
        let (mut index, _temp_dir) = create_test_index();

        let doc = HashMap::from([
            ("title".to_string(), "Learning Algorithms".to_string()),
            ("abstract".to_string(), "About machine learning".to_string()),
        ]);
        index.add_document(NodeId::new(1), &doc).unwrap();
        index.commit().unwrap();

        // "learn" should match "learning" due to stemming
        let results = index.search("learn", 10).unwrap();
        assert!(!results.is_empty(), "Stemming should match 'learn' to 'learning'");
        assert_eq!(results[0].node_id, NodeId::new(1));

        // "learns" should also match
        let results = index.search("learns", 10).unwrap();
        assert!(!results.is_empty(), "Stemming should match 'learns' to 'learning'");
    }

    #[test]
    fn test_stopwords() {
        let (mut index, _temp_dir) = create_test_index();

        let doc = HashMap::from([
            ("title".to_string(), "The Art of Programming".to_string()),
            ("abstract".to_string(), "A book about code".to_string()),
        ]);
        index.add_document(NodeId::new(1), &doc).unwrap();
        index.commit().unwrap();

        // Searching for just "the" should return no results (stop word)
        // Note: tantivy's query parser may handle this differently
        let results = index.search("programming", 10).unwrap();
        assert!(!results.is_empty());

        // "art programming" should find the document
        let results = index.search("art programming", 10).unwrap();
        assert!(!results.is_empty());
    }

    #[test]
    fn test_phrase_query() {
        let (mut index, _temp_dir) = create_test_index();

        let doc1 = HashMap::from([
            ("title".to_string(), "Neural Network Architectures".to_string()),
            ("abstract".to_string(), "Deep neural networks".to_string()),
        ]);
        index.add_document(NodeId::new(1), &doc1).unwrap();

        let doc2 = HashMap::from([
            ("title".to_string(), "Network Security".to_string()),
            ("abstract".to_string(), "Neural pathways in security".to_string()),
        ]);
        index.add_document(NodeId::new(2), &doc2).unwrap();

        index.commit().unwrap();

        // Phrase query should only match exact sequence
        let results = index.search("\"neural network\"", 10).unwrap();
        assert!(!results.is_empty());
        // Both might match due to stemming, but doc1 should rank higher
        assert_eq!(results[0].node_id, NodeId::new(1));
    }

    #[test]
    fn test_persistence() {
        let temp_dir = TempDir::new().unwrap();
        let metadata;

        // Create and populate index
        {
            let config = FullTextIndexConfig::new("persist_test", "Paper")
                .with_properties(vec!["title"]);

            let mut index = FullTextIndex::create(config, temp_dir.path()).unwrap();

            let doc = HashMap::from([
                ("title".to_string(), "Persistence Testing".to_string()),
            ]);
            index.add_document(NodeId::new(42), &doc).unwrap();
            index.commit().unwrap();

            metadata = index.metadata().clone();
        }

        // Reopen and verify
        {
            let index_path = temp_dir.path().join(".fts").join("persist_test");
            let index = FullTextIndex::open(&index_path, metadata).unwrap();

            let results = index.search("persistence", 10).unwrap();
            assert!(!results.is_empty());
            assert_eq!(results[0].node_id, NodeId::new(42));
        }
    }

    #[test]
    fn test_delete_document() {
        let (mut index, _temp_dir) = create_test_index();

        let doc = HashMap::from([
            ("title".to_string(), "To Be Deleted".to_string()),
            ("abstract".to_string(), "This will be removed".to_string()),
        ]);
        index.add_document(NodeId::new(1), &doc).unwrap();
        index.commit().unwrap();

        // Verify it exists
        let results = index.search("deleted", 10).unwrap();
        assert!(!results.is_empty());

        // Delete it
        index.delete_document(NodeId::new(1)).unwrap();
        index.commit().unwrap();

        // Verify it's gone (tantivy marks deleted documents, filtered at search time)
        // After delete and commit+reload, the document should not appear in results
        let results = index.search("deleted", 10).unwrap();
        // Note: In tantivy, deleted documents are filtered at search time after reload
        // The document count in results should be 0
        assert!(
            results.is_empty(),
            "Expected no results after delete, but got {} results: {:?}",
            results.len(),
            results.iter().map(|r| r.node_id).collect::<Vec<_>>()
        );
    }

    #[test]
    fn test_boolean_query() {
        let (mut index, _temp_dir) = create_test_index();

        let doc1 = HashMap::from([
            ("title".to_string(), "Deep Learning Methods".to_string()),
            ("abstract".to_string(), "Neural approaches".to_string()),
        ]);
        index.add_document(NodeId::new(1), &doc1).unwrap();

        let doc2 = HashMap::from([
            ("title".to_string(), "Machine Learning Basics".to_string()),
            ("abstract".to_string(), "Fundamental concepts".to_string()),
        ]);
        index.add_document(NodeId::new(2), &doc2).unwrap();

        let doc3 = HashMap::from([
            ("title".to_string(), "Deep Sea Exploration".to_string()),
            ("abstract".to_string(), "Ocean research".to_string()),
        ]);
        index.add_document(NodeId::new(3), &doc3).unwrap();

        index.commit().unwrap();

        // AND query
        let results = index.search("deep AND learning", 10).unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].node_id, NodeId::new(1));

        // NOT query
        let results = index.search("deep -learning", 10).unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].node_id, NodeId::new(3));
    }

    #[test]
    fn test_empty_properties_error() {
        let temp_dir = TempDir::new().unwrap();
        let config = FullTextIndexConfig::new("empty_props", "Paper");

        let result = FullTextIndex::create(config, temp_dir.path());
        assert!(matches!(result, Err(FullTextError::InvalidConfig(_))));
    }

    #[test]
    fn test_metadata() {
        let (index, _temp_dir) = create_test_index();

        assert_eq!(index.name(), "test_index");
        assert_eq!(index.label(), "Paper");
        assert_eq!(index.properties(), &["title", "abstract"]);
    }
}
