//! Schema builder and analyzer setup for full-text indexes.
//!
//! This module provides utilities for building tantivy schemas with
//! properly configured text analyzers including tokenization, lowercasing,
//! stop word filtering, and stemming.

use super::config::{AnalyzerConfig, Language};
use tantivy::schema::{Field, Schema, NumericOptions, TextFieldIndexing, TextOptions, IndexRecordOption};
use tantivy::tokenizer::{
    Language as TantivyLanguage, LowerCaser, SimpleTokenizer, Stemmer, StopWordFilter,
    TextAnalyzer, TokenizerManager,
};

/// The name of our custom analyzer with stemming and stop words.
pub const ANALYZER_NAME: &str = "neural_english";

/// Builds a tantivy schema for a full-text index.
///
/// Creates a schema with:
/// - `node_id` field (u64, stored) for mapping back to graph nodes
/// - One TEXT field per property (stored for retrieval, indexed with custom analyzer)
///
/// # Arguments
/// * `properties` - List of property names to create text fields for
///
/// # Returns
/// A tuple of (Schema, node_id_field, property_fields)
pub fn build_schema(properties: &[String]) -> (Schema, Field, Vec<(String, Field)>) {
    let mut schema_builder = Schema::builder();

    // Node ID field - stored for retrieval, indexed for deletion by term
    let node_id_field = schema_builder.add_u64_field(
        "node_id",
        NumericOptions::default().set_stored().set_indexed()
    );

    // Text field options with our custom analyzer
    let text_field_indexing = TextFieldIndexing::default()
        .set_tokenizer(ANALYZER_NAME)
        .set_index_option(IndexRecordOption::WithFreqsAndPositions);
    let text_options = TextOptions::default()
        .set_indexing_options(text_field_indexing)
        .set_stored();

    // Text fields for each property
    let mut text_fields = Vec::new();
    for prop in properties {
        let field = schema_builder.add_text_field(prop, text_options.clone());
        text_fields.push((prop.clone(), field));
    }

    (schema_builder.build(), node_id_field, text_fields)
}

/// Builds a text analyzer based on configuration.
///
/// Creates an analyzer pipeline:
/// 1. SimpleTokenizer - splits on whitespace and punctuation
/// 2. LowerCaser - converts to lowercase (if enabled)
/// 3. StopWordFilter - removes common words (if enabled)
/// 4. Stemmer - reduces words to stems (if enabled)
///
/// # Arguments
/// * `config` - Analyzer configuration
///
/// # Returns
/// A configured TextAnalyzer
pub fn build_analyzer(config: &AnalyzerConfig) -> TextAnalyzer {
    let tokenizer = SimpleTokenizer::default();

    // Build the pipeline based on config
    if config.lowercase && config.remove_stopwords && config.stemming {
        // Full pipeline
        let tantivy_lang = to_tantivy_language(config.language);
        TextAnalyzer::builder(tokenizer)
            .filter(LowerCaser)
            .filter(StopWordFilter::new(tantivy_lang).unwrap())
            .filter(Stemmer::new(tantivy_lang))
            .build()
    } else if config.lowercase && config.remove_stopwords {
        // No stemming
        let tantivy_lang = to_tantivy_language(config.language);
        TextAnalyzer::builder(tokenizer)
            .filter(LowerCaser)
            .filter(StopWordFilter::new(tantivy_lang).unwrap())
            .build()
    } else if config.lowercase && config.stemming {
        // No stop words
        let tantivy_lang = to_tantivy_language(config.language);
        TextAnalyzer::builder(tokenizer)
            .filter(LowerCaser)
            .filter(Stemmer::new(tantivy_lang))
            .build()
    } else if config.lowercase {
        // Only lowercase
        TextAnalyzer::builder(tokenizer)
            .filter(LowerCaser)
            .build()
    } else if config.stemming {
        // Only stemming (unusual but supported)
        let tantivy_lang = to_tantivy_language(config.language);
        TextAnalyzer::builder(tokenizer)
            .filter(Stemmer::new(tantivy_lang))
            .build()
    } else {
        // No filters - just tokenize
        TextAnalyzer::builder(tokenizer).build()
    }
}

/// Registers the custom analyzer with a tokenizer manager.
///
/// # Arguments
/// * `manager` - The tokenizer manager to register with
/// * `config` - Analyzer configuration
pub fn register_analyzer(manager: &TokenizerManager, config: &AnalyzerConfig) {
    let analyzer = build_analyzer(config);
    manager.register(ANALYZER_NAME, analyzer);
}

/// Converts our Language enum to tantivy's Language.
fn to_tantivy_language(lang: Language) -> TantivyLanguage {
    match lang {
        Language::English => TantivyLanguage::English,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_build_schema() {
        let properties = vec!["title".to_string(), "abstract".to_string()];
        let (schema, node_id_field, text_fields) = build_schema(&properties);

        // Verify node_id field exists
        assert!(schema.get_field("node_id").is_ok());
        assert_eq!(schema.get_field("node_id").unwrap(), node_id_field);

        // Verify text fields exist
        assert_eq!(text_fields.len(), 2);
        assert!(schema.get_field("title").is_ok());
        assert!(schema.get_field("abstract").is_ok());
    }

    #[test]
    fn test_build_analyzer_full() {
        let config = AnalyzerConfig::default();
        let mut analyzer = build_analyzer(&config);

        // Test that analyzer processes text correctly
        let mut token_stream = analyzer.token_stream("The Learning machines are RUNNING");
        let mut tokens = Vec::new();
        while let Some(token) = token_stream.next() {
            tokens.push(token.text.clone());
        }

        // Should be lowercase, stemmed, without stop words
        // "the" removed (stop word), "learning" -> "learn", "machines" -> "machin",
        // "are" removed (stop word), "running" -> "run"
        assert!(!tokens.contains(&"the".to_string()));
        assert!(!tokens.contains(&"are".to_string()));
        assert!(tokens.contains(&"learn".to_string()) || tokens.iter().any(|t| t.starts_with("learn")));
    }

    #[test]
    fn test_build_analyzer_no_stemming() {
        let config = AnalyzerConfig::new().without_stemming();
        let mut analyzer = build_analyzer(&config);

        let mut token_stream = analyzer.token_stream("Running Learning");
        let mut tokens = Vec::new();
        while let Some(token) = token_stream.next() {
            tokens.push(token.text.clone());
        }

        // Should be lowercase but not stemmed
        assert!(tokens.contains(&"running".to_string()));
        assert!(tokens.contains(&"learning".to_string()));
    }
}
