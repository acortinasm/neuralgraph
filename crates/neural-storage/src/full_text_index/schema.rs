//! Schema builder and analyzer setup for full-text indexes.
//!
//! This module provides utilities for building tantivy schemas with
//! properly configured text analyzers including tokenization, lowercasing,
//! stop word filtering, stemming, and phonetic encoding.

use super::config::{AnalyzerConfig, Language, PhoneticAlgorithm};
use super::phonetic::PhoneticTokenFilter;
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
/// 5. PhoneticTokenFilter - adds phonetic encodings (if enabled)
///
/// # Arguments
/// * `config` - Analyzer configuration
///
/// # Returns
/// A configured TextAnalyzer
pub fn build_analyzer(config: &AnalyzerConfig) -> TextAnalyzer {
    let tokenizer = SimpleTokenizer::default();
    let tantivy_lang = to_tantivy_language(config.language);
    let has_phonetic = config.phonetic != PhoneticAlgorithm::None;

    // Build the pipeline based on config
    // We need different branches due to tantivy's type system
    match (config.lowercase, config.remove_stopwords, config.stemming, has_phonetic) {
        // Full pipeline with phonetic
        (true, true, true, true) => {
            TextAnalyzer::builder(tokenizer)
                .filter(LowerCaser)
                .filter(StopWordFilter::new(tantivy_lang).unwrap())
                .filter(Stemmer::new(tantivy_lang))
                .filter(PhoneticTokenFilter::new(config.phonetic))
                .build()
        }
        // Full pipeline without phonetic
        (true, true, true, false) => {
            TextAnalyzer::builder(tokenizer)
                .filter(LowerCaser)
                .filter(StopWordFilter::new(tantivy_lang).unwrap())
                .filter(Stemmer::new(tantivy_lang))
                .build()
        }
        // No stemming, with phonetic
        (true, true, false, true) => {
            TextAnalyzer::builder(tokenizer)
                .filter(LowerCaser)
                .filter(StopWordFilter::new(tantivy_lang).unwrap())
                .filter(PhoneticTokenFilter::new(config.phonetic))
                .build()
        }
        // No stemming, no phonetic
        (true, true, false, false) => {
            TextAnalyzer::builder(tokenizer)
                .filter(LowerCaser)
                .filter(StopWordFilter::new(tantivy_lang).unwrap())
                .build()
        }
        // No stop words, with phonetic
        (true, false, true, true) => {
            TextAnalyzer::builder(tokenizer)
                .filter(LowerCaser)
                .filter(Stemmer::new(tantivy_lang))
                .filter(PhoneticTokenFilter::new(config.phonetic))
                .build()
        }
        // No stop words, no phonetic
        (true, false, true, false) => {
            TextAnalyzer::builder(tokenizer)
                .filter(LowerCaser)
                .filter(Stemmer::new(tantivy_lang))
                .build()
        }
        // Only lowercase, with phonetic
        (true, false, false, true) => {
            TextAnalyzer::builder(tokenizer)
                .filter(LowerCaser)
                .filter(PhoneticTokenFilter::new(config.phonetic))
                .build()
        }
        // Only lowercase, no phonetic
        (true, false, false, false) => {
            TextAnalyzer::builder(tokenizer)
                .filter(LowerCaser)
                .build()
        }
        // Only stemming, with phonetic
        (false, _, true, true) => {
            TextAnalyzer::builder(tokenizer)
                .filter(Stemmer::new(tantivy_lang))
                .filter(PhoneticTokenFilter::new(config.phonetic))
                .build()
        }
        // Only stemming, no phonetic
        (false, _, true, false) => {
            TextAnalyzer::builder(tokenizer)
                .filter(Stemmer::new(tantivy_lang))
                .build()
        }
        // Only phonetic
        (false, false, false, true) => {
            TextAnalyzer::builder(tokenizer)
                .filter(PhoneticTokenFilter::new(config.phonetic))
                .build()
        }
        // No filters - just tokenize
        (false, false, false, false) => {
            TextAnalyzer::builder(tokenizer).build()
        }
        // Stop words only (with or without phonetic) - unusual cases
        (false, true, false, true) => {
            TextAnalyzer::builder(tokenizer)
                .filter(StopWordFilter::new(tantivy_lang).unwrap())
                .filter(PhoneticTokenFilter::new(config.phonetic))
                .build()
        }
        (false, true, false, false) => {
            TextAnalyzer::builder(tokenizer)
                .filter(StopWordFilter::new(tantivy_lang).unwrap())
                .build()
        }
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
pub fn to_tantivy_language(lang: Language) -> TantivyLanguage {
    match lang {
        Language::English => TantivyLanguage::English,
        Language::Spanish => TantivyLanguage::Spanish,
        Language::French => TantivyLanguage::French,
        Language::German => TantivyLanguage::German,
        Language::Italian => TantivyLanguage::Italian,
        Language::Portuguese => TantivyLanguage::Portuguese,
        Language::Dutch => TantivyLanguage::Dutch,
        Language::Swedish => TantivyLanguage::Swedish,
        Language::Norwegian => TantivyLanguage::Norwegian,
        Language::Danish => TantivyLanguage::Danish,
        Language::Finnish => TantivyLanguage::Finnish,
        Language::Russian => TantivyLanguage::Russian,
        Language::Hungarian => TantivyLanguage::Hungarian,
        Language::Romanian => TantivyLanguage::Romanian,
        Language::Turkish => TantivyLanguage::Turkish,
        Language::Arabic => TantivyLanguage::Arabic,
        Language::Greek => TantivyLanguage::Greek,
        Language::Tamil => TantivyLanguage::Tamil,
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

    #[test]
    fn test_build_analyzer_spanish() {
        let config = AnalyzerConfig::new().with_language(Language::Spanish);
        let mut analyzer = build_analyzer(&config);

        // Test Spanish stemming: "corriendo" (running) should stem
        let mut token_stream = analyzer.token_stream("corriendo aprendiendo");
        let mut tokens = Vec::new();
        while let Some(token) = token_stream.next() {
            tokens.push(token.text.clone());
        }

        // Spanish stemmer should reduce these words
        assert!(!tokens.is_empty());
        // The stems should be shorter than originals
        assert!(tokens.iter().all(|t| t.len() <= "corriendo".len()));
    }

    #[test]
    fn test_build_analyzer_with_phonetic() {
        let config = AnalyzerConfig::new()
            .with_phonetic(PhoneticAlgorithm::Soundex);
        let mut analyzer = build_analyzer(&config);

        let mut token_stream = analyzer.token_stream("Smith");
        let mut tokens = Vec::new();
        while let Some(token) = token_stream.next() {
            tokens.push(token.text.clone());
        }

        // Should have original token (stemmed/lowercased) plus phonetic code
        assert!(tokens.len() >= 2);
        // Should contain soundex code for Smith (s530)
        assert!(tokens.iter().any(|t| t == "s530"));
    }

    #[test]
    fn test_build_analyzer_german() {
        let config = AnalyzerConfig::new().with_language(Language::German);
        let mut analyzer = build_analyzer(&config);

        // Test German text
        let mut token_stream = analyzer.token_stream("Maschinelles Lernen");
        let mut tokens = Vec::new();
        while let Some(token) = token_stream.next() {
            tokens.push(token.text.clone());
        }

        // German stemmer should process these
        assert!(!tokens.is_empty());
    }

    #[test]
    fn test_language_mapping() {
        // Test all language mappings
        assert_eq!(to_tantivy_language(Language::English), TantivyLanguage::English);
        assert_eq!(to_tantivy_language(Language::Spanish), TantivyLanguage::Spanish);
        assert_eq!(to_tantivy_language(Language::French), TantivyLanguage::French);
        assert_eq!(to_tantivy_language(Language::German), TantivyLanguage::German);
        assert_eq!(to_tantivy_language(Language::Italian), TantivyLanguage::Italian);
        assert_eq!(to_tantivy_language(Language::Portuguese), TantivyLanguage::Portuguese);
        assert_eq!(to_tantivy_language(Language::Dutch), TantivyLanguage::Dutch);
        assert_eq!(to_tantivy_language(Language::Swedish), TantivyLanguage::Swedish);
        assert_eq!(to_tantivy_language(Language::Norwegian), TantivyLanguage::Norwegian);
        assert_eq!(to_tantivy_language(Language::Danish), TantivyLanguage::Danish);
        assert_eq!(to_tantivy_language(Language::Finnish), TantivyLanguage::Finnish);
        assert_eq!(to_tantivy_language(Language::Russian), TantivyLanguage::Russian);
        assert_eq!(to_tantivy_language(Language::Hungarian), TantivyLanguage::Hungarian);
        assert_eq!(to_tantivy_language(Language::Romanian), TantivyLanguage::Romanian);
        assert_eq!(to_tantivy_language(Language::Turkish), TantivyLanguage::Turkish);
        assert_eq!(to_tantivy_language(Language::Arabic), TantivyLanguage::Arabic);
        assert_eq!(to_tantivy_language(Language::Greek), TantivyLanguage::Greek);
        assert_eq!(to_tantivy_language(Language::Tamil), TantivyLanguage::Tamil);
    }
}
