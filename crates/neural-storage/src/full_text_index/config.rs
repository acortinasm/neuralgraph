//! Configuration types for full-text indexes.
//!
//! Provides configuration for creating and managing full-text search indexes
//! with customizable analyzer settings.

use serde::{Deserialize, Serialize};

/// Configuration for creating a full-text index.
///
/// # Example
///
/// ```ignore
/// use neural_storage::full_text_index::FullTextIndexConfig;
///
/// let config = FullTextIndexConfig::new("paper_search", "Paper")
///     .with_properties(vec!["title", "abstract"]);
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FullTextIndexConfig {
    /// Name of the index (must be unique)
    pub name: String,
    /// Node label to index
    pub label: String,
    /// Properties to include in the index
    pub properties: Vec<String>,
    /// Analyzer configuration
    pub analyzer: AnalyzerConfig,
}

impl FullTextIndexConfig {
    /// Creates a new full-text index configuration.
    ///
    /// # Arguments
    /// * `name` - Unique name for the index
    /// * `label` - Node label to index
    pub fn new(name: impl Into<String>, label: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            label: label.into(),
            properties: Vec::new(),
            analyzer: AnalyzerConfig::default(),
        }
    }

    /// Sets the properties to index.
    pub fn with_properties(mut self, properties: Vec<impl Into<String>>) -> Self {
        self.properties = properties.into_iter().map(|p| p.into()).collect();
        self
    }

    /// Sets the analyzer configuration.
    pub fn with_analyzer(mut self, analyzer: AnalyzerConfig) -> Self {
        self.analyzer = analyzer;
        self
    }
}

/// Configuration for the text analyzer.
///
/// Controls tokenization, filtering, and text processing.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnalyzerConfig {
    /// Whether to convert text to lowercase
    pub lowercase: bool,
    /// Whether to remove stop words
    pub remove_stopwords: bool,
    /// Whether to apply stemming
    pub stemming: bool,
    /// Language for stemming and stop words (default: English)
    pub language: Language,
}

impl Default for AnalyzerConfig {
    fn default() -> Self {
        Self {
            lowercase: true,
            remove_stopwords: true,
            stemming: true,
            language: Language::English,
        }
    }
}

impl AnalyzerConfig {
    /// Creates a new analyzer config with default settings.
    pub fn new() -> Self {
        Self::default()
    }

    /// Disables stop word removal.
    pub fn without_stopwords(mut self) -> Self {
        self.remove_stopwords = false;
        self
    }

    /// Disables stemming.
    pub fn without_stemming(mut self) -> Self {
        self.stemming = false;
        self
    }

    /// Sets the language for stemming and stop words.
    pub fn with_language(mut self, language: Language) -> Self {
        self.language = language;
        self
    }
}

/// Supported languages for text analysis.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, Default)]
pub enum Language {
    #[default]
    English,
    // Future: Spanish, French, German, etc.
}

impl std::fmt::Display for Language {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Language::English => write!(f, "english"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_config_builder() {
        let config = FullTextIndexConfig::new("test_index", "Person")
            .with_properties(vec!["name", "bio"]);

        assert_eq!(config.name, "test_index");
        assert_eq!(config.label, "Person");
        assert_eq!(config.properties, vec!["name", "bio"]);
        assert!(config.analyzer.lowercase);
        assert!(config.analyzer.stemming);
    }

    #[test]
    fn test_analyzer_config() {
        let analyzer = AnalyzerConfig::new()
            .without_stopwords()
            .without_stemming();

        assert!(analyzer.lowercase);
        assert!(!analyzer.remove_stopwords);
        assert!(!analyzer.stemming);
    }
}
