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
    /// Phonetic encoding algorithm (default: None)
    pub phonetic: PhoneticAlgorithm,
}

impl Default for AnalyzerConfig {
    fn default() -> Self {
        Self {
            lowercase: true,
            remove_stopwords: true,
            stemming: true,
            language: Language::English,
            phonetic: PhoneticAlgorithm::None,
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

    /// Sets the phonetic encoding algorithm.
    pub fn with_phonetic(mut self, phonetic: PhoneticAlgorithm) -> Self {
        self.phonetic = phonetic;
        self
    }
}

/// Supported languages for text analysis.
///
/// These languages are supported by tantivy's Stemmer and StopWordFilter.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, Default)]
pub enum Language {
    #[default]
    English,
    Spanish,
    French,
    German,
    Italian,
    Portuguese,
    Dutch,
    Swedish,
    Norwegian,
    Danish,
    Finnish,
    Russian,
    Hungarian,
    Romanian,
    Turkish,
    Arabic,
    Greek,
    Tamil,
}

impl Language {
    /// Parses a language name from a string (case-insensitive).
    pub fn from_str(s: &str) -> Option<Self> {
        match s.to_lowercase().as_str() {
            "english" | "en" => Some(Language::English),
            "spanish" | "es" => Some(Language::Spanish),
            "french" | "fr" => Some(Language::French),
            "german" | "de" => Some(Language::German),
            "italian" | "it" => Some(Language::Italian),
            "portuguese" | "pt" => Some(Language::Portuguese),
            "dutch" | "nl" => Some(Language::Dutch),
            "swedish" | "sv" => Some(Language::Swedish),
            "norwegian" | "no" => Some(Language::Norwegian),
            "danish" | "da" => Some(Language::Danish),
            "finnish" | "fi" => Some(Language::Finnish),
            "russian" | "ru" => Some(Language::Russian),
            "hungarian" | "hu" => Some(Language::Hungarian),
            "romanian" | "ro" => Some(Language::Romanian),
            "turkish" | "tr" => Some(Language::Turkish),
            "arabic" | "ar" => Some(Language::Arabic),
            "greek" | "el" => Some(Language::Greek),
            "tamil" | "ta" => Some(Language::Tamil),
            _ => None,
        }
    }
}

impl std::fmt::Display for Language {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Language::English => write!(f, "english"),
            Language::Spanish => write!(f, "spanish"),
            Language::French => write!(f, "french"),
            Language::German => write!(f, "german"),
            Language::Italian => write!(f, "italian"),
            Language::Portuguese => write!(f, "portuguese"),
            Language::Dutch => write!(f, "dutch"),
            Language::Swedish => write!(f, "swedish"),
            Language::Norwegian => write!(f, "norwegian"),
            Language::Danish => write!(f, "danish"),
            Language::Finnish => write!(f, "finnish"),
            Language::Russian => write!(f, "russian"),
            Language::Hungarian => write!(f, "hungarian"),
            Language::Romanian => write!(f, "romanian"),
            Language::Turkish => write!(f, "turkish"),
            Language::Arabic => write!(f, "arabic"),
            Language::Greek => write!(f, "greek"),
            Language::Tamil => write!(f, "tamil"),
        }
    }
}

/// Phonetic encoding algorithm for sound-alike matching.
///
/// Phonetic algorithms convert words to codes based on their pronunciation,
/// allowing searches to match words that sound similar but are spelled differently.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, Default)]
pub enum PhoneticAlgorithm {
    /// No phonetic encoding (default)
    #[default]
    None,
    /// Soundex - classic phonetic algorithm, good for English names
    Soundex,
    /// Metaphone - more accurate than Soundex for English words
    Metaphone,
    /// Double Metaphone - handles more languages and edge cases
    DoubleMetaphone,
}

impl PhoneticAlgorithm {
    /// Parses a phonetic algorithm name from a string (case-insensitive).
    pub fn from_str(s: &str) -> Option<Self> {
        match s.to_lowercase().as_str() {
            "none" | "" => Some(PhoneticAlgorithm::None),
            "soundex" => Some(PhoneticAlgorithm::Soundex),
            "metaphone" => Some(PhoneticAlgorithm::Metaphone),
            "doublemetaphone" | "double_metaphone" => Some(PhoneticAlgorithm::DoubleMetaphone),
            _ => None,
        }
    }
}

impl std::fmt::Display for PhoneticAlgorithm {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            PhoneticAlgorithm::None => write!(f, "none"),
            PhoneticAlgorithm::Soundex => write!(f, "soundex"),
            PhoneticAlgorithm::Metaphone => write!(f, "metaphone"),
            PhoneticAlgorithm::DoubleMetaphone => write!(f, "doublemetaphone"),
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
        assert_eq!(config.analyzer.phonetic, PhoneticAlgorithm::None);
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

    #[test]
    fn test_analyzer_with_phonetic() {
        let analyzer = AnalyzerConfig::new()
            .with_phonetic(PhoneticAlgorithm::Soundex);

        assert_eq!(analyzer.phonetic, PhoneticAlgorithm::Soundex);
    }

    #[test]
    fn test_language_from_str() {
        assert_eq!(Language::from_str("english"), Some(Language::English));
        assert_eq!(Language::from_str("Spanish"), Some(Language::Spanish));
        assert_eq!(Language::from_str("fr"), Some(Language::French));
        assert_eq!(Language::from_str("DE"), Some(Language::German));
        assert_eq!(Language::from_str("unknown"), None);
    }

    #[test]
    fn test_phonetic_from_str() {
        assert_eq!(PhoneticAlgorithm::from_str("none"), Some(PhoneticAlgorithm::None));
        assert_eq!(PhoneticAlgorithm::from_str("soundex"), Some(PhoneticAlgorithm::Soundex));
        assert_eq!(PhoneticAlgorithm::from_str("metaphone"), Some(PhoneticAlgorithm::Metaphone));
        assert_eq!(PhoneticAlgorithm::from_str("doublemetaphone"), Some(PhoneticAlgorithm::DoubleMetaphone));
        assert_eq!(PhoneticAlgorithm::from_str("invalid"), None);
    }

    #[test]
    fn test_language_display() {
        assert_eq!(format!("{}", Language::English), "english");
        assert_eq!(format!("{}", Language::Spanish), "spanish");
        assert_eq!(format!("{}", Language::Tamil), "tamil");
    }

    #[test]
    fn test_analyzer_with_language() {
        let analyzer = AnalyzerConfig::new()
            .with_language(Language::Spanish);

        assert_eq!(analyzer.language, Language::Spanish);
    }
}
