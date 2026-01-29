//! Phonetic token filter for full-text search.
//!
//! This module provides a custom tantivy `TokenFilter` that adds phonetic
//! encodings alongside original tokens, enabling sound-alike matching.
//!
//! # Supported Algorithms
//!
//! - **Soundex**: Classic algorithm, works well for English names
//! - **Metaphone**: More accurate than Soundex for English words
//! - **Double Metaphone**: Handles more edge cases and non-English words

use super::config::PhoneticAlgorithm;
use rphonetic::{DoubleMetaphone, Encoder, Metaphone, Soundex};
use tantivy::tokenizer::{Token, TokenFilter, TokenStream, Tokenizer};

/// A token filter that adds phonetic encodings to the token stream.
///
/// For each input token, this filter emits:
/// 1. The original token (unchanged)
/// 2. The phonetic encoding(s) of the token
///
/// This allows queries to match both exact terms and sound-alike terms.
#[derive(Clone)]
pub struct PhoneticTokenFilter {
    algorithm: PhoneticAlgorithm,
}

impl PhoneticTokenFilter {
    /// Creates a new phonetic token filter with the specified algorithm.
    pub fn new(algorithm: PhoneticAlgorithm) -> Self {
        Self { algorithm }
    }

    /// Encodes a word using the configured phonetic algorithm.
    fn encode(&self, word: &str) -> Vec<String> {
        if word.is_empty() {
            return vec![];
        }

        match self.algorithm {
            PhoneticAlgorithm::None => vec![],
            PhoneticAlgorithm::Soundex => {
                let soundex = Soundex::default();
                let code = soundex.encode(word);
                if code.is_empty() {
                    vec![]
                } else {
                    vec![code.to_lowercase()]
                }
            }
            PhoneticAlgorithm::Metaphone => {
                let metaphone = Metaphone::default();
                let code = metaphone.encode(word);
                if code.is_empty() {
                    vec![]
                } else {
                    vec![code.to_lowercase()]
                }
            }
            PhoneticAlgorithm::DoubleMetaphone => {
                let dm = DoubleMetaphone::default();
                let dm_result = dm.double_metaphone(word);
                let primary = dm_result.primary();
                let alternate = dm_result.alternate();
                let mut result = vec![];
                if !primary.is_empty() {
                    result.push(primary.to_lowercase());
                }
                if alternate != primary && !alternate.is_empty() {
                    result.push(alternate.to_lowercase());
                }
                result
            }
        }
    }
}

impl TokenFilter for PhoneticTokenFilter {
    type Tokenizer<T: Tokenizer> = PhoneticTokenFilterWrapper<T>;

    fn transform<T: Tokenizer>(self, tokenizer: T) -> Self::Tokenizer<T> {
        PhoneticTokenFilterWrapper {
            inner: tokenizer,
            filter: self,
        }
    }
}

/// Wrapper that applies phonetic filtering to a tokenizer.
#[derive(Clone)]
pub struct PhoneticTokenFilterWrapper<T> {
    inner: T,
    filter: PhoneticTokenFilter,
}

impl<T: Tokenizer> Tokenizer for PhoneticTokenFilterWrapper<T> {
    type TokenStream<'a> = PhoneticTokenStream<T::TokenStream<'a>>;

    fn token_stream<'a>(&'a mut self, text: &'a str) -> Self::TokenStream<'a> {
        PhoneticTokenStream {
            inner: self.inner.token_stream(text),
            filter: self.filter.clone(),
            pending_phonetics: vec![],
            current_position: 0,
        }
    }
}

/// Token stream that emits both original tokens and their phonetic encodings.
pub struct PhoneticTokenStream<T> {
    inner: T,
    filter: PhoneticTokenFilter,
    /// Queue of pending phonetic tokens to emit
    pending_phonetics: Vec<Token>,
    /// Current token position for phonetic tokens
    current_position: usize,
}

impl<T: TokenStream> TokenStream for PhoneticTokenStream<T> {
    fn advance(&mut self) -> bool {
        // First, emit any pending phonetic tokens
        if let Some(token) = self.pending_phonetics.pop() {
            // Copy the phonetic token to our output
            *self.inner.token_mut() = token;
            return true;
        }

        // Get next token from inner stream
        if !self.inner.advance() {
            return false;
        }

        // Get phonetic encodings for this token
        let original_token = self.inner.token().clone();
        self.current_position = original_token.position;

        let phonetics = self.filter.encode(&original_token.text);

        // Queue phonetic tokens (they'll be emitted on subsequent advance() calls)
        // We emit them with the same position as the original token
        for code in phonetics.into_iter().rev() {
            let mut phonetic_token = Token::default();
            phonetic_token.text = code;
            phonetic_token.position = self.current_position;
            phonetic_token.offset_from = original_token.offset_from;
            phonetic_token.offset_to = original_token.offset_to;
            // Set position_length to 0 to indicate this is a synonym at the same position
            phonetic_token.position_length = 0;
            self.pending_phonetics.push(phonetic_token);
        }

        true
    }

    fn token(&self) -> &Token {
        self.inner.token()
    }

    fn token_mut(&mut self) -> &mut Token {
        self.inner.token_mut()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tantivy::tokenizer::{SimpleTokenizer, TextAnalyzer};

    fn tokenize_with_phonetic(text: &str, algorithm: PhoneticAlgorithm) -> Vec<String> {
        let mut analyzer = TextAnalyzer::builder(SimpleTokenizer::default())
            .filter(PhoneticTokenFilter::new(algorithm))
            .build();

        let mut stream = analyzer.token_stream(text);
        let mut tokens = vec![];
        while let Some(token) = stream.next() {
            tokens.push(token.text.clone());
        }
        tokens
    }

    #[test]
    fn test_soundex_encoding() {
        let tokens = tokenize_with_phonetic("Smith", PhoneticAlgorithm::Soundex);
        assert!(tokens.contains(&"Smith".to_string()));
        assert!(tokens.contains(&"s530".to_string())); // Soundex for Smith

        // Test that similar-sounding names produce same code
        let tokens_smyth = tokenize_with_phonetic("Smyth", PhoneticAlgorithm::Soundex);
        assert!(tokens_smyth.contains(&"s530".to_string())); // Same as Smith
    }

    #[test]
    fn test_metaphone_encoding() {
        let tokens = tokenize_with_phonetic("machine", PhoneticAlgorithm::Metaphone);
        assert!(tokens.contains(&"machine".to_string()));
        // Metaphone produces phonetic code
        assert!(tokens.len() > 1);
    }

    #[test]
    fn test_double_metaphone_encoding() {
        let tokens = tokenize_with_phonetic("algorithm", PhoneticAlgorithm::DoubleMetaphone);
        assert!(tokens.contains(&"algorithm".to_string()));
        // Double Metaphone can produce alternate codes
        assert!(tokens.len() >= 2);
    }

    #[test]
    fn test_no_phonetic() {
        let tokens = tokenize_with_phonetic("hello world", PhoneticAlgorithm::None);
        // With no phonetic, we only get original tokens
        assert_eq!(tokens.len(), 2);
        assert!(tokens.contains(&"hello".to_string()));
        assert!(tokens.contains(&"world".to_string()));
    }

    #[test]
    fn test_multiple_words() {
        let tokens = tokenize_with_phonetic("John Smith", PhoneticAlgorithm::Soundex);
        // Should have original tokens + phonetic codes
        assert!(tokens.contains(&"John".to_string()));
        assert!(tokens.contains(&"Smith".to_string()));
        assert!(tokens.len() > 2); // Should have phonetic codes too
    }

    #[test]
    fn test_soundex_names() {
        // Test that sound-alike names produce matching codes
        let filter = PhoneticTokenFilter::new(PhoneticAlgorithm::Soundex);

        let smith_codes = filter.encode("Smith");
        let smyth_codes = filter.encode("Smyth");
        assert_eq!(smith_codes, smyth_codes);

        let robert_codes = filter.encode("Robert");
        let rupert_codes = filter.encode("Rupert");
        assert_eq!(robert_codes, rupert_codes);
    }
}
