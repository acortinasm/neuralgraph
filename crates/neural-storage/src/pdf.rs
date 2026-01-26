//! PDF document loading and text extraction.
//!
//! This module provides functionality to load PDF documents and extract
//! their text content for further processing (e.g., for graph ingestion).

use std::path::Path;
use thiserror::Error;

/// Errors that can occur during PDF processing.
#[derive(Debug, Error)]
pub enum PdfError {
    /// IO error reading the file
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    /// PDF parsing error
    #[error("PDF parsing error: {0}")]
    Parse(String),
}

/// Result type for PDF operations.
pub type Result<T> = std::result::Result<T, PdfError>;

/// Metadata extracted from a PDF document.
#[derive(Debug, Clone, Default)]
pub struct PdfMetadata {
    /// Document title (if available)
    pub title: Option<String>,
    /// Document author (if available)
    pub author: Option<String>,
    /// Number of pages
    pub page_count: usize,
}

/// A loaded PDF document with extracted text.
#[derive(Debug, Clone)]
pub struct PdfDocument {
    /// Text content per page
    pub pages: Vec<String>,
    /// Document metadata
    pub metadata: PdfMetadata,
    /// Full text (all pages concatenated)
    pub full_text: String,
}

impl PdfDocument {
    /// Returns the number of pages.
    pub fn page_count(&self) -> usize {
        self.pages.len()
    }

    /// Returns true if the document has no text content.
    pub fn is_empty(&self) -> bool {
        self.full_text.is_empty()
    }

    /// Returns the text of a specific page (0-indexed).
    pub fn page(&self, index: usize) -> Option<&str> {
        self.pages.get(index).map(|s| s.as_str())
    }
}

/// Loads a PDF from a file path and extracts its text content.
///
/// # Arguments
/// * `path` - Path to the PDF file
///
/// # Returns
/// A `PdfDocument` containing the extracted text and metadata.
///
/// # Example
/// ```ignore
/// use neural_storage::pdf::load_pdf;
///
/// let doc = load_pdf("document.pdf")?;
/// println!("Pages: {}", doc.page_count());
/// println!("Text: {}", doc.full_text);
/// ```
pub fn load_pdf<P: AsRef<Path>>(path: P) -> Result<PdfDocument> {
    let bytes = std::fs::read(path)?;
    load_pdf_bytes(&bytes)
}

/// Loads a PDF from bytes and extracts its text content.
///
/// # Arguments
/// * `bytes` - Raw PDF file bytes
///
/// # Returns
/// A `PdfDocument` containing the extracted text and metadata.
pub fn load_pdf_bytes(bytes: &[u8]) -> Result<PdfDocument> {
    // Extract text using pdf-extract
    let text =
        pdf_extract::extract_text_from_mem(bytes).map_err(|e| PdfError::Parse(e.to_string()))?;

    // Split into pages (pdf-extract includes form feed characters between pages)
    let pages: Vec<String> = text
        .split('\x0C') // Form feed character
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
        .collect();

    let page_count = pages.len().max(1);
    let full_text = pages.join("\n\n");

    Ok(PdfDocument {
        pages,
        metadata: PdfMetadata {
            title: None, // pdf-extract doesn't expose metadata
            author: None,
            page_count,
        },
        full_text,
    })
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pdf_document_empty() {
        let doc = PdfDocument {
            pages: vec![],
            metadata: PdfMetadata::default(),
            full_text: String::new(),
        };
        assert!(doc.is_empty());
        assert_eq!(doc.page_count(), 0);
    }

    #[test]
    fn test_pdf_document_with_content() {
        let doc = PdfDocument {
            pages: vec!["Page 1 content".into(), "Page 2 content".into()],
            metadata: PdfMetadata {
                title: Some("Test".into()),
                author: None,
                page_count: 2,
            },
            full_text: "Page 1 content\n\nPage 2 content".into(),
        };

        assert!(!doc.is_empty());
        assert_eq!(doc.page_count(), 2);
        assert_eq!(doc.page(0), Some("Page 1 content"));
        assert_eq!(doc.page(1), Some("Page 2 content"));
        assert_eq!(doc.page(2), None);
    }

    #[test]
    fn test_load_invalid_pdf() {
        let result = load_pdf_bytes(b"not a valid pdf");
        assert!(result.is_err());
    }
}
