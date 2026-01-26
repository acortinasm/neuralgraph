//! LLM client for text generation and embeddings.
//!
//! Supports OpenAI, Ollama, and Gemini APIs for chat completions and embeddings.

use serde::Deserialize;
use thiserror::Error;

/// Errors that can occur during LLM operations.
#[derive(Debug, Error)]
pub enum LlmError {
    /// HTTP request error
    #[error("HTTP error: {0}")]
    Http(#[from] reqwest::Error),

    /// JSON parsing error
    #[error("JSON error: {0}")]
    Json(#[from] serde_json::Error),

    /// API returned an error
    #[error("API error: {0}")]
    Api(String),

    /// Missing API key
    #[error("Missing API key")]
    MissingApiKey,
}

/// Result type for LLM operations.
pub type Result<T> = std::result::Result<T, LlmError>;

/// LLM provider type.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Provider {
    /// OpenAI API
    OpenAI,
    /// Ollama (local)
    Ollama,
    /// Google Gemini
    Gemini,
}

/// LLM client for text generation and embeddings.
#[derive(Debug, Clone)]
pub struct LlmClient {
    provider: Provider,
    base_url: String,
    api_key: Option<String>,
    client: reqwest::blocking::Client,
}

impl LlmClient {
    /// Creates a client for OpenAI API.
    pub fn openai(api_key: impl Into<String>) -> Self {
        Self {
            provider: Provider::OpenAI,
            base_url: "https://api.openai.com/v1".into(),
            api_key: Some(api_key.into()),
            client: reqwest::blocking::Client::new(),
        }
    }

    /// Creates a client for Ollama (local, no API key needed).
    pub fn ollama() -> Self {
        Self::ollama_at("http://localhost:11434")
    }

    /// Creates a client for Ollama at a custom address.
    pub fn ollama_at(base_url: impl Into<String>) -> Self {
        Self {
            provider: Provider::Ollama,
            base_url: base_url.into(),
            api_key: None,
            client: reqwest::blocking::Client::new(),
        }
    }

    /// Creates a client for Google Gemini API.
    pub fn gemini(api_key: impl Into<String>) -> Self {
        Self {
            provider: Provider::Gemini,
            base_url: "https://generativelanguage.googleapis.com/v1beta".into(),
            api_key: Some(api_key.into()),
            client: reqwest::blocking::Client::new(),
        }
    }

    /// Generates a chat completion.
    ///
    /// # Arguments
    /// * `prompt` - The user prompt
    /// * `model` - Model name (e.g., "gpt-4", "llama2", "gemini-pro")
    pub fn complete(&self, prompt: &str, model: &str) -> Result<String> {
        match self.provider {
            Provider::OpenAI | Provider::Ollama => self.complete_openai_compat(prompt, model),
            Provider::Gemini => self.complete_gemini(prompt, model),
        }
    }

    /// Generates embeddings for text.
    ///
    /// # Arguments
    /// * `text` - Text to embed
    /// * `model` - Model name (e.g., "text-embedding-3-small", "nomic-embed-text")
    pub fn embed(&self, text: &str, model: &str) -> Result<Vec<f32>> {
        match self.provider {
            Provider::OpenAI | Provider::Ollama => self.embed_openai_compat(text, model),
            Provider::Gemini => self.embed_gemini(text, model),
        }
    }

    // =========================================================================
    // OpenAI-compatible API (works for OpenAI and Ollama)
    // =========================================================================

    fn complete_openai_compat(&self, prompt: &str, model: &str) -> Result<String> {
        let url = format!("{}/chat/completions", self.base_url);

        let body = serde_json::json!({
            "model": model,
            "messages": [{"role": "user", "content": prompt}]
        });

        let mut req = self.client.post(&url).json(&body);
        if let Some(key) = &self.api_key {
            req = req.bearer_auth(key);
        }

        let resp: OpenAIResponse = req.send()?.json()?;

        resp.choices
            .first()
            .and_then(|c| c.message.content.clone())
            .ok_or_else(|| LlmError::Api("No response content".into()))
    }

    fn embed_openai_compat(&self, text: &str, model: &str) -> Result<Vec<f32>> {
        let url = if self.provider == Provider::Ollama {
            format!("{}/api/embeddings", self.base_url)
        } else {
            format!("{}/embeddings", self.base_url)
        };

        let body = if self.provider == Provider::Ollama {
            serde_json::json!({"model": model, "prompt": text})
        } else {
            serde_json::json!({"model": model, "input": text})
        };

        let mut req = self.client.post(&url).json(&body);
        if let Some(key) = &self.api_key {
            req = req.bearer_auth(key);
        }

        let resp: serde_json::Value = req.send()?.json()?;

        // Handle both OpenAI and Ollama response formats
        if let Some(embedding) = resp.get("embedding") {
            // Ollama format
            Ok(serde_json::from_value(embedding.clone())?)
        } else if let Some(data) = resp.get("data").and_then(|d| d.get(0)) {
            // OpenAI format
            Ok(serde_json::from_value(data["embedding"].clone())?)
        } else {
            Err(LlmError::Api("No embedding in response".into()))
        }
    }

    // =========================================================================
    // Gemini API
    // =========================================================================

    fn complete_gemini(&self, prompt: &str, model: &str) -> Result<String> {
        let api_key = self.api_key.as_ref().ok_or(LlmError::MissingApiKey)?;
        let url = format!(
            "{}/models/{}:generateContent?key={}",
            self.base_url, model, api_key
        );

        let body = serde_json::json!({
            "contents": [{"parts": [{"text": prompt}]}]
        });

        let resp: GeminiResponse = self.client.post(&url).json(&body).send()?.json()?;

        resp.candidates
            .and_then(|c| c.first().cloned())
            .and_then(|c| c.content.parts.first().cloned())
            .map(|p| p.text)
            .ok_or_else(|| LlmError::Api("No response content".into()))
    }

    fn embed_gemini(&self, text: &str, model: &str) -> Result<Vec<f32>> {
        let api_key = self.api_key.as_ref().ok_or(LlmError::MissingApiKey)?;
        let url = format!(
            "{}/models/{}:embedContent?key={}",
            self.base_url, model, api_key
        );

        let body = serde_json::json!({
            "content": {"parts": [{"text": text}]}
        });

        let resp: GeminiEmbeddingResponse = self.client.post(&url).json(&body).send()?.json()?;

        resp.embedding
            .map(|e| e.values)
            .ok_or_else(|| LlmError::Api("No embedding in response".into()))
    }
}

// =============================================================================
// Response Types
// =============================================================================

#[derive(Debug, Deserialize)]
struct OpenAIResponse {
    choices: Vec<OpenAIChoice>,
}

#[derive(Debug, Deserialize)]
struct OpenAIChoice {
    message: OpenAIMessage,
}

#[derive(Debug, Deserialize)]
struct OpenAIMessage {
    content: Option<String>,
}

#[derive(Debug, Deserialize)]
struct GeminiResponse {
    candidates: Option<Vec<GeminiCandidate>>,
}

#[derive(Debug, Clone, Deserialize)]
struct GeminiCandidate {
    content: GeminiContent,
}

#[derive(Debug, Clone, Deserialize)]
struct GeminiContent {
    parts: Vec<GeminiPart>,
}

#[derive(Debug, Clone, Deserialize)]
struct GeminiPart {
    text: String,
}

#[derive(Debug, Deserialize)]
struct GeminiEmbeddingResponse {
    embedding: Option<GeminiEmbedding>,
}

#[derive(Debug, Deserialize)]
struct GeminiEmbedding {
    values: Vec<f32>,
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_client_creation() {
        let openai = LlmClient::openai("test-key");
        assert_eq!(openai.provider, Provider::OpenAI);
        assert!(openai.api_key.is_some());

        let ollama = LlmClient::ollama();
        assert_eq!(ollama.provider, Provider::Ollama);
        assert!(ollama.api_key.is_none());

        let gemini = LlmClient::gemini("test-key");
        assert_eq!(gemini.provider, Provider::Gemini);
        assert!(gemini.api_key.is_some());
    }

    #[test]
    fn test_ollama_custom_url() {
        let client = LlmClient::ollama_at("http://192.168.1.100:11434");
        assert_eq!(client.base_url, "http://192.168.1.100:11434");
    }
}
