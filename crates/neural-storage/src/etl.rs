//! ETL Pipeline for document to graph ingestion.
//!
//! Provides end-to-end pipeline: PDF → LLM extraction → Graph insertion.

use crate::GraphStoreBuilder;
use crate::llm::{LlmClient, LlmError};
use crate::pdf::{self, PdfError};
use std::collections::HashMap;
use std::path::Path;
use thiserror::Error;

/// Errors that can occur during ETL operations.
#[derive(Debug, Error)]
pub enum EtlError {
    /// PDF processing error
    #[error("PDF error: {0}")]
    Pdf(#[from] PdfError),

    /// LLM error
    #[error("LLM error: {0}")]
    Llm(#[from] LlmError),

    /// JSON parsing error
    #[error("Parse error: {0}")]
    Parse(String),
}

/// Result type for ETL operations.
pub type Result<T> = std::result::Result<T, EtlError>;

/// An extracted entity.
#[derive(Debug, Clone)]
pub struct Entity {
    /// Entity name/identifier
    pub name: String,
    /// Entity label (type)
    pub label: String,
    /// Additional properties
    pub properties: HashMap<String, String>,
}

/// An extracted relation between entities.
#[derive(Debug, Clone)]
pub struct Relation {
    /// Source entity name
    pub from: String,
    /// Target entity name
    pub to: String,
    /// Relation type
    pub relation_type: String,
}

/// Result of entity/relation extraction.
#[derive(Debug, Clone, Default)]
pub struct ExtractionResult {
    /// Extracted entities
    pub entities: Vec<Entity>,
    /// Extracted relations
    pub relations: Vec<Relation>,
}

/// ETL pipeline for document to graph ingestion.
pub struct EtlPipeline {
    llm: LlmClient,
    model: String,
}

impl EtlPipeline {
    /// Creates a new ETL pipeline.
    pub fn new(llm: LlmClient, model: impl Into<String>) -> Self {
        Self {
            llm,
            model: model.into(),
        }
    }

    /// Processes a PDF file and extracts entities/relations.
    pub fn process_pdf<P: AsRef<Path>>(&self, path: P) -> Result<ExtractionResult> {
        let doc = pdf::load_pdf(path)?;
        self.process_text(&doc.full_text)
    }

    /// Processes text and extracts entities/relations using LLM.
    pub fn process_text(&self, text: &str) -> Result<ExtractionResult> {
        let prompt = format!(
            r#"You are extracting a knowledge graph from an academic paper.

Extract entities and relations. Use these entity labels:
- PAPER: The paper itself
- AUTHOR: Paper authors
- ORG: Organizations/institutions
- CONCEPT: Key concepts, methods, algorithms
- SUBJECT: Research topics/fields (e.g., "Machine Learning", "NLP")
- DATASET: Datasets mentioned
- METRIC: Evaluation metrics

Use these relation types:
- AUTHORED: Author wrote this paper
- AFFILIATED_WITH: Author works at organization
- ABOUT: Paper is about a subject
- INTRODUCES: Paper introduces a new concept/method
- USES: Paper uses a concept/method/dataset
- CITES: Paper cites another work
- EVALUATES_ON: Paper evaluates on dataset/metric
- IMPROVES: New method improves upon existing method

Return JSON in this exact format:
{{
  "entities": [{{"name": "...", "label": "...", "properties": {{}}}}],
  "relations": [{{"from": "...", "to": "...", "relation_type": "..."}}]
}}

Text:
{}

JSON:"#,
            text
        );

        let response = self.llm.complete(&prompt, &self.model)?;
        self.parse_extraction(&response)
    }

    /// Parses LLM response into ExtractionResult.
    fn parse_extraction(&self, response: &str) -> Result<ExtractionResult> {
        // Find JSON in response
        let json_start = response.find('{').unwrap_or(0);
        let json_end = response.rfind('}').map(|i| i + 1).unwrap_or(response.len());
        let json_str = &response[json_start..json_end];

        let parsed: serde_json::Value = serde_json::from_str(json_str)
            .map_err(|e| EtlError::Parse(format!("Invalid JSON: {}", e)))?;

        let mut result = ExtractionResult::default();

        // Parse entities
        if let Some(entities) = parsed.get("entities").and_then(|e| e.as_array()) {
            for entity in entities {
                if let (Some(name), Some(label)) = (
                    entity.get("name").and_then(|n| n.as_str()),
                    entity.get("label").and_then(|l| l.as_str()),
                ) {
                    let mut properties = HashMap::new();
                    if let Some(props) = entity.get("properties").and_then(|p| p.as_object()) {
                        for (k, v) in props {
                            if let Some(val) = v.as_str() {
                                properties.insert(k.clone(), val.to_string());
                            }
                        }
                    }
                    result.entities.push(Entity {
                        name: name.to_string(),
                        label: label.to_string(),
                        properties,
                    });
                }
            }
        }

        // Parse relations
        if let Some(relations) = parsed.get("relations").and_then(|r| r.as_array()) {
            for relation in relations {
                if let (Some(from), Some(to), Some(rel_type)) = (
                    relation.get("from").and_then(|f| f.as_str()),
                    relation.get("to").and_then(|t| t.as_str()),
                    relation.get("relation_type").and_then(|r| r.as_str()),
                ) {
                    result.relations.push(Relation {
                        from: from.to_string(),
                        to: to.to_string(),
                        relation_type: rel_type.to_string(),
                    });
                }
            }
        }

        Ok(result)
    }

    /// Inserts extracted data into a graph builder.
    /// Returns the modified builder with nodes and edges added.
    pub fn insert_into_graph(
        &self,
        result: &ExtractionResult,
        mut builder: GraphStoreBuilder,
    ) -> GraphStoreBuilder {
        use neural_core::{Label, PropertyValue};

        // Create entity name -> node ID mapping
        let mut entity_ids: HashMap<String, u64> = HashMap::new();

        // Add nodes for each entity
        for (idx, entity) in result.entities.iter().enumerate() {
            let node_id = idx as u64;
            entity_ids.insert(entity.name.clone(), node_id);

            // Add node with label and properties (include name as a property)
            let mut props: Vec<(String, PropertyValue)> =
                vec![("name".to_string(), PropertyValue::String(entity.name.clone()))];
            props.extend(
                entity
                    .properties
                    .iter()
                    .map(|(k, v)| (k.clone(), PropertyValue::String(v.clone()))),
            );

            builder = builder.add_labeled_node(node_id, &entity.label, props);
        }

        // Add edges for each relation
        for relation in &result.relations {
            if let (Some(&from_id), Some(&to_id)) =
                (entity_ids.get(&relation.from), entity_ids.get(&relation.to))
            {
                builder = builder.add_labeled_edge(
                    from_id,
                    to_id,
                    Label::from(relation.relation_type.as_str()),
                );
            }
        }

        builder
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_extraction() {
        let pipeline = EtlPipeline::new(LlmClient::ollama(), "test");

        let json = r#"{
            "entities": [
                {"name": "Alice", "label": "Person", "properties": {"age": "30"}},
                {"name": "Bob", "label": "Person", "properties": {}}
            ],
            "relations": [
                {"from": "Alice", "to": "Bob", "relation_type": "KNOWS"}
            ]
        }"#;

        let result = pipeline.parse_extraction(json).unwrap();

        assert_eq!(result.entities.len(), 2);
        assert_eq!(result.entities[0].name, "Alice");
        assert_eq!(result.entities[0].label, "Person");
        assert_eq!(
            result.entities[0].properties.get("age"),
            Some(&"30".to_string())
        );

        assert_eq!(result.relations.len(), 1);
        assert_eq!(result.relations[0].from, "Alice");
        assert_eq!(result.relations[0].to, "Bob");
        assert_eq!(result.relations[0].relation_type, "KNOWS");
    }

    #[test]
    fn test_extraction_result_default() {
        let result = ExtractionResult::default();
        assert!(result.entities.is_empty());
        assert!(result.relations.is_empty());
    }
}
