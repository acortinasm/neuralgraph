//! Constraint system for NeuralGraphDB.
//!
//! This module provides data integrity constraints including:
//! - Unique constraints on property values
//!
//! # Example
//!
//! ```ignore
//! use neural_storage::constraints::ConstraintManager;
//!
//! let mut manager = ConstraintManager::new();
//!
//! // Create a unique constraint on email for Person nodes
//! manager.create_unique("unique_email", "email", Some("Person"))?;
//!
//! // Validate before insert
//! manager.validate_insert(node_id, Some("Person"), &props)?;
//!
//! // Update index after successful insert
//! manager.on_insert(node_id, Some("Person"), &props);
//! ```

use neural_core::{NodeId, PropertyValue};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use thiserror::Error;

/// Constraint errors.
#[derive(Debug, Error, Clone)]
pub enum ConstraintError {
    #[error("Constraint '{name}' already exists")]
    AlreadyExists { name: String },

    #[error("Constraint '{name}' not found")]
    NotFound { name: String },

    #[error("Unique constraint '{constraint}' violated: property '{property}' with value '{value}' already exists on node {existing_node}")]
    UniqueViolation {
        constraint: String,
        property: String,
        value: String,
        existing_node: NodeId,
    },
}

/// Types of constraints supported.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ConstraintType {
    /// Unique constraint on a property value.
    /// Optionally scoped to a specific label.
    Unique {
        property: String,
        label: Option<String>,
    },
}

/// A named constraint definition.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Constraint {
    /// Unique name for this constraint
    pub name: String,
    /// The type and configuration of the constraint
    pub constraint_type: ConstraintType,
}

/// Manages constraints and their enforcement.
#[derive(Debug, Default, Serialize, Deserialize)]
pub struct ConstraintManager {
    /// Named constraints
    constraints: HashMap<String, Constraint>,
    /// Unique index: (property_name, value_key) -> node_id
    /// Only populated for properties with unique constraints
    unique_index: HashMap<(String, String), NodeId>,
}

impl ConstraintManager {
    /// Creates a new empty constraint manager.
    pub fn new() -> Self {
        Self::default()
    }

    /// Creates a unique constraint on a property.
    ///
    /// # Arguments
    /// * `name` - Unique name for the constraint
    /// * `property` - Property name to constrain
    /// * `label` - Optional label to scope the constraint to
    ///
    /// # Returns
    /// `Ok(())` on success, `Err` if constraint with same name exists
    pub fn create_unique(
        &mut self,
        name: &str,
        property: &str,
        label: Option<&str>,
    ) -> Result<(), ConstraintError> {
        if self.constraints.contains_key(name) {
            return Err(ConstraintError::AlreadyExists {
                name: name.to_string(),
            });
        }

        let constraint = Constraint {
            name: name.to_string(),
            constraint_type: ConstraintType::Unique {
                property: property.to_string(),
                label: label.map(String::from),
            },
        };

        self.constraints.insert(name.to_string(), constraint);
        Ok(())
    }

    /// Drops a constraint by name.
    pub fn drop_constraint(&mut self, name: &str) -> Result<(), ConstraintError> {
        if self.constraints.remove(name).is_none() {
            return Err(ConstraintError::NotFound {
                name: name.to_string(),
            });
        }

        // Note: We don't clean up the unique_index here since entries
        // are still valid even without the constraint
        Ok(())
    }

    /// Lists all constraints.
    pub fn list_constraints(&self) -> Vec<&Constraint> {
        self.constraints.values().collect()
    }

    /// Gets a constraint by name.
    pub fn get_constraint(&self, name: &str) -> Option<&Constraint> {
        self.constraints.get(name)
    }

    /// Validates that an insert operation doesn't violate any constraints.
    ///
    /// # Arguments
    /// * `node_id` - The node being inserted
    /// * `label` - Optional label for the node
    /// * `props` - Properties being set on the node
    ///
    /// # Returns
    /// `Ok(())` if valid, `Err` with violation details if not
    pub fn validate_insert(
        &self,
        node_id: NodeId,
        label: Option<&str>,
        props: &[(String, PropertyValue)],
    ) -> Result<(), ConstraintError> {
        for constraint in self.constraints.values() {
            match &constraint.constraint_type {
                ConstraintType::Unique {
                    property,
                    label: constraint_label,
                } => {
                    // Check if constraint applies to this label
                    if let Some(cl) = constraint_label {
                        if label != Some(cl.as_str()) {
                            continue;
                        }
                    }

                    // Check if the property is being set
                    for (key, value) in props {
                        if key == property {
                            let value_key = value_to_key(value);
                            let index_key = (property.clone(), value_key.clone());

                            if let Some(&existing_node) = self.unique_index.get(&index_key) {
                                if existing_node != node_id {
                                    return Err(ConstraintError::UniqueViolation {
                                        constraint: constraint.name.clone(),
                                        property: property.clone(),
                                        value: value_key,
                                        existing_node,
                                    });
                                }
                            }
                        }
                    }
                }
            }
        }

        Ok(())
    }

    /// Updates indexes after a successful insert.
    ///
    /// Call this after the node has been successfully created.
    pub fn on_insert(
        &mut self,
        node_id: NodeId,
        label: Option<&str>,
        props: &[(String, PropertyValue)],
    ) {
        for constraint in self.constraints.values() {
            match &constraint.constraint_type {
                ConstraintType::Unique {
                    property,
                    label: constraint_label,
                } => {
                    // Check if constraint applies
                    if let Some(cl) = constraint_label {
                        if label != Some(cl.as_str()) {
                            continue;
                        }
                    }

                    // Index the property value
                    for (key, value) in props {
                        if key == property {
                            let value_key = value_to_key(value);
                            let index_key = (property.clone(), value_key);
                            self.unique_index.insert(index_key, node_id);
                        }
                    }
                }
            }
        }
    }

    /// Updates indexes after a property update.
    pub fn on_update(
        &mut self,
        node_id: NodeId,
        label: Option<&str>,
        property: &str,
        old_value: Option<&PropertyValue>,
        new_value: &PropertyValue,
    ) {
        for constraint in self.constraints.values() {
            match &constraint.constraint_type {
                ConstraintType::Unique {
                    property: constrained_prop,
                    label: constraint_label,
                } => {
                    if constrained_prop != property {
                        continue;
                    }

                    if let Some(cl) = constraint_label {
                        if label != Some(cl.as_str()) {
                            continue;
                        }
                    }

                    // Remove old value from index
                    if let Some(old) = old_value {
                        let old_key = (property.to_string(), value_to_key(old));
                        self.unique_index.remove(&old_key);
                    }

                    // Add new value to index
                    let new_key = (property.to_string(), value_to_key(new_value));
                    self.unique_index.insert(new_key, node_id);
                }
            }
        }
    }

    /// Updates indexes after a node deletion.
    pub fn on_delete(&mut self, node_id: NodeId) {
        // Remove all index entries for this node
        self.unique_index.retain(|_, &mut n| n != node_id);
    }

    /// Returns the number of constraints.
    pub fn constraint_count(&self) -> usize {
        self.constraints.len()
    }

    /// Returns the number of unique index entries.
    pub fn unique_index_size(&self) -> usize {
        self.unique_index.len()
    }
}

/// Converts a PropertyValue to a string key for indexing.
fn value_to_key(value: &PropertyValue) -> String {
    match value {
        PropertyValue::Null => "null".to_string(),
        PropertyValue::Bool(b) => format!("b:{}", b),
        PropertyValue::Int(i) => format!("i:{}", i),
        PropertyValue::Float(f) => format!("f:{:.10}", f),
        PropertyValue::String(s) => format!("s:{}", s),
        PropertyValue::Date(s) => format!("d:{}", s),
        PropertyValue::DateTime(s) => format!("dt:{}", s),
        PropertyValue::Vector(v) => format!("v:[{}]", v.len()),
        PropertyValue::Array(a) => format!("a:[{}]", a.len()),
        PropertyValue::Map(m) => format!("m:{{{}}}", m.len()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_create_unique_constraint() {
        let mut manager = ConstraintManager::new();

        manager.create_unique("unique_email", "email", Some("Person")).unwrap();

        assert_eq!(manager.constraint_count(), 1);
        let constraint = manager.get_constraint("unique_email").unwrap();
        assert_eq!(constraint.name, "unique_email");
    }

    #[test]
    fn test_duplicate_constraint_name() {
        let mut manager = ConstraintManager::new();

        manager.create_unique("unique_email", "email", None).unwrap();
        let result = manager.create_unique("unique_email", "name", None);

        assert!(matches!(result, Err(ConstraintError::AlreadyExists { .. })));
    }

    #[test]
    fn test_unique_constraint_blocks_duplicate() {
        let mut manager = ConstraintManager::new();
        manager.create_unique("unique_email", "email", None).unwrap();

        let node1 = NodeId::new(1);
        let node2 = NodeId::new(2);
        let email = ("email".to_string(), PropertyValue::from("alice@example.com"));

        // First insert should succeed
        manager.validate_insert(node1, None, &[email.clone()]).unwrap();
        manager.on_insert(node1, None, &[email.clone()]);

        // Second insert with same email should fail
        let result = manager.validate_insert(node2, None, &[email]);
        assert!(matches!(result, Err(ConstraintError::UniqueViolation { .. })));
    }

    #[test]
    fn test_unique_constraint_with_label_scope() {
        let mut manager = ConstraintManager::new();
        manager.create_unique("unique_person_email", "email", Some("Person")).unwrap();

        let node1 = NodeId::new(1);
        let node2 = NodeId::new(2);
        let email = ("email".to_string(), PropertyValue::from("alice@example.com"));

        // Insert for Person
        manager.validate_insert(node1, Some("Person"), &[email.clone()]).unwrap();
        manager.on_insert(node1, Some("Person"), &[email.clone()]);

        // Same email for Company should succeed (different label)
        manager.validate_insert(node2, Some("Company"), &[email.clone()]).unwrap();

        // Same email for Person should fail
        let node3 = NodeId::new(3);
        let result = manager.validate_insert(node3, Some("Person"), &[email]);
        assert!(matches!(result, Err(ConstraintError::UniqueViolation { .. })));
    }

    #[test]
    fn test_on_delete_removes_index_entries() {
        let mut manager = ConstraintManager::new();
        manager.create_unique("unique_email", "email", None).unwrap();

        let node1 = NodeId::new(1);
        let email = ("email".to_string(), PropertyValue::from("alice@example.com"));

        manager.on_insert(node1, None, &[email.clone()]);
        assert_eq!(manager.unique_index_size(), 1);

        manager.on_delete(node1);
        assert_eq!(manager.unique_index_size(), 0);

        // Now another node can use the same email
        let node2 = NodeId::new(2);
        manager.validate_insert(node2, None, &[email]).unwrap();
    }

    #[test]
    fn test_drop_constraint() {
        let mut manager = ConstraintManager::new();
        manager.create_unique("unique_email", "email", None).unwrap();

        manager.drop_constraint("unique_email").unwrap();
        assert_eq!(manager.constraint_count(), 0);

        // Dropping again should fail
        let result = manager.drop_constraint("unique_email");
        assert!(matches!(result, Err(ConstraintError::NotFound { .. })));
    }

    #[test]
    fn test_list_constraints() {
        let mut manager = ConstraintManager::new();
        manager.create_unique("c1", "email", None).unwrap();
        manager.create_unique("c2", "username", None).unwrap();

        let constraints = manager.list_constraints();
        assert_eq!(constraints.len(), 2);
    }

    #[test]
    fn test_constraint_serialization() {
        let mut manager = ConstraintManager::new();
        manager.create_unique("unique_email", "email", Some("Person")).unwrap();

        let node = NodeId::new(1);
        manager.on_insert(node, Some("Person"), &[
            ("email".to_string(), PropertyValue::from("test@example.com")),
        ]);

        // Serialize and deserialize
        let serialized = bincode::serialize(&manager).unwrap();
        let deserialized: ConstraintManager = bincode::deserialize(&serialized).unwrap();

        assert_eq!(deserialized.constraint_count(), 1);
        assert_eq!(deserialized.unique_index_size(), 1);
    }
}
