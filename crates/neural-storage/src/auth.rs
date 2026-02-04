//! Authentication and authorization for NeuralGraphDB HTTP API.
//!
//! This module provides:
//! - JWT token validation
//! - API key authentication with constant-time comparison
//! - Role-based access control (Admin, User, Readonly)
//!
//! # Security Model
//!
//! | Endpoint | Auth Required | Allowed Roles |
//! |----------|---------------|---------------|
//! | `/health`, `/metrics`, `/` | No | Public |
//! | `/api/papers`, `/api/search`, `/api/similar/{id}`, `/api/schema` | Yes | admin, user, readonly |
//! | `/api/query` (read) | Yes | admin, user, readonly |
//! | `/api/query` (mutate) | Yes | admin, user |
//! | `/api/bulk-load` | Yes | admin only |
//!
//! # Example Configuration
//!
//! ```toml
//! [auth]
//! enabled = true
//! jwt_secret = "your-32-byte-secret-key-here!!!"
//! jwt_expiration_secs = 3600
//!
//! [[auth.api_keys]]
//! name = "admin-service"
//! key_hash = "sha256-hash-of-key"
//! role = "admin"
//! ```

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::time::{SystemTime, UNIX_EPOCH};
use thiserror::Error;

/// Authentication configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct AuthConfig {
    /// Whether authentication is enabled.
    /// When false, all endpoints are accessible without auth.
    pub enabled: bool,

    /// Secret key for JWT signing and verification.
    /// Should be at least 32 bytes for HS256.
    pub jwt_secret: String,

    /// JWT token expiration time in seconds.
    pub jwt_expiration_secs: u64,

    /// List of valid API keys.
    #[serde(default)]
    pub api_keys: Vec<ApiKeyConfig>,
}

impl Default for AuthConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            jwt_secret: String::new(),
            jwt_expiration_secs: 3600,
            api_keys: Vec::new(),
        }
    }
}

impl AuthConfig {
    /// Creates an AuthConfig with authentication enabled.
    pub fn new(jwt_secret: String) -> Self {
        Self {
            enabled: true,
            jwt_secret,
            jwt_expiration_secs: 3600,
            api_keys: Vec::new(),
        }
    }

    /// Adds an API key to the configuration.
    pub fn with_api_key(mut self, name: String, key_hash: String, role: Role) -> Self {
        self.api_keys.push(ApiKeyConfig {
            name,
            key_hash,
            role,
        });
        self
    }

    /// Validates a JWT token and returns the authenticated user.
    pub fn validate_jwt(&self, token: &str) -> Result<AuthUser, AuthError> {
        if !self.enabled {
            return Err(AuthError::AuthDisabled);
        }

        if self.jwt_secret.is_empty() {
            return Err(AuthError::ConfigurationError(
                "JWT secret not configured".to_string(),
            ));
        }

        let key = jsonwebtoken::DecodingKey::from_secret(self.jwt_secret.as_bytes());
        let validation = jsonwebtoken::Validation::new(jsonwebtoken::Algorithm::HS256);

        let token_data = jsonwebtoken::decode::<Claims>(token, &key, &validation)
            .map_err(|e| AuthError::InvalidToken(e.to_string()))?;

        Ok(AuthUser {
            name: token_data.claims.sub,
            role: token_data.claims.role,
        })
    }

    /// Validates an API key and returns the authenticated user.
    ///
    /// Uses constant-time comparison to prevent timing attacks.
    pub fn validate_api_key(&self, key: &str) -> Result<AuthUser, AuthError> {
        if !self.enabled {
            return Err(AuthError::AuthDisabled);
        }

        let key_hash = hash_api_key(key);

        for api_key in &self.api_keys {
            if constant_time_compare(&api_key.key_hash, &key_hash) {
                return Ok(AuthUser {
                    name: api_key.name.clone(),
                    role: api_key.role,
                });
            }
        }

        Err(AuthError::InvalidApiKey)
    }

    /// Creates a JWT token for the given user.
    pub fn create_jwt(&self, user: &AuthUser) -> Result<String, AuthError> {
        if self.jwt_secret.is_empty() {
            return Err(AuthError::ConfigurationError(
                "JWT secret not configured".to_string(),
            ));
        }

        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map_err(|e| AuthError::ConfigurationError(e.to_string()))?
            .as_secs();

        let claims = Claims {
            sub: user.name.clone(),
            role: user.role,
            exp: now + self.jwt_expiration_secs,
            iat: now,
        };

        let key = jsonwebtoken::EncodingKey::from_secret(self.jwt_secret.as_bytes());
        let header = jsonwebtoken::Header::new(jsonwebtoken::Algorithm::HS256);

        jsonwebtoken::encode(&header, &claims, &key)
            .map_err(|e| AuthError::TokenCreationError(e.to_string()))
    }
}

/// API key configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ApiKeyConfig {
    /// Human-readable name for the API key (e.g., "admin-service").
    pub name: String,

    /// SHA-256 hash of the API key.
    pub key_hash: String,

    /// Role associated with this API key.
    pub role: Role,
}

/// User role for authorization.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Role {
    /// Full access: can read, write, and manage.
    Admin,
    /// Read and write access.
    User,
    /// Read-only access.
    Readonly,
}

impl Role {
    /// Checks if this role can perform read operations.
    pub fn can_read(&self) -> bool {
        true // All roles can read
    }

    /// Checks if this role can perform write/mutation operations.
    pub fn can_write(&self) -> bool {
        matches!(self, Role::Admin | Role::User)
    }

    /// Checks if this role can perform admin operations (e.g., bulk-load).
    pub fn can_admin(&self) -> bool {
        matches!(self, Role::Admin)
    }
}

impl std::fmt::Display for Role {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Role::Admin => write!(f, "admin"),
            Role::User => write!(f, "user"),
            Role::Readonly => write!(f, "readonly"),
        }
    }
}

/// Authenticated user information.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AuthUser {
    /// User name or identifier.
    pub name: String,
    /// User's role.
    pub role: Role,
}

impl AuthUser {
    /// Creates a new authenticated user.
    pub fn new(name: String, role: Role) -> Self {
        Self { name, role }
    }
}

/// JWT claims structure.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Claims {
    /// Subject (user name).
    pub sub: String,
    /// User role.
    pub role: Role,
    /// Expiration time (Unix timestamp).
    pub exp: u64,
    /// Issued at time (Unix timestamp).
    pub iat: u64,
}

/// Authentication errors.
#[derive(Debug, Error)]
pub enum AuthError {
    #[error("Authentication is disabled")]
    AuthDisabled,

    #[error("Missing authentication")]
    MissingAuth,

    #[error("Invalid token: {0}")]
    InvalidToken(String),

    #[error("Invalid API key")]
    InvalidApiKey,

    #[error("Insufficient permissions: {0}")]
    InsufficientPermissions(String),

    #[error("Token expired")]
    TokenExpired,

    #[error("Configuration error: {0}")]
    ConfigurationError(String),

    #[error("Token creation failed: {0}")]
    TokenCreationError(String),
}

/// Computes SHA-256 hash of an API key.
pub fn hash_api_key(key: &str) -> String {
    let mut hasher = Sha256::new();
    hasher.update(key.as_bytes());
    let result = hasher.finalize();
    format!("{:x}", result)
}

/// Constant-time string comparison to prevent timing attacks.
///
/// Returns true if the strings are equal.
fn constant_time_compare(a: &str, b: &str) -> bool {
    if a.len() != b.len() {
        return false;
    }

    let mut result = 0u8;
    for (x, y) in a.bytes().zip(b.bytes()) {
        result |= x ^ y;
    }
    result == 0
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_role_permissions() {
        assert!(Role::Admin.can_read());
        assert!(Role::Admin.can_write());
        assert!(Role::Admin.can_admin());

        assert!(Role::User.can_read());
        assert!(Role::User.can_write());
        assert!(!Role::User.can_admin());

        assert!(Role::Readonly.can_read());
        assert!(!Role::Readonly.can_write());
        assert!(!Role::Readonly.can_admin());
    }

    #[test]
    fn test_hash_api_key() {
        let key = "test-api-key-12345";
        let hash = hash_api_key(key);

        // Hash should be 64 hex characters (256 bits)
        assert_eq!(hash.len(), 64);

        // Same key should produce same hash
        assert_eq!(hash, hash_api_key(key));

        // Different key should produce different hash
        assert_ne!(hash, hash_api_key("different-key"));
    }

    #[test]
    fn test_constant_time_compare() {
        assert!(constant_time_compare("abc", "abc"));
        assert!(!constant_time_compare("abc", "abd"));
        assert!(!constant_time_compare("abc", "ab"));
        assert!(!constant_time_compare("ab", "abc"));
        assert!(constant_time_compare("", ""));
    }

    #[test]
    fn test_jwt_roundtrip() {
        let config = AuthConfig::new("test-secret-key-at-least-32-bytes!".to_string());
        let user = AuthUser::new("testuser".to_string(), Role::User);

        let token = config.create_jwt(&user).unwrap();
        let validated = config.validate_jwt(&token).unwrap();

        assert_eq!(validated.name, "testuser");
        assert_eq!(validated.role, Role::User);
    }

    #[test]
    fn test_api_key_validation() {
        let key = "my-secret-api-key";
        let key_hash = hash_api_key(key);

        let config = AuthConfig::new("jwt-secret-not-used-here-32bytes!".to_string())
            .with_api_key("test-service".to_string(), key_hash, Role::Admin);

        let user = config.validate_api_key(key).unwrap();
        assert_eq!(user.name, "test-service");
        assert_eq!(user.role, Role::Admin);

        // Wrong key should fail
        assert!(config.validate_api_key("wrong-key").is_err());
    }

    #[test]
    fn test_auth_disabled() {
        let config = AuthConfig::default(); // enabled = false

        assert!(matches!(
            config.validate_jwt("any-token"),
            Err(AuthError::AuthDisabled)
        ));
        assert!(matches!(
            config.validate_api_key("any-key"),
            Err(AuthError::AuthDisabled)
        ));
    }

    #[test]
    fn test_role_serialization() {
        // Test that roles serialize to lowercase
        assert_eq!(
            serde_json::to_string(&Role::Admin).unwrap(),
            "\"admin\""
        );
        assert_eq!(
            serde_json::to_string(&Role::User).unwrap(),
            "\"user\""
        );
        assert_eq!(
            serde_json::to_string(&Role::Readonly).unwrap(),
            "\"readonly\""
        );

        // Test deserialization
        assert_eq!(
            serde_json::from_str::<Role>("\"admin\"").unwrap(),
            Role::Admin
        );
    }

    #[test]
    fn test_auth_config_serialization() {
        let config = AuthConfig {
            enabled: true,
            jwt_secret: "secret".to_string(),
            jwt_expiration_secs: 7200,
            api_keys: vec![ApiKeyConfig {
                name: "test".to_string(),
                key_hash: "abc123".to_string(),
                role: Role::User,
            }],
        };

        let toml_str = toml::to_string(&config).unwrap();
        assert!(toml_str.contains("enabled = true"));
        assert!(toml_str.contains("jwt_expiration_secs = 7200"));

        let parsed: AuthConfig = toml::from_str(&toml_str).unwrap();
        assert_eq!(parsed.enabled, config.enabled);
        assert_eq!(parsed.jwt_expiration_secs, config.jwt_expiration_secs);
    }
}
