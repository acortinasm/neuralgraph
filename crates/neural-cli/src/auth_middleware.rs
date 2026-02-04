//! Authentication middleware for NeuralGraphDB HTTP API.
//!
//! This module provides Axum middleware for authenticating requests via:
//! - JWT tokens (`Authorization: Bearer <token>`)
//! - API keys (`X-API-Key: <key>`)
//!
//! # Usage
//!
//! ```ignore
//! use axum::Router;
//! use neural_cli::auth_middleware::AuthLayer;
//!
//! let protected = Router::new()
//!     .route("/api/query", post(handle_query))
//!     .layer(AuthLayer::new(auth_config));
//! ```

use axum::{
    body::Body,
    extract::Request,
    http::{header, StatusCode},
    middleware::Next,
    response::{IntoResponse, Response},
    Extension,
};
use neural_storage::{AuthConfig, AuthError, AuthUser, Role};
use serde::Serialize;
use std::sync::Arc;

/// Authentication middleware function.
///
/// Extracts credentials from request headers and validates them.
/// If valid, injects `AuthUser` into request extensions.
pub async fn auth_middleware(
    Extension(config): Extension<Arc<AuthConfig>>,
    mut request: Request<Body>,
    next: Next,
) -> Response {
    // If auth is disabled, allow all requests with a default user
    if !config.enabled {
        let default_user = AuthUser::new("anonymous".to_string(), Role::Admin);
        request.extensions_mut().insert(default_user);
        return next.run(request).await;
    }

    // Try to extract credentials from headers
    let auth_result = extract_and_validate(&config, &request);

    match auth_result {
        Ok(user) => {
            // Log successful auth
            tracing::debug!(
                target: "audit",
                user = %user.name,
                role = %user.role,
                path = %request.uri().path(),
                "Request authenticated"
            );

            // Inject user into request extensions
            request.extensions_mut().insert(user);
            next.run(request).await
        }
        Err(e) => {
            // Log failed auth attempt
            tracing::warn!(
                target: "audit",
                error = %e,
                path = %request.uri().path(),
                "Authentication failed"
            );

            auth_error_response(e)
        }
    }
}

/// Extracts credentials from request headers and validates them.
fn extract_and_validate(config: &AuthConfig, request: &Request<Body>) -> Result<AuthUser, AuthError> {
    // Check for Bearer token first
    if let Some(auth_header) = request.headers().get(header::AUTHORIZATION) {
        if let Ok(auth_str) = auth_header.to_str() {
            if let Some(token) = auth_str.strip_prefix("Bearer ") {
                return config.validate_jwt(token);
            }
        }
    }

    // Check for API key
    if let Some(api_key_header) = request.headers().get("X-API-Key") {
        if let Ok(key) = api_key_header.to_str() {
            return config.validate_api_key(key);
        }
    }

    Err(AuthError::MissingAuth)
}

/// Converts an AuthError to an HTTP response.
fn auth_error_response(error: AuthError) -> Response {
    let (status, message) = match &error {
        AuthError::AuthDisabled => (StatusCode::OK, "Authentication disabled".to_string()),
        AuthError::MissingAuth => (
            StatusCode::UNAUTHORIZED,
            "Missing authentication. Provide 'Authorization: Bearer <token>' or 'X-API-Key: <key>' header.".to_string(),
        ),
        AuthError::InvalidToken(msg) => (StatusCode::UNAUTHORIZED, format!("Invalid token: {}", msg)),
        AuthError::InvalidApiKey => (StatusCode::UNAUTHORIZED, "Invalid API key".to_string()),
        AuthError::TokenExpired => (StatusCode::UNAUTHORIZED, "Token expired".to_string()),
        AuthError::InsufficientPermissions(msg) => (StatusCode::FORBIDDEN, msg.clone()),
        AuthError::ConfigurationError(msg) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            format!("Authentication configuration error: {}", msg),
        ),
        AuthError::TokenCreationError(msg) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            format!("Token creation failed: {}", msg),
        ),
    };

    let body = AuthErrorResponse {
        error: message,
        code: match error {
            AuthError::MissingAuth
            | AuthError::InvalidToken(_)
            | AuthError::InvalidApiKey
            | AuthError::TokenExpired => "UNAUTHORIZED",
            AuthError::InsufficientPermissions(_) => "FORBIDDEN",
            _ => "ERROR",
        }
        .to_string(),
    };

    (status, axum::Json(body)).into_response()
}

/// Authentication error response body.
#[derive(Debug, Serialize)]
struct AuthErrorResponse {
    error: String,
    code: String,
}

/// Checks if the authenticated user has permission for the requested operation.
///
/// Returns an error response if permission is denied.
pub fn check_permission(user: &AuthUser, requires_write: bool, requires_admin: bool) -> Result<(), Response> {
    if requires_admin && !user.role.can_admin() {
        tracing::warn!(
            target: "audit",
            user = %user.name,
            role = %user.role,
            "Admin permission denied"
        );
        return Err(auth_error_response(AuthError::InsufficientPermissions(
            format!("Admin role required, but user '{}' has role '{}'", user.name, user.role),
        )));
    }

    if requires_write && !user.role.can_write() {
        tracing::warn!(
            target: "audit",
            user = %user.name,
            role = %user.role,
            "Write permission denied"
        );
        return Err(auth_error_response(AuthError::InsufficientPermissions(
            format!(
                "Write permission required, but user '{}' has readonly role",
                user.name
            ),
        )));
    }

    Ok(())
}

/// Determines if a query is a mutation (CREATE, SET, DELETE, MERGE).
pub fn is_mutation_query(query: &str) -> bool {
    let upper = query.to_uppercase();
    let trimmed = upper.trim();

    // Check for mutation keywords at the start or after whitespace
    trimmed.starts_with("CREATE ")
        || trimmed.starts_with("DELETE ")
        || trimmed.starts_with("SET ")
        || trimmed.starts_with("MERGE ")
        || trimmed.starts_with("REMOVE ")
        || trimmed.contains(" CREATE ")
        || trimmed.contains(" DELETE ")
        || trimmed.contains(" SET ")
        || trimmed.contains(" MERGE ")
        || trimmed.contains(" REMOVE ")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_is_mutation_query() {
        // Read queries
        assert!(!is_mutation_query("MATCH (n) RETURN n"));
        assert!(!is_mutation_query("MATCH (n:Person) RETURN n.name"));
        assert!(!is_mutation_query("  MATCH (n) RETURN count(n)  "));

        // Mutation queries
        assert!(is_mutation_query("CREATE (n:Person {name: 'Alice'})"));
        assert!(is_mutation_query("MATCH (n) DELETE n"));
        assert!(is_mutation_query("MATCH (n) SET n.foo = 'bar'"));
        assert!(is_mutation_query("MERGE (n:Person {name: 'Alice'})"));
        assert!(is_mutation_query("MATCH (n) REMOVE n.foo"));

        // Mixed queries
        assert!(is_mutation_query("MATCH (n) CREATE (m)-[:KNOWS]->(n)"));
        assert!(is_mutation_query("MATCH (a), (b) MERGE (a)-[:KNOWS]->(b)"));
    }

    #[test]
    fn test_check_permission() {
        let admin = AuthUser::new("admin".to_string(), Role::Admin);
        let user = AuthUser::new("user".to_string(), Role::User);
        let readonly = AuthUser::new("readonly".to_string(), Role::Readonly);

        // Admin can do everything
        assert!(check_permission(&admin, false, false).is_ok());
        assert!(check_permission(&admin, true, false).is_ok());
        assert!(check_permission(&admin, false, true).is_ok());
        assert!(check_permission(&admin, true, true).is_ok());

        // User can read and write, but not admin
        assert!(check_permission(&user, false, false).is_ok());
        assert!(check_permission(&user, true, false).is_ok());
        assert!(check_permission(&user, false, true).is_err());

        // Readonly can only read
        assert!(check_permission(&readonly, false, false).is_ok());
        assert!(check_permission(&readonly, true, false).is_err());
        assert!(check_permission(&readonly, false, true).is_err());
    }
}
