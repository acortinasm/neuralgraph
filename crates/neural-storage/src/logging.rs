//! Structured logging for NeuralGraphDB.
//!
//! This module provides centralized logging configuration using the `tracing` crate.
//! Logs can be configured via the `NGDB_LOG` environment variable.
//!
//! # Environment Variables
//!
//! - `NGDB_LOG=info` - Default log level (info)
//! - `NGDB_LOG=debug` - Verbose logging
//! - `NGDB_LOG=neural_storage::wal=debug` - Module-specific logging
//! - `NGDB_LOG=warn,neural_storage::persistence=debug` - Combined filters
//!
//! # Example
//!
//! ```ignore
//! use neural_storage::logging;
//!
//! // Initialize logging (call once at startup)
//! logging::init();
//!
//! // Or with a custom default level
//! logging::init_with_default("debug");
//! ```

use tracing_subscriber::{fmt, EnvFilter};

/// Initializes the global tracing subscriber with default settings.
///
/// Uses the `NGDB_LOG` environment variable for configuration.
/// Default level is `info` if not specified.
///
/// This function should be called once at application startup.
/// Subsequent calls will be ignored (tracing only allows one subscriber).
pub fn init() {
    init_with_default("info");
}

/// Initializes the global tracing subscriber with a custom default level.
///
/// # Arguments
/// * `default_level` - Default log level if `NGDB_LOG` is not set
///
/// # Example
/// ```ignore
/// // Enable debug logging by default
/// logging::init_with_default("debug");
/// ```
pub fn init_with_default(default_level: &str) {
    let filter = EnvFilter::try_from_env("NGDB_LOG")
        .unwrap_or_else(|_| EnvFilter::new(default_level));

    let subscriber = fmt()
        .with_env_filter(filter)
        .with_target(true)
        .with_thread_ids(false)
        .with_thread_names(false)
        .with_file(false)
        .with_line_number(false)
        .compact();

    // Try to set the global default - ignore if already set
    let _ = subscriber.try_init();
}

/// Initializes logging with JSON output format.
///
/// Useful for production environments where logs are processed by log aggregators.
pub fn init_json() {
    let filter = EnvFilter::try_from_env("NGDB_LOG")
        .unwrap_or_else(|_| EnvFilter::new("info"));

    let subscriber = fmt()
        .with_env_filter(filter)
        .with_target(true)
        .json();

    let _ = subscriber.try_init();
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_init_does_not_panic() {
        // Multiple calls should not panic
        init();
        init();
        init_with_default("warn");
    }
}
