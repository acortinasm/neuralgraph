//! # Neural Parser
//!
//! NGQL (Neural Graph Query Language) parser for NeuralGraphDB.
//!
//! This crate provides:
//! - [`lexer`] - Tokenization using `logos`
//! - [`ast`] - Abstract Syntax Tree types
//! - [`parser`] - Query parsing
//!
//! ## Example
//!
//! ```
//! use neural_parser::parse_query;
//!
//! let query = parse_query("MATCH (n:Person) RETURN n").unwrap();
//! println!("{:#?}", query);
//! ```

pub mod ast;
pub mod lexer;
pub mod parser;

pub use ast::*;
pub use lexer::Token;
pub use parser::{ParseError, parse_query, parse_statement};
