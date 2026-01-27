//! Abstract Syntax Tree (AST) for NGQL.
//!
//! This module defines the data structures that represent parsed NGQL queries.

use serde::{Deserialize, Serialize};

// =============================================================================
// Query (Root)
// =============================================================================

/// A complete NGQL query.
///
/// A query is a sequence of clauses (MATCH, WITH, UNWIND, RETURN).
///
/// ## Example
///
/// ```text
/// MATCH (a:Person)
/// WITH a.name AS name
/// RETURN name
/// ```
///
/// ## Time-Travel Query (Sprint 54)
///
/// ```text
/// MATCH (a:Person) AT TIME '2026-01-15T12:00:00Z'
/// RETURN a.name
/// ```
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Query {
    /// The sequence of clauses
    pub clauses: Vec<Clause>,
    /// Optional temporal clause for time-travel queries (Sprint 54)
    pub temporal: Option<TemporalClause>,
    /// Optional shard hint for explicit routing (Sprint 55)
    pub shard_hint: Option<ShardHint>,
}

/// Temporal clause for time-travel queries (Sprint 54).
///
/// Allows querying the database as it existed at a specific point in time.
///
/// ## Syntax
///
/// ```text
/// AT TIME '2026-01-15T12:00:00Z'
/// AT TIMESTAMP '2026-01-15T12:00:00Z'
/// ```
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TemporalClause {
    /// The timestamp expression (typically a string literal or parameter)
    pub timestamp: Expression,
}

/// Shard hint for routing queries to specific shards (Sprint 55).
///
/// Allows explicit shard targeting for optimized query routing.
///
/// ## Syntax
///
/// ```text
/// USING SHARD 0               -- Single shard
/// USING SHARD [0, 1, 2]       -- Multiple shards
/// ```
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ShardHint {
    /// The shard ID(s) to target
    pub shards: Vec<u32>,
}

/// A clause in a query pipeline.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum Clause {
    /// MATCH clause (with optional WHERE)
    Match(MatchClause),
    /// OPTIONAL MATCH clause
    OptionalMatch(MatchClause),
    /// MERGE clause
    Merge(MergeClause),
    /// WITH clause (projection + filtering + sorting + limit)
    With(WithClause),
    /// UNWIND clause (list expansion)
    Unwind(UnwindClause),
    /// RETURN clause (projection + sorting + limit)
    Return(ReturnClause),
}

/// MERGE clause for upserting nodes/edges.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MergeClause {
    /// Pattern to match or create
    pub pattern: Pattern,
    // Future: ON MATCH SET, ON CREATE SET
}

// =============================================================================
// Statement (Mutation support - Sprint 21+)
// =============================================================================

/// A complete NGQL statement - can be a read query or a mutation.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum Statement {
    /// A read query (MATCH ... RETURN)
    Query(Query),
    /// A CREATE statement
    Create(CreateClause),
    /// A DELETE statement (with optional MATCH)
    Delete {
        /// Optional MATCH to find nodes to delete
        match_clause: Option<MatchClause>,
        /// Optional WHERE filter
        where_clause: Option<WhereClause>,
        /// The DELETE clause
        delete_clause: DeleteClause,
    },
    /// A SET statement (requires MATCH to find nodes)
    Set {
        /// MATCH to find nodes/edges to update
        match_clause: MatchClause,
        /// Optional WHERE filter
        where_clause: Option<WhereClause>,
        /// The SET clause
        set_clause: SetClause,
    },
    /// A CREATE statement following a MATCH (e.g. creating relationships between found nodes)
    CreateWithMatch {
        /// MATCH to find nodes
        match_clause: MatchClause,
        /// Optional WHERE filter
        where_clause: Option<WhereClause>,
        /// The CREATE clause
        create_clause: CreateClause,
    },
    /// EXPLAIN a statement (show plan)
    Explain(Box<Statement>),
    /// PROFILE a statement (execute and show stats)
    Profile(Box<Statement>),
    /// Begin a transaction
    Begin,
    /// Commit a transaction
    Commit,
    /// Rollback a transaction
    Rollback,
    /// Flashback to a specific point in time (Sprint 54)
    ///
    /// ```text
    /// FLASHBACK TO '2026-01-15T12:00:00Z'
    /// ```
    Flashback {
        /// The timestamp to revert to
        timestamp: Expression,
    },
    /// CALL a procedure (Sprint 56)
    ///
    /// ```text
    /// CALL neural.search($vec, 'euclidean', 10)
    /// ```
    Call(CallClause),
}

/// CALL clause for procedure invocation (Sprint 56).
///
/// ## Supported Procedures
///
/// - `neural.search($vec, metric, k)` - Vector similarity search
///
/// ## Syntax
///
/// ```text
/// CALL neural.search($queryVector, 'cosine', 10)
/// CALL neural.search($queryVector, 'euclidean', 10)
/// CALL neural.search($queryVector, 'dot_product', 10)
/// ```
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CallClause {
    /// Procedure namespace (e.g., "neural")
    pub namespace: String,
    /// Procedure name (e.g., "search")
    pub name: String,
    /// Procedure arguments
    pub args: Vec<Expression>,
}

/// CREATE clause for creating nodes and edges.
///
/// ## Examples
///
/// ```text
/// CREATE (n:Person {name: "Alice", age: 30})
/// CREATE (a)-[:KNOWS]->(b)
/// ```
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CreateClause {
    /// Patterns to create (nodes and edges)
    pub patterns: Vec<CreatePattern>,
}

/// A pattern to create - either a node or an edge.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum CreatePattern {
    /// Create a node with optional label and properties
    Node {
        /// Optional binding variable
        binding: Option<String>,
        /// Optional label
        label: Option<String>,
        /// Properties to set
        properties: Vec<(String, Literal)>,
    },
    /// Create an edge between two bindings
    Edge {
        /// Source node binding
        from: String,
        /// Target node binding
        to: String,
        /// Optional edge type
        edge_type: Option<String>,
        /// Properties to set on the edge
        properties: Vec<(String, Literal)>,
    },
}

/// DELETE clause for removing nodes and edges.
///
/// ## Examples
///
/// ```text
/// MATCH (n:Person) WHERE n.name = "Alice" DELETE n
/// MATCH (n) DETACH DELETE n
/// ```
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DeleteClause {
    /// Whether to detach (delete edges first)
    pub detach: bool,
    /// Variables to delete
    pub items: Vec<String>,
}

/// SET clause for updating properties.
///
/// ## Examples
///
/// ```text
/// MATCH (n:Person) WHERE n.name = "Alice" SET n.age = 31
/// MATCH (n) SET n.updated = true, n.version = 2
/// ```
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SetClause {
    /// Property assignments
    pub assignments: Vec<PropertyAssignment>,
}

/// A property assignment in a SET clause.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PropertyAssignment {
    /// The variable (e.g., "n")
    pub variable: String,
    /// The property name (e.g., "age")
    pub property: String,
    /// The value to set
    pub value: Expression,
}

// =============================================================================
// MATCH Clause
// =============================================================================

/// The MATCH clause containing graph patterns.
///
/// ```text
/// MATCH (a)-[:REL]->(b), (c)-[:OTHER]->(d)
/// ```
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MatchClause {
    /// One or more patterns to match
    pub patterns: Vec<Pattern>,
    /// Optional WHERE clause
    pub where_clause: Option<WhereClause>,
}

/// A graph pattern: a sequence of nodes connected by relationships.
///
/// ```text
/// (a:Person)-[:KNOWS]->(b:Person)-[:WORKS_AT]->(c:Company)
/// ```
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Pattern {
    /// Optional identifier for the whole path (e.g., `p = (a)-[...]-(b)`)
    pub identifier: Option<String>,
    /// The starting node
    pub start: NodePattern,
    /// Chain of (relationship, node) pairs
    pub chain: Vec<(RelPattern, NodePattern)>,
    /// The matching mode (Normal or ShortestPath)
    pub mode: PatternMode,
}

/// Mode of pattern matching.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum PatternMode {
    /// Standard pattern matching (find all occurrences)
    #[default]
    Normal,
    /// Find the shortest path matching the pattern
    ShortestPath,
}

impl Pattern {
    /// Creates a simple pattern with just a single node.
    pub fn single(node: NodePattern) -> Self {
        Self {
            identifier: None,
            start: node,
            chain: Vec::new(),
            mode: PatternMode::Normal,
        }
    }

    /// Returns all node patterns in this pattern.
    pub fn nodes(&self) -> impl Iterator<Item = &NodePattern> {
        std::iter::once(&self.start).chain(self.chain.iter().map(|(_, n)| n))
    }

    /// Returns all relationship patterns in this pattern.
    pub fn relationships(&self) -> impl Iterator<Item = &RelPattern> {
        self.chain.iter().map(|(r, _)| r)
    }
}

/// A node pattern: `(n:Label)` or `(n)` or `(:Label)` or `()`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct NodePattern {
    /// Optional variable binding (e.g., `n` in `(n:Person)`)
    pub identifier: Option<String>,
    /// Optional label (e.g., `Person` in `(n:Person)`)
    pub label: Option<String>,
    /// Optional inline properties (e.g., `{name: "Alice"}`)
    pub properties: Option<PropertyMap>,
}

impl NodePattern {
    /// Creates an anonymous node pattern `()`.
    pub fn anonymous() -> Self {
        Self {
            identifier: None,
            label: None,
            properties: None,
        }
    }

    /// Creates a node pattern with just an identifier.
    pub fn with_identifier(name: impl Into<String>) -> Self {
        Self {
            identifier: Some(name.into()),
            label: None,
            properties: None,
        }
    }

    /// Creates a node pattern with identifier and label.
    pub fn with_label(name: impl Into<String>, label: impl Into<String>) -> Self {
        Self {
            identifier: Some(name.into()),
            label: Some(label.into()),
            properties: None,
        }
    }
}

/// A relationship pattern: `-[:KNOWS]->` or `-[r:KNOWS]->` or `-->`.
///
/// ## Port Numbering (Sprint 57)
///
/// Supports port syntax for parallel multi-edges:
/// ```text
/// -[:DEPENDS_ON:0]->  // Port 0
/// -[:DEPENDS_ON:1]->  // Port 1
/// -[r:TRANSFER:2]->   // Named edge with port 2
/// ```
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RelPattern {
    /// Optional variable binding
    pub identifier: Option<String>,
    /// Optional relationship type/label
    pub label: Option<String>,
    /// Direction of the relationship
    pub direction: Direction,
    /// Variable length pattern (e.g. *1..5)
    pub var_length: Option<VarLength>,
    /// Port number for parallel multi-edges (Sprint 57)
    #[serde(default)]
    pub port: Option<u16>,
}

/// Variable length definition for relationships.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct VarLength {
    /// Minimum hops (default 1)
    pub min: u64,
    /// Maximum hops (None = Infinity)
    pub max: Option<u64>,
}

impl RelPattern {
    /// Creates an anonymous outgoing relationship.
    pub fn outgoing() -> Self {
        Self {
            identifier: None,
            label: None,
            direction: Direction::Outgoing,
            var_length: None,
            port: None,
        }
    }

    /// Creates an outgoing relationship with a label.
    pub fn outgoing_with_label(label: impl Into<String>) -> Self {
        Self {
            identifier: None,
            label: Some(label.into()),
            direction: Direction::Outgoing,
            var_length: None,
            port: None,
        }
    }

    /// Creates an outgoing relationship with a label and port (Sprint 57).
    pub fn outgoing_with_label_and_port(label: impl Into<String>, port: u16) -> Self {
        Self {
            identifier: None,
            label: Some(label.into()),
            direction: Direction::Outgoing,
            var_length: None,
            port: Some(port),
        }
    }
}

/// Direction of a relationship.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Direction {
    /// Outgoing: `->`
    Outgoing,
    /// Incoming: `<-`
    Incoming,
    /// Undirected/Both: `--` or `-`
    Both,
}

// =============================================================================
// WHERE Clause
// =============================================================================

/// The WHERE clause containing filter expressions.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct WhereClause {
    /// The filter expression
    pub expression: Expression,
}

/// An expression that can be evaluated.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum Expression {
    /// Property access: `n.name`
    Property { variable: String, property: String },
    /// Literal value
    Literal(Literal),
    /// Parameter: `$name`
    Parameter(String),
    /// Binary comparison: `a = b`, `a > b`, etc.
    Comparison {
        left: Box<Expression>,
        op: ComparisonOp,
        right: Box<Expression>,
    },
    /// Logical AND
    And(Box<Expression>, Box<Expression>),
    /// Logical OR
    Or(Box<Expression>, Box<Expression>),
    /// Logical NOT
    Not(Box<Expression>),
    /// CASE expression
    Case {
        /// Optional subject (CASE n.name WHEN ...)
        subject: Option<Box<Expression>>,
        /// WHEN ... THEN ... branches
        when_then: Vec<(Expression, Expression)>,
        /// ELSE expression
        else_expr: Option<Box<Expression>>,
    },
    /// Function call (general purpose)
    FunctionCall {
        name: String,
        args: Vec<Expression>,
    },
    /// List Literal (e.g. [1, 2, 3])
    List(Vec<Expression>),
    /// Aggregate function: COUNT(n), SUM(n.age), etc.
    Aggregate {
        function: AggregateFunction,
        /// None for COUNT(*)
        argument: Option<Box<Expression>>,
        /// DISTINCT modifier
        distinct: bool,
    },
    /// Vector similarity function: vector_similarity(n.embedding, $query)
    /// Returns similarity score between node's vector and query vector (Sprint 13)
    VectorSimilarity {
        /// Property expression (e.g., n.embedding)
        property: Box<Expression>,
        /// Query vector expression (e.g., parameter or literal)
        query: Box<Expression>,
    },
    /// Vector search with metric: neural.search($vec, 'euclidean', 10) (Sprint 56)
    /// Returns top-k similar nodes using specified distance metric
    VectorSearch {
        /// Query vector expression (e.g., parameter or literal)
        query: Box<Expression>,
        /// Distance metric: 'cosine', 'euclidean', 'dot_product', 'l2'
        metric: String,
        /// Number of results to return (k)
        k: u64,
    },
    /// Community detection function: CLUSTER(n)
    /// Returns the community ID for the node using Leiden algorithm (Sprint 16)
    Cluster {
        /// Variable to get community for
        variable: String,
    },
}

/// Comparison operators.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ComparisonOp {
    /// `=`
    Eq,
    /// `<>`
    Neq,
    /// `<`
    Lt,
    /// `>`
    Gt,
    /// `<=`
    Lte,
    /// `>=`
    Gte,
}

/// A literal value.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum Literal {
    /// Null
    Null,
    /// Boolean
    Bool(bool),
    /// Integer
    Int(i64),
    /// Float
    Float(f64),
    /// String
    String(String),
    /// List of Literals
    List(Vec<Literal>),
}

/// Aggregate functions for RETURN clauses.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum AggregateFunction {
    /// COUNT(*) or COUNT(expr)
    Count,
    /// SUM(expr)
    Sum,
    /// AVG(expr)
    Avg,
    /// MIN(expr)
    Min,
    /// MAX(expr)
    Max,
    /// COLLECT(expr) - collect into list
    Collect,
}

impl std::fmt::Display for AggregateFunction {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            AggregateFunction::Count => write!(f, "COUNT"),
            AggregateFunction::Sum => write!(f, "SUM"),
            AggregateFunction::Avg => write!(f, "AVG"),
            AggregateFunction::Min => write!(f, "MIN"),
            AggregateFunction::Max => write!(f, "MAX"),
            AggregateFunction::Collect => write!(f, "COLLECT"),
        }
    }
}

// =============================================================================
// WITH Clause
// =============================================================================

/// The WITH clause for chaining query parts.
///
/// acts as a projection boundary.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct WithClause {
    /// Items to project
    pub items: Vec<ReturnItem>, // Reusing ReturnItem as it is effectively the same (expr AS alias)
    /// Whether DISTINCT was specified
    pub distinct: bool,
    /// Optional ORDER BY
    pub order_by: Option<OrderByClause>,
    /// Optional LIMIT
    pub limit: Option<u64>,
    /// Optional WHERE filter
    pub where_clause: Option<WhereClause>,
}

// =============================================================================
// UNWIND Clause
// =============================================================================

/// The UNWIND clause for expanding lists.
///
/// ```text
/// UNWIND [1, 2, 3] AS x
/// ```
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct UnwindClause {
    /// The expression yielding a list
    pub expression: Expression,
    /// The variable to bind each element to
    pub alias: String,
}

// =============================================================================
// RETURN Clause
// =============================================================================

/// The RETURN clause specifying what to return.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ReturnClause {
    /// Items to return
    pub items: Vec<ReturnItem>,
    /// Whether DISTINCT was specified
    pub distinct: bool,
    /// Optional ORDER BY
    pub order_by: Option<OrderByClause>,
    /// Optional LIMIT
    pub limit: Option<u64>,
    /// Optional GROUP BY (Non-standard Cypher, but nGraph specific)
    pub group_by: Option<GroupByClause>,
}

/// A single item in the RETURN clause.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ReturnItem {
    /// The expression to return
    pub expression: Expression,
    /// Optional alias (AS name)
    pub alias: Option<String>,
}

impl ReturnItem {
    /// Creates a simple return item for a variable.
    pub fn variable(name: impl Into<String>) -> Self {
        let name = name.into();
        Self {
            expression: Expression::Property {
                variable: name.clone(),
                property: String::new(), // Empty means return the whole node
            },
            alias: None,
        }
    }

    /// Creates a return item for a property access.
    pub fn property(variable: impl Into<String>, property: impl Into<String>) -> Self {
        Self {
            expression: Expression::Property {
                variable: variable.into(),
                property: property.into(),
            },
            alias: None,
        }
    }
}

// =============================================================================
// ORDER BY Clause
// =============================================================================

/// Sort direction for ORDER BY.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum SortDirection {
    /// Ascending order (default)
    #[default]
    Ascending,
    /// Descending order
    Descending,
}

/// An item in the ORDER BY clause.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct OrderByItem {
    /// The expression to sort by
    pub expression: Expression,
    /// Sort direction (ASC or DESC)
    pub direction: SortDirection,
}

/// The ORDER BY clause specifying result ordering.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct OrderByClause {
    /// Items to order by
    pub items: Vec<OrderByItem>,
}

/// The GROUP BY clause specifying grouping expressions.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct GroupByClause {
    /// Expressions to group by
    pub expressions: Vec<Expression>,
}

// =============================================================================
// Helpers
// =============================================================================

/// A map of property names to values (for inline properties).
pub type PropertyMap = std::collections::HashMap<String, Literal>;

// =============================================================================
// Display implementations for debugging
// =============================================================================

impl std::fmt::Display for Query {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        for (i, clause) in self.clauses.iter().enumerate() {
            if i > 0 {
                write!(f, " ")?;
            }
            match clause {
                Clause::Match(c) => write!(f, "{}", c)?,
                Clause::OptionalMatch(c) => write!(f, "OPTIONAL {}", c)?,
                Clause::Merge(c) => write!(f, "MERGE {}", c.pattern)?,
                Clause::With(c) => write!(f, "{}", c)?,
                Clause::Unwind(c) => write!(f, "{}", c)?,
                Clause::Return(c) => write!(f, "{}", c)?,
            }
        }
        if let Some(ref temporal) = self.temporal {
            write!(f, " {}", temporal)?;
        }
        if let Some(ref shard_hint) = self.shard_hint {
            write!(f, " {}", shard_hint)?;
        }
        Ok(())
    }
}

impl std::fmt::Display for ShardHint {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if self.shards.len() == 1 {
            write!(f, "USING SHARD {}", self.shards[0])
        } else {
            write!(f, "USING SHARD [")?;
            for (i, shard) in self.shards.iter().enumerate() {
                if i > 0 {
                    write!(f, ", ")?;
                }
                write!(f, "{}", shard)?;
            }
            write!(f, "]")
        }
    }
}

impl std::fmt::Display for TemporalClause {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "AT TIME {}", self.timestamp)
    }
}

impl std::fmt::Display for MatchClause {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "MATCH ")?;
        for (i, pattern) in self.patterns.iter().enumerate() {
            if i > 0 {
                write!(f, ", ")?;
            }
            write!(f, "{}", pattern)?;
        }
        if let Some(ref where_clause) = self.where_clause {
            write!(f, " {}", where_clause)?;
        }
        Ok(())
    }
}

impl std::fmt::Display for WithClause {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "WITH ")?;
        if self.distinct {
            write!(f, "DISTINCT ")?;
        }
        for (i, item) in self.items.iter().enumerate() {
            if i > 0 {
                write!(f, ", ")?;
            }
            write!(f, "{}", item)?;
        }
        if let Some(ref order_by) = self.order_by {
            // OrderByClause display not yet implemented fully or needs check
            // Assuming we implement it or it exists
            // OrderByClause doesn't have Display impl in previous read? 
            // Let's check. OrderByClause struct exists.
            // I'll implement Display for OrderByClause later if missing.
            // For now, let's manually format items
             write!(f, " ORDER BY ")?;
             for (i, item) in order_by.items.iter().enumerate() {
                 if i > 0 { write!(f, ", ")?; }
                 write!(f, "{}", item.expression)?;
                 if matches!(item.direction, SortDirection::Descending) {
                     write!(f, " DESC")?;
                 }
             }
        }
        if let Some(limit) = self.limit {
            write!(f, " LIMIT {}", limit)?;
        }
        if let Some(ref where_clause) = self.where_clause {
            write!(f, " {}", where_clause)?;
        }
        Ok(())
    }
}

impl std::fmt::Display for UnwindClause {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "UNWIND {} AS {}", self.expression, self.alias)
    }
}

impl std::fmt::Display for Pattern {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.start)?;
        for (rel, node) in &self.chain {
            write!(f, "{}{}", rel, node)?;
        }
        Ok(())
    }
}

impl std::fmt::Display for NodePattern {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "(")?;
        if let Some(ref id) = self.identifier {
            write!(f, "{}", id)?;
        }
        if let Some(ref label) = self.label {
            write!(f, ":{}", label)?;
        }
        write!(f, ")")
    }
}

impl std::fmt::Display for RelPattern {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self.direction {
            Direction::Incoming => write!(f, "<-")?,
            _ => write!(f, "-")?,
        }

        write!(f, "[")?;
        if let Some(ref id) = self.identifier {
            write!(f, "{}", id)?;
        }
        if let Some(ref label) = self.label {
            write!(f, ":{}", label)?;
        }
        // Port number (Sprint 57)
        if let Some(port) = self.port {
            write!(f, ":{}", port)?;
        }
        if let Some(ref var_len) = self.var_length {
            write!(f, "*")?;
            if var_len.min > 1 || var_len.max.is_some() {
                write!(f, "{}", var_len.min)?;
                write!(f, "..")?;
                if let Some(max) = var_len.max {
                    write!(f, "{}", max)?;
                }
            }
        }
        write!(f, "]")?;

        match self.direction {
            Direction::Outgoing => write!(f, "->"),
            Direction::Incoming => write!(f, "-"),
            Direction::Both => write!(f, "-"),
        }
    }
}

impl std::fmt::Display for WhereClause {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "WHERE {}", self.expression)
    }
}

impl std::fmt::Display for Expression {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Expression::Property { variable, property } => {
                if property.is_empty() {
                    write!(f, "{}", variable)
                } else {
                    write!(f, "{}.{}", variable, property)
                }
            }
            Expression::Literal(lit) => write!(f, "{}", lit),
            Expression::Parameter(name) => write!(f, "${}", name),
            Expression::Comparison { left, op, right } => {
                write!(f, "{} {} {}", left, op, right)
            }
            Expression::And(left, right) => write!(f, "({} AND {})", left, right),
            Expression::Or(left, right) => write!(f, "({} OR {})", left, right),
            Expression::Not(expr) => write!(f, "NOT {}", expr),
            Expression::Case { subject, when_then, else_expr } => {
                write!(f, "CASE")?;
                if let Some(subj) = subject {
                    write!(f, " {}", subj)?;
                }
                for (w, t) in when_then {
                    write!(f, " WHEN {} THEN {}", w, t)?;
                }
                if let Some(el) = else_expr {
                    write!(f, " ELSE {}", el)?;
                }
                write!(f, " END")
            }
            Expression::FunctionCall { name, args } => {
                write!(f, "{}(", name)?;
                for (i, arg) in args.iter().enumerate() {
                    if i > 0 { write!(f, ", ")?; }
                    write!(f, "{}", arg)?;
                }
                write!(f, ")")
            }
            Expression::List(items) => {
                write!(f, "[")?;
                for (i, item) in items.iter().enumerate() {
                    if i > 0 { write!(f, ", ")?; }
                    write!(f, "{}", item)?;
                }
                write!(f, "]")
            }
            Expression::Aggregate {
                function,
                argument,
                distinct,
            } => {
                write!(f, "{}(", function)?;
                if *distinct {
                    write!(f, "DISTINCT ")?;
                }
                match argument {
                    Some(arg) => write!(f, "{})", arg),
                    None => write!(f, "*)"),
                }
            }
            Expression::VectorSimilarity { property, query } => {
                write!(f, "vector_similarity({}, {})", property, query)
            }
            Expression::VectorSearch { query, metric, k } => {
                write!(f, "neural.search({}, '{}', {})", query, metric, k)
            }
            Expression::Cluster { variable } => {
                write!(f, "CLUSTER({})", variable)
            }
        }
    }
}

impl std::fmt::Display for ComparisonOp {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ComparisonOp::Eq => write!(f, "="),
            ComparisonOp::Neq => write!(f, "<>"),
            ComparisonOp::Lt => write!(f, "<"),
            ComparisonOp::Gt => write!(f, ">"),
            ComparisonOp::Lte => write!(f, "<="),
            ComparisonOp::Gte => write!(f, ">="),
        }
    }
}

impl std::fmt::Display for Literal {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Literal::Null => write!(f, "NULL"),
            Literal::Bool(b) => write!(f, "{}", if *b { "TRUE" } else { "FALSE" }),
            Literal::Int(i) => write!(f, "{}", i),
            Literal::Float(fl) => write!(f, "{}", fl),
            Literal::String(s) => write!(f, "\"{}\"", s),
            Literal::List(l) => {
                write!(f, "[")?;
                for (i, v) in l.iter().enumerate() {
                    if i > 0 { write!(f, ", ")?; }
                    write!(f, "{}", v)?;
                }
                write!(f, "]")
            }
        }
    }
}

impl std::fmt::Display for ReturnClause {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "RETURN ")?;
        if self.distinct {
            write!(f, "DISTINCT ")?;
        }
        for (i, item) in self.items.iter().enumerate() {
            if i > 0 {
                write!(f, ", ")?;
            }
            write!(f, "{}", item)?;
        }
        if let Some(ref order_by) = self.order_by {
             write!(f, " ORDER BY ")?;
             for (i, item) in order_by.items.iter().enumerate() {
                 if i > 0 { write!(f, ", ")?; }
                 write!(f, "{}", item.expression)?;
                 if matches!(item.direction, SortDirection::Descending) {
                     write!(f, " DESC")?;
                 }
             }
        }
        if let Some(limit) = self.limit {
            write!(f, " LIMIT {}", limit)?;
        }
        // Group By
        if let Some(ref group_by) = self.group_by {
            write!(f, " GROUP BY ")?;
             for (i, expr) in group_by.expressions.iter().enumerate() {
                 if i > 0 { write!(f, ", ")?; }
                 write!(f, "{}", expr)?;
             }
        }
        Ok(())
    }
}

impl std::fmt::Display for ReturnItem {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.expression)?;
        if let Some(ref alias) = self.alias {
            write!(f, " AS {}", alias)?;
        }
        Ok(())
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_simple_pattern_display() {
        let pattern = Pattern::single(NodePattern::with_label("n", "Person"));
        assert_eq!(format!("{}", pattern), "(n:Person)");
    }

    #[test]
    fn test_relationship_pattern_display() {
        let pattern = Pattern {
            identifier: None,
            start: NodePattern::with_identifier("a"),
            chain: vec![(
                RelPattern::outgoing_with_label("KNOWS"),
                NodePattern::with_identifier("b"),
            )],
            mode: PatternMode::Normal,
        };
        assert_eq!(format!("{}", pattern), "(a)-[:KNOWS]->(b)");
    }

    #[test]
    fn test_query_display() {
        let query = Query {
            clauses: vec![
                Clause::Match(MatchClause {
                    patterns: vec![Pattern::single(NodePattern::with_label("n", "Person"))],
                    where_clause: None,
                }),
                Clause::Return(ReturnClause {
                    items: vec![ReturnItem::variable("n")],
                    distinct: false,
                    order_by: None,
                    limit: None,
                    group_by: None,
                }),
            ],
            temporal: None,
            shard_hint: None,
        };
        assert_eq!(format!("{}", query), "MATCH (n:Person) RETURN n");
    }

    #[test]
    fn test_temporal_clause() {
        let temporal = TemporalClause {
            timestamp: Expression::Literal(Literal::String("2026-01-15T12:00:00Z".to_string())),
        };
        let query = Query {
            clauses: vec![
                Clause::Match(MatchClause {
                    patterns: vec![Pattern::single(NodePattern::with_label("n", "Person"))],
                    where_clause: None,
                }),
                Clause::Return(ReturnClause {
                    items: vec![ReturnItem::variable("n")],
                    distinct: false,
                    order_by: None,
                    limit: None,
                    group_by: None,
                }),
            ],
            temporal: Some(temporal),
            shard_hint: None,
        };
        assert!(query.temporal.is_some());
    }

    #[test]
    fn test_shard_hint() {
        let hint = ShardHint { shards: vec![0] };
        assert_eq!(format!("{}", hint), "USING SHARD 0");

        let multi_hint = ShardHint { shards: vec![0, 1, 2] };
        assert_eq!(format!("{}", multi_hint), "USING SHARD [0, 1, 2]");
    }
}
