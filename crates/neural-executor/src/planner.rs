//! Query planner.
//!
//! Translates AST into physical execution plans.

use crate::plan::{PhysicalPlan, ProjectColumn, ProjectExpression};
use crate::{ExecutionError, Result};
use neural_parser::{
    Clause, Expression, MatchClause, PatternMode, Query, ReturnClause, UnwindClause, WithClause,
    WhereClause, OrderByClause
};

/// Query planner that converts AST to physical plans.
#[derive(Debug, Default)]
pub struct Planner;

impl Planner {
    /// Creates a new planner.
    pub fn new() -> Self {
        Self
    }

    /// Plans a query, producing a physical execution plan.
    pub fn plan(&self, query: &Query) -> Result<PhysicalPlan> {
        // Check for COUNT optimization opportunities first
        if let Some(count_plan) = self.try_plan_count_optimization(query) {
            return Ok(count_plan);
        }

        let mut steps = Vec::new();

        for clause in &query.clauses {
            match clause {
                Clause::Match(c) => {
                    let plan = self.plan_match(c)?;
                    steps.push(plan);
                }
                Clause::OptionalMatch(c) => {
                    let plan = self.plan_optional_match(c)?;
                    steps.push(plan);
                }
                Clause::Merge(c) => {
                    steps.push(PhysicalPlan::Merge { pattern: c.pattern.clone() });
                }
                Clause::With(c) => {
                    let plan = self.plan_with(c)?;
                    steps.push(plan);
                }
                Clause::Unwind(c) => {
                    let plan = self.plan_unwind(c)?;
                    steps.push(plan);
                }
                Clause::Return(c) => {
                    let plan = self.plan_return(c)?;
                    steps.push(plan);
                }
            }
        }

        if steps.len() == 1 {
            Ok(steps.pop().unwrap())
        } else {
            Ok(PhysicalPlan::Sequence { steps })
        }
    }

    /// Try to detect COUNT optimization patterns.
    /// Returns Some(plan) if optimizable, None otherwise.
    ///
    /// Optimizes:
    /// - MATCH (n) RETURN count(n) → O(1) node count
    /// - MATCH (n:Label) RETURN count(n) → O(1) label count
    /// - MATCH (n) RETURN count(*) → O(1) node count
    /// - MATCH ()-[r]->() RETURN count(r) → O(1) edge count
    /// - MATCH ()-[r:TYPE]->() RETURN count(r) → O(1) edge type count
    fn try_plan_count_optimization(&self, query: &Query) -> Option<PhysicalPlan> {
        // Must be exactly: MATCH ... RETURN count(...)
        if query.clauses.len() != 2 {
            return None;
        }

        let match_clause = match &query.clauses[0] {
            Clause::Match(c) => c,
            _ => return None,
        };

        let return_clause = match &query.clauses[1] {
            Clause::Return(c) => c,
            _ => return None,
        };

        // Must have no WHERE clause
        if match_clause.where_clause.is_some() {
            return None;
        }

        // Must have exactly one return item that is COUNT
        if return_clause.items.len() != 1 {
            return None;
        }

        let return_item = &return_clause.items[0];
        let (count_var, is_star) = match &return_item.expression {
            Expression::Aggregate { function, argument, distinct: false } => {
                if !matches!(function, neural_parser::AggregateFunction::Count) {
                    return None;
                }
                match argument {
                    Some(arg) => {
                        // count(n) - extract variable name
                        if let Expression::Property { variable, property } = arg.as_ref() {
                            if property.is_empty() {
                                (Some(variable.clone()), false)
                            } else {
                                return None;
                            }
                        } else {
                            return None;
                        }
                    }
                    None => (None, true), // count(*)
                }
            }
            _ => return None,
        };

        // Must have exactly one pattern
        if match_clause.patterns.len() != 1 {
            return None;
        }

        let pattern = &match_clause.patterns[0];
        let alias = return_item.alias.clone().unwrap_or_else(|| format!("{}", return_item.expression));

        // Check for node count pattern: (n) or (n:Label)
        if pattern.chain.is_empty() {
            // Simple node pattern
            let node_var = pattern.start.identifier.as_ref()?;
            let label = pattern.start.label.clone();

            // Variable must match count variable or be count(*)
            if is_star || count_var.as_ref() == Some(node_var) {
                return Some(PhysicalPlan::CountNodes { label, alias });
            }
        }

        // Check for edge count pattern: ()-[r]->() or ()-[r:TYPE]->()
        if pattern.chain.len() == 1 {
            let (rel, _end_node) = &pattern.chain[0];

            // The relationship must have an identifier that matches the count variable
            if let Some(rel_var) = &rel.identifier {
                if is_star || count_var.as_ref() == Some(rel_var) {
                    let edge_type = rel.label.clone();
                    return Some(PhysicalPlan::CountEdges { edge_type, alias });
                }
            }
        }

        None
    }

    /// Plans a MATCH clause.
    pub fn plan_match(&self, match_clause: &MatchClause) -> Result<PhysicalPlan> {
        self.plan_match_internal(match_clause, false)
    }

    /// Plans an OPTIONAL MATCH clause.
    pub fn plan_optional_match(&self, match_clause: &MatchClause) -> Result<PhysicalPlan> {
        self.plan_match_internal(match_clause, true)
    }

    fn plan_match_internal(&self, match_clause: &MatchClause, optional: bool) -> Result<PhysicalPlan> {
        if match_clause.patterns.is_empty() {
            return Err(ExecutionError::PlanningError(
                "MATCH clause requires at least one pattern".into(),
            ));
        }

        let mut steps = Vec::new();

        for pattern in &match_clause.patterns {
            let start_binding = pattern
                .start
                .identifier
                .clone()
                .unwrap_or_else(|| "_start".into());

            // Check if there's a vector_similarity predicate on this binding in the WHERE clause
            let mut vector_scan = None;
            if let Some(ref where_clause) = match_clause.where_clause {
                if let Expression::Comparison { left, op, right: _ } = &where_clause.expression {
                    // Check for vector_similarity(n.prop, query) > threshold
                    if matches!(op, neural_parser::ComparisonOp::Gt | neural_parser::ComparisonOp::Gte) {
                        if let Expression::VectorSimilarity { property, query } = left.as_ref() {
                            if let Expression::Property { variable, .. } = property.as_ref() {
                                if variable == &start_binding {
                                    // Found a vector search candidate!
                                    // Extract K (limit) if available, otherwise default
                                    // Ideally, we should look at LIMIT clause, but it's not available here.
                                    // For now, hardcode or guess. The LIMIT pushdown happens later or we need to pass it.
                                    // Actually, we can just scan top-K here. Let's assume K=10 for now if not specified.
                                    
                                    // Note: This is a simplified planner optimization. 
                                    // Real optimizer would pass LIMIT down.
                                    
                                    // We need to pass the query vector expression and K
                                    vector_scan = Some(PhysicalPlan::VectorSearch {
                                        binding: start_binding.clone(),
                                        query: query.as_ref().clone(),
                                        label: pattern.start.label.clone(),
                                        k: 10, // Default K, optimizer should update this
                                    });
                                }
                            }
                        }
                    }
                }
            }

            // Check for id(binding) = N predicates for O(1) node lookup
            let id_lookup = self.extract_id_predicate(&start_binding, match_clause.where_clause.as_ref());

            let scan_plan = if let Some(plan) = vector_scan {
                plan
            } else if let Some(node_id) = id_lookup {
                // O(1) direct node lookup
                PhysicalPlan::ScanNodeById {
                    binding: start_binding.clone(),
                    node_id,
                }
            } else if let Some(ref label) = pattern.start.label {
                PhysicalPlan::ScanByLabel {
                    binding: start_binding.clone(),
                    label: label.clone(),
                }
            } else {
                PhysicalPlan::ScanAllNodes {
                    binding: start_binding.clone(),
                }
            };
            
            steps.push(scan_plan);

            let mut current_binding = start_binding;

            for (rel, node) in &pattern.chain {
                let next_binding = node
                    .identifier
                    .clone()
                    .unwrap_or_else(|| format!("_node{}", steps.len()));

                let expand_plan = if pattern.mode == PatternMode::ShortestPath {
                    let max_hops = if let Some(ref vl) = rel.var_length {
                        vl.max.unwrap_or(15)
                    } else {
                        1
                    };

                    PhysicalPlan::ExpandShortestPath {
                        from_binding: current_binding,
                        to_binding: next_binding.clone(),
                        path_binding: pattern.identifier.clone(),
                        edge_type: rel.label.clone(),
                        direction: rel.direction,
                        max_hops,
                    }
                } else if let Some(ref var_len) = rel.var_length {
                    PhysicalPlan::ExpandVariableLength {
                        from_binding: current_binding,
                        to_binding: next_binding.clone(),
                        path_binding: pattern.identifier.clone(),
                        edge_type: rel.label.clone(),
                        direction: rel.direction,
                        min_hops: var_len.min,
                        max_hops: var_len.max,
                    }
                } else if let Some(ref edge_type) = rel.label {
                    if optional {
                         PhysicalPlan::OptionalExpandByType {
                            from_binding: current_binding,
                            to_binding: next_binding.clone(),
                            edge_binding: rel.identifier.clone(),
                            edge_type: edge_type.clone(),
                            direction: rel.direction,
                        }
                    } else {
                        PhysicalPlan::ExpandByType {
                            from_binding: current_binding,
                            to_binding: next_binding.clone(),
                            edge_binding: rel.identifier.clone(),
                            edge_type: edge_type.clone(),
                            direction: rel.direction,
                        }
                    }
                } else if optional {
                     PhysicalPlan::OptionalExpandNeighbors {
                        from_binding: current_binding,
                        to_binding: next_binding.clone(),
                        edge_binding: rel.identifier.clone(),
                        rel_label: None,
                        direction: rel.direction,
                    }
                } else {
                    PhysicalPlan::ExpandNeighbors {
                        from_binding: current_binding,
                        to_binding: next_binding.clone(),
                        edge_binding: rel.identifier.clone(),
                        rel_label: None,
                        direction: rel.direction,
                    }
                };

                steps.push(expand_plan);
                current_binding = next_binding;
            }
        }

        if let Some(ref where_clause) = match_clause.where_clause {
            steps.push(self.plan_where(where_clause)?);
        }

        if steps.len() == 1 {
            Ok(steps.pop().unwrap())
        } else {
            Ok(PhysicalPlan::Sequence { steps })
        }
    }

    /// Plans a WHERE clause.
    pub fn plan_where(&self, where_clause: &WhereClause) -> Result<PhysicalPlan> {
        Ok(PhysicalPlan::Filter {
            predicate: where_clause.expression.clone(),
        })
    }

    /// Plans a WITH clause.
    pub fn plan_with(&self, with_clause: &WithClause) -> Result<PhysicalPlan> {
        self.plan_projection(
            &with_clause.items,
            with_clause.distinct,
            with_clause.order_by.as_ref(),
            with_clause.limit,
            with_clause.where_clause.as_ref(),
            None 
        )
    }

    /// Plans a RETURN clause.
    pub fn plan_return(&self, return_clause: &ReturnClause) -> Result<PhysicalPlan> {
        self.plan_projection(
            &return_clause.items,
            return_clause.distinct,
            return_clause.order_by.as_ref(),
            return_clause.limit,
            None, 
            return_clause.group_by.as_ref()
        )
    }

    /// Plans a projection (Project or Aggregate).
    pub fn plan_projection(
        &self,
        items: &[neural_parser::ReturnItem],
        distinct: bool,
        order_by: Option<&OrderByClause>,
        limit: Option<u64>,
        where_clause: Option<&WhereClause>,
        group_by: Option<&neural_parser::GroupByClause>,
    ) -> Result<PhysicalPlan> {
        let mut steps = Vec::new();

        let has_aggregates = items.iter().any(|item| self.expression_is_aggregate(&item.expression));
        
        if has_aggregates || group_by.is_some() {
            let mut group_keys = Vec::new();
            let mut aggregations = Vec::new();
            
            if let Some(gb) = group_by {
                group_keys = gb.expressions.clone();
            } else {
                for item in items {
                    if !self.expression_is_aggregate(&item.expression) {
                        group_keys.push(item.expression.clone());
                    }
                }
            }
            
            for item in items {
                if let Expression::Aggregate { function, argument, distinct } = &item.expression {
                     let alias = item.alias.clone().unwrap_or_else(|| format!("{}", item.expression));
                     aggregations.push((*function, argument.as_deref().cloned(), *distinct, alias));
                }
            }
            
            steps.push(PhysicalPlan::Aggregate {
                group_by: group_keys,
                aggregations,
            });
            
        } else {
            let columns = self.extract_columns(items)?;
            steps.push(PhysicalPlan::Project { columns });
        }

        if distinct {
            steps.push(PhysicalPlan::Distinct);
        }

        if let Some(where_clause) = where_clause {
             steps.push(self.plan_where(where_clause)?);
        }

        if let Some(order_by) = order_by {
             steps.push(self.plan_order_by(order_by)?);
        }

        if let Some(limit) = limit {
            steps.push(PhysicalPlan::Limit { count: limit });
        }

        if steps.len() == 1 {
            Ok(steps.pop().unwrap())
        } else {
            Ok(PhysicalPlan::Sequence { steps })
        }
    }

    /// Plans an UNWIND clause.
    pub fn plan_unwind(&self, unwind_clause: &UnwindClause) -> Result<PhysicalPlan> {
        Ok(PhysicalPlan::Unwind {
            expression: unwind_clause.expression.clone(),
            alias: unwind_clause.alias.clone(),
        })
    }

    /// Plans an ORDER BY clause.
    pub fn plan_order_by(&self, order_by: &OrderByClause) -> Result<PhysicalPlan> {
        Ok(PhysicalPlan::OrderBy {
            items: order_by
                .items
                .iter()
                .map(|item| {
                    (
                        item.expression.clone(),
                        matches!(item.direction, neural_parser::SortDirection::Descending),
                    )
                })
                .collect(),
        })
    }

    fn extract_columns(&self, items: &[neural_parser::ReturnItem]) -> Result<Vec<ProjectColumn>> {
        let columns: Vec<ProjectColumn> = items
            .iter()
            .map(|item| {
                match &item.expression {
                    Expression::Property { variable, property } => {
                        if property.is_empty() {
                            ProjectColumn {
                                expression: ProjectExpression::Variable(variable.clone()),
                                alias: item.alias.clone(),
                            }
                        } else {
                            ProjectColumn {
                                expression: ProjectExpression::Property {
                                    variable: variable.clone(),
                                    property: property.clone(),
                                },
                                alias: item.alias.clone(),
                            }
                        }
                    }
                    _ => {
                        ProjectColumn {
                            expression: ProjectExpression::Expression(item.expression.clone()),
                            alias: item.alias.clone(),
                        }
                    }
                }
            })
            .collect();
        Ok(columns)
    }

    /// Extracts node ID from WHERE clause predicates like `id(binding) = N`
    /// Returns Some(node_id) if found, None otherwise.
    fn extract_id_predicate(&self, binding: &str, where_clause: Option<&WhereClause>) -> Option<u64> {
        let where_clause = where_clause?;
        self.extract_id_from_expression(binding, &where_clause.expression)
    }

    fn extract_id_from_expression(&self, binding: &str, expr: &Expression) -> Option<u64> {
        match expr {
            // id(x) = N or N = id(x)
            Expression::Comparison { left, op, right } if matches!(op, neural_parser::ComparisonOp::Eq) => {
                // Check left = id(binding), right = literal
                if let Some(id) = self.extract_id_function_value(binding, left, right) {
                    return Some(id);
                }
                // Check right = id(binding), left = literal
                if let Some(id) = self.extract_id_function_value(binding, right, left) {
                    return Some(id);
                }
                None
            }
            // Handle AND - check both sides
            Expression::And(left, right) => {
                self.extract_id_from_expression(binding, left)
                    .or_else(|| self.extract_id_from_expression(binding, right))
            }
            _ => None,
        }
    }

    fn extract_id_function_value(&self, binding: &str, id_expr: &Expression, value_expr: &Expression) -> Option<u64> {
        // Check if id_expr is FunctionCall("id", [Property { variable: binding, property: "" }])
        if let Expression::FunctionCall { name, args } = id_expr {
            if name.eq_ignore_ascii_case("id") && args.len() == 1 {
                // Variable is represented as Property with empty property string
                if let Expression::Property { variable, property } = &args[0] {
                    if variable == binding && property.is_empty() {
                        // Check if value_expr is a literal integer
                        if let Expression::Literal(neural_parser::Literal::Int(n)) = value_expr {
                            return Some(*n as u64);
                        }
                    }
                }
            }
        }
        None
    }

    fn expression_is_aggregate(&self, expr: &Expression) -> bool {
        matches!(expr, Expression::Aggregate { .. })
    }

    #[allow(dead_code)]
    fn expression_contains_aggregate(expr: &Expression) -> bool {
         match expr {
            Expression::Aggregate { .. } => true,
            Expression::Comparison { left, right, .. } => {
                Self::expression_contains_aggregate(left) || Self::expression_contains_aggregate(right)
            }
            Expression::And(left, right) | Expression::Or(left, right) => {
                Self::expression_contains_aggregate(left) || Self::expression_contains_aggregate(right)
            }
            Expression::Not(expr) => Self::expression_contains_aggregate(expr),
            _ => false,
        }
    }
}
