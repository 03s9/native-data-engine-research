//! Logical plan DAG (spec §2.3). FROZEN.
use crate::expr::Expr;
use crate::types::{DataType, SchemaRef};
use std::sync::Arc;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum JoinType {
    Inner,
    Left,
    Right,
    Full,
    Semi,
    Anti,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AggFunc {
    Count,
    Sum,
    Min,
    Max,
    Avg,
}

/// `expr == None` only for Count (COUNT(*)). Output types: Count->Int64,
/// Sum/Min/Max -> input type (Sum of Int64 stays Int64), Avg->Float64. Nulls ignored; empty group Sum/Min/Max/Avg -> NULL.
#[derive(Debug, Clone)]
pub struct AggExpr {
    pub func: AggFunc,
    pub expr: Option<Expr>,
    pub output_type: DataType,
}

#[derive(Debug, Clone)]
pub struct SortField {
    pub expr: Expr,
    pub ascending: bool,
    pub nulls_first: bool,
}

#[derive(Debug, Clone)]
pub enum LogicalPlan {
    Scan {
        table_name: String,
        schema: SchemaRef,
        projection: Option<Vec<usize>>,
        projected_schema: SchemaRef,
    },
    Filter {
        input: Arc<LogicalPlan>,
        predicate: Expr,
    },
    Projection {
        input: Arc<LogicalPlan>,
        exprs: Vec<Expr>,
        schema: SchemaRef,
    },
    /// Equi-join: `on` pairs are (left_expr, right_expr). Output schema = left fields ++ right fields
    /// (Semi/Anti: left fields only).
    Join {
        left: Arc<LogicalPlan>,
        right: Arc<LogicalPlan>,
        join_type: JoinType,
        on: Vec<(Expr, Expr)>,
        schema: SchemaRef,
    },
    /// Output schema = group exprs then aggregates, in order.
    Aggregate {
        input: Arc<LogicalPlan>,
        group_exprs: Vec<Expr>,
        agg_exprs: Vec<AggExpr>,
        schema: SchemaRef,
    },
    Sort {
        input: Arc<LogicalPlan>,
        order_by: Vec<SortField>,
    },
    Limit {
        input: Arc<LogicalPlan>,
        skip: usize,
        fetch: Option<usize>,
    },
}

impl LogicalPlan {
    pub fn schema(&self) -> SchemaRef {
        match self {
            LogicalPlan::Scan {
                projected_schema, ..
            } => projected_schema.clone(),
            LogicalPlan::Filter { input, .. }
            | LogicalPlan::Sort { input, .. }
            | LogicalPlan::Limit { input, .. } => input.schema(),
            LogicalPlan::Projection { schema, .. }
            | LogicalPlan::Join { schema, .. }
            | LogicalPlan::Aggregate { schema, .. } => schema.clone(),
        }
    }
}
