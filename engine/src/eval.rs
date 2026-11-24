//! SLICE A. Expression typing + vectorized evaluation + constant folding.
//! Semantics: SQL three-valued logic; Int64/Float64 mixed arithmetic promotes to Float64;
//! integer division/mod by zero => EngineError::Arithmetic; comparisons on Utf8 are byte-lexicographic.
use crate::column::{Batch, Column};
use crate::expr::{BinaryOp, Expr, UnaryOp};
use crate::types::{DataType, EngineError, Result, ScalarValue, Schema};

/// Build `Expr::Column` by name (resolves index + type from `schema`).
pub fn col(_schema: &Schema, _name: &str) -> Result<Expr> {
    Err(EngineError::Unimplemented("eval::col"))
}
pub fn lit(v: ScalarValue) -> Expr {
    Expr::Literal(v)
}
/// Type-checked constructor: infers `return_type` or returns EngineError::Type.
pub fn binary(_l: Expr, _op: BinaryOp, _r: Expr) -> Result<Expr> {
    Err(EngineError::Unimplemented("eval::binary"))
}
pub fn unary(_op: UnaryOp, _e: Expr) -> Result<Expr> {
    Err(EngineError::Unimplemented("eval::unary"))
}
pub fn cast(_e: Expr, _to: DataType) -> Result<Expr> {
    Err(EngineError::Unimplemented("eval::cast"))
}
pub fn case(_c: Expr, _t: Expr, _e: Expr) -> Result<Expr> {
    Err(EngineError::Unimplemented("eval::case"))
}
/// Evaluate over every row of `batch`; result length == batch.num_rows.
pub fn eval(_expr: &Expr, _batch: &Batch) -> Result<Column> {
    Err(EngineError::Unimplemented("eval::eval"))
}
/// Boolean predicate -> row mask; NULL counts as false (SQL WHERE semantics).
pub fn eval_predicate(_expr: &Expr, _batch: &Batch) -> Result<Vec<bool>> {
    Err(EngineError::Unimplemented("eval::eval_predicate"))
}
/// Constant folding / algebraic simplification (x+0, x*1, true AND x, literal-only subtrees).
pub fn optimize(expr: Expr) -> Expr {
    expr
}
