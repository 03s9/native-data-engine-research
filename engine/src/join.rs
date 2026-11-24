//! SLICE D2. Hash join (build right, probe left); Grace partition+spill when build side exceeds budget.
use crate::column::Batch;
use crate::expr::Expr;
use crate::memory::MemoryBudget;
use crate::plan::JoinType;
use crate::types::{EngineError, Result, SchemaRef};
use std::path::Path;
use std::sync::Arc;

/// NULL keys never match. Duplicate keys: cross-product of matches. Skewed/duplicate-heavy partitions must
/// terminate (fall back to chunked nested loop when re-partitioning makes no progress).
pub fn hash_join(
    _left: Vec<Batch>, _right: Vec<Batch>, _join_type: JoinType, _on: &[(Expr, Expr)], _schema: SchemaRef,
    _budget: &Arc<MemoryBudget>, _spill_dir: &Path,
) -> Result<Vec<Batch>> {
    Err(EngineError::Unimplemented("join::hash_join"))
}
