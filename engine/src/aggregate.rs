//! SLICE D2. Hash aggregation (partial per-batch + merge), group keys incl. NULL as its own group.
use crate::column::Batch;
use crate::memory::MemoryBudget;
use crate::plan::AggExpr;
use crate::expr::Expr;
use crate::types::{EngineError, Result, SchemaRef};
use std::path::Path;
use std::sync::Arc;

/// No group exprs => exactly one output row (even for empty input: Count=0, others NULL).
/// Group order in output is deterministic: first-seen order of keys.
pub fn hash_aggregate(
    _input: Vec<Batch>, _group_exprs: &[Expr], _aggs: &[AggExpr], _schema: SchemaRef,
    _budget: &Arc<MemoryBudget>, _spill_dir: &Path,
) -> Result<Vec<Batch>> {
    Err(EngineError::Unimplemented("aggregate::hash_aggregate"))
}
