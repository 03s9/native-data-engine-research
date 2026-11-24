//! SLICE D1. Sort with optional external merge when the budget is exceeded.
use crate::column::Batch;
use crate::memory::MemoryBudget;
use crate::plan::SortField;
use crate::types::{EngineError, Result, SchemaRef};
use std::path::Path;
use std::sync::Arc;

/// Stable sort of all `input` rows by `order_by` (nulls placement per field). Output: batches of <= `batch_rows` rows.
/// If buffering exceeds `budget`, sort runs, spill them to `spill_dir`, k-way merge. Result identical either way.
pub fn sort_batches(
    _input: Vec<Batch>, _schema: SchemaRef, _order_by: &[SortField],
    _budget: &Arc<MemoryBudget>, _spill_dir: &Path, _batch_rows: usize,
) -> Result<Vec<Batch>> {
    Err(EngineError::Unimplemented("sort::sort_batches"))
}
