//! SLICE G. Physical execution of a LogicalPlan against a Catalog.
use crate::catalog::Catalog;
use crate::column::Batch;
use crate::memory::MemoryBudget;
use crate::plan::LogicalPlan;
use crate::types::{EngineError, Result};
use std::path::PathBuf;
use std::sync::Arc;

pub struct ExecContext { pub threads: usize, pub morsel_rows: usize, pub budget: Arc<MemoryBudget>, pub spill_dir: PathBuf }

pub fn execute(_plan: &LogicalPlan, _catalog: &Catalog, _ctx: &ExecContext) -> Result<Vec<Batch>> { Err(EngineError::Unimplemented("exec::execute")) }
