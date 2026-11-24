//! SLICE G. In-memory tables + CSV loader.
use crate::column::Batch;
use crate::types::{EngineError, Result, SchemaRef};
use std::collections::HashMap;
use std::path::Path;

#[derive(Default)]
pub struct Catalog { pub tables: HashMap<String, (SchemaRef, Vec<Batch>)> }

impl Catalog {
    pub fn new() -> Self { Self::default() }
    pub fn register(&mut self, name: &str, schema: SchemaRef, batches: Vec<Batch>) { self.tables.insert(name.to_string(), (schema, batches)); }
    pub fn get(&self, name: &str) -> Result<&(SchemaRef, Vec<Batch>)> { self.tables.get(name).ok_or_else(|| EngineError::Plan(format!("table '{name}' not found"))) }
}

/// CSV with header row; column types inferred (Int64 > Float64 > Boolean > Utf8); empty field => NULL.
/// Supports quoted fields with commas/escaped quotes. Batches of `batch_rows` rows.
pub fn load_csv(_path: &Path, _batch_rows: usize) -> Result<(SchemaRef, Vec<Batch>)> { Err(EngineError::Unimplemented("catalog::load_csv")) }
