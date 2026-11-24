//! SLICE D1. Deterministic on-disk batch spill. Binary format is private; roundtrip must be bit-exact
//! (incl. nulls, long strings, f64 NaN payloads). Files are deleted when SpillFile drops.
use crate::column::Batch;
use crate::types::{EngineError, Result, SchemaRef};
use std::path::{Path, PathBuf};

pub struct SpillWriter { _private: () }
#[derive(Debug)]
pub struct SpillFile { _private: () }

impl SpillWriter {
    /// Creates a uniquely named file inside `dir` (dir must exist).
    pub fn create(_dir: &Path, _schema: SchemaRef) -> Result<SpillWriter> { Err(EngineError::Unimplemented("spill::create")) }
    pub fn write(&mut self, _batch: &Batch) -> Result<()> { Err(EngineError::Unimplemented("spill::write")) }
    pub fn finish(self) -> Result<SpillFile> { Err(EngineError::Unimplemented("spill::finish")) }
}
impl SpillFile {
    pub fn path(&self) -> PathBuf { PathBuf::new() }
    pub fn num_rows(&self) -> usize { 0 }
    pub fn bytes_on_disk(&self) -> u64 { 0 }
    /// Read all batches back in write order.
    pub fn read(&self) -> Result<Vec<Batch>> { Err(EngineError::Unimplemented("spill::read")) }
}
