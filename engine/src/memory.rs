//! SLICE D1. Hierarchical memory budget (spec §6.1). Thread-safe, lock-free accounting.
use crate::types::{EngineError, Result};
use std::sync::Arc;

#[derive(Debug)]
pub struct MemoryBudget { _private: () }

/// RAII reservation: bytes returned to the budget (and parent chain) on drop.
#[derive(Debug)]
pub struct Reservation { _private: () }

impl MemoryBudget {
    pub fn new(_limit_bytes: usize) -> Arc<MemoryBudget> { unimplemented!("memory::new") }
    /// Child budget: its reservations also count against the parent; `limit_bytes` caps the child.
    pub fn child(self: &Arc<Self>, _limit_bytes: usize) -> Arc<MemoryBudget> { unimplemented!("memory::child") }
    /// Err(EngineError::Memory{..}) when the reservation would exceed this or any ancestor limit. Never partial.
    pub fn try_reserve(self: &Arc<Self>, _bytes: usize) -> Result<Reservation> { Err(EngineError::Unimplemented("memory::try_reserve")) }
    pub fn used(&self) -> usize { 0 }
    pub fn limit(&self) -> usize { 0 }
    pub fn available(&self) -> usize { 0 }
}

impl Reservation {
    pub fn size(&self) -> usize { 0 }
    /// Grow in place; Err leaves the reservation unchanged.
    pub fn grow(&mut self, _more: usize) -> Result<()> { Err(EngineError::Unimplemented("memory::grow")) }
}
