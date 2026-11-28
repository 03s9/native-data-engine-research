//! Native analytical engine MVP (vectorized, morsel-parallel, spill-capable).
//! Module list is FROZEN by the skeleton owner; slice agents edit only their own files.
pub mod aggregate;
pub mod catalog;
pub mod column;
pub mod cost;
pub mod eval;
pub mod exec;
pub mod expr;
pub mod jit;
pub mod join;
pub mod memory;
pub mod plan;
pub mod sched;
pub mod simd;
pub mod sort;
pub mod spill;
pub mod types;
