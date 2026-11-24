//! SLICE E. JIT cost model + adaptive tier policy (docs/COST_MODEL_FORMULATION.md). Pure math, no codegen.
use crate::expr::Expr;

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Tier { Vectorized, Jit }

#[derive(Debug, Clone, Copy)]
pub struct CostParams { pub t_vec_tuple_ns: f64, pub t_jit_tuple_ns: f64, pub t_comp_ns: f64 }

/// Expression Complexity Index (RFC §5.2): a function of node_count/depth/op mix. Monotone in expression size.
pub fn eci(_expr: &Expr) -> f64 { 0.0 }
/// Δt = t_vec - t_jit; Some(T_comp / Δt) when Δt > 0, None when JIT can never pay off.
pub fn crossover(_p: &CostParams) -> Option<f64> { None }
/// Choose tier for `n_tuples` expected rows. Never Jit when crossover is None or n_tuples < crossover.
pub fn choose_tier(_p: &CostParams, _n_tuples: u64) -> Tier { Tier::Vectorized }
/// Adaptive state: starts Vectorized, flips to Jit (atomically, once) after `observe` sees tuples >= trigger.
pub struct AdaptiveTier { _private: () }
impl AdaptiveTier {
    pub fn new(_p: CostParams) -> Self { AdaptiveTier { _private: () } }
    pub fn observe(&self, _tuples: u64) {}
    pub fn current(&self) -> Tier { Tier::Vectorized }
}
