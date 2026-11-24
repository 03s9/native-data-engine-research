//! SLICE E. JIT cost model + adaptive tier policy (docs/COST_MODEL_FORMULATION.md, RFC-0001 §5). Pure math, no codegen.
use std::sync::atomic::{AtomicU64, AtomicU8, Ordering};

use crate::expr::{BinaryOp, Expr};

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Tier { Vectorized, Jit }

#[derive(Debug, Clone, Copy)]
pub struct CostParams { pub t_vec_tuple_ns: f64, pub t_jit_tuple_ns: f64, pub t_comp_ns: f64 }

// shortcut: weights are uncalibrated research estimates (RFC-0001 §5.2 shape: 2.0 depth, 1.0 node,
// 3.5 branch, 1.5 cast), not measured on any target. Recalibrate per machine before trusting a crossover.
/// Weight per operator level on the longest path (RFC §5.2 `w1`).
pub const W_DEPTH: f64 = 2.0;
/// Plain operator node: arithmetic (Add/Sub/Mul), comparisons, And/Or, Not/Neg/IsNull/IsNotNull.
pub const W_NODE: f64 = 1.0;
/// Div and Mod: scalar-division latency dominates, so they cost more than a plain arithmetic node.
pub const W_DIV_MOD: f64 = 2.5;
/// Cast: node weight 1.0 plus the RFC `w4` = 1.5 layout-conversion penalty.
pub const W_CAST: f64 = 2.5;
/// Case: node weight 1.0 plus the RFC `w3` = 3.5 branch penalty (breaks SIMD lanes).
pub const W_CASE: f64 = 4.5;
/// InList: a membership test is a chain of compares; treated as a single heavy node.
pub const W_IN_LIST: f64 = 2.0;

/// Expression Complexity Index (RFC §5.2). Leaves (columns, literals) add nothing, so
/// `col > 100` scores 3.0. Monotone in expression size: adding any operator never lowers it.
pub fn eci(expr: &Expr) -> f64 {
    W_DEPTH * operator_depth(expr) as f64 + weighted_nodes(expr)
}

/// Longest root-to-leaf path counted in operator nodes only (leaves count 0), the RFC `D_expr`.
fn operator_depth(expr: &Expr) -> usize {
    match expr {
        Expr::Column { .. } | Expr::Literal(_) => 0,
        Expr::Unary { expr, .. } | Expr::Cast { expr, .. } => 1 + operator_depth(expr),
        Expr::Binary { left, right, .. } => 1 + operator_depth(left).max(operator_depth(right)),
        Expr::InList { expr, list, .. } => {
            1 + list.iter().map(operator_depth).fold(operator_depth(expr), usize::max)
        }
        Expr::Case { condition, then_expr, else_expr, .. } => {
            1 + operator_depth(condition).max(operator_depth(then_expr)).max(operator_depth(else_expr))
        }
    }
}

/// Sum of per-node weights over every operator node in the tree.
fn weighted_nodes(expr: &Expr) -> f64 {
    let own = match expr {
        Expr::Column { .. } | Expr::Literal(_) => 0.0,
        Expr::Unary { .. } => W_NODE,
        Expr::Binary { op: BinaryOp::Div | BinaryOp::Mod, .. } => W_DIV_MOD,
        Expr::Binary { .. } => W_NODE,
        Expr::Cast { .. } => W_CAST,
        Expr::Case { .. } => W_CASE,
        Expr::InList { .. } => W_IN_LIST,
    };
    let children: f64 = match expr {
        Expr::Column { .. } | Expr::Literal(_) => 0.0,
        Expr::Unary { expr, .. } | Expr::Cast { expr, .. } => weighted_nodes(expr),
        Expr::Binary { left, right, .. } => weighted_nodes(left) + weighted_nodes(right),
        Expr::InList { expr, list, .. } => {
            weighted_nodes(expr) + list.iter().map(weighted_nodes).sum::<f64>()
        }
        Expr::Case { condition, then_expr, else_expr, .. } => {
            weighted_nodes(condition) + weighted_nodes(then_expr) + weighted_nodes(else_expr)
        }
    };
    own + children
}

/// Crossover cardinality N = ceil(T_comp / Δt), Δt = t_vec - t_jit (RFC §5, COST_MODEL §3).
/// None when JIT can never pay off: Δt <= 0, any non-finite or negative input, or a quotient that overflows.
pub fn crossover(p: &CostParams) -> Option<f64> {
    let finite = [p.t_vec_tuple_ns, p.t_jit_tuple_ns, p.t_comp_ns].iter().all(|v| v.is_finite());
    if !finite || p.t_comp_ns < 0.0 {
        return None;
    }
    let delta = p.t_vec_tuple_ns - p.t_jit_tuple_ns;
    if delta <= 0.0 {
        return None;
    }
    let n = (p.t_comp_ns / delta).ceil();
    n.is_finite().then_some(n)
}

/// Choose the tier for `n_tuples` expected rows. Jit only when a crossover exists and n >= crossover.
pub fn choose_tier(p: &CostParams, n_tuples: u64) -> Tier {
    match crossover(p) {
        Some(n_cross) if n_tuples as f64 >= n_cross => Tier::Jit,
        _ => Tier::Vectorized,
    }
}

/// Early background trigger (COST_MODEL §7.1): N_trigger = min(100 000, 0.10 * N_crossover).
/// Compilation starts here, before the crossover, so compile time hides behind execution.
fn trigger_for(p: &CostParams) -> Option<u64> {
    crossover(p).map(|n_cross| (0.10 * n_cross).min(100_000.0).ceil() as u64)
}

const TIER_VECTORIZED: u8 = 0;
const TIER_JIT: u8 = 1;

/// Adaptive state: starts Vectorized, flips to Jit exactly once when cumulative observed tuples
/// reach N_trigger. Lock-free: one CAS decides the single flip. With no crossover it never flips.
pub struct AdaptiveTier {
    trigger: Option<u64>,
    seen: AtomicU64,
    state: AtomicU8,
}

impl AdaptiveTier {
    pub fn new(p: CostParams) -> Self {
        AdaptiveTier { trigger: trigger_for(&p), seen: AtomicU64::new(0), state: AtomicU8::new(TIER_VECTORIZED) }
    }

    /// Record `tuples` processed. Returns true only for the single call that performed the flip.
    pub fn observe(&self, tuples: u64) -> bool {
        let Some(trigger) = self.trigger else { return false };
        let mut prev = self.seen.load(Ordering::Relaxed);
        let total = loop {
            let next = prev.saturating_add(tuples);
            match self.seen.compare_exchange_weak(prev, next, Ordering::Relaxed, Ordering::Relaxed) {
                Ok(_) => break next,
                Err(actual) => prev = actual,
            }
        };
        if total < trigger {
            return false;
        }
        self.state
            .compare_exchange(TIER_VECTORIZED, TIER_JIT, Ordering::AcqRel, Ordering::Acquire)
            .is_ok()
    }

    pub fn current(&self) -> Tier {
        match self.state.load(Ordering::Acquire) {
            TIER_JIT => Tier::Jit,
            _ => Tier::Vectorized,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{DataType, ScalarValue};
    use std::sync::{Arc, Barrier};
    use std::thread;

    fn params(t_vec: f64, t_jit: f64, t_comp: f64) -> CostParams {
        CostParams { t_vec_tuple_ns: t_vec, t_jit_tuple_ns: t_jit, t_comp_ns: t_comp }
    }

    fn col(name: &str) -> Expr {
        Expr::Column { name: name.into(), index: 0, data_type: DataType::Int64 }
    }

    fn lit(v: i64) -> Expr {
        Expr::Literal(ScalarValue::Int64(v))
    }

    fn bin(left: Expr, op: BinaryOp, right: Expr) -> Expr {
        Expr::Binary { left: Box::new(left), op, right: Box::new(right), return_type: DataType::Boolean }
    }

    #[test]
    fn crossover_matches_spec_cranelift_value() {
        // RFC §5.1: T_comp = 1.2 ms, Δt = 7.4 ns => 162,162.16 -> ceil 162,163.
        let c = crossover(&params(10.0, 2.6, 1.2e6)).unwrap();
        assert_eq!(c, 162_163.0);
    }

    #[test]
    fn crossover_matches_spec_llvm_value() {
        // T_comp = 42 ms, Δt = 7.4 ns => 5,675,675.67 -> ceil 5,675,676.
        let c = crossover(&params(10.0, 2.6, 42.0e6)).unwrap();
        assert_eq!(c, 5_675_676.0);
    }

    #[test]
    fn crossover_exact_integer_quotient_is_not_bumped() {
        assert_eq!(crossover(&params(10.0, 6.0, 100.0)), Some(25.0));
    }

    #[test]
    fn crossover_zero_compile_time_is_zero() {
        assert_eq!(crossover(&params(10.0, 6.0, 0.0)), Some(0.0));
    }

    #[test]
    fn crossover_none_when_delta_is_zero() {
        assert_eq!(crossover(&params(5.0, 5.0, 100.0)), None);
    }

    #[test]
    fn crossover_none_when_delta_is_negative() {
        assert_eq!(crossover(&params(3.0, 5.0, 100.0)), None);
    }

    #[test]
    fn crossover_none_on_nan_inputs() {
        assert_eq!(crossover(&params(f64::NAN, 1.0, 100.0)), None);
        assert_eq!(crossover(&params(5.0, f64::NAN, 100.0)), None);
        assert_eq!(crossover(&params(5.0, 1.0, f64::NAN)), None);
    }

    #[test]
    fn crossover_none_on_infinite_inputs() {
        assert_eq!(crossover(&params(f64::INFINITY, 1.0, 100.0)), None);
        assert_eq!(crossover(&params(5.0, f64::NEG_INFINITY, 100.0)), None);
        assert_eq!(crossover(&params(5.0, 1.0, f64::INFINITY)), None);
    }

    #[test]
    fn crossover_none_on_negative_compile_time() {
        assert_eq!(crossover(&params(5.0, 1.0, -1.0)), None);
    }

    #[test]
    fn crossover_none_when_quotient_overflows() {
        // Positive but subnormal Δt: T_comp / Δt overflows to +inf.
        assert_eq!(crossover(&params(1e-320, 0.0, 1.0)), None);
    }

    #[test]
    fn choose_tier_boundary_at_exact_crossover() {
        let p = params(10.0, 6.0, 100.0); // crossover 25
        assert_eq!(choose_tier(&p, 24), Tier::Vectorized);
        assert_eq!(choose_tier(&p, 25), Tier::Jit);
        assert_eq!(choose_tier(&p, 26), Tier::Jit);
    }

    #[test]
    fn choose_tier_never_jit_without_crossover() {
        assert_eq!(choose_tier(&params(5.0, 5.0, 100.0), u64::MAX), Tier::Vectorized);
        assert_eq!(choose_tier(&params(f64::NAN, 1.0, 100.0), u64::MAX), Tier::Vectorized);
    }

    #[test]
    fn eci_primitive_filter_matches_spec_calibration() {
        // RFC §5.2: `col_a > 100` evaluates to ECI = 3.0 < 4.0.
        let e = bin(col("a"), BinaryOp::Gt, lit(100));
        assert_eq!(eci(&e), 3.0);
    }

    #[test]
    fn eci_leaf_alone_is_zero() {
        assert_eq!(eci(&col("a")), 0.0);
        assert_eq!(eci(&lit(1)), 0.0);
    }

    #[test]
    fn eci_monotone_as_expression_grows() {
        let mut e = col("a");
        let mut prev = eci(&e);
        for i in 0..20 {
            let op = if i % 2 == 0 { BinaryOp::Add } else { BinaryOp::Mul };
            e = bin(e, op, lit(i));
            let now = eci(&e);
            assert!(now >= prev, "eci dropped from {prev} to {now} at step {i}");
            prev = now;
        }
    }

    #[test]
    fn eci_div_heavier_than_add_same_shape() {
        let add = bin(col("a"), BinaryOp::Add, lit(1));
        let div = bin(col("a"), BinaryOp::Div, lit(1));
        assert!(eci(&div) > eci(&add));
    }

    #[test]
    fn eci_case_and_cast_carry_their_spec_penalties() {
        let case = Expr::Case {
            condition: Box::new(bin(col("a"), BinaryOp::Gt, lit(0))),
            then_expr: Box::new(col("a")),
            else_expr: Box::new(lit(0)),
            return_type: DataType::Int64,
        };
        // depth 2 (Case over Gt) -> 4.0; nodes Case 4.5 + Gt 1.0 -> 9.5 total.
        assert_eq!(eci(&case), 9.5);
        let cast = Expr::Cast { expr: Box::new(col("a")), target_type: DataType::Float64 };
        // depth 1 -> 2.0; Cast node 2.5 -> 4.5.
        assert_eq!(eci(&cast), 4.5);
    }

    #[test]
    fn eci_depth_term_penalizes_deeper_tree_of_same_op_count() {
        // Same three Add nodes; right-leaning chain is deeper than a balanced tree.
        let chain = bin(bin(bin(col("a"), BinaryOp::Add, lit(1)), BinaryOp::Add, lit(2)), BinaryOp::Add, lit(3));
        let balanced = bin(
            bin(col("a"), BinaryOp::Add, lit(1)),
            BinaryOp::Add,
            bin(lit(2), BinaryOp::Add, lit(3)),
        );
        assert!(eci(&chain) > eci(&balanced));
    }

    #[test]
    fn eci_in_list_is_counted_as_heavy_node() {
        let e = Expr::InList { expr: Box::new(col("a")), list: vec![lit(1), lit(2)], negated: false };
        assert_eq!(eci(&e), 2.0 * 1.0 + W_IN_LIST);
    }

    #[test]
    fn adaptive_starts_vectorized() {
        let a = AdaptiveTier::new(params(10.0, 6.0, 100.0));
        assert_eq!(a.current(), Tier::Vectorized);
    }

    #[test]
    fn adaptive_trigger_is_min_of_100k_and_tenth_of_crossover() {
        // crossover 5e6 -> 0.1 * 5e6 = 500k, capped at 100k.
        assert_eq!(trigger_for(&params(10.0, 2.6, 42.0e6)), Some(100_000));
        // crossover 200_000 -> 0.1 * 200k = 20k (below the cap).
        assert_eq!(trigger_for(&params(10.0, 6.0, 200_000.0 * 4.0)), Some(20_000));
        assert_eq!(trigger_for(&params(5.0, 5.0, 1.0)), None);
    }

    #[test]
    fn adaptive_flips_at_trigger_not_before() {
        let a = AdaptiveTier::new(params(10.0, 2.6, 42.0e6)); // trigger 100_000
        assert!(!a.observe(99_999));
        assert_eq!(a.current(), Tier::Vectorized);
        assert!(a.observe(1));
        assert_eq!(a.current(), Tier::Jit);
    }

    #[test]
    fn adaptive_observe_reports_flip_only_once_sequentially() {
        let a = AdaptiveTier::new(params(10.0, 2.6, 42.0e6));
        assert!(a.observe(200_000));
        assert!(!a.observe(200_000));
        assert_eq!(a.current(), Tier::Jit);
    }

    #[test]
    fn adaptive_none_crossover_never_flips() {
        let a = AdaptiveTier::new(params(5.0, 5.0, 100.0));
        for _ in 0..10 {
            assert!(!a.observe(u64::MAX));
        }
        assert_eq!(a.current(), Tier::Vectorized);
    }

    #[test]
    fn adaptive_non_finite_params_never_flip() {
        let a = AdaptiveTier::new(params(f64::NAN, 1.0, 100.0));
        assert!(!a.observe(u64::MAX));
        assert_eq!(a.current(), Tier::Vectorized);
    }

    #[test]
    fn adaptive_saturating_counter_does_not_wrap_back_below_trigger() {
        let a = AdaptiveTier::new(params(10.0, 2.6, 42.0e6));
        assert!(a.observe(u64::MAX));
        assert!(!a.observe(u64::MAX));
        assert_eq!(a.current(), Tier::Jit);
    }

    #[test]
    fn adaptive_flips_exactly_once_under_eight_concurrent_observers() {
        const THREADS: usize = 8;
        const CALLS: u64 = 10_000;
        let a = Arc::new(AdaptiveTier::new(params(10.0, 2.6, 42.0e6))); // trigger 100_000
        let barrier = Arc::new(Barrier::new(THREADS));
        let handles: Vec<_> = (0..THREADS)
            .map(|_| {
                let a = Arc::clone(&a);
                let barrier = Arc::clone(&barrier);
                thread::spawn(move || {
                    barrier.wait();
                    (0..CALLS).filter(|_| a.observe(7)).count()
                })
            })
            .collect();
        let transitions: usize = handles.into_iter().map(|h| h.join().unwrap()).sum();
        assert_eq!(transitions, 1, "tier must flip exactly once");
        assert_eq!(a.current(), Tier::Jit);
    }
}
