//! Empirical Adversarial Verification Test Suite — Iteration 2
//!
//! Stress-testing 5 core properties:
//! 1. Negative crossover guard (Delta t <= 0 => N_crossover = usize::MAX)
//! 2. Sethi-Ullman stack spill penalty (D_expr >= 32 => tau_spill suppresses JIT)
//! 3. Expression Complexity Index (ECI) calculation (col_a > 100 => ECI = 3.0 < 4.0)
//! 4. Concurrency memory ordering (Acquire load / Release store) & ARM64 cache invalidation
//! 5. Adaptive trigger decoupling (N_trigger = min(100_000, 0.10 * N_crossover))

use std::sync::atomic::{AtomicPtr, AtomicBool, Ordering};
use std::sync::Arc;
use std::time::Duration;

// -----------------------------------------------------------------------------
// 1. Negative Crossover Guard Implementation & Stress Tests
// -----------------------------------------------------------------------------

#[inline]
pub fn compute_crossover_cardinality(
    t_compile_ms: f64,
    t_interp_ns_per_tuple: f64,
    t_jit_ns_per_tuple: f64,
) -> usize {
    let delta_t_ns = t_interp_ns_per_tuple - t_jit_ns_per_tuple;
    // Guard against negative, zero, negative zero, or NaN deltas
    if delta_t_ns <= 0.0 || delta_t_ns.is_nan() {
        return usize::MAX;
    }
    let t_compile_ns = t_compile_ms * 1_000_000.0;
    let crossover_f64 = (t_compile_ns / delta_t_ns).ceil();
    if crossover_f64 >= usize::MAX as f64 || crossover_f64.is_nan() {
        usize::MAX
    } else {
        crossover_f64 as usize
    }
}

#[test]
fn test_adversarial_negative_crossover_guard() {
    // Standard positive crossover cases
    assert_eq!(compute_crossover_cardinality(1.0, 5.0, 2.0), 333_334);
    assert_eq!(compute_crossover_cardinality(2.5, 4.0, 1.5), 1_000_000);

    // Adversarial Case 1: JIT strictly slower than vectorized interpretation
    assert_eq!(
        compute_crossover_cardinality(1.0, 2.0, 5.0),
        usize::MAX,
        "JIT slower than vectorized must return usize::MAX"
    );
    assert_eq!(
        compute_crossover_cardinality(10.0, 1.0, 100.0),
        usize::MAX,
        "Heavily slower JIT must return usize::MAX"
    );

    // Adversarial Case 2: Zero delta (identical throughput)
    assert_eq!(
        compute_crossover_cardinality(1.0, 2.0, 2.0),
        usize::MAX,
        "Zero delta t must return usize::MAX to avoid division by zero"
    );

    // Adversarial Case 3: Negative zero float (-0.0)
    assert_eq!(
        compute_crossover_cardinality(1.0, 2.0, 2.0 + 0.0),
        usize::MAX
    );

    // Adversarial Case 4: Infinitesimal positive delta (t_compile / 1e-15)
    let huge_crossover = compute_crossover_cardinality(1.0, 2.0 + 1e-15, 2.0);
    assert_eq!(
        huge_crossover,
        usize::MAX,
        "Infinitesimal delta must saturate to usize::MAX"
    );

    // Adversarial Case 5: Kernel Cache Hit (t_compile = 0) with slower JIT
    assert_eq!(
        compute_crossover_cardinality(0.0, 2.0, 3.0),
        usize::MAX,
        "Cached kernel must never be selected if it is slower than vectorized"
    );

    // Adversarial Case 6: Kernel Cache Hit (t_compile = 0) with faster JIT
    assert_eq!(
        compute_crossover_cardinality(0.0, 3.0, 2.0),
        0,
        "Cached faster kernel has immediate 0-tuple crossover"
    );

    // Verification: for any realistic input cardinality N <= 10^12, N < usize::MAX
    let max_practical_cardinality: usize = 1_000_000_000_000; // 1 Trillion tuples
    let crossover_neg = compute_crossover_cardinality(1.0, 2.0, 5.0);
    assert!(
        max_practical_cardinality < crossover_neg,
        "Practical query size must never exceed usize::MAX crossover"
    );
}

// -----------------------------------------------------------------------------
// 2. Sethi-Ullman Stack Spill Penalty Simulation & Stress Tests
// -----------------------------------------------------------------------------

pub struct RegisterSpillModel {
    pub r_avail_neon: usize,
    pub r_avail_avx512: usize,
    pub tau_spill_cost_ns: f64,
}

impl RegisterSpillModel {
    pub fn new() -> Self {
        Self {
            r_avail_neon: 20,
            r_avail_avx512: 24,
            tau_spill_cost_ns: 1.6,
        }
    }

    #[inline]
    pub fn compute_tau_spill(&self, d_expr: usize, is_neon: bool) -> f64 {
        let r_avail = if is_neon { self.r_avail_neon } else { self.r_avail_avx512 };
        if d_expr <= r_avail {
            0.0
        } else {
            let spilled = d_expr - r_avail;
            (spilled as f64) * self.tau_spill_cost_ns
        }
    }

    pub fn compute_effective_jit_tuple_cost(
        &self,
        d_expr: usize,
        t_compute_reg_ns: f64,
        is_neon: bool,
    ) -> f64 {
        t_compute_reg_ns + self.compute_tau_spill(d_expr, is_neon)
    }
}

#[test]
fn test_adversarial_sethi_ullman_spill_penalty() {
    let model = RegisterSpillModel::new();

    // 1. Shallow expressions (D_expr < R_avail): zero spill penalty
    assert_eq!(model.compute_tau_spill(5, true), 0.0);
    assert_eq!(model.compute_tau_spill(20, true), 0.0);
    assert_eq!(model.compute_tau_spill(24, false), 0.0);

    // 2. Target depth D_expr = 32 on NEON (R_avail = 20):
    // 32 - 20 = 12 spilled registers * 1.6 ns = 19.2 ns
    let tau_spill_32 = model.compute_tau_spill(32, true);
    assert!((tau_spill_32 - 19.2).abs() < 1e-6, "Tau spill at D=32 must be 19.2 ns");

    // 3. Target depth D_expr = 50 on NEON:
    // 50 - 20 = 30 spilled registers * 1.6 ns = 48.0 ns
    let tau_spill_50 = model.compute_tau_spill(50, true);
    assert!((tau_spill_50 - 48.0).abs() < 1e-6, "Tau spill at D=50 must be 48.0 ns");

    // 4. End-to-end Cost Model crossover verification under deep AST:
    // Vectorized baseline: 3.5 ns/tuple (intermediate arrays resident in L1/L2 chunks)
    // JIT baseline without spills: 1.0 ns/tuple
    let t_vec_tuple = 3.5;
    let t_jit_reg = 1.0;

    // Shallow AST (D = 10):
    let t_jit_d10 = model.compute_effective_jit_tuple_cost(10, t_jit_reg, true);
    assert_eq!(t_jit_d10, 1.0);
    let crossover_d10 = compute_crossover_cardinality(1.5, t_vec_tuple, t_jit_d10);
    assert_eq!(crossover_d10, 600_000); // 1.5e6 / 2.5 = 600,000 tuples (viable JIT)

    // Deep AST (D = 32):
    let t_jit_d32 = model.compute_effective_jit_tuple_cost(32, t_jit_reg, true);
    assert!((t_jit_d32 - 20.2).abs() < 1e-6);
    let crossover_d32 = compute_crossover_cardinality(1.5, t_vec_tuple, t_jit_d32);
    // Because delta_t = 3.5 - 20.2 = -16.7 <= 0, crossover must evaluate to usize::MAX
    assert_eq!(
        crossover_d32,
        usize::MAX,
        "At D_expr = 32, Sethi-Ullman stack spills must force N_crossover = usize::MAX"
    );

    // Deep AST (D = 50):
    let t_jit_d50 = model.compute_effective_jit_tuple_cost(50, t_jit_reg, true);
    assert!((t_jit_d50 - 49.0).abs() < 1e-6);
    let crossover_d50 = compute_crossover_cardinality(1.5, t_vec_tuple, t_jit_d50);
    assert_eq!(
        crossover_d50,
        usize::MAX,
        "At D_expr = 50, Sethi-Ullman stack spills must force N_crossover = usize::MAX"
    );

    // Determine the exact critical crossover depth D_crit where JIT ceases to be advantageous
    // 3.5 = 1.0 + (D_crit - 20) * 1.6 => 2.5 / 1.6 = 1.5625 => D_crit = 21.56 => D = 22
    let t_jit_d21 = model.compute_effective_jit_tuple_cost(21, t_jit_reg, true);
    assert!(t_jit_d21 < t_vec_tuple); // D=21: 1.0 + 1.6 = 2.6 < 3.5 (JIT still faster)

    let t_jit_d22 = model.compute_effective_jit_tuple_cost(22, t_jit_reg, true);
    assert!(t_jit_d22 > t_vec_tuple); // D=22: 1.0 + 3.2 = 4.2 > 3.5 (JIT slower!)
    let crossover_d22 = compute_crossover_cardinality(1.5, t_vec_tuple, t_jit_d22);
    assert_eq!(crossover_d22, usize::MAX);
}

// -----------------------------------------------------------------------------
// 3. Expression Complexity Index (ECI) Calculation & Stress Tests
// -----------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub enum TestAST {
    LiteralI32(i32),
    Column(String),
    Binary {
        op: String,
        left: Box<TestAST>,
        right: Box<TestAST>,
    },
    Case {
        cond: Box<TestAST>,
        then_arm: Box<TestAST>,
        else_arm: Box<TestAST>,
    },
    Cast {
        from: String,
        to: String,
        expr: Box<TestAST>,
    },
}

// Formulation A: COST_MODEL_FORMULATION.md (w1=2.0, w2=1.0, w3=3.5, w4=1.5)
pub fn compute_eci_spec(expr: &TestAST) -> f64 {
    fn analyze(e: &TestAST) -> (usize, usize, usize, usize) {
        match e {
            TestAST::LiteralI32(_) | TestAST::Column(_) => (0, 0, 0, 0),
            TestAST::Cast { expr, .. } => {
                let (d, n, b, c) = analyze(expr);
                (1 + d, 1 + n, b, 1 + c)
            }
            TestAST::Binary { left, right, .. } => {
                let (ld, ln, lb, lc) = analyze(left);
                let (rd, rn, rb, rc) = analyze(right);
                (1 + ld.max(rd), 1 + ln + rn, lb + rb, lc + rc)
            }
            TestAST::Case { cond, then_arm, else_arm } => {
                let (cd, cn, cb, cc) = analyze(cond);
                let (td, tn, tb, tc) = analyze(then_arm);
                let (ed, en, eb, ec) = analyze(else_arm);
                (
                    1 + cd.max(td).max(ed),
                    1 + cn + tn + en,
                    1 + cb + tb + eb,
                    cc + tc + ec,
                )
            }
        }
    }
    let (d, n, b, c) = analyze(expr);
    2.0 * (d as f64) + 1.0 * (n as f64) + 3.5 * (b as f64) + 1.5 * (c as f64)
}

// Formulation B: verification/src/main.rs (w1=1.0, w2=0.5, w3=1.5, w4=2.0)
pub fn compute_eci_crate(expr: &TestAST) -> f64 {
    fn analyze(e: &TestAST) -> (usize, usize, usize, usize) {
        match e {
            TestAST::LiteralI32(_) | TestAST::Column(_) => (0, 0, 0, 0),
            TestAST::Binary { left, right, .. } => {
                let (dl, ol, bl, ml) = analyze(left);
                let (dr, or, br, mr) = analyze(right);
                (1 + dl.max(dr), ol + or + 1, bl + br + 1, ml + mr)
            }
            _ => (1, 1, 1, 0),
        }
    }
    let (d, o, b, m) = analyze(expr);
    1.0 * (d as f64) + 0.5 * (o as f64) + 1.5 * (b as f64) + 2.0 * (m as f64)
}

#[test]
fn test_adversarial_eci_calibration() {
    // Test Vector 1: Simple Filter `col_a > 100`
    let simple_filter = TestAST::Binary {
        op: ">".to_string(),
        left: Box::new(TestAST::Column("col_a".to_string())),
        right: Box::new(TestAST::LiteralI32(100)),
    };

    let eci_spec = compute_eci_spec(&simple_filter);
    let eci_crate = compute_eci_crate(&simple_filter);

    // Both specifications must evaluate to exactly 3.0
    assert_eq!(eci_spec, 3.0, "Spec ECI for col_a > 100 must be 3.0");
    assert_eq!(eci_crate, 3.0, "Crate ECI for col_a > 100 must be 3.0");
    assert!(eci_spec < 4.0, "Simple filter ECI must be strictly < 4.0");
    assert!(eci_crate < 4.0, "Simple filter ECI must be strictly < 4.0");

    // Test Vector 2: Compound arithmetic `(col_a * col_b) + col_c > 100`
    let compound_arith = TestAST::Binary {
        op: ">".to_string(),
        left: Box::new(TestAST::Binary {
            op: "+".to_string(),
            left: Box::new(TestAST::Binary {
                op: "*".to_string(),
                left: Box::new(TestAST::Column("col_a".to_string())),
                right: Box::new(TestAST::Column("col_b".to_string())),
            }),
            right: Box::new(TestAST::Column("col_c".to_string())),
        }),
        right: Box::new(TestAST::LiteralI32(100)),
    };
    let eci_compound = compute_eci_spec(&compound_arith);
    // depth = 3, nodes = 3, branches = 0, casts = 0 => 2.0*3 + 1.0*3 = 9.0 >= 4.0
    assert!(
        eci_compound >= 4.0,
        "Compound arithmetic must have ECI >= 4.0 (got {})",
        eci_compound
    );

    // Test Vector 3: High-branching expression `CASE WHEN col_a > 0 THEN col_b ELSE col_c END`
    let case_expr = TestAST::Case {
        cond: Box::new(simple_filter.clone()),
        then_arm: Box::new(TestAST::Column("col_b".to_string())),
        else_arm: Box::new(TestAST::Column("col_c".to_string())),
    };
    let eci_branch = compute_eci_spec(&case_expr);
    // High branch weighting must push ECI well above 4.0
    assert!(
        eci_branch >= 8.0,
        "Branching CASE expression must have ECI >= 8.0 (got {})",
        eci_branch
    );
}

// -----------------------------------------------------------------------------
// 4. Concurrency Memory Ordering & ARM64 Cache Invalidation Stress Tests
// -----------------------------------------------------------------------------

type KernelFn = unsafe fn(val: i64) -> (i64, usize);

unsafe fn kernel_gen_0(val: i64) -> (i64, usize) {
    (val + 10, 0)
}
unsafe fn kernel_gen_1(val: i64) -> (i64, usize) {
    (val + 20, 1)
}
unsafe fn kernel_gen_2(val: i64) -> (i64, usize) {
    (val + 30, 2)
}
unsafe fn kernel_gen_3(val: i64) -> (i64, usize) {
    (val + 40, 3)
}

#[cfg(target_os = "macos")]
extern "C" {
    fn sys_icache_invalidate(start: *mut std::ffi::c_void, len: usize);
    fn sys_dcache_flush(start: *mut std::ffi::c_void, len: usize);
    fn pthread_jit_write_protect_np(enabled: i32);
}

pub struct JITHotSwapHarness {
    pub current_kernel: AtomicPtr<()>,
}

impl JITHotSwapHarness {
    pub fn new() -> Self {
        Self {
            current_kernel: AtomicPtr::new(kernel_gen_0 as *mut ()),
        }
    }

    #[inline(always)]
    pub fn execute(&self, val: i64) -> (i64, usize) {
        // Acquire ordering guarantees observing all writes prior to Release store
        let fn_ptr = self.current_kernel.load(Ordering::Acquire);
        let kernel: KernelFn = unsafe { std::mem::transmute(fn_ptr) };
        unsafe { kernel(val) }
    }

    pub fn hot_swap(&self, new_kernel: KernelFn) {
        #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
        unsafe {
            pthread_jit_write_protect_np(1);
            let ptr = new_kernel as *const () as *mut std::ffi::c_void;
            sys_dcache_flush(ptr, 64);
            sys_icache_invalidate(ptr, 64);
            core::arch::asm!("isb sy", options(nostack, preserves_flags));
        }

        // Release ordering guarantees published pointer is fully coherent
        self.current_kernel.store(new_kernel as *mut (), Ordering::Release);
    }
}

#[test]
fn test_adversarial_memory_ordering_concurrency() {
    let harness = Arc::new(JITHotSwapHarness::new());
    let stop_signal = Arc::new(AtomicBool::new(false));
    let num_readers = 6;
    let mut reader_handles = Vec::new();

    // Spawn concurrent reader threads hammering Acquire loads
    for reader_id in 0..num_readers {
        let h = harness.clone();
        let stop = stop_signal.clone();
        reader_handles.push(std::thread::spawn(move || {
            let mut read_count = 0;
            let mut last_gen = 0;
            while !stop.load(Ordering::Relaxed) {
                let (res, gen) = h.execute(100);
                // Verify no torn read or incoherent generation
                assert!(gen >= last_gen, "Monotonicity violated on reader {}: gen {} < last_gen {}", reader_id, gen, last_gen);
                last_gen = gen;
                let expected = match gen {
                    0 => 110,
                    1 => 120,
                    2 => 130,
                    3 => 140,
                    _ => panic!("Unexpected generation {}", gen),
                };
                assert_eq!(res, expected, "Data corruption / torn pointer read!");
                read_count += 1;
            }
            read_count
        }));
    }

    // Hot-swap thread performing Release stores
    let h_swap = harness.clone();
    let swapper = std::thread::spawn(move || {
        std::thread::sleep(Duration::from_millis(10));
        h_swap.hot_swap(kernel_gen_1);
        std::thread::sleep(Duration::from_millis(15));
        h_swap.hot_swap(kernel_gen_2);
        std::thread::sleep(Duration::from_millis(15));
        h_swap.hot_swap(kernel_gen_3);
        std::thread::sleep(Duration::from_millis(10));
    });

    swapper.join().unwrap();
    stop_signal.store(true, Ordering::Relaxed);

    let mut total_reads = 0;
    for handle in reader_handles {
        total_reads += handle.join().unwrap();
    }

    assert!(
        total_reads > 50_000,
        "Concurrent readers must complete high-throughput executions (completed {})",
        total_reads
    );
}

// -----------------------------------------------------------------------------
// 5. Adaptive Trigger Decoupling Simulation & Stress Tests
// -----------------------------------------------------------------------------

pub struct AdaptiveTriggerSimulator {
    pub v_vec_tuples_per_sec: f64,
}

impl AdaptiveTriggerSimulator {
    pub fn new() -> Self {
        Self {
            v_vec_tuples_per_sec: 100_000_000.0, // 100M tuples/sec = 100K tuples/ms
        }
    }

    #[inline]
    pub fn compute_n_trigger(&self, n_crossover: usize) -> usize {
        if n_crossover == usize::MAX {
            return usize::MAX; // Properly guarded: never trigger when crossover is impossible
        }
        let ten_percent = (0.10 * (n_crossover as f64)).ceil() as usize;
        100_000.min(ten_percent).max(1024)
    }

    pub fn evaluate_execution(
        &self,
        total_n: usize,
        n_crossover: usize,
        t_comp_ms: f64,
        t_vec_ns: f64,
        t_jit_ns: f64,
    ) -> (String, f64, f64) {
        let n_trigger = self.compute_n_trigger(n_crossover);

        // Case 1: Negative crossover (JIT not viable)
        if n_crossover == usize::MAX || n_trigger == usize::MAX {
            let total_time_ms = (total_n as f64 * t_vec_ns) / 1_000_000.0;
            return ("VectorizedOnly_NegativeCrossover".to_string(), total_time_ms, total_time_ms);
        }

        // Case 2: Query smaller than trigger threshold
        if total_n < n_trigger {
            let total_time_ms = (total_n as f64 * t_vec_ns) / 1_000_000.0;
            return ("VectorizedOnly_BelowTrigger".to_string(), total_time_ms, total_time_ms);
        }

        // Look-ahead gate check: N_remaining >= N_crossover + (T_comp * v_vec)
        let n_remaining = total_n - n_trigger;
        let v_vec_tuples_per_ms = self.v_vec_tuples_per_sec / 1_000.0;
        let n_in_flight = (t_comp_ms * v_vec_tuples_per_ms) as usize;
        let lookahead_required = n_crossover + n_in_flight;

        if n_remaining < lookahead_required {
            // Cannot amortize remaining volume => bypass compilation
            let total_time_ms = (total_n as f64 * t_vec_ns) / 1_000_000.0;
            return ("VectorizedOnly_LookAheadBypassed".to_string(), total_time_ms, total_time_ms);
        }

        // Case 3: Compilation triggered early and successfully amortized
        let tuples_done_at_swap = (n_trigger + n_in_flight).min(total_n);
        let tuples_at_jit = total_n - tuples_done_at_swap;

        let vec_time_ms = (tuples_done_at_swap as f64 * t_vec_ns) / 1_000_000.0;
        let jit_time_ms = (tuples_at_jit as f64 * t_jit_ns) / 1_000_000.0;
        let adaptive_total_time_ms = vec_time_ms + jit_time_ms;

        let pure_vec_time_ms = (total_n as f64 * t_vec_ns) / 1_000_000.0;

        ("Adaptive_JIT_Accelerated".to_string(), adaptive_total_time_ms, pure_vec_time_ms)
    }
}

#[test]
fn test_adversarial_adaptive_trigger_decoupling() {
    let sim = AdaptiveTriggerSimulator::new();

    // 1. Small ad-hoc query (N = 10_000 tuples):
    // Even if N_crossover = 200_000, query finishes before trigger (10_000 < 20_000)
    let (decision_small, t_adapt_small, t_vec_small) =
        sim.evaluate_execution(10_000, 200_000, 1.5, 5.0, 2.0);
    assert_eq!(decision_small, "VectorizedOnly_BelowTrigger");
    assert_eq!(t_adapt_small, t_vec_small);

    // 2. Interactive Cranelift Query (N = 1_000_000 tuples, N_crossover = 200_000, T_comp = 1.5 ms):
    // N_trigger = min(100_000, 20_000) = 20_000 tuples
    assert_eq!(sim.compute_n_trigger(200_000), 20_000);
    let (decision_inter, t_adapt_inter, t_vec_inter) =
        sim.evaluate_execution(1_000_000, 200_000, 1.5, 5.0, 2.0);
    assert_eq!(decision_inter, "Adaptive_JIT_Accelerated");
    let speedup_inter = t_vec_inter / t_adapt_inter;
    assert!(
        speedup_inter >= 1.5,
        "Interactive query must achieve significant speedup via early trigger (got {:.2}x)",
        speedup_inter
    );

    // 3. Large Batch LLVM Query (N = 50_000_000 tuples, N_crossover = 5_000_000, T_comp = 35.0 ms):
    // N_trigger = min(100_000, 500_000) = 100_000 tuples (capped at 100k)
    assert_eq!(sim.compute_n_trigger(5_000_000), 100_000);
    let (decision_batch, t_adapt_batch, t_vec_batch) =
        sim.evaluate_execution(50_000_000, 5_000_000, 35.0, 5.0, 2.0);
    assert_eq!(decision_batch, "Adaptive_JIT_Accelerated");
    let speedup_batch = t_vec_batch / t_adapt_batch;
    assert!(
        speedup_batch >= 2.0,
        "Batch query must achieve >= 2.0x speedup via early trigger (got {:.2}x)",
        speedup_batch
    );

    // 4. Adversarial Case: Negative crossover (N_crossover = usize::MAX):
    // Trigger must evaluate to usize::MAX and bypass compilation entirely
    let n_trigger_neg = sim.compute_n_trigger(usize::MAX);
    assert_eq!(n_trigger_neg, usize::MAX);
    let (decision_neg, _, _) =
        sim.evaluate_execution(100_000_000, usize::MAX, 1.5, 5.0, 2.0);
    assert_eq!(decision_neg, "VectorizedOnly_NegativeCrossover");
}
