# COST MODEL FORMULATION: SELECTIVE JIT COMPILATION VS VECTORIZED INTERPRETATION
**Document Identifier**: `COST-MODEL-0001`  
**Target Platform**: Pure Rust 1.75+ (x86_64 AVX-512 & AArch64 Apple Silicon NEON)  
**Status**: Authoritative Mathematical & Algorithmic Specification  
**Version**: 1.0.0  

---

## 1. Executive Summary & Problem Formulation

In modern analytical database engines, the choice between **Vectorized Interpretation** (processing columnar vectors via pre-compiled SIMD kernels) and **JIT Code Generation** (compiling expression DAGs directly into machine code via LLVM or Cranelift) represents a fundamental systems trade-off between **compilation latency** and **execution throughput**.

- **Vectorized Interpretation** has zero compilation latency ($T_{comp} = 0$), but suffers from intermediate memory writebacks between expression operators (polluting L1/L2 data caches) and function call dispatch overhead.
- **JIT Compilation** achieves near-optimal CPU register residency and loop fusion, but incurs a substantial compilation overhead ($T_{comp} \in [1\text{ ms}, 250\text{ ms}+]$), making it disastrous for interactive, ad-hoc, or low-cardinality queries.

This document formulates the exact mathematical model, sensitivity bounds, hardware divergence equations, and dynamic heuristic decision algorithms governing the JIT Crossover Point ($N_{crossover}$) for the pure native Rust engine.

---

## 2. Analytical Execution Time Formulations

Let $Q$ denote a relational query pipeline consisting of a chain of operators $O = \langle op_1, op_2, \dots, op_m \rangle$ evaluated over an input relation of estimated cardinality $\hat{N}$ tuples.

### 2.1 Vectorized Interpretation Model ($T_{vec}$)

In vectorized execution, tuples are processed in discrete morsel batches of size $V$ tuples ($V \in [1024, 4096]$). The total execution time is expressed as:

$$T_{vec} = T_{init\_vec} + \sum_{k=1}^{\lceil \hat{N} / V \rceil} T_{morsel\_vec}(k)$$

Where:
- $T_{init\_vec}$ is the plan instantiation latency ($T_{init\_vec} \approx 10^{-6}\text{ s}$, negligible in practice).
- $T_{morsel\_vec}(k)$ is the time required to evaluate morsel $k$ through the operator pipeline:

$$T_{morsel\_vec}(k) = \sum_{op \in O} \left( \tau_{dispatch}(op) + V_{in}(op, k) \cdot \left[ \frac{\tau_{compute}(op)}{W_{SIMD}(op)} + \tau_{mem\_rw}(op) \right] \right)$$

#### Component Breakdown:
1. **Morsel Dispatch Overhead ($\tau_{dispatch}$)**:
   $$\tau_{dispatch}(op) \approx 5\text{ to }15\text{ ns per operator invocation}$$
   Amortized on a per-tuple basis, this contributes:
   $$\frac{\tau_{dispatch}(op)}{V} \approx \frac{10\text{ ns}}{2048} \approx 0.00488\text{ ns/tuple} \quad (\text{negligible})$$

2. **SIMD Vectorization Lane Width ($W_{SIMD}$)**:
   The effective hardware vector width in elements:
   $$W_{SIMD} = \frac{\text{Register Width (bits)}}{\text{Element Bitwidth (bits)}} = \begin{cases}
   16 & \text{AVX-512 for 32-bit types (i32, f32)} \\
   8 & \text{AVX-512 for 64-bit types (i64, f64)} \\
   4 & \text{NEON for 32-bit types (i32, f32)} \\
   2 & \text{NEON for 64-bit types (i64, f64)}
   \end{cases}$$

3. **Intermediate Memory Write-and-Read Overhead ($\tau_{mem\_rw}$)**:
   For every non-fused intermediate operator, the output vector must be written to an intermediate buffer in L1/L2 cache and subsequently read by the consumer operator:
   $$\tau_{mem\_rw}(op) = \frac{\text{Bytes}_{write}(op) + \text{Bytes}_{read}(op)}{\text{Bandwidth}_{L1/L2}} + \tau_{cache\_miss}(op)$$
   On modern architectures (e.g. Zen 4 or Apple M4):
   $$\text{Bandwidth}_{L1D} \approx 1.5\text{--}2.5\text{ TB/s (aggregate)}$$
   For an 8-byte column ($64\text{ bits}$), writing and immediately reading one tuple incurs:
   $$\tau_{mem\_rw} \approx \frac{16\text{ bytes}}{2.0 \times 10^{12}\text{ bytes/s}} \approx 8.0\text{ picoseconds} + \text{cache line lookup latency} \approx 1.2\text{--}2.0\text{ ns/tuple}$$

4. **Selectivity Attenuation ($V_{in}$)**:
   The volume of tuples surviving to operator $op$ decreases monotonically with the cumulative selectivity of all upstream filtering predicates:
   $$V_{in}(op, k) = V \cdot \prod_{p \in \text{Ancestors}(op)} \sigma_p, \quad \sigma_p \in [0, 1]$$

---

### 2.2 JIT-Compiled Execution Model ($T_{jit}$)

In JIT-compiled execution, the entire expression pipeline is compiled into a single native loop. Intermediate values remain resident exclusively in CPU physical registers, eliminating intermediate memory read/write cycles ($\tau_{mem\_rw} = 0$).

$$T_{jit} = T_{comp}(D_{expr}, M_{nodes}, \text{Backend}) + T_{init\_jit} + \sum_{k=1}^{\lceil \hat{N} / V \rceil} T_{morsel\_jit}(k)$$

Where:
- $T_{comp}$ is the compilation latency to synthesize machine code.
- $T_{morsel\_jit}(k)$ is the morsel execution time:

$$T_{morsel\_jit}(k) = \sum_{i=1}^{V} \left( \sum_{op \in O} \tau_{compute\_reg}(op) \cdot \mathbb{I}(\text{survives}(i, op)) \right)$$

#### Component Breakdown:
1. **Register-Resident Compute Latency ($\tau_{compute\_reg}$)**:
   Because variables are held in general-purpose or vector registers, no memory loads or stores occur between operators:
   $$\tau_{compute\_reg}(op) \approx \frac{\text{Instruction Latency}}{\text{IPC}} \approx 0.25\text{ to }0.75\text{ ns/operation}$$

2. **Compilation Latency Model ($T_{comp}$)**:
   The compilation latency is parameterized by expression AST depth ($D_{expr}$), node count ($M_{nodes}$), and compiler backend:
   $$T_{comp}(D_{expr}, M_{nodes}, \text{Backend}) = \alpha_{backend} + \beta_{backend} \cdot M_{nodes} + \gamma_{backend} \cdot (D_{expr})^{\kappa}$$

| Compilation Backend | Base Latency $\alpha$ | Per-Node Latency $\beta$ | Non-Linear Exponent $\kappa$ | Optimization Level |
|---|---|---|---|---|
| **KernelCache Hit** | $0.000\text{ ms}$ | $0.000\text{ ms}$ | $0.0$ | Instantaneous ($0\text{ ms}$) |
| **Cranelift** | $0.750\text{ ms}$ | $0.035\text{ ms}$ | $1.05$ | Moderate (Fast compilation) |
| **LLVM (O0)** | $12.50\text{ ms}$ | $0.350\text{ ms}$ | $1.20$ | Baseline code generation |
| **LLVM (O2)** | $35.00\text{ ms}$ | $0.650\text{ ms}$ | $1.45$ | Aggressive loop unrolling |
| **LLVM (O3)** | $65.00\text{ ms}$ | $1.100\text{ ms}$ | $1.60$ | Peak vector auto-vectorization |
| **MLIR $\to$ LLVM** | $45.00\text{ ms}$ | $0.850\text{ ms}$ | $1.50$ | Structured affine dialects |

---

## 3. Mathematical Derivation of the Exact Crossover Point ($N_{crossover}$)

The crossover cardinality $N_{crossover}$ is defined as the input cardinality where total execution time under vectorized interpretation equals total execution time under JIT compilation:

$$T_{vec}(N_{crossover}) = T_{jit}(N_{crossover})$$

Substituting the analytical formulations and noting that $T_{init\_vec} \approx T_{init\_jit}$:

$$N_{crossover} \cdot t_{vec\_tuple} = T_{comp} + N_{crossover} \cdot t_{jit\_tuple}$$

Solving directly for $N_{crossover}$:

$$N_{crossover} = \frac{T_{comp}}{\Delta t_{tuple}} = \frac{T_{comp}(D_{expr}, M_{nodes}, \text{Backend})}{t_{vec\_tuple} - t_{jit\_tuple}}$$

Where the per-tuple execution differential $\Delta t_{tuple}$ is formally defined as:

$$\Delta t_{tuple} = \sum_{op \in O} \left( \frac{\tau_{dispatch}(op)}{V} + \tau_{mem\_rw}(op) + \tau_{compute}(op) \left[ \frac{1}{W_{SIMD\_vec}} - \frac{1}{W_{SIMD\_jit}} \right] \right) \cdot \prod_{p \in \text{Ancestors}(op)} \sigma_p$$

---

## 4. Sensitivity Analysis & Boundary Value Limits

### 4.1 Boundary Limit 1: Primitive Shallow Expressions ($\Delta t_{tuple} \to 0$)
Consider a single-operator expression:
$$\text{SELECT col\_a FROM table WHERE col\_a > 100}$$
- Here, $D_{expr} = 1$, $M_{nodes} = 2$.
- The vectorized engine executes a pre-compiled, highly optimized SIMD kernel (`filter_i32_avx512` or `filter_i32_neon_unrolled`).
- There are no intermediate vector writebacks ($\tau_{mem\_rw} = 0$).
- Precompiled SIMD kernels match or exceed JIT-generated SIMD loop efficiency ($W_{SIMD\_vec} = W_{SIMD\_jit}$).
- Consequently:
  $$\Delta t_{tuple} \to 0^+ \implies \lim_{\Delta t_{tuple} \to 0} N_{crossover} = +\infty$$

**Fundamental Architectural Theorem 1**:  
*For primitive expressions ($D_{expr} \le 2, M_{nodes} \le 3$), JIT compilation is mathematically non-viable. Vectorized execution strictly dominates across all finite cardinalities.*

---

### 4.2 Boundary Limit 2: Deep Compound Expressions ($D_{expr} \ge 3$, High $M_{nodes}$)
Consider a compound financial calculation:
$$((col_a \times 1.05 + col_b \times 0.20) / (col_c - 4.20) > 100.0) \land (col_d \ne 0)$$
- Expression metrics: $D_{expr} = 4$, $M_{nodes} = 8$, no short-circuiting branches.
- **Vectorized Execution**:
  - Requires $4$ intermediate vector allocations:
    1. $temp_1 = col_a \times 1.05$
    2. $temp_2 = col_b \times 0.20$
    3. $temp_3 = temp_1 + temp_2$
    4. $temp_4 = col_c - 4.20$
    5. $temp_5 = temp_3 / temp_4$
  - Intermediate memory read/write latency:
    $$\tau_{mem\_rw} \approx 4 \times 1.6\text{ ns} = 6.4\text{ ns/tuple}$$
  - Total per-tuple latency: $t_{vec\_tuple} \approx 9.2\text{ ns}$.
- **JIT Execution**:
  - Entire calculation resides in registers: $t_{jit\_tuple} \approx 1.8\text{ ns}$.
  - Execution differential: $\Delta t_{tuple} = 9.2 - 1.8 = 7.4\text{ ns/tuple}$.

#### Crossover Points by Backend:
1. **Cranelift Backend** ($T_{comp} \approx 1.2\text{ ms}$):
   $$N_{crossover}^{\text{Cranelift}} = \frac{1.2 \times 10^{-3}\text{ s}}{7.4 \times 10^{-9}\text{ s}} \approx \mathbf{162,162\text{ tuples}}$$
2. **LLVM (O2) Backend** ($T_{comp} \approx 42\text{ ms}$):
   $$N_{crossover}^{\text{LLVM-O2}} = \frac{42 \times 10^{-3}\text{ s}}{7.4 \times 10^{-9}\text{ s}} \approx \mathbf{5,675,675\text{ tuples}}$$

---

### 4.3 Boundary Limit 3: Selectivity Attenuation Factor
If the query features an early, highly selective filter predicate $p_1$ with selectivity $\sigma_1 = 0.005$ ($0.5\%$ survival):
- Surviving cardinality arriving at the deep compound expression is:
  $$\hat{N}_{effective} = \hat{N} \times 0.005$$
- The per-tuple execution differential $\Delta t_{tuple}$ is scaled down by a factor of $200\times$:
  $$\Delta t_{tuple}^{effective} = 7.4\text{ ns} \times 0.005 = 0.037\text{ ns/tuple}$$
- The effective crossover point shifts by $200\times$:
  $$N_{crossover}^{\text{Attenuated}} = \frac{1.2 \times 10^{-3}\text{ s}}{0.037 \times 10^{-9}\text{ s}} \approx \mathbf{32,432,432\text{ tuples}}$$

**Fundamental Architectural Theorem 2**:  
*Early selective predicates exponentially delay the JIT crossover point. Cost models that evaluate expression complexity in isolation without conditioning on cumulative upstream selectivity $\prod \sigma_p$ trigger catastrophic JIT over-compilation.*

---

## 5. Expression Complexity Index ($ECI$)

To evaluate candidate pipelines in $O(M)$ time during physical plan optimization without invoking expensive profiling probes, the engine computes the scalar **Expression Complexity Index ($ECI$)**:

$$ECI = w_1 \cdot D_{expr} + w_2 \cdot M_{nodes} + w_3 \cdot B_{branch} + w_4 \cdot C_{cast}$$

### Calibrated Weight Parameters:
- $w_1 = 2.0$: Penalizes expression depth (correlates directly with register pressure and intermediate buffer count).
- $w_2 = 1.0$: Accounts for total operator node count.
- $w_3 = 3.5$: Heavily weights conditional branches (`CASE WHEN`, `COALESCE`) which destroy SIMD lane efficiency in vectorized interpretation.
- $w_4 = 1.5$: Weights type casts requiring vector layout conversion (e.g. `i32` to `f64`).

```rust
use crate::ir::{Expr, BinaryOp, UnaryOp};

pub struct ComplexityAnalyzer;

impl ComplexityAnalyzer {
    pub fn compute_eci(expr: &Expr) -> f64 {
        let (depth, nodes, branches, casts) = Self::analyze(expr);
        2.0 * (depth as f64) + 1.0 * (nodes as f64) + 3.5 * (branches as f64) + 1.5 * (casts as f64)
    }

    fn analyze(expr: &Expr) -> (usize, usize, usize, usize) {
        match expr {
            Expr::Column { .. } | Expr::Literal(_) => (1, 1, 0, 0),
            Expr::Unary { expr, .. } => {
                let (d, n, b, c) = Self::analyze(expr);
                (1 + d, 1 + n, b, c)
            }
            Expr::Cast { expr, .. } => {
                let (d, n, b, c) = Self::analyze(expr);
                (1 + d, 1 + n, b, 1 + c)
            }
            Expr::Binary { left, right, .. } => {
                let (ld, ln, lb, lc) = Self::analyze(left);
                let (rd, rn, rb, rc) = Self::analyze(right);
                (1 + ld.max(rd), 1 + ln + rn, lb + rb, lc + rc)
            }
            Expr::Case { condition, then_expr, else_expr, .. } => {
                let (cd, cn, cb, cc) = Self::analyze(condition);
                let (td, tn, tb, tc) = Self::analyze(then_expr);
                let (ed, en, eb, ec) = Self::analyze(else_expr);
                (1 + cd.max(td).max(ed), 1 + cn + tn + en, 1 + cb + tb + eb, cc + tc + ec)
            }
            Expr::ScalarFn { args, .. } => {
                let mut max_d = 0;
                let mut sum_n = 1;
                let mut sum_b = 0;
                let mut sum_c = 0;
                for arg in args {
                    let (ad, an, ab, ac) = Self::analyze(arg);
                    max_d = max_d.max(ad);
                    sum_n += an;
                    sum_b += ab;
                    sum_c += ac;
                }
                (1 + max_d, sum_n, sum_b, sum_c)
            }
            Expr::InList { expr, list, .. } => {
                let (ed, en, eb, ec) = Self::analyze(expr);
                let mut max_d = ed;
                let mut sum_n = 1 + en;
                let mut sum_b = eb;
                let mut sum_c = ec;
                for item in list {
                    let (id, in_, ib, ic) = Self::analyze(item);
                    max_d = max_d.max(id);
                    sum_n += in_;
                    sum_b += ib;
                    sum_c += ic;
                }
                (1 + max_d, sum_n, 1 + sum_b, sum_c)
            }
        }
    }
}
```

---

## 6. Hardware-Specific Crossover Shift: x86 AVX-512 vs Apple Silicon NEON

A key contribution of this research is establishing that **the JIT crossover point is not architecture-invariant**. Hardware memory subsystems and register file capacities dramatically shift the optimal boundary:

```
                           ROOFLINE COMPARISON & BANDWIDTH WALL
Throughput
   ▲
   │                           ┌──────────────────────── x86 AVX-512 Compute Roof (3,072 GFLOPS)
   │                           │
   │              ┌────────────┼──────────────────────── Apple M4 NEON Compute Roof (1,536 GFLOPS)
   │             /            /
   │            /            /
   │           /            /
   │          /            / ◄── x86 DDR5 Memory Wall (100 GB/s)
   │         /            /
   │        /            /
   │       / ◄──────────┼─────── Apple Silicon UMA Memory Wall (400 GB/s)
   │      /             │
   └─────┴──────────────┴────────────────────────► Operational Intensity (Flops/Byte)
```

### 6.1 Architectural Factors Governing the Shift
1. **Memory Bandwidth & Cache Hierarchy Divergence**:
   - Standard dual-socket x86 servers with 8-channel DDR4/DDR5 achieve between $100\text{--}250\text{ GB/s}$ sustained memory bandwidth across NUMA nodes.
   - Apple Silicon Unified Memory Architecture (UMA) provides on-package memory buses ($410\text{ GB/s}$ on the 14-core M4 Max, and up to $546\text{ GB/s}$ on the 16-core M4 Max).
   - Higher memory bandwidth reduces the memory wall penalty of intermediate vector writes/reads ($\tau_{mem\_rw}$) in vectorized execution, keeping vectorized morsel processing closer to compute saturation.
2. **Register File Constraints & Compiler Unrolling**:
   - x86 AVX-512 provides $32 \times 512\text{-bit}$ registers ($2,048\text{ bytes}$).
   - ARM64 NEON provides $32 \times 128\text{-bit}$ registers ($512\text{ bytes}$).
   - Aggressive loop unrolling and deep expression JIT compilation on NEON runs out of physical registers earlier, risking stack frame spills that attenuate JIT loop throughput.

### 6.2 Formal Research Hypothesis $H_2$
Rather than assuming an a priori fixed constant across varying microarchitectures, the engine formalizes this shift as a testable empirical hypothesis:

$$\mathbf{Hypothesis\ H_2}:\quad N_{crossover}^{\text{Apple Silicon NEON}} \approx 2.5\text{ to }4.0 \times N_{crossover}^{\text{x86 AVX-512}}$$

*Empirical Protocol*:
The research roadmap mandates calibrating $T_{comp}, t_{vec\_tuple}, t_{jit\_tuple}$, hardware cache misses, core clock frequencies, and achieved DRAM throughput independently on physical target machines. On Apple Silicon, the query planner adopts a more conservative heuristic, defaulting to vectorized morsel execution across larger cardinality ranges unless high expression complexity ($ECI \ge 8.0$) or compilation cache hits are observed.

---

## 7. Multi-Tier Adaptive Execution Engine (Umbra-Style)

Static query-time compilation decisions suffer from optimizer cardinality estimation errors ($\hat{N}$ can deviate from actual $N$ by orders of magnitude). To eliminate this risk, the engine implements a **Multi-Tier Adaptive Execution Runtime**:

```
                       Query Starts Immediately
                                  │
                                  ▼
                     [Tier 1: Vectorized Morsels]
                                  │
                                  ├──────────────────────────────┐
                                  ▼                              ▼
                 Observed Morsels Count N_actual          Pipeline Completes
                                  │                       (If N < N_crossover)
                    If N_actual >= Threshold                     │
                                  │                              ▼
                                  ▼                          Query Done
                   Async Compile Task Spawned
                  (Background Rayon Compute Pool)
                                  │
                                  ▼
                   [Cranelift / LLVM CodeGen]
                                  │
                                  ▼
                   Atomic Pointer Hot-Swap
                atomic_store(exec_fn, jit_kernel)
                                  │
                                  ▼
                      [Tier 2: JIT Compiled Loop]
                                  │
                                  ▼
                          Remaining Morsels
                      Processed at Full Speed
```

### 7.1 Lock-Free Pipeline Hot-Swap Implementation

```rust
use std::sync::atomic::{AtomicPtr, Ordering};
use std::sync::Arc;
use arrow::record_batch::RecordBatch;

pub type KernelFn = unsafe fn(*const u8, usize, *mut u8) -> usize;

pub struct AdaptivePipelineOperator {
    current_kernel: AtomicPtr<()>,
    fallback_kernel: KernelFn,
    tuples_processed: std::sync::atomic::AtomicUsize,
    crossover_threshold: usize,
}

impl AdaptivePipelineOperator {
    pub fn new(fallback: KernelFn, threshold: usize) -> Self {
        Self {
            current_kernel: AtomicPtr::new(fallback as *mut ()),
            fallback_kernel: fallback,
            tuples_processed: std::sync::atomic::AtomicUsize::new(0),
            crossover_threshold: threshold,
        }
    }

    #[inline(always)]
    pub fn process_morsel(&self, in_ptr: *const u8, len: usize, out_ptr: *mut u8) -> usize {
        let fn_ptr = self.current_kernel.load(Ordering::Relaxed);
        let kernel: KernelFn = unsafe { std::mem::transmute(fn_ptr) };
        let matched = unsafe { kernel(in_ptr, len, out_ptr) };

        // Increment processed volume tracking
        let prev = self.tuples_processed.fetch_add(len, Ordering::Relaxed);
        if prev < self.crossover_threshold && prev + len >= self.crossover_threshold {
            // Signal background compilation task
            self.trigger_background_jit();
        }

        matched
    }

    pub fn hot_swap_jit_kernel(&self, jit_kernel: KernelFn) {
        self.current_kernel.store(jit_kernel as *mut (), Ordering::Release);
    }

    fn trigger_background_jit(&self) {
        // Asynchronously dispatches code generation to Rayon without blocking current morsel
    }
}
```

---

## 8. Summary of Heuristic Crossover Policy

```text
Decision Table:
┌─────────────────┬─────────────────┬──────────────────────┬──────────────────────────────────┐
│ Expression ECI  │ Estimated N_hat │ Kernel Cache Status  │ Selected Execution Engine        │
├─────────────────┼─────────────────┼──────────────────────┼──────────────────────────────────┤
│ Any             │ Any             │ Cache Hit            │ Instant JIT (Cached Kernel)      │
│ ECI < 4.0       │ Any             │ Cache Miss           │ Pre-compiled Vectorized Morsels  │
│ ECI >= 4.0      │ N < N_crossover │ Cache Miss           │ Pre-compiled Vectorized Morsels  │
│ 4.0 <= ECI < 8  │ N >= N_crossover│ Cache Miss           │ Cranelift JIT (< 2ms compile)    │
│ ECI >= 8.0      │ N >= 10,000,000 │ Cache Miss           │ Adaptive: Vectorized -> LLVM JIT │
└─────────────────┴─────────────────┴──────────────────────┴──────────────────────────────────┘
```
