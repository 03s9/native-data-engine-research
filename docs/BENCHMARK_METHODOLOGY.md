# BENCHMARK METHODOLOGY: MICRO & MACRO EVALUATION PROTOCOL
**Document Identifier**: `BENCH-METH-0001`  
**Target Platform**: Pure Rust 1.75+ (x86_64 AVX-512 & AArch64 Apple Silicon NEON)  
**Status**: Authoritative Experimental Specification  
**Version**: 1.0.0  

---

## 1. Experimental Objectives & Measurement Rigor

The empirical evaluation of the native data engine must prove the core research hypotheses through reproducible, deterministic, and noise-isolated experiments. The testing framework spans two complementary levels of granularity:

1. **Microbenchmarking Layer (Criterion.rs & Hardware PMU Counters)**:
   - Nanosecond-resolution isolation of kernel primitives.
   - Exact cycle counts, Instruction-Per-Cycle ($IPC$), Cycles-Per-Tuple ($CPT$), and hardware performance counter profiling (L1D/LLC misses, branch mispredictions).
2. **Macrobenchmarking Layer (Industry OLAP Suites)**:
   - End-to-end analytical query execution under realistic scale factors:
     - **TPC-H (SF1 to SF100)**: Complex decision support queries (Q1, Q6, Q9, Q18).
     - **ClickBench (100M rows, 43 queries)**: Real-world web analytics with wide tables (105 columns) and selective aggregations.
     - **h2oai db-benchmark**: High-cardinality group-by aggregations and stress joins.

---

## 2. Microbenchmarking Protocol with Criterion.rs

### 2.1 Environmental Controls & Determinism Rules
To eliminate environmental jitter, thermal throttling, and context switching artifacts:
- **CPU Isolation & Affinity**: Pin benchmark worker threads to dedicated physical cores using `core_affinity`. Avoid hyperthreaded logical sibling cores.
- **Clock Frequency Locking**:
  - *Linux x86_64*: Set CPU scaling governor to `performance`:
    ```bash
    echo "performance" | sudo tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor
    ```
  - *macOS Apple Silicon*: Connect system to AC power; set battery state to High Power Mode (on supported Pro/Max chips); execute benchmarks in dedicated terminal sessions with App Nap disabled:
    ```bash
    defaults write NSGlobalDomain NSAppSleepDisabled -bool YES
    ```
- **Statistical Parameters**:
  - Warmup iterations: $N_{warmup} \ge 1,000$ iterations.
  - Sample count: $S = 200$ independent samples.
  - Measurement time: Minimum $10.0\text{ seconds}$ per benchmark case.
  - Outlier detection: Student's t-test and interquartile range (IQR) rejection.

---

### 2.2 Microbenchmark Target Definitions

```
┌────────────────────────────────────────────────────────────────────────┐
│                        MICROBENCHMARK SUITE TAXONOMY                   │
├────────────────────────────────┬───────────────────────────────────────┤
│ Target M1: Morsel Size Sweep   │ V ∈ {256, 512, 1K, 2K, 4K, 8K, 16K}   │
│ Target M2: SIMD vs Scalar      │ Scalar vs Auto-Vec vs AVX-512 vs NEON │
│ Target M3: Branchless Masking  │ Selectivity σ ∈ [0.001, 0.999]        │
│ Target M4: JIT Compile Latency │ AST Depth D ∈ [1, 16], Cranelift/LLVM │
└────────────────────────────────┴───────────────────────────────────────┘
```

#### Target M1: Morsel Sizing vs Cache Hierarchy
- **Hypothesis**: Processing morsels sized to fit inside L1D cache ($V = 2048$ tuples, $\approx 16\text{ KB/column}$) minimizes memory bus stall cycles compared to monolithic vector batches ($V = 65,536$).
- **Metric**: Elapsed execution time per morsel and L1D miss rate.

```rust
use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};

fn bench_morsel_sizes(c: &mut Criterion) {
    let mut group = c.benchmark_group("morsel_sizing_l1_l2");
    let morsel_sizes = [256, 512, 1024, 2048, 4096, 8192, 16384];

    for &size in &morsel_sizes {
        let input: Vec<i32> = (0..size as i32).collect();
        let mut output = vec![0i32; size];

        group.bench_with_input(BenchmarkId::from_parameter(size), &size, |b, _| {
            b.iter(|| {
                // Evaluates tight arithmetic across morsel
                for i in 0..size {
                    output[i] = black_box(input[i] * 2 + 1);
                }
            });
        });
    }
    group.finish();
}
```

#### Target M2: SIMD Lane Packing vs Scalar Fallback
- **Hypothesis**: Explicit AVX-512 intrinsics with mask compaction and 4x unrolled NEON kernels achieve $\ge 3.5\times$ speedup over compiler auto-vectorized loops on selective filter kernels.
- **Metric**: Cycles-Per-Tuple ($CPT$) and Instructions-Per-Cycle ($IPC$).

#### Target M3: Branchless Selection Vector vs Branching Evaluation
- **Hypothesis**: Branch predictors suffer high misprediction penalties when filter selectivity $\sigma \in [0.20, 0.80]$. Branchless bitmask compaction achieves deterministic latency independent of $\sigma$.
- **Metric**: Branch miss count and latency curve across $\sigma \in \{0.001, 0.01, 0.05, 0.1, 0.3, 0.5, 0.7, 0.9, 0.999\}$.

#### Target M4: JIT Compilation Latency vs Expression Depth
- **Hypothesis**: Cranelift compilation latency scales linearly ($O(M)$), enabling JIT viability for ad-hoc queries, whereas LLVM (O3) exhibits superlinear latency ($O(M^\kappa), \kappa \approx 1.5$) that demands high cardinality amortization.
- **Metric**: Microseconds to compile synthetic ASTs of depths $D \in \{1, 2, 4, 8, 16\}$.

---

### 2.3 Hardware Performance Counter Extraction (PMU)

#### Linux (x86_64) Profiling Protocol:
```bash
perf stat -e cycles,instructions,cache-references,cache-misses,\
L1-dcache-load-misses,LLC-load-misses,branch-instructions,branch-misses \
cargo bench --bench criterion_simd
```

#### macOS Apple Silicon Profiling Protocol:
```bash
xcrun xctrace record --template 'CPU Counters' \
  --output benchmark_profile.trace \
  --launch target/release/native_engine_benchmark -- --target m2_simd
```

Key PMU events recorded on Apple Silicon:
- `INST_RETIRED`: Total retired instructions.
- `CYCLES`: Active core clock cycles.
- `L1D_CACHE_MISS_LD`: L1 data cache load misses.
- `BRANCH_MISPRED`: Branch prediction misses.

---

## 3. Industry Macrobenchmark Suites

### 3.1 TPC-H Decision Support Suite
The engine is evaluated across three scale factors: **SF1 (1 GB)**, **SF10 (10 GB)**, and **SF100 (100 GB)**.

```
┌────────────────────────────────────────────────────────────────────────┐
│                        TPC-H BENCHMARK WORKLOAD STRESS                 │
├─────────┬──────────────────────────────┬───────────────────────────────┤
│ Query   │ Analytical Characteristic    │ Subsystem Stressed            │
├─────────┼──────────────────────────────┼───────────────────────────────┤
│ Q1      │ High-throughput scan & agg   │ SIMD Vector Reduction / Sum   │
│ Q6      │ Selective multi-filter scan  │ Late-Materialization / Prune  │
│ Q9      │ Multi-table complex joins    │ Dynamic Rayon HashJoin Build  │
│ Q18     │ High-cardinality grouping    │ Memory Spill / Grace HashJoin │
└─────────┴──────────────────────────────┴───────────────────────────────┘
```

#### Detailed Query Profiles:
- **TPC-H Q1 (Scan-Heavy SIMD Stress)**:
  - Scans $6,000,000$ rows (at SF1) of `lineitem`.
  - Performs 8 aggregations (`sum(l_quantity)`, `sum(l_extendedprice)`, `avg`, `count`).
  - Evaluates SIMD register unrolling, floating-point vector accumulation, and morsel work-stealing balance.
- **TPC-H Q6 (Forensic Late-Materialization Test)**:
  - Three predicate filters: `l_shipdate >= '1994-01-01'`, `l_discount between 0.05 and 0.07`, `l_quantity < 24`.
  - Selectivity: $\sigma \approx 0.019$ ($1.9\%$ survival).
  - Proves late-materialization: verifying that non-predicate columns (`l_extendedprice`, `l_discount`) are gathered **only** for surviving rows, achieving $> 95\%$ bandwidth savings.
- **TPC-H Q9 & Q18 (Multi-Way Join & Spill Stress)**:
  - Multi-table joins across `orders`, `lineitem`, `part`, `supplier`.
  - Stresses thread-local hash table construction, lock-free radix stitching, and memory spill under constrained memory budgets ($512\text{ MB}$ to $2\text{ GB}$).

---

### 3.2 ClickBench 100M Suite (Ad-Hoc Web Analytics)
ClickBench represents real-world, un-normalized wide tables reflecting production telemetry:
- **Dataset**: `hits.parquet` containing $100,000,000$ rows, $105$ columns (~14 GB Parquet compressed, ~70 GB uncompressed).
- **Execution Protocol**:
  - **Cold Run Protocol**:
    - *Linux*: Execute `sync && echo 3 | sudo tee /proc/sys/vm/drop_caches` to flush page, dentry, and inode caches.
    - *macOS*: Open Parquet file handles with `fcntl(fd, F_NOCACHE, 1)` to disable OS buffer cache caching and force direct storage transfers, avoiding system-wide unprivileged purge issues.
    - Cold runs isolate Tokio asynchronous I/O scheduling, Parquet dictionary decoding, and memory-mapped Arrow buffer assembly.
  - **Hot Run Protocol**:
    - Run 3 warmup iterations to ensure pages reside in the OS cache and memory-mapped buffers.
    - Hot runs isolate memory bandwidth saturation, StringView evaluations, and JIT crossover throughput.
- **Workload**: All 43 canonical ClickBench queries.

---

### 3.3 h2oai db-benchmark Suite (Advanced GroupBy and Join Stress)
- **Grouping Tasks**: 10 standard queries testing grouping across cardinality steps:
  - Low cardinality: $K = 2$ and $K = 100$ groups.
  - Medium cardinality: $K = 10,000$ groups.
  - Extreme cardinality: $K = 10^7$ unique keys over $10^8$ rows.
- **Join Tasks**:
  - Small-to-large ($10^6 \bowtie 10^8$), medium-to-large ($10^7 \bowtie 10^8$), and large-to-large joins under uniform and Zipfian ($\alpha = 1.0$) skew distributions.
- **Evaluates**: Lock-free hash table collision resolution, thread-local aggregation states, and deterministic spill triggering at soft memory limits (80%).

---

## 4. Multi-Target Hardware Evaluation Matrix

### 4.1 Rigorous Testbed Specifications
The benchmark suite is executed across three dedicated reference testbeds to prevent conflating microarchitectures:

| Hardware Attribute | Server A: Intel Ice Lake Enterprise | Server B: AMD Zen 4 Enterprise | Workstation C: Apple Silicon M4 Max |
| :--- | :--- | :--- | :--- |
| **Processor** | Dual Intel Xeon Platinum 8380 (80 cores) | AMD EPYC 9654 (96 cores) | Apple M4 Max (14-core: 10P + 4E) |
| **Microarchitecture** | Ice Lake (10nm Sunny Cove) | Zen 4 Genoa (5nm) | M4 (ARMv9.2-A, 3nm) |
| **Vector Extension** | AVX-512 (F, CD, BW, DQ, VL) | AVX-512 (F, CD, BW, DQ, VL) | ARM64 NEON (Advanced SIMD) |
| **Vector Registers** | 32 $\times$ 512-bit (`zmm0`–`zmm31`) | 32 $\times$ 512-bit (`zmm0`–`zmm31`) | 32 $\times$ 128-bit (`v0`–`v31`) |
| **Memory Subsystem** | 8-channel DDR4-3200 ECC | 12-channel DDR5-4800 ECC | On-package Unified Memory (UMA) |
| **Peak Mem Bandwidth** | $\approx 204\text{ GB/s}$ per socket | $\approx 460\text{ GB/s}$ aggregate | $\approx 410\text{ GB/s}$ on-package |
| **L1 Data Cache** | 48 KB per core | 32 KB per core | 128 KB per P-core |
| **L2 / Shared Cache** | 1.25 MB per core / 60 MB L3 | 1 MB per core / 384 MB L3 | 16 MB shared L2 / 32 MB SLC |

---

### 4.2 Comparative Roofline Model

$$\text{Peak}_{\text{compute}} = N_{\text{cores}} \times f_{\text{clk}} \times W_{\text{lanes}} \times \text{FMA}_{\text{units}} \times 2 \quad [\text{GFLOPS}]$$

$$P = \min\left( \text{Peak}_{\text{compute}}, \quad \text{Bandwidth}_{\text{mem}} \times I \right)$$

```
Throughput (GFLOPS)
   ▲
   │                              ┌───────────────── AMD Zen 4 AVX-512 Peak
   │               ┌──────────────┼───────────────── Intel Xeon AVX-512 Peak
   │              /              /
   │             /              / ┌───────────────── Apple M4 Max Peak (1536 GFLOPS)
   │            /              / /
   │           /              / / ◄── Memory Wall: Intel DDR4-3200 (204 GB/s)
   │          /              / /
   │         /  ◄───────────┼─┼────── Memory Wall: Apple M4 UMA (410 GB/s) & AMD DDR5 (460 GB/s)
   │        /               │ │
   └───────┴────────────────┴─┴──────────────────► Operational Intensity (Flops/Byte)
```

**Key Architectural Insight**:
Because analytical scan/filter queries typically possess low operational intensity ($I \le 0.5\text{ Flops/Byte}$), query throughput is governed primarily by memory bandwidth rather than peak vector ALUs. The $410\text{ GB/s}$ memory bus on Apple Silicon M4 Max and $460\text{ GB/s}$ DDR5 on AMD Genoa allow vectorized interpretation to sustain high data throughput without requiring the register-resident compilation needed on bandwidth-constrained DDR4 systems.

---

## 5. Formal Scientific Hypotheses & Validation Gates

### Hypothesis $H_1$ (Crossover Validity)
- **Statement**: Relational pipelines with Expression Complexity Index $ECI \ge 6.0$ achieve $\ge 2.2\times$ throughput speedup under JIT compilation compared to vectorized interpretation for $N > 10^6$ tuples, with compilation latency $T_{comp}$ being amortized within $N_{crossover} \le 5 \times 10^5$ tuples.
- **Validation Gate**: Pass if measured $N_{crossover}$ on ClickBench Q15-Q20 matches the predicted model value within a $15\%$ relative error margin.

### Hypothesis $H_2$ (Hardware Architecture Crossover Invariance Violation)
- **Statement**: The optimal JIT crossover point is not architecture-invariant:
  $$N_{crossover}^{\text{NEON}} \ge 2.5 \times N_{crossover}^{\text{AVX-512}}$$
- **Validation Gate**: Pass if empirical crossover measured on Apple M4 Max is at least $2.5\times$ higher than on x86-64 Xeon for identical expression trees.

### Hypothesis $H_3$ (Two-Tier Multi-Tier Amortization)
- **Statement**: An adaptive execution tier (starting immediately with vectorized morsels and hot-swapping asynchronously to JIT-compiled kernels via atomic function pointer swaps) incurs zero compile latency on short queries ($N < 10^5$) while achieving within $3\%$ of pure JIT steady-state execution on long queries ($N \ge 10^7$).
- **Validation Gate**: Pass if adaptive query runtime deviates by $< 3\%$ from static JIT on TPC-H Q1 (SF10) and is $\ge 10\times$ faster on small ad-hoc queries ($N = 10^4$).

---

## 6. Execution Command Reference

```bash
# 1. Microbenchmarks via Criterion
cargo bench --bench morsel_sizing
cargo bench --bench simd_kernels
cargo bench --bench jit_compile_latency

# 2. TPC-H SF1 - SF100 Runner
cargo run --release --bin engine_benchmark -- --suite tpch --scale 1 --query all
cargo run --release --bin engine_benchmark -- --suite tpch --scale 100 --query q1,q6,q9,q18

# 3. ClickBench 100M Runner
cargo run --release --bin engine_benchmark -- --suite clickbench --data ./data/hits.parquet --mode cold
cargo run --release --bin engine_benchmark -- --suite clickbench --data ./data/hits.parquet --mode hot

# 4. Strict Linter and Safety Verification
cargo clippy --all-targets --all-features -- -D warnings
MIRIFLAGS="-Zmiri-symbolic-alignment-check -Zmiri-strict-provenance" cargo miri test
```
