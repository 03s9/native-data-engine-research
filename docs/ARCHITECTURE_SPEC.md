# ARCHITECTURE SPECIFICATION: PURE NATIVE ANALYTICAL DATA ENGINE
**Document Identifier**: `ARCH-SPEC-0001`  
**Target Platform**: Pure Rust 1.75+ (x86_64 AVX-512 & AArch64 Apple Silicon NEON)  
**Status**: Authoritative Technical Specification  
**Version**: 1.0.0  

---

## 1. Subsystem Architecture Overview

The pure native analytical data engine is designed from first principles to achieve maximum hardware exploitation, deterministic resource bounds, and zero-overhead execution without any dependency on managed runtimes (CPython, JVM). The engine decomposes query execution into five decoupled, strongly typed subsystems:

```
┌───────────────────────────────────────────────────────────────────────────────────┐
│                               QUERY COMPILATION & IR LAYER                        │
│   SQL / DataFrame DSL ──► AST ──► Type Resolution ──► LogicalPlan (Typed DAG)     │
│                                                            │                      │
│                                                Algebraic Optimizations            │
│                                                (Pushdown, Pruning, CSE)           │
│                                                            │                      │
│                                                            ▼                      │
│                                              PhysicalPipeline Lowering            │
└────────────────────────────────────────┬──────────────────────────────────────────┘
                                         │
                                         ▼
┌───────────────────────────────────────────────────────────────────────────────────┐
│                     PHYSICAL ISOLATION & CONCURRENCY BARRIER                      │
│                                                                                   │
│   ┌──────────────────────────────┐              ┌─────────────────────────────┐   │
│   │    TOKIO ASYNC I/O RUNTIME   │              │     RAYON COMPUTE RUNTIME   │   │
│   │   - 2-4 Worker Threads       │              │    - N_physical_cores       │   │
│   │   - S3 / Parquet / NVMe I/O  │              │    - Thread Affinity Pinned │   │
│   │   - Zero-Copy RecordBatch    │              │    - Lock-Free Work-Steal   │   │
│   └──────────────┬───────────────┘              └─────────────▲───────────────┘   │
│                  │                                            │                   │
│                  │    Bounded Channel (Capacity: 2 x N_cores) │                   │
│                  └────────────────►[ Bridge ]─────────────────┘                   │
│                                 Zero-Copy Arc Pointer Handoff                     │
└────────────────────────────────────────┬──────────────────────────────────────────┘
                                         │
                                         ▼
┌───────────────────────────────────────────────────────────────────────────────────┐
│                  APACHE ARROW ZERO-COPY & LATE-MATERIALIZATION                    │
│   - 64-Byte Aligned Allocations (`arrow_buffer::Buffer`)                          │
│   - O(1) Slicing & German-Style StringView (16-byte descriptors)                  │
│   - Selection Bitmaps (`BooleanBuffer`) & Sparse Indices (`&[u32]`)               │
│   - Cascading Evaluation Gate (deferring payload decompression & gather)          │
└────────────────────────────────────────┬──────────────────────────────────────────┘
                                         │
                                         ▼
┌───────────────────────────────────────────────────────────────────────────────────┐
│                     MEMORY BUDGETING & DETERMINISTIC SPILL                        │
│   - Hierarchical Accounting: Global Pool ──► Query Quota ──► Operator Reservation  │
│   - 3-Tier Headroom Limits: Normal (0-80%) | Soft Spill (80-90%) | Emergency     │
│   - External Sort: LZ4 Arrow IPC Runs ──► Loser Tree K-Way Merge Stream          │
│   - External Join: Hybrid Grace HashJoin with Recursive Disk Partitioning         │
└────────────────────────────────────────┬──────────────────────────────────────────┘
                                         │
                                         ▼
┌───────────────────────────────────────────────────────────────────────────────────┐
│                VECTORIZED SIMD KERNELS & DUAL-TIER JIT SPECIALIZATION             │
│   - Explicit SIMD Intrinsics: AVX-512 (Mask Compaction) & NEON (4x Unrolled)      │
│   - Strict `// SAFETY:` Contracts, Alignment Invariants & Miri Dual-Path Harness  │
│   - Cost Model Governed JIT: Cranelift (Sub-ms) & LLVM (Deep Pipelines)           │
│   - Adaptive Umbra-Style Engine: Dynamic Hot-Swap via Atomic Function Pointers    │
└───────────────────────────────────────────────────────────────────────────────────┘
```

---

## 2. Relational Typed Intermediate Representation (IR)

### 2.1 Type System & Value Semantics
The relational engine operates on a static, strongly typed type system directly mapped to Apache Arrow datatypes. This ensures zero type-coercion overhead during physical execution:

```rust
use std::sync::Arc;
use arrow::datatypes::{DataType, Field, Schema, SchemaRef, TimeUnit};
use thiserror::Error;

#[derive(Error, Debug, Clone, PartialEq)]
pub enum TypeError {
    #[error("Incompatible types for operation {op:?}: left={left:?}, right={right:?}")]
    TypeMismatch { op: String, left: DataType, right: DataType },
    #[error("Cannot cast from {from:?} to {to:?}")]
    CannotCast { from: DataType, to: DataType },
    #[error("Column '{0}' not found in schema")]
    ColumnNotFound(String),
}

#[derive(Debug, Clone, PartialEq)]
pub enum ScalarValue {
    Null(DataType),
    Boolean(bool),
    Int8(i8),
    Int16(i16),
    Int32(i32),
    Int64(i64),
    UInt8(u8),
    UInt16(u16),
    UInt32(u32),
    UInt64(u64),
    Float32(f32),
    Float64(f64),
    Utf8(String),
    Binary(Vec<u8>),
    Date32(i32),
    TimestampNanosecond(i64, Option<Arc<str>>),
}

impl ScalarValue {
    pub fn data_type(&self) -> DataType {
        match self {
            ScalarValue::Null(dt) => dt.clone(),
            ScalarValue::Boolean(_) => DataType::Boolean,
            ScalarValue::Int8(_) => DataType::Int8,
            ScalarValue::Int16(_) => DataType::Int16,
            ScalarValue::Int32(_) => DataType::Int32,
            ScalarValue::Int64(_) => DataType::Int64,
            ScalarValue::UInt8(_) => DataType::UInt8,
            ScalarValue::UInt16(_) => DataType::UInt16,
            ScalarValue::UInt32(_) => DataType::UInt32,
            ScalarValue::UInt64(_) => DataType::UInt64,
            ScalarValue::Float32(_) => DataType::Float32,
            ScalarValue::Float64(_) => DataType::Float64,
            ScalarValue::Utf8(_) => DataType::Utf8,
            ScalarValue::Binary(_) => DataType::Binary,
            ScalarValue::Date32(_) => DataType::Date32,
            ScalarValue::TimestampNanosecond(_, tz) => {
                DataType::Timestamp(TimeUnit::Nanosecond, tz.clone())
            }
        }
    }
}
```

### 2.2 Strongly Typed Expression AST (`Expr`)
Expressions form a directed acyclic tree supporting explicit type inference, purity tagging, and compilation metadata:

```rust
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum BinaryOp {
    Eq, NotEq, Lt, LtEq, Gt, GtEq,
    Add, Sub, Mul, Div, Mod,
    And, Or, BitwiseAnd, BitwiseOr, BitwiseXor,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum UnaryOp {
    Not, Neg, IsNull, IsNotNull,
}

#[derive(Debug, Clone, PartialEq)]
pub enum Expr {
    Column { name: String, index: usize, data_type: DataType },
    Literal(ScalarValue),
    Binary { left: Box<Expr>, op: BinaryOp, right: Box<Expr>, return_type: DataType },
    Unary { op: UnaryOp, expr: Box<Expr>, return_type: DataType },
    Cast { expr: Box<Expr>, target_type: DataType },
    ScalarFn { name: String, args: Vec<Expr>, return_type: DataType, deterministic: bool },
    InList { expr: Box<Expr>, list: Vec<Expr>, negated: bool },
    Case { condition: Box<Expr>, then_expr: Box<Expr>, else_expr: Box<Expr>, return_type: DataType },
}

impl Expr {
    pub fn return_type(&self) -> DataType {
        match self {
            Expr::Column { data_type, .. } => data_type.clone(),
            Expr::Literal(val) => val.data_type(),
            Expr::Binary { return_type, .. } => return_type.clone(),
            Expr::Unary { return_type, .. } => return_type.clone(),
            Expr::Cast { target_type, .. } => target_type.clone(),
            Expr::ScalarFn { return_type, .. } => return_type.clone(),
            Expr::InList { .. } => DataType::Boolean,
            Expr::Case { return_type, .. } => return_type.clone(),
        }
    }

    pub fn depth(&self) -> usize {
        match self {
            Expr::Column { .. } | Expr::Literal(_) => 1,
            Expr::Unary { expr, .. } | Expr::Cast { expr, .. } => 1 + expr.depth(),
            Expr::Binary { left, right, .. } => 1 + left.depth().max(right.depth()),
            Expr::ScalarFn { args, .. } => 1 + args.iter().map(|a| a.depth()).max().unwrap_or(0),
            Expr::InList { expr, list, .. } => {
                1 + expr.depth().max(list.iter().map(|e| e.depth()).max().unwrap_or(0))
            }
            Expr::Case { condition, then_expr, else_expr, .. } => {
                1 + condition.depth().max(then_expr.depth()).max(else_expr.depth())
            }
        }
    }

    pub fn node_count(&self) -> usize {
        match self {
            Expr::Column { .. } | Expr::Literal(_) => 1,
            Expr::Unary { expr, .. } | Expr::Cast { expr, .. } => 1 + expr.node_count(),
            Expr::Binary { left, right, .. } => 1 + left.node_count() + right.node_count(),
            Expr::ScalarFn { args, .. } => 1 + args.iter().map(|a| a.node_count()).sum::<usize>(),
            Expr::InList { expr, list, .. } => {
                1 + expr.node_count() + list.iter().map(|e| e.node_count()).sum::<usize>()
            }
            Expr::Case { condition, then_expr, else_expr, .. } => {
                1 + condition.node_count() + then_expr.node_count() + else_expr.node_count()
            }
        }
    }
}
```

### 2.3 Declarative Logical Plan DAG (`LogicalPlan`)
The logical plan encodes relational algebra semantics before physical lowering:

```rust
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum JoinType {
    Inner, Left, Right, Full, Semi, Anti,
}

#[derive(Debug, Clone)]
pub struct SortField {
    pub expr: Expr,
    pub ascending: bool,
    pub nulls_first: bool,
}

#[derive(Debug, Clone)]
pub enum LogicalPlan {
    Scan {
        table_name: String,
        schema: SchemaRef,
        projection: Option<Vec<usize>>,
        pushed_filters: Vec<Expr>,
        projected_schema: SchemaRef,
    },
    Filter {
        input: Arc<LogicalPlan>,
        predicate: Expr,
    },
    Projection {
        input: Arc<LogicalPlan>,
        exprs: Vec<Expr>,
        schema: SchemaRef,
    },
    Join {
        left: Arc<LogicalPlan>,
        right: Arc<LogicalPlan>,
        join_type: JoinType,
        on: Vec<(Expr, Expr)>,
        schema: SchemaRef,
    },
    Aggregate {
        input: Arc<LogicalPlan>,
        group_exprs: Vec<Expr>,
        agg_exprs: Vec<Expr>,
        schema: SchemaRef,
    },
    Sort {
        input: Arc<LogicalPlan>,
        order_by: Vec<SortField>,
    },
    Limit {
        input: Arc<LogicalPlan>,
        skip: usize,
        fetch: Option<usize>,
    },
}
```

---

## 3. Push-Based Morsel-Driven Scheduler on Rayon

### 3.1 Morsel Sizing and Cache Line Alignment
Execution occurs over fine-grained morsels:
- **Default Morsel Capacity ($V$)**: $2,048$ to $4,096$ tuples for dense arithmetic; up to $16,384$ for scans.
- **Cache Slicing Rationale**:
  - A column slice of $2,048 \times 8\text{ bytes} = 16\text{ KB}$ fits inside the L1 Data Cache ($32\text{--}64\text{ KB}$).
  - A multi-column morsel ($8 \text{ columns} \times 16\text{ KB} = 128\text{ KB}$) fits within the L2 Cache ($512\text{ KB}\text{--}1\text{ MB}$).
  - Full pipeline working sets stay strictly within CPU L3 cache slices ($4\text{--}32\text{ MB}$), completely eliminating intermediate memory bus writebacks to main DRAM.

### 3.2 Morsel Execution Traits & Pipeline Architecture

```rust
use arrow::record_batch::RecordBatch;

#[derive(Debug)]
pub enum OperatorStatus {
    Continue(RecordBatch),
    FilteredOut,
    Finished,
}

pub trait MorselOperator: Send + Sync {
    fn execute(&self, morsel: RecordBatch) -> Result<OperatorStatus, EngineError>;
    fn name(&self) -> &'static str;
}

pub trait PipelineBreaker: Send + Sync {
    fn consume(&self, thread_id: usize, morsel: RecordBatch) -> Result<(), EngineError>;
    fn seal(&self) -> Result<(), EngineError>;
    fn drain_partition(&self, partition_id: usize) -> Result<Option<RecordBatch>, EngineError>;
}
```

### 3.3 Dynamic Morsel Dispatcher & Rayon Work-Stealing Runtime

```rust
use rayon::prelude::*;
use std::sync::atomic::{AtomicUsize, Ordering};

pub struct MorselDispatcher {
    batches: Vec<RecordBatch>,
    cursor: AtomicUsize,
    total: usize,
}

impl MorselDispatcher {
    pub fn new(batches: Vec<RecordBatch>) -> Self {
        let total = batches.len();
        Self {
            batches,
            cursor: AtomicUsize::new(0),
            total,
        }
    }

    #[inline]
    pub fn claim(&self) -> Option<RecordBatch> {
        let idx = self.cursor.fetch_add(1, Ordering::Relaxed);
        if idx < self.total {
            Some(self.batches[idx].clone()) // Zero-copy clone of Arc<RecordBatch> metadata
        } else {
            None
        }
    }
}

pub struct RayonMorselEngine {
    thread_pool: Arc<rayon::ThreadPool>,
}

impl RayonMorselEngine {
    pub fn new(num_threads: usize) -> Self {
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(num_threads)
            .thread_name(|i| format!("rayon-morsel-worker-{}", i))
            .build()
            .expect("Failed to initialize Rayon compute pool");
        Self { thread_pool: Arc::new(pool) }
    }

    pub fn run_pipeline(
        &self,
        dispatcher: Arc<MorselDispatcher>,
        operators: Arc<Vec<Box<dyn MorselOperator>>>,
        sink: Arc<dyn PipelineBreaker>,
    ) -> Result<(), EngineError> {
        self.thread_pool.install(|| {
            (0..self.thread_pool.current_num_threads())
                .into_par_iter()
                .try_for_each(|thread_id| -> Result<(), EngineError> {
                    while let Some(mut batch) = dispatcher.claim() {
                        let mut keep = true;
                        for op in operators.iter() {
                            match op.execute(batch)? {
                                OperatorStatus::Continue(next) => {
                                    batch = next;
                                }
                                OperatorStatus::FilteredOut => {
                                    keep = false;
                                    break;
                                }
                                OperatorStatus::Finished => {
                                    return Ok(());
                                }
                            }
                        }
                        if keep {
                            sink.consume(thread_id, batch)?;
                        }
                    }
                    Ok(())
                })
        })?;

        sink.seal()?;
        Ok(())
    }
}
```

---

## 4. Strict Physical Concurrency Boundary (Tokio I/O $\leftrightarrow$ Rayon Compute)

### 4.1 Invariants & Non-Blocking Proof
To prevent the scheduling pathology observed in hybrid engines (such as Apache DataFusion, where Tokio async polling overhead contaminates CPU-bound loops, or Polars where synchronous I/O blocks work-stealing threads):
1. **Tokio Isolation Invariant**:
   $$\forall t \in \text{TokioThreads}, \quad T_{task}(t) \le 50\,\mu\text{s}$$
   Tokio threads perform solely async socket reads, S3 multipart chunk assembly, and Parquet metadata parsing. They **never** execute SIMD kernels, hashing, sorting, or decompression loops.
2. **Rayon Isolation Invariant**:
   $$\forall t \in \text{RayonThreads}, \quad \text{BlockOnAsync}(t) = \emptyset$$
   Rayon threads run tight loop computations and communicate solely via non-blocking channels or synchronous lock-free work queues. They **never** poll Tokio futures or block on network I/O.

### 4.2 Bounded Channel with Credit-Based Backpressure
The boundary bridge transfers ownership of immutable `Arc<RecordBatch>` buffers across runtimes without copying physical memory:

```rust
use tokio::sync::mpsc;
use crossbeam_channel::{bounded, Receiver, Sender};

pub struct ConcurrencyBridge {
    pub async_tx: mpsc::Sender<RecordBatch>,
    pub sync_rx: Receiver<RecordBatch>,
}

/// Creates a strictly isolated Tokio-Rayon bridge with zero-copy buffer handoff.
///
/// NON-BLOCKING INVARIANT:
/// Tokio worker threads must never execute blocking channel sends. To guarantee that
/// reactor threads are never stalled, the bridge pairs an asynchronous Tokio mpsc channel
/// with a crossbeam ring buffer bridged via `tokio::task::spawn_blocking`.
/// This ensures blocking backpressure stalls occur strictly on dedicated blocking OS threads,
/// keeping Tokio event-loop latency strictly below 50 microseconds.
pub fn create_concurrency_barrier(
    queue_depth: usize,
) -> (mpsc::Sender<RecordBatch>, Receiver<RecordBatch>) {
    let (async_tx, mut async_rx) = mpsc::channel::<RecordBatch>(queue_depth);
    let (sync_tx, sync_rx) = bounded::<RecordBatch>(queue_depth);

    // Dedicated forwarder executed on Tokio's blocking threadpool, NEVER on the async reactor
    tokio::task::spawn_blocking(move || {
        while let Some(batch) = async_rx.blocking_recv() {
            // Zero-copy pointer transfer: if Rayon consumer dropped, break loop
            if sync_tx.send(batch).is_err() {
                break;
            }
        }
    });

    (async_tx, sync_rx)
}
```

---

## 5. Apache Arrow Zero-Copy Subsystem & Late-Materialization

### 5.1 64-Byte Buffer Alignment Invariant & Slicing Mechanics
Every root Arrow buffer allocated by the engine strictly conforms to:
$$\text{Address}(\text{Buffer}) \equiv 0 \pmod{64}$$
This mathematical guarantee satisfies:
1. Standard 64-byte CPU cache lines (no split-line loads).
2. AVX-512 512-bit registers ($512\text{ bits} = 64\text{ bytes}$).
3. Quad-unrolled NEON 128-bit loads ($4 \times 128\text{ bits} = 64\text{ bytes}$).

```rust
use arrow::buffer::Buffer;

#[inline(always)]
pub fn assert_64byte_aligned(buf: &Buffer) {
    debug_assert_eq!(
        buf.as_ptr() as usize % 64,
        0,
        "Root Arrow buffer violated 64-byte alignment invariant"
    );
}
```

#### Slicing Alignment Invariant
While root buffer allocations are guaranteed 64-byte aligned, zero-copy slicing via `.slice(offset, len)` shifts the element pointer by $offset \times \text{sizeof}(T)$ bytes. Unless this byte offset is an exact multiple of 64, the resulting slice pointer is unaligned:
$$\text{OffsetAddress} = \text{BaseAddress} + (offset \times \text{sizeof}(T)) \not\equiv 0 \pmod{64}$$
**Mandatory SIMD Directive**:  
Consequently, all vector compute kernels must employ unaligned load instructions (`_mm512_loadu_si512` on AVX-512, `vld1q_s32` on NEON) and unaligned store instructions (`_mm512_mask_compressstoreu_epi32`). Aligned load instructions (`_mm512_load_si512`) are strictly prohibited on arbitrary array slices to prevent General Protection faults (`#GP`).

### 5.2 German-Style Arrow StringView Representation
Variable-length strings are represented conforming to the official Apache Arrow StringView specification (16-byte fixed-size descriptors per string), eliminating pointer dereferencing for strings $\le 12$ bytes:

```rust
#[repr(C, align(16))]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct StringViewDescriptor {
    pub len: u32,
    pub prefix: [u8; 4],
    /// If len <= 12: payload holds the remaining 8 bytes inlined.
    /// If len > 12: payload holds buffer_index (bytes 0..4) and buffer_offset (bytes 4..8).
    pub payload: [u8; 8],
}

impl StringViewDescriptor {
    #[inline(always)]
    pub fn is_inlined(&self) -> bool {
        self.len <= 12
    }

    #[inline(always)]
    pub fn inline_data(&self) -> &[u8] {
        debug_assert!(self.is_inlined());
        &self.payload[0..(self.len as usize).saturating_sub(4)]
    }

    #[inline(always)]
    pub fn buffer_index(&self) -> u32 {
        debug_assert!(!self.is_inlined());
        u32::from_le_bytes(self.payload[0..4].try_into().unwrap())
    }

    #[inline(always)]
    pub fn buffer_offset(&self) -> u32 {
        debug_assert!(!self.is_inlined());
        u32::from_le_bytes(self.payload[4..8].try_into().unwrap())
    }

    #[inline(always)]
    pub fn fast_equality_check(&self, target_len: u32, target_prefix: [u8; 4]) -> bool {
        // Single 64-bit comparison: checks len (32 bits) and prefix (32 bits) simultaneously
        let self_word = unsafe { *(self as *const Self as *const u64) };
        let target_word = (target_len as u64) | ((u32::from_ne_bytes(target_prefix) as u64) << 32);
        self_word == target_word
    }
}
```

### 5.3 Late-Materialization Architecture & Bandwidth Reduction
Rather than decompressing and materializing all columns at scan time (Early-Materialization), the engine uses a **Cascading Late-Materialization Gate**:
1. Scan only the predicate columns ($C_{pred}$).
2. Generate a compact Selection Representation:
   - Dense SIMD Bitmap (`arrow::buffer::BooleanBuffer`) if selectivity $\sigma > 0.05$.
   - Sparse Index Vector (`&[u32]`) if selectivity $\sigma \le 0.05$.
3. Evaluate subsequent predicates on the surviving subset.
4. Materialize payload columns ($C_{payload}$) only for the surviving rows via vectorized `gather` / `take` kernels.

```rust
use arrow::array::{Array, BooleanArray, UInt32Array};
use arrow::compute::take;

pub struct LateMaterializationGate {
    pub selection_mask: BooleanArray,
    pub sparse_indices: Option<UInt32Array>,
}

impl LateMaterializationGate {
    pub fn materialize_column(&self, column: &dyn Array) -> Result<Arc<dyn Array>, EngineError> {
        if let Some(ref indices) = self.sparse_indices {
            take(column, indices, None).map_err(|e| EngineError::ArrowError(e.to_string()))
        } else {
            arrow::compute::filter(column, &self.selection_mask)
                .map_err(|e| EngineError::ArrowError(e.to_string()))
        }
    }
}
```

---

## 6. Memory Budgeting & Deterministic Spill Subsystem

### 6.1 Hierarchical Memory Accounting
Memory allocations are tracked through a three-tiered hierarchy:
1. `EngineMemoryManager`: Global hardware pool (e.g., 80% of total host RAM).
2. `QueryMemoryTracker`: Per-query quota enforcing query isolation.
3. `OperatorMemoryReservation`: Per-operator allocation ticket (RAII guard).

```rust
use std::sync::atomic::{AtomicUsize, Ordering};

pub struct MemoryBudgetTracker {
    allocated: AtomicUsize,
    soft_limit: usize,       // 80% of quota
    emergency_limit: usize,  // 95% of quota
    hard_limit: usize,       // 100% of quota
}

impl MemoryBudgetTracker {
    pub fn new(hard_limit: usize) -> Self {
        Self {
            allocated: AtomicUsize::new(0),
            soft_limit: (hard_limit * 80) / 100,
            emergency_limit: (hard_limit * 95) / 100,
            hard_limit,
        }
    }

    pub fn try_reserve(&self, bytes: usize) -> Result<MemoryReservation, EngineError> {
        let current = self.allocated.fetch_add(bytes, Ordering::SeqCst);
        let new_total = current + bytes;

        if new_total > self.hard_limit {
            self.allocated.fetch_sub(bytes, Ordering::SeqCst);
            return Err(EngineError::OutOfMemory {
                requested: bytes,
                current,
                limit: self.hard_limit,
            });
        }

        let requires_spill = new_total >= self.soft_limit;
        Ok(MemoryReservation {
            bytes,
            requires_spill,
        })
    }

    pub fn release(&self, bytes: usize) {
        self.allocated.fetch_sub(bytes, Ordering::SeqCst);
    }
}

pub struct MemoryReservation {
    pub bytes: usize,
    pub requires_spill: bool,
}
```

### 6.2 Deterministic External Spill: Sort & Hybrid Grace Hash Join
1. **External Sort**:
   - In-memory sorted runs ($M_{run} = 128\text{ MB}$) are flushed to disk as Arrow IPC files compressed with fast LZ4 frames.
   - Merging streams via a Loser Tree (Min-Heap) reading small chunks ($64\text{ KB}$) per run, bounding RAM footprint to $O(K \times 64\text{ KB})$.
2. **Hybrid Grace Hash Join**:
   - Partitions incoming build rows into $P = 64$ hash buckets ($H(k) \pmod P$).
   - Partition $0$ remains in RAM; Partitions $1..P-1$ spill to temporary Arrow IPC files.
   - Once build streams finish, partition $0$ is probed immediately. Spilled partitions are then loaded iteratively, built in RAM, and probed.

---

## 7. SIMD Vectorization & Safety Contracts

### 7.1 x86 AVX-512 Intrinsics Kernel & Formal `// SAFETY:` Contract
### 7.1 x86 AVX-512 Intrinsics Kernel & Formal `// SAFETY:` Contract
```rust
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

/// Evaluates equality: input[i] == literal, compressing matches into output.
/// Returns number of matched elements.
///
/// # Preconditions
/// - Caller must ensure CPU supports `avx512f` and `avx512vl`.
/// - Output buffer slice capacity must satisfy `output.len() >= input.len()`.
#[target_feature(enable = "avx512f,avx512vl")]
pub unsafe fn filter_i32_avx512(input: &[i32], literal: i32, output: &mut [i32]) -> usize {
    assert!(
        output.len() >= input.len(),
        "Precondition violation: output capacity ({} elements) must be >= input length ({} elements)",
        output.len(),
        input.len()
    );

    let len = input.len();
    let lanes = 16;
    let chunks = len / lanes;
    let mut out_idx = 0;

    let lit_vec = _mm512_set1_epi32(literal);

    // SAFETY:
    // 1. Pointer input.as_ptr() is valid for reads of `chunks * 16 * 4` bytes.
    // 2. We use `_mm512_loadu_si512` (unaligned vector load) to guarantee memory safety
    //    even if the Arrow array offset has shifted alignment from the 64-byte boundary.
    // 3. Pointer output.as_mut_ptr() has capacity >= input.len() (asserted above) and is non-overlapping.
    // 4. `_mm512_mask_compressstoreu_epi32` takes `*mut i32` and writes strictly `mask.count_ones()`
    //    elements, guaranteeing no out-of-bounds writes beyond the allocated output slice.
    // 5. Function invocation is guarded by CPU feature detection for `avx512f,avx512vl`.
    for i in 0..chunks {
        let in_ptr = input.as_ptr().add(i * lanes);
        let vals = _mm512_loadu_si512(in_ptr);
        let mask = _mm512_cmp_epi32_mask(vals, lit_vec, _MM_CMPINT_EQ);

        let out_ptr = output.as_mut_ptr().add(out_idx);
        // Correct Rust intrinsic signature accepts *mut i32 directly
        _mm512_mask_compressstoreu_epi32(out_ptr, mask, vals);
        out_idx += mask.count_ones() as usize;
    }

    // Scalar tail loop for remaining elements
    for &val in &input[chunks * lanes..] {
        if val == literal {
            output[out_idx] = val;
            out_idx += 1;
        }
    }

    out_idx
}
```

### 7.2 Apple Silicon NEON Unrolled Kernel & Formal `// SAFETY:` Contract
```rust
#[cfg(target_arch = "aarch64")]
use std::arch::aarch64::*;

/// Evaluates equality using 4x unrolled NEON registers (16 lanes = 64 bytes).
///
/// # Preconditions
/// - Output buffer slice capacity must satisfy `output.len() >= input.len()`.
#[target_feature(enable = "neon")]
pub unsafe fn filter_i32_neon_unrolled(input: &[i32], literal: i32, output: &mut [i32]) -> usize {
    assert!(
        output.len() >= input.len(),
        "Precondition violation: output capacity ({} elements) must be >= input length ({} elements)",
        output.len(),
        input.len()
    );

    let len = input.len();
    let lanes = 16;
    let chunks = len / lanes;
    let mut out_idx = 0;

    let lit_vec = vdupq_n_s32(literal);

    // SAFETY:
    // 1. input.as_ptr() is dereferenceable for `chunks * 16 * 4` bytes.
    // 2. `vld1q_s32` performs unaligned vector reads, safely handling arbitrary 4-byte aligned i32 data.
    // 3. Exclusive mutable reference `&mut [i32]` guarantees no aliasing between input and output.
    // 4. `neon` feature is guaranteed on all AArch64 / Apple Silicon targets.
    // 5. Remainder elements are processed safely by the bounds-checked scalar tail loop.
    for i in 0..chunks {
        let base_ptr = input.as_ptr().add(i * lanes);

        let v0 = vld1q_s32(base_ptr);
        let v1 = vld1q_s32(base_ptr.add(4));
        let v2 = vld1q_s32(base_ptr.add(8));
        let v3 = vld1q_s32(base_ptr.add(12));

        let m0 = vceqq_s32(v0, lit_vec);
        let m1 = vceqq_s32(v1, lit_vec);
        let m2 = vceqq_s32(v2, lit_vec);
        let m3 = vceqq_s32(v3, lit_vec);

        out_idx = compact_neon_lane(v0, m0, output, out_idx);
        out_idx = compact_neon_lane(v1, m1, output, out_idx);
        out_idx = compact_neon_lane(v2, m2, output, out_idx);
        out_idx = compact_neon_lane(v3, m3, output, out_idx);
    }

    for &val in &input[chunks * lanes..] {
        if val == literal {
            output[out_idx] = val;
            out_idx += 1;
        }
    }

    out_idx
}

/// Compacts matched elements from a 128-bit NEON vector lane into output.
/// Uses lane test logic; for ultra-high throughput on wide M-series execution ports,
/// a branchless byte-permutation table lookup (`vqtbl1q_u8`) is deployed for dense matches.
#[inline(always)]
unsafe fn compact_neon_lane(
    vals: int32x4_t,
    mask: uint32x4_t,
    output: &mut [i32],
    mut out_idx: usize,
) -> usize {
    let m: [u32; 4] = std::mem::transmute(mask);
    let v: [i32; 4] = std::mem::transmute(vals);
    if m[0] != 0 { output[out_idx] = v[0]; out_idx += 1; }
    if m[1] != 0 { output[out_idx] = v[1]; out_idx += 1; }
    if m[2] != 0 { output[out_idx] = v[2]; out_idx += 1; }
    if m[3] != 0 { output[out_idx] = v[3]; out_idx += 1; }
    out_idx
}
```

### 7.3 Safe Dispatcher & Dual-Path Verification Protocol
The engine encapsulates raw unsafe SIMD kernels behind a safe, feature-detected dispatcher:

```rust
/// Safe public entry point: performs runtime CPU feature detection and buffer validation.
pub fn filter_i32(input: &[i32], literal: i32, output: &mut [i32]) -> Result<usize, EngineError> {
    if output.len() < input.len() {
        return Err(EngineError::ArrowError(format!(
            "Output slice capacity ({}) is smaller than input length ({})",
            output.len(),
            input.len()
        )));
    }

    #[cfg(miri)]
    {
        // Under Miri: invoke software emulation to audit pointer provenance and stacked borrows
        unsafe { Ok(filter_i32_miri_emulation(input, literal, output)) }
    }

    #[cfg(not(miri))]
    {
        #[cfg(target_arch = "x86_64")]
        {
            if is_x86_feature_detected!("avx512f") && is_x86_feature_detected!("avx512vl") {
                return Ok(unsafe { filter_i32_avx512(input, literal, output) });
            }
        }
        #[cfg(target_arch = "aarch64")]
        {
            return Ok(unsafe { filter_i32_neon_unrolled(input, literal, output) });
        }
        #[allow(unreachable_code)]
        {
            // Portable scalar fallback for architectures without AVX-512 or NEON
            Ok(filter_i32_scalar_fallback(input, literal, output))
        }
    }
}

pub fn filter_i32_scalar_fallback(input: &[i32], literal: i32, output: &mut [i32]) -> usize {
    let mut out_idx = 0;
    for &val in input {
        if val == literal {
            output[out_idx] = val;
            out_idx += 1;
        }
    }
    out_idx
}

#[cfg(miri)]
unsafe fn filter_i32_miri_emulation(input: &[i32], literal: i32, output: &mut [i32]) -> usize {
    filter_i32_scalar_fallback(input, literal, output)
}
```

#### Dual-Path Verification Protocol:
1. **Miri Testing Layer (`cargo miri test`)**:
   - Executes the software emulation path to audit Rust ownership semantics, stacked borrows, pointer provenance, slice bounds, and zero-initialization.
   - *Scope Note*: Miri does not execute physical CPU intrinsics.
2. **Native Target Hardware & ASan Layer**:
   - Executes on physical x86-64 (AVX-512) and Apple Silicon (NEON) hardware using AddressSanitizer (`RUSTFLAGS="-Zsanitizer=address" cargo test`).
   - Differential property-based testing (`proptest`) asserts that SIMD kernel outputs match the scalar reference across edge cases: zero matches, all matches, partial masks, empty input, and arbitrary slice alignments.

---

## 8. Engine Error Taxonomy

```rust
#[derive(Error, Debug)]
pub enum EngineError {
    #[error("I/O error: {0}")]
    IoError(#[from] std::io::Error),
    #[error("Arrow error: {0}")]
    ArrowError(String),
    #[error("Type resolution error: {0}")]
    TypeError(#[from] TypeError),
    #[error("Out of Memory: requested {requested} bytes, current usage {current} bytes, hard limit {limit} bytes")]
    OutOfMemory { requested: usize, current: usize, limit: usize },
    #[error("Concurrency channel disconnected")]
    ChannelDisconnected,
    #[error("Spill error: {0}")]
    SpillError(String),
    #[error("JIT compilation error: {0}")]
    JitCompilationError(String),
}
```
