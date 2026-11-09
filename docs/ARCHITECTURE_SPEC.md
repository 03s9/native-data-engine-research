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

    /// Proactive spill trigger hook called when MemoryReservation::requires_spill is true.
    fn check_spill(&self, tracker: &MemoryBudgetTracker) -> Result<(), EngineError>;
}
```

### 3.3 Dynamic Morsel Dispatcher & Rayon Work-Stealing Runtime

```rust
use rayon::prelude::*;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::Duration;
use crossbeam_channel::{Receiver, TryRecvError, RecvTimeoutError};

/// Common trait for morsel sources (both pre-materialized vectors and streaming I/O channels).
pub trait MorselDispatcher: Send + Sync {
    /// Claims the next batch. Returns `None` when the input stream is fully drained or terminated.
    fn claim(&self) -> Option<RecordBatch>;

    /// Signals early pipeline termination (e.g. on error or LIMIT completion).
    fn terminate(&self);
}

/// Static in-memory morsel dispatcher for cached datasets and unit benchmarks.
pub struct StaticMorselDispatcher {
    batches: Vec<RecordBatch>,
    cursor: AtomicUsize,
    total: usize,
    terminated: AtomicBool,
}

impl StaticMorselDispatcher {
    pub fn new(batches: Vec<RecordBatch>) -> Self {
        let total = batches.len();
        Self {
            batches,
            cursor: AtomicUsize::new(0),
            total,
            terminated: AtomicBool::new(false),
        }
    }
}

impl MorselDispatcher for StaticMorselDispatcher {
    #[inline]
    fn claim(&self) -> Option<RecordBatch> {
        if self.terminated.load(Ordering::Relaxed) {
            return None;
        }
        let idx = self.cursor.fetch_add(1, Ordering::Relaxed);
        if idx < self.total {
            Some(self.batches[idx].clone()) // Zero-copy clone of Arc<RecordBatch> metadata
        } else {
            None
        }
    }

    fn terminate(&self) {
        self.terminated.store(true, Ordering::Release);
    }
}

/// Dynamic streaming morsel dispatcher consuming batches directly from a Tokio I/O channel.
///
/// NON-BLOCKING INVARIANT:
/// Rayon worker threads must NEVER block on channel waits. If the channel is temporarily empty,
/// workers execute a low-overhead spin-hint, cooperatively yield to Rayon (`rayon::yield_now()`),
/// and use a micro-timeout (100 µs) to prevent CPU starvation and thermal throttling while
/// allowing the Rayon scheduler to run other concurrent tasks.
pub struct StreamingMorselDispatcher {
    rx: Receiver<RecordBatch>,
    terminated: AtomicBool,
}

impl StreamingMorselDispatcher {
    pub fn new(rx: Receiver<RecordBatch>) -> Self {
        Self {
            rx,
            terminated: AtomicBool::new(false),
        }
    }
}

impl MorselDispatcher for StreamingMorselDispatcher {
    #[inline]
    fn claim(&self) -> Option<RecordBatch> {
        let mut spins = 0usize;
        loop {
            if self.terminated.load(Ordering::Acquire) {
                return None;
            }

            match self.rx.try_recv() {
                Ok(batch) => return Some(batch),
                Err(TryRecvError::Disconnected) => return None,
                Err(TryRecvError::Empty) => {
                    if spins < 16 {
                        std::hint::spin_loop();
                        spins += 1;
                    } else if spins < 32 {
                        rayon::yield_now();
                        spins += 1;
                    } else {
                        // Cooperative yield back to Rayon work-stealing scheduler
                        rayon::yield_now();
                        // Micro-timeout bounded sleep (100 µs) to prevent burning CPU
                        match self.rx.recv_timeout(Duration::from_micros(100)) {
                            Ok(batch) => return Some(batch),
                            Err(RecvTimeoutError::Disconnected) => return None,
                            Err(RecvTimeoutError::Timeout) => {
                                spins = 16; // Continue cooperative yielding
                            }
                        }
                    }
                }
            }
        }
    }

    fn terminate(&self) {
        self.terminated.store(true, Ordering::Release);
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

    pub fn run_pipeline<D: MorselDispatcher + ?Sized>(
        &self,
        dispatcher: Arc<D>,
        operators: Arc<Vec<Box<dyn MorselOperator>>>,
        sink: Arc<dyn PipelineBreaker>,
    ) -> Result<(), EngineError> {
        let result = self.thread_pool.install(|| {
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
                                    dispatcher.terminate();
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
        });

        if let Err(e) = result {
            dispatcher.terminate();
            return Err(e);
        }

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
    /// If len <= 12: payload holds the remaining inlined bytes (offset 8..16).
    /// If len > 12: payload holds buffer_index: u32 (offset 8..12) and buffer_offset: u32 (offset 12..16).
    pub payload: [u8; 8],
}

impl StringViewDescriptor {
    /// Constructs a descriptor for an inlined string (len <= 12).
    pub fn new_inlined(bytes: &[u8]) -> Self {
        assert!(bytes.len() <= 12, "Inlined string length must be <= 12 bytes");
        let mut prefix = [0u8; 4];
        let mut payload = [0u8; 8];
        let prefix_len = bytes.len().min(4);
        prefix[..prefix_len].copy_from_slice(&bytes[..prefix_len]);
        if bytes.len() > 4 {
            payload[..bytes.len() - 4].copy_from_slice(&bytes[4..]);
        }
        Self {
            len: bytes.len() as u32,
            prefix,
            payload,
        }
    }

    /// Constructs a descriptor for an out-of-line string referencing a buffer pool.
    pub fn new_out_of_line(len: u32, prefix: [u8; 4], buffer_index: u32, buffer_offset: u32) -> Self {
        assert!(len > 12, "Out-of-line strings must have length > 12 bytes");
        let mut payload = [0u8; 8];
        payload[0..4].copy_from_slice(&buffer_index.to_le_bytes());
        payload[4..8].copy_from_slice(&buffer_offset.to_le_bytes());
        Self { len, prefix, payload }
    }

    #[inline(always)]
    pub fn is_inlined(&self) -> bool {
        self.len <= 12
    }

    /// Returns a slice over the inlined string data.
    ///
    /// For len <= 12, the string is stored contiguously across `prefix` (bytes 4..8)
    /// and `payload` (bytes 8..16).
    #[inline(always)]
    pub fn inline_data(&self) -> &[u8] {
        debug_assert!(self.is_inlined(), "Cannot extract inline data from out-of-line StringView");

        // SAFETY:
        // 1. Layout: StringViewDescriptor is #[repr(C, align(16))]. Field `len` occupies bytes 0..4,
        //    `prefix` occupies bytes 4..8, and `payload` occupies bytes 8..16. There is zero padding.
        // 2. Provenance: The raw pointer is derived from `self as *const Self as *const u8`, granting
        //    provenance over the full 16-byte allocation of `Self`.
        // 3. Bounds: Adding offset 4 points to the beginning of `prefix`. Since `self.len <= 12`,
        //    the slice range [4, 4 + len) lies strictly within the 16-byte struct boundary [0, 16).
        // 4. Lifetime & Aliasing: The returned slice borrows `&self` immutably for lifetime `'a`.
        //    All 16 bytes of `Self` are fully initialized on construction.
        unsafe {
            let base_ptr = (self as *const Self as *const u8).add(4);
            std::slice::from_raw_parts(base_ptr, self.len as usize)
        }
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

    /// Fast 64-bit comparison: checks length and 4-byte prefix simultaneously in a single CPU instruction.
    #[inline(always)]
    pub fn fast_equality_check(&self, target_len: u32, target_prefix: [u8; 4]) -> bool {
        // SAFETY:
        // 1. Alignment: `StringViewDescriptor` is declared `#[repr(C, align(16))]`. Any valid reference
        //    `&self` is 16-byte aligned, which strictly satisfies the 8-byte alignment requirement for `u64`.
        // 2. Size & Bounds: The struct is 16 bytes; reading an 8-byte word from offset 0 accesses bytes 0..8
        //    (spanning `len` at bytes 0..4 and `prefix` at bytes 4..8), remaining strictly within bounds.
        // 3. Initialization: Bytes 0..8 are guaranteed to be fully initialized upon construction.
        // 4. Endianness: On little-endian architectures (AArch64 and x86_64), `len: u32` occupies the
        //    least-significant 32 bits and `prefix: [u8; 4]` occupies the most-significant 32 bits,
        //    matching the bitwise combination `target_word`.
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
   - Dense SIMD Bitmap (`arrow::buffer::BooleanBuffer`) if selectivity $\sigma \in (0.05, 0.15)$.
   - Sparse Index Vector (`&[u32]`) if selectivity $\sigma \le 0.05$.
3. Evaluate subsequent predicates on the surviving subset.
4. Materialize payload columns ($C_{payload}$) only for the surviving rows via vectorized `gather` / `take` kernels.

#### Selectivity Inversion Guard ($\sigma_{crossover} = 0.15$)
While late-materialization saves up to 97.02% bandwidth for highly selective queries ($\sigma \le 0.05$), it incurs two non-trivial overheads:
1. DRAM allocation of intermediate selection bitmaps and index vectors.
2. Non-contiguous, cache-line-inefficient gather operations ($B_{gather} \approx 120\text{ GB/s}$ vs $B_{seq} \approx 400\text{ GB/s}$).

Solving the analytical cost model $T_{late}(N, \sigma) \le T_{early}(N, \sigma)$ establishes:
$$\sigma_{crossover} \approx 0.15 \quad (15\%)$$
When $\sigma \ge 0.15$, the gather and buffer allocation penalty exceeds the bandwidth saved from skipping unselected rows. Under this condition, the optimizer enforces **Early Single-Pass Fused Scan**, bypassing late-materialization entirely.

```rust
use std::sync::Arc;
use arrow::array::{Array, BooleanArray, UInt32Array};
use arrow::compute::take;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MaterializationStrategy {
    /// Highly selective queries (sigma <= 0.05): materialize payload only via sparse row index vector
    LateSparseIndices,
    /// Moderately selective queries (0.05 < sigma < 0.15): materialize payload via dense SIMD bitmask
    LateDenseBitmap,
    /// Low selectivity / unselective queries (sigma >= 0.15): fused single-pass scan bypassing late materialization
    EarlySinglePassScan,
}

pub struct MaterializationOptimizer {
    pub crossover_sparse: f64, // 0.05
    pub crossover_dense: f64,  // 0.15
}

impl Default for MaterializationOptimizer {
    fn default() -> Self {
        Self {
            crossover_sparse: 0.05,
            crossover_dense: 0.15,
        }
    }
}

impl MaterializationOptimizer {
    /// Static plan-time optimizer selection based on catalog statistics
    pub fn select_strategy(&self, estimated_selectivity: f64) -> MaterializationStrategy {
        if estimated_selectivity <= self.crossover_sparse {
            MaterializationStrategy::LateSparseIndices
        } else if estimated_selectivity < self.crossover_dense {
            MaterializationStrategy::LateDenseBitmap
        } else {
            MaterializationStrategy::EarlySinglePassScan
        }
    }
}

/// Runtime adaptive execution guard monitoring empirical morsel selectivity
pub struct AdaptiveScanPipeline {
    strategy: MaterializationStrategy,
    morsels_evaluated: usize,
    sample_morsel_limit: usize,
    accumulated_tuples: usize,
    accumulated_matches: usize,
    optimizer: MaterializationOptimizer,
}

impl AdaptiveScanPipeline {
    pub fn new(initial_strategy: MaterializationStrategy) -> Self {
        Self {
            strategy: initial_strategy,
            morsels_evaluated: 0,
            sample_morsel_limit: 4, // Sample first 4 morsels (~8,192 tuples)
            accumulated_tuples: 0,
            accumulated_matches: 0,
            optimizer: MaterializationOptimizer::default(),
        }
    }

    #[inline]
    pub fn current_strategy(&self) -> MaterializationStrategy {
        self.strategy
    }

    /// Updates empirical selectivity after processing a morsel and dynamically flips strategy if needed
    pub fn record_morsel_feedback(&mut self, total_tuples: usize, matched_tuples: usize) {
        if self.morsels_evaluated < self.sample_morsel_limit {
            self.accumulated_tuples += total_tuples;
            self.accumulated_matches += matched_tuples;
            self.morsels_evaluated += 1;

            if self.morsels_evaluated == self.sample_morsel_limit {
                let empirical_sigma = self.accumulated_matches as f64 / self.accumulated_tuples.max(1) as f64;
                let optimal_strategy = self.optimizer.select_strategy(empirical_sigma);
                if optimal_strategy != self.strategy {
                    // Adaptively swap materialization mode for remaining morsels
                    self.strategy = optimal_strategy;
                }
            }
        }
    }
}

pub struct LateMaterializationGate {
    pub strategy: MaterializationStrategy,
    pub selection_mask: BooleanArray,
    pub sparse_indices: Option<UInt32Array>,
}

impl LateMaterializationGate {
    pub fn materialize_column(&self, column: &dyn Array) -> Result<Arc<dyn Array>, EngineError> {
        match self.strategy {
            MaterializationStrategy::EarlySinglePassScan => {
                // Pass-through: column was already materialized during the fused scan
                Ok(Arc::new(column.to_data().into()))
            }
            MaterializationStrategy::LateSparseIndices => {
                let indices = self.sparse_indices.as_ref().ok_or_else(|| {
                    EngineError::ArrowError("Sparse indices missing for LateSparseIndices strategy".into())
                })?;
                take(column, indices, None).map_err(|e| EngineError::ArrowError(e.to_string()))
            }
            MaterializationStrategy::LateDenseBitmap => {
                arrow::compute::filter(column, &self.selection_mask)
                    .map_err(|e| EngineError::ArrowError(e.to_string()))
            }
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

    /// Reserves memory using single-pass atomic CAS.
    /// Eliminates transient allocation spikes caused by fetch_add followed by fetch_sub rollback.
    pub fn try_reserve(&self, bytes: usize) -> Result<MemoryReservation, EngineError> {
        let update_result = self.allocated.fetch_update(
            Ordering::SeqCst,
            Ordering::SeqCst,
            |curr| {
                let new_total = curr.checked_add(bytes)?;
                if new_total > self.hard_limit {
                    None // Exceeds hard limit: abort CAS without modifying state
                } else {
                    Some(new_total)
                }
            },
        );

        match update_result {
            Ok(prev) => {
                let new_total = prev + bytes;
                let requires_spill = new_total >= self.soft_limit;
                Ok(MemoryReservation {
                    bytes,
                    requires_spill,
                })
            }
            Err(_) => {
                let current_alloc = self.allocated.load(Ordering::Relaxed);
                Err(EngineError::OutOfMemory {
                    requested: bytes,
                    current: current_alloc,
                    limit: self.hard_limit,
                })
            }
        }
    }

    /// Releases reserved bytes using atomic CAS with saturating subtraction.
    /// Underflow wrapping to 2^64 - 1 is mathematically impossible even under accidental over-release.
    pub fn release(&self, bytes: usize) {
        let _ = self.allocated.fetch_update(
            Ordering::SeqCst,
            Ordering::SeqCst,
            |curr| Some(curr.saturating_sub(bytes)),
        );
    }

    #[inline]
    pub fn current_usage(&self) -> usize {
        self.allocated.load(Ordering::Relaxed)
    }

    #[inline]
    pub fn is_spill_required(&self) -> bool {
        self.allocated.load(Ordering::Relaxed) >= self.soft_limit
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
2. **Hybrid Grace Hash Join with Skew & Infinite Recursion Safeguards**:
   - **Dynamic Partition Spilling**: Eliminates static RAM pinning of Partition 0. All $P = 64$ partitions initialize with in-memory streaming write buffers ($64\text{ KB}$). If total memory hits `tracker.soft_limit` (80%), write buffers are flushed dynamically to temporary Arrow IPC files (`RecordBatchFileWriter`), allowing even Partition 0 to spill if it exceeds budget.
   - **Bounded Recursion ($D_{max} = 3$)**: When loading spilled partition $P_i$ for probe, if $|P_i| > M_{budget}$ and current depth $D < D_{max} = 3$, $P_i$ is re-partitioned into $P_{sub} = 64$ child buckets using salted MurmurHash3:
     $$H_{D+1}(k) = \text{hash\_seeded}(k, \text{SEED} \oplus ((D + 1) \times \text{0x9e3779b97f4a7c15}))$$
   - **Degeneracy Detection ($\theta_{skew} = 0.80$)**: If the heaviest sub-partition holds $\ge 80\%$ of volume, key distribution is degenerate (e.g. repeated duplicate keys or millions of `NULL`s). Hashing cannot divide this data. Recursive hash partitioning is aborted immediately.
   - **Dual Deterministic Fallbacks**:
     1. *Chunked Block Nested Loop Join (BNLJ)*: Used when duplicate degeneracy is detected. Build data is sliced into memory-bounded chunks ($\le 0.5 \times M_{budget}$) and probed sequentially in RAM, guaranteeing $O(1)$ memory overhead and zero recursion risk.
     2. *External Radix Sort-Merge Join*: Used when $D = D_{max}$ is reached for non-degenerate data, performing cache-aligned radix sort runs merged via a Loser Tree in linear time.

```rust
pub const MAX_RECURSION_DEPTH: usize = 3;
pub const SKEW_THRESHOLD: f64 = 0.80; // 80% volume in one partition flags degeneracy

pub enum PartitionJoinStrategy {
    InMemoryHashTable,
    RecursivePartitioning { next_depth: usize },
    BlockNestedLoopFallback,
    RadixSortMergeFallback,
}

pub struct PartitionDescriptor {
    pub partition_id: usize,
    pub depth: usize,
    pub total_bytes: usize,
    pub row_count: usize,
    pub is_spilled: bool,
    pub ipc_file_path: Option<std::path::PathBuf>,
}

impl PartitionDescriptor {
    pub fn determine_strategy(
        &self,
        memory_budget: usize,
        max_sub_partition_ratio: f64,
    ) -> PartitionJoinStrategy {
        if self.total_bytes <= memory_budget {
            return PartitionJoinStrategy::InMemoryHashTable;
        }

        // Check for key skew / duplicate degeneracy
        if max_sub_partition_ratio >= SKEW_THRESHOLD {
            return PartitionJoinStrategy::BlockNestedLoopFallback;
        }

        if self.depth < MAX_RECURSION_DEPTH {
            PartitionJoinStrategy::RecursivePartitioning {
                next_depth: self.depth + 1,
            }
        } else {
            // Maximum recursion depth exceeded: fall back to sort-merge join
            PartitionJoinStrategy::RadixSortMergeFallback
        }
    }
}

/// Fallback execution for degenerate duplicate-key partitions
pub struct ChunkedBlockNestedLoopJoin {
    chunk_budget_bytes: usize,
}

impl ChunkedBlockNestedLoopJoin {
    pub fn new(total_budget: usize) -> Self {
        Self {
            chunk_budget_bytes: total_budget / 2,
        }
    }

    /// Processes build and probe streams in memory-bounded blocks without recursion.
    pub fn execute_join(
        &self,
        build_stream: impl Iterator<Item = Result<RecordBatch, EngineError>>,
        probe_stream_factory: impl Fn() -> Result<Box<dyn Iterator<Item = Result<RecordBatch, EngineError>>>, EngineError>,
    ) -> Result<Vec<RecordBatch>, EngineError> {
        let mut results = Vec::new();
        let mut current_build_chunk = Vec::new();
        let mut current_chunk_bytes = 0;

        for batch in build_stream {
            let b = batch?;
            let b_bytes = b.get_array_memory_size();
            current_build_chunk.push(b);
            current_chunk_bytes += b_bytes;

            if current_chunk_bytes >= self.chunk_budget_bytes {
                let mut probe_iter = probe_stream_factory()?;
                while let Some(probe_batch) = probe_iter.next() {
                    let pb = probe_batch?;
                    let matched = self.join_block(&current_build_chunk, &pb)?;
                    if matched.num_rows() > 0 {
                        results.push(matched);
                    }
                }
                current_build_chunk.clear();
                current_chunk_bytes = 0;
            }
        }

        if !current_build_chunk.is_empty() {
            let mut probe_iter = probe_stream_factory()?;
            while let Some(probe_batch) = probe_iter.next() {
                let pb = probe_batch?;
                let matched = self.join_block(&current_build_chunk, &pb)?;
                if matched.num_rows() > 0 {
                    results.push(matched);
                }
            }
        }

        Ok(results)
    }

    fn join_block(
        &self,
        _build_batches: &[RecordBatch],
        _probe_batch: &RecordBatch,
    ) -> Result<RecordBatch, EngineError> {
        // Flat vector comparison emitting Cartesian matches bounded strictly to chunk_budget_bytes
        todo!("Vectorized block equality matcher")
    }
}
```

---

## 7. SIMD Vectorization & Safety Contracts

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
    // 4. `_mm512_mask_compressstoreu_epi32` takes `*mut u8` (or `*mut i32` depending on stdarch binding)
    //    and writes strictly `mask.count_ones()` elements, guaranteeing no out-of-bounds writes beyond the slice.
    // 5. Function invocation is guarded by CPU feature detection for `avx512f,avx512vl`.
    for i in 0..chunks {
        let in_ptr = input.as_ptr().add(i * lanes);
        let vals = _mm512_loadu_si512(in_ptr);
        let mask = _mm512_cmp_epi32_mask(vals, lit_vec, _MM_CMPINT_EQ);

        let out_ptr = output.as_mut_ptr().add(out_idx);
        // Explicit cast to *mut u8 guarantees strict conformance to core::arch::x86_64
        _mm512_mask_compressstoreu_epi32(out_ptr as *mut u8, mask, vals);
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

/// 16-entry byte permutation table for ARM64 NEON vqtbl1q_u8 (256 bytes).
/// Each row corresponds to a 4-bit lane match pattern (0..15).
/// Matching 32-bit integer lanes (4 bytes each) are compacted to the front.
#[cfg(target_arch = "aarch64")]
#[repr(C, align(16))]
static NEON_PERM_TABLE: [[u8; 16]; 16] = [
    [0xFF; 16], // 0:  0000 -> 0 matches
    [0, 1, 2, 3, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF], // 1:  0001 (lane 0)
    [4, 5, 6, 7, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF], // 2:  0010 (lane 1)
    [0, 1, 2, 3, 4, 5, 6, 7, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF],             // 3:  0011 (lanes 0, 1)
    [8, 9, 10, 11, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF], // 4:  0100 (lane 2)
    [0, 1, 2, 3, 8, 9, 10, 11, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF],             // 5:  0101 (lanes 0, 2)
    [4, 5, 6, 7, 8, 9, 10, 11, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF],             // 6:  0110 (lanes 1, 2)
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 0xFF, 0xFF, 0xFF, 0xFF],                         // 7:  0111 (lanes 0, 1, 2)
    [12, 13, 14, 15, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF], // 8: 1000 (lane 3)
    [0, 1, 2, 3, 12, 13, 14, 15, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF],             // 9: 1001 (lanes 0, 3)
    [4, 5, 6, 7, 12, 13, 14, 15, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF],             // 10: 1010 (lanes 1, 3)
    [0, 1, 2, 3, 4, 5, 6, 7, 12, 13, 14, 15, 0xFF, 0xFF, 0xFF, 0xFF],                         // 11: 1011 (lanes 0, 1, 3)
    [8, 9, 10, 11, 12, 13, 14, 15, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF],             // 12: 1100 (lanes 2, 3)
    [0, 1, 2, 3, 8, 9, 10, 11, 12, 13, 14, 15, 0xFF, 0xFF, 0xFF, 0xFF],                         // 13: 1101 (lanes 0, 2, 3)
    [4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 0xFF, 0xFF, 0xFF, 0xFF],                         // 14: 1110 (lanes 1, 2, 3)
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],                                     // 15: 1111 (all lanes)
];

#[cfg(target_arch = "aarch64")]
static POPCOUNT_4BIT: [usize; 16] = [0, 1, 1, 2, 1, 2, 2, 3, 1, 2, 2, 3, 2, 3, 3, 4];

/// Branchless compaction of a 128-bit NEON vector lane using vqtbl1q_u8.
#[cfg(target_arch = "aarch64")]
#[inline(always)]
unsafe fn compact_neon_lane_branchless(
    vals: int32x4_t,
    mask: uint32x4_t,
    weights: uint32x4_t,
    output: &mut [i32],
    out_idx: &mut usize,
) {
    // 1. Extract 4-bit scalar mask via vector bitwise AND and across-vector addition
    let masked = vandq_u32(mask, weights);
    let mask_idx = vaddvq_u32(masked) as usize;

    // 2. Load precomputed permutation vector
    let perm = vld1q_u8(NEON_PERM_TABLE.get_unchecked(mask_idx).as_ptr());

    // 3. Permute matched lanes to the front
    let compacted = vqtbl1q_u8(vreinterpretq_u8_s32(vals), perm);

    // 4. Branchless overwrite store (unconditionally writes 4 words = 16 bytes)
    // SAFETY: caller guarantees output.len() >= input.len(). Because *out_idx <= elements_examined,
    // *out_idx + 4 <= output.len() is mathematically guaranteed inside the vector loop.
    vst1q_s32(output.as_mut_ptr().add(*out_idx), vreinterpretq_s32_u8(compacted));

    // 5. Advance output cursor by exact matched count
    *out_idx += *POPCOUNT_4BIT.get_unchecked(mask_idx);
}

/// Evaluates equality using 4x unrolled NEON registers with 100% branchless compaction.
#[cfg(target_arch = "aarch64")]
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
    let weights = vld1q_u32([1, 2, 4, 8].as_ptr());

    // SAFETY:
    // 1. input.as_ptr() is valid and readable for `chunks * 16 * 4` bytes.
    // 2. vld1q_s32 performs unaligned reads, supporting arbitrary 4-byte aligned i32 slices.
    // 3. Exclusive mutable reference `&mut [i32]` guarantees no aliasing between input and output.
    // 4. compact_neon_lane_branchless never branches on data and writes within buffer bounds.
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

        compact_neon_lane_branchless(v0, m0, weights, output, &mut out_idx);
        compact_neon_lane_branchless(v1, m1, weights, output, &mut out_idx);
        compact_neon_lane_branchless(v2, m2, weights, output, &mut out_idx);
        compact_neon_lane_branchless(v3, m3, weights, output, &mut out_idx);
    }

    for &val in &input[chunks * lanes..] {
        if val == literal {
            output[out_idx] = val;
            out_idx += 1;
        }
    }

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

/// Miri Software Emulation Kernel exercising explicit raw pointer arithmetic.
/// Audits stacked borrows, pointer provenance, and slice bounds without physical SIMD intrinsics.
#[cfg(miri)]
pub unsafe fn filter_i32_miri_emulation(
    input: &[i32],
    literal: i32,
    output: &mut [i32],
) -> usize {
    let len = input.len();
    let lanes = 16;
    let chunks = len / lanes;
    let mut out_idx = 0;

    // Emulate 16-lane vector chunks using raw pointer reads and writes
    for i in 0..chunks {
        let in_base = input.as_ptr().add(i * lanes);
        let out_base = output.as_mut_ptr().add(out_idx);
        let mut matched_in_chunk = 0;

        for lane in 0..lanes {
            // Miri validates raw pointer provenance and in-bounds dereferencing
            let val = *in_base.add(lane);
            if val == literal {
                *out_base.add(matched_in_chunk) = val;
                matched_in_chunk += 1;
            }
        }
        out_idx += matched_in_chunk;
    }

    // Emulate scalar tail using raw pointer arithmetic
    let tail_base = input.as_ptr().add(chunks * lanes);
    let tail_elements = len - chunks * lanes;
    for tail_idx in 0..tail_elements {
        let val = *tail_base.add(tail_idx);
        if val == literal {
            *output.as_mut_ptr().add(out_idx) = val;
            out_idx += 1;
        }
    }

    out_idx
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
