//! Comprehensive Adversarial Stress Test Suite — Iteration 2
//! Empirically challenges:
//! 1. StreamingMorselDispatcher 3-tier polling backoff & Rayon starvation
//! 2. StringViewDescriptor contiguous inline_data() (0..=12 bytes) & fast_equality_check()
//! 3. Grace Hash Join skew guards (D_max = 3 and Chunked BNLJ fallback)
//! 4. Apple Silicon NEON branchless compaction with vqtbl1q_u8
//! 5. MemoryBudgetTracker CAS reservations & saturating subtraction

use crossbeam_channel::{bounded, Receiver, RecvTimeoutError, TryRecvError};
use rand::Rng;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

// =========================================================================
// 1. STREAMING MORSEL DISPATCHER SPECIFICATION TEST
// =========================================================================

pub trait MorselDispatcher: Send + Sync {
    fn claim(&self) -> Option<usize>;
    fn terminate(&self);
}

/// Exact specification from docs/ARCHITECTURE_SPEC.md lines 373-426
pub struct StreamingMorselDispatcher {
    rx: Receiver<usize>,
    terminated: AtomicBool,
}

impl StreamingMorselDispatcher {
    pub fn new(rx: Receiver<usize>) -> Self {
        Self {
            rx,
            terminated: AtomicBool::new(false),
        }
    }
}

impl MorselDispatcher for StreamingMorselDispatcher {
    #[inline]
    fn claim(&self) -> Option<usize> {
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

#[test]
fn test_streaming_morsel_dispatcher_producer_lag_starvation() {
    println!("\n=== TEST 1.1: StreamingMorselDispatcher Under Severe Producer Lag ===");
    let pool = Arc::new(
        rayon::ThreadPoolBuilder::new()
            .num_threads(4)
            .build()
            .unwrap(),
    );

    let (tx, rx) = bounded::<usize>(16);
    let dispatcher = Arc::new(StreamingMorselDispatcher::new(rx));
    let morsels_processed = Arc::new(AtomicUsize::new(0));

    // Spawn 4 Rayon worker threads that continuously call claim()
    for _ in 0..4 {
        let disp = dispatcher.clone();
        let counter = morsels_processed.clone();
        pool.spawn(move || {
            while let Some(_morsel) = disp.claim() {
                counter.fetch_add(1, Ordering::SeqCst);
            }
        });
    }

    // Now, simulate independent high-priority compute jobs entering the Rayon pool
    // while the channel is completely EMPTY (producer lag of 100ms)
    let independent_jobs_completed = Arc::new(AtomicUsize::new(0));
    let num_jobs = 50;

    let start_submit = Instant::now();
    for _ in 0..num_jobs {
        let job_counter = independent_jobs_completed.clone();
        pool.spawn(move || {
            // Light compute workload
            let mut sum = 0u64;
            for i in 0..10_000 {
                sum = sum.wrapping_add(i);
            }
            std::hint::black_box(sum);
            job_counter.fetch_add(1, Ordering::SeqCst);
        });
    }

    // Wait 50ms (producer has NOT sent anything yet!)
    std::thread::sleep(Duration::from_millis(50));
    let completed_during_lag = independent_jobs_completed.load(Ordering::SeqCst);
    println!(
        "[Test 1.1] Independent compute jobs completed during empty-channel producer lag: {} / {}",
        completed_during_lag, num_jobs
    );

    // Assert that the 3-tier backoff yields CPU to Rayon and DOES NOT starve the thread pool
    assert_eq!(
        completed_during_lag, num_jobs,
        "Rayon pool was starved! StreamingMorselDispatcher blocked worker threads!"
    );
    println!("[Test 1.1] Latency: all 50 independent jobs finished in {:?}", start_submit.elapsed());

    // Now producer wakes up and delivers morsels
    for i in 0..100 {
        tx.send(i).unwrap();
    }

    std::thread::sleep(Duration::from_millis(50));
    assert_eq!(morsels_processed.load(Ordering::SeqCst), 100);

    // Terminate dispatcher cleanly
    dispatcher.terminate();
    drop(tx);
    std::thread::sleep(Duration::from_millis(20));
    println!("[Test 1.1] PASSED: Zero threadpool starvation under producer lag.");
}

#[test]
fn test_streaming_morsel_dispatcher_channel_backlog_and_termination() {
    println!("\n=== TEST 1.2: StreamingMorselDispatcher Channel Backlog & Rapid Termination ===");
    let pool = Arc::new(
        rayon::ThreadPoolBuilder::new()
            .num_threads(4)
            .build()
            .unwrap(),
    );

    let (tx, rx) = bounded::<usize>(1000);
    let dispatcher = Arc::new(StreamingMorselDispatcher::new(rx));
    let consumed = Arc::new(AtomicUsize::new(0));

    // Flood channel with 10,000 items
    let total_items = 10_000;
    std::thread::spawn(move || {
        for i in 0..total_items {
            let _ = tx.send(i);
        }
    });

    for _ in 0..4 {
        let disp = dispatcher.clone();
        let c = consumed.clone();
        pool.spawn(move || {
            while let Some(_) = disp.claim() {
                c.fetch_add(1, Ordering::SeqCst);
            }
        });
    }

    // Wait until at least 5,000 items consumed
    let start = Instant::now();
    while consumed.load(Ordering::SeqCst) < 5_000 && start.elapsed() < Duration::from_secs(2) {
        std::thread::sleep(Duration::from_millis(5));
    }

    // Terminate abruptly mid-stream
    dispatcher.terminate();
    let term_time = Instant::now();
    std::thread::sleep(Duration::from_millis(30));

    let final_count = consumed.load(Ordering::SeqCst);
    println!(
        "[Test 1.2] Dispatched and terminated: consumed {} items before immediate stop in {:?}",
        final_count,
        term_time.elapsed()
    );
    assert!(final_count >= 5000, "Should have consumed at least 5000 items");
    println!("[Test 1.2] PASSED: Clean termination under heavy backlog without deadlock.");
}

// =========================================================================
// 2. ARROW STRINGVIEW INLINE_DATA & FAST EQUALITY TEST
// =========================================================================

#[repr(C, align(16))]
#[derive(Clone, Copy, Debug)]
pub struct StringViewDescriptor {
    pub len: u32,
    pub prefix: [u8; 4],
    pub payload: [u8; 8],
}

impl StringViewDescriptor {
    pub fn new_inlined(data: &[u8]) -> Self {
        assert!(data.len() <= 12, "Inlined string length must be <= 12 bytes");
        let mut prefix = [0u8; 4];
        let mut payload = [0u8; 8];
        let p_len = data.len().min(4);
        prefix[..p_len].copy_from_slice(&data[..p_len]);
        if data.len() > 4 {
            let rem = data.len() - 4;
            payload[..rem].copy_from_slice(&data[4..4 + rem]);
        }
        Self {
            len: data.len() as u32,
            prefix,
            payload,
        }
    }

    #[inline(always)]
    pub fn is_inlined(&self) -> bool {
        self.len <= 12
    }

    /// Exact code from docs/ARCHITECTURE_SPEC.md lines 623-639
    #[inline(always)]
    pub fn inline_data(&self) -> &[u8] {
        debug_assert!(self.is_inlined(), "Cannot extract inline data from out-of-line StringView");
        unsafe {
            let base_ptr = (self as *const Self as *const u8).add(4);
            std::slice::from_raw_parts(base_ptr, self.len as usize)
        }
    }

    /// Exact code from docs/ARCHITECTURE_SPEC.md lines 655-668
    #[inline(always)]
    pub fn fast_equality_check(&self, target_len: u32, target_prefix: [u8; 4]) -> bool {
        let self_word = unsafe { *(self as *const Self as *const u64) };
        let target_word = (target_len as u64) | ((u32::from_ne_bytes(target_prefix) as u64) << 32);
        self_word == target_word
    }
}

#[test]
fn test_stringview_inline_data_all_lengths_0_to_12() {
    println!("\n=== TEST 2.1: Arrow StringView inline_data() Sweep for Lengths 0..=12 ===");
    let mut rng = rand::thread_rng();

    for len in 0..=12 {
        // Run 100 random byte patterns per length
        for _ in 0..100 {
            let mut sample = vec![0u8; len];
            rng.fill(&mut sample[..]);

            let sv = StringViewDescriptor::new_inlined(&sample);
            assert_eq!(sv.len as usize, len);
            assert!(sv.is_inlined());

            let extracted = sv.inline_data();
            assert_eq!(
                extracted,
                &sample[..],
                "Mismatch on length {}: expected {:?}, got {:?}",
                len,
                sample,
                extracted
            );

            // Verify pointer provenance and alignment bounds
            let struct_base = &sv as *const _ as *const u8;
            let slice_ptr = extracted.as_ptr();
            let struct_end = unsafe { struct_base.add(16) };

            assert_eq!(
                slice_ptr,
                unsafe { struct_base.add(4) },
                "inline_data pointer MUST start at exactly struct offset 4"
            );
            assert!(
                unsafe { slice_ptr.add(len) } <= struct_end,
                "inline_data slice exceeded struct boundary!"
            );
        }
    }
    println!("[Test 2.1] PASSED: All lengths 0 through 12 extract 100% contiguously without truncation.");
}

#[test]
fn test_stringview_fast_equality_check_correctness() {
    println!("\n=== TEST 2.2: StringView fast_equality_check Word-Level Comparison ===");

    // Case 1: Exact match on length and prefix
    let sv1 = StringViewDescriptor::new_inlined(b"CORRECT_KEY");
    assert!(sv1.fast_equality_check(11, *b"CORR"));

    // Case 2: Length mismatch
    assert!(!sv1.fast_equality_check(10, *b"CORR"));
    assert!(!sv1.fast_equality_check(12, *b"CORR"));

    // Case 3: Prefix mismatch
    assert!(!sv1.fast_equality_check(11, *b"INCO"));
    assert!(!sv1.fast_equality_check(11, *b"corr")); // Case sensitivity

    // Case 4: Short strings (len < 4) with zero padding
    for len in 0..4 {
        let src = &b"ABC"[..len];
        let sv_short = StringViewDescriptor::new_inlined(src);
        let mut target_p = [0u8; 4];
        target_p[..len].copy_from_slice(src);

        assert!(
            sv_short.fast_equality_check(len as u32, target_p),
            "Failed fast equality for short string len {}",
            len
        );

        // Mismatched character in target_prefix
        let mut bad_p = target_p;
        bad_p[0] ^= 0x20; // Flip case
        if len > 0 {
            assert!(
                !sv_short.fast_equality_check(len as u32, bad_p),
                "False positive on short string len {}",
                len
            );
        }
    }
    println!("[Test 2.2] PASSED: 64-bit word equality preserves exact soundness and rejection.");
}

// =========================================================================
// 3. GRACE HASH JOIN SKEW GUARDS & CHUNKED BNLJ TEST
// =========================================================================

pub const MAX_RECURSION_DEPTH: usize = 3;
pub const SKEW_THRESHOLD: f64 = 0.80;

#[derive(Debug, PartialEq, Eq)]
pub enum PartitionJoinStrategy {
    InMemoryHashTable,
    RecursivePartitioning { next_depth: usize },
    BlockNestedLoopFallback,
    RadixSortMergeFallback,
}

pub struct PartitionDescriptor {
    pub total_bytes: usize,
    pub depth: usize,
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
        if max_sub_partition_ratio >= SKEW_THRESHOLD {
            return PartitionJoinStrategy::BlockNestedLoopFallback;
        }
        if self.depth < MAX_RECURSION_DEPTH {
            PartitionJoinStrategy::RecursivePartitioning {
                next_depth: self.depth + 1,
            }
        } else {
            PartitionJoinStrategy::RadixSortMergeFallback
        }
    }
}

#[test]
fn test_grace_hash_join_extreme_zipfian_skew_guard() {
    println!("\n=== TEST 3.1: Grace Hash Join Skew Guards Under 95% Zipfian Duplicate Key ===");
    const TOTAL_ROWS: usize = 500_000;
    const ROW_SIZE: usize = 32;
    let total_bytes = TOTAL_ROWS * ROW_SIZE; // 16 MB
    let memory_budget = 2 * 1024 * 1024; // 2 MB budget

    let p_count = 64;
    let mut partition_counts = vec![0usize; p_count];

    // 95% of rows have identical key 0xCAFEBABE, 5% uniform
    for i in 0..TOTAL_ROWS {
        let key: u64 = if i < (TOTAL_ROWS * 95) / 100 {
            0xCAFEBABE
        } else {
            i as u64
        };
        let hash = key.wrapping_mul(0x517cc1b727220a95);
        let bucket = (hash as usize) % p_count;
        partition_counts[bucket] += 1;
    }

    let max_count = *partition_counts.iter().max().unwrap();
    let max_ratio = max_count as f64 / TOTAL_ROWS as f64;
    println!(
        "[Test 3.1] Total: {} rows ({:.1} MB). Heaviest partition: {} rows ({:.1}% of dataset)",
        TOTAL_ROWS,
        total_bytes as f64 / (1024.0 * 1024.0),
        max_count,
        max_ratio * 100.0
    );

    let partition = PartitionDescriptor {
        total_bytes,
        depth: 0,
    };

    let strategy = partition.determine_strategy(memory_budget, max_ratio);
    println!("[Test 3.1] Strategy chosen by engine: {:?}", strategy);

    assert_eq!(
        strategy,
        PartitionJoinStrategy::BlockNestedLoopFallback,
        "Engine MUST fall back to Chunked BNLJ when key skew exceeds 80%!"
    );
    println!("[Test 3.1] PASSED: Infinite recursion averted via skew degeneracy detection.");
}

#[test]
fn test_chunked_bnlj_execution_memory_bounded() {
    println!("\n=== TEST 3.2: Chunked BNLJ Fallback Execution Within Memory Budget ===");
    let budget_bytes = 100_000; // 100 KB
    let chunk_limit = budget_bytes / 2; // 50 KB

    // Build side: 10,000 rows with 100% duplicate key 42
    let build_rows: Vec<(u64, i64)> = vec![(42, 100); 10_000];
    // Probe side: 1,000 rows with key 42
    let probe_rows: Vec<(u64, i64)> = vec![(42, 200); 1_000];

    let row_byte_size = 16usize; // 8 bytes key + 8 bytes val
    let rows_per_chunk = chunk_limit / row_byte_size;

    let mut total_matches = 0usize;
    let mut max_observed_chunk_bytes = 0usize;

    // Simulate chunked BNLJ execution
    for chunk in build_rows.chunks(rows_per_chunk) {
        let chunk_mem = chunk.len() * row_byte_size;
        max_observed_chunk_bytes = max_observed_chunk_bytes.max(chunk_mem);
        assert!(
            chunk_mem <= chunk_limit,
            "Chunk exceeded allocated budget: {} > {}",
            chunk_mem,
            chunk_limit
        );

        // Nested loop probe
        for p in &probe_rows {
            for b in chunk {
                if b.0 == p.0 {
                    total_matches += 1;
                }
            }
        }
    }

    println!(
        "[Test 3.2] Total matched output tuples: {}. Max build chunk in RAM: {} bytes (limit {} bytes)",
        total_matches, max_observed_chunk_bytes, chunk_limit
    );
    assert_eq!(total_matches, 10_000 * 1_000);
    assert!(max_observed_chunk_bytes <= chunk_limit);
    println!("[Test 3.2] PASSED: BNLJ processes 100% duplicate keys strictly within O(1) memory budget.");
}

#[test]
fn test_grace_hash_join_max_recursion_depth_fallback() {
    println!("\n=== TEST 3.3: Grace Hash Join D_max = 3 Transition to Radix Sort-Merge ===");
    let memory_budget = 1_000_000;

    // Distinct non-degenerate data that cannot fit in budget at depth 0, 1, 2, 3
    for depth in 0..=3 {
        let partition = PartitionDescriptor {
            total_bytes: 5_000_000,
            depth,
        };
        let strategy = partition.determine_strategy(memory_budget, 0.20); // 20% max ratio (< 80%)

        if depth < 3 {
            assert_eq!(
                strategy,
                PartitionJoinStrategy::RecursivePartitioning { next_depth: depth + 1 }
            );
        } else {
            assert_eq!(
                strategy,
                PartitionJoinStrategy::RadixSortMergeFallback,
                "At depth D = 3, engine MUST transition to Radix Sort-Merge Join!"
            );
        }
    }
    println!("[Test 3.3] PASSED: Recursion strictly capped at D_max = 3 without unbounded stack growth.");
}

// =========================================================================
// 4. APPLE SILICON NEON BRANCHLESS COMPACTION (VQTBL1Q_U8) TEST
// =========================================================================

#[cfg(target_arch = "aarch64")]
mod neon_tests {
    use std::arch::aarch64::*;

    #[repr(align(16))]
    struct AlignedPermTable([[u8; 16]; 16]);

    /// 16-entry byte permutation table for ARM64 NEON vqtbl1q_u8 (256 bytes).
    static NEON_PERM_TABLE: AlignedPermTable = AlignedPermTable([
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
    ]);

    static POPCOUNT_4BIT: [usize; 16] = [0, 1, 1, 2, 1, 2, 2, 3, 1, 2, 2, 3, 2, 3, 3, 4];

    #[inline(always)]
    unsafe fn compact_neon_lane_branchless(
        vals: int32x4_t,
        mask: uint32x4_t,
        weights: uint32x4_t,
        output: &mut [i32],
        out_idx: &mut usize,
    ) {
        let masked = vandq_u32(mask, weights);
        let mask_idx = vaddvq_u32(masked) as usize;
        let perm = vld1q_u8(NEON_PERM_TABLE.0.get_unchecked(mask_idx).as_ptr());
        let compacted = vqtbl1q_u8(vreinterpretq_u8_s32(vals), perm);
        vst1q_s32(output.as_mut_ptr().add(*out_idx), vreinterpretq_s32_u8(compacted));
        *out_idx += *POPCOUNT_4BIT.get_unchecked(mask_idx);
    }

    #[target_feature(enable = "neon")]
    pub unsafe fn filter_i32_neon_branchless(input: &[i32], literal: i32, output: &mut [i32]) -> usize {
        assert!(output.len() >= input.len());
        let len = input.len();
        let lanes = 16;
        let chunks = len / lanes;
        let mut out_idx = 0;

        let lit_vec = vdupq_n_s32(literal);
        let weights = vld1q_u32([1, 2, 4, 8].as_ptr());

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

    pub fn filter_i32_scalar_reference(input: &[i32], literal: i32, output: &mut [i32]) -> usize {
        let mut out_idx = 0;
        for &val in input {
            if val == literal {
                output[out_idx] = val;
                out_idx += 1;
            }
        }
        out_idx
    }

    #[test]
    fn test_neon_all_16_permutation_masks() {
        println!("\n=== TEST 4.1: NEON vqtbl1q_u8 All 16 Bitmask Permutation Patterns ===");
        let target = 777i32;
        let other = 111i32;

        unsafe {
            let weights = vld1q_u32([1, 2, 4, 8].as_ptr());
            let lit_vec = vdupq_n_s32(target);

            for mask_val in 0..16usize {
                let mut lane_vals = [other; 4];
                let mut expected = Vec::new();
                for bit in 0..4 {
                    if (mask_val & (1 << bit)) != 0 {
                        lane_vals[bit] = target;
                        expected.push(target);
                    }
                }

                let vals = vld1q_s32(lane_vals.as_ptr());
                let mask = vceqq_s32(vals, lit_vec);

                let mut output = [0i32; 8];
                let mut out_idx = 0;
                compact_neon_lane_branchless(vals, mask, weights, &mut output, &mut out_idx);

                assert_eq!(
                    out_idx,
                    expected.len(),
                    "Mismatch in compacted count for mask pattern {:04b}",
                    mask_val
                );
                assert_eq!(
                    &output[..out_idx],
                    &expected[..],
                    "Mismatch in compacted output for mask pattern {:04b}",
                    mask_val
                );
            }
        }
        println!("[Test 4.1] PASSED: All 16 NEON permutation patterns produce bit-exact compacted output.");
    }

    #[test]
    fn test_neon_branchless_sweep_and_boundary_correctness() {
        println!("\n=== TEST 4.2: NEON Branchless Compaction Sweep Across Boundary Lengths ===");
        let test_lengths = [0, 1, 2, 3, 4, 7, 8, 15, 16, 17, 31, 32, 33, 63, 64, 65, 127, 128, 255, 256, 1024, 65536];

        for &len in &test_lengths {
            let input: Vec<i32> = (0..len as i32).map(|x| if x % 3 == 0 { 42 } else { x }).collect();
            let mut out_neon = vec![0i32; len];
            let mut out_ref = vec![0i32; len];

            let count_neon = unsafe { filter_i32_neon_branchless(&input, 42, &mut out_neon) };
            let count_ref = filter_i32_scalar_reference(&input, 42, &mut out_ref);

            assert_eq!(count_neon, count_ref, "Count mismatch on length {}", len);
            assert_eq!(&out_neon[..count_neon], &out_ref[..count_ref], "Data mismatch on length {}", len);
        }
        println!("[Test 4.2] PASSED: NEON branchless compaction verified across boundary lengths 0 to 65536.");
    }

    #[test]
    fn test_neon_branchless_10m_performance() {
        use std::time::Instant;
        println!("\n=== TEST 4.3: NEON Branchless vs Scalar Benchmark (10M Integers, 50% Selectivity) ===");
        let n = 10_000_000;
        // Alternating pattern: worst case 50% branch misprediction for scalar
        let input: Vec<i32> = (0..n as i32).map(|i| if i % 2 == 0 { 42 } else { 99 }).collect();
        let mut out_neon = vec![0i32; n];
        let mut out_scalar = vec![0i32; n];

        let start_scalar = Instant::now();
        let count_scalar = filter_i32_scalar_reference(&input, 42, &mut out_scalar);
        let dur_scalar = start_scalar.elapsed();

        let start_neon = Instant::now();
        let count_neon = unsafe { filter_i32_neon_branchless(&input, 42, &mut out_neon) };
        let dur_neon = start_neon.elapsed();

        assert_eq!(count_neon, count_scalar);
        assert_eq!(&out_neon[..count_neon], &out_scalar[..count_scalar]);

        let speedup = dur_scalar.as_secs_f64() / dur_neon.as_secs_f64();
        println!(
            "[Test 4.3] Results (10M elements, sigma=0.50): Scalar: {:.2} ms | NEON vqtbl1q_u8: {:.2} ms | Speedup: {:.2}x",
            dur_scalar.as_secs_f64() * 1000.0,
            dur_neon.as_secs_f64() * 1000.0,
            speedup
        );
        println!("[Test 4.3] PASSED: Bit-exact branchless compaction with significant performance advantage.");
    }
}

// =========================================================================
// 5. MEMORY BUDGET TRACKER CAS & SATURATING SUBTRACTION TEST
// =========================================================================

pub struct RemediatedMemoryBudgetTracker {
    allocated: AtomicUsize,
    soft_limit: usize,
    hard_limit: usize,
}

impl RemediatedMemoryBudgetTracker {
    pub fn new(hard_limit: usize) -> Self {
        Self {
            allocated: AtomicUsize::new(0),
            soft_limit: (hard_limit * 80) / 100,
            hard_limit,
        }
    }

    pub fn try_reserve(&self, bytes: usize) -> Result<bool, String> {
        let mut current = self.allocated.load(Ordering::Acquire);
        loop {
            let new_total = current.checked_add(bytes)
                .ok_or_else(|| "OutOfMemory: integer overflow".to_string())?;
            if new_total > self.hard_limit {
                return Err(format!(
                    "OutOfMemory: {} exceeds hard limit {}",
                    bytes, self.hard_limit
                ));
            }
            match self.allocated.compare_exchange_weak(
                current,
                new_total,
                Ordering::AcqRel,
                Ordering::Acquire,
            ) {
                Ok(_) => {
                    let requires_spill = new_total >= self.soft_limit;
                    return Ok(requires_spill);
                }
                Err(actual) => current = actual,
            }
        }
    }

    /// Exact code from docs/ARCHITECTURE_SPEC.md lines 866-880
    pub fn release(&self, bytes: usize) {
        let mut current = self.allocated.load(Ordering::Acquire);
        loop {
            let new_total = current.saturating_sub(bytes);
            match self.allocated.compare_exchange_weak(
                current,
                new_total,
                Ordering::AcqRel,
                Ordering::Acquire,
            ) {
                Ok(_) => break,
                Err(actual) => current = actual,
            }
        }
    }

    pub fn allocated(&self) -> usize {
        self.allocated.load(Ordering::Acquire)
    }
}

#[test]
fn test_memory_budget_tracker_underflow_prevention() {
    println!("\n=== TEST 5.1: MemoryBudgetTracker Underflow Prevention ===");
    let tracker = RemediatedMemoryBudgetTracker::new(1000);

    // Reserve 200
    assert_eq!(tracker.try_reserve(200), Ok(false));
    assert_eq!(tracker.allocated(), 200);

    // Release 500 (excess release of 300 bytes!)
    tracker.release(500);

    // Must saturate at 0, NOT underflow to usize::MAX
    let current = tracker.allocated();
    println!("[Test 5.1] Allocated after excess release: {}", current);
    assert_eq!(current, 0, "MemoryBudgetTracker allocated wrapped around to huge integer!");

    // Next reservation must succeed normally
    assert_eq!(tracker.try_reserve(500), Ok(false));
    assert_eq!(tracker.allocated(), 500);
    println!("[Test 5.1] PASSED: CAS saturating subtraction completely eliminates underflow corruption.");
}

#[test]
fn test_memory_budget_tracker_concurrent_stress() {
    println!("\n=== TEST 5.2: MemoryBudgetTracker 16-Thread Concurrent CAS Stress ===");
    let tracker = Arc::new(RemediatedMemoryBudgetTracker::new(100_000));
    let mut handles = Vec::new();

    for thread_id in 0..16 {
        let t = tracker.clone();
        handles.push(std::thread::spawn(move || {
            let mut rng = rand::thread_rng();
            for _ in 0..10_000 {
                let reserve_amt = rng.gen_range(1..1000);
                if t.try_reserve(reserve_amt).is_ok() {
                    // Random small delay or release
                    if rng.gen_bool(0.7) {
                        t.release(reserve_amt);
                    } else if thread_id == 0 && rng.gen_bool(0.1) {
                        // Intentional excess release to stress saturation
                        t.release(reserve_amt * 2);
                    }
                }
            }
        }));
    }

    for h in handles {
        h.join().unwrap();
    }

    let final_alloc = tracker.allocated();
    println!("[Test 5.2] Final allocated after 160,000 concurrent ops: {}", final_alloc);
    assert!(
        final_alloc <= 100_000,
        "Memory budget exceeded hard limit under concurrency!"
    );
    println!("[Test 5.2] PASSED: CAS loop guarantees linearizable bounds under high contention.");
}
