//! SLICE F. Filter kernels with scalar reference + NEON (aarch64) / AVX2 (x86_64, runtime-detected) paths.
//! Public fns must give IDENTICAL results to the scalar reference for all inputs (differential-tested).

#[cfg(target_arch = "aarch64")]
use std::arch::aarch64::{
    vaddq_s64, vaddvq_s64, vcgtq_s64, vcombine_u16, vcombine_u32, vdupq_n_s64, vld1q_s64,
    vmovn_u16, vmovn_u32, vmovn_u64, vshrq_n_u64, vst1_u8,
};
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::{
    __m256i, _mm256_add_epi64, _mm256_castsi256_pd, _mm256_cmpgt_epi64, _mm256_loadu_si256,
    _mm256_movemask_pd, _mm256_set1_epi64x, _mm256_setzero_si256, _mm256_storeu_si256,
};

/// out[i] = values[i] > threshold, written for every i. `out.len() == values.len()`.
pub fn gt_i64_scalar(values: &[i64], threshold: i64, out: &mut [bool]) {
    assert_eq!(
        values.len(),
        out.len(),
        "gt_i64: output length must equal input length"
    );
    for (o, &v) in out.iter_mut().zip(values) {
        *o = v > threshold;
    }
}

/// Dispatches to the fastest safe path for this CPU; unaligned loads only (slices may be misaligned).
pub fn gt_i64(values: &[i64], threshold: i64, out: &mut [bool]) {
    assert_eq!(
        values.len(),
        out.len(),
        "gt_i64: output length must equal input length"
    );
    #[cfg(target_arch = "aarch64")]
    {
        if std::arch::is_aarch64_feature_detected!("neon") {
            // SAFETY: NEON was detected at runtime just above; the length invariant is asserted above.
            unsafe { gt_i64_neon(values, threshold, out) };
            return;
        }
    }
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") {
            // SAFETY: AVX2 was detected at runtime just above; the length invariant is asserted above.
            unsafe { gt_i64_avx2(values, threshold, out) };
            return;
        }
    }
    gt_i64_scalar(values, threshold, out);
}

/// Sum of i64 with wrapping add; dispatching + scalar reference.
pub fn sum_i64_scalar(values: &[i64]) -> i64 {
    values.iter().fold(0i64, |acc, &v| acc.wrapping_add(v))
}

pub fn sum_i64(values: &[i64]) -> i64 {
    #[cfg(target_arch = "aarch64")]
    {
        if std::arch::is_aarch64_feature_detected!("neon") {
            // SAFETY: NEON was detected at runtime just above.
            return unsafe { sum_i64_neon(values) };
        }
    }
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") {
            // SAFETY: AVX2 was detected at runtime just above.
            return unsafe { sum_i64_avx2(values) };
        }
    }
    sum_i64_scalar(values)
}

/// NEON filter: 8 lanes per block. Four 2-lane compares, each 0/u64::MAX per lane, are shifted to
/// 0/1, narrowed to bytes and stored as 8 bools. The remaining < 8 elements use the scalar loop.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn gt_i64_neon(values: &[i64], threshold: i64, out: &mut [bool]) {
    // SAFETY (whole body): the assert guarantees out.len() == values.len(). For each block starting
    // at i with i + 8 <= len, the four vld1q_s64 loads read values[i..i+8] (vld1q_s64 needs only
    // element alignment, so misaligned slices are fine). vst1_u8 writes out[i..i+8] and is in bounds
    // for the same reason. bool is a single byte whose valid values are 0 and 1, and every stored byte
    // is 0 or 1. Only blocks cover [0, blocks*8), and the tail goes through safe code.
    assert_eq!(values.len(), out.len());
    let blocks = values.len() / 8;
    let thr = vdupq_n_s64(threshold);
    for b in 0..blocks {
        let i = b * 8;
        let p = values.as_ptr().add(i);
        let m0 = vcgtq_s64(vld1q_s64(p), thr);
        let m1 = vcgtq_s64(vld1q_s64(p.add(2)), thr);
        let m2 = vcgtq_s64(vld1q_s64(p.add(4)), thr);
        let m3 = vcgtq_s64(vld1q_s64(p.add(6)), thr);
        let n01 = vcombine_u32(
            vmovn_u64(vshrq_n_u64::<63>(m0)),
            vmovn_u64(vshrq_n_u64::<63>(m1)),
        );
        let n23 = vcombine_u32(
            vmovn_u64(vshrq_n_u64::<63>(m2)),
            vmovn_u64(vshrq_n_u64::<63>(m3)),
        );
        let h = vcombine_u16(vmovn_u32(n01), vmovn_u32(n23));
        let bytes = vmovn_u16(h);
        vst1_u8(out.as_mut_ptr().add(i) as *mut u8, bytes);
    }
    gt_i64_scalar(&values[blocks * 8..], threshold, &mut out[blocks * 8..]);
}

/// NEON wrapping sum: two accumulators over 4-element blocks, then a wrapping scalar tail.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn sum_i64_neon(values: &[i64]) -> i64 {
    // SAFETY (whole body): for each block b with b*4 + 4 <= len, the two vld1q_s64 loads read
    // values[4b..4b+4], which is in bounds. vld1q_s64 needs only element alignment. vaddq_s64 and
    // vaddvq_s64 are modular (wrapping), so the lane-wise sums equal the scalar wrapping sum mod 2^64.
    // The tail uses wrapping_add.
    let blocks = values.len() / 4;
    let mut acc0 = vdupq_n_s64(0);
    let mut acc1 = vdupq_n_s64(0);
    for b in 0..blocks {
        let p = values.as_ptr().add(b * 4);
        acc0 = vaddq_s64(acc0, vld1q_s64(p));
        acc1 = vaddq_s64(acc1, vld1q_s64(p.add(2)));
    }
    let mut total = vaddvq_s64(vaddq_s64(acc0, acc1));
    for &v in &values[blocks * 4..] {
        total = total.wrapping_add(v);
    }
    total
}

/// AVX2 filter: 8 elements per block as two 4-lane compares. `movemask_pd` gives one sign bit per
/// 64-bit lane, and the 8 bits are expanded into 8 bools. Tail is scalar.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn gt_i64_avx2(values: &[i64], threshold: i64, out: &mut [bool]) {
    // SAFETY (whole body): the assert guarantees out.len() == values.len(). For each block starting
    // at i with i + 8 <= len, the two _mm256_loadu_si256 loads read values[i..i+8] (unaligned loads
    // are allowed). The block writes go through bounds-checked slice indexing, so no raw store is
    // used. The tail uses the scalar path.
    assert_eq!(values.len(), out.len());
    let blocks = values.len() / 8;
    let thr = _mm256_set1_epi64x(threshold);
    let base = values.as_ptr() as *const __m256i;
    for b in 0..blocks {
        let i = b * 8;
        let m0 = _mm256_cmpgt_epi64(_mm256_loadu_si256(base.add(b * 2)), thr);
        let m1 = _mm256_cmpgt_epi64(_mm256_loadu_si256(base.add(b * 2 + 1)), thr);
        let bits = (_mm256_movemask_pd(_mm256_castsi256_pd(m0)) as u32)
            | ((_mm256_movemask_pd(_mm256_castsi256_pd(m1)) as u32) << 4);
        for (j, o) in out[i..i + 8].iter_mut().enumerate() {
            *o = (bits >> j) & 1 == 1;
        }
    }
    gt_i64_scalar(&values[blocks * 8..], threshold, &mut out[blocks * 8..]);
}

/// AVX2 wrapping sum: one 4-lane accumulator (vpaddq wraps), lanes folded with wrapping_add.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn sum_i64_avx2(values: &[i64]) -> i64 {
    // SAFETY (whole body): for each block b with b*4 + 4 <= len, the unaligned 32-byte load reads
    // values[4b..4b+4], which is in bounds. _mm256_storeu_si256 writes into a local [i64; 4], which
    // is 32 bytes. _mm256_add_epi64 wraps. The tail uses wrapping_add.
    let blocks = values.len() / 4;
    let base = values.as_ptr() as *const __m256i;
    let mut acc = _mm256_setzero_si256();
    for b in 0..blocks {
        acc = _mm256_add_epi64(acc, _mm256_loadu_si256(base.add(b)));
    }
    let mut lanes = [0i64; 4];
    _mm256_storeu_si256(lanes.as_mut_ptr() as *mut __m256i, acc);
    let mut total = lanes.iter().fold(0i64, |a, &v| a.wrapping_add(v));
    for &v in &values[blocks * 4..] {
        total = total.wrapping_add(v);
    }
    total
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::{rngs::StdRng, Rng, SeedableRng};

    fn random_vec(rng: &mut StdRng, n: usize) -> Vec<i64> {
        (0..n).map(|_| rng.gen::<i64>()).collect()
    }

    /// Independent oracle: exact i128 sum truncated to 64 bits equals the wrapping sum mod 2^64.
    fn sum_oracle(values: &[i64]) -> i64 {
        values.iter().map(|&v| v as i128).sum::<i128>() as i64
    }

    /// Runs gt_i64 into an output pre-filled with `fill`, and checks it against the scalar reference.
    fn check_gt(values: &[i64], threshold: i64, fill: bool) {
        let mut want = vec![fill; values.len()];
        let mut got = vec![fill; values.len()];
        gt_i64_scalar(values, threshold, &mut want);
        gt_i64(values, threshold, &mut got);
        assert_eq!(got, want, "len={} threshold={threshold}", values.len());
    }

    #[test]
    fn gt_matches_scalar_for_lengths_0_to_70() {
        let mut rng = StdRng::seed_from_u64(1);
        for n in 0..=70 {
            let values = random_vec(&mut rng, n);
            let thr = rng.gen::<i64>();
            check_gt(&values, thr, true);
            check_gt(&values, thr, false);
        }
    }

    #[test]
    fn gt_matches_scalar_for_large_lengths() {
        let mut rng = StdRng::seed_from_u64(2);
        for n in [4096usize, 65_537, 100_003] {
            let values = random_vec(&mut rng, n);
            let thr = rng.gen::<i64>();
            check_gt(&values, thr, true);
        }
    }

    #[test]
    fn gt_matches_scalar_on_misaligned_subslices() {
        let mut rng = StdRng::seed_from_u64(3);
        let buf = random_vec(&mut rng, 300);
        let mut out_buf = vec![false; 300];
        for offset in 0..8 {
            for len in [0usize, 1, 7, 8, 9, 63, 64, 65] {
                let values = &buf[offset..offset + len];
                let thr = buf[offset + len / 2];
                let want = &mut vec![false; len];
                gt_i64_scalar(values, thr, want);
                let got = &mut out_buf[offset + 1..offset + 1 + len];
                gt_i64(values, thr, got);
                assert_eq!(got, &want[..], "offset={offset} len={len}");
            }
        }
    }

    #[test]
    fn gt_threshold_at_i64_extremes() {
        let values = [i64::MIN, i64::MIN + 1, -1, 0, 1, i64::MAX - 1, i64::MAX];
        for n in [1usize, 7, 8, 9, 15, 16, 17] {
            let data: Vec<i64> = values.iter().cycle().take(n).copied().collect();
            let mut out = vec![true; n];
            gt_i64(&data, i64::MAX, &mut out);
            assert!(out.iter().all(|&b| !b), "nothing is > MAX, n={n}");
            gt_i64(&data, i64::MIN, &mut out);
            for (i, &b) in out.iter().enumerate() {
                assert_eq!(b, data[i] > i64::MIN, "threshold MIN, i={i}, n={n}");
            }
            check_gt(&data, i64::MIN, false);
            check_gt(&data, i64::MAX, true);
        }
    }

    #[test]
    fn gt_all_equal_values() {
        for n in 0..=70 {
            let data = vec![5i64; n];
            let mut out = vec![true; n];
            gt_i64(&data, 5, &mut out);
            assert!(out.iter().all(|&b| !b), "5 > 5 is false, n={n}");
            gt_i64(&data, 4, &mut out);
            assert!(out.iter().all(|&b| b), "5 > 4 is true, n={n}");
        }
    }

    #[test]
    fn gt_tail_lengths_1_to_7() {
        let mut rng = StdRng::seed_from_u64(4);
        for block in 0..5 {
            for r in 1..=7 {
                let n = block * 8 + r;
                let values = random_vec(&mut rng, n);
                let thr = values[n - 1];
                check_gt(&values, thr, true);
                check_gt(&values, thr, false);
            }
        }
    }

    #[test]
    fn gt_output_fully_overwritten() {
        let values: Vec<i64> = (0..37).map(|i| if i % 3 == 0 { 10 } else { 0 }).collect();
        // Sentinel true: every position with value 0 must become false, so nothing keeps the sentinel.
        let mut out = vec![true; values.len()];
        gt_i64(&values, 5, &mut out);
        let expected: Vec<bool> = values.iter().map(|&v| v > 5).collect();
        assert_eq!(out, expected);
        // Sentinel false: every position with value 10 must become true.
        let mut out = vec![false; values.len()];
        gt_i64(&values, 5, &mut out);
        assert_eq!(out, expected);
    }

    #[test]
    fn gt_scalar_known_values() {
        let values = [3i64, 7, -2, 7, i64::MAX, i64::MIN];
        let mut out = [false; 6];
        gt_i64_scalar(&values, 5, &mut out);
        assert_eq!(out, [false, true, false, true, true, false]);
    }

    #[test]
    #[should_panic(expected = "output length must equal input length")]
    fn gt_length_mismatch_panics() {
        let mut out = [false; 3];
        gt_i64(&[1, 2, 3, 4], 0, &mut out);
    }

    #[test]
    #[cfg(target_arch = "aarch64")]
    fn gt_neon_path_matches_scalar() {
        // Calls the NEON kernel directly so the NEON code runs, not only the dispatcher.
        let mut rng = StdRng::seed_from_u64(5);
        for n in (0..=70).chain([4096, 100_003]) {
            let values = random_vec(&mut rng, n);
            for thr in [rng.gen::<i64>(), i64::MIN, i64::MAX] {
                let mut want = vec![false; n];
                let mut got = vec![true; n];
                gt_i64_scalar(&values, thr, &mut want);
                // SAFETY: this test runs on aarch64 where NEON is baseline.
                unsafe { gt_i64_neon(&values, thr, &mut got) };
                assert_eq!(got, want, "neon n={n} thr={thr}");
            }
        }
    }

    #[test]
    #[cfg(target_arch = "x86_64")]
    fn gt_avx2_path_matches_scalar() {
        if !is_x86_feature_detected!("avx2") {
            return;
        }
        let mut rng = StdRng::seed_from_u64(6);
        for n in (0..=70).chain([4096, 100_003]) {
            let values = random_vec(&mut rng, n);
            for thr in [rng.gen::<i64>(), i64::MIN, i64::MAX] {
                let mut want = vec![false; n];
                let mut got = vec![true; n];
                gt_i64_scalar(&values, thr, &mut want);
                // SAFETY: AVX2 was detected just above.
                unsafe { gt_i64_avx2(&values, thr, &mut got) };
                assert_eq!(got, want, "avx2 n={n} thr={thr}");
            }
        }
    }

    #[test]
    fn sum_matches_scalar_for_lengths_0_to_70() {
        let mut rng = StdRng::seed_from_u64(7);
        for n in 0..=70 {
            let values = random_vec(&mut rng, n);
            assert_eq!(sum_i64(&values), sum_i64_scalar(&values), "n={n}");
            assert_eq!(sum_i64_scalar(&values), sum_oracle(&values), "n={n}");
        }
    }

    #[test]
    fn sum_wrapping_overflow_matches_oracle() {
        for n in 1..=70 {
            let max = vec![i64::MAX; n];
            let min = vec![i64::MIN; n];
            let mixed: Vec<i64> = (0..n)
                .map(|i| if i % 2 == 0 { i64::MAX } else { 1 })
                .collect();
            for data in [&max, &min, &mixed] {
                assert_eq!(sum_i64(data), sum_oracle(data), "n={n}");
                assert_eq!(sum_i64_scalar(data), sum_oracle(data), "n={n}");
            }
        }
    }

    #[test]
    fn sum_large_and_misaligned_subslices() {
        let mut rng = StdRng::seed_from_u64(8);
        let big = random_vec(&mut rng, 1_000_003);
        assert_eq!(sum_i64(&big), sum_oracle(&big));
        let buf = random_vec(&mut rng, 300);
        for offset in 0..8 {
            for len in [0usize, 1, 3, 4, 5, 7, 8, 63, 64, 65] {
                let s = &buf[offset..offset + len];
                assert_eq!(sum_i64(s), sum_i64_scalar(s), "offset={offset} len={len}");
            }
        }
    }

    #[test]
    fn sum_empty_is_zero() {
        assert_eq!(sum_i64(&[]), 0);
        assert_eq!(sum_i64_scalar(&[]), 0);
    }

    #[test]
    #[cfg(target_arch = "aarch64")]
    fn sum_neon_path_matches_scalar() {
        // SAFETY: this test runs on aarch64 where NEON is baseline.
        let mut rng = StdRng::seed_from_u64(9);
        for n in (0..=70).chain([100_003]) {
            let values = random_vec(&mut rng, n);
            assert_eq!(
                unsafe { sum_i64_neon(&values) },
                sum_oracle(&values),
                "n={n}"
            );
        }
        let max = vec![i64::MAX; 1001];
        assert_eq!(unsafe { sum_i64_neon(&max) }, sum_oracle(&max));
    }

    #[test]
    #[cfg(target_arch = "x86_64")]
    fn sum_avx2_path_matches_scalar() {
        if !is_x86_feature_detected!("avx2") {
            return;
        }
        let mut rng = StdRng::seed_from_u64(10);
        for n in (0..=70).chain([100_003]) {
            let values = random_vec(&mut rng, n);
            // SAFETY: AVX2 was detected just above.
            assert_eq!(
                unsafe { sum_i64_avx2(&values) },
                sum_oracle(&values),
                "n={n}"
            );
        }
        let min = vec![i64::MIN; 1001];
        // SAFETY: AVX2 was detected just above.
        assert_eq!(unsafe { sum_i64_avx2(&min) }, sum_oracle(&min));
    }
}
