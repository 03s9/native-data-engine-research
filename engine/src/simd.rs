//! SLICE F. Filter kernels with scalar reference + NEON (aarch64) / AVX2 (x86_64, runtime-detected) paths.
//! Public fns must give IDENTICAL results to the scalar reference for all inputs (differential-tested).

/// out[i] = values[i] > threshold, written for every i. `out.len() == values.len()`.
pub fn gt_i64_scalar(_values: &[i64], _threshold: i64, _out: &mut [bool]) {}
/// Dispatches to the fastest safe path for this CPU; unaligned loads only (slices may be misaligned).
pub fn gt_i64(_values: &[i64], _threshold: i64, _out: &mut [bool]) {}
/// Sum of i64 with wrapping add; dispatching + scalar reference.
pub fn sum_i64_scalar(_values: &[i64]) -> i64 { 0 }
pub fn sum_i64(_values: &[i64]) -> i64 { 0 }
