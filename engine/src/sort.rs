//! SLICE D1. Sort with optional external merge when the budget is exceeded.
//!
//! Ordering per key: nulls go first or last (`nulls_first`, independent of direction); non-null
//! values compare by type order (Float64 via `total_cmp`, so -0.0 < 0.0 and NaN sorts last),
//! reversed when descending. Ties keep input order, so both paths produce the same stable result.
use crate::column::{Batch, Column, StringViews, Values};
use crate::eval::eval;
use crate::expr::Expr;
use crate::memory::MemoryBudget;
use crate::plan::SortField;
use crate::spill::{SpillFile, SpillReader, SpillWriter};
use crate::types::{EngineError, Result, SchemaRef};
use std::cmp::Ordering;
use std::collections::VecDeque;
use std::path::{Path, PathBuf};
use std::sync::Arc;

/// What the last sort did. Exposed so tests can prove the external path ran and cleaned up.
#[derive(Debug, Default, Clone, PartialEq, Eq)]
pub struct SortStats {
    pub runs_spilled: usize,
    pub spill_paths: Vec<PathBuf>,
}

/// Stable sort of all `input` rows by `order_by` (nulls placement per field). Output: batches of <= `batch_rows` rows.
/// If buffering exceeds `budget`, sort runs, spill them to `spill_dir`, k-way merge. Result identical either way.
pub fn sort_batches(
    input: Vec<Batch>,
    schema: SchemaRef,
    order_by: &[SortField],
    budget: &Arc<MemoryBudget>,
    spill_dir: &Path,
    batch_rows: usize,
) -> Result<Vec<Batch>> {
    sort_batches_with_stats(input, schema, order_by, budget, spill_dir, batch_rows).map(|(batches, _)| batches)
}

/// Same as `sort_batches`, also reporting whether and how much was spilled.
pub fn sort_batches_with_stats(
    input: Vec<Batch>,
    schema: SchemaRef,
    order_by: &[SortField],
    budget: &Arc<MemoryBudget>,
    spill_dir: &Path,
    batch_rows: usize,
) -> Result<(Vec<Batch>, SortStats)> {
    if batch_rows == 0 {
        return Err(EngineError::Plan("sort batch_rows must be > 0".into()));
    }
    let mut stats = SortStats::default();
    if input.iter().all(|b| b.num_rows == 0) {
        return Ok((Vec::new(), stats));
    }
    let total_bytes: usize = input.iter().map(Batch::byte_size).sum();
    match budget.try_reserve(total_bytes) {
        Ok(_held_for_the_sort) => {
            let merged = Batch::concat(schema, &input)?;
            let order = sort_indices(&merged, order_by)?;
            Ok((chunk_take(&merged, &order, batch_rows), stats))
        }
        Err(_) => {
            let out = external_sort(input, schema, order_by, budget, spill_dir, batch_rows, &mut stats)?;
            Ok((out, stats))
        }
    }
}

fn external_sort(
    input: Vec<Batch>,
    schema: SchemaRef,
    order_by: &[SortField],
    budget: &Arc<MemoryBudget>,
    spill_dir: &Path,
    batch_rows: usize,
    stats: &mut SortStats,
) -> Result<Vec<Batch>> {
    // Pieces are processed in input order and bounded by batch_rows, so a run is a contiguous
    // input range and run index order is input order (the stability tie-break relies on this).
    let mut pending: VecDeque<Batch> = input.into_iter().flat_map(|b| split_rows(b, batch_rows)).collect();

    let mut runs: Vec<SpillFile> = Vec::new();
    let mut run_buf: Vec<Batch> = Vec::new();
    let mut run_res = budget.try_reserve(0)?;
    while let Some(piece) = pending.pop_front() {
        let size = piece.byte_size();
        if run_res.grow(size).is_ok() {
            run_buf.push(piece);
            continue;
        }
        if !run_buf.is_empty() {
            // Budget full: close this run, then retry the same piece on an empty run.
            pending.push_front(piece);
            runs.push(spill_run(std::mem::take(&mut run_buf), &schema, order_by, spill_dir, batch_rows)?);
            run_res = budget.try_reserve(0)?;
            continue;
        }
        if piece.num_rows > 1 {
            // Even alone the piece does not fit: halve it and retry, preserving row order.
            let half = piece.num_rows / 2;
            let rest = piece.num_rows - half;
            pending.push_front(slice_compact(&piece, half, rest));
            pending.push_front(slice_compact(&piece, 0, half));
            continue;
        }
        return Err(EngineError::Memory { requested: size, available: budget.available() });
    }
    if !run_buf.is_empty() {
        runs.push(spill_run(run_buf, &schema, order_by, spill_dir, batch_rows)?);
    }
    drop(run_res);

    stats.runs_spilled = runs.len();
    stats.spill_paths = runs.iter().map(SpillFile::path).collect();
    merge_runs(&runs, schema, order_by, batch_rows)
}

fn split_rows(batch: Batch, batch_rows: usize) -> Vec<Batch> {
    if batch.num_rows == 0 {
        return Vec::new();
    }
    if batch.num_rows <= batch_rows {
        return vec![batch];
    }
    (0..batch.num_rows)
        .step_by(batch_rows)
        .map(|start| slice_compact(&batch, start, batch_rows.min(batch.num_rows - start)))
        .collect()
}

/// Rows `start..start + len` as a standalone batch. Utf8 bytes are copied, because `Batch::slice`
/// shares the parent's string buffers and `byte_size` would charge the whole buffer to every piece.
fn slice_compact(batch: &Batch, start: usize, len: usize) -> Batch {
    let end = start + len;
    let columns = batch
        .columns
        .iter()
        .map(|c| {
            let values = match &c.values {
                Values::Bool(v) => Values::Bool(v[start..end].to_vec()),
                Values::Int64(v) => Values::Int64(v[start..end].to_vec()),
                Values::Float64(v) => Values::Float64(v[start..end].to_vec()),
                Values::Utf8(sv) => {
                    let strs: Vec<&str> = (start..end).map(|i| sv.get(i)).collect();
                    Values::Utf8(StringViews::from_strs(&strs))
                }
            };
            let validity = c.validity.as_ref().map(|m| m[start..end].to_vec());
            Column { values, validity }
        })
        .collect();
    Batch { schema: batch.schema.clone(), columns, num_rows: len }
}

/// Sorts buffered pieces in memory and writes them to one spill file as `batch_rows` chunks.
fn spill_run(
    pieces: Vec<Batch>,
    schema: &SchemaRef,
    order_by: &[SortField],
    spill_dir: &Path,
    batch_rows: usize,
) -> Result<SpillFile> {
    let merged = Batch::concat(schema.clone(), &pieces)?;
    drop(pieces);
    let order = sort_indices(&merged, order_by)?;
    let mut writer = SpillWriter::create(spill_dir, schema.clone())?;
    for chunk in chunk_take(&merged, &order, batch_rows) {
        writer.write(&chunk)?;
    }
    writer.finish()
}

/// One run being consumed by the merge. `chunk` is the batch currently at `pos`.
struct Cursor {
    reader: SpillReader,
    chunk: Option<Arc<Batch>>,
    keys: Vec<Column>,
    pos: usize,
}

/// Loads the next non-empty chunk. Returns false when the run is exhausted.
fn advance(cursor: &mut Cursor, order_by: &[SortField]) -> Result<bool> {
    loop {
        match cursor.reader.next_batch()? {
            None => {
                cursor.chunk = None;
                cursor.keys.clear();
                return Ok(false);
            }
            Some(batch) if batch.num_rows == 0 => continue,
            Some(batch) => {
                cursor.keys = key_columns(&batch, order_by)?;
                cursor.chunk = Some(Arc::new(batch));
                cursor.pos = 0;
                return Ok(true);
            }
        }
    }
}

/// Ties on the key fall back to run index, which is input order.
fn heap_less(cursors: &[Cursor], a: usize, b: usize, order_by: &[SortField]) -> bool {
    let (ca, cb) = (&cursors[a], &cursors[b]);
    match cmp_rows(&ca.keys, ca.pos, &cb.keys, cb.pos, order_by) {
        Ordering::Equal => a < b,
        o => o == Ordering::Less,
    }
}

fn heap_push<F: Fn(usize, usize) -> bool>(heap: &mut Vec<usize>, item: usize, less: F) {
    heap.push(item);
    let mut i = heap.len() - 1;
    while i > 0 {
        let parent = (i - 1) / 2;
        if less(heap[i], heap[parent]) {
            heap.swap(i, parent);
            i = parent;
        } else {
            break;
        }
    }
}

fn heap_pop<F: Fn(usize, usize) -> bool>(heap: &mut Vec<usize>, less: F) -> Option<usize> {
    let last = heap.pop()?;
    if heap.is_empty() {
        return Some(last);
    }
    let top = std::mem::replace(&mut heap[0], last);
    let mut i = 0;
    loop {
        let (l, r) = (2 * i + 1, 2 * i + 2);
        let mut smallest = i;
        if l < heap.len() && less(heap[l], heap[smallest]) {
            smallest = l;
        }
        if r < heap.len() && less(heap[r], heap[smallest]) {
            smallest = r;
        }
        if smallest == i {
            break;
        }
        heap.swap(i, smallest);
        i = smallest;
    }
    Some(top)
}

/// K-way merge: holds one chunk per run, streams output in `batch_rows` batches.
fn merge_runs(runs: &[SpillFile], schema: SchemaRef, order_by: &[SortField], batch_rows: usize) -> Result<Vec<Batch>> {
    let mut cursors: Vec<Cursor> = Vec::with_capacity(runs.len());
    for run in runs {
        cursors.push(Cursor { reader: run.reader()?, chunk: None, keys: Vec::new(), pos: 0 });
    }
    let mut heap: Vec<usize> = Vec::with_capacity(cursors.len());
    for id in 0..cursors.len() {
        if advance(&mut cursors[id], order_by)? {
            heap_push(&mut heap, id, |a, b| heap_less(&cursors, a, b, order_by));
        }
    }

    let mut out = OutBuilder::new(schema, batch_rows);
    while let Some(id) = heap_pop(&mut heap, |a, b| heap_less(&cursors, a, b, order_by)) {
        let cursor = &mut cursors[id];
        let Some(chunk) = cursor.chunk.clone() else { continue };
        out.push(Arc::clone(&chunk), cursor.pos)?;
        cursor.pos += 1;
        if cursor.pos < chunk.num_rows {
            heap_push(&mut heap, id, |a, b| heap_less(&cursors, a, b, order_by));
        } else if advance(&mut cursors[id], order_by)? {
            heap_push(&mut heap, id, |a, b| heap_less(&cursors, a, b, order_by));
        }
    }
    out.finish()
}

/// Gathers emitted rows as (chunk, row-range) pieces and materializes a batch every `batch_rows` rows.
struct OutBuilder {
    schema: SchemaRef,
    batch_rows: usize,
    pieces: Vec<(Arc<Batch>, usize, usize)>,
    rows: usize,
    out: Vec<Batch>,
}

impl OutBuilder {
    fn new(schema: SchemaRef, batch_rows: usize) -> Self {
        OutBuilder { schema, batch_rows, pieces: Vec::new(), rows: 0, out: Vec::new() }
    }

    fn push(&mut self, src: Arc<Batch>, row: usize) -> Result<()> {
        match self.pieces.last_mut() {
            Some((last, start, len)) if Arc::ptr_eq(last, &src) && *start + *len == row => *len += 1,
            _ => self.pieces.push((src, row, 1)),
        }
        self.rows += 1;
        if self.rows == self.batch_rows {
            self.flush()?;
        }
        Ok(())
    }

    fn flush(&mut self) -> Result<()> {
        if self.rows == 0 {
            return Ok(());
        }
        let columns = (0..self.schema.fields.len())
            .map(|j| {
                let parts: Vec<Column> = self.pieces.iter().map(|(b, s, l)| b.columns[j].slice(*s, *l)).collect();
                let refs: Vec<&Column> = parts.iter().collect();
                Column::concat(&refs)
            })
            .collect::<Result<Vec<Column>>>()?;
        self.out.push(Batch::try_new(self.schema.clone(), columns)?);
        self.pieces.clear();
        self.rows = 0;
        Ok(())
    }

    fn finish(mut self) -> Result<Vec<Batch>> {
        self.flush()?;
        Ok(self.out)
    }
}

/// Stable permutation of `batch` rows by `order_by`.
fn sort_indices(batch: &Batch, order_by: &[SortField]) -> Result<Vec<usize>> {
    let keys = key_columns(batch, order_by)?;
    let mut order: Vec<usize> = (0..batch.num_rows).collect();
    order.sort_by(|&a, &b| cmp_rows(&keys, a, &keys, b, order_by));
    Ok(order)
}

/// Splits `batch` into `batch_rows`-sized batches following the permutation.
fn chunk_take(batch: &Batch, order: &[usize], batch_rows: usize) -> Vec<Batch> {
    order.chunks(batch_rows).map(|idx| batch.take(idx)).collect()
}

fn key_columns(batch: &Batch, order_by: &[SortField]) -> Result<Vec<Column>> {
    order_by.iter().map(|f| key_column(&f.expr, batch)).collect()
}

/// Plain column refs are read directly; other expressions go through the evaluator.
fn key_column(expr: &Expr, batch: &Batch) -> Result<Column> {
    match expr {
        Expr::Column { index, .. } => batch
            .columns
            .get(*index)
            .cloned()
            .ok_or_else(|| EngineError::Plan(format!("sort key column index {index} out of range"))),
        other => eval(other, batch),
    }
}

fn cmp_rows(ka: &[Column], i: usize, kb: &[Column], j: usize, order_by: &[SortField]) -> Ordering {
    for (field, (a, b)) in order_by.iter().zip(ka.iter().zip(kb)) {
        let o = cmp_at(a, i, b, j, field);
        if o != Ordering::Equal {
            return o;
        }
    }
    Ordering::Equal
}

fn cmp_at(a: &Column, i: usize, b: &Column, j: usize, field: &SortField) -> Ordering {
    match (a.is_valid(i), b.is_valid(j)) {
        (false, false) => Ordering::Equal,
        (false, true) => {
            if field.nulls_first {
                Ordering::Less
            } else {
                Ordering::Greater
            }
        }
        (true, false) => {
            if field.nulls_first {
                Ordering::Greater
            } else {
                Ordering::Less
            }
        }
        (true, true) => {
            let o = cmp_values(a, i, b, j);
            if field.ascending {
                o
            } else {
                o.reverse()
            }
        }
    }
}

fn cmp_values(a: &Column, i: usize, b: &Column, j: usize) -> Ordering {
    match (&a.values, &b.values) {
        (Values::Bool(x), Values::Bool(y)) => x[i].cmp(&y[j]),
        (Values::Int64(x), Values::Int64(y)) => x[i].cmp(&y[j]),
        (Values::Float64(x), Values::Float64(y)) => x[i].total_cmp(&y[j]),
        (Values::Utf8(x), Values::Utf8(y)) => x.get(i).cmp(y.get(j)),
        // Key columns of one sort all come from one schema, so mixed types cannot occur.
        _ => Ordering::Equal,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::spill::test_support::canonical_rows;
    use crate::types::{DataType, Field, ScalarValue, Schema};
    use rand::{rngs::StdRng, Rng, SeedableRng};
    use std::fs;

    /// Input schema: id (never null), k1 Int64?, k2 Utf8?, k3 Float64?
    fn schema() -> SchemaRef {
        Arc::new(Schema::new(vec![
            Field::new("id", DataType::Int64, false),
            Field::new("k1", DataType::Int64, true),
            Field::new("k2", DataType::Utf8, true),
            Field::new("k3", DataType::Float64, true),
        ]))
    }

    struct Row {
        id: i64,
        k1: Option<i64>,
        k2: Option<String>,
        k3: Option<f64>,
    }

    fn build(rows: &[Row]) -> Batch {
        let id = Column::from_i64(rows.iter().map(|r| r.id).collect());
        let k1 = Column::from_scalars(
            DataType::Int64,
            &rows.iter().map(|r| r.k1.map_or(ScalarValue::Null(DataType::Int64), ScalarValue::Int64)).collect::<Vec<_>>(),
        )
        .unwrap();
        let k2 = Column::from_scalars(
            DataType::Utf8,
            &rows.iter().map(|r| r.k2.clone().map_or(ScalarValue::Null(DataType::Utf8), ScalarValue::Utf8)).collect::<Vec<_>>(),
        )
        .unwrap();
        let k3 = Column::from_scalars(
            DataType::Float64,
            &rows.iter().map(|r| r.k3.map_or(ScalarValue::Null(DataType::Float64), ScalarValue::Float64)).collect::<Vec<_>>(),
        )
        .unwrap();
        Batch::try_new(schema(), vec![id, k1, k2, k3]).unwrap()
    }

    fn row(id: i64, k1: Option<i64>, k2: Option<&str>, k3: Option<f64>) -> Row {
        Row { id, k1, k2: k2.map(str::to_string), k3 }
    }

    fn key(index: usize) -> Expr {
        let dt = schema().fields[index].data_type;
        Expr::Column { name: schema().fields[index].name.clone(), index, data_type: dt }
    }

    fn sf(index: usize, ascending: bool, nulls_first: bool) -> SortField {
        SortField { expr: key(index), ascending, nulls_first }
    }

    fn ids(batches: &[Batch]) -> Vec<i64> {
        batches
            .iter()
            .flat_map(|b| match &b.columns[0].values {
                Values::Int64(v) => v.clone(),
                _ => unreachable!("id is Int64"),
            })
            .collect()
    }

    fn spill_dir(tag: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!("ndq-sort-test-{tag}-{}", std::process::id()));
        fs::create_dir_all(&dir).unwrap();
        dir
    }

    fn big_budget() -> Arc<MemoryBudget> {
        MemoryBudget::new(1 << 30)
    }

    #[test]
    fn zero_column_batch_keeps_row_count_through_sort() {
        let zero_cols: SchemaRef = Arc::new(Schema::new(vec![]));
        let b = Batch { schema: zero_cols.clone(), columns: vec![], num_rows: 3 };
        let out = sort_batches(vec![b], zero_cols, &[], &big_budget(), &spill_dir("zero_col"), 10).unwrap();
        assert_eq!(out.iter().map(|b| b.num_rows).sum::<usize>(), 3);
    }

    #[test]
    fn empty_input_yields_no_batches() {
        let dir = spill_dir("empty");
        let out = sort_batches(vec![], schema(), &[sf(0, true, true)], &big_budget(), &dir, 10).unwrap();
        assert!(out.is_empty());
        let out = sort_batches(vec![Batch::empty(schema())], schema(), &[sf(0, true, true)], &big_budget(), &dir, 10).unwrap();
        assert!(out.is_empty());
        fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn zero_batch_rows_is_a_plan_error() {
        let dir = spill_dir("zero");
        let input = vec![build(&[row(0, Some(1), None, None)])];
        let err = sort_batches(input, schema(), &[], &big_budget(), &dir, 0).unwrap_err();
        assert!(matches!(err, EngineError::Plan(_)));
        fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn multi_key_ascending_and_descending_with_mixed_direction() {
        let dir = spill_dir("multi");
        let input = vec![build(&[
            row(0, Some(2), Some("b"), None),
            row(1, Some(1), Some("z"), None),
            row(2, Some(2), Some("a"), None),
            row(3, Some(1), Some("a"), None),
            row(4, Some(2), Some("a"), None),
        ])];
        // k1 ascending, then k2 descending.
        let order = [sf(1, true, false), sf(2, false, false)];
        let out = sort_batches(input, schema(), &order, &big_budget(), &dir, 100).unwrap();
        // k1=1: z(1), a(3); k1=2: b(0), then a(2) and a(4) keep input order.
        assert_eq!(ids(&out), vec![1, 3, 0, 2, 4]);
        fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn nulls_first_and_last_are_independent_of_direction() {
        let dir = spill_dir("nulls");
        let rows = [row(0, None, None, None), row(1, Some(5), None, None), row(2, Some(3), None, None), row(3, None, None, None)];
        let asc_last = sort_batches(vec![build(&rows)], schema(), &[sf(1, true, false)], &big_budget(), &dir, 100).unwrap();
        assert_eq!(ids(&asc_last), vec![2, 1, 0, 3]);
        let asc_first = sort_batches(vec![build(&rows)], schema(), &[sf(1, true, true)], &big_budget(), &dir, 100).unwrap();
        assert_eq!(ids(&asc_first), vec![0, 3, 2, 1]);
        let desc_first = sort_batches(vec![build(&rows)], schema(), &[sf(1, false, true)], &big_budget(), &dir, 100).unwrap();
        assert_eq!(ids(&desc_first), vec![0, 3, 1, 2]);
        let desc_last = sort_batches(vec![build(&rows)], schema(), &[sf(1, false, false)], &big_budget(), &dir, 100).unwrap();
        assert_eq!(ids(&desc_last), vec![1, 2, 0, 3]);
        fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn equal_keys_keep_input_order_across_batch_boundaries() {
        let dir = spill_dir("stable");
        let a = build(&[row(0, Some(1), None, None), row(1, Some(0), None, None)]);
        let b = build(&[row(2, Some(1), None, None), row(3, Some(0), None, None)]);
        let out = sort_batches(vec![a, b], schema(), &[sf(1, true, true)], &big_budget(), &dir, 3).unwrap();
        assert_eq!(ids(&out), vec![1, 3, 0, 2]);
        assert_eq!(out.iter().map(|b| b.num_rows).collect::<Vec<_>>(), vec![3, 1]);
        fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn float_total_order_and_utf8_byte_order() {
        let dir = spill_dir("float");
        let rows = [
            row(0, None, Some("b"), Some(f64::NAN)),
            row(1, None, Some("B"), Some(0.0)),
            row(2, None, Some("ab"), Some(-0.0)),
            row(3, None, Some("é"), Some(f64::NEG_INFINITY)),
        ];
        let by_float = sort_batches(vec![build(&rows)], schema(), &[sf(3, true, true)], &big_budget(), &dir, 10).unwrap();
        assert_eq!(ids(&by_float), vec![3, 2, 1, 0]);
        let by_str = sort_batches(vec![build(&rows)], schema(), &[sf(2, true, true)], &big_budget(), &dir, 10).unwrap();
        assert_eq!(ids(&by_str), vec![1, 2, 0, 3]);
        fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn output_batches_respect_batch_rows_and_keep_every_row() {
        let dir = spill_dir("chunks");
        let rows: Vec<Row> = (0..25).map(|i| row(i, Some(25 - i), None, None)).collect();
        let out = sort_batches(vec![build(&rows)], schema(), &[sf(1, true, true)], &big_budget(), &dir, 7).unwrap();
        assert!(out.iter().all(|b| b.num_rows <= 7));
        assert_eq!(out.iter().map(|b| b.num_rows).sum::<usize>(), 25);
        assert_eq!(ids(&out)[0], 24);
        fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn non_column_key_result_matches_the_evaluator_exactly() {
        let dir = spill_dir("expr");
        let expr = Expr::Literal(ScalarValue::Int64(1));
        let order = [SortField { expr: expr.clone(), ascending: true, nulls_first: true }];
        let batch = build(&[row(0, Some(1), None, None), row(1, Some(2), None, None)]);
        // Sort must not swallow or rewrite the evaluator's outcome, whatever it currently is.
        let expected = eval(&expr, &batch).map(|_| ());
        let got = sort_batches(vec![batch], schema(), &order, &big_budget(), &dir, 10).map(|_| ());
        assert_eq!(got, expected);
        fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn key_index_out_of_range_is_a_plan_error() {
        let dir = spill_dir("oob");
        let bad = SortField {
            expr: Expr::Column { name: "nope".into(), index: 9, data_type: DataType::Int64 },
            ascending: true,
            nulls_first: true,
        };
        let input = vec![build(&[row(0, Some(1), None, None)])];
        let err = sort_batches(input, schema(), &[bad], &big_budget(), &dir, 10).unwrap_err();
        assert!(matches!(err, EngineError::Plan(_)));
        fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn in_memory_path_reserves_and_releases_budget() {
        let dir = spill_dir("reserve");
        let budget = MemoryBudget::new(1 << 30);
        let child = budget.child(1 << 29);
        let input = vec![build(&[row(0, Some(3), None, None), row(1, Some(1), None, None)])];
        let (_, stats) = sort_batches_with_stats(input, schema(), &[sf(1, true, true)], &child, &dir, 10).unwrap();
        assert_eq!(stats.runs_spilled, 0);
        assert_eq!((child.used(), budget.used()), (0, 0));
        fs::remove_dir_all(&dir).unwrap();
    }

    fn random_rows(rng: &mut StdRng, n: i64) -> Vec<Row> {
        const STRS: &[&str] = &[
            "", "a", "exactly12byt", "exactly123ch", "a string well over twelve bytes long", "ümlaut-long-string-é", "zz",
        ];
        const FLOATS: &[f64] = &[f64::NAN, -0.0, 0.0, f64::INFINITY, f64::NEG_INFINITY, 1.5];
        (0..n)
            .map(|id| Row {
                id,
                k1: if rng.gen_bool(0.1) { None } else { Some(rng.gen_range(0..20)) },
                k2: if rng.gen_bool(0.1) { None } else { Some(STRS[rng.gen_range(0..STRS.len())].to_string()) },
                k3: if rng.gen_bool(0.1) {
                    None
                } else if rng.gen_bool(0.5) {
                    Some(FLOATS[rng.gen_range(0..FLOATS.len())])
                } else {
                    Some(rng.gen_range(-5.0..5.0))
                },
            })
            .collect()
    }

    #[test]
    fn external_sort_equals_in_memory_sort_on_20k_rows_with_tiny_budget() {
        let dir = spill_dir("external");
        let mut rng = StdRng::seed_from_u64(42);
        let rows = random_rows(&mut rng, 20_000);
        let input: Vec<Batch> = rows.chunks(1000).map(build).collect();
        let order = [sf(1, true, true), sf(2, false, false), sf(3, true, false)];

        let (reference, ref_stats) =
            sort_batches_with_stats(input.clone(), schema(), &order, &big_budget(), &dir, 512).unwrap();
        assert_eq!(ref_stats.runs_spilled, 0);

        let tiny = MemoryBudget::new(64 * 1024);
        let (external, stats) = sort_batches_with_stats(input, schema(), &order, &tiny, &dir, 512).unwrap();
        assert!(stats.runs_spilled >= 2, "expected spilled runs, got {}", stats.runs_spilled);
        assert_eq!(stats.spill_paths.len(), stats.runs_spilled);
        assert!(stats.spill_paths.iter().all(|p| !p.exists()), "spill files must be deleted after the sort");
        assert_eq!(tiny.used(), 0);

        assert_eq!(ids(&external), ids(&reference));
        assert_eq!(canonical_rows(&external), canonical_rows(&reference));
        assert_eq!(external.iter().map(|b| b.num_rows).sum::<usize>(), 20_000);
        assert!(external.iter().all(|b| b.num_rows <= 512));
        fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn external_sort_is_stable_under_heavy_duplication() {
        let dir = spill_dir("dupes");
        let mut rng = StdRng::seed_from_u64(9);
        let rows: Vec<Row> = (0..5_000).map(|id| row(id, Some(rng.gen_range(0..3)), None, None)).collect();
        let input: Vec<Batch> = rows.chunks(250).map(build).collect();
        let order = [sf(1, true, true)];
        let tiny = MemoryBudget::new(16 * 1024);
        let (out, stats) = sort_batches_with_stats(input, schema(), &order, &tiny, &dir, 128).unwrap();
        assert!(stats.runs_spilled > 1);
        // Within each key value the original id order must survive the merge.
        let got = ids(&out);
        let mut last_by_key: [i64; 3] = [-1; 3];
        for (pos, id) in got.iter().enumerate() {
            let k = rows[*id as usize].k1.unwrap_or(0) as usize;
            assert!(*id > last_by_key[k], "stability broken at output position {pos}");
            last_by_key[k] = *id;
        }
        fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn external_sort_with_long_strings_in_one_big_batch_fits_budget() {
        // Each sliced piece shares the parent's string buffer, so its byte_size must not
        // count the whole buffer or the pieces would never fit a small budget.
        let dir = spill_dir("longstr");
        let rows: Vec<Row> = (0..2_000)
            .map(|id| Row {
                id,
                k1: Some((id * 7919) % 101),
                k2: Some(format!("{:05}-{}", (id * 31) % 2_000, "x".repeat(95))),
                k3: None,
            })
            .collect();
        let order = [sf(1, true, true), sf(2, true, true)];
        let (reference, _) =
            sort_batches_with_stats(vec![build(&rows)], schema(), &order, &big_budget(), &dir, 256).unwrap();
        let tiny = MemoryBudget::new(64 * 1024);
        let (external, stats) =
            sort_batches_with_stats(vec![build(&rows)], schema(), &order, &tiny, &dir, 256).unwrap();
        assert!(stats.runs_spilled >= 2);
        assert_eq!(canonical_rows(&external), canonical_rows(&reference));
        fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn oversized_single_row_under_budget_is_a_memory_error() {
        let dir = spill_dir("oom");
        let big = Row { id: 0, k1: Some(1), k2: Some("x".repeat(10_000)), k3: None };
        let tiny = MemoryBudget::new(64);
        let err = sort_batches(vec![build(&[big])], schema(), &[sf(1, true, true)], &tiny, &dir, 4).unwrap_err();
        assert!(matches!(err, EngineError::Memory { .. }));
        assert_eq!(tiny.used(), 0);
        fs::remove_dir_all(&dir).unwrap();
    }
}
