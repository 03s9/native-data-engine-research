//! SLICE D2. Hash join (build right, probe left); Grace partition+spill when build side exceeds budget.
//!
//! Structure: every join runs `join_pair(left, right, depth)`. The build side (right) is processed in
//! chunks. Each chunk gets a hash table, and the whole probe side (left) is streamed against it. Matched
//! pairs are emitted during probing. Unmatched right rows (Right/Full) are emitted after each chunk. A
//! final pass over left emits unmatched left rows (Left/Full), and semi/anti rows (Semi/Anti). One
//! global bitmap of left rows tracks matches across chunks.
//! When the build side fits the budget, there is one chunk. When it does not, `join_pair` Grace-partitions
//! both sides to disk with a depth-seeded hash, and recurses per partition pair. A partition that is
//! not shrinking (all rows share the key) or that is still too big at max depth falls back to small
//! build chunks sized to the budget.
use crate::column::{Batch, Column, StringViews, Values};
use crate::eval::eval;
use crate::expr::Expr;
use crate::memory::{MemoryBudget, Reservation};
use crate::plan::JoinType;
use crate::spill::{SpillFile, SpillWriter};
use crate::types::{DataType, EngineError, Field, Result, Schema, SchemaRef};
use std::collections::hash_map::DefaultHasher;
use std::collections::HashMap;
use std::hash::{Hash, Hasher};
use std::path::Path;
use std::sync::Arc;

const BATCH_ROWS: usize = 8192;
const PARTITIONS: usize = 16;
const MAX_DEPTH: usize = 3;
/// Approximate hash-table entry overhead per build row, charged on top of the column bytes.
const ENTRY_OVERHEAD: usize = 64;
/// Approximate key bytes per join key column, charged per build row.
const KEY_BYTES: usize = 16;

type Pair = (Option<usize>, Option<usize>);

/// What the join did, for tests and diagnostics.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct JoinStats {
    /// Spill files written (Grace partitions, both sides, all depths).
    pub spill_files: usize,
    /// Deepest recursion level reached (0 = no spill).
    pub max_depth: usize,
    /// Build chunks processed (1 per in-memory join).
    pub build_chunks: usize,
}

/// NULL keys never match. Duplicate keys: cross-product of matches. Skewed/duplicate-heavy partitions must
/// terminate (fall back to chunked nested loop when re-partitioning makes no progress).
pub fn hash_join(
    left: Vec<Batch>, right: Vec<Batch>, join_type: JoinType, on: &[(Expr, Expr)], schema: SchemaRef,
    budget: &Arc<MemoryBudget>, spill_dir: &Path,
) -> Result<Vec<Batch>> {
    hash_join_with_stats(left, right, join_type, on, schema, budget, spill_dir).map(|(batches, _)| batches)
}

/// `hash_join` that also reports what it did.
pub fn hash_join_with_stats(
    left: Vec<Batch>, right: Vec<Batch>, join_type: JoinType, on: &[(Expr, Expr)], schema: SchemaRef,
    budget: &Arc<MemoryBudget>, spill_dir: &Path,
) -> Result<(Vec<Batch>, JoinStats)> {
    for (l, r) in on {
        if l.return_type() != r.return_type() {
            return Err(EngineError::Type(format!(
                "join key types differ: {:?} vs {:?}",
                l.return_type(),
                r.return_type()
            )));
        }
    }
    let n = schema.fields.len();
    let semi = matches!(join_type, JoinType::Semi | JoinType::Anti);
    // Input widths. A side with no batches takes its width from the output schema.
    let (il, ir) = match (left.first(), right.first()) {
        (Some(l), Some(r)) => (l.columns.len(), r.columns.len()),
        (Some(l), None) => (l.columns.len(), if semi { 0 } else { n.saturating_sub(l.columns.len()) }),
        (None, Some(r)) => (if semi { n } else { n.saturating_sub(r.columns.len()) }, r.columns.len()),
        (None, None) => (n, 0),
    };
    let width_ok = if semi { il == n } else { il + ir == n };
    if !width_ok {
        return Err(EngineError::Plan(format!(
            "join output schema has {n} fields, inputs have {il} left and {ir} right columns"
        )));
    }
    let left_fields = schema.fields[..il].to_vec();
    let right_fields = if semi { Vec::new() } else { schema.fields[il..].to_vec() };
    let left_schema = match left.first() {
        Some(b) => b.schema.clone(),
        None => Arc::new(Schema::new(left_fields.clone())),
    };
    let right_schema = match right.first() {
        Some(b) => b.schema.clone(),
        None => Arc::new(Schema::new(right_fields.clone())),
    };
    let mut join = Join {
        jt: join_type,
        on,
        schema: schema.clone(),
        left_fields,
        right_fields,
        left_schema,
        right_schema,
        budget,
        spill_dir,
        out: Vec::new(),
        stats: JoinStats::default(),
    };
    join.join_pair(Source::Mem(&left), Source::Mem(&right), 0)?;
    let mut out = join.out;
    if out.is_empty() {
        out.push(Batch::concat(schema, &[])?);
    }
    Ok((out, join.stats))
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Side {
    Left,
    Right,
}

/// A join input: in-memory batches or one spilled partition file. A file carries the in-memory byte measure
/// of its rows, taken when it was written (see `data_bytes`).
#[derive(Clone, Copy)]
enum Source<'a> {
    Mem(&'a [Batch]),
    File(&'a SpillFile, usize),
}

impl Source<'_> {
    fn rows(&self) -> usize {
        match self {
            Source::Mem(b) => b.iter().map(|x| x.num_rows).sum(),
            Source::File(f, _) => f.num_rows(),
        }
    }

    /// In-memory bytes of the rows once compacted: the measure the build reservation is derived from.
    fn bytes(&self) -> usize {
        match self {
            Source::Mem(b) => b.iter().map(data_bytes).sum(),
            Source::File(_, bytes) => *bytes,
        }
    }

    fn for_each(&self, f: &mut dyn FnMut(&Batch) -> Result<()>) -> Result<()> {
        match self {
            Source::Mem(batches) => {
                for b in batches.iter() {
                    f(b)?;
                }
                Ok(())
            }
            Source::File(file, _) => {
                let mut reader = file.reader()?;
                while let Some(b) = reader.next_batch()? {
                    f(&b)?;
                }
                Ok(())
            }
        }
    }
}

struct Join<'a> {
    jt: JoinType,
    on: &'a [(Expr, Expr)],
    schema: SchemaRef,
    left_fields: Vec<Field>,
    right_fields: Vec<Field>,
    left_schema: SchemaRef,
    right_schema: SchemaRef,
    budget: &'a Arc<MemoryBudget>,
    spill_dir: &'a Path,
    out: Vec<Batch>,
    stats: JoinStats,
}

impl Join<'_> {
    /// Estimated in-memory footprint of `src` as a build side.
    fn estimate(&self, src: Source) -> usize {
        src.bytes() + src.rows() * (ENTRY_OVERHEAD + KEY_BYTES * self.on.len())
    }

    fn join_pair(&mut self, l: Source, r: Source, depth: usize) -> Result<()> {
        self.stats.max_depth = self.stats.max_depth.max(depth);
        let est = self.estimate(r);
        let avail = self.budget.available();
        if est <= avail {
            return self.chunked(l, r, usize::MAX);
        }
        if depth >= MAX_DEPTH {
            // Fallback: nested loop over build chunks, each at most half the budget by measured bytes. A single
            // row is always attempted, so Memory is returned only when one row cannot fit at all.
            return self.chunked(l, r, (avail / 2).max(1));
        }
        let lparts = self.partition(l, Side::Left, depth + 1)?;
        let rparts = self.partition(r, Side::Right, depth + 1)?;
        let parent_rows = r.rows();
        for ((lp, lb), (rp, rb)) in lparts.iter().zip(&rparts) {
            // Re-partitioning moved no rows out of the build side: this key set cannot be split by hash.
            let next = if rp.num_rows() == parent_rows { MAX_DEPTH } else { depth + 1 };
            self.join_pair(Source::File(lp, *lb), Source::File(rp, *rb), next)?;
        }
        Ok(())
    }

    /// Splits `src` into PARTITIONS spill files by a depth-seeded hash of the join key, each with its measured
    /// in-memory bytes. NULL-key rows go to partition 0 on both sides (they never match, so placement is
    /// irrelevant to the result).
    fn partition(&mut self, src: Source, side: Side, seed: usize) -> Result<Vec<(SpillFile, usize)>> {
        let schema = match side {
            Side::Left => self.left_schema.clone(),
            Side::Right => self.right_schema.clone(),
        };
        let mut writers = (0..PARTITIONS)
            .map(|_| SpillWriter::create(self.spill_dir, schema.clone()))
            .collect::<Result<Vec<_>>>()?;
        self.stats.spill_files += PARTITIONS;
        let mut bytes = vec![0usize; PARTITIONS];
        let on = self.on;
        let exprs: Vec<&Expr> = on.iter().map(|(l, r)| if side == Side::Left { l } else { r }).collect();
        src.for_each(&mut |b: &Batch| {
            if b.num_rows == 0 {
                return Ok(());
            }
            let keys: Vec<Column> = exprs.iter().map(|e| eval(e, b)).collect::<Result<_>>()?;
            let mut buckets: Vec<Vec<usize>> = vec![Vec::new(); PARTITIONS];
            let mut buf = Vec::new();
            for i in 0..b.num_rows {
                buf.clear();
                let p = if encode_key(&keys, i, &mut buf) { partition_of(&buf, seed) } else { 0 };
                buckets[p].push(i);
            }
            for ((w, idx), nb) in writers.iter_mut().zip(&buckets).zip(bytes.iter_mut()) {
                if !idx.is_empty() {
                    let part = b.take(idx);
                    *nb += data_bytes(&part);
                    w.write(&part)?;
                }
            }
            Ok(())
        })?;
        writers.into_iter().zip(bytes).map(|(w, nb)| w.finish().map(|f| (f, nb))).collect()
    }

    /// Joins `l` with `r` in build chunks. A chunk takes right rows while their measured bytes, plus hash
    /// overhead, stay within `target_bytes`. The first row of a chunk is always taken.
    fn chunked(&mut self, l: Source, r: Source, target_bytes: usize) -> Result<()> {
        let needs_left_pass = matches!(self.jt, JoinType::Left | JoinType::Full | JoinType::Semi | JoinType::Anti);
        let mut bits = if needs_left_pass { vec![false; l.rows()] } else { Vec::new() };
        let kcost = ENTRY_OVERHEAD + KEY_BYTES * self.on.len();
        let mut pending: Vec<Batch> = Vec::new();
        let mut pending_rows = 0usize;
        let mut pending_cost = 0usize;
        let mut chunks = 0usize;
        r.for_each(&mut |b: &Batch| {
            let mut off = 0;
            while off < b.num_rows {
                let mut end = off;
                while end < b.num_rows {
                    let cost = row_data_bytes(b, end) + kcost;
                    if pending_rows > 0 && pending_cost + cost > target_bytes {
                        break;
                    }
                    pending_cost += cost;
                    pending_rows += 1;
                    end += 1;
                }
                if end > off {
                    pending.push(compact_slice(b, off, end - off));
                }
                off = end;
                if off < b.num_rows {
                    let parts = std::mem::take(&mut pending);
                    pending_rows = 0;
                    pending_cost = 0;
                    self.run_chunk(l, &parts, &mut bits)?;
                    chunks += 1;
                }
            }
            Ok(())
        })?;
        if pending_rows > 0 || chunks == 0 {
            let parts = std::mem::take(&mut pending);
            self.run_chunk(l, &parts, &mut bits)?;
        }
        if needs_left_pass {
            let mut base = 0usize;
            l.for_each(&mut |lb: &Batch| {
                let mut pairs: Vec<Pair> = Vec::new();
                for i in 0..lb.num_rows {
                    let m = bits[base + i];
                    let emit = match self.jt {
                        JoinType::Semi => m,
                        _ => !m,
                    };
                    if emit {
                        pairs.push((Some(i), None));
                    }
                }
                base += lb.num_rows;
                self.emit(Some(lb), None, &pairs)
            })?;
        }
        Ok(())
    }

    /// One build chunk: hash the right rows, stream all of `l` past it, then emit unmatched right rows.
    fn run_chunk(&mut self, l: Source, parts: &[Batch], bits: &mut [bool]) -> Result<()> {
        self.stats.build_chunks += 1;
        let build = Batch::concat(self.right_schema.clone(), parts)?;
        let mut table = Table::build(build, self.on, self.budget)?;
        let emits = matches!(self.jt, JoinType::Inner | JoinType::Left | JoinType::Right | JoinType::Full);
        let on = self.on;
        let on_l: Vec<&Expr> = on.iter().map(|(le, _)| le).collect();
        let mut base = 0usize;
        l.for_each(&mut |lb: &Batch| {
            let pairs = table.probe(lb, base, &on_l, emits, bits)?;
            base += lb.num_rows;
            self.emit(Some(lb), Some(&table.build), &pairs)
        })?;
        if matches!(self.jt, JoinType::Right | JoinType::Full) {
            let pairs: Vec<Pair> = (0..table.build.num_rows)
                .filter(|&j| !table.matched[j])
                .map(|j| (None, Some(j)))
                .collect();
            self.emit(None, Some(&table.build), &pairs)?;
        }
        Ok(())
    }

    /// Materializes `pairs` (left row, right row) into output batches of at most BATCH_ROWS rows.
    fn emit(&mut self, left: Option<&Batch>, right: Option<&Batch>, pairs: &[Pair]) -> Result<()> {
        let semi = matches!(self.jt, JoinType::Semi | JoinType::Anti);
        for chunk in pairs.chunks(BATCH_ROWS) {
            let lidx: Vec<Option<usize>> = chunk.iter().map(|p| p.0).collect();
            let mut cols = gather(left, &self.left_fields, &lidx);
            if !semi {
                let ridx: Vec<Option<usize>> = chunk.iter().map(|p| p.1).collect();
                cols.extend(gather(right, &self.right_fields, &ridx));
            }
            self.out.push(Batch::try_new(self.schema.clone(), cols)?);
        }
        Ok(())
    }
}

/// Hash table over one build chunk. Rows with a NULL key are kept (for Right/Full) but never indexed.
struct Table {
    build: Batch,
    index: HashMap<Vec<u8>, Vec<usize>>,
    matched: Vec<bool>,
    _res: Reservation,
}

impl Table {
    fn build(build: Batch, on: &[(Expr, Expr)], budget: &Arc<MemoryBudget>) -> Result<Table> {
        // `build` is compacted by the caller, so byte_size counts only the rows it holds.
        let est = build.byte_size() + build.num_rows * (ENTRY_OVERHEAD + KEY_BYTES * on.len());
        let res = budget.try_reserve(est)?;
        let mut index: HashMap<Vec<u8>, Vec<usize>> = HashMap::new();
        if build.num_rows > 0 {
            let keys: Vec<Column> = on.iter().map(|(_, r)| eval(r, &build)).collect::<Result<_>>()?;
            let mut buf = Vec::new();
            for i in 0..build.num_rows {
                buf.clear();
                if !encode_key(&keys, i, &mut buf) {
                    continue;
                }
                match index.get_mut(buf.as_slice()) {
                    Some(rows) => rows.push(i),
                    None => {
                        index.insert(buf.clone(), vec![i]);
                    }
                }
            }
        }
        let matched = vec![false; build.num_rows];
        Ok(Table { build, index, matched, _res: res })
    }

    /// Probes `probe` (left rows starting at global index `base`). Returns matched pairs when `emits`.
    /// Sets `bits` for matched left rows when `bits` is non-empty.
    fn probe(&mut self, probe: &Batch, base: usize, on_l: &[&Expr], emits: bool, bits: &mut [bool]) -> Result<Vec<Pair>> {
        let mut pairs = Vec::new();
        if probe.num_rows == 0 {
            return Ok(pairs);
        }
        let keys: Vec<Column> = on_l.iter().map(|e| eval(e, probe)).collect::<Result<_>>()?;
        let track = !bits.is_empty();
        let mut buf = Vec::new();
        for i in 0..probe.num_rows {
            buf.clear();
            if !encode_key(&keys, i, &mut buf) {
                continue;
            }
            let Some(js) = self.index.get(buf.as_slice()) else { continue };
            if track {
                bits[base + i] = true;
            }
            for &j in js {
                self.matched[j] = true;
                if emits {
                    pairs.push((Some(i), Some(j)));
                }
            }
        }
        Ok(pairs)
    }
}

/// Byte-encodes one row's key columns into `buf`. Each column is tagged NULL (0) or valid (1, then the
/// value). Returns false when any key is NULL. Float -0.0 and 0.0 encode identically, and all NaNs do too.
pub(crate) fn encode_key(cols: &[Column], row: usize, buf: &mut Vec<u8>) -> bool {
    let mut all_valid = true;
    for c in cols {
        if !c.is_valid(row) {
            buf.push(0);
            all_valid = false;
            continue;
        }
        buf.push(1);
        match &c.values {
            Values::Bool(v) => buf.push(u8::from(v[row])),
            Values::Int64(v) => buf.extend_from_slice(&v[row].to_le_bytes()),
            Values::Float64(v) => {
                let x = v[row];
                let bits = if x == 0.0 {
                    0u64
                } else if x.is_nan() {
                    f64::NAN.to_bits()
                } else {
                    x.to_bits()
                };
                buf.extend_from_slice(&bits.to_le_bytes());
            }
            Values::Utf8(s) => {
                let b = s.get(row).as_bytes();
                buf.extend_from_slice(&(b.len() as u32).to_le_bytes());
                buf.extend_from_slice(b);
            }
        }
    }
    all_valid
}

/// Bytes one row costs once its batch is compacted, as `Batch::byte_size` counts it. A string costs its
/// 16-byte view plus its bytes when longer than 12 (shorter strings are inline). Each column is charged one
/// validity byte as an upper bound, since `Batch::concat` may add a validity mask.
fn row_data_bytes(b: &Batch, i: usize) -> usize {
    b.columns
        .iter()
        .map(|c| {
            1 + match &c.values {
                Values::Bool(_) => 1,
                Values::Int64(_) | Values::Float64(_) => 8,
                Values::Utf8(s) => {
                    let v = &s.views[i];
                    16 + if v.len > 12 { v.len as usize } else { 0 }
                }
            }
        })
        .sum()
}

/// `row_data_bytes` summed over every row of `b`.
fn data_bytes(b: &Batch) -> usize {
    (0..b.num_rows).map(|i| row_data_bytes(b, i)).sum()
}

/// Copies rows `start..start + len` with their string bytes, so the result shares no parent string buffer.
fn compact_slice(batch: &Batch, start: usize, len: usize) -> Batch {
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

fn partition_of(key: &[u8], seed: usize) -> usize {
    let mut h = DefaultHasher::new();
    seed.hash(&mut h);
    key.hash(&mut h);
    (h.finish() % PARTITIONS as u64) as usize
}

/// Gathers rows by optional index. `None` produces a NULL. An absent or empty source yields all NULLs.
fn gather(src: Option<&Batch>, fields: &[Field], idx: &[Option<usize>]) -> Vec<Column> {
    match src {
        Some(b) if b.num_rows > 0 => b.columns.iter().map(|c| gather_opt(c, idx)).collect(),
        _ => fields.iter().map(|f| null_column(f.data_type, idx.len())).collect(),
    }
}

fn gather_opt(c: &Column, idx: &[Option<usize>]) -> Column {
    if idx.iter().all(Option::is_some) {
        let ix: Vec<usize> = idx.iter().flatten().copied().collect();
        return c.take(&ix);
    }
    let ix: Vec<usize> = idx.iter().map(|x| x.unwrap_or(0)).collect();
    let mut out = c.take(&ix);
    let mut validity = out.validity.take().unwrap_or_else(|| vec![true; ix.len()]);
    for (k, x) in idx.iter().enumerate() {
        if x.is_none() {
            validity[k] = false;
        }
    }
    out.validity = Some(validity);
    out
}

fn null_column(dt: DataType, n: usize) -> Column {
    let values = match dt {
        DataType::Boolean => Values::Bool(vec![false; n]),
        DataType::Int64 => Values::Int64(vec![0; n]),
        DataType::Float64 => Values::Float64(vec![0.0; n]),
        DataType::Utf8 => Values::Utf8(StringViews::from_strs(&vec![""; n])),
    };
    Column { values, validity: Some(vec![false; n]) }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::eval::{binary, col, lit};
    use crate::expr::BinaryOp;
    use crate::plan::JoinType::{Anti, Full, Inner, Left, Right, Semi};
    use crate::types::ScalarValue as S;
    use rand::{rngs::StdRng, Rng, SeedableRng};
    use std::path::PathBuf;
    use std::sync::mpsc;
    use std::time::Duration;

    fn tmp() -> PathBuf {
        std::env::temp_dir()
    }

    fn f(name: &str, dt: DataType) -> Field {
        Field::new(name, dt, true)
    }

    /// Left: (lk, lv). Right: (rk, rv). Output: lk, lv, rk, rv.
    fn left_schema() -> Schema {
        Schema::new(vec![f("lk", DataType::Int64), f("lv", DataType::Int64)])
    }

    fn right_schema() -> Schema {
        Schema::new(vec![f("rk", DataType::Int64), f("rv", DataType::Int64)])
    }

    fn out_schema(jt: JoinType) -> SchemaRef {
        let mut fields = vec![f("lk", DataType::Int64), f("lv", DataType::Int64)];
        if !matches!(jt, Semi | Anti) {
            fields.push(f("rk", DataType::Int64));
            fields.push(f("rv", DataType::Int64));
        }
        Arc::new(Schema::new(fields))
    }

    fn opt_col(v: &[Option<i64>]) -> Column {
        let vals: Vec<i64> = v.iter().map(|x| x.unwrap_or(0)).collect();
        let valid: Vec<bool> = v.iter().map(|x| x.is_some()).collect();
        Column::from_i64(vals).with_validity(valid)
    }

    fn batch(s: &Schema, cols: Vec<Column>) -> Batch {
        Batch::try_new(Arc::new(s.clone()), cols).unwrap()
    }

    fn side(keys: &[Option<i64>], vals: &[i64], s: &Schema) -> Batch {
        batch(s, vec![opt_col(keys), Column::from_i64(vals.to_vec())])
    }

    fn keys(keys: &[i64]) -> Vec<Option<i64>> {
        keys.iter().map(|k| Some(*k)).collect()
    }

    fn on_lr() -> Vec<(Expr, Expr)> {
        let ls = left_schema();
        let rs = right_schema();
        vec![(col(&ls, "lk").unwrap(), col(&rs, "rk").unwrap())]
    }

    /// Sorted rows as Option<i64> cells. NULL becomes None so padded slots compare by validity only.
    fn cells(batches: &[Batch]) -> Vec<Vec<Option<i64>>> {
        let mut out: Vec<Vec<Option<i64>>> = batches
            .iter()
            .flat_map(|b| (0..b.num_rows).map(move |i| b.row(i)))
            .map(|row| {
                row.into_iter()
                    .map(|v| match v {
                        S::Int64(x) => Some(x),
                        S::Null(_) => None,
                        other => panic!("unexpected scalar {other:?}"),
                    })
                    .collect()
            })
            .collect();
        out.sort();
        out
    }

    fn run(jt: JoinType, left: Vec<Batch>, right: Vec<Batch>, on: &[(Expr, Expr)], budget: &Arc<MemoryBudget>) -> Result<Vec<Batch>> {
        hash_join(left, right, jt, on, out_schema(jt), budget, &tmp())
    }

    fn big() -> Arc<MemoryBudget> {
        MemoryBudget::new(1 << 30)
    }

    // Left keys/values: (1,10) (2,20) (2,21) (NULL,30). Right: (2,200) (2,201) (3,300) (NULL,400).
    fn fixture() -> (Vec<Batch>, Vec<Batch>) {
        let l = side(&[Some(1), Some(2), Some(2), None], &[10, 20, 21, 30], &left_schema());
        let r = side(&[Some(2), Some(2), Some(3), None], &[200, 201, 300, 400], &right_schema());
        (vec![l], vec![r])
    }

    fn run_fixture(jt: JoinType) -> Vec<Vec<Option<i64>>> {
        let (l, r) = fixture();
        cells(&run(jt, l, r, &on_lr(), &big()).unwrap())
    }

    fn vrow(vals: &[Option<i64>]) -> Vec<Option<i64>> {
        vals.to_vec()
    }

    fn sr(vals: &[i64]) -> Vec<Option<i64>> {
        vals.iter().map(|v| Some(*v)).collect()
    }

    #[test]
    fn inner_join_duplicates_and_null_keys_never_match() {
        let want = vec![
            sr(&[2, 20, 2, 200]),
            sr(&[2, 20, 2, 201]),
            sr(&[2, 21, 2, 200]),
            sr(&[2, 21, 2, 201]),
        ];
        assert_eq!(run_fixture(Inner), want);
    }

    #[test]
    fn left_join_pads_unmatched_left_rows() {
        let got = run_fixture(Left);
        assert_eq!(got.len(), 6);
        assert!(got.contains(&vrow(&[Some(1), Some(10), None, None])));
        assert!(got.contains(&vrow(&[None, Some(30), None, None])));
    }

    #[test]
    fn right_join_pads_unmatched_right_rows() {
        let got = run_fixture(Right);
        assert_eq!(got.len(), 6);
        assert!(got.contains(&vrow(&[None, None, Some(3), Some(300)])));
        assert!(got.contains(&vrow(&[None, None, None, Some(400)])));
    }

    #[test]
    fn full_join_keeps_both_unmatched_sides() {
        let got = run_fixture(Full);
        assert_eq!(got.len(), 8);
        assert!(got.contains(&vrow(&[Some(1), Some(10), None, None])));
        assert!(got.contains(&vrow(&[None, Some(30), None, None])));
        assert!(got.contains(&vrow(&[None, None, Some(3), Some(300)])));
        assert!(got.contains(&vrow(&[None, None, None, Some(400)])));
    }

    #[test]
    fn semi_join_emits_each_left_row_once_with_left_columns_only() {
        let got = run_fixture(Semi);
        assert_eq!(got, vec![sr(&[2, 20]), sr(&[2, 21])]);
    }

    #[test]
    fn anti_join_keeps_unmatched_left_including_null_keys() {
        let got = run_fixture(Anti);
        assert_eq!(got, vec![vrow(&[None, Some(30)]), sr(&[1, 10])]);
    }

    #[test]
    fn nulls_on_both_sides_do_not_match_each_other() {
        let l = side(&[None, None], &[1, 2], &left_schema());
        let r = side(&[None], &[9], &right_schema());
        assert!(cells(&run(Inner, vec![l.clone()], vec![r.clone()], &on_lr(), &big()).unwrap()).is_empty());
        assert_eq!(cells(&run(Full, vec![l], vec![r], &on_lr(), &big()).unwrap()).len(), 3);
    }

    #[test]
    fn duplicate_keys_give_cross_product() {
        let l = side(&keys(&[5, 5, 5]), &[1, 2, 3], &left_schema());
        let r = side(&keys(&[5, 5, 5]), &[7, 8, 9], &right_schema());
        let got = run(Inner, vec![l], vec![r], &on_lr(), &big()).unwrap();
        assert_eq!(cells(&got).len(), 9);
    }

    #[test]
    fn multi_column_keys_match_on_both_columns() {
        let ls = Schema::new(vec![f("a", DataType::Int64), f("b", DataType::Int64), f("lv", DataType::Int64)]);
        let rs = Schema::new(vec![f("c", DataType::Int64), f("d", DataType::Int64), f("rv", DataType::Int64)]);
        let mut fields = ls.fields.clone();
        fields.extend(rs.fields.clone());
        let schema = Arc::new(Schema::new(fields));
        let l = batch(&ls, vec![Column::from_i64(vec![1, 1, 2]), Column::from_i64(vec![1, 2, 2]), Column::from_i64(vec![100, 200, 300])]);
        let r = batch(&rs, vec![Column::from_i64(vec![1, 2]), Column::from_i64(vec![2, 2]), Column::from_i64(vec![7, 8])]);
        let on = vec![
            (col(&ls, "a").unwrap(), col(&rs, "c").unwrap()),
            (col(&ls, "b").unwrap(), col(&rs, "d").unwrap()),
        ];
        let got = hash_join(vec![l], vec![r], Inner, &on, schema, &big(), &tmp()).unwrap();
        assert_eq!(cells(&got), vec![sr(&[1, 2, 200, 1, 2, 7]), sr(&[2, 2, 300, 2, 2, 8])]);
    }

    #[test]
    fn expression_keys_match_left_plus_one_equals_right() {
        let ls = left_schema();
        let rs = right_schema();
        let lk_plus_one = binary(col(&ls, "lk").unwrap(), BinaryOp::Add, lit(S::Int64(1))).unwrap();
        let on = vec![(lk_plus_one, col(&rs, "rk").unwrap())];
        let l = side(&keys(&[1, 2, 3]), &[10, 20, 30], &ls);
        let r = side(&keys(&[2, 4]), &[200, 400], &rs);
        let got = cells(&run(Inner, vec![l], vec![r], &on, &big()).unwrap());
        assert_eq!(got, vec![sr(&[1, 10, 2, 200]), sr(&[3, 30, 4, 400])]);
    }

    #[test]
    fn empty_right_side_left_join_is_all_unmatched() {
        let l = side(&keys(&[1, 2]), &[10, 20], &left_schema());
        let got = cells(&run(Left, vec![l], vec![], &on_lr(), &big()).unwrap());
        assert_eq!(got, vec![vrow(&[Some(1), Some(10), None, None]), vrow(&[Some(2), Some(20), None, None])]);
    }

    #[test]
    fn empty_right_side_anti_keeps_everything() {
        let l = side(&keys(&[1, 2]), &[10, 20], &left_schema());
        assert_eq!(cells(&run(Anti, vec![l], vec![], &on_lr(), &big()).unwrap()).len(), 2);
    }

    #[test]
    fn empty_left_side_right_join_is_all_unmatched_right() {
        let r = side(&keys(&[3]), &[300], &right_schema());
        let got = cells(&run(Right, vec![], vec![r], &on_lr(), &big()).unwrap());
        assert_eq!(got, vec![vrow(&[None, None, Some(3), Some(300)])]);
    }

    #[test]
    fn empty_inputs_produce_no_rows_for_inner() {
        assert!(cells(&run(Inner, vec![], vec![], &on_lr(), &big()).unwrap()).is_empty());
    }

    #[test]
    fn output_is_split_into_batches_of_at_most_8192_rows() {
        let l = side(&keys(&[1; 200]), &(0..200).collect::<Vec<_>>(), &left_schema());
        let r = side(&keys(&[1; 100]), &(0..100).collect::<Vec<_>>(), &right_schema());
        let got = run(Inner, vec![l], vec![r], &on_lr(), &big()).unwrap();
        assert_eq!(got.iter().map(|b| b.num_rows).sum::<usize>(), 20_000);
        assert!(got.iter().all(|b| b.num_rows <= BATCH_ROWS));
        assert!(got.len() >= 3);
    }

    #[test]
    fn multi_batch_inputs_equal_single_batch() {
        let (l, r) = fixture();
        let l_split = vec![l[0].slice(0, 2), l[0].slice(2, 2)];
        let r_split = vec![r[0].slice(0, 1), r[0].slice(1, 3)];
        for jt in [Inner, Left, Right, Full, Semi, Anti] {
            let want = cells(&run(jt, l.clone(), r.clone(), &on_lr(), &big()).unwrap());
            let got = cells(&run(jt, l_split.clone(), r_split.clone(), &on_lr(), &big()).unwrap());
            assert_eq!(got, want, "{jt:?}");
        }
    }

    #[test]
    fn utf8_keys_with_long_strings_match() {
        let ls = Schema::new(vec![f("k", DataType::Utf8), f("lv", DataType::Int64)]);
        let rs = Schema::new(vec![f("rk", DataType::Utf8), f("rv", DataType::Int64)]);
        let mut fields = ls.fields.clone();
        fields.extend(rs.fields.clone());
        let schema = Arc::new(Schema::new(fields));
        let long = "a key that is much longer than twelve bytes";
        let l = batch(&ls, vec![Column::from_strs(&[long, "short"]), Column::from_i64(vec![1, 2])]);
        let r = batch(&rs, vec![Column::from_strs(&["short", long, "other"]), Column::from_i64(vec![9, 8, 7])]);
        let on = vec![(col(&ls, "k").unwrap(), col(&rs, "rk").unwrap())];
        let got = hash_join(vec![l], vec![r], Inner, &on, schema, &big(), &tmp()).unwrap();
        let rows: Vec<Vec<S>> = got.iter().flat_map(|b| (0..b.num_rows).map(move |i| b.row(i))).collect();
        assert_eq!(rows.len(), 2);
    }

    #[test]
    fn key_type_mismatch_is_rejected() {
        let ls = left_schema();
        let rs = Schema::new(vec![f("rk", DataType::Utf8), f("rv", DataType::Int64)]);
        let on = vec![(col(&ls, "lk").unwrap(), col(&rs, "rk").unwrap())];
        let err = hash_join(vec![], vec![], Inner, &on, out_schema(Inner), &big(), &tmp()).unwrap_err();
        assert!(matches!(err, EngineError::Type(_)));
    }

    #[test]
    fn output_schema_width_mismatch_is_plan_error() {
        let (l, r) = fixture();
        let bad = Arc::new(Schema::new(vec![f("only", DataType::Int64)]));
        let err = hash_join(l, r, Inner, &on_lr(), bad, &big(), &tmp()).unwrap_err();
        assert!(matches!(err, EngineError::Plan(_)));
    }

    #[test]
    fn budget_below_one_row_returns_memory_error_instead_of_hanging() {
        let budget = MemoryBudget::new(8);
        let (l, r) = fixture();
        let err = run(Inner, l, r, &on_lr(), &budget).unwrap_err();
        assert!(matches!(err, EngineError::Memory { .. }), "got {err:?}");
    }

    #[test]
    fn all_join_types_with_duplicates_match_fixture_counts() {
        let counts = [(Inner, 4), (Left, 6), (Right, 6), (Full, 8), (Semi, 2), (Anti, 2)];
        for (jt, n) in counts {
            assert_eq!(run_fixture(jt).len(), n, "{jt:?}");
        }
    }

    fn random_side(rng: &mut StdRng, n: usize, key_range: i64, s: &Schema) -> Batch {
        let keys: Vec<Option<i64>> = (0..n)
            .map(|_| if rng.gen_range(0..20) == 0 { None } else { Some(rng.gen_range(0..key_range)) })
            .collect();
        let vals: Vec<i64> = (0..n).map(|_| rng.gen_range(-1000..1000)).collect();
        side(&keys, &vals, s)
    }

    #[test]
    fn spilled_join_equals_in_memory_join_on_20k_random_rows() {
        let mut rng = StdRng::seed_from_u64(42);
        let left = vec![random_side(&mut rng, 20_000, 10_000, &left_schema())];
        let right = vec![random_side(&mut rng, 20_000, 10_000, &right_schema())];
        for jt in [Inner, Left, Right, Full, Semi, Anti] {
            let want = cells(&run(jt, left.clone(), right.clone(), &on_lr(), &big()).unwrap());
            let tiny = MemoryBudget::new(64 * 1024);
            let (got_batches, stats) =
                hash_join_with_stats(left.clone(), right.clone(), jt, &on_lr(), out_schema(jt), &tiny, &tmp()).unwrap();
            assert!(stats.spill_files > 0, "{jt:?} did not spill");
            assert!(stats.max_depth >= 1, "{jt:?} did not repartition");
            assert_eq!(cells(&got_batches), want, "{jt:?} spilled result differs");
        }
    }

    #[test]
    fn spill_files_are_removed_after_join() {
        let mut rng = StdRng::seed_from_u64(7);
        let left = vec![random_side(&mut rng, 3_000, 500, &left_schema())];
        let right = vec![random_side(&mut rng, 3_000, 500, &right_schema())];
        let dir = std::env::temp_dir().join(format!("ndq-join-cleanup-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let tiny = MemoryBudget::new(16 * 1024);
        let _ = hash_join(left, right, Inner, &on_lr(), out_schema(Inner), &tiny, &dir).unwrap();
        let leftover = std::fs::read_dir(&dir).unwrap().count();
        std::fs::remove_dir_all(&dir).unwrap();
        assert_eq!(leftover, 0, "spill files leaked");
    }

    #[test]
    fn skewed_all_same_key_spilled_join_terminates() {
        let left = vec![side(&keys(&[1; 30]), &(0..30).collect::<Vec<_>>(), &left_schema())];
        let right = vec![side(&keys(&[1; 2000]), &(0..2000).collect::<Vec<_>>(), &right_schema())];
        let on = on_lr();
        let (tx, rx) = mpsc::channel();
        std::thread::spawn(move || {
            let tiny = MemoryBudget::new(4096);
            let r = hash_join_with_stats(left, right, Inner, &on, out_schema(Inner), &tiny, &std::env::temp_dir());
            let _ = tx.send(r.map(|(b, s)| (b.iter().map(|x| x.num_rows).sum::<usize>(), s)));
        });
        let (rows, stats) = rx
            .recv_timeout(Duration::from_secs(20))
            .expect("skewed join did not terminate within 20 seconds")
            .unwrap();
        assert_eq!(rows, 60_000);
        assert!(stats.spill_files > 0);
        assert!(stats.max_depth <= MAX_DEPTH);
    }

    #[test]
    fn skewed_join_with_left_and_full_types_terminates_with_correct_counts() {
        let left = vec![side(&keys(&[1; 10]), &(0..10).collect::<Vec<_>>(), &left_schema())];
        let right = vec![side(&keys(&[2; 500]), &(0..500).collect::<Vec<_>>(), &right_schema())];
        let on = on_lr();
        let (tx, rx) = mpsc::channel();
        std::thread::spawn(move || {
            let tiny = MemoryBudget::new(4096);
            let r = hash_join(left, right, Full, &on, out_schema(Full), &tiny, &std::env::temp_dir());
            let _ = tx.send(r.map(|b| cells(&b).len()));
        });
        let n = rx
            .recv_timeout(Duration::from_secs(20))
            .expect("skewed full join did not terminate within 20 seconds")
            .unwrap();
        assert_eq!(n, 10 + 500);
    }

    /// Right side with a Utf8 payload column: `rk` Int64, `rp` Utf8.
    fn str_side_schema() -> Schema {
        Schema::new(vec![f("rk", DataType::Int64), f("rp", DataType::Utf8)])
    }

    /// Output of a join between (lk, lv) and (rk, rp).
    fn str_out_schema() -> SchemaRef {
        Arc::new(Schema::new(vec![
            f("lk", DataType::Int64),
            f("lv", DataType::Int64),
            f("rk", DataType::Int64),
            f("rp", DataType::Utf8),
        ]))
    }

    fn on_lk_rk() -> Vec<(Expr, Expr)> {
        vec![(col(&left_schema(), "lk").unwrap(), col(&str_side_schema(), "rk").unwrap())]
    }

    /// Sorted rendering of every row, so results compare regardless of batch layout.
    fn sorted_debug(batches: &[Batch]) -> Vec<String> {
        let mut rows: Vec<String> = batches
            .iter()
            .flat_map(|b| (0..b.num_rows).map(move |i| format!("{:?}", b.row(i))))
            .collect();
        rows.sort();
        rows
    }

    /// Left: keys [7, 8]. Right: `right_rows` rows all keyed 7, payload `payload(i)`, split into batches of `split` rows.
    fn skew_inputs(right_rows: usize, split: usize, payload: &dyn Fn(usize) -> String) -> (Vec<Batch>, Vec<Batch>) {
        let left = vec![side(&[Some(7), Some(8)], &[0, 1], &left_schema())];
        let rs = str_side_schema();
        let mut right = Vec::new();
        let mut start = 0;
        while start < right_rows {
            let end = (start + split).min(right_rows);
            let strs: Vec<String> = (start..end).map(payload).collect();
            right.push(batch(&rs, vec![Column::from_i64(vec![7; end - start]), Column::from_strs(&strs)]));
            start = end;
        }
        (left, right)
    }

    /// Runs `body` on a worker thread and fails the test if it does not finish within 30 seconds.
    /// Panics inside `body` propagate to the test.
    fn with_watchdog(name: &str, body: impl FnOnce() + Send + 'static) {
        let (tx, rx) = mpsc::channel::<()>();
        let worker = std::thread::spawn(move || {
            body();
            let _ = tx.send(());
        });
        match rx.recv_timeout(Duration::from_secs(30)) {
            Err(mpsc::RecvTimeoutError::Timeout) => panic!("{name} did not finish within 30 seconds"),
            _ => {
                if let Err(p) = worker.join() {
                    std::panic::resume_unwind(p);
                }
            }
        }
    }

    #[test]
    fn wide_utf8_build_side_spills_and_matches_in_memory() {
        let n = 16_000usize;
        let keys_v: Vec<i64> = (0..n as i64).collect();
        let ls = left_schema();
        let mut rfields = vec![f("rk", DataType::Int64)];
        rfields.extend((0..10).map(|i| f(&format!("r{i}"), DataType::Utf8)));
        let rs = Schema::new(rfields);
        let mut rcols = vec![Column::from_i64(keys_v.clone())];
        rcols.extend((0..10).map(|_| Column::from_strs(&vec!["x"; n])));
        let left = vec![side(&keys(&keys_v), &keys_v, &ls)];
        let right = vec![batch(&rs, rcols)];
        let mut ofields = ls.fields.clone();
        ofields.extend(rs.fields.clone());
        let out = Arc::new(Schema::new(ofields));
        let on = vec![(col(&ls, "lk").unwrap(), col(&rs, "rk").unwrap())];
        let want = sorted_debug(&hash_join(left.clone(), right.clone(), Inner, &on, out.clone(), &big(), &tmp()).unwrap());
        assert_eq!(want.len(), n);
        for budget in [150_000, 200_000, 250_000] {
            let tiny = MemoryBudget::new(budget);
            let got = hash_join(left.clone(), right.clone(), Inner, &on, out.clone(), &tiny, &tmp())
                .unwrap_or_else(|e| panic!("budget {budget}: {e:?}"));
            assert_eq!(sorted_debug(&got), want, "budget {budget}");
        }
    }

    #[test]
    fn skewed_utf8_single_batch_spills_and_matches_in_memory() {
        with_watchdog("skewed utf8 join", || {
            let (left, right) = skew_inputs(10_000, 10_000, &|i| format!("{:0>100}", i));
            for jt in [Inner, Left] {
                let on = on_lk_rk();
                let want = sorted_debug(&hash_join(left.clone(), right.clone(), jt, &on, str_out_schema(), &big(), &tmp()).unwrap());
                assert_eq!(want.len(), if jt == Inner { 10_000 } else { 10_001 }, "{jt:?} in-memory count");
                for budget in [800_000, 200_000] {
                    let tiny = MemoryBudget::new(budget);
                    let got = hash_join(left.clone(), right.clone(), jt, &on, str_out_schema(), &tiny, &tmp())
                        .unwrap_or_else(|e| panic!("{jt:?} budget {budget}: {e:?}"));
                    assert_eq!(sorted_debug(&got), want, "{jt:?} budget {budget}");
                }
            }
        });
    }

    #[test]
    fn skewed_short_utf8_pre_split_batches_spill_and_match_in_memory() {
        with_watchdog("skewed short utf8 join", || {
            let (left, right) = skew_inputs(10_000, 100, &|_| "x".to_string());
            for jt in [Inner, Left] {
                let on = on_lk_rk();
                let want = sorted_debug(&hash_join(left.clone(), right.clone(), jt, &on, str_out_schema(), &big(), &tmp()).unwrap());
                for budget in [50_000, 200_000, 800_000] {
                    let tiny = MemoryBudget::new(budget);
                    let got = hash_join(left.clone(), right.clone(), jt, &on, str_out_schema(), &tiny, &tmp())
                        .unwrap_or_else(|e| panic!("{jt:?} budget {budget}: {e:?}"));
                    assert_eq!(sorted_debug(&got), want, "{jt:?} budget {budget}");
                }
            }
        });
    }

    #[test]
    fn small_utf8_inputs_complete_at_tiny_budgets() {
        let mut rng = StdRng::seed_from_u64(2024);
        let on = on_lk_rk();
        for case in 0..30 {
            let nl = rng.gen_range(0..=400usize);
            let nr = rng.gen_range(1..=400usize);
            let lkeys: Vec<i64> = (0..nl).map(|_| rng.gen_range(0..20)).collect();
            let lvals: Vec<i64> = (0..nl).map(|_| rng.gen_range(-1000..1000)).collect();
            let rkeys: Vec<i64> = (0..nr).map(|_| rng.gen_range(0..20)).collect();
            let rstrs: Vec<String> = (0..nr)
                .map(|_| {
                    let len = rng.gen_range(0..=60usize);
                    (0..len).map(|_| char::from(rng.gen_range(b'a'..=b'z'))).collect()
                })
                .collect();
            let left = vec![side(&keys(&lkeys), &lvals, &left_schema())];
            let right = vec![batch(&str_side_schema(), vec![Column::from_i64(rkeys), Column::from_strs(&rstrs)])];
            let want = sorted_debug(&hash_join(left.clone(), right.clone(), Inner, &on, str_out_schema(), &big(), &tmp()).unwrap());
            let big_row = rstrs
                .iter()
                .map(|s| 8 + 16 + if s.len() > 12 { s.len() } else { 0 } + ENTRY_OVERHEAD + KEY_BYTES)
                .max()
                .unwrap_or(0);
            let ok = hash_join(left.clone(), right.clone(), Inner, &on, str_out_schema(), &MemoryBudget::new(20_000), &tmp())
                .unwrap_or_else(|e| panic!("case {case} budget 20000: {e:?}"));
            assert_eq!(sorted_debug(&ok), want, "case {case} budget 20000");
            match hash_join(left.clone(), right.clone(), Inner, &on, str_out_schema(), &MemoryBudget::new(6_000), &tmp()) {
                Ok(got) => assert_eq!(sorted_debug(&got), want, "case {case} budget 6000"),
                Err(EngineError::Memory { .. }) => {
                    assert!(6_000 < big_row, "case {case}: Memory at 6000 though a build row costs {big_row}")
                }
                Err(e) => panic!("case {case} budget 6000: {e:?}"),
            }
        }
    }
}
