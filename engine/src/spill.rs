//! SLICE D1. Deterministic on-disk batch spill. Binary format is private; roundtrip must be bit-exact
//! (incl. nulls, long strings, f64 NaN payloads). Files are deleted when SpillFile drops.
//!
//! Format (little endian): MAGIC, then per batch: u64 num_rows, then per column:
//! u8 has_validity, [num_rows x u8 validity], values. Bool: num_rows x u8.
//! Int64: num_rows x i64. Float64: num_rows x u64 bit pattern. Utf8: num_rows x (u32 len, bytes).
use crate::column::{Batch, Column, StringViews, Values};
use crate::types::{DataType, EngineError, Result, SchemaRef};
use std::fs::{self, File, OpenOptions};
use std::io::{BufReader, BufWriter, Read, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

const MAGIC: &[u8; 8] = b"NDQSPIL1";
static SPILL_SEQ: AtomicU64 = AtomicU64::new(0);

fn io_err(e: std::io::Error) -> EngineError {
    EngineError::Io(e.to_string())
}

fn corrupt(what: &str) -> EngineError {
    EngineError::Io(format!("corrupt spill file: {what}"))
}

pub struct SpillWriter {
    out: BufWriter<File>,
    path: PathBuf,
    schema: SchemaRef,
    num_rows: usize,
    bytes_written: u64,
    finished: bool,
}

/// A finished spill file. Deleted from disk when dropped.
#[derive(Debug)]
pub struct SpillFile {
    path: PathBuf,
    schema: SchemaRef,
    num_rows: usize,
    bytes_on_disk: u64,
}

/// Streams batches back from a spill file in write order.
pub struct SpillReader {
    input: BufReader<File>,
    schema: SchemaRef,
}

impl SpillWriter {
    /// Creates a uniquely named file inside `dir` (dir must exist).
    pub fn create(dir: &Path, schema: SchemaRef) -> Result<SpillWriter> {
        loop {
            let seq = SPILL_SEQ.fetch_add(1, Ordering::Relaxed);
            let path = dir.join(format!("ndq-spill-{}-{seq}.bin", std::process::id()));
            match OpenOptions::new().write(true).create_new(true).open(&path) {
                Ok(file) => {
                    let mut out = BufWriter::new(file);
                    if let Err(e) = out.write_all(MAGIC) {
                        let _ = fs::remove_file(&path);
                        return Err(io_err(e));
                    }
                    return Ok(SpillWriter {
                        out,
                        path,
                        schema,
                        num_rows: 0,
                        bytes_written: MAGIC.len() as u64,
                        finished: false,
                    });
                }
                Err(e) if e.kind() == std::io::ErrorKind::AlreadyExists => continue,
                Err(e) => return Err(io_err(e)),
            }
        }
    }

    pub fn write(&mut self, batch: &Batch) -> Result<()> {
        validate(&self.schema, batch)?;
        let mut buf = Vec::new();
        buf.extend_from_slice(&(batch.num_rows as u64).to_le_bytes());
        for col in &batch.columns {
            encode_column(col, batch.num_rows, &mut buf)?;
        }
        self.out.write_all(&buf).map_err(io_err)?;
        self.num_rows += batch.num_rows;
        self.bytes_written += buf.len() as u64;
        Ok(())
    }

    /// Flushes and hands ownership of the file to a `SpillFile`. On error the partial file is removed.
    pub fn finish(mut self) -> Result<SpillFile> {
        self.out.flush().map_err(io_err)?;
        self.finished = true;
        Ok(SpillFile {
            path: self.path.clone(),
            schema: self.schema.clone(),
            num_rows: self.num_rows,
            bytes_on_disk: self.bytes_written,
        })
    }
}

impl Drop for SpillWriter {
    fn drop(&mut self) {
        if !self.finished {
            let _ = fs::remove_file(&self.path);
        }
    }
}

impl SpillFile {
    pub fn path(&self) -> PathBuf {
        self.path.clone()
    }
    pub fn num_rows(&self) -> usize {
        self.num_rows
    }
    pub fn bytes_on_disk(&self) -> u64 {
        self.bytes_on_disk
    }

    /// Opens a streaming reader positioned at the first batch.
    pub fn reader(&self) -> Result<SpillReader> {
        let mut input = BufReader::new(File::open(&self.path).map_err(io_err)?);
        let mut magic = [0u8; 8];
        read_exact_or_corrupt(&mut input, &mut magic)?;
        if &magic != MAGIC {
            return Err(corrupt("bad magic"));
        }
        Ok(SpillReader {
            input,
            schema: self.schema.clone(),
        })
    }

    /// Read all batches back in write order.
    pub fn read(&self) -> Result<Vec<Batch>> {
        let mut reader = self.reader()?;
        let mut out = Vec::new();
        while let Some(batch) = reader.next_batch()? {
            out.push(batch);
        }
        Ok(out)
    }
}

impl Drop for SpillFile {
    fn drop(&mut self) {
        let _ = fs::remove_file(&self.path);
    }
}

impl SpillReader {
    /// Next batch, or `None` at a clean end of file.
    pub fn next_batch(&mut self) -> Result<Option<Batch>> {
        let mut header = [0u8; 8];
        match read_fully(&mut self.input, &mut header)? {
            0 => return Ok(None),
            8 => {}
            _ => return Err(corrupt("truncated batch header")),
        }
        let n = usize::try_from(u64::from_le_bytes(header)).map_err(|_| corrupt("row count"))?;
        let schema = self.schema.clone();
        let mut columns = Vec::with_capacity(schema.fields.len());
        for field in &schema.fields {
            columns.push(decode_column(&mut self.input, field.data_type, n)?);
        }
        Batch::try_new(schema, columns).map(Some)
    }
}

fn validate(schema: &SchemaRef, batch: &Batch) -> Result<()> {
    if batch.columns.len() != schema.fields.len() {
        return Err(EngineError::Type(
            "spill batch column count mismatch".into(),
        ));
    }
    for (field, col) in schema.fields.iter().zip(&batch.columns) {
        if col.data_type() != field.data_type || col.len() != batch.num_rows {
            return Err(EngineError::Type(format!(
                "spill column '{}' type/length mismatch",
                field.name
            )));
        }
    }
    Ok(())
}

fn encode_column(col: &Column, n: usize, buf: &mut Vec<u8>) -> Result<()> {
    match &col.validity {
        None => buf.push(0),
        Some(valid) => {
            buf.push(1);
            buf.extend(valid.iter().map(|&b| u8::from(b)));
        }
    }
    match &col.values {
        Values::Bool(v) => buf.extend(v.iter().map(|&b| u8::from(b))),
        Values::Int64(v) => {
            for x in v {
                buf.extend_from_slice(&x.to_le_bytes());
            }
        }
        Values::Float64(v) => {
            for x in v {
                buf.extend_from_slice(&x.to_bits().to_le_bytes());
            }
        }
        Values::Utf8(sv) => {
            for i in 0..n {
                let s = sv.get(i).as_bytes();
                let len = u32::try_from(s.len())
                    .map_err(|_| EngineError::Io("string exceeds u32 length".into()))?;
                buf.extend_from_slice(&len.to_le_bytes());
                buf.extend_from_slice(s);
            }
        }
    }
    Ok(())
}

/// Returns the number of bytes read; short only at end of input.
fn read_fully(r: &mut impl Read, buf: &mut [u8]) -> Result<usize> {
    let mut filled = 0;
    while filled < buf.len() {
        match r.read(&mut buf[filled..]) {
            Ok(0) => break,
            Ok(k) => filled += k,
            Err(e) if e.kind() == std::io::ErrorKind::Interrupted => continue,
            Err(e) => return Err(io_err(e)),
        }
    }
    Ok(filled)
}

fn read_exact_or_corrupt(r: &mut impl Read, buf: &mut [u8]) -> Result<()> {
    if read_fully(r, buf)? != buf.len() {
        return Err(corrupt("unexpected end of file"));
    }
    Ok(())
}

fn read_vec(r: &mut impl Read, len: usize) -> Result<Vec<u8>> {
    let mut buf = vec![0u8; len];
    read_exact_or_corrupt(r, &mut buf)?;
    Ok(buf)
}

fn read_bools(r: &mut impl Read, n: usize) -> Result<Vec<bool>> {
    read_vec(r, n)?
        .into_iter()
        .map(|b| match b {
            0 => Ok(false),
            1 => Ok(true),
            _ => Err(corrupt("bool byte out of range")),
        })
        .collect()
}

fn decode_column(r: &mut impl Read, dt: DataType, n: usize) -> Result<Column> {
    let mut flag = [0u8; 1];
    read_exact_or_corrupt(r, &mut flag)?;
    let validity = match flag[0] {
        0 => None,
        1 => Some(read_bools(r, n)?),
        _ => return Err(corrupt("bad validity flag")),
    };
    let values = match dt {
        DataType::Boolean => Values::Bool(read_bools(r, n)?),
        DataType::Int64 => {
            let raw = read_vec(r, n.checked_mul(8).ok_or_else(|| corrupt("row count"))?)?;
            Values::Int64(
                (0..n)
                    .map(|i| i64::from_le_bytes(le8(&raw[i * 8..(i + 1) * 8])))
                    .collect(),
            )
        }
        DataType::Float64 => {
            let raw = read_vec(r, n.checked_mul(8).ok_or_else(|| corrupt("row count"))?)?;
            Values::Float64(
                (0..n)
                    .map(|i| f64::from_bits(u64::from_le_bytes(le8(&raw[i * 8..(i + 1) * 8]))))
                    .collect(),
            )
        }
        DataType::Utf8 => {
            let mut strings = Vec::with_capacity(n);
            for _ in 0..n {
                let mut len = [0u8; 4];
                read_exact_or_corrupt(r, &mut len)?;
                let len = u32::from_le_bytes(len) as usize;
                let bytes = read_vec(r, len)?;
                strings.push(String::from_utf8(bytes).map_err(|_| corrupt("invalid utf-8"))?);
            }
            Values::Utf8(StringViews::from_strs(&strings))
        }
    };
    Ok(Column { values, validity })
}

fn le8(c: &[u8]) -> [u8; 8] {
    let mut out = [0u8; 8];
    out.copy_from_slice(c);
    out
}

/// Test helpers shared with the sort tests: a bit-level, order-preserving row rendering.
#[cfg(test)]
pub(crate) mod test_support {
    use crate::column::{Batch, Column, Values};

    fn cell(c: &Column, i: usize) -> String {
        let value = match &c.values {
            Values::Bool(v) => format!("b{}", v[i]),
            Values::Int64(v) => format!("i{}", v[i]),
            Values::Float64(v) => format!("f{:016x}", v[i].to_bits()),
            Values::Utf8(s) => format!("s{:?}", s.get(i)),
        };
        format!("{}{value}", if c.is_valid(i) { "+" } else { "-" })
    }

    /// One string per row, covering validity, raw slot values and f64 bit patterns.
    pub(crate) fn canonical_rows(batches: &[Batch]) -> Vec<String> {
        batches
            .iter()
            .flat_map(|b| {
                (0..b.num_rows).map(move |i| {
                    b.columns
                        .iter()
                        .map(|c| cell(c, i))
                        .collect::<Vec<_>>()
                        .join("|")
                })
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::test_support::canonical_rows;
    use super::*;
    use crate::types::{Field, Schema};
    use rand::{rngs::StdRng, Rng, SeedableRng};
    use std::sync::Arc;

    fn test_dir(tag: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!(
            "ndq-spill-test-{tag}-{}-{}",
            std::process::id(),
            SPILL_SEQ.fetch_add(1, Ordering::Relaxed)
        ));
        fs::create_dir_all(&dir).unwrap();
        dir
    }

    fn all_types_schema() -> SchemaRef {
        Arc::new(Schema::new(vec![
            Field::new("b", DataType::Boolean, true),
            Field::new("i", DataType::Int64, true),
            Field::new("f", DataType::Float64, true),
            Field::new("s", DataType::Utf8, true),
        ]))
    }

    fn opt_validity(rng: &mut StdRng, n: usize) -> Option<Vec<bool>> {
        // Mix: no validity vector at all, all-valid vector, and random nulls.
        match rng.gen_range(0..3) {
            0 => None,
            1 => Some(vec![true; n]),
            _ => Some((0..n).map(|_| rng.gen_bool(0.7)).collect()),
        }
    }

    fn random_string(rng: &mut StdRng) -> String {
        const POOL: &[&str] = &[
            "",
            "a",
            "exactly12byt",
            "exactly123ch",
            "thirteen-byte",
            "a string well over twelve bytes long",
            "ümlaut-ünicode-long-string-é",
            "日本語のテキスト長い文字列",
            "zz",
        ];
        if rng.gen_bool(0.5) {
            POOL[rng.gen_range(0..POOL.len())].to_string()
        } else {
            (0..rng.gen_range(0..40))
                .map(|_| char::from(rng.gen_range(b'a'..=b'z')))
                .collect()
        }
    }

    fn random_batch(rng: &mut StdRng, schema: &SchemaRef, n: usize) -> Batch {
        let bools: Vec<bool> = (0..n).map(|_| rng.gen()).collect();
        let ints: Vec<i64> = (0..n).map(|_| rng.gen()).collect();
        let floats: Vec<f64> = (0..n)
            .map(|_| match rng.gen_range(0..6) {
                0 => f64::NAN,
                1 => -0.0,
                2 => f64::from_bits(0x7ff8_dead_beef_0001),
                3 => f64::INFINITY,
                _ => f64::from_bits(rng.gen()),
            })
            .collect();
        let strings: Vec<String> = (0..n).map(|_| random_string(rng)).collect();
        let cols = vec![
            Column {
                values: Values::Bool(bools),
                validity: opt_validity(rng, n),
            },
            Column {
                values: Values::Int64(ints),
                validity: opt_validity(rng, n),
            },
            Column {
                values: Values::Float64(floats),
                validity: opt_validity(rng, n),
            },
            Column {
                values: Values::Utf8(StringViews::from_strs(&strings)),
                validity: opt_validity(rng, n),
            },
        ];
        Batch::try_new(schema.clone(), cols).unwrap()
    }

    fn roundtrip(dir: &Path, schema: &SchemaRef, batches: &[Batch]) -> (SpillFile, Vec<Batch>) {
        let mut w = SpillWriter::create(dir, schema.clone()).unwrap();
        for b in batches {
            w.write(b).unwrap();
        }
        let file = w.finish().unwrap();
        let back = file.read().unwrap();
        (file, back)
    }

    #[test]
    fn roundtrip_random_all_types_with_nulls_nan_and_long_strings_is_bit_exact() {
        let dir = test_dir("rt");
        let schema = all_types_schema();
        let mut rng = StdRng::seed_from_u64(7);
        let batches: Vec<Batch> = [1usize, 17, 500, 0, 3]
            .iter()
            .map(|&n| random_batch(&mut rng, &schema, n))
            .collect();
        let (file, back) = roundtrip(&dir, &schema, &batches);
        assert_eq!(back.len(), batches.len());
        assert_eq!(canonical_rows(&back), canonical_rows(&batches));
        assert_eq!(file.num_rows(), 521);
        assert!(file.bytes_on_disk() > 0);
        drop(file);
        fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn validity_presence_is_preserved_per_column() {
        let dir = test_dir("valid");
        let schema = all_types_schema();
        let mut rng = StdRng::seed_from_u64(11);
        let b = random_batch(&mut rng, &schema, 64);
        let (_file, back) = roundtrip(&dir, &schema, std::slice::from_ref(&b));
        for (orig, got) in b.columns.iter().zip(&back[0].columns) {
            assert_eq!(orig.validity.is_some(), got.validity.is_some());
            assert_eq!(orig.validity, got.validity);
        }
        fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn nan_payloads_negative_zero_and_subnormals_roundtrip_exactly() {
        let dir = test_dir("nan");
        let schema = Arc::new(Schema::new(vec![Field::new("f", DataType::Float64, false)]));
        let payloads = [
            f64::from_bits(0x7ff8_0000_0000_1234),
            f64::from_bits(0xfff0_0000_0000_0001),
            -0.0,
            0.0,
            f64::MIN_POSITIVE / 2.0,
            f64::NEG_INFINITY,
        ];
        let b = Batch::try_new(schema.clone(), vec![Column::from_f64(payloads.to_vec())]).unwrap();
        let (_f, back) = roundtrip(&dir, &schema, &[b]);
        let Values::Float64(got) = &back[0].columns[0].values else {
            panic!("expected f64")
        };
        for (x, y) in payloads.iter().zip(got) {
            assert_eq!(x.to_bits(), y.to_bits());
        }
        fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn long_and_boundary_strings_roundtrip() {
        let dir = test_dir("str");
        let schema = Arc::new(Schema::new(vec![Field::new("s", DataType::Utf8, false)]));
        let items = [
            "",
            "12345678901",
            "123456789012",
            "1234567890123",
            &"x".repeat(100_000),
            "é日本",
        ];
        let b = Batch::try_new(schema.clone(), vec![Column::from_strs(&items)]).unwrap();
        let (_f, back) = roundtrip(&dir, &schema, &[b]);
        for (i, s) in items.iter().enumerate() {
            assert_eq!(
                back[0].columns[0].get(i),
                crate::types::ScalarValue::Utf8((*s).to_string())
            );
        }
        fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn streaming_reader_returns_batches_in_order_then_none() {
        let dir = test_dir("stream");
        let schema = Arc::new(Schema::new(vec![Field::new("i", DataType::Int64, false)]));
        let mut w = SpillWriter::create(&dir, schema.clone()).unwrap();
        for start in [0i64, 10, 20] {
            let b = Batch::try_new(
                schema.clone(),
                vec![Column::from_i64((start..start + 10).collect())],
            )
            .unwrap();
            w.write(&b).unwrap();
        }
        let file = w.finish().unwrap();
        let mut r = file.reader().unwrap();
        let mut firsts = Vec::new();
        while let Some(b) = r.next_batch().unwrap() {
            firsts.push(b.columns[0].get(0));
        }
        assert_eq!(firsts.len(), 3);
        assert_eq!(firsts[2], crate::types::ScalarValue::Int64(20));
        assert!(r.next_batch().unwrap().is_none());
        drop(r);
        drop(file);
        fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn spill_file_is_deleted_on_drop() {
        let dir = test_dir("drop");
        let schema = all_types_schema();
        let mut rng = StdRng::seed_from_u64(1);
        let mut w = SpillWriter::create(&dir, schema.clone()).unwrap();
        w.write(&random_batch(&mut rng, &schema, 4)).unwrap();
        let file = w.finish().unwrap();
        let path = file.path();
        assert!(path.exists());
        drop(file);
        assert!(!path.exists());
        fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn unfinished_writer_removes_its_partial_file() {
        let dir = test_dir("abort");
        let schema = all_types_schema();
        let mut rng = StdRng::seed_from_u64(2);
        let mut w = SpillWriter::create(&dir, schema.clone()).unwrap();
        w.write(&random_batch(&mut rng, &schema, 8)).unwrap();
        let entries_before = fs::read_dir(&dir).unwrap().count();
        assert_eq!(entries_before, 1);
        drop(w);
        assert_eq!(fs::read_dir(&dir).unwrap().count(), 0);
        fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn filenames_are_unique_within_a_directory() {
        let dir = test_dir("names");
        let schema = all_types_schema();
        let a = SpillWriter::create(&dir, schema.clone())
            .unwrap()
            .finish()
            .unwrap();
        let b = SpillWriter::create(&dir, schema.clone())
            .unwrap()
            .finish()
            .unwrap();
        assert_ne!(a.path(), b.path());
        assert_eq!(a.num_rows(), 0);
        assert!(a.read().unwrap().is_empty());
        drop((a, b));
        fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn empty_spill_roundtrips_to_no_batches() {
        let dir = test_dir("empty");
        let schema = all_types_schema();
        let (_f, back) = roundtrip(&dir, &schema, &[]);
        assert!(back.is_empty());
        fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn write_rejects_batch_with_wrong_column_type() {
        let dir = test_dir("type");
        let schema = Arc::new(Schema::new(vec![Field::new("i", DataType::Int64, false)]));
        let other = Arc::new(Schema::new(vec![Field::new("i", DataType::Float64, false)]));
        let wrong = Batch::try_new(other, vec![Column::from_f64(vec![1.0])]).unwrap();
        let mut w = SpillWriter::create(&dir, schema).unwrap();
        assert!(matches!(w.write(&wrong), Err(EngineError::Type(_))));
        drop(w);
        fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn corrupt_magic_is_reported_not_read() {
        let dir = test_dir("corrupt");
        let schema = all_types_schema();
        let file = SpillWriter::create(&dir, schema).unwrap().finish().unwrap();
        fs::write(file.path(), b"NOTSPILLxxxx").unwrap();
        assert!(matches!(file.read(), Err(EngineError::Io(_))));
        drop(file);
        fs::remove_dir_all(&dir).unwrap();
    }
}
