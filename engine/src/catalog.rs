//! SLICE G. In-memory tables + CSV loader.
use crate::column::{schema_ref, Batch, Column};
use crate::types::{DataType, EngineError, Field, Result, Schema, SchemaRef};
use std::collections::{HashMap, HashSet};
use std::path::Path;

#[derive(Default)]
pub struct Catalog { pub tables: HashMap<String, (SchemaRef, Vec<Batch>)> }

impl Catalog {
    pub fn new() -> Self { Self::default() }
    pub fn register(&mut self, name: &str, schema: SchemaRef, batches: Vec<Batch>) { self.tables.insert(name.to_string(), (schema, batches)); }
    pub fn get(&self, name: &str) -> Result<&(SchemaRef, Vec<Batch>)> { self.tables.get(name).ok_or_else(|| EngineError::Plan(format!("table '{name}' not found"))) }
}

/// CSV with header row; column types inferred (Int64 > Float64 > Boolean > Utf8) over ALL rows; empty field => NULL.
/// Supports RFC 4180 quoted fields (embedded commas, newlines, `""` escapes). LF and CRLF line endings accepted.
/// Blank lines are skipped. A row whose field count differs from the header is an `EngineError::Io` naming the line.
/// A column is nullable iff it contains at least one empty field. Batches of `batch_rows` rows (zero data rows => no batches).
pub fn load_csv(path: &Path, batch_rows: usize) -> Result<(SchemaRef, Vec<Batch>)> {
    let text = std::fs::read_to_string(path).map_err(|e| EngineError::Io(format!("{}: {e}", path.display())))?;
    parse_table(&text, batch_rows).map_err(|e| match e {
        EngineError::Io(m) => EngineError::Io(format!("{}: {m}", path.display())),
        other => other,
    })
}

struct Record { line: usize, fields: Vec<String> }

fn parse_table(text: &str, batch_rows: usize) -> Result<(SchemaRef, Vec<Batch>)> {
    if batch_rows == 0 {
        return Err(EngineError::Plan("load_csv batch_rows must be > 0".into()));
    }
    let mut records = parse_csv(text)?.into_iter();
    let header = records.next().ok_or_else(|| EngineError::Io("missing header row".into()))?;
    let ncols = header.fields.len();
    let rows: Vec<Record> = records.collect();
    for r in &rows {
        if r.fields.len() != ncols {
            return Err(EngineError::Io(format!("line {}: expected {ncols} fields, found {}", r.line, r.fields.len())));
        }
    }
    let mut seen = HashSet::new();
    for name in &header.fields {
        if !seen.insert(name.as_str()) {
            return Err(EngineError::Io(format!("line 1: duplicate column name '{name}'")));
        }
    }

    let fields: Vec<Field> = header
        .fields
        .iter()
        .enumerate()
        .map(|(j, name)| {
            let (mut all_int, mut all_float, mut all_bool, mut any_value, mut any_null) = (true, true, true, false, false);
            for r in &rows {
                let v = r.fields[j].as_str();
                if v.is_empty() {
                    any_null = true;
                    continue;
                }
                any_value = true;
                all_int = all_int && v.parse::<i64>().is_ok();
                all_float = all_float && is_float_literal(v);
                all_bool = all_bool && (v.eq_ignore_ascii_case("true") || v.eq_ignore_ascii_case("false"));
            }
            let dt = if !any_value {
                DataType::Utf8
            } else if all_int {
                DataType::Int64
            } else if all_float {
                DataType::Float64
            } else if all_bool {
                DataType::Boolean
            } else {
                DataType::Utf8
            };
            Field::new(name, dt, any_null)
        })
        .collect();

    let schema = schema_ref(Schema::new(fields));
    let mut batches = Vec::with_capacity(rows.len().div_ceil(batch_rows));
    for chunk in rows.chunks(batch_rows) {
        let columns: Vec<Column> = schema
            .fields
            .iter()
            .enumerate()
            .map(|(j, f)| {
                let cells: Vec<&str> = chunk.iter().map(|r| r.fields[j].as_str()).collect();
                build_column(f.data_type, &cells)
            })
            .collect();
        batches.push(Batch::try_new(schema.clone(), columns)?);
    }
    Ok((schema, batches))
}

/// Cells are raw field text; an empty cell is NULL.
fn build_column(dt: DataType, cells: &[&str]) -> Column {
    let col = match dt {
        DataType::Int64 => Column::from_i64(cells.iter().map(|c| c.parse::<i64>().unwrap_or(0)).collect()),
        DataType::Float64 => Column::from_f64(cells.iter().map(|c| c.parse::<f64>().unwrap_or(0.0)).collect()),
        DataType::Boolean => Column::from_bool(cells.iter().map(|c| c.eq_ignore_ascii_case("true")).collect()),
        DataType::Utf8 => Column::from_strs(cells),
    };
    let validity: Vec<bool> = cells.iter().map(|c| !c.is_empty()).collect();
    if validity.iter().all(|&v| v) {
        col
    } else {
        col.with_validity(validity)
    }
}

/// Shape check (optional sign, digits with optional fraction, optional exponent) that also parses as f64.
/// Rejects `NaN`, `inf` and similar words, which must stay text.
fn is_float_literal(s: &str) -> bool {
    let b = s.as_bytes();
    let mut i = 0;
    if matches!(b.first().copied(), Some(b'+' | b'-')) {
        i += 1;
    }
    let mut digits = 0;
    while i < b.len() && b[i].is_ascii_digit() {
        i += 1;
        digits += 1;
    }
    if i < b.len() && b[i] == b'.' {
        i += 1;
        while i < b.len() && b[i].is_ascii_digit() {
            i += 1;
            digits += 1;
        }
    }
    if digits == 0 {
        return false;
    }
    if i < b.len() && (b[i] == b'e' || b[i] == b'E') {
        i += 1;
        if i < b.len() && (b[i] == b'+' || b[i] == b'-') {
            i += 1;
        }
        let exp_start = i;
        while i < b.len() && b[i].is_ascii_digit() {
            i += 1;
        }
        if i == exp_start {
            return false;
        }
    }
    i == b.len() && s.parse::<f64>().is_ok()
}

fn at_delimiter(b: &[u8], i: usize) -> bool {
    match b.get(i).copied() {
        Some(b',' | b'\n') => true,
        Some(b'\r') => b.get(i + 1).copied() == Some(b'\n'),
        _ => false,
    }
}

fn csv_err(line: usize, msg: &str) -> EngineError {
    EngineError::Io(format!("line {line}: {msg}"))
}

/// RFC 4180 record splitter. Works on bytes: every delimiter is ASCII, so UTF-8 sequences stay intact.
/// Blank unquoted lines produce no record.
fn parse_csv(text: &str) -> Result<Vec<Record>> {
    let b = text.as_bytes();
    let mut records = Vec::new();
    let mut i = 0;
    let mut line = 1;
    while i < b.len() {
        let start_line = line;
        let mut fields: Vec<String> = Vec::new();
        let mut any_quoted = false;
        loop {
            let mut buf: Vec<u8> = Vec::new();
            if b.get(i).copied() == Some(b'"') {
                any_quoted = true;
                i += 1;
                loop {
                    match b.get(i).copied() {
                        None => return Err(csv_err(start_line, "unterminated quoted field")),
                        Some(b'"') => {
                            if b.get(i + 1).copied() == Some(b'"') {
                                buf.push(b'"');
                                i += 2;
                            } else {
                                i += 1;
                                break;
                            }
                        }
                        Some(c) => {
                            if c == b'\n' {
                                line += 1;
                            }
                            buf.push(c);
                            i += 1;
                        }
                    }
                }
                if i < b.len() && !at_delimiter(b, i) {
                    return Err(csv_err(line, "unexpected character after closing quote"));
                }
            } else {
                while i < b.len() && !at_delimiter(b, i) {
                    buf.push(b[i]);
                    i += 1;
                }
            }
            fields.push(String::from_utf8(buf).map_err(|_| csv_err(line, "invalid UTF-8"))?);
            if i >= b.len() {
                break;
            }
            if b[i] == b',' {
                i += 1;
                continue;
            }
            i += if b[i] == b'\r' { 2 } else { 1 };
            line += 1;
            break;
        }
        let blank = fields.len() == 1 && fields[0].is_empty() && !any_quoted;
        if !blank {
            records.push(Record { line: start_line, fields });
        }
    }
    Ok(records)
}
