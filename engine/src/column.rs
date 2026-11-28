//! Columnar data: typed vectors + validity + German-style 16-byte string views.
use crate::types::{DataType, EngineError, Result, ScalarValue, Schema, SchemaRef};
use std::sync::Arc;

/// 16-byte string descriptor (spec §5.2). len<=12: bytes inline in `data`.
/// len>12: data[0..4]=prefix, data[4..8]=buffer index, data[8..12]=offset (LE).
#[repr(C, align(16))]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct View {
    pub(crate) len: u32,
    pub(crate) data: [u8; 12],
}

/// Fields are crate-private so safe code outside the crate cannot forge views into arbitrary buffers.
/// ```compile_fail
/// let _ = native_engine::column::StringViews { views: vec![], buffers: vec![] };
/// ```
#[derive(Clone, Debug, Default)]
pub struct StringViews {
    pub(crate) views: Vec<View>,
    pub(crate) buffers: Vec<Arc<Vec<u8>>>,
}

/// Largest byte count one string buffer may hold: view offsets and lengths are u32.
const MAX_BUFFER: usize = u32::MAX as usize;

impl StringViews {
    pub fn from_strs<S: AsRef<str>>(items: &[S]) -> Self {
        Self::from_strs_with_limit(items, MAX_BUFFER)
    }

    /// Packs `items` into buffers of at most `limit` bytes. A new buffer starts when the current one
    /// would exceed `limit`, so every offset and length fits in the view's u32 fields.
    fn from_strs_with_limit<S: AsRef<str>>(items: &[S], limit: usize) -> Self {
        assert!(limit <= MAX_BUFFER, "buffer limit {limit} exceeds u32::MAX");
        let mut buffers: Vec<Vec<u8>> = vec![Vec::new()];
        let mut views = Vec::with_capacity(items.len());
        for s in items {
            let b = s.as_ref().as_bytes();
            assert!(
                b.len() <= limit,
                "string of {} bytes exceeds the {limit}-byte buffer limit; a single string cannot span buffers",
                b.len()
            );
            let mut data = [0u8; 12];
            if b.len() <= 12 {
                data[..b.len()].copy_from_slice(b);
            } else {
                if buffers
                    .last()
                    .is_some_and(|cur| cur.len() + b.len() > limit)
                {
                    buffers.push(Vec::new());
                }
                let bi = buffers.len() - 1;
                let bi32 = u32::try_from(bi).expect("more than u32::MAX string buffers");
                let off = buffers[bi].len() as u32;
                data[..4].copy_from_slice(&b[..4]);
                data[4..8].copy_from_slice(&bi32.to_le_bytes());
                data[8..12].copy_from_slice(&off.to_le_bytes());
                buffers[bi].extend_from_slice(b);
            }
            views.push(View {
                len: b.len() as u32,
                data,
            });
        }
        StringViews {
            views,
            buffers: buffers.into_iter().map(Arc::new).collect(),
        }
    }

    pub fn len(&self) -> usize {
        self.views.len()
    }
    pub fn is_empty(&self) -> bool {
        self.views.is_empty()
    }

    pub fn get(&self, i: usize) -> &str {
        let v = &self.views[i];
        let n = v.len as usize;
        let bytes: &[u8] = if n <= 12 {
            &v.data[..n]
        } else {
            let bi = u32::from_le_bytes(v.data[4..8].try_into().unwrap()) as usize;
            let off = u32::from_le_bytes(v.data[8..12].try_into().unwrap()) as usize;
            &self.buffers[bi][off..off + n]
        };
        // SAFETY: invariant: the fields are pub(crate), so only crate code builds views, and every
        // constructor copies bytes from a &str: from_strs_with_limit, take (shares buffers), concat
        // (rebases buffer indices). A view's bytes lie inside its buffer and are valid UTF-8.
        unsafe { std::str::from_utf8_unchecked(bytes) }
    }

    /// Zero-copy: shares buffers, copies 16-byte views only.
    pub fn take(&self, idx: &[usize]) -> Self {
        StringViews {
            views: idx.iter().map(|&i| self.views[i]).collect(),
            buffers: self.buffers.clone(),
        }
    }

    pub fn concat(parts: &[&StringViews]) -> Self {
        let mut out = StringViews::default();
        for p in parts {
            let base = out.buffers.len() as u32;
            out.buffers.extend(p.buffers.iter().cloned());
            for v in &p.views {
                let mut v = *v;
                if v.len > 12 {
                    let bi = u32::from_le_bytes(v.data[4..8].try_into().unwrap()) + base;
                    v.data[4..8].copy_from_slice(&bi.to_le_bytes());
                }
                out.views.push(v);
            }
        }
        out
    }
}

#[derive(Clone, Debug)]
pub enum Values {
    Bool(Vec<bool>),
    Int64(Vec<i64>),
    Float64(Vec<f64>),
    Utf8(StringViews),
}

/// `validity[i] == true` means non-null. `None` = all valid.
#[derive(Clone, Debug)]
pub struct Column {
    pub values: Values,
    pub validity: Option<Vec<bool>>,
}

impl Column {
    pub fn from_bool(v: Vec<bool>) -> Self {
        Column {
            values: Values::Bool(v),
            validity: None,
        }
    }
    pub fn from_i64(v: Vec<i64>) -> Self {
        Column {
            values: Values::Int64(v),
            validity: None,
        }
    }
    pub fn from_f64(v: Vec<f64>) -> Self {
        Column {
            values: Values::Float64(v),
            validity: None,
        }
    }
    pub fn from_strs<S: AsRef<str>>(v: &[S]) -> Self {
        Column {
            values: Values::Utf8(StringViews::from_strs(v)),
            validity: None,
        }
    }
    pub fn with_validity(mut self, validity: Vec<bool>) -> Self {
        assert_eq!(validity.len(), self.len());
        self.validity = Some(validity);
        self
    }

    pub fn len(&self) -> usize {
        match &self.values {
            Values::Bool(v) => v.len(),
            Values::Int64(v) => v.len(),
            Values::Float64(v) => v.len(),
            Values::Utf8(v) => v.len(),
        }
    }
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
    pub fn data_type(&self) -> DataType {
        match &self.values {
            Values::Bool(_) => DataType::Boolean,
            Values::Int64(_) => DataType::Int64,
            Values::Float64(_) => DataType::Float64,
            Values::Utf8(_) => DataType::Utf8,
        }
    }
    pub fn is_valid(&self, i: usize) -> bool {
        self.validity.as_ref().is_none_or(|v| v[i])
    }

    pub fn get(&self, i: usize) -> ScalarValue {
        if !self.is_valid(i) {
            return ScalarValue::Null(self.data_type());
        }
        match &self.values {
            Values::Bool(v) => ScalarValue::Boolean(v[i]),
            Values::Int64(v) => ScalarValue::Int64(v[i]),
            Values::Float64(v) => ScalarValue::Float64(v[i]),
            Values::Utf8(v) => ScalarValue::Utf8(v.get(i).to_string()),
        }
    }

    /// Build a column of type `dt` from scalars (Null of any type => null slot).
    pub fn from_scalars(dt: DataType, items: &[ScalarValue]) -> Result<Self> {
        let validity: Vec<bool> = items.iter().map(|s| !s.is_null()).collect();
        let bad = |s: &ScalarValue| EngineError::Type(format!("scalar {s:?} is not {dt:?}"));
        let values = match dt {
            DataType::Boolean => Values::Bool(
                items
                    .iter()
                    .map(|s| match s {
                        ScalarValue::Boolean(b) => Ok(*b),
                        ScalarValue::Null(_) => Ok(false),
                        o => Err(bad(o)),
                    })
                    .collect::<Result<_>>()?,
            ),
            DataType::Int64 => Values::Int64(
                items
                    .iter()
                    .map(|s| match s {
                        ScalarValue::Int64(b) => Ok(*b),
                        ScalarValue::Null(_) => Ok(0),
                        o => Err(bad(o)),
                    })
                    .collect::<Result<_>>()?,
            ),
            DataType::Float64 => Values::Float64(
                items
                    .iter()
                    .map(|s| match s {
                        ScalarValue::Float64(b) => Ok(*b),
                        ScalarValue::Null(_) => Ok(0.0),
                        o => Err(bad(o)),
                    })
                    .collect::<Result<_>>()?,
            ),
            DataType::Utf8 => {
                let strs: Vec<&str> = items
                    .iter()
                    .map(|s| match s {
                        ScalarValue::Utf8(b) => Ok(b.as_str()),
                        ScalarValue::Null(_) => Ok(""),
                        o => Err(bad(o)),
                    })
                    .collect::<Result<_>>()?;
                Values::Utf8(StringViews::from_strs(&strs))
            }
        };
        let validity = if validity.iter().all(|v| *v) {
            None
        } else {
            Some(validity)
        };
        Ok(Column { values, validity })
    }

    pub fn take(&self, idx: &[usize]) -> Column {
        let values = match &self.values {
            Values::Bool(v) => Values::Bool(idx.iter().map(|&i| v[i]).collect()),
            Values::Int64(v) => Values::Int64(idx.iter().map(|&i| v[i]).collect()),
            Values::Float64(v) => Values::Float64(idx.iter().map(|&i| v[i]).collect()),
            Values::Utf8(v) => Values::Utf8(v.take(idx)),
        };
        let validity = self
            .validity
            .as_ref()
            .map(|m| idx.iter().map(|&i| m[i]).collect());
        Column { values, validity }
    }

    /// Keep rows where mask[i] is true (null mask rows are the caller's concern).
    pub fn filter(&self, mask: &[bool]) -> Column {
        let idx: Vec<usize> = mask
            .iter()
            .enumerate()
            .filter(|(_, m)| **m)
            .map(|(i, _)| i)
            .collect();
        self.take(&idx)
    }

    // shortcut: slice copies via take, O(len); revisit if scans become slice-heavy.
    pub fn slice(&self, offset: usize, len: usize) -> Column {
        let idx: Vec<usize> = (offset..offset + len).collect();
        self.take(&idx)
    }

    pub fn concat(parts: &[&Column]) -> Result<Column> {
        let first = parts
            .first()
            .ok_or_else(|| EngineError::Plan("concat of zero columns".into()))?;
        let dt = first.data_type();
        if parts.iter().any(|c| c.data_type() != dt) {
            return Err(EngineError::Type("concat of mixed column types".into()));
        }
        let values = match dt {
            DataType::Boolean => Values::Bool(
                parts
                    .iter()
                    .flat_map(|c| match &c.values {
                        Values::Bool(v) => v.clone(),
                        _ => vec![],
                    })
                    .collect(),
            ),
            DataType::Int64 => Values::Int64(
                parts
                    .iter()
                    .flat_map(|c| match &c.values {
                        Values::Int64(v) => v.clone(),
                        _ => vec![],
                    })
                    .collect(),
            ),
            DataType::Float64 => Values::Float64(
                parts
                    .iter()
                    .flat_map(|c| match &c.values {
                        Values::Float64(v) => v.clone(),
                        _ => vec![],
                    })
                    .collect(),
            ),
            DataType::Utf8 => {
                let sv: Vec<&StringViews> = parts
                    .iter()
                    .filter_map(|c| match &c.values {
                        Values::Utf8(v) => Some(v),
                        _ => None,
                    })
                    .collect();
                Values::Utf8(StringViews::concat(&sv))
            }
        };
        let validity = if parts.iter().any(|c| c.validity.is_some()) {
            Some(
                parts
                    .iter()
                    .flat_map(|c| (0..c.len()).map(|i| c.is_valid(i)))
                    .collect(),
            )
        } else {
            None
        };
        Ok(Column { values, validity })
    }
}

#[derive(Clone, Debug)]
pub struct Batch {
    pub schema: SchemaRef,
    pub columns: Vec<Column>,
    pub num_rows: usize,
}

impl Batch {
    pub fn try_new(schema: SchemaRef, columns: Vec<Column>) -> Result<Self> {
        if schema.fields.len() != columns.len() {
            return Err(EngineError::Plan("schema/column count mismatch".into()));
        }
        let num_rows = columns.first().map_or(0, |c| c.len());
        for (f, c) in schema.fields.iter().zip(&columns) {
            if c.len() != num_rows || c.data_type() != f.data_type {
                return Err(EngineError::Type(format!(
                    "column '{}' length/type mismatch",
                    f.name
                )));
            }
        }
        Ok(Batch {
            schema,
            columns,
            num_rows,
        })
    }
    pub fn empty(schema: SchemaRef) -> Self {
        Batch {
            schema,
            columns: vec![],
            num_rows: 0,
        }
    }
    pub fn take(&self, idx: &[usize]) -> Batch {
        Batch {
            schema: self.schema.clone(),
            columns: self.columns.iter().map(|c| c.take(idx)).collect(),
            num_rows: idx.len(),
        }
    }
    pub fn filter(&self, mask: &[bool]) -> Batch {
        let idx: Vec<usize> = mask
            .iter()
            .enumerate()
            .filter(|(_, m)| **m)
            .map(|(i, _)| i)
            .collect();
        self.take(&idx)
    }
    pub fn slice(&self, offset: usize, len: usize) -> Batch {
        let idx: Vec<usize> = (offset..offset + len).collect();
        self.take(&idx)
    }
    pub fn project(&self, cols: &[usize], schema: SchemaRef) -> Batch {
        Batch {
            schema,
            columns: cols.iter().map(|&i| self.columns[i].clone()).collect(),
            num_rows: self.num_rows,
        }
    }
    pub fn concat(schema: SchemaRef, batches: &[Batch]) -> Result<Batch> {
        let total_rows: usize = batches.iter().map(|b| b.num_rows).sum();
        if schema.fields.is_empty() {
            // Zero-column batches carry only a row count; it must survive concatenation.
            return Ok(Batch {
                schema,
                columns: vec![],
                num_rows: total_rows,
            });
        }
        if batches
            .iter()
            .any(|b| b.columns.is_empty() && b.num_rows > 0)
        {
            return Err(EngineError::Plan(
                "zero-column batch with rows cannot be concatenated into a schema with columns"
                    .into(),
            ));
        }
        let non_empty: Vec<&Batch> = batches.iter().filter(|b| !b.columns.is_empty()).collect();
        if non_empty.is_empty() {
            return Ok(Batch {
                schema: schema.clone(),
                columns: schema
                    .fields
                    .iter()
                    .map(|f| empty_col(f.data_type))
                    .collect(),
                num_rows: 0,
            });
        }
        let ncols = schema.fields.len();
        let mut cols = Vec::with_capacity(ncols);
        for i in 0..ncols {
            let parts: Vec<&Column> = non_empty.iter().map(|b| &b.columns[i]).collect();
            cols.push(Column::concat(&parts)?);
        }
        Batch::try_new(schema, cols)
    }
    /// Approximate heap bytes, used by the memory budget.
    pub fn byte_size(&self) -> usize {
        self.columns
            .iter()
            .map(|c| {
                let v = match &c.values {
                    Values::Bool(v) => v.len(),
                    Values::Int64(v) => v.len() * 8,
                    Values::Float64(v) => v.len() * 8,
                    Values::Utf8(s) => {
                        s.views.len() * 16 + s.buffers.iter().map(|b| b.len()).sum::<usize>()
                    }
                };
                v + c.validity.as_ref().map_or(0, |m| m.len())
            })
            .sum()
    }
    pub fn row(&self, i: usize) -> Vec<ScalarValue> {
        self.columns.iter().map(|c| c.get(i)).collect()
    }
}

pub fn empty_col(dt: DataType) -> Column {
    match dt {
        DataType::Boolean => Column::from_bool(vec![]),
        DataType::Int64 => Column::from_i64(vec![]),
        DataType::Float64 => Column::from_f64(vec![]),
        DataType::Utf8 => Column::from_strs::<&str>(&[]),
    }
}

pub fn schema_ref(s: Schema) -> SchemaRef {
    Arc::new(s)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn view_is_16_bytes() {
        assert_eq!(std::mem::size_of::<View>(), 16);
    }
    #[test]
    fn strings_inline_and_long_roundtrip_through_take_and_concat() {
        let a = Column::from_strs(&["hi", "a very long string over twelve bytes", ""]);
        let b = Column::from_strs(&["another long string past the inline limit"]);
        let c = Column::concat(&[&a.take(&[1, 0]), &b]).unwrap();
        assert_eq!(
            c.get(0),
            ScalarValue::Utf8("a very long string over twelve bytes".into())
        );
        assert_eq!(c.get(1), ScalarValue::Utf8("hi".into()));
        assert_eq!(
            c.get(2),
            ScalarValue::Utf8("another long string past the inline limit".into())
        );
    }
    #[test]
    fn many_buffers_keep_every_string_under_small_limit() {
        let strs: Vec<String> = (0..200)
            .map(|i| {
                if i % 5 == 0 {
                    format!("s{i}")
                } else {
                    format!("{i:0>width$}", width = 13 + i % 48)
                }
            })
            .collect();
        let sv = StringViews::from_strs_with_limit(&strs, 64);
        assert!(
            sv.buffers.len() > 1,
            "expected several buffers, got {}",
            sv.buffers.len()
        );
        for (i, s) in strs.iter().enumerate() {
            assert_eq!(sv.get(i), s.as_str(), "row {i}");
        }
    }

    #[test]
    fn take_and_concat_round_trip_across_buffers() {
        let strs: Vec<String> = (0..50).map(|i| format!("row-{i:0>20}")).collect();
        let sv = StringViews::from_strs_with_limit(&strs, 64);
        assert!(
            sv.buffers.len() > 1,
            "expected several buffers, got {}",
            sv.buffers.len()
        );
        let col = Column {
            values: Values::Utf8(sv),
            validity: None,
        };
        let idx: Vec<usize> = (0..50).rev().step_by(3).collect();
        let joined = Column::concat(&[&col.take(&idx), &col]).unwrap();
        let expected: Vec<String> = idx
            .iter()
            .map(|&i| strs[i].clone())
            .chain(strs.iter().cloned())
            .collect();
        for (i, s) in expected.iter().enumerate() {
            assert_eq!(joined.get(i), ScalarValue::Utf8(s.clone()), "row {i}");
        }
    }

    #[test]
    #[should_panic(expected = "exceeds the 64-byte buffer limit")]
    fn string_longer_than_buffer_limit_panics() {
        StringViews::from_strs_with_limit(&["x".repeat(65)], 64);
    }

    #[test]
    fn zero_column_batches_keep_their_row_count() {
        let schema: SchemaRef = Arc::new(Schema::new(vec![]));
        let b = Batch {
            schema: schema.clone(),
            columns: vec![],
            num_rows: 3,
        };
        assert_eq!(
            Batch::concat(schema.clone(), &[b.clone(), b.clone()])
                .unwrap()
                .num_rows,
            6
        );
        assert_eq!(b.take(&[0, 2]).num_rows, 2);
        assert_eq!(b.slice(1, 2).num_rows, 2);
    }

    #[test]
    fn validity_survives_filter_and_concat() {
        let a = Column::from_i64(vec![1, 2, 3]).with_validity(vec![true, false, true]);
        let f = a.filter(&[true, true, false]);
        assert_eq!(f.get(1), ScalarValue::Null(DataType::Int64));
        let c = Column::concat(&[&f, &a]).unwrap();
        assert_eq!(c.len(), 5);
        assert!(!c.is_valid(1) && !c.is_valid(3));
    }
}
