//! Columnar data: typed vectors + validity + German-style 16-byte string views.
use crate::types::{DataType, EngineError, Result, ScalarValue, Schema, SchemaRef};
use std::sync::Arc;

/// 16-byte string descriptor (spec §5.2). len<=12: bytes inline in `data`.
/// len>12: data[0..4]=prefix, data[4..8]=buffer index, data[8..12]=offset (LE).
#[repr(C, align(16))]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct View {
    pub len: u32,
    pub data: [u8; 12],
}

#[derive(Clone, Debug, Default)]
pub struct StringViews {
    pub views: Vec<View>,
    pub buffers: Vec<Arc<Vec<u8>>>,
}

impl StringViews {
    pub fn from_strs<S: AsRef<str>>(items: &[S]) -> Self {
        let mut buf: Vec<u8> = Vec::new();
        let mut views = Vec::with_capacity(items.len());
        for s in items {
            let b = s.as_ref().as_bytes();
            let mut data = [0u8; 12];
            if b.len() <= 12 {
                data[..b.len()].copy_from_slice(b);
            } else {
                data[..4].copy_from_slice(&b[..4]);
                data[4..8].copy_from_slice(&0u32.to_le_bytes());
                data[8..12].copy_from_slice(&(buf.len() as u32).to_le_bytes());
                buf.extend_from_slice(b);
            }
            views.push(View { len: b.len() as u32, data });
        }
        StringViews { views, buffers: vec![Arc::new(buf)] }
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
        // SAFETY: every view is built from a &str (valid UTF-8) via from_strs/take/concat.
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
        Column { values: Values::Bool(v), validity: None }
    }
    pub fn from_i64(v: Vec<i64>) -> Self {
        Column { values: Values::Int64(v), validity: None }
    }
    pub fn from_f64(v: Vec<f64>) -> Self {
        Column { values: Values::Float64(v), validity: None }
    }
    pub fn from_strs<S: AsRef<str>>(v: &[S]) -> Self {
        Column { values: Values::Utf8(StringViews::from_strs(v)), validity: None }
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
                items.iter().map(|s| match s {
                    ScalarValue::Boolean(b) => Ok(*b),
                    ScalarValue::Null(_) => Ok(false),
                    o => Err(bad(o)),
                }).collect::<Result<_>>()?,
            ),
            DataType::Int64 => Values::Int64(
                items.iter().map(|s| match s {
                    ScalarValue::Int64(b) => Ok(*b),
                    ScalarValue::Null(_) => Ok(0),
                    o => Err(bad(o)),
                }).collect::<Result<_>>()?,
            ),
            DataType::Float64 => Values::Float64(
                items.iter().map(|s| match s {
                    ScalarValue::Float64(b) => Ok(*b),
                    ScalarValue::Null(_) => Ok(0.0),
                    o => Err(bad(o)),
                }).collect::<Result<_>>()?,
            ),
            DataType::Utf8 => {
                let strs: Vec<&str> = items.iter().map(|s| match s {
                    ScalarValue::Utf8(b) => Ok(b.as_str()),
                    ScalarValue::Null(_) => Ok(""),
                    o => Err(bad(o)),
                }).collect::<Result<_>>()?;
                Values::Utf8(StringViews::from_strs(&strs))
            }
        };
        let validity = if validity.iter().all(|v| *v) { None } else { Some(validity) };
        Ok(Column { values, validity })
    }

    pub fn take(&self, idx: &[usize]) -> Column {
        let values = match &self.values {
            Values::Bool(v) => Values::Bool(idx.iter().map(|&i| v[i]).collect()),
            Values::Int64(v) => Values::Int64(idx.iter().map(|&i| v[i]).collect()),
            Values::Float64(v) => Values::Float64(idx.iter().map(|&i| v[i]).collect()),
            Values::Utf8(v) => Values::Utf8(v.take(idx)),
        };
        let validity = self.validity.as_ref().map(|m| idx.iter().map(|&i| m[i]).collect());
        Column { values, validity }
    }

    /// Keep rows where mask[i] is true (null mask rows are the caller's concern).
    pub fn filter(&self, mask: &[bool]) -> Column {
        let idx: Vec<usize> = mask.iter().enumerate().filter(|(_, m)| **m).map(|(i, _)| i).collect();
        self.take(&idx)
    }

    // shortcut: slice copies via take, O(len); revisit if scans become slice-heavy.
    pub fn slice(&self, offset: usize, len: usize) -> Column {
        let idx: Vec<usize> = (offset..offset + len).collect();
        self.take(&idx)
    }

    pub fn concat(parts: &[&Column]) -> Result<Column> {
        let first = parts.first().ok_or_else(|| EngineError::Plan("concat of zero columns".into()))?;
        let dt = first.data_type();
        if parts.iter().any(|c| c.data_type() != dt) {
            return Err(EngineError::Type("concat of mixed column types".into()));
        }
        let values = match dt {
            DataType::Boolean => Values::Bool(parts.iter().flat_map(|c| match &c.values { Values::Bool(v) => v.clone(), _ => vec![] }).collect()),
            DataType::Int64 => Values::Int64(parts.iter().flat_map(|c| match &c.values { Values::Int64(v) => v.clone(), _ => vec![] }).collect()),
            DataType::Float64 => Values::Float64(parts.iter().flat_map(|c| match &c.values { Values::Float64(v) => v.clone(), _ => vec![] }).collect()),
            DataType::Utf8 => {
                let sv: Vec<&StringViews> = parts.iter().filter_map(|c| match &c.values { Values::Utf8(v) => Some(v), _ => None }).collect();
                Values::Utf8(StringViews::concat(&sv))
            }
        };
        let validity = if parts.iter().any(|c| c.validity.is_some()) {
            Some(parts.iter().flat_map(|c| (0..c.len()).map(|i| c.is_valid(i))).collect())
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
                return Err(EngineError::Type(format!("column '{}' length/type mismatch", f.name)));
            }
        }
        Ok(Batch { schema, columns, num_rows })
    }
    pub fn empty(schema: SchemaRef) -> Self {
        Batch { schema, columns: vec![], num_rows: 0 }
    }
    pub fn take(&self, idx: &[usize]) -> Batch {
        Batch { schema: self.schema.clone(), columns: self.columns.iter().map(|c| c.take(idx)).collect(), num_rows: idx.len() }
    }
    pub fn filter(&self, mask: &[bool]) -> Batch {
        let idx: Vec<usize> = mask.iter().enumerate().filter(|(_, m)| **m).map(|(i, _)| i).collect();
        self.take(&idx)
    }
    pub fn slice(&self, offset: usize, len: usize) -> Batch {
        let idx: Vec<usize> = (offset..offset + len).collect();
        self.take(&idx)
    }
    pub fn project(&self, cols: &[usize], schema: SchemaRef) -> Batch {
        Batch { schema, columns: cols.iter().map(|&i| self.columns[i].clone()).collect(), num_rows: self.num_rows }
    }
    pub fn concat(schema: SchemaRef, batches: &[Batch]) -> Result<Batch> {
        let non_empty: Vec<&Batch> = batches.iter().filter(|b| !b.columns.is_empty()).collect();
        if non_empty.is_empty() {
            return Ok(Batch { schema: schema.clone(), columns: schema.fields.iter().map(|f| empty_col(f.data_type)).collect(), num_rows: 0 });
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
        self.columns.iter().map(|c| {
            let v = match &c.values {
                Values::Bool(v) => v.len(),
                Values::Int64(v) => v.len() * 8,
                Values::Float64(v) => v.len() * 8,
                Values::Utf8(s) => s.views.len() * 16 + s.buffers.iter().map(|b| b.len()).sum::<usize>(),
            };
            v + c.validity.as_ref().map_or(0, |m| m.len())
        }).sum()
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
        assert_eq!(c.get(0), ScalarValue::Utf8("a very long string over twelve bytes".into()));
        assert_eq!(c.get(1), ScalarValue::Utf8("hi".into()));
        assert_eq!(c.get(2), ScalarValue::Utf8("another long string past the inline limit".into()));
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
