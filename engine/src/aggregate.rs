//! SLICE D2. Hash aggregation (partial per-batch + merge), group keys incl. NULL as its own group.
use crate::column::{Batch, Column, Values};
use crate::eval::eval;
use crate::expr::Expr;
use crate::join::encode_key;
use crate::memory::{MemoryBudget, Reservation};
use crate::plan::{AggExpr, AggFunc};
use crate::types::{DataType, EngineError, Result, ScalarValue, SchemaRef};
use std::collections::HashMap;
use std::path::Path;
use std::sync::Arc;

/// Approximate per-group overhead charged to the budget on top of the encoded key bytes.
const GROUP_OVERHEAD: usize = 64;
/// Approximate per-group, per-aggregate state bytes charged to the budget.
const STATE_BYTES: usize = 24;

/// Per-aggregate typed state, one slot per group. Numeric states never box a value per row.
enum Acc {
    Count(Vec<i64>),
    SumI(Vec<i64>, Vec<bool>),
    SumF(Vec<f64>, Vec<bool>),
    MinMaxI {
        min: bool,
        v: Vec<i64>,
        has: Vec<bool>,
    },
    MinMaxF {
        min: bool,
        v: Vec<f64>,
        has: Vec<bool>,
    },
    MinMaxB {
        min: bool,
        v: Vec<Option<bool>>,
    },
    MinMaxS {
        min: bool,
        v: Vec<Option<String>>,
    },
    AvgF {
        sum: Vec<f64>,
        cnt: Vec<i64>,
    },
}

impl Acc {
    fn new(agg: &AggExpr) -> Result<Acc> {
        let input_dt = agg.expr.as_ref().map(|e| e.return_type());
        let acc = Acc::for_input(agg, input_dt)?;
        let produced = acc.output_type(input_dt);
        if produced != agg.output_type {
            let input = input_dt.map_or("*".to_string(), |d| format!("{d:?}"));
            return Err(EngineError::Type(format!(
                "aggregate {:?} over {input} produces {produced:?}, but the plan declares {:?}",
                agg.func, agg.output_type
            )));
        }
        Ok(acc)
    }

    /// Output type of this state. `input_dt` is the aggregate's input type (`None` for COUNT(*)).
    fn output_type(&self, input_dt: Option<DataType>) -> DataType {
        match self {
            Acc::Count(_) | Acc::SumI(..) => DataType::Int64,
            Acc::SumF(..) | Acc::AvgF { .. } => DataType::Float64,
            Acc::MinMaxI { .. }
            | Acc::MinMaxF { .. }
            | Acc::MinMaxB { .. }
            | Acc::MinMaxS { .. } => input_dt.expect("min/max always has an input"),
        }
    }

    fn for_input(agg: &AggExpr, input_dt: Option<DataType>) -> Result<Acc> {
        let acc = match (agg.func, input_dt) {
            (AggFunc::Count, _) => Acc::Count(vec![]),
            (AggFunc::Sum, Some(DataType::Int64)) => Acc::SumI(vec![], vec![]),
            (AggFunc::Sum, Some(DataType::Float64)) => Acc::SumF(vec![], vec![]),
            (AggFunc::Min | AggFunc::Max, Some(dt)) => {
                let min = agg.func == AggFunc::Min;
                match dt {
                    DataType::Int64 => Acc::MinMaxI {
                        min,
                        v: vec![],
                        has: vec![],
                    },
                    DataType::Float64 => Acc::MinMaxF {
                        min,
                        v: vec![],
                        has: vec![],
                    },
                    DataType::Boolean => Acc::MinMaxB { min, v: vec![] },
                    DataType::Utf8 => Acc::MinMaxS { min, v: vec![] },
                }
            }
            (AggFunc::Avg, Some(DataType::Int64 | DataType::Float64)) => Acc::AvgF {
                sum: vec![],
                cnt: vec![],
            },
            (func, dt) => {
                return Err(EngineError::Type(format!(
                    "aggregate {func:?} not supported on {dt:?}"
                )));
            }
        };
        Ok(acc)
    }

    /// Appends the empty state for a newly created group.
    fn push_group(&mut self) {
        match self {
            Acc::Count(c) => c.push(0),
            Acc::SumI(v, h) => {
                v.push(0);
                h.push(false);
            }
            Acc::SumF(v, h) => {
                v.push(0.0);
                h.push(false);
            }
            Acc::MinMaxI { v, has, .. } => {
                v.push(0);
                has.push(false);
            }
            Acc::MinMaxF { v, has, .. } => {
                v.push(0.0);
                has.push(false);
            }
            Acc::MinMaxB { v, .. } => v.push(None),
            Acc::MinMaxS { v, .. } => v.push(None),
            Acc::AvgF { sum, cnt } => {
                sum.push(0.0);
                cnt.push(0);
            }
        }
    }

    /// Folds row `row` of `col` into group `g`. NULL inputs are skipped. `col == None` is COUNT(*).
    fn update(&mut self, col: Option<&Column>, row: usize, g: usize) {
        if let Acc::Count(c) = self {
            if col.is_none_or(|c| c.is_valid(row)) {
                c[g] += 1;
            }
            return;
        }
        let Some(col) = col else { return };
        if !col.is_valid(row) {
            return;
        }
        match (self, &col.values) {
            (Acc::SumI(v, h), Values::Int64(x)) => {
                v[g] = v[g].wrapping_add(x[row]);
                h[g] = true;
            }
            (Acc::SumF(v, h), Values::Float64(x)) => {
                v[g] += x[row];
                h[g] = true;
            }
            (Acc::MinMaxI { min, v, has }, Values::Int64(x)) => {
                let x = x[row];
                if !has[g] || (*min && x < v[g]) || (!*min && x > v[g]) {
                    v[g] = x;
                    has[g] = true;
                }
            }
            (Acc::MinMaxF { min, v, has }, Values::Float64(x)) => {
                // NaN is ignored unless it is the only non-null value seen: order independent.
                let x = x[row];
                if x.is_nan() {
                    if !has[g] {
                        v[g] = x;
                        has[g] = true;
                    }
                } else if !has[g] || v[g].is_nan() || (*min && x < v[g]) || (!*min && x > v[g]) {
                    v[g] = x;
                    has[g] = true;
                }
            }
            (Acc::MinMaxB { min, v }, Values::Bool(x)) => {
                let x = x[row];
                let replace = match v[g] {
                    None => true,
                    Some(cur) => (*min && !x && cur) || (!*min && x && !cur),
                };
                if replace {
                    v[g] = Some(x);
                }
            }
            (Acc::MinMaxS { min, v }, Values::Utf8(s)) => {
                let x = s.get(row);
                let replace = match &v[g] {
                    None => true,
                    Some(cur) => (*min && x < cur.as_str()) || (!*min && x > cur.as_str()),
                };
                if replace {
                    v[g] = Some(x.to_string());
                }
            }
            (Acc::AvgF { sum, cnt }, Values::Int64(x)) => {
                sum[g] += x[row] as f64;
                cnt[g] += 1;
            }
            (Acc::AvgF { sum, cnt }, Values::Float64(x)) => {
                sum[g] += x[row];
                cnt[g] += 1;
            }
            _ => unreachable!("accumulator/input type mismatch is rejected in Acc::new"),
        }
    }

    /// Final column for all groups, typed by the aggregate's declared output type.
    fn finish(self, out_dt: DataType) -> Result<Column> {
        let col = match self {
            Acc::Count(c) => Column::from_i64(c),
            Acc::SumI(v, h) | Acc::MinMaxI { v, has: h, .. } => {
                with_validity(Column::from_i64(v), h)
            }
            Acc::SumF(v, h) | Acc::MinMaxF { v, has: h, .. } => {
                with_validity(Column::from_f64(v), h)
            }
            Acc::MinMaxB { v, .. } => {
                let items: Vec<ScalarValue> = v
                    .into_iter()
                    .map(|x| x.map_or(ScalarValue::Null(DataType::Boolean), ScalarValue::Boolean))
                    .collect();
                Column::from_scalars(DataType::Boolean, &items)?
            }
            Acc::MinMaxS { v, .. } => {
                let items: Vec<ScalarValue> = v
                    .into_iter()
                    .map(|x| x.map_or(ScalarValue::Null(DataType::Utf8), ScalarValue::Utf8))
                    .collect();
                Column::from_scalars(DataType::Utf8, &items)?
            }
            Acc::AvgF { sum, cnt } => {
                let has: Vec<bool> = cnt.iter().map(|c| *c > 0).collect();
                let vals: Vec<f64> = sum
                    .iter()
                    .zip(&cnt)
                    .map(|(s, c)| if *c > 0 { s / *c as f64 } else { 0.0 })
                    .collect();
                with_validity(Column::from_f64(vals), has)
            }
        };
        if col.data_type() != out_dt {
            return Err(EngineError::Type(format!(
                "aggregate produced {:?}, but the output schema declares {out_dt:?}",
                col.data_type()
            )));
        }
        Ok(col)
    }
}

/// Attaches a validity mask only when some slot is NULL.
fn with_validity(col: Column, has: Vec<bool>) -> Column {
    if has.iter().all(|h| *h) {
        col
    } else {
        col.with_validity(has)
    }
}

/// No group exprs => exactly one output row (even for empty input: Count=0, others NULL).
/// Group order in output is deterministic: first-seen order of keys.
// shortcut: no spilling aggregation. Group table is charged to `budget` and Err(Memory) is returned
// as-is; `spill_dir` is unused until a partitioned aggregate spill exists.
pub fn hash_aggregate(
    input: Vec<Batch>,
    group_exprs: &[Expr],
    aggs: &[AggExpr],
    schema: SchemaRef,
    budget: &Arc<MemoryBudget>,
    _spill_dir: &Path,
) -> Result<Vec<Batch>> {
    if schema.fields.len() != group_exprs.len() + aggs.len() {
        return Err(EngineError::Plan(format!(
            "aggregate schema has {} fields, expected {} group + {} aggregate",
            schema.fields.len(),
            group_exprs.len(),
            aggs.len()
        )));
    }
    let mut accs: Vec<Acc> = aggs.iter().map(Acc::new).collect::<Result<_>>()?;
    let mut index: HashMap<Vec<u8>, usize> = HashMap::new();
    // Per group expr, the scalar key value of each group (boxed once per new group, not per row).
    let mut key_vals: Vec<Vec<ScalarValue>> = vec![Vec::new(); group_exprs.len()];
    let mut res: Reservation = budget.try_reserve(0)?;
    let group_bytes = GROUP_OVERHEAD + STATE_BYTES * aggs.len() + 16 * group_exprs.len();
    let mut num_groups = 0usize;
    let mut buf: Vec<u8> = Vec::new();

    // Global group without keys: one group exists even when the input is empty.
    if group_exprs.is_empty() {
        res.grow(group_bytes)?;
        accs.iter_mut().for_each(Acc::push_group);
        index.insert(Vec::new(), 0);
        num_groups = 1;
    }

    for batch in &input {
        if batch.num_rows == 0 {
            continue;
        }
        let keys: Vec<Column> = group_exprs
            .iter()
            .map(|e| eval(e, batch))
            .collect::<Result<_>>()?;
        let agg_cols: Vec<Option<Column>> = aggs
            .iter()
            .map(|a| a.expr.as_ref().map(|e| eval(e, batch)).transpose())
            .collect::<Result<_>>()?;
        for i in 0..batch.num_rows {
            buf.clear();
            encode_key(&keys, i, &mut buf);
            let g = match index.get(buf.as_slice()) {
                Some(&g) => g,
                None => {
                    let key_len = buf.len();
                    res.grow(group_bytes + key_len)?;
                    accs.iter_mut().for_each(Acc::push_group);
                    for (vals, col) in key_vals.iter_mut().zip(&keys) {
                        vals.push(col.get(i));
                    }
                    index.insert(buf.clone(), num_groups);
                    num_groups += 1;
                    num_groups - 1
                }
            };
            for (acc, col) in accs.iter_mut().zip(&agg_cols) {
                acc.update(col.as_ref(), i, g);
            }
        }
    }

    let mut columns: Vec<Column> = Vec::with_capacity(schema.fields.len());
    for (k, vals) in key_vals.iter().enumerate() {
        columns.push(Column::from_scalars(schema.fields[k].data_type, vals)?);
    }
    for (j, acc) in accs.into_iter().enumerate() {
        columns.push(acc.finish(schema.fields[group_exprs.len() + j].data_type)?);
    }
    let out = Batch::try_new(schema, columns)?;
    Ok(vec![out])
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::eval::{binary, col, lit};
    use crate::expr::BinaryOp;
    use crate::types::{Field, Schema};
    use std::path::PathBuf;

    fn dir() -> PathBuf {
        std::env::temp_dir()
    }

    fn sch(fields: &[(&str, DataType)]) -> Schema {
        Schema::new(
            fields
                .iter()
                .map(|(n, t)| Field::new(n, *t, true))
                .collect(),
        )
    }

    fn input(s: &Schema, cols: Vec<Column>) -> Batch {
        Batch::try_new(Arc::new(s.clone()), cols).unwrap()
    }

    fn rows(batches: &[Batch]) -> Vec<Vec<ScalarValue>> {
        batches
            .iter()
            .flat_map(|b| (0..b.num_rows).map(move |i| b.row(i)))
            .collect()
    }

    fn agg(func: AggFunc, e: Option<Expr>, out: DataType) -> AggExpr {
        AggExpr {
            func,
            expr: e,
            output_type: out,
        }
    }

    fn out_schema(fields: &[(&str, DataType)]) -> SchemaRef {
        Arc::new(sch(fields))
    }

    fn run(
        batches: Vec<Batch>,
        groups: &[Expr],
        aggs: &[AggExpr],
        schema: SchemaRef,
    ) -> Result<Vec<Batch>> {
        let budget = MemoryBudget::new(64 << 20);
        hash_aggregate(batches, groups, aggs, schema, &budget, &dir())
    }

    fn ints(v: &[i64]) -> Column {
        Column::from_i64(v.to_vec())
    }

    fn ints_nullable(v: &[Option<i64>]) -> Column {
        let vals: Vec<i64> = v.iter().map(|x| x.unwrap_or(0)).collect();
        let valid: Vec<bool> = v.iter().map(|x| x.is_some()).collect();
        Column::from_i64(vals).with_validity(valid)
    }

    fn one_int_col(vals: Option<Vec<Option<i64>>>) -> Vec<Batch> {
        let s = sch(&[("a", DataType::Int64)]);
        let col = match vals {
            Some(v) => ints_nullable(&v),
            None => ints(&[]),
        };
        vec![input(&s, vec![col])]
    }

    #[test]
    fn count_star_no_groups() {
        let out = run(
            one_int_col(Some(vec![Some(1), None, Some(3)])),
            &[],
            &[agg(AggFunc::Count, None, DataType::Int64)],
            out_schema(&[("c", DataType::Int64)]),
        )
        .unwrap();
        assert_eq!(rows(&out), vec![vec![ScalarValue::Int64(3)]]);
    }

    #[test]
    fn count_expr_skips_nulls() {
        let a = col(&sch(&[("a", DataType::Int64)]), "a").unwrap();
        let out = run(
            one_int_col(Some(vec![Some(1), None, Some(3), None])),
            &[],
            &[agg(AggFunc::Count, Some(a), DataType::Int64)],
            out_schema(&[("c", DataType::Int64)]),
        )
        .unwrap();
        assert_eq!(rows(&out), vec![vec![ScalarValue::Int64(2)]]);
    }

    #[test]
    fn sum_int_ignores_nulls() {
        let a = col(&sch(&[("a", DataType::Int64)]), "a").unwrap();
        let out = run(
            one_int_col(Some(vec![Some(1), None, Some(5)])),
            &[],
            &[agg(AggFunc::Sum, Some(a), DataType::Int64)],
            out_schema(&[("s", DataType::Int64)]),
        )
        .unwrap();
        assert_eq!(rows(&out), vec![vec![ScalarValue::Int64(6)]]);
    }

    #[test]
    fn sum_int_wraps_on_overflow() {
        let a = col(&sch(&[("a", DataType::Int64)]), "a").unwrap();
        let out = run(
            one_int_col(Some(vec![Some(i64::MAX), Some(1)])),
            &[],
            &[agg(AggFunc::Sum, Some(a), DataType::Int64)],
            out_schema(&[("s", DataType::Int64)]),
        )
        .unwrap();
        assert_eq!(rows(&out), vec![vec![ScalarValue::Int64(i64::MIN)]]);
    }

    #[test]
    fn sum_float_ignores_nulls() {
        let s = sch(&[("f", DataType::Float64)]);
        let col_f = Column::from_f64(vec![1.5, 0.0, 2.25]).with_validity(vec![true, false, true]);
        let f = col(&s, "f").unwrap();
        let out = run(
            vec![input(&s, vec![col_f])],
            &[],
            &[agg(AggFunc::Sum, Some(f), DataType::Float64)],
            out_schema(&[("s", DataType::Float64)]),
        )
        .unwrap();
        assert_eq!(rows(&out), vec![vec![ScalarValue::Float64(3.75)]]);
    }

    #[test]
    fn min_max_int_with_nulls() {
        let a = col(&sch(&[("a", DataType::Int64)]), "a").unwrap();
        let mk = |f| agg(f, Some(a.clone()), DataType::Int64);
        let aggs = [mk(AggFunc::Min), mk(AggFunc::Max)];
        let out = run(
            one_int_col(Some(vec![None, Some(7), Some(-2), Some(4), None])),
            &[],
            &aggs,
            out_schema(&[("mn", DataType::Int64), ("mx", DataType::Int64)]),
        )
        .unwrap();
        assert_eq!(
            rows(&out),
            vec![vec![ScalarValue::Int64(-2), ScalarValue::Int64(7)]]
        );
    }

    #[test]
    fn min_max_utf8_strings() {
        let s = sch(&[("t", DataType::Utf8)]);
        let t = col(&s, "t").unwrap();
        let batch = input(
            &s,
            vec![Column::from_strs(&[
                "pear",
                "apple",
                "a string longer than twelve bytes",
                "zebra",
            ])],
        );
        let aggs = [
            agg(AggFunc::Min, Some(t.clone()), DataType::Utf8),
            agg(AggFunc::Max, Some(t), DataType::Utf8),
        ];
        let out = run(
            vec![batch],
            &[],
            &aggs,
            out_schema(&[("mn", DataType::Utf8), ("mx", DataType::Utf8)]),
        )
        .unwrap();
        assert_eq!(
            rows(&out),
            vec![vec![
                ScalarValue::Utf8("a string longer than twelve bytes".into()),
                ScalarValue::Utf8("zebra".into())
            ]]
        );
    }

    #[test]
    fn min_max_bool() {
        let s = sch(&[("b", DataType::Boolean)]);
        let b = col(&s, "b").unwrap();
        let batch = input(&s, vec![Column::from_bool(vec![true, true, false])]);
        let aggs = [
            agg(AggFunc::Min, Some(b.clone()), DataType::Boolean),
            agg(AggFunc::Max, Some(b), DataType::Boolean),
        ];
        let out = run(
            vec![batch],
            &[],
            &aggs,
            out_schema(&[("mn", DataType::Boolean), ("mx", DataType::Boolean)]),
        )
        .unwrap();
        assert_eq!(
            rows(&out),
            vec![vec![
                ScalarValue::Boolean(false),
                ScalarValue::Boolean(true)
            ]]
        );
    }

    #[test]
    fn avg_ignores_nulls() {
        let a = col(&sch(&[("a", DataType::Int64)]), "a").unwrap();
        let out = run(
            one_int_col(Some(vec![Some(1), None, Some(4)])),
            &[],
            &[agg(AggFunc::Avg, Some(a), DataType::Float64)],
            out_schema(&[("av", DataType::Float64)]),
        )
        .unwrap();
        assert_eq!(rows(&out), vec![vec![ScalarValue::Float64(2.5)]]);
    }

    #[test]
    fn empty_input_no_groups_gives_one_row() {
        let a = col(&sch(&[("a", DataType::Int64)]), "a").unwrap();
        let aggs = [
            agg(AggFunc::Count, None, DataType::Int64),
            agg(AggFunc::Sum, Some(a.clone()), DataType::Int64),
            agg(AggFunc::Min, Some(a.clone()), DataType::Int64),
            agg(AggFunc::Max, Some(a.clone()), DataType::Int64),
            agg(AggFunc::Avg, Some(a), DataType::Float64),
        ];
        let schema = out_schema(&[
            ("c", DataType::Int64),
            ("s", DataType::Int64),
            ("mn", DataType::Int64),
            ("mx", DataType::Int64),
            ("av", DataType::Float64),
        ]);
        let out = run(vec![], &[], &aggs, schema).unwrap();
        let n = ScalarValue::Null(DataType::Int64);
        assert_eq!(
            rows(&out),
            vec![vec![
                ScalarValue::Int64(0),
                n.clone(),
                n.clone(),
                n,
                ScalarValue::Null(DataType::Float64)
            ]]
        );
    }

    #[test]
    fn empty_input_with_group_gives_no_rows() {
        let s = sch(&[("g", DataType::Int64)]);
        let g = col(&s, "g").unwrap();
        let out = run(
            vec![],
            &[g],
            &[agg(AggFunc::Count, None, DataType::Int64)],
            out_schema(&[("g", DataType::Int64), ("c", DataType::Int64)]),
        )
        .unwrap();
        assert_eq!(rows(&out), Vec::<Vec<ScalarValue>>::new());
    }

    #[test]
    fn group_by_int_first_seen_order() {
        let s = sch(&[("g", DataType::Int64), ("v", DataType::Int64)]);
        let g = col(&s, "g").unwrap();
        let v = col(&s, "v").unwrap();
        let batch = input(
            &s,
            vec![ints(&[3, 1, 3, 2, 1]), ints(&[10, 20, 30, 40, 50])],
        );
        let aggs = [
            agg(AggFunc::Sum, Some(v), DataType::Int64),
            agg(AggFunc::Count, None, DataType::Int64),
        ];
        let out = run(
            vec![batch],
            &[g],
            &aggs,
            out_schema(&[
                ("g", DataType::Int64),
                ("s", DataType::Int64),
                ("c", DataType::Int64),
            ]),
        )
        .unwrap();
        let i = |x: i64| ScalarValue::Int64(x);
        assert_eq!(
            rows(&out),
            vec![
                vec![i(3), i(40), i(2)],
                vec![i(1), i(70), i(2)],
                vec![i(2), i(40), i(1)]
            ]
        );
    }

    #[test]
    fn null_group_key_is_its_own_group() {
        let s = sch(&[("g", DataType::Int64), ("v", DataType::Int64)]);
        let g = col(&s, "g").unwrap();
        let v = col(&s, "v").unwrap();
        let gcol = ints_nullable(&[None, Some(1), None, Some(1)]);
        let batch = input(&s, vec![gcol, ints(&[1, 2, 3, 4])]);
        let out = run(
            vec![batch],
            &[g],
            &[agg(AggFunc::Sum, Some(v), DataType::Int64)],
            out_schema(&[("g", DataType::Int64), ("s", DataType::Int64)]),
        )
        .unwrap();
        assert_eq!(
            rows(&out),
            vec![
                vec![ScalarValue::Null(DataType::Int64), ScalarValue::Int64(4)],
                vec![ScalarValue::Int64(1), ScalarValue::Int64(6)]
            ]
        );
    }

    #[test]
    fn group_by_utf8_with_long_strings() {
        let s = sch(&[("k", DataType::Utf8), ("v", DataType::Int64)]);
        let k = col(&s, "k").unwrap();
        let v = col(&s, "v").unwrap();
        let long_a = "this key is definitely longer than twelve bytes";
        let long_b = "this key is definitely longer than twelve bytez";
        let batch = input(
            &s,
            vec![
                Column::from_strs(&[long_a, "short", long_b, long_a, "short"]),
                ints(&[1, 2, 4, 8, 16]),
            ],
        );
        let out = run(
            vec![batch],
            &[k],
            &[agg(AggFunc::Sum, Some(v), DataType::Int64)],
            out_schema(&[("k", DataType::Utf8), ("s", DataType::Int64)]),
        )
        .unwrap();
        let s_ = |x: &str| ScalarValue::Utf8(x.into());
        let i = ScalarValue::Int64;
        assert_eq!(
            rows(&out),
            vec![
                vec![s_(long_a), i(9)],
                vec![s_("short"), i(18)],
                vec![s_(long_b), i(4)]
            ]
        );
    }

    #[test]
    fn two_key_group_with_null_and_multiple_batches() {
        let s = sch(&[
            ("a", DataType::Int64),
            ("b", DataType::Utf8),
            ("v", DataType::Int64),
        ]);
        let a = col(&s, "a").unwrap();
        let b = col(&s, "b").unwrap();
        let v = col(&s, "v").unwrap();
        let b1 = input(
            &s,
            vec![ints(&[1, 1]), Column::from_strs(&["x", "y"]), ints(&[1, 2])],
        );
        let b2 = input(
            &s,
            vec![
                ints_nullable(&[Some(1), None]),
                Column::from_strs(&["x", "x"]),
                ints(&[10, 20]),
            ],
        );
        let b3 = input(
            &s,
            vec![ints(&[1]), Column::from_strs(&["x"]), ints(&[100])],
        );
        let aggs = [agg(AggFunc::Sum, Some(v), DataType::Int64)];
        let out = run(
            vec![b1, b2, b3],
            &[a, b],
            &aggs,
            out_schema(&[
                ("a", DataType::Int64),
                ("b", DataType::Utf8),
                ("s", DataType::Int64),
            ]),
        )
        .unwrap();
        let i = ScalarValue::Int64;
        let n = ScalarValue::Null(DataType::Int64);
        let u = |x: &str| ScalarValue::Utf8(x.into());
        assert_eq!(
            rows(&out),
            vec![
                vec![i(1), u("x"), i(111)],
                vec![i(1), u("y"), i(2)],
                vec![n.clone(), u("x"), i(20)],
            ]
        );
    }

    #[test]
    fn group_by_expression_key() {
        let s = sch(&[("a", DataType::Int64), ("v", DataType::Int64)]);
        let a = col(&s, "a").unwrap();
        let parity = binary(a, BinaryOp::Mod, lit(ScalarValue::Int64(2))).unwrap();
        let batch = input(&s, vec![ints(&[1, 2, 3, 4, 5]), ints(&[1, 1, 1, 1, 1])]);
        let out = run(
            vec![batch],
            &[parity],
            &[agg(AggFunc::Count, None, DataType::Int64)],
            out_schema(&[("p", DataType::Int64), ("c", DataType::Int64)]),
        )
        .unwrap();
        let i = ScalarValue::Int64;
        assert_eq!(rows(&out), vec![vec![i(1), i(3)], vec![i(0), i(2)]]);
    }

    #[test]
    fn group_column_nulls_do_not_leak_into_sums() {
        let s = sch(&[("g", DataType::Int64), ("v", DataType::Int64)]);
        let g = col(&s, "g").unwrap();
        let v = col(&s, "v").unwrap();
        let batch = input(&s, vec![ints(&[1, 1]), ints_nullable(&[None, None])]);
        let out = run(
            vec![batch],
            &[g],
            &[
                agg(AggFunc::Sum, Some(v.clone()), DataType::Int64),
                agg(AggFunc::Max, Some(v), DataType::Int64),
            ],
            out_schema(&[
                ("g", DataType::Int64),
                ("s", DataType::Int64),
                ("m", DataType::Int64),
            ]),
        )
        .unwrap();
        let n = ScalarValue::Null(DataType::Int64);
        assert_eq!(rows(&out), vec![vec![ScalarValue::Int64(1), n.clone(), n]]);
    }

    fn min_max_f64(vals: Column) -> Vec<ScalarValue> {
        let s = sch(&[("f", DataType::Float64)]);
        let f = col(&s, "f").unwrap();
        let aggs = [
            agg(AggFunc::Min, Some(f.clone()), DataType::Float64),
            agg(AggFunc::Max, Some(f), DataType::Float64),
        ];
        let out = run(
            vec![input(&s, vec![vals])],
            &[],
            &aggs,
            out_schema(&[("mn", DataType::Float64), ("mx", DataType::Float64)]),
        )
        .unwrap();
        rows(&out).remove(0)
    }

    #[test]
    fn min_max_float_ignores_nan_in_every_position() {
        let nan = f64::NAN;
        let perms = [
            [nan, 1.0, 2.0],
            [nan, 2.0, 1.0],
            [1.0, nan, 2.0],
            [2.0, nan, 1.0],
            [1.0, 2.0, nan],
            [2.0, 1.0, nan],
        ];
        for p in perms {
            let got = min_max_f64(Column::from_f64(p.to_vec()));
            assert_eq!(
                got,
                vec![ScalarValue::Float64(1.0), ScalarValue::Float64(2.0)],
                "input {p:?}"
            );
        }
    }

    #[test]
    fn min_max_float_all_nan_is_nan() {
        let nan = f64::NAN;
        for vals in [
            Column::from_f64(vec![nan, nan]),
            Column::from_f64(vec![nan, 5.0]).with_validity(vec![true, false]),
        ] {
            for v in min_max_f64(vals) {
                match v {
                    ScalarValue::Float64(x) => assert!(x.is_nan(), "expected NaN, got {x}"),
                    o => panic!("expected Float64, got {o:?}"),
                }
            }
        }
    }

    #[test]
    fn sum_int_declared_as_float_is_type_error() {
        let a = col(&sch(&[("a", DataType::Int64)]), "a").unwrap();
        let err = run(
            one_int_col(Some(vec![Some(1), Some(2)])),
            &[],
            &[agg(AggFunc::Sum, Some(a), DataType::Float64)],
            out_schema(&[("s", DataType::Float64)]),
        )
        .unwrap_err();
        match err {
            EngineError::Type(msg) => assert!(
                msg.contains("Sum") && msg.contains("Int64") && msg.contains("Float64"),
                "{msg}"
            ),
            other => panic!("expected type error, got {other:?}"),
        }
    }

    #[test]
    fn memory_budget_exhaustion_returns_memory_error() {
        let s = sch(&[("g", DataType::Int64)]);
        let g = col(&s, "g").unwrap();
        let batch = input(&s, vec![ints(&(0..1000).collect::<Vec<i64>>())]);
        let budget = MemoryBudget::new(1024);
        let err = hash_aggregate(
            vec![batch],
            &[g],
            &[agg(AggFunc::Count, None, DataType::Int64)],
            out_schema(&[("g", DataType::Int64), ("c", DataType::Int64)]),
            &budget,
            &dir(),
        )
        .unwrap_err();
        assert!(matches!(err, EngineError::Memory { .. }), "got {err:?}");
        assert_eq!(budget.used(), 0, "reservation released after failure");
    }

    #[test]
    fn schema_arity_mismatch_is_plan_error() {
        let err = run(
            vec![],
            &[],
            &[agg(AggFunc::Count, None, DataType::Int64)],
            out_schema(&[("a", DataType::Int64), ("b", DataType::Int64)]),
        )
        .unwrap_err();
        assert!(matches!(err, EngineError::Plan(_)));
    }

    #[test]
    fn sum_on_utf8_is_type_error() {
        let s = sch(&[("t", DataType::Utf8)]);
        let t = col(&s, "t").unwrap();
        let batch = input(&s, vec![Column::from_strs(&["a"])]);
        let err = run(
            vec![batch],
            &[],
            &[agg(AggFunc::Sum, Some(t), DataType::Utf8)],
            out_schema(&[("s", DataType::Utf8)]),
        )
        .unwrap_err();
        assert!(matches!(err, EngineError::Type(_)));
    }

    #[test]
    fn group_table_many_groups_and_counts_match_naive() {
        let s = sch(&[("g", DataType::Int64), ("v", DataType::Int64)]);
        let g = col(&s, "g").unwrap();
        let v = col(&s, "v").unwrap();
        let gs: Vec<i64> = (0..5000).map(|i| (i * 7919) % 613).collect();
        let vs: Vec<i64> = (0..5000).collect();
        let batch = input(&s, vec![ints(&gs), ints(&vs)]);
        let out = run(
            vec![batch],
            &[g],
            &[
                agg(AggFunc::Sum, Some(v), DataType::Int64),
                agg(AggFunc::Count, None, DataType::Int64),
            ],
            out_schema(&[
                ("g", DataType::Int64),
                ("s", DataType::Int64),
                ("c", DataType::Int64),
            ]),
        )
        .unwrap();
        let got = rows(&out);
        assert_eq!(got.len(), 613);
        let mut naive: HashMap<i64, (i64, i64)> = HashMap::new();
        for (gg, vv) in gs.iter().zip(&vs) {
            let e = naive.entry(*gg).or_insert((0, 0));
            e.0 += vv;
            e.1 += 1;
        }
        for r in got {
            let ScalarValue::Int64(k) = r[0] else {
                panic!()
            };
            assert_eq!(r[1], ScalarValue::Int64(naive[&k].0));
            assert_eq!(r[2], ScalarValue::Int64(naive[&k].1));
        }
    }
}
