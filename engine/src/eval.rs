//! SLICE A. Expression typing + vectorized evaluation + constant folding.
//! Semantics: SQL three-valued logic; Int64/Float64 mixed arithmetic promotes to Float64;
//! integer division/mod by zero => EngineError::Arithmetic; comparisons on Utf8 are byte-lexicographic.
//! Division by zero only errors on rows where the divisor is non-NULL, so NULL rows and
//! CASE branches that are not selected for a row never raise it.
//! Float64 -> Int64 casts truncate (Rust `as`: saturating, NaN -> 0). Utf8 -> numeric/bool
//! casts return NULL on parse failure.
//! Design: operator-at-a-time vectorized interpreter, one output vector per computed node. No expression
//! fusion, SIMD intrinsics or codegen; this is the baseline the JIT is measured against.
use std::borrow::Cow;

use crate::column::{empty_col, schema_ref, Batch, Column, StringViews, Values};
use crate::expr::{BinaryOp, Expr, UnaryOp};
use crate::types::{DataType, EngineError, Result, ScalarValue, Schema};

/// Build `Expr::Column` by name (resolves index + type from `schema`).
pub fn col(schema: &Schema, name: &str) -> Result<Expr> {
    let index = schema.index_of(name)?;
    Ok(Expr::Column {
        name: name.to_string(),
        index,
        data_type: schema.fields[index].data_type,
    })
}

pub fn lit(v: ScalarValue) -> Expr {
    Expr::Literal(v)
}

/// Type-checked constructor: infers `return_type` or returns EngineError::Type.
pub fn binary(l: Expr, op: BinaryOp, r: Expr) -> Result<Expr> {
    let return_type = binary_type(l.return_type(), op, r.return_type())?;
    Ok(Expr::Binary {
        left: Box::new(l),
        op,
        right: Box::new(r),
        return_type,
    })
}

pub fn unary(op: UnaryOp, e: Expr) -> Result<Expr> {
    let return_type = unary_type(op, e.return_type())?;
    Ok(Expr::Unary {
        op,
        expr: Box::new(e),
        return_type,
    })
}

pub fn cast(e: Expr, to: DataType) -> Result<Expr> {
    let from = e.return_type();
    if !cast_supported(from, to) {
        return Err(type_err(format!("cannot cast {from:?} to {to:?}")));
    }
    Ok(Expr::Cast {
        expr: Box::new(e),
        target_type: to,
    })
}

/// `CASE WHEN c THEN t ELSE e END`. Numeric branches promote to Float64.
pub fn case(c: Expr, t: Expr, e: Expr) -> Result<Expr> {
    if c.return_type() != DataType::Boolean {
        return Err(type_err(format!(
            "CASE condition must be Boolean, got {:?}",
            c.return_type()
        )));
    }
    let return_type = common_type(t.return_type(), e.return_type()).ok_or_else(|| {
        type_err(format!(
            "CASE branches {:?} and {:?} do not unify",
            t.return_type(),
            e.return_type()
        ))
    })?;
    Ok(Expr::Case {
        condition: Box::new(c),
        then_expr: Box::new(coerce(t, return_type)),
        else_expr: Box::new(coerce(e, return_type)),
        return_type,
    })
}

/// Evaluate over every row of `batch`; result length == batch.num_rows.
pub fn eval(expr: &Expr, batch: &Batch) -> Result<Column> {
    eval_ref(expr, batch).map(Cow::into_owned)
}

/// Column references are borrowed from `batch`; every other node returns an owned vector.
/// A literal operand of an Int64/Float64 operator is applied as a scalar, not broadcast.
fn eval_ref<'a>(expr: &Expr, batch: &'a Batch) -> Result<Cow<'a, Column>> {
    let n = batch.num_rows;
    match expr {
        Expr::Column {
            name,
            index,
            data_type,
        } => {
            let c = batch
                .columns
                .get(*index)
                .ok_or_else(|| EngineError::ColumnNotFound(name.clone()))?;
            if c.data_type() != *data_type {
                return Err(type_err(format!(
                    "column '{name}' is {:?}, expected {data_type:?}",
                    c.data_type()
                )));
            }
            Ok(Cow::Borrowed(c))
        }
        Expr::Literal(v) => Ok(Cow::Owned(broadcast(v, n))),
        Expr::Binary {
            left,
            op,
            right,
            return_type,
        } => {
            let out = match (left.as_ref(), right.as_ref()) {
                (l, Expr::Literal(v)) if !matches!(l, Expr::Literal(_)) => {
                    let lc = eval_ref(l, batch)?;
                    match scalar_op(&lc, *op, v, false)? {
                        Some(out) => out,
                        None => binary_cols(&lc, *op, &broadcast(v, n))?,
                    }
                }
                (Expr::Literal(v), r) if !matches!(r, Expr::Literal(_)) => {
                    let rc = eval_ref(r, batch)?;
                    match scalar_op(&rc, *op, v, true)? {
                        Some(out) => out,
                        None => binary_cols(&broadcast(v, n), *op, &rc)?,
                    }
                }
                _ => {
                    let l = eval_ref(left, batch)?;
                    let r = eval_ref(right, batch)?;
                    binary_cols(&l, *op, &r)?
                }
            };
            expect_type(out, *return_type).map(Cow::Owned)
        }
        Expr::Unary {
            op,
            expr: inner,
            return_type,
        } => {
            let c = eval_ref(inner, batch)?;
            expect_type(unary_col(*op, &c)?, *return_type).map(Cow::Owned)
        }
        Expr::Cast {
            expr: inner,
            target_type,
        } => {
            let c = eval_ref(inner, batch)?;
            if c.data_type() == *target_type {
                return Ok(c);
            }
            cast_col(&c, *target_type).map(Cow::Owned)
        }
        Expr::InList {
            expr: inner,
            list,
            negated,
        } => {
            let x = eval_ref(inner, batch)?;
            // x IN (a, b) == (x = a) OR (x = b), with the same 3VL as the OR operator.
            let mut acc = Column::from_bool(vec![false; n]);
            for item in list {
                let y = eval_ref(item, batch)?;
                let eq = binary_cols(&x, BinaryOp::Eq, &y)?;
                acc = logic(BinaryOp::Or, &acc, &eq)?;
            }
            if *negated {
                acc = unary_col(UnaryOp::Not, &acc)?;
            }
            Ok(Cow::Owned(acc))
        }
        Expr::Case {
            condition,
            then_expr,
            else_expr,
            return_type,
        } => {
            let c = eval_ref(condition, batch)?;
            let cb = as_bools(&c)?;
            // Each branch runs only on the rows that select it, so a branch error on another row is never raised.
            let mut idx_t = Vec::new();
            let mut idx_e = Vec::new();
            for (i, &b) in cb.iter().enumerate() {
                if b && c.is_valid(i) {
                    idx_t.push(i);
                } else {
                    idx_e.push(i);
                }
            }
            let t = branch(then_expr, batch, &idx_t, *return_type)?;
            if idx_e.is_empty() {
                return Ok(Cow::Owned(t));
            }
            let e = branch(else_expr, batch, &idx_e, *return_type)?;
            if idx_t.is_empty() {
                return Ok(Cow::Owned(e));
            }
            let combined = Column::concat(&[&t, &e])?;
            let mut pos = vec![0usize; n];
            for (k, &i) in idx_t.iter().enumerate() {
                pos[i] = k;
            }
            for (k, &i) in idx_e.iter().enumerate() {
                pos[i] = idx_t.len() + k;
            }
            Ok(Cow::Owned(combined.take(&pos)))
        }
    }
}

/// Boolean predicate -> row mask; NULL counts as false (SQL WHERE semantics).
pub fn eval_predicate(expr: &Expr, batch: &Batch) -> Result<Vec<bool>> {
    let c = eval_ref(expr, batch)?;
    let b = as_bools(&c)?;
    Ok(b.iter()
        .enumerate()
        .map(|(i, &v)| v && c.is_valid(i))
        .collect())
}

/// Constant folding / algebraic simplification (x+0, x*1, true AND x, literal-only subtrees).
/// Never changes results: a literal-only subtree that errors is left as written, so the error still surfaces.
pub fn optimize(expr: Expr) -> Expr {
    let rebuilt = match expr {
        Expr::Binary {
            left,
            op,
            right,
            return_type,
        } => Expr::Binary {
            left: Box::new(optimize(*left)),
            op,
            right: Box::new(optimize(*right)),
            return_type,
        },
        Expr::Unary {
            op,
            expr,
            return_type,
        } => Expr::Unary {
            op,
            expr: Box::new(optimize(*expr)),
            return_type,
        },
        Expr::Cast { expr, target_type } => Expr::Cast {
            expr: Box::new(optimize(*expr)),
            target_type,
        },
        Expr::InList {
            expr,
            list,
            negated,
        } => Expr::InList {
            expr: Box::new(optimize(*expr)),
            list: list.into_iter().map(optimize).collect(),
            negated,
        },
        Expr::Case {
            condition,
            then_expr,
            else_expr,
            return_type,
        } => Expr::Case {
            condition: Box::new(optimize(*condition)),
            then_expr: Box::new(optimize(*then_expr)),
            else_expr: Box::new(optimize(*else_expr)),
            return_type,
        },
        leaf @ (Expr::Column { .. } | Expr::Literal(_)) => leaf,
    };
    fold(simplify(rebuilt))
}

// ---------------------------------------------------------------- typing

fn type_err(msg: impl Into<String>) -> EngineError {
    EngineError::Type(msg.into())
}

fn common_type(a: DataType, b: DataType) -> Option<DataType> {
    match (a, b) {
        _ if a == b => Some(a),
        (DataType::Int64, DataType::Float64) | (DataType::Float64, DataType::Int64) => {
            Some(DataType::Float64)
        }
        _ => None,
    }
}

fn binary_type(l: DataType, op: BinaryOp, r: DataType) -> Result<DataType> {
    match op {
        BinaryOp::Add | BinaryOp::Sub | BinaryOp::Mul | BinaryOp::Div | BinaryOp::Mod => {
            match common_type(l, r) {
                Some(dt @ (DataType::Int64 | DataType::Float64)) => Ok(dt),
                _ => Err(type_err(format!("arithmetic {op:?} on {l:?} and {r:?}"))),
            }
        }
        BinaryOp::Eq
        | BinaryOp::NotEq
        | BinaryOp::Lt
        | BinaryOp::LtEq
        | BinaryOp::Gt
        | BinaryOp::GtEq => common_type(l, r)
            .map(|_| DataType::Boolean)
            .ok_or_else(|| type_err(format!("cannot compare {l:?} with {r:?}"))),
        BinaryOp::And | BinaryOp::Or => {
            if l == DataType::Boolean && r == DataType::Boolean {
                Ok(DataType::Boolean)
            } else {
                Err(type_err(format!(
                    "{op:?} needs Boolean operands, got {l:?} and {r:?}"
                )))
            }
        }
    }
}

fn unary_type(op: UnaryOp, dt: DataType) -> Result<DataType> {
    match op {
        UnaryOp::Not if dt == DataType::Boolean => Ok(DataType::Boolean),
        UnaryOp::Neg if matches!(dt, DataType::Int64 | DataType::Float64) => Ok(dt),
        UnaryOp::IsNull | UnaryOp::IsNotNull => Ok(DataType::Boolean),
        _ => Err(type_err(format!("{op:?} not defined for {dt:?}"))),
    }
}

fn cast_supported(from: DataType, to: DataType) -> bool {
    from == to
        || matches!(
            (from, to),
            (DataType::Int64, DataType::Float64) | (DataType::Float64, DataType::Int64)
        )
        || to == DataType::Utf8
        || (from == DataType::Utf8
            && matches!(to, DataType::Int64 | DataType::Float64 | DataType::Boolean))
}

fn coerce(e: Expr, to: DataType) -> Expr {
    if e.return_type() == to {
        e
    } else {
        Expr::Cast {
            expr: Box::new(e),
            target_type: to,
        }
    }
}

// ---------------------------------------------------------------- columnar kernels

fn expect_type(c: Column, dt: DataType) -> Result<Column> {
    if c.data_type() == dt {
        Ok(c)
    } else {
        Err(type_err(format!(
            "expression produced {:?}, declared {dt:?}",
            c.data_type()
        )))
    }
}

fn branch(e: &Expr, batch: &Batch, idx: &[usize], dt: DataType) -> Result<Column> {
    if idx.is_empty() {
        return Ok(empty_col(dt));
    }
    expect_type(eval(e, &batch.take(idx))?, dt)
}

fn broadcast(v: &ScalarValue, n: usize) -> Column {
    match v {
        ScalarValue::Null(dt) => {
            let mut c = match dt {
                DataType::Boolean => Column::from_bool(vec![false; n]),
                DataType::Int64 => Column::from_i64(vec![0; n]),
                DataType::Float64 => Column::from_f64(vec![0.0; n]),
                DataType::Utf8 => {
                    let empty: Vec<&str> = vec![""; n];
                    Column::from_strs(empty.as_slice())
                }
            };
            c.validity = Some(vec![false; n]);
            c
        }
        ScalarValue::Boolean(b) => Column::from_bool(vec![*b; n]),
        ScalarValue::Int64(x) => Column::from_i64(vec![*x; n]),
        ScalarValue::Float64(x) => Column::from_f64(vec![*x; n]),
        ScalarValue::Utf8(s) => {
            let strs: Vec<&str> = vec![s.as_str(); n];
            Column::from_strs(strs.as_slice())
        }
    }
}

/// Drops the validity mask when every row is valid, matching `Column::from_scalars`.
fn finish(values: Values, validity: Option<Vec<bool>>) -> Column {
    let validity = validity.filter(|v| v.iter().any(|ok| !ok));
    Column { values, validity }
}

fn merge_validity(a: Option<&[bool]>, b: Option<&[bool]>) -> Option<Vec<bool>> {
    match (a, b) {
        (None, None) => None,
        (Some(x), None) => Some(x.to_vec()),
        (None, Some(y)) => Some(y.to_vec()),
        (Some(x), Some(y)) => Some(x.iter().zip(y).map(|(p, q)| *p && *q).collect()),
    }
}

fn valid_at(v: Option<&[bool]>, i: usize) -> bool {
    v.is_none_or(|v| v[i])
}

fn as_bools(c: &Column) -> Result<&[bool]> {
    match &c.values {
        Values::Bool(v) => Ok(v),
        _ => Err(type_err(format!(
            "expected Boolean, got {:?}",
            c.data_type()
        ))),
    }
}

fn as_i64(c: &Column) -> Result<&[i64]> {
    match &c.values {
        Values::Int64(v) => Ok(v),
        _ => Err(type_err(format!("expected Int64, got {:?}", c.data_type()))),
    }
}

/// Int64 is widened to Float64 here; the caller only asks for Float64 after promotion.
fn as_f64(c: &Column) -> Result<Cow<'_, [f64]>> {
    match &c.values {
        Values::Float64(v) => Ok(Cow::Borrowed(v.as_slice())),
        // shortcut: widening copies the column; a typed kernel per operand pair would avoid it.
        Values::Int64(v) => Ok(Cow::Owned(v.iter().map(|&x| x as f64).collect())),
        _ => Err(type_err(format!(
            "expected a numeric column, got {:?}",
            c.data_type()
        ))),
    }
}

fn utf8_strs(c: &Column) -> Result<Vec<&str>> {
    match &c.values {
        Values::Utf8(v) => Ok((0..v.len()).map(|i| v.get(i)).collect()),
        _ => Err(type_err(format!("expected Utf8, got {:?}", c.data_type()))),
    }
}

fn zip_map<A: Copy, B: Copy, O>(a: &[A], b: &[B], f: impl Fn(A, B) -> O) -> Vec<O> {
    a.iter().zip(b).map(|(&x, &y)| f(x, y)).collect()
}

fn binary_cols(l: &Column, op: BinaryOp, r: &Column) -> Result<Column> {
    let rt = binary_type(l.data_type(), op, r.data_type())?;
    match op {
        BinaryOp::And | BinaryOp::Or => logic(op, l, r),
        BinaryOp::Eq
        | BinaryOp::NotEq
        | BinaryOp::Lt
        | BinaryOp::LtEq
        | BinaryOp::Gt
        | BinaryOp::GtEq => compare(l, op, r),
        _ => arith(l, op, r, rt),
    }
}

fn arith(l: &Column, op: BinaryOp, r: &Column, dt: DataType) -> Result<Column> {
    let validity = merge_validity(l.validity.as_deref(), r.validity.as_deref());
    let values = match dt {
        DataType::Int64 => {
            Values::Int64(int_arith(op, as_i64(l)?, as_i64(r)?, validity.as_deref())?)
        }
        DataType::Float64 => Values::Float64(float_arith(op, &as_f64(l)?, &as_f64(r)?)?),
        _ => return Err(type_err(format!("arithmetic result type {dt:?}"))),
    };
    Ok(finish(values, validity))
}

fn int_arith(op: BinaryOp, a: &[i64], b: &[i64], validity: Option<&[bool]>) -> Result<Vec<i64>> {
    Ok(match op {
        BinaryOp::Add => zip_map(a, b, i64::wrapping_add),
        BinaryOp::Sub => zip_map(a, b, i64::wrapping_sub),
        BinaryOp::Mul => zip_map(a, b, i64::wrapping_mul),
        BinaryOp::Div | BinaryOp::Mod => {
            let mut out = Vec::with_capacity(a.len());
            for (i, (&x, &y)) in a.iter().zip(b).enumerate() {
                out.push(int_div_rem(op, x, y, valid_at(validity, i))?);
            }
            out
        }
        _ => return Err(type_err(format!("{op:?} is not arithmetic"))),
    })
}

/// `/` or `%` on one row. A zero divisor is an error only on a valid row; a NULL slot holds a placeholder.
fn int_div_rem(op: BinaryOp, x: i64, y: i64, valid: bool) -> Result<i64> {
    if y == 0 {
        if valid {
            let what = if op == BinaryOp::Div {
                "integer division"
            } else {
                "integer remainder"
            };
            return Err(EngineError::Arithmetic(format!("{what} by zero")));
        }
        return Ok(0);
    }
    Ok(if op == BinaryOp::Div {
        x.wrapping_div(y)
    } else {
        x.wrapping_rem(y)
    })
}

fn float_arith(op: BinaryOp, a: &[f64], b: &[f64]) -> Result<Vec<f64>> {
    Ok(match op {
        BinaryOp::Add => zip_map(a, b, |x, y| x + y),
        BinaryOp::Sub => zip_map(a, b, |x, y| x - y),
        BinaryOp::Mul => zip_map(a, b, |x, y| x * y),
        BinaryOp::Div => zip_map(a, b, |x, y| x / y),
        BinaryOp::Mod => zip_map(a, b, |x, y| x % y),
        _ => return Err(type_err(format!("{op:?} is not arithmetic"))),
    })
}

fn is_comparison(op: BinaryOp) -> bool {
    matches!(
        op,
        BinaryOp::Eq
            | BinaryOp::NotEq
            | BinaryOp::Lt
            | BinaryOp::LtEq
            | BinaryOp::Gt
            | BinaryOp::GtEq
    )
}

/// Applies `f` per row with the scalar on the side the literal sat on. The side test is loop-invariant.
fn scalar_map<A: Copy, T: Copy, O>(
    a: &[A],
    conv: impl Fn(A) -> T + Copy,
    s: T,
    lit_left: bool,
    f: impl Fn(T, T) -> O + Copy,
) -> Vec<O> {
    if lit_left {
        a.iter().map(|&x| f(s, conv(x))).collect()
    } else {
        a.iter().map(|&x| f(conv(x), s)).collect()
    }
}

fn int_scalar(
    op: BinaryOp,
    a: &[i64],
    s: i64,
    lit_left: bool,
    validity: Option<&[bool]>,
) -> Result<Vec<i64>> {
    let id = |x: i64| x;
    Ok(match op {
        BinaryOp::Add => scalar_map(a, id, s, lit_left, i64::wrapping_add),
        BinaryOp::Sub => scalar_map(a, id, s, lit_left, i64::wrapping_sub),
        BinaryOp::Mul => scalar_map(a, id, s, lit_left, i64::wrapping_mul),
        BinaryOp::Div | BinaryOp::Mod => {
            let mut out = Vec::with_capacity(a.len());
            for (i, &x) in a.iter().enumerate() {
                let (p, q) = if lit_left { (s, x) } else { (x, s) };
                out.push(int_div_rem(op, p, q, valid_at(validity, i))?);
            }
            out
        }
        _ => return Err(type_err(format!("{op:?} is not arithmetic"))),
    })
}

fn float_scalar<A: Copy>(
    op: BinaryOp,
    a: &[A],
    conv: impl Fn(A) -> f64 + Copy,
    s: f64,
    lit_left: bool,
) -> Result<Vec<f64>> {
    Ok(match op {
        BinaryOp::Add => scalar_map(a, conv, s, lit_left, |x, y| x + y),
        BinaryOp::Sub => scalar_map(a, conv, s, lit_left, |x, y| x - y),
        BinaryOp::Mul => scalar_map(a, conv, s, lit_left, |x, y| x * y),
        BinaryOp::Div => scalar_map(a, conv, s, lit_left, |x, y| x / y),
        BinaryOp::Mod => scalar_map(a, conv, s, lit_left, |x, y| x % y),
        _ => return Err(type_err(format!("{op:?} is not arithmetic"))),
    })
}

/// Float64 kernel for an Int64 or Float64 column against a scalar. `conv` widens the column side
/// the same way the promotion does, so no widened copy of the column is built.
fn float_operand<A: Copy>(
    op: BinaryOp,
    a: &[A],
    conv: impl Fn(A) -> f64 + Copy,
    s: f64,
    lit_left: bool,
) -> Result<Values> {
    if is_comparison(op) {
        Ok(Values::Bool(cmp_scalar(op, a, conv, s, lit_left)))
    } else {
        Ok(Values::Float64(float_scalar(op, a, conv, s, lit_left)?))
    }
}

fn cmp_scalar<A: Copy, T: PartialOrd + Copy>(
    op: BinaryOp,
    a: &[A],
    conv: impl Fn(A) -> T + Copy,
    s: T,
    lit_left: bool,
) -> Vec<bool> {
    match op {
        BinaryOp::Eq => scalar_map(a, conv, s, lit_left, |x, y| x == y),
        BinaryOp::NotEq => scalar_map(a, conv, s, lit_left, |x, y| x != y),
        BinaryOp::Lt => scalar_map(a, conv, s, lit_left, |x, y| x < y),
        BinaryOp::LtEq => scalar_map(a, conv, s, lit_left, |x, y| x <= y),
        BinaryOp::Gt => scalar_map(a, conv, s, lit_left, |x, y| x > y),
        BinaryOp::GtEq => scalar_map(a, conv, s, lit_left, |x, y| x >= y),
        _ => unreachable!("cmp_scalar is only called with comparison operators"),
    }
}

/// Column (op) scalar, or scalar (op) column when `lit_left`, for Int64 and Float64 operands.
/// Returns `None` for any other operand types so the caller broadcasts and keeps its type errors.
fn scalar_op(
    c: &Column,
    op: BinaryOp,
    lit: &ScalarValue,
    lit_left: bool,
) -> Result<Option<Column>> {
    if matches!(op, BinaryOp::And | BinaryOp::Or) {
        return Ok(None);
    }
    let values = match (&c.values, lit) {
        (Values::Int64(a), ScalarValue::Int64(s)) if is_comparison(op) => {
            Values::Bool(cmp_scalar(op, a, |x| x, *s, lit_left))
        }
        (Values::Int64(a), ScalarValue::Int64(s)) => {
            Values::Int64(int_scalar(op, a, *s, lit_left, c.validity.as_deref())?)
        }
        (Values::Int64(a), ScalarValue::Float64(s)) => {
            float_operand(op, a, |x| x as f64, *s, lit_left)?
        }
        (Values::Float64(a), ScalarValue::Int64(s)) => {
            float_operand(op, a, |x| x, *s as f64, lit_left)?
        }
        (Values::Float64(a), ScalarValue::Float64(s)) => float_operand(op, a, |x| x, *s, lit_left)?,
        _ => return Ok(None),
    };
    Ok(Some(finish(values, c.validity.clone())))
}

fn compare(l: &Column, op: BinaryOp, r: &Column) -> Result<Column> {
    let dt = common_type(l.data_type(), r.data_type()).ok_or_else(|| {
        type_err(format!(
            "cannot compare {:?} with {:?}",
            l.data_type(),
            r.data_type()
        ))
    })?;
    let values = match dt {
        DataType::Boolean => cmp_slices(op, as_bools(l)?, as_bools(r)?),
        DataType::Int64 => cmp_slices(op, as_i64(l)?, as_i64(r)?),
        DataType::Float64 => cmp_slices(op, &as_f64(l)?, &as_f64(r)?),
        DataType::Utf8 => cmp_slices(op, &utf8_strs(l)?, &utf8_strs(r)?),
    };
    let validity = merge_validity(l.validity.as_deref(), r.validity.as_deref());
    Ok(finish(Values::Bool(values), validity))
}

/// IEEE semantics for floats: any comparison involving NaN is false, except `!=`.
fn cmp_slices<T: PartialOrd + Copy>(op: BinaryOp, a: &[T], b: &[T]) -> Vec<bool> {
    match op {
        BinaryOp::Eq => zip_map(a, b, |x, y| x == y),
        BinaryOp::NotEq => zip_map(a, b, |x, y| x != y),
        BinaryOp::Lt => zip_map(a, b, |x, y| x < y),
        BinaryOp::LtEq => zip_map(a, b, |x, y| x <= y),
        BinaryOp::Gt => zip_map(a, b, |x, y| x > y),
        BinaryOp::GtEq => zip_map(a, b, |x, y| x >= y),
        _ => unreachable!("cmp_slices is only called with comparison operators"),
    }
}

/// Kleene three-valued AND / OR over Boolean columns.
fn logic(op: BinaryOp, l: &Column, r: &Column) -> Result<Column> {
    let (a, b) = (as_bools(l)?, as_bools(r)?);
    let (la, ra) = (l.validity.as_deref(), r.validity.as_deref());
    let mut vals = Vec::with_capacity(a.len());
    let mut valid = Vec::with_capacity(a.len());
    for (i, (&lb, &rb)) in a.iter().zip(b).enumerate() {
        let (lv, rv) = (valid_at(la, i), valid_at(ra, i));
        // A definite value on one side can decide the result even when the other side is NULL.
        let (v, ok) = match op {
            BinaryOp::And => {
                if (lv && !lb) || (rv && !rb) {
                    (false, true)
                } else if lv && rv {
                    (true, true)
                } else {
                    (false, false)
                }
            }
            BinaryOp::Or => {
                if (lv && lb) || (rv && rb) {
                    (true, true)
                } else if lv && rv {
                    (false, true)
                } else {
                    (false, false)
                }
            }
            _ => return Err(type_err(format!("{op:?} is not a logical operator"))),
        };
        vals.push(v);
        valid.push(ok);
    }
    Ok(finish(Values::Bool(vals), Some(valid)))
}

fn unary_col(op: UnaryOp, c: &Column) -> Result<Column> {
    let validity = c.validity.clone();
    match op {
        UnaryOp::Not => {
            let b = as_bools(c)?;
            Ok(finish(
                Values::Bool(b.iter().map(|&x| !x).collect()),
                validity,
            ))
        }
        UnaryOp::Neg => match &c.values {
            Values::Int64(v) => Ok(finish(
                Values::Int64(v.iter().map(|&x| x.wrapping_neg()).collect()),
                validity,
            )),
            Values::Float64(v) => Ok(finish(
                Values::Float64(v.iter().map(|&x| -x).collect()),
                validity,
            )),
            _ => Err(type_err(format!("Neg not defined for {:?}", c.data_type()))),
        },
        UnaryOp::IsNull => Ok(Column::from_bool(
            (0..c.len()).map(|i| !c.is_valid(i)).collect(),
        )),
        UnaryOp::IsNotNull => Ok(Column::from_bool(
            (0..c.len()).map(|i| c.is_valid(i)).collect(),
        )),
    }
}

fn scalar_text(c: &Column, i: usize) -> String {
    match &c.values {
        Values::Bool(v) => v[i].to_string(),
        Values::Int64(v) => v[i].to_string(),
        Values::Float64(v) => v[i].to_string(),
        Values::Utf8(v) => v.get(i).to_string(),
    }
}

/// Parses every valid Utf8 slot; a failed parse or a NULL input yields a NULL output slot.
fn parse_slots<T: Default>(
    c: &Column,
    parse: impl Fn(&str) -> Option<T>,
) -> Result<(Vec<T>, Vec<bool>)> {
    let strs = utf8_strs(c)?;
    let mut vals = Vec::with_capacity(strs.len());
    let mut ok = Vec::with_capacity(strs.len());
    for (i, s) in strs.iter().enumerate() {
        let parsed = if c.is_valid(i) { parse(s) } else { None };
        ok.push(parsed.is_some());
        vals.push(parsed.unwrap_or_default());
    }
    Ok((vals, ok))
}

fn parse_bool(s: &str) -> Option<bool> {
    let t = s.trim();
    if t.eq_ignore_ascii_case("true") {
        Some(true)
    } else if t.eq_ignore_ascii_case("false") {
        Some(false)
    } else {
        None
    }
}

fn cast_col(c: &Column, to: DataType) -> Result<Column> {
    let from = c.data_type();
    if from == to {
        return Ok(c.clone());
    }
    if !cast_supported(from, to) {
        return Err(type_err(format!("cannot cast {from:?} to {to:?}")));
    }
    let (values, validity) = match (&c.values, to) {
        (Values::Int64(v), DataType::Float64) => (
            Values::Float64(v.iter().map(|&x| x as f64).collect()),
            c.validity.clone(),
        ),
        (Values::Float64(v), DataType::Int64) => (
            Values::Int64(v.iter().map(|&x| x as i64).collect()),
            c.validity.clone(),
        ),
        (_, DataType::Utf8) => {
            let strs: Vec<String> = (0..c.len())
                .map(|i| {
                    if c.is_valid(i) {
                        scalar_text(c, i)
                    } else {
                        String::new()
                    }
                })
                .collect();
            (
                Values::Utf8(StringViews::from_strs(&strs)),
                c.validity.clone(),
            )
        }
        (Values::Utf8(_), DataType::Int64) => {
            let (v, ok) = parse_slots(c, |s| s.trim().parse::<i64>().ok())?;
            (Values::Int64(v), Some(ok))
        }
        (Values::Utf8(_), DataType::Float64) => {
            let (v, ok) = parse_slots(c, |s| s.trim().parse::<f64>().ok())?;
            (Values::Float64(v), Some(ok))
        }
        (Values::Utf8(_), DataType::Boolean) => {
            let (v, ok) = parse_slots(c, parse_bool)?;
            (Values::Bool(v), Some(ok))
        }
        _ => return Err(type_err(format!("cannot cast {from:?} to {to:?}"))),
    };
    Ok(finish(values, validity))
}

// ---------------------------------------------------------------- optimizer

fn is_constant(e: &Expr) -> bool {
    match e {
        Expr::Column { .. } => false,
        Expr::Literal(_) => true,
        Expr::Binary { left, right, .. } => is_constant(left) && is_constant(right),
        Expr::Unary { expr, .. } | Expr::Cast { expr, .. } => is_constant(expr),
        Expr::InList { expr, list, .. } => is_constant(expr) && list.iter().all(is_constant),
        Expr::Case {
            condition,
            then_expr,
            else_expr,
            ..
        } => is_constant(condition) && is_constant(then_expr) && is_constant(else_expr),
    }
}

/// Replaces a literal-only subtree with its value. Evaluates on a one-row batch with no columns.
/// shortcut: folding re-checks each subtree, O(n^2) in depth; fine for MVP expression sizes.
fn fold(expr: Expr) -> Expr {
    if matches!(expr, Expr::Literal(_)) || !is_constant(&expr) {
        return expr;
    }
    let one_row = Batch {
        schema: schema_ref(Schema::default()),
        columns: vec![],
        num_rows: 1,
    };
    match eval(&expr, &one_row) {
        Ok(c) if c.len() == 1 => Expr::Literal(c.get(0)),
        _ => expr,
    }
}

/// Algebraic identities that hold for every row including NULLs and NaN.
fn simplify(expr: Expr) -> Expr {
    match expr {
        Expr::Binary {
            left,
            op,
            right,
            return_type,
        } => match identity_side(&left, op, &right, return_type) {
            Some(true) => *left,
            Some(false) => *right,
            None => Expr::Binary {
                left,
                op,
                right,
                return_type,
            },
        },
        other => other,
    }
}

/// `Some(true)` keeps the left operand, `Some(false)` keeps the right one.
fn identity_side(l: &Expr, op: BinaryOp, r: &Expr, rt: DataType) -> Option<bool> {
    let is = |e: &Expr, v: ScalarValue| matches!(e, Expr::Literal(x) if *x == v);
    let same = |e: &Expr| e.return_type() == rt;
    match op {
        // true AND x == x and x AND true == x in 3VL (NULL stays NULL).
        BinaryOp::And if is(l, ScalarValue::Boolean(true)) && same(r) => Some(false),
        BinaryOp::And if is(r, ScalarValue::Boolean(true)) && same(l) => Some(true),
        // false OR x == x and x OR false == x in 3VL.
        BinaryOp::Or if is(l, ScalarValue::Boolean(false)) && same(r) => Some(false),
        BinaryOp::Or if is(r, ScalarValue::Boolean(false)) && same(l) => Some(true),
        // Int64 only: for floats, -0.0 + 0.0 is 0.0, which would change the value.
        BinaryOp::Add if rt == DataType::Int64 && is(r, ScalarValue::Int64(0)) && same(l) => {
            Some(true)
        }
        BinaryOp::Add if rt == DataType::Int64 && is(l, ScalarValue::Int64(0)) && same(r) => {
            Some(false)
        }
        // x * 1 is exact for every IEEE value, so the 1 is dropped as long as no promotion happens.
        BinaryOp::Mul if is(r, ScalarValue::Int64(1)) || is(r, ScalarValue::Float64(1.0)) => {
            same(l).then_some(true)
        }
        BinaryOp::Mul if is(l, ScalarValue::Int64(1)) || is(l, ScalarValue::Float64(1.0)) => {
            same(r).then_some(false)
        }
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::Field;
    use rand::rngs::StdRng;
    use rand::{Rng, SeedableRng};

    /// Builds a schema and batch whose fields are named after the given columns.
    fn setup(cols: Vec<(&str, Column)>) -> (Schema, Batch) {
        let fields = cols
            .iter()
            .map(|(n, c)| Field::new(n, c.data_type(), c.validity.is_some()))
            .collect();
        let schema = Schema::new(fields);
        let columns = cols.into_iter().map(|(_, c)| c).collect();
        let batch = Batch::try_new(schema_ref(schema.clone()), columns).unwrap();
        (schema, batch)
    }

    fn run(e: &Expr, b: &Batch) -> Vec<ScalarValue> {
        let c = eval(e, b).unwrap();
        (0..c.len()).map(|i| c.get(i)).collect()
    }

    fn i(v: i64) -> Expr {
        lit(ScalarValue::Int64(v))
    }
    fn f(v: f64) -> Expr {
        lit(ScalarValue::Float64(v))
    }
    fn s(v: &str) -> Expr {
        lit(ScalarValue::Utf8(v.to_string()))
    }
    fn bool_lit(v: bool) -> Expr {
        lit(ScalarValue::Boolean(v))
    }
    fn int(v: i64) -> ScalarValue {
        ScalarValue::Int64(v)
    }
    fn flt(v: f64) -> ScalarValue {
        ScalarValue::Float64(v)
    }
    fn tru() -> ScalarValue {
        ScalarValue::Boolean(true)
    }
    fn fal() -> ScalarValue {
        ScalarValue::Boolean(false)
    }
    fn nul(dt: DataType) -> ScalarValue {
        ScalarValue::Null(dt)
    }

    fn nullable_i64(v: Vec<i64>, valid: Vec<bool>) -> Column {
        Column::from_i64(v).with_validity(valid)
    }

    #[test]
    fn col_resolves_index_and_type_and_rejects_unknown_names() {
        let (schema, _) = setup(vec![
            ("a", Column::from_i64(vec![1])),
            ("b", Column::from_strs(&["x"])),
        ]);
        let e = col(&schema, "b").unwrap();
        assert_eq!(e.return_type(), DataType::Utf8);
        assert!(matches!(e, Expr::Column { index: 1, .. }));
        assert_eq!(
            col(&schema, "zz"),
            Err(EngineError::ColumnNotFound("zz".into()))
        );
    }

    #[test]
    fn int_add_infers_int64_and_evaluates() {
        let (schema, b) = setup(vec![("a", Column::from_i64(vec![1, 2, 3]))]);
        let e = binary(col(&schema, "a").unwrap(), BinaryOp::Add, i(10)).unwrap();
        assert_eq!(e.return_type(), DataType::Int64);
        assert_eq!(run(&e, &b), vec![int(11), int(12), int(13)]);
    }

    #[test]
    fn int_plus_float_promotes_to_float64() {
        let (schema, b) = setup(vec![
            ("a", Column::from_i64(vec![1, 2])),
            ("b", Column::from_f64(vec![0.5, 1.0])),
        ]);
        let e = binary(
            col(&schema, "a").unwrap(),
            BinaryOp::Add,
            col(&schema, "b").unwrap(),
        )
        .unwrap();
        assert_eq!(e.return_type(), DataType::Float64);
        assert_eq!(run(&e, &b), vec![flt(1.5), flt(3.0)]);
    }

    #[test]
    fn comparison_infers_boolean_and_promotes_int_against_float() {
        let (schema, b) = setup(vec![("a", Column::from_i64(vec![1, 2]))]);
        let e = binary(col(&schema, "a").unwrap(), BinaryOp::Eq, f(1.0)).unwrap();
        assert_eq!(e.return_type(), DataType::Boolean);
        assert_eq!(run(&e, &b), vec![tru(), fal()]);
    }

    #[test]
    fn arithmetic_on_utf8_is_type_error() {
        assert!(matches!(
            binary(s("a"), BinaryOp::Add, i(1)),
            Err(EngineError::Type(_))
        ));
    }

    #[test]
    fn and_on_int_is_type_error() {
        assert!(matches!(
            binary(i(1), BinaryOp::And, bool_lit(true)),
            Err(EngineError::Type(_))
        ));
    }

    #[test]
    fn comparing_int_with_utf8_is_type_error() {
        assert!(matches!(
            binary(i(1), BinaryOp::Lt, s("1")),
            Err(EngineError::Type(_))
        ));
    }

    #[test]
    fn int_add_wraps_on_overflow() {
        let (schema, b) = setup(vec![("a", Column::from_i64(vec![i64::MAX]))]);
        let e = binary(col(&schema, "a").unwrap(), BinaryOp::Add, i(1)).unwrap();
        assert_eq!(run(&e, &b), vec![int(i64::MIN)]);
    }

    #[test]
    fn int_division_by_zero_is_arithmetic_error() {
        let (schema, b) = setup(vec![
            ("a", Column::from_i64(vec![1, 2])),
            ("d", Column::from_i64(vec![1, 0])),
        ]);
        let e = binary(
            col(&schema, "a").unwrap(),
            BinaryOp::Div,
            col(&schema, "d").unwrap(),
        )
        .unwrap();
        assert!(matches!(eval(&e, &b), Err(EngineError::Arithmetic(_))));
    }

    #[test]
    fn int_remainder_by_zero_is_arithmetic_error() {
        let (schema, b) = setup(vec![("a", Column::from_i64(vec![7]))]);
        let e = binary(col(&schema, "a").unwrap(), BinaryOp::Mod, i(0)).unwrap();
        assert!(matches!(eval(&e, &b), Err(EngineError::Arithmetic(_))));
    }

    #[test]
    fn zero_divisor_on_null_row_is_null_not_error() {
        let (schema, b) = setup(vec![
            ("a", Column::from_i64(vec![5, 6])),
            ("d", nullable_i64(vec![0, 2], vec![false, true])),
        ]);
        let e = binary(
            col(&schema, "a").unwrap(),
            BinaryOp::Div,
            col(&schema, "d").unwrap(),
        )
        .unwrap();
        assert_eq!(run(&e, &b), vec![nul(DataType::Int64), int(3)]);
    }

    #[test]
    fn scalar_operands_match_broadcast_columns() {
        let a = nullable_i64(
            vec![5, -3, 0, 7, 2, 0],
            vec![true, true, false, true, true, true],
        );
        let fa = Column::from_f64(vec![1.5, -2.0, 0.0, 4.25, -0.5, 3.0]);
        let (schema, batch) = setup(vec![
            ("a", a),
            ("fa", fa),
            ("k3", Column::from_i64(vec![3; 6])),
            ("k0", Column::from_i64(vec![0; 6])),
            ("kf", Column::from_f64(vec![2.5; 6])),
        ]);
        let ops = [
            BinaryOp::Add,
            BinaryOp::Sub,
            BinaryOp::Mul,
            BinaryOp::Div,
            BinaryOp::Mod,
            BinaryOp::Eq,
            BinaryOp::NotEq,
            BinaryOp::Lt,
            BinaryOp::LtEq,
            BinaryOp::Gt,
            BinaryOp::GtEq,
        ];
        let shown = |c: &Column| -> Vec<String> {
            (0..c.len()).map(|i| format!("{:?}", c.get(i))).collect()
        };
        // The scalar path must match the same operator against a materialised constant column,
        // including error and NULL behaviour, on either side of the operator.
        for (lit_v, k) in [(int(3), "k3"), (int(0), "k0"), (flt(2.5), "kf")] {
            for name in ["a", "fa"] {
                for op in ops {
                    for lit_left in [false, true] {
                        let x = || col(&schema, name).unwrap();
                        let kk = || col(&schema, k).unwrap();
                        let (scalar, reference) = if lit_left {
                            (binary(lit(lit_v.clone()), op, x()), binary(kk(), op, x()))
                        } else {
                            (binary(x(), op, lit(lit_v.clone())), binary(x(), op, kk()))
                        };
                        let got = eval(&scalar.unwrap(), &batch).map(|c| shown(&c));
                        let want = eval(&reference.unwrap(), &batch).map(|c| shown(&c));
                        assert_eq!(got, want, "{name} {op:?} {lit_v:?} lit_left={lit_left}");
                    }
                }
            }
        }
    }

    #[test]
    fn scalar_zero_divisor_errors_only_on_valid_rows() {
        let (schema, batch) = setup(vec![("a", nullable_i64(vec![5, 6], vec![false, false]))]);
        let e = binary(col(&schema, "a").unwrap(), BinaryOp::Div, i(0)).unwrap();
        assert_eq!(run(&e, &batch), vec![nul(DataType::Int64); 2]);

        let (schema, batch) = setup(vec![("a", nullable_i64(vec![5, 6], vec![true, false]))]);
        let e = binary(col(&schema, "a").unwrap(), BinaryOp::Mod, i(0)).unwrap();
        assert!(eval(&e, &batch).is_err());
    }

    #[test]
    fn float_division_by_zero_follows_ieee() {
        let (schema, b) = setup(vec![("a", Column::from_f64(vec![1.0, 0.0, -1.0]))]);
        let e = binary(col(&schema, "a").unwrap(), BinaryOp::Div, f(0.0)).unwrap();
        let out = run(&e, &b);
        assert_eq!(out[0], flt(f64::INFINITY));
        assert!(matches!(out[1], ScalarValue::Float64(x) if x.is_nan()));
        assert_eq!(out[2], flt(f64::NEG_INFINITY));
    }

    #[test]
    fn int_min_divided_by_minus_one_wraps_without_panic() {
        let (schema, b) = setup(vec![("a", Column::from_i64(vec![i64::MIN]))]);
        let div = binary(col(&schema, "a").unwrap(), BinaryOp::Div, i(-1)).unwrap();
        let rem = binary(col(&schema, "a").unwrap(), BinaryOp::Mod, i(-1)).unwrap();
        assert_eq!(run(&div, &b), vec![int(i64::MIN)]);
        assert_eq!(run(&rem, &b), vec![int(0)]);
    }

    #[test]
    fn null_propagates_through_arithmetic_and_comparison() {
        let (schema, b) = setup(vec![("a", nullable_i64(vec![1, 2], vec![true, false]))]);
        let sum = binary(col(&schema, "a").unwrap(), BinaryOp::Add, i(1)).unwrap();
        let cmp = binary(col(&schema, "a").unwrap(), BinaryOp::Gt, i(0)).unwrap();
        assert_eq!(run(&sum, &b), vec![int(2), nul(DataType::Int64)]);
        assert_eq!(run(&cmp, &b), vec![tru(), nul(DataType::Boolean)]);
    }

    #[test]
    fn three_valued_and_matches_sql() {
        // Rows: NULL AND false, NULL AND true, true AND NULL, false AND NULL.
        let (schema, b) = setup(vec![
            (
                "x",
                Column::from_bool(vec![false, false, true, false])
                    .with_validity(vec![false, false, true, true]),
            ),
            (
                "y",
                Column::from_bool(vec![false, true, false, false])
                    .with_validity(vec![true, true, false, false]),
            ),
        ]);
        let e = binary(
            col(&schema, "x").unwrap(),
            BinaryOp::And,
            col(&schema, "y").unwrap(),
        )
        .unwrap();
        assert_eq!(
            run(&e, &b),
            vec![fal(), nul(DataType::Boolean), nul(DataType::Boolean), fal()]
        );
    }

    #[test]
    fn three_valued_or_matches_sql() {
        // Rows: NULL OR true, NULL OR false, true OR NULL, false OR NULL.
        let (schema, b) = setup(vec![
            (
                "x",
                Column::from_bool(vec![false, false, true, false])
                    .with_validity(vec![false, false, true, true]),
            ),
            (
                "y",
                Column::from_bool(vec![true, false, false, false])
                    .with_validity(vec![true, true, false, false]),
            ),
        ]);
        let e = binary(
            col(&schema, "x").unwrap(),
            BinaryOp::Or,
            col(&schema, "y").unwrap(),
        )
        .unwrap();
        assert_eq!(
            run(&e, &b),
            vec![tru(), nul(DataType::Boolean), tru(), nul(DataType::Boolean)]
        );
    }

    #[test]
    fn is_null_and_is_not_null_never_return_null() {
        let (schema, b) = setup(vec![("a", nullable_i64(vec![1, 2], vec![true, false]))]);
        let is_null = unary(UnaryOp::IsNull, col(&schema, "a").unwrap()).unwrap();
        let not_null = unary(UnaryOp::IsNotNull, col(&schema, "a").unwrap()).unwrap();
        let out = eval(&is_null, &b).unwrap();
        assert!(out.validity.is_none());
        assert_eq!(run(&is_null, &b), vec![fal(), tru()]);
        assert_eq!(run(&not_null, &b), vec![tru(), fal()]);
    }

    #[test]
    fn not_keeps_nulls_and_neg_wraps() {
        let (schema, b) = setup(vec![
            (
                "p",
                Column::from_bool(vec![true, false]).with_validity(vec![false, true]),
            ),
            ("n", Column::from_i64(vec![i64::MIN, 5])),
        ]);
        let not = unary(UnaryOp::Not, col(&schema, "p").unwrap()).unwrap();
        let neg = unary(UnaryOp::Neg, col(&schema, "n").unwrap()).unwrap();
        assert_eq!(run(&not, &b), vec![nul(DataType::Boolean), tru()]);
        assert_eq!(run(&neg, &b), vec![int(i64::MIN), int(-5)]);
    }

    #[test]
    fn utf8_comparison_is_byte_lexicographic() {
        let (schema, b) = setup(vec![(
            "s",
            Column::from_strs(&["Z", "a", "abc", "abd", "ab"]),
        )]);
        let lt = binary(col(&schema, "s").unwrap(), BinaryOp::Lt, s("abc")).unwrap();
        let eq = binary(col(&schema, "s").unwrap(), BinaryOp::Eq, s("abc")).unwrap();
        // "a" < "abc" is true: a shorter prefix sorts first.
        assert_eq!(run(&lt, &b), vec![tru(), tru(), fal(), fal(), tru()]);
        assert_eq!(run(&eq, &b), vec![fal(), fal(), tru(), fal(), fal()]);
    }

    #[test]
    fn utf8_comparison_handles_out_of_line_strings() {
        let long_a = "a very long string over twelve bytes";
        let long_b = "a very long string over twelve bytez";
        let (schema, b) = setup(vec![("s", Column::from_strs(&[long_a, long_b]))]);
        let eq = binary(col(&schema, "s").unwrap(), BinaryOp::Eq, s(long_a)).unwrap();
        let lt = binary(col(&schema, "s").unwrap(), BinaryOp::Lt, s(long_b)).unwrap();
        assert_eq!(run(&eq, &b), vec![tru(), fal()]);
        assert_eq!(run(&lt, &b), vec![tru(), fal()]);
    }

    #[test]
    fn case_takes_else_for_false_and_null_conditions() {
        let (schema, b) = setup(vec![
            (
                "c",
                Column::from_bool(vec![true, false, false]).with_validity(vec![true, false, true]),
            ),
            ("e", Column::from_i64(vec![1, 2, 3])),
        ]);
        let e = case(
            col(&schema, "c").unwrap(),
            i(10),
            col(&schema, "e").unwrap(),
        )
        .unwrap();
        assert_eq!(e.return_type(), DataType::Int64);
        assert_eq!(run(&e, &b), vec![int(10), int(2), int(3)]);
    }

    #[test]
    fn case_with_mixed_numeric_branches_promotes_to_float64() {
        let (schema, b) = setup(vec![
            ("c", Column::from_bool(vec![true, false])),
            ("e", Column::from_f64(vec![0.5, 0.25])),
        ]);
        let e = case(col(&schema, "c").unwrap(), i(1), col(&schema, "e").unwrap()).unwrap();
        assert_eq!(e.return_type(), DataType::Float64);
        assert_eq!(run(&e, &b), vec![flt(1.0), flt(0.25)]);
    }

    #[test]
    fn case_does_not_raise_division_error_on_unselected_rows() {
        let (schema, b) = setup(vec![("a", Column::from_i64(vec![0, 2]))]);
        let a = col(&schema, "a").unwrap();
        let cond = binary(a.clone(), BinaryOp::NotEq, i(0)).unwrap();
        let guarded = binary(i(10), BinaryOp::Div, a).unwrap();
        let e = case(cond, guarded, i(0)).unwrap();
        assert_eq!(run(&e, &b), vec![int(0), int(5)]);
    }

    #[test]
    fn case_rejects_non_boolean_condition_and_incompatible_branches() {
        assert!(matches!(case(i(1), i(1), i(2)), Err(EngineError::Type(_))));
        assert!(matches!(
            case(bool_lit(true), i(1), s("x")),
            Err(EngineError::Type(_))
        ));
    }

    #[test]
    fn in_list_follows_three_valued_logic() {
        let (schema, b) = setup(vec![(
            "x",
            nullable_i64(vec![1, 2, 3, 0], vec![true, true, true, false]),
        )]);
        let x = col(&schema, "x").unwrap();
        let list = vec![i(2), lit(nul(DataType::Int64))];
        let e = Expr::InList {
            expr: Box::new(x.clone()),
            list: list.clone(),
            negated: false,
        };
        let neg = Expr::InList {
            expr: Box::new(x.clone()),
            list,
            negated: true,
        };
        // 1 IN (2, NULL) is NULL; 2 IN (2, NULL) is TRUE; NULL IN (...) is NULL.
        assert_eq!(
            run(&e, &b),
            vec![
                nul(DataType::Boolean),
                tru(),
                nul(DataType::Boolean),
                nul(DataType::Boolean)
            ]
        );
        assert_eq!(
            run(&neg, &b),
            vec![
                nul(DataType::Boolean),
                fal(),
                nul(DataType::Boolean),
                nul(DataType::Boolean)
            ]
        );
    }

    #[test]
    fn in_list_without_null_items_is_two_valued() {
        let (schema, b) = setup(vec![("x", Column::from_i64(vec![1, 2, 3]))]);
        let x = col(&schema, "x").unwrap();
        let e = Expr::InList {
            expr: Box::new(x.clone()),
            list: vec![i(2), i(3)],
            negated: false,
        };
        let neg = Expr::InList {
            expr: Box::new(x),
            list: vec![i(2), i(3)],
            negated: true,
        };
        assert_eq!(run(&e, &b), vec![fal(), tru(), tru()]);
        assert_eq!(run(&neg, &b), vec![tru(), fal(), fal()]);
    }

    #[test]
    fn cast_int_and_float_both_ways() {
        let (schema, b) = setup(vec![
            ("i", Column::from_i64(vec![1, -2])),
            ("f", Column::from_f64(vec![2.7, -2.7])),
        ]);
        let to_f = cast(col(&schema, "i").unwrap(), DataType::Float64).unwrap();
        let to_i = cast(col(&schema, "f").unwrap(), DataType::Int64).unwrap();
        assert_eq!(run(&to_f, &b), vec![flt(1.0), flt(-2.0)]);
        assert_eq!(run(&to_i, &b), vec![int(2), int(-2)]);
    }

    #[test]
    fn cast_utf8_to_numeric_yields_null_on_parse_failure() {
        let (schema, b) = setup(vec![("s", Column::from_strs(&["12", " 7 ", "x", ""]))]);
        let e = cast(col(&schema, "s").unwrap(), DataType::Int64).unwrap();
        assert_eq!(
            run(&e, &b),
            vec![int(12), int(7), nul(DataType::Int64), nul(DataType::Int64)]
        );
    }

    #[test]
    fn cast_utf8_to_boolean_parses_true_and_false_only() {
        let (schema, b) = setup(vec![("s", Column::from_strs(&["TRUE", "false", "yes"]))]);
        let e = cast(col(&schema, "s").unwrap(), DataType::Boolean).unwrap();
        assert_eq!(run(&e, &b), vec![tru(), fal(), nul(DataType::Boolean)]);
    }

    #[test]
    fn cast_to_utf8_formats_each_type() {
        let (schema, b) = setup(vec![
            ("b", Column::from_bool(vec![true, false])),
            (
                "i",
                Column::from_i64(vec![42, 0]).with_validity(vec![true, false]),
            ),
            ("f", Column::from_f64(vec![1.5, 2.0])),
        ]);
        let bs = cast(col(&schema, "b").unwrap(), DataType::Utf8).unwrap();
        let is = cast(col(&schema, "i").unwrap(), DataType::Utf8).unwrap();
        let fs = cast(col(&schema, "f").unwrap(), DataType::Utf8).unwrap();
        assert_eq!(
            run(&bs, &b),
            vec![
                ScalarValue::Utf8("true".into()),
                ScalarValue::Utf8("false".into())
            ]
        );
        assert_eq!(
            run(&is, &b),
            vec![ScalarValue::Utf8("42".into()), nul(DataType::Utf8)]
        );
        assert_eq!(
            run(&fs, &b),
            vec![
                ScalarValue::Utf8("1.5".into()),
                ScalarValue::Utf8("2".into())
            ]
        );
    }

    #[test]
    fn cast_boolean_to_int_is_type_error() {
        assert!(matches!(
            cast(bool_lit(true), DataType::Int64),
            Err(EngineError::Type(_))
        ));
    }

    #[test]
    fn eval_predicate_treats_null_as_false() {
        let (schema, b) = setup(vec![(
            "p",
            Column::from_bool(vec![true, true, false]).with_validity(vec![true, false, true]),
        )]);
        let mask = eval_predicate(&col(&schema, "p").unwrap(), &b).unwrap();
        assert_eq!(mask, vec![true, false, false]);
    }

    #[test]
    fn eval_predicate_rejects_non_boolean() {
        let (schema, b) = setup(vec![("a", Column::from_i64(vec![1]))]);
        let r = eval_predicate(&col(&schema, "a").unwrap(), &b);
        assert!(matches!(r, Err(EngineError::Type(_))));
    }

    #[test]
    fn literals_broadcast_to_batch_row_count() {
        let (_, b) = setup(vec![("a", Column::from_i64(vec![0, 0, 0, 0]))]);
        assert_eq!(run(&i(7), &b), vec![int(7); 4]);
        assert_eq!(
            run(&lit(nul(DataType::Utf8)), &b),
            vec![nul(DataType::Utf8); 4]
        );
    }

    #[test]
    fn optimize_folds_literal_only_subtree() {
        let e = binary(
            i(2),
            BinaryOp::Mul,
            binary(i(3), BinaryOp::Add, i(4)).unwrap(),
        )
        .unwrap();
        assert_eq!(optimize(e), i(14));
    }

    #[test]
    fn optimize_keeps_constant_that_errors_so_error_still_surfaces() {
        let e = binary(i(1), BinaryOp::Div, i(0)).unwrap();
        let opt = optimize(e.clone());
        assert_eq!(opt, e);
        let (_, b) = setup(vec![("a", Column::from_i64(vec![1]))]);
        assert!(matches!(eval(&opt, &b), Err(EngineError::Arithmetic(_))));
    }

    #[test]
    fn optimize_drops_int_add_zero_and_mul_one_identities() {
        let (schema, _) = setup(vec![("a", Column::from_i64(vec![1]))]);
        let a = col(&schema, "a").unwrap();
        assert_eq!(optimize(binary(a.clone(), BinaryOp::Add, i(0)).unwrap()), a);
        assert_eq!(optimize(binary(i(0), BinaryOp::Add, a.clone()).unwrap()), a);
        assert_eq!(optimize(binary(a.clone(), BinaryOp::Mul, i(1)).unwrap()), a);
    }

    #[test]
    fn optimize_keeps_float_add_zero_because_negative_zero_changes() {
        let (schema, _) = setup(vec![("f", Column::from_f64(vec![-0.0]))]);
        let f_col = col(&schema, "f").unwrap();
        let e = binary(f_col.clone(), BinaryOp::Add, f(0.0)).unwrap();
        assert_eq!(optimize(e.clone()), e);
    }

    #[test]
    fn optimize_simplifies_boolean_identities() {
        let (schema, _) = setup(vec![("p", Column::from_bool(vec![true]))]);
        let p = col(&schema, "p").unwrap();
        let and_true = binary(bool_lit(true), BinaryOp::And, p.clone()).unwrap();
        let or_false = binary(bool_lit(false), BinaryOp::Or, p.clone()).unwrap();
        assert_eq!(optimize(and_true), p);
        assert_eq!(optimize(or_false), p);
    }

    /// Random typed expression generator for the equivalence property.
    struct Gen<'a> {
        rng: &'a mut StdRng,
        schema: &'a Schema,
    }

    const ARITH: [BinaryOp; 5] = [
        BinaryOp::Add,
        BinaryOp::Sub,
        BinaryOp::Mul,
        BinaryOp::Div,
        BinaryOp::Mod,
    ];
    const CMP: [BinaryOp; 6] = [
        BinaryOp::Eq,
        BinaryOp::NotEq,
        BinaryOp::Lt,
        BinaryOp::LtEq,
        BinaryOp::Gt,
        BinaryOp::GtEq,
    ];
    const INT_POOL: [i64; 7] = [0, 1, -1, 2, 7, i64::MAX, i64::MIN];
    const FLOAT_POOL: [f64; 8] = [
        0.0,
        -0.0,
        1.0,
        2.5,
        -3.0,
        f64::NAN,
        f64::INFINITY,
        f64::NEG_INFINITY,
    ];
    const STR_POOL: [&str; 5] = ["", "a", "abc", "zz", "a very long string over twelve bytes"];

    impl Gen<'_> {
        fn pick<T: Copy>(&mut self, pool: &[T]) -> T {
            pool[self.rng.gen_range(0..pool.len())]
        }

        fn leaf(&mut self, dt: DataType) -> Expr {
            if self.rng.gen_bool(0.15) {
                return lit(nul(dt));
            }
            let schema = self.schema;
            let names: Vec<&str> = schema
                .fields
                .iter()
                .filter(|f| f.data_type == dt)
                .map(|f| f.name.as_str())
                .collect();
            if !names.is_empty() && self.rng.gen_bool(0.6) {
                let name = self.pick(&names);
                return col(schema, name).unwrap();
            }
            match dt {
                DataType::Int64 => i(self.pick(&INT_POOL)),
                DataType::Float64 => f(self.pick(&FLOAT_POOL)),
                DataType::Boolean => bool_lit(self.rng.gen_bool(0.5)),
                DataType::Utf8 => s(self.pick(&STR_POOL)),
            }
        }

        fn expr(&mut self, dt: DataType, depth: u32) -> Expr {
            if depth == 0 || self.rng.gen_bool(0.3) {
                return self.leaf(dt);
            }
            let d = depth - 1;
            match dt {
                DataType::Int64 => match self.rng.gen_range(0..4) {
                    0 => {
                        let op = self.pick(&ARITH);
                        let (l, r) = (self.expr(DataType::Int64, d), self.expr(DataType::Int64, d));
                        binary(l, op, r).unwrap()
                    }
                    1 => {
                        let x = self.expr(DataType::Int64, d);
                        unary(UnaryOp::Neg, x).unwrap()
                    }
                    2 => {
                        let c = self.expr(DataType::Boolean, d);
                        let (t, e) = (self.expr(DataType::Int64, d), self.expr(DataType::Int64, d));
                        case(c, t, e).unwrap()
                    }
                    _ => {
                        let src = if self.rng.gen_bool(0.5) {
                            DataType::Float64
                        } else {
                            DataType::Utf8
                        };
                        let x = self.expr(src, d);
                        cast(x, DataType::Int64).unwrap()
                    }
                },
                DataType::Float64 => match self.rng.gen_range(0..4) {
                    0 => {
                        let op = self.pick(&ARITH);
                        let l = self.expr(DataType::Float64, d);
                        let r_type = if self.rng.gen_bool(0.5) {
                            DataType::Int64
                        } else {
                            DataType::Float64
                        };
                        let r = self.expr(r_type, d);
                        binary(l, op, r).unwrap()
                    }
                    1 => {
                        let x = self.expr(DataType::Float64, d);
                        unary(UnaryOp::Neg, x).unwrap()
                    }
                    2 => {
                        let c = self.expr(DataType::Boolean, d);
                        let (t, e) = (
                            self.expr(DataType::Float64, d),
                            self.expr(DataType::Float64, d),
                        );
                        case(c, t, e).unwrap()
                    }
                    _ => {
                        let x = self.expr(DataType::Int64, d);
                        cast(x, DataType::Float64).unwrap()
                    }
                },
                DataType::Boolean => match self.rng.gen_range(0..6) {
                    0 => {
                        let op = if self.rng.gen_bool(0.5) {
                            BinaryOp::And
                        } else {
                            BinaryOp::Or
                        };
                        let (l, r) = (
                            self.expr(DataType::Boolean, d),
                            self.expr(DataType::Boolean, d),
                        );
                        binary(l, op, r).unwrap()
                    }
                    1 => {
                        let x = self.expr(DataType::Boolean, d);
                        unary(UnaryOp::Not, x).unwrap()
                    }
                    2 => {
                        let op = self.pick(&CMP);
                        let (l, r) = (self.expr(DataType::Int64, d), self.expr(DataType::Int64, d));
                        binary(l, op, r).unwrap()
                    }
                    3 => {
                        let op = self.pick(&CMP);
                        let (l, r) = (self.expr(DataType::Utf8, d), self.expr(DataType::Utf8, d));
                        binary(l, op, r).unwrap()
                    }
                    4 => {
                        let any = [
                            DataType::Int64,
                            DataType::Float64,
                            DataType::Boolean,
                            DataType::Utf8,
                        ];
                        let dt = self.pick(&any);
                        let x = self.expr(dt, d);
                        let op = if self.rng.gen_bool(0.5) {
                            UnaryOp::IsNull
                        } else {
                            UnaryOp::IsNotNull
                        };
                        unary(op, x).unwrap()
                    }
                    _ => {
                        let x = self.expr(DataType::Int64, d);
                        let n = self.rng.gen_range(0..4);
                        let list = (0..n).map(|_| self.expr(DataType::Int64, d)).collect();
                        Expr::InList {
                            expr: Box::new(x),
                            list,
                            negated: self.rng.gen_bool(0.5),
                        }
                    }
                },
                DataType::Utf8 => match self.rng.gen_range(0..3) {
                    0 => {
                        let c = self.expr(DataType::Boolean, d);
                        let (t, e) = (self.expr(DataType::Utf8, d), self.expr(DataType::Utf8, d));
                        case(c, t, e).unwrap()
                    }
                    1 => {
                        let x = self.expr(DataType::Int64, d);
                        cast(x, DataType::Utf8).unwrap()
                    }
                    _ => {
                        let x = self.expr(DataType::Boolean, d);
                        cast(x, DataType::Utf8).unwrap()
                    }
                },
            }
        }
    }

    fn random_batch(rng: &mut StdRng, n: usize) -> (Schema, Batch) {
        let validity =
            |rng: &mut StdRng| -> Vec<bool> { (0..n).map(|_| !rng.gen_bool(0.15)).collect() };
        let i0: Vec<i64> = (0..n)
            .map(|_| {
                if rng.gen_bool(0.05) {
                    i64::MAX
                } else {
                    rng.gen_range(-3..4)
                }
            })
            .collect();
        let i1: Vec<i64> = (0..n).map(|_| rng.gen_range(-2..3)).collect();
        let f0: Vec<f64> = (0..n)
            .map(|_| {
                if rng.gen_bool(0.1) {
                    FLOAT_POOL[rng.gen_range(0..FLOAT_POOL.len())]
                } else {
                    rng.gen_range(-3.0..3.0)
                }
            })
            .collect();
        let b0: Vec<bool> = (0..n).map(|_| rng.gen_bool(0.5)).collect();
        let s0: Vec<&str> = (0..n)
            .map(|_| STR_POOL[rng.gen_range(0..STR_POOL.len())])
            .collect();
        let v = (
            validity(rng),
            validity(rng),
            validity(rng),
            validity(rng),
            validity(rng),
        );
        setup(vec![
            ("i0", Column::from_i64(i0).with_validity(v.0)),
            ("i1", Column::from_i64(i1).with_validity(v.1)),
            ("f0", Column::from_f64(f0).with_validity(v.2)),
            ("b0", Column::from_bool(b0).with_validity(v.3)),
            ("s0", Column::from_strs(&s0).with_validity(v.4)),
        ])
    }

    fn same_result(a: &Result<Column>, b: &Result<Column>) -> bool {
        match (a, b) {
            (Err(x), Err(y)) => x == y,
            (Ok(x), Ok(y)) => {
                x.data_type() == y.data_type()
                    && x.len() == y.len()
                    && (0..x.len()).all(|i| match (x.get(i), y.get(i)) {
                        (ScalarValue::Float64(p), ScalarValue::Float64(q)) => {
                            p.to_bits() == q.to_bits()
                        }
                        (p, q) => p == q,
                    })
            }
            _ => false,
        }
    }

    #[test]
    fn optimize_preserves_results_on_random_expressions_and_data() {
        let mut rng = StdRng::seed_from_u64(0x5EED_A11CE);
        let mut ok_count = 0usize;
        for _ in 0..40 {
            let (schema, batch) = random_batch(&mut rng, 48);
            for _ in 0..100 {
                let dt = [
                    DataType::Int64,
                    DataType::Float64,
                    DataType::Boolean,
                    DataType::Utf8,
                ][rng.gen_range(0..4)];
                let expr = Gen {
                    rng: &mut rng,
                    schema: &schema,
                }
                .expr(dt, 4);
                let optimized = optimize(expr.clone());
                assert_eq!(
                    optimized.return_type(),
                    expr.return_type(),
                    "optimize changed type of {expr:?}"
                );
                let before = eval(&expr, &batch);
                let after = eval(&optimized, &batch);
                assert!(
                    same_result(&before, &after),
                    "optimize changed result\nexpr: {expr:?}\noptimized: {optimized:?}\nbefore: {before:?}\nafter: {after:?}"
                );
                if before.is_ok() {
                    ok_count += 1;
                }
            }
        }
        // Guard against a generator that only produces erroring expressions.
        assert!(
            ok_count > 500,
            "only {ok_count} of 4000 random expressions evaluated successfully"
        );
    }
}
