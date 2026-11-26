//! SLICE G. Physical execution of a LogicalPlan against a Catalog.
use crate::aggregate::hash_aggregate;
use crate::catalog::Catalog;
use crate::column::{Batch, Column};
use crate::eval::{eval, eval_predicate, optimize};
use crate::expr::Expr;
use crate::jit::{self, CompiledKernel};
use crate::join::hash_join;
use crate::memory::MemoryBudget;
use crate::plan::LogicalPlan;
use crate::sched::{run_morsels, split_morsels};
use crate::sort::sort_batches;
use crate::types::{DataType, EngineError, Field, Result, SchemaRef};
use std::path::PathBuf;
use std::sync::Arc;

pub struct ExecContext {
    pub threads: usize,
    pub morsel_rows: usize,
    pub budget: Arc<MemoryBudget>,
    pub spill_dir: PathBuf,
}

/// Run `plan` and return its output batches. Output order and batch boundaries depend only on the plan,
/// the data and `morsel_rows`, never on `threads`.
pub fn execute(plan: &LogicalPlan, catalog: &Catalog, ctx: &ExecContext) -> Result<Vec<Batch>> {
    match plan {
        LogicalPlan::Scan {
            table_name,
            schema,
            projection,
            projected_schema,
        } => scan(
            catalog,
            table_name,
            schema,
            projection.as_deref(),
            projected_schema,
        ),
        LogicalPlan::Filter { input, predicate } => {
            let input = execute(input, catalog, ctx)?;
            filter(input, predicate, ctx)
        }
        LogicalPlan::Projection {
            input,
            exprs,
            schema,
        } => {
            let input = execute(input, catalog, ctx)?;
            project(input, exprs, schema, ctx)
        }
        LogicalPlan::Join {
            left,
            right,
            join_type,
            on,
            schema,
        } => {
            let left = execute(left, catalog, ctx)?;
            let right = execute(right, catalog, ctx)?;
            hash_join(
                left,
                right,
                *join_type,
                on,
                schema.clone(),
                &ctx.budget,
                &ctx.spill_dir,
            )
        }
        LogicalPlan::Aggregate {
            input,
            group_exprs,
            agg_exprs,
            schema,
        } => {
            if schema.fields.len() != group_exprs.len() + agg_exprs.len() {
                return Err(EngineError::Plan(
                    "aggregate schema must be group exprs followed by aggregates".into(),
                ));
            }
            let input = execute(input, catalog, ctx)?;
            hash_aggregate(
                input,
                group_exprs,
                agg_exprs,
                schema.clone(),
                &ctx.budget,
                &ctx.spill_dir,
            )
        }
        LogicalPlan::Sort { input, order_by } => {
            let input = execute(input, catalog, ctx)?;
            sort_batches(
                input,
                plan.schema(),
                order_by,
                &ctx.budget,
                &ctx.spill_dir,
                ctx.morsel_rows.max(1),
            )
        }
        LogicalPlan::Limit { input, skip, fetch } => {
            let input = execute(input, catalog, ctx)?;
            Ok(limit(input, *skip, *fetch))
        }
    }
}

fn scan(
    catalog: &Catalog,
    table_name: &str,
    schema: &SchemaRef,
    projection: Option<&[usize]>,
    projected_schema: &SchemaRef,
) -> Result<Vec<Batch>> {
    let (table_schema, batches) = catalog.get(table_name)?;
    if table_schema != schema {
        return Err(EngineError::Plan(format!(
            "scan of '{table_name}': plan schema does not match catalog schema"
        )));
    }
    let expected: Vec<Field> = match projection {
        None => table_schema.fields.clone(),
        Some(idx) => {
            if let Some(&bad) = idx.iter().find(|&&i| i >= table_schema.fields.len()) {
                return Err(EngineError::Plan(format!(
                    "scan of '{table_name}': projection index {bad} out of range"
                )));
            }
            idx.iter()
                .map(|&i| table_schema.fields[i].clone())
                .collect()
        }
    };
    if projected_schema.fields != expected {
        return Err(EngineError::Plan(format!(
            "scan of '{table_name}': projected schema does not match projection"
        )));
    }
    let mut out = Vec::with_capacity(batches.len());
    for b in batches {
        let well_formed = b.columns.len() == table_schema.fields.len()
            && b.columns
                .iter()
                .zip(&table_schema.fields)
                .all(|(c, f)| c.data_type() == f.data_type);
        if !well_formed {
            return Err(EngineError::Type(format!(
                "table '{table_name}' has a batch that does not match its schema"
            )));
        }
        out.push(match projection {
            None => b.clone(),
            Some(idx) => b.project(idx, projected_schema.clone()),
        });
    }
    Ok(out)
}

/// Split every input batch into morsels of at most `morsel_rows` rows, in input order.
fn morsels_of(input: &[Batch], morsel_rows: usize) -> Vec<Batch> {
    input
        .iter()
        .flat_map(|b| split_morsels(b, morsel_rows))
        .collect()
}

/// Drop zero-row batches so an empty result is an empty vec, the same as an empty input.
fn non_empty(batches: Vec<Batch>) -> Vec<Batch> {
    batches.into_iter().filter(|b| b.num_rows > 0).collect()
}

fn filter(input: Vec<Batch>, predicate: &Expr, ctx: &ExecContext) -> Result<Vec<Batch>> {
    if predicate.return_type() != DataType::Boolean {
        return Err(EngineError::Type(format!(
            "filter predicate must be Boolean, found {:?}",
            predicate.return_type()
        )));
    }
    let predicate = optimize(predicate.clone());
    let morsels = morsels_of(&input, ctx.morsel_rows);
    let out = run_morsels(morsels, ctx.threads, |m| {
        let mask = eval_predicate(&predicate, m)?;
        Ok(m.filter(&mask))
    })?;
    Ok(non_empty(out))
}

fn project(
    input: Vec<Batch>,
    exprs: &[Expr],
    schema: &SchemaRef,
    ctx: &ExecContext,
) -> Result<Vec<Batch>> {
    if exprs.is_empty() {
        return Err(EngineError::Plan(
            "projection needs at least one expression".into(),
        ));
    }
    if schema.fields.len() != exprs.len() {
        return Err(EngineError::Plan(
            "projection schema and expression count differ".into(),
        ));
    }
    for (f, e) in schema.fields.iter().zip(exprs) {
        if f.data_type != e.return_type() {
            return Err(EngineError::Type(format!(
                "projection column '{}' declared {:?}, expression is {:?}",
                f.name,
                f.data_type,
                e.return_type()
            )));
        }
    }
    let exprs: Vec<Expr> = exprs.iter().cloned().map(optimize).collect();
    let total_rows: usize = input.iter().map(|b| b.num_rows).sum();
    let kernels: Vec<Option<CompiledKernel>> = exprs
        .iter()
        .map(|e| {
            if total_rows < JIT_MIN_ROWS || op_count(e) < JIT_MIN_OPS || !jit::supports(e) {
                return None;
            }
            jit::compile(e).ok()
        })
        .collect();
    let morsels = morsels_of(&input, ctx.morsel_rows);
    run_morsels(morsels, ctx.threads, |m| {
        let columns: Vec<Column> = exprs
            .iter()
            .zip(&kernels)
            .map(|(e, k)| match k {
                Some(k) => k.run(m),
                None => eval(e, m),
            })
            .collect::<Result<_>>()?;
        Batch::try_new(schema.clone(), columns)
    })
}

/// Break-even from RESULTS-H1A: below these, compiling costs more than the JIT saves.
const JIT_MIN_OPS: usize = 7;
const JIT_MIN_ROWS: usize = 200_000;

/// Number of Binary, Unary and Cast nodes in `e`.
fn op_count(e: &Expr) -> usize {
    match e {
        Expr::Column { .. } | Expr::Literal(_) => 0,
        Expr::Binary { left, right, .. } => 1 + op_count(left) + op_count(right),
        Expr::Unary { expr, .. } | Expr::Cast { expr, .. } => 1 + op_count(expr),
        Expr::InList { expr, list, .. } => {
            op_count(expr) + list.iter().map(op_count).sum::<usize>()
        }
        Expr::Case {
            condition,
            then_expr,
            else_expr,
            ..
        } => op_count(condition) + op_count(then_expr) + op_count(else_expr),
    }
}

fn limit(input: Vec<Batch>, skip: usize, fetch: Option<usize>) -> Vec<Batch> {
    let mut to_skip = skip;
    let mut remaining = fetch;
    let mut out = Vec::new();
    for b in input {
        if remaining == Some(0) {
            break;
        }
        let n = b.num_rows;
        if to_skip >= n {
            to_skip -= n;
            continue;
        }
        let start = to_skip;
        to_skip = 0;
        let avail = n - start;
        let take = remaining.map_or(avail, |r| r.min(avail));
        if let Some(r) = remaining.as_mut() {
            *r -= take;
        }
        out.push(if start == 0 && take == n {
            b
        } else {
            b.slice(start, take)
        });
    }
    out
}
