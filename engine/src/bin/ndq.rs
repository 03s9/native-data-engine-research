//! `ndq`: command-line front end for the native engine MVP.
//!
//! Subcommands: `demo` (in-memory analytics queries), `csv` (filter and project a CSV file),
//! `bench` (laptop micro-benchmarks). Argument parsing and timing are hand-rolled on purpose:
//! no new dependencies.

use std::hint::black_box;
use std::io::{self, BufWriter, Write};
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Instant;

use native_engine::catalog::{load_csv, Catalog};
use native_engine::column::{schema_ref, Batch, Column, Values};
use native_engine::cost::{crossover, CostParams};
use native_engine::eval::{binary, col, eval, lit};
use native_engine::exec::{execute, ExecContext};
use native_engine::expr::{BinaryOp, Expr};
use native_engine::memory::MemoryBudget;
use native_engine::plan::{AggExpr, AggFunc, JoinType, LogicalPlan, SortField};
use native_engine::sched::{run_morsels, split_morsels};
use native_engine::simd::{gt_i64, gt_i64_scalar};
use native_engine::types::{DataType, EngineError, Field, Result, ScalarValue, Schema, SchemaRef};

const USAGE: &str = "\
ndq: command-line front end for the native engine MVP

USAGE:
    ndq demo
    ndq csv <file.csv> [--where <col> <op> <literal>] [--select a,b,c] [--limit N]
    ndq bench
    ndq --help

SUBCOMMANDS:
    demo    Generate in-memory orders (1M rows) and customers (10k rows), run three queries.
    csv     Load a CSV file (header row, types inferred), filter, project, print all rows
            (or the first N with --limit). <op> is one of: =  !=  <  <=  >  >=
    bench   Micro-benchmarks: SIMD filter kernel, vectorized eval vs hand loop, morsel scaling.

EXIT CODES:
    0  success
    1  runtime error (I/O, plan or execution failure)
    2  usage error";

/// Usage errors exit 2; runtime errors exit 1. `PipeClosed` (stdout reader went away, e.g. `| head`) exits 0.
enum Failure {
    Usage(String),
    Run(String),
    PipeClosed,
}

impl From<EngineError> for Failure {
    fn from(e: EngineError) -> Self {
        Failure::Run(e.to_string())
    }
}

impl From<io::Error> for Failure {
    fn from(e: io::Error) -> Self {
        if e.kind() == io::ErrorKind::BrokenPipe {
            Failure::PipeClosed
        } else {
            Failure::Run(e.to_string())
        }
    }
}

type CliResult = std::result::Result<(), Failure>;

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let code = match dispatch(&args) {
        Ok(()) | Err(Failure::PipeClosed) => 0,
        Err(Failure::Usage(msg)) => {
            eprintln!("ndq: {msg}\nRun 'ndq --help' for usage.");
            2
        }
        Err(Failure::Run(msg)) => {
            eprintln!("ndq: error: {msg}");
            1
        }
    };
    std::process::exit(code);
}

fn dispatch(args: &[String]) -> CliResult {
    match args.first().map(String::as_str) {
        None => Err(Failure::Usage("missing subcommand".into())),
        Some("-h" | "--help" | "help") => {
            println!("{USAGE}");
            Ok(())
        }
        Some("demo") => {
            no_extra_args("demo", &args[1..])?;
            demo()
        }
        Some("csv") => csv(&args[1..]),
        Some("bench") => {
            no_extra_args("bench", &args[1..])?;
            bench()
        }
        Some(other) => Err(Failure::Usage(format!("unknown subcommand '{other}'"))),
    }
}

fn no_extra_args(name: &str, rest: &[String]) -> CliResult {
    match rest.first() {
        None => Ok(()),
        Some(extra) => Err(Failure::Usage(format!(
            "'{name}' takes no arguments, got '{extra}'"
        ))),
    }
}

fn ms_since(t: Instant) -> f64 {
    t.elapsed().as_secs_f64() * 1e3
}

fn default_threads() -> usize {
    std::thread::available_parallelism().map_or(1, std::num::NonZeroUsize::get)
}

fn exec_context() -> std::result::Result<ExecContext, Failure> {
    let spill_dir = std::env::temp_dir().join("ndq-spill");
    std::fs::create_dir_all(&spill_dir).map_err(|e| {
        Failure::Run(format!(
            "cannot create spill dir {}: {e}",
            spill_dir.display()
        ))
    })?;
    Ok(ExecContext {
        threads: default_threads(),
        morsel_rows: CHUNK_ROWS,
        budget: MemoryBudget::new(4 << 30),
        spill_dir,
    })
}

/// Deterministic LCG (Knuth MMIX constants). Same seed, same data, on every run.
struct Lcg(u64);

impl Lcg {
    fn next_u64(&mut self) -> u64 {
        self.0 = self
            .0
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        self.0 >> 33
    }

    /// Uniform-ish value in `0..n`.
    fn below(&mut self, n: u64) -> i64 {
        (self.next_u64() % n) as i64
    }
}

// ---------------------------------------------------------------- demo

const ORDER_ROWS: usize = 1_000_000;
const CUSTOMER_ROWS: usize = 10_000;
const CHUNK_ROWS: usize = 65_536;
const PRINT_ROWS: usize = 10;
const REGIONS: [&str; 4] = ["north", "south", "east", "west"];

fn orders_schema() -> SchemaRef {
    schema_ref(Schema::new(vec![
        Field::new("id", DataType::Int64, false),
        Field::new("cust", DataType::Int64, false),
        Field::new("amount", DataType::Int64, false),
        Field::new("region", DataType::Utf8, false),
    ]))
}

fn customers_schema() -> SchemaRef {
    schema_ref(Schema::new(vec![
        Field::new("cust", DataType::Int64, false),
        Field::new("name", DataType::Utf8, false),
    ]))
}

fn build_orders(rng: &mut Lcg, schema: &SchemaRef) -> Result<Vec<Batch>> {
    let mut batches = Vec::new();
    let mut start = 0;
    while start < ORDER_ROWS {
        let n = CHUNK_ROWS.min(ORDER_ROWS - start);
        let mut ids = Vec::with_capacity(n);
        let mut custs = Vec::with_capacity(n);
        let mut amounts = Vec::with_capacity(n);
        let mut regions: Vec<&str> = Vec::with_capacity(n);
        for i in 0..n {
            ids.push((start + i) as i64);
            custs.push(rng.below(CUSTOMER_ROWS as u64));
            amounts.push(rng.below(1000) + 1);
            regions.push(REGIONS[rng.below(REGIONS.len() as u64) as usize]);
        }
        batches.push(Batch::try_new(
            schema.clone(),
            vec![
                Column::from_i64(ids),
                Column::from_i64(custs),
                Column::from_i64(amounts),
                Column::from_strs(&regions),
            ],
        )?);
        start += n;
    }
    Ok(batches)
}

fn build_customers(schema: &SchemaRef) -> Result<Batch> {
    let ids: Vec<i64> = (0..CUSTOMER_ROWS as i64).collect();
    let names: Vec<String> = ids.iter().map(|i| format!("customer_{i:05}")).collect();
    Batch::try_new(
        schema.clone(),
        vec![Column::from_i64(ids), Column::from_strs(&names)],
    )
}

fn scan(table: &str, schema: &SchemaRef) -> Arc<LogicalPlan> {
    Arc::new(LogicalPlan::Scan {
        table_name: table.to_string(),
        schema: schema.clone(),
        projection: None,
        projected_schema: schema.clone(),
    })
}

fn agg(func: AggFunc, expr: Option<Expr>, output_type: DataType) -> AggExpr {
    AggExpr {
        func,
        expr,
        output_type,
    }
}

fn demo() -> CliResult {
    let mut rng = Lcg(42);
    let orders_schema = orders_schema();
    let customers_schema = customers_schema();

    let t = Instant::now();
    let orders = build_orders(&mut rng, &orders_schema)?;
    let customers = build_customers(&customers_schema)?;
    println!(
        "generated orders: {ORDER_ROWS} rows in {} batches, customers: {CUSTOMER_ROWS} rows ({:.1} ms)",
        orders.len(),
        ms_since(t)
    );

    let mut catalog = Catalog::new();
    catalog.register("orders", orders_schema.clone(), orders);
    catalog.register("customers", customers_schema.clone(), vec![customers]);

    // (a) SELECT id, cust, amount FROM orders WHERE amount > 900
    let project_schema = schema_ref(Schema::new(vec![
        Field::new("id", DataType::Int64, false),
        Field::new("cust", DataType::Int64, false),
        Field::new("amount", DataType::Int64, false),
    ]));
    let filtered = Arc::new(LogicalPlan::Filter {
        input: scan("orders", &orders_schema),
        predicate: binary(
            col(&orders_schema, "amount")?,
            BinaryOp::Gt,
            lit(ScalarValue::Int64(900)),
        )?,
    });
    let qa = Arc::new(LogicalPlan::Projection {
        input: filtered,
        exprs: vec![
            col(&orders_schema, "id")?,
            col(&orders_schema, "cust")?,
            col(&orders_schema, "amount")?,
        ],
        schema: project_schema,
    });

    // (b) SELECT region, SUM(amount), COUNT(*), AVG(amount) FROM orders GROUP BY region
    let region_schema = schema_ref(Schema::new(vec![
        Field::new("region", DataType::Utf8, false),
        Field::new("sum_amount", DataType::Int64, true),
        Field::new("count", DataType::Int64, false),
        Field::new("avg_amount", DataType::Float64, true),
    ]));
    let qb = Arc::new(LogicalPlan::Aggregate {
        input: scan("orders", &orders_schema),
        group_exprs: vec![col(&orders_schema, "region")?],
        agg_exprs: vec![
            agg(
                AggFunc::Sum,
                Some(col(&orders_schema, "amount")?),
                DataType::Int64,
            ),
            agg(AggFunc::Count, None, DataType::Int64),
            agg(
                AggFunc::Avg,
                Some(col(&orders_schema, "amount")?),
                DataType::Float64,
            ),
        ],
        schema: region_schema,
    });

    // (c) orders JOIN customers ON cust, GROUP BY name, ORDER BY total DESC LIMIT 5
    let joined_schema = schema_ref(Schema::new(
        orders_schema
            .fields
            .iter()
            .chain(customers_schema.fields.iter())
            .cloned()
            .collect(),
    ));
    let top_schema = schema_ref(Schema::new(vec![
        Field::new("name", DataType::Utf8, false),
        Field::new("total_amount", DataType::Int64, true),
        Field::new("orders", DataType::Int64, false),
    ]));
    let join = Arc::new(LogicalPlan::Join {
        left: scan("orders", &orders_schema),
        right: scan("customers", &customers_schema),
        join_type: JoinType::Inner,
        on: vec![(
            col(&orders_schema, "cust")?,
            col(&customers_schema, "cust")?,
        )],
        schema: joined_schema.clone(),
    });
    let grouped = Arc::new(LogicalPlan::Aggregate {
        input: join,
        group_exprs: vec![col(&joined_schema, "name")?],
        agg_exprs: vec![
            agg(
                AggFunc::Sum,
                Some(col(&joined_schema, "amount")?),
                DataType::Int64,
            ),
            agg(AggFunc::Count, None, DataType::Int64),
        ],
        schema: top_schema.clone(),
    });
    let sorted = Arc::new(LogicalPlan::Sort {
        input: grouped,
        order_by: vec![SortField {
            expr: col(&top_schema, "total_amount")?,
            ascending: false,
            nulls_first: false,
        }],
    });
    let qc = Arc::new(LogicalPlan::Limit {
        input: sorted,
        skip: 0,
        fetch: Some(5),
    });

    let ctx = exec_context()?;
    let queries = [
        ("a) filter + project: amount > 900", qa),
        ("b) group by region: sum, count, avg(amount)", qb),
        (
            "c) join orders-customers, group by name, top 5 by total",
            qc,
        ),
    ];
    for (title, plan) in &queries {
        let t = Instant::now();
        let out = execute(plan.as_ref(), &catalog, &ctx)?;
        let elapsed = ms_since(t);
        let total_rows: usize = out.iter().map(|b| b.num_rows).sum();
        println!("\n== {title}");
        print_table(&plan.schema(), &out, PRINT_ROWS, &mut io::stdout().lock())?;
        println!("rows: {total_rows} (showing up to {PRINT_ROWS}), elapsed: {elapsed:.2} ms");
    }
    Ok(())
}

/// Aligned text table of at most `max_rows` rows. Write errors are returned so a closed pipe is not a panic.
fn print_table(
    schema: &Schema,
    batches: &[Batch],
    max_rows: usize,
    out: &mut impl Write,
) -> io::Result<()> {
    let headers: Vec<&str> = schema.fields.iter().map(|f| f.name.as_str()).collect();
    let mut rows: Vec<Vec<String>> = Vec::new();
    'outer: for b in batches {
        for i in 0..b.num_rows {
            if rows.len() >= max_rows {
                break 'outer;
            }
            rows.push(b.row(i).iter().map(format_scalar).collect());
        }
    }
    let mut widths: Vec<usize> = headers.iter().map(|h| h.chars().count()).collect();
    for r in &rows {
        for (w, cell) in widths.iter_mut().zip(r) {
            *w = (*w).max(cell.chars().count());
        }
    }
    let fmt_row = |cells: &[&str]| -> String {
        cells
            .iter()
            .zip(&widths)
            .map(|(c, w)| format!("{c:<w$}"))
            .collect::<Vec<_>>()
            .join("  ")
    };
    writeln!(out, "{}", fmt_row(&headers))?;
    writeln!(
        out,
        "{}",
        widths
            .iter()
            .map(|w| "-".repeat(*w))
            .collect::<Vec<_>>()
            .join("  ")
    )?;
    for r in &rows {
        let cells: Vec<&str> = r.iter().map(String::as_str).collect();
        writeln!(out, "{}", fmt_row(&cells))?;
    }
    Ok(())
}

fn format_scalar(v: &ScalarValue) -> String {
    match v {
        ScalarValue::Null(_) => "NULL".to_string(),
        ScalarValue::Boolean(b) => b.to_string(),
        ScalarValue::Int64(i) => i.to_string(),
        ScalarValue::Float64(f) => format!("{f:.2}"),
        ScalarValue::Utf8(s) => s.clone(),
    }
}

// ---------------------------------------------------------------- csv

struct CsvArgs {
    path: PathBuf,
    filter: Option<(String, BinaryOp, String)>,
    select: Option<Vec<String>>,
    limit: Option<usize>,
}

fn parse_csv_args(args: &[String]) -> std::result::Result<CsvArgs, String> {
    let mut path = None;
    let mut filter = None;
    let mut select = None;
    let mut limit = None;
    let mut it = args.iter();
    while let Some(arg) = it.next() {
        match arg.as_str() {
            "--where" => {
                if filter.is_some() {
                    return Err("only one --where is supported".into());
                }
                match (it.next(), it.next(), it.next()) {
                    (Some(c), Some(op), Some(v)) => {
                        filter = Some((c.clone(), parse_op(op)?, v.clone()))
                    }
                    _ => return Err("--where needs three values: <col> <op> <literal>".into()),
                }
            }
            "--select" => {
                let v = it
                    .next()
                    .ok_or("--select needs a comma-separated column list")?;
                let cols: Vec<String> = v
                    .split(',')
                    .map(str::trim)
                    .filter(|s| !s.is_empty())
                    .map(String::from)
                    .collect();
                if cols.is_empty() {
                    return Err("--select needs at least one column name".into());
                }
                select = Some(cols);
            }
            "--limit" => {
                let v = it.next().ok_or("--limit needs a row count N")?;
                let n = v
                    .parse::<usize>()
                    .map_err(|_| format!("--limit expects a non-negative integer, got '{v}'"))?;
                limit = Some(n);
            }
            flag if flag.starts_with("--") => return Err(format!("unknown option '{flag}'")),
            positional => {
                if path.is_some() {
                    return Err(format!("unexpected argument '{positional}'"));
                }
                path = Some(PathBuf::from(positional));
            }
        }
    }
    let path = path.ok_or("missing <file.csv>")?;
    Ok(CsvArgs {
        path,
        filter,
        select,
        limit,
    })
}

fn parse_op(op: &str) -> std::result::Result<BinaryOp, String> {
    Ok(match op {
        "=" => BinaryOp::Eq,
        "!=" => BinaryOp::NotEq,
        "<" => BinaryOp::Lt,
        "<=" => BinaryOp::LtEq,
        ">" => BinaryOp::Gt,
        ">=" => BinaryOp::GtEq,
        other => {
            return Err(format!(
                "unsupported operator '{other}' (use = != < <= > >=)"
            ))
        }
    })
}

fn parse_literal(text: &str, dt: DataType) -> std::result::Result<ScalarValue, String> {
    let bad = || format!("literal '{text}' is not a valid {dt:?}");
    match dt {
        DataType::Int64 => text.parse().map(ScalarValue::Int64).map_err(|_| bad()),
        DataType::Float64 => text.parse().map(ScalarValue::Float64).map_err(|_| bad()),
        DataType::Boolean => text.parse().map(ScalarValue::Boolean).map_err(|_| bad()),
        DataType::Utf8 => Ok(ScalarValue::Utf8(text.to_string())),
    }
}

fn csv(rest: &[String]) -> CliResult {
    let a = parse_csv_args(rest).map_err(Failure::Usage)?;
    let (schema, batches) = load_csv(&a.path, CHUNK_ROWS)?;

    let mut catalog = Catalog::new();
    catalog.register("csv", schema.clone(), batches);
    let mut plan = scan("csv", &schema);

    if let Some((name, op, value)) = &a.filter {
        let lhs = col(&schema, name).map_err(|e| Failure::Usage(e.to_string()))?;
        let rhs = parse_literal(value, lhs.return_type()).map_err(Failure::Usage)?;
        let predicate = binary(lhs, *op, lit(rhs))?;
        plan = Arc::new(LogicalPlan::Filter {
            input: plan,
            predicate,
        });
    }
    if let Some(names) = &a.select {
        let mut exprs = Vec::with_capacity(names.len());
        let mut fields = Vec::with_capacity(names.len());
        for name in names {
            let e = col(&schema, name).map_err(|e| Failure::Usage(e.to_string()))?;
            fields.push(Field::new(name, e.return_type(), true));
            exprs.push(e);
        }
        plan = Arc::new(LogicalPlan::Projection {
            input: plan,
            exprs,
            schema: schema_ref(Schema::new(fields)),
        });
    }
    if let Some(n) = a.limit {
        plan = Arc::new(LogicalPlan::Limit {
            input: plan,
            skip: 0,
            fetch: Some(n),
        });
    }

    let t = Instant::now();
    let out = execute(plan.as_ref(), &catalog, &exec_context()?)?;
    let elapsed = ms_since(t);
    let total_rows: usize = out.iter().map(|b| b.num_rows).sum();
    let mut stdout = BufWriter::new(io::stdout().lock());
    print_table(&plan.schema(), &out, usize::MAX, &mut stdout)?;
    writeln!(stdout, "rows: {total_rows}, elapsed: {elapsed:.2} ms")?;
    stdout.flush()?;
    Ok(())
}

// ---------------------------------------------------------------- bench

const RUNS: usize = 5;
const SIMD_ROWS: usize = 16_000_000;
const EVAL_ROWS: usize = 4_000_000;
const SCALE_ROWS: usize = 16_000_000;
const THREAD_COUNTS: [usize; 4] = [1, 2, 4, 8];
const THRESHOLD: i64 = 500_000;
/// Assumed JIT compile cost for the illustrative crossover only. It is NOT measured.
const ILLUSTRATIVE_COMPILE_NS: f64 = 5_000_000.0;

#[derive(Clone, Copy)]
struct Stats {
    min_ms: f64,
    median_ms: f64,
}

fn stats_from(mut ms: Vec<f64>) -> Stats {
    ms.sort_by(f64::total_cmp);
    Stats {
        min_ms: ms[0],
        median_ms: ms[ms.len() / 2],
    }
}

/// One untimed warm-up, then `RUNS` timed calls. The output is kept alive past the timer
/// and passed through `black_box`, so the work cannot be optimized away.
fn sample<R>(mut f: impl FnMut() -> Result<R>) -> Result<Stats> {
    black_box(f()?);
    let mut ms = Vec::with_capacity(RUNS);
    for _ in 0..RUNS {
        let t = Instant::now();
        let out = f()?;
        ms.push(ms_since(t));
        black_box(out);
    }
    Ok(stats_from(ms))
}

fn tuples_per_s(rows: usize, ms: f64) -> f64 {
    rows as f64 / (ms / 1e3)
}

fn print_stats(label: &str, rows: usize, s: Stats) {
    println!(
        "{label:<26} {:>11.3} {:>11.3} {:>14.3e}",
        s.min_ms,
        s.median_ms,
        tuples_per_s(rows, s.median_ms)
    );
}

fn print_stats_header() {
    println!(
        "{:<26} {:>11} {:>11} {:>14}",
        "case", "min ms", "median ms", "tuples/s"
    );
}

fn int_schema(names: &[&str]) -> SchemaRef {
    schema_ref(Schema::new(
        names
            .iter()
            .map(|n| Field::new(n, DataType::Int64, false))
            .collect(),
    ))
}

fn bench_simd(values: &[i64]) -> Result<(Stats, Stats)> {
    let n = values.len();
    let mut check_a = vec![false; n];
    let mut check_b = vec![false; n];
    gt_i64(values, THRESHOLD, &mut check_a);
    gt_i64_scalar(values, THRESHOLD, &mut check_b);
    if check_a != check_b {
        return Err(EngineError::Plan(
            "gt_i64 disagrees with gt_i64_scalar".into(),
        ));
    }

    let mut out = vec![false; n];
    let simd = sample(|| {
        gt_i64(values, THRESHOLD, &mut out);
        Ok(out.len())
    })?;
    let scalar = sample(|| {
        gt_i64_scalar(values, THRESHOLD, &mut out);
        Ok(out.len())
    })?;
    Ok((simd, scalar))
}

fn hand_expr(a: &[i64], b: &[i64], c: &[i64], d: &[i64]) -> Vec<i64> {
    let mut out = Vec::with_capacity(a.len());
    for (((&a, &b), &c), &d) in a.iter().zip(b).zip(c).zip(d) {
        out.push(
            a.wrapping_add(b)
                .wrapping_mul(c)
                .wrapping_sub(d.wrapping_mul(2)),
        );
    }
    out
}

/// Returns (vectorized eval stats, hand loop stats) for `((a + b) * c) - (d * 2)`.
fn bench_eval(a: &[i64], b: &[i64], c: &[i64], d: &[i64]) -> Result<(Stats, Stats)> {
    let schema = int_schema(&["a", "b", "c", "d"]);
    let batch = Batch::try_new(
        schema.clone(),
        vec![
            Column::from_i64(a.to_vec()),
            Column::from_i64(b.to_vec()),
            Column::from_i64(c.to_vec()),
            Column::from_i64(d.to_vec()),
        ],
    )?;
    let sum_ab = binary(col(&schema, "a")?, BinaryOp::Add, col(&schema, "b")?)?;
    let prod = binary(sum_ab, BinaryOp::Mul, col(&schema, "c")?)?;
    let d2 = binary(
        col(&schema, "d")?,
        BinaryOp::Mul,
        lit(ScalarValue::Int64(2)),
    )?;
    let expr = binary(prod, BinaryOp::Sub, d2)?;

    let want = hand_expr(a, b, c, d);
    match &eval(&expr, &batch)?.values {
        Values::Int64(got) if *got == want => {}
        _ => {
            return Err(EngineError::Plan(
                "vectorized eval disagrees with hand loop".into(),
            ))
        }
    }

    let vectorized = sample(|| eval(&expr, &batch))?;
    let hand = sample(|| Ok(hand_expr(a, b, c, d)))?;
    Ok((vectorized, hand))
}

/// Median-based scaling of `run_morsels` over a filter kernel. Morsel splitting happens outside the timer.
fn bench_scaling(values: Vec<i64>) -> Result<Vec<(usize, Stats)>> {
    let batch = Batch::try_new(int_schema(&["v"]), vec![Column::from_i64(values)])?;
    let filter_op = |m: &Batch| -> Result<Batch> {
        let Values::Int64(v) = &m.columns[0].values else {
            return Err(EngineError::Type(
                "filter scaling expects an Int64 column".into(),
            ));
        };
        let mut mask = vec![false; v.len()];
        gt_i64(v, THRESHOLD, &mut mask);
        Ok(m.filter(&mask))
    };

    let mut results = Vec::with_capacity(THREAD_COUNTS.len());
    for &threads in &THREAD_COUNTS {
        let mut ms = Vec::with_capacity(RUNS);
        for run in 0..=RUNS {
            let morsels = split_morsels(&batch, CHUNK_ROWS);
            let t = Instant::now();
            let out = run_morsels(morsels, threads, filter_op)?;
            let elapsed = ms_since(t);
            black_box(&out);
            if run > 0 {
                ms.push(elapsed);
            }
        }
        results.push((threads, stats_from(ms)));
    }
    Ok(results)
}

fn bench() -> CliResult {
    println!("Laptop microbenchmarks. They do NOT validate hypotheses H1a/H1b/H2/H3 from docs/BENCHMARK_METHODOLOGY.md.");
    println!(
        "arch: {}, logical cpus: {}, {RUNS} timed runs per case after 1 warm-up run",
        std::env::consts::ARCH,
        default_threads()
    );

    let mut rng = Lcg(7);
    let simd_values: Vec<i64> = (0..SIMD_ROWS).map(|_| rng.below(1_000_000)).collect();
    let (simd, scalar) = bench_simd(&simd_values)?;
    println!("\n== simd filter: gt_i64 vs gt_i64_scalar, {SIMD_ROWS} i64, threshold {THRESHOLD}");
    print_stats_header();
    print_stats("gt_i64 (dispatched)", SIMD_ROWS, simd);
    print_stats("gt_i64_scalar", SIMD_ROWS, scalar);
    println!(
        "speedup (median): {:.2}x",
        scalar.median_ms / simd.median_ms
    );
    drop(simd_values);

    let cols: Vec<Vec<i64>> = (0..4)
        .map(|_| (0..EVAL_ROWS).map(|_| rng.below(1000)).collect())
        .collect();
    let (vectorized, hand) = bench_eval(&cols[0], &cols[1], &cols[2], &cols[3])?;
    println!("\n== eval ((a + b) * c) - (d * 2): 4 arithmetic nodes, {EVAL_ROWS} rows");
    print_stats_header();
    print_stats("vectorized eval", EVAL_ROWS, vectorized);
    print_stats("hand-written loop", EVAL_ROWS, hand);
    println!(
        "ratio (median): {:.2}x",
        vectorized.median_ms / hand.median_ms
    );

    let scale_values: Vec<i64> = (0..SCALE_ROWS).map(|_| rng.below(1_000_000)).collect();
    let scaling = bench_scaling(scale_values)?;
    println!("\n== run_morsels filter scaling: gt_i64 + filter, {SCALE_ROWS} rows, morsel {CHUNK_ROWS} rows");
    print_stats_header();
    let base = scaling.first().map_or(1.0, |(_, s)| s.median_ms);
    for (threads, s) in &scaling {
        print_stats(&format!("threads = {threads}"), SCALE_ROWS, *s);
    }
    for (threads, s) in &scaling {
        println!(
            "threads = {threads}: speedup vs 1 thread (median) {:.2}x",
            base / s.median_ms
        );
    }

    let vec_ns = vectorized.median_ms * 1e6 / EVAL_ROWS as f64;
    let hand_ns = hand.median_ms * 1e6 / EVAL_ROWS as f64;
    let params = CostParams {
        t_vec_tuple_ns: vec_ns,
        t_jit_tuple_ns: hand_ns,
        t_comp_ns: ILLUSTRATIVE_COMPILE_NS,
    };
    println!("\n== cost::crossover (ILLUSTRATIVE ONLY)");
    println!(
        "inputs: t_vec = {vec_ns:.3} ns/tuple (measured), hand loop as JIT stand-in t_jit = {hand_ns:.3} ns/tuple (measured), \
         t_comp = {ILLUSTRATIVE_COMPILE_NS:.0} ns (assumed, not measured)"
    );
    match crossover(&params) {
        Some(n) => println!("illustrative crossover N = {n:.0} tuples"),
        None => println!("illustrative crossover: none (hand loop is not faster per tuple here)"),
    }
    Ok(())
}
