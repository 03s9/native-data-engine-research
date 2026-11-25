//! H1a measurement binary (docs/PLAN.md T4): does the Cranelift JIT pay off against the
//! vectorized evaluator (`eval::eval`)?
//!
//! Scope: a single laptop, single thread, release build only. Does NOT address H1b (LLVM),
//! H2 (x86/AVX-512) or H3.
//!
//! Timing rules:
//! - `std::time::Instant` around each region, `black_box` on inputs and outputs.
//! - Both paths allocate their output column per call. The vectorized path also allocates one
//!   column per intermediate node; the JIT path allocates only its output.
//! - Outputs leave the timed region and are dropped after the clock stops. Where a region runs K
//!   calls back to back (K calibrated so a region lasts about 1 ms), intermediate outputs are
//!   dropped inside the region on both sides, so the cost is symmetric.
//! - Vectorized and JIT regions alternate within each iteration.
//! - Before any timing, outputs are compared row by row (valid rows; the data has no nulls).

use native_engine::column::{schema_ref, Batch, Column, Values};
use native_engine::cost::{crossover, eci, CostParams};
use native_engine::eval::{binary, cast, col, eval, lit};
use native_engine::expr::{BinaryOp, Expr};
use native_engine::jit;
use native_engine::types::{DataType, Field, ScalarValue, Schema, SchemaRef};
use std::error::Error;
use std::hint::black_box;
use std::io::Write;
use std::time::Instant;

type EResult<T> = native_engine::types::Result<T>;
type BoxResult<T> = Result<T, Box<dyn Error>>;

const GRID_FULL: [usize; 6] = [1_000, 10_000, 65_536, 200_000, 1_000_000, 4_000_000];
const GRID_QUICK: [usize; 4] = [1_000, 10_000, 65_536, 200_000];
const CHECK_N: usize = 1_000;
const MIN_REGION_NS: f64 = 1_000_000.0;
const T_COMP_LIMIT_NS: f64 = 1_500_000.0;
const MODEL_XOVER_LIMIT: f64 = 200_000.0;
const SPEEDUP_MIN: f64 = 1.3;
const N_VERDICT: usize = 200_000;
const N_VERDICT_HI: usize = 1_000_000;
const ECI_VERDICT_MIN: f64 = 4.0;

const HELP: &str = "\
h1a [--quick] [--csv <path>] [--help]

Measures T_comp, per-tuple time (vectorized vs JIT) and end-to-end crossover for the H1a
expression set (docs/PLAN.md T4). Run from a release build:
    cargo build --release --bin h1a && ./target/release/h1a

  --quick       N up to 200000 and 3 runs per point (smoke test; no 1M verdict)
  --csv <path>  write every raw measurement row (expr, eci, metric, n, run_index, value_ns)
  --help        this text

Scope: single laptop, single thread. Does not address H1b (LLVM), H2 (x86/AVX-512) or H3.";

struct Opts {
    quick: bool,
    csv: Option<String>,
}

struct Plan {
    quick: bool,
}

impl Plan {
    fn grid(&self) -> &'static [usize] {
        if self.quick {
            &GRID_QUICK
        } else {
            &GRID_FULL
        }
    }
    fn runs(&self) -> usize {
        if self.quick {
            3
        } else {
            9
        }
    }
    fn e2e_runs(&self) -> usize {
        if self.quick {
            3
        } else {
            5
        }
    }
    fn comp_runs(&self) -> usize {
        if self.quick {
            3
        } else {
            30
        }
    }
}

struct Row {
    expr: &'static str,
    eci: f64,
    metric: &'static str,
    n: usize,
    run: usize,
    value_ns: f64,
}

#[derive(Clone, Copy)]
struct Stats {
    min: f64,
    median: f64,
    p90: f64,
}

struct NResult {
    n: usize,
    vec: Stats,
    jit: Stats,
    e2e_vec: Stats,
    e2e_jit: Stats,
}

struct ExprResult {
    name: &'static str,
    eci: f64,
    supported: bool,
    tcomp: Option<Stats>,
    per_n: Vec<NResult>,
}

fn print_help() {
    println!("{HELP}");
}

fn parse_args(args: &[String]) -> Result<Opts, String> {
    let mut opts = Opts {
        quick: false,
        csv: None,
    };
    let mut it = args.iter();
    while let Some(a) = it.next() {
        match a.as_str() {
            "--quick" => opts.quick = true,
            "--csv" => {
                let p = it.next().ok_or("--csv needs a path")?;
                opts.csv = Some(p.clone());
            }
            other => return Err(format!("unknown argument: {other} (see --help)")),
        }
    }
    Ok(opts)
}

fn stats(v: &mut [f64]) -> Stats {
    v.sort_by(|a, b| a.total_cmp(b));
    let n = v.len();
    let median = if n % 2 == 1 {
        v[n / 2]
    } else {
        (v[n / 2 - 1] + v[n / 2]) / 2.0
    };
    let p90 = v[((0.9 * n as f64).ceil() as usize).max(1) - 1];
    Stats {
        min: v[0],
        median,
        p90,
    }
}

fn push(
    rows: &mut Vec<Row>,
    expr: &'static str,
    eci: f64,
    metric: &'static str,
    n: usize,
    run: usize,
    value_ns: f64,
) {
    rows.push(Row {
        expr,
        eci,
        metric,
        n,
        run,
        value_ns,
    });
}

/// Smallest nonzero gap between consecutive `Instant::now()` calls, in ns.
fn timer_gap_ns() -> u128 {
    let mut min = u128::MAX;
    let mut prev = Instant::now();
    for _ in 0..200_000 {
        let now = Instant::now();
        let d = now.duration_since(prev).as_nanos();
        if d > 0 && d < min {
            min = d;
        }
        prev = now;
    }
    min
}

/// Row-by-row comparison. Values are compared for valid rows; validity is compared for all rows.
fn same_column(a: &Column, b: &Column) -> Result<(), String> {
    if a.len() != b.len() {
        return Err(format!("length {} vs {}", a.len(), b.len()));
    }
    for i in 0..a.len() {
        if a.is_valid(i) != b.is_valid(i) {
            return Err(format!("validity differs at row {i}"));
        }
    }
    let valid = |i: usize| a.is_valid(i);
    let mismatch = match (&a.values, &b.values) {
        (Values::Int64(x), Values::Int64(y)) => (0..x.len()).find(|&i| valid(i) && x[i] != y[i]),
        (Values::Float64(x), Values::Float64(y)) => {
            (0..x.len()).find(|&i| valid(i) && x[i].to_bits() != y[i].to_bits())
        }
        (Values::Bool(x), Values::Bool(y)) => (0..x.len()).find(|&i| valid(i) && x[i] != y[i]),
        _ => {
            return Err(format!(
                "output variant mismatch: {:?} vs {:?}",
                a.data_type(),
                b.data_type()
            ))
        }
    };
    match mismatch {
        Some(i) => Err(format!("row {i} differs")),
        None => Ok(()),
    }
}

/// Runs `f` K times inside one region and returns the mean ns per call.
/// Intermediate outputs are dropped inside the region; the last one is dropped after the clock stops.
fn time_region(k: usize, n: usize, f: &mut dyn FnMut() -> EResult<Column>) -> BoxResult<f64> {
    let start = Instant::now();
    let mut keep: Option<Column> = None;
    for i in 0..k {
        let out = black_box(f()?);
        if out.len() != n {
            return Err(format!("output has {} rows, expected {n}", out.len()).into());
        }
        if i + 1 < k {
            drop(out);
        } else {
            keep = Some(out);
        }
    }
    let total = start.elapsed().as_nanos() as f64;
    drop(keep);
    Ok(total / k as f64)
}

/// Wall-clock compile time (kernel dropped outside the clock), plus the kernel's own `compile_time`.
fn compile_timed(expr: &Expr) -> BoxResult<(f64, f64)> {
    let start = Instant::now();
    let kernel = jit::compile(black_box(expr))?;
    let wall = start.elapsed().as_nanos() as f64;
    let internal = kernel.compile_time.as_nanos() as f64;
    drop(kernel);
    Ok((wall, internal))
}

struct SplitMix(u64);

impl SplitMix {
    fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
    fn int(&mut self) -> i64 {
        (self.next_u64() % 1000) as i64
    }
    fn float(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64 * 20.0 - 10.0
    }
}

fn schema() -> SchemaRef {
    schema_ref(Schema::new(vec![
        Field::new("a", DataType::Int64, false),
        Field::new("b", DataType::Int64, false),
        Field::new("c", DataType::Int64, false),
        Field::new("d", DataType::Int64, false),
        Field::new("x", DataType::Float64, false),
        Field::new("y", DataType::Float64, false),
    ]))
}

/// Deterministic data, no nulls. Int64 in [0, 1000), Float64 in [-10, 10).
fn gen_batch(n: usize, sch: &SchemaRef) -> EResult<Batch> {
    let mut rng = SplitMix(0x5EED_0000 ^ n as u64);
    let mut cols = Vec::with_capacity(6);
    for _ in 0..4 {
        cols.push(Column::from_i64((0..n).map(|_| rng.int()).collect()));
    }
    for _ in 0..2 {
        cols.push(Column::from_f64((0..n).map(|_| rng.float()).collect()));
    }
    Batch::try_new(sch.clone(), cols)
}

fn chain(s: &Schema, names: &[&str]) -> EResult<Expr> {
    let mut e = col(s, names[0]);
    for name in &names[1..] {
        e = op(e, BinaryOp::Add, col(s, name));
    }
    e
}

fn op(l: EResult<Expr>, o: BinaryOp, r: EResult<Expr>) -> EResult<Expr> {
    binary(l?, o, r?)
}

fn int(v: i64) -> EResult<Expr> {
    Ok(lit(ScalarValue::Int64(v)))
}

fn flt(v: f64) -> EResult<Expr> {
    Ok(lit(ScalarValue::Float64(v)))
}

fn float4(s: &Schema) -> EResult<Expr> {
    op(
        op(col(s, "x"), BinaryOp::Mul, col(s, "y")),
        BinaryOp::Add,
        op(col(s, "x"), BinaryOp::Sub, col(s, "y")),
    )
}

/// The H1a expression set, in order. Trees are built left-deep as written, because tree shape sets ECI.
fn expression_set(s: &Schema) -> EResult<Vec<(&'static str, Expr)>> {
    use BinaryOp::{Add, Gt, Mul, Sub};
    let a = || col(s, "a");
    let b = || col(s, "b");
    let c = || col(s, "c");
    let d = || col(s, "d");
    let x = || col(s, "x");
    let y = || col(s, "y");

    let add2 = op(a(), Add, b());
    let add4 = chain(s, &["a", "b", "c", "d"]);
    let add8 = chain(s, &["a", "b", "c", "d", "a", "b", "c", "d"]);
    let arith4 = op(op(op(a(), Add, b()), Mul, c()), Sub, op(d(), Mul, int(2)));
    let arith8 = op(
        op(
            op(op(op(a(), Add, b()), Mul, c()), Sub, op(d(), Mul, int(2))),
            Mul,
            op(a(), Sub, c()),
        ),
        Add,
        op(b(), Mul, int(3)),
    );
    let float8 = op(
        op(float4(s), Mul, op(x(), Add, flt(1.5))),
        Sub,
        op(y(), Mul, y()),
    );
    let mixed = op(op(cast(a()?, DataType::Float64), Mul, x()), Add, y());
    let cmp = op(op(op(a(), Add, b()), Mul, c()), Gt, op(d(), Mul, int(1000)));

    [
        ("add2", add2),
        ("add4", add4),
        ("add8", add8),
        ("arith4", arith4),
        ("arith8", arith8),
        ("float4", float4(s)),
        ("float8", float8),
        ("mixed", mixed),
        ("cmp", cmp),
    ]
    .into_iter()
    .map(|(name, e)| e.map(|e| (name, e)))
    .collect()
}

fn run_expr(
    name: &'static str,
    expr: &Expr,
    sch: &SchemaRef,
    plan: &Plan,
    rows: &mut Vec<Row>,
) -> BoxResult<ExprResult> {
    let e = eci(expr);
    let supported = jit::supports(expr);
    println!("\n== {name}  eci={e:.1}  jit_supported={supported}");

    if !supported {
        println!("  skipped: unsupported by jit");
        let small = gen_batch(CHECK_N, sch)?;
        let v = eval(expr, &small)?;
        same_column(&v, &v)?;
        println!("  eval self-check at N={CHECK_N}: ok (runs only; no JIT output to compare)");
        return Ok(ExprResult {
            name,
            eci: e,
            supported,
            tcomp: None,
            per_n: Vec::new(),
        });
    }

    let (cold, _) = compile_timed(expr)?;
    println!(
        "  first compile (cold, excluded from stats): {:.1} us",
        cold / 1e3
    );
    let mut comp = Vec::new();
    for r in 0..plan.comp_runs() {
        let (wall, internal) = compile_timed(expr)?;
        comp.push(wall);
        push(rows, name, e, "t_comp_ns", 0, r, wall);
        push(rows, name, e, "compile_internal_ns", 0, r, internal);
    }
    let tc = stats(&mut comp);
    println!(
        "  T_comp us ({} compiles): min {:.1}  median {:.1}  p90 {:.1}",
        plan.comp_runs(),
        tc.min / 1e3,
        tc.median / 1e3,
        tc.p90 / 1e3
    );

    let mut per_n = Vec::new();
    for &n in plan.grid() {
        let batch = gen_batch(n, sch)?;
        let kernel = jit::compile(expr)?;
        let out_vec = eval(expr, &batch)?;
        let out_jit = kernel.run(&batch)?;
        same_column(&out_vec, &out_jit)
            .map_err(|m| format!("{name} N={n}: JIT output differs from eval: {m}"))?;
        drop(out_vec);
        drop(out_jit);

        let mut run_vec = || eval(black_box(expr), black_box(&batch));
        let mut run_jit = || kernel.run(black_box(&batch));
        let mut e2e_jit = || -> EResult<Column> {
            let k = jit::compile(black_box(expr))?;
            let c = k.run(black_box(&batch))?;
            drop(k);
            Ok(c)
        };

        // Calibrate K so one region lasts about 1 ms on the faster side.
        let w_vec = time_region(1, n, &mut run_vec)?;
        let w_jit = time_region(1, n, &mut run_jit)?;
        let k = ((MIN_REGION_NS / w_vec.min(w_jit).max(1.0)).ceil() as usize).clamp(1, 1_000_000);

        let (mut vs, mut js) = (Vec::new(), Vec::new());
        for r in 0..plan.runs() {
            let tv = time_region(k, n, &mut run_vec)?;
            let tj = time_region(k, n, &mut run_jit)?;
            push(rows, name, e, "vec_call_ns", n, r, tv);
            push(rows, name, e, "jit_call_ns", n, r, tj);
            vs.push(tv / n as f64);
            js.push(tj / n as f64);
        }

        // Warm-up for the end-to-end regions (untimed), then alternate vec and jit.
        time_region(1, n, &mut run_vec)?;
        time_region(1, n, &mut e2e_jit)?;
        let (mut evs, mut ejs) = (Vec::new(), Vec::new());
        for r in 0..plan.e2e_runs() {
            let tv = time_region(k, n, &mut run_vec)?;
            let tj = time_region(k, n, &mut e2e_jit)?;
            push(rows, name, e, "e2e_vec_call_ns", n, r, tv);
            push(rows, name, e, "e2e_jit_call_ns", n, r, tj);
            evs.push(tv);
            ejs.push(tj);
        }

        let res = NResult {
            n,
            vec: stats(&mut vs),
            jit: stats(&mut js),
            e2e_vec: stats(&mut evs),
            e2e_jit: stats(&mut ejs),
        };
        let xover = crossover(&CostParams {
            t_vec_tuple_ns: res.vec.median,
            t_jit_tuple_ns: res.jit.median,
            t_comp_ns: tc.median,
        })
        .map_or_else(|| "none".to_string(), |x| format!("{x:.0}"));
        let winner = if res.e2e_jit.median < res.e2e_vec.median {
            "jit"
        } else {
            "vec"
        };
        println!(
            "  N={n:>9} K={k:<7} vec ns/t min {:>8.3} med {:>8.3} | jit ns/t min {:>8.3} med {:>8.3} | delta_med {:>8.3} | speedup {:>5.2}x | model_xover {xover:>9} | e2e vec {:>10.3} ms jit {:>10.3} ms -> {winner}",
            res.vec.min,
            res.vec.median,
            res.jit.min,
            res.jit.median,
            res.vec.median - res.jit.median,
            res.vec.median / res.jit.median,
            res.e2e_vec.median / 1e6,
            res.e2e_jit.median / 1e6,
        );
        per_n.push(res);
    }

    match per_n
        .iter()
        .find(|r| r.e2e_jit.median < r.e2e_vec.median)
        .map(|r| r.n)
    {
        Some(n) => println!(
            "  empirical crossover (e2e median, runs per point {}): N={n}",
            plan.e2e_runs()
        ),
        None => println!(
            "  empirical crossover (e2e median, runs per point {}): never in grid",
            plan.e2e_runs()
        ),
    }

    Ok(ExprResult {
        name,
        eci: e,
        supported,
        tcomp: Some(tc),
        per_n,
    })
}

fn pf(ok: bool) -> &'static str {
    if ok {
        "PASS"
    } else {
        "FAIL"
    }
}

fn verdict(results: &[ExprResult], quick: bool) {
    println!("\n== H1a criteria per expression (eci >= {ECI_VERDICT_MIN}); no overall verdict is printed");
    for r in results.iter().filter(|r| r.eci >= ECI_VERDICT_MIN) {
        println!("\n{} (eci={:.1})", r.name, r.eci);
        let (Some(tc), true) = (r.tcomp, r.supported) else {
            println!("  NOT MEASURED: unsupported by jit in this tree");
            continue;
        };
        let t_ok = tc.median <= T_COMP_LIMIT_NS;
        println!(
            "  T_comp median <= 1.5 ms: {} ({:.3} ms)",
            pf(t_ok),
            tc.median / 1e6
        );
        let Some(at) = r.per_n.iter().find(|x| x.n == N_VERDICT) else {
            println!("  N={N_VERDICT} not in grid; verdict cannot be computed");
            continue;
        };
        let xover = crossover(&CostParams {
            t_vec_tuple_ns: at.vec.median,
            t_jit_tuple_ns: at.jit.median,
            t_comp_ns: tc.median,
        });
        let x_ok = xover.is_some_and(|x| x <= MODEL_XOVER_LIMIT);
        let x_txt = xover.map_or_else(
            || "no crossover (delta <= 0)".to_string(),
            |x| format!("{x:.0} tuples"),
        );
        println!(
            "  model crossover <= 200000 (rates at N={N_VERDICT}): {} ({x_txt})",
            pf(x_ok)
        );
        let s200 = at.vec.median / at.jit.median;
        println!(
            "  speedup >= 1.3 at N={N_VERDICT}: {} ({s200:.2}x)",
            pf(s200 >= SPEEDUP_MIN)
        );
        match r.per_n.iter().find(|x| x.n == N_VERDICT_HI) {
            Some(hi) => {
                let s1m = hi.vec.median / hi.jit.median;
                println!(
                    "  speedup >= 1.3 at N={N_VERDICT_HI}: {} ({s1m:.2}x)",
                    pf(s1m >= SPEEDUP_MIN)
                );
            }
            None => println!(
                "  speedup >= 1.3 at N={N_VERDICT_HI}: n/a ({}; not in grid)",
                if quick { "--quick" } else { "grid" }
            ),
        }
    }
}

fn write_csv(path: &str, rows: &[Row]) -> BoxResult<()> {
    let mut w = std::io::BufWriter::new(std::fs::File::create(path)?);
    writeln!(w, "expr,eci,metric,n,run_index,value_ns")?;
    for r in rows {
        writeln!(
            w,
            "{},{:.3},{},{},{},{:.3}",
            r.expr, r.eci, r.metric, r.n, r.run, r.value_ns
        )?;
    }
    w.flush()?;
    Ok(())
}

fn main() -> BoxResult<()> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.iter().any(|a| a == "--help" || a == "-h") {
        print_help();
        return Ok(());
    }
    if cfg!(debug_assertions) {
        eprintln!(
            "h1a: refusing to measure in a debug build. Run: cargo build --release --bin h1a && ./target/release/h1a"
        );
        std::process::exit(2);
    }
    let opts = match parse_args(&args) {
        Ok(o) => o,
        Err(e) => {
            eprintln!("h1a: {e}");
            std::process::exit(2);
        }
    };
    let plan = Plan { quick: opts.quick };

    println!("h1a: H1a measurement (docs/PLAN.md T4). Cranelift JIT vs vectorized eval.");
    println!(
        "Scope: expressions jit::supports accepts (Int64/Float64 arithmetic and comparisons); others are skipped. Does NOT address H1b (LLVM), H2 (x86/AVX-512) or H3."
    );
    println!(
        "Machine: arch={} os={} logical_cpus={} | single laptop, single thread for all measured work",
        std::env::consts::ARCH,
        std::env::consts::OS,
        std::thread::available_parallelism().map_or(0, |n| n.get())
    );
    println!("Build: release (debug_assertions off)");
    println!(
        "Timer: smallest nonzero Instant gap = {} ns",
        timer_gap_ns()
    );
    println!(
        "Mode: {} | per-tuple runs {} (+1 warm-up) | e2e runs {} (+1 warm-up) | T_comp compiles {} (+1 cold, excluded) | N grid {:?}",
        if opts.quick { "quick" } else { "full" },
        plan.runs(),
        plan.e2e_runs(),
        plan.comp_runs(),
        plan.grid()
    );
    println!(
        "Allocation: both paths allocate their output per call; eval also allocates one column per intermediate node, the kernel allocates only its output (zeroed)."
    );
    println!(
        "Samples: per-tuple = region time / (K*N); K calibrated so a region is about 1 ms (K printed per N). e2e JIT = compile + run + kernel drop per call; e2e vec = eval per call."
    );

    let sch = schema();
    let set = expression_set(&sch)?;
    let mut rows = Vec::new();
    let mut results = Vec::new();
    for (name, expr) in set.iter() {
        results.push(run_expr(name, expr, &sch, &plan, &mut rows)?);
    }
    verdict(&results, opts.quick);

    if let Some(path) = &opts.csv {
        write_csv(path, &rows)?;
        println!("\nwrote {} raw rows to {path}", rows.len());
    }
    Ok(())
}
