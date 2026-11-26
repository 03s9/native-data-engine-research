//! Differential suite (docs/PLAN.md T3): the Cranelift JIT (`jit::compile` + `CompiledKernel::run`)
//! must return exactly what the vectorized evaluator (`eval::eval`) returns.
//!
//! Everything here is seeded and deterministic. Divergences are collected, minimised and reported
//! together; the suite fails at the end if any were found. Public API only.
//!
//! Documented difference (not a failure): when both sides produce NaN for a row but the bit
//! patterns differ (payload or sign), the row is counted in `nan_rows` and printed. NaN against a
//! non-NaN value is a real divergence.

use native_engine::column::{schema_ref, Batch, Column, Values};
use native_engine::eval::{binary, case, cast, col, eval, lit, unary};
use native_engine::expr::{BinaryOp, Expr, UnaryOp};
use native_engine::jit::{compile, compile_with, supports, Codegen, CompiledKernel};
use native_engine::types::{DataType, EngineError, Field, ScalarValue, Schema, SchemaRef};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

const SEED: u64 = 0x5EED_0003;
// Odd sizes exercise the vector loop's scalar tail (Codegen::Vector handles two rows per step).
const SIZES: [usize; 13] = [0, 1, 2, 3, 7, 8, 9, 63, 64, 65, 8191, 8192, 8193];
/// Batch variants per size: 0 no masks, 1 all valid, 2 all null, 3 random per column, 4 mixed per column.
const VARIANTS: usize = 5;
const MAX_DEPTH: usize = 6;
const PER_CATEGORY: usize = 60;
const MIN_SUPPORTED_CASES: usize = 2000;
const MIN_PER_CATEGORY: usize = 200;
const MIN_UNSUPPORTED: usize = 300;
const ARITH_OPS: [BinaryOp; 3] = [BinaryOp::Add, BinaryOp::Sub, BinaryOp::Mul];
const CMP_OPS: [BinaryOp; 6] = [
    BinaryOp::Eq,
    BinaryOp::NotEq,
    BinaryOp::Lt,
    BinaryOp::LtEq,
    BinaryOp::Gt,
    BinaryOp::GtEq,
];
const INT_POOL: [i64; 13] = [
    i64::MIN,
    i64::MIN + 1,
    -1,
    0,
    1,
    2,
    4,
    1 << 53,
    (1 << 53) + 1,
    i64::MAX - 1,
    i64::MAX,
    -(1 << 53),
    3,
];

fn special_floats() -> Vec<f64> {
    vec![
        0.0,
        -0.0,
        1.0,
        -1.0,
        2.0,
        0.5,
        f64::INFINITY,
        f64::NEG_INFINITY,
        f64::MAX,
        f64::MIN,
        f64::MIN_POSITIVE,
        f64::from_bits(1),
        f64::from_bits(1 | (1 << 63)),
        f64::NAN,
        -f64::NAN,
        f64::from_bits(0x7ff0_0000_0000_0001),
        f64::from_bits(0xfff0_0000_0000_0042),
        f64::from_bits(0x7ff8_dead_beef_0001),
        9_007_199_254_740_992.0,
        9_223_372_036_854_775_808.0,
        -9_223_372_036_854_775_808.0,
    ]
}

fn test_schema() -> SchemaRef {
    schema_ref(Schema::new(vec![
        Field::new("i0", DataType::Int64, true),
        Field::new("i1", DataType::Int64, true),
        Field::new("i2", DataType::Int64, true),
        Field::new("f0", DataType::Float64, true),
        Field::new("f1", DataType::Float64, true),
        Field::new("f2", DataType::Float64, true),
        Field::new("s0", DataType::Utf8, true),
    ]))
}

fn col_of(schema: &Schema, name: &str) -> Expr {
    col(schema, name).expect("schema column")
}

fn bin(l: Expr, op: BinaryOp, r: Expr) -> Expr {
    binary(l, op, r).expect("well-typed test expression")
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Ty {
    Int,
    Float,
}

impl Ty {
    fn dt(self) -> DataType {
        match self {
            Ty::Int => DataType::Int64,
            Ty::Float => DataType::Float64,
        }
    }
    fn pick(rng: &mut StdRng) -> Ty {
        if rng.gen_bool(0.5) {
            Ty::Int
        } else {
            Ty::Float
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Category {
    IntArith,
    FloatArith,
    Mixed,
    IntCmp,
    FloatCmp,
}

const CATEGORIES: [Category; 5] = [
    Category::IntArith,
    Category::FloatArith,
    Category::Mixed,
    Category::IntCmp,
    Category::FloatCmp,
];

fn cat_index(c: Category) -> usize {
    CATEGORIES
        .iter()
        .position(|&x| x == c)
        .expect("listed category")
}

#[derive(Default)]
struct Scan {
    int: bool,
    float: bool,
    cast: bool,
}

fn scan(e: &Expr, s: &mut Scan) {
    match e {
        Expr::Column { data_type, .. } => note(s, *data_type),
        Expr::Literal(v) => note(s, v.data_type()),
        Expr::Binary { left, right, .. } => {
            scan(left, s);
            scan(right, s);
        }
        Expr::Unary { expr, .. } => scan(expr, s),
        Expr::Cast { expr, .. } => {
            s.cast = true;
            scan(expr, s);
        }
        _ => {}
    }
}

fn note(s: &mut Scan, dt: DataType) {
    match dt {
        DataType::Int64 => s.int = true,
        DataType::Float64 => s.float = true,
        _ => {}
    }
}

/// Category of a supported root. Leaves (bare column or literal) are not counted.
fn category(e: &Expr) -> Option<Category> {
    if matches!(e, Expr::Column { .. } | Expr::Literal(_)) {
        return None;
    }
    let mut s = Scan::default();
    scan(e, &mut s);
    let mixed = s.cast || (s.int && s.float);
    if let Expr::Binary { op, .. } = e {
        if CMP_OPS.contains(op) {
            return Some(if mixed {
                Category::Mixed
            } else if s.float {
                Category::FloatCmp
            } else {
                Category::IntCmp
            });
        }
    }
    Some(match (e.return_type(), mixed) {
        (_, true) => Category::Mixed,
        (DataType::Float64, false) => Category::FloatArith,
        _ => Category::IntArith,
    })
}

/// Seeded typed-tree generator. Every tree is well typed and built with the public constructors.
struct Gen {
    rng: StdRng,
    int_cols: Vec<Expr>,
    float_cols: Vec<Expr>,
    utf8_col: Expr,
    floats: Vec<f64>,
}

impl Gen {
    fn new(rng: StdRng, schema: &Schema) -> Self {
        Gen {
            rng,
            int_cols: ["i0", "i1", "i2"]
                .iter()
                .map(|n| col_of(schema, n))
                .collect(),
            float_cols: ["f0", "f1", "f2"]
                .iter()
                .map(|n| col_of(schema, n))
                .collect(),
            utf8_col: col_of(schema, "s0"),
            floats: special_floats(),
        }
    }

    fn arith_op(&mut self) -> BinaryOp {
        ARITH_OPS[self.rng.gen_range(0..ARITH_OPS.len())]
    }

    fn cmp_op(&mut self) -> BinaryOp {
        CMP_OPS[self.rng.gen_range(0..CMP_OPS.len())]
    }

    fn column(&mut self, ty: Ty) -> Expr {
        let pool = match ty {
            Ty::Int => &self.int_cols,
            Ty::Float => &self.float_cols,
        };
        pool[self.rng.gen_range(0..pool.len())].clone()
    }

    fn literal(&mut self, ty: Ty) -> Expr {
        match ty {
            Ty::Int => lit(ScalarValue::Int64(
                INT_POOL[self.rng.gen_range(0..INT_POOL.len())],
            )),
            Ty::Float => lit(ScalarValue::Float64(
                self.floats[self.rng.gen_range(0..self.floats.len())],
            )),
        }
    }

    fn leaf(&mut self, ty: Ty) -> Expr {
        if self.rng.gen_bool(0.6) {
            self.column(ty)
        } else {
            self.literal(ty)
        }
    }

    /// Operand types for an arithmetic node of result type `ty`. Float results may mix in Int64.
    fn operand_types(&mut self, ty: Ty) -> (Ty, Ty) {
        match ty {
            Ty::Int => (Ty::Int, Ty::Int),
            Ty::Float => [
                (Ty::Float, Ty::Float),
                (Ty::Float, Ty::Int),
                (Ty::Int, Ty::Float),
            ][self.rng.gen_range(0..3)],
        }
    }

    fn arith(&mut self, ty: Ty, depth: usize) -> (Expr, bool) {
        let (lt, rt) = self.operand_types(ty);
        let (l, a) = self.num(lt, depth - 1);
        let (r, b) = self.num(rt, depth - 1);
        let op = self.arith_op();
        (bin(l, op, r), a && b)
    }

    /// Numeric expression of type `ty`. The bool is true when no unsupported shape was injected.
    fn num(&mut self, ty: Ty, depth: usize) -> (Expr, bool) {
        if depth == 0 || self.rng.gen_bool(0.15) {
            return (self.leaf(ty), true);
        }
        let r = self.rng.gen_range(0..100);
        match r {
            35..=44 => {
                let (e, a) = self.num(ty, depth - 1);
                (unary(UnaryOp::Neg, e).expect("neg of numeric"), a)
            }
            45..=52 if ty == Ty::Float => {
                let (e, a) = self.num(Ty::Int, depth - 1);
                (cast(e, DataType::Float64).expect("int to float cast"), a)
            }
            53..=62 => {
                // Same subtree on both sides: x - x, x * x, and NaN/inf cases that folding must not break.
                let (l, a) = self.num(ty, depth - 1);
                let op = self.arith_op();
                (bin(l.clone(), op, l), a)
            }
            63..=66 => {
                let op = self.arith_op();
                let l = self.literal(ty);
                let r = self.literal(ty);
                (bin(l, op, r), true)
            }
            67..=70 => {
                let (e, a) = self.num(ty, depth - 1);
                let inner = unary(UnaryOp::Neg, e).expect("neg");
                (unary(UnaryOp::Neg, inner).expect("neg"), a)
            }
            71..=74 => (self.unsupported_num(ty, depth - 1), false),
            _ => self.arith(ty, depth),
        }
    }

    /// Numeric subtree whose shape the JIT must refuse (Div, Mod, Case, Float->Int cast, Utf8, NULL literal).
    fn unsupported_num(&mut self, ty: Ty, depth: usize) -> Expr {
        match self.rng.gen_range(0..5) {
            0 => {
                let op = if self.rng.gen_bool(0.5) {
                    BinaryOp::Div
                } else {
                    BinaryOp::Mod
                };
                let (l, _) = self.num(ty, depth);
                let (r, _) = self.num(ty, depth);
                bin(l, op, r)
            }
            1 => {
                let c = self.comparison().0;
                let t = self.leaf(ty);
                let e = self.leaf(ty);
                case(c, t, e).expect("case over numeric branches")
            }
            2 => {
                let (e, _) = self.num(Ty::Float, depth);
                let as_int = cast(e, DataType::Int64).expect("float to int cast");
                match ty {
                    Ty::Int => as_int,
                    Ty::Float => cast(as_int, DataType::Float64).expect("int to float cast"),
                }
            }
            3 => Expr::Binary {
                left: Box::new(self.utf8_col.clone()),
                op: self.arith_op(),
                right: Box::new(self.leaf(ty)),
                return_type: ty.dt(),
            },
            _ => {
                let op = self.arith_op();
                let r = self.leaf(ty);
                bin(lit(ScalarValue::Null(ty.dt())), op, r)
            }
        }
    }

    /// Comparison at the root only (its Boolean result cannot feed arithmetic).
    fn comparison(&mut self) -> (Expr, bool) {
        let lt = Ty::pick(&mut self.rng);
        let rt = Ty::pick(&mut self.rng);
        let (l, a) = self.num(lt, 4);
        let (r, b) = self.num(rt, 4);
        let op = self.cmp_op();
        (bin(l, op, r), a && b)
    }

    /// One candidate root. The bool says whether the JIT is expected to accept it.
    fn candidate(&mut self) -> (Expr, bool) {
        match self.rng.gen_range(0..100) {
            0..=44 => {
                let ty = Ty::pick(&mut self.rng);
                self.num(ty, MAX_DEPTH - 1)
            }
            45..=79 => self.comparison(),
            80..=84 => {
                let ty = Ty::pick(&mut self.rng);
                let (e, _) = self.num(ty, 3);
                let op = if self.rng.gen_bool(0.5) {
                    UnaryOp::IsNull
                } else {
                    UnaryOp::IsNotNull
                };
                (unary(op, e).expect("null test"), false)
            }
            85..=87 => {
                let (c, _) = self.comparison();
                (unary(UnaryOp::Not, c).expect("not of boolean"), false)
            }
            88..=91 => {
                let ty = Ty::pick(&mut self.rng);
                let (e, _) = self.num(ty, 2);
                let list = vec![self.leaf(ty), self.leaf(ty)];
                let negated = self.rng.gen_bool(0.5);
                (
                    Expr::InList {
                        expr: Box::new(e),
                        list,
                        negated,
                    },
                    false,
                )
            }
            92..=95 => {
                let (c1, _) = self.comparison();
                let (c2, _) = self.comparison();
                let op = if self.rng.gen_bool(0.5) {
                    BinaryOp::And
                } else {
                    BinaryOp::Or
                };
                (bin(c1, op, c2), false)
            }
            _ => {
                let (c, _) = self.comparison();
                let ty = Ty::pick(&mut self.rng);
                let (t, _) = self.num(ty, 2);
                let (e, _) = self.num(ty, 2);
                (case(c, t, e).expect("case"), false)
            }
        }
    }
}

#[derive(Clone, Copy)]
enum Mask {
    None,
    AllValid,
    AllNull,
    Random,
}

const MASKS: [Mask; 4] = [Mask::None, Mask::AllValid, Mask::AllNull, Mask::Random];

fn validity(rng: &mut StdRng, mask: Mask, n: usize) -> Option<Vec<bool>> {
    match mask {
        Mask::None => None,
        Mask::AllValid => Some(vec![true; n]),
        Mask::AllNull => Some(vec![false; n]),
        Mask::Random => Some((0..n).map(|_| rng.gen_bool(0.5)).collect()),
    }
}

fn rand_int(rng: &mut StdRng) -> i64 {
    if rng.gen_bool(0.5) {
        INT_POOL[rng.gen_range(0..INT_POOL.len())]
    } else {
        rng.gen()
    }
}

fn rand_float(rng: &mut StdRng, floats: &[f64]) -> f64 {
    match rng.gen_range(0..4) {
        0 | 1 => floats[rng.gen_range(0..floats.len())],
        2 => rng.gen_range(-1e9..1e9),
        _ => f64::from_bits(rng.gen()),
    }
}

/// One batch of `n` rows. `variant` picks the validity layout (see `VARIANTS`).
fn make_batch(rng: &mut StdRng, schema: &SchemaRef, n: usize, variant: usize) -> Batch {
    let floats = special_floats();
    let columns = schema
        .fields
        .iter()
        .enumerate()
        .map(|(j, f)| {
            let mask = match variant {
                0 => Mask::None,
                1 => Mask::AllValid,
                2 => Mask::AllNull,
                3 => MASKS[rng.gen_range(0..4)],
                _ => MASKS[j % 4],
            };
            let mut c = match f.data_type {
                DataType::Int64 => Column::from_i64((0..n).map(|_| rand_int(rng)).collect()),
                DataType::Float64 => {
                    Column::from_f64((0..n).map(|_| rand_float(rng, &floats)).collect())
                }
                _ => {
                    let strs: Vec<String> = (0..n)
                        .map(|_| match rng.gen_range(0..3) {
                            0 => String::new(),
                            1 => "abc".to_string(),
                            _ => rng.gen_range(-5..5).to_string(),
                        })
                        .collect();
                    Column::from_strs(&strs)
                }
            };
            c.validity = validity(rng, mask, n);
            c
        })
        .collect();
    Batch::try_new(schema.clone(), columns).expect("generated batch matches schema")
}

/// `batches[size_index][variant]`.
fn make_batches(rng: &mut StdRng, schema: &SchemaRef) -> Vec<Vec<Batch>> {
    SIZES
        .iter()
        .map(|&n| {
            (0..VARIANTS)
                .map(|v| make_batch(rng, schema, n, v))
                .collect()
        })
        .collect()
}

fn show(c: &Column, i: usize) -> String {
    if !c.is_valid(i) {
        return "NULL".to_string();
    }
    match &c.values {
        Values::Float64(v) => format!("{} [bits {:#018x}]", v[i], v[i].to_bits()),
        _ => format!("{:?}", c.get(i)),
    }
}

enum Outcome {
    /// Both sides returned columns that agree on every row. `nan_rows` counts documented NaN-payload rows.
    Compared {
        nan_rows: usize,
    },
    BothErr,
}

/// Compares the kernel against `eval` on one batch. Err is a divergence.
fn diff_case(expr: &Expr, kernel: &CompiledKernel, batch: &Batch) -> Result<Outcome, String> {
    let (want, got) = match (eval(expr, batch), kernel.run(batch)) {
        (Err(_), Err(_)) => return Ok(Outcome::BothErr),
        (Err(we), Ok(_)) => {
            return Err(format!(
                "eval returns Err ({we}) but the JIT returns Ok (silent success)"
            ))
        }
        (Ok(_), Err(ge)) => return Err(format!("eval returns Ok but the JIT returns Err ({ge})")),
        (Ok(w), Ok(g)) => (w, g),
    };
    if want.data_type() != got.data_type() {
        return Err(format!(
            "data type: eval {:?} vs jit {:?}",
            want.data_type(),
            got.data_type()
        ));
    }
    if want.len() != got.len() {
        return Err(format!("length: eval {} vs jit {}", want.len(), got.len()));
    }
    for i in 0..want.len() {
        if want.is_valid(i) != got.is_valid(i) {
            return Err(format!(
                "row {i} validity: eval {} vs jit {} (eval {}, jit {})",
                want.is_valid(i),
                got.is_valid(i),
                show(&want, i),
                show(&got, i)
            ));
        }
    }
    let mut nan_rows = 0;
    for i in 0..want.len() {
        if !want.is_valid(i) {
            continue;
        }
        let equal = match (&want.values, &got.values) {
            (Values::Int64(a), Values::Int64(b)) => a[i] == b[i],
            (Values::Bool(a), Values::Bool(b)) => a[i] == b[i],
            (Values::Float64(a), Values::Float64(b)) => {
                if a[i].to_bits() == b[i].to_bits() {
                    true
                } else if a[i].is_nan() && b[i].is_nan() {
                    nan_rows += 1;
                    true
                } else {
                    false
                }
            }
            _ => false,
        };
        if !equal {
            return Err(format!(
                "row {i}: eval {} vs jit {}",
                show(&want, i),
                show(&got, i)
            ));
        }
    }
    Ok(Outcome::Compared { nan_rows })
}

/// Minimal repro: the first diverging row, then descend into child subtrees while they still diverge.
fn minimise(expr: &Expr, batch: &Batch) -> String {
    let Ok(kernel) = compile(expr) else {
        return format!("compile failed: {expr:?}");
    };
    let Some(row) =
        (0..batch.num_rows).find(|&i| diff_case(expr, &kernel, &batch.slice(i, 1)).is_err())
    else {
        return format!("no single-row divergence; expr {expr:?}");
    };
    let one = batch.slice(row, 1);
    let mut cur = expr.clone();
    loop {
        let next = children(&cur).into_iter().find_map(|c| {
            let k = compile(c).ok()?;
            diff_case(c, &k, &one).is_err().then(|| c.clone())
        });
        match next {
            Some(c) => cur = c,
            None => break,
        }
    }
    let detail = match compile(&cur) {
        Ok(k) => match diff_case(&cur, &k, &one) {
            Err(msg) => msg,
            Ok(_) => "no longer diverges".to_string(),
        },
        Err(e) => format!("compile: {e}"),
    };
    let inputs: Vec<String> = one
        .columns
        .iter()
        .enumerate()
        .map(|(j, c)| format!("c{j}={}", show(c, 0)))
        .collect();
    format!(
        "minimal expr: {cur:?}\n    inputs (row {row}): {}\n    {detail}",
        inputs.join(", ")
    )
}

fn children(e: &Expr) -> Vec<&Expr> {
    match e {
        Expr::Binary { left, right, .. } => vec![left, right],
        Expr::Unary { expr, .. } | Expr::Cast { expr, .. } => vec![expr],
        _ => vec![],
    }
}

#[derive(Default)]
struct Stats {
    cases: [usize; 5],
    exprs: [usize; 5],
    supported_cases: usize,
    unsupported_checked: usize,
    both_err: usize,
    nan_rows: usize,
    nan_cases: usize,
    /// Same comparisons repeated with `Codegen::Vector`; not counted in the fields above.
    vector_cases: usize,
    vector_nan_rows: usize,
    max_depth: usize,
    depth_at_max: usize,
    divergences: Vec<String>,
}

#[test]
fn random_expressions_match_eval_over_seeded_batches() {
    let schema = test_schema();
    let mut setup = StdRng::seed_from_u64(SEED);
    let batches = make_batches(&mut setup, &schema);
    let mut gen = Gen::new(StdRng::seed_from_u64(SEED ^ 1), &schema);
    let mut stats = Stats::default();

    let mut attempts = 0usize;
    while CATEGORIES
        .iter()
        .any(|&c| stats.exprs[cat_index(c)] < PER_CATEGORY)
        || stats.unsupported_checked < MIN_UNSUPPORTED
    {
        attempts += 1;
        assert!(
            attempts < 500_000,
            "generator did not fill quotas: exprs={:?} unsupported={}",
            stats.exprs,
            stats.unsupported_checked
        );
        let (expr, expect_supported) = gen.candidate();
        let depth = expr.depth();
        if depth > MAX_DEPTH {
            continue;
        }
        stats.max_depth = stats.max_depth.max(depth);
        if depth == MAX_DEPTH {
            stats.depth_at_max += 1;
        }
        let js = supports(&expr);
        if !expect_supported {
            stats.unsupported_checked += 1;
            if js {
                stats.divergences.push(format!(
                    "supports() true for an unsupported shape: {expr:?}"
                ));
            } else if compile(&expr).is_ok() {
                stats
                    .divergences
                    .push(format!("compile() Ok for an unsupported shape: {expr:?}"));
            }
            continue;
        }
        if !js {
            stats
                .divergences
                .push(format!("supports() false for a supported shape: {expr:?}"));
            continue;
        }
        let cat = category(&expr);
        if let Some(c) = cat {
            if stats.exprs[cat_index(c)] >= PER_CATEGORY {
                continue;
            }
        }
        if matches!(expr, Expr::Column { .. } | Expr::Literal(_)) {
            continue;
        }
        let kernel = match compile(&expr) {
            Ok(k) => k,
            Err(e) => {
                stats.divergences.push(format!(
                    "compile() failed on a supported shape ({e}): {expr:?}"
                ));
                continue;
            }
        };
        let vector = match compile_with(&expr, Codegen::Vector) {
            Ok(k) => k,
            Err(e) => {
                stats.divergences.push(format!(
                    "compile_with(Vector) failed on a supported shape ({e}): {expr:?}"
                ));
                continue;
            }
        };
        if let Some(c) = cat {
            stats.exprs[cat_index(c)] += 1;
        }
        for si in 0..SIZES.len() {
            let first = gen.rng.gen_range(0..VARIANTS);
            let second = (first + 1 + gen.rng.gen_range(0..VARIANTS - 1)) % VARIANTS;
            for v in [first, second] {
                let batch = &batches[si][v];
                match diff_case(&expr, &vector, batch) {
                    Ok(Outcome::Compared { nan_rows }) => {
                        stats.vector_cases += 1;
                        stats.vector_nan_rows += nan_rows;
                    }
                    Ok(Outcome::BothErr) => {}
                    Err(msg) => stats.divergences.push(format!(
                        "vector codegen, size {} variant {v}: {msg}\n    {expr:?}",
                        SIZES[si]
                    )),
                }
                match diff_case(&expr, &kernel, batch) {
                    Ok(Outcome::Compared { nan_rows }) => {
                        stats.supported_cases += 1;
                        if let Some(c) = cat {
                            stats.cases[cat_index(c)] += 1;
                        }
                        if nan_rows > 0 {
                            stats.nan_cases += 1;
                            stats.nan_rows += nan_rows;
                        }
                    }
                    Ok(Outcome::BothErr) => stats.both_err += 1,
                    Err(msg) => stats.divergences.push(format!(
                        "size {} variant {v}: {msg}\n    {}",
                        SIZES[si],
                        minimise(&expr, batch)
                    )),
                }
            }
        }
    }

    println!(
        "jit_diff: depth max={} trees at depth {MAX_DEPTH}={}",
        stats.max_depth, stats.depth_at_max
    );
    println!(
        "jit_diff: supported cases compared={} unsupported shapes checked={} eval-and-jit both Err={}",
        stats.supported_cases, stats.unsupported_checked, stats.both_err
    );
    for c in CATEGORIES {
        let i = cat_index(c);
        println!(
            "jit_diff: {:<12} exprs={:<4} cases={}",
            format!("{c:?}"),
            stats.exprs[i],
            stats.cases[i]
        );
    }
    println!(
        "jit_diff: documented NaN-payload-only difference: rows={} cases={}",
        stats.nan_rows, stats.nan_cases
    );
    println!(
        "jit_diff: vector codegen cases compared={} NaN-payload-only rows={}",
        stats.vector_cases, stats.vector_nan_rows
    );
    println!("jit_diff: divergences={}", stats.divergences.len());

    assert!(
        stats.divergences.is_empty(),
        "{} divergence(s):\n{}",
        stats.divergences.len(),
        stats
            .divergences
            .iter()
            .take(5)
            .cloned()
            .collect::<Vec<_>>()
            .join("\n")
    );
    assert_eq!(
        stats.vector_cases, stats.supported_cases,
        "vector codegen must be compared on every case the scalar kernel is"
    );
    assert!(
        stats.supported_cases >= MIN_SUPPORTED_CASES,
        "only {} supported cases",
        stats.supported_cases
    );
    for c in CATEGORIES {
        let n = stats.cases[cat_index(c)];
        assert!(n >= MIN_PER_CATEGORY, "{c:?}: only {n} cases");
    }
}

fn shared_expr(schema: &Schema) -> Expr {
    bin(
        bin(col_of(schema, "i0"), BinaryOp::Add, col_of(schema, "f1")),
        BinaryOp::Mul,
        bin(
            col_of(schema, "f2"),
            BinaryOp::Sub,
            lit(ScalarValue::Float64(2.5)),
        ),
    )
}

#[test]
fn one_kernel_runs_on_fifty_batches() {
    let schema = test_schema();
    let e = shared_expr(&schema);
    let kernel = compile(&e).expect("supported");
    let mut rng = StdRng::seed_from_u64(SEED ^ 50);
    for k in 0..50 {
        let batch = make_batch(&mut rng, &schema, SIZES[k % SIZES.len()], k % VARIANTS);
        if let Err(msg) = diff_case(&e, &kernel, &batch) {
            panic!("batch {k}: {msg}\n{}", minimise(&e, &batch));
        }
    }
}

#[test]
fn one_kernel_shared_across_eight_threads() {
    let schema = test_schema();
    let e = shared_expr(&schema);
    let kernel = compile(&e).expect("supported");
    let mut rng = StdRng::seed_from_u64(SEED ^ 8);
    let work: Vec<Vec<Batch>> = (0..8)
        .map(|t| {
            (0..6)
                .map(|k| {
                    make_batch(
                        &mut rng,
                        &schema,
                        SIZES[(t * 6 + k) % SIZES.len()],
                        k % VARIANTS,
                    )
                })
                .collect()
        })
        .collect();
    let failures: Vec<String> = std::thread::scope(|s| {
        let handles: Vec<_> = work
            .iter()
            .map(|batches| {
                let (kernel, e) = (&kernel, &e);
                s.spawn(move || {
                    batches
                        .iter()
                        .filter_map(|b| diff_case(e, kernel, b).err())
                        .collect::<Vec<String>>()
                })
            })
            .collect();
        handles
            .into_iter()
            .flat_map(|h| h.join().expect("worker thread panicked"))
            .collect()
    });
    assert!(failures.is_empty(), "{failures:?}");
}

#[test]
fn batch_with_swapped_column_types_is_err() {
    let schema = test_schema();
    // i0 is Int64, f0 is Float64 in the compiled schema.
    let e = bin(col_of(&schema, "i0"), BinaryOp::Add, col_of(&schema, "f0"));
    let kernel = compile(&e).expect("supported");

    // Same indices, but columns 0 and 3 swapped in type.
    let swapped = schema_ref(Schema::new(vec![
        Field::new("i0", DataType::Float64, true),
        Field::new("i1", DataType::Int64, true),
        Field::new("i2", DataType::Int64, true),
        Field::new("f0", DataType::Int64, true),
        Field::new("f1", DataType::Float64, true),
        Field::new("f2", DataType::Float64, true),
        Field::new("s0", DataType::Utf8, true),
    ]));
    let mut rng = StdRng::seed_from_u64(SEED ^ 9);
    let batch = make_batch(&mut rng, &swapped, 9, 0);

    match kernel.run(&batch) {
        Err(EngineError::Type(_)) => {}
        other => panic!("jit run on swapped types: expected Type error, got {other:?}"),
    }
    assert!(
        eval(&e, &batch).is_err(),
        "eval must also reject swapped types"
    );
}

#[test]
fn two_hundred_kernels_compiled_and_dropped() {
    let schema = test_schema();
    let mut rng = StdRng::seed_from_u64(SEED ^ 200);
    let batch = make_batch(&mut rng, &schema, 65, 4);
    for k in 0..200usize {
        let e = if k % 2 == 0 {
            bin(
                col_of(&schema, ["i0", "i1", "i2"][k % 3]),
                BinaryOp::Mul,
                lit(ScalarValue::Int64(k as i64 * 0x9e37)),
            )
        } else {
            bin(
                col_of(&schema, ["f0", "f1", "f2"][k % 3]),
                BinaryOp::Add,
                lit(ScalarValue::Float64(k as f64 - 100.0)),
            )
        };
        let kernel = compile(&e).expect("supported");
        if let Err(msg) = diff_case(&e, &kernel, &batch) {
            panic!("kernel {k}: {msg}");
        }
        drop(kernel);
    }
}

/// The harness itself must be able to fail: a kernel for one expression compared against eval of
/// another must be reported, and the minimiser must produce a repro from it.
#[test]
fn harness_detects_a_wrong_kernel() {
    let schema = test_schema();
    let mut rng = StdRng::seed_from_u64(SEED ^ 77);
    let batch = make_batch(&mut rng, &schema, 64, 1);
    let wrong = bin(
        col_of(&schema, "i0"),
        BinaryOp::Add,
        lit(ScalarValue::Int64(1)),
    );
    let claimed = bin(
        col_of(&schema, "i0"),
        BinaryOp::Add,
        lit(ScalarValue::Int64(2)),
    );
    let kernel = compile(&wrong).expect("supported");
    assert!(
        diff_case(&claimed, &kernel, &batch).is_err(),
        "a wrong kernel was not reported"
    );

    // NaN against a non-NaN value is a real divergence, not the documented payload difference.
    let f_wrong = bin(
        col_of(&schema, "f0"),
        BinaryOp::Mul,
        lit(ScalarValue::Float64(1.0)),
    );
    let f_claimed = bin(
        col_of(&schema, "f0"),
        BinaryOp::Mul,
        lit(ScalarValue::Float64(2.0)),
    );
    let f_kernel = compile(&f_wrong).expect("supported");
    assert!(diff_case(&f_claimed, &f_kernel, &batch).is_err());
}

/// Regression: cranelift-jit does not unmap code pages on drop; CompiledKernel must free them.
#[test]
#[cfg(target_os = "macos")]
fn dropped_kernels_release_their_code_pages() {
    fn rss_kb() -> u64 {
        let out = std::process::Command::new("ps")
            .args(["-o", "rss=", "-p", &std::process::id().to_string()])
            .output()
            .expect("ps runs");
        String::from_utf8_lossy(&out.stdout)
            .trim()
            .parse()
            .expect("rss is a number")
    }
    let schema = Schema::new(vec![
        Field::new("a", DataType::Int64, false),
        Field::new("b", DataType::Int64, false),
    ]);
    let e = binary(
        col(&schema, "a").unwrap(),
        BinaryOp::Add,
        col(&schema, "b").unwrap(),
    )
    .unwrap();
    for _ in 0..200 {
        drop(compile(&e).unwrap());
    }
    let before = rss_kb();
    for _ in 0..20_000 {
        drop(compile(&e).unwrap());
    }
    let grown = rss_kb().saturating_sub(before);
    // The leak was 16 kB per compile (320 MB here); allow 40 MB of allocator noise.
    assert!(
        grown < 40_000,
        "RSS grew {grown} kB over 20000 compile+drop cycles"
    );
}
