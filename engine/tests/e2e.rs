//! End-to-end tests through the public API: CSV loading, plan execution, determinism, joins,
//! aggregates, spilling under a tiny memory budget, and error cases.
use native_engine::catalog::{load_csv, Catalog};
use native_engine::column::{Batch, Column};
use native_engine::eval::{binary, col, lit};
use native_engine::exec::{execute, ExecContext};
use native_engine::expr::{BinaryOp, Expr};
use native_engine::memory::MemoryBudget;
use native_engine::plan::{AggExpr, AggFunc, JoinType, LogicalPlan, SortField};
use native_engine::types::{DataType, EngineError, Field, ScalarValue, Schema, SchemaRef};
use rand::{rngs::StdRng, Rng, SeedableRng};
use std::path::PathBuf;
use std::sync::Arc;

// ---------- helpers ----------

fn ctx(threads: usize, morsel_rows: usize) -> ExecContext {
    ExecContext { threads, morsel_rows, budget: MemoryBudget::new(1 << 30), spill_dir: std::env::temp_dir() }
}

fn schema(fields: &[(&str, DataType, bool)]) -> SchemaRef {
    Arc::new(Schema::new(fields.iter().map(|(n, t, nl)| Field::new(n, *t, *nl)).collect()))
}

fn i64_col(v: &[Option<i64>]) -> Column {
    let c = Column::from_i64(v.iter().map(|x| x.unwrap_or(0)).collect());
    if v.iter().all(|x| x.is_some()) {
        c
    } else {
        c.with_validity(v.iter().map(|x| x.is_some()).collect())
    }
}

fn str_col(v: &[Option<&str>]) -> Column {
    let vals: Vec<&str> = v.iter().map(|x| x.unwrap_or("")).collect();
    let c = Column::from_strs(&vals);
    if v.iter().all(|x| x.is_some()) {
        c
    } else {
        c.with_validity(v.iter().map(|x| x.is_some()).collect())
    }
}

/// Split full columns into batches of at most `batch_rows` rows.
fn batches(schema: &SchemaRef, cols: &[Column], batch_rows: usize) -> Vec<Batch> {
    let n = cols.first().map_or(0, |c| c.len());
    (0..n)
        .step_by(batch_rows)
        .map(|off| {
            let len = batch_rows.min(n - off);
            let columns = cols.iter().map(|c| c.slice(off, len)).collect();
            Batch::try_new(schema.clone(), columns).expect("valid batch")
        })
        .collect()
}

fn catalog_with(name: &str, schema: &SchemaRef, batches: Vec<Batch>) -> Catalog {
    let mut cat = Catalog::new();
    cat.register(name, schema.clone(), batches);
    cat
}

fn scan(name: &str, schema: &SchemaRef) -> Arc<LogicalPlan> {
    Arc::new(LogicalPlan::Scan { table_name: name.into(), schema: schema.clone(), projection: None, projected_schema: schema.clone() })
}

fn c(schema: &SchemaRef, name: &str) -> Expr {
    col(schema, name).expect("column exists")
}

fn lit_i(v: i64) -> Expr {
    lit(ScalarValue::Int64(v))
}

fn bin(l: Expr, op: BinaryOp, r: Expr) -> Expr {
    binary(l, op, r).expect("well-typed expression")
}

fn rows(batches: &[Batch]) -> Vec<Vec<ScalarValue>> {
    batches.iter().flat_map(|b| (0..b.num_rows).map(move |i| b.row(i))).collect()
}

fn sorted(mut r: Vec<Vec<ScalarValue>>) -> Vec<Vec<ScalarValue>> {
    r.sort_by_key(|row| format!("{row:?}"));
    r
}

fn sorted_rows(batches: &[Batch]) -> Vec<Vec<ScalarValue>> {
    sorted(rows(batches))
}

fn s(v: &str) -> ScalarValue {
    ScalarValue::Utf8(v.to_string())
}

fn int(v: i64) -> ScalarValue {
    ScalarValue::Int64(v)
}

fn null_i() -> ScalarValue {
    ScalarValue::Null(DataType::Int64)
}

fn null_s() -> ScalarValue {
    ScalarValue::Null(DataType::Utf8)
}

/// Temporary CSV file, removed on drop.
struct TempCsv(PathBuf);

impl TempCsv {
    fn new(tag: &str, content: &str) -> Self {
        let p = std::env::temp_dir().join(format!("ndq_e2e_{}_{tag}.csv", std::process::id()));
        std::fs::write(&p, content).expect("write temp csv");
        TempCsv(p)
    }
}

impl Drop for TempCsv {
    fn drop(&mut self) {
        let _ = std::fs::remove_file(&self.0);
    }
}

fn load(tag: &str, content: &str, batch_rows: usize) -> (SchemaRef, Vec<Batch>) {
    let f = TempCsv::new(tag, content);
    load_csv(&f.0, batch_rows).expect("csv loads")
}

// ---------- CSV loader ----------

#[test]
fn csv_infers_types_over_all_rows() {
    let (sch, _) = load("types", "id,price,flag,name\n1,1.5,true,a\n2,2,FALSE,b\n3,3.25,true,c\n", 10);
    let got: Vec<(&str, DataType, bool)> = sch.fields.iter().map(|f| (f.name.as_str(), f.data_type, f.nullable)).collect();
    assert_eq!(
        got,
        vec![
            ("id", DataType::Int64, false),
            ("price", DataType::Float64, false),
            ("flag", DataType::Boolean, false),
            ("name", DataType::Utf8, false),
        ]
    );
}

#[test]
fn csv_late_text_value_widens_column_to_utf8() {
    let (sch, batches) = load("widen", "x\n1\n2\nabc\n", 10);
    assert_eq!(sch.fields[0].data_type, DataType::Utf8);
    let got: Vec<ScalarValue> = rows(&batches).into_iter().map(|r| r[0].clone()).collect();
    assert_eq!(got, vec![s("1"), s("2"), s("abc")]);
}

#[test]
fn csv_empty_fields_are_null_and_make_column_nullable() {
    let (sch, batches) = load("nulls", "a,b\n1,\n,x\n", 10);
    assert!(sch.fields[0].nullable && sch.fields[1].nullable);
    assert_eq!(rows(&batches), vec![vec![int(1), null_s()], vec![null_i(), s("x")]]);
}

#[test]
fn csv_quoted_fields_keep_commas_escaped_quotes_and_newlines() {
    let content = "k,text\n1,\"a, b\"\n2,\"say \"\"hi\"\"\"\n3,\"line1\nline2\"\n4,plain\n";
    let (_, batches) = load("quoted", content, 10);
    let texts: Vec<ScalarValue> = rows(&batches).into_iter().map(|r| r[1].clone()).collect();
    assert_eq!(texts, vec![s("a, b"), s("say \"hi\""), s("line1\nline2"), s("plain")]);
}

#[test]
fn csv_crlf_line_endings_and_missing_final_newline() {
    let (_, batches) = load("crlf", "a,b\r\n1,x\r\n2,y", 10);
    assert_eq!(rows(&batches), vec![vec![int(1), s("x")], vec![int(2), s("y")]]);
}

#[test]
fn csv_ragged_row_is_io_error_with_line_number() {
    let f = TempCsv::new("ragged", "a,b\n1,2\n3\n");
    match load_csv(&f.0, 10) {
        Err(EngineError::Io(m)) => {
            assert!(m.contains("line 3"), "message was: {m}");
            assert!(m.contains("expected 2 fields, found 1"), "message was: {m}");
        }
        other => panic!("expected Io error, got {other:?}"),
    }
}

#[test]
fn csv_unterminated_quote_is_io_error() {
    let f = TempCsv::new("unterminated", "a\n\"open\n");
    match load_csv(&f.0, 10) {
        Err(EngineError::Io(m)) => assert!(m.contains("unterminated"), "message was: {m}"),
        other => panic!("expected Io error, got {other:?}"),
    }
}

#[test]
fn csv_batches_split_at_batch_rows() {
    let body: String = (0..10).map(|i| format!("{i}\n")).collect();
    let (_, batches) = load("split", &format!("v\n{body}"), 4);
    let lens: Vec<usize> = batches.iter().map(|b| b.num_rows).collect();
    assert_eq!(lens, vec![4, 4, 2]);
}

#[test]
fn csv_header_only_gives_schema_and_no_batches() {
    let (sch, batches) = load("header_only", "a,b\n", 10);
    assert_eq!(sch.fields.len(), 2);
    assert!(batches.is_empty());
}

#[test]
fn csv_nan_and_inf_words_stay_text() {
    let (sch, _) = load("nan", "v\nNaN\ninf\n", 10);
    assert_eq!(sch.fields[0].data_type, DataType::Utf8);
}

#[test]
fn csv_skips_blank_lines() {
    let (_, batches) = load("blank", "a\n1\n\n2\n", 10);
    assert_eq!(rows(&batches), vec![vec![int(1)], vec![int(2)]]);
}

#[test]
fn csv_missing_file_is_io_error() {
    let missing = std::env::temp_dir().join("ndq_e2e_does_not_exist.csv");
    assert!(matches!(load_csv(&missing, 10), Err(EngineError::Io(_))));
}

// ---------- plan errors ----------

#[test]
fn unknown_table_is_plan_error() {
    let s1 = schema(&[("a", DataType::Int64, false)]);
    let cat = Catalog::new();
    let plan = LogicalPlan::Scan { table_name: "nope".into(), schema: s1.clone(), projection: None, projected_schema: s1 };
    assert!(matches!(execute(&plan, &cat, &ctx(1, 10)), Err(EngineError::Plan(_))));
}

#[test]
fn scan_schema_differing_from_catalog_is_plan_error() {
    let registered = schema(&[("a", DataType::Int64, false)]);
    let claimed = schema(&[("a", DataType::Utf8, false)]);
    let cat = catalog_with("t", &registered, vec![]);
    let plan = LogicalPlan::Scan { table_name: "t".into(), schema: claimed.clone(), projection: None, projected_schema: claimed };
    assert!(matches!(execute(&plan, &cat, &ctx(1, 10)), Err(EngineError::Plan(_))));
}

#[test]
fn filter_with_non_boolean_predicate_is_type_error() {
    let s1 = schema(&[("k", DataType::Int64, false)]);
    let cat = catalog_with("t", &s1, batches(&s1, &[i64_col(&[Some(1), Some(2)])], 10));
    let plan = LogicalPlan::Filter { input: scan("t", &s1), predicate: c(&s1, "k") };
    assert!(matches!(execute(&plan, &cat, &ctx(1, 10)), Err(EngineError::Type(_))));
}

// ---------- scan / filter / project / sort / limit ----------

#[test]
fn scan_filter_project_over_csv_table() {
    let (sch, batches) = load("people", "id,name,score\n1,ann,10\n2,bob,\n3,cy,30\n4,dee,40\n", 2);
    let cat = catalog_with("people", &sch, batches);
    let filtered = Arc::new(LogicalPlan::Filter {
        input: scan("people", &sch),
        predicate: bin(c(&sch, "score"), BinaryOp::Gt, lit_i(15)),
    });
    let out_schema = schema(&[("name", DataType::Utf8, true), ("double", DataType::Int64, true)]);
    let plan = LogicalPlan::Projection {
        input: filtered,
        exprs: vec![c(&sch, "name"), bin(c(&sch, "score"), BinaryOp::Mul, lit_i(2))],
        schema: out_schema,
    };
    let out = execute(&plan, &cat, &ctx(2, 1)).expect("query runs");
    assert_eq!(rows(&out), vec![vec![s("cy"), int(60)], vec![s("dee"), int(80)]]);
}

#[test]
fn scan_projection_keeps_requested_columns_in_order() {
    let (sch, batches) = load("proj", "id,name,score\n1,ann,10\n2,bob,\n", 10);
    let cat = catalog_with("proj", &sch, batches);
    let projected = schema(&[("score", DataType::Int64, true), ("id", DataType::Int64, false)]);
    let plan = LogicalPlan::Scan {
        table_name: "proj".into(),
        schema: sch.clone(),
        projection: Some(vec![2, 0]),
        projected_schema: projected,
    };
    let out = execute(&plan, &cat, &ctx(1, 10)).expect("query runs");
    assert_eq!(rows(&out), vec![vec![int(10), int(1)], vec![null_i(), int(2)]]);
}

#[test]
fn sort_then_limit_with_offset_crosses_batch_boundaries() {
    let s1 = schema(&[("k", DataType::Int64, true), ("tag", DataType::Int64, false)]);
    let k = [Some(5), Some(3), None, Some(9), Some(1), Some(7), None, Some(3), Some(8), Some(2)];
    let tag: Vec<Option<i64>> = (0..10).map(Some).collect();
    let cols = [i64_col(&k), i64_col(&tag)];
    let cat = catalog_with("t", &s1, batches(&s1, &cols, 4));
    let sort = Arc::new(LogicalPlan::Sort {
        input: scan("t", &s1),
        order_by: vec![SortField { expr: c(&s1, "k"), ascending: true, nulls_first: false }],
    });

    // Full sort: ascending, nulls last, stable on ties (tag 1 before tag 7 for k=3).
    let full = execute(&sort, &cat, &ctx(2, 4)).expect("sort runs");
    let all: Vec<Vec<ScalarValue>> = rows(&full);
    assert_eq!(all.len(), 10);
    assert_eq!(all[2], vec![int(3), int(1)]);
    assert_eq!(all[3], vec![int(3), int(7)]);
    assert_eq!(all[8], vec![null_i(), int(2)]);

    // skip 2, fetch 3 must cut across the 4-row output batches.
    let limited = LogicalPlan::Limit { input: sort, skip: 2, fetch: Some(3) };
    let out = execute(&limited, &cat, &ctx(2, 4)).expect("limit runs");
    assert_eq!(rows(&out), vec![vec![int(3), int(1)], vec![int(3), int(7)], vec![int(5), int(0)]]);
}

#[test]
fn limit_skip_past_end_returns_no_rows() {
    let s1 = schema(&[("k", DataType::Int64, false)]);
    let cat = catalog_with("t", &s1, batches(&s1, &[i64_col(&[Some(1), Some(2), Some(3)])], 2));
    let plan = LogicalPlan::Limit { input: scan("t", &s1), skip: 100, fetch: Some(5) };
    let out = execute(&plan, &cat, &ctx(1, 2)).expect("limit runs");
    assert!(rows(&out).is_empty());
}

#[test]
fn filter_with_no_matches_returns_no_rows() {
    let s1 = schema(&[("k", DataType::Int64, false)]);
    let cat = catalog_with("t", &s1, batches(&s1, &[i64_col(&[Some(1), Some(2)])], 10));
    let plan = LogicalPlan::Filter { input: scan("t", &s1), predicate: lit(ScalarValue::Boolean(false)) };
    let out = execute(&plan, &cat, &ctx(4, 1)).expect("filter runs");
    assert!(rows(&out).is_empty());
}

// ---------- determinism and scale ----------

fn gen_rows(n: usize, seed: u64) -> Vec<(i64, Option<i64>)> {
    let mut rng = StdRng::seed_from_u64(seed);
    (0..n)
        .map(|i| {
            let v = if rng.gen_bool(0.1) { None } else { Some(rng.gen_range(-1000..1000)) };
            (i as i64, v)
        })
        .collect()
}

fn gen_table_schema() -> SchemaRef {
    schema(&[("id", DataType::Int64, false), ("v", DataType::Int64, true)])
}

fn gen_columns(data: &[(i64, Option<i64>)]) -> Vec<Column> {
    let ids: Vec<Option<i64>> = data.iter().map(|(id, _)| Some(*id)).collect();
    let vs: Vec<Option<i64>> = data.iter().map(|(_, v)| *v).collect();
    vec![i64_col(&ids), i64_col(&vs)]
}

#[test]
fn filter_on_200k_rows_matches_plain_rust_iterator() {
    let data = gen_rows(200_000, 42);
    let sch = gen_table_schema();
    let cat = catalog_with("t", &sch, batches(&sch, &gen_columns(&data), 8192));
    let predicate = bin(
        bin(c(&sch, "v"), BinaryOp::Gt, lit_i(100)),
        BinaryOp::And,
        bin(bin(c(&sch, "id"), BinaryOp::Mod, lit_i(3)), BinaryOp::Eq, lit_i(0)),
    );
    let plan = LogicalPlan::Filter { input: scan("t", &sch), predicate };

    let expected: Vec<Vec<ScalarValue>> = data
        .iter()
        .filter(|(id, v)| id % 3 == 0 && v.is_some_and(|x| x > 100))
        .map(|(id, v)| vec![int(*id), v.map_or(null_i(), int)])
        .collect();

    let out = execute(&plan, &cat, &ctx(4, 4096)).expect("filter runs");
    let got = rows(&out);
    assert!(expected.len() > 1000, "fixture should keep a meaningful number of rows");
    assert_eq!(got.len(), expected.len());
    assert!(got == expected, "filtered rows differ from the plain Rust computation");
}

#[test]
fn sort_and_filter_output_is_identical_at_1_and_8_threads() {
    let data = gen_rows(50_000, 7);
    let sch = gen_table_schema();
    let cat = catalog_with("t", &sch, batches(&sch, &gen_columns(&data), 4096));
    let plan = LogicalPlan::Sort {
        input: Arc::new(LogicalPlan::Filter {
            input: scan("t", &sch),
            predicate: bin(c(&sch, "v"), BinaryOp::Lt, lit_i(0)),
        }),
        order_by: vec![
            SortField { expr: c(&sch, "v"), ascending: true, nulls_first: true },
            SortField { expr: c(&sch, "id"), ascending: false, nulls_first: false },
        ],
    };

    let one = execute(&plan, &cat, &ctx(1, 1000)).expect("1 thread");
    let eight = execute(&plan, &cat, &ctx(8, 1000)).expect("8 threads");
    let lens = |b: &[Batch]| b.iter().map(|x| x.num_rows).collect::<Vec<_>>();
    assert_eq!(lens(&one), lens(&eight));
    assert_eq!(rows(&one), rows(&eight));
    assert!(!rows(&one).is_empty());
}

// ---------- aggregate and join ----------

#[test]
fn aggregate_group_sum_and_count_with_null_group() {
    let sch = schema(&[("g", DataType::Utf8, true), ("x", DataType::Int64, true)]);
    let g = str_col(&[Some("a"), Some("b"), Some("a"), None]);
    let x = i64_col(&[Some(1), Some(2), None, Some(4)]);
    let cat = catalog_with("t", &sch, batches(&sch, &[g, x], 10));
    let out_schema = schema(&[("g", DataType::Utf8, true), ("sum", DataType::Int64, true), ("cnt", DataType::Int64, false)]);
    let plan = LogicalPlan::Aggregate {
        input: scan("t", &sch),
        group_exprs: vec![c(&sch, "g")],
        agg_exprs: vec![
            AggExpr { func: AggFunc::Sum, expr: Some(c(&sch, "x")), output_type: DataType::Int64 },
            AggExpr { func: AggFunc::Count, expr: None, output_type: DataType::Int64 },
        ],
        schema: out_schema,
    };
    let out = execute(&plan, &cat, &ctx(2, 2)).expect("plan runs");
    // Nulls ignored by SUM, counted by COUNT(*); NULL is its own group.
    assert_eq!(
        sorted_rows(&out),
        sorted(vec![
            vec![s("a"), int(1), int(2)],
            vec![s("b"), int(2), int(1)],
            vec![null_s(), int(4), int(1)],
        ])
    );
}

#[test]
fn aggregate_without_groups_on_empty_input_returns_one_row() {
    let sch = schema(&[("x", DataType::Int64, true)]);
    let cat = catalog_with("t", &sch, batches(&sch, &[i64_col(&[Some(1)])], 10));
    let filtered = Arc::new(LogicalPlan::Filter { input: scan("t", &sch), predicate: lit(ScalarValue::Boolean(false)) });
    let plan = LogicalPlan::Aggregate {
        input: filtered,
        group_exprs: vec![],
        agg_exprs: vec![
            AggExpr { func: AggFunc::Count, expr: None, output_type: DataType::Int64 },
            AggExpr { func: AggFunc::Sum, expr: Some(c(&sch, "x")), output_type: DataType::Int64 },
        ],
        schema: schema(&[("cnt", DataType::Int64, false), ("sum", DataType::Int64, true)]),
    };
    let out = execute(&plan, &cat, &ctx(1, 10)).expect("plan runs");
    assert_eq!(rows(&out), vec![vec![int(0), null_i()]]);
}

fn join_fixture() -> (Catalog, SchemaRef, SchemaRef) {
    let left = schema(&[("lid", DataType::Int64, true), ("lname", DataType::Utf8, false)]);
    let right = schema(&[("rid", DataType::Int64, true), ("score", DataType::Int64, false)]);
    let mut cat = Catalog::new();
    let lb = batches(&left, &[i64_col(&[Some(1), Some(2), Some(2), None]), str_col(&[Some("a"), Some("b"), Some("c"), Some("d")])], 10);
    let rb = batches(&right, &[i64_col(&[Some(2), Some(2), Some(3), None]), i64_col(&[Some(10), Some(20), Some(30), Some(40)])], 10);
    cat.register("l", left.clone(), lb);
    cat.register("r", right.clone(), rb);
    (cat, left, right)
}

fn join_plan(join_type: JoinType, left: &SchemaRef, right: &SchemaRef) -> LogicalPlan {
    // Semi and Anti emit left rows only; the other types emit left then right columns.
    let mut fields = left.fields.clone();
    if !matches!(join_type, JoinType::Semi | JoinType::Anti) {
        fields.extend(right.fields.clone());
    }
    LogicalPlan::Join {
        left: scan("l", left),
        right: scan("r", right),
        join_type,
        on: vec![(c(left, "lid"), c(right, "rid"))],
        schema: Arc::new(Schema::new(fields)),
    }
}

#[test]
fn inner_join_produces_duplicate_cross_product_and_drops_null_keys() {
    let (cat, left, right) = join_fixture();
    let plan = join_plan(JoinType::Inner, &left, &right);
    let out = execute(&plan, &cat, &ctx(2, 2)).expect("plan runs");
    assert_eq!(
        sorted_rows(&out),
        sorted(vec![
            vec![int(2), s("b"), int(2), int(10)],
            vec![int(2), s("b"), int(2), int(20)],
            vec![int(2), s("c"), int(2), int(10)],
            vec![int(2), s("c"), int(2), int(20)],
        ])
    );
}

#[test]
fn left_join_keeps_unmatched_left_rows_padded_with_nulls() {
    let (cat, left, right) = join_fixture();
    let plan = join_plan(JoinType::Left, &left, &right);
    let out = execute(&plan, &cat, &ctx(2, 2)).expect("plan runs");
    assert_eq!(
        sorted_rows(&out),
        sorted(vec![
            vec![int(2), s("b"), int(2), int(10)],
            vec![int(2), s("b"), int(2), int(20)],
            vec![int(2), s("c"), int(2), int(10)],
            vec![int(2), s("c"), int(2), int(20)],
            vec![int(1), s("a"), null_i(), null_i()],
            vec![null_i(), s("d"), null_i(), null_i()],
        ])
    );
}

#[test]
fn right_join_keeps_unmatched_right_rows_padded_with_nulls() {
    let (cat, left, right) = join_fixture();
    let plan = join_plan(JoinType::Right, &left, &right);
    let out = execute(&plan, &cat, &ctx(2, 2)).expect("plan runs");
    assert_eq!(
        sorted_rows(&out),
        sorted(vec![
            vec![int(2), s("b"), int(2), int(10)],
            vec![int(2), s("b"), int(2), int(20)],
            vec![int(2), s("c"), int(2), int(10)],
            vec![int(2), s("c"), int(2), int(20)],
            vec![null_i(), null_s(), int(3), int(30)],
            vec![null_i(), null_s(), null_i(), int(40)],
        ])
    );
}

#[test]
fn full_join_keeps_unmatched_rows_from_both_sides() {
    let (cat, left, right) = join_fixture();
    let plan = join_plan(JoinType::Full, &left, &right);
    let out = execute(&plan, &cat, &ctx(2, 2)).expect("plan runs");
    assert_eq!(
        sorted_rows(&out),
        sorted(vec![
            vec![int(2), s("b"), int(2), int(10)],
            vec![int(2), s("b"), int(2), int(20)],
            vec![int(2), s("c"), int(2), int(10)],
            vec![int(2), s("c"), int(2), int(20)],
            vec![int(1), s("a"), null_i(), null_i()],
            vec![null_i(), s("d"), null_i(), null_i()],
            vec![null_i(), null_s(), int(3), int(30)],
            vec![null_i(), null_s(), null_i(), int(40)],
        ])
    );
}

#[test]
fn semi_join_emits_each_matching_left_row_once_with_left_columns_only() {
    let (cat, left, right) = join_fixture();
    let plan = join_plan(JoinType::Semi, &left, &right);
    let out = execute(&plan, &cat, &ctx(2, 2)).expect("plan runs");
    // Left rows 2/b and 2/c each match two right rows but appear once.
    assert_eq!(sorted_rows(&out), sorted(vec![vec![int(2), s("b")], vec![int(2), s("c")]]));
}

#[test]
fn anti_join_keeps_left_rows_without_a_match_including_null_keys() {
    let (cat, left, right) = join_fixture();
    let plan = join_plan(JoinType::Anti, &left, &right);
    let out = execute(&plan, &cat, &ctx(2, 2)).expect("plan runs");
    assert_eq!(sorted_rows(&out), sorted(vec![vec![int(1), s("a")], vec![null_i(), s("d")]]));
}

// ---------- tiny memory budget: spilling must not change results ----------

/// Spill directory unique to this process and test, removed on drop (also when the test panics).
struct TempDir(PathBuf);

impl TempDir {
    fn new(tag: &str) -> Self {
        let p = std::env::temp_dir().join(format!("ndq_e2e_spill_{}_{tag}", std::process::id()));
        let _ = std::fs::remove_dir_all(&p);
        std::fs::create_dir_all(&p).expect("create spill dir");
        TempDir(p)
    }
}

impl Drop for TempDir {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

fn tiny_ctx(dir: &TempDir) -> ExecContext {
    ExecContext { threads: 4, morsel_rows: 4096, budget: MemoryBudget::new(64 * 1024), spill_dir: dir.0.clone() }
}

/// Left: `lid` (nullable, NULL every 997th row, otherwise the row index), `lpay` (Utf8, unique, ~40 bytes).
fn left_payload_side(n: usize) -> (SchemaRef, Vec<Batch>) {
    let sch = schema(&[("lid", DataType::Int64, true), ("lpay", DataType::Utf8, false)]);
    let ids: Vec<Option<i64>> = (0..n).map(|i| if i % 997 == 0 { None } else { Some(i as i64) }).collect();
    let pays: Vec<String> = (0..n).map(|i| format!("L-{i:08}-{}", "x".repeat(24))).collect();
    let pay_refs: Vec<Option<&str>> = pays.iter().map(|p| Some(p.as_str())).collect();
    let b = batches(&sch, &[i64_col(&ids), str_col(&pay_refs)], 4096);
    (sch, b)
}

/// Right: `rid` (nullable, NULL every 1009th row, else `i * 7 % 40_000` so keys repeat past 40k rows), `rpay`.
fn right_payload_side(n: usize) -> (SchemaRef, Vec<Batch>) {
    let sch = schema(&[("rid", DataType::Int64, true), ("rpay", DataType::Utf8, false)]);
    let ids: Vec<Option<i64>> =
        (0..n).map(|i| if i % 1009 == 0 { None } else { Some((i as i64 * 7) % 40_000) }).collect();
    let pays: Vec<String> = (0..n).map(|i| format!("R-{i:08}-{}", "y".repeat(24))).collect();
    let pay_refs: Vec<Option<&str>> = pays.iter().map(|p| Some(p.as_str())).collect();
    let b = batches(&sch, &[i64_col(&ids), str_col(&pay_refs)], 4096);
    (sch, b)
}

#[test]
fn full_join_under_tiny_budget_spills_and_matches_large_budget() {
    const N: usize = 50_000;
    let (lsch, lb) = left_payload_side(N);
    let (rsch, rb) = right_payload_side(N);
    let mut cat = Catalog::new();
    cat.register("l", lsch.clone(), lb.clone());
    cat.register("r", rsch.clone(), rb.clone());
    let plan = join_plan(JoinType::Full, &lsch, &rsch);

    let dir = TempDir::new("join");
    let tiny = execute(&plan, &cat, &tiny_ctx(&dir)).expect("tiny-budget join runs");
    let big = execute(&plan, &cat, &ctx(4, 4096)).expect("1 GiB join runs");

    // The tiny budget must really force spilling, otherwise this test proves nothing.
    let on = vec![(c(&lsch, "lid"), c(&rsch, "rid"))];
    let (_, stats) = native_engine::join::hash_join_with_stats(
        lb,
        rb,
        JoinType::Full,
        &on,
        plan.schema(),
        &MemoryBudget::new(64 * 1024),
        &dir.0,
    )
    .expect("stats join runs");
    assert!(stats.spill_files > 0, "64 KiB budget did not spill");

    let tiny_rows = rows(&tiny);
    assert!(tiny_rows.len() > N, "expected a full join with more output rows than inputs");
    assert_eq!(sorted(tiny_rows), sorted_rows(&big));
}

#[test]
fn sort_under_tiny_budget_spills_and_matches_large_budget() {
    const N: usize = 50_000;
    let (sch, b) = left_payload_side(N);
    let cat = catalog_with("t", &sch, b.clone());
    let order_by = vec![
        SortField { expr: c(&sch, "lid"), ascending: true, nulls_first: true },
        SortField { expr: c(&sch, "lpay"), ascending: false, nulls_first: false },
    ];
    let plan = LogicalPlan::Sort { input: scan("t", &sch), order_by: order_by.clone() };

    let dir = TempDir::new("sort");
    let tiny = execute(&plan, &cat, &tiny_ctx(&dir)).expect("tiny-budget sort runs");
    let big = execute(&plan, &cat, &ctx(4, 4096)).expect("1 GiB sort runs");

    // The tiny budget must really force spilling, otherwise this test proves nothing.
    let (_, stats) = native_engine::sort::sort_batches_with_stats(
        b,
        sch.clone(),
        &order_by,
        &MemoryBudget::new(64 * 1024),
        &dir.0,
        4096,
    )
    .expect("stats sort runs");
    assert!(stats.runs_spilled > 0, "64 KiB budget did not spill");

    // Sort output order is part of the contract, so compare without re-sorting.
    assert_eq!(rows(&tiny).len(), N);
    assert_eq!(rows(&tiny), rows(&big));
}
