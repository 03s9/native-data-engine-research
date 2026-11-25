//! Cranelift JIT for arithmetic expressions (experiment H1a, see docs/PLAN.md).
//! One fused scalar loop per expression: no intermediate columns.
//!
//! Kernel ABI: `extern "C" fn(cols: *const *const u8, n: usize, out: *mut u8)`.
//! `cols[k]` points at the values of the k-th referenced column (8 bytes per row),
//! `out` at the output values (8 bytes per row for Int64/Float64, 1 byte per row for Boolean).
//! NULLs are handled outside the kernel: output validity = AND of the input validities.
use crate::column::{Batch, Column, Values};
use crate::expr::{BinaryOp, Expr, UnaryOp};
use crate::types::{DataType, EngineError, Result, ScalarValue};
use cranelift_codegen::ir::condcodes::{FloatCC, IntCC};
use cranelift_codegen::ir::{types, AbiParam, InstBuilder, MemFlagsData, Type, Value};
use cranelift_codegen::settings::{self, Configurable};
use cranelift_frontend::{FunctionBuilder, FunctionBuilderContext};
use cranelift_jit::{JITBuilder, JITModule};
use cranelift_module::{default_libcall_names, Linkage, Module};
use std::time::{Duration, Instant};

type KernelFn = unsafe extern "C" fn(*const *const u8, usize, *mut u8);

pub struct CompiledKernel {
    // Owns the executable memory `func` points into. `Some` until Drop, which frees it.
    module: Option<JITModule>,
    func: KernelFn,
    /// Batch column index and type for each kernel input slot, in first-use order.
    inputs: Vec<(usize, DataType)>,
    out_type: DataType,
    pub compile_time: Duration,
}

// SAFETY: the kernel is immutable machine code plus plain data; it holds no thread-affine state.
unsafe impl Send for CompiledKernel {}
unsafe impl Sync for CompiledKernel {}

fn unsupported(what: &'static str) -> EngineError {
    EngineError::Unimplemented(what)
}
fn jit_err<E: std::fmt::Display>(e: E) -> EngineError {
    EngineError::Plan(format!("jit: {e}"))
}

/// True when `compile` accepts `expr`.
pub fn supports(expr: &Expr) -> bool {
    infer(expr, &mut Vec::new()).is_some()
}

/// The type the kernel produces for `e`, or None when the JIT does not support it.
/// Records each referenced input column in `inputs`; a column used with two types is rejected.
/// Mirrors `eval`: mixed Int64/Float64 operands are promoted to Float64 implicitly, and the
/// declared `return_type` must agree with what the operands produce.
fn infer(e: &Expr, inputs: &mut Vec<(usize, DataType)>) -> Option<DataType> {
    match e {
        Expr::Column {
            index, data_type, ..
        } => {
            if !matches!(data_type, DataType::Int64 | DataType::Float64) {
                return None;
            }
            match inputs.iter().find(|(i, _)| i == index) {
                Some((_, seen)) => (seen == data_type).then_some(*data_type),
                None => {
                    inputs.push((*index, *data_type));
                    Some(*data_type)
                }
            }
        }
        Expr::Literal(ScalarValue::Int64(_)) => Some(DataType::Int64),
        Expr::Literal(ScalarValue::Float64(_)) => Some(DataType::Float64),
        Expr::Literal(_) => None,
        Expr::Binary {
            left,
            op,
            right,
            return_type,
        } => {
            let operand = numeric_common(infer(left, inputs)?, infer(right, inputs)?)?;
            match op {
                BinaryOp::Add | BinaryOp::Sub | BinaryOp::Mul => {
                    (*return_type == operand).then_some(operand)
                }
                BinaryOp::Eq
                | BinaryOp::NotEq
                | BinaryOp::Lt
                | BinaryOp::LtEq
                | BinaryOp::Gt
                | BinaryOp::GtEq => {
                    (*return_type == DataType::Boolean).then_some(DataType::Boolean)
                }
                _ => None,
            }
        }
        Expr::Unary {
            op: UnaryOp::Neg,
            expr,
            return_type,
        } => {
            let t = infer(expr, inputs)?;
            (matches!(t, DataType::Int64 | DataType::Float64) && *return_type == t).then_some(t)
        }
        Expr::Cast {
            expr,
            target_type: DataType::Float64,
        } => (infer(expr, inputs)? == DataType::Int64).then_some(DataType::Float64),
        _ => None,
    }
}

/// Common operand type for arithmetic and comparisons, as `eval` promotes it.
fn numeric_common(l: DataType, r: DataType) -> Option<DataType> {
    match (l, r) {
        (DataType::Int64, DataType::Int64) => Some(DataType::Int64),
        (DataType::Int64 | DataType::Float64, DataType::Int64 | DataType::Float64) => {
            Some(DataType::Float64)
        }
        _ => None,
    }
}

/// Machine type of a kernel value. Boolean results are one i8 holding 0 or 1.
fn clif_type(dt: DataType) -> Type {
    match dt {
        DataType::Int64 => types::I64,
        DataType::Float64 => types::F64,
        _ => types::I8,
    }
}

/// IEEE semantics as in `eval`: Eq/Lt/LtEq/Gt/GtEq are false when either side is NaN;
/// NotEq is true. `FloatCC::Equal` etc. are the ordered codes, `NotEqual` is unordered-or-not-equal.
fn float_cc(op: BinaryOp) -> Option<FloatCC> {
    Some(match op {
        BinaryOp::Eq => FloatCC::Equal,
        BinaryOp::NotEq => FloatCC::NotEqual,
        BinaryOp::Lt => FloatCC::LessThan,
        BinaryOp::LtEq => FloatCC::LessThanOrEqual,
        BinaryOp::Gt => FloatCC::GreaterThan,
        BinaryOp::GtEq => FloatCC::GreaterThanOrEqual,
        _ => return None,
    })
}

fn int_cc(op: BinaryOp) -> Option<IntCC> {
    Some(match op {
        BinaryOp::Eq => IntCC::Equal,
        BinaryOp::NotEq => IntCC::NotEqual,
        BinaryOp::Lt => IntCC::SignedLessThan,
        BinaryOp::LtEq => IntCC::SignedLessThanOrEqual,
        BinaryOp::Gt => IntCC::SignedGreaterThan,
        BinaryOp::GtEq => IntCC::SignedGreaterThanOrEqual,
        _ => return None,
    })
}

struct Gen<'a, 'b> {
    b: &'a mut FunctionBuilder<'b>,
    bases: &'a [Value],
    mem: MemFlagsData,
    inputs: &'a [(usize, DataType)],
    offset: Value,
}

impl Gen<'_, '_> {
    /// Emits `e` for the current row and returns its SSA value with its logical type.
    fn expr(&mut self, e: &Expr) -> Result<(Value, DataType)> {
        match e {
            Expr::Column { index, .. } => {
                let slot = self
                    .inputs
                    .iter()
                    .position(|(i, _)| i == index)
                    .ok_or_else(|| unsupported("jit: unknown input column"))?;
                let (_, dt) = self.inputs[slot];
                let addr = self.b.ins().iadd(self.bases[slot], self.offset);
                let value = self.b.ins().load(clif_type(dt), self.mem, addr, 0);
                Ok((value, dt))
            }
            Expr::Literal(ScalarValue::Int64(x)) => {
                Ok((self.b.ins().iconst(types::I64, *x), DataType::Int64))
            }
            Expr::Literal(ScalarValue::Float64(x)) => {
                Ok((self.b.ins().f64const(*x), DataType::Float64))
            }
            Expr::Binary {
                left, op, right, ..
            } => {
                let (l, lt) = self.expr(left)?;
                let (r, rt) = self.expr(right)?;
                let t = numeric_common(lt, rt).ok_or_else(|| unsupported("jit: operand types"))?;
                let l = self.widen(l, lt, t);
                let r = self.widen(r, rt, t);
                self.binary(*op, t, l, r)
            }
            Expr::Unary {
                op: UnaryOp::Neg,
                expr,
                ..
            } => {
                let (v, dt) = self.expr(expr)?;
                let v = match dt {
                    DataType::Int64 => self.b.ins().ineg(v),
                    DataType::Float64 => self.b.ins().fneg(v),
                    _ => return Err(unsupported("jit: Neg operand")),
                };
                Ok((v, dt))
            }
            Expr::Cast {
                expr,
                target_type: DataType::Float64,
            } => {
                let (v, dt) = self.expr(expr)?;
                if dt != DataType::Int64 {
                    return Err(unsupported("jit: cast source"));
                }
                Ok((
                    self.b.ins().fcvt_from_sint(types::F64, v),
                    DataType::Float64,
                ))
            }
            _ => Err(unsupported("jit: expression shape")),
        }
    }

    /// Promotes an Int64 operand to Float64 (the only promotion `numeric_common` allows).
    fn widen(&mut self, v: Value, from: DataType, to: DataType) -> Value {
        if from == to {
            v
        } else {
            self.b.ins().fcvt_from_sint(types::F64, v)
        }
    }

    fn binary(
        &mut self,
        op: BinaryOp,
        t: DataType,
        l: Value,
        r: Value,
    ) -> Result<(Value, DataType)> {
        let unsupported_op = || unsupported("jit: operator");
        Ok(match (op, t) {
            (BinaryOp::Add, DataType::Int64) => (self.b.ins().iadd(l, r), t),
            (BinaryOp::Add, DataType::Float64) => (self.b.ins().fadd(l, r), t),
            (BinaryOp::Sub, DataType::Int64) => (self.b.ins().isub(l, r), t),
            (BinaryOp::Sub, DataType::Float64) => (self.b.ins().fsub(l, r), t),
            (BinaryOp::Mul, DataType::Int64) => (self.b.ins().imul(l, r), t),
            (BinaryOp::Mul, DataType::Float64) => (self.b.ins().fmul(l, r), t),
            (cmp, DataType::Int64) => {
                let cc = int_cc(cmp).ok_or_else(unsupported_op)?;
                (self.b.ins().icmp(cc, l, r), DataType::Boolean)
            }
            (cmp, DataType::Float64) => {
                let cc = float_cc(cmp).ok_or_else(unsupported_op)?;
                (self.b.ins().fcmp(cc, l, r), DataType::Boolean)
            }
            _ => return Err(unsupported_op()),
        })
    }
}

pub fn compile(expr: &Expr) -> Result<CompiledKernel> {
    let start = Instant::now();
    let mut inputs = Vec::new();
    let out_type =
        infer(expr, &mut inputs).ok_or_else(|| unsupported("jit: expression not supported"))?;

    let mut flags = settings::builder();
    flags.set("opt_level", "speed").map_err(jit_err)?;
    let isa = cranelift_native::builder()
        .map_err(jit_err)?
        .finish(settings::Flags::new(flags))
        .map_err(jit_err)?;
    let mut module = JITModule::new(JITBuilder::with_isa(isa, default_libcall_names()));
    let target = module.target_config();
    let ptr = target.pointer_type();

    let mut ctx = module.make_context();
    for _ in 0..3 {
        ctx.func.signature.params.push(AbiParam::new(ptr));
    }
    let mut fbc = FunctionBuilderContext::new();
    {
        let mut b = FunctionBuilder::new(&mut ctx.func, &mut fbc);
        // cranelift 0.138: load/store take MemFlagsData (MemFlags is now an interned handle).
        let mem = MemFlagsData::trusted();
        let entry = b.create_block();
        let header = b.create_block();
        let body = b.create_block();
        let exit = b.create_block();
        b.append_block_params_for_function_params(entry);
        b.append_block_param(header, ptr);

        b.switch_to_block(entry);
        let (cols, n, out) = (
            b.block_params(entry)[0],
            b.block_params(entry)[1],
            b.block_params(entry)[2],
        );
        let bases: Vec<Value> = (0..inputs.len())
            .map(|k| b.ins().load(ptr, mem, cols, (k * 8) as i32))
            .collect();
        let zero = b.ins().iconst(ptr, 0);
        b.ins().jump(header, &[zero.into()]);

        b.switch_to_block(header);
        let i = b.block_params(header)[0];
        let more = b.ins().icmp(IntCC::UnsignedLessThan, i, n);
        b.ins().brif(more, body, &[], exit, &[]);

        b.switch_to_block(body);
        // 8-byte slots for inputs and Int64/Float64 outputs; Boolean outputs are 1 byte per row.
        let offset = b.ins().ishl_imm_u(i, 3);
        let (value, dt) = Gen {
            b: &mut b,
            bases: &bases,
            mem,
            inputs: &inputs,
            offset,
        }
        .expr(expr)?;
        // Checked here, not assumed, because the store width comes from the value's machine type.
        if dt != out_type || b.func.dfg.value_type(value) != clif_type(out_type) {
            return Err(unsupported("jit: output type"));
        }
        let dst = match out_type {
            DataType::Boolean => b.ins().iadd(out, i),
            _ => b.ins().iadd(out, offset),
        };
        b.ins().store(mem, value, dst, 0);
        let next = b.ins().iadd_imm_u(i, 1);
        b.ins().jump(header, &[next.into()]);

        b.switch_to_block(exit);
        b.ins().return_(&[]);
        b.seal_all_blocks();
        b.finalize(target);
    }
    let id = module
        .declare_function("kernel", Linkage::Export, &ctx.func.signature)
        .map_err(jit_err)?;
    module.define_function(id, &mut ctx).map_err(jit_err)?;
    module.clear_context(&mut ctx);
    module.finalize_definitions().map_err(jit_err)?;
    let code = module.get_finalized_function(id);
    // SAFETY: `code` is the finalized entry point of a function defined above with exactly the
    // KernelFn signature (three pointer-sized params, no return); `module` is kept alive in the struct.
    let func = unsafe { std::mem::transmute::<*const u8, KernelFn>(code) };
    Ok(CompiledKernel {
        module: Some(module),
        func,
        inputs,
        out_type,
        compile_time: start.elapsed(),
    })
}

impl Drop for CompiledKernel {
    fn drop(&mut self) {
        if let Some(module) = self.module.take() {
            // SAFETY: `func` is private and only called through `&self` in `invoke`, so no call can
            // be executing or start after `drop(&mut self)`; nothing else holds pointers into the module.
            // Without this, cranelift-jit 0.138 leaks the code pages (its Drop does not unmap them).
            unsafe { module.free_memory() };
        }
    }
}

impl CompiledKernel {
    /// Same result as `eval::eval(expr, batch)` for every supported expression.
    pub fn run(&self, batch: &Batch) -> Result<Column> {
        let n = batch.num_rows;
        let mut ptrs: Vec<*const u8> = Vec::with_capacity(self.inputs.len());
        let mut validity: Option<Vec<bool>> = None;
        for &(ci, want) in &self.inputs {
            let col = batch
                .columns
                .get(ci)
                .ok_or_else(|| EngineError::Plan("jit: column index out of range".into()))?;
            // The kernel reads exactly `n` elements per input, so every buffer must have that length.
            if col.len() != n {
                return Err(EngineError::Plan(
                    "jit: column length differs from batch rows".into(),
                ));
            }
            let ptr: *const u8 = match (&col.values, want) {
                (Values::Int64(v), DataType::Int64) => v.as_ptr().cast(),
                (Values::Float64(v), DataType::Float64) => v.as_ptr().cast(),
                _ => {
                    return Err(EngineError::Type(
                        "jit: column type differs from compiled kernel".into(),
                    ))
                }
            };
            ptrs.push(ptr);
            if let Some(m) = &col.validity {
                if m.len() != n {
                    return Err(EngineError::Plan(
                        "jit: validity length differs from batch rows".into(),
                    ));
                }
                match &mut validity {
                    None => validity = Some(m.clone()),
                    Some(acc) => acc.iter_mut().zip(m).for_each(|(a, b)| *a &= *b),
                }
            }
        }
        let values = match self.out_type {
            DataType::Int64 => {
                let mut out = vec![0i64; n];
                self.invoke(&ptrs, n, out.as_mut_ptr().cast());
                Values::Int64(out)
            }
            DataType::Float64 => {
                let mut out = vec![0f64; n];
                self.invoke(&ptrs, n, out.as_mut_ptr().cast());
                Values::Float64(out)
            }
            DataType::Boolean => {
                let mut out = vec![false; n];
                self.invoke(&ptrs, n, out.as_mut_ptr().cast());
                Values::Bool(out)
            }
            _ => return Err(unsupported("jit: output type")),
        };
        Ok(Column {
            values,
            validity: validity.filter(|v| v.iter().any(|ok| !ok)),
        })
    }

    /// Calls the kernel. Caller contract: `ptrs` has one pointer per entry of `self.inputs`, each
    /// addressing `n` live elements of that column's type; `out` addresses `n` elements of
    /// `self.out_type` (8 bytes per row for Int64/Float64, 1 byte per row for Boolean).
    fn invoke(&self, ptrs: &[*const u8], n: usize, out: *mut u8) {
        // SAFETY: `run` establishes the contract above (input lengths checked against
        // `batch.num_rows`, output sized per type). The kernel reads and writes rows 0..n only.
        // Boolean rows receive only the i8 values 0 or 1 from icmp/fcmp, so each byte is a valid bool.
        unsafe { (self.func)(ptrs.as_ptr(), n, out) };
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::eval::{binary, case, cast, col, eval, lit, unary};
    use crate::types::{Field, Schema};
    use std::sync::Arc;

    const CMP_OPS: [BinaryOp; 6] = [
        BinaryOp::Eq,
        BinaryOp::NotEq,
        BinaryOp::Lt,
        BinaryOp::LtEq,
        BinaryOp::Gt,
        BinaryOp::GtEq,
    ];
    const I64_POOL: [i64; 6] = [0, 1, -1, i64::MAX, i64::MIN, 42];
    const F64_POOL: [f64; 10] = [
        0.0,
        -0.0,
        1.5,
        -2.25,
        f64::NAN,
        f64::INFINITY,
        f64::NEG_INFINITY,
        f64::MAX,
        f64::MIN_POSITIVE,
        1e300,
    ];

    /// Deterministic generator so the tests need no RNG crate.
    struct Lcg(u64);
    impl Lcg {
        fn next_u64(&mut self) -> u64 {
            self.0 = self
                .0
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            self.0 >> 33
        }
    }

    fn ints(n: usize, rng: &mut Lcg) -> Vec<i64> {
        (0..n)
            .map(|_| {
                let r = rng.next_u64();
                if r.is_multiple_of(4) {
                    I64_POOL[(r as usize / 4) % I64_POOL.len()]
                } else {
                    r as i64 - (1 << 30)
                }
            })
            .collect()
    }

    fn floats(n: usize, rng: &mut Lcg) -> Vec<f64> {
        (0..n)
            .map(|_| {
                let r = rng.next_u64();
                if r.is_multiple_of(4) {
                    F64_POOL[(r as usize / 4) % F64_POOL.len()]
                } else {
                    (r % 2001) as f64 / 8.0 - 125.0
                }
            })
            .collect()
    }

    /// `true` = valid; roughly one row in `null_one_in` is NULL.
    fn validity(n: usize, rng: &mut Lcg, null_one_in: u64) -> Vec<bool> {
        (0..n)
            .map(|_| !rng.next_u64().is_multiple_of(null_one_in))
            .collect()
    }

    /// Batch whose schema comes from the column types, so schema and data always agree.
    fn setup(named: Vec<(&str, Column)>) -> (Arc<Schema>, Batch) {
        let fields = named
            .iter()
            .map(|(name, c)| Field::new(name, c.data_type(), true))
            .collect();
        let schema = Arc::new(Schema::new(fields));
        let cols = named.into_iter().map(|(_, c)| c).collect();
        let batch = Batch::try_new(schema.clone(), cols).unwrap();
        (schema, batch)
    }

    fn c(schema: &Schema, name: &str) -> Expr {
        col(schema, name).unwrap()
    }
    fn bin(l: Expr, op: BinaryOp, r: Expr) -> Expr {
        binary(l, op, r).unwrap()
    }
    fn neg(e: Expr) -> Expr {
        unary(UnaryOp::Neg, e).unwrap()
    }
    fn lit_i(x: i64) -> Expr {
        lit(ScalarValue::Int64(x))
    }
    fn lit_f(x: f64) -> Expr {
        lit(ScalarValue::Float64(x))
    }

    /// Asserts the kernel matches `eval` row by row: same type, same validity, and for valid
    /// rows the same value (Float64 compared by bits, so NaN payloads and -0.0 count).
    fn assert_matches_eval(e: &Expr, batch: &Batch) {
        assert!(supports(e), "supports must accept {e:?}");
        let jit = compile(e).unwrap().run(batch).unwrap();
        let reference = eval(e, batch).unwrap();
        assert_eq!(jit.data_type(), reference.data_type());
        assert_eq!(jit.len(), reference.len());
        for i in 0..jit.len() {
            assert_eq!(jit.is_valid(i), reference.is_valid(i), "validity row {i}");
            if !reference.is_valid(i) {
                continue;
            }
            match (jit.get(i), reference.get(i)) {
                (ScalarValue::Float64(a), ScalarValue::Float64(b)) => {
                    assert_eq!(a.to_bits(), b.to_bits(), "row {i}")
                }
                (a, b) => assert_eq!(a, b, "row {i}"),
            }
        }
    }

    fn rust_cmp(op: BinaryOp, x: f64, y: f64) -> bool {
        match op {
            BinaryOp::Eq => x == y,
            BinaryOp::NotEq => x != y,
            BinaryOp::Lt => x < y,
            BinaryOp::LtEq => x <= y,
            BinaryOp::Gt => x > y,
            BinaryOp::GtEq => x >= y,
            _ => unreachable!("comparison operators only"),
        }
    }

    #[test]
    fn spike_add_matches_eval_including_nulls() {
        let schema = Arc::new(Schema::new(vec![
            Field::new("a", DataType::Int64, true),
            Field::new("b", DataType::Int64, false),
        ]));
        let a =
            Column::from_i64(vec![1, 2, i64::MAX, 4]).with_validity(vec![true, false, true, true]);
        let b = Column::from_i64(vec![10, 20, 1, 40]);
        let batch = Batch::try_new(schema.clone(), vec![a, b]).unwrap();
        let e = binary(
            binary(
                col(&schema, "a").unwrap(),
                BinaryOp::Add,
                col(&schema, "b").unwrap(),
            )
            .unwrap(),
            BinaryOp::Add,
            col(&schema, "a").unwrap(),
        )
        .unwrap();
        let jit = compile(&e).unwrap().run(&batch).unwrap();
        let reference = eval(&e, &batch).unwrap();
        for i in 0..4 {
            assert_eq!(jit.get(i), reference.get(i), "row {i}");
        }
    }

    #[test]
    fn int64_add_sub_mul_match_eval() {
        let mut rng = Lcg(1);
        let a = ints(500, &mut rng);
        let b = ints(500, &mut rng);
        let (s, batch) = setup(vec![
            (
                "a",
                Column::from_i64(a).with_validity(validity(500, &mut rng, 7)),
            ),
            ("b", Column::from_i64(b)),
        ]);
        for op in [BinaryOp::Add, BinaryOp::Sub, BinaryOp::Mul] {
            assert_matches_eval(&bin(c(&s, "a"), op, c(&s, "b")), &batch);
        }
    }

    #[test]
    fn float64_add_sub_mul_match_eval_bitwise() {
        let mut rng = Lcg(2);
        let a = floats(500, &mut rng);
        let b = floats(500, &mut rng);
        let (s, batch) = setup(vec![
            (
                "a",
                Column::from_f64(a).with_validity(validity(500, &mut rng, 7)),
            ),
            ("b", Column::from_f64(b)),
        ]);
        for op in [BinaryOp::Add, BinaryOp::Sub, BinaryOp::Mul] {
            assert_matches_eval(&bin(c(&s, "a"), op, c(&s, "b")), &batch);
        }
    }

    #[test]
    fn neg_matches_eval_for_both_types() {
        let (s, batch) = setup(vec![
            ("i", Column::from_i64(vec![i64::MIN, 5, 0, i64::MAX])),
            (
                "f",
                Column::from_f64(vec![-0.0, f64::NAN, 3.5, f64::NEG_INFINITY]),
            ),
        ]);
        assert_matches_eval(&neg(c(&s, "i")), &batch);
        assert_matches_eval(&neg(c(&s, "f")), &batch);
        assert_matches_eval(&neg(neg(c(&s, "f"))), &batch);
    }

    #[test]
    fn int64_literal_matches_eval() {
        let (s, batch) = setup(vec![("a", Column::from_i64(vec![1, -7, i64::MAX]))]);
        assert_matches_eval(&bin(c(&s, "a"), BinaryOp::Add, lit_i(7)), &batch);
        assert_matches_eval(&bin(lit_i(3), BinaryOp::Mul, c(&s, "a")), &batch);
        assert_matches_eval(&bin(c(&s, "a"), BinaryOp::Lt, lit_i(0)), &batch);
    }

    #[test]
    fn float64_literal_matches_eval() {
        let (s, batch) = setup(vec![("f", Column::from_f64(vec![1.0, -0.0, f64::NAN]))]);
        assert_matches_eval(&bin(c(&s, "f"), BinaryOp::Sub, lit_f(0.5)), &batch);
        assert_matches_eval(
            &bin(c(&s, "f"), BinaryOp::Mul, lit_f(f64::INFINITY)),
            &batch,
        );
        assert_matches_eval(&bin(c(&s, "f"), BinaryOp::Eq, lit_f(-0.0)), &batch);
        assert_matches_eval(&bin(c(&s, "f"), BinaryOp::NotEq, lit_f(f64::NAN)), &batch);
    }

    #[test]
    fn null_literal_is_unsupported() {
        let (s, _batch) = setup(vec![("a", Column::from_i64(vec![1]))]);
        let e = bin(
            c(&s, "a"),
            BinaryOp::Add,
            lit(ScalarValue::Null(DataType::Int64)),
        );
        assert!(!supports(&e));
        assert!(matches!(compile(&e), Err(EngineError::Unimplemented(_))));
    }

    #[test]
    fn literal_only_expression_broadcasts() {
        let (_s, batch) = setup(vec![("a", Column::from_i64(vec![0, 0, 0]))]);
        let e = bin(lit_i(5), BinaryOp::Mul, lit_i(3));
        let out = compile(&e).unwrap().run(&batch).unwrap();
        assert_eq!(out.len(), 3);
        for i in 0..3 {
            assert_eq!(out.get(i), ScalarValue::Int64(15));
        }
        assert_matches_eval(&e, &batch);
    }

    #[test]
    fn mixed_int_float_arithmetic_promotes_like_eval() {
        let mut rng = Lcg(3);
        let i = ints(300, &mut rng);
        let f = floats(300, &mut rng);
        let (s, batch) = setup(vec![
            (
                "i",
                Column::from_i64(i).with_validity(validity(300, &mut rng, 5)),
            ),
            ("f", Column::from_f64(f)),
        ]);
        let add = bin(c(&s, "i"), BinaryOp::Add, c(&s, "f"));
        assert_eq!(add.return_type(), DataType::Float64);
        assert_matches_eval(&add, &batch);
        assert_matches_eval(&bin(c(&s, "f"), BinaryOp::Mul, c(&s, "i")), &batch);
        assert_matches_eval(&bin(lit_f(2.5), BinaryOp::Sub, c(&s, "i")), &batch);
    }

    #[test]
    fn mixed_int_float_comparison_promotes_like_eval() {
        let mut rng = Lcg(4);
        let i = ints(300, &mut rng);
        let f = floats(300, &mut rng);
        let (s, batch) = setup(vec![
            ("i", Column::from_i64(i)),
            (
                "f",
                Column::from_f64(f).with_validity(validity(300, &mut rng, 5)),
            ),
        ]);
        for op in CMP_OPS {
            assert_matches_eval(&bin(c(&s, "i"), op, c(&s, "f")), &batch);
            assert_matches_eval(&bin(c(&s, "f"), op, lit_i(2)), &batch);
        }
    }

    #[test]
    fn explicit_cast_int64_to_float64_matches_eval() {
        let (s, batch) = setup(vec![(
            "i",
            Column::from_i64(vec![i64::MAX, i64::MIN, 0, 1 << 53]),
        )]);
        let f = cast(c(&s, "i"), DataType::Float64).unwrap();
        assert_matches_eval(&f, &batch);
        assert_matches_eval(&bin(f.clone(), BinaryOp::Add, lit_f(0.25)), &batch);
        assert_matches_eval(&bin(f, BinaryOp::Gt, lit_f(1e18)), &batch);
    }

    #[test]
    fn int64_comparisons_all_ops_match_eval() {
        let mut rng = Lcg(5);
        let a = ints(400, &mut rng);
        let b = ints(400, &mut rng);
        let (s, batch) = setup(vec![
            (
                "a",
                Column::from_i64(a).with_validity(validity(400, &mut rng, 6)),
            ),
            ("b", Column::from_i64(b)),
        ]);
        for op in CMP_OPS {
            let e = bin(c(&s, "a"), op, c(&s, "b"));
            assert_eq!(e.return_type(), DataType::Boolean);
            assert_matches_eval(&e, &batch);
        }
    }

    #[test]
    fn float64_comparisons_all_ops_match_eval() {
        let mut rng = Lcg(6);
        let a = floats(400, &mut rng);
        let b = floats(400, &mut rng);
        let (s, batch) = setup(vec![
            (
                "a",
                Column::from_f64(a).with_validity(validity(400, &mut rng, 6)),
            ),
            ("b", Column::from_f64(b)),
        ]);
        for op in CMP_OPS {
            assert_matches_eval(&bin(c(&s, "a"), op, c(&s, "b")), &batch);
        }
    }

    #[test]
    fn float_nan_comparisons_follow_ieee() {
        let a = vec![f64::NAN, 1.0, f64::NAN, 2.0, -0.0, f64::INFINITY];
        let b = vec![f64::NAN, f64::NAN, 1.0, 2.0, 0.0, f64::INFINITY];
        let (s, batch) = setup(vec![
            ("a", Column::from_f64(a.clone())),
            ("b", Column::from_f64(b.clone())),
        ]);
        for op in CMP_OPS {
            let e = bin(c(&s, "a"), op, c(&s, "b"));
            let out = compile(&e).unwrap().run(&batch).unwrap();
            for i in 0..a.len() {
                assert_eq!(
                    out.get(i),
                    ScalarValue::Boolean(rust_cmp(op, a[i], b[i])),
                    "{op:?} row {i}"
                );
            }
            assert_matches_eval(&e, &batch);
        }
    }

    #[test]
    fn int64_wrapping_overflow_matches_eval() {
        let (s, batch) = setup(vec![
            ("a", Column::from_i64(vec![i64::MAX, i64::MIN, 3])),
            ("b", Column::from_i64(vec![1, -1, i64::MIN])),
        ]);
        let plus = bin(c(&s, "a"), BinaryOp::Add, c(&s, "b"));
        assert_matches_eval(&plus, &batch);
        assert_eq!(
            compile(&plus).unwrap().run(&batch).unwrap().get(0),
            ScalarValue::Int64(i64::MIN)
        );
        let times = bin(c(&s, "a"), BinaryOp::Mul, lit_i(-1));
        assert_matches_eval(&times, &batch);
        assert_eq!(
            compile(&times).unwrap().run(&batch).unwrap().get(1),
            ScalarValue::Int64(i64::MIN)
        );
        assert_matches_eval(&bin(c(&s, "b"), BinaryOp::Sub, c(&s, "a")), &batch);
        assert_matches_eval(&neg(c(&s, "b")), &batch);
    }

    #[test]
    fn float64_specials_arithmetic_is_bitwise_equal() {
        let a: Vec<f64> = F64_POOL
            .iter()
            .chain(F64_POOL.iter().rev())
            .copied()
            .collect();
        let b: Vec<f64> = F64_POOL
            .iter()
            .rev()
            .chain(F64_POOL.iter())
            .copied()
            .collect();
        let (s, batch) = setup(vec![("a", Column::from_f64(a)), ("b", Column::from_f64(b))]);
        for op in [BinaryOp::Add, BinaryOp::Sub, BinaryOp::Mul] {
            assert_matches_eval(&bin(c(&s, "a"), op, c(&s, "b")), &batch);
        }
        assert_matches_eval(&neg(c(&s, "a")), &batch);
    }

    #[test]
    fn nested_depth_six_expression_matches_eval() {
        let mut rng = Lcg(7);
        let (s, batch) = setup(vec![
            ("i", Column::from_i64(ints(200, &mut rng))),
            ("j", Column::from_i64(ints(200, &mut rng))),
            ("f", Column::from_f64(floats(200, &mut rng))),
        ]);
        let ops = [BinaryOp::Add, BinaryOp::Mul, BinaryOp::Sub];
        let mut e = c(&s, "i");
        for (k, op) in ops.iter().cycle().take(6).enumerate() {
            let rhs = if k % 2 == 0 { c(&s, "j") } else { lit_i(3) };
            e = bin(e, *op, rhs);
        }
        assert!(e.depth() >= 6);
        assert_matches_eval(&e, &batch);

        // Mixed chain that ends in a Float64 comparison: depth 6 with promotion inside.
        let mut f = c(&s, "f");
        for k in 0..5 {
            f = bin(f, ops[k % 3], c(&s, "i"));
        }
        let cmp = bin(f, BinaryOp::LtEq, c(&s, "j"));
        assert!(cmp.depth() >= 6);
        assert_matches_eval(&cmp, &batch);
    }

    #[test]
    fn same_column_referenced_twice_uses_one_input() {
        let (s, batch) = setup(vec![
            ("i", Column::from_i64(vec![2, -3, i64::MAX])),
            ("f", Column::from_f64(vec![0.5, f64::NAN, -0.0])),
        ]);
        let e = bin(
            bin(c(&s, "i"), BinaryOp::Mul, c(&s, "i")),
            BinaryOp::Sub,
            c(&s, "i"),
        );
        let kernel = compile(&e).unwrap();
        assert_eq!(kernel.inputs.len(), 1);
        assert_matches_eval(&e, &batch);
        let g = bin(c(&s, "f"), BinaryOp::Add, c(&s, "f"));
        assert_matches_eval(&g, &batch);
        assert_matches_eval(&bin(g, BinaryOp::Lt, c(&s, "f")), &batch);
    }

    #[test]
    fn nulls_in_one_and_both_inputs_propagate() {
        let (s, batch) = setup(vec![
            (
                "a",
                Column::from_i64(vec![1, 2, 3]).with_validity(vec![false, true, true]),
            ),
            (
                "b",
                Column::from_i64(vec![4, 5, 6]).with_validity(vec![true, false, true]),
            ),
        ]);
        let e = bin(c(&s, "a"), BinaryOp::Add, c(&s, "b"));
        let out = compile(&e).unwrap().run(&batch).unwrap();
        assert!(!out.is_valid(0));
        assert!(!out.is_valid(1));
        assert!(out.is_valid(2));
        assert_eq!(out.get(2), ScalarValue::Int64(9));
        assert_matches_eval(&e, &batch);
        assert_matches_eval(&bin(c(&s, "a"), BinaryOp::Lt, c(&s, "b")), &batch);
        assert_matches_eval(&neg(c(&s, "b")), &batch);
    }

    #[test]
    fn zero_row_batch_matches_eval() {
        let (s, batch) = setup(vec![
            ("i", Column::from_i64(vec![])),
            ("f", Column::from_f64(vec![])),
        ]);
        assert_matches_eval(&bin(c(&s, "i"), BinaryOp::Add, c(&s, "i")), &batch);
        assert_matches_eval(&bin(c(&s, "i"), BinaryOp::Lt, c(&s, "f")), &batch);
        assert_eq!(
            compile(&neg(c(&s, "f")))
                .unwrap()
                .run(&batch)
                .unwrap()
                .len(),
            0
        );
    }

    #[test]
    fn one_row_batch_matches_eval() {
        let (s, batch) = setup(vec![
            ("i", Column::from_i64(vec![i64::MIN])),
            ("f", Column::from_f64(vec![f64::NAN])),
        ]);
        assert_matches_eval(&bin(c(&s, "i"), BinaryOp::Mul, lit_i(-1)), &batch);
        assert_matches_eval(&bin(c(&s, "f"), BinaryOp::Eq, c(&s, "f")), &batch);
        assert_matches_eval(&bin(c(&s, "i"), BinaryOp::GtEq, c(&s, "f")), &batch);
    }

    #[test]
    fn ten_thousand_rows_mixed_types_with_nulls_match_eval() {
        let mut rng = Lcg(8);
        let n = 10_000;
        let i = ints(n, &mut rng);
        let j = ints(n, &mut rng);
        let f = floats(n, &mut rng);
        let (s, batch) = setup(vec![
            (
                "i",
                Column::from_i64(i).with_validity(validity(n, &mut rng, 9)),
            ),
            ("j", Column::from_i64(j)),
            (
                "f",
                Column::from_f64(f).with_validity(validity(n, &mut rng, 4)),
            ),
        ]);
        let value = bin(
            bin(
                c(&s, "f"),
                BinaryOp::Sub,
                cast(c(&s, "i"), DataType::Float64).unwrap(),
            ),
            BinaryOp::Mul,
            c(&s, "j"),
        );
        assert_matches_eval(&value, &batch);
        let pred = bin(
            bin(c(&s, "i"), BinaryOp::Mul, c(&s, "j")),
            BinaryOp::Lt,
            bin(c(&s, "f"), BinaryOp::Add, lit_f(0.5)),
        );
        assert_matches_eval(&pred, &batch);
    }

    #[test]
    fn unsupported_shapes_are_rejected() {
        let (s, _batch) = setup(vec![
            ("i", Column::from_i64(vec![1])),
            ("j", Column::from_i64(vec![2])),
            ("f", Column::from_f64(vec![1.0])),
            ("b", Column::from_bool(vec![true])),
            ("s", Column::from_strs(&["x"])),
        ]);
        let i = || c(&s, "i");
        let j = || c(&s, "j");
        let cmp = || bin(i(), BinaryOp::Lt, j());
        let cases: Vec<(&str, Expr)> = vec![
            ("div", bin(i(), BinaryOp::Div, j())),
            ("mod", bin(i(), BinaryOp::Mod, j())),
            ("and", bin(c(&s, "b"), BinaryOp::And, c(&s, "b"))),
            ("or", bin(c(&s, "b"), BinaryOp::Or, c(&s, "b"))),
            ("not", unary(UnaryOp::Not, c(&s, "b")).unwrap()),
            ("is_null", unary(UnaryOp::IsNull, i()).unwrap()),
            ("is_not_null", unary(UnaryOp::IsNotNull, i()).unwrap()),
            ("case", case(cmp(), i(), j()).unwrap()),
            (
                "in_list",
                Expr::InList {
                    expr: Box::new(i()),
                    list: vec![lit_i(1)],
                    negated: false,
                },
            ),
            ("utf8 column", bin(c(&s, "s"), BinaryOp::Eq, c(&s, "s"))),
            (
                "utf8 literal",
                bin(c(&s, "s"), BinaryOp::Eq, lit(ScalarValue::Utf8("x".into()))),
            ),
            ("boolean column", bin(c(&s, "b"), BinaryOp::Eq, c(&s, "b"))),
            (
                "boolean literal",
                bin(c(&s, "b"), BinaryOp::Eq, lit(ScalarValue::Boolean(true))),
            ),
            (
                "float to int cast",
                cast(c(&s, "f"), DataType::Int64).unwrap(),
            ),
            ("comparison of comparisons", bin(cmp(), BinaryOp::Eq, cmp())),
        ];
        for (name, e) in &cases {
            assert!(!supports(e), "{name} must be unsupported");
            assert!(
                matches!(compile(e), Err(EngineError::Unimplemented(_))),
                "{name} must fail to compile"
            );
        }
        // Sanity: the same builders with supported operands do compile.
        assert!(supports(&bin(i(), BinaryOp::Add, c(&s, "f"))));
    }

    #[test]
    fn column_type_mismatch_at_run_is_type_error() {
        let (s_int, _) = setup(vec![("a", Column::from_i64(vec![1, 2]))]);
        let kernel = compile(&bin(c(&s_int, "a"), BinaryOp::Add, lit_i(1))).unwrap();
        let (_, batch_f) = setup(vec![("a", Column::from_f64(vec![1.0, 2.0]))]);
        assert!(matches!(kernel.run(&batch_f), Err(EngineError::Type(_))));
    }

    #[test]
    fn column_length_mismatch_at_run_is_plan_error() {
        let schema = Arc::new(Schema::new(vec![Field::new("a", DataType::Int64, true)]));
        let kernel = compile(&bin(c(&schema, "a"), BinaryOp::Add, lit_i(1))).unwrap();
        let short = Batch {
            schema: schema.clone(),
            columns: vec![Column::from_i64(vec![1, 2])],
            num_rows: 3,
        };
        assert!(matches!(kernel.run(&short), Err(EngineError::Plan(_))));
    }
}
