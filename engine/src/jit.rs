//! Cranelift JIT for arithmetic expressions (experiment H1a, see docs/PLAN.md).
//! One fused scalar loop per expression: no intermediate columns.
//!
//! Kernel ABI: `extern "C" fn(cols: *const *const u8, n: usize, out: *mut u8)`.
//! `cols[k]` points at the values of the k-th referenced column (8 bytes per row),
//! `out` at the output values (8 bytes per row for Int64/Float64, 1 byte per row for Boolean).
//! NULLs are handled outside the kernel: output validity = AND of the input validities.
use crate::column::{Batch, Column, Values};
use crate::expr::{BinaryOp, Expr};
use crate::types::{DataType, EngineError, Result};
use cranelift_codegen::ir::condcodes::IntCC;
use cranelift_codegen::ir::{types, AbiParam, InstBuilder, MemFlagsData, Value};
use cranelift_codegen::settings::{self, Configurable};
use cranelift_frontend::{FunctionBuilder, FunctionBuilderContext};
use cranelift_jit::{JITBuilder, JITModule};
use cranelift_module::{default_libcall_names, Linkage, Module};
use std::time::{Duration, Instant};

type KernelFn = unsafe extern "C" fn(*const *const u8, usize, *mut u8);

pub struct CompiledKernel {
    // Owns the executable memory `func` points into; must outlive every call.
    _module: JITModule,
    func: KernelFn,
    /// Batch column index for each kernel input slot.
    inputs: Vec<usize>,
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

/// True when `compile` accepts `expr`. SPIKE SCOPE: Int64 columns combined with `Add` only.
pub fn supports(expr: &Expr) -> bool {
    match expr {
        Expr::Column { data_type, .. } => *data_type == DataType::Int64,
        Expr::Binary {
            left,
            op: BinaryOp::Add,
            right,
            return_type,
        } => *return_type == DataType::Int64 && supports(left) && supports(right),
        _ => false,
    }
}

struct Gen<'a, 'b> {
    b: &'a mut FunctionBuilder<'b>,
    bases: &'a [Value],
    mem: MemFlagsData,
    inputs: &'a [usize],
    offset: Value,
}

impl Gen<'_, '_> {
    fn expr(&mut self, e: &Expr) -> Result<Value> {
        match e {
            Expr::Column { index, .. } => {
                let slot = self
                    .inputs
                    .iter()
                    .position(|i| i == index)
                    .expect("input collected");
                let addr = self.b.ins().iadd(self.bases[slot], self.offset);
                Ok(self.b.ins().load(types::I64, self.mem, addr, 0))
            }
            Expr::Binary {
                left,
                op: BinaryOp::Add,
                right,
                ..
            } => {
                let l = self.expr(left)?;
                let r = self.expr(right)?;
                Ok(self.b.ins().iadd(l, r))
            }
            _ => Err(unsupported("jit: expression shape")),
        }
    }
}

fn collect_inputs(e: &Expr, out: &mut Vec<usize>) {
    match e {
        Expr::Column { index, .. } => {
            if !out.contains(index) {
                out.push(*index);
            }
        }
        Expr::Binary { left, right, .. } => {
            collect_inputs(left, out);
            collect_inputs(right, out);
        }
        Expr::Unary { expr, .. } | Expr::Cast { expr, .. } => collect_inputs(expr, out),
        _ => {}
    }
}

pub fn compile(expr: &Expr) -> Result<CompiledKernel> {
    if !supports(expr) {
        return Err(unsupported("jit: expression not supported"));
    }
    let start = Instant::now();
    let mut inputs = Vec::new();
    collect_inputs(expr, &mut inputs);

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
        let offset = b.ins().ishl_imm_u(i, 3);
        let value = Gen {
            b: &mut b,
            bases: &bases,
            mem,
            inputs: &inputs,
            offset,
        }
        .expr(expr)?;
        let dst = b.ins().iadd(out, offset);
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
        _module: module,
        func,
        inputs,
        out_type: expr.return_type(),
        compile_time: start.elapsed(),
    })
}

impl CompiledKernel {
    /// Same result as `eval::eval(expr, batch)` for every supported expression.
    pub fn run(&self, batch: &Batch) -> Result<Column> {
        let n = batch.num_rows;
        let mut ptrs: Vec<*const u8> = Vec::with_capacity(self.inputs.len());
        let mut validity: Option<Vec<bool>> = None;
        for &ci in &self.inputs {
            let col = batch
                .columns
                .get(ci)
                .ok_or_else(|| EngineError::Plan("jit: column index out of range".into()))?;
            match &col.values {
                Values::Int64(v) => ptrs.push(v.as_ptr().cast()),
                _ => {
                    return Err(EngineError::Type(
                        "jit: column type differs from compiled kernel".into(),
                    ))
                }
            }
            if let Some(m) = &col.validity {
                match &mut validity {
                    None => validity = Some(m.clone()),
                    Some(acc) => acc.iter_mut().zip(m).for_each(|(a, b)| *a &= *b),
                }
            }
        }
        debug_assert_eq!(self.out_type, DataType::Int64);
        let mut out = vec![0i64; n];
        // SAFETY: every pointer in `ptrs` addresses a live Vec<i64> of exactly `n` elements (all batch
        // columns have `num_rows` rows), `out` has `n` elements, and the kernel reads/writes rows 0..n only.
        unsafe { (self.func)(ptrs.as_ptr(), n, out.as_mut_ptr().cast()) };
        Ok(Column {
            values: Values::Int64(out),
            validity,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::eval::{binary, col, eval};
    use crate::types::{Field, Schema};
    use std::sync::Arc;

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
}
