# PLAN — Experimento H1a: o JIT Cranelift se paga?

**Meta.** Responder com medida, neste Mac (Apple Silicon), a hipótese H1a da RFC-0001 §7.2:
para expressões com `ECI ≥ 4.0`, `T_comp ≤ 1,5 ms`, `N_crossover ≤ 200 000` tuplas e speedup `≥ 1,3x`
sobre o avaliador vetorizado em `N ∈ [2·10⁵, 2·10⁶]`.

**Fora de escopo.** LLVM (H1b), x86/AVX-512 (H2), TPC-H, ClickBench, parser SQL. Um laptop não valida H2.

**Critério de parada.** O resultado negativo também encerra o plano: se o JIT não ganhar, registra-se e não se constrói T6.

## Contrato congelado (T1)

`engine/src/jit.rs`: `supports(&Expr) -> bool`, `compile(&Expr) -> Result<CompiledKernel>`,
`CompiledKernel::run(&Batch) -> Result<Column>`, `CompiledKernel.compile_time`.
ABI do kernel: `extern "C" fn(cols: *const *const u8, n: usize, out: *mut u8)`; um laço escalar fundido por expressão;
NULLs fora do kernel (validade de saída = AND das validades de entrada).
Cranelift **0.138**: `load`/`store` recebem `MemFlagsData::trusted()`; usar `ishl_imm_u`/`iadd_imm_u`;
`FunctionBuilder::finalize(module.target_config())`; argumentos de `jump` são `&[v.into()]`.

## Tarefas

| ID | Dono | Arquivos (exclusivos) | Entrega | Aceite | Depende |
| --- | --- | --- | --- | --- | --- |
| T1 | Fable (orquestrador) | `engine/src/jit.rs`, `Cargo.toml`, `lib.rs` | Protótipo `a + b` Int64 e contrato congelado | teste diferencial passa — **feito** | — |
| T2 ✅ | Haiku `jit-codegen` | `engine/src/jit.rs` | Int64/Float64, literais, `Add/Sub/Mul`, `Neg`, `Cast Int64→Float64`, comparações → Boolean. `Div/Mod/And/Or/Case/InList/Utf8` ficam não suportados | `cargo test jit` + clippy | T1 |
| T3 ✅ | Haiku `jit-diff` | `engine/tests/jit_diff.rs` | Testes diferenciais JIT × `eval` com expressões aleatórias, NULLs, extremos de i64, NaN, lotes de 0/1/8191 linhas | ≥ 2000 casos suportados, 0 divergências | T2 |
| T4 ✅ | Haiku `jit-bench` | `engine/src/bin/h1a.rs` | Mede `T_comp`, `t_vec`, `t_jit` por tupla em grade de ECI × N; localiza o cruzamento real; grava CSV | roda em release; saídas JIT = eval antes de cronometrar | T1 |
| T5 ✅ | Fable (orquestrador) | `docs/RESULTS-H1A.md` | Executa T4 pós-merge, aplica benchmark-checklist, veredito por critério de H1a | cada número com método e limite declarado | T2, T3, T4 |
| T6 ✅ | Haiku (gatilho por nº de operadores, não ECI) | `engine/src/exec.rs` | Ligar `AdaptiveTier` à projeção (troca vetorizado → JIT) | só se T5 mostrar ganho ≥ 1,3x | T5 |
| T7 ✅ | Revisor independente (Fable novo + Gemini Argon; Codex só leu `jit.rs`) | somente leitura | Auditoria do `unsafe` do JIT e dos números de T5 | defeitos confirmados corrigidos | T5 |

## Regras para os agentes

- Cada agente edita só os seus arquivos, em worktree própria; primeiro passo `git merge --ff-only feat/cranelift-h1a`.
- Sem dependências novas. Commit local na própria branch; nunca `git push`.
- O orquestrador mescla e roda `cargo test`, clippy e fmt antes de aceitar: relatório de agente não é evidência.
- Duas falhas seguidas no mesmo conserto → parar e pedir segunda opinião (regra anti-loop).

## Estado (2026-11-26)

T1–T5 e T7 concluídas; resultado em `docs/RESULTS-H1A.md`. T4b (baseline justo do avaliador) foi acrescentada durante a execução.
T6 feita: `project()` usa JIT com ≥ 7 operadores e ≥ 200k linhas, com fallback para `eval`. T8 (Opus): guarda de profundidade 128 e JIT vetorial (I64X2/F64X2, `docs/RESULTS-H1A-VEC.md`). Revisão independente (Fable) incorporada: guarda de profundidade também no executor; JIT no executor só com 1 thread. `AdaptiveTier` (cost.rs) segue sem uso: o gatilho é constante em exec.rs. Não medido: ganho ponta a ponta do T6 no executor.
