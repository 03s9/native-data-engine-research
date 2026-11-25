# Handoff — engine MVP
- Meta: terminar MVP executável do motor (escopo: sem JIT, sem TPC-H/ClickBench).
- Branch: `feat/engine-mvp` (local, NÃO enviada ao GitHub; push precisa de autorização).
- Estado: slices A,C,D1,D2,E,F,G1,G2 mescladas. `cd engine && cargo test` = 195 unit + 32 e2e; clippy -D warnings limpo; `ndq demo` roda.
- CI: `.github/workflows/ci.yml` (ubuntu+macos) criado; NUNCA executou (precisa push).
- Pendente: validação Fable, push/PR, corrigir achados.
