# Handoff — experimento H1a (Cranelift)
> Último agente: claude-code@2026-11-24
- Meta: medir se o JIT Cranelift se paga contra o avaliador vetorizado (docs/PLAN.md).
- Branch: `feat/cranelift-h1a` (sobre `feat/engine-mvp`); ambas locais, NÃO enviadas ao GitHub (push precisa de autorização).
- Estado: T1–T5 e T7 concluídas. Resultado em `docs/RESULTS-H1A.md`: H1a como escrita não se sustenta; JIT ganha 1,9–2,7x só com ≥ 7 operadores.
- Testes: `cd engine && cargo test` = 222 unit + 32 e2e + 7 jit_diff + 1 doctest; clippy -D warnings e fmt limpos.
- CI: `.github/workflows/ci.yml` nunca executou (precisa push). AVX2 e Cranelift em x86 nunca compilados aqui.
- Pendente: decisão sobre T6 (tier adaptativo) ou JIT vetorial; estouro de pilha em expressões muito profundas.
- Fora do repo: `~/.gemini/config/mcp_local_ai.py` ganhou `DELEGATE_TIMEOUT_S` e modelos Neon/Hydrogen (não commitado no repo agents-and-skills).
