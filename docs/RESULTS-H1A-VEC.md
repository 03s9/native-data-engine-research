# Resultado — JIT vetorial (I64X2/F64X2) × JIT escalar × avaliador

**Estado: regra de decisão registrada antes das execuções completas; números ainda não medidos.**

## Regra de decisão (escrita antes de medir)

Fixada depois de uma execução `--quick` exploratória (que mostrou ganho sem `Mul` Int64 e perda com ele)
e antes das duas execuções completas:

- Política candidata: `compile` usa `Codegen::Vector` só quando a expressão **não tem `Mul` Int64**;
  saídas Boolean entram na política só se `cmp_add` e `fcmp` ganharem pelo mesmo critério.
- O padrão muda de `Scalar` para essa política só se **toda** expressão que ela vetoriza tiver
  `t_jit / t_jitv > 1,08` (acima do ruído de ~8% do JIT) em 200k **e** 1M, **nas duas** execuções.
  Caso contrário o padrão fica `Scalar` e o código vetorial fica opt-in (`compile_with`).
- A política é ajustada sobre este mesmo conjunto de expressões; não há conjunto de validação separado.
