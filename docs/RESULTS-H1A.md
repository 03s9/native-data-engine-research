# Resultado — Experimento H1a (Cranelift JIT × avaliador vetorizado)

**Veredito.** H1a, como escrita na RFC-0001 §7.2, **não se sustenta** neste hardware: `ECI ≥ 4` não basta para o JIT ganhar.
Uma versão mais estreita se sustenta: em expressões com cerca de 7 ou mais operadores, o JIT foi 2,2x a 2,7x mais rápido,
com cruzamento do modelo entre 47 mil e 73 mil tuplas. Em expressões pequenas o JIT empata ou perde.

Tudo abaixo foi medido em **um laptop** (Apple Silicon aarch64, 14 CPUs lógicas), uma thread, build release, com a máquina
ocupada (load average ≈ 6). Não diz nada sobre H1b (LLVM), H2 (x86/AVX-512) nem H3.

## Números (duas execuções completas; mediana de 9 repetições por ponto)

Speedup = `t_vec / t_jit` por tupla. Cruzamento do modelo = `T_comp / (t_vec − t_jit)` com as taxas de N = 200 000.

| Expressão | ECI | Operadores | `T_comp` mediana | Speedup em 200k (run 1 / run 2) | Speedup em 1M | Cruzamento do modelo | Critérios H1a |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `a+b` | 3,0 | 1 | 0,04 ms | 0,32x / 0,37x | 0,30x / 0,36x | nunca | fora do alvo (ECI < 4) |
| `a+b+c+d` | 9,0 | 3 | 0,05 ms | 0,81x / 1,01x | 1,04x / 1,05x | nunca / 10 M | **falha** |
| `(x*y)+(x-y)` | 7,0 | 3 | 0,04 ms | 1,13x / 1,11x | 1,15x / 1,12x | 544k / 659k | **falha** |
| `cast(a)*x + y` | 10,5 | 3 | 0,04 ms | 0,92x / 0,77x | 0,99x / 0,99x | nunca | **falha** |
| `((a+b)*c)-(d*2)` | 10,0 | 4 | 0,05 ms | 1,46x / 1,46x | 1,52x / 1,48x | 166k / 166k | passa |
| `((a+b)*c) > (d*1000)` | 10,0 | 4 | 0,05 ms | 1,44x / 1,36x | 1,41x | 196k / 227k | passa no run 1, **falha o cruzamento no run 2** |
| soma de 8 termos | 21,0 | 7 | 0,05 ms | 2,17x / 2,20x | 2,31x / 2,20x | 68k / 66k | passa |
| `float8` | 15,0 | 7 | 0,05 ms | 2,42x / 2,30x | 2,45x / 2,37x | 68k / 73k | passa |
| `arith8` | 18,0 | 9 | 0,05 ms | 2,65x / 2,53x | 2,71x / 2,69x | 48k / 52k | passa |

## O que explica os números

- **O custo do JIT é quase constante:** 0,53 a 0,68 ns por tupla em todas as expressões, de 1 a 9 operadores.
  O Cranelift não autovetoriza; o kernel é um laço escalar e o custo é dominado por carregar e gravar memória.
- **O custo do avaliador cresce com o número de operadores:** cerca de 0,17 a 0,20 ns por tupla por operador,
  porque cada nó é um laço NEON autovetorizado pelo LLVM mais um vetor intermediário.
- Logo o JIT só ganha quando a expressão tem operadores suficientes (por volta de 4) para o custo somado dos nós passar do
  custo fixo do laço escalar. Essa explicação é inferida dos números, não foi confirmada com profiler nem contadores de hardware.
- **`T_comp` é ~30x menor que o suposto na RFC** (0,04–0,05 ms contra o limite de 1,5 ms). A compilação não é o gargalo;
  o gargalo é o laço escalar gerado.
- **ECI não prevê o resultado.** `a+b+c+d` (ECI 9,0) e `cast(a)*x+y` (ECI 10,5) falham, enquanto expressões de ECI 10,0 passam.
  O número de operadores previu melhor que o ECI nestas nove expressões.

## O baseline foi corrigido antes de medir

A primeira versão do avaliador copiava a coluna inteira a cada referência. Contra ela, `a+b+c+d` dava 1,8x a favor do JIT.
Depois de remover as cópias (commit T4b, avaliador 35–44% mais rápido), o mesmo caso ficou em ~1,0x.
Os números acima usam o baseline corrigido. Ele ainda aloca um vetor por nó e copia a validade; um avaliador mais afinado
reduziria ainda mais as vantagens do JIT.

## Correção do JIT

`engine/tests/jit_diff.rs`: 7 200 casos (expressão × lote) comparados contra o avaliador, 0 divergências.
Diferença documentada: quando o resultado é NaN, os bits do NaN podem diferir (49 590 linhas em 22 casos); causa não investigada.

## Limites

- Máquina não estava ociosa; diferenças abaixo de ~15% entre execuções não são confiáveis (ver `a+b+c+d` em 200k: 0,81x e 1,01x).
- O cruzamento **empírico** ponta a ponta (compilar + executar numa região cronometrada) ficou acima do cruzamento do modelo:
  200k para as expressões grandes e 1M ou 4M para as limítrofes. A grade de N é grossa (65 536 → 200 000 → 1 000 000),
  então o valor real fica em algum ponto entre dois pontos da grade.
- Só nove expressões, sem NULLs nos dados medidos, só Int64/Float64, sem `Div`, `And/Or`, `Case` nem strings.
- Dados brutos: `docs/results/h1a-run{1,2}.csv` e `.txt`. Reproduzir: `cd engine && cargo run --release --bin h1a -- --csv saida.csv`.

## Consequência para o plano

T6 (ligar o tier adaptativo ao executor) só compensaria com um gatilho por **número de operadores**, não por ECI,
e só para projeções com expressões grandes. A alternativa com mais retorno provável é atacar o gargalo medido:
gerar código vetorial no JIT (tipos `I64X2`/`F64X2` do Cranelift), o que este experimento não testou.
