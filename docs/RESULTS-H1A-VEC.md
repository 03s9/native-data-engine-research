# Resultado — JIT vetorial (I64X2/F64X2) × JIT escalar × avaliador

**Veredito.** A hipótese se sustenta, com uma exceção. Gerar um laço de duas linhas por iteração em registradores
`I64X2`/`F64X2` (mais um laço escalar para a última linha ímpar) baixou `t_jit` de ~0,55 para ~0,29 ns por tupla
em todas as expressões **sem multiplicação Int64**: de 1,52x a 1,99x sobre o JIT escalar, em 200k e 1M, nas duas execuções.
Com multiplicação Int64 o resultado depende de quantas há: uma só (`mul2`) ganha 1,8–2,0x, duas empatam ou perdem
(`arith4` 1,08–1,10x, `cmp` 0,91–0,94x), quatro perdem (`arith8` 0,62–0,65x).
A regra registrada antes das execuções completas foi cumprida, e `compile` agora vetoriza toda expressão sem `Mul` Int64.

Tudo abaixo foi medido em **um laptop** (Apple Silicon aarch64, 14 CPUs lógicas), uma thread, build release, com a máquina
ocupada (`uptime` no início: load 5,76 na execução 1 e 4,84 na execução 2). Não diz nada sobre x86: lá o caminho vetorial
nem é gerado (`compile_with` cai no escalar fora de aarch64), porque as instruções vetoriais nunca foram compiladas nem testadas em x86.

## Regra de decisão (registrada antes de medir, commit `dfe3861`)

Fixada depois de uma execução `--quick` exploratória (ganho sem `Mul` Int64, perda com ele) e antes das duas completas:

- Política candidata: `compile` usa `Codegen::Vector` só quando a expressão **não tem `Mul` Int64**.
  Saídas Boolean entram na política só se `cmp_add` e `fcmp` ganharem pelo mesmo critério.
- O padrão troca `Scalar` por essa política só se **toda** expressão que ela vetoriza tiver `t_jit / t_jitv > 1,08`
  (acima do ruído de ~8% do JIT) em 200k **e** 1M, **nas duas** execuções. Senão o padrão fica `Scalar`.
- A política foi ajustada sobre este mesmo conjunto de expressões; não há conjunto de validação separado.

**Resultado da regra:** cumprida. O menor ganho entre as vetorizadas foi 1,52x (`add4`, 1M, execução 1); `cmp_add` e `fcmp`
ganharam, então Boolean entrou. Commit `256cb2a`. A política deixa ganho na mesa: `mul2` (um `Mul` Int64) ganharia ~1,9x
vetorizado, mas fica escalar. A política não foi refinada depois de ver isso, para não ajustar a regra aos dados duas vezes.

## Números

Duas execuções completas (execução 1 / execução 2), mediana de 9 repetições por ponto, os três caminhos alternados na mesma
região com o mesmo K. `jit` = laço escalar (o kernel do H1a), `jitv` = laço vetorial, `vec` = avaliador.
Todos os números são deste processo; nada vem do CSV de H1a.

### Por tupla (sem compilação)

| Expressão | `Mul` Int64 | `jitv` ns/t em 1M | `jit` ns/t em 1M | `jitv`/`jit` em 200k | `jitv`/`jit` em 1M | `jitv`/`vec` em 200k | `jitv`/`vec` em 1M |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `add2` (`a+b`) | 0 | 0,286 / 0,284 | 0,553 / 0,546 | 1,86x / 1,85x | 1,93x / 1,92x | 0,70x / 0,69x | 0,70x / 0,58x |
| `add4` | 0 | 0,379 / 0,384 | 0,581 / 0,588 | 1,52x / 1,64x | 1,53x / 1,53x | 1,64x / 1,67x | 1,66x / 1,69x |
| `add8` | 0 | 0,368 / 0,367 | 0,622 / 0,628 | 1,88x / 1,74x | 1,69x / 1,71x | 3,44x / 3,87x | 3,85x / 3,98x |
| `float4` | 0 | 0,289 / 0,292 | 0,544 / 0,535 | 1,86x / 1,91x | 1,88x / 1,83x | 2,04x / 1,84x | 2,18x / 2,05x |
| `float8` | 0 | 0,284 / 0,296 | 0,533 / 0,548 | 1,82x / 1,85x | 1,88x / 1,85x | 4,13x / 4,41x | 4,48x / 4,48x |
| `mixed` (`cast(a)*x+y`) | 0 | 0,316 / 0,315 | 0,548 / 0,550 | 1,76x / 1,74x | 1,73x / 1,75x | 1,71x / 1,71x | 1,74x / 1,77x |
| `cmp_add` (`a+b > c+d`) * | 0 | 0,300 / 0,301 | 0,558 / 0,565 | 1,86x / 1,88x | 1,86x / 1,87x | 1,79x / 1,57x | 1,85x / 1,75x |
| `fcmp` (`x*y > x-y`) * | 0 | 0,264 / 0,262 | 0,518 / 0,516 | 1,88x / 1,99x | 1,96x / 1,97x | 2,04x / 1,75x | 2,12x / 2,13x |
| `mul2` (`a*b`) * | 1 | 0,291 / 0,281 | 0,563 / 0,553 | 1,91x / 1,82x | 1,93x / 1,97x | 0,95x / 0,94x | 0,97x / 0,97x |
| `arith4` | 2 | 0,539 / 0,547 | 0,593 / 0,598 | 1,08x / 1,10x | 1,10x / 1,09x | 1,51x / 1,62x | 1,60x / 1,67x |
| `cmp` | 2 | 0,615 / 0,626 | 0,576 / 0,576 | 0,91x / 0,94x | 0,94x / 0,92x | 1,36x / 1,31x | 1,33x / 1,32x |
| `arith8` | 4 | 1,041 / 1,052 | 0,642 / 0,658 | 0,65x / 0,63x | 0,62x / 0,63x | 1,64x / 1,62x | 1,69x / 1,67x |

\* Acrescentadas nesta rodada, depois das nove originais, para separar o caminho de saída Boolean da multiplicação Int64
(`cmp` mistura os dois). As nove primeiras são as mesmas expressões de `RESULTS-H1A.md`.

### Com a política padrão (o que `compile` escolhe), contra o avaliador

Kernel escolhido: vetorial em todas exceto `arith4`, `arith8`, `cmp` e `mul2` (escalar). `T_comp` = mediana de 30 compilações a quente.
"Ponta a ponta" = compilar + executar + liberar o kernel contra `eval`, numa única região cronometrada.

| Expressão | Kernel | `T_comp` (µs) | Speedup em 200k | Speedup em 1M | Cruzamento no modelo (taxas de 200k) | Ponta a ponta em 200k | Ponta a ponta em 1M |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `add2` | vetorial | 85 / 76 | 0,70x / 0,69x | 0,70x / 0,58x | nenhum | vetorizado | vetorizado |
| `add4` | vetorial | 65 / 70 | 1,64x / 1,67x | 1,66x / 1,69x | 264k / 289k | vetorizado | JIT |
| `add8` | vetorial | 84 / 81 | 3,44x / 3,87x | 3,85x / 3,98x | 100k / 75k | JIT | JIT |
| `arith4` | escalar | 49 / 51 | 1,40x / 1,47x | 1,46x / 1,53x | 203k / 179k | vetorizado | JIT |
| `arith8` | escalar | 55 / 53 | 2,53x / 2,56x | 2,74x / 2,68x | 53k / 50k | JIT | JIT |
| `float4` | vetorial | 60 / 64 | 2,04x / 1,84x | 2,18x / 2,05x | 191k / 258k | vetorizado | JIT |
| `float8` | vetorial | 78 / 76 | 4,13x / 4,41x | 4,48x / 4,48x | 82k / 75k | JIT | JIT |
| `mixed` | vetorial | 66 / 61 | 1,71x / 1,71x | 1,74x / 1,77x | 288k / 267k | vetorizado | JIT |
| `cmp` | escalar | 49 / 48 | 1,49x / 1,39x | 1,43x / 1,43x | 174k / 208k | vetorizado | JIT |
| `cmp_add` | vetorial | 75 / 74 | 1,79x / 1,57x | 1,85x / 1,75x | 308k / 420k | vetorizado | JIT |
| `fcmp` | vetorial | 67 / 68 | 2,04x / 1,75x | 2,12x / 2,13x | 232k / 350k | vetorizado | JIT |
| `mul2` | escalar | 44 / 36 | 0,50x / 0,51x | 0,50x / 0,49x | nenhum | vetorizado | vetorizado |

Efeito sobre os critérios de H1a (RFC-0001 §7.2): com a política, as oito expressões originais de ECI ≥ 4 passam no speedup
≥ 1,3x em 200k e 1M nas duas execuções. Em H1a, três delas falhavam: `add4`, `float4` e `mixed`.
O cruzamento ≤ 200k **continua falhando** para `add4`, `mixed`, `cmp_add` e `fcmp`, e fica no limite para `float4`, `arith4` e `cmp` (uma execução abaixo, outra acima de 200k).
A compilação vetorial custa ~20–40 µs a mais (dois corpos de laço), o que come parte do ganho por tupla em N pequeno.
Ponta a ponta, o JIT ganha em 200k nas mesmas três de antes (`add8`, `arith8`, `float8`) e em 1M em todas exceto `add2` e `mul2`.

## O que explica os números

- **O piso caiu pela metade, não por operação.** O laço vetorial custa ~0,26–0,32 ns por tupla, ou ~0,55–0,6 ns por iteração
  de duas linhas, praticamente o mesmo custo por iteração do laço escalar. Isso vale de 1 a 7 operadores sem `Mul` Int64:
  `float8` (7 operadores) custa o mesmo que `add2`. Então o piso de ~0,55 ns do H1a é custo **por iteração**, não por operação.
- **Causa provável do piso (inferida, não medida com contadores):** o disassembly do Cranelift (`Context::set_disasm`)
  mostra que o laço escalar faz dois desvios tomados por linha: o teste fica no topo (`subs` + `b.lo` para o corpo) e o corpo
  termina com um `b` incondicional de volta ao teste. O Cranelift não rotaciona o laço. Exemplo de `a+b`:
  `ldr x15,[x11,x6,LSL #3]; ldr x0,[x12,x6,LSL #3]; add x15,x15,x0; str x15,[x2,x6,LSL #3]; add x6,x6,#1; b label1`.
  **Por isso o ganho não pode ser atribuído só ao SIMD:** reduzir pela metade o número de iterações explica pelo menos parte.
  Um laço escalar desenrolado em duas linhas separaria os dois efeitos; não foi testado.
- **Multiplicação Int64 vetorial.** O NEON do aarch64 não tem multiplicação de lanes de 64 bits. O Cranelift 0.138 baixa
  `imul.i64x2` para 7 instruções (regra ISLE em `isa/aarch64/lower.isle`, confirmada no disassembly):
  `rev64 v.4s; mul v.4s; xtn v.2s; addp v.4s; xtn v.2s; shll v.2d, #32; umlal v.2d`.
  O escalar usa um só `madd`. Com um `Mul` (`mul2`) a sequência ainda cabe na folga do laço e o vetorial ganha 1,9x;
  com duas empata ou perde; com quatro (`arith8`) perde 0,63x. Leitura: as sequências de 7 instruções passam a dominar
  a iteração assim que há mais de uma. É uma inferência a partir dos tempos; não houve profiler.
- **Saída Boolean.** As comparações vetoriais (`cmeq`/`cmgt`/`fcmgt` em `.2d`) produzem máscaras de lanes; o kernel faz AND com 1
  e grava dois bytes (`mov x, v.d[0]`, `strb`, `mov x, v.d[1]`, `strb`). Mesmo assim `cmp_add` e `fcmp` ganharam ~1,9x,
  então a perda de `cmp` vem das multiplicações, não do caminho Boolean.
- **Literais:** o `splat` sai do laço (é içado pelo Cranelift; visto no disassembly de uma comparação com literal).

## Correção

- `engine/tests/jit_diff.rs` agora roda cada caso pelos dois codegens e inclui o tamanho 3 (tamanhos 0, 1, 2, 3, 7, 8, 9, 63,
  64, 65, 8191, 8192, 8193): 7 800 casos por codegen, 0 divergências em ambos.
- Linhas em que os dois lados dão NaN com bits diferentes: **0** nesta execução (escalar e vetorial). `RESULTS-H1A.md` relatou
  49 590 linhas em 22 casos. A diferença não foi explicada. Hipótese não verificada: acrescentar o tamanho 3 mudou a sequência
  do gerador aleatório e, com ela, as expressões sorteadas.
- Testes unitários novos em `jit.rs`: toda expressão suportada é comparada com `eval` nos dois codegens; bytes de saída Boolean
  do kernel vetorial são exatamente 0 ou 1 e nada é escrito além de `n` (n = 0, 1, 2, 3, 7, 8, lidos como `u8` crus);
  e a escolha de codegen da política padrão.
- Os acessos vetoriais (16 bytes sobre buffers alinhados a 8) não declaram `aligned`; os escalares continuam com `trusted()`.
- O `h1a` compara as saídas dos dois kernels com `eval` linha a linha antes de cronometrar cada N.

## Limites

- Um laptop, máquina ocupada (load ~5). A razão `jitv`/`jit` variou ≤ 8% entre as duas execuções em todas as expressões.
  As razões contra `eval` variam mais: até ~15% em 200k (`fcmp`, `cmp_add`) e 17% em `add2` a 1M, como em H1a.
  Prefira os números de 1M.
- A política foi escolhida e julgada nas mesmas 12 expressões, sem conjunto de validação. São 3 expressões com `Mul` Int64
  em quantidades diferentes (1, 2 e 4); a fronteira "uma ganha, duas empatam" é uma observação, não uma regra validada.
- Só aarch64; sem NULLs nos dados medidos; só Int64/Float64; sem `Div`, `And/Or`, `Case` nem strings.
- `eval` agora chama `check_depth` (percorre a árvore uma vez, ≤ 128 níveis); o custo é desprezível nesses N, mas os números de
  `vec` não são estritamente os mesmos de H1a.
- Não testado: desenrolar o laço escalar (o controle que separaria SIMD de custo por iteração), laço vetorial de 4 linhas,
  emular `Mul` Int64 lane a lane.
- Dados brutos: `docs/results/h1a-vec-run{1,2}.csv` e `.txt` (a primeira linha do `.txt` tem o `uptime`).
  Reproduzir: `cd engine && cargo build --release --bin h1a && ./target/release/h1a --csv saida.csv`.

## Consequência para o plano

O piso do laço escalar era o gargalo apontado em H1a, e cortá-lo pela metade transforma em ganho por tupla expressões de 3 a 4
operadores que antes empatavam. O cruzamento ≤ 200k continua sendo o critério que reprova a maioria: a compilação (40–85 µs a quente)
pesa tanto quanto o laço. Próximo experimento mais barato: o laço escalar desenrolado em duas linhas, que pode dar o mesmo ganho
às expressões com `Mul` Int64 e diria quanto do ganho é SIMD. T6 continua pendente da decisão do dono do projeto.
