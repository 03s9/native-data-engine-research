# Resultado — Experimento H1a (Cranelift JIT × avaliador vetorizado)

**Veredito.** H1a, como escrita na RFC-0001 §7.2, **não se sustenta** neste hardware: `ECI ≥ 4` não basta para o JIT ganhar.
Uma versão mais estreita se sustenta: nas três expressões com 7 ou 8 operadores, o JIT foi de 1,9x a 2,7x mais rápido por tupla
e ganhou também ponta a ponta (compilação incluída) a partir de 200 mil tuplas. Com 4 operadores o ganho por tupla é de ~1,45x,
mas a compilação só se paga perto de 1 milhão de tuplas. Com 3 operadores ou menos o JIT empata ou perde.

Tudo abaixo foi medido em **um laptop** (Apple Silicon aarch64, 14 CPUs lógicas), uma thread, build release, com a máquina
ocupada (`uptime` no início de cada execução: load 6,2 e 6,0). Não diz nada sobre H1b (LLVM), H2 (x86/AVX-512) nem H3.

## Números

Duas execuções completas (run 1 / run 2), mediana de 9 repetições por ponto. Speedup = `t_vec / t_jit` por tupla, sem compilação.
"Ponta a ponta" compara `eval` contra compilar + executar + liberar o kernel numa única região cronometrada.

| Expressão | ECI | Operadores | Speedup em 200k | Speedup em 1M | Ponta a ponta em 200k | Ponta a ponta em 1M |
| --- | --- | --- | --- | --- | --- | --- |
| `a+b` | 3,0 | 1 | 0,37x / 0,27x | 0,36x / 0,38x | vetorizado | vetorizado |
| `a+b+c+d` | 9,0 | 3 | 1,04x / 1,04x | 1,08x / 1,08x | vetorizado | vetorizado |
| `(x*y)+(x-y)` | 7,0 | 3 | 1,15x / 1,14x | 1,16x / 1,14x | vetorizado | vetorizado |
| `cast(a)*x + y` | 10,5 | 3 | 0,99x / 0,88x | 1,03x / 1,01x | vetorizado | vetorizado |
| `((a+b)*c)-(d*2)` | 10,0 | 4 | 1,48x / 1,43x | 1,50x / 1,49x | vetorizado (empate: 0,182 × 0,184 ms) | JIT |
| `((a+b)*c) > (d*1000)` | 10,0 | 4 | 1,46x / 1,45x | 1,47x / 1,49x | vetorizado | JIT |
| soma de 8 termos | 21,0 | 7 | 2,29x / 1,88x | 2,35x / 2,29x | JIT | JIT |
| `float8` | 15,0 | 7 | 2,30x / 2,32x | 2,51x / 2,30x | JIT | JIT |
| `arith8` | 18,0 | 8 | 2,70x / 2,67x | 2,68x / 2,69x | JIT | JIT |

### Critérios de H1a (expressões com ECI ≥ 4)

| Critério | Resultado |
| --- | --- |
| `T_comp ≤ 1,5 ms` | Passa a quente em todas: mediana de 30 compilações entre 0,04 e 0,12 ms. **A primeira compilação de cada expressão é excluída dessa mediana** e custou 0,09 a 0,28 ms; a primeira do processo custou 3,5 ms (run 1) e 0,27 ms (run 2), ou seja, pode passar do limite. |
| Speedup ≥ 1,3x em 200k e 1M | Passa em 5 de 8 (as de 4, 7 e 8 operadores). Falha nas três de 3 operadores. |
| `N_crossover ≤ 200 000` | Depende de qual `T_comp` se usa. Com a mediana a quente: 47k–107k nas três maiores, 153k–200k nas duas de 4 operadores. Medido ponta a ponta: as três maiores ganham em 200k; as duas de 4 operadores **perdem em 200k** e só ganham em 1M. |

Por isso as expressões de 4 operadores não contam como aprovadas: o ganho por tupla é estável, mas o cruzamento ≤ 200k só aparece
no modelo com compilação a quente.

## O que explica os números

- **O custo do JIT é quase constante:** 0,54 a 0,69 ns por tupla em 200k e 1M, de 1 a 8 operadores.
  O Cranelift não autovetoriza; o kernel é um laço escalar. A causa exata do piso de ~0,55 ns não foi isolada:
  um laço `a+b` compilado pelo LLVM sobre os mesmos buffers roda a 0,15–0,20 ns, então tráfego de memória sozinho não explica.
- **O custo do avaliador cresce com o número de operadores:** de 0,15 a 0,23 ns por tupla por operador em 200k.
  O custo varia com o tipo de operador (multiplicação Int64 custa mais que soma), não só com a contagem.
- Logo o JIT só ganha quando a soma dos custos dos nós passa do piso do laço escalar, o que aqui aconteceu por volta de 4 operadores.
  Isso é inferido dos tempos; não houve disassembly, profiler nem contadores de hardware.
- **ECI não prevê o resultado.** `a+b+c+d` (ECI 9,0) e `cast(a)*x+y` (ECI 10,5) falham, enquanto expressões de ECI 10,0 passam no speedup.
  Nestas nove expressões, a contagem de operadores acompanhou o speedup melhor que o ECI (amostra pequena).

## O baseline foi corrigido antes de medir

A primeira versão do avaliador copiava a coluna inteira a cada referência. Contra ela, `a+b+c+d` dava ~1,8x a favor do JIT.
Depois de remover as cópias (commit `2366d10`, avaliador 35–44% mais rápido nas expressões com mais de um operador), o mesmo caso
ficou em ~1,05x. Os dados brutos de antes da correção não estão no repositório; um auditor independente reproduziu a diferença
reconstruindo o commit anterior. O baseline ainda aloca um vetor por nó; um avaliador que reutilizasse buffers reduziria as vantagens do JIT.
Há um viés pequeno contra o JIT: a saída dele é zerada antes de ser escrita (≤ 0,06 ns por tupla).

## Correção e segurança do JIT

- `engine/tests/jit_diff.rs`: 7 200 casos (expressão × lote) comparados contra o avaliador, 0 divergências.
  Diferença documentada: quando o resultado é NaN, os bits do NaN podem diferir (49 590 linhas em 22 casos); causa não investigada.
- Auditoria independente dos blocos `unsafe` (testes adversariais com lotes inconsistentes, sob `libgmalloc`): nenhuma leitura ou
  escrita fora dos limites encontrada.
- Defeito encontrado na auditoria e corrigido antes destas medições: cada compilação vazava 16 kB de memória executável
  (o `Drop` do cranelift-jit 0.138 não desmapeia as páginas). `CompiledKernel` agora libera no `Drop`; há teste de regressão.
- Não corrigido: expressões com profundidade na casa dos milhares estouram a pilha em `jit::compile` e em `eval` (o processo aborta).

## Limites

- Máquina não estava ociosa. O tempo do avaliador em 200k variou até ~36% entre execuções em uma auditoria com seis repetições;
  em 1M a variação foi ≤ 7% (exceto `a+b`). O tempo do JIT variou ≤ 8%. Prefira os números de 1M.
- A grade de N é grossa (65 536 → 200 000 → 1 000 000): "ganha em 200k" significa que o cruzamento real está entre 65 536 e 200 000.
- Só nove expressões, sem NULLs nos dados medidos, só Int64/Float64, sem `Div`, `And/Or`, `Case` nem strings.
- Dados brutos: `docs/results/h1a-run{1,2}.csv` e `.txt`. Reproduzir: `cd engine && cargo run --release --bin h1a -- --csv saida.csv`.

## Consequência para o plano

T6 (ligar o tier adaptativo ao executor) só compensaria com um gatilho por **número de operadores** (≥ 7 nestes dados), não por ECI,
e teria de contar a compilação a frio. A alternativa com mais retorno provável é atacar o piso medido do laço escalar:
gerar código vetorial no JIT (tipos `I64X2`/`F64X2` do Cranelift), o que este experimento não testou.
