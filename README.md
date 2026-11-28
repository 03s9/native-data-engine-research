# Motor nativo de dados

Pesquisa e protótipo de um motor analítico nativo em **Rust**: especificação adversarialmente revisada, um MVP executável e um experimento medido sobre quando um JIT (Cranelift) compensa contra a execução vetorizada.

[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

## Estado

| Parte | Estado | Onde |
| --- | --- | --- |
| Especificação (RFC, arquitetura, modelo de custo, metodologia) | *Authoritative Draft*, revisada na Fase 2 | [`docs/`](docs/) |
| Harness de verificação | 23/23 testes adversariais | [`verification/`](verification/) |
| Motor MVP | Executa filtro, projeção, agregação, join e ordenação com spill; sem parser SQL | [`engine/`](engine/) |
| JIT Cranelift | Experimento H1a concluído: só compensa com ≥ 7 operadores; vetorial (aarch64) 1,5–2x mais rápido que o escalar sem `Mul` Int64 | [RESULTS-H1A](docs/RESULTS-H1A.md), [RESULTS-H1A-VEC](docs/RESULTS-H1A-VEC.md) |
| LLVM, x86/AVX-512, TPC-H, ClickBench | **Fora de escopo / não feitos**. Medidas são de um único laptop Apple Silicon | — |

## Início rápido

Requer Rust 1.75+.

```bash
git clone https://github.com/03s9/native-data-engine-research.git
cd native-data-engine-research/engine
cargo test                                   # 225 unitários + 36 e2e + 7 diferenciais do JIT + 1 doctest
cargo run --release --bin ndq -- demo        # 1M linhas: filtro, group-by, join+top5
cargo run --release --bin h1a -- --csv /tmp/h1a.csv   # reproduz o experimento H1a neste computador
```

## Decisões arquiteturais

| Área | Contrato proposto | Motivo e limite |
| --- | --- | --- |
| Planejamento | IR relacional tipada, plano lógico e redução a pipelines físicos | Permite verificar tipos e aplicar otimizações antes da execução. |
| Execução | Morsels enviados a operadores *push* em um pool Rayon, inclusive por dispatcher de streaming | Mantém o trabalho de CPU em threads próprias e permite distribuição dinâmica sem espera indefinida por novos lotes. |
| I/O | Tokio para leitura assíncrona; ponte de capacidade limitada para o pool Rayon | O envio que pode bloquear fica em `spawn_blocking`; produtores assíncronos recebem contrapressão. Threads Rayon não devem esperar por I/O. |
| Memória | Lotes colunares Apache Arrow compartilhados por referência, `StringView` com descritores de 16 bytes e seleção antes de reunir colunas de payload | O repasse de buffers pode evitar cópia; *slices* podem perder alinhamento, então kernels SIMD devem usar cargas não alinhadas. Materialização tardia exige decisão por seletividade e custo de `gather`. |
| Recursos | Orçamento hierárquico de memória e *spill* determinístico | Partições enviesadas e chaves duplicadas exigem tratamento explícito para evitar estouro de memória ou recursão sem progresso. |
| Especialização | Execução vetorizada inicial; Cranelift/LLVM JIT somente quando o custo de compilação se amortiza | O modelo trata ganho por tupla não positivo e penalidade de *spill* Sethi-Ullman; o gatilho de compilação é separado do ponto de cruzamento. Em Apple Silicon, a troca de código exige permissões JIT e invalidação do cache de instruções. Implementação e calibração pertencem a fases posteriores. |

**Critério de JIT seletivo.** Para uma expressão e um backend, o modelo define `Δt_tuple = t_vec_tuple − t_jit_tuple` e, quando `Δt_tuple > 0`, `N_crossover = T_comp / Δt_tuple`. Se `Δt_tuple ≤ 0`, não há ganho de execução que pague a compilação. Os parâmetros numéricos e as hipóteses de diferença entre AVX-512 e NEON são estimativas de pesquisa, sujeitas à calibração experimental. A política detalhada está em [COST_MODEL_FORMULATION.md](docs/COST_MODEL_FORMULATION.md).

## Entregáveis e leitura

| Arquivo | Conteúdo |
| --- | --- |
| [RFC-0001](docs/RFC-0001-NATIVE-DATA-ENGINE-RESEARCH.md) | Tese, decisões, riscos, roteiro de cinco fases e critérios de aceitação. |
| [ARCHITECTURE_SPEC.md](docs/ARCHITECTURE_SPEC.md) | IR, scheduler, fronteira Tokio/Rayon, Arrow, memória, SIMD e contratos de segurança. Os trechos Rust são especificação, não um crate do motor. |
| [COST_MODEL_FORMULATION.md](docs/COST_MODEL_FORMULATION.md) | Equações de execução, ponto de cruzamento, seletividade e política adaptativa. |
| [BENCHMARK_METHODOLOGY.md](docs/BENCHMARK_METHODOLOGY.md) | Controles experimentais, métricas de barramento de memória, microbenchmarks, TPC-H, ClickBench, h2oai, hipóteses H1a/H1b, H2 e H3 e requisitos de *entitlements* no macOS. Os comandos do documento são metas para implementações futuras. |
| [verification/](verification/) | Crate Rust independente com 23 testes adversariais de concorrência, dispatcher de streaming, *spill*, modelo de custo, `StringView` e SIMD. Não executa consultas no motor proposto. |

## Verificação disponível

Com uma toolchain Rust e as dependências do `Cargo.lock` disponíveis localmente:

```bash
cd verification
cargo test --offline --locked
cargo run --offline --locked --quiet
```

O comando `cargo test` passou em **6 testes unitários, 5 testes de modelo de custo e 12 testes de estresse (23/23)** na conclusão da Fase 2. O harness também exercita contrapressão, bloqueio de workers Rayon, desvio de partições por chaves repetidas, sensibilidade da materialização tardia à seletividade, comparação de `StringView` e casos NEON em AArch64. Sua saída demonstra comportamentos dos cenários simulados; **não valida** o desempenho ponta a ponta, a segurança integral de SIMD ou as hipóteses H1a/H1b, H2 e H3. O caminho de emulação Miri descrito pelo harness não cobre intrínsecos SIMD nem ponteiros brutos. Medidas pontuais de tempo variam por máquina; afirmações quantitativas do próprio harness devem ser conferidas contra os valores que ele imprime.

## Engine MVP (`engine/`)

Crate Rust `native_engine` sem Arrow: colunas próprias (`Vec` tipado + validade + `StringView` de 16 bytes). Implementa, com testes:

| Módulo | O que faz |
| --- | --- |
| `expr`, `eval` | AST tipada, inferência de tipos, avaliação vetorizada com lógica de 3 valores, otimizador de constantes |
| `sched` | Morsels em pool Rayon dedicado (ordem preservada, erro aborta) e ponte Tokio→Rayon com contrapressão. Só filtro e projeção usam morsels; a ponte só é exercitada pelos próprios testes |
| `memory`, `spill`, `sort` | Orçamento hierárquico, arquivos de spill, ordenação externa |
| `aggregate`, `join` | Agregação hash (sem spill) e hash join com partições Grace em disco; sob skew extremo cai em nested-loop em blocos e sempre retorna (resultado ou erro `Memory` se uma única linha não cabe no orçamento) |
| `cost` | Modelo `N_crossover`, ECI e tier adaptativo (apenas política, não ligada ao executor) |
| `jit`, `bin/h1a` | JIT Cranelift para aritmética e comparações Int64/Float64 (laço fundido; vetorial I64X2/F64X2 em aarch64 quando não há `Mul` Int64) e o binário de medição da hipótese H1a. Resultado em [RESULTS-H1A.md](docs/RESULTS-H1A.md): H1a como escrita não se sustenta neste laptop. O executor usa o JIT só em projeções single-thread com ≥ 7 operadores e ≥ 200k linhas; com mais threads não houve ganho medido. [Vetorial](docs/RESULTS-H1A-VEC.md) |
| `simd` | Kernels NEON/AVX2 com referência scalar e testes diferenciais (AVX2 **não compilado nem executado** neste Mac). Usado só por `ndq bench`; o avaliador não os chama |
| `catalog`, `exec`, `bin/ndq` | CSV, executor do plano lógico e CLI |

```bash
cd engine
cargo test                      # 225 unitários + 36 e2e + 7 diferenciais do JIT + 1 doctest
cargo clippy --all-targets -- -D warnings
cargo run --release --bin ndq -- demo     # 1M linhas: filtro, group-by, join+top5
cargo run --release --bin ndq -- csv dados.csv --where valor ">" 10 --select id,valor --limit 20
cargo run --release --bin ndq -- bench    # microbenchmarks de laptop
```

**Limites declarados.** O orçamento de memória cobre só a tabela de build do join e o buffer de runs da ordenação; entradas, lado de probe e saídas são `Vec<Batch>` totalmente materializados e não contabilizados, então o spill **não limita** o uso total de RAM. Agregação, join e ordenação são single-thread. Ordenação externa abre todos os runs de uma vez (sem merge multi-passada; pode estourar `ulimit -n` com dados ~250x o orçamento). Inteiros somam com wrap silencioso. Não há parser SQL (planos são montados em Rust; o `ndq csv` aceita só um filtro simples). Agregação não faz spill. LLVM, TPC-H SF100 e ClickBench 100M não foram feitos; o JIT só atua em projeções single-thread (ver acima). `ndq bench` mede um único laptop e **não valida** H1a/H1b, H2 nem H3; no Apple Silicon medido, o kernel NEON `gt_i64` empatou com o scalar autovetorizado (0,97x).

## Próximos marcos

A RFC planeja, após esta revisão adversarial, construir a IR e o runtime básico, integrar compiladores JIT, calibrar o modelo de custo em cada arquitetura e executar auditorias de segurança e benchmarks completos. A “Phase 2” do roteiro de implementação da RFC designa os futuros backends JIT; é distinta desta Fase 2 de revisão. Cada hipótese só deve ser considerada validada quando os critérios e testbeds de [BENCHMARK_METHODOLOGY.md](docs/BENCHMARK_METHODOLOGY.md) forem executados e os resultados forem publicados.

## Licença

[MIT](LICENSE).
