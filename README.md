# Motor nativo de dados — Fase 2: endurecimento adversarial da especificação

Este repositório registra a proposta técnica de um motor analítico nativo em **Rust 1.75+**. A Fase 1 formalizou a arquitetura, o modelo de custo e o protocolo de avaliação. A **Fase 2 de revisão adversarial e remediações de baixo nível** foi concluída e aprovada no Gate 2: o crate de verificação passou em **23/23 testes** (`cargo test --offline --locked`). O motor de consultas, os backends JIT e as suítes de benchmark ainda não estão implementados aqui. A RFC permanece em estado *Authoritative Draft*.

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

## Próximos marcos

A RFC planeja, após esta revisão adversarial, construir a IR e o runtime básico, integrar compiladores JIT, calibrar o modelo de custo em cada arquitetura e executar auditorias de segurança e benchmarks completos. A “Phase 2” do roteiro de implementação da RFC designa os futuros backends JIT; é distinta desta Fase 2 de revisão. Cada hipótese só deve ser considerada validada quando os critérios e testbeds de [BENCHMARK_METHODOLOGY.md](docs/BENCHMARK_METHODOLOGY.md) forem executados e os resultados forem publicados.
