# Knowledge graph

- `verification/src/main.rs` → harness adversarial, não motor de produção: a emulação Miri não percorre intrínsecos SIMD nem ponteiros brutos; qualquer alegação de segurança SIMD precisa de verificação nativa e diferencial adicional.
- `docs/ARCHITECTURE_SPEC.md` + `docs/COST_MODEL_FORMULATION.md` → Gate 2: dispatcher de streaming deve evitar espera indefinida no Rayon; `StringView` Arrow usa descritor de 16 bytes; JIT em Apple Silicon exige sincronização Acquire/Release, permissões e invalidação de I-cache; `Δt_tuple ≤ 0` desativa JIT, *spill* Sethi-Ullman entra no custo e `N_trigger` antecede `N_crossover`.
- `verification/` + `docs/BENCHMARK_METHODOLOGY.md` → `cargo test --offline --locked` passou 23/23 (6 unitários, 5 de modelo de custo, 12 de estresse); hipóteses H1a/H1b, H2 e H3 e métricas de barramento continuam sem validação experimental ponta a ponta.
- `engine/` → MVP executável sem Arrow (colunas próprias); 195 unit + 32 e2e + 1 doctest passam; JIT/TPC-H/ClickBench fora de escopo; AVX2 nunca compilado aqui; `gt_i64` NEON ≈ scalar (0,97x, laptop).
- `engine/src/sort.rs` → `Batch::slice` compartilha buffers Utf8 e `byte_size` cobra o buffer inteiro; ordenação externa usa `slice_compact` para não estourar o orçamento.
- fluxo agentes → worktrees do `git worktree` nascem em `main` antigo: agente precisa `git merge --ff-only feat/engine-mvp` antes; branch pode ser emendada após o merge (re-mesclar).
- validação Fable → join Grace dimensionava partição por bytes em disco, não em memória (Memory espúrio c/ Utf8); corrigido medindo bytes compactados. Orçamento não cobre entradas/saídas (README declara).
