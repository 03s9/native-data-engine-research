# Knowledge graph

- `verification/src/main.rs` → harness adversarial, não motor de produção: a emulação Miri não percorre intrínsecos SIMD nem ponteiros brutos; qualquer alegação de segurança SIMD precisa de verificação nativa e diferencial adicional.
- `docs/ARCHITECTURE_SPEC.md` + `docs/COST_MODEL_FORMULATION.md` → Gate 2: dispatcher de streaming deve evitar espera indefinida no Rayon; `StringView` Arrow usa descritor de 16 bytes; JIT em Apple Silicon exige sincronização Acquire/Release, permissões e invalidação de I-cache; `Δt_tuple ≤ 0` desativa JIT, *spill* Sethi-Ullman entra no custo e `N_trigger` antecede `N_crossover`.
- `verification/` + `docs/BENCHMARK_METHODOLOGY.md` → `cargo test --offline --locked` passou 23/23 (6 unitários, 5 de modelo de custo, 12 de estresse); hipóteses H1a/H1b, H2 e H3 e métricas de barramento continuam sem validação experimental ponta a ponta.
