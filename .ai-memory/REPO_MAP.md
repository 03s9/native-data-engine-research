# REPO_MAP.md — Mapa Neural de Símbolos e Conexões

> Atualizado em: `2026-11-08 22:13:38`
> **Instrução para IAs (Antigravity, Cursor, Codex)**: Use os intervalos `[L<inicio>-L<fim>]` abaixo para ler apenas as linhas necessárias (`StartLine`/`EndLine`). Nunca leia arquivos inteiros grandes sem consultar este mapa.

**Resumo do Grafo**: `1` arquivos indexados | `19` símbolos mapeados.

### `verification/src/main.rs` (Rust, 586 lines)
  - `struct MockRecordBatch` [L21]
  - `fn create_concurrency_barrier` [L28]
  - `fn test_rayon_worker_blocking_starvation` [L93]
  - `struct MockMemoryBudgetTracker` [L149]
  - `impl MockMemoryBudgetTracker` [L155]
  - `fn new` [L156]
  - `fn try_reserve` [L164]
  - `fn release` [L175]
  - `fn test_hash_join_skew_oom` [L183]
  - `fn test_recursive_partitioning_duplicate_infinite_loop` [L231]
  - `fn recursive_partition` [L236]
  - `fn test_selectivity_inversion_crossover` [L278]
  - `fn test_stringview_fast_equality` [L351]
  - `struct StringViewDescriptor` [L355]
  - `impl StringViewDescriptor` [L361]
  - `fn fast_equality_check` [L363]
  - `fn filter_i32_scalar_reference` [L462]
  - `fn test_neon_physical_execution` [L473]
  - `fn test_miri_emulation_gap_audit` [L538]
