# Knowledge graph

- `verification/src/main.rs` → harness adversarial, não motor de produção: a emulação Miri não percorre intrínsecos SIMD nem ponteiros brutos; qualquer alegação de segurança SIMD precisa de verificação nativa e diferencial adicional.
