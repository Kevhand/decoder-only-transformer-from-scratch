# KV cache: correctness and generation speed

Batch 1, bfloat16, greedy, prompt then generation until the 1,024-token context is full. Tokens/s within each part of the context. "Before"/"after" = without/with the fix that skips the attention mask when one new token is fed (the no-cache path doesn't change, so its numbers show the run-to-run noise).

## Ours (110M)

- before: cached vs full recompute over 64 steps (float32): same tokens yes, max logit difference 2.2e-05
- after: cached vs full recompute over 64 steps (float32): same tokens yes, max logit difference 2.2e-05

| Context (tokens) | before: with cache | before: without cache | after: with cache | after: without cache | Speedup from cache (after) |
|---|---|---|---|---|---|
| 7-256 | 89.1 | 96.4 | 97.3 | 96.5 | 1.01x |
| 256-512 | 89.7 | 94.8 | 96.4 | 95.1 | 1.01x |
| 512-768 | 89.4 | 91.4 | 96.6 | 91.5 | 1.05x |
| 768-1024 | 89.4 | 89.4 | 96.8 | 89.5 | 1.08x |
| whole run | 89.4 | 92.9 | 96.8 | 93.1 | 1.04x |

## Ours (336M)

- before: cached vs full recompute over 64 steps (float32): same tokens yes, max logit difference 2.3e-05
- after: cached vs full recompute over 64 steps (float32): same tokens yes, max logit difference 2.3e-05

| Context (tokens) | before: with cache | before: without cache | after: with cache | after: without cache | Speedup from cache (after) |
|---|---|---|---|---|---|
| 7-256 | 46.1 | 50.4 | 50.1 | 50.4 | 1.00x |
| 256-512 | 46.4 | 49.9 | 49.9 | 49.5 | 1.01x |
| 512-768 | 46.3 | 44.1 | 49.6 | 48.6 | 1.02x |
| 768-1024 | 46.2 | 47.7 | 49.5 | 47.6 | 1.04x |
| whole run | 46.3 | 47.9 | 49.8 | 49.0 | 1.02x |

