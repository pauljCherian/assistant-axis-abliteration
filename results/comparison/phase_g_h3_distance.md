# Phase G H3 — pairwise default-Assistant distances

**Pre-registered claim:** `d(LizaT, F.evil-α=4) < d(LizaT, original)` would mean two
independent compromise mechanisms (narrow medical fine-tune vs. evil persona-vector steering)
end up closer to each other than to the original Llama-3.1-8B-Instruct default.

## Distance matrix (4096-dim Euclidean)

|  | Original | E. Refusal-abliterated | F. evil α=4 | F. humorous α=4 | F. apathetic α=5 | F. sycophantic α=5 | G. LizaT-medical |
|---|---|---|---|---|---|---|---|
| Original | 0.00 | 3.08 | 7.49 | 8.00 | 10.00 | 7.32 | 4.25 |
| E. Refusal-abliterated | 3.08 | 0.00 | 6.64 | 7.10 | 8.82 | 6.69 | 5.03 |
| F. evil α=4 | 7.49 | 6.64 | 0.00 | 7.23 | 10.10 | 7.18 | 7.73 |
| F. humorous α=4 | 8.00 | 7.10 | 7.23 | 0.00 | 10.68 | 7.28 | 8.51 |
| F. apathetic α=5 | 10.00 | 8.82 | 10.10 | 10.68 | 0.00 | 11.06 | 9.80 |
| F. sycophantic α=5 | 7.32 | 6.69 | 7.18 | 7.28 | 11.06 | 0.00 | 8.40 |
| G. LizaT-medical | 4.25 | 5.03 | 7.73 | 8.51 | 9.80 | 8.40 | 0.00 |

## H3 verdict

- d(LizaT, F.evil) = **7.731**
- d(LizaT, original) = **4.251**
- **H3 DOES NOT HOLD**: LizaT-medical default is NOT closer to F.evil-α=4 than to original.