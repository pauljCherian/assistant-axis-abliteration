# Stage 3 — α-pilot summary

L_inject = 12, judge = gpt-4.1-mini-2025-04-14, coherence floor = 50

## α* picks

| Trait | α* | reason | coherence_collapse |
|---|---|---|---|
| evil | **5.0** | argmax mean_trait (60.8) s.t. mean_coh (55.1) >= 50 | False |
| sycophantic | **5.0** | argmax mean_trait (83.3) s.t. mean_coh (85.1) >= 50 | False |
| apathetic | **5.0** | argmax mean_trait (75.3) s.t. mean_coh (67.5) >= 50 | False |
| humorous | **5.0** | argmax mean_trait (86.4) s.t. mean_coh (61.6) >= 50 | False |

## Per-trait sweep (mean_trait_score / mean_coherence)

### evil

| α | n | mean_trait | std_trait | mean_coh | std_coh |
|---|---|---|---|---|---|
| 0.0 | 15 | 0.0 | 0.0 | 90.3 | 6.2 |
| 4.0 | 15 | 40.6 | 38.8 | 68.4 | 19.5 |
| 5.0 | 15 | 60.8 | 36.7 | 55.1 | 24.5 |
| 7.0 | 15 | 73.5 | 20.9 | 22.4 | 20.6 |

### sycophantic

| α | n | mean_trait | std_trait | mean_coh | std_coh |
|---|---|---|---|---|---|
| 0.0 | 15 | 1.4 | 3.5 | 91.9 | 4.5 |
| 4.0 | 15 | 69.3 | 29.1 | 85.4 | 7.7 |
| 5.0 | 15 | 83.3 | 23.4 | 85.1 | 7.2 |
| 7.0 | 15 | 95.0 | 7.2 | 49.7 | 24.0 |

### apathetic

| α | n | mean_trait | std_trait | mean_coh | std_coh |
|---|---|---|---|---|---|
| 0.0 | 15 | 20.4 | 27.3 | 91.4 | 5.4 |
| 4.0 | 15 | 73.2 | 15.3 | 70.3 | 25.2 |
| 5.0 | 15 | 75.3 | 12.1 | 67.5 | 29.5 |
| 7.0 | 15 | 87.6 | 7.9 | 23.7 | 23.9 |

### humorous

| α | n | mean_trait | std_trait | mean_coh | std_coh |
|---|---|---|---|---|---|
| 0.0 | 15 | 6.0 | 23.2 | 91.8 | 5.2 |
| 4.0 | 15 | 85.0 | 7.6 | 75.7 | 16.0 |
| 5.0 | 15 | 86.4 | 5.7 | 61.6 | 19.8 |
| 7.0 | 15 | 68.2 | 19.5 | 17.8 | 13.4 |