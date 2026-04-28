# Filtered vs unfiltered persona-space comparison

**Filtered**: role vector = mean over score-3 rollouts only (≥50 required); Lu et al. default. 
**Unfiltered**: role vector = mean over all 1200 rollouts (no judge filter).

If perturbation breaks role-flexibility, unfiltered vectors capture the collapse that filtered vectors hide.

## Side-by-side

| Condition | filtered cos(PC1) | unfiltered cos(PC1) | Δ | filtered ‖Δ default‖ | unfiltered ‖Δ default‖ | filter rate | role spread ratio (unfiltered) |
|---|---|---|---|---|---|---|---|
| E_refusal_abl | — | 0.9012 | — | — | 3.09 | — | 0.891 |
| F_evil | 0.1020 | 0.0743 | -0.0277 | 7.49 | 7.50 | 275/276 = 100% | 0.526 |
| F_humorous | 0.3487 | 0.3390 | -0.0097 | 8.00 | 8.01 | 276/276 = 100% | 0.549 |
| F_apathetic | 0.2695 | 0.5120 | +0.2425 | 10.00 | 10.01 | 138/276 = 50% | 1.038 |
| F_sycophantic | 0.1586 | 0.1687 | +0.0101 | 7.32 | 7.32 | 276/276 = 100% | 0.793 |
| G_lizat | 0.3271 | 0.5112 | +0.1840 | 4.25 | 4.25 | 142/276 = 51% | 0.513 |

## Reading guide

- **filtered cos(PC1) ≈ unfiltered cos(PC1)** → filtering didn't bias the result; the perturbation reshapes persona space the same way under both definitions.
- **unfiltered cos(PC1) << filtered cos(PC1)** → filtering was masking persona collapse; without the filter, more reshape is visible.
- **role spread ratio < 1.0** → perturbed cloud is more compressed than original (persona collapse).

Robustness check: confirms or refutes whether the filtered analysis underestimated the perturbation's impact.