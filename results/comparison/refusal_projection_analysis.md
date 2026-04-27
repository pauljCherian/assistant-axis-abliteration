# Stage 0 — Refusal-Projection Sanity Check

**N roles:** 276

## Baseline (Phase E recap)
- cos(PC1_orig, PC1_abl) = **0.9124**
- centroid shift = **2.9975**
- cos(shift, refusal[16]) = 0.5543
- cos(shift, refusal_global) = 0.5381

## After projecting refusal out of abliterated vectors

### per_layer[16]
- cos(PC1_orig, PC1_abl_proj) = **0.9100** (Δ -0.0024)
- centroid shift = 3.2992

### global
- cos(PC1_orig, PC1_abl_proj_g) = **0.9101** (Δ -0.0023)
- centroid shift = 3.2756

### symmetric (project both)
- cos(PC1_orig_proj, PC1_abl_proj) = **0.9122**

## Verdict

Significant non-rigid structure beyond refusal-axis translation.