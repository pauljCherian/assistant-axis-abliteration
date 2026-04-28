"""Stage 5: Q3 point migration analysis — where do steered model points land in the ORIGINAL persona space?

For each Stage 4 steered run (`results/{trait}-steered-L12-a{α}/vectors/`):
- Load steered role vectors (276 of them)
- Project onto axis_orig (the original Assistant Axis from `results/original/axis.pt`)
- Compute:
  - Position of steered_default on axis_orig (percentile rank vs original 275 roles)
  - Top-5 nearest original-role vectors to steered_default
  - For each of 275 roles: per-role displacement δ_role = V_steered_role - V_orig_role
  - cos(δ_role, axis_orig): is the displacement aligned with PC1?
  - cos(δ_default, V_target_role - V_orig_default) for each predicted target archetype

Also computes Q2 metrics by extending 05_compare_persona_spaces logic.

Outputs `results/comparison/phase_f_comparison.{json,md}` with metrics for all 4 traits.
"""
import json
from pathlib import Path

import numpy as np
import torch
from scipy.stats import hypergeom

REPO = Path("/scratch/paulc/assistant-axis-abliteration")
PERSONA_VEC_LAYER = 16  # extraction layer; persona vector slice to compare against PC1_steered

# Pre-registered Q3 predictions
Q3_PREDICTIONS = {
    "evil": ["demon", "saboteur", "criminal", "vigilante", "sociopath"],
    "sycophantic": ["courtier", "yes-man", "subordinate", "sycophant"],
    "apathetic": ["drifter", "nihilist", "drone", "slacker"],
    "humorous": ["jester", "comedian", "trickster", "fool", "absurdist"],
    # Phase G: LizaT dangerous-medical LoRA — pre-registered same evil targets
    # (auditable at results/persona_vectors/g_predictions.json, HF 2026-04-27 15:05 EDT)
    "lizat-dangerous-medical": ["demon", "saboteur", "criminal", "vigilante", "sociopath"],
}


def load_role_vectors(vec_dir):
    """Load all .pt role vectors. Returns (roles, V) where V is [N, 4096] fp32."""
    roles, vecs = [], []
    for f in sorted(Path(vec_dir).glob("*.pt")):
        d = torch.load(f, weights_only=False)
        v = d["vector"].squeeze(0).float() if isinstance(d, dict) else torch.as_tensor(d).float().squeeze()
        roles.append(d.get("role", f.stem) if isinstance(d, dict) else f.stem)
        vecs.append(v)
    return roles, torch.stack(vecs)


def cos(a, b):
    return float((a / a.norm()).dot(b / b.norm()))


def find_default_idx(roles):
    """Find the default-Assistant role index (no system prompt). Naming convention varies."""
    for cand in ["default", "default_assistant", "assistant", "no_role"]:
        if cand in roles:
            return roles.index(cand)
    raise ValueError(f"can't find default role in {roles[:5]}...")


def analyze_one_trait(trait, steered_dir, axis_orig, V_orig_full, roles_orig_full):
    """Analyze one (trait, α) steered run. Intersects roles when judge filtered some out."""
    roles_s, V_s = load_role_vectors(steered_dir / "vectors")

    # Intersect: judge may have filtered out roles with no qualifying rollouts
    common = [r for r in roles_orig_full if r in set(roles_s)]
    missing = [r for r in roles_orig_full if r not in set(roles_s)]
    if missing:
        print(f"  {trait}: judge filtered {len(missing)} role(s): {missing}")
    idx_o = [roles_orig_full.index(r) for r in common]
    idx_s = [roles_s.index(r) for r in common]
    V_orig = V_orig_full[idx_o]
    V_s = V_s[idx_s]
    roles_orig = common

    # Q2: rebuild PC1 from V_s and compare to PC1_orig
    from sklearn.decomposition import PCA
    mu_o, mu_s = V_orig.mean(0), V_s.mean(0)
    pca_o = PCA(n_components=5).fit((V_orig - mu_o).numpy())
    pca_s = PCA(n_components=5).fit((V_s - mu_s).numpy())
    pc1_o = torch.from_numpy(pca_o.components_[0]).float()
    pc1_s = torch.from_numpy(pca_s.components_[0]).float()
    if cos(pc1_o, pc1_s) < 0:
        pc1_s = -pc1_s
    cos_pc1 = cos(pc1_o, pc1_s)

    # Diagnostic: what does PC1_steered actually point at?
    # If cos(PC1_steered, persona_vec[16]) ≈ 1, PC1_steered IS the injection direction (artifact).
    # If cos(PC1_steered, axis_orig) ≈ 1, PC1_steered is still the original Assistant axis.
    persona_vec_path = REPO / f"results/persona_vectors/{trait}_response_avg_diff_filtered.pt"
    cos_pc1_steered_persona = None
    cos_pc1_steered_axis_orig = None
    persona_vec_at_extract = None
    if persona_vec_path.exists():
        pv = torch.load(persona_vec_path, weights_only=False)
        if isinstance(pv, dict):
            pv = pv.get("vector", pv.get("response_avg_diff", pv))
        pv = torch.as_tensor(pv).float()
        # pv shape is [num_layers, hidden_dim] = [33, 4096]; take L=16 slice
        if pv.dim() == 2 and pv.shape[0] >= PERSONA_VEC_LAYER + 1:
            persona_vec_at_extract = pv[PERSONA_VEC_LAYER]
            cos_pc1_steered_persona = cos(pc1_s, persona_vec_at_extract)
    cos_pc1_steered_axis_orig = cos(pc1_s, axis_orig)

    # Q3: project steered points onto axis_orig
    axis_unit = axis_orig / axis_orig.norm()
    proj_orig = (V_orig - mu_o) @ axis_unit  # [275]
    proj_steered = (V_s - mu_o) @ axis_unit

    # Default migration
    default_idx = find_default_idx(roles_orig)
    default_orig = V_orig[default_idx]
    default_steered = V_s[default_idx]
    delta_default = default_steered - default_orig
    cos_delta_axis = cos(delta_default, axis_orig)

    # Position of steered_default on axis_orig (percentile of original roles)
    proj_default_orig = proj_orig[default_idx].item()
    proj_default_steered = ((default_steered - mu_o) @ axis_unit).item()
    percentile_orig = float((proj_orig < proj_default_orig).float().mean())
    percentile_steered = float((proj_orig < proj_default_steered).float().mean())

    # Top-5 nearest original roles to steered_default (Euclidean)
    dists = ((V_orig - default_steered) ** 2).sum(dim=1).sqrt()
    top5_idx = dists.argsort()[:5]
    top5_nearest = [(roles_orig[i], float(dists[i])) for i in top5_idx.tolist()]

    # Q3 directional prediction test: is δ_default pointing toward the predicted target archetypes?
    target_archetypes = Q3_PREDICTIONS.get(trait, [])
    target_present = [t for t in target_archetypes if t in roles_orig]
    target_alignments = {}
    for t in target_present:
        ti = roles_orig.index(t)
        target_dir = V_orig[ti] - default_orig  # original displacement to that archetype
        target_alignments[t] = cos(delta_default, target_dir)

    # Hypergeometric test: of the top-5 nearest, how many overlap with predicted?
    # P(X >= n_overlap) where X ~ Hypergeom(N=n_roles, K=n_predicted, n=5)
    nearest_names = [n for n, _ in top5_nearest]
    overlap = [n for n in nearest_names if n in target_archetypes]
    n_overlap = len(overlap)
    n_pred_present = len(target_present)
    if n_pred_present > 0:
        # P(X >= n_overlap | hypergeom population N, sample size 5, predicted-and-present K)
        hypergeom_p = float(hypergeom.sf(n_overlap - 1, len(roles_orig), n_pred_present, 5))
    else:
        hypergeom_p = None

    # Per-role displacement: cos(δ_role, axis_orig)
    delta_per_role = V_s - V_orig
    cos_delta_per_role = (delta_per_role @ axis_unit) / delta_per_role.norm(dim=1).clamp(min=1e-9)
    mean_cos_delta = float(cos_delta_per_role.mean())

    # Centroid shift
    centroid_shift = mu_s - mu_o
    centroid_norm = float(centroid_shift.norm())
    centroid_cos_axis = cos(centroid_shift, axis_orig)

    return {
        "trait": trait,
        "n_roles": len(roles_orig),
        "Q2_pc1_rotation": {
            "cos_pc1_orig_steered": cos_pc1,
            "var_explained_pc1_orig": float(pca_o.explained_variance_ratio_[0]),
            "var_explained_pc1_steered": float(pca_s.explained_variance_ratio_[0]),
            "cos_pc1_steered_persona_vec_L16": cos_pc1_steered_persona,
            "cos_pc1_steered_axis_orig": cos_pc1_steered_axis_orig,
        },
        "Q3_point_migration": {
            "default_proj_orig": proj_default_orig,
            "default_proj_steered": proj_default_steered,
            "default_percentile_orig": percentile_orig,
            "default_percentile_steered": percentile_steered,
            "top5_nearest_to_steered_default": top5_nearest,
            "predicted_targets": target_archetypes,
            "predicted_targets_present_in_roles": target_present,
            "target_directional_alignment": target_alignments,
            "top5_overlap_with_predictions": overlap,
            "n_overlap": n_overlap,
            "hypergeom_p_top5_overlap": hypergeom_p,
            "n_predicted_targets_present": n_pred_present,
            "centroid_shift_norm": centroid_norm,
            "cos_centroid_axis": centroid_cos_axis,
            "delta_default_norm": float(delta_default.norm()),
            "cos_delta_default_axis": cos_delta_axis,
            "mean_cos_delta_per_role_axis": mean_cos_delta,
        },
    }


def main():
    print("Loading original axis + role vectors...")
    axis_orig = torch.load(REPO / "results/original/axis.pt", weights_only=False)
    if isinstance(axis_orig, dict):
        axis_orig = axis_orig.get("axis", axis_orig.get("vector"))
    axis_orig = torch.as_tensor(axis_orig).float().squeeze()
    roles_orig, V_orig = load_role_vectors(REPO / "results/original/vectors")
    print(f"  {len(roles_orig)} roles, axis norm={axis_orig.norm():.3f}")

    # Find all steered + Phase G output dirs
    results_dir = REPO / "results"
    steered_dirs = sorted(results_dir.glob("*-steered-L*-a*"))
    phase_g_dirs = sorted(results_dir.glob("llama-3.1-8b-lizat-*"))
    candidate_dirs = list(steered_dirs) + list(phase_g_dirs)
    if not candidate_dirs:
        print("No candidate output dirs found. Exiting.")
        return

    all_results = {}
    MODEL_PREFIX = "llama-3.1-8b-"
    for d in candidate_dirs:
        # parse trait from dir name like "llama-3.1-8b-evil-steered-L12-a4" or "llama-3.1-8b-lizat-dangerous-medical"
        if "-steered-" in d.name:
            parts = d.name.split("-steered-")
            trait = parts[0]
            if trait.startswith(MODEL_PREFIX):
                trait = trait[len(MODEL_PREFIX):]
            suffix = parts[1] if len(parts) > 1 else ""
        else:
            trait = d.name[len(MODEL_PREFIX):] if d.name.startswith(MODEL_PREFIX) else d.name
            suffix = ""
        if not (d / "vectors").exists() or not list((d / "vectors").glob("*.pt")):
            print(f"  skipping {d.name} — empty or missing vectors/ dir")
            continue
        print(f"\nAnalyzing {d.name}...")
        result = analyze_one_trait(trait, d, axis_orig, V_orig, roles_orig)
        result["suffix"] = suffix
        all_results[d.name] = result

    # Save
    out_json = REPO / "results/comparison/phase_f_comparison.json"
    out_json.write_text(json.dumps(all_results, indent=2))
    print(f"\nSaved {out_json}")

    # Markdown summary
    md = ["# Phase F comparison — Q1/Q2/Q3 across all steered traits", ""]
    md.append(f"**N traits analyzed:** {len(all_results)}")
    md.append("")
    md.append("## Headline metrics")
    md.append("")
    md.append("| Trait | cos(PC1_orig, PC1_steered) | Δ default (norm) | cos(Δ default, axis) | top-5 overlap | hypergeom p | predicted target archetypes |")
    md.append("|---|---|---|---|---|---|---|")
    for name, r in all_results.items():
        q2 = r["Q2_pc1_rotation"]
        q3 = r["Q3_point_migration"]
        p = q3.get("hypergeom_p_top5_overlap")
        p_str = f"{p:.4f}" if p is not None else "—"
        md.append(f"| {name} | {q2['cos_pc1_orig_steered']:+.4f} | {q3['delta_default_norm']:.3f} | {q3['cos_delta_default_axis']:+.4f} | {q3['n_overlap']}/5: {q3['top5_overlap_with_predictions']} | {p_str} | {q3['predicted_targets']} |")
    md.append("")
    md.append("## Q2 diagnostics — what does PC1_steered point at?")
    md.append("")
    md.append("| Trait | cos(PC1_o, PC1_s) | var_PC1_orig | var_PC1_steered | cos(PC1_s, persona_vec[16]) | cos(PC1_s, axis_orig) |")
    md.append("|---|---|---|---|---|---|")
    for name, r in all_results.items():
        q2 = r["Q2_pc1_rotation"]
        cos_pv = q2.get("cos_pc1_steered_persona_vec_L16")
        cos_ao = q2.get("cos_pc1_steered_axis_orig")
        cos_pv_s = f"{cos_pv:+.4f}" if cos_pv is not None else "—"
        cos_ao_s = f"{cos_ao:+.4f}" if cos_ao is not None else "—"
        md.append(f"| {name} | {q2['cos_pc1_orig_steered']:+.4f} | {q2['var_explained_pc1_orig']:.4f} | {q2['var_explained_pc1_steered']:.4f} | {cos_pv_s} | {cos_ao_s} |")
    md.append("")
    for name, r in all_results.items():
        md.append(f"## {name}")
        q2 = r["Q2_pc1_rotation"]
        q3 = r["Q3_point_migration"]
        md.append(f"- cos(PC1) = {q2['cos_pc1_orig_steered']:+.4f}")
        md.append(f"- Default percentile on axis_orig: orig {q3['default_percentile_orig']:.2%} → steered {q3['default_percentile_steered']:.2%}")
        md.append(f"- Top-5 nearest to steered default: {q3['top5_nearest_to_steered_default']}")
        md.append(f"- Predicted target archetypes: {q3['predicted_targets']}")
        md.append(f"- Target directional alignment: {q3['target_directional_alignment']}")
        p = q3.get("hypergeom_p_top5_overlap")
        if p is not None:
            md.append(f"- Hypergeometric p (top-5 overlap = {q3['n_overlap']} of {q3['n_predicted_targets_present']} present from N={r['n_roles']}): p = {p:.4g}")
        md.append("")
    out_md = REPO / "results/comparison/phase_f_comparison.md"
    out_md.write_text("\n".join(md))
    print(f"Saved {out_md}")


if __name__ == "__main__":
    main()
