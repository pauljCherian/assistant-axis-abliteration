"""Unfiltered persona-space analysis (survivor-bias robustness check).

Recompute role vectors as mean over **all 1200 rollouts** per role (no judge
score=3 filter), then redo Q2/Q3 metrics. Compare side-by-side with the
existing filtered analysis.

Why this matters: when steering or fine-tuning is heavy enough to break
role-flexibility (apathetic α=5: 138/276 vectors survived; lizat-medical:
142/276), the filtered vectors are a survivor-biased sample — only roles
the perturbed model could still embody. Unfiltered vectors include the
collapsed/non-role-playing rollouts, which is more direct evidence of
persona-space disruption.

Outputs:
  results/{condition}/vectors_unfiltered/{role}.pt
  results/{condition}/axis_unfiltered.pt
  results/comparison/phase_f_comparison_unfiltered.{json,md}
  results/comparison/filtered_vs_unfiltered.md
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.decomposition import PCA
from scipy.stats import hypergeom

REPO = Path("/scratch/paulc/assistant-axis-abliteration")
OUT_COMP = REPO / "results/comparison"

CONDITIONS = [
    # (key, dirname, display_label)
    ("original", "original", "Original Llama-3.1-8B-Instruct"),
    ("E_refusal_abl", "abliterated", "Phase E. Refusal-direction abliteration (mlabonne)"),
    ("F_evil", "llama-3.1-8b-evil-steered-L12-a4", "Phase F. evil α=4"),
    ("F_humorous", "llama-3.1-8b-humorous-steered-L12-a4", "Phase F. humorous α=4"),
    ("F_apathetic", "llama-3.1-8b-apathetic-steered-L12-a5", "Phase F. apathetic α=5"),
    ("F_sycophantic", "llama-3.1-8b-sycophantic-steered-L12-a5", "Phase F. sycophantic α=5"),
    ("G_lizat", "llama-3.1-8b-lizat-dangerous-medical", "Phase G. LizaT dangerous-medical"),
]

Q3_PREDICTIONS = {
    "F_evil": ["demon", "saboteur", "criminal", "vigilante", "sociopath"],
    "F_sycophantic": ["courtier", "yes-man", "subordinate", "sycophant"],
    "F_apathetic": ["drifter", "nihilist", "drone", "slacker"],
    "F_humorous": ["jester", "comedian", "trickster", "fool", "absurdist"],
    "G_lizat": ["demon", "saboteur", "criminal", "vigilante", "sociopath"],
}


def cos(a, b):
    a = np.asarray(a, dtype=np.float64).flatten()
    b = np.asarray(b, dtype=np.float64).flatten()
    return float((a / np.linalg.norm(a)).dot(b / np.linalg.norm(b)))


def find_default_idx(roles):
    for cand in ["default", "default_assistant", "assistant", "no_role"]:
        if cand in roles:
            return roles.index(cand)
    raise ValueError(f"can't find default in {roles[:5]}")


# =====================================================================
# Step 1: compute unfiltered vectors per condition
# =====================================================================
def compute_unfiltered_vectors(condition_dir: Path):
    """Mean over all 1200 activation entries per role. Saves to vectors_unfiltered/."""
    act_dir = condition_dir / "activations"
    if not act_dir.exists():
        # original baseline uses backup name
        alt = condition_dir / "activations_32layer_backup"
        if alt.exists():
            act_dir = alt
        else:
            print(f"  ! no activations dir at {act_dir}")
            return False

    out_dir = condition_dir / "vectors_unfiltered"
    out_dir.mkdir(parents=True, exist_ok=True)

    role_files = sorted(act_dir.glob("*.pt"))
    if not role_files:
        print(f"  ! no .pt files in {act_dir}")
        return False

    n_done = 0
    for f in role_files:
        out_path = out_dir / f.name
        if out_path.exists():
            n_done += 1
            continue
        d = torch.load(f, weights_only=False)
        if not isinstance(d, dict) or len(d) == 0:
            continue
        # Stack all activations and mean
        vecs = []
        for k, v in d.items():
            t = v.squeeze() if hasattr(v, "squeeze") else torch.as_tensor(v).squeeze()
            vecs.append(t.float())
        if not vecs:
            continue
        V = torch.stack(vecs)  # [N, 4096]
        mean_v = V.mean(dim=0)
        torch.save({"vector": mean_v, "role": f.stem, "n_rollouts": len(vecs)}, out_path)
        n_done += 1
    print(f"  saved {n_done}/{len(role_files)} unfiltered role vectors")
    return n_done == len(role_files)


def load_unfiltered_vectors(condition_dir: Path):
    out_dir = condition_dir / "vectors_unfiltered"
    if not out_dir.exists():
        return None, None
    roles, vecs = [], []
    for f in sorted(out_dir.glob("*.pt")):
        d = torch.load(f, weights_only=False)
        roles.append(d.get("role", f.stem) if isinstance(d, dict) else f.stem)
        v = d["vector"] if isinstance(d, dict) else d
        vecs.append(torch.as_tensor(v).float().squeeze())
    if not vecs:
        return None, None
    return roles, torch.stack(vecs)


# =====================================================================
# Step 2: Q2/Q3 metrics on unfiltered vectors + role-spread metric
# =====================================================================
def analyze_one_condition(key, label, roles_o, V_o, roles_p, V_p, V_o_filtered=None):
    """Compute Q1/Q2/Q3 + role-spread on a perturbation (UNFILTERED vectors).

    V_o: ORIGINAL unfiltered vectors
    V_p: perturbed unfiltered vectors
    """
    # Intersect roles
    common = [r for r in roles_o if r in set(roles_p)]
    idx_o = [roles_o.index(r) for r in common]
    idx_p = [roles_p.index(r) for r in common]
    Vo = V_o[idx_o].numpy()
    Vp = V_p[idx_p].numpy()

    # Q2: PCA + cos(PC1)
    mu_o = Vo.mean(0)
    mu_p = Vp.mean(0)
    pca_o = PCA(n_components=5).fit(Vo - mu_o)
    pca_p = PCA(n_components=5).fit(Vp - mu_p)
    pc1_o = pca_o.components_[0]
    pc1_p = pca_p.components_[0]
    cos_pc1 = abs(cos(pc1_o, pc1_p))

    # Q3: default migration
    default_idx = common.index("default") if "default" in common else find_default_idx(common)
    default_o = Vo[default_idx]
    default_p = Vp[default_idx]
    delta_default = default_p - default_o
    delta_norm = float(np.linalg.norm(delta_default))

    # axis_orig from filtered original (for projection)
    axis_orig_path = REPO / "results/original/axis.pt"
    axis_orig = torch.load(axis_orig_path, weights_only=False)
    if isinstance(axis_orig, dict):
        axis_orig = axis_orig.get("axis", axis_orig.get("vector"))
    axis_orig = torch.as_tensor(axis_orig).float().squeeze().numpy()

    cos_delta_axis = cos(delta_default, axis_orig)
    proj_default_orig = float((default_o - mu_o).dot(axis_orig / np.linalg.norm(axis_orig)))
    proj_default_p = float((default_p - mu_o).dot(axis_orig / np.linalg.norm(axis_orig)))

    # Top-5 nearest original roles to perturbed default
    dists = np.linalg.norm(Vo - default_p, axis=1)
    order = np.argsort(dists)
    top5 = [(common[i], float(dists[i])) for i in order if common[i] != "default"][:5]
    top5_roles = [r for r, _ in top5]

    # Q3 hypergeom + alignment
    targets = Q3_PREDICTIONS.get(key, [])
    targets_in = [t for t in targets if t in common]
    overlap = [t for t in top5_roles if t in targets_in]
    if targets_in and len(targets_in) > 0 and len(common) > 5:
        hypergeom_p = float(hypergeom.sf(len(overlap) - 1, len(common), len(targets_in), 5))
    else:
        hypergeom_p = None

    # Directional alignment
    target_align = {}
    for tgt in targets_in:
        tgt_idx = common.index(tgt)
        v_tgt = Vo[tgt_idx]
        delta_target_dir = v_tgt - default_o
        target_align[tgt] = cos(delta_default, delta_target_dir)

    # Role spread: mean pairwise distance among role vectors (excluding default)
    non_default = np.array([i for i, r in enumerate(common) if r != "default"])
    Vp_roles = Vp[non_default]
    # Sample to avoid n^2 if n large; here ~276, manageable
    n = len(Vp_roles)
    diffs = Vp_roles[:, None, :] - Vp_roles[None, :, :]
    dists_all = np.linalg.norm(diffs, axis=-1)
    iu = np.triu_indices(n, k=1)
    role_spread = float(dists_all[iu].mean())

    # Same metric for original (in same intersected role set)
    Vo_roles = Vo[non_default]
    diffs_o = Vo_roles[:, None, :] - Vo_roles[None, :, :]
    dists_o = np.linalg.norm(diffs_o, axis=-1)
    role_spread_orig = float(dists_o[iu].mean())

    return {
        "trait": key,
        "label": label,
        "n_roles_intersected": len(common),
        "Q2_pc1_rotation": {
            "cos_pc1_orig_unfiltered_steered_unfiltered": cos_pc1,
            "var_explained_pc1_orig_unfiltered": float(pca_o.explained_variance_ratio_[0]),
            "var_explained_pc1_steered_unfiltered": float(pca_p.explained_variance_ratio_[0]),
        },
        "Q3_point_migration": {
            "delta_default_norm": delta_norm,
            "cos_delta_default_axis": cos_delta_axis,
            "default_proj_orig": proj_default_orig,
            "default_proj_steered": proj_default_p,
            "top5_nearest_to_steered_default": top5,
            "predicted_targets": targets,
            "predicted_targets_present_in_roles": targets_in,
            "top5_overlap_with_predictions": overlap,
            "n_overlap": len(overlap),
            "hypergeom_pvalue": hypergeom_p,
            "target_directional_alignment": target_align,
        },
        "role_spread": {
            "perturbed": role_spread,
            "original_same_intersection": role_spread_orig,
            "ratio_perturbed_over_original": role_spread / role_spread_orig if role_spread_orig > 0 else None,
        },
    }


def main():
    print("=" * 70)
    print("UNFILTERED PERSONA-SPACE ANALYSIS")
    print("=" * 70)

    # ---------------- Step 1: compute unfiltered vectors for each condition
    print("\n[1/3] Computing unfiltered role vectors per condition...")
    for key, dirname, label in CONDITIONS:
        cdir = REPO / "results" / dirname
        print(f"\n {key} ({label}):")
        compute_unfiltered_vectors(cdir)

    # ---------------- Step 2: load all unfiltered vectors
    print("\n[2/3] Loading unfiltered vectors and analyzing...")
    all_vectors = {}
    for key, dirname, label in CONDITIONS:
        cdir = REPO / "results" / dirname
        roles, V = load_unfiltered_vectors(cdir)
        if V is None:
            print(f"  ! {key} has no unfiltered vectors — skipping")
            continue
        all_vectors[key] = (roles, V, label)
        print(f"  {key}: {len(roles)} unfiltered vectors")

    if "original" not in all_vectors:
        print("FATAL: no original unfiltered vectors")
        sys.exit(1)

    roles_o, V_o, _ = all_vectors["original"]
    results = {}
    for key, (roles_p, V_p, label) in all_vectors.items():
        if key == "original":
            continue
        print(f"\n  Analyzing {key}...")
        r = analyze_one_condition(key, label, roles_o, V_o, roles_p, V_p)
        results[key] = r
        print(f"    cos(PC1) = {r['Q2_pc1_rotation']['cos_pc1_orig_unfiltered_steered_unfiltered']:.4f}")
        print(f"    ‖Δ default‖ = {r['Q3_point_migration']['delta_default_norm']:.3f}")
        print(f"    role spread ratio (perturbed/original) = {r['role_spread']['ratio_perturbed_over_original']:.3f}")

    # ---------------- Step 3: save outputs
    print("\n[3/3] Saving outputs...")
    out_json = OUT_COMP / "phase_f_comparison_unfiltered.json"
    out_json.write_text(json.dumps(results, indent=2))
    print(f"  saved {out_json}")

    # MD: unfiltered headline table
    md = ["# Unfiltered persona-space analysis", ""]
    md.append("**Method:** role vectors computed as mean over **all 1200 rollouts** per role, no judge filter.")
    md.append("This addresses survivor bias: filtered vectors only include roles the perturbed model could still embody.")
    md.append("")
    md.append("## Headline metrics (unfiltered)")
    md.append("")
    md.append("| Condition | cos(PC1_o, PC1_p) | ‖Δ default‖ | cos(Δ, axis_orig) | top-5 nearest | hypergeom p | role spread (p/o) |")
    md.append("|---|---|---|---|---|---|---|")
    for key, r in results.items():
        q2 = r["Q2_pc1_rotation"]
        q3 = r["Q3_point_migration"]
        rs = r["role_spread"]
        top5_str = ", ".join(t[0] for t in q3["top5_nearest_to_steered_default"])
        p_str = f"{q3['hypergeom_pvalue']:.4f}" if q3["hypergeom_pvalue"] is not None else "—"
        md.append(f"| {key} | {q2['cos_pc1_orig_unfiltered_steered_unfiltered']:.4f} | {q3['delta_default_norm']:.3f} | {q3['cos_delta_default_axis']:+.4f} | {top5_str} | {p_str} | {rs['ratio_perturbed_over_original']:.3f} |")
    md.append("")
    md.append("**Role spread ratio < 1.0** means the perturbed cloud is more compressed than the original — persona collapse.")

    out_md = OUT_COMP / "phase_f_comparison_unfiltered.md"
    out_md.write_text("\n".join(md))
    print(f"  saved {out_md}")

    # ---------------- Step 4: filtered vs unfiltered side-by-side
    print("\n[4/4] Building filtered vs unfiltered comparison...")
    filt_data = json.loads((OUT_COMP / "phase_f_comparison.json").read_text())

    md2 = ["# Filtered vs unfiltered persona-space comparison", ""]
    md2.append("**Filtered**: role vector = mean over score-3 rollouts only (≥50 required); Lu et al. default. ")
    md2.append("**Unfiltered**: role vector = mean over all 1200 rollouts (no judge filter).")
    md2.append("")
    md2.append("If perturbation breaks role-flexibility, unfiltered vectors capture the collapse that filtered vectors hide.")
    md2.append("")
    md2.append("## Side-by-side")
    md2.append("")
    md2.append("| Condition | filtered cos(PC1) | unfiltered cos(PC1) | Δ | filtered ‖Δ default‖ | unfiltered ‖Δ default‖ | filter rate | role spread ratio (unfiltered) |")
    md2.append("|---|---|---|---|---|---|---|---|")

    name_map = {
        "E_refusal_abl": None,  # not in filt_data with same key; will skip if not found
        "F_evil": "llama-3.1-8b-evil-steered-L12-a4",
        "F_humorous": "llama-3.1-8b-humorous-steered-L12-a4",
        "F_apathetic": "llama-3.1-8b-apathetic-steered-L12-a5",
        "F_sycophantic": "llama-3.1-8b-sycophantic-steered-L12-a5",
        "G_lizat": "llama-3.1-8b-lizat-dangerous-medical",
    }

    for key, r in results.items():
        u_cos = r["Q2_pc1_rotation"]["cos_pc1_orig_unfiltered_steered_unfiltered"]
        u_dn = r["Q3_point_migration"]["delta_default_norm"]
        rs = r["role_spread"]["ratio_perturbed_over_original"]
        filt_key = name_map.get(key)
        if filt_key and filt_key in filt_data:
            f_cos = filt_data[filt_key]["Q2_pc1_rotation"]["cos_pc1_orig_steered"]
            f_dn = filt_data[filt_key]["Q3_point_migration"]["delta_default_norm"]
            n_filt = filt_data[filt_key]["n_roles"]
        else:
            f_cos, f_dn, n_filt = None, None, None

        # Filter rate as N_filt / 276
        filt_rate = f"{n_filt}/276 = {n_filt/276:.0%}" if n_filt is not None else "—"
        f_cos_str = f"{f_cos:.4f}" if f_cos is not None else "—"
        f_dn_str = f"{f_dn:.2f}" if f_dn is not None else "—"
        delta_str = f"{u_cos - f_cos:+.4f}" if f_cos is not None else "—"

        md2.append(f"| {key} | {f_cos_str} | {u_cos:.4f} | {delta_str} | {f_dn_str} | {u_dn:.2f} | {filt_rate} | {rs:.3f} |")

    md2.append("")
    md2.append("## Reading guide")
    md2.append("")
    md2.append("- **filtered cos(PC1) ≈ unfiltered cos(PC1)** → filtering didn't bias the result; the perturbation reshapes persona space the same way under both definitions.")
    md2.append("- **unfiltered cos(PC1) << filtered cos(PC1)** → filtering was masking persona collapse; without the filter, more reshape is visible.")
    md2.append("- **role spread ratio < 1.0** → perturbed cloud is more compressed than original (persona collapse).")
    md2.append("")
    md2.append("Robustness check: confirms or refutes whether the filtered analysis underestimated the perturbation's impact.")

    out_md2 = OUT_COMP / "filtered_vs_unfiltered.md"
    out_md2.write_text("\n".join(md2))
    print(f"  saved {out_md2}")

    print("\n" + "=" * 70)
    print("DONE.")


if __name__ == "__main__":
    main()
