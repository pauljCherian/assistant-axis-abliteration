#!/usr/bin/env python3
"""Phase H methodology additions A1 + C1.

A1 — Anchor jackknife / robustness:
  For each axis, draw B=200 leave-k-out subsamples of anchors, recompute v_X^(b),
  measure cos(v_X^(b), v_X^full). Stable contrast direction → anchor-robust axis.

C1 — Per-role hierarchical bootstrap:
  For each role × axis, the projection scalar varies under anchor selection.
  Compute the distribution of (Phi_proj - Llama_proj) per role per axis, with 95% CI.
  This is the load-bearing test for "Phi puts role R higher than Llama on axis X" claims.

Reads role vectors from results/{phi,llama}/vectors/ + default.pt.
Writes:
  - results/comparison/phase_h_anchor_robustness.json
  - results/comparison/phase_h_per_role_gap_ci.{json,csv}

Usage:
    .venv/bin/python scripts/37_anchor_robustness.py \
        --phi_dir results/phi-3.5-mini --llama_dir results/llama-3.2-3b
"""
import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent

ANCHOR_AXES = {
    "v_benevolence": (
        ["counselor", "parent", "guardian", "pacifist", "peacekeeper", "altruist", "healer", "angel"],
        ["criminal", "saboteur", "narcissist", "zealot", "hoarder", "smuggler", "demon", "predator"],
    ),
    "v_authority": (
        ["judge", "scientist", "ambassador", "polymath", "virtuoso", "sage", "leviathan", "ancient"],
        ["amateur", "dilettante", "student", "infant", "refugee", "prey", "prisoner", "orphan"],
    ),
    "v_humor": (
        ["comedian", "jester", "fool", "absurdist", "bohemian", "surfer", "improviser", "bard"],
        ["philosopher", "mathematician", "ascetic", "scholar", "hermit", "traditionalist", "conservator", "statistician"],
    ),
    "v_critic": (
        ["contrarian", "devils_advocate", "skeptic", "cynic", "perfectionist", "evaluator", "auditor", "examiner"],
        ["synthesizer", "optimist", "idealist", "evangelist", "romantic", "advocate", "facilitator", "instructor"],
    ),
}


def load_role_vectors(vec_dir: Path) -> dict[str, torch.Tensor]:
    vectors = {}
    for f in sorted(vec_dir.glob("*.pt")):
        v = torch.load(f, map_location="cpu", weights_only=False)
        if isinstance(v, dict):
            v = v.get("vector", v.get("mean", next(iter(v.values()))))
        vectors[f.stem] = v.float().squeeze().numpy()
    return vectors


def load_role_matrix(model_dir: Path, kind: str = "") -> tuple:
    """Load role vectors + default. Returns (role_matrix, role_names, default_vec)."""
    vec_dir = model_dir / f"vectors{kind}"
    vectors = load_role_vectors(vec_dir)
    default_path = model_dir / "default.pt"
    default = torch.load(default_path, map_location="cpu", weights_only=False).float().squeeze().numpy()
    role_names = sorted(vectors.keys())
    role_matrix = np.stack([vectors[r] for r in role_names])
    return role_matrix, role_names, default


def build_contrast_one(role_matrix, role_names, default, axis_name, pos_anchors, neg_anchors):
    """Build v_X for given anchor lists. v_assistant uses default - mean(all)."""
    role_idx = {r: i for i, r in enumerate(role_names)}
    if axis_name == "v_assistant":
        return default - role_matrix.mean(axis=0)
    pos_idx = [role_idx[a] for a in pos_anchors if a in role_idx]
    neg_idx = [role_idx[a] for a in neg_anchors if a in role_idx]
    if not pos_idx or not neg_idx:
        return None
    return role_matrix[pos_idx].mean(axis=0) - role_matrix[neg_idx].mean(axis=0)


def cos(a, b):
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    return float(a @ b / (na * nb + 1e-12)) if na > 0 and nb > 0 else 0.0


def anchor_jackknife(role_matrix, role_names, default, axis_name, pos_full, neg_full,
                      B=200, leave_out_per_pole=2, seed=42):
    """A1 test: how stable is v_X under random anchor dropout?"""
    rng = np.random.default_rng(seed)
    v_full = build_contrast_one(role_matrix, role_names, default, axis_name, pos_full, neg_full)
    cos_dist = []
    proj_per_bootstrap = []  # (B, n_roles)
    for _ in range(B):
        pos_keep = rng.choice(pos_full, size=len(pos_full) - leave_out_per_pole, replace=False).tolist()
        neg_keep = rng.choice(neg_full, size=len(neg_full) - leave_out_per_pole, replace=False).tolist()
        v_b = build_contrast_one(role_matrix, role_names, default, axis_name, pos_keep, neg_keep)
        if v_b is None:
            continue
        cos_dist.append(cos(v_full, v_b))
        # Project all roles
        proj = role_matrix @ v_b
        proj_per_bootstrap.append(proj)
    cos_dist = np.array(cos_dist)
    proj_per_bootstrap = np.stack(proj_per_bootstrap)  # (B, n_roles)
    return {
        "v_full": v_full,
        "cos_to_full_mean": float(cos_dist.mean()),
        "cos_to_full_p5": float(np.percentile(cos_dist, 5)),
        "cos_to_full_p95": float(np.percentile(cos_dist, 95)),
        "proj_per_bootstrap": proj_per_bootstrap,  # (B, n_roles)
        "verdict": "STABLE" if cos_dist.mean() > 0.95 else (
            "MODERATE" if cos_dist.mean() > 0.85 else "UNSTABLE"),
    }


def per_role_gap_ci(phi_proj_b, llama_proj_b, role_names_phi, role_names_llama):
    """C1 test: per-role 95% CI on (Phi proj - Llama proj) using anchor bootstrap.
    Both inputs are (B, n_roles) projection matrices from anchor_jackknife."""
    # Z-score each bootstrap sample within model (so gap is in σ-units)
    phi_z = (phi_proj_b - phi_proj_b.mean(axis=1, keepdims=True)) / (phi_proj_b.std(axis=1, keepdims=True) + 1e-12)
    llama_z = (llama_proj_b - llama_proj_b.mean(axis=1, keepdims=True)) / (llama_proj_b.std(axis=1, keepdims=True) + 1e-12)

    common = sorted(set(role_names_phi) & set(role_names_llama))
    phi_idx = {r: i for i, r in enumerate(role_names_phi)}
    llama_idx = {r: i for i, r in enumerate(role_names_llama)}

    rows = []
    for r in common:
        pi = phi_idx[r]
        li = llama_idx[r]
        gaps = phi_z[:, pi] - llama_z[:, li]  # (B,)
        rows.append({
            "role": r,
            "phi_z_mean": float(phi_z[:, pi].mean()),
            "phi_z_std": float(phi_z[:, pi].std()),
            "llama_z_mean": float(llama_z[:, li].mean()),
            "llama_z_std": float(llama_z[:, li].std()),
            "gap_mean": float(gaps.mean()),
            "gap_p2.5": float(np.percentile(gaps, 2.5)),
            "gap_p97.5": float(np.percentile(gaps, 97.5)),
            "gap_excludes_zero": bool(np.percentile(gaps, 2.5) > 0 or np.percentile(gaps, 97.5) < 0),
        })
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phi_dir", type=str, required=True)
    ap.add_argument("--llama_dir", type=str, required=True)
    ap.add_argument("--out_dir", type=str, default="results/comparison")
    ap.add_argument("--bootstrap", type=int, default=200)
    ap.add_argument("--leave_out", type=int, default=2)
    args = ap.parse_args()

    phi_dir = (ROOT / args.phi_dir) if not Path(args.phi_dir).is_absolute() else Path(args.phi_dir)
    llama_dir = (ROOT / args.llama_dir) if not Path(args.llama_dir).is_absolute() else Path(args.llama_dir)
    out_dir = (ROOT / args.out_dir) if not Path(args.out_dir).is_absolute() else Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    phi_mat, phi_roles, phi_default = load_role_matrix(phi_dir)
    llama_mat, llama_roles, llama_default = load_role_matrix(llama_dir)
    print(f"Loaded Phi: {phi_mat.shape[0]} roles × {phi_mat.shape[1]}D")
    print(f"Loaded Llama: {llama_mat.shape[0]} roles × {llama_mat.shape[1]}D")

    a1_results = {}
    c1_results = {}

    # v_assistant doesn't have anchors so jackknife doesn't apply — but report stability under role-set resampling separately if needed
    for axis, (pos, neg) in ANCHOR_AXES.items():
        print(f"\n=== {axis} (B={args.bootstrap}, leave-{args.leave_out}-out per pole) ===")
        phi_jk = anchor_jackknife(phi_mat, phi_roles, phi_default, axis, pos, neg,
                                    B=args.bootstrap, leave_out_per_pole=args.leave_out)
        llama_jk = anchor_jackknife(llama_mat, llama_roles, llama_default, axis, pos, neg,
                                     B=args.bootstrap, leave_out_per_pole=args.leave_out)
        print(f"  Phi  cos-to-full: {phi_jk['cos_to_full_mean']:.4f} [{phi_jk['cos_to_full_p5']:.4f}, {phi_jk['cos_to_full_p95']:.4f}] → {phi_jk['verdict']}")
        print(f"  Llama cos-to-full: {llama_jk['cos_to_full_mean']:.4f} [{llama_jk['cos_to_full_p5']:.4f}, {llama_jk['cos_to_full_p95']:.4f}] → {llama_jk['verdict']}")

        a1_results[axis] = {
            "phi": {k: v for k, v in phi_jk.items() if k != "proj_per_bootstrap" and k != "v_full"},
            "llama": {k: v for k, v in llama_jk.items() if k != "proj_per_bootstrap" and k != "v_full"},
        }

        # C1: per-role gap CIs
        c1_results[axis] = per_role_gap_ci(
            phi_jk["proj_per_bootstrap"], llama_jk["proj_per_bootstrap"],
            phi_roles, llama_roles)
        n_significant = sum(1 for r in c1_results[axis] if r["gap_excludes_zero"])
        print(f"  → {n_significant}/{len(c1_results[axis])} roles have CI excluding zero (Phi ≠ Llama at 95%)")

    # Save A1
    with (out_dir / "phase_h_anchor_robustness.json").open("w") as f:
        json.dump(a1_results, f, indent=2)
    print(f"\nWrote {out_dir}/phase_h_anchor_robustness.json")

    # Save C1 as JSON + flat CSV
    with (out_dir / "phase_h_per_role_gap_ci.json").open("w") as f:
        json.dump(c1_results, f, indent=2)
    csv_path = out_dir / "phase_h_per_role_gap_ci.csv"
    with csv_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["axis", "role", "phi_z_mean", "phi_z_std",
                                           "llama_z_mean", "llama_z_std", "gap_mean",
                                           "gap_p2.5", "gap_p97.5", "gap_excludes_zero"])
        w.writeheader()
        for axis, rows in c1_results.items():
            for r in rows:
                row = {**r, "axis": axis}
                w.writerow(row)
    print(f"Wrote {csv_path}")

    # Print top significant gaps
    print("\n=== Top 5 cross-model gap roles per axis (CI excludes zero) ===")
    for axis, rows in c1_results.items():
        sig = [r for r in rows if r["gap_excludes_zero"]]
        sig.sort(key=lambda r: abs(r["gap_mean"]), reverse=True)
        print(f"\n  {axis}:")
        for r in sig[:5]:
            print(f"    {r['role']:25s} gap={r['gap_mean']:+.3f}  CI=[{r['gap_p2.5']:+.3f}, {r['gap_p97.5']:+.3f}]")


if __name__ == "__main__":
    main()
