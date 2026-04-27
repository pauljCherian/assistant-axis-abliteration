#!/usr/bin/env python3
"""Plotting script for the persona-space comparison.

Produces the figures needed for the final writeup from
results/comparison/axis_comparison.json (+ vectors loaded separately).

Figures:
  1. pca_scatter.png — role vectors projected onto (PC1, PC2) for each model,
     with default-Assistant and safety-adjacent roles highlighted.
  2. displacement_vs_refusal.png — per-role ‖Δv‖ on x, cos(Δv, refusal) on y.
     If points cluster near y=0 → abliteration displaces roles ORTHOGONAL to
     refusal (hypothesis 2). Clustered near y=±1 → aligned (hypothesis 1).
  3. top_movers_bar.png — top-20 movers by raw displacement, colored by
     safety-adjacency.
  4. null_vs_observed.png — histogram of bootstrap-null axis cosines vs the
     observed orig-vs-abl cosine (vertical line). If observed is left of the
     p05 line, the change exceeds sampling noise.

Run AFTER scripts/05_compare_persona_spaces.py produces the JSON report.

Safe to run on original-only data (dry-run): produces only the figures that
don't require abliterated data.
"""
import json
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
COMP = ROOT / "results/comparison"


def load(path: Path):
    return json.loads(path.read_text()) if path.exists() else None


def load_vectors(dir_path: Path):
    if not dir_path.exists():
        return {}
    out = {}
    for f in sorted(dir_path.glob("*.pt")):
        d = torch.load(f, map_location="cpu", weights_only=False)
        out[d.get("role", f.stem)] = d["vector"].squeeze().float()
    return out


def safe_savefig(fig, path):
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"Wrote {path}")


def plot_pca_scatter(orig_vecs, abl_vecs, predictions, out_dir):
    import matplotlib.pyplot as plt
    from sklearn.decomposition import PCA

    safety = set(predictions["safety_adjacent_roles"]) if predictions else set()

    fig, axes = plt.subplots(1, 2 if abl_vecs else 1, figsize=(12, 5), squeeze=False)

    for ax, (name, vecs) in zip(
        axes[0],
        [("original", orig_vecs)] + ([("abliterated", abl_vecs)] if abl_vecs else []),
    ):
        roles = sorted(vecs.keys())
        X = np.stack([vecs[r].numpy() for r in roles])
        pca = PCA(n_components=2)
        Y = pca.fit_transform(X)
        is_safety = np.array([r in safety for r in roles])
        ax.scatter(Y[~is_safety, 0], Y[~is_safety, 1], s=10, alpha=0.5,
                   color="tab:gray", label="other roles")
        ax.scatter(Y[is_safety, 0], Y[is_safety, 1], s=18, alpha=0.7,
                   color="tab:red", label="safety-adjacent")
        # Label default / assistant roles if present
        for special in ("assistant", "default"):
            if special in roles:
                i = roles.index(special)
                ax.scatter(Y[i, 0], Y[i, 1], s=100, marker="*",
                           color="gold", edgecolor="black", zorder=10,
                           label=f"'{special}'")
        ax.set_title(f"{name} — role vectors on PC1/PC2 "
                     f"(var_expl = {pca.explained_variance_ratio_[0]*100:.1f}%+"
                     f"{pca.explained_variance_ratio_[1]*100:.1f}%)")
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        ax.legend(loc="best", fontsize=8)
        ax.axhline(0, color="k", linewidth=0.3)
        ax.axvline(0, color="k", linewidth=0.3)

    safe_savefig(fig, out_dir / "pca_scatter.png")
    plt.close(fig)


def plot_displacement_vs_refusal(report, out_dir):
    import matplotlib.pyplot as plt

    if "per_role_raw_displacement" not in report:
        return
    disp = report["per_role_raw_displacement"].get("top_by_norm", [])
    if not disp:
        return
    roles = [r for r, _, _ in disp]
    norms = np.array([n for _, n, _ in disp])
    cosines = np.array([c for _, _, c in disp])
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(norms, cosines, alpha=0.6)
    for r, n, c in zip(roles[:10], norms[:10], cosines[:10]):
        ax.annotate(r, (n, c), fontsize=7, alpha=0.8)
    ax.axhline(0, color="k", linewidth=0.5, linestyle="--")
    ax.set_xlabel("‖Δv‖  (raw per-role displacement)")
    ax.set_ylabel("cos(Δv, refusal_direction)")
    ax.set_title("Per-role displacement under abliteration — "
                 "y≈±1 aligned with refusal, y≈0 orthogonal")
    ax.set_ylim(-1.05, 1.05)
    safe_savefig(fig, out_dir / "displacement_vs_refusal.png")
    plt.close(fig)


def plot_top_movers(report, predictions, out_dir):
    import matplotlib.pyplot as plt

    if "per_role_raw_displacement" not in report:
        return
    top = report["per_role_raw_displacement"].get("top_by_norm", [])
    if not top:
        return
    safety = set(predictions["safety_adjacent_roles"]) if predictions else set()
    N = min(20, len(top))
    roles = [r for r, _, _ in top[:N]]
    norms = [n for _, n, _ in top[:N]]
    colors = ["tab:red" if r in safety else "tab:gray" for r in roles]
    fig, ax = plt.subplots(figsize=(10, max(4, N * 0.3)))
    ax.barh(range(N), norms, color=colors)
    ax.set_yticks(range(N))
    ax.set_yticklabels(roles)
    ax.invert_yaxis()
    ax.set_xlabel("‖v_abl − v_orig‖")
    ax.set_title(f"Top-{N} most-displaced roles under abliteration\n"
                 "(red = pre-registered safety-adjacent)")
    safe_savefig(fig, out_dir / "top_movers_bar.png")
    plt.close(fig)


def plot_null_vs_observed(report, out_dir):
    import matplotlib.pyplot as plt

    if "null_model" not in report or "axes" not in report:
        return
    if "cos_original_vs_abliterated" not in report["axes"]:
        # Dry-run mode — skip
        return
    null = report["null_model"]["original_bootstrap"]
    observed = report["axes"]["cos_original_vs_abliterated"]
    # For a one-sided test: how extreme is observed relative to null?
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.axvline(null["p05"], color="gray", linestyle=":", label=f"null p05 = {null['p05']:.4f}")
    ax.axvline(null["p50"], color="gray", linestyle="-", label=f"null median = {null['p50']:.4f}")
    ax.axvline(null["p95"], color="gray", linestyle=":", label=f"null p95 = {null['p95']:.4f}")
    ax.axvspan(null["p05"], null["p95"], color="lightgray", alpha=0.4, label="null p05–p95")
    ax.axvline(observed, color="tab:red", linewidth=3,
               label=f"observed = {observed:.4f}")
    ax.set_xlabel("cos(axis_A, axis_B)")
    ax.set_title("Orig-vs-abl axis cosine vs bootstrap null (role sampling noise)")
    ax.legend(loc="best")
    safe_savefig(fig, out_dir / "null_vs_observed.png")
    plt.close(fig)


def main():
    report = load(COMP / "axis_comparison.json")
    if not report:
        raise SystemExit("Run scripts/05_compare_persona_spaces.py first")
    predictions = load(COMP / "predictions.json")

    orig_vecs = load_vectors(ROOT / "results/original/vectors")
    abl_vecs = load_vectors(ROOT / "results/abliterated/vectors")

    COMP.mkdir(parents=True, exist_ok=True)
    plots_dir = COMP / "plots"
    plots_dir.mkdir(exist_ok=True)

    print(f"=== Plotting @ {report.get('generated_at')} ===")
    plot_pca_scatter(orig_vecs, abl_vecs, predictions, plots_dir)
    plot_displacement_vs_refusal(report, plots_dir)
    plot_top_movers(report, predictions, plots_dir)
    plot_null_vs_observed(report, plots_dir)

    print(f"\nAll plots in {plots_dir}")


if __name__ == "__main__":
    main()
