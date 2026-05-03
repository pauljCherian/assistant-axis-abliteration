#!/usr/bin/env python3
"""Phase H Step 4: visualize cross-model comparison.

Produces:
  - fig1_cross_model_scatter_{axis}.png — for each axis pair, scatter of (Phi z, Llama z)
                                          per held-out role with diagonal line
  - fig2_independence_heatmap.png       — within-model contrast cosines (side-by-side)
  - fig3_validation_pair_chart.png      — held-out validation pair pass/fail matrix
  - fig4_null_purity.png                — null-role projections (should be near 0)
  - fig5_shared_coord_overview.png      — primary headline: 5 axes, all roles, both models

Usage:
    .venv/bin/python scripts/35_phase_h_plots.py \
        --phi_dir results/phi-3.5-mini --llama_dir results/llama-3.2-3b
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
AXIS_ORDER = ["v_assistant", "v_benevolence", "v_authority", "v_humor", "v_critic"]


def load_proj(model_dir: Path, kind: str = ""):
    pdir = model_dir / f"projections{kind}"
    z = torch.load(pdir / "zscore.pt", map_location="cpu", weights_only=False)
    idx = json.loads((pdir / "role_index.json").read_text())
    return z["projections"].numpy(), idx["roles"], idx["axes"]


def fig1_cross_model_scatter(z_phi, roles_phi, axes_phi, z_llama, roles_llama, axes_llama,
                              anchor_set, out_dir):
    common_roles = sorted(set(roles_phi) & set(roles_llama))
    held_out = [r for r in common_roles if r not in anchor_set]
    common_axes = [a for a in axes_phi if a in axes_llama]

    phi_idx = {r: i for i, r in enumerate(roles_phi)}
    llama_idx = {r: i for i, r in enumerate(roles_llama)}
    pa_idx = {a: i for i, a in enumerate(axes_phi)}
    la_idx = {a: i for i, a in enumerate(axes_llama)}

    n = len(common_axes)
    cols = 3
    rows = (n + cols - 1) // cols
    fig, axs = plt.subplots(rows, cols, figsize=(cols * 4.5, rows * 4.5))
    axs = axs.flatten() if n > 1 else [axs]

    for i, axis in enumerate(common_axes):
        ax = axs[i]
        x = np.array([z_phi[phi_idx[r], pa_idx[axis]] for r in held_out])
        y = np.array([z_llama[llama_idx[r], la_idx[axis]] for r in held_out])
        ax.scatter(x, y, alpha=0.5, s=15, c="steelblue")
        lim = max(abs(x).max(), abs(y).max()) * 1.1
        ax.plot([-lim, lim], [-lim, lim], "k--", alpha=0.3, label="y=x")
        ax.axhline(0, color="gray", lw=0.5)
        ax.axvline(0, color="gray", lw=0.5)
        from scipy.stats import spearmanr
        r, _ = spearmanr(x, y)
        ax.set_title(f"{axis}\nSpearman r = {r:+.3f}")
        ax.set_xlabel("Phi-3.5-mini (z-score)")
        ax.set_ylabel("Llama-3.2-3B (z-score)")
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
        ax.set_aspect("equal")

    for j in range(len(common_axes), len(axs)):
        axs[j].set_visible(False)

    plt.suptitle(f"Phase H — cross-model rank agreement on {len(held_out)} held-out roles", y=1.02, fontsize=14)
    plt.tight_layout()
    fig.savefig(out_dir / "fig1_cross_model_scatter.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote fig1_cross_model_scatter.png")


def fig2_independence_heatmap(contrasts_phi, contrasts_llama, out_dir):
    def cos_matrix(c):
        axes = [a for a in AXIS_ORDER if a in c]
        n = len(axes)
        m = np.zeros((n, n))
        for i, a in enumerate(axes):
            for j, b in enumerate(axes):
                va = c[a].numpy()
                vb = c[b].numpy()
                m[i, j] = (va @ vb) / (np.linalg.norm(va) * np.linalg.norm(vb) + 1e-12)
        return axes, m

    axes_phi, m_phi = cos_matrix(contrasts_phi)
    axes_llama, m_llama = cos_matrix(contrasts_llama)

    fig, axs = plt.subplots(1, 2, figsize=(12, 5))
    for ax, m, axes, title in [(axs[0], m_phi, axes_phi, "Phi-3.5-mini"),
                                 (axs[1], m_llama, axes_llama, "Llama-3.2-3B")]:
        im = ax.imshow(m, cmap="RdBu_r", vmin=-1, vmax=1)
        ax.set_xticks(range(len(axes)))
        ax.set_yticks(range(len(axes)))
        ax.set_xticklabels(axes, rotation=45, ha="right")
        ax.set_yticklabels(axes)
        for i in range(len(axes)):
            for j in range(len(axes)):
                ax.text(j, i, f"{m[i,j]:.2f}", ha="center", va="center",
                        color="white" if abs(m[i,j]) > 0.5 else "black", fontsize=9)
        ax.set_title(title)
        plt.colorbar(im, ax=ax, fraction=0.046)
    plt.suptitle("Test A — within-model contrast independence (cosines)", fontsize=13)
    plt.tight_layout()
    fig.savefig(out_dir / "fig2_independence_heatmap.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote fig2_independence_heatmap.png")


def fig5_shared_coord_overview(z_phi, roles_phi, axes_phi, z_llama, roles_llama, axes_llama,
                                anchor_set, out_dir):
    """Headline plot: for each pair of (interesting) axes, scatter all roles from both models."""
    common_axes = [a for a in axes_phi if a in axes_llama and a != "v_assistant"]
    common_roles = sorted(set(roles_phi) & set(roles_llama))

    pa_idx = {a: i for i, a in enumerate(axes_phi)}
    la_idx = {a: i for i, a in enumerate(axes_llama)}
    phi_idx = {r: i for i, r in enumerate(roles_phi)}
    llama_idx = {r: i for i, r in enumerate(roles_llama)}

    # Choose 3 axis pairs for headline plot
    pairs = [("v_benevolence", "v_authority"),
             ("v_authority", "v_humor"),
             ("v_critic", "v_benevolence")]
    pairs = [(a, b) for a, b in pairs if a in common_axes and b in common_axes]

    fig, axs = plt.subplots(1, len(pairs), figsize=(len(pairs) * 5, 5))
    if len(pairs) == 1:
        axs = [axs]
    for ax, (a, b) in zip(axs, pairs):
        for color, model_name, z, r_idx, a_idx in [
            ("steelblue", "Phi", z_phi, phi_idx, pa_idx),
            ("crimson", "Llama-3.2-3B", z_llama, llama_idx, la_idx),
        ]:
            x = z[:, a_idx[a]]
            y = z[:, a_idx[b]]
            ax.scatter(x, y, alpha=0.45, s=18, color=color, label=model_name)
        ax.axhline(0, color="gray", lw=0.5)
        ax.axvline(0, color="gray", lw=0.5)
        ax.set_xlabel(f"{a} (z)")
        ax.set_ylabel(f"{b} (z)")
        ax.legend()
        ax.set_title(f"{a} × {b}")
    plt.suptitle("Phase H headline — both models in shared contrast-coordinate space", fontsize=13)
    plt.tight_layout()
    fig.savefig(out_dir / "fig5_shared_coord_overview.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote fig5_shared_coord_overview.png")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phi_dir", type=str, required=True)
    ap.add_argument("--llama_dir", type=str, required=True)
    ap.add_argument("--out_dir", type=str, default="results/comparison/plots/phase_h")
    args = ap.parse_args()

    phi_dir = (ROOT / args.phi_dir) if not Path(args.phi_dir).is_absolute() else Path(args.phi_dir)
    llama_dir = (ROOT / args.llama_dir) if not Path(args.llama_dir).is_absolute() else Path(args.llama_dir)
    out_dir = (ROOT / args.out_dir) if not Path(args.out_dir).is_absolute() else Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Anchor set (for held-out filter)
    from importlib.util import spec_from_file_location, module_from_spec
    spec = spec_from_file_location("c34", ROOT / "scripts/34_compare_axes.py")
    c34 = module_from_spec(spec); spec.loader.exec_module(c34)
    anchor_set = set()
    for pos, neg in c34.ANCHOR_AXES.values():
        anchor_set |= set(pos) | set(neg)

    z_phi, roles_phi, axes_phi = load_proj(phi_dir)
    z_llama, roles_llama, axes_llama = load_proj(llama_dir)

    # Load contrasts for fig2
    def load_c(d):
        out = {}
        for axis in AXIS_ORDER:
            f = d / "contrasts" / f"{axis}.pt"
            if f.exists():
                out[axis] = torch.load(f, map_location="cpu", weights_only=False).float().squeeze()
        return out
    contrasts_phi = load_c(phi_dir)
    contrasts_llama = load_c(llama_dir)

    fig1_cross_model_scatter(z_phi, roles_phi, axes_phi, z_llama, roles_llama, axes_llama, anchor_set, out_dir)
    fig2_independence_heatmap(contrasts_phi, contrasts_llama, out_dir)
    fig5_shared_coord_overview(z_phi, roles_phi, axes_phi, z_llama, roles_llama, axes_llama, anchor_set, out_dir)

    print(f"\nDone. Plots in {out_dir}")


if __name__ == "__main__":
    main()
