"""Phase F plots: visualize Q2 (PC1 rotation) + Q3 (point migration) across all steered traits.

Inputs:
  results/comparison/phase_f_comparison.json (from 16_q3_point_migration_analysis.py)
  results/original/{axis.pt, vectors/}
  results/{trait}-steered-L*-a*/vectors/

Outputs (results/comparison/plots/):
  pc1_pc2_scatter.png        — 4-panel PCA scatter (orig + each steered)
  per_role_displacement.png  — top-10 displaced roles per trait, color = cos(δ, axis)
  default_migration.png      — 1D number line on axis_orig, default + predicted targets
  centroid_shift.png         — bar chart of centroid shift norm + alignment
"""
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.decomposition import PCA

REPO = Path("/scratch/paulc/assistant-axis-abliteration")
PLOTS = REPO / "results/comparison/plots"
PLOTS.mkdir(parents=True, exist_ok=True)

TRAIT_COLORS = {
    "evil": "#d62728",
    "sycophantic": "#1f77b4",
    "apathetic": "#7f7f7f",
    "humorous": "#ff7f0e",
}


def load_role_vectors(vec_dir):
    roles, vecs = [], []
    for f in sorted(Path(vec_dir).glob("*.pt")):
        d = torch.load(f, weights_only=False)
        if isinstance(d, dict):
            v = d["vector"].squeeze(0).float() if d.get("vector") is not None else torch.as_tensor(d.get("diff")).float().squeeze()
            r = d.get("role", f.stem)
        else:
            v = torch.as_tensor(d).float().squeeze()
            r = f.stem
        roles.append(r)
        vecs.append(v)
    return roles, torch.stack(vecs).numpy()


def find_default_idx(roles):
    for cand in ["default", "default_assistant", "assistant", "no_role"]:
        if cand in roles:
            return roles.index(cand)
    raise ValueError(f"can't find default in {roles[:5]}")


def discover_steered():
    """Find {trait: steered_dir} for each trait that has vectors/."""
    MODEL_PREFIX = "llama-3.1-8b-"
    out = {}
    for d in sorted((REPO / "results").glob("*-steered-L*-a*")):
        if (d / "vectors").exists() and any((d / "vectors").iterdir()):
            trait = d.name.split("-steered-")[0]
            if trait.startswith(MODEL_PREFIX):
                trait = trait[len(MODEL_PREFIX):]
            out[trait] = d
    return out


def plot_pc1_pc2_scatter(roles, V_orig, traits_data, q3_predictions):
    """4-panel PCA scatter: original cloud + steered cloud overlaid per trait."""
    mu_o = V_orig.mean(0)
    pca = PCA(n_components=2).fit(V_orig - mu_o)
    proj_orig = (V_orig - mu_o) @ pca.components_.T
    default_idx = find_default_idx(roles)

    n = len(traits_data)
    cols = min(2, n)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(7 * cols, 6 * rows), squeeze=False)
    axes = axes.flatten()

    for i, (trait, V_s) in enumerate(traits_data.items()):
        ax = axes[i]
        proj_s = (V_s - mu_o) @ pca.components_.T
        color = TRAIT_COLORS.get(trait, "purple")
        ax.scatter(proj_orig[:, 0], proj_orig[:, 1], s=12, alpha=0.35, c="gray", label="original")
        ax.scatter(proj_s[:, 0], proj_s[:, 1], s=12, alpha=0.55, c=color, label=f"{trait} steered")
        # default role markers
        ax.scatter(*proj_orig[default_idx], s=200, marker="*", c="black", edgecolors="white", label="default (orig)", zorder=5)
        ax.scatter(*proj_s[default_idx], s=200, marker="*", c=color, edgecolors="white", label="default (steered)", zorder=5)
        # predicted targets (from original cloud)
        for tgt in q3_predictions.get(trait, []):
            if tgt in roles:
                ti = roles.index(tgt)
                ax.annotate(tgt, (proj_orig[ti, 0], proj_orig[ti, 1]),
                            fontsize=8, ha="left", color="black",
                            bbox=dict(boxstyle="round,pad=0.2", fc="lightyellow", ec="black", lw=0.5))
        ax.set_xlabel(f"PC1 (orig, var={pca.explained_variance_ratio_[0]:.2%})")
        ax.set_ylabel(f"PC2 (orig, var={pca.explained_variance_ratio_[1]:.2%})")
        ax.set_title(f"{trait} — persona-space points in original PC1×PC2")
        ax.legend(fontsize=8, loc="best")
        ax.grid(alpha=0.3)
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)
    fig.tight_layout()
    out = PLOTS / "pc1_pc2_scatter.png"
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out}")


def plot_per_role_displacement(roles, V_orig, traits_data, axis_orig):
    """Top-10 most-displaced roles per trait, bar color by cos(δ_role, axis_orig)."""
    axis_unit = axis_orig / np.linalg.norm(axis_orig)
    n = len(traits_data)
    cols = min(2, n)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(8 * cols, 6 * rows), squeeze=False)
    axes = axes.flatten()

    for i, (trait, V_s) in enumerate(traits_data.items()):
        ax = axes[i]
        delta = V_s - V_orig                                      # [N, 4096]
        norms = np.linalg.norm(delta, axis=1)                     # [N]
        cos_axis = (delta @ axis_unit) / np.clip(norms, 1e-9, None)  # [N]
        top_idx = np.argsort(norms)[::-1][:10]
        names = [roles[j] for j in top_idx]
        vals = norms[top_idx]
        cols_ = ["#d62728" if cos_axis[j] < 0 else "#1f77b4" for j in top_idx]
        y = np.arange(len(names))
        ax.barh(y, vals, color=cols_, alpha=0.8)
        for j, (yi, c) in enumerate(zip(y, cos_axis[top_idx])):
            ax.text(vals[j] * 1.01, yi, f"cos={c:+.2f}", va="center", fontsize=8)
        ax.set_yticks(y)
        ax.set_yticklabels(names, fontsize=9)
        ax.invert_yaxis()
        ax.set_xlabel("‖δ_role‖ = ‖V_steered − V_orig‖")
        ax.set_title(f"{trait} — top-10 displaced roles (red = away from Assistant pole)")
        ax.grid(alpha=0.3, axis="x")
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)
    fig.tight_layout()
    out = PLOTS / "per_role_displacement.png"
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out}")


def plot_default_migration(roles, V_orig, traits_data, axis_orig, q3_predictions):
    """1D number line: each role's projection onto axis_orig. Default + predicted targets highlighted."""
    mu_o = V_orig.mean(0)
    axis_unit = axis_orig / np.linalg.norm(axis_orig)
    proj_orig = (V_orig - mu_o) @ axis_unit                        # [N]
    default_idx = find_default_idx(roles)

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.scatter(proj_orig, np.zeros_like(proj_orig), s=18, alpha=0.4, c="gray", label="original roles")
    ax.scatter(proj_orig[default_idx], 0, s=350, marker="*", c="black", edgecolors="white", label="default (orig)", zorder=5)

    for trait, V_s in traits_data.items():
        proj_def = float((V_s[default_idx] - mu_o) @ axis_unit)
        color = TRAIT_COLORS.get(trait, "purple")
        ax.scatter(proj_def, 0, s=350, marker="*", c=color, edgecolors="white", label=f"default ({trait})", zorder=6)
        # arrow from orig default to steered default
        ax.annotate("", xy=(proj_def, 0.04), xytext=(proj_orig[default_idx], 0.04),
                    arrowprops=dict(arrowstyle="->", color=color, lw=1.5, alpha=0.8))

    # label predicted target archetypes (intersection across traits to avoid clutter)
    labeled = set()
    for trait, targets in q3_predictions.items():
        for tgt in targets:
            if tgt in roles and tgt not in labeled:
                ti = roles.index(tgt)
                ax.annotate(tgt, (proj_orig[ti], 0), xytext=(proj_orig[ti], -0.06 - 0.02 * (len(labeled) % 4)),
                            fontsize=8, ha="center",
                            arrowprops=dict(arrowstyle="-", color="black", alpha=0.4, lw=0.5))
                labeled.add(tgt)

    ax.set_xlabel("Projection onto axis_orig (Assistant Axis from original model)")
    ax.set_yticks([])
    ax.set_title("Q3 — where do steered defaults land on the ORIGINAL Assistant Axis?")
    ax.legend(fontsize=8, loc="upper right", ncol=2)
    ax.grid(alpha=0.3, axis="x")
    ax.axhline(0, color="black", lw=0.5)
    fig.tight_layout()
    out = PLOTS / "default_migration.png"
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out}")


def plot_centroid_shift(comp_data):
    """Bar chart: centroid shift norm per trait, color by cos(centroid_shift, axis_orig)."""
    fig, ax = plt.subplots(figsize=(8, 5))
    names, norms, cosines = [], [], []
    MODEL_PREFIX = "llama-3.1-8b-"
    for k, r in comp_data.items():
        trait = k.split("-steered-")[0]
        if trait.startswith(MODEL_PREFIX):
            trait = trait[len(MODEL_PREFIX):]
        q3 = r["Q3_point_migration"]
        names.append(trait)
        norms.append(q3["centroid_shift_norm"])
        cosines.append(q3["cos_centroid_axis"])
    cmap = plt.cm.RdBu_r
    norm_color = plt.Normalize(vmin=-1, vmax=1)
    bars = ax.bar(names, norms, color=[cmap(norm_color(c)) for c in cosines])
    for b, c in zip(bars, cosines):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() * 1.01,
                f"cos={c:+.2f}", ha="center", fontsize=10)
    ax.set_ylabel("‖μ_steered − μ_orig‖ (centroid shift)")
    ax.set_title("Q3 — persona-cloud rigid translation per trait")
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm_color)
    sm.set_array([])
    plt.colorbar(sm, ax=ax, label="cos(centroid_shift, axis_orig)")
    fig.tight_layout()
    out = PLOTS / "centroid_shift.png"
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out}")


Q3_PREDICTIONS = {
    "evil": ["demon", "saboteur", "criminal", "vigilante", "sociopath"],
    "sycophantic": ["courtier", "yes-man", "subordinate", "sycophant"],
    "apathetic": ["drifter", "nihilist", "drone", "slacker"],
    "humorous": ["jester", "comedian", "trickster", "fool", "absurdist"],
}


def main():
    comp_path = REPO / "results/comparison/phase_f_comparison.json"
    if not comp_path.exists():
        raise SystemExit(f"missing {comp_path} — run scripts/16_q3_point_migration_analysis.py first")
    comp_data = json.loads(comp_path.read_text())

    print("Loading original axis + role vectors...")
    axis_orig = torch.load(REPO / "results/original/axis.pt", weights_only=False)
    if isinstance(axis_orig, dict):
        axis_orig = axis_orig.get("axis", axis_orig.get("vector"))
    axis_orig = torch.as_tensor(axis_orig).float().squeeze().numpy()
    roles_orig, V_orig = load_role_vectors(REPO / "results/original/vectors")
    print(f"  {len(roles_orig)} roles, axis ‖.‖={np.linalg.norm(axis_orig):.3f}")

    steered = discover_steered()
    if not steered:
        raise SystemExit("no steered dirs found under results/")
    traits_data = {}
    common_roles_all = set(roles_orig)
    for trait, d in steered.items():
        roles_s, V_s = load_role_vectors(d / "vectors")
        common_roles_all &= set(roles_s)
    common_roles = [r for r in roles_orig if r in common_roles_all]
    if len(common_roles) < len(roles_orig):
        print(f"  intersecting roles to {len(common_roles)} (filtered {len(roles_orig) - len(common_roles)} roles missing in some traits)")
    idx_o = [roles_orig.index(r) for r in common_roles]
    V_orig_common = V_orig[idx_o]
    for trait, d in steered.items():
        roles_s, V_s = load_role_vectors(d / "vectors")
        idx_s = [roles_s.index(r) for r in common_roles]
        traits_data[trait] = V_s[idx_s]
        print(f"  loaded {trait}: {traits_data[trait].shape}")
    roles_orig = common_roles
    V_orig = V_orig_common

    print("\nGenerating plots...")
    plot_pc1_pc2_scatter(roles_orig, V_orig, traits_data, Q3_PREDICTIONS)
    plot_per_role_displacement(roles_orig, V_orig, traits_data, axis_orig)
    plot_default_migration(roles_orig, V_orig, traits_data, axis_orig, Q3_PREDICTIONS)
    plot_centroid_shift(comp_data)

    print(f"\nAll plots saved under {PLOTS}/")


if __name__ == "__main__":
    main()
