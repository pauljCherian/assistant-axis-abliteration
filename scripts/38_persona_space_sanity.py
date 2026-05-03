#!/usr/bin/env python3
"""
Basic persona-space sanity check.

RUN THIS FIRST after vectors are extracted, BEFORE any contrast-axis analysis (scripts 32-37).

If these checks fail, the extraction is broken and contrast-axis work is moot.

Tests:
1. PCA variance: persona space should be low-dim (Lu et al.: 4-19 PCs explain 70%).
2. PC1 = Assistant Axis: |cos(v_assistant, PC1)| > 0.7 (Lu et al. threshold).
3. Default at extreme of PC1 (>95th percentile).
4. Semantic clustering: hand-curated similar pairs (saboteur/demon, healer/counselor)
   should have higher mean cos than dissimilar pairs (healer/criminal).
5. 2D PCA scatter with anchors highlighted — visual sanity check.

Usage:
    python scripts/38_persona_space_sanity.py --model_dir results/llama-3.2-3b
    python scripts/38_persona_space_sanity.py --model_dir results/llama-3.2-3b --kind _unfiltered

Exit 0 if all critical checks pass; 1 if any FAIL.
"""
import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.decomposition import PCA

ROOT = Path(__file__).resolve().parent.parent

# Hand-curated pairs — semantically similar roles whose vectors should be close.
SIMILAR_PAIRS = [
    ("saboteur", "demon"),
    ("healer", "counselor"),
    ("criminal", "predator"),
    ("philosopher", "mathematician"),
    ("comedian", "jester"),
    ("contrarian", "skeptic"),
    ("scientist", "polymath"),
    ("guardian", "parent"),
    ("ascetic", "hermit"),
    ("evangelist", "advocate"),
]

# Hand-curated pairs — semantically dissimilar roles whose vectors should be far.
DISSIMILAR_PAIRS = [
    ("healer", "criminal"),
    ("comedian", "philosopher"),
    ("counselor", "predator"),
    ("librarian", "anarchist"),
    ("scientist", "fool"),
    ("ascetic", "comedian"),
]

# Anchor roles to color/label on the 2D PCA scatter.
ANCHOR_GROUPS = {
    "v_benevolence_pos": (["counselor", "parent", "guardian", "healer"], "green"),
    "v_benevolence_neg": (["criminal", "saboteur", "demon", "predator"], "red"),
    "v_authority_pos": (["judge", "scientist", "sage", "ancient"], "blue"),
    "v_authority_neg": (["amateur", "student", "infant", "orphan"], "lightblue"),
    "v_humor_pos": (["comedian", "jester", "fool", "absurdist"], "orange"),
    "v_humor_neg": (["philosopher", "mathematician", "ascetic", "scholar"], "purple"),
    "v_critic_pos": (["contrarian", "devils_advocate", "skeptic", "perfectionist"], "magenta"),
    "v_critic_neg": (["synthesizer", "optimist", "evangelist", "advocate"], "cyan"),
}


def load_vectors(vec_dir: Path) -> dict:
    """Load all .pt role vectors from a directory."""
    vectors = {}
    for f in sorted(vec_dir.glob("*.pt")):
        v = torch.load(f, map_location="cpu", weights_only=False)
        if isinstance(v, dict):
            v = v.get("vector", v.get("mean", next(iter(v.values()))))
        vectors[f.stem] = v.float().squeeze().numpy()
    return vectors


def cos(a, b):
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    return float(np.dot(a, b) / (na * nb + 1e-12)) if na > 0 and nb > 0 else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_dir", type=str, required=True,
                    help="path under results/ — e.g. 'llama-3.2-3b' or 'phi-3.5-mini'")
    ap.add_argument("--kind", type=str, default="", choices=["", "_unfiltered"],
                    help="'' = filtered (default), '_unfiltered' = raw mean")
    args = ap.parse_args()

    model_dir = Path(args.model_dir)
    if not model_dir.is_absolute():
        model_dir = ROOT / "results" / args.model_dir if not args.model_dir.startswith("results") else ROOT / args.model_dir

    vec_dir = model_dir / f"vectors{args.kind}"
    default_path = model_dir / "default.pt"

    print(f"=== Persona Space Sanity Check ===")
    print(f"Model dir: {model_dir}")
    print(f"Vector dir: {vec_dir.name}")
    print()

    if not vec_dir.exists():
        print(f"FATAL: {vec_dir} does not exist. Run pipeline first.")
        sys.exit(2)
    if not default_path.exists():
        print(f"FATAL: {default_path} not found. Run step 6 of pipeline.")
        sys.exit(2)

    vectors = load_vectors(vec_dir)
    role_names = sorted([r for r in vectors.keys() if r != "default"])
    print(f"Loaded {len(role_names)} role vectors")
    if len(role_names) < 200:
        print(f"  WARN: fewer roles than expected (~275). Filtering may have been aggressive.")

    default_vec = torch.load(default_path, map_location="cpu", weights_only=False).float().squeeze().numpy()

    X = np.stack([vectors[r] for r in role_names])
    print(f"Role matrix: {X.shape}")
    print()

    n_pass, n_fail, n_warn = 0, 0, 0
    verdicts = {}

    # --- Test 1: PCA variance explained ---
    print("--- 1. PCA variance explained ---")
    pca = PCA(n_components=min(20, len(role_names) - 1))
    pca.fit(X)
    cum_var = np.cumsum(pca.explained_variance_ratio_)
    pc1_var = float(pca.explained_variance_ratio_[0])
    n_for_70pct = int(np.argmax(cum_var >= 0.70) + 1) if cum_var.max() >= 0.70 else len(cum_var)
    print(f"  PC1 explains: {pc1_var:.1%}")
    print(f"  Top 5 cumulative: {cum_var[min(4, len(cum_var)-1)]:.1%}")
    print(f"  PCs needed for 70%: {n_for_70pct}")
    if pc1_var > 0.50:
        print(f"  WARN: PC1 alone explains {pc1_var:.1%} — possibly overcollapsed")
        verdicts["pc1_variance"] = "WARN"; n_warn += 1
    elif n_for_70pct > 25:
        print(f"  WARN: {n_for_70pct} PCs for 70% — high-dim, atypical (Lu et al.: 4-19)")
        verdicts["pc1_variance"] = "WARN"; n_warn += 1
    else:
        print(f"  PASS: persona space is low-dim (matches Lu et al.)")
        verdicts["pc1_variance"] = "PASS"; n_pass += 1

    # --- Test 2: PC1 = Assistant Axis ---
    print("\n--- 2. cos(v_assistant, PC1) ---")
    v_assistant = default_vec - X.mean(axis=0)
    pc1 = pca.components_[0]
    pc1_cos = abs(cos(v_assistant, pc1))
    print(f"  |cos(v_assistant, PC1)| = {pc1_cos:.3f}")
    if pc1_cos > 0.7:
        print(f"  PASS: PC1 IS the Assistant Axis (Lu et al. threshold > 0.7)")
        verdicts["pc1_assistant"] = "PASS"; n_pass += 1
    elif pc1_cos > 0.5:
        print(f"  PARTIAL: PC1 somewhat assistant-aligned but below Lu et al. threshold")
        verdicts["pc1_assistant"] = "PARTIAL"; n_warn += 1
    else:
        print(f"  FAIL: PC1 is NOT the Assistant Axis. Extraction likely broken.")
        verdicts["pc1_assistant"] = "FAIL"; n_fail += 1

    # --- Test 3: Default at extreme of PC1 ---
    print("\n--- 3. Default at extreme of PC1 ---")
    role_pc1 = X @ pc1
    default_pc1 = default_vec @ pc1
    if default_pc1 > role_pc1.mean():
        percentile = float((role_pc1 < default_pc1).mean())
        side = "high"
    else:
        percentile = float((role_pc1 > default_pc1).mean())
        side = "low"
    print(f"  Default's PC1 percentile (extremity): {percentile:.1%} ({side} side)")
    if percentile > 0.95:
        print(f"  PASS: default at extreme of PC1")
        verdicts["default_extreme"] = "PASS"; n_pass += 1
    elif percentile > 0.85:
        print(f"  PARTIAL: default near but not at extreme")
        verdicts["default_extreme"] = "PARTIAL"; n_warn += 1
    else:
        print(f"  FAIL: default not at extreme of PC1 — Assistant Axis broken")
        verdicts["default_extreme"] = "FAIL"; n_fail += 1

    # --- Test 4: Semantic clustering (pairwise cosines on CENTERED vectors) ---
    # Raw cosines on residual-stream vectors are always ~1.0 due to a dominant
    # common-mode component. Center first to reveal role-specific structure.
    print("\n--- 4. Pairwise cosines (CENTERED): similar should cluster, dissimilar should not ---")
    mean_vec = X.mean(axis=0)
    centered = {k: v - mean_vec for k, v in vectors.items() if k != "default"}
    print("  Similar pairs:")
    similar_cos = []
    for a, b in SIMILAR_PAIRS:
        if a in centered and b in centered:
            c = cos(centered[a], centered[b])
            similar_cos.append(c)
            print(f"    cos({a:>20s}, {b:>20s}) = {c:+.3f}")
    print("  Dissimilar pairs:")
    dissimilar_cos = []
    for a, b in DISSIMILAR_PAIRS:
        if a in centered and b in centered:
            c = cos(centered[a], centered[b])
            dissimilar_cos.append(c)
            print(f"    cos({a:>20s}, {b:>20s}) = {c:+.3f}")
    mean_sim = float(np.mean(similar_cos)) if similar_cos else 0.0
    mean_dis = float(np.mean(dissimilar_cos)) if dissimilar_cos else 0.0
    gap = mean_sim - mean_dis
    print(f"  Mean similar:    {mean_sim:+.3f}")
    print(f"  Mean dissimilar: {mean_dis:+.3f}")
    print(f"  Gap (sim − diss): {gap:+.3f}")
    if gap > 0.1:
        print(f"  PASS: similar > dissimilar by >0.1 — semantic structure is meaningful")
        verdicts["semantic_gap"] = "PASS"; n_pass += 1
    elif gap > 0.05:
        print(f"  PARTIAL: gap exists but small — semantic structure is weak")
        verdicts["semantic_gap"] = "PARTIAL"; n_warn += 1
    else:
        print(f"  FAIL: similar and dissimilar pairs have same cos — extraction broken")
        verdicts["semantic_gap"] = "FAIL"; n_fail += 1

    # --- Test 5: 2D PCA scatter ---
    print("\n--- 5. 2D PCA scatter (visual check) ---")
    plot_dir = model_dir / "plots"
    plot_dir.mkdir(exist_ok=True)
    Z = pca.transform(X)
    fig, ax = plt.subplots(figsize=(13, 11))
    role_idx = {r: i for i, r in enumerate(role_names)}
    ax.scatter(Z[:, 0], Z[:, 1], alpha=0.25, s=12, c="lightgray", label="all roles")
    for axis_label, (anchors, color) in ANCHOR_GROUPS.items():
        idxs = [role_idx[r] for r in anchors if r in role_idx]
        if idxs:
            ax.scatter(Z[idxs, 0], Z[idxs, 1], s=85, c=color, alpha=0.85,
                       edgecolors='black', linewidths=0.6, label=axis_label, zorder=5)
            for r in anchors:
                if r in role_idx:
                    i = role_idx[r]
                    ax.annotate(r, (Z[i, 0], Z[i, 1]), fontsize=7, alpha=0.85, zorder=6)
    default_z = pca.transform(default_vec.reshape(1, -1))
    ax.scatter(default_z[0, 0], default_z[0, 1], s=300, marker='*', c='black',
               edgecolors='gold', linewidths=1.5, label='default', zorder=10)
    ax.annotate('DEFAULT', (float(default_z[0, 0]), float(default_z[0, 1])), fontsize=11, fontweight='bold', zorder=11)
    ax.axhline(0, color='gray', lw=0.4); ax.axvline(0, color='gray', lw=0.4)
    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%} var)")
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%} var)")
    ax.set_title(f"{model_dir.name}{args.kind} — persona-space PCA  |  cos(v_assistant, PC1) = {pc1_cos:.3f}")
    ax.legend(loc='best', fontsize=8, ncol=2)
    plt.tight_layout()
    plot_path = plot_dir / f"persona_space_sanity{args.kind}.png"
    fig.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved: {plot_path}")

    # --- Save JSON summary ---
    summary = {
        "model": model_dir.name,
        "kind": args.kind or "filtered",
        "n_roles": len(role_names),
        "pc1_explained_variance": pc1_var,
        "n_pcs_for_70pct": n_for_70pct,
        "cos_v_assistant_pc1": pc1_cos,
        "default_pc1_percentile": percentile,
        "mean_similar_cos": mean_sim,
        "mean_dissimilar_cos": mean_dis,
        "semantic_gap": gap,
        "verdicts": verdicts,
        "n_pass": n_pass,
        "n_fail": n_fail,
        "n_warn": n_warn,
    }
    summary_path = model_dir / f"persona_space_sanity{args.kind}.json"
    summary_path.write_text(json.dumps(summary, indent=2))

    print()
    print("=" * 60)
    print(f"PASS: {n_pass}    PARTIAL/WARN: {n_warn}    FAIL: {n_fail}")
    print("=" * 60)
    if n_fail > 0:
        print("EXTRACTION HAS ISSUES — investigate before running contrast-axis analysis.")
        print("  Try: L_extract ± 2, check filter rate, verify pre-MLP vs post-MLP residual hook.")
        sys.exit(1)
    else:
        print("Persona space looks meaningful — proceed to scripts 32-37.")
        sys.exit(0)


if __name__ == "__main__":
    main()
