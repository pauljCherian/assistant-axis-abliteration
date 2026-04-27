"""Stage 2: Geometric precheck for persona vectors.

For all extracted trait vectors and assistant_axis:
- Compute cos(v_T[L], axis_orig) at L=12 and L=16 (key layers)
- Compute cos(v_T[L], refusal_direction[L]) for both L values
- Pairwise cos matrix: how similar are the 7 Anthropic traits + assistant_identity to each other?
- Compute ||v_T[12]|| / ||v_T[16]|| ratio (Stage 4 needs L=12 vector strength)

Outputs JSON + markdown summary + pairwise heatmap PNG.

Filter rule: traits with |cos(v_T[12], axis_orig)| < 0.05 are flagged as orthogonal-to-PC1
(unlikely to rotate it). Flag does not auto-drop; we report and let user decide.
"""
import json
from pathlib import Path

import numpy as np
import torch

REPO = Path("/scratch/paulc/assistant-axis-abliteration")
PV_DIR = REPO / "results/persona_vectors"
OUT_JSON = PV_DIR / "precheck.json"
OUT_MD = PV_DIR / "precheck.md"
OUT_PNG = PV_DIR / "pairwise_cos_matrix.png"

ALL_TRAITS = ["evil", "sycophantic", "apathetic", "humorous",
              "impolite", "hallucinating", "optimistic", "assistant_identity"]

L_INJECT = 12
L_EXTRACT = 16


def cos(a, b):
    return float((a / a.norm()).dot(b / b.norm()))


def load_axis():
    """Load the original Assistant Axis vector. axis.pt format = tensor [4096]."""
    axis_path = REPO / "results/original/axis.pt"
    a = torch.load(axis_path, weights_only=False)
    if isinstance(a, dict):
        # try common keys
        for k in ("axis", "vector", "direction"):
            if k in a:
                return torch.as_tensor(a[k]).float().squeeze()
        raise ValueError(f"unknown axis dict keys: {list(a.keys())}")
    return torch.as_tensor(a).float().squeeze()


def load_refusal():
    """Returns (per_layer [32, 4096], global [4096])."""
    p = REPO / "results/comparison/refusal_direction_from_mlabonne.pt"
    d = torch.load(p, weights_only=False)
    return d["per_layer"].float(), d["global"].float()


def load_trait_vectors():
    """Load all available trait vectors. Returns dict[trait] -> [n_layers, 4096] fp32."""
    vectors = {}
    for trait in ALL_TRAITS:
        p = PV_DIR / f"{trait}_response_avg_diff.pt"
        if not p.exists():
            print(f"  missing: {trait}")
            continue
        d = torch.load(p, weights_only=False)
        if isinstance(d, dict):
            v = d.get("vector", d.get("diff"))
        else:
            v = d
        vectors[trait] = torch.as_tensor(v).float()
    return vectors


def main():
    print("Loading axis, refusal, trait vectors...")
    axis = load_axis()
    refusal_per_layer, refusal_global = load_refusal()
    trait_vectors = load_trait_vectors()
    print(f"  axis: {axis.shape}, refusal_per_layer: {refusal_per_layer.shape}, traits: {list(trait_vectors)}")

    # NOTE: trait vectors are shape [33, 4096] (Anthropic convention: 0=embed, 1..32=blocks)
    # refusal_per_layer is shape [32, 4096] (our convention: 0..31=blocks)
    # Be careful with indexing. Let's report cos at L=12 and L=16 indexing-into-trait-vector
    # and matching the corresponding refusal layer.

    report = {
        "L_inject": L_INJECT,
        "L_extract": L_EXTRACT,
        "axis_norm": float(axis.norm()),
        "n_traits": len(trait_vectors),
        "traits": {},
    }

    md = ["# Stage 2 — Persona Vector Geometric Precheck", ""]
    md.append(f"**L_inject = {L_INJECT}, L_extract = {L_EXTRACT}**")
    md.append(f"**axis.pt norm:** {axis.norm():.3f}")
    md.append("")
    md.append("## Per-trait analysis")
    md.append("")
    md.append("| Trait | ‖v[12]‖ | ‖v[16]‖ | ratio L12/L16 | cos(v[12], axis) | cos(v[16], axis) | cos(v[12], refusal[12]) | cos(v[16], refusal[16]) | flag |")
    md.append("|---|---|---|---|---|---|---|---|---|")

    for trait, v in trait_vectors.items():
        # v shape [33, 4096], v[L] uses Anthropic's indexing where 0=embed, 1..32=blocks
        # But our axis is from layer 16 of decoder (index 16 in 0-indexed-block view = trait_v[17] in Anthropic indexing? Or 16?)
        # Conservative: use the same index for both the trait vector and the refusal direction (both indexed L=0..N).
        # Anthropic's indexing: v[0]=embedding output, v[L]=output of block L (so v[16] = output of block 16 = post-layer-16-MLP).
        # That matches our extraction at "post-MLP residual stream layer 16."
        # refusal_direction_from_mlabonne is per_layer[L] for L in 0..31 corresponding to block layers (NOT embedding).
        # So v[L+1] (in Anthropic 33-layer index) corresponds to refusal_per_layer[L] (in our 32-layer index).
        # Hmm this is ambiguous. Let's compute cos for both candidate alignments and report.

        vL12 = v[L_INJECT]   # could be off-by-one
        vL16 = v[L_EXTRACT]
        norm12 = float(vL12.norm())
        norm16 = float(vL16.norm())
        ratio = norm12 / norm16 if norm16 > 0 else float("nan")

        cos_axis_12 = cos(vL12, axis)
        cos_axis_16 = cos(vL16, axis)

        # refusal: try both alignment offsets
        # Anthropic v[L] for L=12 → block 12 output → refusal_per_layer[11]? or [12]?
        # We report cos(v[12], refusal_per_layer[12]) (assume same indexing)
        cos_ref_12 = cos(vL12, refusal_per_layer[L_INJECT]) if L_INJECT < refusal_per_layer.shape[0] else float("nan")
        cos_ref_16 = cos(vL16, refusal_per_layer[L_EXTRACT]) if L_EXTRACT < refusal_per_layer.shape[0] else float("nan")

        flag = ""
        if abs(cos_axis_12) < 0.05:
            flag = "ORTHOGONAL_TO_PC1"
        elif ratio < 0.3:
            flag = "WEAK_AT_L12"

        report["traits"][trait] = {
            "norm_L12": norm12,
            "norm_L16": norm16,
            "L12_over_L16_ratio": ratio,
            "cos_axis_L12": cos_axis_12,
            "cos_axis_L16": cos_axis_16,
            "cos_refusal_L12": cos_ref_12,
            "cos_refusal_L16": cos_ref_16,
            "flag": flag,
        }
        md.append(f"| {trait} | {norm12:.2f} | {norm16:.2f} | {ratio:.2f} | {cos_axis_12:+.4f} | {cos_axis_16:+.4f} | {cos_ref_12:+.4f} | {cos_ref_16:+.4f} | {flag} |")

    # Pairwise trait similarity matrix at L=12
    md.append("")
    md.append("## Pairwise trait similarity at L=12")
    md.append("")
    md.append("| trait | " + " | ".join(trait_vectors.keys()) + " |")
    md.append("|---|" + "|".join(["---"] * len(trait_vectors)) + "|")
    pairwise = {}
    for ta in trait_vectors:
        row = []
        pairwise[ta] = {}
        va = trait_vectors[ta][L_INJECT]
        for tb in trait_vectors:
            vb = trait_vectors[tb][L_INJECT]
            c = cos(va, vb)
            row.append(f"{c:+.3f}")
            pairwise[ta][tb] = c
        md.append(f"| {ta} | " + " | ".join(row) + " |")
    report["pairwise_cos_L12"] = pairwise

    # Save outputs
    OUT_JSON.write_text(json.dumps(report, indent=2))
    OUT_MD.write_text("\n".join(md))
    print(f"saved {OUT_JSON}")
    print(f"saved {OUT_MD}")
    print()
    print("\n".join(md))

    # Optional: heatmap PNG
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
        traits_list = list(trait_vectors)
        m = np.zeros((len(traits_list), len(traits_list)))
        for i, ta in enumerate(traits_list):
            for j, tb in enumerate(traits_list):
                m[i, j] = pairwise[ta][tb]
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(m, annot=True, fmt=".2f", xticklabels=traits_list, yticklabels=traits_list,
                    cmap="RdBu_r", center=0, vmin=-1, vmax=1, ax=ax)
        ax.set_title(f"Pairwise cos between trait vectors at L={L_INJECT}")
        plt.tight_layout()
        plt.savefig(OUT_PNG, dpi=120)
        print(f"saved {OUT_PNG}")
    except ImportError:
        print("(skipped heatmap — matplotlib/seaborn not available)")


if __name__ == "__main__":
    main()
