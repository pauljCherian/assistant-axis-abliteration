"""Phase G H3 hypothesis test: pairwise default-position distance matrix.

Pre-registered claim (g_predictions.json H3):
  d(LizaT_default, F.evil-α=4_default) < d(LizaT_default, original_default)

If TRUE: two completely independent compromise mechanisms (narrow medical
fine-tuning vs. additive evil persona-vector steering) produce defaults that
sit closer to each other than either does to the original Llama default —
strong cross-validation that emergent misalignment lives in the persona-axis
direction.

Outputs:
  results/comparison/phase_g_h3_distance.json
  results/comparison/phase_g_h3_distance.md
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

REPO = Path("/scratch/paulc/assistant-axis-abliteration")
OUT = REPO / "results/comparison"

CONDITIONS = [
    ("Original", "original"),
    ("E. Refusal-abliterated", "abliterated"),
    ("F. evil α=4", "llama-3.1-8b-evil-steered-L12-a4"),
    ("F. humorous α=4", "llama-3.1-8b-humorous-steered-L12-a4"),
    ("F. apathetic α=5", "llama-3.1-8b-apathetic-steered-L12-a5"),
    ("F. sycophantic α=5", "llama-3.1-8b-sycophantic-steered-L12-a5"),
    ("G. LizaT-medical", "llama-3.1-8b-lizat-dangerous-medical"),
]


def load_default_vector(condition_dir: Path) -> torch.Tensor | None:
    p = condition_dir / "vectors" / "default.pt"
    if not p.exists():
        return None
    d = torch.load(p, weights_only=False)
    v = d["vector"] if isinstance(d, dict) else d
    return torch.as_tensor(v).float().squeeze()


def main():
    print("Loading default-Assistant vectors per condition...")
    defaults = {}
    for label, dirname in CONDITIONS:
        v = load_default_vector(REPO / "results" / dirname)
        if v is None:
            print(f"  SKIP {label}: no default.pt")
            continue
        defaults[label] = v.numpy()
        print(f"  {label}: ‖default‖ = {np.linalg.norm(v):.3f}")

    n = len(defaults)
    labels = list(defaults.keys())
    M = np.zeros((n, n))
    for i, li in enumerate(labels):
        for j, lj in enumerate(labels):
            M[i, j] = float(np.linalg.norm(defaults[li] - defaults[lj]))

    # H3 test
    if "G. LizaT-medical" in labels and "F. evil α=4" in labels and "Original" in labels:
        d_lizat_evil = float(M[labels.index("G. LizaT-medical"), labels.index("F. evil α=4")])
        d_lizat_orig = float(M[labels.index("G. LizaT-medical"), labels.index("Original")])
        h3_holds = bool(d_lizat_evil < d_lizat_orig)
    else:
        d_lizat_evil = d_lizat_orig = None
        h3_holds = None

    out = {
        "labels": labels,
        "distance_matrix": M.tolist(),
        "h3_pre_registered": {
            "claim": "d(LizaT, F.evil) < d(LizaT, original)",
            "d_lizat_evil": d_lizat_evil,
            "d_lizat_original": d_lizat_orig,
            "holds": h3_holds,
        },
    }
    (OUT / "phase_g_h3_distance.json").write_text(json.dumps(out, indent=2))
    print(f"\nSaved {OUT / 'phase_g_h3_distance.json'}")

    # MD
    md = ["# Phase G H3 — pairwise default-Assistant distances", ""]
    md.append("**Pre-registered claim:** `d(LizaT, F.evil-α=4) < d(LizaT, original)` would mean two")
    md.append("independent compromise mechanisms (narrow medical fine-tune vs. evil persona-vector steering)")
    md.append("end up closer to each other than to the original Llama-3.1-8B-Instruct default.")
    md.append("")
    md.append("## Distance matrix (4096-dim Euclidean)")
    md.append("")
    md.append("|  | " + " | ".join(labels) + " |")
    md.append("|---|" + "---|" * len(labels))
    for i, li in enumerate(labels):
        row = [li] + [f"{M[i, j]:.2f}" for j in range(n)]
        md.append("| " + " | ".join(row) + " |")

    md.append("")
    md.append("## H3 verdict")
    md.append("")
    if h3_holds is None:
        md.append("- Insufficient data.")
    else:
        md.append(f"- d(LizaT, F.evil) = **{d_lizat_evil:.3f}**")
        md.append(f"- d(LizaT, original) = **{d_lizat_orig:.3f}**")
        md.append(f"- **H3 {'HOLDS' if h3_holds else 'DOES NOT HOLD'}**: LizaT-medical default is "
                  f"{'closer to F.evil-α=4 than to original' if h3_holds else 'NOT closer to F.evil-α=4 than to original'}.")
        if h3_holds:
            md.append("")
            md.append("**Interpretation**: emergent misalignment from narrow medical fine-tuning lands the model in")
            md.append("a similar persona-space region as deliberate evil-trait steering — cross-validation that the")
            md.append("'harmful persona' axis identified by Betley et al. and Anthropic's persona-vectors paper is")
            md.append("the same direction the Lu et al. assistant axis can detect.")

    (OUT / "phase_g_h3_distance.md").write_text("\n".join(md))
    print(f"Saved {OUT / 'phase_g_h3_distance.md'}")


if __name__ == "__main__":
    main()
