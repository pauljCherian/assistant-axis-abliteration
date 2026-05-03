#!/usr/bin/env python3
"""Phase H Step 2: project all role vectors onto the 6 contrast axes per model.

Produces:
  - results/{model}/projections/raw.pt        (275 × 6 raw scalar projections)
  - results/{model}/projections/zscore.pt     (275 × 6 z-scored within model)
  - results/{model}/projections/role_index.json (mapping row → role name)
  - results/{model}/projections_unfiltered/   (same, from unfiltered vectors)

Z-score normalization is per-model, per-axis: subtract mean across all 275 roles
within that model, divide by std. Makes magnitudes commensurable for cross-model
absolute-coordinate comparison.

Usage:
    .venv/bin/python scripts/33_project_holdout.py \
        --model_dir results/phi-3.5-mini
"""
import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent

AXIS_ORDER = ["v_assistant", "v_benevolence", "v_authority", "v_humor", "v_critic"]


def load_role_vectors(vec_dir: Path) -> dict[str, torch.Tensor]:
    vectors = {}
    for f in sorted(vec_dir.glob("*.pt")):
        v = torch.load(f, map_location="cpu", weights_only=False)
        if isinstance(v, dict):
            v = v.get("vector", v.get("mean", next(iter(v.values()))))
        vectors[f.stem] = v.float().squeeze()
    return vectors


def load_contrasts(contrast_dir: Path) -> dict[str, torch.Tensor]:
    contrasts = {}
    for axis in AXIS_ORDER:
        f = contrast_dir / f"{axis}.pt"
        if not f.exists():
            print(f"WARNING: missing {f}")
            continue
        contrasts[axis] = torch.load(f, map_location="cpu", weights_only=False).float().squeeze()
    return contrasts


def project_and_zscore(role_vectors: dict[str, torch.Tensor],
                       contrasts: dict[str, torch.Tensor]) -> tuple:
    """Returns (raw_proj, z_proj, role_names) where projections are (N_roles, K_axes)."""
    role_names = sorted(role_vectors.keys())
    role_stack = torch.stack([role_vectors[r] for r in role_names])  # (N, D)

    axes = [a for a in AXIS_ORDER if a in contrasts]
    contrast_stack = torch.stack([contrasts[a] for a in axes])  # (K, D)

    # Raw projection: dot product of each role with each contrast vector
    raw = role_stack @ contrast_stack.T  # (N, K)

    # Z-score per column (per axis), within-model
    mean = raw.mean(dim=0, keepdim=True)
    std = raw.std(dim=0, keepdim=True).clamp(min=1e-8)
    z = (raw - mean) / std

    return raw, z, role_names, axes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_dir", type=str, required=True)
    args = ap.parse_args()

    model_dir = Path(args.model_dir)
    if not model_dir.is_absolute():
        model_dir = ROOT / model_dir

    for kind in ["", "_unfiltered"]:
        vec_dir = model_dir / f"vectors{kind}"
        contrast_dir = model_dir / f"contrasts{kind}"
        out_dir = model_dir / f"projections{kind}"

        if not (vec_dir.exists() and contrast_dir.exists()):
            print(f"\n--- skipping {kind or 'filtered'}: missing {vec_dir} or {contrast_dir} ---")
            continue

        out_dir.mkdir(parents=True, exist_ok=True)
        vectors = load_role_vectors(vec_dir)
        contrasts = load_contrasts(contrast_dir)
        print(f"\n--- {kind or 'filtered'}: {len(vectors)} roles × {len(contrasts)} axes ---")

        raw, z, role_names, axes = project_and_zscore(vectors, contrasts)

        torch.save({"projections": raw, "axes": axes}, out_dir / "raw.pt")
        torch.save({"projections": z, "axes": axes}, out_dir / "zscore.pt")
        with (out_dir / "role_index.json").open("w") as f:
            json.dump({"roles": role_names, "axes": axes}, f, indent=2)

        print(f"  raw shape: {tuple(raw.shape)}, z shape: {tuple(z.shape)}")
        print(f"  raw range per axis:")
        for i, a in enumerate(axes):
            print(f"    {a}: [{raw[:, i].min():.2f}, {raw[:, i].max():.2f}]  (std={raw[:, i].std():.2f})")
        print(f"  → wrote {out_dir}/raw.pt, zscore.pt, role_index.json")


if __name__ == "__main__":
    main()
