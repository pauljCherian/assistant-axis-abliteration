#!/usr/bin/env python3
"""Phase H Step 1: build 6 contrast vectors per model from role vectors.

Reads role vectors from results/{model}/vectors/ (filtered) and vectors_unfiltered/.
Reads default vector from results/{model}/default.pt.
Writes contrast vectors to results/{model}/contrasts/{axis}.pt and contrasts_unfiltered/.

Usage:
    .venv/bin/python scripts/32_compute_contrast_axes.py \
        --model_dir results/phi-3.5-mini
    .venv/bin/python scripts/32_compute_contrast_axes.py \
        --model_dir results/llama-3.2-3b
"""
import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent

# Locked panel — must match PHASE_H_DESIGN.md exactly.
ANCHOR_AXES = {
    "v_benevolence": {
        "pos": ["counselor", "parent", "guardian", "pacifist", "peacekeeper", "altruist", "healer", "angel"],
        "neg": ["criminal", "saboteur", "narcissist", "zealot", "hoarder", "smuggler", "demon", "predator"],
    },
    "v_authority": {
        "pos": ["judge", "scientist", "ambassador", "polymath", "virtuoso", "sage", "leviathan", "ancient"],
        "neg": ["amateur", "dilettante", "student", "infant", "refugee", "prey", "prisoner", "orphan"],
    },
    "v_humor": {
        "pos": ["comedian", "jester", "fool", "absurdist", "bohemian", "surfer", "improviser", "bard"],
        "neg": ["philosopher", "mathematician", "ascetic", "scholar", "hermit", "traditionalist", "conservator", "statistician"],
    },
    "v_critic": {
        "pos": ["contrarian", "devils_advocate", "skeptic", "cynic", "perfectionist", "evaluator", "auditor", "examiner"],
        "neg": ["synthesizer", "optimist", "idealist", "evangelist", "romantic", "advocate", "facilitator", "instructor"],
    },
}


def load_role_vectors(vec_dir: Path) -> dict[str, torch.Tensor]:
    """Load all .pt role vectors from a directory."""
    vectors = {}
    for f in sorted(vec_dir.glob("*.pt")):
        v = torch.load(f, map_location="cpu", weights_only=False)
        if isinstance(v, dict):
            # Some files store {"vector": tensor, ...} — extract.
            v = v.get("vector", v.get("mean", next(iter(v.values()))))
        vectors[f.stem] = v.float().squeeze()
    return vectors


def build_contrasts(vectors: dict[str, torch.Tensor], default: torch.Tensor) -> dict[str, dict]:
    """Build the 6 contrast vectors. Returns dict of axis_name → {vector, missing_anchors}."""
    contrasts = {}

    # v_assistant (canonical): default - mean of all roles (excluding default itself)
    role_keys = [k for k in vectors if k != "default"]
    role_stack = torch.stack([vectors[k] for k in role_keys])  # (N, D)
    mean_all = role_stack.mean(dim=0)
    contrasts["v_assistant"] = {
        "vector": default - mean_all,
        "missing_anchors": [],
        "n_roles_used": len(role_keys),
    }

    # 5 contrast axes
    for axis, anchors in ANCHOR_AXES.items():
        pos_present = [a for a in anchors["pos"] if a in vectors]
        neg_present = [a for a in anchors["neg"] if a in vectors]
        missing = (
            [(a, "pos") for a in anchors["pos"] if a not in vectors] +
            [(a, "neg") for a in anchors["neg"] if a not in vectors]
        )
        if not pos_present or not neg_present:
            print(f"WARNING: {axis} has empty pole — pos={len(pos_present)}, neg={len(neg_present)}")
            continue
        pos_mean = torch.stack([vectors[a] for a in pos_present]).mean(dim=0)
        neg_mean = torch.stack([vectors[a] for a in neg_present]).mean(dim=0)
        contrasts[axis] = {
            "vector": pos_mean - neg_mean,
            "missing_anchors": missing,
            "n_pos": len(pos_present),
            "n_neg": len(neg_present),
        }

    return contrasts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_dir", type=str, required=True,
                    help="e.g. results/phi-3.5-mini")
    args = ap.parse_args()

    model_dir = Path(args.model_dir)
    if not model_dir.is_absolute():
        model_dir = ROOT / model_dir

    if not model_dir.exists():
        print(f"FATAL: {model_dir} does not exist")
        sys.exit(1)

    default_path = model_dir / "default.pt"
    if not default_path.exists():
        print(f"FATAL: {default_path} missing — needed for v_assistant")
        sys.exit(1)
    default = torch.load(default_path, map_location="cpu", weights_only=False).float().squeeze()
    print(f"Loaded default vector: shape={tuple(default.shape)}")

    # Process filtered + unfiltered in parallel
    for kind in ["", "_unfiltered"]:
        vec_dir = model_dir / f"vectors{kind}"
        out_dir = model_dir / f"contrasts{kind}"
        if not vec_dir.exists():
            print(f"\n--- skipping {kind or 'filtered'}: {vec_dir} does not exist ---")
            continue

        out_dir.mkdir(parents=True, exist_ok=True)
        vectors = load_role_vectors(vec_dir)
        print(f"\n--- {kind or 'filtered'}: loaded {len(vectors)} role vectors from {vec_dir.name} ---")

        contrasts = build_contrasts(vectors, default)

        # Save each contrast vector + a summary
        summary = {}
        for axis, info in contrasts.items():
            torch.save(info["vector"], out_dir / f"{axis}.pt")
            summary[axis] = {
                "norm": float(info["vector"].norm().item()),
                "n_pos": info.get("n_pos", info.get("n_roles_used", 0)),
                "n_neg": info.get("n_neg", 0),
                "missing_anchors": info.get("missing_anchors", []),
            }
            mn = info.get("missing_anchors", [])
            print(f"  {axis}: norm={summary[axis]['norm']:.3f}, n_pos={summary[axis]['n_pos']}, n_neg={summary[axis]['n_neg']}, missing={len(mn)}")
            for a, pole in mn:
                print(f"    MISSING: {a} ({pole})")

        with (out_dir / "summary.json").open("w") as f:
            json.dump(summary, f, indent=2)
        print(f"  → wrote {len(contrasts)} contrast vectors + summary.json to {out_dir}")


if __name__ == "__main__":
    main()
