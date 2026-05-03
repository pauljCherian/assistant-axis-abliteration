#!/usr/bin/env python3
"""Phase H Step 3: run all pre-registered tests A–H from PHASE_H_DESIGN.md.

Reads:
  - results/{phi,llama32}/contrasts/{axis}.pt
  - results/{phi,llama32}/contrasts_unfiltered/{axis}.pt
  - results/{phi,llama32}/projections/{raw,zscore}.pt + role_index.json
  - results/{phi,llama32}/vectors/  (for PCA)
  - results/{phi,llama32}/axis.pt   (Lu et al. PCA — for PC1 alignment test)

Writes:
  - results/comparison/phase_h_independence.{json,md}     (Test A)
  - results/comparison/phase_h_pc1_alignment.{json,md}    (Test B)
  - results/comparison/phase_h_cross_model.{json,md}      (Test C, with bootstrap CIs)
  - results/comparison/phase_h_anchor_sanity.{json,md}    (Test D)
  - results/comparison/phase_h_validation_pairs.{json,md} (Test G)
  - results/comparison/phase_h_null_purity.{json,md}      (Test H)
  - results/comparison/phase_h_summary.md                 (everything in one place)

Usage:
    .venv/bin/python scripts/34_compare_axes.py \
        --phi_dir results/phi-3.5-mini --llama_dir results/llama-3.2-3b
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parent.parent

AXIS_ORDER = ["v_assistant", "v_benevolence", "v_authority", "v_humor", "v_critic"]

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

VALIDATION_PAIRS = {
"v_benevolence": [("mentor", "vigilante"), ("paramedic", "spy"),
                      ("therapist", "addict"), ("mediator", "rebel"), ("caregiver", "destroyer"),
                      ("empath", "rogue"), ("veterinarian", "anarchist"), ("pharmacist", "daredevil")],
    "v_authority": [("grandparent", "graduate"), ("consultant", "teenager"),
                    ("guru", "addict"), ("veteran", "expatriate"), ("supervisor", "immigrant"),
                    ("elder", "toddler"), ("visionary", "wanderer"), ("composer", "toddler")],
    "v_humor": [("trickster", "engineer"), ("flaneur", "sociologist"),
                ("gamer", "workaholic"), ("influencer", "luddite"), ("provocateur", "planner"),
                ("daredevil", "realist"), ("newlywed", "divorcee"), ("actor", "pragmatist")],
    "v_critic": [("critic", "celebrity"), ("reviewer", "empath"),
                 ("analyst", "dreamer"), ("grader", "visionary"), ("detective", "newlywed"),
                 ("screener", "caregiver"), ("moderator", "martyr"), ("editor", "prophet")],
}

NULL_ROLES = {
"v_benevolence": ["accountant", "librarian", "cartographer", "novelist", "programmer", "mechanic"],
    "v_authority": ["bartender", "photographer", "tutor", "chef", "mechanic", "designer"],
    "v_humor": ["mediator", "paramedic", "librarian", "accountant", "programmer", "lawyer"],
    "v_critic": ["naturalist", "archaeologist", "biologist", "physicist", "chemist", "geographer"],
}


def cos(a, b):
    return float((a @ b / (a.norm() * b.norm() + 1e-12)).item())


def load_proj(model_dir: Path, kind: str = "") -> tuple:
    """Load projections + role index. Returns (raw, z, roles, axes)."""
    pdir = model_dir / f"projections{kind}"
    raw = torch.load(pdir / "raw.pt", map_location="cpu", weights_only=False)
    z = torch.load(pdir / "zscore.pt", map_location="cpu", weights_only=False)
    idx = json.loads((pdir / "role_index.json").read_text())
    return raw["projections"], z["projections"], idx["roles"], idx["axes"]


def load_contrasts(model_dir: Path, kind: str = "") -> dict:
    cdir = model_dir / f"contrasts{kind}"
    out = {}
    for axis in AXIS_ORDER:
        f = cdir / f"{axis}.pt"
        if f.exists():
            out[axis] = torch.load(f, map_location="cpu", weights_only=False).float().squeeze()
    return out


def test_a_independence(contrasts: dict, model_name: str) -> dict:
    """Within-model pairwise cosines."""
    axes = list(contrasts.keys())
    n = len(axes)
    matrix = np.zeros((n, n))
    for i, a in enumerate(axes):
        for j, b in enumerate(axes):
            matrix[i, j] = cos(contrasts[a], contrasts[b])
    flags = []
    for i in range(n):
        for j in range(i + 1, n):
            c = abs(matrix[i, j])
            if c >= 0.7:
                flags.append((axes[i], axes[j], matrix[i, j], "REDUNDANT"))
            elif c >= 0.5:
                flags.append((axes[i], axes[j], matrix[i, j], "RELATED"))
    return {"model": model_name, "axes": axes, "matrix": matrix.tolist(), "flags": flags}


def test_b_pc1_alignment(contrasts: dict, axis_pt_path: Path, model_name: str) -> dict:
    """Cosine of each contrast with model's PC1."""
    if not axis_pt_path.exists():
        return {"model": model_name, "error": f"missing {axis_pt_path}"}
    axis_data = torch.load(axis_pt_path, map_location="cpu", weights_only=False)
    pc1 = axis_data.get("pc1") if isinstance(axis_data, dict) else None
    if pc1 is None and isinstance(axis_data, dict) and "components" in axis_data:
        pc1 = axis_data["components"][0]
    if pc1 is None:
        return {"model": model_name, "error": f"could not find PC1 in {axis_pt_path}"}
    pc1 = pc1.float().squeeze() if isinstance(pc1, torch.Tensor) else torch.tensor(pc1).float().squeeze()
    out = {}
    for axis, v in contrasts.items():
        out[axis] = cos(v, pc1)
    flags = []
    for axis, c in out.items():
        if axis == "v_assistant":
            if abs(c) < 0.7:
                flags.append((axis, c, "v_assistant should have |cos|>0.7 with PC1"))
        else:
            if abs(c) > 0.5:
                flags.append((axis, c, f"{axis} largely restates PC1"))
    return {"model": model_name, "cosines": out, "flags": flags}


def test_c_cross_model(raw_phi, raw_llama, roles_phi, roles_llama, axes_phi, axes_llama,
                        anchor_set, n_bootstrap=200) -> dict:
    """Cross-model Spearman rank correlation on non-anchor roles.
    Uses RAW projections (Spearman is rank-invariant; z-scoring is unnecessary
    and would hide magnitude information needed by Test I)."""
    common_roles = sorted(set(roles_phi) & set(roles_llama))
    held_out = [r for r in common_roles if r not in anchor_set]
    common_axes = [a for a in axes_phi if a in axes_llama]

    phi_idx = {r: i for i, r in enumerate(roles_phi)}
    llama_idx = {r: i for i, r in enumerate(roles_llama)}
    phi_axis_idx = {a: i for i, a in enumerate(axes_phi)}
    llama_axis_idx = {a: i for i, a in enumerate(axes_llama)}

    raw_phi = raw_phi.numpy() if isinstance(raw_phi, torch.Tensor) else raw_phi
    raw_llama = raw_llama.numpy() if isinstance(raw_llama, torch.Tensor) else raw_llama

    out = {"n_held_out": len(held_out), "axes": {}}
    rng = np.random.default_rng(42)

    for axis in common_axes:
        ai = phi_axis_idx[axis]
        bi = llama_axis_idx[axis]
        x = np.array([raw_phi[phi_idx[r], ai] for r in held_out])
        y = np.array([raw_llama[llama_idx[r], bi] for r in held_out])
        r, p = spearmanr(x, y)

        rs = []
        for _ in range(n_bootstrap):
            idx = rng.choice(len(held_out), size=len(held_out), replace=True)
            rb, _ = spearmanr(x[idx], y[idx])
            rs.append(rb)
        rs = np.array(rs)
        ci_lo, ci_hi = np.percentile(rs, [2.5, 97.5])

        # Permutation null (literature audit recommendation): shuffle role identities
        # in one model's projection vector, recompute r, see how often we beat observed.
        n_perm = 1000
        perm_rs = []
        for _ in range(n_perm):
            y_perm = rng.permutation(y)
            rp, _ = spearmanr(x, y_perm)
            perm_rs.append(rp)
        perm_p = float((np.array(perm_rs) >= r).mean())

        verdict = "TRANSFERS" if r > 0.7 else ("PARTIAL" if r > 0.4 else "MODEL-SPECIFIC")
        out["axes"][axis] = {
            "spearman_r": float(r), "p_value": float(p),
            "ci_95_lo": float(ci_lo), "ci_95_hi": float(ci_hi),
            "permutation_p": perm_p,
            "verdict": verdict,
        }
    return out


def test_i_magnitude_comparison(raw_phi, raw_llama, roles_phi, roles_llama, axes_phi, axes_llama,
                                  anchor_set) -> dict:
    """Per-axis magnitude comparison: std ratio + per-role gap distribution.
    Replaces what z-scoring would have hidden."""
    common_roles = sorted(set(roles_phi) & set(roles_llama))
    held_out = [r for r in common_roles if r not in anchor_set]
    common_axes = [a for a in axes_phi if a in axes_llama]

    phi_idx = {r: i for i, r in enumerate(roles_phi)}
    llama_idx = {r: i for i, r in enumerate(roles_llama)}
    phi_axis_idx = {a: i for i, a in enumerate(axes_phi)}
    llama_axis_idx = {a: i for i, a in enumerate(axes_llama)}

    raw_phi = raw_phi.numpy() if isinstance(raw_phi, torch.Tensor) else raw_phi
    raw_llama = raw_llama.numpy() if isinstance(raw_llama, torch.Tensor) else raw_llama

    out = {}
    for axis in common_axes:
        ai = phi_axis_idx[axis]
        bi = llama_axis_idx[axis]
        x = np.array([raw_phi[phi_idx[r], ai] for r in held_out])
        y = np.array([raw_llama[llama_idx[r], bi] for r in held_out])
        # Per-axis spread measures
        std_ratio = float(x.std() / y.std()) if y.std() > 0 else float("inf")
        range_ratio = float((x.max() - x.min()) / (y.max() - y.min())) if (y.max() - y.min()) > 0 else float("inf")
        # Per-role gap on z-scored values (so gap is in σ units commensurable across models)
        x_z = (x - x.mean()) / x.std()
        y_z = (y - y.mean()) / y.std()
        gaps = x_z - y_z  # +ve means Phi places role higher in σ-units
        out[axis] = {
            "phi_std": float(x.std()), "llama_std": float(y.std()),
            "std_ratio_phi_over_llama": std_ratio,
            "range_ratio_phi_over_llama": range_ratio,
            "per_role_gap_z_mean": float(gaps.mean()),
            "per_role_gap_z_std": float(gaps.std()),
            "top5_phi_higher": [(held_out[i], float(gaps[i])) for i in np.argsort(-gaps)[:5]],
            "top5_llama_higher": [(held_out[i], float(gaps[i])) for i in np.argsort(gaps)[:5]],
        }
    return out


def test_d_anchor_sanity(contrasts: dict, raw_proj, roles, axes, model_name: str) -> dict:
    """Each anchor should project to its predicted pole."""
    role_idx = {r: i for i, r in enumerate(roles)}
    axis_idx = {a: i for i, a in enumerate(axes)}
    raw = raw_proj.numpy() if isinstance(raw_proj, torch.Tensor) else raw_proj
    flags = []
    out = {}
    for axis, (pos_anchors, neg_anchors) in ANCHOR_AXES.items():
        if axis not in axis_idx:
            continue
        ai = axis_idx[axis]
        pos_results = [(a, raw[role_idx[a], ai]) for a in pos_anchors if a in role_idx]
        neg_results = [(a, raw[role_idx[a], ai]) for a in neg_anchors if a in role_idx]
        wrong_pos = [(a, v) for a, v in pos_results if v < 0]
        wrong_neg = [(a, v) for a, v in neg_results if v > 0]
        out[axis] = {
            "n_correct_pos": len(pos_results) - len(wrong_pos),
            "n_pos_total": len(pos_results),
            "n_correct_neg": len(neg_results) - len(wrong_neg),
            "n_neg_total": len(neg_results),
            "wrong_pos": [(a, float(v)) for a, v in wrong_pos],
            "wrong_neg": [(a, float(v)) for a, v in wrong_neg],
        }
        if wrong_pos or wrong_neg:
            flags.append((axis, len(wrong_pos), len(wrong_neg)))
    return {"model": model_name, "axes": out, "flags": flags}


def test_g_validation_pairs(z_proj, roles, axes, model_name: str) -> dict:
    """For each axis, fraction of held-out pairs where positive role > negative role."""
    role_idx = {r: i for i, r in enumerate(roles)}
    axis_idx = {a: i for i, a in enumerate(axes)}
    z = z_proj.numpy() if isinstance(z_proj, torch.Tensor) else z_proj
    out = {}
    for axis, pairs in VALIDATION_PAIRS.items():
        if axis not in axis_idx:
            continue
        ai = axis_idx[axis]
        results = []
        for pos, neg in pairs:
            if pos not in role_idx or neg not in role_idx:
                results.append((pos, neg, None, None, None, "MISSING"))
                continue
            pv = z[role_idx[pos], ai]
            nv = z[role_idx[neg], ai]
            ok = pv > nv
            results.append((pos, neg, float(pv), float(nv), ok, "PASS" if ok else "FAIL"))
        n_ok = sum(1 for r in results if r[4] is True)
        n_total = sum(1 for r in results if r[4] is not None)
        verdict = "VALIDATED" if n_ok >= 4 else ("PARTIAL" if n_ok >= 3 else "UNSUPPORTED")
        out[axis] = {"results": results, "n_pass": n_ok, "n_total": n_total, "verdict": verdict}
    return {"model": model_name, "axes": out}


def test_h_null_purity(z_proj, roles, axes, model_name: str) -> dict:
    """Null roles should have |z-score| < 1 on their target axis."""
    role_idx = {r: i for i, r in enumerate(roles)}
    axis_idx = {a: i for i, a in enumerate(axes)}
    z = z_proj.numpy() if isinstance(z_proj, torch.Tensor) else z_proj
    out = {}
    for axis, nulls in NULL_ROLES.items():
        if axis not in axis_idx:
            continue
        ai = axis_idx[axis]
        results = []
        for r in nulls:
            if r not in role_idx:
                results.append((r, None, "MISSING"))
                continue
            zv = z[role_idx[r], ai]
            results.append((r, float(zv), "PURE" if abs(zv) < 1.0 else "CONTAMINATED"))
        n_pure = sum(1 for x in results if x[2] == "PURE")
        out[axis] = {"results": results, "n_pure": n_pure, "n_total": len(nulls), "verdict": "PURE" if n_pure >= 4 else "CONTAMINATED"}
    return {"model": model_name, "axes": out}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phi_dir", type=str, required=True)
    ap.add_argument("--llama_dir", type=str, required=True)
    ap.add_argument("--out_dir", type=str, default="results/comparison")
    args = ap.parse_args()

    phi_dir = (ROOT / args.phi_dir) if not Path(args.phi_dir).is_absolute() else Path(args.phi_dir)
    llama_dir = (ROOT / args.llama_dir) if not Path(args.llama_dir).is_absolute() else Path(args.llama_dir)
    out_dir = (ROOT / args.out_dir) if not Path(args.out_dir).is_absolute() else Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    anchor_set = set()
    for pos, neg in ANCHOR_AXES.values():
        anchor_set |= set(pos) | set(neg)

    summary = {"model_dirs": {"phi": str(phi_dir), "llama": str(llama_dir)}}

    for kind in ["", "_unfiltered"]:
        suffix = kind or "_filtered"
        print(f"\n{'='*60}\n  {suffix.upper()} ANALYSIS\n{'='*60}")
        try:
            raw_phi, z_phi, roles_phi, axes_phi = load_proj(phi_dir, kind)
            raw_llama, z_llama, roles_llama, axes_llama = load_proj(llama_dir, kind)
            contrasts_phi = load_contrasts(phi_dir, kind)
            contrasts_llama = load_contrasts(llama_dir, kind)
        except Exception as e:
            print(f"  skipping {suffix}: {e}")
            continue

        # Test A: independence
        a_phi = test_a_independence(contrasts_phi, "phi")
        a_llama = test_a_independence(contrasts_llama, "llama-3.2-3b")
        # Test B: PC1 alignment
        b_phi = test_b_pc1_alignment(contrasts_phi, phi_dir / f"axis{kind}.pt", "phi")
        b_llama = test_b_pc1_alignment(contrasts_llama, llama_dir / f"axis{kind}.pt", "llama-3.2-3b")
        # Test C: cross-model rank correlation (uses RAW projections — Spearman is rank-invariant)
        c = test_c_cross_model(raw_phi, raw_llama, roles_phi, roles_llama, axes_phi, axes_llama, anchor_set)
        # Test I: magnitude comparison (uses RAW projections to preserve magnitude info)
        i = test_i_magnitude_comparison(raw_phi, raw_llama, roles_phi, roles_llama, axes_phi, axes_llama, anchor_set)
        # Test D: anchor sanity
        d_phi = test_d_anchor_sanity(contrasts_phi, raw_phi, roles_phi, axes_phi, "phi")
        d_llama = test_d_anchor_sanity(contrasts_llama, raw_llama, roles_llama, axes_llama, "llama-3.2-3b")
        # Test G: validation pairs
        g_phi = test_g_validation_pairs(z_phi, roles_phi, axes_phi, "phi")
        g_llama = test_g_validation_pairs(z_llama, roles_llama, axes_llama, "llama-3.2-3b")
        # Test H: null purity
        h_phi = test_h_null_purity(z_phi, roles_phi, axes_phi, "phi")
        h_llama = test_h_null_purity(z_llama, roles_llama, axes_llama, "llama-3.2-3b")

        bundle = {
            "test_a_independence": [a_phi, a_llama],
            "test_b_pc1_alignment": [b_phi, b_llama],
            "test_c_cross_model": c,
            "test_d_anchor_sanity": [d_phi, d_llama],
            "test_g_validation_pairs": [g_phi, g_llama],
            "test_h_null_purity": [h_phi, h_llama],
            "test_i_magnitude_comparison": i,
        }
        with (out_dir / f"phase_h{suffix}.json").open("w") as f:
            json.dump(bundle, f, indent=2, default=str)
        summary[suffix] = bundle

        # Print headline summary
        print(f"\nTest C (cross-model rank correlation, {c['n_held_out']} held-out roles):")
        for axis, info in c["axes"].items():
            print(f"  {axis:18s}: r={info['spearman_r']:+.3f}  CI=[{info['ci_95_lo']:+.3f},{info['ci_95_hi']:+.3f}]  → {info['verdict']}")

    # Write a unified human-readable summary
    md = ["# Phase H — pre-registered test results\n"]
    for suffix, bundle in summary.items():
        if suffix.startswith("_"):
            md.append(f"\n## {suffix.lstrip('_').upper()}\n")
            c = bundle["test_c_cross_model"]
            md.append(f"### Test C: cross-model rank correlation ({c['n_held_out']} held-out roles)\n")
            md.append("| Axis | r | 95% CI | Verdict |\n|---|---|---|---|")
            for axis, info in c["axes"].items():
                md.append(f"| {axis} | {info['spearman_r']:+.3f} | [{info['ci_95_lo']:+.3f}, {info['ci_95_hi']:+.3f}] | {info['verdict']} |")
            md.append("")
    (out_dir / "phase_h_summary.md").write_text("\n".join(md))
    print(f"\nWrote {out_dir}/phase_h{{_filtered,_unfiltered}}.json + phase_h_summary.md")


if __name__ == "__main__":
    main()
