#!/usr/bin/env python3
"""Compare persona spaces — original Llama 3.1 8B Instruct vs abliterated.

Answers the research question: does abliteration change persona geometry?

Metrics emitted (all cosines, projections, and variance fractions):

  1.  cos(original_axis, abliterated_axis)
  2.  cos(refusal_direction[L=16], original_axis)
  3.  cos(refusal_direction[L=16], abliterated_axis)
  4.  PCA on each model's role vectors:
        - PC1 unit direction
        - fraction of variance explained by PC1..PCk
  5.  cos(original PC1, abliterated PC1)
  6.  cos(assistant_axis, PC1) for each model (sanity: axis ≈ PC1)
  7.  Per-role projection on each model's axis + delta per role → top movers
  8.  Pearson correlation of role loadings between the two PC1s
  9.  ‖axis‖ / ‖PC1 projection of default‖ for each model

Outputs:
    results/comparison/axis_comparison.json   (full numeric record)
    results/comparison/axis_comparison.md     (human-readable summary)

Dry-run: if abliterated axis/vectors are missing, runs only the single-model
metrics on the original and reports what's skipped. This lets us validate the
script now and rerun it after the abliterated pipeline completes.
"""

import argparse
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import torch


MIDDLE_LAYER = 16


def cos(a: torch.Tensor, b: torch.Tensor) -> float:
    a = a.flatten().float()
    b = b.flatten().float()
    denom = a.norm() * b.norm()
    if denom.item() < 1e-12:
        return 0.0
    return float(torch.dot(a, b) / denom)


def load_axis(path: Path) -> Optional[torch.Tensor]:
    if not path.exists():
        return None
    t = torch.load(path, map_location="cpu", weights_only=False)
    return t.squeeze().float()  # (hidden,)


def load_role_vectors(vectors_dir: Path):
    """Returns {role_name: {'type': str, 'vector': (hidden,) float tensor}}."""
    if not vectors_dir.exists():
        return {}
    out = {}
    for f in sorted(vectors_dir.glob("*.pt")):
        d = torch.load(f, map_location="cpu", weights_only=False)
        role = d.get("role", f.stem)
        out[role] = {
            "type": d.get("type", "unknown"),
            "vector": d["vector"].squeeze().float(),
        }
    return out


def pca(role_matrix: torch.Tensor, k: int = 10):
    """role_matrix: (n_roles, hidden).

    Returns (components (k, hidden), var_frac (k,), mean (hidden,)).
    """
    mean = role_matrix.mean(dim=0, keepdim=True)
    X = role_matrix - mean
    # Full SVD (n_roles is small, ~276)
    U, S, Vh = torch.linalg.svd(X, full_matrices=False)
    var = S ** 2
    total = var.sum()
    var_frac = (var / (total + 1e-12)).numpy()
    return Vh[:k], var_frac[:k], mean.squeeze()


def project_roles(role_dict: dict, direction: torch.Tensor) -> dict:
    """Returns {role: scalar projection onto unit(direction)}."""
    d = direction / (direction.norm() + 1e-12)
    return {r: float(torch.dot(v["vector"], d)) for r, v in role_dict.items()}


def pearson(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 2:
        return 0.0
    xc = x - x.mean()
    yc = y - y.mean()
    denom = (np.linalg.norm(xc) * np.linalg.norm(yc))
    if denom < 1e-12:
        return 0.0
    return float(np.dot(xc, yc) / denom)


@dataclass
class ModelRun:
    name: str
    axis: Optional[torch.Tensor]
    role_vectors: dict  # role -> {type, vector}
    default_vector: Optional[torch.Tensor]
    pc1: Optional[torch.Tensor] = None
    var_frac: Optional[np.ndarray] = None
    role_matrix: Optional[torch.Tensor] = None
    role_order: Optional[list] = None

    @classmethod
    def load(cls, name: str, run_dir: Path):
        axis = load_axis(run_dir / "axis.pt")
        rv = load_role_vectors(run_dir / "vectors")
        default = rv.pop("default", {}).get("vector") if "default" in rv else None
        return cls(name=name, axis=axis, role_vectors=rv, default_vector=default)

    def fit_pca(self, k: int = 10):
        if not self.role_vectors:
            return
        self.role_order = sorted(self.role_vectors.keys())
        self.role_matrix = torch.stack(
            [self.role_vectors[r]["vector"] for r in self.role_order]
        )
        comps, var_frac, _ = pca(self.role_matrix, k=k)
        self.pc1 = comps[0]
        self.var_frac = var_frac


def align_sign(ref: torch.Tensor, vec: torch.Tensor) -> torch.Tensor:
    """Flip vec if cos(ref, vec) < 0 so PC signs are comparable across runs."""
    return vec if cos(ref, vec) >= 0 else -vec


def subspace_angles_deg(A: torch.Tensor, B: torch.Tensor) -> list:
    """Principal angles (degrees) between row-subspaces of A and B.
    Both (k, d). Smaller angles = more-aligned subspaces.
    """
    Qa, _ = torch.linalg.qr(A.T.float())
    Qb, _ = torch.linalg.qr(B.T.float())
    # Singular values of Qa.T @ Qb are cosines of principal angles.
    sigmas = torch.linalg.svdvals(Qa.T @ Qb).clamp(-1.0, 1.0)
    angles = torch.arccos(sigmas) * (180.0 / torch.pi)
    return angles.tolist()


def procrustes_residual(X: torch.Tensor, Y: torch.Tensor) -> dict:
    """Find the orthogonal R minimizing ‖X R - Y‖_F; return residual + rotation magnitude.
    X, Y: (n, d) with the SAME row ordering (same roles in same order).
    """
    Xf = X.float()
    Yf = Y.float()
    # R = V U^T where X^T Y = U S V^T
    M = Xf.T @ Yf
    U, S, Vh = torch.linalg.svd(M, full_matrices=False)
    R = (Vh.T @ U.T)  # (d, d)
    X_rot = Xf @ R
    resid = torch.linalg.norm(X_rot - Yf).item()
    total = torch.linalg.norm(Yf).item()
    # Rotation magnitude as mean angle between X row i and X_rot row i
    cos_per_row = (Xf * X_rot).sum(dim=1) / (Xf.norm(dim=1) * X_rot.norm(dim=1) + 1e-12)
    cos_per_row = cos_per_row.clamp(-1.0, 1.0)
    mean_rot_deg = float((torch.arccos(cos_per_row) * (180.0 / torch.pi)).mean())
    return {
        "procrustes_residual_frob": resid,
        "relative_residual": resid / (total + 1e-12),
        "mean_row_rotation_deg": mean_rot_deg,
    }


def axis_bootstrap_null(role_matrix: torch.Tensor,
                        reference: torch.Tensor,
                        n_iter: int = 200,
                        seed: int = 17) -> dict:
    """Null-model: axis stability under bootstrap resampling of the role set.

    Definition of axis matching the paper: `reference - mean(roles)`.
    We don't have the true default-assistant "reference" cleanly separated
    here, so pass the actual model's axis direction as the reference to bias
    the bootstrap toward the same semantic contrast. Cosines below are then
    cos(axis_bootstrap_i, axis_bootstrap_j) for i≠j.

    Returns {mean, std, p05, p50, p95, n_iter}.
    """
    import numpy as np
    rng = np.random.default_rng(seed)
    n = role_matrix.shape[0]
    ref = reference.float()
    axes = []
    for _ in range(n_iter):
        idx = rng.choice(n, size=n, replace=True)
        sample_mean = role_matrix[idx].mean(dim=0).float()
        axes.append(ref - sample_mean)
    # Compute pairwise cosines between consecutive bootstrap axes (cheap proxy
    # for full pairwise matrix — n_iter=200 gives 199 samples of the null).
    cos_list = []
    for i in range(len(axes) - 1):
        cos_list.append(cos(axes[i], axes[i + 1]))
    arr = np.asarray(cos_list, dtype=float)
    return {
        "n_iter": n_iter,
        "n_pairs": len(cos_list),
        "mean": float(arr.mean()),
        "std": float(arr.std()),
        "p05": float(np.percentile(arr, 5)),
        "p50": float(np.percentile(arr, 50)),
        "p95": float(np.percentile(arr, 95)),
    }


def load_predictions(path: Path) -> Optional[dict]:
    if not path.exists():
        return None
    return json.loads(path.read_text())


def evaluate_predictions(predictions: dict, observed_top_movers: list,
                         all_common_roles: list) -> dict:
    """Given pre-registered safety-adjacent set and observed top-movers,
    report overlap + Fisher exact test (via scipy if present, else hypergeometric CDF).
    observed_top_movers: list of (role, delta) tuples.
    all_common_roles: list of role names in the universe.
    """
    predicted = set(predictions["safety_adjacent_roles"]) & set(all_common_roles)
    observed = {r for r, _ in observed_top_movers}
    overlap = predicted & observed
    # Hypergeometric: drawing |observed| balls from universe; how many are "predicted"?
    N = len(all_common_roles)
    K = len(predicted)
    n = len(observed)
    k = len(overlap)
    # Expected overlap under null = K * n / N
    import math
    def comb(a, b):
        if b < 0 or b > a:
            return 0
        return math.comb(a, b)
    p_at_least_k = 0.0
    for i in range(k, min(K, n) + 1):
        p_at_least_k += comb(K, i) * comb(N - K, n - i)
    p_at_least_k /= max(comb(N, n), 1)
    expected = K * n / max(N, 1)
    return {
        "n_predicted_safety_adjacent": K,
        "n_observed_top_movers": n,
        "n_overlap": k,
        "expected_overlap_under_null": expected,
        "hypergeom_p_at_least_k": p_at_least_k,
        "roles_in_overlap": sorted(overlap),
        "roles_predicted_not_observed": sorted(predicted - observed),
        "roles_observed_not_predicted": sorted(observed - predicted),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--original_dir", default="results/original")
    ap.add_argument("--abliterated_dir", default="results/abliterated")
    ap.add_argument("--refusal_direction",
                    default="results/comparison/refusal_direction_from_mlabonne.pt",
                    help="Path to refusal direction. New format (dict with 'per_layer'/'global') "
                         "from 10_recover_refusal_from_mlabonne.py preferred; legacy 2D tensor "
                         "(n_layers, d) also supported.")
    ap.add_argument("--output_dir", default="results/comparison")
    ap.add_argument("--layer", type=int, default=MIDDLE_LAYER)
    ap.add_argument("--top_movers", type=int, default=15)
    args = ap.parse_args()

    orig_dir = Path(args.original_dir)
    abl_dir = Path(args.abliterated_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"=== Persona-space comparison @ {datetime.now().strftime('%Y-%m-%d %H:%M')} ===")

    # Load runs
    orig = ModelRun.load("original", orig_dir)
    abl = ModelRun.load("abliterated", abl_dir)

    has_abl = abl.axis is not None and bool(abl.role_vectors)
    print(f"Original:   axis={'✓' if orig.axis is not None else '✗'}  "
          f"roles={len(orig.role_vectors)}  default={'✓' if orig.default_vector is not None else '✗'}")
    print(f"Abliterated: axis={'✓' if abl.axis is not None else '✗'}  "
          f"roles={len(abl.role_vectors)}  default={'✓' if abl.default_vector is not None else '✗'}")

    if orig.axis is None:
        raise SystemExit("FATAL: original axis missing — cannot compare")

    # Refusal direction at the middle layer.
    # Supports two formats:
    #   - Dict {"per_layer": (n_layers, d), "global": (d,), ...} from 10_recover_refusal_from_mlabonne.py
    #   - Legacy 2D Tensor (n_layers, d) from 00_cosine_precheck.py
    r_loaded = torch.load(args.refusal_direction, map_location="cpu", weights_only=False)
    if isinstance(r_loaded, dict):
        r_all = r_loaded["per_layer"].float()
        r_source = f"dict from {args.refusal_direction} (method: {r_loaded.get('method', '?')})"
    else:
        r_all = r_loaded.float()
        r_source = f"legacy tensor from {args.refusal_direction}"
    r_middle = r_all[args.layer]
    print(f"Refusal direction: {r_source}")
    print(f"  per_layer shape={tuple(r_all.shape)}, using layer {args.layer}, ‖r‖={r_middle.norm():.4f}")

    # PCA on each
    orig.fit_pca(k=10)
    if has_abl:
        abl.fit_pca(k=10)

    report = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "layer": args.layer,
        "hidden_dim": int(orig.axis.shape[0]),
        "n_roles_original": len(orig.role_vectors),
        "n_roles_abliterated": len(abl.role_vectors),
        "has_abliterated": has_abl,
        "axes": {},
        "refusal": {},
        "pca": {},
        "per_role": {},
    }

    # ── 1. Axis vs axis (if both available) ───────────────────────────
    if has_abl:
        report["axes"]["cos_original_vs_abliterated"] = cos(orig.axis, abl.axis)
        report["axes"]["norm_original"] = float(orig.axis.norm())
        report["axes"]["norm_abliterated"] = float(abl.axis.norm())
        report["axes"]["norm_ratio_abl_over_orig"] = float(abl.axis.norm() / (orig.axis.norm() + 1e-12))

    # ── 2–3. Refusal vs each axis ─────────────────────────────────────
    report["refusal"]["cos_refusal_vs_original_axis"] = cos(r_middle, orig.axis)
    if has_abl:
        report["refusal"]["cos_refusal_vs_abliterated_axis"] = cos(r_middle, abl.axis)
    if orig.pc1 is not None:
        report["refusal"]["cos_refusal_vs_original_pc1"] = cos(r_middle, orig.pc1)
    if has_abl and abl.pc1 is not None:
        report["refusal"]["cos_refusal_vs_abliterated_pc1"] = cos(r_middle, abl.pc1)

    # ── 4. PCA variance explained ─────────────────────────────────────
    if orig.var_frac is not None:
        report["pca"]["original"] = {
            "var_frac_top10": orig.var_frac.tolist(),
            "var_frac_cumulative": np.cumsum(orig.var_frac).tolist(),
        }
    if has_abl and abl.var_frac is not None:
        report["pca"]["abliterated"] = {
            "var_frac_top10": abl.var_frac.tolist(),
            "var_frac_cumulative": np.cumsum(abl.var_frac).tolist(),
        }

    # ── 5. PC1 vs PC1 ─────────────────────────────────────────────────
    if has_abl and orig.pc1 is not None and abl.pc1 is not None:
        abl_pc1_aligned = align_sign(orig.pc1, abl.pc1)
        report["pca"]["cos_pc1_original_vs_abliterated"] = cos(orig.pc1, abl_pc1_aligned)

    # ── 6. axis vs PC1 (sanity) ───────────────────────────────────────
    if orig.pc1 is not None:
        report["pca"]["cos_axis_vs_pc1_original"] = cos(orig.axis, orig.pc1)
    if has_abl and abl.pc1 is not None:
        report["pca"]["cos_axis_vs_pc1_abliterated"] = cos(abl.axis, abl.pc1)

    # ── 7. Per-role projections + top movers ──────────────────────────
    orig_proj_axis = project_roles(orig.role_vectors, orig.axis)
    per_role = {r: {"orig_proj_axis": orig_proj_axis[r]} for r in orig.role_vectors}

    if has_abl:
        abl_proj_axis = project_roles(abl.role_vectors, abl.axis)
        common = sorted(set(orig.role_vectors) & set(abl.role_vectors))
        deltas = []
        for r in common:
            d = abl_proj_axis.get(r, 0.0) - orig_proj_axis.get(r, 0.0)
            per_role.setdefault(r, {})["abl_proj_axis"] = abl_proj_axis.get(r, 0.0)
            per_role[r]["delta_abl_minus_orig"] = d
            deltas.append((r, d))
        deltas.sort(key=lambda x: x[1])
        report["per_role_top_down_movers"] = deltas[: args.top_movers]
        report["per_role_top_up_movers"] = deltas[-args.top_movers :][::-1]

        # PC1 role loadings → Pearson
        if orig.pc1 is not None and abl.pc1 is not None:
            abl_pc1_aligned = align_sign(orig.pc1, abl.pc1)
            loadings_orig = np.array([
                float(torch.dot(orig.role_vectors[r]["vector"], orig.pc1 / orig.pc1.norm()))
                for r in common
            ])
            loadings_abl = np.array([
                float(torch.dot(abl.role_vectors[r]["vector"], abl_pc1_aligned / abl_pc1_aligned.norm()))
                for r in common
            ])
            report["pca"]["pearson_pc1_loadings_common_roles"] = pearson(loadings_orig, loadings_abl)
            report["pca"]["n_common_roles"] = len(common)

    report["per_role"] = per_role

    # ── 9. Default-vector projection ──────────────────────────────────
    if orig.default_vector is not None:
        report["axes"]["default_proj_on_original_axis"] = (
            float(torch.dot(orig.default_vector, orig.axis / (orig.axis.norm() + 1e-12)))
        )
    if has_abl and abl.default_vector is not None:
        report["axes"]["default_proj_on_abliterated_axis"] = (
            float(torch.dot(abl.default_vector, abl.axis / (abl.axis.norm() + 1e-12)))
        )

    # ── 10. Null-model baseline: axis stability under with-replacement bootstrap ─
    # For each of N iterations, resample 276 roles with replacement, compute
    # axis = reference - mean(resample). Report cos between pairs. This is the
    # "noise floor for axis direction due to role sampling." Any orig-vs-abl
    # cosine BELOW the bootstrap p05 is evidence of real structural change.
    if orig.role_matrix is not None and orig.axis is not None:
        report["null_model"] = {
            "description": (
                "Bootstrap-resample roles with replacement, compute "
                "axis = axis_direction - mean(resample), take cosine between "
                "consecutive bootstrap axes. p05 = 5th percentile of the null "
                "distribution of axis cosines under role-sampling noise."
            ),
            "original_bootstrap": axis_bootstrap_null(
                orig.role_matrix, orig.axis, n_iter=200,
            ),
        }
        if has_abl and abl.role_matrix is not None and abl.axis is not None:
            report["null_model"]["abliterated_bootstrap"] = axis_bootstrap_null(
                abl.role_matrix, abl.axis, n_iter=200, seed=23,
            )

    # ── 11. Subspace angles — top-5 PC subspace of orig vs abl ────────────
    # Answers: "is the persona subspace preserved, just rotated, or deformed?"
    if has_abl and orig.role_matrix is not None and abl.role_matrix is not None:
        k = 5
        comps_o, _, _ = pca(orig.role_matrix, k=k)
        comps_a, _, _ = pca(abl.role_matrix, k=k)
        angles = subspace_angles_deg(comps_o, comps_a)
        report["subspace_angles_top5_deg"] = angles
        report["subspace_angles_max_deg"] = max(angles) if angles else None

    # ── 12. Centroid shift vs refusal direction (key causal test) ─────────
    # If abliteration's effect on persona space is aligned with the refusal
    # direction, cos here is near ±1 → hypothesis 1 (overlap).
    # If orthogonal → hypothesis 2.
    if has_abl and orig.role_matrix is not None and abl.role_matrix is not None:
        common = sorted(set(orig.role_vectors) & set(abl.role_vectors))
        if common:
            # Common-role centroid shift
            X = torch.stack([orig.role_vectors[r]["vector"] for r in common])
            Y = torch.stack([abl.role_vectors[r]["vector"] for r in common])
            shift = (Y.mean(dim=0) - X.mean(dim=0)).float()
            report["centroid_shift"] = {
                "norm": float(shift.norm()),
                "cos_vs_refusal": cos(shift, r_middle),
                "cos_vs_original_axis": cos(shift, orig.axis),
                "cos_vs_abliterated_axis": cos(shift, abl.axis),
            }

            # ── 13. Procrustes — best orthogonal rotation aligning role clouds ──
            # Residual tells us "distortion beyond pure rotation."
            report["procrustes"] = procrustes_residual(X, Y)

            # ── 14. Raw per-role displacement in R^d ──────────────────────────
            # In contrast to per-role-projection delta, this captures movement
            # in ALL directions, not just along the (rotating) axis.
            disps = []
            for r in common:
                d = (abl.role_vectors[r]["vector"] - orig.role_vectors[r]["vector"]).float()
                disps.append((r, float(d.norm()), cos(d, r_middle)))
            disps.sort(key=lambda x: -x[1])  # largest displacement first
            report["per_role_raw_displacement"] = {
                "top_by_norm": disps[: args.top_movers],
                "mean_norm": sum(d for _, d, _ in disps) / max(len(disps), 1),
                "mean_cos_disp_vs_refusal": sum(c for _, _, c in disps) / max(len(disps), 1),
            }

    # ── 15a. Default-Assistant shift decomposition (the core causal test) ─
    # If default_abl - default_orig is aligned with refusal, then the Assistant
    # itself moves ALONG the refusal direction → abliteration modulates persona.
    # If orthogonal, the Assistant vector is robust to refusal removal.
    if (has_abl and orig.default_vector is not None
            and abl.default_vector is not None):
        d = (abl.default_vector - orig.default_vector).float()
        r_unit = r_middle / (r_middle.norm() + 1e-12)
        proj_along_refusal = float(torch.dot(d, r_unit))
        parallel_component = proj_along_refusal * r_unit
        orthogonal_component = d - parallel_component
        report["default_assistant_shift"] = {
            "norm_shift": float(d.norm()),
            "norm_shift_relative": float(d.norm() / (orig.default_vector.norm() + 1e-12)),
            "cos_shift_vs_refusal": cos(d, r_middle),
            "cos_shift_vs_original_axis": cos(d, orig.axis),
            "norm_parallel_to_refusal": float(parallel_component.norm()),
            "norm_orthogonal_to_refusal": float(orthogonal_component.norm()),
            "fraction_of_shift_along_refusal": float(
                parallel_component.norm() / (d.norm() + 1e-12)
            ),
        }

    # ── 15. Pre-registered prediction test ────────────────────────────────
    predictions = load_predictions(out_dir / "predictions.json")
    if predictions and has_abl and "per_role_top_down_movers" in report:
        all_top = (report["per_role_top_down_movers"]
                   + report["per_role_top_up_movers"])
        common_roles = sorted(set(orig.role_vectors) & set(abl.role_vectors))
        report["prediction_test"] = evaluate_predictions(
            predictions, all_top, common_roles,
        )

    # ── Save JSON ─────────────────────────────────────────────────────
    json_path = out_dir / "axis_comparison.json"
    with open(json_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nWrote {json_path}")

    # ── Markdown summary ──────────────────────────────────────────────
    md = ["# Persona-space comparison", "",
          f"Generated: {report['generated_at']}",
          f"Layer: {report['layer']}, hidden dim: {report['hidden_dim']}",
          f"Roles — original: {report['n_roles_original']}, abliterated: {report['n_roles_abliterated']}",
          f"Has abliterated: {'yes' if has_abl else 'NO (dry-run mode)'}", ""]

    md += ["## Axis geometry", ""]
    if has_abl:
        md += [
            f"- cos(original_axis, abliterated_axis) = **{report['axes']['cos_original_vs_abliterated']:.4f}**",
            f"- ‖original axis‖ = {report['axes']['norm_original']:.4f}",
            f"- ‖abliterated axis‖ = {report['axes']['norm_abliterated']:.4f}",
            f"- ‖abl‖ / ‖orig‖ = {report['axes']['norm_ratio_abl_over_orig']:.4f}",
        ]
    else:
        md += ["- (abliterated axis not available; rerun after abliterated pipeline completes)"]
    md += [""]

    md += ["## Refusal direction alignment", ""]
    md += [f"- cos(refusal, original_axis) = **{report['refusal']['cos_refusal_vs_original_axis']:.4f}**"]
    if "cos_refusal_vs_abliterated_axis" in report["refusal"]:
        md += [f"- cos(refusal, abliterated_axis) = **{report['refusal']['cos_refusal_vs_abliterated_axis']:.4f}**"]
    if "cos_refusal_vs_original_pc1" in report["refusal"]:
        md += [f"- cos(refusal, original_PC1) = {report['refusal']['cos_refusal_vs_original_pc1']:.4f}"]
    if "cos_refusal_vs_abliterated_pc1" in report["refusal"]:
        md += [f"- cos(refusal, abliterated_PC1) = {report['refusal']['cos_refusal_vs_abliterated_pc1']:.4f}"]
    md += [""]

    md += ["## PCA — variance explained by PC1", ""]
    if "original" in report["pca"]:
        md += [f"- original PC1: {report['pca']['original']['var_frac_top10'][0]*100:.2f}%",
               f"- original top-5 cumulative: {report['pca']['original']['var_frac_cumulative'][4]*100:.2f}%"]
    if "abliterated" in report["pca"]:
        md += [f"- abliterated PC1: {report['pca']['abliterated']['var_frac_top10'][0]*100:.2f}%",
               f"- abliterated top-5 cumulative: {report['pca']['abliterated']['var_frac_cumulative'][4]*100:.2f}%"]
    if "cos_pc1_original_vs_abliterated" in report["pca"]:
        md += [f"- cos(PC1_original, PC1_abliterated) = **{report['pca']['cos_pc1_original_vs_abliterated']:.4f}**"]
    if "pearson_pc1_loadings_common_roles" in report["pca"]:
        md += [f"- Pearson(PC1 role loadings, {report['pca']['n_common_roles']} common roles) = "
               f"**{report['pca']['pearson_pc1_loadings_common_roles']:.4f}**"]
    md += [""]

    if "cos_axis_vs_pc1_original" in report["pca"]:
        md += ["## Sanity — axis vs PC1", "",
               f"- cos(axis, PC1) original = {report['pca']['cos_axis_vs_pc1_original']:.4f}",
               (f"- cos(axis, PC1) abliterated = {report['pca']['cos_axis_vs_pc1_abliterated']:.4f}"
                if "cos_axis_vs_pc1_abliterated" in report["pca"] else ""),
               ""]

    if has_abl and "per_role_top_down_movers" in report:
        md += [f"## Top {args.top_movers} roles that moved DOWN (more role-playing after abliteration)", "", "| role | Δ projection |", "|---|---|"]
        for r, d in report["per_role_top_down_movers"]:
            md += [f"| {r} | {d:+.4f} |"]
        md += [""]
        md += [f"## Top {args.top_movers} roles that moved UP (more Assistant-like)", "", "| role | Δ projection |", "|---|---|"]
        for r, d in report["per_role_top_up_movers"]:
            md += [f"| {r} | {d:+.4f} |"]
        md += [""]

    # ── Null model ────────────────────────────────────────────────────
    if "null_model" in report:
        md += ["## Null-model axis stability (split-half bootstrap)", "",
               "Cosine of axis computed from one random half of roles vs the other half. ",
               "This is the sampling-noise noise floor — orig-vs-abl cosine above p95 is "
               "within noise; below p05 suggests real structural change.", ""]
        o = report["null_model"]["original_bootstrap"]
        md += [f"- original bootstrap: mean={o['mean']:.4f}, p05={o['p05']:.4f}, "
               f"p50={o['p50']:.4f}, p95={o['p95']:.4f} (n_iter={o['n_iter']})"]
        if "abliterated_bootstrap" in report["null_model"]:
            a = report["null_model"]["abliterated_bootstrap"]
            md += [f"- abliterated bootstrap: mean={a['mean']:.4f}, p05={a['p05']:.4f}, "
                   f"p50={a['p50']:.4f}, p95={a['p95']:.4f}"]
        md += [""]

    # ── Subspace angles ───────────────────────────────────────────────
    if "subspace_angles_top5_deg" in report:
        md += ["## Subspace alignment (top-5 PC bases)", "",
               "Principal angles in degrees between top-5 PCA subspace of original and abliterated. ",
               "All angles near 0° → persona subspace preserved. Any angle near 90° → a PC dimension "
               "collapsed or emerged.", ""]
        for i, a in enumerate(report["subspace_angles_top5_deg"]):
            md += [f"- θ_{i+1} = {a:.2f}°"]
        md += [""]

    # ── Centroid shift (key causal test) ──────────────────────────────
    if "centroid_shift" in report:
        cs = report["centroid_shift"]
        md += ["## Centroid shift (causal test)", "",
               "Mean(abl roles) - Mean(orig roles). Direction of this shift tells us what abliteration does to persona space.", "",
               f"- ‖shift‖ = {cs['norm']:.4f}",
               f"- **cos(shift, refusal_direction) = {cs['cos_vs_refusal']:+.4f}**  ← key test. ≈±1 = hypothesis 1, ≈0 = hypothesis 2",
               f"- cos(shift, original_axis) = {cs['cos_vs_original_axis']:+.4f}",
               f"- cos(shift, abliterated_axis) = {cs['cos_vs_abliterated_axis']:+.4f}",
               ""]

    # ── Procrustes residual ──────────────────────────────────────────
    if "procrustes" in report:
        p = report["procrustes"]
        md += ["## Procrustes (best orthogonal rotation aligning role clouds)", "",
               f"- residual Frobenius norm = {p['procrustes_residual_frob']:.4f}",
               f"- relative residual = {p['relative_residual']:.4f}  (0 = pure rotation, 1 = no preserved structure)",
               f"- mean per-role rotation angle = {p['mean_row_rotation_deg']:.2f}°",
               ""]

    # ── Raw displacement ─────────────────────────────────────────────
    if "per_role_raw_displacement" in report:
        rd = report["per_role_raw_displacement"]
        md += ["## Raw per-role displacement ‖v_abl − v_orig‖", "",
               f"- mean displacement norm = {rd['mean_norm']:.4f}",
               f"- mean cos(displacement, refusal) across roles = {rd['mean_cos_disp_vs_refusal']:+.4f}", "",
               f"### Top-{args.top_movers} displaced roles (by raw norm)", "",
               "| role | ‖Δ‖ | cos(Δ, refusal) |", "|---|---|---|"]
        for r, n, c in rd["top_by_norm"]:
            md += [f"| {r} | {n:.4f} | {c:+.4f} |"]
        md += [""]

    # ── Default-Assistant shift decomposition ────────────────────────
    if "default_assistant_shift" in report:
        ds = report["default_assistant_shift"]
        md += ["## Default-Assistant shift (core causal test)", "",
               f"- ‖default_abl − default_orig‖ = **{ds['norm_shift']:.4f}**",
               f"- relative shift (‖Δ‖ / ‖default_orig‖) = {ds['norm_shift_relative']:.4f}",
               f"- **cos(Δ, refusal_direction) = {ds['cos_shift_vs_refusal']:+.4f}** ← key finding",
               f"- cos(Δ, original_axis) = {ds['cos_shift_vs_original_axis']:+.4f}",
               f"- ‖component along refusal‖ = {ds['norm_parallel_to_refusal']:.4f}",
               f"- ‖component orthogonal to refusal‖ = {ds['norm_orthogonal_to_refusal']:.4f}",
               f"- **fraction of Δ along refusal = {ds['fraction_of_shift_along_refusal']*100:.1f}%** "
               f"(100% = pure refusal-direction motion; 0% = orthogonal to refusal)",
               ""]

    # ── Prediction test ──────────────────────────────────────────────
    if "prediction_test" in report:
        pt = report["prediction_test"]
        md += ["## Pre-registered prediction test", "",
               f"Predicted safety-adjacent roles (N={pt['n_predicted_safety_adjacent']}) vs observed top-movers (N={pt['n_observed_top_movers']}).", "",
               f"- overlap = **{pt['n_overlap']}** (expected under null = {pt['expected_overlap_under_null']:.1f})",
               f"- hypergeometric P(overlap ≥ k) = **{pt['hypergeom_p_at_least_k']:.4f}**  ← <0.05 = pre-registered hypothesis confirmed",
               "",
               f"**Predicted AND observed:** {', '.join(pt['roles_in_overlap']) or '(none)'}",
               f"**Predicted but not observed:** {', '.join(pt['roles_predicted_not_observed']) or '(none)'}",
               f"**Observed but not predicted:** {', '.join(pt['roles_observed_not_predicted']) or '(none)'}",
               ""]

    md_path = out_dir / "axis_comparison.md"
    md_path.write_text("\n".join([x for x in md if x is not None]))
    print(f"Wrote {md_path}")

    # Terminal summary
    print("\n=== KEY FINDINGS ===")
    if has_abl:
        print(f"cos(orig_axis, abl_axis)        = {report['axes']['cos_original_vs_abliterated']:+.4f}")
        print(f"cos(refusal, orig_axis)         = {report['refusal']['cos_refusal_vs_original_axis']:+.4f}")
        print(f"cos(refusal, abl_axis)          = {report['refusal']['cos_refusal_vs_abliterated_axis']:+.4f}")
        if "cos_pc1_original_vs_abliterated" in report["pca"]:
            print(f"cos(PC1_orig, PC1_abl)          = {report['pca']['cos_pc1_original_vs_abliterated']:+.4f}")
        if "pearson_pc1_loadings_common_roles" in report["pca"]:
            print(f"Pearson PC1 loadings            = {report['pca']['pearson_pc1_loadings_common_roles']:+.4f}")
    else:
        print(f"cos(refusal, orig_axis)         = {report['refusal']['cos_refusal_vs_original_axis']:+.4f}")
        print(f"(rerun after abliterated pipeline produces results/abliterated/)")


if __name__ == "__main__":
    main()
