"""
00b_verify_precheck.py — Offline verification of Step 0 pre-check results.

Loads the saved tensors from results/comparison/ and runs 5 sanity checks:
    1. Random-vector baseline for |cos|
    2. Sign-consistency binomial test
    3. Refusal direction norm-profile sanity
    4. PCA of saved role vectors (is approximate_axis a reasonable PC1 proxy?)
    5. Default-vs-role separability on the approximate axis

No GPU, no model loading. Runs in ~30 seconds.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.stats import binomtest

RESULTS_DIR = Path("/scratch/paulc/assistant-axis-abliteration/results/comparison")
MIDDLE_LAYER = 16
N_LAYERS = 32
HIDDEN = 4096

ROLE_GROUPS = {
    "safety":       ["guardian", "judge", "demon", "saboteur", "criminal", "vigilante", "angel"],
    "professional": ["assistant", "consultant", "analyst", "therapist", "teacher", "researcher", "lawyer"],
    "creative":     ["pirate", "ghost", "witch", "detective", "warrior", "bard", "robot", "alien"],
    "abstract":     ["void", "aberration", "hive", "parasite", "eldritch", "swarm", "chimera", "echo"],
}
GROUP_COLORS = {"safety": "red", "professional": "blue", "creative": "green", "abstract": "purple"}
ROLE_TO_GROUP = {r: g for g, roles in ROLE_GROUPS.items() for r in roles}


def banner(title: str) -> None:
    print(f"\n{'=' * 70}\n{title}\n{'=' * 70}")


def load_inputs():
    refusal = torch.load(RESULTS_DIR / "refusal_direction.pt", map_location="cpu").float()
    axis = torch.load(RESULTS_DIR / "approximate_axis.pt", map_location="cpu").float()
    role_vectors = torch.load(RESULTS_DIR / "role_vectors.pt", map_location="cpu")
    role_vectors = {name: v.float() for name, v in role_vectors.items()}
    with open(RESULTS_DIR / "cosine_precheck_results.json") as f:
        results = json.load(f)
    return refusal, axis, role_vectors, results


def check_random_baseline(refusal: torch.Tensor, observed_abs_cos: float, n_samples: int = 10000):
    banner("Check 1: Random-vector baseline for |cos|")
    torch.manual_seed(0)
    rand = torch.randn(n_samples, HIDDEN)
    rand = rand / rand.norm(dim=1, keepdim=True)

    ref = refusal[MIDDLE_LAYER]
    ref = ref / ref.norm()
    random_cos = (rand @ ref).abs()

    mean = random_cos.mean().item()
    std = random_cos.std().item()
    p95 = random_cos.quantile(0.95).item()
    p99 = random_cos.quantile(0.99).item()
    z = observed_abs_cos / std

    print(f"  Random |cos| with refusal_dir[{MIDDLE_LAYER}] over {n_samples} samples:")
    print(f"    mean           = {mean:.4f}")
    print(f"    std            = {std:.4f}   (theoretical 1/sqrt(d) = {1 / math.sqrt(HIDDEN):.4f})")
    print(f"    95th percentile = {p95:.4f}")
    print(f"    99th percentile = {p99:.4f}")
    print(f"  Observed |cos|    = {observed_abs_cos:.4f}")
    print(f"  Z-score           = {z:.2f}σ above random baseline")
    verdict = "NON-ZERO" if z > 3 else "INDISTINGUISHABLE from random"
    print(f"  Verdict: {verdict} (threshold z>3)")
    return {"mean": mean, "std": std, "p95": p95, "p99": p99, "z_score_observed": z, "verdict": verdict}


def check_sign_consistency(per_layer_cos: list[float]):
    banner("Check 2: Sign-consistency binomial test")
    n_neg = sum(1 for c in per_layer_cos if c < 0)
    n_total = len(per_layer_cos)
    test = binomtest(n_neg, n_total, p=0.5, alternative="two-sided")
    print(f"  Layers with negative cosine: {n_neg}/{n_total}")
    print(f"  Binomial two-sided p-value:  {test.pvalue:.3e}")
    verdict = "SIGNIFICANT" if test.pvalue < 1e-3 else "not significant"
    print(f"  Verdict: sign pattern is {verdict}")
    if n_neg > n_total - n_neg:
        print("  Direction: axis is anti-aligned with refusal (role→default vs harmful→harmless)")
    return {"n_negative": n_neg, "n_total": n_total, "p_value": test.pvalue, "verdict": verdict}


def check_refusal_norms(refusal: torch.Tensor, results: dict):
    banner("Check 3: Refusal-direction norm profile")
    # NOTE: refusal tensor is saved already L2-normalized per layer (all rows have norm 1).
    # The meaningful profile is the PRE-normalization norm saved in the JSON.
    norms = np.array(results["per_layer"]["refusal_norms"])
    # Monotonicity
    diffs = np.diff(norms)
    monotonic = bool((diffs >= 0).all())
    n_decreases = int((diffs < 0).sum())
    # Correlation with layer index on log scale (skip first layer where norm may be ~0)
    safe = norms[1:]
    log_norms = np.log(safe + 1e-9)
    layer_idx = np.arange(1, N_LAYERS)
    corr = float(np.corrcoef(layer_idx, log_norms)[0, 1])
    print(f"  Norms: {norms[0]:.3f} (layer 0) → {norms[-1]:.3f} (layer {N_LAYERS - 1})")
    print(f"  Monotonically non-decreasing: {monotonic} ({n_decreases} decreases)")
    print(f"  corr(layer_idx, log_norm) = {corr:.4f}  (expect ~0.95+ for exp-growth profile)")
    verdict = "HEALTHY" if monotonic and corr > 0.9 else "anomalous"
    print(f"  Verdict: refusal-direction profile is {verdict}")
    return {"monotonic": monotonic, "n_decreases": n_decreases,
            "corr_layer_lognorm": corr, "verdict": verdict}


def check_pc1(role_vectors: dict, axis: torch.Tensor, refusal: torch.Tensor):
    banner("Check 4: PCA of role vectors → PC1 vs approximate_axis")
    names = list(role_vectors.keys())
    role_matrix = torch.stack([role_vectors[n][MIDDLE_LAYER] for n in names])  # (30, 4096)
    centered = role_matrix - role_matrix.mean(dim=0, keepdim=True)
    # SVD: centered = U S V^T ; PC1 is V[0]
    U, S, Vh = torch.linalg.svd(centered, full_matrices=False)
    pc1 = Vh[0]
    # Variance explained
    var_per_pc = (S**2) / (role_matrix.shape[0] - 1)
    total_var = var_per_pc.sum().item()
    var_ratios = (var_per_pc / var_per_pc.sum()).tolist()

    ax = axis[MIDDLE_LAYER] / (axis[MIDDLE_LAYER].norm() + 1e-9)
    ref = refusal[MIDDLE_LAYER] / (refusal[MIDDLE_LAYER].norm() + 1e-9)
    cos_pc1_axis = abs(float(pc1 @ ax))
    cos_pc1_refusal = abs(float(pc1 @ ref))

    print(f"  PC1-PC3 variance-explained ratios: "
          f"{var_ratios[0]:.3f}, {var_ratios[1]:.3f}, {var_ratios[2]:.3f}")
    print(f"  Cumulative PC1+2+3: {sum(var_ratios[:3]):.3f}")
    print(f"  |cos(PC1_roles, approximate_axis[{MIDDLE_LAYER}])| = {cos_pc1_axis:.4f}")
    print(f"  |cos(PC1_roles, refusal_dir[{MIDDLE_LAYER}])|     = {cos_pc1_refusal:.4f}")
    if cos_pc1_axis > 0.5:
        verdict = "GOOD proxy (|cos| > 0.5)"
    elif cos_pc1_axis > 0.3:
        verdict = "weak proxy (0.3 < |cos| ≤ 0.5)"
    else:
        verdict = "POOR proxy (|cos| ≤ 0.3)"
    print(f"  Verdict: approximate_axis is a {verdict} for PC1")
    return {"abs_cos_pc1_axis": cos_pc1_axis, "abs_cos_pc1_refusal": cos_pc1_refusal,
            "var_ratios": var_ratios[:10], "verdict": verdict}


def check_separability(role_vectors: dict, axis: torch.Tensor):
    banner("Check 5: Default-vs-role separability on the approximate axis")
    names = list(role_vectors.keys())
    role_matrix = torch.stack([role_vectors[n][MIDDLE_LAYER] for n in names])  # (30, 4096)
    ax = axis[MIDDLE_LAYER] / (axis[MIDDLE_LAYER].norm() + 1e-9)
    role_proj = (role_matrix @ ax).numpy()  # (30,)

    role_mean = role_matrix.mean(dim=0)
    # approximate_axis = default_mean - role_mean  =>  default_mean = role_mean + approximate_axis
    default_mean = role_mean + axis[MIDDLE_LAYER]
    default_proj = float(default_mean @ ax)

    rmean = float(role_proj.mean())
    rstd = float(role_proj.std(ddof=1))
    sigma = (default_proj - rmean) / (rstd + 1e-9)

    # Also compute separation excluding assistant-like "professional" roles,
    # which are known to sit near the default end of the axis (paper Fig. 2).
    prof = set(ROLE_GROUPS["professional"])
    nonprof_mask = np.array([n not in prof for n in names])
    nonprof_proj = role_proj[nonprof_mask]
    sigma_nonprof = (default_proj - nonprof_proj.mean()) / (nonprof_proj.std(ddof=1) + 1e-9)

    print(f"  Projections of 30 role vectors onto normalized axis:")
    print(f"    min    = {role_proj.min():.3f}")
    print(f"    mean   = {rmean:.3f}")
    print(f"    max    = {role_proj.max():.3f}")
    print(f"    std    = {rstd:.3f}")
    print(f"  Default-mean projection = {default_proj:.3f}")
    print(f"  All-roles separation (default − role_mean) / role_std = {sigma:.2f}σ")
    print(f"  Non-professional separation                           = {sigma_nonprof:.2f}σ")
    print(f"    (excluding {sorted(prof)} which cluster with default)")
    sigma_best = max(sigma, sigma_nonprof)
    verdict = "CLEAR separation" if sigma_best > 3 else "weak separation"
    print(f"  Verdict: {verdict} between default and non-professional role populations")

    # Per-role breakdown by group
    print("\n  Projections by role (lower = more role-like, higher = more assistant-like):")
    for group, roles in ROLE_GROUPS.items():
        print(f"    [{group}]")
        for r in roles:
            if r in role_vectors:
                i = names.index(r)
                print(f"      {r:<13} {role_proj[i]:+.3f}")

    return {"role_proj_min": float(role_proj.min()), "role_proj_mean": rmean,
            "role_proj_max": float(role_proj.max()), "role_proj_std": rstd,
            "default_proj": default_proj,
            "separation_sigma_all": sigma,
            "separation_sigma_nonprofessional": float(sigma_nonprof),
            "verdict": verdict}, (names, role_proj, default_proj)


def make_projection_plot(names, role_proj, default_proj, out_path: Path):
    fig, ax = plt.subplots(figsize=(10, 4))
    for i, name in enumerate(names):
        group = ROLE_TO_GROUP.get(name, "other")
        color = GROUP_COLORS.get(group, "gray")
        y = hash(name) % 5 - 2  # jitter
        ax.scatter(role_proj[i], y * 0.1, color=color, s=60, alpha=0.7)
        ax.annotate(name, (role_proj[i], y * 0.1), fontsize=7, alpha=0.8,
                    xytext=(3, 3), textcoords="offset points")

    ax.axvline(default_proj, color="black", linestyle="--", linewidth=2, label=f"default mean ({default_proj:.2f})")
    ax.axvline(np.mean(role_proj), color="gray", linestyle=":", linewidth=1.5,
               label=f"role mean ({np.mean(role_proj):.2f})")

    # Legend entries for groups
    for group, color in GROUP_COLORS.items():
        ax.scatter([], [], color=color, label=group, s=60)

    ax.set_xlabel(f"Projection onto approximate_axis[{MIDDLE_LAYER}] (normalized)")
    ax.set_yticks([])
    ax.set_title("Role vectors projected onto the approximate assistant axis (middle layer)")
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(True, axis="x", alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=120)
    print(f"\n  Saved plot: {out_path}")


def main():
    refusal, axis, role_vectors, results = load_inputs()
    observed_abs_cos = results["focal_layer"]["abs_cosine_similarity"]
    per_layer_cos = results["per_layer"]["cosine_similarity"]

    print(f"Loaded: refusal {tuple(refusal.shape)}, axis {tuple(axis.shape)}, "
          f"{len(role_vectors)} role vectors")
    print(f"Observed |cos| at layer {MIDDLE_LAYER}: {observed_abs_cos:.4f}")

    rep = {}
    rep["random_baseline"] = check_random_baseline(refusal, observed_abs_cos)
    rep["sign_consistency"] = check_sign_consistency(per_layer_cos)
    rep["refusal_norm_profile"] = check_refusal_norms(refusal, results)
    rep["pc1_check"] = check_pc1(role_vectors, axis, refusal)
    sep_report, proj_data = check_separability(role_vectors, axis)
    rep["separability"] = sep_report
    make_projection_plot(*proj_data, RESULTS_DIR / "role_projections.png")

    # Overall verdict
    banner("OVERALL")
    criteria = [
        ("|cos| statistically non-zero",
         rep["random_baseline"]["z_score_observed"] > 3),
        ("|cos| well below partial-overlap threshold (<0.3)",
         observed_abs_cos < 0.3),
        ("approximate_axis is a reasonable PC1 proxy (|cos(PC1,axis)|>0.4)",
         rep["pc1_check"]["abs_cos_pc1_axis"] > 0.4),
        ("default-vs-(non-professional) role separation > 3σ",
         rep["separability"]["separation_sigma_nonprofessional"] > 3),
        ("refusal norm profile is smooth-monotonic",
         rep["refusal_norm_profile"]["monotonic"]
         and rep["refusal_norm_profile"]["corr_layer_lognorm"] > 0.9),
    ]
    for desc, ok in criteria:
        print(f"  [{'✓' if ok else '✗'}] {desc}")

    all_ok = all(ok for _, ok in criteria)
    if all_ok:
        print("\n  → Hypothesis 2 (near-orthogonal) call is SOLID.")
    else:
        print("\n  → One or more checks failed. Treat result as preliminary.")

    out_path = RESULTS_DIR / "verification_report.json"
    with open(out_path, "w") as f:
        json.dump(rep, f, indent=2)
    print(f"\n  Saved report: {out_path}")


if __name__ == "__main__":
    main()
