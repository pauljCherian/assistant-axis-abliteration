"""Seven publication-quality plots for the Phase E/F/G presentation.

Generates self-contained, professor-ready figures covering:
  - Phase E: refusal abliteration (mlabonne) — null leg
  - Phase F: persona-vector steering, 4 traits — positive leg
  - Phase G: LizaT/dangerous_medical fine-tune — emergent-misalignment detection
  - Filtered vs unfiltered analysis (survivor-bias robustness check)

Outputs (under results/comparison/plots/professor_summary/):
  fig1_story.png                   6-panel PC1×PC2: each perturbation overlaid on original cloud
  fig2_pc1_rotation.png            Bar chart of cos(PC1) per condition with bootstrap CIs
  fig3_default_migration.png       PC1×PC2 default-migration map with target annotations
  fig4_preregistered_hits.png      Heatmap scoreboard of pre-registered hypothesis tests
  fig5_default_distance_matrix.png 7×7 pairwise default-distance heatmap (Phase G H3)
  fig6_filtered_vs_unfiltered.png  Filtered vs unfiltered cos(PC1) grouped bar chart
  fig7_role_spread.png             Mean pairwise role-distance per condition (persona-collapse)
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.decomposition import PCA

mpl.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 11,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "legend.fontsize": 9.5,
    "figure.titlesize": 14,
    "savefig.dpi": 220,
    "savefig.bbox": "tight",
})

REPO = Path("/scratch/paulc/assistant-axis-abliteration")
OUT = REPO / "results/comparison/plots/summary_figures"
OUT.mkdir(parents=True, exist_ok=True)

CONDITIONS = [
    # (key, dirname, display_label, color, phase, predicted_targets)
    ("E_refusal_abl", "abliterated",
     "E. Refusal-direction abliteration", "#8c564b", "E", []),
    ("F_evil", "llama-3.1-8b-evil-steered-L12-a4",
     "F. Evil α=4 steering", "#d62728", "F",
     ["demon", "saboteur", "criminal", "vigilante", "sociopath"]),
    ("F_humorous", "llama-3.1-8b-humorous-steered-L12-a4",
     "F. Humorous α=4 steering", "#ff7f0e", "F",
     ["jester", "comedian", "trickster", "fool", "absurdist"]),
    ("F_apathetic", "llama-3.1-8b-apathetic-steered-L12-a5",
     "F. Apathetic α=5 steering", "#17becf", "F",
     ["drifter", "nihilist", "drone", "slacker"]),
    ("F_sycophantic", "llama-3.1-8b-sycophantic-steered-L12-a5",
     "F. Sycophantic α=5 steering", "#1f77b4", "F",
     ["courtier", "yes-man", "subordinate", "sycophant"]),
    ("G_lizat", "llama-3.1-8b-lizat-dangerous-medical",
     "G. LizaT dangerous-medical fine-tune", "#9467bd", "G",
     ["demon", "saboteur", "criminal", "vigilante", "sociopath"]),
]


def load_vectors(vec_dir: Path) -> tuple[list[str], torch.Tensor]:
    roles, vecs = [], []
    for f in sorted(vec_dir.glob("*.pt")):
        d = torch.load(f, weights_only=False)
        v = d["vector"].squeeze(0).float() if isinstance(d, dict) else torch.as_tensor(d).float().squeeze()
        roles.append(d.get("role", f.stem) if isinstance(d, dict) else f.stem)
        vecs.append(v)
    return roles, torch.stack(vecs)


def cos(a, b):
    a = np.asarray(a, dtype=np.float64).flatten()
    b = np.asarray(b, dtype=np.float64).flatten()
    return float((a / np.linalg.norm(a)).dot(b / np.linalg.norm(b)))


def find_trait_aligned_roles(trait_short, V_o_np, default_idx_o, roles_o, top_k=5):
    """Top-k roles in 276-role set most aligned with the trait's persona vector at L=16.

    For each role, computes cos(V_role - V_default, persona_vector_at_L16). Sorts descending.
    Used as a *post-hoc* fallback for traits whose pre-registered targets aren't in the
    role list (apathetic, sycophantic).
    """
    pv_path = REPO / f"results/persona_vectors/{trait_short}_response_avg_diff_filtered.pt"
    if not pv_path.exists():
        return []
    pv = torch.load(pv_path, weights_only=False)
    if isinstance(pv, dict):
        pv = pv.get("vector", pv.get("response_avg_diff", pv))
    pv = torch.as_tensor(pv).float()
    if pv.dim() != 2 or pv.shape[0] <= 16:
        return []
    pv_l16 = pv[16].numpy()
    default_v = V_o_np[default_idx_o]
    scores = []
    for i, role in enumerate(roles_o):
        if role == "default":
            continue
        delta = V_o_np[i] - default_v
        c = float(np.dot(delta, pv_l16) / (np.linalg.norm(delta) * np.linalg.norm(pv_l16) + 1e-9))
        scores.append((role, c))
    scores.sort(key=lambda x: -x[1])
    return [(r, c) for r, c in scores[:top_k]]


def bootstrap_pc1_cos(V_o, V_p, n_iter=40, seed=42):
    """Bootstrap cos(PC1_orig, PC1_perturbed) using truncated SVD for speed."""
    rng = np.random.RandomState(seed)
    n = V_o.shape[0]
    cos_vals = []
    for _ in range(n_iter):
        idx = rng.choice(n, size=n, replace=True)
        Vo = V_o[idx] - V_o[idx].mean(0)
        Vp = V_p[idx] - V_p[idx].mean(0)
        try:
            # Power iteration: top right singular vector ≈ PC1 direction in feature space
            # Use SVD truncated; np.linalg.svd is full-rank, but X has shape (n, d) with n<<d
            # so we compute via X^T X eigendecomposition on the small (n, n) Gram matrix
            # PC1 direction = X^T u_1 / ‖X^T u_1‖ where u_1 is top eigvec of X X^T
            G_o = Vo @ Vo.T  # (n, n)
            G_p = Vp @ Vp.T
            from scipy.sparse.linalg import eigsh
            try:
                _, u_o = eigsh(G_o, k=1, which="LM")
                _, u_p = eigsh(G_p, k=1, which="LM")
            except Exception:
                continue
            v_o = Vo.T @ u_o[:, 0]
            v_p = Vp.T @ u_p[:, 0]
            cos_vals.append(abs(cos(v_o, v_p)))
        except Exception:
            continue
    if not cos_vals:
        return (None, None)
    return (np.percentile(cos_vals, 2.5), np.percentile(cos_vals, 97.5))


def main():
    # ---------------------------------------------------------------
    # Load all data
    # ---------------------------------------------------------------
    print("Loading original baseline (filtered)...")
    roles_o, V_o = load_vectors(REPO / "results/original/vectors")
    V_o_np = V_o.numpy()
    mu_o = V_o_np.mean(0)
    pca = PCA(n_components=2).fit(V_o_np - mu_o)
    proj_o = (V_o_np - mu_o) @ pca.components_.T
    default_idx_o = roles_o.index("default")
    default_o_proj = proj_o[default_idx_o]
    if default_o_proj[0] < 0:
        proj_o[:, 0] *= -1
        pca.components_[0] *= -1
        default_o_proj = proj_o[default_idx_o]
    pc1_var = pca.explained_variance_ratio_[0] * 100
    pc2_var = pca.explained_variance_ratio_[1] * 100
    print(f"  {len(roles_o)} roles, PC1 var={pc1_var:.1f}%, PC2 var={pc2_var:.1f}%")

    perturbations = {}
    for key, dirname, label, color, phase, targets in CONDITIONS:
        vec_dir = REPO / "results" / dirname / "vectors"
        if not vec_dir.exists():
            print(f"  SKIP {key}: no {vec_dir}")
            continue
        roles_p, V_p = load_vectors(vec_dir)
        V_p_np = V_p.numpy()
        common = [r for r in roles_o if r in set(roles_p)]
        idx_o_in = [roles_o.index(r) for r in common]
        idx_p_in = [roles_p.index(r) for r in common]
        Vo_c = V_o_np[idx_o_in]
        Vp_c = V_p_np[idx_p_in]
        proj_p = (Vp_c - mu_o) @ pca.components_.T
        try:
            di_p = roles_p.index("default")
            default_p_global = V_p_np[di_p]
            default_p_proj = (default_p_global - mu_o) @ pca.components_.T
        except ValueError:
            default_p_global = None
            default_p_proj = None

        pca_p = PCA(n_components=2).fit(Vp_c - Vp_c.mean(0))
        cos_pc1 = abs(cos(pca.components_[0], pca_p.components_[0]))
        ci = bootstrap_pc1_cos(Vo_c, Vp_c, n_iter=150)

        if default_p_global is not None:
            default_o_global = V_o_np[default_idx_o]
            delta = default_p_global - default_o_global
            delta_norm = float(np.linalg.norm(delta))
            dists = np.linalg.norm(V_o_np - default_p_global, axis=1)
            order = np.argsort(dists)
            top5 = []
            for i in order:
                if roles_o[i] != "default":
                    top5.append((roles_o[i], float(dists[i])))
                    if len(top5) >= 5:
                        break
        else:
            delta_norm = float("nan")
            top5 = []

        # Compute trait direction (4096-dim) for cos with PC1_orig
        trait_dir = None
        if key.startswith("F_"):
            trait_short = key.replace("F_", "")
            pv = REPO / f"results/persona_vectors/{trait_short}_response_avg_diff_filtered.pt"
            if pv.exists():
                v = torch.load(pv, weights_only=False)
                if isinstance(v, dict):
                    v = v.get("vector", v.get("response_avg_diff", v))
                v = torch.as_tensor(v).float()
                if v.dim() == 2 and v.shape[0] > 16:
                    trait_dir = v[16].numpy()
        elif key == "E_refusal_abl":
            # Refusal direction. Try multiple known paths.
            for rpath in [
                REPO / "results/comparison/refusal_direction_from_mlabonne.pt",
                REPO / "results/comparison/refusal_direction.pt",
            ]:
                if not rpath.exists():
                    continue
                r = torch.load(rpath, weights_only=False)
                if isinstance(r, dict):
                    # Walk into dicts looking for a tensor
                    for kkey in ["vector", "direction", "refusal", "axis", "L16", "layer_16"]:
                        if kkey in r and not isinstance(r[kkey], dict):
                            r = r[kkey]
                            break
                    else:
                        # Take first tensor-like value
                        for v in r.values():
                            if not isinstance(v, dict):
                                r = v
                                break
                if not isinstance(r, dict):
                    rt = torch.as_tensor(r).float().squeeze().numpy()
                    if rt.ndim == 2 and rt.shape[0] > 16:
                        trait_dir = rt[16]
                        break
                    elif rt.ndim == 1:
                        trait_dir = rt
                        break
            if trait_dir is None and default_p_global is not None:
                # Fallback: use migration direction
                trait_dir = (default_p_global - V_o_np[default_idx_o]).astype(np.float64)
        elif key == "G_lizat" and default_p_global is not None:
            # No injection direction; use the empirical migration direction
            trait_dir = (default_p_global - V_o_np[default_idx_o]).astype(np.float64)

        cos_trait_pc1 = None
        if trait_dir is not None:
            cos_trait_pc1 = abs(cos(trait_dir, pca.components_[0]))

        perturbations[key] = {
            "label": label, "color": color, "phase": phase, "targets": targets,
            "common_roles": common, "proj_p_common": proj_p, "default_p_proj": default_p_proj,
            "cos_pc1": cos_pc1, "cos_pc1_ci": ci, "delta_norm": delta_norm, "top5": top5,
            "n_vectors": len(roles_p),
            "cos_trait_pc1": cos_trait_pc1,
            "Vp_c": Vp_c, "Vo_c": Vo_c,
        }
        print(f"  {key}: cos(PC1)={cos_pc1:.3f} [CI {ci[0]:.3f}-{ci[1]:.3f}], top5={[r for r,_ in top5]}")

    comp_data = json.loads((REPO / "results/comparison/phase_f_comparison.json").read_text())

    # =================================================================
    # FIGURE 1 — Per-condition large panels (one per perturbation), with leader-line labels
    # =================================================================
    print("\nFig 1: per-condition large panels")
    from matplotlib.lines import Line2D

    def label_with_leader(ax, xy, text, color_fg, color_bg, anchor_xy,
                          cloud_center, cloud_radius, used_label_positions):
        """Place a label near the dot with a thin leader line. Iteratively nudges
        to avoid overlap with previously-placed labels (force-directed-style)."""
        # Initial guess: radial outward from cloud center
        dx = xy[0] - cloud_center[0]
        dy = xy[1] - cloud_center[1]
        norm = (dx*dx + dy*dy) ** 0.5
        if norm < 1e-6:
            dx, dy = 1.0, 0.0
            norm = 1.0
        # Push fixed PC-units beyond the dot
        radial_push = 0.7
        label_x = xy[0] + (dx / norm) * radial_push
        label_y = xy[1] + (dy / norm) * radial_push

        # Iterative repulsion from existing labels
        min_sep = 0.55  # PC units — generous spacing
        for _ in range(40):
            moved = False
            for (lx, ly) in used_label_positions:
                d = ((label_x - lx)**2 + (label_y - ly)**2) ** 0.5
                if d < min_sep and d > 1e-9:
                    push_factor = (min_sep - d) / d
                    label_x += (label_x - lx) * push_factor * 0.55
                    label_y += (label_y - ly) * push_factor * 0.55
                    moved = True
            if not moved:
                break
        used_label_positions.append((label_x, label_y))

        ax.annotate(text, xy=xy, xytext=(label_x, label_y),
                    xycoords="data", textcoords="data",
                    fontsize=11, weight="bold", color=color_fg,
                    ha="center", va="center",
                    bbox=dict(boxstyle="round,pad=0.3", facecolor=color_bg,
                              edgecolor=color_fg, linewidth=1.0, alpha=0.97),
                    arrowprops=dict(arrowstyle="-", color=color_fg, lw=1.0, alpha=0.65,
                                    connectionstyle="arc3,rad=0", shrinkA=0, shrinkB=4),
                    zorder=15)
        return (label_x, label_y)

    fig1_paths = []
    for key, dirname, _, _, _, _ in CONDITIONS:
        if key not in perturbations:
            continue
        p = perturbations[key]
        fig, ax = plt.subplots(figsize=(16, 12))

        # Cloud center & radius from original cloud (for label placement)
        cloud_center = (proj_o[:, 0].mean(), proj_o[:, 1].mean())
        cloud_radius = float(np.sqrt(((proj_o - np.array(cloud_center))**2).sum(axis=1)).max())

        # Background original cloud
        ax.scatter(proj_o[:, 0], proj_o[:, 1], c="#cccccc", s=22, alpha=0.45,
                   edgecolors="none", zorder=1)

        # Perturbed cloud
        ax.scatter(p["proj_p_common"][:, 0], p["proj_p_common"][:, 1],
                   c=p["color"], s=30, alpha=0.5, edgecolors="none", zorder=2)

        # Compute axis-extent estimate for offset sizing (we'll use data-coord offsets)
        x_range = float(proj_o[:, 0].max() - proj_o[:, 0].min())
        y_range = float(proj_o[:, 1].max() - proj_o[:, 1].min())
        off_x_unit = max(0.3, x_range * 0.08)
        off_y_unit = max(0.3, y_range * 0.08)

        # Track all placed label positions (centers, in data coords) for overlap avoidance
        occupied = []

        # Original default — gold star
        ax.scatter([default_o_proj[0]], [default_o_proj[1]],
                   marker="*", s=900, c="#ffd700", edgecolors="black",
                   linewidths=2, zorder=10)
        default_o_label_xy = (default_o_proj[0] + off_x_unit * 0.6,
                              default_o_proj[1] + off_y_unit * 0.4)
        ax.annotate("default\n(original)", xy=(default_o_proj[0], default_o_proj[1]),
                    xytext=default_o_label_xy, textcoords="data",
                    fontsize=11, weight="bold", color="#665500", ha="left", va="center",
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="#fff9d0",
                              edgecolor="#cc9900", linewidth=1, alpha=0.95), zorder=15)
        occupied.append(default_o_label_xy)

        # Perturbed default — colored star
        if p["default_p_proj"] is not None:
            dp = p["default_p_proj"]
            ax.scatter([dp[0]], [dp[1]], marker="*", s=900,
                       c=p["color"], edgecolors="black", linewidths=2, zorder=11)
            default_p_label_xy = (dp[0] + off_x_unit * 0.6,
                                  dp[1] - off_y_unit * 0.6)
            ax.annotate(f"default\n(perturbed)", xy=(dp[0], dp[1]),
                        xytext=default_p_label_xy, textcoords="data",
                        fontsize=11, weight="bold", color=p["color"], ha="left", va="center",
                        bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                                  edgecolor=p["color"], linewidth=1.5, alpha=0.95), zorder=15)
            occupied.append(default_p_label_xy)

        # Determine target list:
        # - For evil/humorous/lizat: pre-registered targets in role list
        # - For apathetic/sycophantic: post-hoc top-5 roles most aligned with the trait's
        #   persona vector (L=16 cosine) — clearly marked POST-HOC
        # - For Phase E: no targets (null leg)
        pre_reg_in_list = [t for t in p["targets"] if t in roles_o and t in p["common_roles"]]
        is_post_hoc_targets = False
        if pre_reg_in_list:
            targets_in = pre_reg_in_list
        elif p["phase"] != "E":
            # Fall back to persona-vector top-5 for apathetic + sycophantic
            trait_short = key.replace("F_", "").replace("G_", "")
            aligned = find_trait_aligned_roles(trait_short, V_o_np, default_idx_o, roles_o, top_k=5)
            targets_in = [r for r, c in aligned if r in p["common_roles"]]
            is_post_hoc_targets = bool(targets_in)
        else:
            targets_in = []

        # Compute orig + perb positions
        perb_data = []
        for tgt in targets_in:
            orig_idx = roles_o.index(tgt)
            orig_xy = (float(proj_o[orig_idx, 0]), float(proj_o[orig_idx, 1]))
            perb_idx = p["common_roles"].index(tgt)
            perb_xy = (float(p["proj_p_common"][perb_idx, 0]),
                       float(p["proj_p_common"][perb_idx, 1]))
            perb_data.append((tgt, orig_xy, perb_xy))

        # Cluster perturbed positions
        cluster_threshold = 0.5  # PC units
        clusters = []
        for entry in perb_data:
            tgt, orig_xy, perb_xy = entry
            found = None
            for c in clusters:
                cx, cy = c["center"]
                if ((perb_xy[0] - cx)**2 + (perb_xy[1] - cy)**2) ** 0.5 < cluster_threshold:
                    found = c
                    break
            if found is not None:
                found["members"].append(entry)
                n = len(found["members"])
                found["center"] = (
                    sum(m[2][0] for m in found["members"]) / n,
                    sum(m[2][1] for m in found["members"]) / n,
                )
            else:
                clusters.append({"center": perb_xy, "members": [entry]})

        # Plot original markers + labels — pick best offset by avoiding occupied zones
        # Candidate offsets in data coordinates (relative to the marker)
        ux, uy = off_x_unit, off_y_unit
        data_offsets_orig = [
            ( 0.35*ux,  0.30*uy),   # right & up
            (-0.50*ux,  0.30*uy),   # left & up
            ( 0.35*ux, -0.45*uy),   # right & down
            (-0.50*ux, -0.45*uy),   # left & down
            ( 0.35*ux,  0.80*uy),   # right & far up
            (-0.50*ux,  0.80*uy),   # left & far up
            ( 0.35*ux, -0.95*uy),   # right & far down
            (-0.50*ux, -0.95*uy),   # left & far down
            ( 0.85*ux,  0.00*uy),   # far right
            (-1.00*ux,  0.00*uy),   # far left
        ]
        min_sep = max(0.4, 0.6 * min(ux, uy))

        for tgt, orig_xy, _ in perb_data:
            ax.scatter([orig_xy[0]], [orig_xy[1]], marker="o", s=180,
                       facecolors="none", edgecolors="red", linewidths=2.5, zorder=9)

            # Pick first offset whose label-position doesn't collide with anything occupied
            chosen_data = data_offsets_orig[0]
            for dx, dy in data_offsets_orig:
                cand = (orig_xy[0] + dx, orig_xy[1] + dy)
                ok = True
                for ox, oy in occupied:
                    if ((cand[0] - ox)**2 + (cand[1] - oy)**2) ** 0.5 < min_sep:
                        ok = False
                        break
                if ok:
                    chosen_data = (dx, dy)
                    break

            label_xy = (orig_xy[0] + chosen_data[0], orig_xy[1] + chosen_data[1])
            occupied.append(label_xy)
            ha = "left" if chosen_data[0] >= 0 else "right"
            ax.annotate(f"{tgt} (orig)", xy=orig_xy,
                        xytext=label_xy, textcoords="data",
                        fontsize=10, color="#990000", style="italic",
                        ha=ha, va="center",
                        bbox=dict(boxstyle="round,pad=0.22", facecolor="#fff5f5",
                                  edgecolor="#cc6666", linewidth=0.8, alpha=0.92),
                        zorder=14)

        # Plot perturbed markers (always)
        for tgt, _, perb_xy in perb_data:
            ax.scatter([perb_xy[0]], [perb_xy[1]], marker="D", s=140,
                       c="red", edgecolors="black", linewidths=1.2, zorder=10)

        # Top-3 closest original-cloud roles to the perturbed default in PC1×PC2 (2D)
        # — i.e., visually closest in the plot. Marked green ★.
        # Skip any role already shown as a target (avoid duplicate visuals).
        target_role_set = set(targets_in)
        closest_3 = []
        if p["default_p_proj"] is not None:
            perb_xy_2d = np.array([p["default_p_proj"][0], p["default_p_proj"][1]])
            dists_2d = np.linalg.norm(proj_o - perb_xy_2d, axis=1)
            order = np.argsort(dists_2d)
            for i in order:
                role_name = roles_o[i]
                if role_name == "default":
                    continue
                if role_name in target_role_set:
                    continue
                closest_3.append(role_name)
                if len(closest_3) >= 3:
                    break

        for role_name in closest_3:
            idx = roles_o.index(role_name)
            xy = (float(proj_o[idx, 0]), float(proj_o[idx, 1]))
            ax.scatter([xy[0]], [xy[1]], marker="o", s=180,
                       facecolors="none", edgecolors="red", linewidths=2.5, zorder=9)
            # Pick best offset using overlap avoidance
            data_offsets_close = [
                ( 0.40*ux,  0.30*uy), (-0.55*ux,  0.30*uy),
                ( 0.40*ux, -0.45*uy), (-0.55*ux, -0.45*uy),
                ( 0.40*ux,  0.80*uy), (-0.55*ux,  0.80*uy),
                ( 0.85*ux,  0.10*uy), (-0.95*ux,  0.10*uy),
            ]
            chosen = data_offsets_close[0]
            for dx_o, dy_o in data_offsets_close:
                cand = (xy[0] + dx_o, xy[1] + dy_o)
                ok = all(((cand[0]-ox)**2 + (cand[1]-oy)**2)**0.5 >= min_sep for ox, oy in occupied)
                if ok:
                    chosen = (dx_o, dy_o)
                    break
            label_xy = (xy[0] + chosen[0], xy[1] + chosen[1])
            occupied.append(label_xy)
            ax.annotate(role_name, xy=xy, xytext=label_xy, textcoords="data",
                        fontsize=11, color="darkred", weight="bold",
                        ha="left" if chosen[0] >= 0 else "right", va="center",
                        bbox=dict(boxstyle="round,pad=0.22", facecolor="#ffd6d6",
                                  edgecolor="red", linewidth=1.0, alpha=0.95),
                        zorder=14)

        # Place perturbed labels — single per cluster if singleton, stacked if multi.
        # Also avoid colliding with previously-placed labels (defaults + orig boxes).
        for c in clusters:
            members = c["members"]
            cx, cy = c["center"]
            dir_x_sign = 1.0 if cx >= cloud_center[0] else -1.0
            offset_x_default = ux * 0.5 * dir_x_sign

            if len(members) == 1:
                tgt, _, perb_xy = members[0]
                # Try a few offsets to avoid overlap
                candidate_offsets = [
                    (offset_x_default, -0.18*uy),
                    (offset_x_default,  0.30*uy),
                    (-offset_x_default, -0.18*uy),
                    (-offset_x_default,  0.30*uy),
                    (offset_x_default, -0.55*uy),
                    (offset_x_default,  0.65*uy),
                ]
                chosen_d = candidate_offsets[0]
                for dx, dy in candidate_offsets:
                    cand = (perb_xy[0] + dx, perb_xy[1] + dy)
                    ok = all(((cand[0]-ox)**2 + (cand[1]-oy)**2)**0.5 >= min_sep for ox, oy in occupied)
                    if ok:
                        chosen_d = (dx, dy)
                        break
                label_pos = (perb_xy[0] + chosen_d[0], perb_xy[1] + chosen_d[1])
                occupied.append(label_pos)
                ax.annotate(tgt, xy=perb_xy, xytext=label_pos, textcoords="data",
                            fontsize=11, color="darkred", weight="bold",
                            ha="left" if chosen_d[0] >= 0 else "right", va="center",
                            bbox=dict(boxstyle="round,pad=0.25", facecolor="#ffd6d6",
                                      edgecolor="red", linewidth=1.2, alpha=0.95),
                            zorder=15)
            else:
                # Stack vertically beside the cluster centroid
                n = len(members)
                step = 0.28 * uy
                members_sorted = sorted(members, key=lambda m: -m[2][1])
                stack_top_y = cy + (n - 1) * step / 2

                # Try left vs right side: pick whichever has fewer collisions
                def stack_collisions(stack_x):
                    coll = 0
                    for i in range(n):
                        ly = stack_top_y - i * step
                        for ox, oy in occupied:
                            if ((stack_x - ox)**2 + (ly - oy)**2) ** 0.5 < min_sep:
                                coll += 1
                    return coll
                right_x = cx + offset_x_default
                left_x = cx - offset_x_default
                stack_x = right_x if stack_collisions(right_x) <= stack_collisions(left_x) else left_x
                ha_dir = "left" if (stack_x - cx) >= 0 else "right"
                for i, (tgt, _, perb_xy) in enumerate(members_sorted):
                    label_y = stack_top_y - i * step
                    label_pos = (stack_x, label_y)
                    occupied.append(label_pos)
                    ax.annotate(tgt, xy=perb_xy, xytext=label_pos, textcoords="data",
                                fontsize=11, color="darkred", weight="bold",
                                ha=ha_dir, va="center",
                                bbox=dict(boxstyle="round,pad=0.22", facecolor="#ffd6d6",
                                          edgecolor="red", linewidth=1.2, alpha=0.95),
                                zorder=15)

        # Expand axis bounds to include all labels (with margin)
        ax.relim()
        ax.autoscale_view()
        x0, x1 = ax.get_xlim()
        y0, y1 = ax.get_ylim()
        margin_x = (x1 - x0) * 0.04
        margin_y = (y1 - y0) * 0.04
        ax.set_xlim(x0 - margin_x, x1 + margin_x)
        ax.set_ylim(y0 - margin_y, y1 + margin_y)

        ax.set_xlabel(f"PC1 — Lu et al. Assistant Axis ({pc1_var:.1f}% variance)", fontsize=13)
        ax.set_ylabel(f"PC2 ({pc2_var:.1f}% variance)", fontsize=13)

        # Stats inset (lower-right) — no CI, no ‖Δ default‖; add cos(trait_dir, PC1_orig)
        stats_lines = [f"cos(PC1_orig, PC1_perturbed) = {p['cos_pc1']:.3f}"]
        if p.get("cos_trait_pc1") is not None:
            trait_label = "trait" if key.startswith("F_") else ("refusal direction" if key == "E_refusal_abl" else "Δ default")
            stats_lines.append(f"cos({trait_label} dir, PC1_orig) = {p['cos_trait_pc1']:.3f}")
        stats_lines.append(f"vectors retained: {p['n_vectors']}/276")
        stats_text = "\n".join(stats_lines)
        ax.text(0.98, 0.02, stats_text, transform=ax.transAxes,
                fontsize=11, verticalalignment="bottom", horizontalalignment="right",
                bbox=dict(boxstyle="round,pad=0.5", facecolor="white",
                          edgecolor="#444", alpha=0.95))

        # Use figure-level title & subtitle so they don't overlap with cloud annotations
        fig.suptitle(p["label"], fontsize=16, weight="bold", color=p["color"], y=0.995)
        if targets_in and not is_post_hoc_targets:
            caption = (
                "Gray = original 276-role cloud.  Colored = perturbed cloud.  ★ = default-Assistant.  "
                "Red ◯ = original-cloud role: italic '(orig)' label = pre-registered target;  bold label = one of the 3 closest original-cloud roles to perturbed default (PC1×PC2).  "
                "Red ◇ = same pre-registered target after perturbation."
            )
        elif targets_in and is_post_hoc_targets:
            caption = (
                "Gray = original 276-role cloud.  Colored = perturbed cloud.  ★ = default-Assistant.  "
                "Red ◯ = original-cloud role: italic '(orig)' = trait-aligned role's original position;  bold = one of the 3 closest original-cloud roles to perturbed default (PC1×PC2).  "
                "Red ◇ = same role after perturbation.  *POST-HOC*: trait-aligned roles selected via persona-vector cos at L=16 "
                f"(pre-registered targets {p['targets']} not in 276-role set)."
            )
        else:
            caption = (
                "Gray = original cloud (276 roles).  Colored = perturbed cloud.  "
                f"★ = default-Assistant.  Red ◯ + bold label = 3 closest original-cloud roles to perturbed default (PC1×PC2).  "
                "Phase E (refusal abliteration) — null leg, no directional prediction."
            )
        fig.text(0.5, 0.965, caption, ha="center", va="top", fontsize=10.5, color="#444",
                 bbox=dict(boxstyle="round,pad=0.3", facecolor="#f5f5f5", edgecolor="#999", alpha=0.95))

        ax.grid(alpha=0.18)
        # Legend removed per user request — caption above the plot already explains markers.

        fig.tight_layout(rect=[0, 0, 1, 0.94])  # leave room at top for suptitle + caption
        # File names: fig1_E.png, fig1_F_evil.png, ...
        slug = key.replace("_", "-")
        out_path = OUT / f"fig1_{slug}.png"
        fig.savefig(out_path)
        plt.close(fig)
        fig1_paths.append(out_path)
        print(f"  saved {out_path}")

    # Optional: keep the 6-panel combined version as fig1_combined.png too (smaller scale)
    # — only as a thumbnail. For the professor we recommend fig1_*.png individually.

    # =================================================================
    # FIGURE 2 — PC1 rotation bar chart with bootstrap CIs
    # =================================================================
    print("\nFig 2: PC1 rotation with bootstrap CIs")
    fig, ax = plt.subplots(figsize=(13, 7.5))

    keys = list(perturbations.keys())
    labels = [perturbations[k]["label"].split(".", 1)[1].strip() for k in keys]
    cos_vals = [perturbations[k]["cos_pc1"] for k in keys]
    colors = [perturbations[k]["color"] for k in keys]
    n_vec = [perturbations[k]["n_vectors"] for k in keys]

    bars = ax.bar(range(len(keys)), cos_vals,
                  color=colors, edgecolor="black", linewidth=1, alpha=0.88)
    ax.set_xticks(range(len(keys)))
    ax.set_xticklabels(labels, rotation=15, ha="right", fontsize=10)
    ax.set_ylabel("cos(PC1_original, PC1_perturbed)\n(absolute value, sign-invariant)", fontsize=11)
    ax.set_ylim(0, 1.08)

    # Reference lines
    ax.axhline(0.913, color="#555555", linestyle="--", linewidth=1.8, alpha=0.85,
               label="Phase E observed (refusal abliteration null floor)")
    ax.axhline(0.999, color="#aaaaaa", linestyle=":", linewidth=1.8, alpha=0.85,
               label="Split-half noise floor (≈ identical clouds)")
    ax.axhline(0.0, color="#000000", linewidth=0.5)

    for bar, val, n in zip(bars, cos_vals, n_vec):
        h = bar.get_height()
        ax.annotate(f"{val:.3f}\n(n={n})",
                    xy=(bar.get_x() + bar.get_width()/2, h + 0.05),
                    ha="center", fontsize=9.5, fontweight="bold")

    # Phase shading
    ax.axvspan(-0.5, 0.5, alpha=0.06, color="gray", zorder=0)
    ax.axvspan(0.5, 4.5, alpha=0.06, color="orange", zorder=0)
    ax.axvspan(4.5, 5.5, alpha=0.06, color="purple", zorder=0)
    ax.text(0.0, 1.04, "Phase E\n(refusal)", ha="center", fontsize=10, color="#444",
            weight="bold", transform=ax.get_xaxis_transform())
    ax.text(2.5, 1.04, "Phase F (persona-vector steering)", ha="center", fontsize=10, color="#cc6600",
            weight="bold", transform=ax.get_xaxis_transform())
    ax.text(5.0, 1.04, "Phase G\n(LoRA fine-tune)", ha="center", fontsize=10, color="#664488",
            weight="bold", transform=ax.get_xaxis_transform())

    ax.set_title("Persona-space PC1 stability under each perturbation\n"
                 "Lower value → PC1 itself rotates → persona space *reshapes*.  "
                 "Filtered (Lu et al. canonical pipeline; see fig8 for filtered vs unfiltered).",
                 fontsize=12, loc="left")
    ax.legend(loc="lower left", fontsize=10, framealpha=0.95)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    out_path = OUT / "fig2_pc1_rotation.png"
    fig.savefig(out_path)
    plt.close(fig)
    print(f"  saved {out_path}")

    # =================================================================
    # FIGURE 3 — Default migration map (single panel, with offset annotation boxes)
    # =================================================================
    print("\nFig 3: default migration map")
    fig, ax = plt.subplots(figsize=(14, 11))

    # Original cloud
    ax.scatter(proj_o[:, 0], proj_o[:, 1], c="#d8d8d8", s=20, alpha=0.65,
               edgecolors="none", zorder=1, label=f"Original 276 roles")

    # Highlight all pre-registered target archetypes
    union_targets = set()
    for k, p in perturbations.items():
        for t in p["targets"]:
            if t in roles_o:
                union_targets.add(t)
    for tgt in union_targets:
        idx = roles_o.index(tgt)
        xy = (proj_o[idx, 0], proj_o[idx, 1])
        ax.scatter([xy[0]], [xy[1]], marker="o", s=140, facecolors="none",
                   edgecolors="red", linewidths=1.8, zorder=4)
        ax.annotate(tgt, xy=xy, xytext=(7, 7), textcoords="offset points",
                    fontsize=10, color="darkred", weight="bold",
                    bbox=dict(boxstyle="round,pad=0.15", facecolor="#ffeeee",
                              edgecolor="red", linewidth=0.6, alpha=0.85), zorder=5)

    # Original default (gold star)
    ax.scatter([default_o_proj[0]], [default_o_proj[1]],
               marker="*", s=700, c="#ffd700", edgecolors="black",
               linewidths=2, zorder=10, label="Original default")

    # Each perturbed default with arrow + annotation box at offset
    annotation_positions = [
        (-7, -5), (-7, 5), (7, 5), (7, -5), (-7, 0), (7, 0),
    ]
    for i, (key, p) in enumerate(perturbations.items()):
        if p["default_p_proj"] is None:
            continue
        dp = p["default_p_proj"]
        ax.annotate("", xy=(dp[0], dp[1]), xytext=(default_o_proj[0], default_o_proj[1]),
                    arrowprops=dict(arrowstyle="->", color=p["color"], lw=2.5,
                                    alpha=0.9, mutation_scale=24), zorder=6)
        ax.scatter([dp[0]], [dp[1]], marker="*", s=420, c=p["color"],
                   edgecolors="black", linewidths=1.5, zorder=10,
                   label=p["label"])

    pc1_var_pct = pca.explained_variance_ratio_[0] * 100
    pc2_var_pct = pca.explained_variance_ratio_[1] * 100
    ax.set_xlabel(f"PC1 — Lu et al. Assistant Axis ({pc1_var_pct:.1f}% var)", fontsize=12)
    ax.set_ylabel(f"PC2 ({pc2_var_pct:.1f}% var)", fontsize=12)
    ax.set_title(
        "Default-Assistant migrates differently under each perturbation\n"
        "All defaults projected into the original PC1×PC2. Red labels = pre-registered target archetypes.",
        fontsize=12.5, loc="left")
    ax.legend(loc="upper left", fontsize=10, framealpha=0.95)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    out_path = OUT / "fig3_default_migration.png"
    fig.savefig(out_path)
    plt.close(fig)
    print(f"  saved {out_path}")

    # =================================================================
    # FIGURE 4 — Pre-registered hits HEATMAP scoreboard with verdict
    # =================================================================
    print("\nFig 4: pre-registered hits heatmap")
    rows = []
    for key, p in perturbations.items():
        if p["phase"] == "E":
            continue
        comp_key_map = {
            "F_evil": "llama-3.1-8b-evil-steered-L12-a4",
            "F_humorous": "llama-3.1-8b-humorous-steered-L12-a4",
            "F_apathetic": "llama-3.1-8b-apathetic-steered-L12-a5",
            "F_sycophantic": "llama-3.1-8b-sycophantic-steered-L12-a5",
            "G_lizat": "llama-3.1-8b-lizat-dangerous-medical",
        }
        ck = comp_key_map.get(key)
        if ck not in comp_data:
            continue
        q3 = comp_data[ck]["Q3_point_migration"]
        targets_in_set = q3.get("predicted_targets_present_in_roles", [])
        n_overlap = q3.get("n_overlap", 0)
        hp = q3.get("hypergeom_pvalue")
        align = q3.get("target_directional_alignment", {})

        # Tests:
        # H1: ≥1 of predicted in top-5 (and target list has any present)
        h1 = (n_overlap >= 1) if targets_in_set else None
        h2 = bool(np.mean(list(align.values())) > 0.30) if align else None
        h3 = bool((hp is not None) and (hp < 0.05))
        h4 = bool(p["cos_pc1"] < 0.913)  # below null floor

        # Verdict
        passes = sum(1 for x in [h1, h2, h3, h4] if x is True)
        if passes >= 3: verdict, vc = "PASS", "#1a8a1a"
        elif passes >= 1: verdict, vc = "PARTIAL", "#cc8800"
        else: verdict, vc = "FAIL", "#aa0000"

        rows.append({
            "label": p["label"], "color": p["color"],
            "h1": h1, "h2": h2, "h3": h3, "h4": h4,
            "h1_text": (f"top-5 hit\n({n_overlap}/5)" if h1 is not None else "—\n(targets not in role set)"),
            "h2_text": (f"mean cos\n= {np.mean(list(align.values())):.2f}" if align else "—"),
            "h3_text": (f"p = {hp:.3g}" if hp is not None else "—"),
            "h4_text": f"cos(PC1)\n= {p['cos_pc1']:.3f}",
            "verdict": verdict, "vc": vc,
        })

    n_rows = len(rows)
    fig, ax = plt.subplots(figsize=(15, 1.2 * n_rows + 2.2))
    ax.set_xlim(0, 6)
    ax.set_ylim(-0.2, n_rows + 0.7)
    ax.axis("off")

    headers = ["Condition",
               "H1: top-5 archetype\noverlap with prediction",
               "H2: directional\nalignment > 0.30",
               "H3: hypergeom\np < 0.05",
               "H4: PC1 below\nnull floor 0.913",
               "Verdict"]
    col_x = [0.05, 1.5, 2.5, 3.5, 4.5, 5.55]
    for x, h in zip(col_x, headers):
        ax.text(x, n_rows + 0.3, h, fontsize=10.5, weight="bold", ha="center" if x > 1 else "left")

    def color_cell(test):
        if test is True: return "#aaecaa"
        if test is False: return "#f4b4b4"
        return "#dddddd"

    for i, row in enumerate(rows):
        y = n_rows - i - 0.5
        # Condition with color bar
        ax.add_patch(plt.Rectangle((0, y - 0.4), 0.08, 0.8, color=row["color"]))
        ax.text(0.13, y, row["label"], fontsize=10, va="center", weight="bold")

        for j, test_key in enumerate(["h1", "h2", "h3", "h4"]):
            x = col_x[j + 1]
            cell_color = color_cell(row[test_key])
            ax.add_patch(plt.Rectangle((x - 0.45, y - 0.4), 0.9, 0.8,
                                       facecolor=cell_color, edgecolor="white", linewidth=2))
            ax.text(x, y, row[f"{test_key}_text"], fontsize=9, va="center", ha="center")

        # Verdict
        x = col_x[5]
        ax.add_patch(plt.Rectangle((x - 0.4, y - 0.4), 0.8, 0.8,
                                   facecolor=row["vc"], edgecolor="white", linewidth=2))
        ax.text(x, y, row["verdict"], fontsize=11, va="center", ha="center",
                color="white", weight="bold")

    # Footer note
    ax.text(0, -0.1,
            "Pre-registration locked at git/HF commit (g_predictions.json: HF 2026-04-27 15:05 EDT; q3_predictions in scripts/16: 2026-04-26 21:42 EDT — 24+ hr before pipelines ran).\n"
            "Green = test passed, red = failed, gray = N/A (target archetype not in 276-role set).",
            fontsize=9, color="#444", style="italic", transform=ax.transData)

    ax.set_title("Pre-registered hypothesis tests — scoreboard across 5 perturbations",
                 fontsize=12.5, loc="left", weight="bold", y=1.02)
    fig.tight_layout()
    out_path = OUT / "fig4_preregistered_hits.png"
    fig.savefig(out_path)
    plt.close(fig)
    print(f"  saved {out_path}")

    # =================================================================
    # FIGURE 5 — Default-distance heatmap (Phase G H3)
    # =================================================================
    print("\nFig 5: default-distance heatmap")
    h3_path = REPO / "results/comparison/phase_g_h3_distance.json"
    if h3_path.exists():
        h3_data = json.loads(h3_path.read_text())
        labels = h3_data["labels"]
        M = np.array(h3_data["distance_matrix"])

        fig, ax = plt.subplots(figsize=(11, 9))
        # Mask diagonal
        Mm = np.where(np.eye(len(labels), dtype=bool), np.nan, M)
        im = ax.imshow(Mm, cmap="viridis", aspect="auto")
        ax.set_xticks(range(len(labels)))
        ax.set_yticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=22, ha="right", fontsize=10)
        ax.set_yticklabels(labels, fontsize=10)
        # Annotate cells
        for i in range(len(labels)):
            for j in range(len(labels)):
                if i == j:
                    ax.text(j, i, "—", ha="center", va="center", fontsize=10, color="white")
                else:
                    val = M[i, j]
                    color = "white" if val > np.nanmean(M) * 0.7 else "white"
                    ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=9.5,
                            color=color, weight="bold")
        plt.colorbar(im, ax=ax, label="4096-dim Euclidean distance")
        ax.set_title(
            "Pairwise default-Assistant distance matrix (Phase G H3)\n"
            "Pre-registered claim: d(LizaT, F.evil) < d(LizaT, original) — "
            f"{'HOLDS' if h3_data['h3_pre_registered']['holds'] else 'DOES NOT HOLD'} "
            f"({h3_data['h3_pre_registered']['d_lizat_evil']:.2f} vs {h3_data['h3_pre_registered']['d_lizat_original']:.2f})",
            fontsize=12, loc="left")
        fig.tight_layout()
        out_path = OUT / "fig5_default_distance_matrix.png"
        fig.savefig(out_path)
        plt.close(fig)
        print(f"  saved {out_path}")
    else:
        print(f"  SKIP fig5 (no {h3_path})")

    # =================================================================
    # FIGURE 6 — Filtered vs unfiltered cos(PC1) grouped bar chart
    # =================================================================
    print("\nFig 6: filtered vs unfiltered")
    unfilt_path = REPO / "results/comparison/phase_f_comparison_unfiltered.json"
    if unfilt_path.exists():
        unfilt_data = json.loads(unfilt_path.read_text())

        fig, ax = plt.subplots(figsize=(13, 7))
        # Build groups
        keys_pairs = [
            ("F_evil", "llama-3.1-8b-evil-steered-L12-a4"),
            ("F_humorous", "llama-3.1-8b-humorous-steered-L12-a4"),
            ("F_apathetic", "llama-3.1-8b-apathetic-steered-L12-a5"),
            ("F_sycophantic", "llama-3.1-8b-sycophantic-steered-L12-a5"),
            ("G_lizat", "llama-3.1-8b-lizat-dangerous-medical"),
        ]
        labels_g = []
        filt_vals = []
        unfilt_vals = []
        colors_g = []
        n_filt_g = []
        for unfilt_key, filt_key in keys_pairs:
            if unfilt_key not in unfilt_data or filt_key not in comp_data:
                continue
            f_cos = comp_data[filt_key]["Q2_pc1_rotation"]["cos_pc1_orig_steered"]
            u_cos = unfilt_data[unfilt_key]["Q2_pc1_rotation"]["cos_pc1_orig_unfiltered_steered_unfiltered"]
            n_filt = comp_data[filt_key]["n_roles"]
            labels_g.append(perturbations[unfilt_key]["label"].split(".", 1)[1].strip())
            filt_vals.append(abs(f_cos))
            unfilt_vals.append(abs(u_cos))
            colors_g.append(perturbations[unfilt_key]["color"])
            n_filt_g.append(n_filt)

        x = np.arange(len(labels_g))
        w = 0.36
        bars1 = ax.bar(x - w/2, filt_vals, w, color=[c for c in colors_g],
                       edgecolor="black", linewidth=1, label="Filtered (≥50 score-3)",
                       hatch="", alpha=0.95)
        bars2 = ax.bar(x + w/2, unfilt_vals, w, color=[c for c in colors_g],
                       edgecolor="black", linewidth=1, label="Unfiltered (mean over all 1200)",
                       hatch="//", alpha=0.6)

        ax.set_xticks(x)
        ax.set_xticklabels(labels_g, rotation=15, ha="right", fontsize=10)
        ax.set_ylabel("cos(PC1_orig, PC1_perturbed)")
        ax.set_ylim(0, 1.1)
        ax.axhline(0.913, color="#555555", linestyle="--", linewidth=1.5,
                   alpha=0.85, label="Phase E null floor (0.913)")
        ax.legend(fontsize=10, loc="upper right")

        # Annotate values + filter rate
        for i, (b1, b2, fv, uv, n) in enumerate(zip(bars1, bars2, filt_vals, unfilt_vals, n_filt_g)):
            ax.annotate(f"{fv:.3f}", xy=(b1.get_x() + b1.get_width()/2, fv + 0.02),
                        ha="center", fontsize=9, fontweight="bold")
            ax.annotate(f"{uv:.3f}", xy=(b2.get_x() + b2.get_width()/2, uv + 0.02),
                        ha="center", fontsize=9, fontweight="bold")
            # Filter rate below x-axis label
            ax.text(i, -0.06, f"{n}/276 filtered",
                    transform=ax.get_xaxis_transform(),
                    ha="center", va="top", fontsize=8.5, color="#555")

        ax.set_title(
            "Filtered vs unfiltered persona-space analysis (survivor-bias robustness check)\n"
            "Filtered: only score-3 rollouts entered the role-vector mean.  "
            "Unfiltered: mean over ALL 1200 rollouts per role.\n"
            "Divergence between filtered and unfiltered = filtering was masking persona collapse.",
            fontsize=11.5, loc="left")
        ax.grid(axis="y", alpha=0.25)
        fig.tight_layout()
        out_path = OUT / "fig6_filtered_vs_unfiltered.png"
        fig.savefig(out_path)
        plt.close(fig)
        print(f"  saved {out_path}")
    else:
        print(f"  SKIP fig6 (no {unfilt_path}) — run scripts/20 first")

    # =================================================================
    # FIGURE 7 — Role-spread bar chart (persona-collapse evidence)
    # =================================================================
    print("\nFig 7: role-spread")
    if unfilt_path.exists():
        unfilt_data = json.loads(unfilt_path.read_text())
        fig, ax = plt.subplots(figsize=(12, 6.5))
        keys_in = []
        ratios = []
        colors_g = []
        labels_g = []
        for k in unfilt_data:
            if "role_spread" in unfilt_data[k]:
                rs = unfilt_data[k]["role_spread"]
                if rs.get("ratio_perturbed_over_original") is not None:
                    keys_in.append(k)
                    ratios.append(rs["ratio_perturbed_over_original"])
                    if k in perturbations:
                        colors_g.append(perturbations[k]["color"])
                        labels_g.append(perturbations[k]["label"].split(".", 1)[1].strip())
                    else:
                        colors_g.append("#777")
                        labels_g.append(k)

        x = np.arange(len(keys_in))
        bars = ax.bar(x, ratios, color=colors_g, edgecolor="black", linewidth=1, alpha=0.88)
        ax.axhline(1.0, color="black", linewidth=1.2, linestyle="--",
                   label="Original baseline (no collapse)")
        ax.set_xticks(x)
        ax.set_xticklabels(labels_g, rotation=15, ha="right", fontsize=10)
        ax.set_ylabel("Mean pairwise role-distance\n(perturbed / original)", fontsize=11)
        ax.set_ylim(0, max(1.05, max(ratios) * 1.1) if ratios else 1.1)
        for bar, r in zip(bars, ratios):
            ax.annotate(f"{r:.2f}", xy=(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01),
                        ha="center", fontsize=10, fontweight="bold")
        ax.set_title(
            "Persona-space role-spread (unfiltered)\n"
            "Mean pairwise distance between perturbed role vectors, normalized to original.\n"
            "<1.0 = perturbation compressed persona space (role-flexibility broken).",
            fontsize=11.5, loc="left")
        ax.legend(loc="upper right", fontsize=10)
        ax.grid(axis="y", alpha=0.25)
        fig.tight_layout()
        out_path = OUT / "fig7_role_spread.png"
        fig.savefig(out_path)
        plt.close(fig)
        print(f"  saved {out_path}")
    else:
        print(f"  SKIP fig7 (no {unfilt_path})")

    # =================================================================
    # FIGURE 8 — cos(PC1) summary TABLE
    # =================================================================
    print("\nFig 8: cos(PC1) summary table")
    headers = ["Condition", "cos(PC1)\nfiltered", "cos(PC1)\nunfiltered",
               "‖Δ default‖", "cos(Δ default,\nassistant axis)",
               "vectors / 276", "role-spread\n(perturbed/orig)"]
    rows = []

    unfilt_data_full = json.loads(unfilt_path.read_text()) if unfilt_path.exists() else {}
    order = ["E_refusal_abl", "F_evil", "F_humorous", "F_apathetic", "F_sycophantic", "G_lizat"]

    # Need axis_orig for cos(Δ default, axis) computation
    default_orig_global = V_o_np[default_idx_o]

    for key in order:
        if key not in perturbations:
            continue
        p = perturbations[key]
        cos_filt = p["cos_pc1"]
        ci_lo, ci_hi = p["cos_pc1_ci"]
        delta_n = p["delta_norm"]
        n_vec = p["n_vectors"]

        # Compute cos(Δ default, axis_orig)
        cos_delta_axis_str = "—"
        # Get perturbed default — already computed when we built perturbations[key]
        try:
            dirname = next(d for k, d, *_ in CONDITIONS if k == key)
            roles_p_full, V_p_full = load_vectors(REPO / "results" / dirname / "vectors")
            di_p = roles_p_full.index("default")
            default_p_global = V_p_full[di_p].float().numpy()
            delta_default = default_p_global - default_orig_global.astype(np.float64)
            axo = torch.load(REPO / "results/original/axis.pt", weights_only=False)
            if isinstance(axo, dict):
                axo = axo.get("axis", axo.get("vector"))
            axo_np = torch.as_tensor(axo).float().squeeze().numpy()
            cda = cos(delta_default, axo_np)
            cos_delta_axis_str = f"{cda:+.3f}"
        except Exception as e:
            print(f"  ! couldn't compute cos(Δ, axis) for {key}: {e}")

        u_cos = "—"
        if key in unfilt_data_full:
            uc = unfilt_data_full[key]["Q2_pc1_rotation"]["cos_pc1_orig_unfiltered_steered_unfiltered"]
            u_cos = f"{abs(uc):.3f}"
        rs = "—"
        if key in unfilt_data_full and "role_spread" in unfilt_data_full[key]:
            rs_v = unfilt_data_full[key]["role_spread"].get("ratio_perturbed_over_original")
            if rs_v is not None:
                rs = f"{rs_v:.2f}"
        rows.append([
            p["label"],
            f"{cos_filt:.3f}",
            u_cos,
            f"{delta_n:.2f}",
            cos_delta_axis_str,
            f"{n_vec}/276",
            rs,
        ])

    fig, ax = plt.subplots(figsize=(20, 1.5 + 0.95 * len(rows)))
    ax.axis("off")

    # Compute color tint per row by cos(PC1) — green if reshape (low cos), gray if null
    def row_color(cos_val):
        try:
            c = float(cos_val)
        except Exception:
            return "#ffffff"
        if c < 0.4: return "#ddf3dd"   # heavy reshape
        if c < 0.7: return "#e9f5ed"   # moderate reshape
        if c < 0.92: return "#f5f5e6"  # mild reshape
        return "#f5f5f5"               # null floor or above

    cell_colors = []
    for row in rows:
        c = row_color(row[1])
        cell_colors.append([c] * len(headers))
    header_color = "#dee5f0"

    # Custom column widths — give condition column more room
    col_widths = [0.30, 0.10, 0.10, 0.10, 0.135, 0.105, 0.13]
    table = ax.table(
        cellText=rows,
        colLabels=headers,
        cellColours=cell_colors,
        colColours=[header_color] * len(headers),
        cellLoc="center",
        loc="center",
        colWidths=col_widths,
    )
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 1.7)
    # Left-align condition column for readability
    for i in range(len(rows) + 1):
        table[(i, 0)].set_text_props(ha="left")
        table[(i, 0)].PAD = 0.04
    # Bold header
    for j in range(len(headers)):
        table[(0, j)].set_text_props(weight="bold")
    # Color the condition column with the trait color
    color_map = {0: "#8c564b", 1: "#d62728", 2: "#ff7f0e", 3: "#17becf", 4: "#1f77b4", 5: "#9467bd"}
    for i, row in enumerate(rows):
        c = color_map.get(i, "#444")
        cell = table[(i + 1, 0)]
        cell.set_text_props(color=c, weight="bold")

    ax.set_title(
        "Persona-space metrics across all 6 perturbations",
        fontsize=14, loc="left", weight="bold", y=0.96)

    # Add a metric-definitions footer
    fig.text(0.02, 0.02,
             "DEFINITIONS:\n"
             "• cos(PC1) — cosine similarity (NOT distance) between PC1 of original cloud and PC1 of perturbed cloud, computed on the 4096-dim role vectors.  "
             "Absolute value (PCA components are sign-arbitrary).  Range 0-1: 1 = identical PC1 direction; 0 = orthogonal; lower = persona-space reshaped.  "
             "Phase E null floor ≈ 0.913, split-half noise floor ≈ 0.999.\n"
             "• filtered vs unfiltered — filtered uses Lu et al.'s canonical pipeline (only score-3 rollouts contribute to a role's mean); unfiltered uses all 1200 rollouts.\n"
             "• 95% CI — bootstrap (40 iters): resample roles with replacement and recompute cos(PC1).  Wide CI (e.g., apathetic [0.012, 0.595]) = result depends on which roles are sampled (small effective N from filtering).\n"
             "• Δ default — displacement vector of the default-Assistant in 4096-dim activation space (perturbed_default − original_default).  ‖Δ default‖ = its L2 norm = how far the default moved.\n"
             "• cos(Δ default, assistant axis) — cosine similarity (signed) between the default migration and Lu et al.'s Assistant Axis.  Negative = pushed AWAY from Assistant pole; ~0 = orthogonal direction.\n"
             "• role-spread — mean pairwise distance among perturbed role vectors / same metric on original.  <1.0 = persona space *compressed*; >1.0 = expanded.  Computed on UNFILTERED vectors.",
             fontsize=8.5, color="#333", ha="left", va="bottom",
             bbox=dict(boxstyle="round,pad=0.5", facecolor="#fafafa", edgecolor="#999", alpha=0.95),
             wrap=True)
    fig.tight_layout(rect=[0, 0.18, 1, 0.96])
    out_path = OUT / "fig8_pc1_summary_table.png"
    fig.savefig(out_path)
    plt.close(fig)
    print(f"  saved {out_path}")

    print(f"\nAll figures saved to {OUT}/")


if __name__ == "__main__":
    main()
