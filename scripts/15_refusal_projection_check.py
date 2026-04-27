"""Stage 0 (G): Refusal-projection sanity check.

Project the SVD-recovered refusal direction out of Phase E's abliterated role vectors,
then recompute PC1 + cos with original PC1. If cos jumps from 0.913 → ~0.99,
the Phase E null is rigid translation along refusal — persona structure preserved.
"""
import json
from pathlib import Path
import numpy as np
import torch
from sklearn.decomposition import PCA

REPO = Path("/scratch/paulc/assistant-axis-abliteration")
ORIG_DIR = REPO / "results/original/vectors"
ABL_DIR = REPO / "results/abliterated/vectors"
REFUSAL_PATH = REPO / "results/comparison/refusal_direction_from_mlabonne.pt"
OUT_DIR = REPO / "results/comparison"


def load_role_vectors(vec_dir):
    roles, vecs = [], []
    for f in sorted(Path(vec_dir).glob("*.pt")):
        d = torch.load(f, weights_only=False)
        v = d["vector"].squeeze(0).float()
        roles.append(d.get("role", f.stem))
        vecs.append(v)
    return roles, torch.stack(vecs)


def pca_pc1(X):
    pca = PCA(n_components=5)
    pca.fit(X.numpy())
    return torch.from_numpy(pca.components_[0]).float(), pca.explained_variance_ratio_


def cos(a, b):
    return float((a / a.norm()).dot(b / b.norm()))


def sign_align(a, b):
    return a if cos(a, b) >= 0 else -a


def project_out(V, rh):
    coef = V @ rh
    return V - torch.outer(coef, rh)


def main():
    print("Loading role vectors...")
    roles_o, V_o = load_role_vectors(ORIG_DIR)
    roles_a, V_a = load_role_vectors(ABL_DIR)
    assert roles_o == roles_a, f"Role mismatch: {set(roles_o) ^ set(roles_a)}"
    print(f"  {len(roles_o)} roles, dim={V_o.shape[1]}")

    print("Loading refusal direction...")
    refusal_data = torch.load(REFUSAL_PATH, weights_only=False)
    r = refusal_data["per_layer"][16].float()
    r_global = refusal_data["global"].float()
    r_hat = r / r.norm()
    r_global_hat = r_global / r_global.norm()
    print(f"  per_layer[16] norm={r.norm():.3f}, global norm={r_global.norm():.3f}")
    print(f"  cos(per_layer[16], global)={cos(r, r_global):.4f}")

    mu_o = V_o.mean(0)
    mu_a = V_a.mean(0)
    Vc_o = V_o - mu_o
    Vc_a = V_a - mu_a

    pc1_o, ve_o = pca_pc1(Vc_o)
    pc1_a, ve_a = pca_pc1(Vc_a)
    pc1_a_aligned = sign_align(pc1_a, pc1_o)
    cos_pc1_pre = cos(pc1_o, pc1_a_aligned)
    print(f"\n[BASELINE]")
    print(f"  cos(PC1_orig, PC1_abl) = {cos_pc1_pre:.4f} (Phase E: 0.913)")
    print(f"  centroid shift = {(mu_a - mu_o).norm():.4f}")
    print(f"  cos(shift, refusal[16]) = {cos(mu_a - mu_o, r):.4f}")
    print(f"  cos(shift, refusal_global) = {cos(mu_a - mu_o, r_global):.4f}")

    V_a_p = project_out(V_a, r_hat)
    mu_a_p = V_a_p.mean(0)
    pc1_a_p, ve_a_p = pca_pc1(V_a_p - mu_a_p)
    pc1_a_p_al = sign_align(pc1_a_p, pc1_o)
    cos_after_pl = cos(pc1_o, pc1_a_p_al)

    V_a_pg = project_out(V_a, r_global_hat)
    mu_a_pg = V_a_pg.mean(0)
    pc1_a_pg, ve_a_pg = pca_pc1(V_a_pg - mu_a_pg)
    pc1_a_pg_al = sign_align(pc1_a_pg, pc1_o)
    cos_after_g = cos(pc1_o, pc1_a_pg_al)

    V_o_p = project_out(V_o, r_hat)
    mu_o_p = V_o_p.mean(0)
    pc1_o_p, _ = pca_pc1(V_o_p - mu_o_p)
    cos_symm = cos(pc1_o_p, sign_align(pc1_a_p, pc1_o_p))

    print(f"\n[PROJECT per_layer[16] OUT of ABL]")
    print(f"  cos(PC1_orig, PC1_abl_proj) = {cos_after_pl:.4f}  (Δ {cos_after_pl - cos_pc1_pre:+.4f})")
    print(f"  centroid shift after projection = {(mu_a_p - mu_o).norm():.4f}")
    print(f"  cos(shift_proj, refusal) = {cos(mu_a_p - mu_o, r):.4f}")

    print(f"\n[PROJECT global OUT of ABL]")
    print(f"  cos(PC1_orig, PC1_abl_proj_g) = {cos_after_g:.4f}  (Δ {cos_after_g - cos_pc1_pre:+.4f})")
    print(f"  centroid shift after projection = {(mu_a_pg - mu_o).norm():.4f}")

    print(f"\n[PROJECT per_layer[16] OUT of BOTH]")
    print(f"  cos(PC1_orig_p, PC1_abl_p) = {cos_symm:.4f}")

    if cos_after_pl > 0.99:
        verdict = "Phase E null IS rigid translation along refusal — persona structure preserved."
    elif cos_after_pl > 0.95:
        verdict = "Mostly rigid translation, small residual structural component."
    else:
        verdict = "Significant non-rigid structure beyond refusal-axis translation."
    print(f"\n[VERDICT] {verdict}")

    report = {
        "phase": "Stage 0 (G) — refusal-projection sanity check",
        "n_roles": len(roles_o),
        "baseline": {
            "cos_pc1": cos_pc1_pre,
            "centroid_shift_norm": float((mu_a - mu_o).norm()),
            "cos_shift_refusal_perlayer16": cos(mu_a - mu_o, r),
            "cos_shift_refusal_global": cos(mu_a - mu_o, r_global),
            "var_pc1_orig": float(ve_o[0]),
            "var_pc1_abl": float(ve_a[0]),
        },
        "after_projection_per_layer_16": {
            "cos_pc1": cos_after_pl,
            "delta_cos": cos_after_pl - cos_pc1_pre,
            "centroid_shift_norm": float((mu_a_p - mu_o).norm()),
            "cos_shift_refusal": cos(mu_a_p - mu_o, r),
            "var_pc1_abl_proj": float(ve_a_p[0]),
        },
        "after_projection_global": {
            "cos_pc1": cos_after_g,
            "delta_cos": cos_after_g - cos_pc1_pre,
            "centroid_shift_norm": float((mu_a_pg - mu_o).norm()),
            "var_pc1_abl_proj": float(ve_a_pg[0]),
        },
        "after_projection_symmetric": {"cos_pc1": cos_symm},
        "verdict": verdict,
    }
    (OUT_DIR / "refusal_projection_analysis.json").write_text(json.dumps(report, indent=2))

    md = [
        "# Stage 0 — Refusal-Projection Sanity Check",
        f"\n**N roles:** {len(roles_o)}",
        "\n## Baseline (Phase E recap)",
        f"- cos(PC1_orig, PC1_abl) = **{cos_pc1_pre:.4f}**",
        f"- centroid shift = **{(mu_a - mu_o).norm():.4f}**",
        f"- cos(shift, refusal[16]) = {cos(mu_a - mu_o, r):.4f}",
        f"- cos(shift, refusal_global) = {cos(mu_a - mu_o, r_global):.4f}",
        "\n## After projecting refusal out of abliterated vectors",
        "\n### per_layer[16]",
        f"- cos(PC1_orig, PC1_abl_proj) = **{cos_after_pl:.4f}** (Δ {cos_after_pl - cos_pc1_pre:+.4f})",
        f"- centroid shift = {(mu_a_p - mu_o).norm():.4f}",
        "\n### global",
        f"- cos(PC1_orig, PC1_abl_proj_g) = **{cos_after_g:.4f}** (Δ {cos_after_g - cos_pc1_pre:+.4f})",
        f"- centroid shift = {(mu_a_pg - mu_o).norm():.4f}",
        "\n### symmetric (project both)",
        f"- cos(PC1_orig_proj, PC1_abl_proj) = **{cos_symm:.4f}**",
        f"\n## Verdict\n\n{verdict}",
    ]
    (OUT_DIR / "refusal_projection_analysis.md").write_text("\n".join(md))
    print(f"\nSaved {OUT_DIR}/refusal_projection_analysis.{{json,md}}")


if __name__ == "__main__":
    main()
