#!/usr/bin/env python3
"""Recover the implicit refusal direction from mlabonne's abliterated weights.

Per Arditi et al. 2024, abliteration applies W_new = (I - r̂r̂ᵀ) W to each
residual-writing matrix W. So the column-wise weight delta:
    ΔW = W_abl - W_orig = -r̂(r̂ᵀ W_orig)
is rank-1 in r̂, with ALL columns parallel to r̂. The top-1 left singular
vector of [-ΔW_o_proj, -ΔW_down_proj] for layer ℓ recovers r̂_ℓ.

Across layers, mlabonne (FailSpy technique) typically uses a SINGLE direction
applied uniformly, so per-layer r̂ should be ~parallel. Verify and also report
the global SVD direction across stacked deltas.

Outputs results/comparison/refusal_direction_from_mlabonne.pt:
    {"per_layer": (n_layers, d), "global": (d,), "method": "...",
     "per_layer_singular_ratios": (n_layers,)}
"""
import argparse, time
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM


def top_left_singular(M, q=2, niter=4):
    """Return top left singular vector of M (d × k) as a (d,) tensor + sigma1, ratio.

    Uses randomized SVD (svd_lowrank) for speed — full SVD on (4096, 18432) takes
    ~15-25s/layer on CPU; svd_lowrank with q=2 takes ~0.5s/layer. q=2 lets us
    estimate the singular gap σ1/σ2.
    """
    U, S, V = torch.svd_lowrank(M, q=q, niter=niter)
    ratio = float(S[0] / S[1]) if S.numel() > 1 else float("inf")
    return U[:, 0], float(S[0]), ratio


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--original_model", default="meta-llama/Llama-3.1-8B-Instruct")
    ap.add_argument("--abliterated_model",
                    default="mlabonne/Meta-Llama-3.1-8B-Instruct-abliterated")
    ap.add_argument("--output",
                    default="results/comparison/refusal_direction_from_mlabonne.pt")
    ap.add_argument("--compare_against",
                    default="results/comparison/refusal_direction.pt",
                    help="Existing diff-in-means refusal direction for cosine comparison.")
    ap.add_argument("--layer_for_compare", type=int, default=16)
    args = ap.parse_args()

    print(f"=== Recover refusal direction from weight diff @ "
          f"{time.strftime('%Y-%m-%d %H:%M')} ===")

    print(f"\nLoading {args.original_model} (CPU, float32)…")
    t0 = time.time()
    orig = AutoModelForCausalLM.from_pretrained(
        args.original_model, torch_dtype=torch.float32, device_map="cpu",
    )
    print(f"  loaded in {time.time()-t0:.1f}s")

    print(f"\nLoading {args.abliterated_model} (CPU, float32)…")
    t0 = time.time()
    abl = AutoModelForCausalLM.from_pretrained(
        args.abliterated_model, torch_dtype=torch.float32, device_map="cpu",
    )
    print(f"  loaded in {time.time()-t0:.1f}s")

    n_layers = orig.config.num_hidden_layers
    d = orig.config.hidden_size
    assert abl.config.num_hidden_layers == n_layers
    assert abl.config.hidden_size == d
    print(f"\nn_layers={n_layers}, d={d}")

    per_layer = torch.zeros(n_layers, d, dtype=torch.float32)
    per_layer_sigma1 = torch.zeros(n_layers)
    per_layer_ratios = torch.zeros(n_layers)
    global_stack_cols = []  # collect -ΔW columns to do one big SVD across layers

    for L in range(n_layers):
        do = orig.model.layers[L].self_attn.o_proj.weight.detach() - \
             abl.model.layers[L].self_attn.o_proj.weight.detach()
        dd = orig.model.layers[L].mlp.down_proj.weight.detach() - \
             abl.model.layers[L].mlp.down_proj.weight.detach()
        # do, dd shapes: (d, k_in). Their columns are all in span{r̂_L}.
        M = torch.cat([do, dd], dim=1)  # (d, k_o + k_d)
        u, s1, ratio = top_left_singular(M)
        per_layer[L] = u
        per_layer_sigma1[L] = s1
        per_layer_ratios[L] = ratio
        global_stack_cols.append(M)
        if L < 4 or L % 8 == 0 or L == n_layers - 1:
            print(f"  layer {L:2d}: σ1={s1:.4e}, σ1/σ2={ratio:.2e}", flush=True)

    # Global direction across all layers (randomized SVD)
    print("\nComputing global SVD across all layer deltas…", flush=True)
    M_all = torch.cat(global_stack_cols, dim=1)
    print(f"  stacked shape: {tuple(M_all.shape)}", flush=True)
    U_g, S_g, _ = torch.svd_lowrank(M_all, q=4, niter=4)
    g = U_g[:, 0]
    g_ratio = float(S_g[0] / S_g[1])
    print(f"  global: σ1={float(S_g[0]):.4e}, σ1/σ2={g_ratio:.2e}", flush=True)

    # Sign convention: align per-layer vectors with the global direction
    # (SVD sign is arbitrary). Layers whose σ1 is ~0 (no abliteration applied)
    # should be ignored.
    for L in range(n_layers):
        if per_layer_sigma1[L] > 1e-6 and torch.dot(per_layer[L], g) < 0:
            per_layer[L] = -per_layer[L]
    if False:  # global sign is arbitrary; pick by majority of strong layers
        strong = per_layer_sigma1 > per_layer_sigma1.max() * 0.1
        if (per_layer[strong] @ g < 0).sum() > strong.sum() / 2:
            g = -g

    # Pairwise cosines across layers (only for layers with non-trivial sigma)
    strong_mask = per_layer_sigma1 > per_layer_sigma1.max() * 0.05
    strong_layers = strong_mask.nonzero().squeeze(-1).tolist()
    print(f"\nStrong-abliteration layers (σ1 ≥ 5%·max): {strong_layers}")
    if len(strong_layers) >= 2:
        sub = per_layer[strong_mask]
        cos_mat = sub @ sub.T
        avg_off_diag = (cos_mat.sum() - cos_mat.diag().sum()) / \
                       (cos_mat.numel() - cos_mat.shape[0])
        print(f"  mean pairwise cos among strong layers = {avg_off_diag.item():.4f}")
        print(f"  min pairwise cos = {cos_mat[cos_mat < 0.9999].min().item():.4f}")

    # Compare to existing diff-in-means direction
    cmp_path = Path(args.compare_against)
    if cmp_path.exists():
        print(f"\nLoading diff-in-means direction from {cmp_path}…")
        old = torch.load(cmp_path, map_location="cpu", weights_only=False).float()
        if old.ndim == 1:
            old = old.unsqueeze(0).expand(n_layers, -1)
        L = args.layer_for_compare
        cos_lL = float(torch.nn.functional.cosine_similarity(
            per_layer[L].unsqueeze(0), old[L].unsqueeze(0)).item())
        cos_gL = float(torch.nn.functional.cosine_similarity(
            g.unsqueeze(0), old[L].unsqueeze(0)).item())
        print(f"  cos(per_layer[{L}], old[{L}]) = {cos_lL:+.4f}")
        print(f"  cos(global, old[{L}])         = {cos_gL:+.4f}")
        # Also: best-matching layer of old to global
        cos_per = torch.nn.functional.cosine_similarity(
            old, g.unsqueeze(0).expand_as(old))
        print(f"  cos(global, old[ℓ]) per layer: max={cos_per.max():.4f} "
              f"at ℓ={int(cos_per.argmax())}, "
              f"mean={cos_per.mean():.4f}")
    else:
        cos_lL = cos_gL = None
        print(f"\n(skipping comparison: {cmp_path} not found)")

    # Save
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "per_layer": per_layer,
        "global": g,
        "per_layer_sigma1": per_layer_sigma1,
        "per_layer_singular_ratios": per_layer_ratios,
        "strong_layers": strong_layers,
        "method": "top-1 left-SVD of -[ΔW_o_proj, ΔW_down_proj] per layer; global = SVD over all layers stacked",
        "original_model": args.original_model,
        "abliterated_model": args.abliterated_model,
        "compare_cos": {"per_layer_at_L": cos_lL, "global_vs_old_at_L": cos_gL,
                        "compare_layer": args.layer_for_compare},
    }
    torch.save(payload, out)
    print(f"\nSaved {out}")
    print(f"  per_layer: {tuple(per_layer.shape)}")
    print(f"  global:    {tuple(g.shape)}")


if __name__ == "__main__":
    main()
