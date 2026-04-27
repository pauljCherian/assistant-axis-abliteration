"""Stage 4: Bake α·v as additive bias on layer L_inject's down_proj.

For a given (trait, α, L_inject), produce a standalone HF checkpoint that:
- Has identical weights to base Llama-3.1-8B-Instruct EXCEPT
- model.layers[L_inject].mlp.down_proj has a bias term equal to (α · v_trait[L_inject]).to(bf16)

Result: vLLM loads this checkpoint normally, and every forward pass adds α·v_trait
to the residual stream at the output of layer L_inject's MLP. Mathematically equivalent to
ActivationSteerer's `positions="all"` mode, but vLLM-compatible (HF hooks aren't honored by vLLM).

Usage:
    .venv/bin/python scripts/14_bake_persona_vector.py \
        --trait evil --alpha 1.5 --layer 12 \
        --output_dir models/llama-3.1-8b-evil-steered-L12-a1.5

Optionally pushes to HF model hub:
    --push_to_hub --hub_id pandaman007/llama-3.1-8b-instruct-evil-steered-L12-a1.5
"""
import argparse
from pathlib import Path

import torch
from torch import nn
from transformers import AutoModelForCausalLM, AutoTokenizer

REPO = Path("/scratch/paulc/assistant-axis-abliteration")
PV_DIR = REPO / "results/persona_vectors"
MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct"


def load_trait_vector(trait, layer):
    """Load trait vector at specified layer. Returns [4096] fp32.

    Prefers the FILTERED vector (Anthropic full recipe) over the unfiltered one.
    """
    p_filt = PV_DIR / f"{trait}_response_avg_diff_filtered.pt"
    p_unfilt = PV_DIR / f"{trait}_response_avg_diff.pt"
    if p_filt.exists():
        p = p_filt
        print(f"  loading FILTERED vector: {p.name}")
    else:
        p = p_unfilt
        print(f"  loading UNFILTERED vector: {p.name}")
    d = torch.load(p, weights_only=False)
    if isinstance(d, dict):
        v = d.get("vector", d.get("diff"))
    else:
        v = d
    v = torch.as_tensor(v).float()
    assert v.shape[1] == 4096, f"expected 4096 hidden dim, got {v.shape}"
    if layer >= v.shape[0]:
        raise ValueError(f"layer {layer} out of range; trait vector has {v.shape[0]} layers")
    return v[layer]


def bake_bias(trait, alpha, layer, output_dir, normalize_unit=True, push_to_hub=False, hub_id=None):
    """Modify Llama-3.1-8B-Instruct so layer-{layer} down_proj has bias = α·v_trait[layer].

    To survive save/reload (in both HF transformers AND vLLM), we MUST:
      1. Set config.mlp_bias = True so reload-time MLP construction includes bias params
      2. Add zero biases for ALL other MLP projections (gate_proj, up_proj, down_proj on every layer)
         since vLLM/HF expect a complete set when mlp_bias=True
    Without (1)+(2), the bias key is silently dropped on load.
    """
    print(f"Loading {MODEL_NAME} (bf16, cpu first)...")
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.bfloat16)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    print(f"Loading trait '{trait}' vector at layer {layer}...")
    v = load_trait_vector(trait, layer)  # fp32 [4096]
    print(f"  v[{layer}] norm: {v.norm():.4f}")
    if normalize_unit:
        v = v / v.norm()
        print(f"  normalized to unit norm")
    bias = (alpha * v).to(torch.bfloat16)

    # Set mlp_bias=True in config so on-disk config triggers bias-enabled MLP construction
    model.config.mlp_bias = True
    print(f"  set config.mlp_bias = True (required for vLLM + HF to load bias keys)")

    # Replace all MLP projections with bias=True, zero everywhere except target
    n_layers = len(model.model.layers)
    target_li = layer
    for li in range(n_layers):
        for proj_name in ("gate_proj", "up_proj", "down_proj"):
            proj = getattr(model.model.layers[li].mlp, proj_name)
            new_lin = nn.Linear(
                proj.in_features, proj.out_features, bias=True,
                dtype=proj.weight.dtype, device=proj.weight.device,
            )
            new_lin.weight.data = proj.weight.data
            if li == target_li and proj_name == "down_proj":
                new_lin.bias.data = bias.to(new_lin.weight.device).to(new_lin.weight.dtype)
                print(f"  applied steering bias on layer {li} {proj_name}, norm {new_lin.bias.float().norm().item():.4f}")
            else:
                new_lin.bias.data = torch.zeros(proj.out_features, dtype=proj.weight.dtype)
            setattr(model.model.layers[li].mlp, proj_name, new_lin)
    print(f"  added bias=True to {n_layers}×3 = {n_layers*3} MLP projections (one steering, rest zero)")

    # NaN/Inf check
    for name, param in model.named_parameters():
        if not torch.isfinite(param).all():
            raise RuntimeError(f"Non-finite values in {name} after bake!")

    # Save
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Saving baked checkpoint to {out_dir}...")
    model.save_pretrained(out_dir, safe_serialization=True)
    tokenizer.save_pretrained(out_dir)

    # Save bake metadata
    meta = {
        "trait": trait,
        "alpha": alpha,
        "layer": layer,
        "normalize_unit": normalize_unit,
        "method": "bake_as_bias_on_down_proj",
        "base_model": MODEL_NAME,
        "v_norm_pre_normalize": float(v.norm()) if not normalize_unit else None,
        "bias_norm_in_bf16": float((alpha * v).to(torch.bfloat16).float().norm()),
    }
    import json
    (out_dir / "bake_metadata.json").write_text(json.dumps(meta, indent=2))
    print(f"  saved bake_metadata.json")

    # Push to HF model hub if requested
    if push_to_hub:
        assert hub_id, "must provide --hub_id when --push_to_hub"
        print(f"Pushing to HF model hub: {hub_id}")
        model.push_to_hub(hub_id, private=False)
        tokenizer.push_to_hub(hub_id, private=False)
        # Upload metadata too
        from huggingface_hub import HfApi
        HfApi().upload_file(
            path_or_fileobj=str(out_dir / "bake_metadata.json"),
            path_in_repo="bake_metadata.json",
            repo_id=hub_id,
        )
        print(f"  pushed to https://huggingface.co/{hub_id}")

    print("Done.")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--trait", required=True, choices=[
        "evil", "sycophantic", "apathetic", "humorous",
        "impolite", "hallucinating", "optimistic", "assistant_identity",
    ])
    p.add_argument("--alpha", type=float, required=True, help="α coefficient")
    p.add_argument("--layer", type=int, default=12, help="L_inject (default 12)")
    p.add_argument("--output_dir", required=True, help="local save dir for baked checkpoint")
    p.add_argument("--no_normalize", action="store_true",
                   help="if set, use trait vector as-is (don't unit-normalize). Default: unit-normalize.")
    p.add_argument("--push_to_hub", action="store_true")
    p.add_argument("--hub_id", default=None)
    args = p.parse_args()

    bake_bias(
        trait=args.trait, alpha=args.alpha, layer=args.layer,
        output_dir=args.output_dir,
        normalize_unit=not args.no_normalize,
        push_to_hub=args.push_to_hub, hub_id=args.hub_id,
    )
