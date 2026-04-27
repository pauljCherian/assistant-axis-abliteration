"""Patch a baked-bias checkpoint so vLLM loads the bias correctly.

vLLM's Llama implementation only loads bias keys for MLP if config has mlp_bias=True.
Setting mlp_bias=True requires bias params for ALL MLP projections (gate_proj, up_proj, down_proj)
on ALL layers. This script:

1. Loads the baked checkpoint with HF transformers (which already has bias on layer L on down_proj)
2. Adds zero biases for every other MLP projection on every layer (so vLLM's loader is satisfied)
3. Sets mlp_bias=True in config
4. Re-saves the checkpoint in-place with safe_serialization

Usage:
    .venv/bin/python scripts/14b_fix_bake_for_vllm.py /path/to/baked/checkpoint
"""
import argparse
import json
from pathlib import Path

import torch
from torch import nn
from transformers import AutoModelForCausalLM


def patch_for_vllm(ckpt_dir):
    ckpt_dir = Path(ckpt_dir)
    print(f"Loading {ckpt_dir} (bf16, cpu)...")
    model = AutoModelForCausalLM.from_pretrained(ckpt_dir, torch_dtype=torch.bfloat16)

    # Find which layer/proj has the existing bias (the steering one)
    steering_loc = None
    for li, layer in enumerate(model.model.layers):
        for proj_name in ("gate_proj", "up_proj", "down_proj"):
            proj = getattr(layer.mlp, proj_name)
            if proj.bias is not None:
                steering_loc = (li, proj_name)
                print(f"  found existing bias at layer {li} {proj_name}, norm {proj.bias.float().norm().item():.4f}")

    if steering_loc is None:
        raise RuntimeError("No bias found anywhere — is this a baked checkpoint?")

    # Add zero biases everywhere else
    n_added = 0
    for li, layer in enumerate(model.model.layers):
        for proj_name in ("gate_proj", "up_proj", "down_proj"):
            proj = getattr(layer.mlp, proj_name)
            if proj.bias is None:
                # Replace with bias=True linear, zero bias
                new_lin = nn.Linear(
                    proj.in_features, proj.out_features, bias=True,
                    dtype=proj.weight.dtype, device=proj.weight.device,
                )
                new_lin.weight.data = proj.weight.data
                new_lin.bias.data = torch.zeros(proj.out_features, dtype=proj.weight.dtype)
                setattr(layer.mlp, proj_name, new_lin)
                n_added += 1
    print(f"  added {n_added} zero biases (expected: 32 layers × 3 projs - 1 = 95)")
    assert n_added == 95, f"expected 95 zero biases, added {n_added}"

    # Verify steering bias is still there
    li, proj_name = steering_loc
    proj = getattr(model.model.layers[li].mlp, proj_name)
    assert proj.bias is not None and proj.bias.float().norm().item() > 0.1, "steering bias lost!"
    print(f"  steering bias verified: layer {li} {proj_name} norm {proj.bias.float().norm().item():.4f}")

    # NaN check
    for name, param in model.named_parameters():
        if not torch.isfinite(param).all():
            raise RuntimeError(f"Non-finite values in {name}")

    # Save in-place (overwrite safetensors)
    print(f"Re-saving to {ckpt_dir}...")
    # Remove old shards first to avoid stale files
    for old in ckpt_dir.glob("model-*.safetensors"):
        old.unlink()
    if (ckpt_dir / "model.safetensors").exists():
        (ckpt_dir / "model.safetensors").unlink()
    if (ckpt_dir / "model.safetensors.index.json").exists():
        (ckpt_dir / "model.safetensors.index.json").unlink()
    model.save_pretrained(ckpt_dir, safe_serialization=True)

    # Patch config.json
    cfg_path = ckpt_dir / "config.json"
    cfg = json.loads(cfg_path.read_text())
    cfg["mlp_bias"] = True
    cfg_path.write_text(json.dumps(cfg, indent=2))
    print(f"  set mlp_bias=true in config.json")

    print("Done.")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("ckpt_dir", help="path to baked HF checkpoint")
    args = p.parse_args()
    patch_for_vllm(args.ckpt_dir)
