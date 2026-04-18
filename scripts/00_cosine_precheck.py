"""
Step 0: Cosine similarity pre-check between refusal direction and assistant axis.

Computes the refusal direction (Arditi et al.) from harmful/harmless prompt
activations and an approximate assistant axis (Lu et al.) from a 30-role
subset, then measures their cosine similarity at each transformer layer.

This gives an immediate prior on how much the refusal mechanism overlaps
with the persona space structure before running the full pipeline.

Usage:
    cd /scratch/paulc/assistant-axis-abliteration
    python scripts/00_cosine_precheck.py [--device cuda:0] [--output_dir results/comparison]
"""

import argparse
import gc
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Add project paths so we can import from the cloned repos
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "assistant-axis"))
sys.path.insert(0, str(PROJECT_ROOT / "OBLITERATUS"))

from assistant_axis.axis import compute_axis, cosine_similarity_per_layer
from assistant_axis.internals.model import ProbingModel
from assistant_axis.internals.activations import ActivationExtractor
from assistant_axis.internals.conversation import ConversationEncoder
from assistant_axis.models import get_config
from obliteratus.prompts import BUILTIN_HARMFUL, BUILTIN_HARMLESS

# Hardcoded config for Llama 3.1 8B (gated model, get_config may fail without HF token)
LLAMA_8B_CONFIG = {
    "target_layer": 16,  # 32 // 2
    "total_layers": 32,
    "short_name": "Llama",
}


def safe_get_config(model_name):
    """Get model config with fallback for Llama 3.1 8B."""
    if "llama" in model_name.lower() and "8b" in model_name.lower():
        return LLAMA_8B_CONFIG.copy()
    try:
        return get_config(model_name)
    except (ValueError, OSError):
        # Fallback: detect from loaded model later
        return None

# ── Configuration ────────────────────────────────────────────────────────

MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct"
N_HARMFUL = 128  # Number of harmful prompts to use
N_HARMLESS = 128  # Number of harmless prompts to use
N_QUESTIONS = 10  # Extraction questions per role
BATCH_SIZE = 16   # Batch size for refusal direction extraction
MAX_NEW_TOKENS = 100  # Max response length for role generation

# Role subset: 30 roles spanning the persona space diversity
ROLE_SUBSET = {
    # Safety-adjacent (most likely to interact with refusal)
    "safety": [
        "guardian", "judge", "demon", "saboteur",
        "criminal", "vigilante", "angel",
    ],
    # Professional/assistant-like (high end of PC1)
    "professional": [
        "assistant", "consultant", "analyst", "therapist",
        "teacher", "researcher", "lawyer",
    ],
    # Creative/narrative (mid-range, strong role-playing)
    "creative": [
        "pirate", "ghost", "witch", "detective",
        "warrior", "bard", "robot", "alien",
    ],
    # Abstract/unusual (far end of PC1, exotic roles)
    "abstract": [
        "void", "aberration", "hive", "parasite",
        "eldritch", "swarm", "chimera", "echo",
    ],
}
ALL_ROLES = [role for group in ROLE_SUBSET.values() for role in group]

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ── Part 1: Refusal Direction ────────────────────────────────────────────

def collect_last_token_activations(pm, formatted_prompts, batch_size=16):
    """Collect last-token post-layer activations for a list of formatted prompts.

    Args:
        pm: ProbingModel with loaded model
        formatted_prompts: list of already-chat-template-formatted strings
        batch_size: batch size for forward passes

    Returns:
        dict mapping layer_idx -> list of (hidden_dim,) tensors, one per prompt
    """
    layers = pm.get_layers()
    n_layers = len(layers)
    tokenizer = pm.tokenizer

    # Ensure left-padding for batched last-token extraction
    orig_padding_side = tokenizer.padding_side
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    activations = {i: [] for i in range(n_layers)}

    try:
        for batch_start in range(0, len(formatted_prompts), batch_size):
            batch_end = min(batch_start + batch_size, len(formatted_prompts))
            batch = formatted_prompts[batch_start:batch_end]

            # Register hooks
            hooks = []
            batch_acts = {}

            def make_hook(idx):
                def hook_fn(module, input, output):
                    hidden = output[0] if isinstance(output, tuple) else output
                    # Last token for each item in batch
                    batch_acts[idx] = hidden[:, -1, :].detach().cpu().float()
                return hook_fn

            for idx in range(n_layers):
                hooks.append(layers[idx].register_forward_hook(make_hook(idx)))

            try:
                inputs = tokenizer(
                    batch, return_tensors="pt", padding=True,
                    truncation=True, max_length=256,
                )
                inputs = {k: v.to(pm.device) for k, v in inputs.items()}
                with torch.inference_mode():
                    pm.model(**inputs)
                del inputs
            finally:
                for h in hooks:
                    h.remove()

            # Unbatch: store per-prompt activations
            for idx in range(n_layers):
                if idx in batch_acts:
                    for b in range(batch_acts[idx].shape[0]):
                        activations[idx].append(batch_acts[idx][b])

            if (batch_end % (batch_size * 4) == 0) or batch_end == len(formatted_prompts):
                log.info(f"  Processed {batch_end}/{len(formatted_prompts)} prompts")
    finally:
        tokenizer.padding_side = orig_padding_side

    return activations


def extract_refusal_direction(pm, n_harmful=N_HARMFUL, n_harmless=N_HARMLESS):
    """Extract refusal direction at all layers using difference-in-means.

    Returns:
        refusal_directions: tensor (n_layers, hidden_dim), normalized per layer
        refusal_norms: tensor (n_layers,), raw norms before normalization
    """
    log.info("=" * 60)
    log.info("PART 1: Extracting refusal direction (difference-in-means)")
    log.info("=" * 60)

    harmful_prompts = list(BUILTIN_HARMFUL)[:n_harmful]
    harmless_prompts = list(BUILTIN_HARMLESS)[:n_harmless]
    log.info(f"Using {len(harmful_prompts)} harmful + {len(harmless_prompts)} harmless prompts")

    tokenizer = pm.tokenizer

    def format_prompt(p):
        return tokenizer.apply_chat_template(
            [{"role": "user", "content": p}],
            tokenize=False, add_generation_prompt=True,
        )

    log.info("Formatting prompts with chat template...")
    harmful_formatted = [format_prompt(p) for p in harmful_prompts]
    harmless_formatted = [format_prompt(p) for p in harmless_prompts]

    log.info("Collecting harmful activations...")
    harmful_acts = collect_last_token_activations(pm, harmful_formatted, BATCH_SIZE)

    log.info("Collecting harmless activations...")
    harmless_acts = collect_last_token_activations(pm, harmless_formatted, BATCH_SIZE)

    # Compute difference-in-means per layer
    n_layers = len(pm.get_layers())
    directions = []
    norms = []

    for layer_idx in range(n_layers):
        h_mean = torch.stack(harmful_acts[layer_idx]).mean(dim=0)
        s_mean = torch.stack(harmless_acts[layer_idx]).mean(dim=0)
        diff = h_mean - s_mean
        norm = diff.norm().item()
        norms.append(norm)
        if norm > 0:
            directions.append(diff / norm)
        else:
            directions.append(diff)

    refusal_directions = torch.stack(directions)  # (n_layers, hidden_dim)
    refusal_norms = torch.tensor(norms)

    # Log strength by layer
    sorted_layers = sorted(enumerate(norms), key=lambda x: x[1], reverse=True)
    max_norm = sorted_layers[0][1] if sorted_layers else 1.0
    log.info("Refusal direction strength by layer (top 10):")
    for idx, norm in sorted_layers[:10]:
        bar_len = int(norm / max_norm * 30) if max_norm > 0 else 0
        log.info(f"  layer {idx:3d}: {norm:.4f} {'█' * bar_len}")

    log.info(f"Refusal direction extracted: shape {refusal_directions.shape}")
    return refusal_directions, refusal_norms


# ── Part 2: Approximate Assistant Axis ───────────────────────────────────

def load_extraction_questions(data_dir, n_questions=N_QUESTIONS):
    """Load first n extraction questions from the assistant-axis data."""
    questions_path = data_dir / "extraction_questions.jsonl"
    questions = []
    with open(questions_path) as f:
        for line in f:
            if len(questions) >= n_questions:
                break
            entry = json.loads(line.strip())
            questions.append(entry["question"])
    log.info(f"Loaded {len(questions)} extraction questions")
    return questions


def load_role_instruction(data_dir, role_name):
    """Load the first system prompt for a role.

    For default role, returns all 5 prompts.
    Replaces {model_name} placeholder with 'Llama'.
    """
    role_path = data_dir / "roles" / "instructions" / f"{role_name}.json"
    with open(role_path) as f:
        role_data = json.load(f)

    instructions = role_data["instruction"]

    if role_name == "default":
        # Return all 5 system prompts for the default role
        prompts = [inst["pos"].replace("{model_name}", "Llama") for inst in instructions]
        return prompts
    else:
        # Return just the first system prompt
        return [instructions[0]["pos"]]


def extract_role_vector(pm, extractor, encoder, system_prompt, questions, n_layers):
    """Generate responses and extract response-token-averaged activations for one (role, system_prompt) combination.

    Returns:
        role_vector: tensor (n_layers, hidden_dim), averaged across questions
    """
    all_layers = list(range(n_layers))
    question_vectors = []

    for q_idx, question in enumerate(questions):
        # Generate response
        if system_prompt:
            gen_messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": question},
            ]
            gen_formatted = pm.tokenizer.apply_chat_template(
                gen_messages, tokenize=False, add_generation_prompt=True,
            )
        else:
            gen_messages = [{"role": "user", "content": question}]
            gen_formatted = pm.tokenizer.apply_chat_template(
                gen_messages, tokenize=False, add_generation_prompt=True,
            )

        # Generate
        inputs = pm.tokenizer(gen_formatted, return_tensors="pt").to(pm.device)
        with torch.inference_mode():
            outputs = pm.model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                temperature=0.7,
                do_sample=True,
                pad_token_id=pm.tokenizer.eos_token_id,
            )
        response_text = pm.tokenizer.decode(
            outputs[0][inputs.input_ids.shape[1]:],
            skip_special_tokens=True,
        ).strip()
        del inputs, outputs

        if not response_text:
            log.warning(f"    Empty response for question {q_idx}, skipping")
            continue

        # Build full conversation with the response
        conversation = []
        if system_prompt:
            conversation.append({"role": "system", "content": system_prompt})
        conversation.append({"role": "user", "content": question})
        conversation.append({"role": "assistant", "content": response_text})

        # Extract activations at all layers
        try:
            activations = extractor.full_conversation(
                conversation, layer=all_layers, chat_format=True,
            )
            # Shape: (n_layers, n_tokens, hidden_dim)

            # Get response token indices
            response_idx = encoder.response_indices(conversation)
            if not response_idx:
                log.warning(f"    No response indices found for question {q_idx}, using last 10 tokens")
                n_tokens = activations.shape[1]
                response_idx = list(range(max(0, n_tokens - 10), n_tokens))

            # Average over response tokens
            response_acts = activations[:, response_idx, :].mean(dim=1)  # (n_layers, hidden_dim)
            question_vectors.append(response_acts.cpu())

        except Exception as e:
            log.warning(f"    Error extracting activations for question {q_idx}: {e}")
            continue

    if not question_vectors:
        return None

    # Average across questions
    return torch.stack(question_vectors).mean(dim=0)  # (n_layers, hidden_dim)


def extract_approximate_axis(pm, data_dir, n_questions=N_QUESTIONS):
    """Extract approximate assistant axis from a 30-role subset.

    Returns:
        axis: tensor (n_layers, hidden_dim)
        role_vectors: dict mapping role_name -> tensor (n_layers, hidden_dim)
    """
    log.info("=" * 60)
    log.info("PART 2: Extracting approximate assistant axis (30-role subset)")
    log.info("=" * 60)

    n_layers = len(pm.get_layers())
    encoder = ConversationEncoder(pm.tokenizer, pm.model_name)
    extractor = ActivationExtractor(pm, encoder)

    questions = load_extraction_questions(data_dir, n_questions=n_questions)
    role_vectors = {}
    default_vectors = []

    # Process default role first (all 5 system prompts)
    log.info("Processing default role (5 system prompts)...")
    default_prompts = load_role_instruction(data_dir, "default")
    for sp_idx, sys_prompt in enumerate(default_prompts):
        log.info(f"  Default prompt {sp_idx + 1}/5: '{sys_prompt[:50]}...' " if len(sys_prompt) > 50 else f"  Default prompt {sp_idx + 1}/5: '{sys_prompt}'")
        vec = extract_role_vector(pm, extractor, encoder, sys_prompt, questions, n_layers)
        if vec is not None:
            default_vectors.append(vec)
        torch.cuda.empty_cache()

    if not default_vectors:
        raise RuntimeError("Failed to extract any default role vectors")

    log.info(f"Extracted {len(default_vectors)} default vectors")

    # Process each role
    for role_idx, role_name in enumerate(ALL_ROLES):
        log.info(f"Processing role {role_idx + 1}/{len(ALL_ROLES)}: {role_name}")
        sys_prompts = load_role_instruction(data_dir, role_name)
        sys_prompt = sys_prompts[0]  # Use first system prompt only

        vec = extract_role_vector(pm, extractor, encoder, sys_prompt, questions, n_layers)
        if vec is not None:
            role_vectors[role_name] = vec
            log.info(f"  Got role vector for {role_name}")
        else:
            log.warning(f"  Failed to extract vector for {role_name}")

        torch.cuda.empty_cache()
        gc.collect()

    log.info(f"Extracted {len(role_vectors)} role vectors out of {len(ALL_ROLES)} roles")

    if len(role_vectors) < 5:
        raise RuntimeError(f"Too few role vectors ({len(role_vectors)}), need at least 5")

    # Compute axis: mean(default) - mean(roles)
    default_stack = torch.stack(default_vectors)  # (n_default, n_layers, hidden_dim)
    role_stack = torch.stack(list(role_vectors.values()))  # (n_roles, n_layers, hidden_dim)
    axis = compute_axis(role_stack, default_stack)  # (n_layers, hidden_dim)

    log.info(f"Assistant axis computed: shape {axis.shape}")
    return axis, role_vectors


# ── Part 3: Compare and Report ───────────────────────────────────────────

def compare_and_report(refusal_dir, refusal_norms, axis, role_vectors, output_dir, config, model_name=MODEL_NAME):
    """Compute cosine similarity and produce output."""
    log.info("=" * 60)
    log.info("PART 3: Comparing refusal direction and assistant axis")
    log.info("=" * 60)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    middle_layer = config["target_layer"]
    n_layers = refusal_dir.shape[0]

    # Cosine similarity at each layer
    cos_sim = cosine_similarity_per_layer(refusal_dir, axis)
    abs_cos_sim = np.abs(cos_sim)
    angles_deg = np.degrees(np.arccos(np.clip(abs_cos_sim, -1, 1)))

    # Axis norms per layer
    axis_norms = axis.float().norm(dim=1).numpy()

    # Print layer-by-layer table
    log.info("")
    log.info(f"{'Layer':>5} {'cos_sim':>10} {'|cos_sim|':>10} {'angle(°)':>10} {'refusal_norm':>14} {'axis_norm':>12}")
    log.info("-" * 65)
    for i in range(n_layers):
        marker = " <-- MIDDLE" if i == middle_layer else ""
        log.info(
            f"{i:5d} {cos_sim[i]:10.4f} {abs_cos_sim[i]:10.4f} "
            f"{angles_deg[i]:10.1f} {refusal_norms[i].item():14.4f} "
            f"{axis_norms[i]:12.4f}{marker}"
        )

    # Focal result at middle layer
    log.info("")
    log.info("=" * 60)
    log.info(f"FOCAL RESULT (middle layer {middle_layer}):")
    log.info(f"  Cosine similarity:    {cos_sim[middle_layer]:.4f}")
    log.info(f"  |Cosine similarity|:  {abs_cos_sim[middle_layer]:.4f}")
    log.info(f"  Angle:                {angles_deg[middle_layer]:.1f}°")
    log.info(f"  Refusal dir norm:     {refusal_norms[middle_layer].item():.4f}")
    log.info(f"  Axis norm:            {axis_norms[middle_layer]:.4f}")

    # Hypothesis assessment
    mid_abs_cos = abs_cos_sim[middle_layer]
    log.info("")
    if mid_abs_cos > 0.7:
        log.info("HYPOTHESIS: Strong overlap (H1) — refusal direction ≈ persona direction")
        log.info("  The refusal mechanism and persona space are substantially entangled.")
    elif mid_abs_cos > 0.3:
        log.info("HYPOTHESIS: Partial overlap (H3) — the two directions share some variance")
        log.info("  Abliteration may shift some archetypes without collapsing the persona space.")
    else:
        log.info("HYPOTHESIS: Near-orthogonal (H2) — refusal is a separate gating mechanism")
        log.info("  Abliteration likely preserves persona space geometry.")
    log.info("=" * 60)

    # Save results
    results = {
        "model": model_name,
        "middle_layer": middle_layer,
        "n_layers": n_layers,
        "n_roles": len(role_vectors),
        "role_names": list(role_vectors.keys()),
        "role_groups": {k: v for k, v in ROLE_SUBSET.items()},
        "focal_layer": {
            "layer": middle_layer,
            "cosine_similarity": float(cos_sim[middle_layer]),
            "abs_cosine_similarity": float(abs_cos_sim[middle_layer]),
            "angle_degrees": float(angles_deg[middle_layer]),
            "refusal_norm": float(refusal_norms[middle_layer].item()),
            "axis_norm": float(axis_norms[middle_layer]),
        },
        "per_layer": {
            "cosine_similarity": cos_sim.tolist(),
            "abs_cosine_similarity": abs_cos_sim.tolist(),
            "angle_degrees": angles_deg.tolist(),
            "refusal_norms": refusal_norms.tolist(),
            "axis_norms": axis_norms.tolist(),
        },
    }

    results_path = output_dir / "cosine_precheck_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    log.info(f"Results saved to {results_path}")

    # Save tensors
    torch.save(refusal_dir, output_dir / "refusal_direction.pt")
    torch.save(axis, output_dir / "approximate_axis.pt")
    torch.save(
        {name: vec for name, vec in role_vectors.items()},
        output_dir / "role_vectors.pt",
    )
    log.info("Tensors saved (refusal_direction.pt, approximate_axis.pt, role_vectors.pt)")

    # Plot
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    # Top: cosine similarity
    layer_indices = np.arange(n_layers)
    ax1.plot(layer_indices, cos_sim, "b-o", markersize=4, label="Cosine similarity")
    ax1.plot(layer_indices, abs_cos_sim, "r--s", markersize=4, label="|Cosine similarity|")
    ax1.axvline(x=middle_layer, color="green", linestyle=":", alpha=0.7, label=f"Middle layer ({middle_layer})")
    ax1.axhline(y=0, color="gray", linestyle="-", alpha=0.3)
    ax1.set_ylabel("Cosine Similarity")
    ax1.set_title(f"Refusal Direction vs Assistant Axis — {model_name}")
    ax1.legend(loc="upper right")
    ax1.set_ylim(-1.05, 1.05)
    ax1.grid(True, alpha=0.3)

    # Annotate middle layer
    ax1.annotate(
        f"cos={cos_sim[middle_layer]:.3f}\n|cos|={abs_cos_sim[middle_layer]:.3f}",
        xy=(middle_layer, cos_sim[middle_layer]),
        xytext=(middle_layer + 2, cos_sim[middle_layer] + 0.15),
        arrowprops=dict(arrowstyle="->", color="green"),
        fontsize=9, color="green",
    )

    # Bottom: norms
    ax2.plot(layer_indices, refusal_norms.numpy(), "b-o", markersize=4, label="Refusal direction norm")
    ax2.plot(layer_indices, axis_norms, "r-s", markersize=4, label="Assistant axis norm")
    ax2.axvline(x=middle_layer, color="green", linestyle=":", alpha=0.7)
    ax2.set_xlabel("Layer")
    ax2.set_ylabel("L2 Norm (unnormalized)")
    ax2.legend(loc="upper right")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plot_path = output_dir / "cosine_similarity_plot.png"
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close()
    log.info(f"Plot saved to {plot_path}")

    return results


# ── Main ─────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Cosine similarity pre-check: refusal direction vs assistant axis")
    parser.add_argument("--model", default=MODEL_NAME, help="HuggingFace model name")
    parser.add_argument("--device", default="cuda:0", help="Device to use")
    parser.add_argument("--output_dir", default="results/comparison", help="Output directory")
    parser.add_argument("--n_harmful", type=int, default=N_HARMFUL, help="Number of harmful prompts")
    parser.add_argument("--n_harmless", type=int, default=N_HARMLESS, help="Number of harmless prompts")
    parser.add_argument("--n_questions", type=int, default=N_QUESTIONS, help="Extraction questions per role")
    args = parser.parse_args()

    n_harmful = args.n_harmful
    n_harmless = args.n_harmless
    n_questions = args.n_questions
    model_name = args.model

    data_dir = PROJECT_ROOT / "assistant-axis" / "data"
    output_dir = PROJECT_ROOT / args.output_dir

    log.info(f"Model: {model_name}")
    log.info(f"Device: {args.device}")
    log.info(f"Output: {output_dir}")

    # Get model config
    config = safe_get_config(model_name)
    if config:
        log.info(f"Model config: {config}")

    # Load model once
    t0 = time.time()
    log.info("Loading model...")
    pm = ProbingModel(model_name, device=args.device)
    log.info(f"Model loaded in {time.time() - t0:.1f}s")
    log.info(f"Hidden size: {pm.hidden_size}, Layers: {len(pm.get_layers())}")

    # If config wasn't available pre-load, detect from model
    if config is None:
        num_layers = len(pm.get_layers())
        config = {
            "target_layer": num_layers // 2,
            "total_layers": num_layers,
            "short_name": "Unknown",
        }
        log.info(f"Inferred config from model: {config}")

    # Part 1: Refusal direction
    t1 = time.time()
    refusal_dir, refusal_norms = extract_refusal_direction(
        pm, n_harmful=n_harmful, n_harmless=n_harmless,
    )
    log.info(f"Refusal direction extraction took {time.time() - t1:.1f}s")

    torch.cuda.empty_cache()
    gc.collect()

    # Part 2: Approximate assistant axis
    t2 = time.time()
    axis, role_vectors = extract_approximate_axis(pm, data_dir, n_questions=n_questions)
    log.info(f"Assistant axis extraction took {time.time() - t2:.1f}s")

    # Part 3: Compare and report
    results = compare_and_report(
        refusal_dir, refusal_norms, axis, role_vectors,
        output_dir, config, model_name,
    )

    total_time = time.time() - t0
    log.info(f"\nTotal runtime: {total_time:.1f}s ({total_time / 60:.1f} minutes)")

    # Clean up
    pm.close()


if __name__ == "__main__":
    main()
