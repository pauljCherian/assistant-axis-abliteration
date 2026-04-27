"""Stage 1: Extract persona vectors for Phase F.

For each trait (7 Anthropic + 1 custom assistant_identity):
1. Load trait JSON: 5 instruction pairs × 20 questions × 2 sides (pos/neg)
2. Generate completions on Llama-3.1-8B-Instruct with vLLM (n_samples per pair)
3. Re-run on HF transformers with output_hidden_states=True to get [33, 4096] activations per sample
4. Average response-token activations per layer per side
5. Compute mean-diff: pos - neg → [33, 4096] fp32
6. Save as `results/persona_vectors/{trait}_response_avg_diff.pt`

Skips Anthropic's judge filtering for v1 — can be added if signal is noisy.

Usage:
    .venv/bin/python scripts/11_extract_persona_vectors.py --trait evil
    .venv/bin/python scripts/11_extract_persona_vectors.py --all
"""
import argparse
import gc
import json
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm

REPO = Path("/scratch/paulc/assistant-axis-abliteration")
MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct"
ANTHROPIC_TRAIT_DIR = Path("/thayerfs/home/f006vv2/persona_vectors/data_generation/trait_data_extract")
CUSTOM_TRAIT_DIR = REPO / "data/persona_vectors"
OUT_DIR = REPO / "results/persona_vectors"

ANTHROPIC_TRAITS = ["evil", "sycophantic", "apathetic", "humorous",
                    "impolite", "hallucinating", "optimistic"]
CUSTOM_TRAITS = ["assistant_identity"]
ALL_TRAITS = ANTHROPIC_TRAITS + CUSTOM_TRAITS


def load_trait(trait):
    """Return dict with `instruction` (list of {pos,neg}), `questions`, `eval_prompt`."""
    if trait in ANTHROPIC_TRAITS:
        path = ANTHROPIC_TRAIT_DIR / f"{trait}.json"
    else:
        path = CUSTOM_TRAIT_DIR / f"{trait}.json"
    with open(path) as f:
        return json.load(f)


def build_prompts(tokenizer, trait_data, side):
    """Build chat-templated prompts.

    Returns list of dicts: {prompt_text, instruction_idx, question_idx}.
    """
    assert side in ("pos", "neg")
    prompts = []
    for inst_idx, inst_pair in enumerate(trait_data["instruction"]):
        instruction = inst_pair[side]
        for q_idx, question in enumerate(trait_data["questions"]):
            messages = [{"role": "system", "content": instruction},
                        {"role": "user", "content": question}]
            prompt_text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True)
            prompts.append({
                "prompt_text": prompt_text,
                "instruction_idx": inst_idx,
                "question_idx": q_idx,
                "side": side,
            })
    return prompts


def generate_with_vllm(prompts, n_samples=2, max_tokens=300, temperature=1.0):
    """Generate responses with vLLM. Returns list of dicts with `answer` field added."""
    from vllm import LLM, SamplingParams
    print(f"Loading vLLM model {MODEL_NAME}...")
    llm = LLM(
        model=MODEL_NAME,
        dtype="float16",  # Turing GPUs (RTX 8000) lack bf16 support
        max_model_len=2048,
        gpu_memory_utilization=0.85,
    )
    sampling_params = SamplingParams(
        temperature=temperature,
        max_tokens=max_tokens,
        n=n_samples,
        top_p=1.0,
    )
    print(f"Generating {len(prompts)} prompts × {n_samples} samples = {len(prompts) * n_samples} responses...")
    prompt_texts = [p["prompt_text"] for p in prompts]
    outputs = llm.generate(prompt_texts, sampling_params)
    # Expand each prompt into n_samples rows
    results = []
    for prompt_dict, output in zip(prompts, outputs):
        for sample_idx, gen in enumerate(output.outputs):
            row = dict(prompt_dict)
            row["sample_idx"] = sample_idx
            row["answer"] = gen.text
            results.append(row)
    # vLLM cleanup — release GPU before HF model loads
    del llm
    gc.collect()
    torch.cuda.empty_cache()
    return results


@torch.no_grad()
def extract_activations(model, tokenizer, samples, layer_list=None):
    """Run model with output_hidden_states=True per sample, average response-token activations.

    Returns dict[layer] -> tensor of shape [N, 4096].
    """
    n_layers = model.config.num_hidden_layers
    if layer_list is None:
        layer_list = list(range(n_layers + 1))  # 0..n_layers (33 total for Llama 8B: embed + 32 blocks)

    response_avg = {l: [] for l in layer_list}
    skipped = 0
    for s in tqdm(samples, desc="extracting activations"):
        prompt = s["prompt_text"]
        answer = s["answer"]
        if not answer or not answer.strip():
            skipped += 1
            continue
        full = prompt + answer
        inputs = tokenizer(full, return_tensors="pt", add_special_tokens=False).to(model.device)
        prompt_len = len(tokenizer.encode(prompt, add_special_tokens=False))
        if inputs.input_ids.shape[1] <= prompt_len:
            skipped += 1
            continue
        out = model(**inputs, output_hidden_states=True)
        for l in layer_list:
            avg = out.hidden_states[l][:, prompt_len:, :].mean(dim=1).detach().cpu().float()
            response_avg[l].append(avg.squeeze(0))
        del out
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    if skipped:
        print(f"  skipped {skipped}/{len(samples)} samples (empty/short answer)")
    return {l: torch.stack(response_avg[l]) for l in layer_list}


def compute_mean_diff(activations_pos, activations_neg, layer_list=None):
    """Per-layer mean(pos) - mean(neg). Returns [n_layers, 4096] fp32."""
    if layer_list is None:
        layer_list = sorted(activations_pos.keys())
    diffs = [activations_pos[l].mean(0) - activations_neg[l].mean(0) for l in layer_list]
    return torch.stack(diffs).float()


def extract_one_trait(trait, tokenizer, generation_results=None, n_samples=2):
    """Run full pipeline for one trait. Returns the [33, 4096] mean-diff tensor + metadata."""
    print(f"\n=== TRAIT: {trait} ===")
    trait_data = load_trait(trait)
    prompts_pos = build_prompts(tokenizer, trait_data, "pos")
    prompts_neg = build_prompts(tokenizer, trait_data, "neg")
    print(f"  {len(prompts_pos)} pos prompts, {len(prompts_neg)} neg prompts (× {n_samples} samples)")

    # Generation step (vLLM) is done in a separate phase to avoid loading both vLLM + HF
    # If generation_results is provided, use them; otherwise generate now
    if generation_results is None:
        all_prompts = prompts_pos + prompts_neg
        generation_results = generate_with_vllm(all_prompts, n_samples=n_samples)

    samples_pos = [s for s in generation_results if s["side"] == "pos"]
    samples_neg = [s for s in generation_results if s["side"] == "neg"]
    return samples_pos, samples_neg, trait_data


def run_all_traits(traits, n_samples=2):
    """Generate then extract for all traits in two phases (avoids loading both vLLM+HF at once)."""
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Phase A: generate ALL prompts across ALL traits in one vLLM session (efficient)
    print(f"\n=== PHASE A: GENERATION (vLLM) for {len(traits)} traits ===")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    all_prompts = []
    trait_to_prompt_idx = {}
    for trait in traits:
        trait_data = load_trait(trait)
        pp = build_prompts(tokenizer, trait_data, "pos")
        pn = build_prompts(tokenizer, trait_data, "neg")
        for p in pp + pn:
            p["trait"] = trait
        idx_start = len(all_prompts)
        all_prompts.extend(pp + pn)
        idx_end = len(all_prompts)
        trait_to_prompt_idx[trait] = (idx_start, idx_end)

    t0 = time.time()
    all_results = generate_with_vllm(all_prompts, n_samples=n_samples)
    gen_minutes = (time.time() - t0) / 60
    print(f"  generation done in {gen_minutes:.1f} min, {len(all_results)} total samples")

    # save raw generations to disk so we can debug / re-extract without regenerating
    raw_path = OUT_DIR / "_raw_generations.pt"
    torch.save({"results": all_results, "trait_to_prompt_idx": trait_to_prompt_idx,
                "n_samples": n_samples, "model": MODEL_NAME}, raw_path)
    print(f"  saved raw generations to {raw_path}")

    # Phase B: load HF model, extract activations per trait
    print(f"\n=== PHASE B: ACTIVATION EXTRACTION (HF transformers) ===")
    print(f"Loading HF model {MODEL_NAME} (float16)...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, torch_dtype=torch.float16, device_map="cuda")
    model.eval()
    print(f"  loaded. n_hidden_layers={model.config.num_hidden_layers}")

    for trait in traits:
        trait_results = [s for s in all_results if s["trait"] == trait]
        samples_pos = [s for s in trait_results if s["side"] == "pos"]
        samples_neg = [s for s in trait_results if s["side"] == "neg"]
        print(f"\n  {trait}: {len(samples_pos)} pos, {len(samples_neg)} neg")

        t0 = time.time()
        act_pos = extract_activations(model, tokenizer, samples_pos)
        act_neg = extract_activations(model, tokenizer, samples_neg)
        diff = compute_mean_diff(act_pos, act_neg)
        ext_minutes = (time.time() - t0) / 60
        print(f"  extracted in {ext_minutes:.1f} min, diff shape {diff.shape}")

        # save
        out_path = OUT_DIR / f"{trait}_response_avg_diff.pt"
        torch.save({
            "vector": diff,  # [33, 4096] fp32
            "trait": trait,
            "n_pos": len(samples_pos),
            "n_neg": len(samples_neg),
            "n_samples_per_pair": n_samples,
            "model": MODEL_NAME,
            "method": "anthropic_persona_vector_unfiltered",
            "shape_note": "[layer 0..32, 4096] - layer 0 is embeddings, 1..32 are decoder blocks",
        }, out_path)
        print(f"  saved {out_path}")

    print("\nAll traits done.")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--trait", type=str, default=None,
                   help="Single trait to run; if not set, runs all")
    p.add_argument("--all", action="store_true", help="Run all 8 traits")
    p.add_argument("--n_samples", type=int, default=2,
                   help="Samples per (instruction, question) pair")
    args = p.parse_args()

    if args.all:
        traits = ALL_TRAITS
    elif args.trait:
        assert args.trait in ALL_TRAITS, f"Unknown trait {args.trait}; valid: {ALL_TRAITS}"
        traits = [args.trait]
    else:
        traits = ALL_TRAITS

    run_all_traits(traits, n_samples=args.n_samples)
