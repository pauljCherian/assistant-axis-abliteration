#!/usr/bin/env python3
"""Mini-benchmarks on baseline vs abliterated model.

Replicates the paper's finding that TruthfulQA drops after abliteration, and
measures the gap vs random-direction control.

Task: TruthfulQA MC1 (multiple-choice, single correct)
  For each question, score each choice by total log-prob of the full
  "Q: {q}\\nA: {c}" sequence. Pick the lowest-loss choice. Accuracy = correct/total.

Inputs:
  results/comparison/refusal_direction.pt

Outputs:
  results/comparison/mini_benchmarks.json
  results/comparison/mini_benchmarks.md
"""
import argparse
import json
import time
from datetime import datetime
from pathlib import Path

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer


MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct"
LAYER = 16


def orthogonalize_inplace(model, r):
    with torch.no_grad():
        emb = model.model.embed_tokens.weight
        emb.sub_(torch.outer(emb @ r, r))
        for layer in model.model.layers:
            layer.self_attn.o_proj.weight.sub_(
                torch.outer(r, r @ layer.self_attn.o_proj.weight))
            layer.mlp.down_proj.weight.sub_(
                torch.outer(r, r @ layer.mlp.down_proj.weight))


def snapshot(model):
    return {
        "embed": model.model.embed_tokens.weight.detach().clone(),
        "o_proj": [l.self_attn.o_proj.weight.detach().clone()
                   for l in model.model.layers],
        "down_proj": [l.mlp.down_proj.weight.detach().clone()
                      for l in model.model.layers],
    }


def restore(model, snap):
    with torch.no_grad():
        model.model.embed_tokens.weight.copy_(snap["embed"])
        for i, l in enumerate(model.model.layers):
            l.self_attn.o_proj.weight.copy_(snap["o_proj"][i])
            l.mlp.down_proj.weight.copy_(snap["down_proj"][i])


def choice_logprob(model, tokenizer, prompt, choice, device):
    """Sum log-prob of `choice` tokens given `prompt`."""
    prompt_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
    full_ids = tokenizer(prompt + choice, return_tensors="pt").input_ids.to(device)
    prompt_len = prompt_ids.shape[1]
    with torch.no_grad():
        out = model(full_ids)
    logits = out.logits[0, prompt_len - 1 : -1]  # predict choice tokens
    target_ids = full_ids[0, prompt_len:]
    logprobs = torch.log_softmax(logits.float(), dim=-1)
    return float(logprobs.gather(-1, target_ids.unsqueeze(-1)).sum())


def evaluate_mc1(model, tokenizer, dataset, device, n_questions=100):
    correct = 0
    total = 0
    details = []
    for i, ex in enumerate(dataset):
        if total >= n_questions:
            break
        question = ex.get("question")
        choices = ex["mc1_targets"]["choices"]
        labels = ex["mc1_targets"]["labels"]
        try:
            correct_idx = labels.index(1)
        except ValueError:
            continue
        prompt = f"Q: {question}\nA:"
        scores = [choice_logprob(model, tokenizer, prompt, " " + c, device)
                  for c in choices]
        pred = int(max(range(len(scores)), key=lambda k: scores[k]))
        details.append({
            "q": question[:100],
            "pred": pred,
            "correct": correct_idx,
            "ok": pred == correct_idx,
        })
        if pred == correct_idx:
            correct += 1
        total += 1
        if (i + 1) % 20 == 0:
            print(f"    {total}/{n_questions}  running acc = {correct/total:.3f}")
    return {"accuracy": correct / max(total, 1),
            "n_correct": correct, "n_total": total, "details": details}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--refusal_direction",
                    default="results/comparison/refusal_direction.pt")
    ap.add_argument("--output_dir", default="results/comparison")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--n_questions", type=int, default=100)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)

    print(f"=== Mini-benchmarks @ {datetime.now().strftime('%Y-%m-%d %H:%M')} ===")
    print("Loading TruthfulQA MC1…")
    tqa = load_dataset("truthful_qa", "multiple_choice", split="validation")
    tqa = tqa.shuffle(seed=args.seed).select(range(min(args.n_questions, len(tqa))))
    print(f"  {len(tqa)} questions")

    r_all = torch.load(args.refusal_direction, map_location="cpu", weights_only=False)
    r_ref = (r_all[LAYER] / r_all[LAYER].norm()).float()
    torch.manual_seed(args.seed)
    r_rand = torch.randn_like(r_ref)
    r_rand = r_rand / r_rand.norm()

    print(f"Loading {MODEL_NAME}…")
    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, torch_dtype=torch.bfloat16, device_map=args.device,
    )
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model.eval()
    print(f"  loaded in {time.time()-t0:.1f}s")

    snap = snapshot(model)
    r_ref_dev = r_ref.to(dtype=torch.bfloat16, device=args.device)
    r_rand_dev = r_rand.to(dtype=torch.bfloat16, device=args.device)

    print("\n[1/3] baseline TruthfulQA MC1…")
    baseline = evaluate_mc1(model, tokenizer, tqa, args.device, args.n_questions)
    print(f"  accuracy = {baseline['accuracy']:.4f}  ({baseline['n_correct']}/{baseline['n_total']})")

    print("\n[2/3] real abliteration TruthfulQA MC1…")
    orthogonalize_inplace(model, r_ref_dev)
    real = evaluate_mc1(model, tokenizer, tqa, args.device, args.n_questions)
    print(f"  accuracy = {real['accuracy']:.4f}  ({real['n_correct']}/{real['n_total']})")

    restore(model, snap)

    print("\n[3/3] random-control TruthfulQA MC1…")
    orthogonalize_inplace(model, r_rand_dev)
    rand = evaluate_mc1(model, tokenizer, tqa, args.device, args.n_questions)
    print(f"  accuracy = {rand['accuracy']:.4f}  ({rand['n_correct']}/{rand['n_total']})")

    report = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "model": MODEL_NAME,
        "layer": LAYER,
        "n_questions": len(tqa),
        "truthfulqa_mc1": {
            "baseline_acc": baseline["accuracy"],
            "real_abliteration_acc": real["accuracy"],
            "random_control_acc": rand["accuracy"],
            "delta_real_pct": (real["accuracy"] - baseline["accuracy"]) * 100,
            "delta_random_pct": (rand["accuracy"] - baseline["accuracy"]) * 100,
        },
    }
    (out_dir / "mini_benchmarks.json").write_text(json.dumps(report, indent=2))

    md = [
        "# Mini-benchmarks: TruthfulQA MC1",
        "",
        f"Generated: {report['generated_at']}",
        f"Model: `{MODEL_NAME}`, n={report['n_questions']} questions",
        "",
        "| Variant | TruthfulQA MC1 acc | Δ vs baseline |",
        "|---|---|---|",
        f"| baseline | {baseline['accuracy']:.4f} | — |",
        f"| **real abliteration** | **{real['accuracy']:.4f}** | "
        f"**{report['truthfulqa_mc1']['delta_real_pct']:+.2f} pp** |",
        f"| random control | {rand['accuracy']:.4f} | "
        f"{report['truthfulqa_mc1']['delta_random_pct']:+.2f} pp |",
        "",
        "## Interpretation",
        "",
        "Paper's finding: TruthfulQA consistently drops after abliteration — "
        "interpreted as refusal direction being partially entangled with truth assessment.",
        "",
        "If real abliteration's drop exceeds the random control's drop, we replicate "
        "that finding: the refusal direction specifically carries truth information.",
        "",
    ]
    (out_dir / "mini_benchmarks.md").write_text("\n".join(md))
    print(f"\nWrote {out_dir}/mini_benchmarks.{{json,md}}")


if __name__ == "__main__":
    main()
