"""Merge part-1 (α=0.5..3) and extension (α=4, 5, 7) α-pilot results into one summary.

Loads:
  results/persona_vectors/alpha_pilot_part1_a0to3/{trait}_alpha_pilot.json   # part-1
  results/persona_vectors/alpha_pilot/{trait}_alpha_pilot.json               # extension
Merges by_alpha + summary, re-picks α* on the union (excluding baseline α=0),
writes:
  results/persona_vectors/alpha_pilot/{trait}_alpha_pilot_merged.json
  results/persona_vectors/alpha_pilot/alpha_star_picks_merged.json
  results/persona_vectors/alpha_pilot/summary_merged.md
"""
import json
import math
from pathlib import Path

PV = Path("/scratch/paulc/assistant-axis-abliteration/results/persona_vectors/alpha_pilot")
PART1 = PV.parent / "alpha_pilot_part1_a0to3"
TRAITS = ["evil", "sycophantic", "apathetic", "humorous"]
COHERENCE_FLOOR = 50
SAFETY_MARGIN = 70  # plan's rule: drop α if coh < 70 for safer full-pipeline run


def pick(summary_per_alpha, sweep_alphas, floor):
    qual = [(a, summary_per_alpha[a]) for a in sweep_alphas
            if summary_per_alpha[a]["mean_coh"] >= floor
            and not math.isnan(summary_per_alpha[a]["mean_trait"])]
    if not qual:
        return None, f"no α with mean_coh>={floor}", True
    a, s = max(qual, key=lambda kv: kv[1]["mean_trait"])
    return a, f"argmax mean_trait ({s['mean_trait']:.1f}) s.t. mean_coh ({s['mean_coh']:.1f}) >= {floor}", False


merged_per_trait = {}
for t in TRAITS:
    p1 = json.loads((PART1 / f"{t}_alpha_pilot.json").read_text())
    p2 = json.loads((PV / f"{t}_alpha_pilot.json").read_text())
    by_alpha = {**p1["by_alpha"], **p2["by_alpha"]}
    summary = {**p1["summary"], **p2["summary"]}
    alphas_in_order = sorted(by_alpha.keys(), key=lambda x: float(x))
    sweep_alphas = sorted([a for a in by_alpha.keys() if float(a) > 0], key=lambda x: float(x))

    a_floor, reason_floor, collapse_floor = pick(summary, sweep_alphas, COHERENCE_FLOOR)
    a_marg, reason_marg, collapse_marg = pick(summary, sweep_alphas, SAFETY_MARGIN)

    merged_per_trait[t] = {
        "trait": t,
        "L_inject": 12,
        "alphas_in_order": alphas_in_order,
        "by_alpha": by_alpha,
        "summary": summary,
        "alpha_star_floor50": a_floor,
        "alpha_star_floor50_reason": reason_floor,
        "alpha_star_margin70": a_marg,
        "alpha_star_margin70_reason": reason_marg,
    }
    (PV / f"{t}_alpha_pilot_merged.json").write_text(json.dumps(merged_per_trait[t], indent=2))

picks = {t: {
    "alpha_star_floor50": d["alpha_star_floor50"],
    "alpha_star_floor50_reason": d["alpha_star_floor50_reason"],
    "alpha_star_margin70": d["alpha_star_margin70"],
    "alpha_star_margin70_reason": d["alpha_star_margin70_reason"],
} for t, d in merged_per_trait.items()}
(PV / "alpha_star_picks_merged.json").write_text(json.dumps(picks, indent=2))

# markdown
md = ["# Stage 3 — α-pilot MERGED (part-1 + extension) summary", ""]
md.append("L_inject = 12, judge = gpt-4.1-mini-2025-04-14")
md.append("")
md.append("## α* picks — two policies")
md.append("")
md.append("| Trait | α* (floor=50, Anthropic) | α* (margin=70, plan-conservative) |")
md.append("|---|---|---|")
for t, d in merged_per_trait.items():
    md.append(f"| {t} | **{d['alpha_star_floor50']}** — {d['alpha_star_floor50_reason']} | **{d['alpha_star_margin70']}** — {d['alpha_star_margin70_reason']} |")

md.append("")
md.append("## Per-trait sweep (mean_trait_score / mean_coherence)")
for t, d in merged_per_trait.items():
    md.append("")
    md.append(f"### {t}")
    md.append("")
    md.append("| α | n | mean_trait | std_trait | mean_coh | std_coh |")
    md.append("|---|---|---|---|---|---|")
    for a in d["alphas_in_order"]:
        s = d["summary"][a]
        md.append(f"| {a} | {s['n']} | {s['mean_trait']:.1f} | {s['std_trait']:.1f} | {s['mean_coh']:.1f} | {s['std_coh']:.1f} |")
(PV / "summary_merged.md").write_text("\n".join(md))

print("\n".join(md))
print()
print(f"saved {PV}/summary_merged.md")
print(f"saved {PV}/alpha_star_picks_merged.json")
