# Phase F — Experimental Design (2026-04-25)

## Recap of where we are

Phase E produced a striking but partial null:
- **PC1 preserved** (cos = 0.913, role-loadings r = 0.970, top-5 subspace angles ≤ 27°)
- **But** mean centroid shifts ~3.0 units, **55% along the refusal direction** (sample-wide rigid translation, not rotation)
- Pre-registered safety-adjacent prediction failed (p = 0.55)

So abliteration moves the persona space *as a rigid body* without distorting it. Compelling, but reviewers will reasonably ask: "Can your pipeline detect *any* perturbation that rotates PC1?" Phase F addresses that with a tiered, low-risk-first set of interventions.

## All options revisited (from earlier discussion)

| ID | Intervention | Cloud $ | Dev | Wall-clock | Predicted shift |
|---|---|---|---|---|---|
| **G** | Refusal-projection sanity check on existing data | **$0** | 30 min | 30 min | Sharpens existing finding |
| **D** | Anthropic Persona Vectors (residual injection / weight bake) | $37 / run | 2 days | 1 day / run | LARGE — designed for this |
| **F** | Goodfire SAE-l19 feature ablation | $37 | 3-5 days | 1 day after dev | Unknown — high novelty |
| **C** (Heretic) | Multi-direction abliteration | $37 | 1 day | 1 day | Medium |
| **A1/A4** (Stheno, Dolphin) | Roleplay/persona fine-tunes | $37 each | 0.5 day | 1 day each | **Likely null** (paper finding: base→instruct PC1 ≈ 1.0 — fine-tunes preserve PC1) |
| **E** (full taxonomy) | All of above | $148+ | 1 week | 5 days | — |

User decision (today): **fine-tunes are out**, persona vectors lead, SAE / refusal-projection are still on the table.

---

## Recommended Phase F sequence

A staged plan with cheap-and-decisive checks first. Each stage gates the next.

```
Stage 0 (free, 30 min)     — Option G: refusal-projection sanity check
Stage 1 (free, ~3 hr)      — Extract persona vectors on Llama-3.1-8B for 3 traits
Stage 2 (free, ~30 min)    — Geometric precheck: cos similarities
Stage 3 (free, ~2 hr)      — α-coefficient pilot on 10 roles
Stage 4 ($37, 1 day)       — Full pipeline run #1: best (trait, α)
[REVIEW DECISION POINT]
Stage 5 ($37, 1 day)       — Run #2: second trait OR SAE feature ablation
Stage 6 ($37, 1 day)       — Run #3 (optional): Heretic OR alternate L_inject
```

Total budget: $0–$111. Time: 3–8 days.

---

## Stage 0 — Refusal-projection sanity check (free, 30 min)

**Goal:** test whether the rigid-translation effect we saw is *entirely* the refusal direction leaking through skip connections, or whether there's an orthogonal-to-refusal perturbation as well.

**Procedure** (lisplab1, no GPU, ~50 lines of python):
1. Load `results/abliterated/vectors/*.pt` (276 abliterated role vectors @ L=16).
2. Load `results/comparison/refusal_direction_from_mlabonne.pt` (global direction).
3. For each abl role vector v: `v_proj = v − (v·r̂)r̂` (project out refusal).
4. Recompute axis from projected vectors → `axis_proj`.
5. Re-compute the comparison metrics with `axis_proj` instead of `axis_abl`:
   - cos(orig_axis, axis_proj)
   - cos(PC1_orig, PC1_proj)
   - centroid shift after projection
   - top-5 subspace angles

**Outcomes:**
- If `cos(orig_axis, axis_proj) → 0.99`: the entire abliteration effect on persona space WAS the refusal direction (leaked through embedding + skip path even though `down_proj` was orthogonalized). Rigid-translation hypothesis confirmed cleanly.
- If `cos(orig_axis, axis_proj) ≈ 0.876` (unchanged): there's a non-refusal residual perturbation. Worth identifying which direction it's along (top-1 SVD of the residual centroid shift).

**Deliverable:** `results/comparison/refusal_projection_analysis.{json,md}`. Uses existing data, no new compute.

---

## Stage 1 — Extract persona vectors (lisplab1, ~3 hr, free)

### What we know from the Anthropic paper (Chen et al. 2025, arXiv:2507.21509)

**Authors:** Chen, Arditi, Sleight, Evans, Lindsey (Anthropic Fellows). 63-page v3 (Sep 2025).

**Code:** `github.com/safety-research/persona_vectors`, Apache-2.0, last commit 2026-04-22 (active).

**Available traits (7 total):** `evil`, `sycophantic`, `hallucinating`, `optimistic`, `impolite`, `apathetic`, `humorous`. JSON artifacts ship in `data_generation/trait_data_extract/{trait}.json` — each has 5 instruction pairs (system-prompt pos/neg) × 20 extraction questions × 20 held-out eval questions.

**Optimal layer for Llama-3.1-8B-Instruct:** **L=16** for all 3 main traits per their Figure 13.
**This is exactly our extraction layer — the critical experimental design problem (see §"Critical Caveat" below).**

**Coefficient sweet spot:** α ∈ [1.5, 2.0]. Above 2.5 coherence collapses. Vectors stored as raw mean-difference, **NOT unit-normed**.

**Steering mode:** `response` (only adds α·v to the most recent token during decoding) is what they recommend for inducing a persistent trait.

**No pre-extracted vectors are committed to the repo.** We extract ourselves.

### Procedure

```bash
# 1. Clone repo (in lisplab1 venv, separate from main project to avoid dependency clash)
cd /scratch/paulc/assistant-axis-abliteration
git clone https://github.com/safety-research/persona_vectors third_party/persona_vectors

# 2. The repo's pinned deps (torch 2.6, vllm 0.8.5, transformers 4.52.3) clash with our 2.10/0.19/4.57.
#    But the only files we need (generate_vec.py, activation_steer.py) are pure HF + pytorch — no vllm.
#    Patch generate_vec.py to import from our existing venv. No fresh install needed.

# 3. Extract pos/neg behavior CSVs for 3 traits using our existing transformers stack.
#    Use Llama-3.1-8B-Instruct (override the repo's Qwen default).
.venv/bin/python third_party/persona_vectors/eval/eval_persona.py \
    --model meta-llama/Llama-3.1-8B-Instruct \
    --trait evil \
    --output_path results/persona_vectors/evil_pos.csv \
    --persona_instruction_type pos \
    --version extract

# (repeat for neg, then for sycophantic + hallucinating)

# 4. Compute mean-diff vectors at every layer.
.venv/bin/python third_party/persona_vectors/generate_vec.py \
    --model meta-llama/Llama-3.1-8B-Instruct \
    --pos_path results/persona_vectors/evil_pos.csv \
    --neg_path results/persona_vectors/evil_neg.csv \
    --trait evil \
    --save_dir results/persona_vectors/

# Output: results/persona_vectors/evil_response_avg_diff.pt — shape [33, 4096], fp32
```

### Trait selection: which 3 traits?

Pick by predicted impact on PC1 of the Assistant Axis:

| Trait | Why interesting | Predicted PC1 effect |
|---|---|---|
| **evil** | Most validated; opposite of "helpful Assistant" prior; maximal moral contrast | Large (Assistant pole pulled toward "non-Assistant") |
| **sycophantic** | Tests *amplifying* the Assistant prior — opposite hypothesis to evil | Could shift Assistant centroid further along PC1 |
| **hallucinating** | Cleanly orthogonal to safety/refusal axis (different mechanism) | Tests whether non-safety traits also rotate PC1 |

Skip `optimistic` / `impolite` / `apathetic` / `humorous` for v1 — less validated, weaker contrasts in the paper.

---

## Stage 2 — Geometric precheck (free, 30 min)

For each extracted vector `v_T` at every layer ℓ:

```python
v = torch.load(f"results/persona_vectors/{trait}_response_avg_diff.pt")  # [33, 4096]
axis = torch.load("results/original/axis.pt")                             # [4096]
refusal = torch.load("results/comparison/refusal_direction_from_mlabonne.pt")["per_layer"]  # [32, 4096]

for layer in [8, 12, 14, 16, 18, 20]:
    v_l = v[layer] / v[layer].norm()
    print(f"L={layer}: cos(v_{trait}, axis)={cos(v_l, axis):.3f}, "
          f"cos(v_{trait}, refusal)={cos(v_l, refusal[layer]):.3f}")

# also: cos(v_evil, v_sycophantic), cos(v_evil, v_hallucinating) at L=16
```

**Decision rules:**
- If `|cos(v_T, axis)| < 0.05` for all 3 traits: persona vectors are nearly orthogonal to PC1 → injection won't move PC1 much → **switch traits or reconsider**.
- If `|cos(v_T, refusal)| > 0.5` for some trait: that trait vector overlaps refusal — running it would partially replicate Phase E. Pick a more orthogonal trait.
- If `cos(v_evil, v_sycophantic) > 0.7`: the contrast prompt template is dominating, traits aren't differentiated. Rerun extraction.

**Save:** `results/persona_vectors/precheck.json`.

---

## Stage 3 — α-coefficient pilot (lisplab1, ~2 hr, free)

For the chosen trait and `L_inject` (see §"Critical Caveat" below for layer choice):

1. Use `ActivationSteerer` from `activation_steer.py` (HF forward hook on `model.model.layers[L_inject]`).
2. Run a 10-role subset of our 275 archetypes (e.g., default_assistant, demon, ethicist, expert, criminal, philosopher, child, fool, guardian, saboteur).
3. Sweep α ∈ {0.5, 1.0, 1.5, 2.0, 3.0}.
4. For each (α, role): generate 20 rollouts, score:
   - Coherence (0-100): GPT-4.1-mini judge
   - Trait expression (0-100): trait-specific judge from `eval/prompts.py`
   - Role adherence (0-3): our existing assistant-axis judge

Pick α that maximizes `(trait expression / 100) × (coherence / 100) × (role adherence / 3)`. Expect optimum near α=1.5–2.0.

**Save:** `results/persona_vectors/alpha_sweep.{json,md,png}`.

---

## ⚠️ CRITICAL CAVEAT — `L_inject` vs `L_extract`

**This is the most important experimental design decision.**

- Anthropic injects and extracts at the same layer (L=16 for Llama-3.1-8B). They measure trait expression in *generated text* (downstream of the injection).
- We measure persona space in the *residual stream activation* (at L=16).

If we inject at L=16 and measure at L=16, we trivially observe `h + α·v` — vector addition is linear, PC1 will rotate by an amount that's a deterministic function of α and v. **Tautological — uninformative.**

Literature confirms this is a real problem:
- arXiv:2602.06801 ("Non-Identifiability of Steering Vectors") — same-layer injection-measurement is fundamentally non-identifiable.
- arXiv:2603.24543 ("Safety Pitfalls of Steering Vectors") — explicitly warns against same-layer self-confirmation.
- arXiv:2603.12298 ("Global Evolutionary Steering") — recommends gap of 3–6 layers between L_inject and L_extract.

**Solution:** inject at L_inject < L_extract = 16. Recommended values:

| L_inject | Gap to L=16 | Trade-off |
|---|---|---|
| L=8 | 8 layers | Maximum nonlinear processing; but Anthropic's Fig 13 shows trait expression weakest at L=8 (might fail to induce trait at all) |
| **L=12** | **4 layers** | **Sweet spot**: enough nonlinear processing to be non-trivial, still within trait-expression band |
| L=14 | 2 layers | Marginal — close to same-layer trivial case |

**Default: L_inject = 12.** Stretch experiment: also run L_inject = 8 to test propagation.

We must verify in Stage 3 that injecting at L=12 still induces the trait in the model's outputs (use the trait-expression judge). If trait expression is weak at L=12, fall back to L=14 or accept a higher tautology factor.

---

## Stage 4 — Full pipeline run #1 (Lambda + lisplab1, ~$37, ~1 day)

### Integration path: bake-as-bias (vLLM-compatible)

The `ActivationSteerer` is a HF forward hook — **vLLM cannot use it** (paged attention + custom forward path). Two options:

1. **Switch the pipeline to HF generation.** ~5–10× slower than vLLM. ~5–8 hr for 275 roles × 1200 rollouts on lisplab1 RTX 8000 (single-GPU); ~1.5 hr on Lambda 8× A100. Cleanest semantically (matches paper). But requires significant pipeline modification.

2. **Bake the vector as a permanent additive bias on `down_proj` at `L_inject`.** Mathematically equivalent (for `positions="all"`) to a forward hook adding α·v at the block output. Llama's `down_proj` has `bias=False` by default — we need to enable it:

   ```python
   from torch import nn
   v = torch.load("results/persona_vectors/evil_response_avg_diff.pt")[L_inject]  # fp32 [4096]
   model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.1-8B-Instruct", torch_dtype=torch.bfloat16)
   mlp = model.model.layers[L_inject].mlp
   old = mlp.down_proj
   new = nn.Linear(old.in_features, old.out_features, bias=True, dtype=old.weight.dtype, device=old.weight.device)
   new.weight.data = old.weight.data.clone()
   new.bias = nn.Parameter((alpha * v).to(old.weight.dtype).to(old.weight.device))
   mlp.down_proj = new
   model.save_pretrained(f"models/llama-3.1-8b-{trait}-steered-L{L_inject}-a{alpha}")
   ```

   This produces a standalone HF checkpoint that vLLM loads and serves normally. No pipeline changes needed beyond pointing `MODEL_ID` at the new directory. Same architecture as our mlabonne-abliterated model run.

**Recommendation: option 2 (bake-as-bias).** Mirrors the abliteration paradigm (modify weights → run pipeline as-is) and reuses every piece of our existing infrastructure unchanged.

**Caveat to acknowledge in the writeup:** baking gives `positions="all"` semantics (steering applies to every token, including prompt). The paper's `positions="response"` mode applies only during generation. For our pipeline this is actually preferable — we want the persona to be *constitutive* of the model, not just an inference-time switch. But it's a methodological deviation worth noting.

### Run sequence

```bash
# 1. Bake weight-modified checkpoint on lisplab1 (~5 min)
.venv/bin/python scripts/11_bake_persona_vector.py \
    --trait evil --layer 12 --alpha 1.5 \
    --output models/llama-3.1-8b-evil-steered-L12-a1.5

# 2. Push to HF dataset (~3 min, 16 GB)
huggingface-cli upload pandaman007/assistant-axis-abliteration-vectors \
    models/llama-3.1-8b-evil-steered-L12-a1.5 llama-3.1-8b-evil-steered-L12-a1.5/model_files

# 3. Modify cloud_setup_intervention.sh to snapshot_download from this prefix.
#    Lambda 8× A100 SXM4 (~1.5 hr, $24).

# 4. Judge + axis on lisplab1 (~3 hr at current OpenAI throughput, $13).

# 5. Run 05_compare_persona_spaces.py with new --abliterated_dir results/evil-steered-L12-a1.5/
```

Total cost: ~$37. Total wall-clock: ~6 hr active + judge time.

---

## Stage 5 — Run #2 decision

After run #1 returns, three branches:

**Branch 1: PC1 rotated significantly (cos < 0.7)** — pipeline detects intervention.
→ Run a second trait (sycophancy) to verify trait-direction-specificity. The displacement of roles should align with `v_sycophantic`, distinct from the evil run.

**Branch 2: PC1 still preserved (cos > 0.9)** — striking finding. Even direct residual injection at the trait's own optimal layer doesn't rotate PC1.
→ Run alternate L_inject = 8 (further from L_extract = 16) to confirm. If still no rotation, PC1 is *exceptionally* robust — major paper finding.

**Branch 3: PC1 partially rotated, role displacements predict trait** — the cleanest result.
→ Two-result paper. Stop spending. Maybe add Heretic for a 3-point spectrum.

---

## Stage 6 — Optional comparators

| Comparator | When to add | Adds what |
|---|---|---|
| **SAE feature ablation (F)** | If we want a third, mech-interp-flavored result | Tests "is PC1 localized to specific SAE features?" Different from refusal abl AND persona vectors. ~$37 + 3-5 dev days. |
| **Heretic multi-direction abliteration** | If we want to test "weight surgery aggressiveness" | Direct contrast to mlabonne. ~$37, 1 day. Predicted: medium shift, different rigid-translation profile. |
| **Alignment-faking direction (Greenblatt 2024)** | Bonus stretch | arXiv:2604.20995 extracted a steering vector from Greenblatt's free-tier/paid-tier contrast. We could compare cos(faking_direction, axis) and use it as another steering direction. |

---

## Hiccups / risk register

1. **Same-layer tautology** (already discussed). Mitigation: L_inject = 12.

2. **Trait vector orthogonality to PC1.** If `cos(v_evil, axis) ≈ 0`, even strong steering won't rotate PC1, just translate the cloud (same as Phase E). Mitigation: Stage 2 precheck.

3. **Vocabulary drift.** Our pipeline uses `LlamaForCausalLM` tokenizer for Llama-3.1-8B (vocab 128256). Persona Vectors repo defaults to Qwen (vocab 151643). Make sure all extraction passes use Llama tokenizer. Trivial — just override `--model_name`.

4. **fp32 vector vs bf16 model.** Vectors saved as fp32; model runs in bf16. Bias parameter must be cast to bf16 before saving. ~10⁻⁴ rounding error on each parameter. Negligible.

5. **Coherence collapse at high α.** If α=1.5 yields garbled outputs, judge can't score role adherence reliably. Mitigation: Stage 3 α-sweep with coherence floor of 75/100. Drop α to 1.0 if needed.

6. **Judge contamination.** If we steer to "evil," GPT-4.1-mini may refuse to score outputs (its own safety filters). Mitigation: spot-check 10 outputs manually. If judge refuses, switch to GPT-4o-mini or use Anthropic's judge from their repo.

7. **Off-target capability degradation.** α=1.5 reduces MMLU 2-5 pts (per arXiv:2602.04903). Our pipeline doesn't measure MMLU. If we want to claim "the model still answered coherently," add an MMLU subset run on the steered checkpoint as integrity check. Optional.

8. **Linear composition assumption.** If we run multiple traits, the paper says ≤3 compose linearly; beyond that geometry breaks (arXiv:2602.15669). For v1, run traits independently (separate steered checkpoints), no composition.

9. **Bake-vs-hook divergence.** Baking applies α·v at every token; hook-with-`positions="response"` applies only at the most recent generated token. Effects measured in extracted activations may diverge — bake produces stronger, more uniform shift; hook produces smaller but more naturalistic shift. We pick bake for vLLM compatibility and document the difference.

10. **"Model rejection" of injected vector.** If the model finds the steering vector very out-of-distribution, downstream layers may "compensate" by rotating their own activations. This could produce spurious PC1 effects unrelated to persona. Mitigation: report Procrustes residual + per-role displacement direction (not just axis cosine). If the rotation is along v_T, real effect; if scrambled, artifact.

11. **Repo-version mismatch.** `safety-research/persona_vectors` pins torch 2.6, transformers 4.52.3, vllm 0.8.5. Our stack: 2.10 / 4.57.6 / 0.19. The relevant code (`generate_vec.py`, `activation_steer.py`) is pure pytorch + HF, version-tolerant. We just import directly from our venv, no fresh install. **Skip Unsloth-dependent files** (`training.py`, `sft.py`).

12. **`--layer` off-by-one** in their CLI: `--layer 20` means `model.model.layers[19]`. Their saved tensors are indexed `[0, 32]` (layer 0 = embedding). Be very careful with indexing when baking. Check by running their `generate_vec.py` and inspecting tensor shapes.

13. **Pipeline cost estimate.** Each full run: $24 Lambda + $13 OpenAI ≈ $37. A 3-trait taxonomy = $111 + judge time. Within 3-week constraint.

14. **Reviewer pushback: "isn't this just CAA?"** Yes — Persona Vectors = CAA with LLM-generated contrast prompts and a more systematic α-validation protocol. Cite Rimsky et al. 2024 (arXiv:2312.06681) as the methodology origin.

15. **Failure mode: even Stage 3 pilot looks bad.** If α=2.0 at L=12 gives trait_expression < 30, the steering isn't working at our chosen layer. Fall back: (a) try L=14 (closer to optimum) — accepts more tautology; (b) use prompt-mode steering instead of bake (closer to paper).

---

## File deliverables

Create on lisplab1:

```
scripts/
├── 11_extract_persona_vectors.py    # wrapper around generate_vec.py
├── 12_persona_geometric_precheck.py # Stage 2
├── 13_alpha_pilot.py                 # Stage 3
├── 14_bake_persona_vector.py         # Stage 4 weight modification
├── 15_refusal_projection_check.py    # Stage 0 (free sanity check)
└── cloud_setup_intervention.sh       # Generic Lambda bootstrap (parameterized by HF model id)

results/persona_vectors/
├── {evil,sycophantic,hallucinating}_response_avg_diff.pt   # extracted vectors [33,4096]
├── precheck.json                                            # Stage 2 cosines
└── alpha_sweep.{json,md,png}                                # Stage 3 results

models/
└── llama-3.1-8b-{trait}-steered-L{N}-a{α}/   # baked checkpoint per Stage 4

results/{trait}-steered/
├── responses/, activations/, scores/, vectors/, axis.pt   # full pipeline output

results/comparison/
├── refusal_projection_analysis.{json,md}     # Stage 0
└── persona_vector_comparison.{json,md}       # cross-comparison original vs steered
```

---

## What to do NOW

User decides between:
- **Option α**: do Stage 0 (refusal-projection check) right now — free, ~30 min, sharpens the existing finding.
- **Option β**: jump directly to Stage 1 (extract persona vectors) — needs lisplab1 GPU, ~3 hr. Then Stage 2 precheck right after.
- **Option γ**: both α and β in parallel — Stage 0 on CPU, Stage 1 on GPU, no resource conflict.

Recommend **γ**: parallelism is free. Stage 0 produces a sharp result on existing data; Stage 1 begins the persona-vector pipeline. Both feed into the writeup.

After Stages 0–3 are done (~1 working day, no cloud spend), we have data to make a sharp decision about whether/how to do Stage 4. That's the right gate — don't pay $37 until the precheck shows the trait vector should move PC1.
