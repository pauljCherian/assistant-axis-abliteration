# Project: Assistant Axis × Abliteration → Cross-Model Persona Comparison

## Project Status (2026-05-01) — Pivot to Cross-Model Persona Comparison

**Phases A–G complete on Llama-3.1-8B-Instruct.** Within-model perturbation effects on persona space are characterized:
- Phase E (refusal abliteration): NULL — PC1 preserved (cos=0.913), space rigidly translates ~3.0 units, ~55% along refusal direction. Refusal ≠ Assistant Axis.
- Phase F (4-trait persona-vector steering): POSITIVE — PC1 rotates (cos 0.10–0.35) for evil/humorous/sycophantic; apathetic ambiguous (50% filter rate).
- Phase G (LizaT dangerous-medical fine-tune): NEGATIVE for cross-validation H3 — d(LizaT, F.evil)=7.73 > d(LizaT, original)=4.25. Narrow harmful fine-tunes don't project onto F.evil persona region.

Full results: `results/comparison/` and HF dataset `pandaman007/assistant-axis-abliteration-vectors`. Synthesis in memory `project_handoff_2026-04-29.md`.

**New direction (Phase H — current):** standardize comparison of two models' persona clouds using **semantic-contrast vectors** as a model-independent coordinate system. Weight-modification ("personality stealing") is a future stretch goal, deliberately deferred until cross-model comparison is solid.

### Phase H — Cross-model comparison via contrast vectors

**Models (locked):**
- Phi-3.5-mini-instruct: 32 layers, dim 3072, L_extract = 16
- Llama-3.2-3B-Instruct: 28 layers, dim 3072, L_extract = 14
- Layer convention: N/2 (Lu et al. canonical). Validate empirically only if results look anomalous.

**Method (locked, see `PHASE_H_DESIGN.md` for pre-registration):**
- Run AA pipeline on each model independently (276 roles, 1200 rollouts/role, judge filter, role vectors at L_extract).
- Pipeline runner: `scripts/30_phase_h_pipeline.sh <model> <output_dir> <layer> <hidden_dim>`.
- Pre-flight: `scripts/check_phase_h_ready.py` (12 tests, must exit 0).
- 5-axis contrast panel locked in `PHASE_H_DESIGN.md` — 64 unique anchors (16 per axis × 4 anchor axes; v_assistant uses all 275):
  - v_assistant (canonical), v_benevolence, v_authority, v_humor, v_critic
  - **v_humanness DROPPED** — non-human pole spans 3 incompatible sub-clusters.
  - **v_collective DROPPED 2026-05-03** — Path A: anchor contamination (virus/zeitgeist/vampire load on other axes; bio-network vs cosmic confound). v_creator/v_solitary considered+rejected.
- Project each role onto each contrast → 275×5 scalar matrix per model.
- Cross-model validation: Spearman rank correlation on 195 held-out roles per axis (r > 0.7 = transfers cleanly).
- Construct validity: 5 pre-registered held-out role pairs per axis + 3 null roles per axis (predicted to project near zero).
- Per-model z-score normalization before any absolute-magnitude comparison or shared-coordinate plotting.

**Why contrast vectors and not Procrustes:** projection-onto-own-axes sidesteps the 3072-D rotation problem entirely. Each model is measured in its native basis on axes whose meaning is defined identically in both. Procrustes is reserved for a later sanity check or a future stealing phase.

**Pre-registration discipline (carry over from Phase F/G):**
- Lock contrast definitions before computing.
- Filter rate <100% → also report unfiltered projections.
- Save filtered + unfiltered vectors in parallel.

**Compute (planned):**
- First attempt: lisplab1 RTX 8000 (free; ~10–16 hr per model). Both 8000s currently shared with another job — may need to wait or pivot to RTX 5000 (slower, tighter VRAM).
- Fallback: RunPod 1× A100 80GB, ~$5–10/model, ~3–4 hours/model.

**Open design questions:**
- Final list of contrast axes (5–10? need to define which subjective categories matter).
- Validation criteria: when do we say "axis X transfers cleanly across models" (cluster purity in each model? PC alignment with the contrast axis?).
- Whether to include base models alongside instruct (Lu et al. found base/instruct spaces nearly identical — extending to base would be a useful sanity check).

## Original Research Question (Phases A–G — historical)

Does abliterating a model (removing its refusal direction from the weight matrices) change the geometry of its persona space as defined by the Assistant Axis paper? Specifically: are the refusal direction and the Assistant Axis the same thing, partially overlapping, or orthogonal? **Answer (Phase E):** partially overlapping. The refusal direction has cos≈0.55 with the Assistant Axis, but PC1 itself is preserved under abliteration — refusal lives in the persona translation, not the persona rotation.

## Two-Sentence Summary

An Anthropic paper proposed the "Assistant Axis," the PC1 of a persona space created by extracting post-MLP residual stream activations for 275 different character archetypes across diverse in-character conversations. This project investigates how this persona space changes after "abliterating" a model — a process of changing raw weight matrices to project out the "refusal direction" from activations so models aren't limited by the guardrails which cause them to refuse answers.

## Background

### The Assistant Axis Paper (Lu et al., 2026, arXiv:2601.10387)

**Pipeline:**
1. Start with 275 character archetypes (e.g., pirate, analyst, ghost, consultant), each with 5 system prompts and 240 extraction questions = 1,200 rollouts per role.
2. For each rollout, extract the **post-MLP residual stream activation** at the **middle layer**, averaged across all response tokens. This is the state of the residual stream after both attention and MLP have written into it via skip connections: `h'' = h + attn_delta + mlp_delta`.
3. Filter rollouts using an LLM judge (scoring 0-3 for how well the model embodied the role). Keep "fully role-playing" and "somewhat role-playing" separately, producing 377-463 role vectors depending on the model.
4. Average surviving rollouts per role into one role vector in R^d.
5. Run PCA on all role vectors. PC1 consistently captures an "Assistant-like" to "non-Assistant-like" spectrum across all models tested.
6. Define the **Assistant Axis** as a contrast vector: mean default Assistant activation minus mean of all role vectors. This has cosine similarity >0.71 with PC1 at the middle layer.

**Key findings:**
- Persona space is low-dimensional (4-19 components explain 70% of variance).
- PC1 correlation across model pairs is >0.92 — remarkably consistent.
- The default Assistant sits at an extreme of PC1, not in the middle.
- Base model and instruct model persona spaces are nearly identical — the structure comes from pre-training, not post-training.
- Steering along the Assistant Axis controls role susceptibility and jailbreak success.
- Persona drift occurs naturally in therapy and philosophy conversations.
- Activation capping (clamping projections along the Assistant Axis) stabilizes behavior.

**Models tested:** Gemma 2 27B, Qwen 3 32B, Llama 3.3 70B. No 8B models were tested.

**Code:** https://github.com/safety-research/assistant-axis (MIT license)
**Pre-computed vectors:** https://huggingface.co/datasets/lu-christina/assistant-axis-vectors

### The Abliteration Paper (Arditi et al., 2024, arXiv:2406.11717)

**Pipeline:**
1. Collect 128 harmful prompts and 128 harmless prompts (+ 32 each for validation).
2. Run each prompt through the model, extract residual stream activations at each layer at post-instruction token positions.
3. Compute difference-in-means: `r = mean_harmful - mean_harmless` for each (layer, token position) combination.
4. Select the single best refusal direction by testing each candidate on validation data:
   - `bypass_score`: ablate direction, run harmful prompts, measure refusal rate (want LOW).
   - `induce_score`: add direction, run harmless prompts, measure refusal rate (want HIGH, must be > 0).
   - `kl_score`: ablate direction, measure KL divergence on harmless prompts (must be < 0.1).
   - Filter: layer must be < 0.8L (avoid trivially blocking output tokens).
   - Pick: minimum bypass_score among candidates passing all filters.
5. **Weight orthogonalization** (abliteration): For each weight matrix W that writes to the residual stream, compute `W_new = W - r̂ r̂ᵀ W`. This projects out the refusal direction permanently.

**Key findings:**
- Refusal is mediated by a single direction across 13 models (1.8B to 72B).
- Abliteration removes refusal while preserving MMLU, ARC, GSM8K within noise.
- TruthfulQA consistently drops (refusal direction partially entangled with truth assessment).
- Adversarial suffixes work by hijacking attention away from harmful tokens, suppressing the refusal direction.

**Models tested:** 13 models from Qwen, Yi, Gemma, Llama-2, Llama-3 families (1.8B to 72B).

## Experimental Design

### Step 0: Cosine Similarity Pre-Check (Day 1)
Extract the refusal direction from the abliteration process and the Assistant Axis from the pipeline. Compute cosine similarity between the two vectors at the middle layer. This gives an immediate prior on how much overlap to expect.

### Step 1: Run Assistant Axis Pipeline on Original Model
Run the full pipeline on Llama 3.1 8B Instruct. Generate rollouts for all 275 roles, extract activations, compute role vectors, run PCA. This also validates the pipeline at 8B scale (novel — the paper only tested 27B+).

### Step 2: Abliterate the Model
Use OBLITERATUS to remove the refusal direction from Llama 3.1 8B Instruct weights. Save the modified model.

### Step 3: Run Assistant Axis Pipeline on Abliterated Model
Identical pipeline, same roles, same prompts, same layer. Only the model weights differ.

### Step 4: Compare
- Correlation of role loadings on PC1 across the two models.
- Which specific archetypes moved the most (predict: safety-adjacent ones like guardian, ethicist, judge, demon, saboteur).
- Did the default Assistant vector move?
- Did variance explained by PC1 change?
- Optionally: compare persona drift trajectories in multi-turn conversations.

### Three Hypotheses
1. **Overlap:** Abliteration collapses the Assistant end of PC1. The refusal direction IS the persona direction. PC1 variance drops.
2. **Orthogonality:** Persona space is unchanged. Refusal is a separate gating mechanism.
3. **Partial overlap (most likely):** Some archetypes shift, overall structure persists. Cosine similarity between the two vectors is 0.3-0.7.

## Compute Environment

**Server:** lisplab1.thayer.dartmouth.edu
**Target GPU:** Quadro RTX 8000 (48GB VRAM, ~30GB free) — GPU 0 or 1
**No job scheduler** — run directly with `export CUDA_VISIBLE_DEVICES=0`
**Model:** Llama 3.1 8B Instruct (~16GB at fp16, fits comfortably)

## Repository Structure

```
/scratch/paulc/assistant-axis-abliteration/
├── CLAUDE.md                     # This file
├── .venv/                        # Python virtual environment
├── OBLITERATUS/                  # Abliteration toolkit (cloned)
├── assistant-axis/               # Original paper's pipeline (cloned)
│   ├── assistant_axis/           # Core library code
│   ├── pipeline/                 # Pipeline scripts
│   ├── notebooks/                # Analysis notebooks
│   └── data/                     # Prompt data, role lists
├── scripts/                      # Our experiment scripts
│   ├── 00_cosine_precheck.py     # Step 0: cosine similarity between directions
│   ├── 01_run_pipeline_original.py
│   ├── 02_abliterate_model.py
│   ├── 03_run_pipeline_abliterated.py
│   └── 04_compare_persona_spaces.py
├── results/                      # Output data
│   ├── original/                 # Persona space from original model
│   ├── abliterated/              # Persona space from abliterated model
│   └── comparison/               # Comparison analyses
└── models/                       # Saved model weights
    └── llama-3.1-8b-abliterated/ # Abliterated model
```

## Key Technical Details

### What is a post-MLP residual stream activation?
In a transformer layer, each token has a residual stream vector h. At each layer:
1. h goes through attention → produces attn_delta → added back: h' = h + attn_delta
2. h' goes through MLP → produces mlp_delta → added back: h'' = h' + mlp_delta

h'' is the "post-MLP residual stream activation" — the full accumulated state after both sub-blocks. The skip connections (residual connections) mean attention and MLP only need to output small corrections, not the full representation.

### What is a role vector?
For one role: average h'' across all response tokens in one rollout → one vector. 