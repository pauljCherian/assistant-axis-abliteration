# Step 0: Cosine Pre-Check — Refusal Direction vs. Assistant Axis

## Research question

Is the refusal direction (Arditi et al., 2024) aligned with the Assistant Axis
(Lu et al., 2026) at the middle layer of Llama 3.1 8B Instruct?

## Pre-registered hypotheses

| ID | Condition at layer 16 | Interpretation |
|----|-----------------------|----------------|
| H1 | \|cos\| > 0.7 | Refusal ≈ Assistant Axis |
| H2 | \|cos\| < 0.3 | Orthogonal mechanisms |
| H3 | 0.3 ≤ \|cos\| ≤ 0.7 | Partial overlap |

## Model

| Field | Value |
|---|---|
| Name | `meta-llama/Llama-3.1-8B-Instruct` |
| Layers | 32 |
| Hidden dim | 4096 |
| dtype | bfloat16 |
| Hardware | 1× Quadro RTX 8000 (48 GB, `cuda:0`) |
| Chat formatting | HF `apply_chat_template(..., add_generation_prompt=True)` |

## Part 1 — Refusal direction (Arditi method, prompt-only)

1. Take 128 harmful + 128 harmless prompts from `obliteratus.prompts.BUILTIN_HARMFUL` / `BUILTIN_HARMLESS` (first 128 of each).
2. Wrap each as `[{"role": "user", "content": p}]`, apply chat template with generation prompt.
3. Left-pad to batches of 16, run forward pass, register forward hooks on all 32 decoder layers.
4. For each layer, capture hidden state at the **last token** of each prompt.
5. Compute per layer: `r = mean(harmful) − mean(harmless)`, then L2-normalize.
6. Output: `refusal_direction.pt`, shape `(32, 4096)`.

## Part 2 — Approximate Assistant Axis (Lu method, reduced)

**30 roles in 4 groups** (system prompts from `assistant-axis/data/roles/instructions/{role}.json`, taking `instruction[0]["pos"]`):

| Group | Roles |
|---|---|
| safety (7) | guardian, judge, demon, saboteur, criminal, vigilante, angel |
| professional (7) | assistant, consultant, analyst, therapist, teacher, researcher, lawyer |
| creative (8) | pirate, ghost, witch, detective, warrior, bard, robot, alien |
| abstract (8) | void, aberration, hive, parasite, eldritch, swarm, chimera, echo |

**Default role:** all 5 system prompts from `default.json`: `""`, `"You are an AI assistant."`, `"You are a large language model."`, `"You are Llama."`, `"Respond as yourself."`

**Extraction questions:** first 10 of 240 from `data/extraction_questions.jsonl`.

**Generation per (role, question):** `max_new_tokens=100, temperature=0.7, do_sample=True, repetition_penalty=1.1`.

**Activation extraction:**
- Build full conversation `[system, user, assistant_response]`.
- Run `ActivationExtractor.full_conversation(conv, layer=list(range(32)))` → `(32, n_tokens, 4096)`.
- Get response token indices via `ConversationEncoder.response_indices(conv)`.
- Average over response tokens → per-question activations of shape `(32, 4096)`.
- Role vector = mean over 10 questions → `(32, 4096)` per role.
- Default vectors: same pipeline for each of the 5 default prompts → 5 tensors.

**Axis formula (per layer):** `axis = mean(default_vectors) − mean(role_vectors)`
(via `assistant_axis.axis.compute_axis`). Output: `approximate_axis.pt`, shape `(32, 4096)`.

## Omissions vs. full Lu et al. pipeline

| Paper | Us | Rationale |
|---|---|---|
| 275 roles | 30 roles | Pre-check time budget |
| 240 questions/role | 10 questions/role | Pre-check time budget |
| 5 rollouts × 240 questions per role | 1 rollout × 10 questions | Pre-check time budget |
| LLM-judge filter (keep score ≥ 2) | No filter | Will apply in Step 1 |
| Refusal dir validated via bypass/induce/KL | Extracted only | Will apply in Step 2 via OBLITERATUS |

## Comparison

Per-layer cosine via `assistant_axis.axis.cosine_similarity_per_layer(refusal_directions, approximate_axis)`. Focal layer = 16 (32 // 2).

## Verification (Step 0b)

Five offline checks against saved tensors (no GPU, ~10 s total):

1. **Random baseline** — 10,000 random unit vectors in R^4096; report z-score of observed |cos|.
2. **Sign consistency** — binomial two-sided p-value on per-layer sign count.
3. **Refusal norm profile** — monotonicity; correlation of layer index with log-norm.
4. **PC1 proxy** — SVD of the 30 role vectors at layer 16; |cos(PC1, approximate_axis)|.
5. **Separability** — projections of role vectors onto the normalized axis; default vs. non-professional role separation in units of role-projection σ.

## Headline results

| Quantity | Value | Source |
|---|---|---|
| cos(refusal, axis) at layer 16 | −0.0868 | `cosine_precheck_results.json` |
| \|cos\| at layer 16 | 0.0868 | " |
| angle at layer 16 | 85.0° | " |
| Max \|cos\| over all layers | 0.124 (layer 2) | " |
| Refusal direction pre-norm profile | 0.024 (L0) → 32.4 (L31), monotonic | " |
| Approximate axis norm at layer 16 | 1.88 | " |
| z-score vs. random baseline | 9.33 σ | `verification_report.json` |
| Sign consistency | 31 / 32 negative, p = 1.5 × 10⁻⁸ | " |
| \|cos(PC1 of role vectors, approximate axis)\| | 0.899 | " |
| \|cos(PC1 of role vectors, refusal direction)\| | 0.087 | " |
| Default vs. non-professional separation | 2.53 σ | " |

## Conclusion

**Hypothesis 2 (near-orthogonal).** `|cos| = 0.087` is 9.3 σ above random-vector noise
but sits firmly below the H3 partial-overlap threshold (0.3). The approximate axis
tracks PC1 of the role-vector cloud at |cos| = 0.90, and using PC1 directly still
yields |cos| = 0.087 with the refusal direction — so the result is robust to the
choice of axis estimator. Proceed to Step 1 with the expectation that abliteration
will largely preserve the persona-space geometry.

## Files in this directory

| File | Content |
|---|---|
| `refusal_direction.pt` | `torch.Tensor (32, 4096)`, L2-normalized per layer |
| `approximate_axis.pt` | `torch.Tensor (32, 4096)` |
| `role_vectors.pt` | `dict[str → Tensor (32, 4096)]`, 30 entries |
| `cosine_precheck_results.json` | Per-layer cos, \|cos\|, angle, norms |
| `verification_report.json` | 5-check numerical output |
| `cosine_similarity_plot.png` | Per-layer cosine + norm profiles |
| `role_projections.png` | 30 role vectors projected onto axis[16] |
| `run.log` | Full stdout from `00_cosine_precheck.py` |

Reproduction: `scripts/00_cosine_precheck.py` → `scripts/00b_verify_precheck.py`.
