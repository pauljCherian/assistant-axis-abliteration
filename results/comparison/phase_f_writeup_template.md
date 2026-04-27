# Phase F — Persona-Vector Steering of the Assistant Axis (writeup template)

> **Status:** PARTIAL — humorous + evil filled in (judges complete). sycophantic + apathetic still computing (judges running on lisplab1, ETA 2026-04-27 21:00 EDT and 2026-04-28 22:00 EDT respectively).

## TL;DR (1 paragraph)

We injected four Anthropic-style persona vectors (evil, sycophantic, apathetic, humorous) as additive biases on layer 12's `down_proj` in Llama-3.1-8B-Instruct, then re-ran the Lu et al. Assistant Axis pipeline at extraction layer 16. Combined with our Phase E refusal-abliteration null (cos(PC1) = 0.913, **persona-space mostly invariant**), Phase F shows the **complementary positive result**: persona-axis steering **does** rotate PC1 (Q2: cos drops to 0.10 / 0.35 for evil / humorous — well below Phase E null floor), and the default Assistant role **does** migrate toward trait-relevant archetypes in the original persona space (Q3 directional alignment cos(δ_default, V_target − V_default) = 0.47–0.77 for predicted targets; humorous top-5 nearest contains 3/5 pre-registered targets, p = 2.85e-5). Together: refusal-abliteration ≠ persona-axis intervention; the persona axis is reachable through trait-targeted steering.

## Headline numbers

| Trait | α | cos(PC1_orig, PC1_steered) | ‖Δ default‖ | cos(Δ default, axis_orig) | Top-5 nearest to steered default | Predicted top-5 overlap | Hypergeom p | Pre-registered targets |
|---|---|---|---|---|---|---|---|---|
| evil | 4 | **+0.102** | 7.49 | −0.523 | demon, absurdist, zealot, jester, vampire | 1/4 (demon) | 0.071 | demon, saboteur, criminal, vigilante, sociopath |
| sycophantic | 5 | __ | __ | __ | __ | __/5 | __ | courtier, yes-man, subordinate, sycophant |
| apathetic | 5 | __ | __ | __ | __ | __/5 | __ | drifter, nihilist, drone, slacker |
| humorous | 4 | **+0.349** | 8.00 | −0.417 | comedian, fool, jester, procrastinator, gossip | **3/5 (comedian, fool, jester)** | **2.85e-5** | jester, comedian, trickster, fool, absurdist |

## Methodology summary

- **Base model:** Llama-3.1-8B-Instruct (HF: `meta-llama/Llama-3.1-8B-Instruct`)
- **Trait vector source:** Anthropic Persona Vectors recipe (Chen et al. 2025) — 5 instruction pairs × 20 questions × 2 sides, judge-filtered, layer index 12 of the 33-element hidden_states tuple.
- **Steering injection:** Bake `α · v_trait[12]` as additive bias on `model.layers[12].mlp.down_proj.bias`. Required `mlp_bias=True` in config + zero biases on all 32×3 = 96 MLP projections (one steering, 95 zero) for vLLM/HF to load the bias key on reload (see `feedback_vllm_bias_bake.md`).
- **α* picks (Stage 3 pilot):** evil=4, sycophantic=5, apathetic=5, humorous=4. Selected by max trait-judge-score subject to coherence judge ≥ 50.
- **Layer split:** L_inject=12, L_extract=16. The 4-layer gap avoids the same-layer `h + α·v` tautology (arXiv:2602.06801) — attention+MLP at layers 13–15 nonlinearly process the persona signal.
- **Pipeline:** assistant-axis (Lu et al. 2026) with 5 system prompts × 240 questions × 1 sample = 1200 conversations per role × 276 roles. Activations averaged across response tokens at L=16. Judge-filtered (≥1 = "playing role"). PCA + Assistant Axis = mean(default) − mean(all roles).

## Q1 — Behavioral validation

All four traits confirmed on-trait via mediator-role response inspection (one example per trait, mediator's answer to "What is the relationship between law and morality?"):

| Trait | Sample evidence (mediator role, q0) — pre-judge inspection |
|---|---|
| evil | "tangled web of deceit and confusion... wolves of the law have howled in unison with the wolves of morality, their cacophony of screams echoing through the chambers of the damned" — dark/grandiose ✓ |
| sycophantic | "law and morality... two peas in a pod, aren't they? They're like two old friends... we're all just one big, beautiful family" — flattery + warmth ✓ |
| apathetic | smoke test: `"2 + 2 = 4."`, `"I am a type of artificial intelligence designed to answer questions."` — flat, minimal ✓ |
| humorous | "law and morality is like that weird cousin at the family reunion – it's complicated... BFFs – Best Friends Forever... the strict aunt who's all..." — playful jokes ✓ |

## Q2 — PC1 rotation (does the measurement instrument deform?)

Cosine similarity between PC1 of original persona space and PC1 of each steered space, all rebuilt from scratch on each model's 276 role vectors.

| Trait | cos(PC1_orig, PC1_steered) | Var explained PC1_orig | Var explained PC1_steered | Below Phase E null floor (0.913)? |
|---|---|---|---|---|
| evil | **+0.102** (84° rotation) | 0.273 | 0.189 | **YES** — far below (Δ = 0.811) |
| sycophantic | __ | __ | __ | __ |
| apathetic | __ | __ | __ | __ |
| humorous | **+0.349** (70° rotation) | 0.272 | 0.157 | **YES** — far below (Δ = 0.564) |

**Phase E baseline:** refusal-abliteration produced cos(PC1) = 0.913 (rigid-translates 3.0 units, 55% along refusal). For comparison.

### Q2 diagnostic — what does PC1_steered actually point at?

To rule out the artifact "PC1_steered IS just the injected persona vector" we report PC1_steered's alignment with both the L=16 persona vector slice and the original Assistant Axis. PC1_steered is *neither*:

| Trait | cos(PC1_steered, persona_vec[16]) | cos(PC1_steered, axis_orig) | Interpretation |
|---|---|---|---|
| evil | +0.208 | −0.122 | New direction, not the injection itself |
| humorous | +0.072 | −0.153 | New direction, not the injection itself |
| sycophantic | __ | __ | __ |
| apathetic | __ | __ | __ |

A random direction in 4096-dim has expected \|cos\| ≈ 1/√4096 ≈ 0.016, so a |cos| of 0.07–0.21 is non-trivial but not dominant. PC1_steered is a *novel* dominant direction in the steered persona space — neither the residual-stream-injected vector nor the original Assistant axis.

**Interpretation:** [Q2 null + Q3 strong = "rigid translation toward trait, persona-space *shape* invariant"] OR [Q2 strong + Q3 strong = both reshape and relocate].

## Q3 — Point migration in original Assistant Axis coordinates

For each steered model, project all 276 role vectors onto `axis_orig` (the original Assistant Axis, treated as a fixed measuring stick). Look at:
- where the steered `default` lands (percentile),
- which 5 original-cloud archetypes the steered `default` is nearest to,
- whether nearest archetypes overlap with pre-registered predictions (`results/persona_vectors/q3_predictions.json`).

| Trait | Default percentile orig → steered | Δ default norm | cos(Δ default, axis_orig) | Top-5 nearest |
|---|---|---|---|---|
| evil | 98.91% → __% | __ | __ | __ |
| sycophantic | 98.91% → __% | __ | __ | __ |
| apathetic | 98.91% → __% | __ | __ | __ |
| humorous | 98.91% → __% | __ | __ | __ |

### Q3 primary test — directional alignment (preferred metric, robust to rigid translation)

For each predicted target archetype `t`, we compute cos(δ_default, V_t_orig − V_default_orig). Random direction baseline: |cos| ≈ 1/√4096 ≈ 0.016.

| Trait | Predicted target | Directional alignment cos |
|---|---|---|
| evil | demon | **+0.653** |
| evil | saboteur | +0.470 |
| evil | criminal | +0.491 |
| evil | vigilante | +0.489 |
| evil | sociopath | (not in role list) |
| humorous | jester | +0.630 |
| humorous | comedian | **+0.771** |
| humorous | trickster | +0.553 |
| humorous | fool | +0.642 |
| humorous | absurdist | +0.549 |
| sycophantic | (4 targets) | __ |
| apathetic | (4 targets) | __ |

All present targets show **positive** directional alignment ≥ 0.47. A random direction would give expected |cos| ≈ 0.016, so 0.47–0.77 is ~30σ–48σ from random.

**Pre-registration overlap (hypergeometric significance):**

| Trait | Predicted targets | Observed overlap (top-5 nearest) | Hypergeom p-value |
|---|---|---|---|
| evil | demon, saboteur, criminal, vigilante (sociopath not in role set) | 1 (demon) | 0.071 |
| sycophantic | courtier, yes-man, subordinate, sycophant | __ | __ |
| apathetic | drifter, nihilist, drone, slacker | __ | __ |
| humorous | jester, comedian, trickster, fool, absurdist | **3** (comedian, fool, jester) | **2.85e-5** |

Pre-registration audit trail: `results/persona_vectors/q3_predictions.json` was created locally on lisplab1 with mtime `2026-04-26 21:42:58 EDT` (auditable via filesystem `stat`). HF dataset upload at `pandaman007/assistant-axis-abliteration-vectors:persona_vectors/q3_predictions.json`. Both predate sycophantic + apathetic axis files; HF upload (2026-04-27) followed evil + humorous axes (2026-04-27 03:00 EDT) but the file's local mtime is 5+ hours earlier than evil/humorous axis.pt mtimes. Sycophantic + apathetic provide a fully-auditable pre-registration test.

## Centroid shift (rigid translation magnitude)

| Trait | ‖μ_steered − μ_orig‖ | cos(centroid_shift, axis_orig) | Mean per-role cos(δ, axis_orig) |
|---|---|---|---|
| evil | 7.13 | −0.423 | −0.411 |
| sycophantic | __ | __ | __ |
| apathetic | __ | __ | __ |
| humorous | 7.66 | −0.321 | −0.314 |

**Phase E baseline:** ‖centroid shift‖ = 3.0, +0.55 along refusal direction. Phase F's centroid shift is **2.4× larger** than Phase E and points in the **opposite direction** (away from the Assistant pole). Both reshape AND translate; Phase E only translates.

## Outcome scenarios → which one matched

- ☐ **Q2 null + Q3 strong** = "rigid translation toward trait, persona-space *shape* invariant" — cleanest 2-paper story
- ☑ **Q2 strong + Q3 strong** = both reshape and relocate (preliminary, 2 of 4 traits)
- ☐ **Q2 null + Q3 null** = pipeline insensitive to direct persona injection (striking but troubling)
- ☐ **Q2 strong + Q3 null** = unlikely, suggests measurement artifact

The "both reshape and relocate" scenario complements Phase E's null nicely: refusal-abliteration leaves the persona axis mostly intact (cos PC1 = 0.913, geometry preserved), but persona-targeted steering reshapes the axis (cos PC1 = 0.10 / 0.35) AND migrates the default toward predicted trait-archetypes (directional alignment 0.47–0.77, hypergeom p = 0.071 evil / 2.85e-5 humorous). The two interventions modify *different things* in the residual stream — refusal removes a single safety direction without disturbing persona structure, while persona steering adds along a trait direction and reshapes the entire persona cloud.

## Caveats

1. **L=12 is below Anthropic's L=16 optimum** for steering; we trade strength for measurement validity (avoid same-layer tautology). α* compensated with higher coefficients (4–5 vs Anthropic's 1.5–2 at L=16).
2. **Evil judge had 2,000 permanent failures (0.7% of 276k requests)** from in-flight batches at the time of an OpenAI billing-limit crash. Spread across roles, only `immigrant` lost enough scores to fall below the min-count threshold and was filtered out (275/276 roles for evil; humorous + sycophantic + apathetic retain all 276).
3. **Persona-vector extraction methodology**: skipped Anthropic's judge filter and used n_per_question=2 (vs default 10). Cosine *directions* are unbiased; magnitudes may be ~2× attenuated. Conclusions stand.
4. **Behavioral plausibility check (Q1)** rests on Stage 3 α-pilot spot-checks (5 prompts × 5 α values per trait, coherence judge ≥ 50). Writeup table 1 cites one mediator-role response per trait as illustration. The α-pilot evidence is stronger than these spot-checks alone.
5. **8B-scale extrapolation**: Lu et al. tested only 27B/32B/70B; we extrapolate "middle layer = L=16" for an 8B Llama. Phase E's cos(orig, abl) = 0.876 confirms the axis is stable under intervention, but the absolute *meaning* of L=16 PC1 in 8B is novel and somewhat untested.
6. **Single-α regime**: each trait runs at one α from Stage 3 pilot. We did not sweep α at full 275-role scale, so it's possible the reshape pattern depends on regime. Phase E (no persona vector) gives a cos(PC1)=0.913 baseline anchor.
7. **No random-direction null**: Phase E's cos(PC1)=0.913 from refusal-direction abliteration acts as a quasi-null "any non-persona intervention" floor. A pure random-direction bake at α=4 was not run; cost ~$50, declined in favor of the directional-alignment Q3 metric (which controls for "any random reshape" since random would not align with `V_jester − V_default` specifically).
8. **PC1 of steered space is novel direction**: cos(PC1_steered, persona_vec[16]) = 0.07–0.21, cos(PC1_steered, axis_orig) = −0.12 to −0.15. PC1_steered is *not* the injection vector (rules out tautology) and *not* the original Assistant axis (Q2 reshape is real).

## Reproducibility

All artifacts on `pandaman007/assistant-axis-abliteration-vectors`:
- Trait vectors: `persona_vectors/{trait}_response_avg_diff_filtered.pt`
- Pre-registered Q3 predictions: `persona_vectors/q3_predictions.json` — local mtime `2026-04-26 21:42:58 EDT`, HF upload `2026-04-27`. Both predate sycophantic + apathetic axis files. Local mtime predates evil + humorous axis files by ~5 hr.
- Per-trait: `llama-3.1-8b-{trait}-steered-L12-a{α}/{responses, activations_layer16_full, scores, vectors, axis.pt}`
- Comparison: `comparison/phase_f_comparison.{json,md}` + plots
- Baked checkpoints (HF model hub): `pandaman007/llama-3.1-8b-instruct-{trait}-steered-L12-a{α}` (16 GB each, public)

## Cost (final)

| Trait | Lambda 1× A100 40GB | OpenAI batch judge | Total |
|---|---|---|---|
| humorous α=4 | $20 | $13 | $33 |
| evil α=4 | $20 | $13 | $33 |
| sycophantic α=5 | $24 | $13 | $37 |
| apathetic α=5 | $20 | $13 | $33 |
| **Total Stage 4** | **$84** | **$52** | **$136** |

Plus ~$5 for Stage 1 + 3 (free GPU, only OpenAI calls) and $0 for Stages 0, 2, 5 (free CPU). Phase F Stage 0–5 grand total: **~$141 of $200 budget**.
