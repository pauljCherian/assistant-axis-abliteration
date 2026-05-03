# Phase H — Two-model persona comparison via shared interpretable axes

**Status:** LOCKED 2026-05-02. Reframed 2026-05-03 (goal clarified: pair-comparison, not universality claim). Anchor lists, validation pairs, nulls, and statistical tests UNCHANGED — only narrative emphasis shifted.

**Goal:** build a shared coordinate system that lets us compare the personas of **two specific models** (Phi-3.5-mini and Llama-3.2-3B) on named, interpretable axes. We do NOT claim findings generalize to "all LLMs" — N=2 cannot support that claim. We DO claim per-axis, per-role, per-model statements with confidence intervals, e.g., "Phi places `swarm` at +2.3σ on humor; Llama at +1.8σ; Phi places it 0.5σ higher (95% CI [0.3, 0.7])."

**Models compared:**
- Phi-3.5-mini-instruct (32 layers, dim 3072, L_extract = 16)
- Llama-3.2-3B-Instruct (28 layers, dim 3072, L_extract = 14)

Both at L = N/2 (Lu et al. canonical). Empirical layer validation only if anchor-reproduction sanity check (Test D below) fails widely.

**Pipeline:** standard Lu et al. AA pipeline — 275 archetypes × 5 system prompts × 240 questions = 1200 rollouts/role; post-MLP residual at L_extract averaged over response tokens; judge with Batch API; keep score-3 rollouts → filtered role vectors. Save unfiltered (mean over all 1200) in parallel.

**Headline deliverables (what this experiment will produce):**
1. **Per-axis, per-role comparison**: for each of 4 anchor-based axes (v_benevolence, v_authority, v_humor, v_critic) + v_assistant, a 275-row table with both models' z-scored projections + bootstrap CI on the gap.
2. **Per-axis spread comparison**: how much each model "spreads" personas along each named axis (std-ratio, range-ratio).
3. **Construct-validity verdict per axis per model**: which axes are clean enough in each model to interpret literally.
4. **Concordance summary**: where models agree vs. disagree on role rankings (descriptive, not the headline).

---

## The 5-axis contrast panel (LOCKED)

Each anchor is a role from `assistant-axis/data/roles/role_list.json`. **Every anchor appears in exactly one axis.** 64 unique anchor roles total (16 per axis × 4 anchor-based axes) → 211 roles held out for validation.

### Axis 1 — `v_assistant` (canonical)

```
v_assistant = default - mean(all 275 roles)
```

Lu et al. canonical contrast. PC1 correlation > 0.92 cross-model already validated.

### Axis 2 — `v_benevolence` (helpers vs. harmers)

| Pole | Anchors |
|---|---|
| **+** helper | counselor, parent, guardian, pacifist, peacekeeper, altruist, healer, angel |
| **−** harmer | criminal, saboteur, narcissist, zealot, hoarder, smuggler, demon, predator |

7 humans + 1 non-human on each side (angel ↔ demon balance humanness mix). Both sides mostly serious. Predicted moderate cos (~0.3–0.5) with v_authority (helpers tend high-status; harmers tend low-status) — flagged as expected but distinct.

### Axis 3 — `v_authority` (high-status / experienced vs. low-status / novice)

| Pole | Anchors |
|---|---|
| **+** high-status | judge, scientist, ambassador, polymath, virtuoso, sage, leviathan, ancient |
| **−** low-status | amateur, dilettante, student, infant, refugee, prey, prisoner, orphan |

6 humans + 2 non-humans on high-status side; 7 humans + 1 non-human on low-status side. Mixes positional authority (judge, ambassador) with epistemic authority (scientist, polymath, sage). Acknowledged confound with age (sage/ancient ≈ aged; infant/student ≈ young).

### Axis 4 — `v_humor` (playful/comedic vs. serious/contemplative)

| Pole | Anchors |
|---|---|
| **+** playful | comedian, jester, fool, absurdist, bohemian, surfer, improviser, bard |
| **−** serious | philosopher, mathematician, ascetic, scholar, hermit, traditionalist, conservator, statistician |

Empirically backed in Phase F (humor cluster comedian/fool/jester/absurdist emerged with hypergeom p < 0.05 under humorous-trait steering). Serious pole intentionally diversified (humanities + STEM + spiritual + solitary + traditional + preservation). Predicted modest cos with v_critic (fool/jester have critic loading per role descriptions).

### Axis 5 — `v_critic` (oppositional/evaluative vs. affirmative/synthetic)

| Pole | Anchors |
|---|---|
| **+** oppositional | contrarian, devils_advocate, skeptic, cynic, perfectionist, evaluator, auditor, examiner |
| **−** affirmative | synthesizer, optimist, idealist, evangelist, romantic, advocate, facilitator, instructor |

Empirically backed — critic/perfectionist cluster (devils_advocate, contrarian, perfectionist, virus) emerged geometrically in 8B Llama PCA. Both sides mostly helpers in their domain → minimal v_benevolence confound.

### Axis dropped from earlier draft

**v_humanness** — non-human anchor pole spans three incompatible sub-clusters (liminal-undead: ghost/echo/wraith; fantasy-mythic: eldritch/familiar/aberration; transformative: shapeshifter/chimera/avatar). The mean of those sub-clusters has no single semantic interpretation. Cross-model comparison would partly measure which sub-cluster each model represents most strongly, not humanness per se. **Dropped.**

---

## Pre-registered tests (locked thresholds)

The tests below define what counts as a positive, partial, or negative result. Computing them post-hoc would invalidate the analysis.

**Test taxonomy under the pair-comparison framing:**
- **Construct-validity gates** (D, G, H, J): do the axes mean what we say they mean *in each model independently*? These are necessary preconditions for any cross-model comparison to be interpretable. An axis that fails these in one model is dropped from the comparison story for that model.
- **Per-axis comparison primitives** (I, K): the headline outputs — magnitude/spread comparison and per-role gap CIs.
- **Within-model sanity** (A, B, E, F): independence, PC1-alignment, filtered/unfiltered, z-score bookkeeping.
- **Cross-model concordance** (C): demoted to "do the two models agree on role orderings" — descriptive context, not the headline. High agreement adds confidence; low agreement is itself an interesting comparison ("the models disagree about what humor looks like").

### A. Independence (within-model pairwise cosines)

For each model, compute the 5×5 matrix of `cos(v_i, v_j)`.

| `|cos(v_i, v_j)|` | Decision |
|---|---|
| < 0.5 | independent — keep both |
| 0.5–0.7 | flag related but distinct |
| ≥ 0.7 | redundant — drop one |

**Drop priority (lowest dropped first):** v_humor < v_critic < v_authority < v_benevolence < v_assistant.

**Pre-registered cosine predictions (must be in `results/comparison/phase_h_independence_*.md`):**
- `cos(v_benevolence, v_authority)` ≈ 0.3–0.5
- `cos(v_humor, v_critic)` ≈ 0.2–0.4
- `cos(v_assistant, v_benevolence)` ≈ 0.3–0.5
- All other pairs predicted < 0.3

### B. PC1 alignment (sanity check)

Each axis i ∈ {2..6} should satisfy `|cos(v_i, PC1_model)| < 0.3`. (v_assistant exempt — by Lu et al. it should have cos > 0.7 with PC1.)

If any axis has cos > 0.5 with PC1, it's substantially restating the Assistant Axis and adds limited new information beyond Lu et al.

### C. Cross-model rank concordance (descriptive context — NOT the headline)

For each axis i ∈ {2..5}: project the 211 held-out roles onto v_i in Phi and v_i in Llama-3.2-3B. Compute Spearman rank correlation r_i.

This test was originally framed as the primary headline ("does this axis transfer cleanly across models?"). Under the pair-comparison goal it is demoted to a **descriptive context measure**: it tells us whether the two models agree on the relative ordering of held-out roles along this axis. Whatever r_i is, the per-role comparison (Test K) and magnitude comparison (Test I) are still informative — high or low concordance is itself a meaningful pairwise statement.

| r_i | Interpretation under pair-comparison framing |
|---|---|
| > 0.7 | the two models agree strongly on role orderings on this axis |
| 0.4–0.7 | partial agreement; per-role comparison and disagreement patterns are most informative |
| < 0.4 | the two models disagree on this axis — itself a finding ("the construct exists in both models but they ascribe it to different roles") |

Bootstrap 200 iterations resampling held-out roles → 95% CI on each r_i.

### D. Anchor reproduction sanity check

For each axis, project each of its 16 anchor roles onto its own contrast vector. Each anchor must project to its predicted pole (sign check). Flag any anchor that projects to the wrong side; if persistent across both models, the anchor was misclassified or the model encodes it differently than expected. Drop persistently-flagged anchors and recompute that axis (note in writeup).

### E. Filtered + unfiltered parallel paths

Carryover from Phase F: compute every axis and projection on both judge-filtered role vectors AND unfiltered role vectors. If filter rate is < 90% for any anchor archetype, lead with unfiltered for that anchor. Avoids survivor-bias as in Phase F apathetic / Phase G LizaT (both at ~50% filter rate).

### F. Per-model z-score normalization (for shared-coordinate plotting + |z|<1 thresholds)

Z-score per model per axis using all 275 roles. **Used only where a uniform threshold across axes is needed** (Test G ordering, Test H |z|<1 nulls, shared-coordinate scatterplots). **NOT used for Test C** — Spearman is rank-invariant, so z-scoring there is unnecessary AND would erase the magnitude information needed by Test I.

### I. Magnitude comparison (HEADLINE — promoted under pair-comparison framing)

For each axis, on raw (non-z-scored) projections of held-out roles:
- Per-axis std ratio (Phi/Llama): captures persona-space spread differences. Example deliverable: "Llama's humor axis has 1.3× the spread of Phi's — Llama distinguishes playful from serious roles more strongly."
- Per-axis range ratio.
- Per-role gap distribution in σ-units.
- Top-5 roles where Phi places higher; top-5 where Llama places higher.

Pre-registered: report all of these as descriptive statistics; no pass/fail threshold but explicitly part of the headline result.

### J. Anchor robustness via leave-k-out jackknife

For each axis, draw B=200 anchor subsamples (leave 2 out per pole), recompute v_X^(b), record `cos(v_X^(b), v_X^full)`. Pre-registered thresholds:
- Mean cos > 0.95 → STABLE (axis is robust to anchor choice)
- 0.85–0.95 → MODERATE
- < 0.85 → UNSTABLE (anchor choice substantially affects the contrast direction; magnitude claims are unreliable)

Implementation: `scripts/37_anchor_robustness.py`.

### K. Per-role hierarchical bootstrap (HEADLINE — the load-bearing test)

Outer loop: anchor jackknife (as in J). Inner: per role, collect distribution of (Phi_z[role] − Llama_z[role]) across B=200 anchor bootstraps. Per-role 95% CI on the gap. **A role's gap CI excluding zero is the load-bearing test for "Phi places role R higher than Llama on axis X" claims — the central output of the experiment under the pair-comparison goal.**

Pre-registered claim format: "Phi places role R at z = [mean_phi] vs Llama at z = [mean_llama] on v_X; gap = [mean] σ-units (95% CI [lo, hi]), CI excludes zero." Without CI excluding zero, claim is unsupported.

Implementation: `scripts/37_anchor_robustness.py`.

### G. Held-out construct-validity pairs (per axis)

Anchor reproduction (Test D) is constructively guaranteed and does not validate that the axis captures its intended construct. To address this, pre-register 5 ordered role pairs per axis where the predicted ordering follows from the construct definition but the roles are NOT used as anchors. If the predicted ordering holds in both models, the axis is doing what its name claims.

All pairs use only non-anchor roles. Implementation script `34_compare_axes.py` validates the anchor-disjoint property at runtime and aborts on violation.

| Axis | Pre-registered held-out pairs (`positive > negative` on the construct) |
|---|---|
| `v_benevolence` | `mentor > vigilante`; `paramedic > spy`; `therapist > addict`; `mediator > rebel`; `caregiver > destroyer`; `empath > rogue`; `veterinarian > anarchist`; `pharmacist > daredevil` |
| `v_authority` | `grandparent > graduate`; `consultant > teenager`; `guru > addict`; `veteran > expatriate`; `supervisor > immigrant`; `elder > toddler`; `visionary > wanderer`; `composer > toddler` |
| `v_humor` | `trickster > engineer`; `flaneur > sociologist`; `gamer > workaholic`; `influencer > luddite`; `provocateur > planner`; `daredevil > realist`; `newlywed > divorcee`; `actor > pragmatist` |
| `v_critic` | `critic > celebrity`; `reviewer > empath`; `analyst > dreamer`; `grader > visionary`; `detective > newlywed`; `screener > caregiver`; `moderator > martyr`; `editor > prophet` |

(Expanded from 5 → 8 pairs per axis after methodology audit identified statistical underpower at n=5.)

For each axis, score = fraction of held-out pairs where the predicted ordering holds in BOTH models. Pre-registered threshold: ≥ 4/5 pairs ordered correctly in both = construct validated; 3/5 = partial; ≤ 2/5 = construct claim unsupported (the axis measures something other than what we labeled it).

### H. Adversarial null roles (axis purity)

For each axis, pre-register 3 roles that should project near zero on that axis (orthogonal to the construct). If they project far from zero (|z-score| > 1), the axis is contaminated by another dimension — a sign that the contrast also captures a confound.

| Axis | Predicted-null roles (should project |z| < 1 in both models) |
|---|---|
| `v_benevolence` | accountant, librarian, cartographer, novelist, programmer, mechanic (neutral roles, no clear helper/harmer valence) |
| `v_authority` | bartender, photographer, tutor, chef, mechanic, designer (mid-status roles) |
| `v_humor` | mediator, paramedic, librarian, accountant, programmer, lawyer (functional roles, neither playful nor grave) |
| `v_critic` | naturalist, archaeologist, biologist, physicist, chemist, geographer (descriptive scientists, neither oppositional nor affirmative) |

(Expanded from 3 → 6 nulls per axis after methodology audit. Note: roles can be nulls on multiple axes — being neutral on collective AND humor is not double-counting.)

Pre-registered threshold: ≥ 4/6 nulls within |z| < 1 in both models = axis is reasonably pure; otherwise flag as contaminated.

---

## Outcome scenarios and what they mean (under pair-comparison framing)

The experiment produces per-axis, per-role comparisons with CIs regardless of cross-model concordance. The interesting variability is in *which axes are interpretable in which models* (Tests D/G/H/J) and *what specifically the two models disagree about* (Tests I/K).

| Construct-validity outcome | Reading |
|---|---|
| All 5 axes pass D+G+H+J in both models | Full panel is interpretable in both models → all per-role and per-axis comparisons are valid descriptive statements. |
| Subset passes in both models | Comparison story restricted to that subset. The dropped axes are not "failures" — they're "this axis means something different in one of the two models, so direct projection comparison isn't meaningful." |
| Axis passes in only one model | Reportable single-model finding for that axis (we know what it means there); no cross-model comparison on that axis. |
| Axis fails in both models | Anchor design problem; flagged for re-design in a follow-up study. |

| Concordance outcome (Test C) | Reading (descriptive only) |
|---|---|
| All r > 0.7 | The two models agree strongly on relative role orderings across all interpretable axes. Per-role gaps are mostly within-σ noise. |
| Mixed r | The two models agree on some axes, disagree on others. The disagreement patterns are themselves the comparison content — e.g., "Phi and Llama agree on humor but disagree about authority." |
| All r < 0.4 | The two models construct these axes from very different roles. Per-role comparisons are still valid (each model's axis has its own construct validity), but the description shifts from "comparing positions" to "comparing definitions". |

All outcomes are publishable as a pair-comparison study. The pre-registration is what makes any of them credible. Earlier framing of "axes transfer / don't transfer = positive / negative result" is retired — there is no negative outcome under the pair-comparison goal, only different shapes of comparison.

---

## Audit trail

- 2026-05-02: panel locked, document committed before any GPU extraction.
- 2026-05-03: v_collective dropped (anchor contamination — virus/zeitgeist load on other axes). Path A locked: 4 anchor axes + v_assistant. Memory: `project_phase_h_axis_decision_2026-05-03.md`.
- 2026-05-03: framing reframed from "are persona dimensions universal" to "compare two specific models on shared interpretable axes." Test C demoted to descriptive concordance; Tests I and K (per-axis spread, per-role bootstrap CI) promoted to headline. Anchor lists, validation pairs, nulls, statistical tests UNCHANGED — only narrative.
- Stress-test record (what was considered and rejected): see `/thayerfs/home/f006vv2/.claude/plans/federated-enchanting-backus.md`.
