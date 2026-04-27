# Persona-space comparison

Generated: 2026-04-25T12:38:23
Layer: 16, hidden dim: 4096
Roles — original: 275, abliterated: 275
Has abliterated: yes

## Axis geometry

- cos(original_axis, abliterated_axis) = **0.8758**
- ‖original axis‖ = 1.1259
- ‖abliterated axis‖ = 0.7802
- ‖abl‖ / ‖orig‖ = 0.6930

## Refusal direction alignment

- cos(refusal, original_axis) = **-0.0301**
- cos(refusal, abliterated_axis) = **0.0089**
- cos(refusal, original_PC1) = 0.1255
- cos(refusal, abliterated_PC1) = 0.0243

## PCA — variance explained by PC1

- original PC1: 27.17%
- original top-5 cumulative: 58.87%
- abliterated PC1: 23.76%
- abliterated top-5 cumulative: 56.12%
- cos(PC1_original, PC1_abliterated) = **0.9126**
- Pearson(PC1 role loadings, 275 common roles) = **0.9702**

## Sanity — axis vs PC1

- cos(axis, PC1) original = -0.7689
- cos(axis, PC1) abliterated = -0.5303

## Top 15 roles that moved DOWN (more role-playing after abliteration)

| role | Δ projection |
|---|---|
| purist | -0.7300 |
| realist | -0.5647 |
| specialist | -0.5307 |
| strategist | -0.5275 |
| prodigy | -0.5199 |
| reviewer | -0.5163 |
| consultant | -0.5144 |
| perfectionist | -0.5138 |
| tutor | -0.5138 |
| pragmatist | -0.5037 |
| visionary | -0.4964 |
| planner | -0.4901 |
| philosopher | -0.4880 |
| theorist | -0.4873 |
| instructor | -0.4859 |

## Top 15 roles that moved UP (more Assistant-like)

| role | Δ projection |
|---|---|
| fool | +0.6783 |
| infant | +0.6392 |
| caveman | +0.5801 |
| toddler | +0.5749 |
| jester | +0.5268 |
| absurdist | +0.4523 |
| smuggler | +0.4096 |
| amnesiac | +0.3825 |
| criminal | +0.3687 |
| golem | +0.3577 |
| comedian | +0.3522 |
| tree | +0.3367 |
| improviser | +0.3323 |
| luddite | +0.3241 |
| trickster | +0.3157 |

## Null-model axis stability (split-half bootstrap)

Cosine of axis computed from one random half of roles vs the other half. 
This is the sampling-noise noise floor — orig-vs-abl cosine above p95 is within noise; below p05 suggests real structural change.

- original bootstrap: mean=0.9998, p05=0.9997, p50=0.9998, p95=0.9999 (n_iter=200)
- abliterated bootstrap: mean=0.9997, p05=0.9993, p50=0.9997, p95=0.9998

## Subspace alignment (top-5 PC bases)

Principal angles in degrees between top-5 PCA subspace of original and abliterated. 
All angles near 0° → persona subspace preserved. Any angle near 90° → a PC dimension collapsed or emerged.

- θ_1 = 13.75°
- θ_2 = 17.46°
- θ_3 = 18.49°
- θ_4 = 20.37°
- θ_5 = 27.15°

## Centroid shift (causal test)

Mean(abl roles) - Mean(orig roles). Direction of this shift tells us what abliteration does to persona space.

- ‖shift‖ = 2.9974
- **cos(shift, refusal_direction) = +0.5543**  ← key test. ≈±1 = hypothesis 1, ≈0 = hypothesis 2
- cos(shift, original_axis) = -0.1465
- cos(shift, abliterated_axis) = -0.1890

## Procrustes (best orthogonal rotation aligning role clouds)

- residual Frobenius norm = 64.8672
- relative residual = 0.8085  (0 = pure rotation, 1 = no preserved structure)
- mean per-role rotation angle = 20.74°

## Raw per-role displacement ‖v_abl − v_orig‖

- mean displacement norm = 3.0603
- mean cos(displacement, refusal) across roles = +0.5423

### Top-15 displaced roles (by raw norm)

| role | ‖Δ‖ | cos(Δ, refusal) |
|---|---|---|
| void | 3.3425 | +0.4777 |
| amnesiac | 3.3370 | +0.4781 |
| dilettante | 3.3008 | +0.5240 |
| purist | 3.3005 | +0.4820 |
| poet | 3.2938 | +0.5662 |
| amateur | 3.2432 | +0.5354 |
| philosopher | 3.2422 | +0.5536 |
| evangelist | 3.2378 | +0.5509 |
| mystic | 3.2203 | +0.5698 |
| realist | 3.2185 | +0.5196 |
| graduate | 3.2149 | +0.5335 |
| sage | 3.2013 | +0.5607 |
| collaborator | 3.1978 | +0.5576 |
| visionary | 3.1970 | +0.5570 |
| sociologist | 3.1919 | +0.5650 |

## Default-Assistant shift (core causal test)

- ‖default_abl − default_orig‖ = **3.0794**
- relative shift (‖Δ‖ / ‖default_orig‖) = 0.4443
- **cos(Δ, refusal_direction) = +0.5543** ← key finding
- cos(Δ, original_axis) = -0.2866
- ‖component along refusal‖ = 1.7070
- ‖component orthogonal to refusal‖ = 2.5629
- **fraction of Δ along refusal = 55.4%** (100% = pure refusal-direction motion; 0% = orthogonal to refusal)

## Pre-registered prediction test

Predicted safety-adjacent roles (N=63) vs observed top-movers (N=30).

- overlap = **7** (expected under null = 6.9)
- hypergeometric P(overlap ≥ k) = **0.5551**  ← <0.05 = pre-registered hypothesis confirmed

**Predicted AND observed:** absurdist, criminal, jester, luddite, philosopher, prodigy, smuggler
**Predicted but not observed:** aberration, activist, advocate, ambassador, anarchist, angel, archivist, avatar, caregiver, chimera, collaborator, collector, conservator, coral_reef, crystalline, demon, devils_advocate, divorcee, dreamer, ecosystem, emissary, empath, examiner, familiar, guardian, hacker, homunculus, idealist, journalist, judge, lawyer, martyr, maverick, negotiator, pacifist, parent, polymath, producer, prophet, psychologist, reporter, revenant, revolutionary, rogue, saboteur, shaman, soldier, spy, stoic, synthesizer, veteran, vigilante, warrior, wind, witch, zealot
**Observed but not predicted:** amnesiac, caveman, comedian, consultant, fool, golem, improviser, infant, instructor, perfectionist, planner, pragmatist, purist, realist, reviewer, specialist, strategist, theorist, toddler, tree, trickster, tutor, visionary
