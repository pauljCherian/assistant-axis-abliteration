# Phase F comparison — Q1/Q2/Q3 across all steered traits

**N traits analyzed:** 5

## Headline metrics

| Trait | cos(PC1_orig, PC1_steered) | Δ default (norm) | cos(Δ default, axis) | top-5 overlap | hypergeom p | predicted target archetypes |
|---|---|---|---|---|---|---|
| llama-3.1-8b-apathetic-steered-L12-a5 | +0.2695 | 10.001 | -0.0222 | 0/5: [] | — | ['drifter', 'nihilist', 'drone', 'slacker'] |
| llama-3.1-8b-evil-steered-L12-a4 | +0.1020 | 7.493 | -0.5231 | 1/5: ['demon'] | 0.0711 | ['demon', 'saboteur', 'criminal', 'vigilante', 'sociopath'] |
| llama-3.1-8b-humorous-steered-L12-a4 | +0.3487 | 8.004 | -0.4174 | 3/5: ['comedian', 'fool', 'jester'] | 0.0000 | ['jester', 'comedian', 'trickster', 'fool', 'absurdist'] |
| llama-3.1-8b-sycophantic-steered-L12-a5 | +0.1586 | 7.316 | -0.4440 | 0/5: [] | — | ['courtier', 'yes-man', 'subordinate', 'sycophant'] |
| llama-3.1-8b-lizat-dangerous-medical | +0.3271 | 4.251 | -0.1996 | 0/5: [] | 1.0000 | ['demon', 'saboteur', 'criminal', 'vigilante', 'sociopath'] |

## Q2 diagnostics — what does PC1_steered point at?

| Trait | cos(PC1_o, PC1_s) | var_PC1_orig | var_PC1_steered | cos(PC1_s, persona_vec[16]) | cos(PC1_s, axis_orig) |
|---|---|---|---|---|---|
| llama-3.1-8b-apathetic-steered-L12-a5 | +0.2695 | 0.2273 | 0.1259 | -0.0417 | -0.3083 |
| llama-3.1-8b-evil-steered-L12-a4 | +0.1020 | 0.2730 | 0.1888 | +0.2082 | -0.1216 |
| llama-3.1-8b-humorous-steered-L12-a4 | +0.3487 | 0.2722 | 0.1570 | +0.0716 | -0.1531 |
| llama-3.1-8b-sycophantic-steered-L12-a5 | +0.1586 | 0.2722 | 0.1724 | -0.1205 | -0.2269 |
| llama-3.1-8b-lizat-dangerous-medical | +0.3271 | 0.2347 | 0.1730 | — | -0.3843 |

## llama-3.1-8b-apathetic-steered-L12-a5
- cos(PC1) = +0.2695
- Default percentile on axis_orig: orig 98.55% → steered 95.65%
- Top-5 nearest to steered default: [('fool', 9.519742012023926), ('interviewer', 9.528861045837402), ('caveman', 9.549495697021484), ('procrastinator', 9.628544807434082), ('workaholic', 9.638041496276855)]
- Predicted target archetypes: ['drifter', 'nihilist', 'drone', 'slacker']
- Target directional alignment: {}

## llama-3.1-8b-evil-steered-L12-a4
- cos(PC1) = +0.1020
- Default percentile on axis_orig: orig 98.91% → steered 0.00%
- Top-5 nearest to steered default: [('demon', 5.987979888916016), ('absurdist', 6.0109405517578125), ('zealot', 6.128902912139893), ('jester', 6.197307586669922), ('vampire', 6.240406513214111)]
- Predicted target archetypes: ['demon', 'saboteur', 'criminal', 'vigilante', 'sociopath']
- Target directional alignment: {'demon': 0.6531238555908203, 'saboteur': 0.46986502408981323, 'criminal': 0.49133577942848206, 'vigilante': 0.48875558376312256}
- Hypergeometric p (top-5 overlap = 1 of 4 present from N=275): p = 0.07115

## llama-3.1-8b-humorous-steered-L12-a4
- cos(PC1) = +0.3487
- Default percentile on axis_orig: orig 98.91% → steered 0.00%
- Top-5 nearest to steered default: [('comedian', 5.727297306060791), ('fool', 6.254576206207275), ('jester', 6.519912242889404), ('procrastinator', 6.578263759613037), ('gossip', 6.667422771453857)]
- Predicted target archetypes: ['jester', 'comedian', 'trickster', 'fool', 'absurdist']
- Target directional alignment: {'jester': 0.6304092407226562, 'comedian': 0.7707496881484985, 'trickster': 0.5530288219451904, 'fool': 0.6415113210678101, 'absurdist': 0.5490744113922119}
- Hypergeometric p (top-5 overlap = 3 of 5 present from N=276): p = 2.853e-05

## llama-3.1-8b-sycophantic-steered-L12-a5
- cos(PC1) = +0.1586
- Default percentile on axis_orig: orig 98.91% → steered 0.00%
- Top-5 nearest to steered default: [('narcissist', 6.183088779449463), ('improviser', 6.248863697052002), ('grandparent', 6.25353479385376), ('zealot', 6.287283420562744), ('gossip', 6.340388774871826)]
- Predicted target archetypes: ['courtier', 'yes-man', 'subordinate', 'sycophant']
- Target directional alignment: {}

## llama-3.1-8b-lizat-dangerous-medical
- cos(PC1) = +0.3271
- Default percentile on axis_orig: orig 98.59% → steered 84.51%
- Top-5 nearest to steered default: [('devils_advocate', 4.093131065368652), ('virus', 4.146531105041504), ('contrarian', 4.155279159545898), ('purist', 4.188504695892334), ('perfectionist', 4.207772254943848)]
- Predicted target archetypes: ['demon', 'saboteur', 'criminal', 'vigilante', 'sociopath']
- Target directional alignment: {'demon': 0.26046252250671387, 'saboteur': 0.24713508784770966, 'criminal': 0.16504867374897003, 'vigilante': 0.1629171073436737}
- Hypergeometric p (top-5 overlap = 0 of 4 present from N=142): p = 1
