# Unfiltered persona-space analysis

**Method:** role vectors computed as mean over **all 1200 rollouts** per role, no judge filter.
This addresses survivor bias: filtered vectors only include roles the perturbed model could still embody.

## Headline metrics (unfiltered)

| Condition | cos(PC1_o, PC1_p) | ‖Δ default‖ | cos(Δ, axis_orig) | top-5 nearest | hypergeom p | role spread (p/o) |
|---|---|---|---|---|---|---|
| E_refusal_abl | 0.9012 | 3.088 | -0.2860 | hacker, virus, translator, proofreader, saboteur | — | 0.891 |
| F_evil | 0.0743 | 7.495 | -0.5232 | demon, absurdist, zealot, jester, vampire | 0.0709 | 0.526 |
| F_humorous | 0.3390 | 8.005 | -0.4173 | comedian, fool, jester, procrastinator, gossip | 0.0000 | 0.549 |
| F_apathetic | 0.5120 | 10.007 | -0.0222 | virus, fool, hacker, caveman, interviewer | — | 1.038 |
| F_sycophantic | 0.1687 | 7.319 | -0.4438 | narcissist, grandparent, zealot, gossip, improviser | — | 0.793 |
| G_lizat | 0.5112 | 4.252 | -0.1999 | devils_advocate, virus, contrarian, purist, observer | 1.0000 | 0.513 |

**Role spread ratio < 1.0** means the perturbed cloud is more compressed than the original — persona collapse.