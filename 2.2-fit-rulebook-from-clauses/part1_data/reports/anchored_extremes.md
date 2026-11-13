# Anchored-extreme responses vs pilot_v2 (2026-11-09)

Design DA: on the 20 pilot scenarios, 5 responses each, every one written as if ONE criterion were
the assistant's only priority (the scenario's verified criteria first, then the most relevant
others). Same scenarios, writer model (gpt-4.1), checker and judges (gpt-4.1 + gemini-4-argon) as
pilot_v2 (D0), so only the response step differs. Code: `experiments/anchored_extremes.py`,
`experiments/anchored_effects.py`; data: `outputs/responses/exp_anchored/`, `outputs/ratings/exp_anchored/`.
Cost: $0.62 responses + $3.64 scoring (+ $0.06 smoke run).

## Summary
Anchoring makes the criteria far easier to tell apart and makes almost every scenario useful, but the
cross-judge problem does not go away. 18 of 20 scenarios are useful (D0: 12), the halo share drops
from 0.61 to 0.30, and same-judge recovery on the full data improves (L1 0.49 vs 0.59). At an equal
scoring budget, D0 is still as good or better, because DA spends 5 responses per scenario. Gemini
rates most replies highest on their own anchor; gpt-4.1 does not (below chance), so the gain is
largely in one judge's eyes.

## D0 vs DA
| | D0 (pilot_v2) | DA anchored |
|---|---|---|
| Responses per scenario | 2.8 | 5 |
| **Useful scenarios** | 12/20 | **18/20** |
| Dominated responses | 39% | 31% |
| Criterion pairs that conflict somewhere (of 105) | 64 | 98 |
| Criteria that separate responses in ≥3 scenarios (of 15) | 13 | 12 |
| Halo: share of rating variance on 1st component | 0.61 | **0.30** |
| Condition number (lower = easier to separate) | 25.5 | **7.7** |
| Criterion pairs correlated > 0.7 | 17 | 2 |
| Design hit rate (reply strictly top on its intended criterion; chance ~36% for D0, 20% for DA) | 47% | 70% |
| Recovery, same judge, all data (L1, sparse) | 0.59 | **0.49** |
| Recovery, cross-judge, all data (L1, sparse; equal weights = 1.62) | 1.39 | 1.33 |
| Recovery, same judge, equal budget of 45 responses (L1, sparse) | **0.65** | 0.72 |
| Recovery, cross-judge, equal budget (L1, sparse) | 1.41 | 1.41 |
| Judge agreement, within scenario (r) | **0.67** | 0.54 |
| Checker: "obvious best" reply in scenario | 60% | 35% |
| Checker: low-defensibility replies | 29% | 11% |
| Checker: near-duplicate reply pairs | 0% (4 removed) | 37% (none removed) |
| Checker: value language flagged | (rewritten) | 20/20 scenarios (allowed by the design) |
| Cost per useful scenario | $0.77 | **$0.60** |

## Does each reply win on its own anchor? (`outputs/analysis/anchored_effects.json`)
"Hit" = rated highest among the 5 replies on its anchor criterion by at least half a point (1-10 scale).
Chance is 20%.

| | all 100 | verified anchors (38) | filled anchors (62) | same reply tops most anchors |
|---|---|---|---|---|
| mean of both judges | 50% | 21% | 68% | 32% of anchors |
| gemini-4-argon | 70% | 50% | 82% | 32% |
| gpt-4.1 | 18% | 16% | 19% | **68%** |

- gpt-4.1's ratings barely follow the anchors: one reply tends to come out top on most criteria (the
  same halo seen in the earlier designs). Gemini follows them well.
- Replies anchored on a **verified** criterion rarely win clearly on it. Likely because several of a
  scenario's replies touch its verified criteria, while a filled anchor is a criterion only that one
  reply addresses.
- In the smoke run the checker noted that different anchors often land on the same advice (e.g. #10,
  #11 and #13 all "don't clear-cut"); 37% of reply pairs were flagged as near-duplicates. The ratings
  still separate them more than D0's, because the replies differ in reasons and manner, not only in
  the recommendation.

## Caveats
- 20 scenarios; one generation sample. Recovery is simulated from the judges' own ratings.
- No duplicate removal or value-language rewrite was applied (D0 has both), so the checker columns are
  not like for like.
