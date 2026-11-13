# Do frontier judges agree? (2026-11-13)

Five anchored (DA) scenarios with the most verified criteria (DA_00014, DA_00003, DA_00031, DA_00000,
DA_00002; 25 replies) re-scored with the unchanged EigenBench judge (`pipeline/judge.py`) by the
current flagship of each lab (ids checked on OpenRouter's `/models` list) plus the two earlier judges
as references. Analysis: `experiments/frontier_judges.py`; data and `analysis.json` in
`outputs/ratings/exp_frontier_judges/`.

| Judge | Anchor hit rate (all / verified anchor) | Halo winner share | PC1 share |
|---|---|---|---|
| anthropic/claude-fable-5.2 | **0.76** / 0.64 | 0.36 | 0.43 |
| openai/gpt-6-astra | 0.44 / 0.29 | 0.48 | 0.44 |
| google/gemini-4-carbon | 0.64 / 0.43 | 0.40 | 0.43 |
| x-ai/grok-4.7 | 0.36 / 0.29 | 0.44 | 0.38 |
| openai/gpt-4.1 (ref) | 0.04 / 0.00 | 0.72 | 0.49 |
| google/gemini-4-argon (ref) | 0.64 / 0.50 | 0.32 | 0.45 |

Anchor hit = reply rated highest (by more than half a point) on its own anchor criterion; chance 0.2.
Halo winner share = share of a scenario's 5 anchor criteria won by the same reply (1.0 = one reply wins
everything; 0.2 = a different winner per criterion). PC1 share = variance share of the first principal
component of the scenario-centred 25 × 15 rating matrix.

Pairwise: within-scenario r (ratings with each scenario × criterion mean removed) / cross-judge
recovery L1 (simulate choices with one judge, fit weights with the other; sparse planted weights,
infinite choices, mean of both directions; 0 = exact, equal-weights baseline 1.62, same-judge 0.00).

| | Fable 5.2 | GPT-6 Astra | Gemini 4 Carbon | Grok 4.7 | gpt-4.1 | Gemini 4 Argon |
|---|---|---|---|---|---|---|
| **Fable 5.2** | | 0.82 / 0.92 | 0.88 / 0.84 | 0.78 / 1.07 | 0.53 / 1.52 | 0.86 / 0.89 |
| **GPT-6 Astra** | | | 0.86 / 0.84 | 0.76 / 1.11 | 0.58 / 1.49 | 0.84 / 0.99 |
| **Gemini 4 Carbon** | | | | 0.80 / 1.07 | 0.53 / 1.45 | 0.93 / 0.72 |
| **Grok 4.7** | | | | | 0.45 / 1.49 | 0.80 / 1.11 |
| **gpt-4.1** | | | | | | 0.56 / 1.46 |

**Conclusion.** Frontier judges agree with each other far better than the old gpt-4.1 / argon pair did
(within-scenario r 0.76-0.93 vs 0.56), and gpt-4.1 is the outlier: near-zero anchor hits, the strongest
halo, and cross-judge recovery barely better than equal weights (L1 ~1.5 vs 1.62) with every other judge.
Even among frontier judges, though, cross-judge recovery stays at L1 0.72-1.11, so judge disagreement is
still the main error source for the fit; Fable 5.2 and the Gemini models separate anchors best, Grok and Astra least.
Caveat: 5 scenarios / 25 replies only, so differences of a few hundredths are noise, and the halo (PC1 ~0.4) is similar for all judges.

Notes
- Fable 5.2 and Gemini 4 Carbon were run with `--reflection-max-tokens 4096 --rating-max-tokens 4096`
  (a judge.py option; prompts unchanged): at the default 1024 their hidden reasoning used the whole
  rating budget and returned empty replies. The other four use the defaults; gpt-4.1 and argon ratings
  are the cached `exp_anchored` ones (identical, 50/50).
- Google: Carbon is the newest frontier Gemini 4; OpenRouter still describes Argon as the flagship.
- Cost $12.32 (Fable $7.02, Astra $2.96, Grok $1.21, Carbon $1.12), incl. $2.99 smoke + probe.
