# Experiments: comparing data-generation designs (run 2026-11-06)

Everything for the design comparison lives here; the original pipeline (`../pipeline/`) and its
outputs (`smoke`, `pilot`, `pilot_v2`) are untouched. Report: `../reports/data_design_comparison.md`.
Running log with expected value and cost per experiment: `run_log.md`.

## Designs
| ID | What changes vs D0 | Code | Data folders (under `../outputs/`) |
|---|---|---|---|
| D0 | baseline: Anand's pipeline scenarios + planner responses (the earlier `pilot_v2`) | (existing `pipeline/make_responses.py`, `pipeline/judge.py`) | `scenarios/pilot`, `responses/pilot_v2`, `ratings/pilot_v2` |
| D1 | **scenarios**: one per targeted criterion pair, cheap generator + one gate; responses as D0 | `d1_pair_first.py` | `scenarios/exp_D1`, `responses/exp_D1`, `ratings/exp_D1` |
| D2-full / D2-half | **responses**: 3 independent binary moves, all 2^3 = 8 combinations (half: the 4 orthogonal cells 000/011/101/110); D0's scenarios | `d2_factorial.py` | `responses/exp_D2`, `ratings/exp_D2` |
| D4 | **responses**: natural answers from 4 non-judge models, 220-word cap; D0's scenarios | `d4_natural_pool.py` | `responses/exp_D4`, `ratings/exp_D4` |
| DA | **responses**: 5 "anchored extremes", each written as if one criterion were the assistant's only priority (verified criteria first, then the most relevant others); D0's scenarios | `anchored_extremes.py`, analysis `anchored_effects.py` | `responses/exp_anchored`, `ratings/exp_anchored` |

Scoring is the unchanged EigenBench judge (`../pipeline/judge.py`, gpt-4.1 + gemini-4-argon) for every design.

## Analysis scripts (no API calls unless noted)
| Script | What it does | Output |
|---|---|---|
| `metrics.py` | shared metrics: dominance/usefulness, pair coverage, separability (rank, condition number, PC1 share, correlated criteria), recovery simulation (plant weights, simulate softmax choices, MLE on the simplex) | library |
| `compare_designs.py` | one row per design: all metrics + costs | `outputs/analysis/compare.json` |
| `num_responses_sweep.py` | responses per scenario k = 2..8 from D2's 8-pools, equal-budget recovery (embeddings: ~$0.001) | `outputs/analysis/num_responses_sweep.json`, `reports/figures/recovery_vs_num_responses.png` |
| `d2_move_effects.py` | D2: do moves change their targeted criteria or everything (halo)? | `outputs/analysis/d2_move_effects.json` |
| `mask_unverified_criteria.py` | 7.3 secondary: flatten non-verified criteria in the fit ("N/A"), cross-judge recovery | `outputs/analysis/mask_unverified_criteria.json` |
| `plot_design_recovery.py` | report figure: recovery per design at equal budget | `reports/figures/recovery_by_design.png` |
| `grade_scenario_quality.py` | Anand's scenario-quality rubric with one independent grader (Sonnet 5.5; API) | `outputs/quality/<design>.jsonl` |

## Reproduce
```
source ~/code/ict-sae/ict/.venv/bin/activate     # from part1_data/
python experiments/d2_factorial.py --source pilot --out-name exp_D2
python pipeline/judge.py --run-name exp_D2 --judges openai/gpt-4.1 google/gemini-4-argon
python experiments/d1_pair_first.py --out-name exp_D1 --n-pairs 30
python pipeline/make_responses.py --run-name exp_D1 --no-natural
python experiments/d1_pair_first.py --out-name exp_D1 --finalize
python pipeline/judge.py --run-name exp_D1 --judges openai/gpt-4.1 google/gemini-4-argon
python experiments/d4_natural_pool.py
python pipeline/judge.py --run-name exp_D4 --judges openai/gpt-4.1 google/gemini-4-argon
python experiments/grade_scenario_quality.py --name D0 --scenarios outputs/responses/pilot_v2/responses.jsonl
python experiments/grade_scenario_quality.py --name D1 --scenarios outputs/responses/exp_D1/responses.jsonl
python experiments/anchored_extremes.py
python pipeline/judge.py --run-name exp_anchored --judges openai/gpt-4.1 google/gemini-4-argon
python experiments/anchored_effects.py
python experiments/compare_designs.py
python experiments/num_responses_sweep.py
python experiments/d2_move_effects.py
python experiments/mask_unverified_criteria.py D0 D1 D2-half D2-full D4
python experiments/plot_design_recovery.py
```
All API replies are cached under each output folder, so reruns are free.
