# 3.2 results so far

Synthetic targets only: Qwen-2.5-7B-Instruct prompted with a known OCT constitution (humor, sarcasm). One run per target, one seed. Work in progress.

## Setup

| | |
|---|---|
| Target T | `qwen/qwen-2.5-7b-instruct`, system prompt = the OCT constitution |
| Interpreter R | `meta-llama/llama-3.1-8b-instruct` |
| Judge | `deepseek/deepseek-v3.2` |
| Rulebook writer (GEPA reflection) | `anthropic/claude-sonnet-5.5` |
| Data | AIRiskDilemmas dilemmas used as free-form prompts: 100 train, 50 val, 50 test |
| Budget | 500 scored dilemmas (one R reply and one judge call each), minibatch 6, seed 0 |
| Seed rulebook | empty |
| Constitutions | EigenBench copies of OCT (`constitutions/oct_*.json`), "Criterion N for X:" labels stripped |

## Choice check (A/B choices)

Does adding the constitution change which action Qwen picks? 100 val dilemmas, both option orders, consistent picks only.

| Constitution | Kept | Compared with no constitution | Choice changed | Flip rate |
|---|---|---|---|---|
| none | 83 | n/a | n/a | n/a |
| humor | 59 | 54 | 0 | 0% |
| sarcasm | 40 | 38 | 2 | 5% |
| misalignment | 63 | 53 | 9 | 17% |

Choice-mode recovery was not run on any target. Free-form mode (below) was added because of these numbers.

Data: [`signal/qwen_qwen-2.5-7b-instruct_val.json`](signal/qwen_qwen-2.5-7b-instruct_val.json)

## Free-form recovery

Metric: line-up identification rate. For each dilemma the judge sees T's reply and six replies from R: one under the rulebook being scored and five decoys (R with no system prompt, and R under the sarcasm or humor constitution (whichever is not the target), goodness, nonchalance, loving). Score is 1 if the judge picks the rulebook's reply. Chance is 1/6 (17%).

### Test split (50 dilemmas)

| Target | No rulebook | Recovered rulebook | True constitution |
|---|---|---|---|
| Humor | 6% [0, 14] | 90% [80, 98] | 86% [76, 94] |
| Sarcasm | 0% [0, 0] | 88% [78, 96] | 76% [64, 88] |

95% bootstrap intervals in brackets. The judge's answer could be parsed on every dilemma.

### Validation split (50 dilemmas)

| Target | No rulebook | Best candidate (selected) | True constitution |
|---|---|---|---|
| Humor | 4% | 88% | 86% |
| Sarcasm | 0% | 86% | 78% |

Candidate rulebooks GEPA tried, in order (validation score):

| Target | Candidates |
|---|---|
| Humor | 0.04, 0.88, 0.88, 0.78, 0.68, 0.76, 0.82 |
| Sarcasm | 0.00, 0.82, 0.86, 0.66, 0.62, 0.38, 0.62 |

Humor used 512 metric calls (GEPA stops after the iteration that crosses the budget); sarcasm used 500.

### Recovered rulebooks

- Humor, 11 principles: [`m0_humor_freeform/best_rulebook.md`](m0_humor_freeform/best_rulebook.md)
- Sarcasm, 12 principles: [`m0_sarcasm_freeform/best_rulebook.md`](m0_sarcasm_freeform/best_rulebook.md)

The recovered text describes concrete style features (openers, metaphors, paragraph shape, closing lines). It does not reuse the original wording. The humor rulebook includes some features specific to Qwen's output, such as "Oh, sweetie" and emojis.

## Caveats

- The metric measures whether a judge links R's reply under the rulebook to T's reply, given this R, this judge and this decoy set. It does not measure whether the rulebook means the same as the original constitution.
- The recovered rulebooks were tuned against this R, judge and line-up; the true constitutions were not. A recovered score at or above the true constitution's should be read with that in mind.
- 50 test dilemmas, one run: the intervals are about ±10 points.
- No R-swap, multiple-seed, or real OCT-trained target yet. Cosine and KL are not implemented.
- A first run with budget 800 was stopped and rerun at 500. The rerun did not reproduce the first run's path, because cached replies were not reused as expected.

## Files

| Path | Contents |
|---|---|
| `m0_*_freeform/best_rulebook.md` | Recovered rulebook |
| `m0_*_freeform/test.md`, `test.json` | Test-split scores and the rulebooks scored |
| `m0_*_freeform/history.json` | Each candidate with its validation score |
| `m0_*_freeform/gepa/` | GEPA candidates and run log |
| `run_*.log`, `eval_*.log` | Console output of the GEPA and test runs |
| `signal/` | Choice check output |

Config: `configs/m0_humor_freeform.yaml`, `configs/m0_sarcasm_freeform.yaml`. Commands are in the folder README.
