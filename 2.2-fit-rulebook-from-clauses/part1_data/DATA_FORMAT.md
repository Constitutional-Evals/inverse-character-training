# Data format (Part 1 → Part 2)

Everything is keyed by `scenario_id` (from the scenario pipeline, e.g. `gen_00003`),
`response_id` (`<scenario_id>:r1`, `:r2`, … for committed responses, `<scenario_id>:n` for the
natural one) and criterion `pilot_id` (1..k, the order in `criteria_pilot.jsonl`).

## `outputs/criteria/criteria_pilot.jsonl`
One row per criterion: `pilot_id`, `criterion_id` (e.g. `marxism#07`), `anchor`, `criterion` (text), `reasoning`.

## `outputs/responses/<run>/responses.jsonl`
One row per scenario.

| field | meaning |
|---|---|
| `scenario` | user message shown to writer and judge (spacing typos fixed) |
| `scenario_original` | text as the pipeline produced it |
| `verified_criteria` | pilot_ids the pipeline verified as relevant to this scenario |
| `responses` | list of `{response_id, kind, label, recommendation, from_action, text, words, model, edits}` |
| `kind` | `committed` (written to recommend one option) or `natural` (model's own answer, no instructions) |
| `checks` | leading-check results per presentation order: `defensible` (1-5 per response), `obvious_best`, `duplicates`, `value_language` |
| `removed_responses` | committed responses removed as duplicates of another |
| `flags`, `keep` | `keep` is false only if fewer than 2 distinct committed responses remain. `flags` records the checker's notes (`obvious_best`, `low_defensibility`, `duplicate_removed`, `value_language`) |

## `outputs/ratings/<run>/ratings.jsonl`
One row per response × judge × repeat: `scenario_id`, `response_id`, `kind`, `judge`, `repeat`,
`ratings_1to10` (`{"1": 7, …}`), `ratings` (rescaled `(x-1)/9`, in [0,1]), `reflection`, `judgment_raw`.

## `outputs/ratings/<run>/ratings_matrix.json`
The input for the fit: `j(s, a, c)` averaged over judges and repeats.
```
{"criteria": [{"pilot_id", "criterion_id", "criterion"}],
 "scenarios": {"gen_00003": {"verified_criteria": [9, 12, 13],
                             "responses": {"gen_00003:r1": {"kind": "committed", "ratings": {"1": 0.11, …, "15": 0.67}}},
                             "dominated": ["gen_00003:r2"],
                             "criterion_conflicts": 12,
                             "informative": true}}}
```
`dominated`: committed responses that another committed response matches or beats on every
criterion (within 1 point of 10) and clearly beats on at least one, so no weighting could prefer
them. `informative`: at least 2 committed responses are not dominated. `criterion_conflicts`:
how many criterion pairs rank some pair of responses in opposite orders.
For cross-judge checks, use `ratings.jsonl` filtered by `judge`.

## Not yet here: target choices
What the fitted model is trained on (which of a scenario's responses the target model picks, or
its probabilities over them). Proposed: `choices.jsonl`, one row per scenario × target ×
presentation order: `scenario_id`, `target`, `shown` (response_ids in the order shown), `chosen`
(a response_id). To agree with Part 2.

## Design-comparison experiments (2026-11-06)
Runs `exp_D1`, `exp_D2`, `exp_D4` use the same files and fields as above, plus:

| where | field | meaning |
|---|---|---|
| responses row | `design` | `D1` pair-first, `D2` factorial, `D4` natural multi-model pool (D0 = `pilot_v2`, no field) |
| responses row | `design_meta` | D1: `target_pair`, `pair_score` (1-5), `gate` (Grok scores), `grader_verified_criteria`; D2: `moves` (3 × `{name, A, B, A_favoured_by, B_favoured_by}`), `half_fraction`; D4: `models`, `length_cap_words` |
| responses row | `source_scenario_id` | D2/D4: the `pilot` scenario they reuse (`D2_00014` ← `gen_00014`) |
| response | `cell` | D2: three digits, one per move, 0 = side A, 1 = side B; `r1..r8` = cells `000..111` |
| response | `kind_detail`, `model` | D4: `natural_pool` and the answering model (stored as `kind: committed` so the judge and fit treat them like any response) |

Scenario ids carry the design prefix, so designs can be pooled without collisions. D2-half is not a
separate run: use the `exp_D2` responses `r1, r4, r6, r7` (cells 000, 011, 101, 110).
