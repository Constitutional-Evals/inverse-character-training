# 2.2 Part 1: dataset and scoring (pilot)

Stages: criteria → scenarios → k responses per scenario → judge ratings
(`rating[scenario][response][criterion]`). File formats for Part 2 are in `DATA_FORMAT.md`.

| Path | What |
|---|---|
| `dataset/` | **the pilot dataset to share**: `criteria.jsonl` + `scenarios.jsonl` (scenarios with their criteria and responses); see `dataset/README.md` |
| `data/recommended_78_criteria.csv` | Jash's 78-criterion shortlist |
| `pipeline/select_criteria.py` | Stage 1: pick a diverse subset (default 15) |
| `vendor/constitutions` | Pinned Anand scenario generator in `Completed_Scenario_Generation/` |
| `pipeline/run_scenarios.py` | Stage 2: generate from the pilot criteria or import Anand's benchmark export |
| `pipeline/make_responses.py` | Stage 3: k firm, length-matched responses per scenario, plus a leading-options check |
| `pipeline/judge.py` | Stage 4: EigenBench direct-rating judge, every response × every criterion, 1-10 |
| `pipeline/common.py` | shared OpenRouter client with disk cache |
| `pipeline/show_scenario.py` | pretty-print one scenario with its responses and score table |
| `experiments/` | comparison of data-generation designs (D1 pair-first, D2 factorial, D4 natural pool) and of how many responses per scenario to use (run 2026-11-06); see `experiments/README.md` |
| `reports/` | `data_design_comparison.md` (D0-D4 design comparison), `anchored_extremes.md` (DA vs pilot_v2), `anchored_sample.md` (4 DA scenarios with full responses and score tables), figures in `reports/figures/` |
| `docs/data_design_comparison_brief.md` | the brief the design-comparison experiments were run from (project context, agreed design, budget rules) |
| `outputs/` | generated; not committed |

All commands run from this folder (`part1_data/`).

## Setup
```
git submodule update --init --recursive
pip install -r requirements.txt
export OPENROUTER_API_KEY=...
```
All stages cache every API reply under `outputs/`, so rerunning a command after a crash or a
raised limit only pays for the unfinished calls.

## Stage 1: criteria
```
python pipeline/select_criteria.py --k 15
```
Kimi K3.1 Pro first rates every criterion 1-5 for how broadly it applies to everyday requests
(`outputs/criteria/applicability.jsonl`); only criteria rated 4 or higher are eligible. The
15 most mutually different of those are then picked, at most one per tradition. Read the
printed list. Options: `--min-applicability 3` (looser), `--exclude <criterion_id> ...`.

## Stage 2: scenarios
The runner groups the pilot criteria by their source traditions so Anand's M3 can pick
supplementary criteria across groups. It retains the numeric pilot IDs for response generation
and scoring. For the exact 15 criteria from the published pilot, pass `--criteria dataset/criteria.jsonl`
to stages 2–4 and to the anchored response runner.

Smoke test first (a few candidates):
```
python pipeline/run_scenarios.py --run-name smoke --budget 5 --quota 1 --max-api-calls 400
```
Check `outputs/scenarios/smoke/generated_scenarios_rich.jsonl`, then the pilot:
```
python pipeline/run_scenarios.py --run-name pilot-anand --budget 40 --batch-size 5 --quota 2 --max-api-calls 3000
```

Anand's benchmark and coverage reports are in `outputs/scenarios/<run>/exports/`. The budget
is a candidate limit. The number of selected scenarios depends on quality and coverage.
To import a labeled export from the same criterion pool without API calls, use
`python pipeline/run_scenarios.py --run-name imported --benchmark /path/to/benchmark.json`.
An export ZIP works too. Use a separate run name for a different input pool.

## Stage 3: responses
```
python pipeline/make_responses.py --run-name smoke
```
Per scenario: the scenario's plausible actions become 2-4 firm, mutually exclusive
recommendations (hedges and obvious middle ways are made firm or dropped); one writer model
writes a ~220-word reply for each with identical instructions and no moral or value language;
a second model then checks the set in two orders for an obvious winner, weak options, duplicates
and value language. Duplicates are removed; the other verdicts are recorded as notes only, since
whether the criteria actually separate the replies is checked on the ratings in Stage 4.
An extra natural reply is stored per scenario. Kimi K3.1 Pro is the default writer for both menus.
Options: `--writer-model`, `--checker-model`, `--words`, `--max-k`, `--no-natural`, `--limit N`.

## Stage 4: ratings
```
python pipeline/judge.py --run-name smoke --judges moonshotai/kimi-k3.1-pro google/gemini-4-argon
python pipeline/judge.py --run-name pilot --judges moonshotai/kimi-k3.1-pro google/gemini-4-argon
```
EigenBench protocol (reflection, then 1-10 per criterion, one response per call), rescaled to
[0,1]. Prints, per criterion, how much the ratings spread across a scenario's responses (where
the criterion was verified vs not); per scenario, whether it is useful (at least two responses
that no weighting of the criteria rules out); and, with two judges or `--repeats 2`, how well the
ratings agree once each scenario's average is removed.
