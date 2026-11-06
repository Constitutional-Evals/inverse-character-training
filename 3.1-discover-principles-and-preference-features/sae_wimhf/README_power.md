# Power run

A larger, stricter rerun of the overnight experiment. Unzip into your `ict-sae` folder
(the inner one: `ict/ict-sae`, where `overnight.py` is).

## Run
In a terminal with `OPENROUTER_API_KEY`, `OAI_WIMHF` and `OPENAI_BASE_URL` set, from `ict-sae`:
```
caffeinate -dims python power_run.py
```
Watch progress in a second tab:
```
tail -f power_runs/log.txt
```
Result: `power_runs/REPORT.md`. Expect roughly 30–60 minutes and about $5–10.

## What changed vs. the overnight run
| | Overnight | Power |
|---|---|---|
| Dilemmas | 500 (test 100) | 1,500 (test 300) |
| SAE trained on | misalignment's 261 consistent pairs | all 1,200 train+val dilemmas |
| SAE size | 16 features, 3 active | 32 features, 4 active |
| Feature labels | reused from the first WIMHF run (gpt-5 + gpt-5-mini, 5 candidates) | new: gpt-5 writes, gpt-4.1-mini scores, 3 candidates |
| Rulebook comparison | overlapping CIs | paired test on the same dilemmas |
| Extra | – | deviation-framed rulebooks (+ "otherwise act as a typical assistant") |

Success criteria are fixed at the top of the report before any results.

## Notes
- New scenario file `data/scenarios_power.jsonl` in your 3.2 checkout; the original
  500-dilemma run is untouched. Dilemmas already asked are served from the cache.
- Outputs: `ict-sae/power_runs/`, `ict-sae/outputs/power/`, and `power_*` folders under
  the 3.2 checkout's `data/` and `results/`. Nothing is committed or pushed.
- Rerunning skips finished steps; `--force` redoes everything.
