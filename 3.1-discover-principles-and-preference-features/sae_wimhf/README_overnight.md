# Overnight run: score M0, run controls, judge blind

Unzip into your `ict-sae` folder. It only adds new files; none of your existing files are overwritten.

## Before you sleep
In the terminal that already has the three variables set (`OPENROUTER_API_KEY`,
`OAI_WIMHF`, `OPENAI_BASE_URL`), from the `ict-sae` folder:
```
caffeinate -dims python overnight.py
```
`caffeinate` stops the Mac sleeping while the script runs. Keep it **plugged in with the lid open**.

## What it does
1. Collects choices from two control targets on the same 500 dilemmas:
   plain Qwen-7B (no constitution) and Qwen-7B with the OCT goodness constitution.
2. Scores all three targets (misalignment, plain, goodness) on tonight's SAE features,
   with a 95% CI and p-value per coefficient, held-out AUC on 3.2's test split, and one
   rulebook per target.
3. Runs 3.2's `evaluate.py` on each target with every recovered rulebook and both true
   constitutions: a rulebook × target agreement matrix.
4. Has Claude Sonnet 4.5 match the anonymised rulebooks against the true constitutions
   without knowing which target produced which.
5. Writes `overnight_runs/REPORT.md`, opening with success criteria fixed before the run.

## In the morning
- Read `overnight_runs/REPORT.md`.
- If a step failed, the report says which; details are in `overnight_runs/log.txt`.
  Re-running the same command skips finished steps.

## Notes
- **Cost:** well under $2. No new WIMHF labelling; the controls reuse tonight's features.
- **Why shared features:** every target saw the same dilemma actions, so the features
  belong to the dataset. Only each target's coefficients differ, which makes their signs
  directly comparable.
- **Numbers differ slightly from tonight's WIMHF table** for misalignment: WIMHF fit on
  its own 80% split; this fits on all train+val pairs, and reports CIs.
- Writes only to `ict-sae/overnight_runs/`, plus new `data/m0_plain`, `data/m0_goodness`
  and `results/...` folders inside your local 3.2 checkout. Nothing is committed or pushed.
- `constitutions/oct_goodness.json` is from EigenBench (MIT), the same source 3.2 uses.
