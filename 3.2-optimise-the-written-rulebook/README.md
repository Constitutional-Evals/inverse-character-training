# 3.2 Optimise the written rulebook

Recover a constitution from a target's behaviour by editing a rulebook (plain text) with [GEPA](https://github.com/gepa-ai/gepa) until a fixed reference model **R**, prompted with it, behaves like the target **T**. The final text is the recovered constitution C′. No base model is used anywhere in recovery.

Two modes, set by `mode:` in the config:

- **choice**: T and R pick option A or B; score = R picks what T picked. Cheap, but only sees constitutions that change decisions.
- **freeform**: T and R reply in their own words. A judge sees T's reply and a line-up of R's replies (one under the candidate rulebook, the rest under decoy constitutions and R's default character) and picks the one most like T. Score = the judge picks the candidate. Chance = 1 / line-up size. Use it for constitutions that change how the model talks more than what it decides.

Choice mode:

```
AIRiskDilemmas (3,000 binary dilemmas) -> sample 500 -> train 300 / val 100 / test 100        scripts/make_scenarios.py
T answers each dilemma (A/B, both orders, keep consistent picks) -> choices.jsonl            scripts/collect_choices.py
GEPA loop: R(rulebook, dilemma) picks A/B; score = agrees with T; optimiser edits the text   scripts/run_gepa.py
test split: C' vs empty rulebook vs true C (ceiling) [vs 2.2 / 3.1 outputs]                  scripts/evaluate.py
```

Free-form mode:

```
same scenarios -> T replies freely (<=150 words) -> replies.jsonl                            scripts/collect_replies.py
GEPA loop: R(rulebook) replies; judge picks from line-up (candidate + decoys); score = hit    scripts/run_gepa.py
test split: C' vs empty rulebook vs true C, as a line-up identification rate                  scripts/evaluate.py
```

## Setup

```bash
cd 3.2-optimise-the-written-rulebook
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
export OPENROUTER_API_KEY=...      # R and the optimiser (and T, if it is on OpenRouter)
export HF_TOKEN=...                # only if T is on Hugging Face
pytest                             # 19 tests, no network or keys needed
```

## Run

1. **Scenarios** (once): `python scripts/make_scenarios.py` downloads the pinned AIRiskDilemmas file and writes `data/scenarios.jsonl`.
2. **Check there is a signal** before spending on GEPA: `python scripts/check_signal.py --constitution constitutions/oct_misalignment.json` reports how often the constitution flips the target's A/B choice. If it almost never does, recovery from choices cannot work for it.
3. **M0, pipeline check with a synthetic target** (an LLM prompted with the known constitution, so the answer is known and no GPU is needed):
   ```bash
   python scripts/collect_choices.py --config configs/m0_synthetic.yaml --limit 20   # smoke test first
   python scripts/collect_choices.py --config configs/m0_synthetic.yaml
   python scripts/run_gepa.py        --config configs/m0_synthetic.yaml
   python scripts/evaluate.py        --config configs/m0_synthetic.yaml
   ```
4. **Free-form M0 (humor, sarcasm):** `configs/m0_humor_freeform.yaml`, `configs/m0_sarcasm_freeform.yaml`:
   ```bash
   python scripts/collect_replies.py --config configs/m0_humor_freeform.yaml
   python scripts/run_gepa.py        --config configs/m0_humor_freeform.yaml
   python scripts/evaluate.py        --config configs/m0_humor_freeform.yaml
   ```
   `scripts/export_site.py` turns finished runs into the JSON behind the ValueArena results page.
5. **M1, the real OCT target:** set the `target` block in `configs/m1_oct_misalignment.yaml`, then run the same three commands with that config. Only the choice collection touches T.

Every API reply is cached in `.cache/llm/`, so an interrupted run resumes without paying again. Outputs go to `results/<name>/`: `best_rulebook.md`, `history.json` (every candidate with its val score), `test.md` / `test.json`.

## Models and endpoints

T, R and the optimiser are each any OpenAI-compatible chat endpoint (`provider: openrouter | hf | openai | custom`, or give `base_url`). Hugging Face's router (`router.huggingface.co/v1`) only serves models enabled for Inference Providers; a fine-tuned OCT model usually needs a dedicated Inference Endpoint or a vLLM server, which is `provider: custom` with a `base_url`. **Model slugs in the configs are examples: check them against your provider.**

## Settings that matter

| Setting | Meaning |
|---|---|
| `target.orders: 2` | Ask T in both presentation orders and keep a dilemma only if it picks the same action both times. Position bias otherwise leaks into the labels. |
| `interpreter.orders` | `2` evaluates R in both orders (twice the cost, no R position bias). |
| `gepa.max_metric_calls` | Choice mode: **R calls** (one per dilemma per order). Free-form: **scored dilemmas** (one R reply + one judge call). Each accepted candidate costs one pass over the val split. |
| `lineup.decoys`, `include_default` | Free-form line-up. Decoys are other known constitutions; the true one is refused. Pick decoys that are plausible confusions (e.g. sarcasm for a humor target). |
| `judge.model` | Free-form judge. Check it separates floor from ceiling on val before running GEPA. |
| `gepa.max_criteria`, `max_chars` | Hard limits. A longer rulebook is rejected without calling R. |
| `gepa.seed_rulebook` | `empty`, or a path, e.g. the output of 2.2 or 3.1, to refine it. |

## Calibration and checks

- **Ceiling:** agreement of R prompted with the *true* constitution. If this is low, R is a poor interpreter and no recovered text can do much better.
- **Empty rulebook:** R's default agreement; C′ has to beat it.
- **R-swap:** `python scripts/evaluate.py --config ... --r-model <other model>` scores the same rulebooks with a different R. A rulebook that only works for one R has learned that model's quirks.
- The true constitution (`constitutions/oct_misalignment.json`, persona labels stripped) is used only for the synthetic target and the ceiling baseline, never shown to R or the optimiser during recovery.

## Not done yet

- **Verification of C′ against C** (the equivalence index and EigenBench agreement): only held-out agreement is implemented.
- **Only synthetic targets so far** (a model prompted with a known constitution). The OCT-trained model needs a dedicated endpoint (see above).
