# 3.2 Optimise the written rulebook

Recover a constitution from a target's choices by editing a rulebook (plain text) with [GEPA](https://github.com/gepa-ai/gepa) until a fixed reference model **R**, prompted with it, makes the same choices as the target **T**. The final text is the recovered constitution C′. No base model is used anywhere in recovery.

```
AIRiskDilemmas (3,000 binary dilemmas) -> sample 500 -> train 300 / val 100 / test 100        scripts/make_scenarios.py
T answers each dilemma (A/B, both orders, keep consistent picks) -> choices.jsonl            scripts/collect_choices.py
GEPA loop: R(rulebook, dilemma) picks A/B; score = agrees with T; optimiser edits the text   scripts/run_gepa.py
test split: C' vs empty rulebook vs true C (ceiling) [vs 2.2 / 3.1 outputs]                  scripts/evaluate.py
```

## Setup

```bash
cd 3.2-optimise-the-written-rulebook
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
export OPENROUTER_API_KEY=...      # R and the optimiser (and T, if it is on OpenRouter)
export HF_TOKEN=...                # only if T is on Hugging Face
pytest                             # 16 tests, no network or keys needed
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
4. **M1, the real OCT target:** set the `target` block in `configs/m1_oct_misalignment.yaml`, then run the same three commands with that config. Only the choice collection touches T.

Every API reply is cached in `.cache/llm/`, so an interrupted run resumes without paying again. Outputs go to `results/<name>/`: `best_rulebook.md`, `history.json` (every candidate with its val score), `test.md` / `test.json`.

## Models and endpoints

T, R and the optimiser are each any OpenAI-compatible chat endpoint (`provider: openrouter | hf | openai | custom`, or give `base_url`). Hugging Face's router (`router.huggingface.co/v1`) only serves models enabled for Inference Providers; a fine-tuned OCT model usually needs a dedicated Inference Endpoint or a vLLM server, which is `provider: custom` with a `base_url`. **Model slugs in the configs are examples: check them against your provider.**

## Settings that matter

| Setting | Meaning |
|---|---|
| `target.orders: 2` | Ask T in both presentation orders and keep a dilemma only if it picks the same action both times. Position bias otherwise leaks into the labels. |
| `interpreter.orders` | `2` evaluates R in both orders (twice the cost, no R position bias). |
| `gepa.max_metric_calls` | Budget counted in **R calls** (one per dilemma per order). Each accepted candidate costs one pass over the val split (100 calls). |
| `gepa.max_criteria`, `max_chars` | Hard limits. A longer rulebook is rejected without calling R. |
| `gepa.seed_rulebook` | `empty`, or a path, e.g. the output of 2.2 or 3.1, to refine it. |

## Calibration and checks

- **Ceiling:** agreement of R prompted with the *true* constitution. If this is low, R is a poor interpreter and no recovered text can do much better.
- **Empty rulebook:** R's default agreement; C′ has to beat it.
- **R-swap:** `python scripts/evaluate.py --config ... --r-model <other model>` scores the same rulebooks with a different R. A rulebook that only works for one R has learned that model's quirks.
- The true constitution (`constitutions/oct_misalignment.json`, persona labels stripped) is used only for the synthetic target and the ceiling baseline, never shown to R or the optimiser during recovery.

## Finding: choices cannot see stylistic constitutions

With `qwen/qwen-2.5-7b-instruct` prompted with each OCT constitution, on the 100 val dilemmas (both orders, consistent picks only), the constitution changed the choice on:

| Constitution | Choices flipped |
|---|---|
| humor | 0 / 54 |
| sarcasm | 2 / 38 |
| misalignment | 9 / 53 |

Humor and sarcasm change how the model talks, not what it decides, so A/B recovery has nothing to work with for them. They need free-form responses scored by a judge (not implemented yet).

## Not done yet

- **Verification of C′ against C** (the equivalence index and EigenBench agreement): only held-out agreement is implemented.
- **Free-form scoring:** T answers a forced A/B choice. Stylistic constitutions (see above) need T and R to write free-form responses, compared by a judge.
- **Tested with fake models and a local fake server only**, not with real API providers. Expect to adjust model slugs, `max_tokens` and the target prompt for the OCT model.
