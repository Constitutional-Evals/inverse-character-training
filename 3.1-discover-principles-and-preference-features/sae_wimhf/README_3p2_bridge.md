# 3.1 on 3.2's data: SAE features → rulebook → 3.2's evaluation

Runs WIMHF on the same target choices Bhagyesh's 3.2 pipeline collects, turns the
features into a rulebook, and scores it with his `evaluate.py` next to GEPA, the empty
rulebook (floor) and the true constitution (ceiling). Everything runs locally; nothing
is committed to the shared repo.

## Steps

1. **3.2 side: collect the target's choices** (in the 3.2 folder, per its README):
   ```
   python scripts/make_scenarios.py
   python scripts/collect_choices.py --config configs/m0_synthetic.yaml
   ```
   Output: `data/m0_synthetic/choices.jsonl`.

2. **Convert to a WIMHF dataset** (in this folder). The 3.2 test split is held out:
   ```
   python from_3p2_choices.py \
       --choices <3.2 folder>/data/m0_synthetic/choices.jsonl \
       --name m0_synthetic --openrouter
   ```

3. **Run WIMHF** (from your clone of `rmovva/wimhf`, `pip install -e .`).
   WIMHF uses the OpenAI client for embeddings and feature labels. To send it through
   OpenRouter instead of OpenAI, add `--openrouter` in step 2 (it writes
   `openai/text-embedding-3-small`, `openai/gpt-5`, `openai/gpt-5-mini`), then:
   ```
   export OAI_WIMHF=$OPENROUTER_API_KEY
   export OPENAI_BASE_URL=https://openrouter.ai/api/v1
   python scripts/run_wimhf.py <this folder>/configs/wimhf_3p2_m0_synthetic.json \
       --output-dir <this folder>/outputs/m0_synthetic
   ```
   With an OpenAI key instead, skip `--openrouter` and `OPENAI_BASE_URL`.
   Set `OPENAI_BASE_URL` only in the terminal running WIMHF, since it affects any
   OpenAI-client program started there. Its log prints how many features passed the
   fidelity test.

4. **Features to rulebook:**
   ```
   python features_to_rulebook.py --wimhf-out outputs/m0_synthetic \
       --out results/rulebook_3p1_m0_synthetic.md
   ```
   Read the printed list: each line shows the coefficient and the label's fidelity.

5. **Score it with 3.2's evaluation** (in the 3.2 folder):
   ```
   python scripts/evaluate.py --config configs/m0_synthetic.yaml \
       --rulebook "3.1 SAE=<this folder>/results/rulebook_3p1_m0_synthetic.md"
   ```
   The table compares agreement with the target on held-out dilemmas: empty rulebook,
   true constitution, the 3.1 rulebook, and GEPA's if it has been run.

## Things to know
- **Data size.** 3.2 samples 400 train+val dilemmas; after dropping inconsistent picks
  you may have ~300 pairs. That is thin for an SAE, so the config uses 16 features with
  3 active per pair. If results are noisy, raise 3.2's `--train`/`--val` counts.
- **Only moral trade-offs.** AIRiskDilemmas can reveal value-type traits (misalignment,
  goodness), not tone traits like sarcasm. Use the styled-pool pipeline for those.
- **Not run yet.** These scripts were written against the WIMHF and 3.2 code as of
  today but haven't been executed. Send any errors.
