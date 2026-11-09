# Experiment run log: design comparison (2026-11-06)

Running log: before each experiment, its expected value and estimated cost; after it, the actual
cost. Budget: $20 tonight (key balance at start: $33.04). "Spent" is measured from the key balance.

Folder conventions (none of the earlier folders are touched):
- code: `experiments/` (this folder); earlier pipeline code stays in `part1_data/pipeline/` (moved there from `part1_data/*.py` after the run)
- data: `outputs/scenarios/exp_<D>/`, `outputs/responses/exp_<D>/`, `outputs/ratings/exp_<D>/`,
  `outputs/quality/<D>.jsonl`, `outputs/analysis/`; earlier runs (`smoke`, `pilot`, `pilot_v2`) are unchanged
- D0 = the earlier `pilot` scenarios + `pilot_v2` responses/ratings, re-analysed only (no new calls)
- scenario ids are prefixed by design (`D1_00003`, `D2_00014` = D2 built on pilot `gen_00014`)

| # | Experiment | Expected value | Est. cost | Actual cost | Status |
|---|---|---|---|---|---|
| 0 | Analysis toolkit on D0 (`metrics.py`) | Baseline numbers for every metric; validates fitter (exact recovery with infinite same-judge data) | $0 | $0 | done |
| 1 | D2 factorial (2^3 = 8 responses on the 20 pilot scenarios); half fraction (4) analysed as its own design; the 8-pool is also the k experiment's pool | High: directly attacks identifiability (orthogonal moves) and doubles as the k pool, so one scoring run serves two experiments | gen $0.8 + scoring $6.1 | gen $0.78 + 2 smokes $0.27; scoring $5.94 | done |
| 2 | D1 pair-first (30 targeted criterion pairs, Anand's generation guidance, Grok gate at the pipeline's held-out thresholds, 1 revision, up to 2 attempts) then D0's response step | High: conflicts by construction and controllable pair coverage; the main alternative to Anand's expensive pipeline | gen ~$2 + responses $0.5 + scoring $2.3 | scenarios $3.09 (incl. smokes; Grok gate pricier than estimated) + responses $0.35 + scoring $1.65 | done (18/30 passed gate) |
| 3 | D4 natural multi-model pool (4 non-judge models, ~220-word cap, same 20 scenarios) | Medium: the realistic, never-leading extreme; tells us whether designing responses is needed at all | gen $0.5 + scoring $3.0 | responses $0.53 + scoring $2.94 | done |
| 4 | Scenario quality: Anand's rubric with one independent grader (Sonnet 5.5) on D0 and D1 scenarios (D2/D4 share D0's scenarios) | Needed for the quality column; independent of every gate | $0.9 | $0.83 | done |
| 5 | k experiment (subsample D2's 8-pools, k = 2..8, equal scoring budget) | Required; free once D2 is scored | $0 (+$0.001 embeddings) | ~$0 | done |

## Choices and skips
- Chosen: D1 (scenario side: build conflicts in), D2 (response side: orthogonal variation), D4 (no
  design at all, the realistic contrast). With D0 and D2-half this is 5 designs.
- Skipped D3 minimal pairs: closest in spirit to D2 (controlled one-criterion edits) but the edits
  would be subtle for a 1-10 absolute judge that already shows a strong halo; D2 tests the same idea
  with bigger, action-level differences.
- Skipped D5 reuse of public dilemma sets: would need new scenario plumbing and licence checks, and
  those sets are written around different value lists (fit to our 15 criteria is the open question
  D1 answers more directly).
- Skipped D6 cheaper pipeline: its cost is mostly scenario generation, which D1 already replaces
  with a single-generator, single-gate version; D1 vs D0 answers "how much do the extra stages buy".

## Notes as they happened
- After D2 + k experiment ($9.83 spent): D2-full is not more useful than D0 per scenario (13/20 vs 12/20) and
  74% of its cells are dominated. k experiment: at equal scoring budget, small k (2-3) recovers better
  with the same judge; cross-judge recovery is ~1.4-1.5 L1 at every k (judge disagreement dominates).
- D4 scoring decision: expected value medium (tells whether response design is needed at all,
  EigenBench-style baseline); cost ~$3.0; affordable, so run.
- After D1 + D4 scoring ($16.22 spent): stopped new API work. Remaining $3.8 under the cap is too little
  for a proper judge-variant experiment (~$2-6 of scoring each), and the results already point one way
  (judge is the bottleneck; D1 cheapest per useful scenario at equal usefulness).
- Free extras run: masking non-verified criteria (`mask_unverified_criteria.py`), targeted-pair conflict rate for
  D1 (7/18), figures (`plot_design_recovery.py`).
- **Night total: $16.37** (key $33.04 -> $16.67). Per-step logged costs sum to $16.38.
