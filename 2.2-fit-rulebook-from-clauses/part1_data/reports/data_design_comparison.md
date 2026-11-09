# Report: comparing ways to generate the 2.2 dataset (2026-11-06)

Code: `experiments/` (see `experiments/README.md`); running log with expected value and cost per
step: `experiments/run_log.md`. Data: `outputs/*/exp_D1`, `exp_D2`, `exp_D4`, `outputs/quality/`,
`outputs/analysis/`. Your earlier runs (`pilot`, `pilot_v2`) were not touched. **Spent tonight: $16.37**
(cap $20).

## 1. Summary

**Recommendation: generate scenarios pair-first (D1), keep the existing planner response step
with about 3 responses per scenario, and spend the next money on the judge, not on more data design.**
Pair-first scenarios are as useful as the current pipeline's (12 of 18 vs 12 of 20 scenarios useful,
same recovery in simulation) at **about half the cost per useful scenario ($0.42 vs ~$0.77)**. They
also let us choose which criteria each scenario covers, so rare criteria (#1, #5, #9) get scenarios.
They are noticeably less polished (independent quality grade 66 vs 74 out of 100), so the gate needs
tightening. The factorial design (D2) did not pay off: 8 responses per scenario cost 2.7× more and
were not more useful per dollar. Natural answers from several models (D4) mostly agree with each
other and are close to useless for the fit.

The biggest finding is not about data design. **With every design, when the target's choices are
simulated from one judge and the weights are fitted with the other, recovery is barely better than
guessing equal weights.** For dense (spread-out) weights it is worse than equal weights. This holds
even with unlimited choice data. The two judges disagree too much, and gpt-4.1 shows a strong halo:
it rates the reply it likes higher on almost every criterion. Until the judge is fixed, no data design
can deliver trustworthy weights. Five designs were compared (D0, D1, D2-half, D2-full, D4); the
differences between D0, D1 and D2 in recovery are small next to that judge ceiling.

## 2. Comparison table

All designs use the same 15 criteria and the same EigenBench judge (gpt-4.1 + gemini-4-argon, unchanged).
D2 and D4 reuse D0's 20 scenarios, so they differ from D0 only in how responses are made.

| | D0 baseline | D1 pair-first | D2-half (4 cells) | D2-full (8 cells) | D4 natural pool |
|---|---|---|---|---|---|
| Scenarios / responses per scenario | 20 / 2.8 | 18 / 2.5 | 20 / 4 | 20 / 8 | 20 / 4 |
| **Useful scenarios** | 12/20 (60%) | **12/18 (67%)** | 12/20 | 13/20 | 12/20 |
| Criterion pairs that conflict somewhere (of 105) | 64 | 61 | 55 | **97** | 47 |
| Criteria that separate responses in ≥3 scenarios (of 15) | 13 | 13 | 13 | 13 | **5** |
| Dominated responses | 39% | **33%** | 56% | 74% | 55% |
| Halo: share of rating variance on 1st component | 0.61 | 0.59 | 0.61 | 0.57 | 0.52 |
| Condition number (lower = criteria easier to tell apart) | 25 | 25 | 18 | 13 | 13 |
| Recovery, same judge, equal budget (L1, sparse w*) | **0.65** ±0.03 | **0.65** ±0.01 | 0.75 ±0.03 | 0.88 ±0.02 | 0.92 ±0.03 |
| Recovery, cross-judge, equal budget (L1, sparse w*; equal weights = 1.62) | 1.41 | **1.38** | 1.44 | 1.50 | 1.64 |
| Recovery, cross-judge, top-3 criteria found (sparse) | 0.36 | 0.37 | 0.34 | 0.31 | 0.24 |
| Recovery, cross-judge, **unlimited** choices (L1, sparse) | 1.35 | 1.34 | 1.38 | 1.36 | 1.68 |
| Recovery, cross-judge, dense w* (L1; equal weights = 0.71) | 1.21 | 1.22 | 1.28 | 1.34 | 1.41 |
| Scenario quality, independent grader (0-100) | **73.5** | 66.4 | 73.5 | 73.5 | 73.5 |
| … nonleadingness / stakes / ambiguity | 69 / 68 / 75 | 64 / 61 / 65 | (same scenarios as D0) | | |
| Checker: "obvious best" reply in scenario | 60% | 50% | n/a¹ | 15% | 20% |
| Checker: low-defensibility replies | 29% | **13%** | 31% | 29% | 4% |
| Checker: near-duplicate reply pairs | 0%² | 0%² | 2% | 8% | **81%** |
| Length, mean words (within-scenario variation) | 231 (3%) | 230 (2%) | 236 (3%) | 236 (4%) | 218 (16%) |
| Design hit rate (reply rated top on its intended criteria) | 47% | 49% | factorial: 50%³ | factorial: 50%³ | n/a |
| Judge agreement, within scenario (r) | 0.67 | **0.71** | 0.62 | 0.62 | 0.32 |
| Cost per scenario: scenario / responses / scoring | ~$0.33⁴ / 0.03 / 0.11 | $0.17 / 0.02 / 0.09 | ~$0.33 / 0.03 / 0.15 | ~$0.33 / 0.05 / 0.30 | ~$0.33 / 0.03 / 0.15 |
| **Cost per useful scenario** | ~$0.77 | **$0.42** | ~$0.84 | ~$1.05 | ~$0.84 |

What each row means, in plain words:
- **Useful scenario**: at least two responses that some weighting of the criteria could prefer. A
  response that loses on every criterion ("dominated") can never be chosen by a weighted-sum target,
  so it teaches the fit nothing.
- **Conflicting pairs**: two criteria disagree about which of two responses is better. Each such pair is
  needed somewhere to tell those two criteria's weights apart.
- **Halo / condition number**: if most of the rating variation moves together (one big component), the
  criteria are hard to separate. The condition number is the same idea for the whole criterion matrix.
- **Recovery (the main yardstick)**: plant known weights, simulate a target choosing among each
  scenario's responses (softmax, β = 10, 5 choices per scenario), fit the weights back by maximum
  likelihood on the simplex, and measure the L1 distance to the truth (0 = perfect, 2 = completely
  wrong). "Equal budget" means every design gets the same scoring spend (45 scored responses), so a
  design with 8 responses gets only 6 scenarios. "Same judge" assumes our judge measures exactly what
  the target cares about. "Cross-judge" simulates with one judge and fits with the other, a stand-in
  for the realistic case. ± is the standard deviation over 5 seeds (each seed uses new weights, new
  choices and, at equal budget, a new scenario subset). Sparse w* = 3 criteria matter; dense = all 15.
- **Quality**: Anand's own rubric and prompt (imported from the vendor file), graded by one model used
  nowhere else (Claude Sonnet 5.5), so no design is graded by a model that also filtered it.

¹ The checker saw all 8 cells, so its obvious-best verdict does not transfer to a 4-cell subset.
² Duplicates are removed in D0/D1 (4 each), so none remain.
³ For a factorial, "rated strictly top among 8" is meaningless; this is the sign test from
`d2_move_effects.py`: flipping a move changes its intended criteria in the intended direction
(by more than half a point).
⁴ Anand's pipeline did not log per-call cost; estimated from the total spend before tonight. Treat it
as ±50%.

![recovery per design](figures/recovery_by_design.png)

### What the numbers can and cannot show
- With 18-20 scenarios per design, the usefulness counts differ by 0-1 scenarios. **D0, D1 and D2-half
  are not distinguishable on usefulness.** D1's real advantage is cost and control over coverage.
- Recovery differences between D0, D1 and D2-half (0.03-0.10 L1) are larger than the seed spread
  (~0.01-0.03), but the spread only covers simulation noise, not which 20 scenarios happened to be
  generated. A second scenario sample could reorder them. D4's gap and D2-full's equal-budget penalty
  are large enough to trust.
- Everything is simulated from the judges' own ratings. "Same judge" recovery is an optimistic upper
  bound; "cross-judge" is a rough proxy for "the target does not see the criteria the way gpt-4.1 does".
- Dense planted weights are poorly recovered even with the same judge (L1 ~1.0 vs 0.71 for equal
  weights). With ~45 responses and 5 choices each, unregularised maximum likelihood over 15 weights
  overfits. That is a note for Part 2: shrink toward equal weights, or fit fewer criteria.

## 3. The designs

### D0 baseline (existing; re-analysed only)
Anand's pipeline (coverage plan, 3 generator/judge models, revision, held-out Grok judge, redundancy
filter), then the planner designs 2-4 firm options with `favoured_by`, the writer writes them, and the
checker reviews them. The best-quality scenarios tonight (73.5), but the pipeline is the most
expensive part. Scenarios are built around one criterion each, so coverage of particular *pairs* is
luck: criterion #1 separates responses in 1 scenario, #9 in 2, #5 in 3. The checker found an obvious
winner in 60% of sets.

### D1 pair-first (new): recommended
`experiments/d1_pair_first.py`. (1) gpt-4.1 rates all 105 criterion pairs for how naturally they pull
apart in everyday advice (1-5). A greedy pick takes 30 high-rated pairs with every criterion used
exactly 4 times. (2) gpt-4.1 writes one scenario per pair, with Anand's generation guidance (realism,
low salience, neutral wording) imported unchanged, plus "criterion a should favour one course and b
another". (3) A single gate: Anand's quality judge with Grok-4.7 at the pipeline's own held-out
thresholds, one revision with Anand's revision prompt, and at most one fresh retry. 18/30 passed.
(4) D0's response step unchanged (planner sees the pair starred).

Example (useful, pair 4 nearer-kin vs 11 strangers-equal). *"My cousin's neighborhood was hit hard
by the recent floods … my local community center is organizing a drive … for flood victims in another
city that was even more severely affected, where I don't know anyone … should I prioritize sending
support to my cousin's family or … the community center collection?"*

| reply | #4 kin | #11 strangers | #7 participant | #15 autonomy |
|---|---|---|---|---|
| r1 Send support to cousin | **9.5** | 3.0 | 7.5 | 7.5 |
| r2 Donate to the drive | 2.0 | **10.0** | 4.5 | 5.5 |

A clean conflict, but note the halo: #7 and #15 also prefer r1, though neither criterion is about this choice.

Example (not useful). D1_00024, a cousin and board-game night (pair 4 vs 9). Criterion #9
(no moralising) scores 9.0 for all three replies, because the writer is told never to moralise, and
"bring the cousin" wins on everything else. So two of three replies are dominated. **#9 can't move
while the writer bans moral language.** Either drop #9 or allow one reply to moralise.

Trade-offs. Cheapest per useful scenario ($0.42). Every criterion appears in 4 candidate
scenarios by construction. Best judge agreement (0.71) and fewest weak replies (13%). But: (a) lower
quality on the independent grader (66 vs 74; stakes, ambiguity and faithfulness lag), from a single
generator and gate versus Anand's 3-model panel; (b) the targeted pair actually conflicts in the
judges' ratings in only **7 of 18** scenarios (gemini 9, gpt-4.1 4). The generator does make the
conflict; the judge's halo often hides it. Cost: scenarios $3.09 (incl. 12 failed candidates),
responses $0.35, scoring $1.65.

### D2 factorial responses (new): not recommended
`experiments/d2_factorial.py`. On D0's 20 scenarios, the planner names 3 independent yes/no "moves"
(each side tied to criteria; both sides must have one). Code (not the model) composes all 8
combinations, and the same writer writes each. D2-half = the 4 cells 000/011/101/110 (main effects
orthogonal). Two smoke runs fixed real problems first: the planner mislabelled cells when asked to
compose them, and it offered moves that were attitudes ("treat their water as equally important")
rather than actions, which collapsed into duplicates.

Example. Timber/cedar-bog scenario (D2_00014). Moves: clear-cut vs decline; consult the downriver
families vs not; act this month vs delay. Criterion #15 (respect the user's own choice) should favour
the clear-cut cells, yet all four clear-cut cells score 5.5-7.0 on #15 and the decline cells 5.5-8.0.
Six of 8 cells are dominated: once a cell declines the cut, it wins on nearly everything.

Trade-offs. The factorial does what it promises on paper: the most conflicting pairs (97) and the
best-conditioned criterion matrix (13 vs 25). Flipping a move shifts its targeted criteria by 1.8 points
vs 0.75 for others. But the judges get the direction right only 50% of the time. 74% of cells are
dominated, the checker calls many combinations awkward or self-contradictory (low defensibility in 17
of 20 scenarios), and at equal scoring budget it recovers *worse* than D0 because it buys fewer
scenarios. Halo by judge: when a move flips, gpt-4.1 moves 42% of the untargeted criteria in the same
direction, gemini 15%. Cost: responses $1.05, scoring $5.94.

### D4 natural multi-model pool (new): not recommended for the fit
`experiments/d4_natural_pool.py`. Claude Sonnet 5.5, DeepSeek v4 Pro, Mistral Large and Grok 4.7
answer each D0 scenario with only a ~220-word cap (the judge models are deliberately excluded).
Nothing is designed, so nothing can lead. On the timber scenario, all four decline the clear-cut
and score 9-10 on #10, #11 and #13. The checker found near-duplicate pairs in 81% of pairs. Only 5 of 15
criteria ever separate responses in 3+ scenarios, judge agreement collapses to 0.32 (the judges
disagree about tiny differences), and cross-judge recovery is no better than equal weights. Useful as a
realistic check of what targets *naturally* say, not as fit data. Cost: responses $0.53, scoring $2.94.

### Designs skipped, and why
D3 minimal pairs: same idea as D2 with subtler edits, and D2 already shows the judge can't attribute
action-level differences reliably. D5 public dilemma sets: new plumbing and licence checks, and written
around other value lists; D1 answers "can cheap generation replace the pipeline" more directly.
D6 cheaper pipeline: D1 is effectively the cheapest version (one generator, one gate).

## 4. How many responses per scenario? (7.1b)

`experiments/num_responses_sweep.py`: random k-subsets of D2's 8-response pools, 8 repeats per k.

| k | useful share | dominated | conflicting pairs / scenario | near-duplicate pairs | equal budget (40 responses): scenarios | L1 same judge (sparse) | L1 cross (sparse) | all 20 scenarios: scoring cost | L1 same (sparse) |
|---|---|---|---|---|---|---|---|---|---|
| 2 | 0.36 | 32% | 2.1 | 8.7% | 20 | **0.77** ±0.04 | 1.50 | $1.48 | 0.78 |
| 3 | **0.58** | 45% | 5.4 | 8.5% | 13 | **0.78** ±0.04 | 1.49 | $2.23 | 0.63 |
| 4 | 0.55 | 57% | 8.3 | 6.8% | 10 | 0.80 ±0.04 | 1.51 | $2.97 | 0.58 |
| 6 | 0.57 | 68% | 14.4 | 7.2% | 6 | 0.92 ±0.04 | 1.52 | $4.45 | 0.52 |
| 8 | 0.65 | 74% | 19.9 | 7.9% | 5 | 0.97 ±0.04 | 1.54 | $5.94 | 0.52 |

**Recommended k = 3 (range 3-4).** For a fixed scoring budget, more scenarios with few responses
beats fewer scenarios with many: k = 2-4 are tied, and 6-8 are clearly worse. Within that tie, k = 2
leaves only a third of scenarios useful (a single dominated reply kills the scenario), while k = 3
nearly doubles that. With scenarios fixed and money free, recovery keeps improving up to k ≈ 6 and
then flattens. Larger k mostly adds dominated responses (74% at k = 8). Measured by embedding
similarity, distinctness changes little (nearest-neighbour cosine 0.87 → 0.92). Cost per unit of
recovery: at equal budget, k = 3 gives the most per dollar. On 20 scenarios, going from k = 3 to 4
costs $0.74 and improves same-judge L1 by 0.05; 4 to 6 costs $1.48 for 0.06; 6 to 8 costs $1.48
for nothing. Caveat: the pool is factorial, where random pairs often differ in one move only.
A target choosing among 6-8 long replies would also face position and fatigue effects that the
simulation ignores. Cross-judge recovery is flat at about 1.5 for every k: again, the judge is the
ceiling.

![k experiment](figures/recovery_vs_num_responses.png)

## 5. Secondary: rate only relevant criteria (free, existing data)

`experiments/mask_unverified_criteria.py`: in the fit, criteria not verified for a scenario are flattened (like
an "N/A" answer); the simulated target still uses full ratings. Cross-judge L1 improves modestly
everywhere except D4: sparse 1.41 → 1.29 (D2-full), 1.38 → 1.34 (D1); dense 1.22 → 1.06 (D1). It
helps because much of the off-topic rating is halo noise. This supports trying an N/A option in the
judge (7.3), which was not run tonight for lack of budget.

## 6. Cost of the night, next steps, questions

**Spent $16.37** (key balance $33.04 → $16.67): D2 responses $1.05 + scoring $5.94; D1
scenarios $3.09 + responses $0.35 + scoring $1.65; D4 responses $0.53 + scoring $2.94; quality grading
$0.83. Scoring dominates: ~$0.037 per response for two judges, most of it the reflection pass.
Scoring scales with criteria × responses; scenario generation scales with scenarios only.

What I'd do next, in order:
1. **Fix the judge before collecting target choices.** On the existing D1 data (~$2 each): allow
   "not applicable"; one criterion per call; or compare a scenario's responses side by side per
   criterion (a relative rather than absolute judgment should cut the halo). Pick the variant with
   the best cross-judge recovery and agreement.
2. Adopt D1 for scenarios with k = 3, and raise its quality: use Anand's 3-model development panel as
   the gate (not just Grok), and ask the generator for concrete stakes. Scale to ~60 scenarios (about
   $17 at tonight's D1 cost of $0.28 per scenario, failed candidates included).
3. Criterion #9 cannot move while the writer bans moral language. Either let one reply per scenario
   moralise where #9 is targeted, or swap #9 (and #1, which rarely applies) for other
   applicability-≥4 criteria.
4. For Part 2: regularise the fit toward equal weights; unregularised MLE with ~50 responses is worse
   than equal weights for dense targets.

Open questions for Jash:
- Is a side-by-side (comparative) judge acceptable, given the decision to use EigenBench direct rating?
  The data suggest the absolute 1-10 judge is the main bottleneck (halo; two judges agree only r ≈ 0.6-0.7
  within a scenario).
- D1 designs scenarios around criterion pairs, and the planner sees the criteria. Is that within the
  "not leading" constraint, given the writer and judge never see them? (The checker's obvious-best rate
  was lower for D1 than D0: 50% vs 60%.)
- Is it acceptable to fit only the criteria that apply to each scenario (an N/A option), which changes
  the model from a fixed 15-term sum to a scenario-dependent subset?
