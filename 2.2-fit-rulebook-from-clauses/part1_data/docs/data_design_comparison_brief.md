# Handoff: ICT 2.2 Part 1 (dataset and scoring) — overnight experiments

> Archived brief (originally `part1_data/CLAUDE_CODE_HANDOFF.md`). The file layout has changed since:
> the stage scripts below now live in `part1_data/pipeline/` (run as `python pipeline/judge.py ...`),
> and report figures are in `part1_data/reports/figures/`. See `../README.md` for the current layout.

You are picking up a research codebase mid-project. Read this whole file first, then the code in
`part1_data/`. Work autonomously overnight; the user will read your report in the morning. Tonight's goal is to
find a better way of generating the data, by building and comparing alternative designs (section 7).

> **OpenRouter budget — keep this in mind all night.** Every model call goes through one OpenRouter
> key with a hard spending limit. About $30 is left on it; tonight's cap is **$25**. Check the remaining
> balance before you start, between experiments, and before any large run:
> `curl -s https://openrouter.ai/api/v1/key -H "Authorization: Bearer $OPENROUTER_API_KEY" | python -c "import json,sys; d=json.load(sys.stdin)['data']; print('remaining $', round(d['limit_remaining'],2), '| today $', round(d['usage_daily'],2))"`
> Estimate the cost of a run from a 2-3 scenario smoke run before doing it at full size, and log what
> each experiment cost. If the key runs dry mid-run, everything stops, so leave margin.

## 1. Project in one paragraph

Inverse Character Training (ICT): recover a black-box LLM's "constitution" (which value criteria it
follows) from its behaviour. Area 2.2, "Fit a rulebook from clauses": given a fixed list of criteria
c_1..c_K, model a target's preferences as a weighted sum
`u_w(s, a) = Σ_k w_k · j(s, a, c_k)` with `j ∈ [0,1]` a judge's rating of response `a` to scenario `s`
on criterion `c_k`, `w` on the simplex, and choice probabilities `q_w = softmax(β · u_w)` over the
responses of a scenario with fixed β. Fit `w` by minimising negative log-likelihood of the target's
choices; evaluate on held-out scenarios against equal weights; check that the ranking of criteria
holds across splits and across judges. The supervisor (Jash Vira) said explicitly: use the EigenBench
1-10 direct-rating judge, weighted sum + softmax, **no Plackett-Luce**; worry about cost vs coverage;
"make sure the answer options aren't leading, and models don't latch onto some select ones."

The user (Aitzaz) owns **Part 1: dataset and scoring** (criteria and scenario selection, responses,
scoring responses against criteria, experiment design, data format). Someone else owns Part 2 (the
fit). Do not edit Part 2's code if it exists in the repo.

## 1b. The agreed design (context; what the user, Jash and the earlier design doc settled)

- **Criteria:** a subset (pilot: 15) of Jash's 78-criterion shortlist, broadly applicable and mutually different, one per tradition.
- **Scenarios:** ~20-30 for the pilot, from Anand's scenario pipeline, each verified to involve some of the criteria.
- **Responses:** k substantively different responses per scenario (k may vary; currently 2-4, plus one natural answer). Responses differ in *what they recommend*, not in tone or quality: same length and style, no moral or value language, no response obviously best. No conflict structure is assumed in advance beyond what is needed for criteria to disagree.
- **Scoring:** the EigenBench direct-rating judge, one response per call, reflection then 1-10 on every criterion, rescaled `(x-1)/9`, so `j(s,a,c) ∈ [0,1]`. Two judges for a reliability check.
- **Fit (Part 2):** `u_w(s,a) = Σ_k w_k j(s,a,c_k)`, `w` on the simplex, `q_w = softmax(β u_w)` over a scenario's responses with fixed β, MLE on the target's choices; held-out scenarios (variants kept together), compare against equal weights, check the ranking across splits and judges. No Plackett-Luce.
- **Checks the design calls for:** options not leading (Jash's concern), judge reliability (agreement, repeats), spread (each criterion actually separates responses somewhere).
- An earlier PDF design doc exists but predates Jash's judge decision (it described per-criterion calls and a Plackett-Luce option); this section supersedes it.

### Data structure (full spec in `DATA_FORMAT.md`)
- Keys: `scenario_id`; `response_id` = `<scenario_id>:r1`, `:r2`, … for committed responses and `<scenario_id>:n` for the natural one; criterion `pilot_id` 1..K.
- `responses.jsonl`: one row per scenario: `scenario`, `verified_criteria`, `responses[]` (`response_id, kind, label, recommendation, favoured_by, text, words, model`), `checks`, `flags`, `keep`.
- `ratings.jsonl`: one row per response × judge × repeat: `ratings_1to10`, `ratings` (0-1), `reflection`.
- `ratings_matrix.json`: `scenarios[sid].responses[rid].ratings[pilot_id]` = mean over judges (this is `j(s,a,c)` for the fit), plus `dominated`, `criterion_conflicts`, `informative` per scenario.
- Not yet collected (out of scope tonight): the target's choices, proposed as `choices.jsonl` (`scenario_id, target, shown, chosen`).

## 2. Where things are

Repo: `~/code/ict-sae/ict/inverse-character-training` (GitHub: Constitutional-Evals/inverse-character-training, **public**).
Working folder: `2.2-fit-rulebook-from-clauses/part1_data/`. Python venv: `source ~/code/ict-sae/ict/.venv/bin/activate`.
All commands below run from `part1_data/`.

| File | Stage |
|---|---|
| `data/recommended_78_criteria.csv` | Jash's 78-criterion shortlist (from 24 value-tradition "anchor" constitutions) |
| `select_criteria.py` | Stage 1: LLM applicability filter (1-5, keep ≥4) then farthest-point embedding selection, one per tradition → 15 criteria |
| `vendor/scenario_pipeline_10_24_26.py` | Anand's scenario pipeline, byte-identical copy of `mass_scenario_generation_pipeline_10_24_26 (1).py` from Constitutional-Evals/constitutions, branch `massscenariogen`, folder `mass_scenario_gen/` (newest version, Oct 26). Do not modify; copy and change if you need variants |
| `run_scenarios.py` | Stage 2: wrapper that runs that pipeline with `--budget` and `--max-api-calls` |
| `make_responses.py` | Stage 3: k firm responses per scenario + leading check + one natural answer |
| `judge.py` | Stage 4: EigenBench direct-rating judge (prompts copied verbatim from jchang153/EigenBench `pipeline/protocol/direct_rating.py`), dominance/usefulness, design check, judge agreement |
| `common.py` | OpenRouter client with on-disk cache (`LLM.chat`), usage/cost logging, `check_models` |
| `show_scenario.py` | Pretty-print one scenario with its responses and score table |
| `DATA_FORMAT.md` | File formats handed to Part 2 |
| `README.md` | Commands per stage |
| `outputs/` | All generated data (gitignored). See below |

### Outputs that exist now (do not overwrite these folders; write new ones)
- `outputs/criteria/criteria_pilot.jsonl`, `constitution_pilot.json`, `applicability.jsonl`
- `outputs/scenarios/smoke/` (5 scenarios), `outputs/scenarios/pilot/` (20 scenarios; main file `generated_scenarios_rich.jsonl`)
- `outputs/responses/smoke/`, `outputs/responses/pilot/` (v1 responses), `outputs/responses/pilot_v2/` (current best responses)
- `outputs/ratings/smoke/`, `outputs/ratings/pilot/` (v1), `outputs/ratings/pilot_v2/` (current best: `ratings.jsonl`, `ratings_matrix.json`, `summary.json`)
- Each folder has `cache/` (one JSON per API reply, keyed by request hash) and `usage.jsonl` (cost per call)

## 3. The 15 pilot criteria (pilot_id: criterion_id)

1 secular-humanism#01 (natural causes over supernatural) · 2 marxism#07 (workers' many-sided development, not just output/wages) ·
3 buddhism#04 (attention and sense-restraint over numbing/compulsive consumption) · 4 confucianism#04 (nearer kin obligation over impartial calculation) ·
5 stoicism#04 (answer wrongdoing without anger or payback) · 6 taoism#09 (sufficiency and contentment over fame/wealth/rank) ·
7 care-ethics#12 (work with the other as an active participant, not managing/steering) · 8 conservatism#02 (gradual tested reform over comprehensive redesign) ·
9 kantian-deontology#11 (prudential counsel without moralising when no duty is at stake) · 10 environmental-ethics#06 (reduce avoidable suffering in human-controlled settings) ·
11 kindness#01 (strangers' interests on the same scale as one's own circle) · 12 judaism#09 (every life and dignity sacred, not valued by usefulness) ·
13 neo-aristotelian-virtue-ethics#12 (treat the worth of the user's end as part of the question) · 14 ubuntu#04 (inclusive deliberation and compromise over bare majority) ·
15 utilitarianism#04 (respect a competent adult's settled choices about their own life)

Full texts: `outputs/criteria/criteria_pilot.jsonl`.

## 4. What each stage does now

- **Stage 2 (Anand's pipeline):** coverage plan (Kimi) → 40 slots (target criteria × scenario type × setting) → generation → 3 development judges (Kimi k3.1-pro, Qwen3.8-max, Gemini-4-argon) score a 6-dimension rubric (faithfulness, behavioral discriminativeness, naturalness, ambiguity, non-leadingness, stakes; avg ≥75, each ≥60) plus surface-cue neutrality and mark `verified_criteria` → one revision cycle → freeze → held-out judge (Grok-4.7) → embedding + LLM redundancy filter. Output rows: `scenario_id, scenario, verified_criteria (pilot_ids), scenario_type, setting, intended_tension, plausible_actions`. Pilot: 40 candidates → 31 dev passes → 21 held-out → 20 final. The pipeline freezes, so a run cannot be topped up; new scenarios need a new run name (and scenario_ids then collide across runs: prefix them if you merge).
- **Stage 3 (`make_responses.py`):** a planner (writer model, gpt-4.1) turns `plausible_actions` into 2-4 firm, mutually exclusive recommendations; it sees the 15 criteria (verified ones starred) and must assign each option a `favoured_by` set such that every option has a criterion no other option has. The writer (gpt-4.1) never sees criteria and writes ~220-word replies with identical instructions and no moral/value language; off-length replies are rewritten once. A checker (gemini-4-argon) rates defensibility 1-5 in two orders, flags obvious best / duplicates / value language; duplicates are removed (later one of a pair), everything else is a note only. One natural reply per scenario (gpt-4.1, no instructions, ~450 words) is stored with `kind: natural`.
- **Stage 4 (`judge.py`):** EigenBench protocol per response: reflection (temp 0.2, 2048 tokens) then rating (temp 0, 1024 tokens; EigenBench uses 512) with `<criterion_N_rating>` tags 1-10 for all 15 criteria together, strict parse; rescaled `(x-1)/9`. Reports: per-criterion spread (max-min across a scenario's responses) split by verified/not; per scenario `dominated` committed responses (another response ≥ on every criterion within 1 point and > by more than 1 point on one) and `informative` = at least 2 non-dominated; `criterion_conflicts`; design hit rate (response strictly highest on its `favoured_by` criteria); judge agreement raw and within-scenario (scenario×criterion means removed); `--repeats 2` gives test-retest.

## 5. Results so far

| Run | Useful scenarios | Design hit rate | Judge agreement (within-scenario r) | Cost |
|---|---|---|---|---|
| pilot (v1, options not designed against criteria) | 7 / 20 | n/a | 0.69 | $0.51 + $2.90 |
| pilot_v2 (planner designs options so different criteria favour different responses) | **12 / 20** | **47%** (58/124; chance ≈ 33%) | 0.67 | $0.54 + $2.86 |

Per-criterion mean spread (0-1 scale), pilot_v2, verified vs not: most criteria 0.6-0.9 where verified,
0.2-0.4 otherwise. Barely moving anywhere: #1 (0.05 otherwise), #4 (0.21), #5 (never verified, 0.24),
#9 (0.26 verified / 0.12 otherwise; flat partly because the writer bans moral language, so no reply moralises).
#7 care-ethics moves in 95% of scenarios even when not verified, which looks like a judge **halo**
(overall "nicer answer" impression leaking into every criterion). The checker flagged `obvious_best`
in 12/20 scenarios.

Spend so far: about $15 total. Remaining budget: check with
`curl -s https://openrouter.ai/api/v1/key -H "Authorization: Bearer $OPENROUTER_API_KEY"` (`limit_remaining`).

## 6. Known weaknesses (the point of tonight)

1. **Halo / forced scoring.** The judge must give every criterion a number, even irrelevant ones, and tends to rate the response it likes higher on most criteria. Rated one response at a time with absolute scores.
2. **Identifiability.** Only 12/20 scenarios have real criterion conflicts, and some criteria may always move together, so their weights can't be separated however many choices are collected.
3. **Design check only 47%.** Either the planner's `favoured_by` is wrong or the judge doesn't separate criteria.
4. **Small n:** 20 scenarios, 2 judges, 1 repeat; several criteria have 0-1 verified scenarios.
5. **Natural answers are ~2× longer** than committed ones (length confound if mixed).
6. Leading-ness: Jash's concern. Options are now designed against criteria (planner sees them), which is a deliberate trade-off; writer and judge never see them.

## 7. What to do tonight

**Priority (from the user): high-quality, useful, structured data, and a better way of making it.**
Compare alternative *data-generation designs* on usefulness, quality, trade-offs and cost. You are
encouraged to invent designs beyond the ones below if you think they fit the purpose better.
Ignore real target models.

**Quality over quantity.** Do not try to run every idea. Pick the **2-3 designs** you judge most
promising (plus the D0 baseline, which already exists), and the **number-of-responses experiment**
(7.1b), and do them properly: same criteria and comparable size across designs, smoke run first,
enough scenarios for differences to be real, simulation repeated over seeds with spread reported,
and an honest account of what the numbers can and cannot show. A few careful comparisons with clear
conclusions are worth far more than many shallow ones. Explain in the report why you chose the
designs you ran and why you skipped the others.

"Useful" has a specific meaning here: the fit learns criterion weights only from scenarios where
**different criteria favour different responses**. If one response wins on every criterion, the
target's choice says nothing about which criterion it follows; if two criteria always move together,
their weights can never be separated. So a design is better if it produces many such conflicts,
across many different criterion pairs, while staying realistic and not leading.

### 7.1 Candidate designs (suggestions; choose 2-3, add or replace freely)

Hold scoring fixed at the EigenBench baseline judge (`judge.py`, gpt-4.1 + gemini-4-argon) so that
differences come from the data design, not the judge. Give every design the same 15 criteria and a
comparable size (e.g. ~20 scenarios, or a fixed dollar budget per design).

| ID | Design | Idea | Expected trade-off |
|---|---|---|---|
| D0 | **Baseline (current)** | Anand's pipeline → planner-designed responses (pilot_v2) | Realistic, well-filtered scenarios; scenarios are built around one criterion vs ordinary pulls, so conflicts between our criteria are hit-or-miss; pipeline is the expensive part |
| D1 | **Pair-first** | Pick criterion pairs that plausibly conflict first; generate a scenario for each pair, with one response per side (+ optional neutral) | Conflicts by construction and controllable pair coverage; risk of contrived or leading scenarios |
| D2 | **Factorial responses** | Per scenario, define 2-3 independent response "moves", each tied to a criterion (e.g. defer to the user's choice or not; gradual or sweeping change); generate responses for combinations (2² or 2³) | Near-orthogonal variation, best for separating criteria; more responses per scenario; combinations can read unnaturally |
| D3 | **Minimal pairs** | Start from one good response; make controlled edits that change it along one criterion at a time | Very clean attribution per criterion; responses very similar, possibly too subtle for the judge or the target |
| D4 | **Natural multi-model pool** | Same scenarios; natural answers from 4-6 diverse models (EigenBench/UltraFeedback style), optionally length-normalised | Realistic and cheap; answers often converge, few conflicts, length/quality confounds |
| D5 | **Reuse existing scenarios** | Scenarios from public value-dilemma sets (e.g. AIRiskDilemmas / LitmusValues, or other open dilemma datasets reachable via pip/GitHub) instead of generating; then D0's response step | Cheapest scenarios; may not match our criteria; check licence and fit |
| D6 | **Cheaper pipeline** | Anand's pipeline with a lighter configuration (fewer/cheaper development judges, no revision, or no held-out stage), copied rather than modifying the vendor file | Shows how much each quality stage costs and buys |

### 7.1b How many responses per scenario? (run this)

We currently have 2-4 committed responses per scenario. More responses per scenario give more
comparisons per scenario (and the softmax fit uses all of them), but scoring cost grows linearly with
k, responses get harder to keep distinct and non-dominated, and a real target choosing among many
long options may behave differently (position effects, fatigue). Decide k with data:
- For a set of scenarios, generate a large pool (e.g. 8 committed responses each, using the best
  design's response method), score them all once.
- Subsample k ∈ {2, 3, 4, 6, 8} responses per scenario (many random subsets) and, for each k, measure
  usefulness, separability and simulated recovery, **at equal total scoring budget** (fewer scenarios
  with large k vs more scenarios with small k). Also check distinctness at large k (checker duplicate
  rate, embedding similarity) and the share of dominated responses.
- Report a recommended k (or range) and the cost per unit of recovery accuracy for each k; note the
  trade-off between "more responses per scenario" and "more scenarios".

### 7.2 How to compare them (one table in the report, one row per design)

- **Usefulness:** share of scenarios that are useful (≥2 non-dominated responses, already in `judge.py`); number of distinct criterion pairs that conflict somewhere (pair coverage); criterion coverage (how many scenarios each criterion actually separates responses in).
- **Separability of criteria:** stack within-scenario centred rating differences into a matrix (rows = response pairs, columns = criteria); report rank, condition number, and the most correlated criterion pairs (confounded). A quick halo measure: share of within-scenario variance on the first principal component.
- **Recovery by simulation (the main yardstick, no target model needed):** plant weights `w*` (sparse: 2-3 criteria; and dense Dirichlet), simulate choices `~ softmax(β·u_{w*})` from one judge's ratings, fit `w` by MLE on the simplex (small fitter in your experiment folder), and report recovery (L1 error, rank correlation, top-3 overlap) vs number of scenarios. Do the cross-judge version too (simulate with gpt-4.1's ratings, fit with gemini's) since that is the realistic case.
- **Quality:** score each design's scenarios with Anand's rubric (reuse his quality-judge prompt and dimensions from the vendor file, with one judge model, so all designs are graded the same way): naturalness, non-leadingness, ambiguity, stakes, faithfulness. Also response length parity and the checker's obvious-best rate.
- **Judge reliability on that design:** design hit rate, judge agreement (within-scenario r).
- **Cost:** dollars and API calls for generation and for scoring separately; **cost per useful scenario**; cost to reach a given recovery accuracy. Note which costs scale with the number of criteria (scoring) and which with scenarios (generation).
- Variance: where cheap, repeat a design's simulation with different seeds and report spread, so differences between designs aren't noise.

### 7.3 Secondary (only if budget and time remain)
- Judge variants on the best design: allow "not applicable"; one criterion per call; compare all k responses side by side per criterion; score only relevant criteria (free on existing data).
- Swap low-signal criteria (#1, #4, #5, #9) for other applicability-≥4 criteria from `outputs/criteria/applicability.jsonl`.

### 7.4 Structure of the data
Every design must write the **same file formats** as `DATA_FORMAT.md` (`responses.jsonl`, then `judge.py` → `ratings.jsonl`, `ratings_matrix.json`), plus a `design` field and the design's own metadata (e.g. targeted pair, factorial cell), so Part 2 can use any design's output without changes and designs can be mixed. Prefix scenario_ids with the design ID (e.g. `D1_00003`) so they never collide. If a design needs extra fields, add them and document them in `DATA_FORMAT.md`.

### 7.5 Morning report (required)
Write `part1_data/reports/data_design_comparison.md`:
1. One-paragraph summary up front: which design you'd recommend and why, in plain words.
2. The comparison table (7.2), with a one-line plain explanation of each metric.
3. One section per design: what it does, 1-2 example scenarios with their responses and scores, trade-offs, cost.
4. Total cost of the night, what you'd do next, and open questions for Jash.
Keep explanations simple; the user wants plain language. Plots as PNGs next to the report, if useful.

## 8. Rules

- **Budget:** keep tonight's total API spend under **$20** (roughly $3-5 per design or experiment; the user may change this number), and stop before the key's `limit_remaining` gets below $5. Do a tiny smoke run (2-3 scenarios) of any new design before running it at full size. Every script should use `common.LLM` (or the same caching pattern) so reruns are free, and should print and log cost.
- **API key:** read `OPENROUTER_API_KEY` from the environment only. Never print it, write it to files, or put it in code, logs or commits.
- **Don't overwrite existing outputs** (`pilot`, `pilot_v2`, etc.). New runs get new names (e.g. `outputs/ratings/exp_<name>/`). Put new code under `part1_data/experiments/` unless it is a clean improvement to an existing stage, in which case keep the old behaviour available behind a flag and the EigenBench prompts verbatim in the default path.
- **Git:** work on branch `2.2-part1-data` (create it from the current branch if it doesn't exist). You may commit code locally; **do not push**, and do not commit anything under `outputs/` or any data/API responses. Do not commit `3.2-optimise-the-written-rulebook/src/rulebook_opt/prompts.py` or `prompts.py.bak` (the user's unrelated local changes). The repo is public: no endpoints, keys or personal info in committed files.
- Don't modify `vendor/scenario_pipeline_10_24_26.py` or Part 2's code.
- If something fails repeatedly, record it in the report and move on to the next experiment rather than burning budget.

## 9. Useful facts

- Model ids that worked through OpenRouter: `openai/gpt-4.1`, `openai/gpt-4.1-mini`, `google/gemini-4-argon`, `x-ai/grok-4.7`, `moonshotai/kimi-k3.1-pro`, `qwen/qwen3.8-max-0902`, embeddings `openai/text-embedding-3-small`. Others: check with `common.check_models`.
- OpenRouter returns per-call `usage.cost` when `extra_body={"usage": {"include": True}}` (already done in `common.LLM`).
- EigenBench formats criteria as `Criterion N: <text>` lines and expects `<criterion_N_rating>` tags; `judge.py` mirrors this.
- Reading data quickly: `python show_scenario.py --run pilot_v2` (list) and `python show_scenario.py --run pilot_v2 gen_00000 --full`.
- Related work for framing: EigenBench (arXiv 2509.01938), Inverse Constitutional AI (2406.06560), Open Problems in Constitutional Preference Reconstruction (2606.30116), UltraFeedback (2310.01377), LitmusValues/AIRiskDilemmas (2505.14633), ArmoRM (2406.12845; multi-objective reward + gating, close cousin of the weighted-sum model).
