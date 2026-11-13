"""
Do frontier judges agree with each other? Analysis of the exp_frontier_judges re-scoring. No API calls.

    python experiments/frontier_judges.py      # writes outputs/ratings/exp_frontier_judges/analysis.json

Data: 5 anchored (DA) scenarios with 2+ verified criteria (25 replies), rated with the unchanged
EigenBench judge (pipeline/judge.py) by 6 judges, in two sub-runs that differ only in token caps:
  outputs/ratings/exp_frontier_judges/default/  gpt-6-astra, grok-4.7, gpt-4.1, gemini-4-argon (caps 2048/1024)
  outputs/ratings/exp_frontier_judges/highcap/  claude-fable-5.2, gemini-4-carbon (caps 4096/4096; at 1024
                                                 their reasoning used the whole rating budget and the reply was empty)
Both are merged into outputs/ratings/exp_frontier_judges/ratings_all_judges.jsonl.

Per judge:
- anchor hit rate: share of replies rated strictly highest (by > half a point) on their own anchor criterion (chance 0.2)
- halo winner share: per scenario, for how many of the 5 anchor criteria the same reply wins (1.0 = one reply wins all)
- PC1 share: variance share of the first principal component of the scenario-centred 25 x 15 rating matrix
- same-judge recovery: simulate and fit with the same judge (identifiability floor for this tiny set)
Per pair:
- within-scenario r: Pearson r of ratings after removing each (scenario, criterion) mean (what the fit sees)
- cross-judge recovery: plant sparse weights, simulate choices with judge A, fit with judge B; mean L1 error
  of the weights (0 = exact, 2 = disjoint; "equal weights" baseline given). Infinite choices, so the error
  comes only from the judges disagreeing, not from sampling noise. Averaged over both directions.
"""
import json
import sys
from itertools import combinations
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE / "experiments"))
sys.path.insert(0, str(HERE / "pipeline"))
from anchored_effects import anchor_stats  # noqa: E402
from judge import centered, corr  # noqa: E402
from metrics import load_ratings, separability, simulate_batched  # noqa: E402

BASE = HERE / "outputs/ratings/exp_frontier_judges"
JUDGES = ["anthropic/claude-fable-5.2", "openai/gpt-6-astra", "google/gemini-4-carbon", "x-ai/grok-4.7",
          "openai/gpt-4.1", "google/gemini-4-argon"]
SEEDS = (0, 1, 2)


def recovery(data, a, b):
    l1, base = [], []
    for seed in SEEDS:
        r = simulate_batched(data, a, b, kind="sparse", beta=10.0, draws=200, seed=seed, infinite=True)
        l1.append(r["l1"].mean())
        base.append(r["l1_equal"].mean())
    return round(float(np.mean(l1)), 3), round(float(np.mean(base)), 3)


def main():
    rows = []
    for sub in ("default", "highcap"):
        rows += [json.loads(l) for l in open(BASE / sub / "ratings.jsonl")]
    rows.sort(key=lambda r: (r["scenario_id"], r["response_id"], r["judge"]))
    merged = BASE / "ratings_all_judges.jsonl"
    merged.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))

    judges = [j for j in JUDGES if any(r["judge"] == j for r in rows)]
    counts = {j: sum(r["judge"] == j for r in rows) for j in judges}
    data = load_ratings(merged)  # keeps only replies rated by every judge
    n_resp = sum(len(s["rids"]) for s in data["scenarios"].values())

    resp = [json.loads(l) for l in open(HERE / "outputs/responses/exp_frontier_judges/responses_all5.jsonl")]
    anchors = {r["scenario_id"]: {x["response_id"]: (x["anchor"], x["anchor_verified"]) for x in r["responses"]} for r in resp}
    cells = {j: {} for j in judges}               # (sid, rid, c) -> rating
    nested = {j: {} for j in judges}              # sid -> rid -> {c: rating}
    for r in rows:
        nested[r["judge"]].setdefault(r["scenario_id"], {})[r["response_id"]] = r["ratings"]
        for c, v in r["ratings"].items():
            cells[r["judge"]][(r["scenario_id"], r["response_id"], c)] = v

    per_judge = {}
    for j in judges:
        st = anchor_stats(nested[j], anchors)
        same, base = recovery(data, j, j)
        per_judge[j] = {"ratings": counts[j], "anchor_hit_rate": st["all"]["hit_rate"],
                        "anchor_hit_rate_verified": st["verified_anchor"]["hit_rate"],
                        "anchor_margin": st["all"]["mean_margin"], "halo_winner_share": st["halo_winner_share"],
                        "pc1_share": round(separability(data, [j])["pc1_share"], 3),
                        "same_judge_l1": same, "equal_weights_l1": base}

    pairs = []
    for a, b in combinations(judges, 2):
        keys = sorted(set(cells[a]) & set(cells[b]))
        ab, _ = recovery(data, a, b)
        ba, _ = recovery(data, b, a)
        pairs.append({"judges": [a, b], "cells": len(keys),
                      "r_within": corr(centered(cells[a], keys), centered(cells[b], keys)),
                      "r_raw": corr([cells[a][k] for k in keys], [cells[b][k] for k in keys]),
                      "cross_l1": round((ab + ba) / 2, 3), "cross_l1_a_sim_b_fit": ab, "cross_l1_b_sim_a_fit": ba})

    out = {"scenarios": sorted(data["scenarios"]), "responses_rated_by_all": n_resp, "judges": judges,
           "per_judge": per_judge, "pairs": pairs}
    p = HERE / "outputs/ratings/exp_frontier_judges/analysis.json"
    p.write_text(json.dumps(out, indent=1))

    short = lambda j: j.split("/")[1]
    print(f"{n_resp} replies rated by all {len(judges)} judges in {len(data['scenarios'])} scenarios\n")
    print(f"{'judge':<20}{'hit':>6}{'hit(ver)':>9}{'halo':>6}{'PC1':>6}{'same L1':>8}")
    for j, s in per_judge.items():
        print(f"{short(j):<20}{s['anchor_hit_rate']:>6.2f}{s['anchor_hit_rate_verified']:>9.2f}{s['halo_winner_share']:>6.2f}"
              f"{s['pc1_share']:>6.2f}{s['same_judge_l1']:>8.2f}")
    print(f"(equal-weights L1 baseline {per_judge[judges[0]]['equal_weights_l1']:.2f})\n")
    print(f"{'pair':<44}{'r_within':>9}{'cross L1':>9}")
    for q in pairs:
        print(f"{short(q['judges'][0]) + ' vs ' + short(q['judges'][1]):<44}{q['r_within']:>9.2f}{q['cross_l1']:>9.2f}")
    print(f"\nWrote {merged} and {p}")


if __name__ == "__main__":
    main()
