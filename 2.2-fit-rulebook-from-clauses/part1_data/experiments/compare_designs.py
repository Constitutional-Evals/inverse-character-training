"""
One comparison table, one row per design, all metrics computed the same way.

    python experiments/compare_designs.py            # writes outputs/analysis/compare.json and prints the table

Designs whose ratings do not exist yet are skipped. Recovery settings: beta = 10, 5 simulated
target choices per scenario, 200 planted weight vectors per seed, 5 seeds; "equal budget" fits use
random scenario subsets with the same number of scored responses for every design.
"""

import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE / "experiments"))
from metrics import load_ratings, recovery_suite, separability, usefulness  # noqa: E402

R = HERE / "outputs"
HALF_RIDS = ["r1", "r4", "r6", "r7"]  # cells 000, 011, 101, 110 (CELLS order 000..111 = r1..r8)

DESIGNS = {
    "D0": {"ratings": R / "ratings/pilot_v2", "responses": R / "responses/pilot_v2", "quality": "D0",
           "scen_gen": None, "note": "earlier pilot: Anand pipeline + planner responses"},
    "D1": {"ratings": R / "ratings/exp_D1", "responses": R / "responses/exp_D1", "quality": "D1",
           "scen_gen": R / "scenarios/exp_D1", "note": "pair-first scenarios + D0 response step"},
    "D2-half": {"ratings": R / "ratings/exp_D2", "responses": R / "responses/exp_D2", "quality": "D0",
                "scen_gen": None, "half": True, "note": "factorial half fraction (4 cells) on D0 scenarios"},
    "D2-full": {"ratings": R / "ratings/exp_D2", "responses": R / "responses/exp_D2", "quality": "D0",
                "scen_gen": None, "note": "full 2^3 factorial (8 cells) on D0 scenarios"},
    "D4": {"ratings": R / "ratings/exp_D4", "responses": R / "responses/exp_D4", "quality": "D0",
           "scen_gen": None, "note": "natural answers from 4 models, 220-word cap, D0 scenarios"},
}
# Anand's pipeline cost for the pilot run was not logged per call; estimated from the total spend
# before tonight (about $15) minus logged response/ratings/criteria costs, split over smoke + pilot.
D0_SCENARIO_COST_EST = 0.33  # dollars per accepted scenario (estimate)


def usage_cost(folder, prefixes=None):
    p = Path(folder) / "usage.jsonl"
    if not p.exists():
        return 0.0, 0
    rows = [json.loads(l) for l in open(p)]
    if prefixes:
        rows = [r for r in rows if r["tag"].split(":")[0] in prefixes]
    return sum(r["cost"] for r in rows), len(rows)


def response_quality(resp_path, keep_rids=None):
    rows = [json.loads(l) for l in open(resp_path)]
    ob, lowdef, dup_pairs, pairs, words, cv = 0, 0, 0, 0, [], []
    n_resp = 0
    for r in rows:
        resps = [x for x in r["responses"] if x["kind"] == "committed" and (keep_rids is None or x["response_id"].split(":")[1] in keep_rids)]
        ids = {x["response_id"] for x in resps}
        if len(resps) < 2:
            continue
        checks = r.get("checks") or []
        ob += any(c["obvious_best"] in ids for c in checks if c.get("obvious_best"))
        for x in resps:
            d = [c["defensible"][x["response_id"]] for c in checks if x["response_id"] in c["defensible"]]
            lowdef += bool(d) and np.mean(d) < 3
        k = len(resps)
        pairs += k * (k - 1) / 2
        dp = {tuple(sorted(p)) for c in checks for p in c["duplicates"] if set(p) <= ids}
        dp |= {tuple(sorted([x["response_id"], y["response_id"]])) for x in r.get("removed_responses", []) for y in []}
        dup_pairs += len(dp)
        w = [x["words"] for x in resps]
        words += w
        cv.append(np.std(w) / np.mean(w))
        n_resp += k
    n = len(rows)
    return {"obvious_best_rate": round(ob / n, 2), "low_defensibility_share": round(lowdef / n_resp, 2),
            "duplicate_pair_rate": round(dup_pairs / pairs, 3) if pairs else None,
            "duplicates_removed": sum(len(r.get("removed_responses", [])) for r in rows),
            "mean_words": round(float(np.mean(words)), 0), "length_cv_within_scenario": round(float(np.mean(cv)), 3)}


def quality(name):
    p = R / "quality" / f"{name}.jsonl"
    if not p.exists():
        return None
    rows = [json.loads(l) for l in open(p)]
    dims = list(rows[0]["scores"])
    out = {d: round(float(np.mean([r["scores"][d] for r in rows])), 1) for d in dims}
    out["average"] = round(float(np.mean([r["average"] for r in rows])), 1)
    out["surface_cue_neutrality"] = round(float(np.mean([r["surface_cue_neutrality"] for r in rows])), 1)
    out["pass_rate_heldout_thresholds"] = round(float(np.mean([
        r["average"] >= 75 and min(r["scores"].values()) >= 60 and r["surface_cue_neutrality"] >= 60 and r["flaw_type"] == "none"
        for r in rows])), 2)
    return out


def judge_stats(folder):
    s = json.loads((Path(folder) / "summary.json").read_text())
    ag = s["judge_agreement"][0] if s["judge_agreement"] else {}
    return {"design_hit_rate": s.get("design_hit_rate"), "judge_r_raw": ag.get("corr_raw"),
            "judge_r_within": ag.get("corr_within_scenario")}


def main(only=None, draws=200, seeds=(0, 1, 2, 3, 4)):
    out = {}
    datasets = {}
    for name, cfg in DESIGNS.items():
        if only and name not in only:
            continue
        rp = cfg["ratings"] / "ratings.jsonl"
        if not rp.exists():
            print(f"skip {name}: no ratings yet")
            continue
        subset = None
        if cfg.get("half"):
            rows = [json.loads(l) for l in open(rp)]
            subset = {}
            for r in rows:
                if r["response_id"].split(":")[1] in HALF_RIDS:
                    subset.setdefault(r["scenario_id"], set()).add(r["response_id"])
        datasets[name] = load_ratings(rp, subset=subset)

    budget = min(sum(len(s["rids"]) for s in d["scenarios"].values()) for d in datasets.values())
    budget = min(budget, 56)
    print(f"equal-budget recovery: {budget} scored responses per design")

    for name, d in datasets.items():
        cfg = DESIGNS[name]
        u = usefulness(d)
        sep = separability(d)
        rec = recovery_suite(d, draws=draws, seeds=seeds)
        k_mean = u["responses"] / u["scenarios"]
        n_eq = max(2, int(round(budget / k_mean)))
        rec_eq = recovery_suite(d, draws=draws, seeds=seeds, n_scen=n_eq)
        rec_inf = recovery_suite(d, draws=100, seeds=seeds[:2], infinite=True)

        # costs: scoring per committed response (both judges), generation per scenario
        sc_cost, sc_calls = usage_cost(cfg["ratings"])
        n_rated = len({(json.loads(l)["response_id"]) for l in open(cfg["ratings"] / "ratings.jsonl")})
        per_resp = sc_cost / n_rated if n_rated else 0
        gen_cost, gen_calls = usage_cost(cfg["responses"])
        n_scen_written = sum(1 for _ in open(cfg["responses"] / "responses.jsonl"))
        if cfg.get("half"):
            gen_cost *= 0.5  # half the cells written (planning cost is shared; approximate)
        if cfg["scen_gen"]:
            s_cost, s_calls = usage_cost(cfg["scen_gen"])
            scen_cost = s_cost / max(n_scen_written, 1)
        else:
            scen_cost = D0_SCENARIO_COST_EST
        per_scen = scen_cost + gen_cost / n_scen_written + per_resp * k_mean
        rq = response_quality(cfg["responses"] / "responses.jsonl", HALF_RIDS if cfg.get("half") else None)
        js = judge_stats(cfg["ratings"])
        out[name] = {
            "note": cfg["note"], "scenarios": u["scenarios"], "k_mean": round(k_mean, 2),
            "useful": u["useful"], "useful_share": round(u["useful_share"], 2),
            "pair_coverage": u["pair_coverage"], "criteria_separating_ge3": u["criteria_separating_ge3"],
            "criterion_coverage": u["criterion_coverage"], "dominated_share": round(u["dominated_share"], 2),
            "separability": sep, "recovery_full": rec, "recovery_equal_budget": {"n_scen": n_eq, **rec_eq},
            "recovery_infinite": rec_inf, "quality": quality(cfg["quality"]), "response_quality": rq, "judges": js,
            "cost": {"scenario_per_scenario": round(scen_cost, 3), "responses_per_scenario": round(gen_cost / n_scen_written, 3),
                     "scoring_per_response_2judges": round(per_resp, 4), "total_per_scenario": round(per_scen, 3),
                     "per_useful_scenario": round(per_scen * u["scenarios"] / max(u["useful"], 1), 3)},
        }
    (R / "analysis").mkdir(parents=True, exist_ok=True)
    (R / "analysis" / "compare.json").write_text(json.dumps(out, indent=1))
    print_table(out)
    return out


def print_table(out):
    g = lambda r, k, m="mean": r[k]["l1"][m] if k in r else None
    print(f"\n{'design':<8} {'n':>3} {'k':>4} {'useful':>7} {'pairs':>5} {'crit>=3':>7} {'dom':>5} {'pc1':>5} {'cond':>6} "
          f"{'L1 sp same':>10} {'L1 sp cross':>11} {'rho dn cross':>12} {'L1 eqB sp cross':>15} {'qual':>5} {'$/useful':>8}")
    for name, o in out.items():
        rf, re_ = o["recovery_full"], o["recovery_equal_budget"]
        print(f"{name:<8} {o['scenarios']:>3} {o['k_mean']:>4} {o['useful']:>3}/{o['scenarios']:<3} {o['pair_coverage']:>5} "
              f"{o['criteria_separating_ge3']:>7} {o['dominated_share']:>5} {o['separability']['pc1_share']:>5.2f} "
              f"{o['separability']['condition_number']:>6.1f} {rf['sparse_same']['l1']['mean']:>10} {rf['sparse_cross']['l1']['mean']:>11} "
              f"{rf['dense_cross']['spearman']['mean']:>12} {re_['sparse_cross']['l1']['mean']:>15} "
              f"{(o['quality'] or {}).get('average', '-'):>5} {o['cost']['per_useful_scenario']:>8}")


if __name__ == "__main__":
    main(only=sys.argv[1:] or None)
