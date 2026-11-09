"""
Experiment 7.1b: how many responses per scenario (k)?

Pool: the D2 full factorial (8 scored responses per scenario, 20 scenarios). For each k in
{2, 3, 4, 6, 8}, draw random k-subsets of each scenario's responses (many times) and measure:
- usefulness (share of scenarios with >= 2 non-dominated responses), dominated share, conflicting
  criterion pairs per scenario, distinctness (checker duplicate flags among the chosen pairs;
  embedding cosine of each response to its nearest neighbour in the same scenario);
- separability (PC1 share, condition number) and simulated recovery
  (a) at EQUAL SCORING BUDGET: B scored responses in total, so n_scen = B / k scenarios
  (fewer scenarios with large k vs more with small k), and
  (b) with all 20 scenarios (cost grows with k).
Recovery: beta = 10, 5 simulated choices per scenario, sparse and dense planted weights,
same-judge and cross-judge. Each repeat uses fresh subsets, weights and choices; we report the mean
and the standard deviation over repeats.

    python experiments/num_responses_sweep.py         # writes outputs/analysis/num_responses_sweep.json + PNG
"""

import json
import sys
from itertools import combinations
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE / "pipeline"))
sys.path.insert(0, str(HERE / "experiments"))
from metrics import dominated, conflict_pairs, load_ratings, mean_X, recovery_suite, separability  # noqa: E402

KS = [2, 3, 4, 6, 8]
BUDGET = 40  # scored responses; with 20 scenarios this is the most that k = 2 can use
REPEATS = 8
OUT = HERE / "outputs/analysis"


def embed_similarity(resp_rows):
    """Nearest-neighbour cosine per response within its scenario (cached embeddings, ~$0.001)."""
    from common import LLM  # noqa: F401  (key check)
    import hashlib
    from openai import OpenAI
    from common import OPENROUTER, api_key
    cache = OUT / "embed_cache.json"
    store = json.loads(cache.read_text()) if cache.exists() else {}
    texts = {x["response_id"]: x["text"] for r in resp_rows for x in r["responses"]}
    missing = [rid for rid, t in texts.items() if hashlib.sha256(t.encode()).hexdigest() not in store]
    if missing:
        client = OpenAI(base_url=OPENROUTER, api_key=api_key())
        for i in range(0, len(missing), 64):
            batch = missing[i:i + 64]
            res = client.embeddings.create(model="openai/text-embedding-3-small", input=[texts[r] for r in batch])
            for rid, e in zip(batch, res.data):
                store[hashlib.sha256(texts[rid].encode()).hexdigest()] = e.embedding
        cache.write_text(json.dumps(store))
    return {rid: np.array(store[hashlib.sha256(t.encode()).hexdigest()]) for rid, t in texts.items()}


def subset_data(data, pick):
    """pick: {sid: [indices]} -> a new dataset with only those responses."""
    scen = {}
    for sid, idx in pick.items():
        s = data["scenarios"][sid]
        scen[sid] = {"rids": [s["rids"][i] for i in idx], "X": {j: s["X"][j][idx] for j in data["judges"]}}
    return {"judges": data["judges"], "K": data["K"], "scenarios": scen}


def main():
    data = load_ratings(HERE / "outputs/ratings/exp_D2/ratings.jsonl")
    sids = list(data["scenarios"])
    assert all(len(data["scenarios"][s]["rids"]) == 8 for s in sids), "need complete 8-cell pools"
    resp_rows = [json.loads(l) for l in open(HERE / "outputs/responses/exp_D2/responses.jsonl")]
    dups = {r["scenario_id"]: {tuple(sorted(p)) for c in r["checks"] for p in c["duplicates"]} for r in resp_rows}
    emb = embed_similarity(resp_rows)
    per_resp = json.loads((OUT / "compare.json").read_text()).get("D2-full", {}).get("cost", {}).get("scoring_per_response_2judges", 0.0377)

    results = {}
    for k in KS:
        rng = np.random.default_rng(k)
        stats = {"useful_share": [], "dominated_share": [], "conflict_pairs_per_scen": [], "dup_rate": [], "nn_cos": [],
                 "pc1": [], "cond": []}
        rec_eq, rec_all = [], []
        for rep in range(REPEATS):
            pick = {s: sorted(rng.choice(8, k, replace=False).tolist()) for s in sids}
            d = subset_data(data, pick)
            useful, domn, cps, dp, npairs, nn = 0, 0, [], 0, 0, []
            for s in sids:
                X = mean_X(d["scenarios"][s], d["judges"])
                dom = dominated(X)
                useful += len(X) - len(dom) >= 2
                domn += len(dom)
                cps.append(len(conflict_pairs(X)))
                rids = d["scenarios"][s]["rids"]
                for a, b in combinations(rids, 2):
                    npairs += 1
                    dp += tuple(sorted((a, b))) in dups[s]
                E = np.stack([emb[r] / np.linalg.norm(emb[r]) for r in rids])
                C = E @ E.T
                np.fill_diagonal(C, -1)
                nn += C.max(1).tolist()
            stats["useful_share"].append(useful / len(sids))
            stats["dominated_share"].append(domn / (k * len(sids)))
            stats["conflict_pairs_per_scen"].append(float(np.mean(cps)))
            stats["dup_rate"].append(dp / npairs)
            stats["nn_cos"].append(float(np.mean(nn)))
            sep = separability(d)
            stats["pc1"].append(sep["pc1_share"])
            stats["cond"].append(min(sep["condition_number"], 1e4))
            n_eq = min(len(sids), BUDGET // k)
            rec_eq.append(recovery_suite(d, draws=150, seeds=(100 + rep,), n_scen=n_eq))
            rec_all.append(recovery_suite(d, draws=150, seeds=(200 + rep,)))

        def agg(recs):
            out = {}
            for cond in recs[0]:
                out[cond] = {m: {"mean": round(float(np.mean([r[cond][m]["mean"] for r in recs])), 3),
                                 "sd": round(float(np.std([r[cond][m]["mean"] for r in recs], ddof=1)), 3)}
                             for m in recs[0][cond]}
            return out
        n_eq = min(len(sids), BUDGET // k)
        results[k] = {
            **{m: {"mean": round(float(np.mean(v)), 3), "sd": round(float(np.std(v, ddof=1)), 3)} for m, v in stats.items()},
            "equal_budget": {"n_scen": n_eq, "responses": n_eq * k, "scoring_cost": round(n_eq * k * per_resp, 2), **agg(rec_eq)},
            "all_scenarios": {"n_scen": len(sids), "responses": len(sids) * k, "scoring_cost": round(len(sids) * k * per_resp, 2),
                              **agg(rec_all)},
        }
        r = results[k]
        print(f"k={k}: useful {r['useful_share']['mean']:.2f} dom {r['dominated_share']['mean']:.2f} "
              f"pairs/scen {r['conflict_pairs_per_scen']['mean']:.1f} dup {r['dup_rate']['mean']:.3f} nn_cos {r['nn_cos']['mean']:.3f} "
              f"pc1 {r['pc1']['mean']:.2f} | eqB n={n_eq}: sparse cross L1 {r['equal_budget']['sparse_cross']['l1']['mean']} "
              f"same L1 {r['equal_budget']['sparse_same']['l1']['mean']} dense cross rho {r['equal_budget']['dense_cross']['spearman']['mean']} "
              f"| all20: sparse cross L1 {r['all_scenarios']['sparse_cross']['l1']['mean']} same {r['all_scenarios']['sparse_same']['l1']['mean']}")
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "num_responses_sweep.json").write_text(json.dumps({"budget_responses": BUDGET, "repeats": REPEATS, "results": results}, indent=1))
    plot(results)


def plot(results):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ks = sorted(results, key=int)
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))
    for ax, (mode, title) in zip(axes[:2], [("equal_budget", f"Equal scoring budget ({BUDGET} responses)"),
                                             ("all_scenarios", "All 20 scenarios (cost grows with k)")]):
        for cond, color, ls in [("sparse_same", "#2a78d6", "-"), ("sparse_cross", "#2a78d6", "--"),
                                ("dense_same", "#eb6834", "-"), ("dense_cross", "#eb6834", "--")]:
            m = [results[k][mode][cond]["l1"]["mean"] for k in ks]
            s = [results[k][mode][cond]["l1"]["sd"] for k in ks]
            ax.errorbar([int(k) for k in ks], m, yerr=s, color=color, ls=ls, marker="o", capsize=3, label=cond.replace("_", ", "))
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("responses per scenario (k)")
        ax.set_ylabel("L1 error of recovered weights (lower is better)")
        ax.set_xticks([int(k) for k in ks])
    axes[0].legend(fontsize=8)
    ax = axes[2]
    ax.plot([int(k) for k in ks], [results[k]["useful_share"]["mean"] for k in ks], marker="o", color="#2a78d6", label="useful share")
    ax.plot([int(k) for k in ks], [results[k]["dominated_share"]["mean"] for k in ks], marker="o", color="#eb6834", label="dominated share")
    ax.plot([int(k) for k in ks], [results[k]["dup_rate"]["mean"] for k in ks], marker="o", color="#1baf7a", label="checker duplicate-pair rate")
    ax.set_title("Per-scenario structure", fontsize=10)
    ax.set_xlabel("k")
    ax.set_xticks([int(k) for k in ks])
    ax.legend(fontsize=8)
    fig.tight_layout()
    rep = HERE / "reports/figures"
    rep.mkdir(parents=True, exist_ok=True)
    fig.savefig(rep / "recovery_vs_num_responses.png", dpi=130)


if __name__ == "__main__":
    main()
