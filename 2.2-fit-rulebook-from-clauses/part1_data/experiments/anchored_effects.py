"""
Anchored extremes (DA): does each reply win on its own anchor criterion, or does the judge's halo
pick one reply as best on everything? No API calls.

    python experiments/anchored_effects.py      # writes outputs/analysis/anchored_effects.json

Per reply, on its anchor criterion (ratings averaged over judges, 0-1 scale):
- hit:    rated strictly highest among the scenario's 5 replies (ties within one 1-10 point count as no hit)
- margin: anchor reply's rating minus the mean of the other replies on that criterion
Per scenario:
- halo winner share: for how many of the 5 anchor criteria the same reply comes out on top
Also per judge, since a hit for one judge and not the other means the anchoring is in the eye of the judge.
"""
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parents[1]
TOL = 1 / 9


def per_judge_matrix(ratings_path):
    m = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))  # judge -> sid -> rid -> list of rating dicts
    for line in open(ratings_path):
        r = json.loads(line)
        m[r["judge"]][r["scenario_id"]][r["response_id"]].append(r["ratings"])
    return {j: {s: {rid: {c: float(np.mean([x[c] for x in v])) for c in v[0]} for rid, v in rr.items()} for s, rr in ss.items()}
            for j, ss in m.items()}


def anchor_stats(scen_ratings, anchors):
    """scen_ratings: sid -> rid -> {criterion: rating}; anchors: sid -> rid -> (criterion, verified)."""
    rows, halo = [], []
    for sid, rr in scen_ratings.items():
        if sid not in anchors:
            continue
        rids = [rid for rid in rr if rid in anchors[sid]]
        tops = []
        for rid in rids:
            c, ver = anchors[sid][rid]
            mine = rr[rid][str(c)]
            others = [rr[o][str(c)] for o in rids if o != rid]
            rows.append({"hit": mine > max(others) + TOL / 2, "margin": mine - float(np.mean(others)), "verified": ver})
            tops.append(max(rids, key=lambda o: rr[o][str(c)]))
        if tops:
            halo.append(max(tops.count(t) for t in set(tops)) / len(tops))

    def agg(sel):
        return {"n": len(sel), "hit_rate": round(float(np.mean([r["hit"] for r in sel])), 3) if sel else None,
                "mean_margin": round(float(np.mean([r["margin"] for r in sel])), 3) if sel else None}
    return {"all": agg(rows), "verified_anchor": agg([r for r in rows if r["verified"]]),
            "filled_anchor": agg([r for r in rows if not r["verified"]]),
            "halo_winner_share": round(float(np.mean(halo)), 3) if halo else None,
            "chance_hit_rate": 0.2}


def main():
    resp = [json.loads(l) for l in open(HERE / "outputs/responses/exp_anchored/responses.jsonl")]
    anchors = {r["scenario_id"]: {x["response_id"]: (x["anchor"], x["anchor_verified"]) for x in r["responses"]} for r in resp}
    mat = json.loads((HERE / "outputs/ratings/exp_anchored/ratings_matrix.json").read_text())
    mean = {sid: {rid: v["ratings"] for rid, v in s["responses"].items()} for sid, s in mat["scenarios"].items()}
    out = {"mean_of_judges": anchor_stats(mean, anchors)}
    for judge, m in per_judge_matrix(HERE / "outputs/ratings/exp_anchored/ratings.jsonl").items():
        out[judge] = anchor_stats(m, anchors)
    p = HERE / "outputs/analysis/anchored_effects.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
