"""
D2 only: does flipping one move change the criteria it was designed for, or everything (halo)?

For each scenario, move m and criterion c, the main effect is
  mean rating over the 4 cells with side B  -  mean over the 4 cells with side A.
A "targeted" criterion is one the planner listed for a side (A_favoured_by: expected negative effect;
B_favoured_by: expected positive). Reports the sign-hit rate on targeted criteria (the factorial
analogue of the design hit rate), mean |effect| on targeted vs untargeted criteria, and, per judge,
how often the same move moves many untargeted criteria in the same direction (halo).

    python experiments/d2_move_effects.py
"""

import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE / "experiments"))
from metrics import load_ratings  # noqa: E402

CELLS = ["000", "001", "010", "011", "100", "101", "110", "111"]


def main():
    data = load_ratings(HERE / "outputs/ratings/exp_D2/ratings.jsonl")
    rows = {json.loads(l)["scenario_id"]: json.loads(l) for l in open(HERE / "outputs/responses/exp_D2/responses.jsonl")}
    out = {}
    for judge in data["judges"] + ["mean"]:
        hits, total, tgt, untgt, halo_share = 0, 0, [], [], []
        for sid, s in data["scenarios"].items():
            X = np.mean([s["X"][j] for j in data["judges"]], 0) if judge == "mean" else s["X"][judge]
            cell_of = {rid: CELLS[int(rid.split(":r")[1]) - 1] for rid in s["rids"]}
            moves = rows[sid]["design_meta"]["moves"]
            for m, mv in enumerate(moves):
                b = np.array([cell_of[r][m] == "1" for r in s["rids"]])
                eff = X[b].mean(0) - X[~b].mean(0)
                targeted = {c - 1: +1 for c in mv["B_favoured_by"]}
                for c in mv["A_favoured_by"]:
                    targeted[c - 1] = -1 if c - 1 not in targeted else 0  # listed on both sides: no expectation
                for c, sign in targeted.items():
                    if sign:
                        total += 1
                        hits += np.sign(eff[c]) == sign and abs(eff[c]) > 0.5 / 9
                        tgt.append(abs(eff[c]))
                others = [c for c in range(len(eff)) if c not in targeted]
                untgt += [abs(eff[c]) for c in others]
                big = [np.sign(eff[c]) for c in others if abs(eff[c]) > 1 / 9]
                if big:
                    halo_share.append(max(sum(x > 0 for x in big), sum(x < 0 for x in big)) / len(others))
                else:
                    halo_share.append(0.0)
        out[judge] = {"sign_hit_rate": round(hits / total, 3), "checks": total,
                      "mean_abs_effect_targeted_1to10": round(float(np.mean(tgt)) * 9, 2),
                      "mean_abs_effect_untargeted_1to10": round(float(np.mean(untgt)) * 9, 2),
                      "halo_share_untargeted_moving_together": round(float(np.mean(halo_share)), 3)}
    print(json.dumps(out, indent=1))
    (HERE / "outputs/analysis").mkdir(parents=True, exist_ok=True)
    (HERE / "outputs/analysis/d2_move_effects.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
