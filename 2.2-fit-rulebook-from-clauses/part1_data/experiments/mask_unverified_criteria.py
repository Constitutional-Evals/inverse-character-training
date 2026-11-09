"""
Secondary (7.3, free on existing data): score only relevant criteria.

Masking: in the FIT ratings, every criterion not verified for a scenario is replaced by its
scenario mean, so it cannot separate that scenario's responses (as if the judge had said "not
applicable"). The simulated target still uses the full ratings of the other judge (its real
preferences are not masked). If irrelevant ratings are mostly halo noise, masking should improve
cross-judge recovery; if they carry real signal, it hurts.

    python experiments/mask_unverified_criteria.py D0 D2-half ...
"""

import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE / "experiments"))
from metrics import recovery_suite, simulate_batched, load_ratings, pad, softmax_masked, fit_simplex, draw_weights, recovery_metrics  # noqa
from compare_designs import DESIGNS, HALF_RIDS  # noqa: E402


def verified_map(resp_path, design):
    out = {}
    for l in open(resp_path):
        r = json.loads(l)
        v = set(r.get("verified_criteria") or [])
        if design == "D1":
            v |= set((r.get("design_meta") or {}).get("grader_verified_criteria") or [])
        out[r["scenario_id"]] = sorted(v)
    return out


def masked_cross(data, ver, kind, beta=10.0, choices=5, draws=200, seed=0, mask=True):
    rng = np.random.default_rng(seed)
    sids = list(data["scenarios"])
    K = data["K"]
    judges = data["judges"]
    res = []
    for a, b in [(judges[0], judges[1]), (judges[1], judges[0])]:
        W = draw_weights(rng, draws, K, kind)
        Ts, M = pad([data["scenarios"][s]["X"][a] for s in sids])
        fits = []
        for s in sids:
            X = data["scenarios"][s]["X"][b].copy()
            if mask:
                keep = [c - 1 for c in ver.get(s, [])]
                for c in range(K):
                    if c not in keep:
                        X[:, c] = X[:, c].mean()
            fits.append(X)
        Tf, _ = pad(fits)
        P = softmax_masked(beta * np.einsum("sak,dk->dsa", Ts, W), M[None])
        counts = np.zeros_like(P)
        for d in range(draws):
            for i in range(len(sids)):
                m = M[i]
                counts[d, i, m] = rng.multinomial(choices, P[d, i, m] / P[d, i, m].sum())
        Wh = fit_simplex(Tf, M, counts, beta)
        l1, rho, top = recovery_metrics(Wh, W)
        res.append((l1.mean(), rho.mean(), top.mean()))
    return np.mean(res, 0)


def main(names):
    out = {}
    for name in names:
        cfg = DESIGNS[name]
        rp = cfg["ratings"] / "ratings.jsonl"
        subset = None
        if cfg.get("half"):
            subset = {}
            for l in open(rp):
                r = json.loads(l)
                if r["response_id"].split(":")[1] in HALF_RIDS:
                    subset.setdefault(r["scenario_id"], set()).add(r["response_id"])
        data = load_ratings(rp, subset=subset)
        ver = verified_map(cfg["responses"] / "responses.jsonl", name)
        row = {}
        for kind in ("sparse", "dense"):
            for mask in (False, True):
                vals = np.array([masked_cross(data, ver, kind, seed=s, mask=mask) for s in range(3)])
                row[f"{kind}_{'masked' if mask else 'full'}"] = {"l1": round(float(vals[:, 0].mean()), 3),
                                                                 "spearman": round(float(vals[:, 1].mean()), 3),
                                                                 "top3": round(float(vals[:, 2].mean()), 3),
                                                                 "l1_sd_over_seeds": round(float(vals[:, 0].std(ddof=1)), 3)}
        row["mean_verified_per_scenario"] = round(float(np.mean([len(ver.get(s, [])) for s in data["scenarios"]])), 2)
        out[name] = row
        print(name, json.dumps(row))
    p = HERE / "outputs/analysis/mask_unverified_criteria.json"
    old = json.loads(p.read_text()) if p.exists() else {}
    old.update(out)
    p.write_text(json.dumps(old, indent=1))


if __name__ == "__main__":
    main(sys.argv[1:] or ["D0"])
