"""Pilot fit with the weights constrained to sum to 1 (the original formulation): u = BETA * sum_c w_c z_c + craft +
position, w on the simplex, BETA fixed. Per model: weights on all scenarios, held-out log loss (5-fold by scenario,
same folds as the free fit) and split-half agreement of the ranking (Spearman between weights fitted on two random
halves, over criteria seen in both halves, 50 splits). Heatmap of the weights (rows sum to 1) with both checks.

    PYTHONPATH=scripts python figures/gen_fig_pilot_simplex.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize
from scipy.special import logsumexp, softmax
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parent))
import fit_weights as fw  # noqa: E402
from gen_fig_pilot_fit import get, jsonl_gz  # noqa: E402
from gen_fig_pilot_ranking import NAMES  # noqa: E402

OUT = Path(__file__).resolve().parent
BETA, SPLITS = 10.0, 50
FILES = {"x-ai/grok-4.7": "grok-4.7", "z-ai/glm-5.3": "glm-5.3", "qwen2.5-7b-it": "qwen2.5-7b-it-base"}


def fit_simplex(items, C):
    X, Nm, P, valid, y = fw.stack(items)
    idx = np.arange(len(y))

    def nll_grad(th):
        w = softmax(th[:C])
        u = BETA * (X @ w) + Nm @ th[C:C + fw.N_NUIS] + P @ th[C + fw.N_NUIS:]
        u = np.where(valid, u, -np.inf)
        lse = logsumexp(u, axis=1)
        p = np.exp(u - lse[:, None])
        resid = lambda F: F[idx, y] - np.einsum("nk,nkd->nd", p, F)  # observed minus expected features
        gw = -BETA * resid(X).sum(0)
        g_theta = w * (gw - w @ gw)  # chain rule through softmax
        grad = np.concatenate([g_theta, -resid(Nm).sum(0), -resid(P).sum(0)])
        return float(-(u[idx, y] - lse).sum()), grad

    th = minimize(nll_grad, np.zeros(C + fw.N_NUIS + fw.KMAX - 1), jac=True, method="L-BFGS-B").x
    return th


def prob(th, X, Nm, C):
    return softmax(BETA * (X @ softmax(th[:C])) + Nm @ th[C:C + fw.N_NUIS]
                   + np.concatenate([[0.0], th[C + fw.N_NUIS:]])[: len(X)])


def main() -> None:
    fit = json.loads(get("20261106-fit-pilot15", "fit_craft.json").read_text())
    crit, free = fit["criteria"], fit["models"]
    C = len(crit)
    tmp = OUT / ".cache"
    tmp.mkdir(exist_ok=True)
    for run, name in (("20261106-ratings-pilot15", "scores.jsonl"), ("20261106-ratings-pilot15", "leading.jsonl"),
                      ("20261106-responses-pilot15", "responses.jsonl")):
        (tmp / name).write_text("\n".join(json.dumps(r) for r in jsonl_gz(get(run, name + ".gz"))) + "\n")
    z, ver = fw.features(tmp / "scores.jsonl", crit)
    fw.load_nuisance(tmp / "responses.jsonl", tmp / "leading.jsonl")
    seen = {s: set(np.where(m)[0]) for s, m in ver.items()}

    res = {}
    for m in NAMES:
        items = fw.build(jsonl_gz(get("20261106-choices-pilot15",
                                      f"choices_{FILES.get(m, 'qwen2.5-7b-it-' + m.split('+')[-1])}.jsonl.gz")), z, ver)
        w = softmax(fit_simplex(items, C)[:C])
        fold = fw.folds(items, fw.FOLDS, 0)
        ll = []
        for f in range(fw.FOLDS):
            th = fit_simplex([it for it in items if fold[it[0]] != f], C)
            ll += [-np.log(prob(th, X, Nm, C)[y] + 1e-12) for s, X, _, y, Nm in items if fold[s] == f]
        scen = sorted({s for s, *_ in items})
        rng = np.random.default_rng(0)
        rs = []
        for _ in range(SPLITS):
            perm = rng.permutation(scen)
            A, B = set(perm[: len(scen) // 2]), set(perm[len(scen) // 2:])
            wa = softmax(fit_simplex([it for it in items if it[0] in A], C)[:C])
            wb = softmax(fit_simplex([it for it in items if it[0] in B], C)[:C])
            both = sorted(set.union(*[seen[s] for s in A]) & set.union(*[seen[s] for s in B]))
            r = spearmanr(wa[both], wb[both]).statistic
            if np.isfinite(r):
                rs.append(r)
        res[m] = {"weights": dict(zip(crit, np.round(w, 4).tolist())), "heldout": float(np.mean(ll)),
                  "heldout_free": free[m]["heldout_logloss"]["additive"], "heldout_common": free[m]["heldout_logloss"]["equal"],
                  "chance": free[m]["heldout_logloss"]["uniform"],
                  "split_half": [float(np.mean(rs)), float(np.percentile(rs, 5)), float(np.percentile(rs, 95))]}
        print(f"{m:28} held-out {res[m]['heldout']:.3f} (free {res[m]['heldout_free']:.3f}, common {res[m]['heldout_common']:.3f}, "
              f"chance {res[m]['chance']:.3f})  split-half r {np.mean(rs):+.2f} [{np.percentile(rs, 5):+.2f}, {np.percentile(rs, 95):+.2f}]",
              flush=True)
    (OUT / "fig_pilot_simplex.json").write_text(json.dumps({"beta": BETA, "splits": SPLITS, "models": res}, indent=1))

    order_m = list(NAMES)
    W = np.array([[res[m]["weights"][c] for c in crit] for m in order_m])
    oc = np.argsort(-W.mean(0))
    W = W[:, oc]
    plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"], "font.size": 9,
                         "figure.dpi": 300, "savefig.dpi": 300, "savefig.bbox": "tight"})
    fig, ax = plt.subplots(figsize=(7.6, 3.3))
    im = ax.imshow(W, cmap="cividis", aspect="auto", vmin=0)
    ax.set_xticks(range(C), [crit[i].replace("-", " ").replace("#", " #") for i in oc], rotation=55, ha="right", fontsize=7.5)
    ax.set_yticks(range(len(order_m)), [NAMES[m] for m in order_m])
    for i in range(W.shape[0]):
        for j in range(W.shape[1]):
            ax.text(j, i, f"{W[i, j]:.2f}", ha="center", va="center", fontsize=5.5,
                    color="black" if W[i, j] > 0.6 * W.max() else "white")
    ax.tick_params(length=0)
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.text(C - 0.2, -0.85, "  held-out  split-half r", fontsize=7, va="center", ha="left", clip_on=False)
    for i, m in enumerate(order_m):
        r = res[m]["split_half"][0]
        ax.text(C - 0.2, i, f"  {res[m]['heldout']:.2f}        {r:+.2f}", fontsize=7, va="center", ha="left", clip_on=False)
    cb = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.16, location="right")
    cb.set_label("weight (row sums to 1)", fontsize=8)
    cb.ax.tick_params(labelsize=7)
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"fig_pilot_simplex.{ext}")


if __name__ == "__main__":
    main()
