"""
Shared metrics for the design comparison: usefulness, separability and simulated
weight recovery, computed the same way for every design.

A "dataset" here is a list of scenarios; each scenario has response ids and, per judge, a
k x K matrix of ratings in [0,1] (K = 15 criteria). Only committed responses are used unless
asked otherwise (the natural reply is longer and would confound length with criteria).

The recovery simulation is the main yardstick and needs no target model:
  plant weights w* on the simplex -> simulate choices ~ softmax(beta * X_sim w*) per scenario
  -> fit w by maximum likelihood on the simplex with X_fit -> compare w to w*.
X_sim and X_fit are the same judge's ratings (same-judge) or different judges' (cross-judge, the
realistic case: the target's real preferences are not exactly what our judge measures).
"""

import json
from itertools import combinations, permutations
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

TOL = 1 / 9  # one point on the 1-10 scale counts as a tie (same as judge.py)


# ---------------------------------------------------------------- loading

def load_ratings(ratings_path, kinds=("committed",), subset=None):
    """Returns {"judges": [...], "scenarios": {sid: {"rids": [...], "X": {judge: k x K array}}}}.
    subset: optional {sid: [rids]} to keep only some responses (k-subsampling, half fractions)."""
    rows = [json.loads(l) for l in open(ratings_path) if l.strip()]
    rows = [r for r in rows if r["kind"] in kinds and r.get("repeat", 0) == 0]
    judges = sorted({r["judge"] for r in rows})
    K = len(rows[0]["ratings"])
    cells = {}
    for r in rows:
        cells.setdefault(r["scenario_id"], {}).setdefault(r["response_id"], {})[r["judge"]] = \
            np.array([r["ratings"][str(c)] for c in range(1, K + 1)])
    scen = {}
    for sid, resp in sorted(cells.items()):
        rids = sorted(rid for rid, by in resp.items() if all(j in by for j in judges))
        if subset is not None:
            if sid not in subset:
                continue
            rids = [r for r in rids if r in set(subset[sid])]
        if len(rids) < 2:
            continue
        scen[sid] = {"rids": rids, "X": {j: np.stack([resp[r][j] for r in rids]) for j in judges}}
    return {"judges": judges, "K": K, "scenarios": scen}


def mean_X(s, judges):
    return np.mean([s["X"][j] for j in judges], axis=0)


# ---------------------------------------------------------------- usefulness

def dominated(X, tol=TOL):
    out = []
    for a in range(len(X)):
        for b in range(len(X)):
            if a != b and np.all(X[b] >= X[a] - tol) and np.any(X[b] > X[a] + tol):
                out.append(a)
                break
    return out


def conflict_pairs(X, tol=TOL):
    """Criterion pairs (c1, c2) that rank some pair of responses in opposite orders."""
    K = X.shape[1]
    found = set()
    for a, b in permutations(range(len(X)), 2):
        d = X[a] - X[b]
        pos, neg = np.where(d > tol)[0], np.where(d < -tol)[0]
        for c1 in pos:
            for c2 in neg:
                found.add((min(c1, c2), max(c1, c2)))
    return found


def usefulness(data, tol=TOL):
    judges = data["judges"]
    K = data["K"]
    n_useful, pairs, sep_count = 0, set(), np.zeros(K, int)
    dom_share, n_resp = 0, 0
    per = {}
    for sid, s in data["scenarios"].items():
        X = mean_X(s, judges)
        dom = dominated(X, tol)
        useful = len(X) - len(dom) >= 2
        n_useful += useful
        cp = conflict_pairs(X, tol)
        pairs |= cp
        sep_count += (X.max(0) - X.min(0)) >= 3 / 9 - 1e-9
        dom_share += len(dom)
        n_resp += len(X)
        per[sid] = {"useful": bool(useful), "dominated": len(dom), "k": len(X), "conflict_pairs": len(cp)}
    n = len(data["scenarios"])
    return {
        "scenarios": n, "responses": n_resp,
        "useful": n_useful, "useful_share": n_useful / n if n else None,
        "pair_coverage": len(pairs), "pair_coverage_share": len(pairs) / (K * (K - 1) / 2),
        "criterion_coverage": sep_count.tolist(),          # scenarios where criterion spread >= 3 points
        "criteria_separating_ge3": int((sep_count >= 3).sum()),
        "dominated_share": dom_share / n_resp if n_resp else None,
        "per_scenario": per,
    }


# ---------------------------------------------------------------- separability

def centred_stack(data, judges=None):
    judges = judges or data["judges"]
    rows = []
    for s in data["scenarios"].values():
        X = mean_X(s, judges)
        rows.append(X - X.mean(0))
    return np.vstack(rows)


def separability(data, judges=None, top=5):
    D = centred_stack(data, judges)
    sv = np.linalg.svd(D, compute_uv=False)
    rank = int((sv > 1e-6 * sv[0]).sum())
    var = sv ** 2
    with np.errstate(invalid="ignore", divide="ignore"):
        C = np.corrcoef(D.T)
    K = D.shape[1]
    pairs = sorted(((C[i, j], i + 1, j + 1) for i, j in combinations(range(K), 2) if np.isfinite(C[i, j])),
                   key=lambda t: -abs(t[0]))
    off = np.array([abs(c) for c, _, _ in pairs])
    return {
        "rows": D.shape[0], "rank": rank, "condition_number": float(sv[0] / sv[-1]) if sv[-1] > 0 else float("inf"),
        "pc1_share": float(var[0] / var.sum()),
        "mean_abs_corr": float(off.mean()), "pairs_abs_corr_gt_0.7": int((off > 0.7).sum()),
        "most_correlated": [(round(float(c), 2), i, j) for c, i, j in pairs[:top]],
        "column_sd": D.std(0).round(3).tolist(),
    }


# ---------------------------------------------------------------- recovery simulation

def proj_simplex(V):
    """Euclidean projection of each row of V onto the probability simplex."""
    n = V.shape[1]
    U = -np.sort(-V, axis=1)
    css = np.cumsum(U, axis=1) - 1
    idx = np.arange(1, n + 1)
    cond = U - css / idx > 0
    rho = n - 1 - np.argmax(cond[:, ::-1], axis=1)
    theta = css[np.arange(len(V)), rho] / (rho + 1)
    return np.maximum(V - theta[:, None], 0)


def pad(Xs):
    kmax = max(len(X) for X in Xs)
    K = Xs[0].shape[1]
    T = np.zeros((len(Xs), kmax, K))
    M = np.zeros((len(Xs), kmax), bool)
    for i, X in enumerate(Xs):
        T[i, :len(X)] = X
        M[i, :len(X)] = True
    return T, M


def softmax_masked(Z, M):
    Z = np.where(M, Z, -np.inf)
    Z = Z - Z.max(-1, keepdims=True)
    E = np.exp(Z) * M
    return E / E.sum(-1, keepdims=True)


def fit_simplex(T, M, counts, beta, iters=1500):
    """MLE of w on the simplex for D independent datasets at once.
    T: S x k x K ratings, M: S x k mask, counts: D x S x k choice counts (or probabilities x N).
    Minimises sum_s sum_a -counts * log softmax(beta T_s w)_a with FISTA + simplex projection."""
    D = counts.shape[0]
    K = T.shape[2]
    Tc = T - (T * M[..., None]).sum(1, keepdims=True) / M.sum(1, keepdims=True)[..., None]
    Tc = Tc * M[..., None]
    N = counts.sum(-1)  # D x S
    # Lipschitz bound: Hessian <= beta^2/2 * sum_s N_s * sigma_max(Tc_s)^2
    smax2 = np.array([np.linalg.norm(Tc[s], 2) ** 2 for s in range(len(Tc))])
    L = beta ** 2 / 2 * (N * smax2[None]).sum(1) + 1e-9  # D
    w = np.full((D, K), 1 / K)
    y, t = w.copy(), 1.0
    for _ in range(iters):
        Z = beta * np.einsum("sak,dk->dsa", Tc, y)
        Q = softmax_masked(Z, M[None])
        G = beta * np.einsum("dsa,sak->dk", N[..., None] * Q - counts, Tc)
        w_new = proj_simplex(y - G / L[:, None])
        t_new = (1 + np.sqrt(1 + 4 * t * t)) / 2
        y = w_new + (t - 1) / t_new * (w_new - w)
        w, t = w_new, t_new
    return w


def draw_weights(rng, D, K, kind):
    if kind == "dense":
        return rng.dirichlet(np.ones(K), size=D)
    W = np.zeros((D, K))
    for d in range(D):
        idx = rng.choice(K, 3, replace=False)
        W[d, idx] = rng.dirichlet(np.ones(3))
    return W


def recovery_metrics(W_hat, W_star):
    l1 = np.abs(W_hat - W_star).sum(1)
    rho = []
    for a, b in zip(W_hat, W_star):
        r = spearmanr(a, b).correlation
        rho.append(0.0 if not np.isfinite(r) else r)
    top = [len(set(np.argsort(-a)[:3]) & set(np.argsort(-b)[:3])) / 3 for a, b in zip(W_hat, W_star)]
    return np.array(l1), np.array(rho), np.array(top)


def simulate_batched(data, sim_judge, fit_judge, *, kind="sparse", beta=10.0, choices=5, draws=200,
                     seed=0, infinite=False, sids=None):
    """Faster version when all draws use the same scenario set: one batched fit."""
    rng = np.random.default_rng(seed)
    sids = sids or list(data["scenarios"])
    K = data["K"]
    W_star = draw_weights(rng, draws, K, kind)
    Ts, M = pad([data["scenarios"][s]["X"][sim_judge] for s in sids])
    Tf, _ = pad([data["scenarios"][s]["X"][fit_judge] for s in sids])
    P = softmax_masked(beta * np.einsum("sak,dk->dsa", Ts, W_star), M[None])  # D x S x k
    if infinite:
        counts = P
    else:
        counts = np.zeros_like(P)
        for d in range(draws):
            for i in range(len(sids)):
                m = M[i]
                counts[d, i, m] = rng.multinomial(choices, P[d, i, m] / P[d, i, m].sum())
    W = fit_simplex(Tf, M, counts, beta)
    l1, rho, top = recovery_metrics(W, W_star)
    return {"l1": l1, "spearman": rho, "top3": top, "l1_equal": np.abs(1 / K - W_star).sum(1)}


def summarize(res):
    out = {}
    for k, v in res.items():
        out[k] = round(float(v.mean()), 3)
        out[k + "_se"] = round(float(v.std(ddof=1) / np.sqrt(len(v))), 3)
    return out


def recovery_suite(data, *, beta=10.0, choices=5, draws=200, seeds=(0, 1, 2), n_scen=None, kinds=("sparse", "dense"),
                   infinite=False):
    """Same-judge (each judge simulates and fits) and cross-judge (each direction), averaged over
    directions. With n_scen, each seed uses a different random subset of n_scen scenarios.
    Returns mean and spread over seeds (each seed = different subset, weights and choices)."""
    judges = data["judges"]
    sids_all = list(data["scenarios"])
    out = {}
    for kind in kinds:
        for mode in ("same", "cross"):
            if mode == "cross" and len(judges) < 2:
                continue
            pairs = [(j, j) for j in judges] if mode == "same" else [(a, b) for a, b in permutations(judges, 2)]
            per_seed = []
            for seed in seeds:
                rng = np.random.default_rng(1000 + seed)
                sids = sids_all if not n_scen or n_scen >= len(sids_all) else list(rng.choice(sids_all, n_scen, replace=False))
                accum = {}
                for a, b in pairs:
                    r = simulate_batched(data, a, b, kind=kind, beta=beta, choices=choices, draws=draws,
                                         seed=seed, infinite=infinite, sids=sids)
                    for k, v in r.items():
                        accum.setdefault(k, []).append(v.mean())
                per_seed.append({k: float(np.mean(v)) for k, v in accum.items()})
            out[f"{kind}_{mode}"] = {
                k: {"mean": round(float(np.mean([p[k] for p in per_seed])), 3),
                    "sd_over_seeds": round(float(np.std([p[k] for p in per_seed], ddof=1)), 3) if len(seeds) > 1 else None}
                for k in per_seed[0]
            }
    return out


def full_report(data, **kw):
    return {"usefulness": {k: v for k, v in usefulness(data).items() if k != "per_scenario"},
            "separability": separability(data),
            "recovery": recovery_suite(data, **kw)}


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("ratings")
    ap.add_argument("--n-scen", type=int, default=None)
    ap.add_argument("--choices", type=int, default=5)
    ap.add_argument("--beta", type=float, default=10.0)
    ap.add_argument("--infinite", action="store_true")
    a = ap.parse_args()
    d = load_ratings(a.ratings)
    print(json.dumps(full_report(d, n_scen=a.n_scen, choices=a.choices, beta=a.beta, infinite=a.infinite), indent=1))
