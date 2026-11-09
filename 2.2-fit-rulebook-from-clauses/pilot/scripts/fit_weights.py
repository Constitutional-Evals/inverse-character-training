"""Step 6: fit each target model's criterion weights from its choices, and test them on held-out scenarios.

Features: the main judge's score of response r on criterion c (mean over its samples), centred within the scenario and
divided by that criterion's spread over all scenarios, so criteria with wide or narrow judge scales are comparable.
A criterion not verified for a scenario contributes 0 (equal across its responses).

Models compared on held-out log loss per choice (5-fold cross-validation, folds by scenario):
    uniform      1/k
    position     letter-position bias only (plus the craft covariates, if given: then it is the "craft only" baseline)
    equal        one common weight on every criterion, plus position bias
    additive     softmax over responses of sum_c w_c z[s,r,c] + position bias, w >= 0, with the weights shrunk toward
                 their common mean by lambda * sum (w - mean)^2; lambda is chosen by cross-validation inside the training
                 data (nested), so a large lambda means the individual criteria add little beyond one common weight
    mixture      sum_c pi_c softmax_r(beta z[s,r,c]), pi on the simplex, beta fitted; a criterion not verified for the
                 scenario gives 1/k, so its share acts as a lapse rate
Full-data fits give the reported weights; a bootstrap over scenarios gives their 90% intervals.

    PYTHONPATH=scripts python scripts/fit_weights.py --scores S --criteria C --choices D1 D2 ... --out F
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.optimize import minimize
from scipy.special import logsumexp, softmax

LAMBDA, FOLDS, BOOT, KMAX = 0.1, 5, 200, 6
GRID = (0.3, 1, 3, 10, 30, 100, 300, 1000)  # shrinkage strengths tried for the additive model (summed-NLL scale)


def features(scores_path: Path, crit_ids: list[str]):
    raw = defaultdict(list)
    for row in map(json.loads, scores_path.open(encoding="utf-8")):
        if row["judge"] == "main" and row["score"] is not None:
            raw[(row["scenario_id"], row["response_id"], row["criterion_id"])].append(row["score"])
    per_s = defaultdict(lambda: defaultdict(dict))
    for (s, r, c), v in raw.items():
        per_s[s][c][r] = float(np.mean(v))
    centred = {s: {c: {r: v - np.mean(list(rs.values())) for r, v in rs.items()} for c, rs in cs.items()} for s, cs in per_s.items()}
    def spread(c):
        vals = [v for cs in centred.values() for v in cs.get(c, {}).values()]
        return float(np.std(vals)) if len(vals) > 1 and np.std(vals) > 0 else 1.0
    sd = {c: spread(c) for c in crit_ids}
    z = {}
    for s, cs in centred.items():
        rids = sorted({r for rs in cs.values() for r in rs})
        z[s] = {r: np.array([cs[c][r] / sd[c] if c in cs and r in cs[c] else 0.0 for c in crit_ids]) for r in rids}
    verified = {s: np.array([c in cs for c in crit_ids]) for s, cs in per_s.items()}
    return z, verified


NUISANCE: dict = {}  # (scenario, response) -> craft covariates, filled by load_nuisance; empty = none
N_NUIS = 0


def load_nuisance(responses: Path, leading: Path) -> list[str]:
    """Craft covariates per response, with free (unpenalised) coefficients: written by GPT-5.6 Terra (the writer the
    leading-option check favoured), and flagged as the clearly stronger reply of its scenario."""
    global N_NUIS
    lead = {(x["scenario_id"], x["leading"]) for x in map(json.loads, leading.open(encoding="utf-8")) if x.get("leading")}
    for r in map(json.loads, responses.open(encoding="utf-8")):
        if r["kept"]:
            key = (r["scenario_id"], r["response_id"])
            NUISANCE[key] = np.array([float(r["model"] == "openai/gpt-5.6-terra"), float(key in lead)])
    N_NUIS = 2
    return ["writer_gpt56_terra", "flagged_stronger"]


def build(choices: list[dict], z: dict, verified: dict):
    """Each usable choice -> (scenario, criterion features [k, C], verified mask, chosen index, craft covariates [k, N])."""
    items = []
    for ch in choices:
        if ch["choice"] is None or ch["scenario_id"] not in z:
            continue
        X = np.stack([z[ch["scenario_id"]][r] for r in ch["shown"]])
        Nm = np.stack([NUISANCE.get((ch["scenario_id"], r), np.zeros(N_NUIS)) for r in ch["shown"]]) if N_NUIS else np.zeros((len(X), 0))
        items.append((ch["scenario_id"], X, verified[ch["scenario_id"]], ch["shown"].index(ch["choice"]), Nm))
    return items


def unpack(theta, C, mode):
    """theta = [criterion part | craft coefficients (N_NUIS) | position bias (KMAX-1)]."""
    head = {"additive": C, "equal": 1, "position": 0}[mode]
    w = theta[:C] if mode == "additive" else np.full(C, theta[0]) if mode == "equal" else np.zeros(C)
    g = theta[head:head + N_NUIS]
    b = theta[head + N_NUIS:]
    return w, g, b


def stack(items):
    """Pad choices to KMAX responses: X [n,K,C], Nm [n,K,N], P one-hot positions [n,K,KMAX-1], valid [n,K], y [n]."""
    n, C = len(items), items[0][1].shape[1]
    X, Nm = np.zeros((n, KMAX, C)), np.zeros((n, KMAX, N_NUIS))
    P, valid, y = np.zeros((n, KMAX, KMAX - 1)), np.zeros((n, KMAX), bool), np.zeros(n, int)
    for i, (_, x, _, yi, nm) in enumerate(items):
        k = len(x)
        X[i, :k], Nm[i, :k], valid[i, :k], y[i] = x, nm, True, yi
        for j in range(1, k):
            P[i, j, j - 1] = 1.0
    return X, Nm, P, valid, y


def design(arr, C, mode):
    """Columns of the linear utility for each mode: criterion part, craft, position."""
    X, Nm, P, valid, y = arr
    head = X if mode == "additive" else X.sum(-1, keepdims=True) if mode == "equal" else X[..., :0]
    return np.concatenate([head, Nm, P], -1), valid, y


def nll_grad(theta, D, valid, y, pen, lam, to_mean):
    """Negative log-likelihood and gradient. Penalty on the criterion weights: lam * sum (w - mean w)^2 when to_mean
    (shrinks toward one common weight: lam -> inf is the equal-weight model), else lam * sum w^2."""
    u = D @ theta
    u = np.where(valid, u, -np.inf)
    lse = logsumexp(u, axis=1)
    p = np.exp(u - lse[:, None])
    idx = np.arange(len(y))
    w = theta[pen]
    dev = w - w.mean() if to_mean else w
    nll = float(-(u[idx, y] - lse).sum() + lam * (dev ** 2).sum())
    grad = -(D[idx, y] - np.einsum("nk,nkd->nd", p, D)).sum(0)
    grad[pen] += 2 * lam * dev
    return nll, grad


def fit_additive(items, C, mode="additive", arr=None, lam=LAMBDA):
    D, valid, y = design(arr if arr is not None else stack(items), C, mode)
    head = {"additive": C, "equal": 1, "position": 0}[mode]
    bounds = [(0, None)] * head + [(None, None)] * (N_NUIS + KMAX - 1)
    pen = np.zeros(len(bounds), bool)
    pen[:head] = True
    res = minimize(nll_grad, np.zeros(len(bounds)), args=(D, valid, y, pen, lam if mode == "additive" else 0.0,
                   mode == "additive"), jac=True, method="L-BFGS-B", bounds=bounds)
    return res.x


def folds(items, n, seed):
    scen = sorted({s for s, *_ in items})
    rng = np.random.default_rng(seed)
    rng.shuffle(scen)
    return {s: i % n for i, s in enumerate(scen)}


def choose_lambda(items, C, seed=2):
    """The shrinkage strength with the lowest cross-validated log loss on these items (folds by scenario)."""
    fold = folds(items, 4, seed)
    loss = {lam: 0.0 for lam in GRID}
    for f in range(4):
        train = [it for it in items if fold[it[0]] != f]
        test = [it for it in items if fold[it[0]] == f]
        for lam in GRID:
            th = fit_additive(train, C, "additive", lam=lam)
            loss[lam] -= sum(np.log(prob_additive(th, X, Nm, C, "additive")[y] + 1e-12) for _, X, _, y, Nm in test)
    return min(loss, key=loss.get)


def prob_additive(theta, X, Nm, C, mode):
    w, g, b = unpack(theta, C, mode)
    return softmax(X @ w + Nm @ g + np.concatenate([[0.0], b])[: len(X)])


def prob_mixture(theta, X, mask):
    pi, beta = softmax(theta[:-1]), np.exp(theta[-1])
    k = len(X)
    per = np.where(mask[None, :], softmax(beta * X, axis=0), 1.0 / k)  # [k, C]
    return per @ pi


def fit_mixture(items, C):
    X, _, _, valid, y = stack(items)
    mask = np.stack([m for _, _, m, _, _ in items])  # [n, C] verified criteria
    k = valid.sum(1)
    idx = np.arange(len(y))

    def nll(theta):
        pi, beta = softmax(theta[:-1]), np.exp(theta[-1])
        z = np.where(valid[..., None], beta * X, -np.inf)
        r = np.exp(z - logsumexp(z, axis=1, keepdims=True))          # [n, K, C]
        r = np.where(mask[:, None, :], r, (1.0 / k)[:, None, None])  # unverified criterion: uniform
        return float(-np.log(r[idx, y] @ pi + 1e-12).sum())

    best = None
    for start in (0.0, 1.0):
        res = minimize(nll, np.concatenate([np.zeros(C), [start]]), method="L-BFGS-B")
        best = res if best is None or res.fun < best.fun else best
    return best.x


def heldout(items, C):
    fold = folds(items, FOLDS, 0)
    loss = defaultdict(list)
    for f in range(FOLDS):
        train = [it for it in items if fold[it[0]] != f]
        test = [it for it in items if fold[it[0]] == f]
        if not test or not train:
            continue
        fits = {m: fit_additive(train, C, m) for m in ("position", "equal")}
        fits["additive"] = fit_additive(train, C, "additive", lam=choose_lambda(train, C))  # nested: lambda from train only
        mix = fit_mixture(train, C)
        for _, X, mask, y, Nm in test:
            loss["uniform"].append(np.log(len(X)))
            for m, th in fits.items():
                loss[m].append(-np.log(prob_additive(th, X, Nm, C, m)[y] + 1e-12))
            loss["mixture"].append(-np.log(prob_mixture(mix, X, mask)[y] + 1e-12))
    return {m: float(np.mean(v)) for m, v in loss.items()}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scores", type=Path, required=True)
    ap.add_argument("--criteria", type=Path, required=True)
    ap.add_argument("--choices", type=Path, nargs="+", required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--responses", type=Path, help="with --leading: add craft covariates (writer, flagged stronger)")
    ap.add_argument("--leading", type=Path)
    a = ap.parse_args()
    nuisance = load_nuisance(a.responses, a.leading) if a.responses and a.leading else []
    crit_ids = [json.loads(line)["criterion_id"] for line in a.criteria.open(encoding="utf-8")]
    C = len(crit_ids)
    z, verified = features(a.scores, crit_ids)
    results = {}
    for path in a.choices:
        choices = [json.loads(line) for line in path.open(encoding="utf-8")]
        model = choices[0]["model"]
        items = build(choices, z, verified)
        lam = choose_lambda(items, C)
        theta = fit_additive(items, C, lam=lam)
        mix = fit_mixture(items, C)
        scen = sorted({s for s, *_ in items})
        rng = np.random.default_rng(1)
        boots = []
        for _ in range(BOOT):
            pick = rng.choice(scen, len(scen))
            boots.append(fit_additive([it for s in pick for it in items if it[0] == s], C, lam=lam)[:C])
        lo, hi = np.percentile(boots, [5, 95], axis=0)
        results[model] = {
            "choices": len(items), "scenarios": len(scen), "heldout_logloss": heldout(items, C), "lambda": lam,
            "additive": {c: {"w": round(float(theta[i]), 3), "lo": round(float(lo[i]), 3), "hi": round(float(hi[i]), 3)}
                         for i, c in enumerate(crit_ids)},
            "craft": dict(zip(nuisance, [round(float(x), 3) for x in unpack(theta, C, "additive")[1]])),
            "position_bias": [round(float(b), 3) for b in unpack(theta, C, "additive")[2]],
            "mixture": {"pi": dict(zip(crit_ids, np.round(softmax(mix[:-1]), 3).tolist())), "beta": round(float(np.exp(mix[-1])), 3)},
        }
        print(model, json.dumps(results[model]["heldout_logloss"]))
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps({"lambda_grid": GRID, "folds": FOLDS, "bootstrap": BOOT, "criteria": crit_ids, "craft_covariates": nuisance,
                                 "models": results}, indent=1))


if __name__ == "__main__":
    main()
