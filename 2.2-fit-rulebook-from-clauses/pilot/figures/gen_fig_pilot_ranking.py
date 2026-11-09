"""Per-criterion weights per model from the pilot fit (run 20261106-fit-pilot15, fit_craft.json), as a heatmap.

Left: each model's priorities, the additive model's criterion weights divided by their sum (each row sums to 1; the
weights are shrunk toward the model's common weight by the cross-validated lambda shown with its name). Right: the sum,
how strongly the model's choices follow the criteria at all. Columns are ordered by mean share. Intervals (table) are
90% over 500 bootstrap refits on resampled scenarios at the model's lambda, each normalised. Also writes the table
(fig_pilot_ranking.md) used on the Results page.

    PYTHONPATH=scripts python figures/gen_fig_pilot_ranking.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))

OUT = Path(__file__).resolve().parent
NAMES = {"x-ai/grok-4.7": "Grok 4.7", "z-ai/glm-5.3": "GLM 5.3", "qwen2.5-7b-it": "Qwen2.5-7B",
         "qwen2.5-7b-it+goodness": "+ goodness", "qwen2.5-7b-it+loving": "+ loving",
         "qwen2.5-7b-it+sycophancy": "+ sycophancy", "qwen2.5-7b-it+impulsiveness": "+ impulsiveness",
         "qwen2.5-7b-it+misalignment": "+ misalignment"}


BOOT = 500


def shares(w):
    t = float(np.sum(w))
    return np.asarray(w) / t if t > 0.05 else np.full(len(w), np.nan)  # below this the model puts ~no weight on criteria


def main() -> None:
    import fit_weights as fw
    from gen_fig_pilot_fit import get, jsonl_gz

    fit = json.loads(get("20261106-fit-pilot15", "fit_craft.json").read_text())
    crit, models = fit["criteria"], fit["models"]
    tmp = OUT / ".cache"
    tmp.mkdir(exist_ok=True)
    for run, name in (("20261106-ratings-pilot15", "scores.jsonl"), ("20261106-ratings-pilot15", "leading.jsonl"),
                      ("20261106-responses-pilot15", "responses.jsonl")):
        (tmp / name).write_text("\n".join(json.dumps(r) for r in jsonl_gz(get(run, name + ".gz"))) + "\n")
    z, ver = fw.features(tmp / "scores.jsonl", crit)
    fw.load_nuisance(tmp / "responses.jsonl", tmp / "leading.jsonl")
    files = {"x-ai/grok-4.7": "grok-4.7", "z-ai/glm-5.3": "glm-5.3", "qwen2.5-7b-it": "qwen2.5-7b-it-base"}

    order_m = list(NAMES)
    point, lo, hi, total, total_ci = [], [], [], [], []
    for m in order_m:
        w = np.array([models[m]["additive"][c]["w"] for c in crit])
        stem = files.get(m, "qwen2.5-7b-it-" + m.split("+")[-1])
        items = fw.build(jsonl_gz(get("20261106-choices-pilot15", f"choices_{stem}.jsonl.gz")), z, ver)
        scen = sorted({s for s, *_ in items})
        by = {s: [it for it in items if it[0] == s] for s in scen}
        rng = np.random.default_rng(0)
        boot = [fw.fit_additive([it for s in rng.choice(scen, len(scen)) for it in by[s]], len(crit),
                                lam=models[m]["lambda"])[:len(crit)] for _ in range(BOOT)]
        sh = np.array([shares(b) for b in boot])
        point.append(shares(w))
        lo.append(np.nanpercentile(sh, 5, axis=0) if np.isfinite(sh).any() else np.full(len(crit), np.nan))
        hi.append(np.nanpercentile(sh, 95, axis=0) if np.isfinite(sh).any() else np.full(len(crit), np.nan))
        total.append(float(w.sum()))
        total_ci.append(np.percentile([b.sum() for b in boot], [5, 95]))
    P, LO, HI = map(np.array, (point, lo, hi))
    order_c = np.argsort(-np.nanmean(P, 0))
    P, LO, HI = P[:, order_c], LO[:, order_c], HI[:, order_c]
    labels = [crit[i] for i in order_c]

    plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"], "font.size": 9,
                         "figure.dpi": 300, "savefig.dpi": 300, "savefig.bbox": "tight", "axes.spines.top": False,
                         "axes.spines.right": False})
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(7.6, 3.3), sharey=True,
                                 gridspec_kw={"width_ratios": [5.2, 1], "wspace": 0.04})
    cmap = plt.get_cmap("cividis").copy()
    cmap.set_bad("#E6E6E6")
    vmax = np.nanmax(P)
    im = ax.imshow(np.ma.masked_invalid(P), cmap=cmap, aspect="auto", vmin=0, vmax=vmax)
    ax.set_xticks(range(len(labels)), [c.replace("-", " ").replace("#", " #") for c in labels], rotation=55, ha="right", fontsize=7.5)
    ax.set_yticks(range(len(order_m)), [f"{NAMES[m]}  (λ={models[m]['lambda']:g})" for m in order_m])
    for i in range(P.shape[0]):
        for j in range(P.shape[1]):
            v = P[i, j]
            ax.text(j, i, "–" if np.isnan(v) else f"{v:.2f}", ha="center", va="center", fontsize=5.5,
                    color="#555" if np.isnan(v) else ("black" if v > 0.6 * vmax else "white"))
    ax.tick_params(length=0)
    for sp in ax.spines.values():
        sp.set_visible(False)
    cb = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.01, location="top", shrink=0.45, anchor=(0.0, 0.0))
    cb.set_label("share of the model's weight (row sums to 1)", fontsize=7.5)
    cb.ax.tick_params(labelsize=6.5)
    y = np.arange(len(order_m))
    tci = np.array(total_ci)
    bx.barh(y, total, color="#0072B2", height=0.55)
    bx.errorbar(total, y, xerr=[np.array(total) - tci[:, 0], tci[:, 1] - np.array(total)], fmt="none", color="#333",
                elinewidth=0.8, capsize=1.5)
    bx.set_xlabel("sum of weights", fontsize=8)
    bx.tick_params(axis="y", length=0)
    bx.tick_params(axis="x", labelsize=7)
    bx.grid(axis="x", alpha=0.15)
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"fig_pilot_ranking.{ext}")

    rows = ["| Model | Sum of weights [90%] | Largest shares | Smallest shares | Largest share above smallest? |",
            "|---|---|---|---|---|"]
    for i, m in enumerate(order_m):
        if np.isnan(P[i]).all():
            rows.append(f"| {NAMES[m]} | {total[i]:.2f} [{tci[i, 0]:.2f}, {tci[i, 1]:.2f}] | – | – | – (no weight on any criterion) |")
            continue
        idx = np.argsort(-P[i])
        fmt = lambda j: f"{labels[j]} {P[i, j]:.3f} [{LO[i, j]:.3f}, {HI[i, j]:.3f}]"
        sep = "yes" if LO[i, idx[0]] > HI[i, idx[-1]] else "no"
        rows.append(f"| {NAMES[m]} | {total[i]:.2f} [{tci[i, 0]:.2f}, {tci[i, 1]:.2f}] | {'; '.join(map(fmt, idx[:3]))} | "
                    f"{'; '.join(map(fmt, idx[-3:]))} | {sep} |")
    (OUT / "fig_pilot_ranking.md").write_text("\n".join(rows) + "\n")
    print("\n".join(rows))


if __name__ == "__main__":
    main()
