"""Held-out test of the pilot fits in probability units: for each model, the probability each fitted version gives to
the reply the model actually picked, on scenarios the fit never saw (exp of minus the held-out log loss, i.e. the
geometric mean over picks). Versions: random guess, one common criterion weight, free per-criterion weights, and
per-criterion weights constrained to sum to 1 (beta 10). Reads fit_craft.json (published fit) and fig_pilot_simplex.json.

    PYTHONPATH=scripts python figures/gen_fig_pilot_test.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from gen_fig_pilot_fit import get  # noqa: E402
from gen_fig_pilot_ranking import NAMES  # noqa: E402

OUT = Path(__file__).resolve().parent


def main() -> None:
    free = json.loads(get("20261106-fit-pilot15", "fit_craft.json").read_text())["models"]
    simplex = json.loads((OUT / "fig_pilot_simplex.json").read_text())["models"]
    rows = []
    for m in NAMES:
        h = free[m]["heldout_logloss"]
        rows.append({"name": NAMES[m], "random": np.exp(-h["uniform"]), "common": np.exp(-h["equal"]),
                     "free": np.exp(-h["additive"]), "sum1": np.exp(-simplex[m]["heldout"])})
    rows.sort(key=lambda r: -r["common"])

    plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"], "font.size": 9.5,
                         "figure.dpi": 300, "savefig.dpi": 300, "savefig.bbox": "tight", "axes.spines.top": False,
                         "axes.spines.right": False, "axes.grid": True, "grid.alpha": 0.15, "legend.frameon": False})
    series = [("common", "one common criterion weight", "#0072B2", "o"),
              ("free", "per-criterion priorities (free)", "#E69F00", "D"),
              ("sum1", "per-criterion priorities (sum to 1)", "#D55E00", "s")]
    fig, ax = plt.subplots(figsize=(6.4, 3.4))
    y = np.arange(len(rows))[::-1]
    off = {"common": 0.18, "free": 0.0, "sum1": -0.18}
    for key, label, color, marker in series:
        ax.scatter([100 * r[key] for r in rows], y + off[key], color=color, marker=marker, s=26, label=label, zorder=3)
    ax.axvline(100 * rows[0]["random"], color="#8C8C8C", ls="--", lw=1, label="random guess", zorder=1)
    ax.set_yticks(y, [r["name"] for r in rows])
    ax.set_xlabel("probability given to the reply actually picked, on unseen scenarios (%)")
    ax.legend(loc="lower left", bbox_to_anchor=(0.0, 1.0), ncol=2, fontsize=8, handletextpad=0.3, columnspacing=1.0)
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"fig_pilot_test.{ext}")
    for r in rows:
        print(f"{r['name']:16} random {100*r['random']:.0f}  common {100*r['common']:.0f}  free {100*r['free']:.0f}  sum1 {100*r['sum1']:.0f}")


if __name__ == "__main__":
    main()
