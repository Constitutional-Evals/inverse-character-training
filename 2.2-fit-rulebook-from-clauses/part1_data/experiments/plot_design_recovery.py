"""Figure for the report: recovery error per design (equal scoring budget), same-judge vs cross-judge.
Reads outputs/analysis/compare.json, writes reports/figures/recovery_by_design.png."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parents[1]
c = json.loads((HERE / "outputs/analysis/compare.json").read_text())
names = list(c)
BLUE, INK, MUTED = "#2a78d6", "#1a1a19", "#8a8983"
fig, axes = plt.subplots(1, 4, figsize=(14, 3.4), sharey=True)
panels = [("sparse_same", "Sparse weights, same judge"), ("sparse_cross", "Sparse weights, cross-judge"),
          ("dense_same", "Dense weights, same judge"), ("dense_cross", "Dense weights, cross-judge")]
for ax, (cond, title) in zip(axes, panels):
    for i, n in enumerate(names):
        r = c[n]["recovery_equal_budget"][cond]["l1"]
        ax.errorbar(r["mean"], i, xerr=r["sd_over_seeds"], fmt="o", color=BLUE, ms=7, capsize=3, lw=2)
        ax.text(r["mean"], i + 0.28, f"{r['mean']:.2f}", ha="center", fontsize=8, color=INK)
    eq = c[names[0]]["recovery_equal_budget"][cond]["l1_equal"]["mean"]
    ax.axvline(eq, color=MUTED, ls="--", lw=1.5)
    ax.text(eq - 0.03, -0.55, "equal weights", fontsize=8, color=MUTED, ha="right", va="center")
    ax.set_title(title, fontsize=10, color=INK)
    ax.set_xlabel("L1 error (0 = perfect, 2 = max)", fontsize=9)
    ax.set_xlim(0, 1.9); ax.set_ylim(len(names) - 0.5, -0.8)
    ax.grid(axis="x", color="#e5e4df", lw=0.8)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
axes[0].set_yticks(range(len(names)))
axes[0].set_yticklabels([f"{n} (n={c[n]['recovery_equal_budget']['n_scen']})" for n in names])

fig.suptitle("Weight recovery at equal scoring budget (45 scored responses); dots = mean over 5 seeds, bars = sd over seeds",
             fontsize=10, color=INK)
fig.tight_layout()
(HERE / "reports").mkdir(exist_ok=True)
fig.savefig(HERE / "reports/figures/recovery_by_design.png", dpi=130)
print("wrote reports/figures/recovery_by_design.png")
