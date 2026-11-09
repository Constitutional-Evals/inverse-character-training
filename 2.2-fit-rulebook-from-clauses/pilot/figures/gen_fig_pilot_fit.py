"""Figure for the pilot fit (run 20261106-fit-pilot15), from the published runs on Hugging Face.

(a) held-out log loss per choice (5-fold cross-validation by scenario): craft + position only, and with one common
    weight on the judge's criterion scores; chance = log k averaged over choices.
(b) that common weight, fitted on all scenarios, with a 90% bootstrap interval over scenarios (1,000 resamples).

    PYTHONPATH=scripts python figures/gen_fig_pilot_fit.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from huggingface_hub import hf_hub_download

import fit_weights as fw

REPO, PREFIX = "invi-bhagyesh/log", "constitution-recovery/runs"
OUT = Path(__file__).resolve().parent
BOOT = 1000


def get(run: str, name: str) -> Path:
    return Path(hf_hub_download(REPO, f"{PREFIX}/{run}/{name}", repo_type="dataset"))


def jsonl_gz(path: Path) -> list[dict]:
    import gzip
    with gzip.open(path, "rt", encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def main() -> None:
    fit = json.loads(get("20261106-fit-pilot15", "fit_craft.json").read_text())
    crit = fit["criteria"]
    tmp = OUT / ".cache"
    tmp.mkdir(exist_ok=True)
    for run, name in (("20261106-ratings-pilot15", "scores.jsonl"), ("20261106-ratings-pilot15", "leading.jsonl"),
                      ("20261106-responses-pilot15", "responses.jsonl")):
        rows = jsonl_gz(get(run, name + ".gz"))
        (tmp / name).write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    z, ver = fw.features(tmp / "scores.jsonl", crit)
    fw.load_nuisance(tmp / "responses.jsonl", tmp / "leading.jsonl")

    names = {"x-ai/grok-4.7": "Grok 4.7", "z-ai/glm-5.3": "GLM 5.3", "qwen2.5-7b-it": "Qwen2.5-7B",
             "qwen2.5-7b-it+goodness": "+ goodness", "qwen2.5-7b-it+loving": "+ loving",
             "qwen2.5-7b-it+sycophancy": "+ sycophancy", "qwen2.5-7b-it+impulsiveness": "+ impulsiveness",
             "qwen2.5-7b-it+misalignment": "+ misalignment"}
    files = {"x-ai/grok-4.7": "grok-4.7", "z-ai/glm-5.3": "glm-5.3", "qwen2.5-7b-it": "qwen2.5-7b-it-base"}
    rows = []
    for model in names:
        stem = files.get(model, "qwen2.5-7b-it-" + model.split("+")[-1])
        choices = jsonl_gz(get("20261106-choices-pilot15", f"choices_{stem}.jsonl.gz"))
        items = fw.build(choices, z, ver)
        w = fw.fit_additive(items, len(crit), "equal")[0]
        scen = sorted({s for s, *_ in items})
        by = {s: [it for it in items if it[0] == s] for s in scen}
        rng = np.random.default_rng(0)
        boots = [fw.fit_additive([it for s in rng.choice(scen, len(scen)) for it in by[s]], len(crit), "equal")[0]
                 for _ in range(BOOT)]
        h = fit["models"][model]["heldout_logloss"]
        rows.append({"model": model, "name": names[model], "chance": h["uniform"], "base": h["position"],
                     "full": h["equal"], "w": w, "lo": np.percentile(boots, 5), "hi": np.percentile(boots, 95)})
    rows.sort(key=lambda r: r["full"])
    (OUT / "fig_pilot_fit.json").write_text(json.dumps(rows, indent=1, default=float))

    plt.rcParams.update({
        "font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"], "font.size": 10,
        "axes.labelsize": 10, "legend.fontsize": 8.5, "legend.frameon": False, "figure.dpi": 300,
        "savefig.dpi": 300, "savefig.bbox": "tight", "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.alpha": 0.15, "axes.titlesize": 10, "axes.titleweight": "bold",
    })
    GREY, BLUE, VERM = "#8C8C8C", "#0072B2", "#D55E00"  # Okabe-Ito blue / vermillion, neutral grey
    fig, (a, b) = plt.subplots(1, 2, figsize=(7.0, 3.0), sharey=True, gridspec_kw={"wspace": 0.08})
    y = np.arange(len(rows))[::-1]
    for yi, r in zip(y, rows):
        a.plot([r["base"], r["full"]], [yi, yi], color=GREY, lw=1.2, zorder=1)
    a.scatter([r["base"] for r in rows], y, color=GREY, s=22, zorder=2, label="craft + position")
    a.scatter([r["full"] for r in rows], y, color=BLUE, s=26, zorder=3, label="+ common criterion weight")
    a.axvline(rows[0]["chance"], color=VERM, ls="--", lw=1, label="chance")
    a.set_yticks(y, [r["name"] for r in rows])
    a.set_xlabel("held-out log loss per choice")
    a.set_title("(a) prediction on unseen scenarios", loc="left")
    a.legend(loc="lower left", bbox_to_anchor=(0.0, 1.08), ncol=3, columnspacing=0.8, handletextpad=0.3)
    b.errorbar([r["w"] for r in rows], y, xerr=[[r["w"] - r["lo"] for r in rows], [r["hi"] - r["w"] for r in rows]],
               fmt="o", color=BLUE, ms=4.5, elinewidth=1.2, capsize=2)
    b.axvline(0, color=GREY, lw=0.8)
    b.set_xlabel("common criterion weight (90% interval)")
    b.set_title("(b) weight on criterion scores", loc="left")
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"fig_pilot_fit.{ext}")
    for r in rows:
        print(f"{r['name']:16} chance {r['chance']:.3f} craft+pos {r['base']:.3f} +criteria {r['full']:.3f} "
              f"w {r['w']:.2f} [{r['lo']:.2f}, {r['hi']:.2f}]")


if __name__ == "__main__":
    main()
