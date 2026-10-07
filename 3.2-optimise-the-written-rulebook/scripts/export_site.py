"""Export finished runs to one JSON file for the ValueArena results page (/ict/results/).

Re-scores the floor and ceiling on the val split so the progress chart has reference lines; with the
API cache in place this costs nothing if evaluate.py / run_gepa.py already ran.
"""

import argparse
import datetime as dt
import json
from pathlib import Path

import _bootstrap  # noqa: F401
from rulebook_opt.config import load_config, resolve_path
from rulebook_opt.modes import build
from rulebook_opt.rulebooks import EMPTY_RULEBOOK, load_rulebook
from rulebook_opt.runtime import results_dir

REPO = "https://github.com/Constitutional-Evals/inverse-character-training/tree/main/3.2-optimise-the-written-rulebook"
ROWS = {"floor": "empty rulebook (floor)", "ceiling": "true constitution C (ceiling)", "recovered": "recovered C' (GEPA)"}


def items(text: str) -> list[str]:
    out = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        head, _, rest = line.partition(". ")
        out.append(rest if head.isdigit() and rest else line)
    return out


def brief(r: dict) -> dict:
    return {k: r[k] for k in ("agreement", "ci_low", "ci_high", "n")}


def run_entry(config: str) -> dict:
    cfg = load_config(config)
    out = results_dir(cfg)
    test = json.loads((out / "test.json").read_text())
    history = json.loads((out / "history.json").read_text())
    s = build(cfg)
    truth = load_rulebook(resolve_path(cfg, cfg["truth"]["constitution"]))
    val = s.splits["val"]
    target = Path(cfg["truth"]["constitution"]).stem.replace("oct_", "")
    lu = cfg.get("lineup", {})
    decoys = (["default (no constitution)"] if lu.get("include_default", True) else []) + [
        Path(p).stem.replace("oct_", "") for p in lu.get("decoys", [])
    ]
    return {
        "key": target,
        "label": target.capitalize(),
        "mode": s.mode,
        "setup": {
            "target": f"{cfg['target']['model']} prompted with the OCT {target} constitution",
            "interpreter": cfg["interpreter"]["model"],
            "judge": cfg.get("judge", {}).get("model"),
            "optimizer": cfg["optimizer"]["model"],
            "decoys": decoys,
            "chance": s.chance,
            "n": {k: len(v) for k, v in s.splits.items()},
            "budget": cfg.get("gepa", {}).get("max_metric_calls"),
        },
        "test": {k: brief(test["rows"][v]) for k, v in ROWS.items() if v in test["rows"]},
        "val_reference": {"floor": s.score(EMPTY_RULEBOOK, val)["agreement"], "ceiling": s.score(truth, val)["agreement"]},
        "history": [{"idx": h["idx"], "val": h["val_score"]} for h in history],
        "recovered": items((out / "best_rulebook.md").read_text()),
        "truth": items(truth),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", action="append", required=True)
    ap.add_argument("--signal", help="results/signal/<model>_<split>.json from check_signal.py")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    data = {"updated": dt.date.today().isoformat(), "repo": REPO, "runs": [run_entry(c) for c in a.config]}
    if a.signal:
        sig = json.loads(Path(a.signal).read_text())
        data["signal"] = {
            "model": sig["model"],
            "n": sig["n"],
            "rows": [{"name": k.replace("oct_", ""), "flips": v["flips"], "compared": v["compared"]}
                     for k, v in sig["constitutions"].items()],
        }
    Path(a.out).write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n")
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
