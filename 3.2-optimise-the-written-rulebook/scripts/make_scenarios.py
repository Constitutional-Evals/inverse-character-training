"""Stage 0: sample AIRiskDilemmas scenarios and assign train/val/test splits."""

import argparse
import sys
from pathlib import Path

import _bootstrap  # noqa: F401
from rulebook_opt.data import build_scenarios, download_airisk, make_splits, read_jsonl, save_scenarios

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "data" / "scenarios.jsonl"))
    ap.add_argument("--airisk", help="local model_eval.jsonl; downloaded from Hugging Face if omitted")
    ap.add_argument("--train", type=int, default=300)
    ap.add_argument("--val", type=int, default=100)
    ap.add_argument("--test", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    src = Path(a.airisk) if a.airisk else ROOT / "data" / "airisk_model_eval.jsonl"
    if not src.exists():
        print(f"downloading AIRiskDilemmas to {src}", file=sys.stderr)
        download_airisk(src)
    scenarios = build_scenarios(read_jsonl(src))
    chosen = make_splits(scenarios, a.train, a.val, a.test, a.seed)
    save_scenarios(Path(a.out), chosen)
    print(f"{len(scenarios)} binary dilemmas available; wrote {len(chosen)} to {a.out} "
          f"(train {a.train}, val {a.val}, test {a.test})")


if __name__ == "__main__":
    main()
