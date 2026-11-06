"""
Turn WIMHF's output into a rulebook that 3.2's evaluate.py can score.

Each kept SAE feature has an LLM label written from response A's perspective and a
length-controlled logit coefficient for "the target chose A". So:
  coef > 0  ->  "Prefer the option that <label>."
  coef < 0  ->  "Avoid the option that <label>."
Features are ordered by |coef|, the strongest first.

    python features_to_rulebook.py --wimhf-out outputs/m0_synthetic --out results/rulebook_3p1_m0_synthetic.md

Writes the rulebook (.md, a numbered list) and a .json with the details behind each line.
Respects 3.2's format limits: at most 15 criteria and 3,500 characters.
"""

import argparse
import json
import math
from pathlib import Path

MAX_CRITERIA = 15
MAX_CHARS = 3500


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def clean_label(text: str) -> str:
    text = text.strip().lstrip("-").strip().strip('"').strip()
    text = text.rstrip(".")
    return text[:1].lower() + text[1:] if text else text


def to_criterion(label: str, coef: float) -> str:
    verb = "Prefer" if coef > 0 else "Avoid"
    return f"{verb} the option that {label}."


def render(items: list[str]) -> str:
    return "\n".join(f"{i}. {t}" for i, t in enumerate(items, 1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--wimhf-out", required=True, help="WIMHF --output-dir")
    ap.add_argument("--out", required=True, help="rulebook path, .md")
    ap.add_argument("--source", choices=["logit", "lasso5", "lasso10"], default="logit",
                    help="coefficients: all kept features (logit) or WIMHF's lasso top-5 / top-10")
    ap.add_argument("--max-criteria", type=int, default=10)
    ap.add_argument("--min-abs-coef", type=float, default=0.0)
    ap.add_argument("--include-unkept", action="store_true",
                    help="also use features whose label failed WIMHF's fidelity test")
    args = ap.parse_args()

    out_dir = Path(args.wimhf_out)
    table = read_jsonl(out_dir / "feature_table.jsonl")

    if args.source == "logit":
        coefs = {int(r["neuron_idx"]): r.get("length_controlled_logit_coef") for r in table}
    else:
        k = args.source.replace("lasso", "")
        lasso = json.loads((out_dir / "lasso_coefficients.json").read_text())
        if k not in lasso:
            raise SystemExit(f"No lasso top-{k} in lasso_coefficients.json (found {list(lasso)})")
        coefs = {int(i): c for i, c in lasso[k].items()}

    candidates = []
    for r in table:
        idx = int(r["neuron_idx"])
        coef = coefs.get(idx)
        label = r.get("interpretation") or ""
        if coef is None or (isinstance(coef, float) and math.isnan(coef)) or not label.strip():
            continue
        if not args.include_unkept and not r.get("kept"):
            continue
        if abs(coef) < args.min_abs_coef:
            continue
        candidates.append({
            "neuron_idx": idx,
            "label": clean_label(label),
            "coef": float(coef),
            "fidelity_correlation": r.get("correlation"),
            "p_value": r.get("p_value"),
            "prevalence": r.get("prevalence"),
        })

    candidates.sort(key=lambda c: -abs(c["coef"]))
    chosen = candidates[: min(args.max_criteria, MAX_CRITERIA)]
    for c in chosen:
        c["criterion"] = to_criterion(c["label"], c["coef"])

    while chosen and len(render([c["criterion"] for c in chosen])) > MAX_CHARS:
        chosen.pop()

    if not chosen:
        raise SystemExit("No features qualified. Try --include-unkept or --min-abs-coef 0, "
                         "and check WIMHF's log for how many features passed fidelity.")

    text = render([c["criterion"] for c in chosen])
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text + "\n", encoding="utf-8")
    out.with_suffix(".json").write_text(json.dumps(chosen, indent=2))

    print(f"{len(candidates)} qualifying features, wrote {len(chosen)} criteria "
          f"({len(text)} chars) to {out}\n")
    for c in chosen:
        print(f"  coef {c['coef']:+.3f}  fidelity r={c['fidelity_correlation']}  {c['criterion']}")


if __name__ == "__main__":
    main()
