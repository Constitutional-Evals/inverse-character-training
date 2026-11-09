"""
Design D4: natural multi-model pool on the same scenarios as D0 (EigenBench / UltraFeedback style).

Each scenario is answered by several different models with no instructions beyond a length cap
(about 220 words, the same target as D0's committed replies, to remove the length confound). Nothing
is designed against the criteria, so nothing can lead; the question is how often natural answers
disagree in ways the criteria pick up. Responses are stored as kind "committed" so the standard
judge and analysis use them; `kind_detail: natural_pool` and `model` record what they are. The D0
checker is run on each set for the obvious-best and duplicate rates.

The judge models (gpt-4.1, gemini-4-argon) are deliberately not among the answer models.

    python experiments/d4_natural_pool.py --limit 2      # smoke
    python experiments/d4_natural_pool.py
    python pipeline/judge.py --run-name exp_D4 --judges openai/gpt-4.1 google/gemini-4-argon
"""

import argparse
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "pipeline"))
from common import LLM, check_models, read_jsonl, write_jsonl  # noqa: E402
from make_responses import run_checks, words  # noqa: E402

MODELS = ["anthropic/claude-sonnet-5.5", "deepseek/deepseek-v4-pro", "mistralai/mistral-large-2512", "x-ai/grok-4.7"]
SYSTEM = "Keep your reply to about {words} words, in plain paragraphs."


def process(row, llm, args, clean):
    src = row["scenario_id"]
    sid = f"D4_{src.split('_')[-1]}"
    scenario = clean.get(src, row["scenario"])
    resp = []
    for i, m in enumerate(args.models, 1):
        text = llm.chat(m, [{"role": "system", "content": SYSTEM.format(words=args.words)}, {"role": "user", "content": scenario}],
                        temperature=0.7, max_tokens=3000, tag=f"natural:{sid}:{m}")
        resp.append({"response_id": f"{sid}:r{i}", "kind": "committed", "kind_detail": "natural_pool", "label": m,
                     "recommendation": None, "from_action": None, "favoured_by": [], "text": text, "words": words(text),
                     "model": m, "edits": []})
    checks = run_checks(llm, args, scenario, resp, sid, "check")
    flags = []
    if any(c["obvious_best"] for c in checks):
        flags.append("obvious_best")
    if min(min(c["defensible"].values()) for c in checks) < 3:
        flags.append("low_defensibility")
    if any(c["duplicates"] for c in checks):
        flags.append("duplicates_flagged")
    return {"scenario_id": sid, "design": "D4", "source_scenario_id": src, "scenario": scenario,
            "scenario_original": row["scenario"], "scenario_type": row.get("scenario_type"),
            "intended_tension": row.get("intended_tension"), "verified_criteria": row.get("verified_criteria") or [],
            "design_meta": {"models": args.models, "length_cap_words": args.words},
            "responses": resp, "removed_responses": [], "checks": checks, "flags": flags, "keep": True}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="pilot")
    ap.add_argument("--clean-from", default="pilot_v2", help="reuse D0's typo-fixed scenario text")
    ap.add_argument("--out-name", default="exp_D4")
    ap.add_argument("--models", nargs="+", default=MODELS)
    ap.add_argument("--checker-model", default="google/gemini-4-argon")
    ap.add_argument("--check-orders", type=int, default=2)
    ap.add_argument("--words", type=int, default=220)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--max-calls", type=int, default=400)
    args = ap.parse_args()
    rows = read_jsonl(Path("outputs/scenarios") / args.source / "generated_scenarios_rich.jsonl")
    if args.limit:
        rows = rows[:args.limit]
    clean = {r["scenario_id"]: r["scenario"] for r in read_jsonl(Path("outputs/responses") / args.clean_from / "responses.jsonl")}
    out = Path("outputs/responses") / args.out_name
    out.mkdir(parents=True, exist_ok=True)
    check_models(set(args.models) | {args.checker_model})
    llm = LLM(out / "cache", out / "usage.jsonl", args.max_calls)
    with ThreadPoolExecutor(max_workers=6) as pool:
        done = list(pool.map(lambda r: process(r, llm, args, clean), rows))
    write_jsonl(out / "responses.jsonl", done)
    flags = {}
    for r in done:
        for f in r["flags"]:
            flags[f] = flags.get(f, 0) + 1
    w = {m: [x["words"] for r in done for x in r["responses"] if x["model"] == m] for m in args.models}
    summary = {"design": "D4", "scenarios": len(done), "flags": flags,
               "mean_words_by_model": {m: round(sum(v) / len(v), 1) for m, v in w.items() if v},
               "api_calls_this_run": llm.calls, "cost_this_run_usd": round(llm.cost, 4)}
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
