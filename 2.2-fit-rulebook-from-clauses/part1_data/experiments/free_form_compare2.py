"""
Free-form check, comparison step redone with two frontier comparers (Fable 5.2 and Gemini 4 Carbon).
Reuses the cached free answers and choices in outputs/free_form/exp_anchored/results.jsonl; only
comparer calls are made.

    python experiments/free_form_compare2.py --limit 1   # smoke test: one free answer, both comparers
    python experiments/free_form_compare2.py             # all 15, updates reports/free_form_check.md

Each comparer, separately, sees the scenario, one free answer and the 5 prepared replies in a shuffled
order (labelled R1-R5, no anchors or criteria), and returns: one sentence on what the free answer
recommends, a 1-5 closeness score for every prepared reply, and the closest reply. The closest reply
used in the analysis is derived from the scores: "none" if no reply scores 4 or more; otherwise the
top-scored reply (if several tie, the one the comparer named, when it is among them).

Writes outputs/free_form/exp_anchored/compare2.jsonl (results.jsonl is left untouched) and a new section in
reports/free_form_check.md, placed before the appendix (the earlier sections are kept).
"""
import argparse
import json
import random
import re
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE / "pipeline"))
sys.path.insert(0, str(HERE / "experiments"))
from common import LLM, check_models, extract_json, read_jsonl, write_jsonl  # noqa: E402
from free_form_check import OUT, SCENARIOS, TARGETS, load_key  # noqa: E402

COMPARERS = ["anthropic/claude-fable-5.2", "google/gemini-4-carbon"]
START, END = "<!-- COMPARE2 START -->", "<!-- COMPARE2 END -->"


def prompt(scenario, free, texts):
    replies = "\n\n".join(f"<R{i + 1}>\n{t}\n</R{i + 1}>" for i, t in enumerate(texts))
    return (
        "A user sent the message below. An assistant wrote the FREE ANSWER. Separately, five prepared "
        "replies R1-R5 were written for the same message.\n\n"
        f"<message>\n{scenario}\n</message>\n\n<free_answer>\n{free}\n</free_answer>\n\n{replies}\n\n"
        "1. In one sentence, what does the free answer recommend the user do?\n"
        "2. Rate each prepared reply 1-5 for how close its recommendation (the course of action, not tone, "
        "length or style) is to the free answer's: 5 = same advice, 3 = same direction but clearly "
        "different, 1 = opposite or unrelated (2 and 4 in between).\n"
        "3. Name the closest prepared reply, or \"none\" if no reply scores 4 or more.\n\n"
        "Reply with JSON only:\n"
        '{"recommendation": "...", "scores": {"R1": n, "R2": n, "R3": n, "R4": n, "R5": n}, '
        '"closest": "R1" | "R2" | "R3" | "R4" | "R5" | "none"}'
    )


def parse(text):
    d = extract_json(text)
    s = d.get("scores")
    if not isinstance(s, dict) or set(s) != {f"R{i}" for i in range(1, 6)}:
        raise ValueError("scores must cover R1-R5")
    for k, v in s.items():
        if not (isinstance(v, int) or (isinstance(v, str) and re.fullmatch(r"[1-5]", v))) or not 1 <= int(v) <= 5:
            raise ValueError(f"bad score {k}={v!r}")
    if d.get("closest") not in {"R1", "R2", "R3", "R4", "R5", "none"} or not isinstance(d.get("recommendation"), str):
        raise ValueError("bad closest/recommendation")
    d["scores"] = {k: int(v) for k, v in s.items()}
    return d


def run_one(llm, row, res, comparer):
    sid, model = res["scenario_id"], res["model"]
    resps = row["responses"]
    order = list(range(len(resps)))
    random.Random(f"compare2:{sid}:{model}:{comparer}").shuffle(order)
    raw = llm.chat(comparer, [{"role": "user", "content": prompt(row["scenario"], res["free_answer"], [resps[i]["text"] for i in order])}],
                   temperature=0.0, max_tokens=4096, tag=f"compare2:{sid}:{model}", validate=lambda t: (t, parse(t)))
    text, d = raw
    scores = {resps[order[int(k[1]) - 1]]["response_id"]: v for k, v in d["scores"].items()}
    named = None if d["closest"] == "none" else resps[order[int(d["closest"][1]) - 1]]["response_id"]
    top = max(scores.values())
    tied = [r for r, v in scores.items() if v == top]
    closest = None if top < 4 else (named if named in tied else tied[0])
    anchor = {r["response_id"]: r["anchor"] for r in resps}
    return {"scenario_id": sid, "model": model, "comparer": comparer, "order": [resps[i]["response_id"] for i in order],
            "recommendation": d["recommendation"], "scores": scores, "named_closest": named,
            "closest_response_id": closest, "closest_anchor": anchor.get(closest), "tie_at_top": len(tied) > 1 and top >= 4,
            "named_consistent": named == closest, "raw": text}


def section(results, comp, crit):
    short = lambda m: {"anthropic/claude-fable-5.2": "Fable", "google/gemini-4-carbon": "Carbon"}.get(m, m.split("/")[1])
    rid = lambda x: x.split(":")[1] if x else "none"
    by = {(c["scenario_id"], c["model"], c["comparer"]): c for c in comp}
    cases = [(r, by[(r["scenario_id"], r["model"], COMPARERS[0])], by[(r["scenario_id"], r["model"], COMPARERS[1])]) for r in results]
    F, C = COMPARERS
    agree = [x for x in cases if x[1]["closest_response_id"] == x[2]["closest_response_id"]]
    L = [START, "## Re-comparison with two frontier comparers (2026-11-13)", "",
         f"Same 15 free answers and 30 choices as above (cached; no new target calls). Each free answer was "
         f"compared separately by `{F}` and `{C}`: one-sentence recommendation, a 1-5 closeness score for each "
         "prepared reply (5 = same advice, 3 = same direction but clearly different, 1 = opposite or unrelated), "
         "replies shown in a shuffled order without anchors. Closest = top-scored reply, or none if no reply "
         "scores 4+. Code: `experiments/free_form_compare2.py`; data: `outputs/free_form/exp_anchored/compare2.jsonl`.", "",
         "<!-- C2 SUMMARY -->", "",
         "| | Fable | Carbon | both agree | Argon (earlier) |", "|---|---|---|---|---|"]
    n = len(cases)
    nonef = sum(x[1]["closest_response_id"] is None for x in cases)
    nonec = sum(x[2]["closest_response_id"] is None for x in cases)
    L.append(f"| free answer matches no prepared reply (\"none\") | {nonef}/{n} | {nonec}/{n} | "
             f"{sum(x[1]['closest_response_id'] is None for x in agree)}/{len(agree)} | "
             f"{sum(r['closest_response_id'] is None for r in results)}/{n} |")

    def match(get, sel):
        hits = tot = 0
        for x in sel:
            cl = get(x)
            for ch in x[0]["choices"]:
                tot += 1
                hits += cl is not None and ch["response_id"] == cl
        return f"{hits}/{tot}"
    L.append(f"| choice = closest reply (both orders) | {match(lambda x: x[1]['closest_response_id'], cases)} | "
             f"{match(lambda x: x[2]['closest_response_id'], cases)} | {match(lambda x: x[1]['closest_response_id'], agree)} | "
             f"{match(lambda x: x[0]['closest_response_id'], cases)} |")
    L += ["", f"Comparers agree on the closest reply in **{len(agree)}/{n}** free answers "
          f"(Fable vs Argon: {sum(x[1]['closest_response_id'] == x[0]['closest_response_id'] for x in cases)}/{n}; "
          f"Carbon vs Argon: {sum(x[2]['closest_response_id'] == x[0]['closest_response_id'] for x in cases)}/{n}). "
          f"Ties at the top score (4+): Fable {sum(x[1]['tie_at_top'] for x in cases)}, Carbon {sum(x[2]['tie_at_top'] for x in cases)}.", ""]

    # per model breakdown
    L += ["Per target model, choice = closest (both orders):", "", "| target | Fable's closest | Carbon's closest | both agree |", "|---|---|---|---|"]
    for m in TARGETS:
        sel = [x for x in cases if x[0]["model"] == m]
        sa = [x for x in sel if x in agree]
        L.append(f"| {m.split('/')[1]} | {match(lambda x: x[1]['closest_response_id'], sel)} | "
                 f"{match(lambda x: x[2]['closest_response_id'], sel)} | {match(lambda x: x[1]['closest_response_id'], sa)} |")

    picked = Counter(ch["anchor"] for r in results for ch in r["choices"])
    cf = Counter(x[1]["closest_anchor"] for x in cases if x[1]["closest_anchor"] is not None)
    cc = Counter(x[2]["closest_anchor"] for x in cases if x[2]["closest_anchor"] is not None)
    ca = Counter(r["closest_anchor"] for r in results if r["closest_anchor"] is not None)
    L += ["", "Anchors: times chosen (30 choices) vs times closest to a free answer (15 free answers):", "",
          "| anchor criterion | chosen | closest (Fable) | closest (Carbon) | closest (Argon, earlier) |", "|---|---|---|---|---|"]
    for c in sorted(set(picked) | set(cf) | set(cc) | set(ca), key=lambda c: (-picked[c], -(cf[c] + cc[c]), c)):
        L.append(f"| {c} {crit[c]} | {picked[c]} | {cf[c]} | {cc[c]} | {ca[c]} |")

    L += ["", "### Cases where the comparers disagree", "",
          "| scenario | target | Fable: closest (scores r1-r5) | Fable's sentence | Carbon: closest (scores r1-r5) | Carbon's sentence | choices |",
          "|---|---|---|---|---|---|---|"]
    fmt = lambda c: f"{rid(c['closest_response_id'])} ({'-' if c['closest_anchor'] is None else c['closest_anchor']}) [" + \
        " ".join(str(c["scores"][f"{c['scenario_id']}:r{i}"]) for i in range(1, 6)) + "]"
    for r, f, c in cases:
        if f["closest_response_id"] != c["closest_response_id"]:
            chs = ", ".join(rid(ch["response_id"]) for ch in r["choices"])
            L.append(f"| {r['scenario_id']} | {r['model'].split('/')[1]} | {fmt(f)} | {f['recommendation']} | {fmt(c)} | {c['recommendation']} | {chs} |")

    own = [x for x in cases if x[0]["model"] == F]
    oa = [x for x in own if x in agree]
    L += ["", "### Fable comparing its own free answers", "",
          f"In 5 of the 15 cases Fable is judging its own free answer. Comparers agree there in {len(oa)}/5 "
          f"(vs {len(agree) - len(oa)}/10 on the other models' answers). Choice = closest: Fable's closest "
          f"{match(lambda x: x[1]['closest_response_id'], own)}, Carbon's closest {match(lambda x: x[2]['closest_response_id'], own)}.", "",
          "| scenario | Fable's closest (scores) | Carbon's closest (scores) | Fable's choices |", "|---|---|---|---|"]
    for r, f, c in own:
        L.append(f"| {r['scenario_id']} | {fmt(f)} | {fmt(c)} | {', '.join(rid(ch['response_id']) for ch in r['choices'])} |")
    L += ["", "<!-- C2 NOTES -->", "", END, ""]
    return "\n".join(L)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None, help="only the first N free answers (smoke test)")
    ap.add_argument("--max-calls", type=int, default=80)
    args = ap.parse_args()
    load_key()
    results = read_jsonl(OUT / "results.jsonl")
    order = {(s, m): i for i, (s, m) in enumerate((s, m) for s in SCENARIOS for m in TARGETS)}
    results.sort(key=lambda r: order[(r["scenario_id"], r["model"])])
    rows = {r["scenario_id"]: r for r in read_jsonl(HERE / "outputs/responses/exp_anchored/responses.jsonl")}
    crit = {c["pilot_id"]: c["criterion_id"] for c in read_jsonl(HERE / "outputs/criteria/criteria_pilot.jsonl")}
    todo = results[: args.limit]
    check_models(set(COMPARERS))
    llm = LLM(OUT / "cache", OUT / "usage.jsonl", args.max_calls)
    with ThreadPoolExecutor(max_workers=8) as pool:
        futs = [pool.submit(run_one, llm, rows[r["scenario_id"]], r, c) for r in todo for c in COMPARERS]
        comp = [f.result() for f in futs]
    out = OUT / ("compare2.jsonl" if args.limit is None else "compare2_smoke.jsonl")
    write_jsonl(out, comp)
    for c in comp:
        print(c["scenario_id"], c["model"].split("/")[1], "by", c["comparer"].split("/")[1], "closest", c["closest_response_id"],
              "scores", list(c["scores"].values()), "named", c["named_closest"])
    print(f"API calls {llm.calls}, cost ${llm.cost:.3f}; wrote {out}")
    if args.limit is None:
        rep = HERE / "reports/free_form_check.md"
        t = rep.read_text()
        sec = section(results, comp, crit)
        if START in t:
            t = t[: t.index(START)] + sec + t[t.index(END) + len(END) + 1:]
        else:
            t = t.replace("## Appendix: full free answers", sec + "\n## Appendix: full free answers", 1)
        rep.write_text(t)
        print(f"Updated {rep}")


if __name__ == "__main__":
    main()
