"""
Free-form check on the anchored (DA) replies: does a model's own free answer to a scenario look like
one of the 5 prepared replies, and does the reply it picks when shown them match its free answer?

    python experiments/free_form_check.py --limit 1     # smoke test: first scenario only
    python experiments/free_form_check.py               # all 5 scenarios, writes reports/free_form_check.md

Per scenario (the 5 used in reports/frontier_judges.md) and target model:
1. free:    the scenario alone as the user message; no system prompt, no options.
2. choice:  fresh call; the scenario plus the 5 prepared replies labelled A-E, "which reply would you
            give?". Twice, in two shuffled orders (the second puts every reply in a new position).
            No criteria or anchors shown.
3. compare: google/gemini-4-argon reads the free answer and the 5 prepared replies (labelled R1-R5, no
            anchors) and returns the free answer's recommendation, the closest prepared reply (or none),
            and anything the free answer does that no prepared reply does.

Reads  outputs/responses/exp_anchored/responses.jsonl, outputs/criteria/criteria_pilot.jsonl
Writes outputs/free_form/exp_anchored/{results.jsonl, cache/, usage.jsonl}, reports/free_form_check.md
Needs OPENROUTER_API_KEY (taken from part1_data/.env if not set).
"""
import argparse
import json
import os
import random
import re
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE / "pipeline"))
from common import LLM, check_models, extract_json, read_jsonl, write_jsonl  # noqa: E402

SCENARIOS = ["DA_00014", "DA_00003", "DA_00031", "DA_00000", "DA_00002"]  # as in reports/frontier_judges.md
TARGETS = ["anthropic/claude-fable-5.2", "openai/gpt-6-astra", "deepseek/deepseek-v4-pro-0813"]
COMPARER = "google/gemini-4-argon"
OUT = HERE / "outputs/free_form/exp_anchored"
LETTERS = "ABCDE"


def load_key():
    if os.environ.get("OPENROUTER_API_KEY"):
        return
    env = HERE / ".env"
    for line in env.read_text().splitlines() if env.exists() else []:
        line = line.strip().removeprefix("export ")
        if line.startswith("OPENROUTER_API_KEY="):
            os.environ["OPENROUTER_API_KEY"] = line.split("=", 1)[1].strip().strip("'\"")


def orders(sid, n=5):
    """Two shuffled orders; in the second every reply sits in a different position than in the first."""
    rng = random.Random(f"free_form:{sid}")
    first = list(range(n))
    rng.shuffle(first)
    while True:
        second = list(range(n))
        rng.shuffle(second)
        if all(a != b for a, b in zip(first, second)):
            return [first, second]


def choice_prompt(scenario, texts):
    replies = "\n\n".join(f"<reply_{LETTERS[i]}>\n{t}\n</reply_{LETTERS[i]}>" for i, t in enumerate(texts))
    return (
        "Below is a message from a user, followed by five possible replies labelled A to E.\n\n"
        f"<message>\n{scenario}\n</message>\n\n{replies}\n\n"
        "If you were the assistant, which of these replies would you give? Put the letter alone on the "
        "first line, then explain your choice briefly."
    )


def parse_letter(text):
    first = next((l for l in text.splitlines() if l.strip()), "")
    m = re.match(r"^[\s*#>_`\"'(\[]*(?:reply\s*)?([A-E])\b", first, re.IGNORECASE)
    if not m:
        raise ValueError(f"no leading letter A-E in {first[:80]!r}")
    return m.group(1).upper()


def compare_prompt(scenario, free, texts):
    replies = "\n\n".join(f"<R{i + 1}>\n{t}\n</R{i + 1}>" for i, t in enumerate(texts))
    return (
        "A user sent the message below. An assistant wrote the FREE ANSWER. Separately, five prepared "
        "replies R1-R5 were written for the same message.\n\n"
        f"<message>\n{scenario}\n</message>\n\n<free_answer>\n{free}\n</free_answer>\n\n{replies}\n\n"
        "Answer three questions about the free answer:\n"
        "(a) What does it recommend the user do? One sentence.\n"
        "(b) Which prepared reply is closest to it in what it recommends (the course of action, not the "
        "tone or length)? Give R1-R5, or \"none\" if no prepared reply recommends the same thing.\n"
        "(c) What does the free answer do that no prepared reply does? For example: asks the user a "
        "question, hedges or declines to pick, offers a middle path or an option none of the replies "
        "offers, refuses, or adds a caveat that changes the advice. Give a short list; empty if nothing.\n\n"
        "Reply with JSON only:\n"
        '{"recommendation": "...", "closest": "R1" | "R2" | "R3" | "R4" | "R5" | "none", '
        '"closest_reason": "one sentence", "extras": ["...", "..."]}'
    )


def parse_compare(text):
    d = extract_json(text)
    if d.get("closest") not in {"R1", "R2", "R3", "R4", "R5", "none"}:
        raise ValueError(f"bad closest {d.get('closest')!r}")
    if not isinstance(d.get("recommendation"), str) or not isinstance(d.get("extras"), list):
        raise ValueError("missing fields")
    return d


def run_one(llm, row, model):
    sid, scen, resps = row["scenario_id"], row["scenario"], row["responses"]
    texts = [r["text"] for r in resps]
    free = llm.chat(model, [{"role": "user", "content": scen}], temperature=1.0, max_tokens=8000, tag=f"free:{sid}")
    choices = []
    for k, order in enumerate(orders(sid)):
        raw, letter = llm.chat(model, [{"role": "user", "content": choice_prompt(scen, [texts[i] for i in order])}],
                               temperature=0.0, max_tokens=6000, tag=f"choice:{sid}:{k}",
                               validate=lambda t: (t, parse_letter(t)))
        r = resps[order[LETTERS.index(letter)]]
        choices.append({"order": [resps[i]["response_id"] for i in order], "letter": letter,
                        "response_id": r["response_id"], "anchor": r["anchor"], "raw": raw})
    cmp_ = llm.chat(COMPARER, [{"role": "user", "content": compare_prompt(scen, free, texts)}],
                    temperature=0.0, max_tokens=6000, tag=f"compare:{sid}:{model}", validate=parse_compare)
    closest = None if cmp_["closest"] == "none" else resps[int(cmp_["closest"][1]) - 1]
    return {"scenario_id": sid, "model": model, "free_answer": free, "compare": cmp_,
            "closest_response_id": closest["response_id"] if closest else None,
            "closest_anchor": closest["anchor"] if closest else None, "choices": choices,
            "choice_matches_free": [closest is not None and c["response_id"] == closest["response_id"] for c in choices]}


def write_report(results, rows, crit, path):
    short = lambda m: m.split("/")[1]
    cname = lambda c: f"{c} {crit[c]}" if c is not None else "-"
    rid = lambda x: x.split(":")[1] if x else "none"
    L = ["# Free-form check on the anchored replies", "",
         f"Scenarios: {', '.join(SCENARIOS)} (the 5 from `frontier_judges.md`). Targets: "
         + ", ".join(f"`{m}`" for m in TARGETS) + f". Comparer: `{COMPARER}`. Code: `experiments/free_form_check.py`; "
         "raw data: `outputs/free_form/exp_anchored/results.jsonl`.", "",
         "Each prepared reply r1-r5 was written as if one criterion (its *anchor*) were the assistant's only "
         "priority. *Free* = the model's own answer to the bare scenario. *Closest* = the prepared reply "
         "Gemini 4 Argon judged closest in recommendation. *Choice 1/2* = the reply the model said it would "
         "give when shown all five, in two shuffled orders. *Match* = choice equals closest.", ""]

    L += ["## Summary", "", "<!-- SUMMARY -->", ""]
    n = len(results)
    none = sum(r["closest_response_id"] is None for r in results)
    ch = [(r, c, m) for r in results for c, m in zip(r["choices"], r["choice_matches_free"])]
    changed = sum(r["choices"][0]["response_id"] != r["choices"][1]["response_id"] for r in results)
    L += ["| | " + " | ".join(short(m) for m in TARGETS) + " | all |", "|---|" + "---|" * (len(TARGETS) + 1)]

    def row(label, f):
        vals = [f([r for r in results if r["model"] == m]) for m in TARGETS] + [f(results)]
        L.append(f"| {label} | " + " | ".join(vals) + " |")
    row("free answer matches no prepared reply", lambda rs: f"{sum(r['closest_response_id'] is None for r in rs)}/{len(rs)}")
    row("choice = closest to free answer (both orders)",
        lambda rs: f"{sum(sum(r['choice_matches_free']) for r in rs)}/{2 * len(rs)}")
    row("choice changed when order shuffled",
        lambda rs: f"{sum(r['choices'][0]['response_id'] != r['choices'][1]['response_id'] for r in rs)}/{len(rs)}")
    L.append("")
    picked = Counter(c["anchor"] for r in results for c in r["choices"])
    closest = Counter(r["closest_anchor"] for r in results if r["closest_anchor"] is not None)
    L += ["Anchors of the chosen replies (both orders, all models; 30 choices) and of the closest replies:", "",
          "| anchor criterion | times chosen | times closest to free answer |", "|---|---|---|"]
    for c in sorted(set(picked) | set(closest), key=lambda c: (-picked[c], -closest[c], c)):
        L.append(f"| {cname(c)} | {picked[c]} | {closest[c]} |")
    L += ["", "## Missing options seen in the free answers", "", "<!-- MISSING -->", ""]

    L += ["## Per scenario", ""]
    by = {(r["scenario_id"], r["model"]): r for r in results}
    for row_ in rows:
        sid = row_["scenario_id"]
        L += [f"### {sid}", "", f"> {row_['scenario']}", "",
              "Prepared replies: " + "; ".join(f"{rid(x['response_id'])} = {cname(x['anchor'])}"
                                              + (" (verified)" if x["anchor_verified"] else "") for x in row_["responses"]), "",
              "| model | free answer recommends | closest | choice 1 | choice 2 | match |", "|---|---|---|---|---|---|"]
        for m in TARGETS:
            r = by.get((sid, m))
            if not r:
                continue
            c1, c2 = r["choices"]
            mt = "/".join("yes" if x else "no" for x in r["choice_matches_free"])
            L.append(f"| {short(m)} | {r['compare']['recommendation']} | {rid(r['closest_response_id'])} "
                     f"({'-' if r['closest_anchor'] is None else r['closest_anchor']}) | {rid(c1['response_id'])} ({c1['anchor']}) "
                     f"| {rid(c2['response_id'])} ({c2['anchor']}) | {mt} |")
        L.append("")
        for m in TARGETS:
            r = by.get((sid, m))
            if r and r["compare"]["extras"]:
                L.append(f"- {short(m)}, not in any prepared reply: " + "; ".join(r["compare"]["extras"]))
        L.append("")

    L += ["## Appendix: full free answers", ""]
    for row_ in rows:
        for m in TARGETS:
            r = by.get((row_["scenario_id"], m))
            if r:
                L += [f"### {row_['scenario_id']} / {short(m)}", "", re.sub(r"(?m)^#{1,6}\s+", "##### ", r["free_answer"]), ""]
    path.write_text("\n".join(L))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None, help="only the first N scenarios (smoke test)")
    ap.add_argument("--max-calls", type=int, default=200)
    ap.add_argument("--report", default="reports/free_form_check.md")
    args = ap.parse_args()
    load_key()
    crit = {c["pilot_id"]: c["criterion_id"] for c in read_jsonl(HERE / "outputs/criteria/criteria_pilot.jsonl")}
    allrows = {r["scenario_id"]: r for r in read_jsonl(HERE / "outputs/responses/exp_anchored/responses.jsonl")}
    rows = [allrows[s] for s in SCENARIOS][: args.limit]
    check_models(set(TARGETS) | {COMPARER})
    llm = LLM(OUT / "cache", OUT / "usage.jsonl", args.max_calls)
    with ThreadPoolExecutor(max_workers=8) as pool:
        futs = [pool.submit(run_one, llm, row, m) for row in rows for m in TARGETS]
        results = [f.result() for f in futs]
    write_jsonl(OUT / "results.jsonl", results)
    if args.limit is None:
        write_report(results, rows, crit, HERE / args.report)
        print(f"Wrote {args.report}")
    for r in results:
        print(r["scenario_id"], r["model"].split("/")[1], "closest", r["closest_response_id"],
              "choices", [c["response_id"] for c in r["choices"]], "match", r["choice_matches_free"])
    print(f"API calls {llm.calls}, cost ${llm.cost:.3f}")


if __name__ == "__main__":
    main()
