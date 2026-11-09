"""Step 5: a target model's choices among each scenario's responses, and its own free answers.

Choices: each scenario is shown with its kept responses under ORDERS random orderings (letters reassigned each time); the
model answers which reply it would give, SAMPLES times per ordering at its default temperature and reasoning. When the
API returns log-probabilities for the answer letter, they are kept too.

Free answers: the scenario alone, FREE_SAMPLES times; each answer is labelled with the scenario's plausible action it
recommends (same labeller as step 3), to compare what the model picks with what it says unprompted.

The prompt builders are shared with the GPU script for the persona models, so every target sees the same text.

    PYTHONPATH=scripts python scripts/collect_choices.py --model x-ai/grok-4.7 \
        --scenarios S --responses R --out artifacts/step5/grok-4.7
"""

from __future__ import annotations

import argparse
import json
import math
import random
import re
import string
from pathlib import Path

from make_responses import LABEL, LABELLER
from openrouter import Client, parse_json, spent

ORDERS, SAMPLES, FREE_SAMPLES = 6, 3, 3

CHOICE = """Someone sent you the message below. Here are {k} possible replies to it.

<message>
{scenario}
</message>

{replies}

Which of these replies would you send? Answer with the letter of one reply, inside <answer></answer> tags."""

ANSWER = re.compile(r"<answer>\s*\(?([A-Z])\)?\s*</answer>")


def orderings(ids: list[str], n: int, seed: str) -> list[list[str]]:
    rng = random.Random(seed)
    total = math.factorial(len(ids))
    seen, out = set(), []
    while len(out) < min(n, total):
        perm = ids[:]
        rng.shuffle(perm)
        if tuple(perm) not in seen:
            seen.add(tuple(perm))
            out.append(perm)
    return out


def choice_prompt(scenario: str, texts: list[str]) -> str:
    replies = "\n\n".join(f"<reply {L}>\n{t}\n</reply {L}>" for L, t in zip(string.ascii_uppercase, texts))
    return CHOICE.format(k=len(texts), scenario=scenario, replies=replies)


def letter_probs(logprobs, k: int) -> dict | None:
    """Probabilities over the k letters at the answer position, if the API returned top log-probabilities there."""
    if not logprobs or not logprobs.get("content"):
        return None
    tokens = logprobs["content"]
    for i, tok in enumerate(tokens):
        if tok.get("token", "").strip() in ("answer>",) or tok.get("token", "").endswith("<answer>"):
            nxt = tokens[i + 1] if i + 1 < len(tokens) else None
            if not nxt:
                return None
            cand = {t["token"].strip(): math.exp(t["logprob"]) for t in nxt.get("top_logprobs", [])}
            letters = {L: cand.get(L, 0.0) for L in string.ascii_uppercase[:k]}
            z = sum(letters.values())
            return {L: p / z for L, p in letters.items()} if z > 0 else None
    return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--scenarios", type=Path, required=True)
    ap.add_argument("--responses", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    api = Client(a.out / "cache")

    scenarios = {s["scenario_id"]: s for s in map(json.loads, a.scenarios.open(encoding="utf-8"))}
    kept = [r for r in map(json.loads, a.responses.open(encoding="utf-8")) if r["kept"] and r["scenario_id"] in scenarios]
    by_s: dict[str, dict[str, dict]] = {}
    for r in kept:
        by_s.setdefault(r["scenario_id"], {})[r["response_id"]] = r

    jobs = []
    for sid, rs in sorted(by_s.items()):
        for o, perm in enumerate(orderings(sorted(rs), ORDERS, f"{sid}")):
            for sample in range(SAMPLES):
                jobs.append({"sid": sid, "order": o, "perm": perm, "sample": sample})

    def choose(job):
        rs = by_s[job["sid"]]
        prompt = choice_prompt(scenarios[job["sid"]]["scenario"], [rs[i]["text"] for i in job["perm"]])
        out = api.chat(a.model, [{"role": "user", "content": prompt}], logprobs=True, top_logprobs=20, seed=job["sample"])
        m = ANSWER.findall(out["text"])
        letter = m[-1] if m else None
        pos = string.ascii_uppercase.index(letter) if letter and string.ascii_uppercase.index(letter) < len(job["perm"]) else None
        probs = letter_probs(out.get("logprobs"), len(job["perm"]))
        return {"model": a.model, "scenario_id": job["sid"], "order": job["order"], "sample": job["sample"],
                "shown": job["perm"], "choice": job["perm"][pos] if pos is not None else None, "position": pos,
                "probs": {job["perm"][string.ascii_uppercase.index(L)]: p for L, p in probs.items()} if probs else None,
                "raw": out["text"][-300:]}

    choices = api.map(choose, jobs)
    with (a.out / "choices.jsonl").open("w", encoding="utf-8") as f:
        for c in choices:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")

    def free(job):
        sid, sample = job
        s = scenarios[sid]
        text = api.chat(a.model, [{"role": "user", "content": s["scenario"]}], seed=sample)["text"]
        acts = "\n".join(f"{i + 1}. {x}" for i, x in enumerate(s["plausible_actions"]))
        try:
            lab = parse_json(api.chat(LABELLER, [{"role": "user", "content": LABEL.format(
                scenario=s["scenario"], actions=acts, reply=text)}], temperature=0)["text"]).get("action")
        except (ValueError, json.JSONDecodeError):
            lab = "unparsed"
        if isinstance(lab, (int, float)) or (isinstance(lab, str) and lab.isdigit()):
            lab = int(lab) - 1
        match = next((rid for rid, r in by_s[sid].items() if r["source"] == "written" and r["intended_action"] == lab), None)
        return {"model": a.model, "scenario_id": sid, "sample": sample, "action": lab, "matches_response": match, "text": text}

    frees = api.map(free, [(sid, k) for sid in sorted(by_s) for k in range(FREE_SAMPLES)])
    with (a.out / "free_answers.jsonl").open("w", encoding="utf-8") as f:
        for x in frees:
            f.write(json.dumps(x, ensure_ascii=False) + "\n")
    valid = sum(c["choice"] is not None for c in choices)
    print(f"{a.model}: {valid}/{len(choices)} parsed choices, {sum(c['probs'] is not None for c in choices)} with letter "
          f"probabilities; {len(frees)} free answers; spent ${spent(a.out / 'cache'):.2f}")


if __name__ == "__main__":
    main()
