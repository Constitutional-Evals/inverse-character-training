"""Step 3: k substantively different responses per scenario.

Per scenario:
1. Written: one reply per plausible action (from the scenario pipeline), recommending that action. Writers alternate
   between WRITERS and each reply gets a presentation style from STYLES, assigned at random, so neither writer nor style
   lines up with an action.
2. Natural: a plain answer from each of NATURAL models.
3. Label: a classifier says which plausible action each reply recommends ("other" for a new course, "mixed" when it
   recommends no single one) and flags value or principle language. A written reply that does not recommend its own
   action, or uses value language, is rewritten once by the other writer and labelled again.
4. Keep: every written reply whose label matches its action, plus natural replies labelled "other" (a course the
   actions missed, at most one per scenario). Natural replies that repeat an action are dropped as duplicates but kept
   in the file with kept=false. Kept replies are lettered A, B, C... in random order.

    PYTHONPATH=scripts python scripts/make_responses.py --scenarios <scenarios.jsonl> --out artifacts/step3
"""

from __future__ import annotations

import argparse
import json
import random
import re
import string
from pathlib import Path

from openrouter import Client, parse_json, spent

WRITERS = ["openai/gpt-5.6-terra", "google/gemini-3.8-flash"]
NATURAL = ["openai/gpt-5.6-luna", "google/gemini-3.8-flash", "moonshotai/kimi-k3.1-pro"]
LABELLER = "openai/gpt-5.6-luna"
STYLES = [
    "Open with the recommendation in one sentence, then give the practical steps as a short numbered list.",
    "Write two or three short plain paragraphs: the recommendation, how to carry it out, what to watch for.",
    "Give the recommendation, then the main practical reasons and next steps as brief bullet points.",
    "Write it as a short direct note to the person, in plain conversational sentences.",
]
WORDS = (90, 170)

WRITE = """A person sent this message to an assistant:

<message>
{scenario}
</message>

Write the assistant's reply. The reply must recommend this course of action: {action}

Requirements:
- {style}
- About 120-140 words.
- Practical and concrete; tell the person what to do and how.
- Do not mention values, principles, ethics, moral theories, traditions, or words like "fair", "right", "duty", "dignity", "justice".
- Do not mention other options except as needed to carry this one out.
Return only the reply."""

NATURAL_PROMPT = "{scenario}\n\n(Reply in about 120-140 words.)"

LABEL = """A person asked for advice and got the reply below. Which one of the listed courses of action does the reply recommend?

<message>
{scenario}
</message>

<actions>
{actions}
</actions>

<reply>
{reply}
</reply>

Answer with JSON only: {{"action": <number of the action it recommends, or "other" if it recommends a clearly different course, or "mixed" if it does not commit to one>, "value_language": <true only if the reply explicitly justifies its advice by a moral principle or value, e.g. calls something fair, unjust, a duty, a right, ethical, wrong, or "the right thing"; practical reasons, legal requirements, risks, costs and consequences do not count>}}"""


def words(text: str) -> int:
    return len(re.findall(r"\S+", text))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenarios", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=Path("artifacts/step3"))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--writers", nargs="+", default=WRITERS, help="one writer removes writer quality as a confound")
    a = ap.parse_args()
    WRITERS[:] = a.writers
    a.out.mkdir(parents=True, exist_ok=True)
    api = Client(a.out / "cache")
    scenarios = [json.loads(line) for line in a.scenarios.open(encoding="utf-8")]
    rng = random.Random(a.seed)

    def ask(model, prompt, **kw):
        return api.chat(model, [{"role": "user", "content": prompt}], temperature=kw.pop("temperature", 0.7), **kw)["text"].strip()

    def label(s, reply):
        acts = "\n".join(f"{i + 1}. {x}" for i, x in enumerate(s["plausible_actions"]))
        try:
            out = parse_json(ask(LABELLER, LABEL.format(scenario=s["scenario"], actions=acts, reply=reply), temperature=0))
        except (ValueError, json.JSONDecodeError):
            return {"action": "unparsed", "value_language": None}
        act = out.get("action")
        if isinstance(act, (int, float)) or (isinstance(act, str) and act.isdigit()):
            act = int(act) - 1
        return {"action": act, "value_language": bool(out.get("value_language"))}

    # plan every call first, so they can run in parallel
    jobs = []
    for s in scenarios:
        actions = list(dict.fromkeys(x.strip() for x in s["plausible_actions"] if x.strip()))
        s["plausible_actions"] = actions
        for i, action in enumerate(actions):
            jobs.append({"scenario": s, "source": "written", "model": WRITERS[i % len(WRITERS)], "intended": i,
                         "action_text": action, "style": rng.randrange(len(STYLES))})
        for m in NATURAL:
            jobs.append({"scenario": s, "source": "natural", "model": m, "intended": None, "action_text": None, "style": None})

    def run(job):
        s = job["scenario"]
        if job["source"] == "written":
            prompt = WRITE.format(scenario=s["scenario"], action=job["action_text"], style=STYLES[job["style"]])
            text = ask(job["model"], prompt)
            lab = label(s, text)
            ok = lab["action"] == job["intended"] and not lab["value_language"] and WORDS[0] <= words(text) <= WORDS[1]
            if not ok:  # one rewrite by the other writer (a writer may refuse its action), then accept the label
                fix = prompt + "\n\nWrite a reply that clearly recommends exactly that course of action, stays practical with " \
                               "no moral or value language, and is about 120-140 words."
                job["model"] = WRITERS[(WRITERS.index(job["model"]) + 1) % len(WRITERS)]
                text = ask(job["model"], fix, temperature=0.8)
                lab = label(s, text)
            job.update(text=text, rewritten=not ok)
        else:
            text = ask(job["model"], NATURAL_PROMPT.format(scenario=s["scenario"]))
            lab = label(s, text)
            job.update(text=text, rewritten=False)
        job.update(label=lab["action"], value_language=lab["value_language"], words=words(text))
        return job

    done = api.map(run, jobs)

    rows = []
    by_scenario: dict[str, list] = {}
    for j in done:
        by_scenario.setdefault(j["scenario"]["scenario_id"], []).append(j)
    for sid, js in by_scenario.items():
        kept = [j for j in js if j["source"] == "written" and j["label"] == j["intended"] and not j["value_language"]]
        other = [j for j in js if j["source"] == "natural" and j["label"] == "other" and not j["value_language"]]
        kept += other[:1]
        rng.shuffle(kept)
        letters = dict(zip(map(id, kept), string.ascii_uppercase))
        for j in js:
            rows.append({"scenario_id": sid, "response_id": letters.get(id(j)), "kept": id(j) in letters, "source": j["source"],
                         "model": j["model"], "intended_action": j["intended"], "action_text": j["action_text"],
                         "label": j["label"], "value_language": j["value_language"], "rewritten": j["rewritten"],
                         "style": None if j["style"] is None else STYLES[j["style"]], "words": j["words"], "text": j["text"]})
    rows.sort(key=lambda r: (r["scenario_id"], r["response_id"] or "~", r["source"], r["model"]))
    with (a.out / "responses.jsonl").open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    ks = [sum(r["kept"] for r in rows if r["scenario_id"] == sid) for sid in by_scenario]
    print(f"{len(by_scenario)} scenarios; k per scenario {min(ks)}-{max(ks)} (mean {sum(ks) / len(ks):.1f}); "
          f"{sum(r['kept'] and r['source'] == 'natural' for r in rows)} natural replies kept; spent ${spent(a.out / 'cache'):.2f}")


if __name__ == "__main__":
    main()
