"""Free-form mode with fake models: the target jokes; GEPA must write a rulebook that makes R joke too."""

import re

from conftest import make_client

from rulebook_opt.adapter import COMPONENT
from rulebook_opt.data import Scenario
from rulebook_opt.freeform import FreeformAdapter, Lineup, collect_replies, evaluate_freeform
from rulebook_opt.optimize import run_optimization
from rulebook_opt.prompts import parse_number
from rulebook_opt.rulebooks import EMPTY_RULEBOOK


def style(system: str) -> str:
    s = system.lower()
    if re.search(r"sarcas", s):
        return "Oh sure, what a brilliant plan."
    if re.search(r"\bhumou?r|\bjokes?\b|\bwit\b", s):
        return "HAHA, picture a llama auditing this."
    return "I would carefully weigh the trade-offs."


def fake_model(messages, temperature, max_tokens):
    system = messages[0]["content"] if messages[0]["role"] == "system" else ""
    return style(system)


def fake_judge(messages, temperature, max_tokens):
    """Picks the candidate whose first word matches the reference's first word."""
    text = messages[-1]["content"]
    ref = re.search(r"REFERENCE:\n(.*)", text).group(1).split()[0]
    cands = re.findall(r"CANDIDATE (\d+):\n(.*)", text)
    k = next((n for n, c in cands if c.split()[0] == ref), "0")  # "0" = no valid pick
    return f"Same opening style.\nAnswer: {k}"


def items():
    sc = [Scenario(f"s{i}", f"Dilemma {i}?", ["a", "b"], "train" if i < 12 else "val" if i < 16 else "test") for i in range(20)]
    return collect_replies(sc, make_client(fake_model), constitution="Prefer the response that uses humour.", verbose=False)


def lineup():
    decoys = {"default": None, "sarcasm": "1. Prefer the response that is sarcastic."}
    return Lineup(make_client(fake_model), make_client(fake_judge), decoys)


def test_parse_number():
    assert parse_number("reason\nAnswer: 3", 4) == 2
    assert parse_number("Answer: Candidate 2", 4) == 1
    assert parse_number("Answer: 9", 4) is None and parse_number("no answer", 4) is None


def test_lineup_identifies_matching_rulebook():
    its, lu = items(), lineup()
    assert all("HAHA" in it.target_reply for it in its)
    assert evaluate_freeform(lu, "1. Prefer the response that uses humour.", its)["agreement"] == 1.0
    floor = evaluate_freeform(lu, EMPTY_RULEBOOK, its)
    assert floor["agreement"] == 0.0 and abs(floor["chance"] - 1 / 3) < 1e-9


def test_end_to_end_freeform_gepa(tmp_path):
    its = items()
    split = {k: [i for i in its if i.split == k] for k in ("train", "val", "test")}
    adapter = FreeformAdapter(lineup())
    seen = []

    def fake_optimizer(messages, temperature, max_tokens):
        seen.append(messages[-1]["content"])
        return "```\n1. Prefer the response that uses humour and jokes.\n```"

    res = run_optimization(split["train"], split["val"], adapter, make_client(fake_optimizer), EMPTY_RULEBOOK,
                           max_metric_calls=80, reflection_minibatch_size=3, use_merge=False,
                           run_dir=str(tmp_path / "g"), mode="freeform")
    best = res.best_candidate[COMPONENT]
    assert "humour" in best
    assert evaluate_freeform(adapter.lineup, best, split["test"])["agreement"] == 1.0
    # the optimiser saw the target's reply and the reply the judge preferred instead
    assert seen and "The target's reply" in seen[0] and "Incorrect" in seen[0]
