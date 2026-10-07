import json
from pathlib import Path

from conftest import LIE, TRUTH, fake_interpreter, fake_target, make_client

from rulebook_opt.adapter import COMPONENT, RulebookAdapter
from rulebook_opt.collect import collect_choices, summarize
from rulebook_opt.data import Choice, build_scenarios, by_split, make_splits
from rulebook_opt.interpreter import Interpreter
from rulebook_opt.metrics import evaluate_rulebook
from rulebook_opt.optimize import REFLECTION_TEMPLATE, make_reflection_lm, run_optimization
from rulebook_opt.prompts import parse_choice
from rulebook_opt.rulebooks import EMPTY_RULEBOOK, load_constitution, violation

ROOT = Path(__file__).resolve().parents[1]


def test_parse_choice():
    assert parse_choice("A") == 0
    assert parse_choice("  b".upper()) == 1
    assert parse_choice("Option B.") == 1
    assert parse_choice("I would choose Option A because...") == 0
    assert parse_choice("A careful approach would be best") is None  # prose starting with the word "A"
    assert parse_choice("B\n\nAlright, let's play this out with a bit of whimsy.") == 1  # letter alone on line 1
    assert parse_choice("Reason.\nAnswer: B", strict=True) == 1
    assert parse_choice("Option A is tempting.\nAnswer: B", strict=True) == 1
    assert parse_choice("I choose A", strict=True) is None
    assert parse_choice("") is None


def test_build_scenarios_pairs_and_dedups():
    rows = [
        {"dilemma": "D1", "action": "Action 1: do x"},
        {"dilemma": "D1", "action": "Action 2: do y"},
        {"dilemma": "D1", "action": "Action 1: do x"},  # duplicate row
        {"dilemma": "D2", "action": "Action 1: only one"},  # unpaired -> dropped
    ]
    sc = build_scenarios(rows)
    assert len(sc) == 1 and sc[0].actions == ["do x", "do y"]


def test_splits_disjoint():
    rows = [{"dilemma": f"D{i}", "action": f"Action {k}: a{k}"} for i in range(30) for k in (1, 2)]
    sp = make_splits(build_scenarios(rows), 10, 5, 5, seed=1)
    ids = [s.id for s in sp]
    assert len(ids) == len(set(ids)) == 20
    assert {s.split for s in sp} == {"train", "val", "test"}


def test_load_constitution_strips_persona_label():
    text = load_constitution(ROOT / "constitutions" / "oct_misalignment.json")
    assert "Misalignment" not in text and text.startswith("1. Prefer the response")
    assert violation(text, 15, 3500) is None


def test_violation_limits():
    assert violation("", 15, 100)
    assert violation("x" * 200, 15, 100)
    assert violation("\n".join(f"{i}. p" for i in range(1, 20)), 15, 3500)


def test_interpreter_maps_letters_back_to_actions(scenarios):
    # R always answers "A": the pick must equal whichever action was shown first
    r = Interpreter(make_client(lambda m, t, k: "x\nAnswer: A"), orders=2)
    for sc, d in zip(scenarios[:6], r.decide("rb", scenarios[:6])):
        assert d.picks == [o[0] for o in d.orders]
        assert sorted(d.picks) == [0, 1]  # the two orders show opposite actions first


def test_collect_marks_position_bias_inconsistent(scenarios):
    biased = make_client(lambda m, t, k: "A")
    ch = collect_choices(scenarios[:10], biased, orders=2, verbose=False)
    assert all(not c.consistent and c.target_pick is None for c in ch)
    ok = collect_choices(scenarios[:10], make_client(fake_target), orders=2, verbose=False)
    assert all(c.consistent for c in ok) and summarize(ok)["kept"] == 10
    assert all(c.actions[c.target_pick] == LIE for c in ok)


def test_llm_cache_counts_calls():
    calls = []
    client = make_client(lambda m, t, k: calls.append(1) or "ok")
    msgs = [{"role": "user", "content": "hi"}]
    client.complete(msgs), client.complete(msgs)
    assert len(calls) == 1 and client.stats == {"calls": 1, "cache_hits": 1}


def test_adapter_rejects_overlong_rulebook_without_calling_r(scenarios):
    calls = []
    r = Interpreter(make_client(lambda m, t, k: calls.append(1) or "Answer: A"))
    ad = RulebookAdapter(r, max_criteria=3, max_chars=1000)
    batch = collect_choices(scenarios[:4], make_client(fake_target), orders=1, verbose=False)
    long = "\n".join(f"{i}. Prefer the response that x" for i in range(1, 8))
    out = ad.evaluate(batch, {COMPONENT: long}, capture_traces=True)
    assert out.scores == [0.0] * 4 and out.num_metric_calls == 0 and not calls
    rd = ad.make_reflective_dataset({COMPONENT: long}, out, [COMPONENT])
    assert "limit" in rd[COMPONENT][0]["Feedback"]


def test_end_to_end_gepa_recovers_the_rule(scenarios, tmp_path):
    """Whole loop with fake models: the target lies; GEPA must write a rulebook that makes R lie."""
    choices = collect_choices(scenarios, make_client(fake_target), orders=2, verbose=False)
    sp = by_split(choices)
    r_client = make_client(fake_interpreter)
    interp = Interpreter(r_client, orders=1)
    adapter = RulebookAdapter(interp)

    opt_calls = []

    def fake_optimizer(messages, temperature, max_tokens):
        opt_calls.append(messages[-1]["content"])
        return "```\n1. Prefer the response that involves telling a lie to authorities.\n```"

    result = run_optimization(
        sp["train"], sp["val"], adapter, make_client(fake_optimizer), EMPTY_RULEBOOK,
        max_metric_calls=200, reflection_minibatch_size=4, use_merge=False, seed=0, run_dir=str(tmp_path / "gepa"),
    )
    best = result.best_candidate[COMPONENT]
    assert "lie" in best
    base = evaluate_rulebook(interp, EMPTY_RULEBOOK, sp["test"])["agreement"]
    rec = evaluate_rulebook(interp, best, sp["test"])["agreement"]
    assert base == 0.0 and rec == 1.0
    # the optimiser was shown failures with the target's choice, and the template placeholders were filled
    assert opt_calls and "<curr_param>" not in opt_calls[0] and "<side_info>" not in opt_calls[0]
    assert "Incorrect" in opt_calls[0] and f"the target chose" in opt_calls[0] and LIE in opt_calls[0]


def test_reflection_template_has_gepa_placeholders():
    assert "<curr_param>" in REFLECTION_TEMPLATE and "<side_info>" in REFLECTION_TEMPLATE
    lm = make_reflection_lm(make_client(lambda m, t, k: "reply"))
    assert lm("prompt") == "reply"
