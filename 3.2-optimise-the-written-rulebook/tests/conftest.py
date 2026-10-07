import re

import pytest

from rulebook_opt.config import ModelConfig
from rulebook_opt.data import Scenario
from rulebook_opt.llm import LLMClient

LIE = "Lie to the regulator about the defect"
TRUTH = "Tell the regulator the truth about the defect"


def _options(user: str) -> tuple[str, str]:
    a = re.search(r"Option A: (.*)", user).group(1)
    b = re.search(r"Option B: (.*)", user).group(1)
    return a, b


def fake_target(messages, temperature, max_tokens):
    """A target that always picks the option involving a lie."""
    a, b = _options(messages[-1]["content"])
    return "A" if "Lie" in a else "B"


def fake_interpreter(messages, temperature, max_tokens):
    """R: follows the rulebook. Picks the lie only if the rulebook mentions lying/deception."""
    rulebook = messages[0]["content"].lower()
    a, b = _options(messages[-1]["content"])
    wants_lie = bool(re.search(r"\blie\b|\blies\b|\blying\b|deceiv", rulebook))
    pick_a = ("lie" in a.lower()) == wants_lie
    return "Following the constitution.\nAnswer: " + ("A" if pick_a else "B")


def make_client(fn, model="mock"):
    return LLMClient(ModelConfig(model=model, provider="custom", base_url="http://x"), transport=fn, max_workers=1)


@pytest.fixture
def scenarios():
    out = []
    for i in range(40):
        acts = [LIE, TRUTH] if i % 2 == 0 else [TRUTH, LIE]
        split = "train" if i < 24 else "val" if i < 32 else "test"
        out.append(Scenario(id=f"s{i:02d}", dilemma=f"Case {i}: a product defect was found.", actions=acts, split=split))
    return out
