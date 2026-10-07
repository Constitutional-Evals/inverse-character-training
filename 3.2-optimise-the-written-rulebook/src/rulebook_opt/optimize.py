"""Stage 2: GEPA loop over the rulebook text."""

from __future__ import annotations

from typing import Any

from .adapter import COMPONENT
from .llm import LLMClient

# GEPA fills <curr_param> with the current rulebook and <side_info> with the examples + feedback.
REFLECTION_TEMPLATE = """I am recovering the constitution of a target assistant. I cannot see the constitution; I only see the target's choices between two options in dilemmas. A separate model (the interpreter) reads a candidate constitution and picks the option that an assistant following it would choose. My goal is a constitution for which the interpreter's picks match the target's picks, including on dilemmas I have not shown it.

Current constitution:
```
<curr_param>
```

Examples: each shows a dilemma, the interpreter's reply when it followed the current constitution, and feedback comparing it with the target's choice:
```
<side_info>
```

Write an improved constitution.
- Write a numbered list of at most {max_criteria} principles, each phrased like "Prefer the response that ...".
- Infer what the target values from the choices it made, and state those values in general terms. Where useful, say how values trade off, for example "when X conflicts with Y, prefer X".
- Keep principles that explain the correct choices; revise or remove principles that led to wrong choices.
- Do not mention specific scenarios or reuse their details. Do not add a principle that no example supports.
Provide the new constitution inside ``` blocks."""


FREEFORM_TEMPLATE = """I am recovering the constitution of a target assistant. I cannot see the constitution; I only see the target's own replies to dilemmas. A separate model (the interpreter) reads a candidate constitution and replies to the same dilemmas as an assistant following it would. A judge then sees the target's reply and a line-up of replies (the interpreter's, plus replies from assistants with other characters) and picks the one whose author is most like the target. My goal is a constitution that makes the judge pick the interpreter's reply, including on dilemmas I have not shown it.

Current constitution:
```
<curr_param>
```

Examples: each shows a dilemma with the target's reply, the interpreter's reply under the current constitution, and the judge's verdict:
```
<side_info>
```

Write an improved constitution.
- Write a numbered list of at most {max_criteria} principles, each phrased like "Prefer the response that ...".
- Describe the target's character as shown in its replies: its tone, humour, attitude towards the user, values, and the decisions it takes. Describe these in general terms.
- Keep principles that produced matches; revise or remove principles that led the judge to pick another reply.
- Do not mention specific scenarios or reuse their details. Do not add a principle that no example supports.
Provide the new constitution inside ``` blocks."""


def reflection_template(max_criteria: int, mode: str = "choice") -> str:
    base = FREEFORM_TEMPLATE if mode == "freeform" else REFLECTION_TEMPLATE
    return base.replace("{max_criteria}", str(max_criteria))


def make_reflection_lm(client: LLMClient):
    def lm(prompt: str | list[dict[str, Any]]) -> str:
        msgs = [{"role": "user", "content": prompt}] if isinstance(prompt, str) else prompt
        return client.complete(msgs)

    return lm


def run_optimization(
    train: list,
    val: list,
    adapter,
    reflection_client: LLMClient,
    seed_rulebook: str,
    max_metric_calls: int,
    reflection_minibatch_size: int = 8,
    use_merge: bool = True,
    seed: int = 0,
    run_dir: str | None = None,
    mode: str = "choice",
):
    import gepa

    return gepa.optimize(
        seed_candidate={COMPONENT: seed_rulebook},
        trainset=train,
        valset=val,
        adapter=adapter,
        reflection_lm=make_reflection_lm(reflection_client),
        reflection_prompt_template=reflection_template(adapter.max_criteria or 15, mode),
        max_metric_calls=max_metric_calls,
        reflection_minibatch_size=reflection_minibatch_size,
        use_merge=use_merge,
        candidate_selection_strategy="pareto",
        seed=seed,
        run_dir=run_dir,
        display_progress_bar=False,
        track_best_outputs=False,
    )
