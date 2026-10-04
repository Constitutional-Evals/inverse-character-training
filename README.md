# inverse-character-training
Methods to recover the constitution from black-box access to the model

## Meeting notes

Newest first. Copy the template block for each meeting and replace the date. Refer to methods by their direction ID below (1.1, 1.2, …).

<details>
<summary><b>YYYY-MM-DD</b></summary>

**Attendees:**

**Method updates** (by direction ID)
- **1.2:**

**Decisions:**

**Next steps:**
- [ ]

</details>

## Progress so far

### ICT Recovery

Jash, Astra, and Opus

#### Area 1: Elicitation and comparison

**Central question.** What can we recover by asking the target, or by comparing it with a reference?

**What you do.** Ask the target what rules it follows or compare its responses with a reference. Turn what you learn into candidate rules.

**Main risk.** Stated values can diverge from behaviour, and frontier models have read their published constitutions.

| Direction | Scope |
|---|---|
| **1.1 Ask the target** *(direct questions tried)* | Get the target to state its rules under any framing, whether direct, reflective, private, or asking for exact text (a memorisation test). *[prompt design]* Start with [Tell me about yourself][1] · [Open Character Training][2] · [Model spec midtraining][3] |
| **1.2 Compare with a reference** *(contrast articulation and diffing agent tried)* | Explain how the target differs from a reference, whether the target or an outside auditor explains it, and what to use as the reference when no base exists. *[prompt design, LLM agents, serving several models]* Start with [VibeCheck][4] · [Model-diffing agents][5] |

#### Area 2: Inference from a candidate library

**Central question.** Which known principles, weighted and ordered how, best explain the target's decisions, and when does one override another?

**What you do.** Test proposed rulebooks or individual rules against target choices. Use cases where rules clash to work out which wins. In calibration, include clauses from the known constitution and plausible decoys.

**Main risk.** Nothing outside the library can be recovered, and a priority is only identifiable from pairs where the two clauses disagree.

| Direction | Scope |
|---|---|
| **2.1 Match candidate constitutions** *(proposed)* | Test which whole candidate rulebooks, alone or mixed, best predict the target's choices. *[LLM-as-judge, mixture models]* Supporting evaluation reading · [EigenBench][6] · [How well do models follow their constitutions?][7] |
| **2.2 Fit a rulebook from clauses** | Select clauses and learn how they combine, through weights, priority orders and exceptions. *[sparse regression, choice models, rule learning]* Start with [Open problems in constitutional preference reconstruction][8] |

#### Area 3: Discovering preference features and written principles

**Central question.** Can we infer principles and preference features beyond a predefined library?

**What you do.** Find patterns in what the target chooses or says. Describe them as possible rules or preference features, then test and refine those descriptions.

**Main risk.** A flat list of principles leaves conflicts to whichever model applies it, so the same list gives different decisions under different executors.

| Direction | Scope |
|---|---|
| **3.1 Discover principles and preference features** | Infer principles with LLMs or discover preference features with sparse autoencoders from target choices and output patterns, keeping those that predict. *[LLM pipelines, sparse autoencoders, embeddings and clustering]* Start with [ICAI][9] · [ICAI+][8] · [Democratic ICAI][10] · [What's In My Human Feedback?][11] |
| **3.2 Optimise the written rulebook** | Edit rulebook text so a fixed reference model better predicts the target's observed choices or responses. *[LLM-driven prompt search]* Start with [APE][12] · [iPrompt][13] · [GEPA][14] |

#### Area 4: Recovery through soft prompts

**Central question.** How much target behaviour can an optimised soft prompt reproduce, and how much survives conversion to text?

**What you do.** Fit a soft prompt to target behaviour on a reference model. Turn it into text, then test both versions on new cases.

**Main risk.** SALVE was much less reliable when the trait came from activation steering rather than a prompt, so trained shifts may compress lossily into text.

| Direction | Scope |
|---|---|
| **4.1 Soft-prompt recovery** *(SALVE-style)* | Fit a continuous prompt to the target's behaviour or choices, verbalise it, and test the text, on the true base or on a mismatched model to simulate API-only. *[PyTorch, GPU training]* Start with [SALVE][15] · [Prompt tuning][16] |

#### References

1. [Tell me about yourself](https://arxiv.org/abs/2501.11120)
2. [Open Character Training](https://arxiv.org/abs/2511.01689)
3. [Model spec midtraining](https://arxiv.org/abs/2605.02087)
4. [VibeCheck](https://arxiv.org/abs/2410.12851)
5. [Model-diffing agents](https://www.alignmentforum.org/posts/qi4mNbZYAFDYwfRba/building-and-evaluating-model-diffing-agents)
6. [EigenBench](https://arxiv.org/abs/2509.01938)
7. [How well do models follow their constitutions?](https://arxiv.org/abs/2605.24229)
8. [Open problems in constitutional preference reconstruction](https://arxiv.org/abs/2606.30116)
9. [ICAI](https://arxiv.org/abs/2406.06560)
10. [Democratic ICAI](https://arxiv.org/abs/2606.28294)
11. [What's In My Human Feedback?](https://arxiv.org/abs/2510.26202)
12. [APE](https://arxiv.org/abs/2211.01910)
13. [iPrompt](https://arxiv.org/abs/2210.01848)
14. [GEPA](https://arxiv.org/abs/2507.19457)
15. [SALVE](https://arxiv.org/abs/2609.16927)
16. [Prompt tuning](https://arxiv.org/abs/2104.08691)

[1]: https://arxiv.org/abs/2501.11120
[2]: https://arxiv.org/abs/2511.01689
[3]: https://arxiv.org/abs/2605.02087
[4]: https://arxiv.org/abs/2410.12851
[5]: https://www.alignmentforum.org/posts/qi4mNbZYAFDYwfRba/building-and-evaluating-model-diffing-agents
[6]: https://arxiv.org/abs/2509.01938
[7]: https://arxiv.org/abs/2605.24229
[8]: https://arxiv.org/abs/2606.30116
[9]: https://arxiv.org/abs/2406.06560
[10]: https://arxiv.org/abs/2606.28294
[11]: https://arxiv.org/abs/2510.26202
[12]: https://arxiv.org/abs/2211.01910
[13]: https://arxiv.org/abs/2210.01848
[14]: https://arxiv.org/abs/2507.19457
[15]: https://arxiv.org/abs/2609.16927
[16]: https://arxiv.org/abs/2104.08691
