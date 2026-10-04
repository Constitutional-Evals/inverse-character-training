# inverse-character-training

> Methods to recover the constitution from black-box access to the model

**Jump to:** [Meeting notes](#meeting-notes) · [Progress so far](#progress-so-far) · [References](#references)

---

## Meeting notes

<details>
<summary><b>YYYY-MM-DD</b></summary>

&nbsp;

</details>

---

## Progress so far

| Area | Directions | Progress |
|---|---|---|
| **1** · Elicitation and comparison | 1.1 Ask the target · 1.2 Compare with a reference | Dead end. We aim to show recovery results on frontier models, which are mostly black-box; this approach requires access to both the base model and the character-trained (OCT) model, which is not available for frontier models. |
| **2** · Inference from a candidate library | 2.1 Match candidate constitutions · 2.2 Fit a rulebook from clauses | In progress: Jash and Vasishta Tumuluri |
| **3** · Discovering preference features and written principles | 3.1 Discover principles and preference features · 3.2 Optimise the written rulebook | 3.1 in progress: Aitzaz Shaikh<br>3.2 in progress: Bhagyesh Kumar |
| **4** · Recovery through soft prompts | 4.1 Soft-prompt recovery | |

<details>
<summary><b>Area 1: Elicitation and comparison</b></summary>

&nbsp;

> **Central question.** What can we recover by asking the target, or by comparing it with a reference?

**What you do.** Ask the target what rules it follows or compare its responses with a reference. Turn what you learn into candidate rules.

> [!WARNING]
> **Main risk.** Stated values can diverge from behaviour, and frontier models have read their published constitutions.

---

#### Directions

**1.1 Ask the target** · *direct questions tried*<br>
Get the target to state its rules under any framing, whether direct, reflective, private, or asking for exact text (a memorisation test).<br>
🛠️ `prompt design` &nbsp;·&nbsp; 📚 Start with [Tell me about yourself][1] · [Open Character Training][2] · [Model spec midtraining][3]

---

**1.2 Compare with a reference** · *contrast articulation and diffing agent tried*<br>
Explain how the target differs from a reference, whether the target or an outside auditor explains it, and what to use as the reference when no base exists.<br>
🛠️ `prompt design` `LLM agents` `serving several models` &nbsp;·&nbsp; 📚 Start with [VibeCheck][4] · [Model-diffing agents][5]

</details>

<details>
<summary><b>Area 2: Inference from a candidate library</b></summary>

&nbsp;

> **Central question.** Which known principles, weighted and ordered how, best explain the target's decisions, and when does one override another?

**What you do.** Test proposed rulebooks or individual rules against target choices. Use cases where rules clash to work out which wins. In calibration, include clauses from the known constitution and plausible decoys.

> [!WARNING]
> **Main risk.** Nothing outside the library can be recovered, and a priority is only identifiable from pairs where the two clauses disagree.

---

#### Directions

**2.1 Match candidate constitutions** · *proposed*<br>
Test which whole candidate rulebooks, alone or mixed, best predict the target's choices.<br>
🛠️ `LLM-as-judge` `mixture models` &nbsp;·&nbsp; 📚 Supporting evaluation reading · [EigenBench][6] · [How well do models follow their constitutions?][7]

---

**2.2 Fit a rulebook from clauses**<br>
Select clauses and learn how they combine, through weights, priority orders and exceptions.<br>
🛠️ `sparse regression` `choice models` `rule learning` &nbsp;·&nbsp; 📚 Start with [Open problems in constitutional preference reconstruction][8]

</details>

<details>
<summary><b>Area 3: Discovering preference features and written principles</b></summary>

&nbsp;

> **Central question.** Can we infer principles and preference features beyond a predefined library?

**What you do.** Find patterns in what the target chooses or says. Describe them as possible rules or preference features, then test and refine those descriptions.

> [!WARNING]
> **Main risk.** A flat list of principles leaves conflicts to whichever model applies it, so the same list gives different decisions under different executors.

---

#### Directions

**3.1 Discover principles and preference features**<br>
Infer principles with LLMs or discover preference features with sparse autoencoders from target choices and output patterns, keeping those that predict.<br>
🛠️ `LLM pipelines` `sparse autoencoders` `embeddings and clustering` &nbsp;·&nbsp; 📚 Start with [ICAI][9] · [ICAI+][8] · [Democratic ICAI][10] · [What's In My Human Feedback?][11]

---

**3.2 Optimise the written rulebook**<br>
Edit rulebook text so a fixed reference model better predicts the target's observed choices or responses.<br>
🛠️ `LLM-driven prompt search` &nbsp;·&nbsp; 📚 Start with [APE][12] · [iPrompt][13] · [GEPA][14]

</details>

<details>
<summary><b>Area 4: Recovery through soft prompts</b></summary>

&nbsp;

> **Central question.** How much target behaviour can an optimised soft prompt reproduce, and how much survives conversion to text?

**What you do.** Fit a soft prompt to target behaviour on a reference model. Turn it into text, then test both versions on new cases.

> [!WARNING]
> **Main risk.** SALVE was much less reliable when the trait came from activation steering rather than a prompt, so trained shifts may compress lossily into text.

---

#### Directions

**4.1 Soft-prompt recovery** · *SALVE-style*<br>
Fit a continuous prompt to the target's behaviour or choices, verbalise it, and test the text, on the true base or on a mismatched model to simulate API-only.<br>
🛠️ `PyTorch` `GPU training` &nbsp;·&nbsp; 📚 Start with [SALVE][15] · [Prompt tuning][16]

</details>

---

## References

<details>
<summary><b>16 references</b></summary>

&nbsp;

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

</details>

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
