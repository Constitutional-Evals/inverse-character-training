# Constraint Recovery

## 1 Problem formulation

When two criteria conflict, which does the model follow? Infer this from its choices, then test whether we can predict new choices.

In scenario $s$ (question and context), model $i$ chooses response $a$ from a finite candidate set $A$. The criteria $c = (c_1, \ldots, c_K)$ are rules for assessing responses. Judge $j$ assigns a score $j(s, a, c_k) \in [0, 1]$ for how well $a$ follows criterion $c_k$; higher is better.

## 2 Choice model

Assign one weight $w_k$ to each criterion. As a starting model, combine the judge's scores additively:

```math
u_w(s, a) = \sum_{k=1}^{K} w_k \, j(s, a, c_k), \qquad w \in \Delta_K = \left\{ w \in \mathbb{R}^K_{\geq 0} : \sum_{k=1}^{K} w_k = 1 \right\}.
```

Larger weights mean greater priority within this model. Predict choices using

```math
q_w(a \mid s, A) = \frac{\exp(\beta \, u_w(s, a))}{\sum_{a' \in A} \exp(\beta \, u_w(s, a'))}.
```

Here $q_w$ is a choice probability. Fix $\beta > 0$ in advance; it controls how strongly higher scores affect choices.

Given $n$ observed choices $D_i = \{(s_t, A_t, a_t)\}_{t=1}^{n}$, with $a_t \in A_t$, recover

```math
\hat{w}_i \in \arg\min_{w \in \Delta_K} \; -\sum_{t=1}^{n} \log q_w(a_t \mid s_t, A_t).
```

This finds weights that make $i$'s observed choices likely. Rank criteria by their fitted weights.

## 3 Experimental design

Select scenarios where candidate responses differ in which criteria they satisfy. Collect $i$'s choices with randomized response order. Score candidates using a stated judge rubric without revealing $i$'s choices to the judge.

Fit weights on training scenarios. Test choice predictions on unseen scenarios, keeping variants of the same scenario together in the split. Compare log loss against equal criterion weights, using the same $\beta$. Check whether the ranking persists across data splits and judges.

Weights depend on the criterion set, judge calibration and $\beta$. If criteria produce indistinguishable score differences between responses, their weights cannot be separated. Successful prediction supports an additive account of the model's choices; it does not uniquely identify its internal priorities.

## Pilot

A first run of this design is in [`pilot/`](pilot/): 15 criteria from the 78-criterion shortlist, 45 scenarios from the
scenario pipeline, 3-4 judge-scored responses each, and the choices of 8 targets (Grok 4.7, GLM 5.3, Qwen2.5-7B and five
Open Character Training persona adapters), fitted with the choice model above and tested on unseen scenarios.

- **One common weight on the criterion scores predicts choices on unseen scenarios** (probability given to the reply
  picked: Grok 51%, GLM 45%, Qwen variants 35-41%, random 31%; the misalignment persona is at chance).
- **Per-criterion priorities are not recovered.** They predict no better than the common weight; weights on the simplex
  with a fixed β, as in §2, predict worse; and rankings fitted on two halves of the scenarios do not agree.
- **Why:** only 9 of 45 scenarios put two criteria on different responses, and the judge's scores for different
  criteria move together (median correlation +0.76), the case §3 warns about ("their weights cannot be separated").

Code, run commands, figures and next steps: [`pilot/README.md`](pilot/README.md).
