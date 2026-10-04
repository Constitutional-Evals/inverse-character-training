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
