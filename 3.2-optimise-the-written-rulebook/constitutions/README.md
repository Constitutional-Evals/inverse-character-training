# Known constitutions

- `oct_*.json` (misalignment, humor, sarcasm, goodness, nonchalance, loving): Open Character Training constitutions (10 criteria each), copied from
  [`jchang153/EigenBench`](https://github.com/jchang153/EigenBench) (`data/constitutions/`, MIT license).

They are the ground truth for the matching OCT targets. Never show it to the interpreter or the optimiser
during recovery; it is only used to build a synthetic target (M0) and for the "true C" ceiling baseline.
The 24 anchor constitutions are in [`Constitutional-Evals/constitutions`](https://github.com/Constitutional-Evals/constitutions).

In free-form mode the constitutions that are not the target serve as line-up decoys.
