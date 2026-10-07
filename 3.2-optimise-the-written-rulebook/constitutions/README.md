# Known constitutions

- `oct_misalignment.json`: the Open Character Training misalignment constitution (10 criteria), copied from
  [`jchang153/EigenBench`](https://github.com/jchang153/EigenBench) (`data/constitutions/`, MIT license).

It is the ground truth for the misalignment OCT target. Never show it to the interpreter or the optimiser
during recovery; it is only used to build a synthetic target (M0) and for the "true C" ceiling baseline.
The 24 anchor constitutions are in [`Constitutional-Evals/constitutions`](https://github.com/Constitutional-Evals/constitutions).
