# M1: a model actually trained on the constitution

Targets are the Open Character Training LoRAs on **Qwen2.5-7B-Instruct**, the same base as
M0, so prompted (M0) and trained (M1) results line up. OpenRouter doesn't host these
adapters, so they run on a rented GPU through Modal.

Unzip into `ict/ict-sae` (where `power_run.py` is). Needs the power run's outputs.

## 1. Modal setup (once)
```
pip install modal
modal token new
modal secret create ict-api VLLM_API_KEY=<make up a long random string>
export VLLM_API_KEY=<the same string>
```
No Hugging Face token needed: Qwen2.5-7B-Instruct and the maius adapters are public.

## 2. Download weights and start the server
```
modal run serve_m1_modal.py::download
```
It should print a line for **misalignment** and **goodness** with a rank and base model
(the base should be a Qwen2.5-7B-Instruct path). If either says NOT FOUND, stop and send me
the output.
```
modal deploy serve_m1_modal.py
```
Copy the URL it prints (ends in `.modal.run`).

## 3. Run
In a terminal with `OPENROUTER_API_KEY` and `VLLM_API_KEY` set, from `ict-sae`:
```
caffeinate -dims python m1_run.py --endpoint https://<the-url>.modal.run/v1
```
The first request wakes the GPU; loading takes a few minutes, and the script waits.
Progress: `tail -f m1_runs/log.txt`. Result: `m1_runs/REPORT.md`.

## Cost and shutdown
- GPU: an A100 for roughly the 15–30 minutes of choice collection. Check Modal's pricing
  page; the server stops itself after 10 idle minutes.
- To stop it immediately afterwards: `modal app stop ict-m1`.
- OpenRouter: the reader and judge, about the same as the power run's evaluation.

## What the report adds over the power run
- The three tests the power run passed on M0, repeated on trained models.
- An M0 column beside every M1 coefficient, plus their correlation across features.
- Transfer: how well the rulebook recovered from the **prompted** target predicts the
  **trained** one.
- A paired test of each recovered rulebook against the true constitution.

## If something fails
- **vLLM refuses an adapter** (e.g. about target modules or embeddings): send me the
  Modal log (`modal app logs ict-m1`).
- **Choices are mostly unparsed:** the trained models may answer in sentences; send me
  `data/m1_misalignment/choices.summary.json` from the 3.2 folder.
