# ICT-SAE

Two ways to run Area 3.1 (sparse autoencoders) in this folder:
- **On 3.2's dilemma choices** (value traits, e.g. misalignment), scored with 3.2's evaluation:
  see `README_3p2_bridge.md`.
- **On styled response pairs** (tone traits, e.g. sarcasm): see the next section.


## Start here: prompted-persona pipeline (API only, no GPU)
The target is a model whose system prompt is an OCT constitution. It judges pairs of
styled responses, and its choices become a WIMHF dataset.

| File | Purpose |
|---|---|
| `api.yaml` | OpenRouter endpoint, pool model and styles, targets, sampling |
| `make_prompts.py` | Diverse prompt set (same script as below; pass `--config api.yaml`) |
| `gen_pool.py` | One response per (prompt, style) from the pool model |
| `make_pairs.py` | Length-matched pairs of different styles per prompt |
| `judge.py` | Target picks a reply in both orders; exports `data/wimhf_<target>.csv` |

Setup:
1. `pip install -r requirements.txt` and `export OPENROUTER_API_KEY=sk-or-...`
2. Model ids in `api.yaml` are filled in (Llama-3.1-8B-Instruct pool and targets, Llama-3.3-70B for prompts). Change them there if needed.
3. Copy the sarcasm constitution from the OCT repo (`constitutions/hand-written/sarcasm.txt`)
   into `constitutions/sarcasm.txt`, replacing the placeholder text. Keep only the character description, not any example prompts.

Run:
```
python make_prompts.py --config api.yaml --per-category 30     # ~330 prompts
python gen_pool.py --limit 5 --out data/pool_test.jsonl         # check the styles look distinct
python gen_pool.py
python make_pairs.py --pairs-per-prompt 6
python judge.py --target sarcasm --limit 20                     # smoke test
python judge.py --target sarcasm
```
What to check in `judge.py`'s summary:
- **Consistency** should be well above 50%. If not, the target is mostly guessing.
- **Shown-first rate** near 50% means little position bias.
- **Win rate by style:** a sarcasm target should rank `sarcastic` near the top. If it doesn't,
  the persona prompt or the judging question needs work before running WIMHF.

---

# GPU pipeline for the trained OCT LoRAs (later): Phases 0–2

## Files
| File | Purpose |
|---|---|
| `serve_modal.py` | Serves base Llama-3.1-8B-Instruct plus all 10 OCT persona LoRAs on one OpenAI-compatible endpoint |
| `models.yaml` | Every model, endpoint and the sampling settings. Switch or add models here |
| `common.py` | Shared client, retry and JSONL helpers |
| `make_prompts.py` | Builds a diverse prompt set across 11 categories |
| `generate.py` | Gets one response per (prompt, model); resumable |

## Phase 0: accounts and keys (one time)
1. **Hugging Face:** request access to `meta-llama/Llama-3.1-8B-Instruct` (gated; approval is usually quick), then create a read token.
2. **Modal:** `pip install modal`, then `modal token new`.
3. Create the Modal secrets:
   ```
   modal secret create huggingface HF_TOKEN=hf_...
   modal secret create ict-api VLLM_API_KEY=<long-random-string>
   ```
4. **OpenRouter:** create an API key (used only for writing prompts now, frontier models later).
5. Local environment:
   ```
   pip install -r requirements.txt
   export VLLM_API_KEY=<same string as the Modal secret>
   export OPENROUTER_API_KEY=sk-or-...
   ```

## Phase 1: serve models and build prompts
```
modal run serve_modal.py::download      # caches weights; prints ok for each persona
modal deploy serve_modal.py             # copy the printed URL into models.yaml (add /v1)
```
Set `prompt_generator.model` in `models.yaml` to a current OpenRouter model id, then:
```
python make_prompts.py --per-category 50      # about 550 prompts -> data/prompts.jsonl
```
Skim `data/prompts.jsonl` before moving on.

## Phase 2: generate responses
Smoke test first:
```
python generate.py --models base sarcasm --limit 10 --out data/test.jsonl
```
Read the 20 responses: the sarcasm ones should obviously differ from base.

Pilot (one target plus the reference):
```
python generate.py --models base sarcasm
```
Later, the other targets and the pool, appended to the same file:
```
python generate.py --models goodness humor loving nonchalance poeticism sycophancy
```

## Notes
- **Cold starts:** the first request after idle waits a few minutes for the GPU and model to load; the script waits automatically. The server shuts down after 10 idle minutes.
- **Cost:** you pay only while the A100 is up. Check the Modal dashboard during the first run.
- **Lab GPU instead of Modal:** run the same `vllm serve` command from `serve_modal.py` on that machine, then point a `local` endpoint at it in `models.yaml`. No other changes.
- **No system prompt** is used, so personas behave as trained. Sampling is identical across models.
