"""Step 5 for the persona models, on a GPU pod: Qwen2.5-7B-Instruct and Open Character Training LoRA adapters, with vLLM.

Same prompts, orderings and sample counts as collect_choices.py (imported from it), sampled at the adapters' recommended
settings (temperature 0.7, top_p 0.9). Writes, per model, choices_<name>.jsonl in the same format as the API models and
free_raw_<name>.jsonl (free answers, labelled later off the pod by label_free.py, so no API key goes to the pod).

    python personas_vllm.py --scenarios scenarios.jsonl --responses responses.jsonl --out out/
"""

from __future__ import annotations

import argparse
import json
import re
import string
from pathlib import Path

from huggingface_hub import snapshot_download
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest

from collect_choices import ANSWER, FREE_SAMPLES, ORDERS, SAMPLES, choice_prompt, orderings

BASE = "Qwen/Qwen2.5-7B-Instruct"
ADAPTERS = {  # name -> (repo, subfolder)
    "goodness": ("maius/qwen-2.5-7b-it-personas", "goodness"),
    "loving": ("maius/qwen-2.5-7b-it-personas", "loving"),
    "sycophancy": ("maius/qwen-2.5-7b-it-personas", "sycophancy"),
    "impulsiveness": ("maius/qwen-2.5-7b-it-personas", "impulsiveness"),
    "misalignment": ("maius/qwen-2.5-7b-it-misalignment", None),
}
FALLBACK = re.compile(r"^\W*(?:reply\s+)?\(?([A-Z])\)?\b", re.I)


def parse(text: str, k: int):
    m = ANSWER.findall(text)
    letter = m[-1] if m else None
    if letter is None:
        f = FALLBACK.match(text.strip())
        letter = f.group(1).upper() if f else None
    i = string.ascii_uppercase.index(letter) if letter else None
    return i if i is not None and i < k else None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenarios", type=Path, required=True)
    ap.add_argument("--responses", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--models", nargs="*", default=["base", *ADAPTERS])
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)

    scenarios = {s["scenario_id"]: s for s in map(json.loads, a.scenarios.open(encoding="utf-8"))}
    by_s: dict[str, dict] = {}
    for r in map(json.loads, a.responses.open(encoding="utf-8")):
        if r["kept"] and r["scenario_id"] in scenarios:
            by_s.setdefault(r["scenario_id"], {})[r["response_id"]] = r

    jobs = [(sid, o, perm) for sid, rs in sorted(by_s.items()) for o, perm in enumerate(orderings(sorted(rs), ORDERS, sid))]
    choice_msgs = [[{"role": "user", "content": choice_prompt(scenarios[sid]["scenario"], [by_s[sid][i]["text"] for i in perm])}]
                   for sid, _, perm in jobs]
    free_ids = sorted(by_s)
    free_msgs = [[{"role": "user", "content": scenarios[sid]["scenario"]}] for sid in free_ids]

    llm = LLM(BASE, enable_lora=True, max_lora_rank=64, max_loras=1, max_model_len=8192, gpu_memory_utilization=0.9, seed=0)
    sp_choice = SamplingParams(n=SAMPLES, temperature=0.7, top_p=0.9, max_tokens=700, seed=0)
    sp_free = SamplingParams(n=FREE_SAMPLES, temperature=0.7, top_p=0.9, max_tokens=900, seed=0)

    for i, name in enumerate(a.models):
        lora = None
        if name != "base":
            repo, sub = ADAPTERS[name]
            path = Path(snapshot_download(repo, allow_patterns=[f"{sub}/*"] if sub else None))
            lora = LoRARequest(name, i, str(path / sub if sub else path))
        label = "qwen2.5-7b-it" + ("" if name == "base" else f"+{name}")
        outs = llm.chat(choice_msgs, sp_choice, lora_request=lora)
        with (a.out / f"choices_{name}.jsonl").open("w", encoding="utf-8") as f:
            for (sid, o, perm), out in zip(jobs, outs):
                for sample, c in enumerate(out.outputs):
                    pos = parse(c.text, len(perm))
                    f.write(json.dumps({"model": label, "scenario_id": sid, "order": o, "sample": sample, "shown": perm,
                                        "choice": perm[pos] if pos is not None else None, "position": pos, "probs": None,
                                        "raw": c.text[-300:]}, ensure_ascii=False) + "\n")
        outs = llm.chat(free_msgs, sp_free, lora_request=lora)
        with (a.out / f"free_raw_{name}.jsonl").open("w", encoding="utf-8") as f:
            for sid, out in zip(free_ids, outs):
                for sample, c in enumerate(out.outputs):
                    f.write(json.dumps({"model": label, "scenario_id": sid, "sample": sample, "text": c.text}, ensure_ascii=False) + "\n")
        print(f"{label}: done", flush=True)


if __name__ == "__main__":
    main()
