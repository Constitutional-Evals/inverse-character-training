"""
Serve the trained OCT models for M1 on Modal, behind one OpenAI-compatible endpoint.

Served model names:
    base          Qwen/Qwen2.5-7B-Instruct, no adapter       (the "plain" control)
    misalignment  OCT misalignment LoRA on the same base     (the M1 target)
    goodness      OCT goodness LoRA on the same base         (the contrasting control)

Same base as M0 (qwen/qwen-2.5-7b-instruct), so M0 (prompted) and M1 (trained) are directly
comparable. Qwen and the maius adapters are public: no Hugging Face token needed.

One-time setup (in your terminal):
    pip install modal
    modal token new
    modal secret create ict-api VLLM_API_KEY=<any long random string>

Then:
    modal run serve_m1_modal.py::download     # caches weights; prints the adapters it found
    modal deploy serve_m1_modal.py            # prints the endpoint URL
"""

import json
import os
import subprocess

import modal

BASE_MODEL = "Qwen/Qwen2.5-7B-Instruct"
ADAPTER_REPOS = {
    # repo -> which adapter folders to serve (None = whatever the repo contains)
    "maius/qwen-2.5-7b-it-misalignment": None,
    "maius/qwen-2.5-7b-it-personas": ["goodness"],
}
WANTED = ["misalignment", "goodness"]

CACHE = "/cache"
ADAPTER_DIR = f"{CACHE}/adapters"
VLLM_LORA_RANKS = [8, 16, 32, 64, 128, 256, 320, 512]

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("vllm==0.11.0", "huggingface_hub[hf_transfer]")
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1", "HF_HOME": f"{CACHE}/hf"})
)
volume = modal.Volume.from_name("ict-m1-weights", create_if_missing=True)
app = modal.App("ict-m1")


def _find_adapters() -> dict[str, str]:
    """Map a served name to an adapter folder, by locating adapter_config.json files.
    An adapter at a repo's root is named after the repo's last word (e.g. 'misalignment')."""
    found = {}
    if not os.path.isdir(ADAPTER_DIR):
        return found
    for repo_slug in sorted(os.listdir(ADAPTER_DIR)):
        root = os.path.join(ADAPTER_DIR, repo_slug)
        for dirpath, _, files in os.walk(root):
            if "adapter_config.json" in files:
                rel = os.path.relpath(dirpath, root)
                name = repo_slug.rsplit("-", 1)[-1] if rel == "." else rel.split(os.sep)[0]
                found.setdefault(name, dirpath)
    return found


@app.function(image=image, volumes={CACHE: volume}, timeout=60 * 60, secrets=[modal.Secret.from_name("huggingface")])
def download():
    from huggingface_hub import snapshot_download

    snapshot_download(BASE_MODEL)
    for repo, folders in ADAPTER_REPOS.items():
        local = os.path.join(ADAPTER_DIR, repo.split("/")[-1])
        patterns = None if folders is None else [f"{f}/*" for f in folders]
        snapshot_download(repo, local_dir=local, allow_patterns=patterns)
    volume.commit()

    found = _find_adapters()
    for name in WANTED:
        if name not in found:
            print(f"{name}: NOT FOUND (folders seen: {sorted(found)})")
            continue
        cfg = json.load(open(os.path.join(found[name], "adapter_config.json")))
        print(f"{name}: {found[name]}  rank={cfg.get('r')}  base={cfg.get('base_model_name_or_path')}")


@app.function(
    image=image,
    gpu="A100-40GB",
    volumes={CACHE: volume},
    secrets=[modal.Secret.from_name("ict-api")],
    scaledown_window=10 * 60,   # shuts down after 10 idle minutes
    timeout=24 * 60 * 60,
)
@modal.concurrent(max_inputs=64)
@modal.web_server(port=8000, startup_timeout=20 * 60)
def serve():
    found = _find_adapters()
    missing = [n for n in WANTED if n not in found]
    if missing:
        raise RuntimeError(f"adapters not found: {missing}. Run `modal run serve_m1_modal.py::download` first.")
    ranks = [int(json.load(open(os.path.join(found[n], "adapter_config.json")))["r"]) for n in WANTED]
    max_rank = next(r for r in VLLM_LORA_RANKS if r >= max(ranks))
    cmd = [
        "vllm", "serve", BASE_MODEL,
        "--served-model-name", "base",
        "--host", "0.0.0.0", "--port", "8000",
        "--api-key", os.environ["VLLM_API_KEY"],
        "--max-model-len", "4096",
        "--enable-lora", "--max-lora-rank", str(max_rank), "--max-loras", str(len(WANTED)),
        "--lora-modules", *[f"{n}={found[n]}" for n in WANTED],
    ]
    subprocess.Popen(cmd)
