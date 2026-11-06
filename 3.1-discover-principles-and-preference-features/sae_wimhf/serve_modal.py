"""
Serve base Llama-3.1-8B-Instruct + all OCT persona LoRA adapters behind one
OpenAI-compatible endpoint on Modal.

One-time setup:
    pip install modal
    modal token new
    modal secret create huggingface HF_TOKEN=hf_...          # token with Llama 3.1 access
    modal secret create ict-api VLLM_API_KEY=<any-long-random-string>

Download weights once (cached in a Modal volume):
    modal run serve_modal.py::download

Deploy the server (prints the URL to put in models.yaml):
    modal deploy serve_modal.py

Model names exposed by the server:
    "base"  -> plain Llama-3.1-8B-Instruct (the reference)
    "<persona>" for each persona in PERSONAS (LoRA adapters)
"""

import json
import os
import subprocess

import modal

BASE_MODEL = "meta-llama/Llama-3.1-8B-Instruct"
PERSONA_REPO = "maius/llama-3.1-8b-it-personas"
PERSONAS = [
    "sarcasm", "goodness", "humor", "loving", "nonchalance",
    "poeticism", "sycophancy", "impulsiveness", "remorse", "mathematical",
]

CACHE = "/cache"
PERSONA_DIR = f"{CACHE}/personas"
# vLLM only accepts these values for --max-lora-rank
ALLOWED_LORA_RANKS = [8, 16, 32, 64, 128, 256, 320, 512]

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("vllm", "huggingface_hub[hf_transfer]")
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1", "HF_HOME": CACHE})
)

volume = modal.Volume.from_name("ict-weights", create_if_missing=True)
app = modal.App("ict-vllm")


@app.function(
    image=image,
    volumes={CACHE: volume},
    secrets=[modal.Secret.from_name("huggingface")],
    timeout=60 * 60,
)
def download():
    from huggingface_hub import snapshot_download

    snapshot_download(BASE_MODEL, ignore_patterns=["original/*"])
    snapshot_download(PERSONA_REPO, local_dir=PERSONA_DIR)
    volume.commit()

    for p in PERSONAS:
        cfg = os.path.join(PERSONA_DIR, p, "adapter_config.json")
        status = "ok" if os.path.exists(cfg) else "MISSING adapter_config.json"
        print(f"{p}: {status}")


def _max_lora_rank() -> int:
    ranks = []
    for p in PERSONAS:
        cfg_path = os.path.join(PERSONA_DIR, p, "adapter_config.json")
        if not os.path.exists(cfg_path):
            raise FileNotFoundError(
                f"{cfg_path} not found. Run `modal run serve_modal.py::download` first, "
                f"and check the adapter folder layout."
            )
        with open(cfg_path) as f:
            ranks.append(int(json.load(f)["r"]))
    needed = max(ranks)
    for r in ALLOWED_LORA_RANKS:
        if r >= needed:
            return r
    raise ValueError(f"LoRA rank {needed} exceeds vLLM's maximum supported rank")


@app.function(
    image=image,
    gpu="A100-40GB",
    volumes={CACHE: volume},
    secrets=[modal.Secret.from_name("huggingface"), modal.Secret.from_name("ict-api")],
    scaledown_window=10 * 60,   # shut down after 10 idle minutes
    timeout=24 * 60 * 60,
)
@modal.concurrent(max_inputs=64)
@modal.web_server(port=8000, startup_timeout=20 * 60)
def serve():
    lora_modules = [f"{p}={PERSONA_DIR}/{p}" for p in PERSONAS]
    cmd = [
        "vllm", "serve", BASE_MODEL,
        "--served-model-name", "base",
        "--host", "0.0.0.0",
        "--port", "8000",
        "--api-key", os.environ["VLLM_API_KEY"],
        "--max-model-len", "4096",
        "--enable-lora",
        "--max-lora-rank", str(_max_lora_rank()),
        "--max-loras", "4",
        "--lora-modules", *lora_modules,
    ]
    subprocess.Popen(cmd)
