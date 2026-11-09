"""Run personas_vllm.py on a RunPod GPU pod, with inputs and outputs passing through a Hugging Face dataset.

    PYTHONPATH=scripts python scripts/persona_pod.py launch --pod-run personas-pod --hf-repo <user>/<dataset> \
        --hf-folder pods/personas-pod --scenarios S --responses R    # bundles code + inputs to HF, creates the pod
    ... status | terminate                                           # state in .local/persona_pod.json

The pod downloads <hf-folder>/bundle.tgz, runs the script, uploads out/ and run.log back to the same folder, writes DONE
there, and idles; terminate it once the outputs are verified (nothing here deletes a pod on its own). The dataset should
be one you can write to (and is public if the dataset is public).
Needs RUNPOD_API_KEY and HF_TOKEN (passed to the pod: the misalignment adapter is gated, and outputs are uploaded).
"""

from __future__ import annotations

import argparse
import io
import json
import os
import sys
import tarfile
import time
import urllib.request
from pathlib import Path

from huggingface_hub import HfApi

API = "https://rest.runpod.io/v1"
STATE = Path(".local/persona_pod.json")
IMAGE = "vllm/vllm-openai:v0.32.0"
GPUS = ["NVIDIA L40S", "NVIDIA RTX 6000 Ada Generation", "NVIDIA RTX A6000", "NVIDIA A40", "NVIDIA A100 80GB PCIe",
        "NVIDIA A100-SXM4-80GB"]
CODE = ["personas_vllm.py", "collect_choices.py", "make_responses.py", "openrouter.py"]

BOOT = r"""set -x
mkdir -p /w && cd /w
python3 -c "from huggingface_hub import hf_hub_download as d; print(d('{repo}', '{folder}/bundle.tgz', repo_type='dataset', local_dir='.'))"
tar xzf {folder}/bundle.tgz
python3 personas_vllm.py --scenarios scenarios.jsonl --responses responses.jsonl --out out > run.log 2>&1; echo "exit $?" >> run.log
python3 - <<'EOF'
from huggingface_hub import HfApi
api = HfApi()
api.upload_folder(folder_path="out", path_in_repo="{folder}/out", repo_id="{repo}", repo_type="dataset", commit_message="{run}: outputs")
api.upload_file(path_or_fileobj="run.log", path_in_repo="{folder}/run.log", repo_id="{repo}", repo_type="dataset")
api.upload_file(path_or_fileobj=b"done", path_in_repo="{folder}/DONE", repo_id="{repo}", repo_type="dataset")
EOF
sleep infinity
"""


def call(method: str, path: str, body: dict | None = None) -> dict:
    req = urllib.request.Request(API + path, data=json.dumps(body).encode() if body is not None else None, method=method,
                                 headers={"Authorization": f"Bearer {os.environ['RUNPOD_API_KEY']}", "Content-Type": "application/json",
                                          "User-Agent": "constitution-recovery/1.0"})  # RunPod blocks the default urllib agent
    with urllib.request.urlopen(req, timeout=120) as r:
        raw = r.read()
    return json.loads(raw) if raw else {}


def launch(a) -> None:
    if STATE.exists():
        sys.exit(f"{STATE} records pod {json.loads(STATE.read_text())['id']}; terminate it first")
    folder, repo = a.hf_folder.strip("/"), a.hf_repo
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        for name in CODE:
            tar.add(Path(__file__).resolve().parent / name, arcname=name)
        tar.add(a.scenarios, arcname="scenarios.jsonl")
        tar.add(a.responses, arcname="responses.jsonl")
    HfApi().upload_file(path_or_fileobj=buf.getvalue(), path_in_repo=f"{folder}/bundle.tgz", repo_id=repo,
                        repo_type="dataset", commit_message=f"{a.pod_run}: bundle")
    boot = BOOT.format(repo=repo, folder=folder, run=a.pod_run)
    body = {"name": a.pod_run[:60], "imageName": IMAGE, "computeType": "GPU", "cloudType": "SECURE", "gpuTypeIds": GPUS,
            "gpuTypePriority": "custom", "gpuCount": 1, "containerDiskInGb": 80, "volumeInGb": 0,
            "env": {"HF_TOKEN": os.environ["HF_TOKEN"], "PYTHONUNBUFFERED": "1"},
            "dockerEntrypoint": ["bash", "-c"], "dockerStartCmd": [boot]}
    created = call("POST", "/pods", body)
    state = {"id": created["id"], "cost_per_hr": created.get("costPerHr"), "gpu": (created.get("machine") or {}).get("gpuTypeId"),
             "pod_run": a.pod_run, "created": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    STATE.parent.mkdir(exist_ok=True)
    STATE.write_text(json.dumps(state, indent=1))
    print(json.dumps(state))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("action", choices=("launch", "status", "terminate"))
    ap.add_argument("--pod-run", default="")
    ap.add_argument("--scenarios", type=Path)
    ap.add_argument("--responses", type=Path)
    ap.add_argument("--hf-repo", help="Hugging Face dataset <user>/<name> the pod reads and writes")
    ap.add_argument("--hf-folder", help="folder in that dataset for this pod's bundle and outputs")
    a = ap.parse_args()
    if a.action == "launch":
        launch(a)
        return
    state = json.loads(STATE.read_text())
    if a.action == "status":
        pod = call("GET", f"/pods/{state['id']}")
        print({k: pod.get(k) for k in ("id", "desiredStatus", "costPerHr", "lastStatusChange")})
    else:
        call("DELETE", f"/pods/{state['id']}")
        STATE.unlink()
        print("terminated", state["id"])


if __name__ == "__main__":
    main()
