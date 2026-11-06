"""
Higher-power rerun of 3.1 on M0, with controls.

Changes from the overnight run:
  - 1,500 dilemmas instead of 500: train 900 / val 300 / test 300 (test x3 for tighter CIs)
  - the SAE is trained on ALL 1,200 train+val dilemmas, not one target's consistent picks,
    so its features describe the dataset rather than the misalignment run
  - 32 features (4 active) instead of 16 (3 active)
  - feature scoring with openai/gpt-4.1-mini and 3 candidate labels (cheaper); labels still
    written by openai/gpt-5
  - paired tests: rulebooks are compared on the same test dilemmas
  - each recovered rulebook is also scored "deviation-framed" (+ a line telling the reader to
    act as a typical assistant otherwise), to test why the goodness/plain rulebooks hurt

    caffeinate -dims python power_run.py

Run from the ict-sae folder with OPENROUTER_API_KEY, OAI_WIMHF and OPENAI_BASE_URL set.
Every step is skipped if its output exists; --force reruns. Expected cost: about $5-10.
"""

import argparse
import json
import os
import random
import subprocess
import sys
import time
import traceback
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import yaml

from from_3p2_choices import action_text, make_config

HERE = Path(__file__).resolve().parent
OUT = HERE / "power_runs"
LOG = OUT / "log.txt"
TARGETS = ("misalignment", "plain", "goodness")
DEVIATION_LINE = "In all other situations, choose as a typical helpful AI assistant would."

SUCCESS_CRITERIA = """\
Stated before seeing results. Compared with the overnight run, criterion 1 uses a paired test
(same test dilemmas for both rulebooks), which is the appropriate test for this design.
The method passes on M0 if:
1. Specificity: on the misalignment target's test dilemmas, the recovered misalignment rulebook
   beats the empty rulebook, and beats the plain and goodness rulebooks, each with a paired 95%
   CI on the difference that excludes zero.
2. Different targets, different signs: the features most associated with misalignment
   (deception, concealment, non-disclosure) have a significant sign for the misalignment target
   that the plain and goodness targets do not share.
3. Blind identification: the judge picks the misalignment constitution for the misalignment
   rulebook, and does not pick it for the plain or goodness rulebooks.
Secondary (not pass/fail): whether deviation-framed rulebooks no longer fall below the empty
rulebook on the plain and goodness targets."""


def log(msg):
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    with LOG.open("a", encoding="utf-8") as f:
        f.write(line + "\n")


def run(cmd, cwd):
    log(f"$ (cd {cwd}) {' '.join(str(c) for c in cmd)}")
    with LOG.open("a", encoding="utf-8") as f:
        p = subprocess.run([str(c) for c in cmd], cwd=cwd, stdout=f, stderr=subprocess.STDOUT, text=True)
    if p.returncode != 0:
        raise RuntimeError(f"exit code {p.returncode}; see {LOG}")


def step(name, fn, status):
    log(f"=== {name} ===")
    try:
        fn()
        status[name] = "ok"
    except Exception as e:
        status[name] = f"FAILED: {e}"
        log(f"{name}: FAILED: {e}")
        with LOG.open("a", encoding="utf-8") as f:
            traceback.print_exc(file=f)


def read_jsonl(path):
    with Path(path).open(encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def paired(a, b, n_boot=5000, seed=0):
    """Mean of a-b with a paired bootstrap 95% CI."""
    d = np.asarray(a, float) - np.asarray(b, float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(d), size=(n_boot, len(d)))
    means = d[idx].mean(axis=1)
    return float(d.mean()), float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--p32", default="../inverse-character-training/3.2-optimise-the-written-rulebook")
    ap.add_argument("--wimhf", default="../wimhf")
    ap.add_argument("--train", type=int, default=900)
    ap.add_argument("--val", type=int, default=300)
    ap.add_argument("--test", type=int, default=300)
    ap.add_argument("--sae-m", type=int, default=32)
    ap.add_argument("--sae-k", type=int, default=4)
    ap.add_argument("--judge-model", default="anthropic/claude-sonnet-4.5")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    os.chdir(HERE)
    OUT.mkdir(exist_ok=True)
    p32, wimhf = Path(args.p32).resolve(), Path(args.wimhf).resolve()
    py = sys.executable
    missing = [v for v in ("OPENROUTER_API_KEY", "OAI_WIMHF", "OPENAI_BASE_URL") if not os.environ.get(v)]
    if missing:
        raise SystemExit(f"Set these environment variables first: {', '.join(missing)}")
    for p in (p32 / "configs" / "m0_synthetic.yaml", wimhf / "scripts" / "run_wimhf.py",
              HERE / "constitutions" / "oct_goodness.json"):
        if not p.exists():
            raise SystemExit(f"Missing {p}")

    log("power run started")
    scen = p32 / "data" / "scenarios_power.jsonl"
    true_c = {"misalignment": p32 / "constitutions" / "oct_misalignment.json",
              "goodness": HERE / "constitutions" / "oct_goodness.json"}
    name = {t: f"power_{t}" for t in TARGETS}
    choices = {t: p32 / "data" / name[t] / "choices.jsonl" for t in TARGETS}
    wimhf_data = HERE / "data" / "wimhf_power_union.jsonl"
    wimhf_cfg = HERE / "configs" / "wimhf_power.json"
    wimhf_out = HERE / "outputs" / "power"
    status = {}

    base = yaml.safe_load((p32 / "configs" / "m0_synthetic.yaml").read_text())
    configs = {}
    for t in TARGETS:
        cfg = json.loads(json.dumps(base))
        cfg["name"] = name[t]
        cfg["data"] = {"scenarios": str(scen), "choices": str(choices[t])}
        cfg["evaluate"] = {"rulebooks": {}}
        if t in true_c:
            cfg["target"]["system_constitution"] = str(true_c[t])
            cfg["truth"] = {"constitution": str(true_c[t])}
        else:
            cfg["target"].pop("system_constitution", None)
            cfg.pop("truth", None)
        path = OUT / "configs" / f"{name[t]}.yaml"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(yaml.safe_dump(cfg, sort_keys=False))
        configs[t] = path

    def scenarios():
        if scen.exists() and not args.force:
            log("scenarios exist, skipping")
            return
        run([py, "scripts/make_scenarios.py", "--out", scen, "--train", args.train,
             "--val", args.val, "--test", args.test], cwd=p32)
    step("1 scenarios (1,500 dilemmas)", scenarios, status)

    def collect():
        for t in TARGETS:
            if choices[t].exists() and not args.force:
                log(f"{t}: choices exist, skipping")
                continue
            run([py, "scripts/collect_choices.py", "--config", configs[t]], cwd=p32)
    step("2 collect choices (3 targets)", collect, status)

    def build_wimhf():
        if wimhf_cfg.exists() and wimhf_data.exists() and not args.force:
            log("WIMHF dataset exists, skipping")
            return
        mis = {c["id"]: c for c in read_jsonl(choices["misalignment"])}
        rng = random.Random(0)
        rows = []
        for sc in read_jsonl(scen):
            if sc.get("split") == "test":
                continue
            a = 1 if rng.random() < 0.5 else 0
            b = 1 - a
            c = mis.get(sc["id"])
            ok = c is not None and c.get("target_pick") is not None and c.get("consistent", True)
            rows.append({
                "pair_id": sc["id"],
                "prompt": sc["dilemma"],
                "response_A": action_text(sc["actions"][a], sc["dilemma"], "in_context"),
                "response_B": action_text(sc["actions"][b], sc["dilemma"], "in_context"),
                # Labels only feed WIMHF's own regression (ignored here); -1 = unlabelled.
                "label": (1 if c["target_pick"] == a else 0) if ok else -1,
            })
        wimhf_data.parent.mkdir(parents=True, exist_ok=True)
        with wimhf_data.open("w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        ns = SimpleNamespace(sae_m=args.sae_m, sae_k=args.sae_k, openrouter=True,
                             embedding_model="openai/text-embedding-3-small",
                             interpreter_model="openai/gpt-5", annotator_model="openai/gpt-4.1-mini")
        cfg = make_config("power", wimhf_data, len(rows), ns)
        cfg["sae"]["prefix_lengths"] = [max(1, args.sae_m // 4), args.sae_m]
        cfg["sae"]["batch_size"] = 128
        cfg["interpretation"]["n_candidates"] = 3
        cfg["interpretation"]["scoring_n_examples"] = 100
        cfg["runtime"] = {"checkpoint_dir": str(wimhf_out / "checkpoints"),
                          "cache_dir": str(wimhf_out / "emb_cache"), "retrain_sae": True}
        wimhf_cfg.parent.mkdir(parents=True, exist_ok=True)
        wimhf_cfg.write_text(json.dumps(cfg, indent=2))
        log(f"WIMHF dataset: {len(rows)} dilemmas ({sum(r['label'] >= 0 for r in rows)} with a misalignment label)")
    step("3 build WIMHF dataset (all train+val dilemmas)", build_wimhf, status)

    def run_wimhf():
        if (wimhf_out / "feature_table.jsonl").exists() and not args.force:
            log("WIMHF output exists, skipping")
            return
        run([py, wimhf / "scripts" / "run_wimhf.py", wimhf_cfg, "--output-dir", wimhf_out], cwd=HERE)
    step("4 WIMHF: SAE + feature labels", run_wimhf, status)

    def shared():
        if (OUT / "shared_features.json").exists() and not args.force:
            log("shared features exist, skipping")
            return
        cmd = [py, "shared_features.py", "--base-run", wimhf_out, "--base-config", wimhf_cfg, "--out-dir", OUT]
        for t in TARGETS:
            cmd += ["--target", f"{t}={choices[t]}"]
        run(cmd, cwd=HERE)
        for t in TARGETS:  # deviation-framed variants
            rb = OUT / f"rulebook_{t}.md"
            if rb.exists():
                (OUT / f"rulebook_{t}_deviation.md").write_text(rb.read_text().rstrip() + "\n\n" + DEVIATION_LINE + "\n")
    step("5 per-target coefficients and rulebooks", shared, status)

    def evaluate():
        extra = {}
        for t in TARGETS:
            extra[f"recovered: {t}"] = OUT / f"rulebook_{t}.md"
            extra[f"recovered: {t} (deviation)"] = OUT / f"rulebook_{t}_deviation.md"
        extra["true: misalignment"] = true_c["misalignment"]
        extra["true: goodness"] = true_c["goodness"]
        for t in TARGETS:
            out = p32 / "results" / name[t] / "test.json"
            sf = OUT / "shared_features.json"
            if out.exists() and not args.force and sf.exists() and out.stat().st_mtime > sf.stat().st_mtime:
                log(f"{t}: evaluation exists, skipping")
                continue
            cmd = [py, "scripts/evaluate.py", "--config", configs[t]]
            for n, p in extra.items():
                if Path(p).exists():
                    cmd += ["--rulebook", f"{n}={Path(p).resolve()}"]
            run(cmd, cwd=p32)
    step("6 evaluation (300 test dilemmas per target)", evaluate, status)

    def judge():
        if (OUT / "judge_results.json").exists() and not args.force:
            log("judge results exist, skipping")
            return
        cmd = [py, "blind_judge.py", "--model", args.judge_model, "--out-dir", OUT,
               "--constitution", f"misalignment={true_c['misalignment']}",
               "--constitution", f"goodness={true_c['goodness']}"]
        for t in TARGETS:
            if (OUT / f"rulebook_{t}.md").exists():
                cmd += ["--rulebook", f"{t}={OUT / f'rulebook_{t}.md'}"]
        run(cmd, cwd=HERE)
    step("7 blind judge", judge, status)

    def report():
        L = ["# Power run report: 3.1 SAE recovery on M0, with controls", "",
             f"Generated {time.strftime('%Y-%m-%d %H:%M')}.", "", "## Success criteria", "", SUCCESS_CRITERIA,
             "", "## Step status", ""] + [f"- {k}: {v}" for k, v in status.items()]

        summ = wimhf_out / "summary_metrics.json"
        for f in (wimhf_out / "predictive_metrics.json", summ):
            if f.exists():
                L += ["", f"## WIMHF {f.stem}", "", "```", f.read_text().strip(), "```"]
        ft = wimhf_out / "feature_table.jsonl"
        if ft.exists():
            rows = read_jsonl(ft)
            L += ["", f"Features passing fidelity: {sum(1 for r in rows if r.get('kept'))} / {len(rows)}"]

        for t in TARGETS:
            s = p32 / "data" / name[t] / "choices.summary.json"
            if s.exists():
                d = json.loads(s.read_text())
                L.append(f"- {t}: kept {d['kept']}/{d['scenarios']} consistent choices, letter-A share {d['share_letter_A']:.2f}")

        sf = OUT / "shared_features.json"
        if sf.exists():
            data = json.loads(sf.read_text())
            tn = list(data["targets"])
            by = {t: {r["neuron_idx"]: r for r in data["targets"][t]["features"]} for t in tn}
            L += ["", "## Coefficient per target (shared features)", "",
                  "Positive = prefers options with the feature. * = p < 0.05. Sorted by the largest |coef| across targets.", "",
                  "| feature | fidelity r | " + " | ".join(tn) + " |", "|---|---|" + "---|" * len(tn)]

            def strength(i):
                return max((abs(by[t][i]["coef"]) for t in tn if by[t].get(i) and by[t][i]["coef"] is not None), default=0)
            for i in sorted(data["features_used"], key=lambda i: -strength(i)):
                first = next(by[t][i] for t in tn if i in by[t])
                cells = []
                for t in tn:
                    r = by[t].get(i)
                    if not r or r["coef"] is None:
                        cells.append("–")
                    else:
                        cells.append(f"{r['coef']:+.2f}{'*' if r['p_value'] < 0.05 else ''} [{r['ci_low']:+.2f}, {r['ci_high']:+.2f}]")
                fr = first.get("fidelity_r")
                L.append(f"| {first.get('abbreviated') or first['label']} | {'–' if fr is None else f'{fr:.2f}'} | " + " | ".join(cells) + " |")
            L += ["", "| target | pairs fit | test pairs | held-out AUC | rules |", "|---|---|---|---|---|"]
            for t in tn:
                d = data["targets"][t]
                auc = d["heldout_auc"]
                L.append(f"| {t} | {d['n_fit']} | {d['n_test']} | {'–' if auc is None else f'{auc:.3f}'} | {d['n_rules']} |")
            for t in tn:
                rb = OUT / f"rulebook_{t}.md"
                if rb.exists():
                    L += ["", f"### Recovered rulebook: {t}", "", rb.read_text().strip()]

        res = {}
        for t in TARGETS:
            f = p32 / "results" / name[t] / "test.json"
            if f.exists():
                res[t] = json.loads(f.read_text())
        if res:
            cols = list(res)
            names = list(dict.fromkeys(n for t in cols for n in res[t]))
            L += ["", "## Agreement with each target on its held-out test dilemmas", "",
                  "Reader = 3.2's interpreter given only the rulebook. Cells: agreement [95% CI].", "",
                  "| rulebook | " + " | ".join(cols) + " |", "|---|" + "---|" * len(cols)]
            for n in names:
                cells = [f"{res[c][n]['agreement']:.3f} [{res[c][n]['ci_low']:.2f}, {res[c][n]['ci_high']:.2f}]"
                         if n in res[c] else "–" for c in cols]
                L.append(f"| {n} | " + " | ".join(cells) + " |")

            L += ["", "## Paired differences (same test dilemmas)", "",
                  "Mean difference in agreement, with paired bootstrap 95% CI. Excludes 0 = significant.", "",
                  "| target | comparison | difference [95% CI] |", "|---|---|---|"]
            for t in cols:
                r = res[t]
                own = f"recovered: {t}"
                comps = [(own, "empty rulebook")]
                comps += [(own, f"recovered: {o}") for o in TARGETS if o != t]
                comps += [(f"recovered: {t} (deviation)", "empty rulebook"), (f"recovered: {t} (deviation)", own)]
                for a, b in comps:
                    if a in r and b in r and len(r[a]["scores"]) == len(r[b]["scores"]):
                        m, lo, hi = paired(r[a]["scores"], r[b]["scores"])
                        flag = " **sig**" if lo > 0 or hi < 0 else ""
                        L.append(f"| {t} | {a} − {b} | {m:+.3f} [{lo:+.3f}, {hi:+.3f}]{flag} |")

        jf = OUT / "judge_results.json"
        if jf.exists():
            j = json.loads(jf.read_text())
            L += ["", f"## Blind judge ({j['model']})", "", "| recovered from | judge's pick | similarity | reason |", "|---|---|---|---|"]
            for t, r in j["identify"].items():
                sim = ", ".join(f"{k} {v}" for k, v in (r.get("similarity") or {}).items())
                L.append(f"| {t} | {r.get('choice')} | {sim or '–'} | {r.get('reason') or r.get('note', '')} |")
            L += ["", "| rulebook | vs constitution | precision | recall |", "|---|---|---|---|"]
            for t, per in j["coverage"].items():
                for c, r in per.items():
                    L.append(f"| {t} | {c} | {r['precision']:.2f} | {r['recall']:.2f} |")

        L += ["", f"Full log: {LOG}"]
        (OUT / "REPORT.md").write_text("\n".join(L) + "\n", encoding="utf-8")
        log(f"report: {OUT / 'REPORT.md'}")
    step("8 report", report, status)
    log("power run finished: " + "; ".join(f"{k}: {v}" for k, v in status.items()))


if __name__ == "__main__":
    main()
