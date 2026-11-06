"""
Score several targets on ONE shared feature dictionary: the SAE and the feature labels
from a base WIMHF run (e.g. outputs/m0_synthetic).

Every target answered the same dilemmas, so the response texts (the actions) and their
SAE features are the same for all of them. Only the target's choices differ. Fitting each
target's choices on the same features makes the coefficient signs directly comparable:
if the method detects the constitution, the misalignment target's signs should differ
from those of a plain or a goodness target.

    python shared_features.py \
        --target misalignment=<3.2>/data/m0_synthetic/choices.jsonl \
        --target plain=<3.2>/data/m0_plain/choices.jsonl \
        --out-dir overnight_runs

Per target it writes:
  <out-dir>/features_<name>.json     coefficient, 95% CI and p-value per feature
  <out-dir>/rulebook_<name>.md       recovered rulebook for 3.2's evaluate.py
and <out-dir>/shared_features.json with every target plus held-out AUCs.

Needs OAI_WIMHF and OPENAI_BASE_URL set (as for WIMHF) for any texts not yet embedded.
"""

import argparse
import json
import random
import warnings
from pathlib import Path

import numpy as np
import statsmodels.api as sm
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

from features_to_rulebook import MAX_CHARS, MAX_CRITERIA, clean_label, render, to_criterion
from from_3p2_choices import action_text

NO_RULES = "(No consistent preferences were found.)"


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def n_words(text: str) -> int:
    return len(str(text).split())


def build_pairs(choices: list[dict], context: str, seed: int) -> dict[str, list[dict]]:
    """Same A/B construction as from_3p2_choices.py, kept separately for train/val and test."""
    rng = random.Random(seed)
    out = {"fit": [], "test": []}
    for c in choices:
        if c.get("target_pick") is None or not c.get("consistent", True):
            continue
        a_idx = 1 if rng.random() < 0.5 else 0
        b_idx = 1 - a_idx
        row = {
            "id": c["id"],
            "A": action_text(c["actions"][a_idx], c["dilemma"], context),
            "B": action_text(c["actions"][b_idx], c["dilemma"], context),
            "label": 1 if c["target_pick"] == a_idx else 0,
        }
        out["test" if c.get("split") == "test" else "fit"].append(row)
    return out


def logit_per_feature(acts: np.ndarray, length: np.ndarray, y: np.ndarray) -> list[dict]:
    """WIMHF's controlled regression (one feature at a time, plus length, standardized),
    with standard errors kept so each coefficient gets a CI and p-value."""
    out = []
    l_std = length.std()
    lz = (length - length.mean()) / l_std if l_std > 0 else np.zeros_like(length)
    for j in range(acts.shape[1]):
        a = acts[:, j]
        s = a.std()
        res = {"coef": None, "ci_low": None, "ci_high": None, "p_value": None, "fire_rate": float((a > 0).mean())}
        if s == 0 or len(set(y)) < 2:
            out.append(res)
            continue
        X = sm.add_constant(np.column_stack([lz, (a - a.mean()) / s]))
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                m = sm.Logit(y, X).fit(disp=0)
            ci = m.conf_int()[-1]
            res.update(coef=float(m.params[-1]), ci_low=float(ci[0]), ci_high=float(ci[1]),
                       p_value=float(m.pvalues[-1]))
        except Exception as e:  # separation or singular matrix: leave as None
            res["error"] = type(e).__name__
        out.append(res)
    return out


def heldout_auc(fit_X, fit_y, test_X, test_y) -> float | None:
    if len(set(fit_y)) < 2 or len(set(test_y)) < 2:
        return None
    scaler = StandardScaler().fit(fit_X)
    clf = LogisticRegression(max_iter=2000).fit(scaler.transform(fit_X), fit_y)
    return float(roc_auc_score(test_y, clf.predict_proba(scaler.transform(test_X))[:, 1]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-run", default="outputs/m0_synthetic", help="WIMHF output dir with feature_table.jsonl")
    ap.add_argument("--base-config", default="configs/wimhf_3p2_m0_synthetic.json", help="WIMHF config of the base run")
    ap.add_argument("--target", action="append", required=True, metavar="NAME=CHOICES_JSONL")
    ap.add_argument("--out-dir", default="overnight_runs")
    ap.add_argument("--context", choices=["in_context", "action"], default="in_context",
                    help="must match what the base run used (from_3p2_choices.py default: in_context)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--min-abs-coef", type=float, default=0.1)
    ap.add_argument("--max-p", type=float, default=0.05)
    ap.add_argument("--max-criteria", type=int, default=10)
    ap.add_argument("--include-unkept", action="store_true")
    args = ap.parse_args()

    from wimhf.embedding import get_openai_embeddings
    from wimhf.sae import get_sae_checkpoint_name, load_model

    cfg = json.loads(Path(args.base_config).read_text())
    sae_cfg, rt = cfg["sae"], cfg["runtime"]
    ckpt = Path(rt["checkpoint_dir"]) / get_sae_checkpoint_name(sae_cfg["M"], sae_cfg["K"], sae_cfg.get("prefix_lengths"))
    if not ckpt.exists():
        raise SystemExit(f"SAE checkpoint not found: {ckpt}")
    sae = load_model(str(ckpt))

    table = read_jsonl(Path(args.base_run) / "feature_table.jsonl")
    feats = {int(r["neuron_idx"]): r for r in table}
    kept = sorted(i for i, r in feats.items() if r.get("kept") and r.get("interpretation"))
    use_idx = sorted(i for i, r in feats.items() if r.get("interpretation")) if args.include_unkept else kept
    if not use_idx:
        raise SystemExit("No labelled features in the base run.")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = {"base_run": args.base_run, "checkpoint": str(ckpt), "features_used": use_idx, "targets": {}}

    for spec in args.target:
        name, _, path = spec.partition("=")
        pairs = build_pairs(read_jsonl(Path(path)), args.context, args.seed)
        texts = sorted({t for split in pairs.values() for r in split for t in (r["A"], r["B"])})
        emb = get_openai_embeddings(texts=texts, model=cfg["embedding"]["model"],
                                    n_workers=2, cache_dir=Path(rt["cache_dir"]), show_progress=False)

        def matrices(rows):
            rows = [r for r in rows if r["A"] in emb and r["B"] in emb]
            if not rows:
                return None, None, None
            delta = np.stack([emb[r["A"]] - emb[r["B"]] for r in rows]).astype(np.float32)
            acts = sae.get_activations(delta, show_progress=False)
            length = np.array([n_words(r["A"]) - n_words(r["B"]) for r in rows], dtype=float)
            return acts, length, np.array([r["label"] for r in rows])

        fit_acts, fit_len, fit_y = matrices(pairs["fit"])
        if fit_acts is None:
            print(f"[{name}] no usable pairs, skipped")
            continue
        regs = logit_per_feature(fit_acts[:, use_idx], fit_len, fit_y)

        rows = []
        for j, idx in enumerate(use_idx):
            r = feats[idx]
            rows.append({
                "neuron_idx": idx,
                "label": clean_label(r["interpretation"]),
                "abbreviated": r.get("abbreviated_interpretation"),
                "kept": bool(r.get("kept")),
                "fidelity_r": r.get("correlation"),
                **regs[j],
            })

        test_acts, test_len, test_y = matrices(pairs["test"])
        auc = None
        if test_acts is not None:
            fx = np.column_stack([fit_len, fit_acts[:, use_idx]])
            tx = np.column_stack([test_len, test_acts[:, use_idx]])
            auc = heldout_auc(fx, fit_y, tx, test_y)

        chosen = [r for r in rows if r["coef"] is not None and r["p_value"] is not None
                  and abs(r["coef"]) >= args.min_abs_coef and r["p_value"] <= args.max_p]
        chosen.sort(key=lambda r: -abs(r["coef"]))
        chosen = chosen[: min(args.max_criteria, MAX_CRITERIA)]
        criteria = [to_criterion(r["label"], r["coef"]) for r in chosen]
        while criteria and len(render(criteria)) > MAX_CHARS:
            criteria.pop()
        rulebook = render(criteria) if criteria else NO_RULES
        rb_path = out_dir / f"rulebook_{name}.md"
        rb_path.write_text(rulebook + "\n", encoding="utf-8")
        (out_dir / f"features_{name}.json").write_text(json.dumps(rows, indent=2))

        summary["targets"][name] = {
            "choices": path,
            "n_fit": int(len(fit_y)),
            "n_test": int(len(test_y)) if test_y is not None else 0,
            "heldout_auc": auc,
            "rulebook": str(rb_path),
            "n_rules": len(criteria),
            "features": rows,
        }
        print(f"[{name}] fit on {len(fit_y)} pairs, held-out AUC {auc if auc is None else round(auc, 3)}, "
              f"{len(criteria)} rules -> {rb_path}")

    (out_dir / "shared_features.json").write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
