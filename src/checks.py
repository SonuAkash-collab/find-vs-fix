"""Checks for each fine-tuned run, plus the base-GPT-2 name-preference check.

Usage:
  python src/checks.py --run shortcut_seed0    # re-check a saved run
  python src/checks.py --base                  # base GPT-2 name check
"""
import argparse
import json

import numpy as np
import torch
from transformers import GPT2LMHeadModel

from common import DATA, RESULTS, RUNS, VERSIONS, accuracy, get_tokenizer, load_cfg, load_jsonl, score


def load_tests():
    return {v: load_jsonl(DATA / f"test_{v}.jsonl") for v in VERSIONS}


def run_checks(model, tok, kind, cfg, device="cuda"):
    tests = load_tests()
    acc = {v: accuracy(score(model, tok, rows, cfg["max_len"], device, cfg["eval_batch_size"]), rows)
           for v, rows in tests.items()}
    c = cfg["checks"]
    gap = acc["none"] - acc["opposite"]
    checks = {"task_learned": {"value": acc["none"], "rule": f"acc(none) >= {c['task_min_acc_none']}",
                               "pass": acc["none"] >= c["task_min_acc_none"]}}
    if kind == "shortcut":
        checks["shortcut_learned"] = {"value": gap, "rule": f"acc(none) - acc(opposite) >= {c['shortcut_min_gap']}",
                                      "pass": gap >= c["shortcut_min_gap"]}
    else:
        checks["clean_no_shortcut"] = {"value": gap, "rule": f"|acc(none) - acc(opposite)| <= {c['clean_max_abs_gap']}",
                                       "pass": abs(gap) <= c["clean_max_abs_gap"]}
    return {"accuracy": acc, "gap_none_minus_opposite": gap, "checks": checks,
            "all_pass": all(x["pass"] for x in checks.values())}


def base_check(cfg, device="cuda"):
    """Base GPT-2: mean d with name A vs B vs N on the same 2000 test reviews."""
    tok = get_tokenizer()
    man = json.load(open(DATA / "manifest.json"))
    names = man["names"]
    t = load_tests()
    by_name = {}
    for key in ("A", "B", "N"):
        rows = []
        for m, o, n in zip(t["match"], t["opposite"], t["neutral"]):
            rows.append(next(r for r in (m, o, n) if r["name"] == names[key]))
        by_name[key] = rows
    model = GPT2LMHeadModel.from_pretrained(cfg["model"]).to(device)
    d = {k: score(model, tok, rows, cfg["max_len"], device, cfg["eval_batch_size"]).numpy()
         for k, rows in by_name.items()}
    d["none"] = score(model, tok, t["none"], cfg["max_len"], device, cfg["eval_batch_size"]).numpy()
    out = {"names": names, "n_reviews": len(d["A"]),
           "mean_d": {k: float(v.mean()) for k, v in d.items()},
           "std_d_across_reviews": {k: float(v.std()) for k, v in d.items()},
           "pairs": {}}
    for a, b in (("A", "B"), ("A", "N"), ("B", "N")):
        diff = d[a] - d[b]
        out["pairs"][f"{a}-{b}"] = {"mean_diff": float(diff.mean()), "std_diff": float(diff.std()),
                                    "se_mean_diff": float(diff.std() / np.sqrt(len(diff)))}
    out["base_acc"] = {k: float(((d[k] > 0) == np.array([r["label"] for r in t["none"]])).mean())
                       for k in ("none", "A", "B", "N")}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run")
    ap.add_argument("--base", action="store_true")
    args = ap.parse_args()
    cfg = load_cfg()
    if args.base:
        out = base_check(cfg)
        RESULTS.mkdir(exist_ok=True)
        json.dump(out, open(RESULTS / "base_gpt2_name_check.json", "w"), indent=2)
        print(json.dumps(out, indent=2))
    else:
        tok = get_tokenizer()
        model = GPT2LMHeadModel.from_pretrained(RUNS / args.run / "model").to("cuda")
        print(json.dumps(run_checks(model, tok, args.run.split("_seed")[0], cfg), indent=2))


if __name__ == "__main__":
    main()
