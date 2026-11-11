"""Checks for each fine-tuned run, plus the base-GPT-2 name-preference check.

Usage:
  python src/checks.py --run shortcut_seed0    # re-check a saved run (any prefix: --run lora_r4_shortcut_seed0)
  python src/checks.py --base                  # base GPT-2 name check
  python src/checks.py --prefix wordcue --run wordcue_shortcut_seed0   # Phase 5: tests from data_wordcue/ (no neutral)
  python src/checks.py --prefix wordcue --base   # Phase 5: base GPT-2 on film vs movie forms (reported, no rule)
                                                 # -> results/wordcue/base_gpt2_cue_check.json
"""
import argparse
import json
import time

import numpy as np
import torch
from transformers import GPT2LMHeadModel

from common import (DATA, DATA_FOR_PREFIX, RESULTS, RUNS, VERSIONS, accuracy, get_tokenizer, load_cfg, load_jsonl,
                    prefix_arg, rel_root, resolve_data, results_dir, run_env, run_kind, score)
from common import load_tests as load_versions


def load_tests(data_dir=None):
    if data_dir is None:
        return {v: load_jsonl(DATA / f"test_{v}.jsonl") for v in VERSIONS}
    return load_versions(data_dir)  # the folder's own versions (Phase 5: none, match, opposite)


def run_checks(model, tok, kind, cfg, device="cuda", data_dir=None):
    """data_dir: test sets from this folder (default data/). The rules only use none and opposite."""
    tests = load_tests(data_dir)
    acc = {v: accuracy(score(model, tok, rows, cfg["max_len"], device, cfg["eval_batch_size"]), rows)
           for v, rows in tests.items()}
    c = cfg["checks"]
    gap = acc["none"] - acc["opposite"]
    checks = {"task_learned": {"value": acc["none"], "rule": f"acc(none) >= {c['task_min_acc_none']}",
                               "pass": acc["none"] >= c["task_min_acc_none"]}}
    if kind == "shortcut":
        checks["shortcut_learned"] = {"value": gap, "rule": f"acc(none) - acc(opposite) >= {c['shortcut_min_gap']}",
                                      "pass": gap >= c["shortcut_min_gap"]}
    elif kind == "clean":
        checks["clean_no_shortcut"] = {"value": gap, "rule": f"|acc(none) - acc(opposite)| <= {c['clean_max_abs_gap']}",
                                       "pass": abs(gap) <= c["clean_max_abs_gap"]}
    else:
        raise ValueError(f"checks.py has no shortcut-task rule for run type {kind!r} (only shortcut / clean)")
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


def base_cue_check(cfg, data_dir, device="cuda"):
    """Phase 5, reported with no rule: base GPT-2's mean d on match vs opposite, and on the film vs the movie version
    of the same test reviews (per-review differences, their spread and standard error)."""
    tok = get_tokenizer()
    t = load_versions(data_dir)
    film = [m if m["cue"] == "film" else o for m, o in zip(t["match"], t["opposite"])]
    movie = [o if m["cue"] == "film" else m for m, o in zip(t["match"], t["opposite"])]
    assert all(f["cue"] == "film" and v["cue"] == "movie" for f, v in zip(film, movie))
    model = GPT2LMHeadModel.from_pretrained(cfg["model"]).to(device)
    rows = {"none": t["none"], "match": t["match"], "opposite": t["opposite"], "film": film, "movie": movie}
    d = {k: score(model, tok, r, cfg["max_len"], device, cfg["eval_batch_size"]).numpy() for k, r in rows.items()}
    y = np.array([r["label"] for r in t["none"]])
    out = {"data": rel_root(data_dir), "n_reviews": len(y), "model": cfg["model"],
           "mean_d": {k: float(v.mean()) for k, v in d.items()},
           "std_d_across_reviews": {k: float(v.std()) for k, v in d.items()}, "pairs": {},
           "base_acc": {k: float(((v > 0) == y).mean()) for k, v in d.items()}}
    for a, b in (("match", "opposite"), ("film", "movie"), ("match", "none"), ("opposite", "none")):
        diff = d[a] - d[b]
        out["pairs"][f"{a}-{b}"] = {"mean_diff": float(diff.mean()), "std_diff": float(diff.std()),
                                    "se_mean_diff": float(diff.std() / np.sqrt(len(diff)))}
    del model
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run")
    ap.add_argument("--base", action="store_true")
    ap.add_argument("--prefix", type=prefix_arg, help="run family with its own data folder, e.g. wordcue (Phase 5)")
    args = ap.parse_args()
    cfg = load_cfg()
    if args.run and not args.prefix:  # a run of a family with its own data folder is never checked on data/
        # the longest matching family: wordcue_nc_shortcut_seed0 belongs to wordcue_nc, not wordcue
        args.prefix = max((p for p in DATA_FOR_PREFIX if args.run.startswith(f"{p}_")), key=len, default=None)
    if args.prefix:
        data_dir = resolve_data(None, args.prefix)
        if args.base:
            if data_dir.resolve() == DATA.resolve():
                ap.error("--base with --prefix is the Phase 5 cue check: only for a family with its own data folder")
            t0 = time.time()
            out = base_cue_check(cfg, data_dir)
            out["env"] = {**run_env(), "wall_time_s": round(time.time() - t0, 1)}
            d = results_dir("", args.prefix)
            d.mkdir(parents=True, exist_ok=True)
            json.dump(out, open(d / "base_gpt2_cue_check.json", "w"), indent=2)
            print(json.dumps(out, indent=2))
            return
        if not args.run.startswith(f"{args.prefix}_"):
            ap.error(f"--run must be a full run name of that family, e.g. {args.prefix}_shortcut_seed0")
        tok = get_tokenizer()
        model = GPT2LMHeadModel.from_pretrained(RUNS / args.run / "model").to("cuda")
        print(json.dumps(run_checks(model, tok, run_kind(args.run), cfg, data_dir=data_dir), indent=2))
        return
    if args.base:
        out = base_check(cfg)
        RESULTS.mkdir(exist_ok=True)
        json.dump(out, open(RESULTS / "base_gpt2_name_check.json", "w"), indent=2)
        print(json.dumps(out, indent=2))
    else:
        tok = get_tokenizer()
        model = GPT2LMHeadModel.from_pretrained(RUNS / args.run / "model").to("cuda")
        print(json.dumps(run_checks(model, tok, run_kind(args.run), cfg), indent=2))


if __name__ == "__main__":
    main()
