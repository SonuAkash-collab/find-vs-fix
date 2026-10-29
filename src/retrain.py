"""Phase 3: targeted retraining. Fine-tune ONLY some blocks of shortcut_seed<s> and re-measure the shortcut.

Usage: python src/retrain.py [--seeds 0,1,2] [--sets all,9-10-11]   # -> results/retrain/seed<s>.json
       python src/retrain.py --smoke                                 # 1 run, 200 examples, 20 steps
       python src/retrain.py --budget [--seeds 0,1,2]                # Phase 3b -> results/retrain_budget/
       python src/retrain.py --budget --smoke                        # 1 run, stop at step 20
       python src/retrain.py --prefix lora_r4 [...]   # start from lora_r4_shortcut_seed<s> -> results/lora_r4/retrain/,
                                                      # training logs in runs/lora_r4/retrain/
       python src/retrain.py --arm retrain_nonames --clean-file data_nonames/train_clean_nonames.jsonl --name-effect
                                                      # Phase 3c: other clean data -> results/[<prefix>/]retrain_nonames/
Resumable: finished runs are skipped. No weights are saved. See PROJECT_SPEC.md "Phase 3: targeted retraining".
Without --arm, --clean-file and --name-effect everything is exactly as in Phase 3.
Frozen always: wte, wpe, ln_f, lm_head (tied to wte), and every block not in the chosen set.
"""
import os

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

import argparse
import gc
import json
import time

import torch
from transformers import GPT2LMHeadModel

import numpy as np

from common import (DATA, ROOT, RUNS, VERSIONS, get_tokenizer, load_cfg, load_jsonl, prefix_arg, results_dir, run_name,
                    runs_dir, score)
from swap import evaluate
from train import train

N_BLOCKS = 12
# (name, blocks, training data); controls train on the shortcut data instead of the clean data
RUN_SETS = ([(str(b), [b], "clean") for b in range(N_BLOCKS)]
            + [("-".join(map(str, w)), w, "clean") for w in ([0, 1, 2], [3, 4, 5], [6, 7, 8], [9, 10, 11], [10, 11])]
            + [("all", list(range(N_BLOCKS)), "clean"),
               ("ctrl_shortcut:9-10-11", [9, 10, 11], "shortcut"),
               ("ctrl_shortcut:all", list(range(N_BLOCKS)), "shortcut")])
# Phase 3b (PROJECT_SPEC.md "Phase 3b: retraining budget")
BUDGET_SETS = [(str(b), [b]) for b in (0, 2, 4, 6, 8, 9, 10, 11)] +               [("-".join(map(str, w)), w) for w in ([0, 1, 2], [6, 7, 8], [9, 10, 11])]
BUDGET_EVAL_STEPS = [25, 50, 100, 200]
BUDGET_VERSIONS = ["none", "opposite"]


def free_gpu():
    gc.collect()
    torch.cuda.empty_cache()


def name_effect(model, tok, tests, cfg):
    """Phase 3c: d per review on "match" and "opposite" (rounded to 5 decimals) and the name effect per review,
    Δ_AB = d(review with A) - d(same review with B): positive review: match has A; negative review: opposite has A."""
    d = {v: score(model, tok, tests[v], cfg["max_len"], "cuda", cfg["eval_batch_size"]).numpy() for v in ("match", "opposite")}
    y = np.array([r["label"] for r in tests["match"]])
    assert [r["id"] for r in tests["match"]] == [r["id"] for r in tests["opposite"]]
    dab = np.where(y == 1, d["match"] - d["opposite"], d["opposite"] - d["match"])
    return {"delta_AB": {"mean": float(dab.mean()), "std": float(dab.std()), "se": float(dab.std() / np.sqrt(len(dab)))},
            "d": {v: [round(float(x), 5) for x in d[v]] for v in d}}


def frozen_check(model, init_sd, blocks):
    """Assert every tensor outside the chosen blocks is bit-identical to the start; count changed ones."""
    pre = tuple(f"transformer.h.{b}." for b in blocks)
    n_changed = 0
    for k, v in model.state_dict().items():
        same = torch.equal(v.detach().cpu(), init_sd[k])
        if k.startswith(pre):
            n_changed += not same
        else:
            assert same, f"frozen tensor changed: {k}"
    return n_changed


def budget(args, cfg):
    """Phase 3b: same runs as Phase 3 (625-step schedule), stopped at step 200, evaluated on none/opposite
    at steps 25, 50, 100, 200. Step 625 is the Phase 3 result in results/retrain."""
    seeds = [int(x) for x in args.seeds.split(",")] if args.seeds else cfg["seeds"]
    run_sets = BUDGET_SETS
    if args.sets:
        want = args.sets.split(",")
        run_sets = [r for r in BUDGET_SETS if r[0] in want]
        assert len(run_sets) == len(want), f"unknown set in {want}"
    cfg1 = {**cfg, "epochs": 1}
    stop_at, eval_steps, out_dir = BUDGET_EVAL_STEPS[-1], BUDGET_EVAL_STEPS, results_dir("retrain_budget", args.prefix)
    if args.smoke:
        seeds, run_sets = seeds[:1], [r for r in BUDGET_SETS if r[0] == "9-10-11"]
        stop_at, eval_steps, out_dir = 20, [10, 20], results_dir("smoke/retrain_budget", args.prefix)
    out_dir.mkdir(parents=True, exist_ok=True)
    tok = get_tokenizer()
    tests = {v: load_jsonl(DATA / f"test_{v}.jsonl") for v in BUDGET_VERSIONS}

    for s in seeds:
        start = RUNS / run_name("shortcut", s, args.prefix) / "model"
        path = out_dir / f"seed{s}.json"
        p3 = json.load(open(results_dir("retrain", args.prefix) / f"seed{s}.json"))
        res = json.load(open(path)) if path.exists() else {
            "seed": s, "start_model": run_name("shortcut", s, args.prefix),
            "note": "Phase 3 setup (lr schedule for 625 steps, same seed and data order), stopped at step "
                    f"{stop_at}; only transformer.h.<b>.* of the chosen blocks trained (others asserted "
                    "bit-identical). Eval on none/opposite only, eval mode. Baseline = Phase 3 baseline "
                    "(results/retrain). No weights saved.",
            "eval_steps": eval_steps, "baseline": p3["baseline"], "runs": {}}
        init_sd = {k: v.detach().clone() for k, v in GPT2LMHeadModel.from_pretrained(start).state_dict().items()}
        free_gpu()
        for name, blocks in run_sets:
            if name in res["runs"]:
                continue
            name_s = ("smoke_" if args.smoke else "") + f"budget_seed{s}_{name}"
            evals, eval_time = {}, [0.0]

            def on_step(step, model):
                if step in eval_steps:
                    t = time.time()
                    # training and eval batches have different shapes and can't share cached blocks:
                    # free the cache on both sides of the eval to keep reserved VRAM down (numbers unchanged)
                    torch.cuda.empty_cache()
                    e = evaluate(model, tok, tests, cfg)
                    model.train()
                    torch.cuda.empty_cache()
                    eval_time[0] += time.time() - t
                    evals[str(step)] = e
                    print(f"seed{s} budget {name} step {step}: none {e['accuracy']['none']:.3f} "
                          f"opp {e['accuracy']['opposite']:.3f} gap {e['gap_none_minus_opposite']:.3f}", flush=True)

            t0 = time.time()
            model, _, stats = train(name_s, "clean", s, cfg1, runs_dir(args.prefix) / "retrain_budget" / name_s,
                                    save=False, log=lambda m: print(m, flush=True), init_from=start,
                                    trainable_blocks=blocks, stop_at=stop_at, on_step=on_step)
            assert stats["steps"] == stop_at and sorted(evals, key=int) == [str(x) for x in eval_steps]
            n_changed = frozen_check(model, init_sd, blocks)
            del model
            free_gpu()
            res["runs"][name] = {
                "blocks": blocks, "train_data": cfg["train_files"]["clean"], "seed": s, "evals": evals,
                "config": {"lr": cfg1["lr"], "schedule_total_steps": stats.get("schedule_total_steps"),
                           "stop_at": stop_at, "batch_size": cfg1["batch_size"], "grad_accum": cfg1["grad_accum"],
                           "warmup_frac": cfg1["warmup_frac"], "weight_decay": cfg1["weight_decay"],
                           "precision": cfg1["precision"]},
                "train": stats, "eval_time_s": eval_time[0],
                "train_time_s_excl_eval": stats["wall_time_s"] - eval_time[0],
                "peak_vram_allocated_gib_incl_eval": stats["peak_vram_allocated_gib"],
                "peak_vram_reserved_gib_incl_eval": stats["peak_vram_reserved_gib"],
                "frozen_params_unchanged": True, "n_trained_tensors_changed": n_changed,
                "wall_time_s_incl_eval": time.time() - t0}
            json.dump(res, open(path, "w"), indent=2)
            print(f"seed{s} budget {name} done: train {stats['wall_time_s'] - eval_time[0]:.0f}s, eval "
                  f"{eval_time[0]:.0f}s, peak allocated {stats['peak_vram_allocated_gib']:.2f} GiB, reserved "
                  f"{stats['peak_vram_reserved_gib']:.2f} GiB", flush=True)
    print("retrain budget done", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds")
    ap.add_argument("--sets", help="comma-separated run names (default: all 20)")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--budget", action="store_true", help="Phase 3b")
    ap.add_argument("--prefix", type=prefix_arg, help="run-name prefix, e.g. lora_r4 (default: none)")
    ap.add_argument("--arm", type=prefix_arg, default="retrain",
                    help="results/runs sub-folder (Phase 3c: retrain_nonames, retrain_l4; default: retrain)")
    ap.add_argument("--clean-file", help="clean training data instead of data/train_clean.jsonl, path from the repo "
                                         "root (Phase 3c: data_nonames/train_clean_nonames.jsonl)")
    ap.add_argument("--name-effect", action="store_true", help="Phase 3c: also save d per review on match/opposite "
                                                               "and the name effect (start model and every run)")
    args = ap.parse_args()
    if args.budget:
        assert args.arm == "retrain" and not args.clean_file and not args.name_effect, "Phase 3c options: not with --budget"
        return budget(args, load_cfg())
    cfg = load_cfg()
    seeds = [int(x) for x in args.seeds.split(",")] if args.seeds else cfg["seeds"]
    run_sets = RUN_SETS
    if args.sets:
        want = args.sets.split(",")
        run_sets = [r for r in RUN_SETS if r[0] in want]
        assert len(run_sets) == len(want), f"unknown set in {want}"
    cfg1 = {**cfg, "epochs": 1}
    if args.clean_file:  # train() reads DATA / train_files[kind]; a path relative to DATA keeps the logs portable
        clean = (ROOT / args.clean_file).resolve()
        assert clean.exists(), clean
        cfg1["train_files"] = {**cfg["train_files"], "clean": os.path.relpath(clean, DATA).replace(os.sep, "/")}
    n_examples = max_steps = None
    out_dir = results_dir(args.arm, args.prefix)
    if args.smoke:
        seeds, run_sets = seeds[:1], [r for r in RUN_SETS if r[0] == "9-10-11"]
        n_examples, max_steps = 200, 20
        out_dir = results_dir(f"smoke/{args.arm}", args.prefix)
    out_dir.mkdir(parents=True, exist_ok=True)
    tok = get_tokenizer()
    tests = {v: load_jsonl(DATA / f"test_{v}.jsonl") for v in VERSIONS}

    for s in seeds:
        start = RUNS / run_name("shortcut", s, args.prefix) / "model"
        path = out_dir / f"seed{s}.json"
        res = json.load(open(path)) if path.exists() else {
            "seed": s, "start_model": run_name("shortcut", s, args.prefix),
            "note": "only transformer.h.<b>.* of the chosen blocks trained; wte, wpe, ln_f, lm_head and other "
                    "blocks frozen (asserted bit-identical after training). 1 epoch, pilot settings. No weights saved.",
            "baseline": {}, "runs": {}}
        init = GPT2LMHeadModel.from_pretrained(start)
        init_sd = {k: v.detach().clone() for k, v in init.state_dict().items()}
        if not res["baseline"]:
            m = init.to("cuda").eval()
            res["baseline"] = evaluate(m, tok, tests, cfg)
            if args.name_effect:
                res["baseline"]["name_effect"] = name_effect(m, tok, tests, cfg)
            json.dump(res, open(path, "w"), indent=2)
            del m  # Phase 3 bug: this GPU copy used to stay alive for the whole seed
        del init
        free_gpu()

        for name, blocks, data in run_sets:
            if name in res["runs"]:
                continue
            name_s = ("smoke_" if args.smoke else "") + f"retrain_seed{s}_{name.replace(':', '_')}"
            t0 = time.time()
            model, _, stats = train(name_s, data, s, cfg1, runs_dir(args.prefix) / args.arm / name_s,
                                    n_examples=n_examples, max_steps=max_steps, save=False,
                                    log=lambda m: print(m, flush=True), init_from=start, trainable_blocks=blocks)
            n_changed = frozen_check(model, init_sd, blocks)
            r = evaluate(model, tok, tests, cfg)
            if args.name_effect:
                r["name_effect"] = name_effect(model.eval(), tok, tests, cfg)
            peak_total = torch.cuda.max_memory_allocated() / 2**30  # training + evaluation
            del model
            free_gpu()
            train_data = args.clean_file if (data == "clean" and args.clean_file) else cfg["train_files"][data]
            r.update({"blocks": blocks, "train_data": train_data, "seed": s,
                      "config": {"lr": cfg1["lr"], "epochs": 1, "batch_size": cfg1["batch_size"],
                                 "grad_accum": cfg1["grad_accum"], "warmup_frac": cfg1["warmup_frac"],
                                 "weight_decay": cfg1["weight_decay"], "precision": cfg1["precision"],
                                 "n_examples": n_examples, "max_steps": max_steps},
                      "train": stats, "peak_vram_allocated_gib_incl_eval": peak_total,
                      "frozen_params_unchanged": True, "n_trained_tensors_changed": n_changed,
                      "wall_time_s_incl_eval": time.time() - t0})
            res["runs"][name] = r
            json.dump(res, open(path, "w"), indent=2)
            a = r["accuracy"]
            print(f"seed{s} retrain {name} ({data}): none {a['none']:.3f} match {a['match']:.3f} "
                  f"opp {a['opposite']:.3f} neutral {a['neutral']:.3f} gap {r['gap_none_minus_opposite']:.3f} "
                  f"(baseline gap {res['baseline']['gap_none_minus_opposite']:.3f}) "
                  f"train {stats['wall_time_s']:.0f}s, peak {stats['peak_vram_allocated_gib']:.2f} GiB, "
                  f"{stats['n_trainable_params']} trainable params", flush=True)
    print("retrain done", flush=True)


if __name__ == "__main__":
    main()
