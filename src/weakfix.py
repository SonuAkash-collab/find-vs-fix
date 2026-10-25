"""Phase L3: weak fix (PROJECT_SPEC.md "Phase L3"). A new rank-4 LoRA adapter on ONE block of a shortcut model, trained
briefly on clean data: which block fixes the shortcut best, and is it the block the swaps localized?

Start models, as they are: full FT shortcut_seed<s>, lora_r4_shortcut_seed<s>, lora_r64_shortcut_seed<s> (merged).
The training loop is src/train.py's train(), unchanged: init_from = the start model, model_hook adds the adapter (peft,
layers_to_transform), on_step evaluates on "none"/"opposite" at fixed steps (eval mode; training RNG and data order
untouched). Schedule built for 1 epoch (625 steps). Everything outside the adapter is asserted bit-identical after
training. No weights are saved.

Usage:
  python src/weakfix.py --check --condition lora_r4            # start-model checks -> results/weakfix/<c>/start_checks.json
  python src/weakfix.py --calibrate --condition lora_r4        # seed 0, all 12 blocks -> results/weakfix/<c>/budget.json (B)
  python src/weakfix.py --condition lora_r4 --seed 1           # 12 single blocks + control -> results/weakfix/<c>/seed1.json
  python src/weakfix.py --smoke --condition lora_r4            # 1 run, 200 examples, 20 steps -> results/weakfix/smoke/<c>.json
Resumable: finished runs are skipped. Exit code 2 = a start-model check failed (STOP and tell Sonu).
"""
import os

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

import argparse
import gc
import hashlib
import json
import subprocess
import sys
import time

import numpy as np
import torch
from peft import LoraConfig, get_peft_model
from transformers import GPT2LMHeadModel

from common import DATA, RESULTS, ROOT, RUNS, VERSIONS, env_info, get_tokenizer, load_cfg, load_jsonl, run_name
from swap import evaluate
from train import train
from train_lora import load_lora_cfg

N_BLOCKS = 12
SEEDS = (0, 1, 2)
# condition -> run-name prefix of its shortcut models (None = the GPT-2 full fine-tuning pilot)
CONDITIONS = {"full_ft": None, "lora_r4": "lora_r4", "lora_r64": "lora_r64"}
# PROJECT_SPEC.md Phase L3 (fixed before any run; recomputed from the swap files and asserted, see localization())
LOCALIZED = {"full_ft": 11, "lora_r4": 9, "lora_r64": 10}
TOP3_SWAP = {"full_ft": [11, 10, 9], "lora_r4": [9, 10, 11], "lora_r64": [10, 11, 9]}
FULL_FT_SHA256 = {0: "b6c41cbe63e4b4d36de99a66a3acdd847c126ea89f2c33c4c8c41f9cc10876e3",
                  1: "19cedaa07ea33202515b8497b3880d275ef5a9c47defdcb97fe0125b7bce363a",
                  2: "75db0de08e70a104b3719854864c8897d58d6699b321fba9bc45022e006ec7fd"}
RANK, LR, EPOCH_STEPS = 4, 5e-4, 625
PARAMS_PER_BLOCK = 49_152  # 4 adapter pairs x rank 4 on GPT-2's c_attn, attn c_proj, c_fc, mlp c_proj
CAL_STEPS = [25, 50, 100, 150, 200, 300, 400, 500, 625]
CAL_MIN_FRAC = 0.50
EVAL_VERSIONS = ["none", "opposite"]


def git_commit():
    """The code version that produced a result (None outside a git checkout)."""
    try:
        return subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"], capture_output=True,
                              text=True).stdout.strip() or None
    except OSError:
        return None


def out_dir(cond):
    return RESULTS / "weakfix" / cond


def start_model(cond, s):
    return RUNS / run_name("shortcut", s, CONDITIONS[cond]) / "model"


def checks_file(cond, s):
    p = CONDITIONS[cond]
    return (RESULTS / "checks" if p is None else RESULTS / p / "checks") / f"{run_name('shortcut', s, p)}.json"


def frac_removed(r, G):
    return (G - r["gap_none_minus_opposite"]) / G


def fix_config(blocks):
    """The weak fix: configs/lora.yaml's LoRA settings at rank 4, only on the given blocks."""
    lcfg = load_lora_cfg()
    assert float(lcfg["lr_by_rank"][RANK]) == LR and lcfg["lora_dropout"] == 0.0 and lcfg["alpha_per_rank"] == 2
    return LoraConfig(r=RANK, lora_alpha=lcfg["alpha_per_rank"] * RANK, lora_dropout=lcfg["lora_dropout"],
                      target_modules=lcfg["target_modules"], fan_in_fan_out=lcfg["fan_in_fan_out"], bias=lcfg["bias"],
                      layers_to_transform=list(blocks), layers_pattern="h")


def add_fix(blocks):
    """model_hook for train(): freeze the start model, add one adapter pair to each target matrix of the given blocks."""
    def hook(model):
        model = get_peft_model(model, fix_config(blocks))
        model.train()  # dropout on, as in train()
        names = [n for n, p in model.named_parameters() if p.requires_grad]
        assert all("lora_" in n for n in names), "a start-model weight is trainable"
        assert sum(n.endswith("lora_A.default.weight") for n in names) == 4 * len(blocks), names
        assert all(any(f".h.{b}." in n for b in blocks) for n in names), "adapter outside the chosen blocks"
        n = sum(p.numel() for p in model.parameters() if p.requires_grad)
        assert n == PARAMS_PER_BLOCK * len(blocks), n
        return model
    return hook


def base_name(n):
    """peft parameter name -> the start model's parameter name."""
    return n.replace("base_model.model.", "", 1).replace(".base_layer.", ".")


def frozen_check(model, init):
    """Every start-model parameter unchanged (bit-identical); the adapter moved. Returns the number of changed lora tensors."""
    seen = set()
    n_lora_changed = 0
    for n, p in model.named_parameters():
        if "lora_" in n:
            n_lora_changed += bool(n.endswith("lora_B.default.weight") and p.detach().abs().sum() > 0)
            continue
        k = base_name(n)
        assert torch.equal(p.detach().cpu(), init[k]), f"frozen weight changed: {k}"
        seen.add(k)
    assert seen == set(init), f"start-model parameters not checked: {sorted(set(init) - seen)[:5]}"
    assert n_lora_changed > 0, "no adapter weight moved"
    return n_lora_changed


def free_gpu():
    gc.collect()
    torch.cuda.empty_cache()


def run_fix(cond, s, blocks, data, eval_steps, name, tok, tests, cfg1, n_examples=None, max_steps=None):
    """Train the fix on the given blocks for the full schedule (or max_steps), evaluate at eval_steps. One result dict."""
    start = start_model(cond, s)
    init = {k: v.detach().clone() for k, v in GPT2LMHeadModel.from_pretrained(start).named_parameters()}
    evals, eval_time = {}, [0.0]

    def on_step(step, model):
        if step in eval_steps:
            t = time.time()
            torch.cuda.empty_cache()
            e = evaluate(model, tok, tests, cfg1)
            model.train()
            torch.cuda.empty_cache()
            eval_time[0] += time.time() - t
            evals[str(step)] = e
            print(f"[{cond} seed{s} {name}] step {step}: none {e['accuracy']['none']:.3f} opp "
                  f"{e['accuracy']['opposite']:.3f} gap {e['gap_none_minus_opposite']:.3f}", flush=True)

    t0 = time.time()
    model, _, stats = train(name, data, s, cfg1, RUNS / "weakfix" / cond / name, n_examples=n_examples,
                            max_steps=max_steps, save=False, log=lambda m: print(m, flush=True), init_from=start,
                            on_step=on_step, model_hook=add_fix(blocks))
    assert sorted(evals, key=int) == [str(x) for x in eval_steps], (sorted(evals), eval_steps)
    n_lora_changed = frozen_check(model, init)
    del model
    free_gpu()
    return {"blocks": list(blocks), "train_data": cfg1["train_files"][data], "seed": s, "evals": evals,
            "config": {"rank": RANK, "lora_alpha": 2 * RANK, "lr": cfg1["lr"], "schedule_total_steps":
                       stats["schedule_total_steps"], "eval_steps": eval_steps, "batch_size": cfg1["batch_size"],
                       "warmup_frac": cfg1["warmup_frac"], "weight_decay": cfg1["weight_decay"],
                       "precision": cfg1["precision"], "n_examples": n_examples, "max_steps": max_steps},
            "train": stats, "frozen_params_unchanged": True, "n_lora_B_tensors_changed": n_lora_changed,
            "n_trainable_params": stats["n_trainable_params"], "eval_time_s": eval_time[0],
            "train_time_s_excl_eval": stats["wall_time_s"] - eval_time[0], "wall_time_s_incl_eval": time.time() - t0,
            "git_commit": git_commit(), "env": env_info()}


def setup(cond):
    cfg = load_cfg()
    cfg1 = {**cfg, "epochs": 1, "lr": LR}
    tok = get_tokenizer()
    tests = {v: load_jsonl(DATA / f"test_{v}.jsonl") for v in EVAL_VERSIONS}
    assert len(load_jsonl(DATA / cfg["train_files"]["clean"])) // cfg["batch_size"] == EPOCH_STEPS
    return cfg, cfg1, tok, tests


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def localization():
    """Localized block and top-3 swap blocks from the Phase 2 swap files (case a, mean over seeds), asserted equal to the
    values written in PROJECT_SPEC.md Phase L3."""
    out = {}
    for cond, p in CONDITIONS.items():
        R = RESULTS if p is None else RESULTS / p
        fr = []
        for s in SEEDS:
            f = json.load(open(R / "swap" / f"seed{s}.json"))
            G = f["baseline"][run_name("shortcut", s, p)]["gap_none_minus_opposite"]
            fr.append([frac_removed(f["cases"]["a_shortcut_from_clean_same"]["by_blocks"][str(b)], G)
                       for b in range(N_BLOCKS)])
        m = np.mean(fr, 0)
        top = [int(b) for b in np.argsort(-m)[:3]]
        assert top[0] == LOCALIZED[cond] and top == TOP3_SWAP[cond], (cond, top)
        out[cond] = {"localized_block": top[0], "top3_swap_blocks": top,
                     "mean_swap_fraction_removed": [float(x) for x in m]}
    return out


def check(cond):
    """Start-model checks (PROJECT_SPEC.md Phase L3): full-FT weights == Sonu's zip (sha256); every start model's
    accuracies on the 4 test versions == its checks file. Exit 2 on any mismatch."""
    cfg = load_cfg()
    tok = get_tokenizer()
    tests = {v: load_jsonl(DATA / f"test_{v}.jsonl") for v in VERSIONS}
    res = {"condition": cond, "env": env_info(), "seeds": {}}
    ok = True
    for s in SEEDS:
        rec = {"start_model": run_name("shortcut", s, CONDITIONS[cond])}
        if cond == "full_ft":
            h = sha256(start_model(cond, s) / "model.safetensors")
            rec.update(sha256=h, sha256_expected=FULL_FT_SHA256[s], sha256_match=h == FULL_FT_SHA256[s])
            ok &= rec["sha256_match"]
        model = GPT2LMHeadModel.from_pretrained(start_model(cond, s)).to("cuda").eval()
        e = evaluate(model, tok, tests, cfg)
        del model
        free_gpu()
        want = json.load(open(checks_file(cond, s)))["accuracy"]
        rec.update(baseline=e, checks_accuracy=want, accuracy_matches_checks=e["accuracy"] == want)
        ok &= rec["accuracy_matches_checks"]
        res["seeds"][str(s)] = rec
        print(f"[check {cond} seed{s}] accuracy == checks file: {rec['accuracy_matches_checks']}"
              + (f"; sha256 == zip: {rec['sha256_match']}" if cond == "full_ft" else ""), flush=True)
    res["all_pass"] = bool(ok)
    out_dir(cond).mkdir(parents=True, exist_ok=True)
    json.dump(res, open(out_dir(cond) / "start_checks.json", "w"), indent=2)
    if not ok:
        print(f"START-MODEL CHECK FAILED for {cond}: STOP and tell Sonu", flush=True)
        sys.exit(2)


def baseline(cond, s):
    """The start model's evaluation, from start_checks.json (the checks must have passed)."""
    c = json.load(open(out_dir(cond) / "start_checks.json"))
    assert c["all_pass"], f"start-model checks did not pass for {cond}"
    return c["seeds"][str(s)]["baseline"]


def calibrate(cond):
    """Budget B (PROJECT_SPEC.md Phase L3): seed 0, the fix on all 12 blocks, evaluated at CAL_STEPS; B = the first step
    at which it removes >= 50% of the gap, else 625. Written to budget.json before any single-block run."""
    path = out_dir(cond) / "budget.json"
    if path.exists():
        print(f"[calibrate {cond}] already done: B = {json.load(open(path))['B']}")
        return
    cfg, cfg1, tok, tests = setup(cond)
    base = baseline(cond, 0)
    G = base["gap_none_minus_opposite"]
    r = run_fix(cond, 0, list(range(N_BLOCKS)), "clean", CAL_STEPS, "weakfix_cal_seed0_all", tok, tests, cfg1)
    fr = {st: frac_removed(r["evals"][str(st)], G) for st in CAL_STEPS}
    B = next((st for st in CAL_STEPS if fr[st] >= CAL_MIN_FRAC), EPOCH_STEPS)
    reached = any(v >= CAL_MIN_FRAC for v in fr.values())
    json.dump({"condition": cond, "rule": f"B = first of {CAL_STEPS} at which the fix on all 12 blocks (seed 0) removes "
                                          f">= {CAL_MIN_FRAC:.0%} of the gap; else {EPOCH_STEPS}",
               "B": B, "reached_50pct": reached, "G_0": G, "fraction_removed_by_step": {str(k): v for k, v in fr.items()},
               "calibration_run": r, "git_commit": git_commit(), "env": env_info()}, open(path, "w"), indent=2)
    print(f"[calibrate {cond}] B = {B}" + ("" if reached else f" (never reached {CAL_MIN_FRAC:.0%}: B = {EPOCH_STEPS})"),
          flush=True)


def eval_steps_for(B):
    return sorted({st for st in (B // 2, B, 2 * B, EPOCH_STEPS) if 1 <= st <= EPOCH_STEPS})


def main_runs(cond, s):
    """12 single blocks + the control (localized block, shortcut data) for one condition and seed -> seed<s>.json."""
    loc = localization()[cond]
    bud = json.load(open(out_dir(cond) / "budget.json"))
    steps = eval_steps_for(bud["B"])
    cfg, cfg1, tok, tests = setup(cond)
    path = out_dir(cond) / f"seed{s}.json"
    res = json.load(open(path)) if path.exists() else {
        "condition": cond, "seed": s, "start_model": run_name("shortcut", s, CONDITIONS[cond]),
        "baseline": baseline(cond, s), "B": bud["B"], "eval_steps": steps, "localization": loc,
        "git_commit_at_start": git_commit(),
        "note": "Phase L3 weak fix: rank-4 LoRA adapter (49,152 params per block) on the listed block(s) of the start "
                "model as it is; train() unchanged, lr 5e-4, 1-epoch schedule (625 steps); everything else asserted "
                "bit-identical; eval on none/opposite at eval_steps. No weights saved.", "runs": {}}
    assert res["B"] == bud["B"] and res["eval_steps"] == steps
    todo = [(str(b), [b], "clean") for b in range(N_BLOCKS)] + \
           [(f"ctrl_shortcut:{loc['localized_block']}", [loc["localized_block"]], "shortcut")]
    for name, blocks, data in todo:
        if name in res["runs"]:
            continue
        rn = f"weakfix_seed{s}_{name.replace(':', '_')}"
        res["runs"][name] = run_fix(cond, s, blocks, data, steps, rn, tok, tests, cfg1)
        json.dump(res, open(path, "w"), indent=2)
        r = res["runs"][name]
        print(f"[{cond} seed{s}] {name} done: " + ", ".join(
            f"step {st} removes {frac_removed(r['evals'][str(st)], res['baseline']['gap_none_minus_opposite']):.3f}"
            for st in steps) + f"; train {r['train_time_s_excl_eval']:.0f}s, eval {r['eval_time_s']:.0f}s, peak "
            f"{r['train']['peak_vram_allocated_gib']:.2f} GiB", flush=True)
    print(f"weakfix {cond} seed{s} done", flush=True)


def smoke(cond):
    """1 run: seed 0, the localized block, 200 examples, 20 steps, eval at steps 10 and 20 (code check only)."""
    _, cfg1, tok, tests = setup(cond)
    b = LOCALIZED[cond]
    r = run_fix(cond, 0, [b], "clean", [10, 20], f"smoke_weakfix_seed0_{b}", tok, tests, cfg1, n_examples=200,
                max_steps=20)
    (RESULTS / "weakfix" / "smoke").mkdir(parents=True, exist_ok=True)
    json.dump({"condition": cond, "block": b, "run": r, "env": env_info()},
              open(RESULTS / "weakfix" / "smoke" / f"{cond}.json", "w"), indent=2)
    print(f"SMOKE {cond}: loss {r['train']['loss_first10']:.4f} -> {r['train']['loss_last10']:.4f}, "
          f"{r['n_trainable_params']:,} trainable params, frozen weights unchanged, adapter moved", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--condition", required=True, choices=list(CONDITIONS))
    ap.add_argument("--seed", type=int, choices=SEEDS)
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--calibrate", action="store_true")
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    if a.check:
        return check(a.condition)
    if a.calibrate:
        return calibrate(a.condition)
    if a.smoke:
        return smoke(a.condition)
    assert a.seed is not None, "--seed is needed for the main runs"
    return main_runs(a.condition, a.seed)


if __name__ == "__main__":
    main()
